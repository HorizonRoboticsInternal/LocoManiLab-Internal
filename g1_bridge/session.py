"""The host-agnostic half of G1_BRIDGE: everything that is teleoperation
rather than robosuite.

``TeleopSession`` owns the sim2real policy runtime (``Tracking``/``RobotIO``),
the PICO source, the ModeManager, the active backend, and the standing-reference
bookkeeping. It knows nothing about robosuite, MuJoCo scenes or robocasa: it
reads robot state through a :class:`TeleopHost` and hands back joint targets.

``G1BridgeComposite`` (composite.py) is one host -- the robosuite adapter.
A second host driving the real robot (or sim2real's own MuJoCo loop) needs only
the handful of methods in :class:`TeleopHost`, and gets the identical control
path, which is the point: the controller that recorded a dataset and the
controller that replays a policy against it must be the same code.

What deliberately does NOT live here
------------------------------------
The per-episode settling window. It exists because robocasa spawns the G1
airborne and lets it free-fall onto the floor -- a property of that host, not of
teleoperation -- so it stays in the composite, which drives this session's
``set_anchor_heading``/``push_standing_frame``/``seed_reference`` while it runs.
Anything genuinely host-specific belongs on the host for the same reason.

This module is importable without robosuite. That is a load-bearing property
(see CLAUDE.md, "Layering"), not an accident -- keep robosuite imports out.
"""

from __future__ import annotations

import numpy as np
from termcolor import colored

from g1_bridge.pico.mode_manager import ManagerMode, ModeManager
from g1_bridge.pico.vr_source import PicoVRSource, make_pico_source
from g1_bridge.g1_runtime import G1_JOINT_NAMES, policy_due
from g1_bridge.vendored.wbt.controllers.wbt_controller_sonic import (
    SONIC_DEFAULT_ANGLES_MUJOCO,
    SONIC_KDS,
    SONIC_KPS,
)

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from sim2real.config.robots.g1 import G1_CFG
from sim2real.rl_policy.robot_io.base import RobotState

from g1_bridge.backends import TickInput, make_backend
from g1_bridge.reference.default import default_reference_frame


# G1_CFG.joint_names is identical (same names, same order) to the repo's own
# g1_runtime.G1_JOINT_NAMES: 6 left leg, 6 right leg, 3 waist, 7 left arm,
# 7 right arm ("Unitree"/MuJoCo order). Asserted rather than commented, because
# a silent reordering upstream would mis-wire every joint in the PD loop while
# still producing plausible-looking motion.
_JOINT_NAMES = G1_CFG.joint_names
_NUM_JOINTS = len(_JOINT_NAMES)
assert tuple(_JOINT_NAMES) == G1_JOINT_NAMES, (
    "sim2real G1_CFG.joint_names diverged from g1_runtime.G1_JOINT_NAMES:\n"
    f"  sim2real:   {tuple(_JOINT_NAMES)}\n"
    f"  g1_runtime: {G1_JOINT_NAMES}"
)

# sim2real's Tracking defaults to rl_rate=50Hz (dt=0.02s); throttle policy
# inference to that rate and PD-hold the cached target in between substeps,
# mirroring wbt_g1_composite.WholeBodyG1Composite._POLICY_DT.
_POLICY_DT = 0.02


class _SharedResources:
    """Process-wide cache for the expensive-to-build pieces: PICO SDK
    connection, sim2real Tracking/ONNX session, GMR subprocess/relay.

    robosuite's ``LeggedRobot._load_controller()`` constructs a **brand new**
    composite controller instance via ``composite_controller_factory`` on
    every single ``env.reset()`` (confirmed empirically -- this is not
    specific to this composite; ``WholeBodyG1Composite`` has the exact same
    characteristic). An instance-level guard in ``load_controller_config``
    therefore can never fire -- ``self._tracking`` is always freshly ``None``
    -- so with ``--pico-source real`` every single task-success/reset during
    a live operator session would otherwise reconnect XRoboToolkit and
    reload the ONNX model. This cache lives at module scope instead, keyed
    by ``(backend, policy_config_path)`` so a genuine config change (e.g. a
    different checkpoint mid-process) still rebuilds cleanly.

    A fresh ``TeleopSession`` is built per episode alongside the composite and
    pulls from this cache; per-episode state therefore starts clean while the
    expensive handles survive, which is exactly the previous behaviour.
    """

    def __init__(self):
        self.key: tuple | None = None
        self.tracking = None
        self.io = None
        self.pico_source_key: tuple | None = None
        self.pico_source: PicoVRSource | None = None
        self.backend_resources = None

    def matches(self, key: tuple) -> bool:
        return self.key == key

    def set_tracking(self, key, tracking, io, backend_resources=None):
        self.key = key
        self.tracking = tracking
        self.io = io
        self.backend_resources = backend_resources

    def pico_source_matches(self, source_key) -> bool:
        return self.pico_source is not None and self.pico_source_key == source_key

    def set_pico_source(self, source_key, source):
        self.pico_source_key = source_key
        self.pico_source = source


_shared = _SharedResources()


class TeleopSession:
    """The teleop control path, implementing the ``Sim2RealContext`` protocol
    the backends are written against (see backends/base.py).

    Constructed per episode; heavy resources come from the module-level
    ``_shared`` cache. The host is expected to build the joint/actuator address
    maps and the hand mapping before calling :meth:`configure`, because the
    backends read them straight off this object.
    """

    def __init__(self, host):
        self._host = host
        self._pico_source: PicoVRSource | None = None
        self._mode_manager: ModeManager | None = None
        self._backend = None
        self._backend_impl = None
        self._ctrl_type = None
        self._tracking = None
        self._io = None
        self._cached_target_q = None
        self._cached_kps = None
        self._cached_kds = None
        self._last_policy_time = -1e9
        self._left_trigger = 0.0
        self._right_trigger = 0.0
        self._prev_manager_mode = ManagerMode.OFF  # for PLANNER-entry edge detection
        # See push_synthetic_default_once: primed once per episode, at the end
        # of the settling window, and then held. RealtimeMotionBuffer derives
        # body_lin_vel_w/body_ang_vel_w by finite-differencing consecutive pushed
        # frames over the time between them -- so re-pushing a frame that keeps
        # moving hands the policy a large, noisy reference velocity. A frame
        # pushed once leaves a single entry, which the buffer reports with zero
        # velocity outright (its len(timestamps_ns) == 1 branch). (The buffer now
        # runs on the host clock -- sim time in sim -- so the dt is at least
        # physically meaningful; pushing once is still the right call.)
        self._synthetic_default_primed = False
        # The standing frame the host's startup window holds, captured once at
        # the spawn pose -- see push_standing_frame. _qpos is the same pose as a
        # flat vector, for backends that need it as a blend anchor rather than
        # as a payload. _quat is the frozen heading every synthetic default
        # frame is built against; None means "use the live pelvis heading".
        self._settle_payload = None
        self._settle_qpos = None
        self._settle_quat = None
        # Policy default pose in G1_JOINT_NAMES order; see standing_joints().
        self._standing_joints = None

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def configure(self, cfg):
        """Build the backend, ModeManager, policy runtime and PICO source.

        Called from the host's own load/reset path. The ModeManager MUST go
        back to OFF every episode (matches WBT's own existing UX: operator
        re-engages with A+B+X+Y for every new demo, so an old "engaged" state
        never carries into a just-randomized scene). But the PICO SDK
        connection, sim2real Tracking/ONNX session, and GMR subprocess/relay
        must NOT be torn down and recreated each episode -- with
        ``--pico-source real`` that would mean reconnecting XRoboToolkit and
        reloading the ONNX model on every single task success/reset during a
        live operator session. Those are guarded separately via ``_shared``.
        """
        self._backend = cfg.get("sim2real_backend", "sonic")
        self._ctrl_type = f"g1_{self._backend}"
        # Every backend is a Sim2RealBackend -- see backends/base.py. Built
        # before the ModeManager because it owns which chord profile the
        # manager listens to: the GMR tracking backends have no kPlanner and no
        # VR-3pt controller, so B+Y / L-stick must not reach the table at all.
        self._backend_impl = make_backend(self._backend)
        self._mode_manager = ModeManager(
            initial_mode=ManagerMode.OFF,
            verbose=bool(cfg.get("mode_manager_verbose", True)),
            profile=self._backend_impl.mode_profile,
        )

        policy_config_path = str(cfg["sim2real_policy_config"])
        inference_backend = cfg.get("sim2real_inference_backend", "onnx-cpu")

        tracking_key = (self._backend, policy_config_path, inference_backend)
        if not _shared.matches(tracking_key):
            res = self._backend_impl.build_tracking(
                cfg, policy_config_path, inference_backend
            )
            # GMR sidecars (subprocess handle, button relay) stay inside
            # res.extra -- the backend that created them reads them back from
            # there (generic.py), and GMRSubprocess tears itself down via
            # atexit; the session never needs its own copy.
            _shared.set_tracking(
                tracking_key,
                res.tracking,
                res.io,
                backend_resources=res,
            )
        self._tracking = _shared.tracking
        self._io = _shared.io
        self._standing_joints = self._resolve_standing_joints()

        pico_key = (
            cfg.get("pico_source", "mock"),
            tuple(sorted((cfg.get("pico_source_kwargs") or {}).items())),
        )
        if not _shared.pico_source_matches(pico_key):
            self._init_pico_source(cfg)
            _shared.set_pico_source(pico_key, self._pico_source)
        else:
            self._pico_source = _shared.pico_source

        # UNGUARDED by design: unlike build_tracking above, this runs on every
        # env.reset() so per-episode state is re-seeded. See
        # Sim2RealBackend.on_load.
        #
        # Passed the CACHED resources, not a fresh object: build_tracking only
        # ran if the cache key changed, and what it decided once (the checkpoint
        # flavor, sidecar paths) must survive every later reset.
        self._backend_impl.on_load(self, cfg, _shared.backend_resources)

        # Put the reference-motion buffer on the HOST clock (sim time in a
        # headless sim, wall time on hardware) instead of its wall-clock
        # default. Under a sim host, wall time and sim time diverge -- and by a
        # machine-load-dependent amount -- so a wall-clocked buffer samples its
        # sim-tick-stamped frames at a drifting offset, making the tracking
        # reference (and the velocities finite-differenced from it) irreproducible
        # across runs. Every producer stamps with the same host clock (see
        # generic.py / adapters.py / _build_synthetic_default_payload), so one
        # monotonic clock governs both ends. On hardware the host clock IS wall
        # time, so this is a no-op there. See motion_buffer._default_clock_ns and
        # docs/sim_validation.md.
        buf = getattr(self._tracking, "motion_buffer", None)
        if buf is not None and hasattr(buf, "set_clock"):
            buf.set_clock(self._host_clock_ns)

    def _init_pico_source(self, cfg):
        source_type = cfg.get("pico_source", "mock")
        source_kwargs = cfg.get("pico_source_kwargs", {}) or {}
        # heft/scalebfm already run a GMR subprocess that owns its own
        # xrobotoolkit_sdk connection for full-body tracking. Opening a SECOND
        # real connection here (just for buttons/trigger) starves whichever
        # connection isn't "primary" of live body data -- verified live: the
        # non-primary connection's body pose freezes and its body timestamp
        # sticks at 0, which is exactly the "hand works, body doesn't follow"
        # (and its mirror image) bug reported in practice. Route buttons/
        # trigger through GMR's own ZMQ publisher instead -- see
        # gmr_button_relay.py for the full trade-off (VR_3PT unreachable).
        source = self._backend_impl.make_pico_source(cfg)
        if source is not None:
            self._pico_source = source
            return
        try:
            self._pico_source = make_pico_source(source_type, **source_kwargs)
            self._pico_source.start()
            print(f"[g1_bridge] PICO VR source: {source_type}")
        except Exception as e:
            if source_type == "real":
                raise RuntimeError(
                    "PICO source 'real' failed to start; refusing to fall "
                    "back to MockPicoVRSource because real hardware was "
                    "explicitly requested."
                ) from e
            print(
                f"[g1_bridge] Failed to start PICO source {source_type!r}: "
                f"{e}\n             Falling back to MockPicoVRSource."
            )
            self._pico_source = make_pico_source("mock")
            self._pico_source.start()

    def _resolve_standing_joints(self):
        """The policy's ``default_dof_angles``, in G1_JOINT_NAMES order.

        Which list indexes that vector is the whole subtlety, and it is NOT the
        obvious one. ``BasePolicy.__init__`` sizes it from
        ``joint_names_simulation`` but *fills* it via
        ``resolve_matching_names_values(..., self.action_manager.joint_names)``
        -- so the ordering is the ACTION MANAGER's, i.e. the robot cfg's, while
        ``joint_names_simulation`` is the policy's own (IsaacLab) order and
        differs for sonic and amo. Keying a permutation off
        ``joint_names_simulation`` therefore scrambles an already-correct
        vector: it lands knee angles on hip-roll joints and elbow angles on
        ankle-roll joints, and the robot settles into a wide, splayed crouch
        that still looks deliberate enough to be mistaken for a policy problem.

        So permute from ``action_manager.joint_names`` when it is available, and
        only when it actually differs. The module-level assert above already
        pins ``G1_CFG.joint_names == G1_JOINT_NAMES``, so in practice this is a
        no-op identity -- but doing it by name means a future robot cfg whose
        order does differ is handled instead of silently mis-wired.
        """
        tracking = self._tracking
        angles = getattr(tracking, "default_dof_angles", None)
        if angles is None:
            return None
        angles = np.asarray(angles, dtype=np.float64).reshape(-1)
        if angles.shape[0] != len(G1_JOINT_NAMES):
            return None
        action_manager = getattr(tracking, "action_manager", None)
        names = list(getattr(action_manager, "joint_names", []) or [])
        if not names:
            # No way to know the ordering; the vector is documented to follow
            # the robot cfg, which is what G1_JOINT_NAMES is asserted to be.
            return angles
        if len(names) != angles.shape[0]:
            return None
        if tuple(names) == G1_JOINT_NAMES:
            return angles
        try:
            order = [names.index(n) for n in G1_JOINT_NAMES]
        except ValueError as e:
            print(
                colored(
                    "[g1_bridge] policy action_manager.joint_names does "
                    f"not cover G1_JOINT_NAMES ({e}); falling back to the robot "
                    "cfg's default standing pose.",
                    "yellow",
                )
            )
            return None
        return angles[order]

    # ------------------------------------------------------------------ #
    # Sim2RealContext: the surface backends are allowed to use. Public
    # accessors for the private state above, so backends never depend on this
    # class's internals (and can be tested against a fake).
    # ------------------------------------------------------------------ #
    @property
    def tracking(self):
        return self._tracking

    @property
    def io(self):
        return self._io

    @property
    def pico_source(self):
        return self._pico_source

    @property
    def mode_manager(self):
        return self._mode_manager

    @property
    def backend_name(self):
        return self._backend

    @property
    def backend(self):
        """The active backend implementation (None before configure()).

        For hosts that need a backend-specific read the generic surface does
        not carry -- e.g. the eval client asking amo for its torso command
        state (``AmoBackend.torso_command_state``). Read-only use.
        """
        return self._backend_impl

    @property
    def ctrl_type(self):
        return self._ctrl_type

    @property
    def cached_target_q(self):
        return self._cached_target_q

    @property
    def left_trigger(self) -> float:
        return self._left_trigger

    @property
    def right_trigger(self) -> float:
        return self._right_trigger

    @property
    def settle_qpos(self):
        """The standing pose the host's startup window held, as a flat qpos."""
        return self._settle_qpos

    def standing_joints(self):
        """The joint angles this episode's policy stands in, in G1_JOINT_NAMES
        order. ``None`` when the checkpoint declares no default pose.

        Exposed on the context because backends build standing references too
        (seed_reference, the OFF ramp).
        """
        return self._standing_joints

    # -- host passthrough. Backends read robot state (and, for amo's joint
    # teleport, the raw sim) through the context rather than holding the host.
    @property
    def sim(self):
        """The MuJoCo sim, or None on a host that has none.

        A sim-only escape hatch: backends use it for their own wall-free clock
        (``sim.data.time``) and, in amo's case, to teleport joints to the
        policy's default pose at reset. A hardware host cannot honour the
        latter -- see AmoBackend.reset.
        """
        return getattr(self._host, "sim", None)

    @property
    def robot_model(self):
        return getattr(self._host, "robot_model", None)

    @property
    def hand_mapping(self):
        return getattr(self._host, "hand_mapping", None)

    @property
    def qpos_addrs(self):
        return getattr(self._host, "qpos_addrs", None)

    @property
    def qvel_addrs(self):
        return getattr(self._host, "qvel_addrs", None)

    def read_hand_state(self):
        """Measured finger joints from the host, or None when it has no hands."""
        read = getattr(self._host, "read_hand_state", None)
        return read() if read is not None else None

    def clock_s(self) -> float:
        """The host's clock in seconds -- sim time in sim, wall time on hardware.

        Backends rate-limit their own logging with this; going through the host
        (rather than ``ctx.sim.data.time``) is what lets a GMR tracking backend
        run on a host that has no ``sim`` at all.
        """
        return self._host.clock_ms() / 1000.0

    def _host_clock_ns(self) -> int:
        """The host clock in nanoseconds -- the motion buffer's time base.

        Same clock as :meth:`clock_s` (host ``clock_ms``), scaled to ns to
        match the buffer's frame timestamps and the ``publish_t_ns`` producers
        stamp with. Sim time under a sim host, wall time on hardware; installed
        onto the buffer in :meth:`configure`.
        """
        return int(self._host.clock_ms() * 1_000_000)

    def read_joint_state(self):
        return self._host.read_joint_state()

    def read_pelvis_state(self):
        return self._host.read_pelvis_state()

    def read_root_pos(self):
        return self._host.read_root_pos()

    def read_body_pose_w(self, body_name: str):
        return self._host.read_body_pose_w(body_name)

    # ------------------------------------------------------------------ #
    # Reference bookkeeping
    # ------------------------------------------------------------------ #
    def _build_synthetic_default_payload(self, qpos):
        """The reference the session holds while disengaged: sim2real's own
        paused frame, standing at the robot's current xy and heading.

        Delegates to ``g1_bridge.reference.default``, which is a port of
        ``pico_retarget_pub._capture_paused_qpos`` + ``_pose_arrays_from_qpos``
        -- see that module for why reproducing the publisher's recipe (rather
        than approximating it with the robot's own live body poses, which is
        what this used to do) is what makes the frame a standing pose, keeps its
        body layout identical to the live GMR stream's, and stops the robot's own
        roll/pitch from tilting the thing it is being asked to track.

        ``qpos`` is unused -- the frame's joint angles come from the policy's own
        default pose, not from the robot's current ones -- but stays in the
        signature because ``push_synthetic_default_once`` is part of the
        Sim2RealContext protocol the backends call through.
        """
        return default_reference_frame(self._tracking.robot_cfg).payload(
            self.read_root_pos(),
            # The host's frozen startup heading once it exists -- rebuilding
            # from the live pelvis each tick is what let the reference chase the
            # robot's yaw instead of correcting it. See push_standing_frame.
            self._settle_quat
            if self._settle_quat is not None
            else self.read_pelvis_state()[0],
            self.standing_joints(),
        )

    def set_anchor_heading(self, quat) -> None:
        """Freeze the heading every synthetic standing frame is built against.

        Called once per episode by the host, at the start of its startup window
        and against the robot's real post-spawn pose. Until then the live pelvis
        heading is used, which is the right default for a host whose robot is
        already standing (i.e. hardware).
        """
        self._settle_quat = np.asarray(quat).copy()

    def push_standing_frame(self) -> None:
        """The standing frame the host's startup window holds: the robot's
        current xy, but the heading FROZEN at what it spawned with.

        The split is the point, and each half fixes a different failure:

        * **Yaw frozen.** Re-reading the robot's heading every tick makes the
          heading error the policy sees independent of how far it has actually
          turned -- ``rel_quat = conj(q_robot) . (dyaw . yaw(q_robot))`` reduces
          to a constant, because yaw quaternions commute. The reference chases
          the robot instead of opposing it, so a yaw disturbance is never
          corrected and can wind up without bound. Freezing yaw is what makes
          the error real and therefore correctable.
        * **xy still tracked.** Robocasa spawns the G1 airborne and it slides
          0.2-0.3 m while the legs absorb the landing (measured). Pinning the
          reference to the spawn xy would leave the robot holding station
          against a target it has already slid away from; the original design
          note -- "the target stays a pose worth reaching for while the legs
          absorb the impact" -- is right about this half.

        The host then hands this exact frame on (``push_synthetic_default_once``)
        instead of rebuilding one, so the episode's standing reference is
        continuous through the hand-off.
        """
        if getattr(self._backend_impl, "owns_policy_runtime", False):
            return
        self._settle_payload = self._build_synthetic_default_payload(None)
        self._settle_qpos = default_reference_frame(
            self._tracking.robot_cfg
        ).default_qpos_at(
            self.read_root_pos(), self._settle_quat, self.standing_joints()
        )
        self.prime_state_processor()
        self._tracking.motion_buffer.push_payload(dict(self._settle_payload))

    def push_synthetic_default_once(self, qpos):
        """Push ``_build_synthetic_default_payload`` into the motion buffer
        exactly once per episode (guarded by ``self._synthetic_default_primed``,
        cleared in ``reset()``), NOT every tick.

        Why once and not per tick: sim2real's ``RealtimeMotionBuffer`` computes
        ``body_lin_vel_w``/``body_ang_vel_w`` by finite-differencing consecutive
        pushed frames over the time between them (see motion_buffer.py's
        ``__append_payload``/``_fill_sample_frames_locked``). The buffer runs on
        the host clock -- sim time under a sim host, since configure() installs
        it -- so the dt is now physically meaningful rather than a tiny wall-time
        interval; but a frame re-pushed every tick still finite-differences any
        residual movement in the "held" pose into a noisy reference velocity fed
        straight to the policy. A single push leaves exactly one frame in the
        buffer, which ``_fill_sample_frames_locked`` explicitly reports with
        zero velocity (its ``len(timestamps_ns) == 1`` branch).

        First reached from the host's startup window via the backend's
        ``seed_reference``. It re-uses that window's own final frame rather than
        building a fresh one: both are "the standing pose at the robot's current
        place", but rebuilding re-reads the heading, and the robot's heading has
        moved by the time it has finished landing. Measured 3.1 deg of yaw and
        0.2-0.3 m of xy between the two -- an unflagged step in the reference at
        the exact hand-off, which sonic's encoder sees directly (its motion
        anchor is the reference root relative to the live pelvis). Backends that
        call it again on later ticks get a no-op.
        """
        if self._synthetic_default_primed:
            return
        if self._settle_payload is not None:
            self.prime_state_processor()
            self._tracking.motion_buffer.push_payload(dict(self._settle_payload))
        else:
            # No startup window ran (a backend reaching here on its own, with
            # no GMR payload yet) -- nothing to be continuous with, so build one.
            self._push_default_reference(qpos)
        self._synthetic_default_primed = True

    def _push_default_reference(self, qpos) -> None:
        """Prime the state processor from the live robot and push one default
        standing frame, anchored to the robot's current xy/heading.

        Deliberately does NOT run ``Tracking.reset()``: the observations'
        heading alignment is calibrated exactly once per episode, in the host's
        startup window, and sim2real's motion-stream contract forbids redoing it
        mid-episode (``docs/policy_yaw_alignment.md``) -- a reference frame that
        gets re-defined at runtime creates exactly the reference jumps the
        contract exists to prevent.
        """
        self.prime_state_processor()
        self._tracking.motion_buffer.push_payload(
            self._build_synthetic_default_payload(qpos)
        )

    def prime_state_processor(self) -> None:
        """Push the robot's actual current state into state_processor before
        Tracking.reset() runs.

        BridgeTracking.__init__ already calls reset() once (via
        set_policy_mode()) before any real state has ever been pushed --
        state_processor starts all-zeros, i.e. a degenerate/non-unit root
        quaternion. Some observations calibrate something from
        state_processor.root_quat_w at reset time (e.g. SONIC's
        sonic_motion_anchor_ori_b_mf_nonflat / sonic_smpl_root_ori_b_
        multi_future heading-offset, used to yaw-align the reference motion
        to the robot at episode start): confirmed this bakes in a bogus
        ~180 degree yaw offset for the *entire* episode when computed
        against that degenerate quaternion (projected_yaw_quat([0,0,0,0])
        returns a 180-degree-yaw quaternion, not identity). Priming with the
        real pose here -- for every backend, not just SONIC -- means every
        Tracking.reset() (including the implicit one inside
        BridgeTracking.__init__) sees a valid orientation instead.
        """
        qpos, _qvel = self.read_joint_state()
        quat, ang_vel = self.read_pelvis_state()
        state = RobotState(
            qpos=np.concatenate([np.zeros(3), quat, qpos]).astype(np.float64),
            qvel=np.concatenate([np.zeros(3), ang_vel, np.zeros(_NUM_JOINTS)]).astype(
                np.float64
            ),
            joint_torque=np.zeros(_NUM_JOINTS, dtype=np.float64),
            tick=self._host.clock_ms(),
        )
        self._io.push_state(state)
        self._tracking.state_processor._prepare_low_state()

    def calibrate_alignment(self) -> None:
        """Arm the policy, calibrating the observations' one-shot
        robot-vs-motion yaw alignment against whatever reference was just
        pushed. The host calls this once per episode, with the robot in its
        real pose and a reference anchored to that same pose.
        """
        if getattr(self._backend_impl, "owns_policy_runtime", False):
            return
        self._tracking.set_policy_mode(source="g1_bridge")

    # ------------------------------------------------------------------ #
    # Control path
    # ------------------------------------------------------------------ #
    def step_policy(self, qpos):
        quat, ang_vel = self.read_pelvis_state()
        # Real joint velocities, NOT zeros. Zeroing them here (the original
        # code did) leaves the policy's joint_vel observations pinned at 0
        # for the whole episode -- the policy runs blind to its own joint
        # motion, so any wobble goes undamped and amplifies tick over tick
        # until the sim explodes. Confirmed the hard way for sonic-g1: an
        # offline A/B against the proven-working WBTControllerSonic showed
        # both pipelines BIT-IDENTICAL (enc/dec inputs and per-tick targets)
        # on a static state, yet in-env sonic diverged 0.9 -> 4 -> 16 rad
        # within 3 ticks -- the only in-env difference being that WBT's
        # _update_history feeds real sim joint_vel while this fed zeros.
        # (The outer PD damping term in the host's write always used real
        # qvel -- only the policy's observations were blinded.)
        _qpos_now, qvel = self.read_joint_state()
        state = RobotState(
            qpos=np.concatenate([np.zeros(3), quat, qpos]).astype(np.float64),
            qvel=np.concatenate([np.zeros(3), ang_vel, qvel]).astype(np.float64),
            joint_torque=np.zeros(_NUM_JOINTS, dtype=np.float64),
            tick=self._host.clock_ms(),
        )
        self._io.push_state(state)
        # sim2real's own loops (BasePolicy.run, integrated_sim2sim) maintain this
        # counter; nothing did on the bridge path, where it stayed absent and
        # getattr'd to -1 forever. The scalebfm and amo observations use
        # (id(data), total_inference_cnt) to collapse the nine same-tick
        # update() calls their nine obs objects make into one -- with the
        # counter pinned, that key degenerates to the address of the temporary
        # dict Tracking.update() rebuilds each tick (measured: 19 distinct
        # addresses across 32 ticks), so two consecutive ticks landing on the
        # same address would silently freeze the whole observation history.
        self._tracking.total_inference_cnt = (
            getattr(self._tracking, "total_inference_cnt", 0) + 1
        )
        self._tracking.step()
        if self._io.last_command is None:
            return self.hold_pose()
        q_target, _dq_target, _tau_ff, kp, kd = self._io.last_command
        return q_target, kp, kd

    def hold_pose(self):
        """Last-resort fallback only -- used before the policy has ever
        produced a command (e.g. no GMR payload received yet at all). Once
        any payload exists, OFF uses _step_generic's frozen default instead
        of this."""
        return SONIC_DEFAULT_ANGLES_MUJOCO.astype(np.float32), SONIC_KPS, SONIC_KDS

    def begin_tick(self, sample, settling: bool):
        """Advance the ModeManager from this tick's buttons and refresh the
        gripper triggers. Returns ``(mode, buttons)``.

        On entry into PLANNER from a non-PLANNER mode the backend re-syncs its
        yaw accumulator to the robot's actual heading (see
        ``_resync_facing_to_robot``) so steering behaves like angular-velocity
        control across mode switches, not absolute-orientation control --
        ported from WholeBodyG1Composite's identical PLANNER-family entry check.
        The family is PLANNER + PLANNER_VR_3PT (see backends.sonic's
        _in_planner_family); PLANNER_FROZEN_UPPER is still not implemented
        here.
        """
        buttons = sample.get("buttons") if sample is not None else None
        self._mode_manager.tick(buttons)
        mode = self._mode_manager.current
        engaged = mode != ManagerMode.OFF

        if mode != self._prev_manager_mode:
            self._backend_impl.on_mode_change(self, self._prev_manager_mode, mode)
        self._prev_manager_mode = mode

        # Gripper trigger. The default (Sim2RealBackend.read_triggers) gates on
        # ModeManager engagement, matching WBT's convention exactly: hands
        # forced open in OFF. Backends that don't use engagement for control at
        # all override it -- see AmoBackend.read_triggers. Hands stay open
        # through the host's startup window regardless of backend: the robot is
        # not under operator control yet (amo's read_triggers override
        # deliberately ignores engagement, so gating on `engaged` alone would
        # let it grip mid-landing).
        self._left_trigger, self._right_trigger = (
            (0.0, 0.0)
            if settling
            else self._backend_impl.read_triggers(buttons, engaged)
        )
        return mode, buttons

    def sample_pico(self):
        return self._pico_source.get_latest() if self._pico_source is not None else None

    def policy_due(self, sim_time) -> bool:
        return self._cached_target_q is None or policy_due(
            sim_time, self._last_policy_time, _POLICY_DT
        )

    def cache_command(self, target_q, kps, kds, sim_time):
        self._cached_target_q = np.asarray(target_q, dtype=np.float32)
        self._cached_kps = np.asarray(kps, dtype=np.float32)
        self._cached_kds = np.asarray(kds, dtype=np.float32)
        self._last_policy_time = sim_time

    @property
    def cached_command(self):
        return self._cached_target_q, self._cached_kps, self._cached_kds

    def step_backend(self, tick: TickInput):
        return self._backend_impl.step(self, tick)

    def vla_step_backend(self, tick: TickInput, action: dict):
        """One policy-driven tick, the inverse of :meth:`step_backend`.

        The backend supplies the command space; the action supplies the values.
        No operator sample and no ModeManager gating is consulted -- there is
        no operator.
        """
        return self._backend_impl.vla_step(self, tick, action)

    def vla_hand_targets(self, action: dict):
        return self._backend_impl.vla_hand_targets(action)

    def vla_reset(self) -> None:
        self._backend_impl.vla_reset()

    @property
    def vla_action_space(self) -> str:
        return self._backend_impl.vla_action_space

    def seed_reference(self, tick: TickInput) -> None:
        """Hand the episode's real reference to the policy.

        The backend's first push after this is automatically flagged as a new
        motion segment (its own continuity tracker is still unset for the
        episode), so ``RealtimeMotionBuffer`` re-attaches that reference to the
        settled pose the policy has been holding rather than teleporting it.
        """
        self._backend_impl.seed_reference(self, tick)

    # ------------------------------------------------------------------ #
    # Episode lifecycle
    # ------------------------------------------------------------------ #
    def reset(self):
        """Clear per-episode state and the inner controller's own history.

        This runs while the robot still holds the PREVIOUS/stale pose --
        robosuite resets controllers before the scene teleports the robot to
        its spawn pose -- so the Tracking.reset() below is deliberately NOT
        the one that calibrates anything. Everything pose-dependent (the
        observations' one-shot robot-vs-motion yaw alignment) is done in the
        host's startup window, on the first policy tick, once the robot is
        actually standing where the episode put it. See calibrate_alignment.
        """
        if self._tracking is not None and not getattr(self._backend_impl, "owns_policy_runtime", False):
            # BEFORE Tracking.reset(), which reads the buffer immediately.
            # The motion buffer is cached process-wide (see _shared) while this
            # session is rebuilt every episode, so anything left in it --
            # frames, and above all the segment transform -- would otherwise
            # carry into a scene the robot has just been teleported into. A
            # stale transform silently rotates the next episode's reference,
            # which reads as the robot spinning up and falling over on the
            # second demo of a session. See RealtimeMotionBuffer.reset_stream.
            # hasattr, not a bare call: the SMPL buffer has no reset_stream.
            buf = getattr(self._tracking, "motion_buffer", None)
            if hasattr(buf, "reset_stream"):
                buf.reset_stream()
            self.prime_state_processor()
            self._tracking.reset()
        self._cached_target_q = None
        self._cached_kps = None
        self._cached_kds = None
        self._last_policy_time = -1e9
        self._prev_manager_mode = ManagerMode.OFF
        self._synthetic_default_primed = False
        self._settle_payload = None
        self._settle_qpos = None
        self._settle_quat = None

        self._backend_impl.reset(self)

    # ------------------------------------------------------------------ #
    # Operator surface
    # ------------------------------------------------------------------ #
    def handle_keypress(self, keycode):
        """'E' toggles OFF<->PLANNER and 'P' toggles PLANNER<->POSE -- the same
        two transitions A+B+X+Y and A+X drive; use these if the chords aren't
        registering (if the keys work but the robot still doesn't move, look at
        GMRRelay/motion payload staleness instead, not the chord). Together they
        make the whole state machine drivable with ``--pico-source mock``, which
        is the only way to exercise it without hardware. For the sonic backend
        once in PLANNER mode (kPlanner locomotion armed), also handles the
        arrow-key/gait-select/facing/speed/height keys ported from
        WholeBodyG1Composite's handle_keypress -- see KEY_HELP for the full
        list."""
        if keycode == ord("E") and self._mode_manager is not None:
            new_mode = (
                ManagerMode.PLANNER
                if self._mode_manager.current == ManagerMode.OFF
                else ManagerMode.OFF
            )
            self._mode_manager.force_mode(new_mode)
            return
        # 'P', not 'T': TeleopViewer binds T to the HUD toggle and is offered
        # keys first (demo_empty_teleop's handler), so a T here would be dead.
        if keycode == ord("P") and self._mode_manager is not None:
            # From OFF this engages straight into POSE rather than doing
            # nothing -- 'E' is the way to reach the idle state deliberately.
            new_mode = (
                ManagerMode.PLANNER
                if self._mode_manager.current == ManagerMode.POSE
                else ManagerMode.POSE
            )
            self._mode_manager.force_mode(new_mode)
            return

        self._backend_impl.handle_keypress(self, keycode)

    @property
    def hud_cheatsheet(self):
        """The viewer's on-screen control map, delegated to the backend.

        Every backend means something different by the same chord (see
        Sim2RealBackend.hud_cheatsheet), and the 'E'/'P' keyboard fallbacks that
        live in ``handle_keypress`` above are already folded into each backend's
        rows. Empty before a backend is attached, which drops the panel."""
        backend = self._backend_impl
        if backend is None:
            return ()
        return backend.hud_cheatsheet

    def teleop_sample(self) -> dict:
        """This tick's recording sample, from the active backend."""
        return self._backend_impl.teleop_sample(self)

    def on_episode_start(self) -> None:
        """Tell the backend a new recorded episode begins.

        Forwarded rather than letting a recorder reach into ``_backend_impl``:
        the session is the only object the collectors hold, on either host.
        """
        self._backend_impl.on_episode_start()
