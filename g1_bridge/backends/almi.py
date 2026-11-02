"""ALMI: PICO-stick locomotion + GMR-retargeted waist and arms.

ALMI (mjlab-loco ``Mjlab-Velocity-Flat-Unitree-G1-Dual``) is the first
dual-branch backend here: two neural actors run in parallel every tick, one
walking on the 12 leg joints and one driving the 17 waist/arm joints. Together
they cover all 29, so unlike amo nothing downstream patches ``q_target``.

It is also the first backend whose command is *split by body half*:

    legs        twist = [vx, vy, wz, base_height]     <- PICO sticks
    waist+arms  q_ref(17) + torso-frame EE pose        <- GMR whole-body retarget

That split is not a UX choice, it is what the checkpoint was trained on. Read
back out of the weights: the twist block's normalizer statistics match uniform
command sampling, while ``q_ref_error``'s std is 0.05-0.34 -- far above the
0.0058 a permanently-zero term plus its uniform noise would leave -- so the
upper branch really did see a live motion reference. A deployment that drives
the arms through EE targets alone (which is correct for the *slope* checkpoint,
whose ``q_ref_error`` std really is 0.0058) would be throwing away the channel
this one learned on. Still true of the 2026-08-17 retrain under the
``half_body`` observation layout (std 0.052-0.311), whose command set is
unchanged -- what moved there is what the two branches OBSERVE, not what this
backend has to publish. See sim2real/rl_policy/observations/almi.py.

Three states, sharing the ``"tracking"`` chord profile with heft/scalebfm:

    OFF   --A+B+X+Y-->  IDLE  --A+X-->  POSE  --A+X-->  IDLE
     ^                                                    |
     +------------------- A+B+X+Y ----------------- (any) +

  POSE  legs follow the sticks, waist and arms follow the operator's body.
  IDLE  twist zeroed to a standing command, upper reference *held* at its last
        value rather than snapped back -- the arms stay where the operator left
        them instead of whipping to a default pose.
  OFF   same hold, and nothing is recorded.

Unlike heft/scalebfm this backend never pushes the motion buffer: ALMI consumes
commands, not a reference trajectory. The buffer still exists (BridgeTracking
builds one) and quietly serves its own standing default frame, which nothing
reads.

Deliberate difference from mjlab-loco's own PICO script
(``deploy_mujoco/slope/reference.py`` on the deploy box): that script puts base
height on the two index triggers. Here the triggers belong to the Dex3 hands,
which every backend in this repo shares, so height moves to the right stick's Y
axis -- integrated, with the same "turning wins" suppression amo uses, because
a thumb pushing sideways to turn unavoidably carries a vertical component.
"""

from __future__ import annotations

import numpy as np
from termcolor import colored

from g1_bridge.pico.mode_manager import ManagerMode

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from sim2real.rl_policy.families import get_family
from sim2real.rl_policy.observations.almi import (
    NEUTRAL_EE_POS,
    NEUTRAL_EE_QUAT,
    NUM_UPPER_REF_JOINTS,
    AlmiCommands,
)
from sim2real.utils.dex3 import grip_to_joint_targets
from sim2real.utils.math import quat_conjugate, quat_mul, quat_rotate_inverse_numpy

from g1_bridge.backends.base import (
    BackendResources,
    Sim2RealBackend,
    Sim2RealContext,
    TickInput,
    read_hand_state,
)
from g1_bridge.reference.default import default_reference_frame

_POLICY_DT = 0.02

#: Bodies the EE command is expressed between. mjlab's MotionEECommand measures
#: the two wrist-yaw links in the torso frame; using a different pair (the
#: hands, the wrist-pitch links) shifts every target by a constant offset the
#: policy has no way to know about.
_TORSO_BODY = "torso_link"
_EE_BODIES = ("left_wrist_yaw_link", "right_wrist_yaw_link")

#: Stick shaping. Deadzones match amo's so the two backends feel the same in the
#: hand; the output scale is ALMI's own trained range (see almi_command_limits).
_STICK_DEADZONE = 0.15
_VX_MAX = 0.8
_VY_MAX = 0.5
_WZ_MAX = 1.8

#: Base height is integrated, not proportional: releasing the stick must hold
#: the height the operator dialled in, not snap back to a baseline.
_HEIGHT_RATE = 0.30  # m/s at full deflection
_YAW_DEADZONE = 0.20  # suppresses height while turning (see module docstring)
_YAW_RELEASE = 0.15  # Schmitt release, so height does not stutter on/off

#: Ticks of missing GMR payload before we say so once. At 50 Hz this is ~0.5 s.
_DROPOUT_WARN_TICKS = 25


def _deadzone(value: float, threshold: float = _STICK_DEADZONE) -> float:
    """Proportional past the deadzone, rescaled so the output still reaches 1."""
    magnitude = abs(value)
    if magnitude < threshold:
        return 0.0
    scaled = (magnitude - threshold) / (1.0 - threshold)
    return float(np.clip(scaled, 0.0, 1.0)) * (1.0 if value > 0 else -1.0)


class AlmiBackend(Sim2RealBackend):
    """Dual-branch locomotion + whole-body-retargeted upper body."""

    name = "almi"
    controller_json = "g1_generic.json"
    #: Consumes the GMR motion stream, so the PICO source must come through
    #: GMRButtonRelay rather than a second xrobotoolkit connection.
    uses_gmr = True
    mode_profile = "tracking"
    #: The split command this backend records and a policy must emit. Neither
    #: ``velocity_cmd`` (no EE/joint reference) nor ``ref_qpos`` (no twist), so
    #: it is its own space -- see sim2real.rl_policy.families.
    vla_action_space = "twist_ee"

    POLICY_CONFIGS = {"almi": get_family("almi").policy_config}

    @property
    def hud_cheatsheet(self):
        return (
            ("", "-- MODE (PICO chords) --"),
            ("A+B+X+Y / E", "OFF <-> IDLE (engage / e-stop)"),
            ("A+X / P", "IDLE <-> POSE (drive the robot)"),
            ("", "-- POSE --"),
            ("L-stick", "walk (vx / vy)"),
            ("R-stick X", "turn (wz)"),
            ("R-stick Y", "body height (held on release)"),
            ("your body", "waist + arms (GMR retargeting)"),
            ("triggers", "close hands"),
        )

    def __init__(self):
        self._relay = None
        self._limits = {}
        self._policy_config = {}
        self._upper_names = []
        # The standing pose, expressed exactly the way a GMR frame is. Filled in
        # on_load; see _resolve_neutral_reference for why it is not a constant.
        self._fk = None
        self._neutral = None
        self._configured_base_height = None
        # Command state carried across ticks. base_height is an integrator, and
        # the upper reference is *held* rather than reset when we leave POSE.
        self._base_height = 0.65
        self._turning = False
        self._q_ref = None
        self._ee_pos = None
        self._ee_quat = None
        self._commands = AlmiCommands.neutral()
        # `None` until a real GMR frame has been consumed -- this is the
        # recording readiness gate, so a dataset never opens with a synthesized
        # reference the operator did not produce.
        self._have_reference = False
        self._dropout_ticks = 0
        self._warned_dropout = False
        # Recording: IDLE (non-POSE) ticks have no operator command behind them
        # -- the twist is zero and the upper reference is HELD from the last POSE
        # frame -- so, like heft, they are not recorded unless the controller
        # config says so (sim2real_record_idle, configs/g1_generic.json).
        self._record_idle = False
        self._driving = False
        # The height an episode opens at (resolved in reset(), where the ctx is
        # available); on_episode_start() returns to it without a ctx.
        self._episode_base_height = self._base_height

    # ------------------------------------------------------------------ #
    # Construction (cache-key-guarded: about once per process)
    # ------------------------------------------------------------------ #
    def build_tracking(self, cfg, policy_config_path, inference_backend):
        from g1_bridge.bridge_runtime import make_bridge_tracking
        from g1_bridge.gmr_utils.publisher_proc import start_gmr_if_available
        from g1_bridge.gmr_utils.relay import GMRRelay

        gmr_proc = start_gmr_if_available(cfg)  # None on a host without the pico venv (eval)
        connect = cfg.get("gmr_motion_zmq_connect") or "tcp://127.0.0.1:28701"
        gmr_relay = GMRRelay(connect=connect)
        # push, not zmq: we subscribe ourselves, because we read joint angles
        # and body poses out of the payload to build a command rather than
        # feeding it to the buffer as a reference trajectory.
        tracking, io = make_bridge_tracking(
            policy_config_path,
            motion_mode="push",
            inference_backend=inference_backend,
        )
        print(
            f"[g1_bridge] G1_BRIDGE armed, backend={self.name} "
            f"(policy_config={policy_config_path}, GMR subprocess {'live' if gmr_proc else 'not started'}). "
            "A+B+X+Y engages (IDLE, stands still); A+X starts driving -- "
            "left stick walks, right stick turns and sets body height, and "
            "your waist and arms are retargeted onto the robot."
        )
        return BackendResources(
            tracking=tracking,
            io=io,
            extra={"gmr_proc": gmr_proc, "gmr_relay": gmr_relay},
        )

    def make_pico_source(self, cfg):
        """Buttons/sticks over GMR's own ZMQ publisher when hardware is live.

        Same reason as the other GMR backends: the publisher already owns an
        xrobotoolkit connection for body tracking, and a second real connection
        starves whichever one is not primary of live body data.
        """
        if cfg.get("pico_source", "mock") != "real":
            return None
        from g1_bridge.gmr_utils.button_relay import GMRButtonRelay

        source = GMRButtonRelay()
        source.start()
        return source

    # ------------------------------------------------------------------ #
    # Per env.reset()
    # ------------------------------------------------------------------ #
    def on_load(self, ctx: Sim2RealContext, cfg: dict, res: BackendResources) -> None:
        self._relay = res.extra["gmr_relay"]
        self._policy_config = ctx.tracking.policy_config
        self._upper_names = self._resolve_upper_joint_names(self._policy_config)
        limits = dict(self._policy_config.get("almi_command_limits", {}))
        self._limits = {
            key: np.asarray(value, dtype=np.float32)
            for key, value in limits.items()
        }
        missing = {
            "twist_min",
            "twist_max",
            "waist_rpy_min",
            "waist_rpy_max",
            "ee_pos_min",
            "ee_pos_max",
        } - set(self._limits)
        if missing:
            raise ValueError(
                f"policy.yaml is missing almi_command_limits entries {sorted(missing)}. "
                "They are what keeps a retargeted operator inside the command "
                "distribution the policy was trained on."
            )
        requested = cfg.get("almi_base_height")
        self._configured_base_height = (
            None if requested is None else float(requested)
        )
        self._record_idle = bool(cfg.get("sim2real_record_idle", False))
        self._fk = default_reference_frame(ctx.tracking.robot_cfg)
        self._neutral = self._resolve_neutral_reference(ctx)

    def _resolve_neutral_reference(self, ctx: Sim2RealContext):
        """The upper reference to hold before (and between) real GMR frames.

        Forward-kinematics the policy's own standing pose and read it back
        through the SAME two extractors a live payload goes through, so the
        neutral is a real, self-consistent pose rather than a set of constants:
        ``q_ref``, the waist derived from it, and the EE pose that pose actually
        produces all agree, because they come from one frame.

        This is load-bearing, not tidiness. The first version of this backend
        used ``deploy_mujoco/reference_sources.py``'s ``_DEFAULT_EE_POS`` /
        ``_DEFAULT_EE_QUAT`` with ``q_ref`` left None (so the observation fell
        back to ``q_ref = q_current``). Those constants belong to the *slope*
        rig, not this checkpoint: they put the hands at z=+0.20 where this
        robot's standing hands are at z=-0.068, with an identity wrist
        orientation where the real one is w=0.917 -- and paired with a q_ref
        that tracked the live pose, the EE target and the joint reference
        disagreed every tick. Measured on IThorKitchenCloseFridge over 20 s:
        with those constants the robot walked backwards 1.52 m and never
        stopped; with this neutral it comes to rest at 0.18 m and stays there
        (tilt 1.1 deg), which is what the same checkpoint does in mjlab's own
        sim2sim. Nothing about the plant changed between those two runs.
        """
        joint_pos = np.asarray(ctx.tracking.default_dof_angles, dtype=np.float64)
        qpos = self._fk.default_qpos_at(
            [0.0, 0.0, self._fk.default_root_height],
            [1.0, 0.0, 0.0, 0.0],
            joint_pos,
        )
        payload = self._fk.payload_from_qpos(qpos)
        q_ref = self._q_ref_from_payload(payload)
        ee_pos, ee_quat = self._ee_from_payload(payload)
        return q_ref, ee_pos, ee_quat

    def reset(self, ctx: Sim2RealContext) -> None:
        self._base_height = self._initial_base_height(ctx)
        self._episode_base_height = self._base_height
        self._driving = False
        self._turning = False
        # Start from the standing pose rather than "no reference": the upper
        # branch runs every tick whether or not an operator is streaming, so it
        # needs a coherent pose to hold, not a placeholder. `_have_reference`
        # stays False so nothing is recorded until a real frame arrives.
        self._q_ref, self._ee_pos, self._ee_quat = (
            (None, None, None) if self._neutral is None else
            (self._neutral[0].copy(), self._neutral[1].copy(), self._neutral[2].copy())
        )
        self._commands = AlmiCommands.neutral()
        self._have_reference = False
        self._dropout_ticks = 0
        self._warned_dropout = False

    def on_episode_start(self) -> None:
        """Per-recorded-episode state (see Sim2RealBackend.on_episode_start).

        ``reset()`` is per ``env.reset()`` -- once per hardware session -- so
        the two things a dataset defines per episode are re-taken here:

        * the recording gate closes again (``_have_reference``): the next
          episode's first recorded frame must come from a reference the
          operator produced IN that episode, not the pose held over from the
          previous take;
        * the height integrator returns to the episode-opening height, so a
          crouch dialled in during one take is not the next take's opening
          command (sim: ``reset()`` did this already; this is a no-op there).

        The held upper reference itself is kept -- the robot is physically
        still holding it, and clearing it would snap the arms.
        """
        self._have_reference = False
        self._base_height = float(self._episode_base_height)
        self._turning = False

    def _initial_base_height(self, ctx: Sim2RealContext) -> float:
        """The height an episode opens at: the robot's own standing height.

        Derived, not a constant, so ALMI opens at the same posture every other
        controller holds -- they all sit at ``RobotCfg.default_qpos``'s root
        height, and a locomotion policy is the only one here that could pick
        something else. Clipped into the trained band, which is what makes the
        0.80 m natural stand come out as 0.79.

        This used to be a flat 0.65, i.e. a 15 cm crouch. Measured on
        IThorKitchenCloseFridge over 15 s, with everything else identical:

            commanded   pelvis_z   knee   tilt   drift
            0.65        -0.19      1.40   2.3    -0.36 .. -1.31 (varies per run)
            0.75        -0.048     0.73   2.7    -0.14
            0.79        -0.043     0.71   1.5    -0.083
            heft ref    -0.025     0.55   0.7    +0.012

        A knee-1.4 crouch is only marginally stable -- hence the run-to-run
        spread at 0.65 -- and standing up removes it. ``almi_base_height`` in
        the controller spec overrides this (``lab collect --set``) for an
        operator who wants to start low.
        """
        lo, hi = self._twist_min()[3], self._twist_max()[3]
        requested = self._configured_base_height
        if requested is None:
            requested = float(self._fk.default_root_height)
        return float(np.clip(requested, lo, hi))

    def _twist_min(self):
        return self._limits.get("twist_min", np.array([-0.8, -0.5, -1.8, 0.35], np.float32))

    def _twist_max(self):
        return self._limits.get("twist_max", np.array([0.8, 0.5, 1.8, 0.79], np.float32))

    # ------------------------------------------------------------------ #
    # Per tick
    # ------------------------------------------------------------------ #
    def step(self, ctx: Sim2RealContext, tick: TickInput):
        driving = tick.mode == ManagerMode.POSE
        self._driving = driving
        twist = self._twist_from_sticks(tick, driving)
        if driving:
            self._update_reference_from_gmr()

        self._commands = AlmiCommands(
            twist=twist,
            # Overwritten from q_ref inside validated() whenever a reference
            # exists; only the no-reference case leaves the waist level.
            waist_rpy=np.zeros(3, dtype=np.float32),
            ee_pos=NEUTRAL_EE_POS if self._ee_pos is None else self._ee_pos,
            ee_quat=NEUTRAL_EE_QUAT if self._ee_quat is None else self._ee_quat,
            upper_ref_joint_pos=self._q_ref,
        ).validated()
        ctx.tracking.almi_commands = self._commands
        return ctx.step_policy(tick.qpos)

    def _twist_from_sticks(self, tick: TickInput, driving: bool) -> np.ndarray:
        """Left stick walks, right stick turns and integrates body height.

        Height is a posture the operator dials in, not a motion, so disengaging
        *holds* it rather than zeroing it -- zeroing would command a crouch into
        the floor every time they paused. It does not keep integrating while
        disengaged either: a thumb resting on the stick between takes would
        otherwise walk the command away silently, and the robot would change
        height the moment POSE was re-entered. Only the velocity channels are
        zeroed, so IDLE and OFF stand still at whatever height was set.
        """
        buttons = tick.buttons or {}
        lx, ly = buttons.get("left_axis", (0.0, 0.0))
        rx, ry = buttons.get("right_axis", (0.0, 0.0))

        lo, hi = self._twist_min(), self._twist_max()
        if not driving:
            return np.array([0.0, 0.0, 0.0, self._base_height], dtype=np.float32)

        # Turning wins over height: both live on the right stick, and a thumb
        # pushing sideways to turn carries a vertical component that would
        # otherwise walk the height command away on every turn. Suppression is
        # only tolerable because height is an integrator -- it means "hold",
        # not "output zero".
        if abs(rx) >= _YAW_DEADZONE:
            self._turning = True
        elif abs(rx) < _YAW_RELEASE:
            self._turning = False
        if not self._turning:
            self._base_height += _HEIGHT_RATE * _deadzone(ry) * _POLICY_DT
        self._base_height = float(np.clip(self._base_height, lo[3], hi[3]))

        twist = np.array(
            [
                _deadzone(ly) * _VX_MAX,
                _deadzone(-lx) * _VY_MAX,
                _deadzone(-rx) * _WZ_MAX,
                self._base_height,
            ],
            dtype=np.float32,
        )
        return np.clip(twist, lo, hi).astype(np.float32)

    def _update_reference_from_gmr(self) -> None:
        """Turn one retargeted payload into q_ref + torso-frame EE pose.

        A dropped frame holds the last reference rather than reverting to a
        neutral pose: the arms are mid-motion, and snapping them to a default
        because one ZMQ message was late is worse than being 20 ms stale.
        """
        payload = self._relay.latest() if self._relay is not None else None
        if payload is None:
            self._note_dropout("no GMR payload yet")
            return

        try:
            q_ref = self._q_ref_from_payload(payload)
            ee_pos, ee_quat = self._ee_from_payload(payload)
        except (KeyError, ValueError) as exc:
            self._note_dropout(f"unusable GMR payload ({exc})")
            return

        if not (
            np.all(np.isfinite(q_ref))
            and np.all(np.isfinite(ee_pos))
            and np.all(np.isfinite(ee_quat))
        ):
            self._note_dropout("GMR payload has non-finite values")
            return

        self._set_upper_reference(q_ref, ee_pos, ee_quat)
        self._dropout_ticks = 0
        self._warned_dropout = False

    def _set_upper_reference(self, q_ref, ee_pos, ee_quat) -> None:
        """Clip an upper-body reference into the trained box and hold it.

        Shared by the operator path and the VLA path on purpose. The box is
        what keeps whoever is driving inside the command distribution the
        policy was trained on, and a retargeted operator and an extrapolating
        model need that guarantee for exactly the same reason -- so this cannot
        be a thing one of the two paths remembers to do.

        Clipping lands on the waist entries of ``q_ref`` itself, not on the
        derived ``waist_rpy`` command: ``AlmiCommands.validated()`` re-derives
        one from the other, so clipping the derived copy would silently put the
        two back out of sync.
        """
        q_ref = np.asarray(q_ref, dtype=np.float32).copy()
        waist_lo = self._limits["waist_rpy_min"]
        waist_hi = self._limits["waist_rpy_max"]
        # q_ref waist order is (yaw, roll, pitch); the limits are (roll, pitch, yaw).
        q_ref[0] = np.clip(q_ref[0], waist_lo[2], waist_hi[2])
        q_ref[1] = np.clip(q_ref[1], waist_lo[0], waist_hi[0])
        q_ref[2] = np.clip(q_ref[2], waist_lo[1], waist_hi[1])

        self._q_ref = q_ref
        self._ee_pos = np.clip(
            ee_pos, self._limits["ee_pos_min"], self._limits["ee_pos_max"]
        ).astype(np.float32)
        self._ee_quat = np.asarray(ee_quat, dtype=np.float32).reshape(-1)
        self._have_reference = True

    def _q_ref_from_payload(self, payload: dict) -> np.ndarray:
        names = payload.get("joint_names")
        values = payload.get("joint_pos")
        if not names or values is None:
            raise KeyError("payload has no joint_names/joint_pos")
        index = {str(name): i for i, name in enumerate(names)}
        joint_pos = np.asarray(values, dtype=np.float32).reshape(-1)
        missing = [name for name in self._upper_names if name not in index]
        if missing:
            raise KeyError(f"GMR payload missing joints {missing}")
        return np.array(
            [joint_pos[index[name]] for name in self._upper_names], dtype=np.float32
        )

    @staticmethod
    def _resolve_upper_joint_names(policy_config: dict) -> list[str]:
        """The 17 reference joints, read from the same place the policy reads.

        Taken off the observation term's own kwargs rather than hardcoded, so
        the order the backend publishes and the order the observation consumes
        cannot drift apart -- they are literally the same list.
        """
        group = policy_config["observation"]["upper_obs"]["upper_obs"]
        names = [str(name) for name in group["upper_joint_names"]]
        if len(names) != NUM_UPPER_REF_JOINTS:
            raise ValueError(
                f"policy.yaml declares {len(names)} upper_joint_names, expected "
                f"{NUM_UPPER_REF_JOINTS}"
            )
        return names

    def _ee_from_payload(self, payload: dict) -> tuple[np.ndarray, np.ndarray]:
        """Wrist poses in the torso frame of the same retargeted pose.

        The publisher already forward-kinematicked the retargeted qpos, so the
        world poses are in the payload and no second FK model is needed here --
        which also means this cannot drift from what the publisher computed.
        """
        names = payload.get("body_names")
        pos_w = payload.get("body_pos_w")
        quat_w = payload.get("body_quat_w")
        if not names or pos_w is None or quat_w is None:
            raise KeyError("payload has no body_names/body_pos_w/body_quat_w")
        index = {str(name): i for i, name in enumerate(names)}
        for required in (_TORSO_BODY, *_EE_BODIES):
            if required not in index:
                raise KeyError(f"GMR payload missing body {required!r}")
        pos_w = np.asarray(pos_w, dtype=np.float32).reshape(len(names), 3)
        quat_w = np.asarray(quat_w, dtype=np.float32).reshape(len(names), 4)

        torso_pos = pos_w[index[_TORSO_BODY]]
        torso_quat = quat_w[index[_TORSO_BODY]]
        torso_conj = quat_conjugate(torso_quat.reshape(1, 4))

        positions, quaternions = [], []
        for body in _EE_BODIES:
            i = index[body]
            positions.append(
                quat_rotate_inverse_numpy(
                    torso_quat.reshape(1, 4), (pos_w[i] - torso_pos).reshape(1, 3)
                ).reshape(3)
            )
            quaternions.append(
                quat_mul(torso_conj, quat_w[i].reshape(1, 4)).reshape(4)
            )
        return (
            np.concatenate(positions).astype(np.float32),
            np.concatenate(quaternions).astype(np.float32),
        )

    def _note_dropout(self, reason: str) -> None:
        self._dropout_ticks += 1
        if self._dropout_ticks >= _DROPOUT_WARN_TICKS and not self._warned_dropout:
            self._warned_dropout = True
            held = "holding last arm reference" if self._have_reference else (
                "no operator frame yet -- waist and arms hold the standing pose "
                "and nothing is recorded until body tracking streams"
            )
            print(
                colored(
                    f"[g1_bridge][almi] {reason}; {held}. Check the GMR "
                    "subprocess, the headset pairing, and body-tracking "
                    "calibration.",
                    "yellow",
                )
            )

    # ------------------------------------------------------------------ #
    # VLA-driven operation (the inverse of step(): the policy replaces both
    # the operator's thumbs and the operator's body)
    # ------------------------------------------------------------------ #
    def vla_sink(self, ctx: Sim2RealContext):
        """The command lives on this backend, not on the policy runtime.

        Same as amo, and for the same reason: ALMI consumes commands rather
        than a reference trajectory, so there is no motion buffer to push into.
        """
        return self

    def set_twist_ee_command(self, cmd: dict) -> None:
        """Fill this tick's split command from policy output.

        The fields are ``teleop_sample``'s, one for one, and they are consumed
        the same way the operator's are -- the same clip box on the upper
        reference, the same twist limits. The only difference from
        ``_twist_from_sticks`` is height: the operator dials it in through an
        integrator because a thumb gives a rate, while the recorded column (and
        therefore the prediction) is the absolute posture in metres. Re-feeding
        that through the integrator would double-integrate a value that is
        already the answer -- the same trap amo's ``target_yaw`` documents.
        """
        lo, hi = self._twist_min(), self._twist_max()
        lin = cmd["lin_vel"]
        self._base_height = float(np.clip(cmd["base_height"][0], lo[3], hi[3]))
        twist = np.array(
            [lin[0], lin[1], cmd["ang_vel_z"][0], self._base_height],
            dtype=np.float32,
        )
        self._set_upper_reference(
            cmd["ref_joint_pos"],
            np.concatenate([cmd["left_ee_pos"], cmd["right_ee_pos"]]),
            np.concatenate([cmd["left_ee_quat"], cmd["right_ee_quat"]]),
        )
        self._commands = AlmiCommands(
            twist=np.clip(twist, lo, hi).astype(np.float32),
            # Overwritten from the reference inside validated(); see the
            # adapter's `derived_keys`.
            waist_rpy=np.zeros(3, dtype=np.float32),
            ee_pos=self._ee_pos,
            ee_quat=self._ee_quat,
            upper_ref_joint_pos=self._q_ref,
        ).validated()

    def vla_step(self, ctx: Sim2RealContext, tick: TickInput, action: dict):
        """Commands from the policy instead of from the sticks and the headset.

        Deliberately does not consult ``tick.mode``: under a VLA there is no
        operator to engage, and the sim host does not even sample the PICO. The
        POSE/IDLE distinction the operator path applies is about *whose* command
        wins, and here there is only one candidate.
        """
        self.apply_vla_action(ctx, action)
        ctx.tracking.almi_commands = self._commands
        return ctx.step_policy(tick.qpos)

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def teleop_sample(self, ctx: Sim2RealContext) -> dict:
        """State = measured joint angles by limb; action = the command stream.

        The action side is exactly what was fed to ALMI this tick, which is
        what a policy replacing the operator has to emit. The 29-D joint target
        the two branches produced is deliberately *not* recorded: it is a
        function of this command plus the checkpoint, so storing it would add a
        second, redundant action space for a VLA to learn a shortcut through.

        ``upper_ref_joint_pos`` is None until the first real retargeted frame,
        which is the layout's readiness source -- ticks before that are skipped
        rather than written with a synthesized reference. It is None again on
        every non-POSE tick unless ``sim2real_record_idle`` is set: IDLE holds
        the last reference motionless with a zero twist, and recording that
        would assert a command the operator is not giving (the contract
        g1_generic.json states and heft implements). A VLA-driven tick
        (``vla_step``) is never ``_driving`` either, so eval/DAgger runs that
        happen to have a recorder attached do not write policy output as if it
        were teleop.
        """
        if not self._have_reference or not (self._driving or self._record_idle):
            return {"upper_ref_joint_pos": None}
        qpos, _qvel = ctx.read_joint_state()
        commands = self._commands
        hands = read_hand_state(ctx)
        zeros7 = np.zeros(7, dtype=np.float32)
        return {
            "left_leg": np.asarray(qpos[0:6], dtype=np.float32),
            "right_leg": np.asarray(qpos[6:12], dtype=np.float32),
            "waist": np.asarray(qpos[12:15], dtype=np.float32),
            "left_arm": np.asarray(qpos[15:22], dtype=np.float32),
            "right_arm": np.asarray(qpos[22:29], dtype=np.float32),
            "left_hand_state": np.asarray(hands.get("left", zeros7), dtype=np.float32),
            "right_hand_state": np.asarray(hands.get("right", zeros7), dtype=np.float32),
            "twist_lin_vel": commands.twist[0:2].astype(np.float32),
            "twist_ang_vel_z": commands.twist[2:3].astype(np.float32),
            "twist_base_height": commands.twist[3:4].astype(np.float32),
            "waist_rpy": commands.waist_rpy.astype(np.float32),
            "left_ee_pos": commands.ee_pos[0:3].astype(np.float32),
            "right_ee_pos": commands.ee_pos[3:6].astype(np.float32),
            "left_ee_quat": commands.ee_quat[0:4].astype(np.float32),
            "right_ee_quat": commands.ee_quat[4:8].astype(np.float32),
            "upper_ref_joint_pos": np.asarray(
                commands.upper_ref_joint_pos, dtype=np.float32
            ),
            "left_hand_action": grip_to_joint_targets(
                float(np.clip(ctx.left_trigger, 0.0, 1.0)), "left"
            ).astype(np.float32),
            "right_hand_action": grip_to_joint_targets(
                float(np.clip(ctx.right_trigger, 0.0, 1.0)), "right"
            ).astype(np.float32),
        }
