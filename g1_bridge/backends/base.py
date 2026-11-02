"""The per-backend strategy interface for ``G1BridgeComposite``.

``G1_BRIDGE`` drives four very different whole-body policies (sonic, amo,
heft, scalebfm). That used to be a ``self._backend`` string compared at
~20 sites, with every backend's state sharing one flat ``__init__`` -- so adding
a backend meant editing three files in about ten places, with nothing to tell
you which ones you missed.

A backend is now an object. The composite keeps the parts that are genuinely
common (address maps, mode manager, the 50 Hz throttle, the PD write, the hand
torques) and delegates the rest through the hooks below.

Two rules keep this working:

* **The context is a Protocol, not the composite class.** Backends never import
  ``composite.py``; if they did, ``composite -> backends -> composite`` would be
  a cycle. It also makes a backend testable against a fake for the first time.
* **Class attributes must be readable without instantiating.**
  ``load_sim2real_controller_config`` runs before ``robosuite.make()``, so there
  is no composite and no sim yet. Heavy imports (torch, pinocchio, onnxruntime)
  therefore stay function-local, and the registry in ``__init__`` is lazy.

The lifecycle split is the part that is easy to get wrong in both directions,
so it is stated on each hook: ``build_tracking``/``make_pico_source`` are
cache-key-guarded and run about once per process, while ``on_load`` runs on
*every* ``env.reset()``. Getting that backwards is silent either way -- a
per-episode reseed that stops happening, or an ONNX session leak that only
shows up dozens of episodes in.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


#: The on-screen key list for the G1_BRIDGE control path. It lives here, next to
#: the keys themselves (the chords are ModeManager's, the kPlanner keys are
#: sonic's ``handle_keypress``), rather than on the composite that happens to
#: print it: a backend wanting to print it used to reach *up* into
#: ``composite/bridge.py``, which is the one import direction this module's
#: docstring forbids -- and which becomes a hard cross-package cycle once the
#: composite is the only part left on the sim side. The composite still exposes
#: it as ``G1BridgeComposite.KEY_HELP`` for the viewer loop; this is the one copy.
BRIDGE_KEY_HELP = (
    "  [sim2real] focus the MuJoCo window; PICO A+B+X+Y (all 4, same "
    "tick) toggles OFF<->PLANNER engagement, A+X toggles\n"
    "             PLANNER<->POSE. On heft/scalebfm that reads as "
    "OFF -> IDLE (hold) -> POSE (track your body) -> IDLE (freeze\n"
    "             where you left it); re-entering POSE blends from the "
    "frozen pose back to your live one.\n"
    "    E          - keyboard fallback for the OFF<->PLANNER toggle "
    "(use if the 4-button chord isn't registering)\n"
    "    P          - keyboard fallback for the A+X PLANNER<->POSE "
    "toggle\n"
    "  sonic backend only: PICO L-stick click toggles PLANNER<->VR_3PT "
    "(legs keep walking, arms follow\n"
    "             your controllers); holding the PICO left menu button in "
    "POSE pauses it (POSE_PAUSE) until released.\n"
    "  sonic backend, PLANNER mode (kPlanner locomotion) only:\n"
    "    Up/Down/Left/Right - walk forward/back/strafe (body frame)\n"
    "    B          - brake (movement_direction = 0)\n"
    "    , / .      - rotate facing left / right\n"
    "    0..9       - locomotion mode (0=IDLE 1=SLOW_WALK 2=WALK 3=RUN ...)\n"
    "    - / =      - target speed -/+ \n"
    "    ; / '      - target height -/+ \n"
    "    N          - re-init kPlanner from the current robot pose\n"
    "    /          - print current MovementState + this help\n"
)


@dataclass
class BackendResources:
    """Whatever ``build_tracking`` produced, cached process-wide.

    ``tracking``/``io`` are the only fields the composite itself touches (its
    ``_step_policy`` needs them). ``extra`` is opaque to the composite and is
    handed straight back to the backend -- it exists so backend-specific
    resources (a GMR subprocess, a relay, a checkpoint sidecar path) stop
    widening the shared cache's signature the way they used to.
    """

    tracking: Any
    io: Any
    flavor: str = "default"
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TickInput:
    """One ``run_controller`` tick, assembled once by the composite."""

    qpos: np.ndarray  # (29,) in G1_JOINT_NAMES order
    qvel: np.ndarray  # (29,)
    sim_time: float
    mode: Any  # ManagerMode
    buttons: dict | None
    sample: dict | None  # pico_source.get_latest()

    @property
    def engaged(self) -> bool:
        from g1_bridge.pico.mode_manager import ManagerMode

        return self.mode != ManagerMode.OFF


@runtime_checkable
class Sim2RealContext(Protocol):
    """Exactly what the composite exposes to its backend.

    Structural, not nominal: backends type-annotate against this and never
    import the composite.
    """

    sim: Any
    robot_model: Any
    tracking: Any
    io: Any
    pico_source: Any
    mode_manager: Any
    #: Flat qpos of the standing frame the settling window held, or None before
    #: settling has begun. The right blend anchor for an episode's first engage:
    #: it is exactly the pose already sitting in the motion buffer.
    settle_qpos: Any
    #: The last (target_q, kps, kds) the 50 Hz throttle cached, or None before
    #: the first policy tick. Backends read it to record what actually drove
    #: the robot this tick (sonic/amo teleop_sample).
    cached_target_q: Any
    #: Latest hand-trigger values in [0, 1]; the generic/amo teleop_samples
    #: record the grip command straight from these.
    left_trigger: float
    right_trigger: float
    #: MuJoCo qpos/qvel addresses for G1_JOINT_NAMES, or None on a host with
    #: no simulator (amo's reset-time joint teleport checks before using).
    qpos_addrs: Any
    qvel_addrs: Any

    def read_joint_state(self):
        ...

    def read_hand_state(self):
        """Measured finger joint angles, ``{"left": (7,), "right": (7,)}`` or None.

        A host answers this from wherever its fingers live -- MuJoCo qpos in sim,
        the Dex3 driver's DDS feedback on hardware -- so a backend that records
        hand state never has to know which one it is running on. ``None`` means
        this host has no hands attached, which is a legitimate configuration
        (``--dex3 none``) and not an error.
        """
        ...

    def clock_s(self) -> float:
        """The host's own clock, in seconds -- sim time in sim, wall time on
        hardware. Backends must use this rather than ``ctx.sim.data.time``:
        the latter does not exist on a host with no simulator, and it is the
        only reason a GMR tracking backend would need ``sim`` at all."""
        ...

    def _host_clock_ns(self) -> int:
        """The host clock in nanoseconds -- the same clock as ``clock_s`` (sim
        time in sim, wall time on hardware), scaled for the motion buffer's
        timestamps. A backend stamps a reference frame it feeds the buffer with
        this so producer and consumer share one clock; see
        motion_buffer._default_clock_ns."""
        ...

    def standing_joints(self):
        """The joint angles the running policy stands in (G1_JOINT_NAMES
        order), or None. Not ``RobotCfg.default_qpos`` -- see
        ``default_reference.DefaultReferenceFrame.standing_joints``."""
        ...

    def read_pelvis_state(self):
        ...

    def read_root_pos(self):
        ...

    def read_body_pose_w(self, body_name: str):
        ...

    def step_policy(self, qpos):
        ...

    def hold_pose(self):
        ...

    def push_synthetic_default_once(self, qpos) -> None:
        ...


def read_hand_state(ctx) -> dict:
    """Measured finger joints as ``{"left": (7,), "right": (7,)}``, always a dict.

    Through the host, so a backend never learns which one it runs on: MuJoCo
    qpos in sim, the Dex3 driver's DDS feedback on hardware.

    Two absences collapse to the same empty dict on purpose, because both mean
    "this run has no fingers to report" and neither is an error: a host with no
    hands attached (``--dex3 none``) answers None, and a context that predates
    the method -- a test double, or a host built before hand state was recorded
    -- does not have it at all. ``Sim2RealSession.read_hand_state`` reaches for
    its own host the same defensive way.

    Callers fill the missing side with zeros (see the recorders), matching what
    ``build_blob_frame_fields`` does for any absent group.
    """
    reader = getattr(ctx, "read_hand_state", None)
    return (reader() if reader is not None else None) or {}


class Sim2RealBackend(abc.ABC):
    """One whole-body policy family behind ``G1_BRIDGE``."""

    # -- class attributes: readable at config time, before any sim exists ----
    #: ``--backend`` value.
    name: str = ""
    #: which controller JSON under ``g1_bridge/configs/`` to load.
    controller_json: str = "g1_generic.json"
    #: True when this backend runs the GMR retargeting subprocess. Drives both
    #: the ``--no-viewer`` overlay and the PICO-source swap (a GMR backend must
    #: NOT open a second xrobotoolkit connection -- see gmr_button_relay).
    uses_gmr: bool = True
    #: Which ModeManager chord profile this backend's UX needs (see
    #: mode_manager._PROFILE_CHORDS). "sonic" = the full 5-mode machine;
    #: "tracking" = OFF/PLANNER(idle)/POSE only, for backends with no kPlanner
    #: and no VR-3pt controller.
    mode_profile: str = "sonic"

    @property
    def hud_cheatsheet(self):
        """``(key, meaning)`` rows for the MuJoCo-window cheat sheet (an empty
        key is a section heading), surfaced through the composite's own
        ``hud_cheatsheet``.

        The default is exactly the chords :attr:`mode_profile` leaves live, so
        a backend that adds no input of its own gets a correct sheet for free
        and cannot advertise a chord the ModeManager mutes. Backends that do add
        input (sonic's kPlanner steering, amo's joystick mapping) override it.
        A property, not a class attribute: amo's sheet depends on a config flag
        that is only known after ``on_load``.
        """
        if self.mode_profile == "tracking":
            return (
                ("", "-- MODE (PICO chords) --"),
                ("A+B+X+Y / E", "OFF <-> IDLE (engage / e-stop)"),
                ("A+X / P", "IDLE <-> POSE (track your body)"),
                ("triggers", "close hands (engaged only)"),
            )
        return (
            ("", "-- MODE (PICO chords) --"),
            ("A+B+X+Y / E", "OFF <-> PLANNER (engage / e-stop)"),
            ("A+X / P", "PLANNER <-> POSE"),
            ("B+Y", "POSE <-> FROZEN"),
            ("L-click", "PLANNER/FROZEN <-> VR_3PT"),
            ("triggers", "close hands (engaged only)"),
        )

    @property
    def modality_layout(self):
        """This backend's LeRobot recording schema.

        Resolved through ``g1_bridge.modality`` rather than held as a
        class attribute so that module -- which promises to be importable
        without the sim2real environment -- never has to import a backend.
        """
        from g1_bridge.modality import G1_BACKEND_TELEOP_KEYS

        return G1_BACKEND_TELEOP_KEYS[self.name]

    # -- config time (no instance state, no sim) -----------------------------
    @classmethod
    def config_overlay(cls, *, pico_source: str, gmr_viewer: bool) -> dict:
        """Extra ``composite_controller_specific_configs`` entries.

        Default: suppress GMR's viser web visualizer, which otherwise competes
        with MuJoCo rendering and policy inference for CPU/GPU badly enough to
        visibly stall the retargeting loop.
        """
        if cls.uses_gmr and not gmr_viewer:
            return {"gmr_extra_args": ["--no-viewer"]}
        return {}

    # -- construction: GUARDED by (name, policy_config, inference_backend) ---
    @abc.abstractmethod
    def build_tracking(
        self, cfg: dict, policy_config_path: str, inference_backend: str
    ) -> BackendResources:
        """Build the sim2real Tracking/RobotIO pair (plus any sidecar).

        Runs only when the cache key changes -- about once per process, NOT per
        episode. Anything that must be re-seeded per episode belongs in
        ``on_load``. Owns its own "armed" banner.
        """

    def make_pico_source(self, cfg: dict):
        """Build this backend's PICO input source.

        Also cache-key-guarded. Default is the shared implementation; override
        only if the backend needs a different source (the GMR backends do).
        Returning None means "use the composite's default".
        """
        return None

    # -- per env.reset(): UNGUARDED, runs every episode ----------------------
    @abc.abstractmethod
    def on_load(self, ctx: Sim2RealContext, cfg: dict, res: BackendResources) -> None:
        """Per-episode (re)initialisation from config.

        Runs on EVERY ``load_controller_config``, i.e. every ``env.reset()``.
        This is where per-episode reseeding lives (sonic's MovementState, amo's
        IK solver). Moving any of it into ``build_tracking`` makes it stick at
        episode 1's values, which looks like a policy problem, not a bug.
        """

    @abc.abstractmethod
    def reset(self, ctx: Sim2RealContext) -> None:
        """Clear per-episode control state."""

    def seed_reference(self, ctx: Sim2RealContext, tick: "TickInput") -> None:
        """Put this episode's opening reference into the motion buffer WITHOUT
        running the policy.

        Called once per episode, at the end of the composite's settling window
        (``G1BridgeComposite._end_settling``), to swap the self-anchored
        landing reference for the one this episode will actually track.

        The composite has already calibrated the observations' one-shot
        robot-vs-motion yaw alignment by this point, and sim2real's
        motion-stream contract forbids recalibrating mid-episode
        (``docs/policy_yaw_alignment.md``: "Policy 只在 observation reset() 时处理
        机器人当前 yaw 和 motion 初始 yaw 不一致的问题"). So an implementation
        must flag its first push as a new motion segment
        (``motion_first_frame``) and let ``RealtimeMotionBuffer`` re-attach the
        reference to the pose the policy is currently holding -- not assume it
        can re-anchor the observation frame instead.

        Default is a no-op, which is correct for backends that never feed the
        motion buffer (amo is command-velocity driven, not motion-tracking).
        """

    # -- per tick ------------------------------------------------------------
    def on_mode_change(self, ctx: Sim2RealContext, prev, new) -> None:
        """ModeManager transition, at substep rate.

        Called outside the policy throttle deliberately: it reads live pelvis
        state, so deferring it to the next policy tick would both delay the
        edge by up to 20 ms and capture a different heading.
        """

    @abc.abstractmethod
    def step(self, ctx: Sim2RealContext, tick: TickInput):
        """Return ``(target_q_29, kps, kds)``. Throttled to 50 Hz."""

    @staticmethod
    def read_triggers(buttons: dict | None, engaged: bool) -> tuple[float, float]:
        """Hand trigger values for this tick.

        Default follows the ModeManager convention: hands forced open unless
        engaged. amo overrides -- it never uses ModeManager engagement for
        control, so gating its hands on it would mean they only respond after
        an unrelated A+B+X+Y chord.
        """
        if buttons is not None and engaged:
            return (
                float(buttons.get("left_trigger", 0.0)),
                float(buttons.get("right_trigger", 0.0)),
            )
        return 0.0, 0.0

    def handle_keypress(self, ctx: Sim2RealContext, keycode: int) -> bool:
        """Backend-specific keys. Return True if consumed."""
        return False

    def on_episode_start(self) -> None:
        """A new RECORDED episode begins. Default: nothing.

        Distinct from :meth:`reset`, and the difference is the whole reason this
        exists. ``reset`` runs on ``env.reset()``, which is a per-episode event
        in sim and does not happen at all between an operator's episodes on
        hardware -- one hardware run is many episodes inside one ``reset``. So a
        backend that records anything measured *from the start of the episode*
        (heft's episode-relative reference root, see backends/generic) cannot
        take its origin in ``reset`` without silently sharing episode 1's origin
        with every later episode of the same run.

        Fired from ``EpisodeRecorder.reset_episode()`` -- the one episode
        boundary both hosts already go through.
        """

    @abc.abstractmethod
    def teleop_sample(self, ctx: Sim2RealContext) -> dict:
        """This tick's recording sample.

        Keys must match the sources declared by :attr:`modality_layout` -- the
        two live on the same object so they cannot drift apart unnoticed.
        """

    # -- VLA-driven operation (the inverse of teleop) -------------------------
    #: The command space a policy must emit to drive this backend, as named in
    #: ``sim2real.rl_policy.families`` / ``sim2real.vla.adapters``. Empty means
    #: this backend has no VLA path yet.
    #:
    #: It is not a free choice: it is whatever :meth:`teleop_sample` records,
    #: because that recording IS the training target. Change one and the other
    #: is already wrong.
    vla_action_space: str = ""

    def vla_sink(self, ctx: Sim2RealContext):
        """The object the adapter delivers into.

        Default is the policy runtime, which is what the motion-buffer command
        spaces (``ref_qpos``, ``smpl``, ``motion_token``) write to. Backends
        that own their command state -- amo's velocity/limb targets -- return
        themselves instead.
        """
        return ctx.tracking

    def make_vla_adapter(self, ctx: Sim2RealContext):
        """Build this backend's adapter. Override to supply constructor args
        the adapter cannot obtain on its own (a forward-kinematics function, a
        payload builder) -- those belong to the host, not to the command space."""
        from sim2real.vla.adapters import get_adapter

        return get_adapter(self.vla_action_space)

    def apply_vla_action(self, ctx: Sim2RealContext, action: dict) -> None:
        """Deliver one timestep of policy output into this backend."""
        if not self.vla_action_space:
            raise NotImplementedError(
                f"backend {self.name!r} has no VLA command space yet. "
                "Set vla_action_space (and implement whatever consumes it) to "
                "make it drivable by a policy."
            )
        if getattr(self, "_vla_adapter", None) is None:
            self._vla_adapter = self.make_vla_adapter(ctx)
        self._vla_adapter.apply(self.vla_sink(ctx), action)

    def vla_hand_targets(self, action: dict):
        """``(left_7, right_7)`` finger joint targets, or None to keep the
        trigger-driven grip.

        Hands are a separate PD loop from the whole-body policy in every
        backend here, so they are also a separate prediction: a policy that
        outputs them drives the fingers directly instead of through a grip
        scalar that has no way to express a specific grasp.

        Three spellings, because a server returns whatever ITS OWN embodiment
        config declares -- which is baked into the checkpoint at training time,
        not read from this repo:

        * ``{left,right}_hand_joints`` -- sonic's legacy per-column layout.
        * ``{left,right}_hand`` -- what every blob-schema backend (heft,
          scalebfm, amo, almi) declares as the ``action.hands`` groups, and
          what a checkpoint trained against a current embodiment config emits.
        * ``hands`` -- the same 14 values UNSPLIT, left 7 then right 7. This is
          what the heft embodiment config asked for before 2026-08-11, so every
          checkpoint trained on a GR00T that predates that change returns it,
          whatever this repo now says.

        Missing a spelling is silent by construction: ``hands`` is returned,
        found under none of the names, and the fingers fall back to the
        operator's trigger -- which, in an eval with no operator, is 0.0, a
        hand that never closes. The gr00t_heft_qfix checkpoint predicted the
        grasp to within 2% of the recording and still evaluated 0/5 with the
        basket never leaving the floor, because those 14 numbers were dropped
        between the server and the fingers with nothing logged either side.
        """
        import numpy as np

        from sim2real.vla.client import action_field

        for left_key, right_key in (
            ("left_hand_joints", "right_hand_joints"),
            ("left_hand", "right_hand"),
        ):
            try:
                left = np.asarray(action_field(action, left_key), np.float32)
                right = np.asarray(action_field(action, right_key), np.float32)
            except KeyError:
                continue
            return left.reshape(-1), right.reshape(-1)

        try:
            packed = np.asarray(action_field(action, "hands"), np.float32).reshape(-1)
        except KeyError:
            return None
        if packed.size != 14:
            raise ValueError(
                f"action key 'hands' has {packed.size} values, expected 14 "
                "(left 7 then right 7 Dex3 finger targets). A 2-D open/close "
                "scalar is the pre-2026-08-11 heft schema, which no current "
                "backend drives."
            )
        return packed[:7], packed[7:]

    def vla_step(self, ctx: Sim2RealContext, tick: TickInput, action: dict):
        """One VLA-driven tick: apply the action, then run the policy.

        The teleop counterpart of :meth:`step`, and deliberately a separate
        method: ``step`` reads the operator sample and the ModeManager, neither
        of which exists here, so sharing it would mean threading "is there an
        operator?" through every backend.
        """
        self.apply_vla_action(ctx, action)
        return ctx.step_policy(tick.qpos)

    def vla_reset(self) -> None:
        """Forget per-episode VLA state."""
        adapter = getattr(self, "_vla_adapter", None)
        if adapter is not None:
            adapter.reset()
