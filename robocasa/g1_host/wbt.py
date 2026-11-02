"""WholeBodyG1Composite — robosuite composite controller that delegates the
inner control law to a wbt (whole_body_teleop) controller.

Registration: importing this module runs ``@register_composite_controller`` and
makes ``WHOLE_BODY_G1`` constructible. robosuite's own
``controllers/composite/__init__.py`` used to do that import; it no longer can,
because this module lives here and robosuite must not depend on g1_bridge
(see README.md). ``robocasa/__init__.py`` triggers it instead, so
every robocasa consumer sees the controller registered exactly as before. Code
that uses this controller *without* importing robocasa must import this module
itself, or ``composite_controller_factory`` will raise "not imported or loaded".

Supports ``wbt_controller_type``:
  - ``default_pos``   — wbt's DefaultPosController (no policy, just PD-hold).
  - ``sonic``         — wbt's WBTControllerSonic with a constant default-pose
                        reference (Sonic does not stand still on its own —
                        useful only as a "Sonic plumbing smoke test").
  - ``sonic_planner`` — wbt's WBTControllerSonic driven by the GR00T kinematic
                        planner (``planner_sonic.onnx``).  This is what
                        matches SONIC official deploy: a 10 Hz planner emits a
                        50 Hz reference motion sequence, Sonic tracks it.
                        Supports keyboard-driven MovementState changes.

``gentle`` / ``wbt`` will be slotted into ``_make_wbt_controller`` when needed.

Design notes
------------
* wbt's controllers expect a 29-DoF MuJoCo joint order (left leg 6, right leg 6,
  waist 3, left arm 7, right arm 7) and return ``(target_q, kps, kds)``.  We
  build a mapping from this order to robosuite's actuator/qpos indices, padding
  absent joints (e.g. legs in G1FixedLowerBody) with zeros.
* Torques are written directly to ``sim.data.ctrl`` here.  ``run_controller``
  returns an empty dict so ``Robot.control()`` does not double-write via its
  part-keyed dispatch.  This sidesteps the right/left ``_joint_split_idx``
  heuristic in robosuite, which assumes right-first actuator ordering — G1's
  MJCF lists left first.
* Hand actuators (14 finger motors) are left at zero ctrl; wbt's 29-D output
  does not include fingers.
* For floating-base variants the pelvis quaternion + body-frame angular vel
  are read from the freejoint, so Sonic gets a real gravity-direction signal.
"""

from __future__ import annotations

import os
from typing import Dict

import numpy as np

from robosuite.controllers.composite.composite_controller import (
    CompositeController,
    register_composite_controller,
)
from g1_bridge.sonic_utils.kplanner import (
    LocomotionMode,
    MovementState,
    SonicKPlanner,
    STATIC_MODES,
)

# Sim-layer primitives + robosuite-facing shell shared with the other G1
# composites.
from robocasa.g1_host.base import G1CompositeBase
from g1_bridge.g1_runtime import (
    G1_JOINT_NAMES,
    heading_yaw_from_quat,
    quat_inv,
    quat_mul,
    relative_anchor6,
    resolve_ckpt_path,
    rotate_in_xy,
    rotate_yaw,
)

# Vendored wbt controllers — see wbt_vendored/__init__.py for provenance.
from g1_bridge.vendored.wbt.controllers.base_controller import (
    BaseController,  # noqa: F401 — re-exported for downstream isinstance checks
)
from g1_bridge.vendored.wbt.controllers.default_pos_controller import (
    DefaultPosController,
)
from g1_bridge.vendored.wbt.controllers.wbt_controller_sonic import (
    SONIC_DEFAULT_ANGLES,
    SONIC_DEFAULT_ANGLES_MUJOCO,
    WBTControllerSonic,
)
from g1_bridge.vendored.wbt.controllers.wbt_controller_sonic_teleop import (
    WBTControllerSonicTeleop,
)
from g1_bridge.vendored.wbt.controllers.wbt_controller_sonic_smpl import (
    WBTControllerSonicSmpl,
)
from g1_bridge.pico.vr_source import (
    PicoVRSource,
    make_pico_source,
    process_smpl_pose_full,
)
from g1_bridge.pico.mode_manager import (
    ManagerMode,
    CalibrationEvent,
    ModeManager,
)
from g1_bridge.vendored.wbt.utils.bydmimic_utils import (
    isaaclab_to_mujoco_reindex as _ISAACLAB_TO_MUJOCO_LIST,
    mujoco_to_isaaclab_reindex as _MUJOCO_TO_ISAACLAB_LIST,
)


# wbt's 29-D MuJoCo joint order (matches whole_body_teleop/utils/
# bydmimic_utils.py). Verified byte-identical to g1_runtime's copy, which is
# now the single definition; kept as an alias for external importers.
_WBT_JOINT_NAMES = G1_JOINT_NAMES


# resolve_ckpt_path now lives in g1_runtime (imported above) so every G1
# composite resolves checkpoints identically -- relative paths land under the
# sim2real repo root; re-exported here because several modules import it from
# this one.


# Cache of heavy ONNX-session-backed objects (SONIC controllers + kPlanner),
# keyed by (class/path) so they are built ONCE per process and REUSED across
# env.reset(). robosuite recreates the composite controller on every reset; if
# each reset built fresh ONNX CUDA sessions, the old sessions' cuDNN handles /
# GPU memory leak until cudnnCreate fails (CUDNN_STATUS_INTERNAL_ERROR) after a
# few dozen episodes. These objects are sim-agnostic (state is fed in per step),
# so a single cached instance is safe; we just reset its internal buffers.
_WBT_ONNX_CACHE = {}

# wbt_controller_type -> SONIC controller class
_SONIC_CTRL_CLASSES = {
    "sonic": WBTControllerSonic,
    "sonic_planner": WBTControllerSonic,
    "sonic_teleop": WBTControllerSonicTeleop,
    "sonic_pose": WBTControllerSonicSmpl,
    # VLA deploy: same whole-body decoder as sonic_planner (mode_id=0), but fed an
    # externally-supplied motion token (from a GR00T policy) instead of the encoder.
    "sonic_vla": WBTControllerSonic,
    # VLA deploy (reference-motion model): the GR00T policy predicts the SONIC
    # encoder INPUT -- 24 SMPL joint positions -- instead of the token. We run the
    # FULL POSE encoder (mode_id=2) -> decoder here (set_pose_control_var + .step),
    # so the predicted reference motion is encoded and decoded on the fly.
    "sonic_pose_vla": WBTControllerSonicSmpl,
}


def _make_wbt_controller(ctrl_type: str, cfg: dict):
    if ctrl_type == "default_pos":
        return DefaultPosController()
    if ctrl_type == "heft_tracker":
        # HEFT's frozen tracker, run through onnxruntime CPU. ⚠️ Imported LAZILY,
        # and that is a layering rule rather than a style choice: the controller
        # lives in `rl/kimodo/heft/` (it reads that package's observation builder
        # and gain tables), so a module-level import here would make robocasa
        # depend on rl -- which depends on robocasa. Nothing outside this branch
        # touches rl, so every other consumer imports this module unchanged.
        from rl.kimodo.heft.tracker_controller import HeftTrackerController

        onnx = cfg.get("heft_policy_onnx") or None
        key = ("heft_tracker", str(onnx))
        ctrl = _WBT_ONNX_CACHE.get(key)
        if ctrl is None:
            ctrl = HeftTrackerController(onnx_path=onnx)
            _WBT_ONNX_CACHE[key] = ctrl
        else:
            ctrl.reset()  # reuse the session, drop per-episode state
        return ctrl
    if ctrl_type in _SONIC_CTRL_CLASSES:
        enc = resolve_ckpt_path(cfg["sonic_encoder_path"])
        dec = resolve_ckpt_path(cfg["sonic_decoder_path"])
        if not (os.path.exists(enc) and os.path.exists(dec)):
            raise FileNotFoundError(f"Sonic ONNX models not found: {enc}, {dec}")
        cls = _SONIC_CTRL_CLASSES[ctrl_type]
        key = ("sonic_ctrl", cls.__name__, enc, dec)
        ctrl = _WBT_ONNX_CACHE.get(key)
        if ctrl is None:
            ctrl = cls(encoder_path=enc, decoder_path=dec, use_trt=False)
            _WBT_ONNX_CACHE[key] = ctrl
        elif hasattr(ctrl, "reset"):
            ctrl.reset()  # reuse the ONNX sessions; clear per-episode state
        return ctrl
    if ctrl_type in ("wbt", "gentle"):
        raise NotImplementedError(
            f"wbt_controller_type='{ctrl_type}' not wired up yet."
        )
    raise ValueError(f"Unknown wbt_controller_type: {ctrl_type!r}")


# Reindex maps from wbt's bydmimic_utils — converted to ndarrays once at import.
_MUJOCO_TO_ISAACLAB = np.asarray(_MUJOCO_TO_ISAACLAB_LIST, dtype=np.int64)
_ISAACLAB_TO_MUJOCO = np.asarray(_ISAACLAB_TO_MUJOCO_LIST, dtype=np.int64)


def _ensure_reindex_maps():
    """No-op kept for backwards source compatibility — maps are loaded at import."""
    return


class _RobotShim:
    """Adapter so wbt controllers can call ``robot.get_state()``."""

    def __init__(self, state: Dict[str, np.ndarray]):
        self._state = state

    def get_state(self):
        return self._state


# The quaternion helpers now live in g1_runtime (imported above). Kept under
# their original private names so this module's own call sites -- and the
# handful of modules that import them from here -- keep working.
_quat_mul = quat_mul
_quat_inv = quat_inv
_relative_anchor6 = relative_anchor6


@register_composite_controller
class WholeBodyG1Composite(G1CompositeBase):
    name = "WHOLE_BODY_G1"

    # Sonic / Gentle / WBT policies run at 50 Hz (dt = 0.02 s) in wbt's
    # deployment.  robosuite's run_controller is invoked once per sim sub-step
    # (~500 Hz), so we throttle the policy call to ~50 Hz and reuse the cached
    # (target_q, kp, kd) for in-between PD updates.  default_pos is cheap and
    # stateless enough to run every step.
    _POLICY_DT = 0.02
    # SONIC kPlanner runs on a dedicated 10 Hz thread in the C++ deploy; we
    # throttle inline at the same rate.
    _PLANNER_DT = 0.1
    # Periodic replan timers per locomotion category (seconds) — mirror the
    # C++ values in g1_deploy_onnx_ref.cpp (replan_interval_running_ etc.).
    _REPLAN_INTERVAL_RUNNING = 0.1
    _REPLAN_INTERVAL_CRAWLING = 0.2
    _REPLAN_INTERVAL_BOXING = 1.0
    _REPLAN_INTERVAL_DEFAULT = 1.0

    def __init__(self, sim, robot_model, grippers):
        super().__init__(sim, robot_model, grippers)
        self._wbt_ctrl = None
        self._ctrl_type = None
        # Joint/actuator address maps and the root freejoint addresses are
        # declared and rebuilt by G1CompositeBase.
        # Cached policy output for sub-step PD updates.
        self._cached_target_q = None
        self._cached_kps = None
        self._cached_kds = None
        self._last_policy_time = -1e9
        # Anchor orientation (6D) captured at first policy tick (sonic mode only).
        self._anchor_ori = None
        # ----- sonic_planner state -----
        self._kplanner = None  # SonicKPlanner or None
        self._kplanner_path = None
        self._planner_initialized = False
        self._current_motion_frame = 0  # cursor into kplanner.planner_motion_50hz
        self._last_planner_check_time = -1e9
        self._last_replan_time = -1e9
        self._replan_interval_counter = 0.0
        self._movement_state = MovementState()  # current commanded MovementState
        self._last_movement_state = (
            MovementState()
        )  # last committed (for change detection)
        self._movement_state_dirty = False  # set by keyboard handler
        # Manually selected walking gait (default SLOW_WALK). The live
        # locomotion_mode follows this when the stick is pushed and drops to
        # IDLE when it is released (push-to-walk; no marching in place).
        self._selected_gait = int(LocomotionMode.SLOW_WALK)
        # Last sim time the PICO planner input tick ran (for ~50 Hz throttling so
        # the yaw accumulator doesn't advance once per sim substep).
        self._last_pico_tick_time = -1e9
        # ----- sonic_teleop state (mode_id=1, PICO VR upper-body) -----
        # The teleop mode reuses the kPlanner for lower-body reference AND
        # adds a PICO VR 3-point input for the upper body.
        self._pico_source: PicoVRSource | None = None
        # ----- sonic_manager state (Phase A+B: 5 modes) -----
        # Manager mode pre-loads up to THREE controllers and swaps based on
        # PICO button state.  Inactive controllers get fresh refs every tick
        # but aren't .step()'d.
        self._mode_manager: ModeManager | None = None
        self._planner_ctrl = None  # WBTControllerSonic instance (mode_id=0)
        self._teleop_ctrl = None  # WBTControllerSonicTeleop instance (mode_id=1)
        self._pose_ctrl = None  # WBTControllerSonicSmpl instance (mode_id=2)
        # Upper-body 17-joint snapshot for PLANNER_FROZEN_UPPER mode (IsaacLab
        # order, 29-D vector where lower-body slots stay as kPlanner output).
        self._upper_body_snapshot_isaac = None  # (29,) float32 or None
        # Gripper trigger state (0 = open, 1 = closed) per hand.  Updated from
        # PICO buttons in sonic_manager mode; the hands are PD-held to these
        # every control step (see run_controller) so the fingers don't flop when
        # not teleoperating.
        self._left_trigger = 0.0
        self._right_trigger = 0.0
        # VLA deploy (sonic_vla): the latest action from the GR00T policy, set via
        # set_vla_action(). The motion token feeds the decoder; the hand-joint
        # targets (7 per hand, actuator order) PD-drive the fingers.
        self._vla_token = None  # (64,) float32 or None
        self._vla_left_hand = None  # (7,) float32 or None
        self._vla_right_hand = None  # (7,) float32 or None
        # VLA deploy (sonic_pose_vla): the GR00T reference-motion policy predicts
        # the 24 SMPL joint positions (encoder input) instead of a token; pushed to
        # the POSE encoder via set_pose_control_var(). anchor/wrist are NOT
        # predicted -> left at the POSE controller's defaults (identity / zero),
        # which is a good approximation for a near-stationary, forward-facing task.
        self._vla_smpl_joints = None  # (24, 3) float32 or None
        # POSE-mode heading alignment.  PLANNER tracks the *commanded* facing
        # while POSE tracks the operator's *absolute* torso yaw; switching
        # between them would snap the robot's heading.  On each entry into POSE
        # we capture a world-frame offset (``_pose_root_align``) that maps the
        # operator's current torso orientation onto the robot's current heading,
        # so the orientation anchor is continuous across the switch.
        self._pose_root_align = None  # (4,) wxyz world-frame offset or None
        self._pose_align_pending = False  # set on POSE entry, captured next tick
        self._prev_manager_mode = ManagerMode.OFF  # for mode-entry edge detection
        # PLANNER steering state.  ``_yaw_angle_rad`` is the commanded heading
        # the right stick integrates; it is re-synced to the robot's ACTUAL
        # heading on entry into a planner-family mode (see _resync_facing_to_robot)
        # so switching back from POSE doesn't snap the robot to a stale facing
        # (i.e. turning behaves like angular-velocity control, not absolute
        # orientation control).
        self._yaw_angle_rad = 0.0
        self._prev_ab_mode = False
        self._prev_xy_mode = False
        # Hand actuator/joint mappings (set by _build_hand_mapping).
        self._hand_left_addrs = None
        self._hand_right_addrs = None

    # ------------------------------------------------------------------ #
    # Init flow: skip the parent's part-controller machinery; we write
    # torques directly in run_controller.
    # ------------------------------------------------------------------ #
    def load_controller_config(
        self, part_controller_config, composite_controller_specific_config=None
    ):
        self.composite_controller_specific_config = (
            composite_controller_specific_config or {}
        )
        self._init_robosuite_state(part_controller_config)

        self._build_mappings()
        self._build_hand_mapping()
        self._ctrl_type = self.composite_controller_specific_config.get(
            "wbt_controller_type", "default_pos"
        )
        if self._ctrl_type == "sonic_manager":
            # Pre-load all SONIC controllers (5-mode coverage).  The active
            # one is chosen by ModeManager based on PICO button input.
            cfg = self.composite_controller_specific_config
            self._planner_ctrl = _make_wbt_controller("sonic_planner", cfg)
            self._teleop_ctrl = _make_wbt_controller("sonic_teleop", cfg)
            self._pose_ctrl = _make_wbt_controller("sonic_pose", cfg)
            self._wbt_ctrl = self._planner_ctrl  # default — will be swapped
            self._init_kplanner(cfg)
            self._init_pico_source(cfg)
            self._mode_manager = ModeManager(
                initial_mode=ManagerMode.OFF,
                verbose=bool(cfg.get("mode_manager_verbose", True)),
            )
            print(
                "[composite] sonic_manager mode armed (5 modes: OFF/PLANNER/VR_3PT/POSE/FROZEN)"
            )
            print(
                "[composite]   A+B+X+Y=engage/stop  A+X=PLANNER<->POSE  B+Y=POSE<->FROZEN_UPPER  L-Click=PLANNER<->VR_3PT"
            )
        else:
            self._wbt_ctrl = _make_wbt_controller(
                self._ctrl_type, self.composite_controller_specific_config
            )
            if self._ctrl_type == "sonic_planner":
                self._init_kplanner(self.composite_controller_specific_config)
            elif self._ctrl_type == "sonic_teleop":
                # Teleop mode needs both the kPlanner (for lower-body reference)
                # AND a PICO source (for upper-body VR 3-point).
                self._init_kplanner(self.composite_controller_specific_config)
                self._init_pico_source(self.composite_controller_specific_config)

    # _build_mappings (G1CompositeBase) builds the joint/actuator and freejoint
    # address maps; JOINT_NAMES below is what it maps.

    # The no-env-action stubs (setup_action_split_idx / set_goal /
    # update_state / action_limits / create_action_vector /
    # create_action_dict_from_action_vector) come from G1CompositeBase.

    def reset(self):
        if self._wbt_ctrl is not None and hasattr(self._wbt_ctrl, "reset"):
            self._wbt_ctrl.reset()
        # Reset the other pre-loaded controllers too when in manager mode.
        for ctrl in (self._planner_ctrl, self._teleop_ctrl, self._pose_ctrl):
            if (
                ctrl is not None
                and ctrl is not self._wbt_ctrl
                and hasattr(ctrl, "reset")
            ):
                ctrl.reset()
        self._cached_target_q = None
        self._cached_kps = None
        self._cached_kds = None
        self._last_policy_time = -1e9
        self._anchor_ori = None
        self._pose_root_align = None
        self._pose_align_pending = False
        self._prev_manager_mode = ManagerMode.OFF
        self._yaw_angle_rad = 0.0
        self._prev_ab_mode = False
        self._prev_xy_mode = False
        self._selected_gait = int(LocomotionMode.SLOW_WALK)
        self._last_pico_tick_time = -1e9
        # Force kPlanner re-initialisation on the next tick.
        self._planner_initialized = False
        self._current_motion_frame = 0
        self._last_planner_check_time = -1e9
        self._last_replan_time = -1e9
        self._replan_interval_counter = 0.0

    # ------------------------------------------------------------------ #
    # Reference motion: feed default-pose hold to Sonic.  Called once per
    # episode (after reset()).  For sonic, the reference is constant so we
    # just push it in lazily on first run_controller call.
    # ------------------------------------------------------------------ #
    @staticmethod
    def _quat_to_anchor6(quat_wxyz):
        """First two columns of the rotation matrix, flattened ROW-MAJOR to a
        6D vector [r00, r01, r10, r11, r20, r21] — matches the convention used
        by SONIC C++ (``GatherMotionAnchorOrientationMutiFrame``) and wbt's
        ``RefMotionNodeSonic.get_wbt_ctrl_vars`` (``rot_mat[:, :2].reshape(-1)``).
        """
        w, x, y, z = quat_wxyz
        r00 = 1 - 2 * (y * y + z * z)
        r10 = 2 * (x * y + w * z)
        r20 = 2 * (x * z - w * y)
        r01 = 2 * (x * y - w * z)
        r11 = 1 - 2 * (x * x + z * z)
        r21 = 2 * (y * z + w * x)
        return np.array([r00, r01, r10, r11, r20, r21], dtype=np.float32)

    def _update_default_pose_reference(self, ctrl=None):
        """Push a 'hold default pose' reference.  Sonic uses set_control_var
        once per policy tick to refresh the encoder buffers.  Anchor frame is
        captured at first call (matches wbt's yaw-alignment design: reference
        frame == robot pose at handshake time).

        Args:
            ctrl: target controller.  Defaults to ``self._wbt_ctrl``.  In
                sonic_manager OFF mode we pass ``self._planner_ctrl`` so the
                planner controller is the one being fed the default reference.
        """
        if self._ctrl_type not in ("sonic", "sonic_planner", "sonic_manager"):
            return
        target = ctrl if ctrl is not None else self._wbt_ctrl
        if target is None or not hasattr(target, "set_control_var"):
            return
        sonic_default_isaac = SONIC_DEFAULT_ANGLES.astype(np.float32)
        if self._anchor_ori is None:
            # The default standing-pose reference (SONIC_DEFAULT_ANGLES +
            # root_z) is robot-local — it does not depend on world-frame yaw.
            # Use an identity anchor so Sonic interprets the reference in the
            # robot's own frame.  A non-identity anchor (e.g. from a yaw≠0
            # spawn) can confuse the encoder, which was trained with
            # identity-anchored standing references.
            self._anchor_ori = np.array(
                [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32
            )
        ref = {
            "joint_pos_29": sonic_default_isaac,
            "joint_vel_29": np.zeros(29, dtype=np.float32),
            "motion_anchor_orientation": self._anchor_ori,
            "root_z": 0.793,
        }
        target.set_control_var(ref)

    # ------------------------------------------------------------------ #
    # sonic_planner mode: drive Sonic from kPlanner-generated motion.
    # ------------------------------------------------------------------ #
    def _init_kplanner(self, cfg):
        planner_path = resolve_ckpt_path(cfg.get("planner_path"))
        if not planner_path:
            raise ValueError(
                "wbt_controller_type='sonic_planner' requires "
                "composite_controller_specific_configs.planner_path"
            )
        self._kplanner_path = planner_path
        default_height = float(cfg.get("planner_default_height", 0.788740))
        seed = int(cfg.get("planner_seed", 1234))
        # Reuse the ONNX-backed planner across resets (see _WBT_ONNX_CACHE): it
        # loads planner_sonic.onnx, so rebuilding it every env.reset() leaks GPU
        # sessions too. The composite re-inits the planner from the current robot
        # pose each episode (self._planner_initialized), so a cached instance is
        # safe.
        key = ("kplanner", planner_path, default_height, seed)
        kp = _WBT_ONNX_CACHE.get(key)
        if kp is None:
            kp = SonicKPlanner(
                planner_path=planner_path,
                default_height=default_height,
                seed=seed,
            )
            _WBT_ONNX_CACHE[key] = kp
        elif hasattr(kp, "reset"):
            kp.reset()
        self._kplanner = kp
        # Seed the commanded MovementState from the config (optional).
        ms_cfg = cfg.get("initial_movement_state", {}) or {}
        self._movement_state = MovementState(
            locomotion_mode=int(ms_cfg.get("locomotion_mode", LocomotionMode.IDLE)),
            movement_direction=np.array(
                ms_cfg.get("movement_direction", [0.0, 0.0, 0.0]), dtype=np.float32
            ),
            facing_direction=np.array(
                ms_cfg.get("facing_direction", [1.0, 0.0, 0.0]), dtype=np.float32
            ),
            movement_speed=float(ms_cfg.get("movement_speed", -1.0)),
            height=float(ms_cfg.get("height", -1.0)),
        )
        self._last_movement_state = self._movement_state.copy()
        self._movement_state_dirty = False

    def _init_pico_source(self, cfg):
        """Construct + start the PICO VR data source for sonic_teleop mode."""
        source_type = cfg.get("pico_source", "mock")
        source_kwargs = cfg.get("pico_source_kwargs", {}) or {}
        try:
            self._pico_source = make_pico_source(source_type, **source_kwargs)
            self._pico_source.start()
            print(f"[composite] PICO VR source: {source_type}")
        except Exception as e:
            if source_type == "real":
                raise RuntimeError(
                    "PICO source 'real' failed to start; refusing to fall back "
                    "to MockPicoVRSource because real hardware was explicitly "
                    "requested."
                ) from e
            # Fall back to mock so the env still loads — but log loudly.
            print(
                f"[composite] Failed to start PICO source {source_type!r}: {e}\n"
                "             Falling back to MockPicoVRSource."
            )
            self._pico_source = make_pico_source("mock")
            self._pico_source.start()

    def _replan_interval_for_mode(self, mode):
        if mode == LocomotionMode.RUN:
            return self._REPLAN_INTERVAL_RUNNING
        if mode in (LocomotionMode.CRAWLING, LocomotionMode.ELBOW_CRAWLING):
            return self._REPLAN_INTERVAL_CRAWLING
        if mode in (
            LocomotionMode.LEFT_PUNCH,
            LocomotionMode.RIGHT_PUNCH,
            LocomotionMode.RANDOM_PUNCH,
            LocomotionMode.LEFT_HOOK,
            LocomotionMode.RIGHT_HOOK,
        ):
            return self._REPLAN_INTERVAL_BOXING
        return self._REPLAN_INTERVAL_DEFAULT

    def _lazy_init_planner_if_needed(self):
        """Initialise the kPlanner from the current robot pose on first tick."""
        if self._planner_initialized or self._kplanner is None:
            return
        qpos, _ = self._read_joint_state()
        quat, _ = self._read_pelvis_state()
        # Derive spawn yaw from the actual pelvis quaternion and sync the
        # composite controller's facing_direction BEFORE kPlanner.initialize()
        # so both sides agree on the robot's heading from the very first tick.
        spawn_yaw = heading_yaw_from_quat(quat)
        self._yaw_angle_rad = spawn_yaw
        self._movement_state.facing_direction = np.array(
            [np.cos(spawn_yaw), np.sin(spawn_yaw), 0.0], dtype=np.float32
        )
        ok = self._kplanner.initialize(
            base_quat_wxyz=quat,
            joint_positions_mujoco=qpos,
            mode=self._movement_state.locomotion_mode,
        )
        if ok:
            self._planner_initialized = True
            self._current_motion_frame = 0
            self._last_movement_state = self._movement_state.copy()
            self._movement_state_dirty = False
            self._replan_interval_counter = 0.0
            self._last_replan_time = float(self.sim.data.time)
            n = self._kplanner.num_frames
            print(
                f"[kPlanner] initialised — mode={self._movement_state.locomotion_mode} "
                f"({n} frames @ 50Hz, {n / 50.0:.2f}s)"
            )

    def _tick_planner(self, sim_time):
        """Run the 10Hz kPlanner check: replan if needed, advance frame cursor."""
        if not self._planner_initialized or self._kplanner is None:
            return
        if sim_time - self._last_planner_check_time < self._PLANNER_DT - 1e-6:
            return
        dt = sim_time - self._last_planner_check_time
        self._last_planner_check_time = sim_time
        if dt > 0:
            self._replan_interval_counter += dt

        cur = self._movement_state
        last = self._last_movement_state
        mode_changed = cur.locomotion_mode != last.locomotion_mode
        facing_changed = not np.allclose(
            cur.facing_direction, last.facing_direction, atol=1e-6
        )
        height_changed = float(cur.height) != float(last.height)
        speed_changed = float(cur.movement_speed) != float(last.movement_speed)
        dir_changed = not np.allclose(
            cur.movement_direction, last.movement_direction, atol=1e-6
        )

        is_static = cur.locomotion_mode in STATIC_MODES
        interval = self._replan_interval_for_mode(cur.locomotion_mode)
        time_to_replan = self._replan_interval_counter >= interval

        need_replan = False
        if mode_changed or facing_changed or height_changed:
            need_replan = True
        elif not is_static and (
            speed_changed
            or dir_changed
            or (time_to_replan and float(cur.movement_speed) != 0.0)
        ):
            need_replan = True

        # If the cursor is about to fall off the end of the current motion,
        # force a replan so we never starve the controller.
        if not need_replan and self._kplanner.num_frames > 0:
            remaining = self._kplanner.num_frames - self._current_motion_frame
            if remaining <= 10:
                need_replan = True

        if not need_replan:
            return

        ok = self._kplanner.update_planning(
            current_frame=self._current_motion_frame,
            movement_state=cur,
        )
        if ok:
            self._current_motion_frame = 0
            self._last_movement_state = cur.copy()
            self._movement_state_dirty = False
            self._replan_interval_counter = 0.0
            self._last_replan_time = sim_time

    def _update_planner_reference(self, ctrl=None):
        """Push a reference to Sonic via set_control_var.

        IDLE (and any other static mode whose planner output drifts) is
        special-cased: we just hold the SONIC default standing pose with zero
        joint velocity, which matches what the C++ deploy's ``idle_readapt``
        machinery achieves and avoids the planner's intrinsic per-inference
        drift (waist_pitch grows ~0.14 rad over a single 1.5 s IDLE clip;
        with naive replan from motion-end the drift compounds).

        Non-static modes use the planner frame at the current cursor with
        finite-diff joint velocities.

        Args:
            ctrl: target controller.  Defaults to ``self._wbt_ctrl``.  In
                sonic_manager VR_3PT mode we pass ``self._planner_ctrl`` to
                keep the inactive controller's buffers fresh.
        """
        target = ctrl if ctrl is not None else self._wbt_ctrl
        if target is None or not hasattr(target, "set_control_var"):
            return
        if self._kplanner is None or self._kplanner.planner_motion_50hz is None:
            return

        mode = self._movement_state.locomotion_mode
        if mode == LocomotionMode.IDLE:
            # Hold canonical standing pose — no drift, no waist sway.
            self._update_default_pose_reference(ctrl=target)
            return

        frame_idx = self._current_motion_frame
        frame_qpos = self._kplanner.sample_frame(frame_idx)  # (36,) MuJoCo qpos
        frame_dq = self._kplanner.sample_joint_velocity(frame_idx)  # (29,) MuJoCo order

        _ensure_reindex_maps()
        joint_pos_mujoco = frame_qpos[7:].astype(np.float32, copy=False)
        joint_pos_isaac = joint_pos_mujoco[_MUJOCO_TO_ISAACLAB]
        joint_vel_isaac = frame_dq[_MUJOCO_TO_ISAACLAB]
        # Static non-IDLE modes (SQUAT/KNEEL/...) still benefit from zeroing
        # the velocity to behave as "hold the target pose".
        if mode in STATIC_MODES:
            joint_vel_isaac = np.zeros_like(joint_vel_isaac)

        # Anchor orientation = relative rotation from current robot pose to
        # the reference root.  Matches RefMotionNodeSonic.get_wbt_ctrl_vars.
        ref_quat = frame_qpos[3:7].astype(np.float32, copy=False)
        robot_quat, _ = self._read_pelvis_state()
        anchor6 = _relative_anchor6(robot_quat, ref_quat)

        # PLANNER_FROZEN_UPPER: overlay the snapshot upper-body 17 joints on
        # top of the kPlanner motion frame.  Lower body still walks per
        # kPlanner, but waist + arms + wrists stay frozen at the snapshot.
        if (
            self._ctrl_type == "sonic_manager"
            and self._mode_manager is not None
            and self._mode_manager.current == ManagerMode.PLANNER_FROZEN_UPPER
            and self._upper_body_snapshot_isaac is not None
        ):
            joint_pos_isaac = joint_pos_isaac.copy()
            joint_pos_isaac[
                self._UPPER_BODY_ISAAC_INDICES
            ] = self._upper_body_snapshot_isaac[self._UPPER_BODY_ISAAC_INDICES]
            # Zero out upper-body velocities so the policy sees "frozen".
            joint_vel_isaac = joint_vel_isaac.copy()
            joint_vel_isaac[self._UPPER_BODY_ISAAC_INDICES] = 0.0

        ctrl_vars = {
            "joint_pos_29": joint_pos_isaac,
            "joint_vel_29": joint_vel_isaac,
            "motion_anchor_orientation": anchor6,
            "root_z": float(frame_qpos[2]),
        }
        target.set_control_var(ctrl_vars)

    # ------------------------------------------------------------------ #
    # sonic_teleop mode: PICO VR upper-body + kPlanner lower-body
    # ------------------------------------------------------------------ #
    def _update_teleop_reference(self, ctrl=None):
        """Push the latest reference frame to the SONIC teleop controller.

        Args:
            ctrl: target controller (must support ``set_lowerbody_control_var``
                and ``set_vr_3point``).  Defaults to ``self._wbt_ctrl``.

        The teleop encoder (mode_id=1) wants:
          - lower-body planner reference (12 leg joints × 10 frames)
          - VR 3-point upper-body target (positions + quaternions)
          - motion_anchor_orientation
        """
        target = ctrl if ctrl is not None else self._wbt_ctrl
        if target is None or not hasattr(target, "set_lowerbody_control_var"):
            return
        if self._kplanner is None or self._kplanner.planner_motion_50hz is None:
            return
        if self._pico_source is None:
            return

        # ---- Lower-body reference from kPlanner ----
        mode = self._movement_state.locomotion_mode
        frame_idx = self._current_motion_frame
        frame_qpos = self._kplanner.sample_frame(frame_idx)  # (36,) MuJoCo qpos
        frame_dq = self._kplanner.sample_joint_velocity(frame_idx)  # (29,) MuJoCo order

        joint_pos_mujoco = frame_qpos[7:].astype(np.float32, copy=False)
        joint_pos_isaac = joint_pos_mujoco[_MUJOCO_TO_ISAACLAB]
        joint_vel_isaac = frame_dq[_MUJOCO_TO_ISAACLAB]
        if mode in STATIC_MODES:
            joint_vel_isaac = np.zeros_like(joint_vel_isaac)

        # ---- Anchor orientation: planner ref quat vs robot pelvis quat ----
        ref_quat = frame_qpos[3:7].astype(np.float32, copy=False)
        robot_quat, _ = self._read_pelvis_state()
        anchor6 = _relative_anchor6(robot_quat, ref_quat)

        target.set_lowerbody_control_var(
            {
                "joint_pos_29": joint_pos_isaac,
                "joint_vel_29": joint_vel_isaac,
                "motion_anchor_orientation": anchor6,
            }
        )

        # ---- VR 3-point upper-body target from PICO ----
        sample = self._pico_source.get_latest()
        if sample is not None:
            target.set_vr_3point(
                sample["vr_3point_local_target"],
                sample["vr_3point_local_orn_target"],
            )

    # ------------------------------------------------------------------ #
    # POSE mode (mode_id=2): full-body SMPL teleop
    # ------------------------------------------------------------------ #
    def _update_pose_reference(self, ctrl=None, compute_from_raw=True):
        """Push the latest full-body SMPL data from PICO to the POSE
        controller (mode_id=2 encoder).

        Args:
            ctrl: target controller (must implement ``set_pose_control_var``).
                Defaults to ``self._wbt_ctrl``.
        """
        target = ctrl if ctrl is not None else self._wbt_ctrl
        if target is None or not hasattr(target, "set_pose_control_var"):
            return
        if self._pico_source is None:
            return
        sample = self._pico_source.get_latest()
        if sample is None:
            return

        # The encoder's smpl_anchor_orientation is NOT the raw SMPL root 6D.
        # The deploy (GatherMotionAnchorOrientationMutiFrame, orientation_mode=0)
        # computes it live as the 6D of  conj(robot_base_quat) * smpl_root_quat
        # — i.e. the reference root expressed relative to the robot's current
        # base.  Mirror that here using the same _relative_anchor6 helper the
        # working PLANNER path uses.
        anchor6 = sample.get("pose_anchor_6d")
        smpl_root_quat = sample.get("pose_smpl_root_quat")
        smpl_joints_local = sample.get("pose_smpl_joints_local")
        wrist_joints = sample.get("pose_wrist_joints")
        if (
            smpl_joints_local is None
            and compute_from_raw
            and sample.get("raw_smpl") is not None
        ):
            pose_data = process_smpl_pose_full(sample["raw_smpl"])
            smpl_joints_local = pose_data["smpl_joints_local"]
            smpl_root_quat = pose_data["smpl_root_quat"]
            anchor6 = pose_data["anchor_6d"]
            wrist_joints = pose_data["wrist_joints"]
        if smpl_joints_local is None or anchor6 is None or wrist_joints is None:
            return

        if smpl_root_quat is not None:
            robot_quat, _ = self._read_pelvis_state()
            # Capture the heading-alignment offset on the first POSE tick after
            # entry: a world-frame rotation R with  R * smpl_root = robot_quat,
            # so the anchor is identity at the switch instant (no sudden turn).
            # Subsequent torso rotation then drives heading relative to that
            # captured zero.  Only captured on the *active* POSE path (the
            # buffer-warming calls from other modes pass ctrl=self._pose_ctrl).
            if self._pose_align_pending and ctrl is None:
                self._pose_root_align = _quat_mul(robot_quat, _quat_inv(smpl_root_quat))
                self._pose_align_pending = False
                print("[composite] POSE heading aligned to current robot facing")
            if self._pose_root_align is not None:
                smpl_root_quat = _quat_mul(self._pose_root_align, smpl_root_quat)
            anchor6 = _relative_anchor6(robot_quat, smpl_root_quat)

        target.set_pose_control_var(
            {
                "smpl_joints_local": smpl_joints_local,
                "anchor_6d": anchor6,
                "wrist_joints": wrist_joints,
            }
        )

    # ------------------------------------------------------------------ #
    # Data collection: expose the live teleop reference + low-level target.
    # ------------------------------------------------------------------ #
    def get_teleop_data_sample(self) -> dict:
        """Snapshot the current teleop command for data collection.

        Returns a dict of float32 arrays describing what drove the robot this
        step — the low-level 29-joint PD target the controller actually
        commanded, the manager/locomotion mode, and (when a pico-* mode is
        engaged) the upper-body teleop reference from the latest PICO sample
        (VR 3-point + full-body SMPL + wrist joints). Fields that are
        unavailable (e.g. before engage, or in keyboard mode) are returned as
        None so the recorder can decide how to handle them. Mirrors the GR00T
        Sonic-VLA ``action`` schema as far as the sim exposes it.
        """

        def _arr(v):
            return None if v is None else np.asarray(v, dtype=np.float32).reshape(-1)

        out = {
            # Low-level 29-joint PD target actually commanded (IsaacLab order).
            "target_q_29": _arr(self._cached_target_q),
            "locomotion_mode": np.array(
                [int(self._movement_state.locomotion_mode)], dtype=np.float32
            ),
            "facing_direction": _arr(self._movement_state.facing_direction),
        }
        mode = self._mode_manager.current if self._mode_manager is not None else None
        out["stream_mode"] = np.array(
            [int(mode) if mode is not None else -1], dtype=np.float32
        )
        # Upper-body teleop reference from the latest PICO sample (if any).
        sample = (
            self._pico_source.get_latest() if self._pico_source is not None else None
        )
        out["vr_3point_pos"] = _arr(
            sample.get("vr_3point_local_target") if sample else None
        )
        out["vr_3point_orn"] = _arr(
            sample.get("vr_3point_local_orn_target") if sample else None
        )
        out["smpl_joints_local"] = _arr(
            sample.get("pose_smpl_joints_local") if sample else None
        )  # (24,3) -> 72
        out["smpl_root_quat"] = _arr(
            sample.get("pose_smpl_root_quat") if sample else None
        )
        out["wrist_joints"] = _arr(sample.get("pose_wrist_joints") if sample else None)
        # --- GR00T unitree_g1_sonic action fields ---
        # motion_token (64D): the SONIC encoder latent currently driving the
        # whole-body decoder (== proprio["token_state"] in gear_sonic). Read from
        # the active inner controller (planner/teleop/pose); zeros before the
        # first policy tick.
        out["motion_token"] = _arr(getattr(self._wbt_ctrl, "token_state", None))
        # left/right hand joints (7D each): the live finger joint positions, in
        # the actuator order (thumb_0,1,2, index_0,1, middle_0,1) that
        # _write_gripper_torques drives -- so collection and any in-sim deploy
        # apply them consistently.
        def _hand_qpos(addrs):
            if addrs is None:
                return None
            return np.asarray(
                [
                    float(self.sim.data.qpos[a]) if a is not None else 0.0
                    for a in addrs["qpos"]
                ],
                dtype=np.float32,
            )

        out["left_hand_joints"] = _hand_qpos(self._hand_left_addrs)
        out["right_hand_joints"] = _hand_qpos(self._hand_right_addrs)
        # schema v6 readiness key (g1_bridge SONIC_TELEOP_KEYS[0]): the 43-wide
        # action.wbc = body target + the hand values above in the state's
        # finger slots. Function-local import: robocasa imports this module
        # unconditionally and must not require g1_backend for non-G1 use.
        if self._cached_target_q is not None:
            from g1_bridge.recording.frames import sonic_action_43

            out["target_q_43"] = sonic_action_43(
                self._cached_target_q, out["left_hand_joints"], out["right_hand_joints"]
            ).astype(np.float32)
        else:
            out["target_q_43"] = None
        return out

    # ------------------------------------------------------------------ #
    # heft_tracker seam
    # ------------------------------------------------------------------ #
    # ⚠️ Why the Kimodo rollout goes through the composite at all, when
    # `rl/kimodo/heft/heft_driver.py` can drive the tracker directly: the driver
    # bypasses `env.step()`, and `_post_action` hold counters, `_check_success`,
    # camera observations and `EpisodeRecorder` all live on that path.
    # `record_step` reads `composite.get_teleop_data_sample()["target_q_29"]`;
    # with no composite in the loop that is None, `build_frame` returns None, and
    # every frame is silently dropped -- an empty dataset with no error.
    def set_heft_reference(self, ref_qpos36, reboot: bool = True) -> None:
        """Push a reference chunk to the HEFT tracker. Peer of set_vla_action.

        ``ref_qpos36`` is [T, 36] at 50 Hz in dataset joint order -- what
        ``rl.kimodo.heft_bridge`` produces from a Kimodo chunk.

        ``reboot`` re-arms the boot indicator, which tells the tracker a new
        command has started. Mid-episode continuation should pass False: the
        chunk was generated from the robot's realized pose, so announcing a
        reboot at every boundary is one of the three stacked causes of the
        documented chunk-boundary stall.

        NOTE: ``env.reset()`` rebuilds the composite, so re-fetch
        ``env.robots[0].composite_controller`` afterwards and set the reference
        again -- a cached handle silently drives nothing.
        """
        if self._ctrl_type != "heft_tracker":
            raise RuntimeError(
                f"set_heft_reference requires wbt_controller_type='heft_tracker', "
                f"got {self._ctrl_type!r}"
            )
        self._wbt_ctrl.set_reference(ref_qpos36, reboot=reboot)

    @property
    def heft_reference_done(self) -> bool:
        """True once the tracker has consumed the last frame of its chunk."""
        return self._ctrl_type == "heft_tracker" and self._wbt_ctrl.reference_done

    def heft_tick_count(self) -> int:
        """Policy ticks since reset. Assert 1:10 against physics substeps."""
        if self._ctrl_type != "heft_tracker":
            return -1
        return self._wbt_ctrl.n_ticks

    def set_heft_replay(self, targets_ds) -> None:
        """Replay a recorded action.wbc stream through the tracker's seat.

        An exported episode's actions are only *shown* to be sufficient to
        reproduce it by driving with them.
        """
        if self._ctrl_type != "heft_tracker":
            raise RuntimeError(
                f"set_heft_replay requires wbt_controller_type='heft_tracker', "
                f"got {self._ctrl_type!r}"
            )
        self._wbt_ctrl.set_replay(targets_ds)

    def heft_ticks_without_reference(self) -> int:
        """Ticks commanded before any reference arrived. Should be 0.

        Each one is a step in which the robot was falling: HEFT's gains do not
        hold the default pose statically (0.72 m of collapse over 200 ticks).
        A harness that warms up the sim before loading a chunk starts tracking
        from a heap on the floor, and the metrics read as bad tracking.
        """
        if self._ctrl_type != "heft_tracker":
            return -1
        return self._wbt_ctrl.n_ticks_no_ref

    # ------------------------------------------------------------------ #
    # VLA deploy: inject a GR00T policy action (sonic_vla mode).
    # ------------------------------------------------------------------ #
    def set_vla_action(
        self,
        motion_token=None,
        left_hand_joints=None,
        right_hand_joints=None,
        smpl_joints=None,
    ):
        """Set the latest GR00T policy action for sonic_vla / sonic_pose_vla mode.

        ``sonic_vla`` (token model): ``motion_token`` (64,) is fed to the SONIC
        decoder (replacing the encoder) to drive the 29-joint whole body.

        ``sonic_pose_vla`` (reference-motion model): ``smpl_joints`` (72, == 24×3
        SMPL joint positions, the POSE-encoder INPUT) is pushed to the POSE encoder
        and the full encoder->decoder is run (see _update_vla_pose_reference).

        ``{left,right}_hand_joints`` (7 each, actuator order) PD-drive the fingers
        in both modes. Call once per new policy chunk step (the composite re-decodes
        at its 50 Hz policy rate using the held action).
        """
        if motion_token is not None:
            self._vla_token = np.asarray(motion_token, dtype=np.float32).reshape(-1)
        if smpl_joints is not None:
            self._vla_smpl_joints = np.asarray(smpl_joints, dtype=np.float32).reshape(
                -1, 3
            )  # (24, 3)
        if left_hand_joints is not None:
            self._vla_left_hand = np.asarray(
                left_hand_joints, dtype=np.float32
            ).reshape(-1)
        if right_hand_joints is not None:
            self._vla_right_hand = np.asarray(
                right_hand_joints, dtype=np.float32
            ).reshape(-1)

    def _update_vla_pose_reference(self):
        """Push the VLA-predicted SMPL joints to the POSE encoder (sonic_pose_vla).

        Only ``smpl_joints_local`` is supplied; ``anchor_6d`` (root orientation)
        and ``wrist_joints`` are left at WBTControllerSonicSmpl's defaults (identity
        6D / zeros) because the reference-motion model does not predict them. For a
        near-stationary, forward-facing task this is a close approximation; if it
        proves too lossy, fill ``wrist_joints`` from the robot's current wrist
        joints here for a closed-loop proxy.
        """
        if self._vla_smpl_joints is None:
            return
        if not hasattr(self._wbt_ctrl, "set_pose_control_var"):
            return
        self._wbt_ctrl.set_pose_control_var(
            {"smpl_joints_local": self._vla_smpl_joints}
        )

    # ------------------------------------------------------------------ #
    # PLANNER_FROZEN_UPPER: kPlanner drives the lower body, the operator's
    # upper-body pose at the moment of mode entry is frozen and overlaid
    # on each kPlanner motion frame before it's pushed to the encoder.
    # ------------------------------------------------------------------ #
    # IsaacLab indices of the 17 non-leg joints (waist + arms + wrists).
    # Derived as: set(range(29)) − set(lower_body_indices [0,3,6,9,13,17,1,4,7,10,14,18]).
    _UPPER_BODY_ISAAC_INDICES = np.array(
        [2, 5, 8, 11, 12, 15, 16, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
        dtype=np.int64,
    )

    def _capture_upper_body_snapshot(self) -> None:
        """Snapshot the robot's current upper-body 17 joints (IsaacLab order)
        into ``_upper_body_snapshot_isaac``.  Called once on entry into
        PLANNER_FROZEN_UPPER mode.
        """
        qpos, _ = self._read_joint_state()  # MuJoCo order, 29
        qpos_isaac = qpos[_MUJOCO_TO_ISAACLAB]  # IsaacLab order, 29
        self._upper_body_snapshot_isaac = qpos_isaac.astype(np.float32, copy=True)
        ub = self._upper_body_snapshot_isaac[self._UPPER_BODY_ISAAC_INDICES]
        print(
            f"[composite] upper-body snapshot captured (17 joints, "
            f"|q|={np.linalg.norm(ub):.3f} rad)"
        )

    # ------------------------------------------------------------------ #
    # Keyboard handler — invoked by the MuJoCo viewer's key_callback.
    # All updates set self._movement_state and flip _movement_state_dirty;
    # _tick_planner picks them up on its next 10 Hz tick.
    # ------------------------------------------------------------------ #
    _MOV_SPEED_STEP = 0.5  # m/s per WASD press (resolved by the model)
    _FACING_YAW_STEP = 0.2618  # 15 degrees per Q/E press
    _VEL_STEP = 0.2  # m/s per [/] press
    _HEIGHT_STEP = 0.05  # m per ;/' press

    # GLFW arrow-key codes (mujoco.viewer reports these through key_callback).
    _GLFW_KEY_RIGHT = 262
    _GLFW_KEY_LEFT = 263
    _GLFW_KEY_DOWN = 264
    _GLFW_KEY_UP = 265

    # NOTE on key choice: MuJoCo's passive viewer wires its own GLFW key
    # handlers in libmujoco and we have no Python hook to suppress them, so
    # our callback runs *in addition* to theirs.  We avoid the destructive
    # built-ins (Space=pause, Backspace=reset, Tab=cycle camera, Esc=quit)
    # by remapping to keys whose built-in actions are only visual toggles
    # (e.g. W=wireframe) — annoying but harmless.  Arrow keys in passive
    # mode have no replay-scrub binding (passive doesn't record history),
    # so they're free for our use.
    KEY_HELP = (
        "G1 keyboard controls (sonic_planner / sonic_teleop modes):\n"
        "  Up / Down  - movement forward / back\n"
        "  Left/Right - movement strafe left / right\n"
        "  B          - brake (movement_direction = 0)\n"
        "  , / .      - rotate facing left / right (15 deg per press)\n"
        "  1..9, 0    - locomotion mode\n"
        "               0:IDLE   1:SLOW_WALK  2:WALK   3:RUN  4:SQUAT\n"
        "               5:KNEEL_2 6:KNEEL_1   7:LYING  8:CRAWL 9:IDEL_BOXING\n"
        "  - / =      - target_vel -/+ 0.2 m/s (-1 = mode default)\n"
        "  ; / '      - target height -/+ 0.05 m (-1 = disabled)\n"
        "  N          - re-init planner from current robot pose\n"
        "  C          - calibrate PICO VR (sonic_teleop only)\n"
        "  /          - print current MovementState + this help\n"
        "NOTE: MuJoCo's built-in single-letter shortcuts (W=wireframe, etc.)\n"
        "still fire — they're visual toggles only and won't affect control.\n"
        "Avoided: [ ] (mjviewer camera switch + robocasa wall-transparency).\n"
    )

    #: ``TeleopViewer`` cheat sheet for the 5-mode PICO state machine
    #: (``sonic_manager``). The transitions are ModeManager's, not this class's
    #: -- see mode_manager.py for the table these rows summarise. Rows are
    #: ``(key, meaning)``; an empty key is a section heading.
    _CHEATSHEET_MANAGER = (
        ("", "-- MODE (PICO chords) --"),
        ("A+B+X+Y", "OFF <-> PLANNER (engage / e-stop)"),
        ("L-click", "PLANNER <-> VR_3PT (legs walk, arms VR)"),
        ("A+X", "PLANNER <-> POSE (full-body tracking)"),
        ("B+Y", "POSE <-> FROZEN (legs walk, arms held)"),
        ("", "-- DRIVE (PLANNER / VR_3PT) --"),
        ("L-stick", "walk / strafe"),
        ("R-stick X", "turn"),
        ("A+B / X+Y", "gait + / -  (SLOW_WALK WALK RUN)"),
        ("triggers", "close hands"),
    )

    #: Same, for the keyboard-driven planner modes, which have no mode machine
    #: at all -- the robot is always engaged and the arrow keys are the only
    #: way to move it.
    _CHEATSHEET_KEYBOARD = (
        ("", "-- KEYBOARD (always engaged) --"),
        ("arrows", "walk forward/back, strafe"),
        (", / .", "turn left / right"),
        ("0..9", "gait (0 IDLE 1 SLOW 2 WALK 3 RUN)"),
        ("B", "brake"),
        ("- / =", "speed - / +"),
        ("; / '", "height - / +"),
        ("N", "re-init planner at current pose"),
        ("C", "calibrate PICO (sonic_teleop)"),
    )

    @property
    def hud_cheatsheet(self):
        """Rows for the MuJoCo-window cheat sheet, per ctrl type.

        Read every frame by the viewer loop rather than captured once, because
        ``env.reset()`` rebuilds this object and a stale sheet would outlive the
        controller it describes."""
        if self._ctrl_type == "sonic_manager":
            return self._CHEATSHEET_MANAGER
        if self._ctrl_type in ("sonic_planner", "sonic_teleop"):
            return self._CHEATSHEET_KEYBOARD
        # sonic / sonic_vla / sonic_pose_vla take no operator input at all.
        return ()

    def handle_keypress(self, keycode):
        """Mutate the commanded MovementState based on a GLFW keycode."""
        if (
            self._ctrl_type not in ("sonic_planner", "sonic_teleop", "sonic_manager")
            or self._kplanner is None
        ):
            return
        ms = self._movement_state
        dirty = True

        # Movement: arrow keys (body frame, rotated into world by facing_dir).
        if keycode == self._GLFW_KEY_UP:
            ms.movement_direction = (
                self._rotate_in_xy(
                    np.array([1.0, 0.0, 0.0], dtype=np.float32), ms.facing_direction
                )
                * self._MOV_SPEED_STEP
            )
        elif keycode == self._GLFW_KEY_DOWN:
            ms.movement_direction = (
                self._rotate_in_xy(
                    np.array([-1.0, 0.0, 0.0], dtype=np.float32), ms.facing_direction
                )
                * self._MOV_SPEED_STEP
            )
        elif keycode == self._GLFW_KEY_LEFT:
            ms.movement_direction = (
                self._rotate_in_xy(
                    np.array([0.0, 1.0, 0.0], dtype=np.float32), ms.facing_direction
                )
                * self._MOV_SPEED_STEP
            )
        elif keycode == self._GLFW_KEY_RIGHT:
            ms.movement_direction = (
                self._rotate_in_xy(
                    np.array([0.0, -1.0, 0.0], dtype=np.float32), ms.facing_direction
                )
                * self._MOV_SPEED_STEP
            )
        elif keycode == ord("B"):
            ms.movement_direction[:] = 0.0
        # Facing yaw: , left, . right.  (Avoid [/] — mjviewer uses them for
        # camera switching and robocasa installs a wall-transparency hotkey
        # on them too.)
        elif keycode == ord(","):
            ms.facing_direction = self._rotate_yaw(
                ms.facing_direction, +self._FACING_YAW_STEP
            )
        elif keycode == ord("."):
            ms.facing_direction = self._rotate_yaw(
                ms.facing_direction, -self._FACING_YAW_STEP
            )
        elif ord("0") <= keycode <= ord("9"):
            new_mode = keycode - ord("0")
            ms.locomotion_mode = int(new_mode)
        elif keycode == ord("-"):
            v = ms.movement_speed if ms.movement_speed > 0 else 0.0
            ms.movement_speed = max(-1.0, v - self._VEL_STEP) if v > 0 else -1.0
        elif keycode == ord("="):
            v = ms.movement_speed if ms.movement_speed > 0 else 0.0
            ms.movement_speed = v + self._VEL_STEP
        elif keycode == ord(";"):
            h = ms.height if ms.height >= 0 else 0.5
            ms.height = max(0.1, h - self._HEIGHT_STEP)
        elif keycode == ord("'"):
            h = ms.height if ms.height >= 0 else 0.5
            ms.height = min(1.0, h + self._HEIGHT_STEP)
        elif keycode == ord("N"):
            # Force a re-init on the next tick (resync planner to robot).
            self._planner_initialized = False
            print("[kPlanner] re-init requested via 'N'.")
            return
        elif keycode == ord("C"):
            # Trigger PICO calibration (sonic_teleop only — no-op for sonic_planner).
            if self._pico_source is not None:
                ok = self._pico_source.calibrate()
                print(f"[PicoVR] calibrate() returned {ok}")
            else:
                print("[PicoVR] calibrate ignored — not in sonic_teleop mode.")
            return
        elif keycode == ord("/"):
            print(self.KEY_HELP)
            print(
                f"[kPlanner] current - mode={ms.locomotion_mode} "
                f"mov={ms.movement_direction.tolist()} "
                f"face={ms.facing_direction.tolist()} "
                f"vel={ms.movement_speed} height={ms.height}"
            )
            return
        else:
            dirty = False

        if dirty:
            self._movement_state_dirty = True

    _rotate_in_xy = staticmethod(rotate_in_xy)
    _rotate_yaw = staticmethod(rotate_yaw)

    # ------------------------------------------------------------------ #
    # The actual control law.  Called every sim step.
    # _read_pelvis_state / _read_joint_state come from G1CompositeBase.

    # ------------------------------------------------------------------ #
    # sonic_manager helpers (Phase A)
    # ------------------------------------------------------------------ #
    # Joystick handling — ports gear_sonic/scripts/pico_manager_thread_server.py
    # PlannerStreamer.run_once verbatim.
    _JOYSTICK_DEADZONE = 0.15
    _YAW_GAIN = 1.0  # rad/s scale for right-stick X (1.5 was too twitchy,
    #                           0.6 too sluggish; 1.0 ~= 57 deg/s full stick)
    _PICO_INPUT_DT = 0.02  # joystick dt (50 Hz, matches policy)
    # G1 hand actuators — 14 motors driving 14 finger joints (7 per hand).
    # Names match robot.xml; ctrlrange comes from MJCF (thumb_0 is ±2.45,
    # everything else ±0.7).
    _HAND_LEFT_NAMES = (
        "left_hand_thumb_0",
        "left_hand_thumb_1",
        "left_hand_thumb_2",
        "left_hand_index_0",
        "left_hand_index_1",
        "left_hand_middle_0",
        "left_hand_middle_1",
    )
    _HAND_RIGHT_NAMES = (
        "right_hand_thumb_0",
        "right_hand_thumb_1",
        "right_hand_thumb_2",
        "right_hand_index_0",
        "right_hand_index_1",
        "right_hand_middle_0",
        "right_hand_middle_1",
    )
    # Per-joint target when fully closed (trigger=1.0).  Values come from the
    # ACTUAL joint range="..." attributes in robot.xml (NOT the motor ctrlrange
    # which is torque limits).  Left / right are NOT symmetric — the MJCF
    # mirrors fingers by flipping joint ranges, so closing directions differ.
    #
    # Open pose = 0 rad for all (default standing pose).
    # Close pose = end of each joint range that represents full curl into palm.
    _HAND_LEFT_CLOSE_TARGETS = np.array(
        [
            0.00,  # thumb_0 (opposition, axis y): 0 = thumb curled toward the
            #   fingers; +/-1.04 swing it AWAY (a thumb-tip sweep over
            #   thumb_0 is symmetric about 0, min at 0). NOT the range end.
            +1.04,  # thumb_1 (range [-0.72, +1.04], flex toward palm)
            +2.09,  # thumb_2 (range [0, +2.09], flex)
            -1.83,  # index_0 (range [-1.83, +0.19], close = NEGATIVE for left)
            -2.09,  # index_1 (range [-2.09, 0])
            -1.83,  # middle_0
            -2.09,  # middle_1
        ],
        dtype=np.float32,
    )
    _HAND_RIGHT_CLOSE_TARGETS = np.array(
        [
            0.00,  # thumb_0 (opposition): 0 = curled toward fingers (mirror of left)
            -1.04,  # thumb_1 (range [-1.04, +0.72], flex)
            -2.09,  # thumb_2 (range [-2.09, 0], flex)
            +1.83,  # index_0 (range [-0.19, +1.83], close = POSITIVE for right)
            +2.09,  # index_1 (range [0, +2.09])
            +1.83,  # middle_0
            +2.09,  # middle_1
        ],
        dtype=np.float32,
    )
    _HAND_KP = 8.0
    _HAND_KD = 0.3

    # Rest pose for the 14 arm joints (MuJoCo body order, slots 15..28 of the
    # 29-D body target): arms hang straight down (shoulder_pitch/elbow = 0) with
    # a small shoulder_roll abduction so they clear the torso. Overlaid on the
    # policy's target whenever the upper body isn't being teleoperated, so the
    # idle/standing pose is "arms down" instead of the SONIC default reach pose.
    _ARM_TARGET_SLICE = slice(15, 29)
    # Idle "arms hang low at the sides" target. shoulder_pitch=0 does NOT put the
    # arm vertical in this joint convention (it leaves the forearm bent up at belly
    # height); a positive shoulder_pitch rotates the whole arm down so the hands
    # drop ~6 cm and tuck closer to the torso (eef z ~0.83 -> ~0.76), a more
    # natural standing/init pose. elbow stays 0 (straight) -- flexing it is what
    # raised the hands. A small shoulder_roll abduction keeps the arms off the
    # torso (and symmetric: bending the elbows instead makes the forearms collide).
    # NOTE: this is applied ONLY in the disengaged OFF mode (see arms_down_hold
    # below), so it does not affect PLANNER / VR_3PT / POSE teleop after engage.
    _ARM_REST_POSE = np.array(
        [
            0.6,
            0.12,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # left  arm: pitch roll yaw elbow wr_roll wr_pitch wr_yaw
            0.6,
            -0.12,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,  # right arm
        ],
        dtype=np.float32,
    )
    # Stiffness used to hold the arms at _ARM_REST_POSE (the SONIC-output arm kp
    # is too soft, letting the arms sag forward). Arms are light, so a firm hold
    # doesn't affect balance (legs/waist keep their SONIC gains).
    _ARM_HOLD_KP = 60.0
    _ARM_HOLD_KD = 2.5

    # Locomotion mode bounds for A+B / X+Y cycling.  Cap matches the highest
    # value our LOCAL kPlanner supports (FORWARD_JUMP=17 in
    # sonic_kplanner.py).  SONIC's PICO server caps at 19 (INJURED_WALK) but
    # our planner doesn't carry those — extending requires adding entries to
    # the LocomotionMode class AND verifying kPlanner ONNX accepts them.
    _LOCOMOTION_MIN = LocomotionMode.IDLE
    _LOCOMOTION_MAX = LocomotionMode.FORWARD_JUMP
    # Lower bound for the manually selected gait (A+B/X+Y). IDLE is excluded
    # because it is now applied automatically when the stick is released, so the
    # selectable gaits start at SLOW_WALK (the user never has to dial down to a
    # "stand" gait — letting go of the stick does that).
    _GAIT_MIN = LocomotionMode.SLOW_WALK

    @staticmethod
    def _locomotion_name(mode_id: int) -> str:
        """Return symbolic name for a locomotion mode id, or 'UNKNOWN(<id>)'.
        LocomotionMode is a plain class (not IntEnum) so we reverse-lookup
        from class attributes."""
        for name, value in vars(LocomotionMode).items():
            if not name.startswith("_") and isinstance(value, int) and value == mode_id:
                return name
        return f"UNKNOWN({mode_id})"

    def _tick_mode_manager(self, buttons: dict | None) -> None:
        """Advance ModeManager + fire side-effect events.  Called once per
        ``run_controller`` tick in sonic_manager mode.
        """
        if self._mode_manager is None:
            return
        _new_mode, event = self._mode_manager.tick(buttons)
        if event == CalibrationEvent.NONE:
            return
        if event == CalibrationEvent.FREEZE_SNAPSHOT:
            self._capture_upper_body_snapshot()
            return
        # CALIB_FULL and CALIB_WRISTS both currently resolve to the same
        # ThreePointPose.reset() call (refinement deferred — see
        # ThreePointPose._capture_calibration for how SONIC distinguishes
        # head-vs-wrist re-zero).
        if self._pico_source is not None:
            ok = self._pico_source.calibrate()
        else:
            ok = False
        tag = "CALIB_FULL" if event == CalibrationEvent.CALIB_FULL else "CALIB_WRISTS"
        print(f"[composite] {tag} requested; pico_source.calibrate()={ok}")

    def _robot_heading_yaw(self) -> float:
        """Yaw (rad) of the robot's current heading, from the pelvis quat.

        The robot's forward (+x body axis) rotated into world, projected to the
        xy plane: yaw = atan2(fwd_y, fwd_x). Matches the convention used for
        ``facing_direction = [cos(yaw), sin(yaw), 0]``.
        """
        quat, _ = self._read_pelvis_state()  # wxyz
        return heading_yaw_from_quat(quat)

    def _resync_facing_to_robot(self) -> None:
        """Re-zero the yaw accumulator + facing_direction to the robot's ACTUAL
        current heading. Called on entry into a planner-family mode so steering
        is angular-velocity-like (relative to where the robot now faces) and
        switching back from POSE/OFF doesn't snap it to a stale commanded
        heading."""
        yaw = self._robot_heading_yaw()
        self._yaw_angle_rad = yaw
        self._movement_state.facing_direction = np.array(
            [np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32
        )

    def _tick_pico_planner_input(self, buttons: dict | None) -> None:
        """Translate PICO joystick + A+B/X+Y chord into MovementState updates.

        Faithful port of ``PlannerStreamer.run_once`` from
        ``gear_sonic/scripts/pico_manager_thread_server.py`` lines 1677-1735:

          - A+B rising edge  -> next locomotion mode
          - X+Y rising edge  -> previous locomotion mode
          - Left stick magnitude < deadzone -> IDLE
          - Else movement = rotated [-lx, ly], speed scaled by current mode
          - Right stick X    -> continuous yaw heading accumulator

        Note: ABXY (all four) is filtered out so it only acts as engage/stop,
        not as next-mode.  Same with A+X / B+Y (reserved for Phase B).
        """
        if buttons is None or self._kplanner is None:
            return
        # run_controller (and thus this tick) is called once per *sim substep*
        # (~480 Hz at control_freq=30/sim=500 Hz), but the yaw accumulator below
        # advances by a fixed step assuming _PICO_INPUT_DT (50 Hz). Throttle to
        # ~50 Hz so turning happens at the intended rate (otherwise it spun ~10x
        # too fast / "violently").
        now = float(self.sim.data.time)
        if now - self._last_pico_tick_time < self._PICO_INPUT_DT - 1e-6:
            return
        self._last_pico_tick_time = now
        a, b = bool(buttons.get("a", False)), bool(buttons.get("b", False))
        x, y = bool(buttons.get("x", False)), bool(buttons.get("y", False))

        # Chord guards: only fire AB/XY if the other pair is NOT also held.
        # This stops A+B+X+Y from being interpreted as "next mode".
        ab_now = a and b and not (x and y)
        xy_now = x and y and not (a and b)

        ms = self._movement_state
        dirty = False
        # A+B / X+Y cycle the *selected* walking gait (applied to the live mode
        # only while the stick is pushed; see below). Clamped to [SLOW_WALK, MAX].
        if ab_now and not self._prev_ab_mode:
            new_gait = min(self._LOCOMOTION_MAX, self._selected_gait + 1)
            if new_gait != self._selected_gait:
                self._selected_gait = int(new_gait)
                print(
                    f"[PicoInput] gait -> {self._locomotion_name(new_gait)} ({new_gait})"
                )
            else:
                print(
                    f"[PicoInput] already at max gait "
                    f"({self._locomotion_name(new_gait)}); A+B ignored"
                )
        if xy_now and not self._prev_xy_mode:
            new_gait = max(self._GAIT_MIN, self._selected_gait - 1)
            if new_gait != self._selected_gait:
                self._selected_gait = int(new_gait)
                print(
                    f"[PicoInput] gait -> {self._locomotion_name(new_gait)} ({new_gait})"
                )
            else:
                print(
                    f"[PicoInput] already at min gait "
                    f"({self._locomotion_name(new_gait)}); X+Y ignored"
                )
        self._prev_ab_mode = ab_now
        self._prev_xy_mode = xy_now

        # ----- joystick axes -----
        lx, ly = buttons.get("left_axis", (0.0, 0.0))
        rx, _ry = buttons.get("right_axis", (0.0, 0.0))

        # Right stick X -> yaw accumulator (SONIC's YawAccumulator inlined).
        if abs(rx) >= self._JOYSTICK_DEADZONE:
            self._yaw_angle_rad += self._YAW_GAIN * (-rx) * self._PICO_INPUT_DT
            new_facing = np.array(
                [np.cos(self._yaw_angle_rad), np.sin(self._yaw_angle_rad), 0.0],
                dtype=np.float32,
            )
            if not np.allclose(ms.facing_direction, new_facing, atol=1e-3):
                ms.facing_direction = new_facing
                dirty = True

        # Left stick -> movement direction + speed (SONIC's exact formula).
        raw_mag = float(np.clip(np.hypot(lx, ly), 0.0, 1.0))
        pushed = raw_mag >= self._JOYSTICK_DEADZONE
        turning = abs(rx) >= self._JOYSTICK_DEADZONE
        # Push-to-walk: keep the selected gait active while the left stick moves
        # OR the right stick turns (so the robot can turn in place — otherwise a
        # turn-only input fell through to IDLE, a static mode that ignores the
        # facing change, and the robot wouldn't rotate). Drop to IDLE only when
        # both sticks are neutral. A static selected gait (SQUAT/...) is held.
        if pushed or turning or self._selected_gait in STATIC_MODES:
            desired_mode = self._selected_gait
        else:
            desired_mode = int(LocomotionMode.IDLE)
        if ms.locomotion_mode != desired_mode:
            ms.locomotion_mode = int(desired_mode)
            dirty = True
        if not pushed:
            new_dir = np.zeros(3, dtype=np.float32)
            new_speed = -1.0
        else:
            mag = (raw_mag - self._JOYSTICK_DEADZONE) / (1.0 - self._JOYSTICK_DEADZONE)
            mag = min(mag, 1.0)
            # SONIC's speed mapping per locomotion mode (lines 1717-1724).
            mode = ms.locomotion_mode
            if mode == LocomotionMode.SLOW_WALK:
                new_speed = 0.1 + 0.5 * mag
            elif mode == LocomotionMode.WALK:
                # Stick-proportional speed (0.5~1.5 m/s) so WALK actually
                # responds to how far you push the stick, bridging SLOW_WALK
                # (0.1~0.6) and RUN (1.5~4.5). SONIC originally fixed this at -1
                # (the mode's default speed), so pushing harder never went
                # faster -> felt very insensitive for teleop.
                new_speed = 0.5 + 1.0 * mag
            elif mode == LocomotionMode.RUN:
                new_speed = 1.5 + 3.0 * mag
            else:
                new_speed = mag
            denom = raw_mag if raw_mag > 0 else 1.0
            scale = mag / denom
            # SONIC line 1728: movement_local = [-lx, ly] * scale
            mov_local = np.array([-lx, ly], dtype=np.float32) * scale
            facing = ms.facing_direction[:2]
            perp = np.array([-facing[1], facing[0]], dtype=np.float32)
            mov_global = perp * mov_local[0] + facing * mov_local[1]
            new_dir = np.array([mov_global[0], mov_global[1], 0.0], dtype=np.float32)

        if (
            not np.allclose(ms.movement_direction, new_dir, atol=1e-3)
            or abs(ms.movement_speed - new_speed) > 1e-3
        ):
            ms.movement_direction = new_dir
            ms.movement_speed = new_speed
            dirty = True

        if dirty:
            self._movement_state_dirty = True

    def _write_gripper_torques(self, left_trigger: float, right_trigger: float) -> None:
        """Drive G1 hand actuators with simple PD-to-curl-target.
        trigger=0 -> hand open (joints at 0), trigger=1 -> hand fully closed
        (joints at _HAND_{LEFT,RIGHT}_CLOSE_TARGETS — values match the joint
        range attribute in robot.xml).  Mirrors SONIC's trigger-to-grip
        behaviour but without the dexterous IK solver — Phase B can swap in
        ``G1GripperInverseKinematicsSolver`` for finer per-finger control.
        """
        if self._hand_left_addrs is None:
            return  # mapping not built (older config / no hand actuators)
        lt = float(np.clip(left_trigger, 0.0, 1.0))
        rt = float(np.clip(right_trigger, 0.0, 1.0))
        for (act_id, qp_addr, qv_addr, target) in zip(
            self._hand_left_addrs["act_ids"],
            self._hand_left_addrs["qpos"],
            self._hand_left_addrs["qvel"],
            self._HAND_LEFT_CLOSE_TARGETS,
        ):
            if act_id is None or qp_addr is None:
                continue
            q = float(self.sim.data.qpos[qp_addr])
            dq = float(self.sim.data.qvel[qv_addr])
            torque = self._HAND_KP * (lt * target - q) - self._HAND_KD * dq
            self.sim.data.ctrl[act_id] = torque
        for (act_id, qp_addr, qv_addr, target) in zip(
            self._hand_right_addrs["act_ids"],
            self._hand_right_addrs["qpos"],
            self._hand_right_addrs["qvel"],
            self._HAND_RIGHT_CLOSE_TARGETS,
        ):
            if act_id is None or qp_addr is None:
                continue
            q = float(self.sim.data.qpos[qp_addr])
            dq = float(self.sim.data.qvel[qv_addr])
            torque = self._HAND_KP * (rt * target - q) - self._HAND_KD * dq
            self.sim.data.ctrl[act_id] = torque

    def _write_gripper_joint_targets(self, left_targets, right_targets) -> None:
        """PD-drive the finger joints to explicit per-joint targets (VLA deploy).

        ``left_targets`` / ``right_targets`` are (7,) absolute joint angles in
        actuator order (thumb_0,1,2, index_0,1, middle_0,1); ``None`` -> open
        (zeros). Same PD gains as the trigger path, but the target is the policy's
        predicted joint rather than ``trigger * close_pose``.
        """
        if self._hand_left_addrs is None:
            return
        for addrs, targets in (
            (self._hand_left_addrs, left_targets),
            (self._hand_right_addrs, right_targets),
        ):
            for i, (act_id, qp_addr, qv_addr) in enumerate(
                zip(addrs["act_ids"], addrs["qpos"], addrs["qvel"])
            ):
                if act_id is None or qp_addr is None:
                    continue
                tgt = 0.0 if targets is None or i >= len(targets) else float(targets[i])
                q = float(self.sim.data.qpos[qp_addr])
                dq = float(self.sim.data.qvel[qv_addr])
                self.sim.data.ctrl[act_id] = (
                    self._HAND_KP * (tgt - q) - self._HAND_KD * dq
                )

    def _build_hand_mapping(self) -> None:
        """Look up hand actuator + joint indices once.  Stores ``None``
        entries for hands that aren't present (e.g. G1FixedLowerBody with
        custom MJCF without hands)."""
        self._hand_left_addrs = {"act_ids": [], "qpos": [], "qvel": []}
        self._hand_right_addrs = {"act_ids": [], "qpos": [], "qvel": []}
        prefix = self.robot_model.naming_prefix
        for store, names in (
            (self._hand_left_addrs, self._HAND_LEFT_NAMES),
            (self._hand_right_addrs, self._HAND_RIGHT_NAMES),
        ):
            for name in names:
                act_id = None
                for full_act in (prefix + name, prefix + name + "_joint"):
                    try:
                        act_id = self.sim.model.actuator_name2id(full_act)
                        break
                    except Exception:
                        pass
                full_joint = prefix + name + "_joint"
                qp_addr, qv_addr = None, None
                try:
                    jid = self.sim.model.joint_name2id(full_joint)
                    qp_addr = self.sim.model.jnt_qposadr[jid]
                    qv_addr = self.sim.model.jnt_dofadr[jid]
                except Exception:
                    pass
                store["act_ids"].append(act_id)
                store["qpos"].append(qp_addr)
                store["qvel"].append(qv_addr)
        n_l = sum(1 for a in self._hand_left_addrs["act_ids"] if a is not None)
        n_r = sum(1 for a in self._hand_right_addrs["act_ids"] if a is not None)
        print(f"[composite] hand actuators detected: L={n_l}/7, R={n_r}/7")

    def run_controller(self, enabled_parts):
        qpos, qvel = self._read_joint_state()
        sim_time = float(self.sim.data.time)

        # sonic_manager: poll PICO state ONCE, then dispatch (state machine,
        # joystick input, gripper, controller selection).
        manager_mode = None
        if self._ctrl_type == "sonic_manager":
            sample = self._pico_source.get_latest() if self._pico_source else None
            buttons = sample.get("buttons") if sample is not None else None
            self._tick_mode_manager(buttons)
            manager_mode = self._mode_manager.current
            prev_mode = self._prev_manager_mode

            # On entry into POSE (from any parent), re-align the orientation
            # anchor to the robot's current heading so it doesn't snap from
            # PLANNER's commanded facing to the operator's absolute torso yaw.
            # The actual offset is captured on the next POSE tick (in
            # _update_pose_reference), where the live smpl_root_quat is read.
            if manager_mode == ManagerMode.POSE and prev_mode != ManagerMode.POSE:
                self._pose_align_pending = True

            # On entry into a planner-family mode (which steers via the commanded
            # facing_direction) FROM a non-planner mode (OFF/POSE, which don't),
            # re-sync the yaw accumulator to the robot's actual heading so the
            # robot doesn't snap to a stale facing — i.e. turning behaves like
            # angular-velocity control across mode switches.
            _planner_family = (
                ManagerMode.PLANNER,
                ManagerMode.VR_3PT,
                ManagerMode.PLANNER_FROZEN_UPPER,
            )
            if manager_mode in _planner_family and prev_mode not in _planner_family:
                self._resync_facing_to_robot()
            self._prev_manager_mode = manager_mode

            # Joystick / chord -> MovementState (only while engaged).
            if manager_mode in (ManagerMode.PLANNER, ManagerMode.VR_3PT):
                self._tick_pico_planner_input(buttons)

            # Gripper from triggers — controllable in any engaged mode.
            # OFF -> open (zero trigger).  The actual PD write happens once,
            # unconditionally, at the end of run_controller (below) so the hands
            # are held for every ctrl type, not just sonic_manager.
            if buttons is not None and manager_mode != ManagerMode.OFF:
                self._left_trigger = float(buttons.get("left_trigger", 0.0))
                self._right_trigger = float(buttons.get("right_trigger", 0.0))
            else:
                self._left_trigger = 0.0
                self._right_trigger = 0.0

            # Active body controller selection.
            if manager_mode == ManagerMode.VR_3PT:
                self._wbt_ctrl = self._teleop_ctrl
            elif manager_mode in (ManagerMode.POSE, ManagerMode.POSE_PAUSE):
                # POSE_PAUSE holds the POSE controller: pausing must freeze the
                # operator input, not hand the body to a different controller.
                self._wbt_ctrl = self._pose_ctrl
            else:  # OFF, PLANNER, PLANNER_FROZEN_UPPER all route through planner ctrl
                self._wbt_ctrl = self._planner_ctrl

        # Keep the kPlanner warm whenever it could be needed.  In OFF/POSE
        # we don't strictly need the planner output, but we keep it
        # initialised so transitions back to PLANNER/VR_3PT/FROZEN are instant.
        if self._ctrl_type in ("sonic_planner", "sonic_teleop") or (
            self._ctrl_type == "sonic_manager"
            and manager_mode
            in (
                ManagerMode.PLANNER,
                ManagerMode.VR_3PT,
                ManagerMode.PLANNER_FROZEN_UPPER,
            )
        ):
            self._lazy_init_planner_if_needed()
            self._tick_planner(sim_time)

        need_policy = (
            self._ctrl_type == "default_pos"
            or self._cached_target_q is None
            or (sim_time - self._last_policy_time) >= self._POLICY_DT - 1e-6
        )

        if need_policy:
            quat, ang_vel = self._read_pelvis_state()
            # Refresh the reference (no-op for default_pos):
            if self._ctrl_type == "sonic":
                self._update_default_pose_reference()
            elif self._ctrl_type == "sonic_planner":
                self._update_planner_reference()
            elif self._ctrl_type == "sonic_teleop":
                self._update_teleop_reference()
            elif self._ctrl_type == "sonic_manager":
                if manager_mode == ManagerMode.OFF:
                    # SONIC encoder mode_id=0 with the fixed default-pose ref
                    # (this is the g1_standing demo's behaviour).  The robot
                    # stands using the policy, not naive PD.
                    self._update_default_pose_reference(ctrl=self._planner_ctrl)
                elif manager_mode == ManagerMode.PLANNER:
                    self._update_planner_reference()
                    # Warm the inactive controllers' buffers (cheap numpy writes).
                    self._update_teleop_reference(ctrl=self._teleop_ctrl)
                    self._update_pose_reference(
                        ctrl=self._pose_ctrl, compute_from_raw=False
                    )
                elif manager_mode == ManagerMode.VR_3PT:
                    self._update_teleop_reference()
                    self._update_planner_reference(ctrl=self._planner_ctrl)
                    self._update_pose_reference(
                        ctrl=self._pose_ctrl, compute_from_raw=False
                    )
                elif manager_mode == ManagerMode.POSE:
                    self._update_pose_reference()
                    self._update_planner_reference(ctrl=self._planner_ctrl)
                    self._update_teleop_reference(ctrl=self._teleop_ctrl)
                elif manager_mode == ManagerMode.POSE_PAUSE:
                    # Deliberately refresh nothing: the pose controller keeps
                    # driving the last reference it was handed, which is what
                    # upstream's "stop the streamer" pause amounts to.
                    pass
                elif manager_mode == ManagerMode.PLANNER_FROZEN_UPPER:
                    # Planner controller, but _update_planner_reference will
                    # overlay the upper-body snapshot internally based on
                    # _mode_manager.current.
                    self._update_planner_reference()
                    self._update_teleop_reference(ctrl=self._teleop_ctrl)
                    self._update_pose_reference(
                        ctrl=self._pose_ctrl, compute_from_raw=False
                    )
            elif self._ctrl_type == "heft_tracker":
                # No reference refresh here: the chunk is pushed in from the
                # rollout harness via set_heft_reference(), and the controller
                # advances its own cursor once per policy tick.
                pass
            state = {
                "joint_pos": qpos,
                "joint_vel": qvel,
                "quaternion": quat,
                "gyroscope": ang_vel,
            }
            if self._ctrl_type == "sonic_vla":
                # Decode-only with the VLA-supplied motion token (zeros until the
                # first policy action arrives -> holds the SONIC default pose).
                token = (
                    self._vla_token
                    if self._vla_token is not None
                    else np.zeros(64, dtype=np.float32)
                )
                target_q, kps, kds = self._wbt_ctrl.decode_with_token(
                    _RobotShim(state), token
                )
            elif self._ctrl_type == "sonic_pose_vla":
                # Reference-motion model: push the predicted SMPL joints into the
                # POSE encoder, then run the FULL encoder->decoder. Before the first
                # policy action the smpl buffer is zeros -> the encoder produces the
                # standing/default token (robot just stands).
                self._update_vla_pose_reference()
                target_q, kps, kds = self._wbt_ctrl.step(_RobotShim(state))
            else:
                target_q, kps, kds = self._wbt_ctrl.step(_RobotShim(state))
            self._cached_target_q = np.asarray(target_q, dtype=np.float32)
            self._cached_kps = np.asarray(kps, dtype=np.float32)
            self._cached_kds = np.asarray(kds, dtype=np.float32)
            self._last_policy_time = sim_time
            # Advance the planner motion cursor once per 50 Hz tick — but
            # only in non-static modes.  Static modes (IDLE/SQUAT/etc.) hold
            # frame 0 so the reference is constant; the IDLE branch in
            # _update_planner_reference uses the SONIC default pose directly.
            planner_active = self._ctrl_type in ("sonic_planner", "sonic_teleop") or (
                self._ctrl_type == "sonic_manager"
                and self._mode_manager.current
                in (ManagerMode.PLANNER, ManagerMode.VR_3PT)
            )
            if (
                planner_active
                and self._kplanner is not None
                and self._kplanner.num_frames > 0
                and self._movement_state.locomotion_mode not in STATIC_MODES
            ):
                self._current_motion_frame = min(
                    self._current_motion_frame + 1, self._kplanner.num_frames - 1
                )

        target_q = self._cached_target_q
        kps = self._cached_kps
        kds = self._cached_kds

        # Hold the "arms hang down" rest pose ONLY while idle/disengaged (OFF),
        # so the standing/init pose looks natural. In every engaged mode the
        # upper body is left to its own controller: PLANNER owns the arms (so
        # they swing/track the planner motion rather than freezing in one pose),
        # and VR_3PT / POSE drive them from the operator.
        arms_down_hold = (
            self._ctrl_type == "sonic_manager" and manager_mode == ManagerMode.OFF
        )
        if arms_down_hold:
            target_q = np.array(target_q, dtype=np.float32, copy=True)
            kps = np.array(kps, dtype=np.float32, copy=True)
            kds = np.array(kds, dtype=np.float32, copy=True)
            target_q[self._ARM_TARGET_SLICE] = self._ARM_REST_POSE
            kps[self._ARM_TARGET_SLICE] = self._ARM_HOLD_KP
            kds[self._ARM_TARGET_SLICE] = self._ARM_HOLD_KD

        self._write_body_torques(target_q, kps, kds, qpos, qvel)

        # Hold the hand actuators every control step (all ctrl types). The body
        # loop above only writes the 29 body actuators, so the hands need their
        # own PD. Triggers default to 0 (open) and are only raised by PICO in
        # sonic_manager mode. Without this the fingers were uncommanded outside
        # sonic_manager (keyboard / planner / data collection) and drifted — the
        # left hand's negative joint range curled it shut at rest.
        if self._ctrl_type in ("sonic_vla", "sonic_pose_vla"):
            # VLA deploy: PD-drive each finger to the policy's predicted joint
            # targets (7 per hand, actuator order). Falls back to open (zeros)
            # until the first action arrives.
            self._write_gripper_joint_targets(self._vla_left_hand, self._vla_right_hand)
        else:
            self._write_gripper_torques(self._left_trigger, self._right_trigger)

        return {}
