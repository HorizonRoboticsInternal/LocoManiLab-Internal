"""The SONIC backend: kPlanner locomotion + SMPL pose tracking.

Two control paths behind one checkpoint family, chosen per tick by ModeManager
mode and by which checkpoint flavor is loaded:

* **PLANNER** -- the kinematic planner (``planner_sonic.onnx``) generates a
  50 Hz reference motion from a locomotion mode + heading + speed, driven by
  the PICO joysticks or the keyboard. Only wired up for the g1 split-onnx
  flavor; the smpl flavor's encoder has no locomotion-capable input at all.
* **POSE** -- the smpl sub-encoder is fed the operator's live SMPL body pose,
  which is what the official deployment does for "follow me".

The flavor is decided once in on_load rather than re-derived from
``isinstance(tracking, ...)`` at three separate tick-time sites.
"""

from __future__ import annotations

import pathlib
import time

import numpy as np

from g1_bridge.pico.mode_manager import ManagerMode
from g1_bridge.pico.vr_source import process_smpl_pose_full
from g1_bridge.g1_runtime import (
    G1_JOINT_NAMES,
    heading_yaw_from_quat,
    quat_inv as _quat_inv,
    quat_mul as _quat_mul,
    relative_anchor6 as _relative_anchor6,
    resolve_ckpt_path,
    rotate_in_xy,
    rotate_yaw,
)
from g1_bridge.sonic_utils.kplanner import (
    STATIC_MODES,
    LocomotionMode,
    MovementState,
    SonicKPlanner,
)
from g1_bridge.vendored.wbt.controllers.wbt_controller_sonic import (
    SONIC_DEFAULT_ANGLES_MUJOCO,
    SONIC_KDS,
    SONIC_KPS,
)
from sim2real.utils.dex3 import grip_to_joint_targets

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from sim2real.rl_policy.families import get_family
from g1_bridge.backends.base import (
    BRIDGE_KEY_HELP,
    BackendResources,
    Sim2RealBackend,
    Sim2RealContext,
    TickInput,
)
from g1_bridge.bridge_runtime import SplitOnnxBridgeTracking, make_bridge_tracking
from g1_bridge.reference.default import default_reference_frame
from g1_bridge.reference.blend import PoseBlender, qpos_from_payload
from g1_bridge.pico.smpl_payload import build_payload as build_smpl_payload
from g1_bridge.recording.frames import sonic_action_43
from g1_bridge.sonic_utils.encoder_obs import (
    sonic_g1_low_latency_encoder_input,
    sonic_g1_v1_1_encoder_input,
)

#: The split-onnx ``g1_input`` observation classes, one per sonic encoder
#: variant. Only low_latency implements the smpl sub-encoder.
_SPLIT_ONNX_ENCODER_OBS = (
    sonic_g1_low_latency_encoder_input,
    sonic_g1_v1_1_encoder_input,
)

from sim2real.config.robots.base import MOTION_FIRST_FRAME_KEY

#: Seconds to ramp a reference in when a mode switch would otherwise step it.
#: Shared spelling with the GMR backends (sim2real_pose_blend_s); 0 disables.
_DEFAULT_BLEND_S = 2.0

#: A PICO pose older than this records as zeros rather than as a frozen
#: operator -- 100 ms, the official collector's own threshold
#: (gear_sonic/scripts/run_data_exporter.py _add_sonic_pose_features).
_SMPL_STALE_S = 0.1

#: How often to repeat the "body pose stale" warning while it persists.
_SMPL_STALE_WARN_EVERY_S = 2.0


# The kPlanner ONNX is ~739MB and its session must survive env.reset(): robosuite
# rebuilds the composite every episode, and a fresh CUDA session per reset leaks
# cuDNN handles until cudnnCreate() fails a few dozen episodes in. Cached
# process-wide, keyed by (path, height, seed); the planner holds no sim state, so
# reuse is safe -- reset() clears its buffers.
_KPLANNER_CACHE: dict = {"key": None, "obj": None}


import os

# SIM2REAL_PICO_DEBUG=1 -> ~1 Hz dump of the raw PICO joystick axes next to
# the MovementState they produce, from inside _tick_pico_planner_input. Off
# by default (costs nothing). This is the one diagnostic that separates the
# two ways PLANNER-mode locomotion can appear dead: axes arriving as
# (0.0, 0.0) means the input never reaches us (headset/SDK/relay side),
# whereas live axes plus an unchanging MovementState/locomotion_mode means
# the input is fine and kPlanner is the one not acting on it. Mode buttons
# are handled separately by ModeManager.
_PICO_INPUT_DEBUG = os.environ.get("SIM2REAL_PICO_DEBUG", "0").lower() not in (
    "0",
    "",
    "false",
    "no",
)


def _composite_key_help() -> str:
    """The on-screen key list -- the same one the viewer loop prints.

    One copy, in ``backends/base.py``: the keys it documents are the mode
    manager's chords and this backend's own kPlanner keys, and reaching up into
    ``composite/bridge.py`` for it was the single import edge from a backend to
    the host adapter.
    """
    return BRIDGE_KEY_HELP


#: Split-onnx (g1) vs fused (smpl) checkpoint flavor.
FLAVOR_SPLIT_ONNX = "split_onnx"
FLAVOR_SMPL = "smpl"

#: Modes driven by the kPlanner lower body. They share the yaw accumulator and
#: the reference blend, so entry/exit bookkeeping keys off the family rather
#: than on PLANNER alone -- otherwise an L-click into VR_3PT cancels a blend
#: that is still ramping and steps the reference.
_PLANNER_FAMILY = (ManagerMode.PLANNER, ManagerMode.PLANNER_VR_3PT)


def _in_planner_family(mode) -> bool:
    return mode in _PLANNER_FAMILY


class SonicBackend(Sim2RealBackend):
    """GR00T-SONIC whole-body policy with kPlanner locomotion."""

    name = "sonic"
    POLICY_CONFIGS = {"sonic": get_family("sonic").policy_config}
    #: The policy predicts SONIC's own latent, so the encoder is skipped and
    #: the token goes straight to the decoder (SplitOnnxPolicyMixin
    #: .set_motion_token). Nothing but SONIC can read this space.
    vla_action_space = "motion_token"
    controller_json = "g1_sonic.json"
    #: SONIC gets its reference from PICO directly; no GMR subprocess.
    uses_gmr = False

    # Rotation helpers used by the steering keys; module functions bound as
    # class attributes so the moved call sites keep their original spelling.
    _rotate_in_xy = staticmethod(rotate_in_xy)
    _rotate_yaw = staticmethod(rotate_yaw)

    _PLANNER_DT = 0.1
    _REPLAN_INTERVAL_RUNNING = 0.1
    _REPLAN_INTERVAL_CRAWLING = 0.2
    _REPLAN_INTERVAL_BOXING = 1.0
    _REPLAN_INTERVAL_DEFAULT = 1.0
    _JOYSTICK_DEADZONE = 0.15
    _YAW_GAIN = 1.0
    _PICO_INPUT_DT = 0.02
    _GLFW_KEY_RIGHT = 262
    _GLFW_KEY_LEFT = 263
    _GLFW_KEY_DOWN = 264
    _GLFW_KEY_UP = 265
    _MOV_SPEED_STEP = 0.5
    _FACING_YAW_STEP = 0.2618
    _VEL_STEP = 0.2
    _HEIGHT_STEP = 0.05

    @property
    def hud_cheatsheet(self):
        """The 5-mode chords plus kPlanner steering.

        Two things the shared default cannot say: PLANNER here is real
        locomotion (the sticks use a fixed SLOW_WALK gait), and
        VR_3PT is reachable on the ModeManager but does nothing yet -- ``step``
        falls through to ``hold_pose`` for it. Advertising a chord that silently
        freezes the robot is worse than not listing it, so the L-stick click is
        marked rather than shown as a mode."""
        return (
            ("", "-- MODE (PICO chords) --"),
            ("A+B+X+Y / E", "OFF <-> PLANNER (engage / e-stop)"),
            ("A+X / P", "PLANNER <-> POSE (track your body)"),
            ("L-click", "VR_3PT - not implemented, holds"),
            ("", "-- DRIVE (PLANNER) --"),
            ("L-stick", "slow walk / strafe (fixed gait)"),
            ("R-stick X", "turn"),
            ("arrows , .", "same, from the keyboard"),
            ("0..9 / N", "gait / re-init planner here"),
            ("triggers", "close hands (engaged only)"),
        )

    def __init__(self):
        self._flavor = FLAVOR_SMPL
        self._human_joints_info_path = None
        self._kplanner = None
        self._movement_state = MovementState()
        self._last_movement_state = self._movement_state.copy()
        self._movement_state_dirty = False
        # Recording-only state (schema v6, see recording.features):
        # the pelvis quaternion captured at engage (OFF -> anything), constant
        # for the session; and a counter of distinct PICO samples seen, so a
        # dataset row says which operator sample it was built from.
        self._init_base_quat = None
        self._pico_frame_index = 0
        self._last_pico_sample_ts = None
        self._planner_initialized = False
        self._current_motion_frame = 0
        self._last_planner_check_time = -1e9
        self._last_replan_time = -1e9
        self._replan_interval_counter = 0.0
        self._yaw_angle_rad = 0.0
        # Reference blending -- see _step_kplanner_pose and _push_pose_reference
        # for the two channels and why they cannot share one mechanism.
        self._blend_s = _DEFAULT_BLEND_S
        self._planner_blender = PoseBlender(_DEFAULT_BLEND_S)
        self._planner_blend_pending = False
        self._planner_anchor_qpos = None
        self._pose_blend_pending = False
        self._pose_blend_t0 = None
        # The operator's SMPL pose at the engage chord, i.e. standing. Anchor
        # for the POSE ramp; None until captured (mock source, or POSE entered
        # without ever passing through the chord).
        self._standing_smpl = None
        self._last_pico_tick_time = -1e9
        self._pico_dbg_t = -1e9
        self._sonic_enc_obs = None
        self._pose_root_align = None  # (4,) wxyz, or None
        self._pose_align_pending = False
        # One SMPL retarget per PICO sample, shared by the recorder and the
        # encoder -- see _pose_data.
        self._pose_data_cache = None
        self._pose_data_ts = None
        # Operator-facing staleness reporting -- see _note_smpl_staleness.
        self._smpl_stale_since = None
        self._smpl_stale_warned = -1e9

    # ------------------------------------------------------------------ #
    # Construction (cache-key-guarded: about once per process)
    # ------------------------------------------------------------------ #
    def build_tracking(self, cfg, policy_config_path, inference_backend):
        tracking, io = make_bridge_tracking(
            policy_config_path, motion_mode="push", inference_backend=inference_backend
        )
        human_joints_info_path = str(
            pathlib.Path(policy_config_path).parent / "human_joints_info.pkl"
        )
        group = getattr(tracking, "observations", {}).get("g1_input")
        funcs = getattr(group, "funcs", {}) if group is not None else {}
        enc_obs = next(
            (o for o in funcs.values() if isinstance(o, _SPLIT_ONNX_ENCODER_OBS)), None
        )
        sub = [
            m
            for m in ("smpl", "teleop")
            if enc_obs is None or self._encoder_supports(enc_obs, m)
        ]
        modes = (
            f"sub-encoders: g1 + {' + '.join(sub)}"
            if sub
            else "g1 mode only (no smpl/teleop sub-encoder); POSE and VR_3PT are refused"
        )
        print(
            "[g1_bridge] G1_BRIDGE armed, backend=sonic "
            f"(policy_config={policy_config_path}). {modes}; "
            "OFF/FROZEN/POSE_PAUSE hold the standing pose."
        )
        return BackendResources(
            tracking=tracking,
            io=io,
            flavor=(
                FLAVOR_SPLIT_ONNX
                if isinstance(tracking, SplitOnnxBridgeTracking)
                else FLAVOR_SMPL
            ),
            extra={"human_joints_info_path": human_joints_info_path},
        )

    # ------------------------------------------------------------------ #
    # Per env.reset() -- UNGUARDED. _init_kplanner belongs here, not in
    # build_tracking: it re-seeds MovementState from config every episode, and
    # a facing_direction carried over from the previous episode sends the
    # robot walking off toward a stale heading.
    # ------------------------------------------------------------------ #
    def on_load(self, ctx: Sim2RealContext, cfg: dict, res: BackendResources) -> None:
        self._flavor = res.flavor
        self._human_joints_info_path = res.extra.get("human_joints_info_path")
        self._blend_s = float(cfg.get("sim2real_pose_blend_s", _DEFAULT_BLEND_S))
        self._planner_blender = PoseBlender(self._blend_s)
        self._init_kplanner(cfg)

    def reset(self, ctx: Sim2RealContext) -> None:
        self._pose_root_align = None
        self._pose_align_pending = False
        self._planner_blender.cancel()
        self._planner_blend_pending = False
        self._planner_anchor_qpos = None
        self._pose_blend_pending = False
        self._pose_blend_t0 = None
        self._standing_smpl = None
        self._pose_data_cache = None
        self._pose_data_ts = None
        self._smpl_stale_since = None
        self._smpl_stale_warned = -1e9
        self._reset_kplanner_state()

    # ------------------------------------------------------------------ #
    # Per tick
    # ------------------------------------------------------------------ #
    def on_mode_change(self, ctx: Sim2RealContext, prev, new) -> None:
        """Re-anchor on entry into a steering/tracking mode.

        PLANNER: re-sync the yaw accumulator to the robot's actual heading so
        steering behaves like angular-velocity control across mode switches
        rather than absolute-orientation control.

        POSE: re-align the smpl encoder's orientation anchor to the current
        heading, so it doesn't snap from PLANNER's commanded facing to the
        operator's absolute torso yaw. The offset itself is captured on the
        next POSE tick, where the live smpl_root_quat is read -- see
        _push_pose_reference. That fixes the *orientation* half of the switch;
        the body shape is ramped separately, armed here.

        OFF -> anything: capture the operator's SMPL pose as the "standing"
        anchor for that ramp. This is the engage chord, the one moment the
        operator is documented to be standing in the calibration pose -- and
        the moment the robot is standing too, so the two agree on what
        "standing" means. Nothing else in the session offers that.
        """
        if new == ManagerMode.POSE and prev != ManagerMode.POSE:
            # At the chord press, while the operator can still do something
            # about it -- rather than one tick later with the robot committed.
            self._reject_pose_without_smpl_encoder(ctx)
        if _in_planner_family(new) and not _in_planner_family(prev):
            self._resync_facing_to_robot(ctx)
        if new == ManagerMode.PLANNER_VR_3PT and prev != ManagerMode.PLANNER_VR_3PT:
            # Same fail-at-the-chord rule as POSE: refuse here, not one tick
            # deeper, so a checkpoint without the teleop sub-encoder reads as
            # "VR_3PT does not exist" rather than as VR_3PT working badly.
            self._reject_vr_3pt_without_teleop_encoder(ctx)
        if prev == ManagerMode.OFF and new != ManagerMode.OFF:
            self._capture_standing_smpl(ctx)
            # observation.init_base_quat (schema v6): the robot's pelvis
            # attitude at the engage chord -- upstream's "IMU quaternion
            # captured at init". Captured once per session (first engage),
            # like upstream; re-engaging after an emergency stop keeps it.
            if self._init_base_quat is None:
                quat, _ = ctx.read_pelvis_state()
                self._init_base_quat = np.asarray(quat, dtype=np.float64).reshape(4).copy()
        if new == ManagerMode.POSE and prev != ManagerMode.POSE:
            self._pose_align_pending = True
            self._pose_blend_pending = True
        # The blend ramps the kPlanner reference in; VR_3PT rides the same
        # planner lower body, so entering it from PLANNER must NOT restart or
        # cancel the ramp -- the reference is continuous across that switch.
        if _in_planner_family(new) and not _in_planner_family(prev):
            self._planner_blend_pending = True
        if not _in_planner_family(new):
            self._planner_blender.cancel()
        if new != ManagerMode.POSE:
            self._pose_blend_pending = False
            self._pose_blend_t0 = None

    def seed_reference(self, ctx: Sim2RealContext, tick: TickInput) -> None:
        """Both flavors open every episode against the self-anchored default
        frame: split-onnx keeps it there permanently (no live reference wired up
        yet), and smpl replaces it with the operator's pose on the first POSE
        tick. Either way it has to be in the buffer before the yaw alignment is
        calibrated -- see Sim2RealBackend.seed_reference."""
        ctx.push_synthetic_default_once(tick.qpos)

    def step(self, ctx: Sim2RealContext, tick: TickInput):
        # PLANNER (kPlanner locomotion) is only wired up for the split-onnx
        # flavor -- checked BEFORE the catch-all below, which would otherwise
        # swallow PLANNER into POSE-mode stepping.
        # POSE_PAUSE means "stop taking operator input, keep tracking what is
        # already in flight" -- upstream implements it by stopping the pose
        # streamer (pico_manager_thread_server.py:2106-2112), so the policy
        # runs on the last reference it was given. Stepping with no push does
        # exactly that. Falling through to the g1 branch instead would snap the
        # robot from following the operator to a standing pose, which is the
        # opposite of a pause and dangerous mid-motion.
        if tick.mode == ManagerMode.POSE_PAUSE:
            return ctx.step_policy(tick.qpos)
        if tick.mode == ManagerMode.PLANNER and self._flavor == FLAVOR_SPLIT_ONNX:
            return self._step_kplanner_pose(ctx, tick.qpos, tick.sim_time, tick.buttons)
        # VR_3PT (mode_id=1): kPlanner legs, PICO 3-point arms. Same planner
        # machinery as PLANNER -- only the encoder mode and the extra 3-point
        # channel differ -- so it is dispatched next to it, not through
        # _step_sonic_pose's g1/smpl switch.
        if (
            tick.mode == ManagerMode.PLANNER_VR_3PT
            and self._flavor == FLAVOR_SPLIT_ONNX
        ):
            return self._step_vr_3pt(ctx, tick.qpos, tick.sim_time, tick.buttons)
        # The split-onnx flavor has no live PICO/GMR reference wired up yet, so
        # (unlike smpl) it runs in every mode, not just POSE.
        if tick.mode == ManagerMode.POSE or self._flavor == FLAVOR_SPLIT_ONNX:
            return self._step_sonic_pose(ctx, tick.qpos, tick.mode)
        # OFF / PLANNER_FROZEN_UPPER / POSE_PAUSE / VR_3PT: not implemented for
        # the fused smpl flavor -- hold.
        return ctx.hold_pose()

    def handle_keypress(self, ctx: Sim2RealContext, keycode: int) -> bool:
        if self._kplanner is None:
            return False
        return self._handle_planner_key(ctx, keycode)

    def teleop_sample(self, ctx: Sim2RealContext) -> dict:
        def _arr(v):
            return None if v is None else np.asarray(v, dtype=np.float32).reshape(-1)

        ms = self._movement_state
        out = {
            # Real values once kPlanner is armed (self._kplanner is not None --
            # see _init_kplanner); otherwise (no planner_path configured, or
            # non-PLANNER modes) these stay at the MovementState default
            # (IDLE / facing +x / zero movement / speed & height -1 = "mode
            # default"), which is also what the columns read before v6 for
            # the two that existed then.
            "locomotion_mode": np.array([int(ms.locomotion_mode)], dtype=np.float32),
            "facing_direction": ms.facing_direction.astype(np.float32),
            "planner_movement": _arr(ms.movement_direction),
            "planner_speed": np.array([float(ms.movement_speed)], dtype=np.float32),
            "planner_height": np.array([float(ms.height)], dtype=np.float32),
        }
        mode = ctx.mode_manager.current if ctx.mode_manager is not None else None
        out["stream_mode"] = np.array(
            [int(mode) if mode is not None else -1], dtype=np.float32
        )
        sample = ctx.pico_source.get_latest() if ctx.pico_source is not None else None

        # ONE freshness decision per tick, covering every column derived from
        # the operator's body frame. The official collector gates its two
        # groups separately (pose <=100 ms, planner <=200 ms,
        # run_data_exporter._add_sonic_pose_features) because they arrive on
        # two ZMQ topics at different rates; here the 3-point pose and the SMPL
        # pose are computed from the SAME body frame
        # (vr_source: three_point.process_smpl_pose(body_poses_np)), so a
        # second threshold would only let one half of a frame go stale while
        # the other half zeroed -- the two columns would then disagree about
        # where the operator was at the same instant.
        fresh = self._smpl_is_fresh(sample)
        self._note_smpl_staleness(sample, fresh)

        vr_pos = sample.get("vr_3point_local_target") if sample and fresh else None
        vr_orn = sample.get("vr_3point_local_orn_target") if sample and fresh else None
        out["vr_3point_pos"] = _arr(vr_pos)
        out["vr_3point_orn"] = _arr(vr_orn)
        # Derived through _pose_data rather than read straight off the sample:
        # the source only pre-computes pose_smpl_* when its background pose
        # pipeline is switched on, and it is OFF by default even on the real
        # robot, so these three recorded as zeros through every real-PICO
        # collection while POSE-mode teleop itself worked (the encoder path
        # re-derived them and kept the result to itself).
        #
        # A pose older than the gate is recorded as zeros rather than as a
        # frozen operator, so a dead SMPL stream is visible in the data instead
        # of looking like someone standing perfectly still.
        pose = self._pose_data(sample) if fresh else None
        out["smpl_joints_local"] = _arr(pose.get("smpl_joints_local") if pose else None)
        out["smpl_root_quat"] = _arr(pose.get("smpl_root_quat") if pose else None)
        out["wrist_joints"] = _arr(pose.get("wrist_joints") if pose else None)
        # smpl_pose (63): the SMPL body pose as axis-angle, joints 1..21 -- the
        # quantity upstream records under teleop.smpl_pose, and what the smpl
        # flavor's encoder consumes (build_smpl_payload's smpl_body_pose_aa).
        # Derived straight from the raw (24,7) PICO sample so it is recorded in
        # EVERY mode and flavor whenever SMPL is streaming; None (-> zeros)
        # otherwise, which is upstream's own fallback.
        out["smpl_pose"] = self._smpl_pose_from_sample(sample) if fresh else None
        # pico_frame_index: how many distinct PICO samples this session has
        # seen, bumped when the sample's own timestamp moves. Every host-side
        # consumer of the sample already keys freshness off that timestamp.
        ts = sample.get("timestamp") if sample else None
        if ts is not None and ts != self._last_pico_sample_ts:
            self._last_pico_sample_ts = ts
            self._pico_frame_index += 1
        out["pico_frame_index"] = np.array([self._pico_frame_index], dtype=np.int64)
        out["init_base_quat"] = (
            None if self._init_base_quat is None else self._init_base_quat.copy()
        )

        # --- GR00T unitree_g1_sonic action fields ---
        # motion_token (64D): the SONIC encoder latent currently driving the
        # whole-body decoder (== proprio["token_state"] in gear_sonic). None
        # before the first policy tick, and on runtimes with no split encoder.
        out["motion_token"] = _arr(getattr(ctx.tracking, "token_state", None))

        # left/right hand joints (7D each): the COMMANDED finger targets this
        # tick, in the Dex3 actuator order (thumb_0,1,2, index_0,1, middle_0,1)
        # that write_gripper_torques drives -- identical to what amo, heft and
        # almi record for their own hand action, and to what
        # Sim2RealBackend.vla_hand_targets feeds straight back into the fingers
        # at deploy.
        #
        # Until 2026-08-11 these were ctx.read_hand_state(), i.e. the MEASURED
        # finger angles. G1_modality.json declares them under `action`, so the
        # sonic VLA's target vector carried 14 dimensions of "where the fingers
        # are right now" -- which observation.state already carries (blocks
        # 22:29 and 36:43), so the model could satisfy that part of the loss by
        # copying its own input, and at deploy it predicted the current pose
        # instead of the grasp to make. The legacy npz recorder these column
        # names came from (sim2real/rl_policy/recording/extras/sonic_extra.py)
        # always took them from the teleop stream, never from the robot.
        left_trigger = float(np.clip(ctx.left_trigger, 0.0, 1.0))
        right_trigger = float(np.clip(ctx.right_trigger, 0.0, 1.0))
        left = grip_to_joint_targets(left_trigger, "left")
        right = grip_to_joint_targets(right_trigger, "right")
        out["left_hand_joints"] = left.astype(np.float32)
        out["right_hand_joints"] = right.astype(np.float32)
        # The raw triggers too (schema v6): *_hand_joints is ONE mapping of the
        # trigger (grip_to_joint_targets); recording the input lets any other
        # mapping -- upstream's binary IK gesture, a different close pose -- be
        # re-derived offline instead of needing a re-collection.
        out["left_trigger"] = np.array([left_trigger], dtype=np.float32)
        out["right_trigger"] = np.array([right_trigger], dtype=np.float32)

        # action.wbc (43, schema v6): the 29 body targets with the commanded
        # fingers above interleaved in observation.state's slots -- see
        # recording.frames.sonic_action_43. None until the first policy tick
        # (this is the readiness sentinel, SONIC_TELEOP_KEYS[0]).
        out["target_q_43"] = (
            None
            if ctx.cached_target_q is None
            else sonic_action_43(ctx.cached_target_q, left, right).astype(np.float32)
        )
        return out

    def _note_smpl_staleness(self, sample, fresh: bool) -> None:
        """Tell the operator, once, when the body stream dies and when it comes
        back.

        The official collector does this too -- ``_print_and_say("Sonic pose
        stale (...ms old), using zeros")`` every tenth consecutive failure --
        and without it the only evidence is a column of zeros discovered after
        the session. Rate-limited by TIME rather than by a failure count,
        because this runs at the 50 Hz control rate instead of the exporter's
        loop: upstream's "every tenth" would be five lines a second here.
        """
        if fresh:
            if self._smpl_stale_since is not None:
                held = time.monotonic() - self._smpl_stale_since
                print(
                    f"[g1_bridge] PICO body stream recovered after {held:.1f}s "
                    "-- SMPL columns are being recorded again"
                )
                self._smpl_stale_since = None
                self._smpl_stale_warned = -1e9
            return
        if sample is None:
            # No sample at all is a different failure (source not started, or
            # dead); the host reports that on its own.
            return
        now = time.monotonic()
        if self._smpl_stale_since is None:
            self._smpl_stale_since = now
        if now - self._smpl_stale_warned < _SMPL_STALE_WARN_EVERY_S:
            return
        self._smpl_stale_warned = now
        ts = sample.get("body_timestamp", sample.get("timestamp"))
        age_ms = (now - float(ts)) * 1e3 if ts is not None else float("nan")
        print(
            f"[g1_bridge] PICO body pose stale ({age_ms:.0f} ms old) -- "
            "recording SMPL and 3-point columns as ZEROS. Check the headset's "
            "body tracking; another PICO connection may own the body stream."
        )

    def _smpl_is_fresh(self, sample) -> bool:
        """Is this sample's SMPL recent enough to record? Samples with no
        timestamp are trusted (that is upstream's fallback too)."""
        if sample is None:
            return False
        # ``body_timestamp``, not ``timestamp``: the isolated PICO worker
        # (ProcessPicoVRSource, the source --pico-source real actually uses)
        # re-stamps ``timestamp`` on every loop of a body-tracking outage so
        # buttons and device poses keep flowing, which would make a dead SMPL
        # stream look permanently fresh. Sources that predate the split fall
        # back to the sample clock.
        ts = sample.get("body_timestamp", sample.get("timestamp"))
        if ts is None:
            return True
        return (time.monotonic() - float(ts)) <= _SMPL_STALE_S

    def _pose_data(self, sample) -> dict | None:
        """The operator's full SMPL derivation for one PICO sample, or None.

        The source only fills ``pose_smpl_*`` when its background pose
        pipeline is enabled (``PICO_ENABLE_BACKGROUND_POSE_DATA``, default
        OFF -- see pico.vr_source); otherwise only the raw ``(24, 7)`` sample
        is there and running ``process_smpl_pose_full`` is ours to do. Doing
        it here rather than at each call site is what keeps the RECORDED
        ``teleop.smpl_*`` columns in step with what the encoder is actually
        tracking: before this, POSE mode retargeted for the policy in
        _push_pose_reference and threw the result away, so with the env flag
        off (the default, on the real robot too) every smpl column recorded
        as zeros while teleop itself looked perfectly healthy.

        Memoized on the sample's own timestamp, so a tick that both records
        and drives the encoder retargets once, not three times.
        """
        if sample is None:
            return None
        if sample.get("pose_smpl_joints_local") is not None:
            return {
                "smpl_joints_local": sample.get("pose_smpl_joints_local"),
                "smpl_root_quat": sample.get("pose_smpl_root_quat"),
                "anchor_6d": sample.get("pose_anchor_6d"),
                "wrist_joints": sample.get("pose_wrist_joints"),
            }
        raw = sample.get("raw_smpl")
        if raw is None:
            return None
        ts = sample.get("timestamp")
        if self._pose_data_cache is not None and ts is not None and ts == self._pose_data_ts:
            return self._pose_data_cache
        try:
            data = process_smpl_pose_full(raw)
        except Exception as e:
            # Same policy as the source's own pipeline: a retargeting glitch
            # must not take VR_3PT (or the recording) down with it.
            print(f"[g1_bridge] process_smpl_pose_full failed: {e}")
            return None
        self._pose_data_cache = data
        self._pose_data_ts = ts
        return data

    @staticmethod
    def _smpl_pose_from_sample(sample) -> np.ndarray | None:
        """``teleop.smpl_pose`` (63,) from the PICO sample's raw (24,7) SMPL
        body poses, or None when no SMPL stream is present (mock/keyboard
        sources, or a real source before the operator's SMPL tracking is up)."""
        raw = sample.get("raw_smpl") if sample else None
        if raw is None:
            return None
        from sim2real.teleop.smpl_stream import smpl_body_pose_aa_from_xrobot_raw

        body_pose_aa, _root_quat = smpl_body_pose_aa_from_xrobot_raw(
            np.asarray(raw, dtype=np.float32)
        )
        return np.asarray(body_pose_aa, dtype=np.float32).reshape(-1)

    def _handle_planner_key(self, ctx, keycode):
        """Arrow-key / gait-select / facing / speed / height keys, ported
        from WholeBodyG1Composite.handle_keypress. Only reachable once
        kPlanner is armed -- see KEY_HELP for the full list."""
        ms = self._movement_state
        dirty = True
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
        elif keycode == ord(","):
            ms.facing_direction = self._rotate_yaw(
                ms.facing_direction, +self._FACING_YAW_STEP
            )
        elif keycode == ord("."):
            ms.facing_direction = self._rotate_yaw(
                ms.facing_direction, -self._FACING_YAW_STEP
            )
        elif ord("0") <= keycode <= ord("9"):
            ms.locomotion_mode = int(keycode - ord("0"))
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
            self._planner_initialized = False
            print("[kPlanner] re-init requested via 'N'.")
        elif keycode == ord("/"):
            print(_composite_key_help())
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
        return True

    def _init_kplanner(self, cfg):
        """Build/fetch the cached SonicKPlanner + (re)seed MovementState from
        config. No-op (self._kplanner stays None) if the config has no
        planner_path -- PLANNER mode then just falls through to _hold_pose(),
        same as before this port for a config that hasn't opted in."""
        planner_path = cfg.get("planner_path")
        if not planner_path:
            self._kplanner = None
            return
        planner_path = resolve_ckpt_path(planner_path)
        default_height = float(cfg.get("planner_default_height", 0.788740))
        seed = int(cfg.get("planner_seed", 1234))
        key = ("kplanner", planner_path, default_height, seed)
        if _KPLANNER_CACHE["key"] == key and _KPLANNER_CACHE["obj"] is not None:
            kp = _KPLANNER_CACHE["obj"]
            if hasattr(kp, "reset"):
                kp.reset()
        else:
            kp = SonicKPlanner(
                planner_path=planner_path, default_height=default_height, seed=seed
            )
            _KPLANNER_CACHE.update(key=key, obj=kp)
        self._kplanner = kp

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

    def _lazy_init_planner_if_needed(self, ctx):
        """Initialise the kPlanner from the current robot pose on first tick."""
        if self._planner_initialized or self._kplanner is None:
            return
        qpos, _ = ctx.read_joint_state()
        quat, _ = ctx.read_pelvis_state()
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
            self._last_replan_time = ctx.clock_s()
            n = self._kplanner.num_frames
            print(
                f"[kPlanner] initialised — mode={self._movement_state.locomotion_mode} "
                f"({n} frames @ 50Hz, {n / 50.0:.2f}s)"
            )

    def _tick_planner(self, sim_time):
        """Run the 10Hz kPlanner check: replan if needed, advance frame cursor."""
        if not self._planner_initialized or self._kplanner is None:
            return
        # ``sim_time <`` guard: see the identical one on need_policy in
        # run_controller. A rewound sim clock (MuJoCo divergence recovery)
        # otherwise parks _last_planner_check_time in the future and this
        # 10 Hz tick never fires again -- kPlanner stops replanning for the
        # rest of the episode.
        rewound = sim_time < self._last_planner_check_time
        if (
            not rewound
            and sim_time - self._last_planner_check_time < self._PLANNER_DT - 1e-6
        ):
            return
        dt = self._PLANNER_DT if rewound else sim_time - self._last_planner_check_time
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

    def _robot_heading_yaw(self, ctx) -> float:
        """Yaw (rad) of the robot's current heading, from the pelvis quat."""
        quat, _ = ctx.read_pelvis_state()  # wxyz
        return heading_yaw_from_quat(quat)

    def _resync_facing_to_robot(self, ctx) -> None:
        """Re-zero the yaw accumulator + facing_direction to the robot's
        ACTUAL current heading. Called on entry into PLANNER from a
        non-PLANNER mode so steering is angular-velocity-like."""
        yaw = self._robot_heading_yaw(ctx)
        self._yaw_angle_rad = yaw
        self._movement_state.facing_direction = np.array(
            [np.cos(yaw), np.sin(yaw), 0.0], dtype=np.float32
        )

    def _tick_pico_planner_input(self, ctx, buttons: dict | None) -> None:
        """Drive SLOW_WALK with the PICO sticks; use IDLE when both are neutral.

        Button chords only select the manager mode, never the walking gait.
        In particular, releasing ABXY unevenly must not turn a remaining A+B
        into a SLOW_WALK -> WALK change.
        """
        if buttons is None or self._kplanner is None:
            return
        now = ctx.clock_s()
        # ``now <`` guard: same rewound-sim-clock hazard as need_policy /
        # _tick_planner. This one is the most user-visible of the three --
        # once _last_pico_tick_time is stranded in the future this returns
        # early every tick, so the joysticks go completely dead in PLANNER
        # mode while buttons (read elsewhere, unthrottled) keep working.
        if (
            now >= self._last_pico_tick_time
            and now - self._last_pico_tick_time < self._PICO_INPUT_DT - 1e-6
        ):
            return
        self._last_pico_tick_time = now
        ms = self._movement_state
        dirty = False

        lx, ly = buttons.get("left_axis", (0.0, 0.0))
        rx, _ry = buttons.get("right_axis", (0.0, 0.0))

        if abs(rx) >= self._JOYSTICK_DEADZONE:
            self._yaw_angle_rad += self._YAW_GAIN * (-rx) * self._PICO_INPUT_DT
            new_facing = np.array(
                [np.cos(self._yaw_angle_rad), np.sin(self._yaw_angle_rad), 0.0],
                dtype=np.float32,
            )
            if not np.allclose(ms.facing_direction, new_facing, atol=1e-3):
                ms.facing_direction = new_facing
                dirty = True

        raw_mag = float(np.clip(np.hypot(lx, ly), 0.0, 1.0))
        pushed = raw_mag >= self._JOYSTICK_DEADZONE
        turning = abs(rx) >= self._JOYSTICK_DEADZONE
        if pushed or turning:
            desired_mode = int(LocomotionMode.SLOW_WALK)
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
            new_speed = 0.1 + 0.5 * mag
            denom = raw_mag if raw_mag > 0 else 1.0
            scale = mag / denom
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

        if _PICO_INPUT_DEBUG and now - getattr(self, "_pico_dbg_t", -1e9) >= 1.0:
            self._pico_dbg_t = now
            print(
                f"[PicoInput][debug] axes L=({lx:+.3f},{ly:+.3f}) R.x={rx:+.3f} "
                f"| deadzone={self._JOYSTICK_DEADZONE} pushed={pushed} turning={turning} "
                f"| gait={self._locomotion_name(ms.locomotion_mode)} "
                f"loco_mode={ms.locomotion_mode} speed={ms.movement_speed:+.3f} "
                f"|dir|={float(np.linalg.norm(ms.movement_direction)):.3f} "
                f"planner={'ready' if self._kplanner is not None else 'None'}",
                flush=True,
            )

        if dirty:
            self._movement_state_dirty = True

    def _build_kplanner_payload(self):
        """Build a push_payload()-compatible reference frame from the
        kPlanner's current motion cursor. Mirrors _step_generic's GMR payload
        shape (joint_pos/body_pos_w/body_quat_w/joint_names/body_names) --
        that's the interface sim2real's own SONIC observation code already
        knows how to consume (only needs one body entry, root_body_name=
        "pelvis" per the g1 policy configs printed at armed-time). Unlike
        WBT's _update_planner_reference (which pushes IsaacLab-order
        joint_pos_29/motion_anchor_orientation/root_z straight into its own
        vendored controller's set_control_var), this composite's sonic
        backend runs sim2real's own BridgeTracking, whose reference input is
        this higher-level pose payload -- the anchor/orientation math happens
        inside sim2real's own encoder-input construction, not here.

        Simplification vs. WBT: no IDLE drift special-case (WBT holds a
        static default pose for IDLE to avoid ~0.14 rad waist_pitch drift
        over a replayed clip; here IDLE is simply a STATIC_MODES entry so the
        frame cursor never advances -- see the call site -- which avoids the
        drift-by-replanning-from-a-later-frame case but not necessarily
        every source of drift WBT worked around. Flagged as a known
        simplification, not verified against real hardware."""
        if self._kplanner is None or self._kplanner.planner_motion_50hz is None:
            return None
        frame_qpos = self._kplanner.sample_frame(self._current_motion_frame)  # (36,)
        return {
            "joint_pos": frame_qpos[7:].astype(np.float32).tolist(),
            "body_pos_w": [frame_qpos[0:3].astype(np.float32).tolist()],
            "body_quat_w": [frame_qpos[3:7].astype(np.float32).tolist()],
            "joint_names": list(G1_JOINT_NAMES),
            "body_names": ["pelvis"],
        }

    def _blend_planner_payload(self, ctx, payload, sim_time):
        """Ease the kPlanner reference in from the pose being held.

        Entering PLANNER swaps the reference from the frozen standing default
        to frame 0 of a freshly planned trajectory, in one tick. kPlanner plans
        from the robot's current *root*, so the root is continuous, but the
        joint angles are whatever the gait's first frame happens to be -- which
        is not the standing pose the robot is actually in.

        Two things happen here, and only on the entry edge:

        * the first blended frame is stamped ``motion_first_frame``. sonic has
          never stamped anything (grep: the constant appears only in the GMR
          backend), so ``RealtimeMotionBuffer`` was not even doing its yaw/xy
          re-attachment at this switch -- contrary to what
          ``Sim2RealBackend.seed_reference`` documents as the contract.
        * the joint angles ramp from the held pose to the planner's over
          ``sim2real_pose_blend_s``.

        Emitted in the kPlanner payload's own single-body shape rather than
        FK'd through DefaultReferenceFrame: that frame declares 33 bodies and
        this one declares 1, and a layout change makes the buffer drop its
        frames and reset the segment transform -- exactly at the handover that
        needs them.

        Alpha runs on sim time here (there is no motion-stream clock on this
        path -- the payload is generated locally, not received).
        """
        if self._blend_s <= 0.0:
            return payload
        frame = qpos_from_payload(payload, 0)
        if self._planner_blend_pending:
            anchor = self._planner_anchor_qpos
            if anchor is None:
                anchor = ctx.settle_qpos
            if anchor is None:
                anchor = default_reference_frame(
                    ctx.tracking.robot_cfg
                ).default_qpos_at(
                    ctx.read_root_pos(),
                    ctx.read_pelvis_state()[0],
                    ctx.standing_joints(),
                )
            self._planner_blender.begin(anchor, frame, int(sim_time * 1e9))
            self._planner_blend_pending = False
            stamped = True
        else:
            stamped = False

        out, _alpha = self._planner_blender.step(frame, int(sim_time * 1e9))
        self._planner_anchor_qpos = np.asarray(out, dtype=np.float64).reshape(-1)
        blended = {
            **payload,
            "joint_pos": out[7:].astype(np.float32).tolist(),
            "body_pos_w": [out[0:3].astype(np.float32).tolist()],
            "body_quat_w": [out[3:7].astype(np.float32).tolist()],
        }
        if stamped:
            blended[MOTION_FIRST_FRAME_KEY] = True
        return blended

    def _step_kplanner_pose(self, ctx, qpos, sim_time, buttons):
        """PLANNER mode (sonic backend, g1 split-onnx flavor only): drive
        locomotion via the ported SonicKPlanner."""
        return self._step_planner_family(ctx, qpos, sim_time, buttons, "g1")

    def _step_vr_3pt(self, ctx, qpos, sim_time, buttons):
        """VR_3PT mode: the same kPlanner lower body as PLANNER, with the
        operator's calibrated 3-point PICO pose steering the upper body
        through the encoder's teleop sub-encoder (mode_id=1).

        The lower-body slots the teleop block wants are the leg subset of the
        very reference PLANNER already pushes into the motion buffer, so the
        only additions over _step_kplanner_pose are the 3-point channel and
        the encoder mode -- see encoder_obs._build_teleop_obs.
        """
        return self._step_planner_family(ctx, qpos, sim_time, buttons, "teleop")

    def _step_planner_family(self, ctx, qpos, sim_time, buttons, encoder_mode):
        """Shared body of PLANNER and VR_3PT. ``encoder_mode`` picks which
        sub-encoder consumes the planner reference this tick."""
        self._lazy_init_planner_if_needed(ctx)
        self._tick_planner(sim_time)
        self._tick_pico_planner_input(ctx, buttons)
        payload = self._build_kplanner_payload()
        if payload is None:
            return ctx.hold_pose()
        # Put the unitoken encoder back in g1 mode. Mandatory, not defensive:
        # the mode is sticky, POSE sets it to smpl, and PLANNER is dispatched
        # here rather than through _step_sonic_pose (which owns the other
        # half of the switch). Without this, an OFF -> PLANNER -> POSE ->
        # PLANNER round trip leaves the encoder reading its smpl blocks
        # forever, so the joint-angle reference pushed just below -- which
        # only ever lands in the g1 blocks -- is silently ignored: joysticks
        # move MovementState and kPlanner replans a perfectly good walking
        # trajectory, and the robot still does not walk.
        enc_obs = self._sonic_encoder_obs(ctx)
        if enc_obs is not None:
            if encoder_mode == "teleop":
                # Backstop for on_mode_change, same role as POSE's: never fall
                # back to g1 silently -- the operator would see the legs walk
                # and the arms ignore them, with nothing in the log.
                self._reject_vr_3pt_without_teleop_encoder(ctx, enc_obs)
                self._push_vr_3point(ctx, enc_obs)
            enc_obs.set_encoder_mode(encoder_mode)
        ctx.tracking.motion_buffer.push_payload(
            self._blend_planner_payload(ctx, payload, sim_time)
        )
        target_q, kps, kds = ctx.step_policy(qpos)
        if (
            self._kplanner is not None
            and self._kplanner.num_frames > 0
            and self._movement_state.locomotion_mode not in STATIC_MODES
        ):
            self._current_motion_frame = min(
                self._current_motion_frame + 1, self._kplanner.num_frames - 1
            )
        return target_q, kps, kds

    @staticmethod
    def _locomotion_name(mode_id: int) -> str:
        """Symbolic name for a locomotion mode id, or 'UNKNOWN(<id>)'.
        LocomotionMode is a plain class (not IntEnum), so reverse-look it up
        from the class attributes.
        """
        for name, value in vars(LocomotionMode).items():
            if not name.startswith("_") and isinstance(value, int) and value == mode_id:
                return name
        return f"UNKNOWN({mode_id})"

    def _reset_kplanner_state(self):
        self._yaw_angle_rad = 0.0
        self._last_pico_tick_time = -1e9
        # Force kPlanner re-initialisation on the next tick.
        self._planner_initialized = False
        self._current_motion_frame = 0
        self._last_planner_check_time = -1e9
        self._last_replan_time = -1e9
        self._replan_interval_counter = 0.0

    @staticmethod
    def _encoder_supports(enc_obs, mode: str) -> bool:
        """Whether this checkpoint's ``g1_input`` class can populate ``mode``.

        Asked of the observation rather than inferred from its type: the block
        layout is what decides, and the observation is the only object that
        knows it. Both shipping classes support g1+smpl; neither wires up the
        VR 3-point teleop sub-encoder.
        """
        return mode in getattr(enc_obs, "SUPPORTED_ENCODER_MODES", ("g1",))

    def _reject_vr_3pt_without_teleop_encoder(self, ctx, enc_obs=None) -> None:
        """Refuse VR_3PT on a checkpoint whose teleop sub-encoder isn't wired
        up. Mirrors _reject_pose_without_smpl_encoder, and for the same
        reason: an unpopulated teleop block is a perfectly valid encoder
        input, so the failure would otherwise be silent."""
        if enc_obs is None:
            enc_obs = self._sonic_encoder_obs(ctx)
        if enc_obs is not None and not self._encoder_supports(enc_obs, "teleop"):
            raise RuntimeError(
                "VR_3PT teleop needs the teleop sub-encoder (mode_id=1), which "
                f"{type(enc_obs).__name__} does not populate -- its teleop "
                "block would stay zero and the arms would ignore you."
            )

    def _push_vr_3point(self, ctx, enc_obs) -> None:
        """Feed this tick's calibrated 3-point PICO pose to the teleop
        sub-encoder. A missing sample leaves the previous one in place: the
        encoder buffer is self-anchored (identity quats at reset), and holding
        the last operator pose beats stepping the arms to a default mid-motion.
        """
        sample = ctx.pico_source.get_latest() if ctx.pico_source is not None else None
        if sample is None:
            return
        target = sample.get("vr_3point_local_target")
        orn = sample.get("vr_3point_local_orn_target")
        if target is None or orn is None:
            return
        enc_obs.set_vr_3point(target, orn)

    def _reject_pose_without_smpl_encoder(self, ctx, enc_obs=None) -> None:
        """Refuse POSE on a checkpoint whose smpl sub-encoder isn't wired up.

        No shipping variant is in that position now that sonic_v1_1 has one, but
        the guard stays: an unpopulated smpl block is a perfectly valid encoder
        input that produces a token the robot will follow into nothing, with no
        runtime symptom to notice. This is the only place it can be caught, and
        the next variant is one manifest away.
        """
        if enc_obs is None:
            enc_obs = self._sonic_encoder_obs(ctx)
        if enc_obs is not None and not self._encoder_supports(enc_obs, "smpl"):
            raise RuntimeError(
                "POSE ('follow me') teleop needs the smpl sub-encoder, which "
                f"{type(enc_obs).__name__} does not populate -- its smpl block "
                "would stay zero and the robot would track nothing."
            )

    def _sonic_encoder_obs(self, ctx):
        """The live split-onnx ``g1_input`` observation instance sim2real built
        from policy.yaml's ``g1_input`` group -- whichever encoder variant the
        manifest names (see ``_SPLIT_ONNX_ENCODER_OBS``). Resolved lazily (the
        observation objects only exist after Tracking construction) and
        cached; returns None for any checkpoint that doesn't use one."""
        if self._sonic_enc_obs is None:
            group = getattr(ctx.tracking, "observations", {}).get("g1_input")
            funcs = getattr(group, "funcs", {}) if group is not None else {}
            for obs in funcs.values():
                if isinstance(obs, _SPLIT_ONNX_ENCODER_OBS):
                    self._sonic_enc_obs = obs
                    break
        return self._sonic_enc_obs

    def _step_sonic_pose(self, ctx, qpos, mode):
        """Drive the body 29-joint target from sim2real's SONIC checkpoint.

        g1 split-onnx flavor (SplitOnnxBridgeTracking) -- the low_latency
        **unitoken** encoder, one graph holding all three sub-encoders (see
        sonic_g1_encoder_obs.py). Which one runs is chosen here:

        * POSE: smpl sub-encoder (mode_id=2), fed the operator's live SMPL
          body pose from PICO. This is exactly what the official deployment
          does for "follow me" -- ``pico_manager_thread_server.py`` streams
          smpl_joints/smpl_anchor plus a 6-value wrist vector and leaves the
          other 23 entries of its ``joint_pos`` at zero, i.e. no full-body
          retargeting is involved and no GMR subprocess is needed. Mirrors
          WholeBodyG1Composite._update_pose_reference, including its
          heading-alignment capture on POSE entry.
        * every other mode: g1 sub-encoder (mode_id=0) against the composite's
          default standing frame (``g1_bridge/default_reference.py``) --
          the same pose ``RealtimeMotionBuffer``'s built-in frame is derived
          from, but placed at the robot's own xy/heading instead of at the world
          origin. The built-in frame bakes in an absolute world pose unrelated
          to wherever this scene actually spawned the robot, and was confirmed
          (live diagnostic) to drive the policy into continuous
          toppling/twitching.

        smpl flavor (a separate, fused-ONNX checkpoint): fed by tele's own
        PICO SMPL sample through sim2real's smpl motion buffer, pushed each
        tick.
        """
        if self._flavor == FLAVOR_SPLIT_ONNX:
            # Keep a valid frame in the motion buffer regardless of mode: the
            # g1 sub-observations still run their update() every tick (they
            # raise if motion_data is None), even when POSE ignores their
            # output. Idempotent -- see _push_synthetic_default_once.
            ctx.push_synthetic_default_once(qpos)
            enc_obs = self._sonic_encoder_obs(ctx)
            if mode == ManagerMode.POSE:
                # Backstop for on_mode_change: fail before _push_pose_reference
                # calls set_pose_reference() three frames deeper, and above all
                # before falling through to g1 mode -- a silent fallback reads
                # as POSE working badly rather than POSE not existing.
                self._reject_pose_without_smpl_encoder(ctx, enc_obs)
            if mode == ManagerMode.POSE and self._push_pose_reference(ctx, enc_obs):
                enc_obs.set_encoder_mode("smpl")
            elif enc_obs is not None:
                enc_obs.set_encoder_mode("g1")
            return ctx.step_policy(qpos)

        sample = ctx.pico_source.get_latest() if ctx.pico_source is not None else None
        if sample is None or sample.get("raw_smpl") is None:
            return ctx.hold_pose()

        wrist_joints = sample.get("pose_wrist_joints")
        if wrist_joints is None:
            wrist_joints = process_smpl_pose_full(sample["raw_smpl"])["wrist_joints"]
        payload = build_smpl_payload(
            sample["raw_smpl"],
            wrist_joints,
            human_joints_info_path=self._human_joints_info_path,
        )
        ctx.tracking.motion_buffer.push_payload(payload)
        return ctx.step_policy(qpos)

    def _push_pose_reference(self, ctx, enc_obs) -> bool:
        """Feed this tick's operator pose into the smpl sub-encoder. Returns
        False (caller falls back to g1 mode) if no usable PICO pose is
        available -- e.g. the mock source, or before the headset streams
        body data.

        Verbatim port of ``WholeBodyG1Composite._update_pose_reference``,
        which is itself the sim-side mirror of the official deployment:

        * the encoder's ``smpl_anchor_orientation`` is NOT the raw SMPL root
          6D -- the deploy's ``GatherMotionAnchorOrientationMutiFrame``
          (orientation_mode=0) recomputes it live as the 6D of
          ``conj(robot_base_quat) * smpl_root_quat``, i.e. the reference root
          expressed relative to the robot's CURRENT base. Recomputed here
          every tick via the same ``_relative_anchor6`` the working PLANNER
          path uses, never taken from the sample's cached ``pose_anchor_6d``
          when a root quat is available.
        * on the first tick after entering POSE, capture a world-frame
          offset R with ``R * smpl_root == robot_quat`` so the anchor reads
          identity at the switch instant. Without it the robot snaps from
          PLANNER's commanded facing to the operator's absolute torso yaw --
          the same phantom-heading failure mode as the reset-ordering bug
          fixed in _push_synthetic_default_once.
        """
        if enc_obs is None or ctx.pico_source is None:
            return False
        sample = ctx.pico_source.get_latest()
        if sample is None:
            return False

        # No freshness gate here, deliberately: this is the control path, and
        # holding the last operator pose is a better failure mode for a robot
        # mid-motion than dropping to g1 mode. Only the recording zeroes out.
        pose_data = self._pose_data(sample) or {}
        smpl_joints_local = pose_data.get("smpl_joints_local")
        smpl_root_quat = pose_data.get("smpl_root_quat")
        anchor6 = pose_data.get("anchor_6d")
        wrist_joints = pose_data.get("wrist_joints")
        if smpl_joints_local is None or anchor6 is None or wrist_joints is None:
            return False

        if smpl_root_quat is not None:
            robot_quat, _ = ctx.read_pelvis_state()
            if self._pose_align_pending:
                self._pose_root_align = _quat_mul(robot_quat, _quat_inv(smpl_root_quat))
                self._pose_align_pending = False
                print("[g1_bridge] POSE heading aligned to current robot facing")
            if self._pose_root_align is not None:
                smpl_root_quat = _quat_mul(self._pose_root_align, smpl_root_quat)
            # The frame the loaded checkpoint's smpl anchor slot expects --
            # body for low_latency, heading (yaw-only) for v1_1. They agree
            # while the operator is upright, so getting this wrong is invisible
            # right up until someone leans; ask, never assume.
            anchor6 = _relative_anchor6(
                robot_quat,
                smpl_root_quat,
                frame=getattr(enc_obs, "POSE_ANCHOR_FRAME", "body"),
            )

        smpl_joints_local, wrist_joints, anchor6 = self._blend_pose_reference(
            ctx, smpl_joints_local, wrist_joints, anchor6
        )
        enc_obs.set_pose_reference(smpl_joints_local, anchor6, wrist_joints)
        return True

    def _capture_standing_smpl(self, ctx) -> None:
        """Remember the operator's pose at the engage chord as "standing"."""
        pose = self._read_operator_pose(ctx)
        if pose is None:
            self._standing_smpl = None
            return
        joints, wrists = pose
        self._standing_smpl = (joints.copy(), wrists.copy())

    def _read_operator_pose(self, ctx):
        """(smpl_joints_local, wrist_joints) from the live PICO sample, or None."""
        if ctx.pico_source is None:
            return None
        pose_data = self._pose_data(ctx.pico_source.get_latest()) or {}
        joints = pose_data.get("smpl_joints_local")
        wrists = pose_data.get("wrist_joints")
        if joints is None or wrists is None:
            return None
        return (
            np.asarray(joints, dtype=np.float32),
            np.asarray(wrists, dtype=np.float32),
        )

    def _blend_pose_reference(self, ctx, smpl_joints_local, wrist_joints, anchor6):
        """Ramp the operator's pose in from the standing one, on POSE entry.

        This is the "猛弹" the operator docs warn about: at the switch the
        reference jumps from whatever the g1 encoder block was commanding (a
        standing pose) to the operator's live full-body pose, which may be
        nothing like it. ``_pose_root_align`` above already removes the
        *orientation* discontinuity; this removes the shape one.

        The blend runs in the encoder's own input space -- 24x3 root-local SMPL
        joint positions plus the 6 wrist angles -- because that is the only
        space the two ends share. Cross-fading the encoder's g1 and smpl blocks
        instead would be out of distribution: ``encoder_mode_4`` is a one-hot
        and the blocks are disjoint slices, never both populated in training.

        The anchor is the operator's pose at the engage chord (see
        ``_capture_standing_smpl``), NOT the robot's current pose -- there is no
        robot->SMPL retargeting available to build the latter. Those agree only
        because both are "standing", which is exactly what the chord's
        calibration posture guarantees. When no anchor was captured the ramp is
        skipped rather than guessed at.

        ``anchor6`` is left alone: it is derived from the root quaternion one
        line above and is already continuous through ``_pose_root_align``.
        """
        if self._blend_s <= 0.0 or self._standing_smpl is None:
            self._pose_blend_pending = False
            return smpl_joints_local, wrist_joints, anchor6

        now = ctx.clock_s()
        if self._pose_blend_pending:
            self._pose_blend_t0 = now
            self._pose_blend_pending = False
            print(
                "[g1_bridge] POSE shape ramping in from the standing "
                f"pose over {self._blend_s:.1f}s"
            )
        if self._pose_blend_t0 is None:
            return smpl_joints_local, wrist_joints, anchor6

        alpha = float(np.clip((now - self._pose_blend_t0) / self._blend_s, 0.0, 1.0))
        if alpha >= 1.0:
            self._pose_blend_t0 = None
            return smpl_joints_local, wrist_joints, anchor6

        stand_joints, stand_wrists = self._standing_smpl
        joints = np.asarray(smpl_joints_local, dtype=np.float32)
        wrists = np.asarray(wrist_joints, dtype=np.float32)
        if stand_joints.shape != joints.shape or stand_wrists.shape != wrists.shape:
            # Operator pose layout changed mid-session; don't blend garbage.
            self._pose_blend_t0 = None
            return smpl_joints_local, wrist_joints, anchor6
        return (
            (1.0 - alpha) * stand_joints + alpha * joints,
            (1.0 - alpha) * stand_wrists + alpha * wrists,
            anchor6,
        )
