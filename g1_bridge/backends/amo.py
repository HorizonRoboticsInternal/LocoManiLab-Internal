"""The AMO backend: command-velocity locomotion + 3-point IK arm teleop.

Unlike the motion-tracking backends, AMO takes no reference motion at all. The
legs and waist come from a TorchScript policy driven by PICO joystick velocity
commands; the shoulders/elbows/wrists are overwritten with targets from an
incremental DLS IK solve against the operator's 3-point VR pose.

Ported from the standalone ``AMOG1Composite`` (deleted -- it had exactly one
consumer and that path never actually reached it; see the Stage 3 commit).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
from scipy.spatial.transform import Rotation as R
from termcolor import colored

from g1_bridge.g1_runtime import G1_JOINT_NAMES
from g1_bridge.pico.vr_source import (
    REF_GESTURE_CTRL_POS,
    REF_GESTURE_CTRL_ROT,
    head_relative_pose,
)

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from sim2real.rl_policy.families import get_family
from sim2real.utils.dex3 import grip_to_joint_targets

#: sim2real's Tracking runs at 50 Hz; the composite throttles policy inference
#: to that rate, and the joystick command integrator below uses the same dt.
_POLICY_DT = 0.02

from g1_bridge.backends.base import (
    BackendResources,
    Sim2RealBackend,
    Sim2RealContext,
    TickInput,
    read_hand_state,
)

# AMO's own 23-DOF order (legs + waist + shoulder/elbow) and the 6 wrist
# joints it doesn't model at all -- verbatim from AMOG1Composite.
_AMO_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint", "left_elbow_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint",
)

_AMO_ARM_NAMES = _AMO_JOINT_NAMES[15:]  # shoulder/elbow only (8) -- IK-driven

_AMO_WRIST_NAMES = (
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

_AMO_DEFAULT_DOF_POS = np.array(
    [
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        -0.1, 0.0, 0.0, 0.3, -0.2, 0.0,
        0.0, 0.0, 0.0,
        0.5, 0.0, 0.2, 0.3,
        0.5, 0.0, -0.2, 0.3,
    ],
    dtype=np.float32,
)

# IK reduced-model joint order: L[sh p,r,y, elbow, wr r,p,y], R[same] (14).
# AMO arm_action order (8): L[sh p,r,y, elbow], R[same].
# _AMO_WRIST_NAMES order: L[wr r,p,y], R[wr r,p,y] (6).
_IK_TO_AMO_ARM = [0, 1, 2, 3, 7, 8, 9, 10]   # IK14 -> 8 shoulder/elbow

_IK_TO_WRIST = [4, 5, 6, 11, 12, 13]         # IK14 -> 6 wrist

# The pose _reset_amo writes into sim, in IK-14 order: AMO's arm defaults with
# the wrists at zero. Used to prime the IK's output filter at episode reset.
_AMO_DEFAULT_Q14 = np.array(
    [0.5, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0, 0.5, 0.0, -0.2, 0.3, 0.0, 0.0, 0.0],
    dtype=float,
)

_AMO_ARM_POS_SCALE = 1.0    # operator hand delta -> robot wrist delta scale

# --------------------------------------------------------------------------- #
# Arm-teleop safety envelope
#
# The operator's hand delta is applied 1:1 with no bound on where it lands, so
# before the solver was fixed the first ordinary reaching motion put the IK
# target outside the arm's reachable set and the solve went chaotic (measured:
# 115 rad/s peak commanded joint velocity at a 0.50 m reach, vs 0.5 rad/s at
# 0.10 m). The solver fix (g1_arm_ik.py) is what removed that; the three
# constants below bound the target, the output, and the blast radius of a
# single bad solve, so that no *future* pathological input can reach the
# operator as a twitch. Ablated on the fixed solver, the rate limit is the one
# that still does real work (unreachable wrist orientations: 8.97 -> 3.00 rad/s
# and 2 twitch events -> 0); the other two are cheap insurance that stays inert.
# --------------------------------------------------------------------------- #

# Max distance from the shoulder-pitch origin to the wrist EE target, in m.
# A RUNAWAY GUARD, not a workspace model -- deliberately set just outside the
# arm's true maximum reach (measured on this model: 0.3646 m at the rest pose,
# 0.4598 m absolute maximum over all 7 joints of one arm), so it is inert
# during every reachable motion and only bounds targets that have left the
# building. It exists because _rel_target integrates the operator's hand delta
# with no bound at all: walk your hands away and the target goes to infinity.
#
# It is deliberately NOT tuned down to the reachable boundary. Measured with
# the fixed solver, tightening it to 0.42 buys almost nothing (peak commanded
# velocity 1.62 -> 1.41 rad/s on a 0.50 m reach) but saturates 63% of an
# ordinary 0.25 m reach -- i.e. it would freeze the operator's last few cm of
# travel in exchange for noise. Unreachable targets are better handled where
# they already are: the IK's position-over-orientation weighting stretches the
# arm as far as it goes, which is the behaviour an operator expects.
_AMO_ARM_REACH_MAX = 0.50

# Max commanded arm joint speed, rad/s, applied to the IK solution before it
# becomes a PD target. ~2x the fastest joint rate hand-tracked teleop actually
# produces, so it is inert in normal use and only bounds nonsense.
_AMO_ARM_VEL_LIMIT = 3.0

# Per-joint jump threshold in the IK-14 order [sh_pitch, sh_roll, sh_yaw, elbow,
# wr_roll, wr_pitch, wr_yaw] x 2 -- verbatim from Psi0's safelySetMotor
# (master_whole_body.py:230-242), whose 29-DOF slice 15..28 has this same
# layout. Note wrist_roll deliberately gets the tight pi/3; only wrist
# pitch/yaw get pi.
_AMO_JUMP_THRESH = np.array(([np.pi / 3] * 5 + [np.pi] * 2) * 2)

# Consecutive rejected solves before we stop holding and disengage arm teleop.
# Psi0 has no such escape: its guard holds the last command forever, so a
# persistently bad solution freezes the arms with no operator recourse. 25 ticks
# = 0.5 s at 50 Hz.
_AMO_JUMP_REJECT_MAX = 25

#: Per-tick IK diagnostics -> ring buffer + 1 Hz summary + .npz on disengage.
#: Same env-var convention as sonic's _PICO_INPUT_DEBUG.
_AMO_IK_DEBUG = os.environ.get("SIM2REAL_AMO_IK_DEBUG", "0").lower() not in (
    "0",
    "",
    "false",
    "no",
)

#: The three device poses logged alongside each solve, in this row order.
#: A subset of vr_source.DEVICE_POSE_KEYS -- the two non-pose entries there
#: (vr_pose_valid, vr_pose_stamp_ns) are not part of the (3, 7) block.
_LOG_DEVICE_POSE_KEYS = (
    "vr_head_pose",
    "vr_left_ctrl_pose",
    "vr_right_ctrl_pose",
)


# --------------------------------------------------------------------------- #
# Absolute wrist teleop
#
# The arm targets come from the operator's HEAD-RELATIVE controller pose, not
# from the SMPL 3-point set. The 3-point data is SONIC-encoder input: measured,
# it is expressed relative to whatever pose the operator happened to hold when
# ThreePointPose calibration fired, so the same physical hand motion produced
# a differently-rotated robot motion on every session (30 deg about x came out
# as 30 deg about x, or about (-0.5,0,-0.866), or about (-0.82,-0.57,0),
# depending only on the calibration stance). It also carries no anchoring to
# any robot link frame at all -- _capture_calibration references the operator's
# wrist to IDENTITY, with a live "TODO(phase2): use real G1 FK".
#
# Here the anchor is real G1 FK: one calibration gesture ties the operator's
# controller frame to the robot's own EE frame at q14 = 0, and from then on the
# mapping is absolute -- palm orientation in, palm orientation out, independent
# of the robot's pose when the operator engaged.
# --------------------------------------------------------------------------- #

#: Controller-frame -> EE-frame remap, RIGHT-multiplied (`R_target = R_rel @ M`),
#: used until a calibration gesture replaces it. Derived from the documented
#: reference gesture so the two cannot disagree; that gesture's rotation is in
#: turn Psi0's hand2inspire prior. It is a PRIOR, not a measurement for a HELD
#: CONTROLLER -- expect a systematic wrist offset until the operator calibrates.
_M_DEFAULT = {s: R.T for s, R in REF_GESTURE_CTRL_ROT.items()}

#: Calibration chord: both menu buttons, held. Menu is the only input this
#: backend does not already use (ABXY = torso pitch/roll, grips = torso yaw,
#: right stick click = arm engage, triggers = hands, sticks = locomotion, left
#: stick click = ModeManager), and the hold makes it un-hittable by accident.
_AMO_CALIB_HOLD_S = 1.0
_AMO_CALIB_FRAMES = 30

_AMO_ARM_SOURCES = ("ik", "gmr")


def resolve_arm_source(cli_value: str | None, pico_source: str) -> str:
    """The single place ``amo_arm_source`` is decided, for a whole run.

    Precedence: an explicit ``--amo-arm-source`` wins, then
    ``SIM2REAL_AMO_ARM_SOURCE``, then a default that depends on what is
    driving the arms.

    That default is "gmr" **only for real PICO hardware**, and the asymmetry
    is not a hedge -- GMR retargets a 24-joint body skeleton, which only the
    real source produces. Under mock/keyboard there is nothing to retarget, so
    defaulting to "gmr" there would start a worker with no input and park the
    arms. It also matters well outside teleop: ``lab eval`` builds this same
    backend with ``pico_source="mock"`` and drives the arms from a policy
    through ``vla_step`` (which never calls ``_tick_amo_arms`` at all), so a
    blanket "gmr" default would spawn a retargeting subprocess for every
    evaluation -- and hard-fail any machine without the pico venv.

    A typo is refused rather than ignored: silently falling back would look
    exactly like GMR having no effect.
    """
    value = (cli_value or os.environ.get("SIM2REAL_AMO_ARM_SOURCE", "")).strip().lower()
    if not value:
        return "gmr" if pico_source == "real" else "ik"
    if value not in _AMO_ARM_SOURCES:
        raise ValueError(
            f"amo arm source {value!r} is not one of {_AMO_ARM_SOURCES}"
        )
    return value


def _amo_gmr_height_default() -> float:
    value = os.environ.get("SIM2REAL_AMO_GMR_HEIGHT", "").strip()
    return float(value) if value else 1.60


#: Fallback only, for paths that build the backend without going through
#: load_sim2real_controller_config (tests, direct construction). Every real run
#: gets an explicit value in the controller config from resolve_arm_source, and
#: that wins. "ik" here so a path that forgot to resolve degrades to the
#: hardware-free solver rather than to one that needs a subprocess.
_AMO_ARM_SOURCE_DEFAULT = "ik"
_AMO_GMR_HEIGHT_DEFAULT = _amo_gmr_height_default()
#: SIM2REAL_AMO_GMR_VIEWER=1 serves the retargeted robot on localhost (viser),
#: optionally on a fixed port. Off by default: it is a debugging view, and it
#: shares the machine with MuJoCo and the policy.
_AMO_GMR_VIEWER_DEFAULT = os.environ.get("SIM2REAL_AMO_GMR_VIEWER", "0").lower() not in (
    "0",
    "",
    "false",
    "no",
)
_AMO_GMR_VIEWER_PORT_DEFAULT = (
    int(os.environ["SIM2REAL_AMO_GMR_VIEWER_PORT"])
    if os.environ.get("SIM2REAL_AMO_GMR_VIEWER_PORT", "").strip()
    else None
)

_AMO_VR_DEFAULTS = {
    # Where the 14 arm joints come from.
    #
    # "ik"  -- the 3-point DLS solve below: head + two controllers -> EE targets
    #          -> G1ArmIK. Metric, anchored to the controller by wrist
    #          calibration, and the default.
    # "gmr" -- retarget the operator's 24-joint PICO body skeleton instead
    #          (gmr_utils/arm_worker.py). Costs the controller anchoring; buys
    #          the elbow.
    #
    # The difference is not solver quality, it is what the input determines.
    # With only the hand pose fixed, the elbow can still swing about the
    # shoulder-wrist axis: that DoF is *undetermined by 3-point input*, and the
    # solve settles it with a posture prior (_Q_REF) that has never heard of
    # the operator's elbow. Measured on a real 807-tick session (scripts/
    # gmr_arm_ab.py): upper-arm direction error 35-37 deg with a std of 14 deg
    # and a 2.6-67 deg range, uncorrelated with pose -- i.e. not "worse when
    # extended", just arbitrary. GMR sees all 24 joints and matches shoulder/
    # elbow/wrist link orientations, and on the same session tracked to a std
    # of 0.5-1.2 deg. It was also calmer: median commanded joint speed 0.92 vs
    # 1.62 rad/s, and 9.4% vs 19.0% of ticks above the 3.0 rad/s rate limit.
    #
    # Settable per run as SIM2REAL_AMO_ARM_SOURCE=gmr -- the same env-var
    # convention SIM2REAL_AMO_IK_DEBUG and TELE_VR_ARM_CALIB already use, and
    # the only one that reaches here without threading a new flag through both
    # LocoManiLab's CLI and the benchmark's arg parser (amo_torso_control, the
    # existing precedent for an AMO mode switch, has no CLI path either). An
    # explicit controller-JSON key still wins over the env var.
    "amo_arm_source": _AMO_ARM_SOURCE_DEFAULT,
    #: Only read when amo_arm_source == "gmr". Set BELOW the operator's real
    #: height; the offline sweep showed the arm angles barely move with it.
    #: Env override: SIM2REAL_AMO_GMR_HEIGHT.
    "amo_gmr_human_height": _AMO_GMR_HEIGHT_DEFAULT,
    "amo_vr_arm_ori_mode": "absolute",  # "relative" == the pre-fix behaviour
    "amo_vr_arm_pos_mode": "absolute",
    "amo_vr_arm_pos_scale": 1.0,
    "amo_vr_ori_ref": "head_full",  # "head_yaw" ignores head pitch/roll
    "amo_vr_engage_blend_s": 1.5,
    "amo_vr_recover_blend_s": 0.5,
    "amo_vr_pose_stale_s": 0.25,
    "amo_vr_pose_hold_s": 0.5,
    "amo_vr_calib_path": None,
    "amo_vr_calib_max_jitter_deg": 3.0,
    "amo_vr_calib_max_jitter_m": 0.02,
}

_AMO_CALIB_VERSION = 1


def default_calib_path() -> str:
    """Where the wrist calibration lives.

    NOT ``robocasa/macros_private.py``: that file is regenerated wholesale by
    ``robocasa.scripts.setup_macros``, so anything written there is destroyed on
    the next setup run.
    """
    override = os.environ.get("TELE_VR_ARM_CALIB")
    if override:
        return override
    base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "tele_humanoid_benchmark", "vr_arm_calib.json")


def se3_interp(T_a, T_b, alpha):
    """Interpolate two 4x4 SE3 poses: slerp on rotation, lerp on translation.

    ``alpha=0`` returns ``T_a`` exactly, which is the property the engage blend
    relies on -- the first commanded target IS the robot's current pose, so the
    step is zero by construction rather than merely bounded.
    """
    a = float(np.clip(alpha, 0.0, 1.0))
    T = np.eye(4)
    Ra, Rb = T_a[:3, :3], T_b[:3, :3]
    rv = R.from_matrix(Ra.T @ Rb).as_rotvec()
    T[:3, :3] = Ra @ R.from_rotvec(a * rv).as_matrix()
    T[:3, 3] = (1.0 - a) * T_a[:3, 3] + a * T_b[:3, 3]
    return T


def clamp_reach(pos, shoulder, r_max=_AMO_ARM_REACH_MAX):
    """Project an EE target back onto the shoulder-centred reach sphere.

    Returns ``(pos, was_clamped)``. Pure function so the offline smoothness
    test can drive it without a sim.
    """
    v = np.asarray(pos, dtype=float) - shoulder
    d = float(np.linalg.norm(v))
    if d <= r_max or d < 1e-9:
        return np.asarray(pos, dtype=float), False
    return shoulder + v * (r_max / d), True


def rate_limit(q_new, q_prev, max_delta):
    """Bound ``q_new`` to ``max_delta`` (inf-norm) from ``q_prev``.

    Scales the WHOLE vector by one factor (Psi0's ``delta / max(motion_scale,
    1)``) rather than clipping per joint: uniform scaling preserves the posture
    direction and keeps the two arms coordinated, whereas per-joint clipping
    distorts the pose mid-motion, which reads worse than the jump it prevents.

    Returns ``(q, was_limited)``.
    """
    delta = np.asarray(q_new, dtype=float) - q_prev
    peak = float(np.max(np.abs(delta)))
    if peak <= max_delta or peak < 1e-12:
        return np.asarray(q_new, dtype=float), False
    return q_prev + delta * (max_delta / peak), True

_AMO_TORSO_YAW_GRIP_THRESHOLD = 0.5  # left/right grip threshold for yaw nudge

# Same 14 joints, same order as G1ArmIK's own reduced model (L[sh p,r,y,
# elbow, wr r,p,y], R[same]) -- used as PinkIKSolver's controlled_joint_names
# too (its ArmPostureTask regularizes exactly these).
_AMO_IK14_JOINT_NAMES = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

_AMO_BODY_IK_FRAME_NAMES = {
    "left": "left_hand_palm_link", "right": "right_hand_palm_link", "head": "head_site",
}

_AMO_TORSO_HEIGHT_DEFAULT = 0.75

# Joystick-driven torso commands (height + roll/pitch/yaw) -- an alternative
# to _tick_amo_arms' head-tracking-driven multi-target IK, selected via
# composite_controller_specific_config["amo_torso_control"] ("joystick",
# the default, or "head_ik"). Unlike locomotion vx/vy (which snap back to 0
# when the stick is released) and vyaw (which integrates into a persistent
# target heading), pitch/roll/yaw here are also persistent -- holding a
# button nudges the target and releasing holds it, matching vyaw's own
# "nudge and hold" feel, since there's no natural "centered" torso lean the
# way there's a natural "walking speed 0".
_AMO_TORSO_HEIGHT_RANGE = 0.30   # max +/- from the 0.75 baseline (right_axis ry)

_AMO_TORSO_HEIGHT_DEADZONE = 0.15

#: Torso height slew while the stick is held, m/s -- the full +/-0.30 m of
#: travel takes ~1.5 s. Height is an integrated "nudge and hold" channel like
#: torso roll/pitch/yaw, NOT a proportional one; see _tick_amo_commands.
_AMO_TORSO_HEIGHT_RATE = 0.20

#: Right-stick X thresholds. _AMO_YAW_DEADZONE is BOTH the turn threshold (in
#: _amo_scale_vyaw) and the threshold that suppresses the height axis, so the
#: two can never drift apart -- if they did, the symptom would be a stick
#: position that turns the robot while still driving height, i.e. the bug this
#: pair exists to fix. _AMO_YAW_RELEASE is the Schmitt-trigger release point:
#: without hysteresis an rx hovering near the threshold makes the height
#: integrator stutter on and off.
_AMO_YAW_DEADZONE = 0.2
_AMO_YAW_RELEASE = 0.15

_AMO_TORSO_RPY_RATE = 0.5        # rad/s while a button is held

_AMO_TORSO_RPY_MAX = 0.35        # rad clamp on pitch/roll/yaw

def _runtime_owns_the_robot(ctx) -> bool:
    """True while the sim2real runtime's own output is what reaches the joints.

    AMO is the one backend that edits ``target_q`` *after* ``step_policy``
    returns it -- the arm and wrist slots come from teleop, not from the
    locomotion policy. That edit has to respect the runtime's control mode,
    because outside ``"policy"`` mode ``target_q`` is not a policy output at
    all: it is the measured pose (``zero``) or a 10-second interpolation
    toward the default stance (``init``). Writing teleop targets over those
    two hands 14 joints to teleop while the operator has explicitly said the
    robot must not move yet.

    On hardware that was not subtle. Measured on the first tick of a real
    collection run, before the operator answered the startup gate: legs and
    waist held to 0.0000 rad of error, while the arms were commanded up to
    1.97 rad away from where they actually were -- ~60 Nm of instantaneous PD
    torque on wrist roll, shoulder yaw and both elbows, in one 20 ms tick.
    That is the "the robot lurches into a pose before I press y" report, and
    the follow-on instability: the arms end up thrown forward while the legs
    are still only PD-holding a stance, with AMO's balance policy not yet
    running.

    Returns True when the mode cannot be determined, which keeps every host
    that does not manage control modes -- the simulator, where
    ``BridgeTracking.__init__`` sets policy mode once and never leaves it --
    on exactly the path it was on before.
    """
    state = getattr(getattr(ctx, "tracking", None), "state_dict", None)
    if not isinstance(state, dict):
        return True
    return state.get("control_mode", "policy") == "policy"


def _amo_rpy_to_quat_wxyz(rpy):
    from scipy.spatial.transform import Rotation as R

    quat_xyzw = R.from_euler("xyz", list(rpy)).as_quat()
    return np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])

def _amo_scale_vx(v):
    return 0.0 if abs(v) < 0.3 else (0.6 if v > 0 else -0.6)

def _amo_scale_vy(v):
    return 0.0 if abs(v) < 0.7 else (0.5 if v > 0 else -0.5)

def _amo_scale_vyaw(v):
    if abs(v) < _AMO_YAW_DEADZONE:
        return 0.0
    return (0.3 if abs(v) < 0.5 else 0.5) * (1.0 if v > 0 else -1.0)

def _amo_world_yaw(quat_wxyz):
    """World yaw of a ``[w, x, y, z]`` quaternion.

    Deliberately the same arctan2 expression sim2real's
    ``rl_policy/observations/amo.py:_quat_to_euler`` uses, because the number
    this produces is subtracted from that one (``dyaw = wrap(rpy[2] -
    commands[1])``). A different Euler convention here -- scipy's
    ``as_euler("xyz")``, say -- would agree only while roll and pitch are
    exactly zero, i.e. everywhere except the walking robot this exists for.
    """
    qw, qx, qy, qz = (float(v) for v in np.asarray(quat_wxyz).reshape(4))
    return float(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))

def _amo_height_rate(ry):
    """Right-stick Y -> normalised height slew in [-1, 1].

    Deadzone-renormalised, ``(|ry| - dz) / (1 - dz)``, the same shape the sonic
    path uses on its left stick. Two reasons that matters here:

    * It removes a discontinuity. This used to return an absolute offset
      (``clip(ry) * 0.30``) computed from the RAW ``ry``, so the instant the
      stick crossed the 0.15 deadzone the torso height jumped
      ``0.15 * 0.30 = 0.045 m``. That step is what made an accidental nudge
      read as the torso lurching rather than drifting.
    * It gives proportional speed -- a light push creeps, a full push slews --
      instead of "light push = instant 4.5 cm".
    """
    mag = abs(float(ry))
    if mag < _AMO_TORSO_HEIGHT_DEADZONE:
        return 0.0
    norm = (mag - _AMO_TORSO_HEIGHT_DEADZONE) / (1.0 - _AMO_TORSO_HEIGHT_DEADZONE)
    return float(np.clip(norm, 0.0, 1.0)) * (1.0 if ry > 0 else -1.0)


class _WristCalibration:
    """The operator's controller frame tied to the robot's EE frame.

    Holds, per side, a rotation ``M`` (right-multiplied: ``R_target = R_rel @
    M``) and a translation ``d`` (``p_target = k * p_rel + d``). Both are solved
    from ONE gesture, which is what makes the two channels agree: at the
    gesture the commanded target is exactly ``FK(q14=0)``, a pose the arm
    reaches with every wrist joint at zero. Solving them separately (or taking
    one from a gesture and the other from an engage anchor) lets the channels
    disagree about which physical hand pose is being asked for, and with
    ``G1ArmIK``'s 50:1 position-over-orientation weighting the solver resolves
    that disagreement by bending the wrist to its limit -- the reported bug.
    """

    def __init__(self, ee_ref: dict, pos_scale: float = 1.0):
        #: side -> 4x4 FK of L_ee/R_ee at q14 = 0, read from the live model.
        self.ee_ref = ee_ref
        self.pos_scale = float(pos_scale)
        self.M = {s: _M_DEFAULT[s].copy() for s in ("left", "right")}
        self.d = {
            s: ee_ref[s][:3, 3] - self.pos_scale * REF_GESTURE_CTRL_POS[s]
            for s in ("left", "right")
        }
        self.source = "default"
        self.residual_deg = float("nan")
        self.residual_m = float("nan")

    # -- solve ---------------------------------------------------------- #
    def solve_from(self, rel_frames: dict, max_jitter_deg: float, max_jitter_m: float):
        """Fit M/d from N head-relative controller poses per side.

        Returns ``(ok, message)``. Rejects a capture the operator moved during:
        a smeared gesture silently becomes a permanently wrong mapping, and the
        operator has no way to see that from the robot's behaviour.
        """
        M_new, d_new, worst_deg, worst_m = {}, {}, 0.0, 0.0
        for side in ("left", "right"):
            frames = rel_frames.get(side) or []
            if len(frames) < 2:
                return False, f"only {len(frames)} valid frames for the {side} hand"
            rots = R.from_matrix(np.stack([T[:3, :3] for T in frames]))
            mean_rot = rots.mean()
            mean_pos = np.mean([T[:3, 3] for T in frames], axis=0)
            dev_deg = np.degrees(
                np.linalg.norm((mean_rot.inv() * rots).as_rotvec(), axis=1)
            ).max()
            dev_m = float(
                np.max(np.linalg.norm([T[:3, 3] - mean_pos for T in frames], axis=1))
            )
            worst_deg, worst_m = max(worst_deg, float(dev_deg)), max(worst_m, dev_m)
            if dev_deg > max_jitter_deg or dev_m > max_jitter_m:
                return False, (
                    f"{side} hand moved during capture "
                    f"({dev_deg:.1f} deg / {dev_m*100:.1f} cm) -- hold still and retry"
                )
            R_rel0 = mean_rot.as_matrix()
            M_new[side] = R_rel0.T @ self.ee_ref[side][:3, :3]
            d_new[side] = self.ee_ref[side][:3, 3] - self.pos_scale * mean_pos
        self.M, self.d = M_new, d_new
        self.source = "captured"
        self.residual_deg, self.residual_m = worst_deg, worst_m
        return True, f"residual {worst_deg:.2f} deg / {worst_m*100:.2f} cm"

    # -- persistence ---------------------------------------------------- #
    def save(self, path: str) -> bool:
        try:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            payload = {
                "version": _AMO_CALIB_VERSION,
                "pos_scale": self.pos_scale,
                "residual_deg": self.residual_deg,
                "residual_m": self.residual_m,
            }
            for side in ("left", "right"):
                payload[side] = {
                    "M_quat_wxyz": R.from_matrix(self.M[side])
                    .as_quat(scalar_first=True)
                    .tolist(),
                    "d_xyz": np.asarray(self.d[side], dtype=float).tolist(),
                }
            with open(path, "w") as f:
                json.dump(payload, f, indent=2)
            return True
        except OSError as e:
            print(f"[amo-vr] could not save calibration to {path}: {e}")
            return False

    def load(self, path: str) -> bool:
        """Load M/d from disk. A missing or unusable file is not an error --
        the defaults stay in place and the caller announces it."""
        try:
            with open(path) as f:
                payload = json.load(f)
            if int(payload.get("version", -1)) != _AMO_CALIB_VERSION:
                print(f"[amo-vr] ignoring calibration {path}: version mismatch")
                return False
            M, d = {}, {}
            for side in ("left", "right"):
                blob = payload[side]
                M[side] = R.from_quat(
                    np.asarray(blob["M_quat_wxyz"], dtype=float), scalar_first=True
                ).as_matrix()
                d[side] = np.asarray(blob["d_xyz"], dtype=float).reshape(3)
            self.M, self.d = M, d
            self.source = path
            self.residual_deg = float(payload.get("residual_deg", float("nan")))
            self.residual_m = float(payload.get("residual_m", float("nan")))
            return True
        except FileNotFoundError:
            return False
        except (OSError, KeyError, ValueError) as e:
            print(f"[amo-vr] ignoring calibration {path} ({type(e).__name__}: {e})")
            return False

    def target(self, side: str, rel: np.ndarray) -> np.ndarray:
        """Head-relative controller pose -> absolute EE target, in the robot's
        pelvis frame. ``M`` right-multiplies (change of BODY frame)."""
        T = np.eye(4)
        T[:3, :3] = rel[:3, :3] @ self.M[side]
        T[:3, 3] = self.pos_scale * rel[:3, 3] + self.d[side]
        return T


class _ArmIkLog:
    """Fixed-size ring buffer of per-tick arm-IK diagnostics.

    Exists because "it feels smoother now" is not evidence when the operator is
    in the loop with real hardware. The headline A/B numbers are peak/p99
    commanded joint velocity, the twitch count, and the limiter duty cycles --
    the last of which is the tell that a limiter is papering over a broken
    upstream rather than sitting idle as intended.

    Preallocated and written by index: this runs inside the 50 Hz control tick,
    so it must not allocate.

    It also carries the *operator* alongside the IK's answer for them -- the raw
    24-joint PICO body skeleton and the 3-point device poses -- so one recorded
    session can be replayed through a different retargeter offline and scored
    against the same ground truth. See ``scripts/gmr_arm_ab.py``, which does
    exactly that with GMR.
    """

    COLUMNS = (
        "sim_time",
        "reach_l",
        "reach_r",
        "clamped_l",
        "clamped_r",
        "dq_raw_inf",
        "dq_cmd_inf",
        "iters",
        "grad_inf",
        "err_norm",
        "jump_rejected",
        "rate_limited",
        "ik_failed",
    )
    _N_SCALAR = len(COLUMNS)

    def __init__(self, capacity=60000, nq=14):
        self._buf = np.zeros((capacity, self._N_SCALAR + 3 * nq), dtype=np.float32)
        # NaN, not zero, for both operator blocks below: a mock source and a
        # body-tracking dropout alike supply no data, and zeros would replay
        # offline as a perfectly valid skeleton collapsed at the origin. The
        # reader skips non-finite frames; it cannot skip plausible garbage.
        # ~40 MB at the default capacity (20 min at 50 Hz), allocated once and
        # only when SIM2REAL_AMO_IK_DEBUG is set.
        self._smpl = np.full((capacity, 24, 7), np.nan, dtype=np.float32)
        self._dev = np.full(
            (capacity, len(_LOG_DEVICE_POSE_KEYS), 7), np.nan, dtype=np.float32
        )
        self._n = 0
        self._nq = nq
        self._dropped = 0
        self._last_report_t = -1e9
        self._seq = 0

    def add(self, scalars, q_raw, q_cmd, q_meas, sample=None):
        if self._n >= self._buf.shape[0]:
            self._dropped += 1
            return
        row = self._buf[self._n]
        row[: self._N_SCALAR] = scalars
        k = self._N_SCALAR
        row[k : k + self._nq] = q_raw
        row[k + self._nq : k + 2 * self._nq] = q_cmd
        row[k + 2 * self._nq :] = q_meas
        if sample is not None:
            smpl = sample.get("raw_smpl")
            if smpl is not None and np.shape(smpl) == (24, 7):
                self._smpl[self._n] = smpl
            # Gate on vr_pose_valid, never on the poses themselves: a source
            # with no headset returns identity poses, which are indistinguish-
            # able from a real one standing at the origin (empty_device_poses).
            valid = sample.get("vr_pose_valid")
            for i, key in enumerate(_LOG_DEVICE_POSE_KEYS):
                if valid is not None and not valid[i]:
                    continue
                pose = sample.get(key)
                if pose is not None and np.shape(pose) == (7,):
                    self._dev[self._n, i] = pose
        self._n += 1

    def maybe_report(self, sim_time, period=1.0):
        """One line per second of sim time, so a live session is readable."""
        if sim_time - self._last_report_t < period:
            return
        self._last_report_t = sim_time
        rows = self._buf[: self._n]
        if not len(rows):
            return
        dq = rows[:, self.COLUMNS.index("dq_cmd_inf")]
        print(
            f"[amo-ik] t={sim_time:7.2f} n={self._n} "
            f"vel p99={np.percentile(dq, 99) / _POLICY_DT:6.2f} "
            f"max={dq.max() / _POLICY_DT:6.2f} rad/s | "
            f"twitch={int((dq > 0.1).sum())} "
            f"clamp={float(rows[:, 3:5].max(axis=1).mean()):.2f} "
            f"ratelim={float(rows[:, self.COLUMNS.index('rate_limited')].mean()):.2f} "
            f"jump={int(rows[:, self.COLUMNS.index('jump_rejected')].sum())} "
            f"iters={float(rows[:, self.COLUMNS.index('iters')].mean()):.1f}"
        )

    def dump(self):
        if not self._n:
            return None
        out_dir = os.environ.get("SIM2REAL_AMO_IK_LOG_DIR", ".")
        path = os.path.join(out_dir, f"amo_ik_log_{self._seq:03d}.npz")
        self._seq += 1
        rows = self._buf[: self._n].copy()
        k = self._N_SCALAR
        try:
            np.savez_compressed(
                path,
                columns=np.array(self.COLUMNS),
                scalars=rows[:, :k],
                q_raw=rows[:, k : k + self._nq],
                q_cmd=rows[:, k + self._nq : k + 2 * self._nq],
                q_meas=rows[:, k + 2 * self._nq :],
                # (T, 24, 7) raw xrobotoolkit_sdk.get_body_joints_pose(): Unity
                # frame, quaternions scalar-LAST. Not converted here -- whoever
                # replays it owns the convention, and the offline GMR path
                # already has that transform (xrobot_utils).
                raw_smpl=self._smpl[: self._n],
                dev_pose=self._dev[: self._n],
                dev_pose_keys=np.array(_LOG_DEVICE_POSE_KEYS),
            )
        except OSError as e:  # a failed log must never take the session down
            print(f"[amo-ik] log dump failed ({type(e).__name__}: {e})")
            path = None
        if self._dropped:
            print(f"[amo-ik] ring buffer full, dropped {self._dropped} ticks")
        self._n = 0
        self._dropped = 0
        if path:
            print(f"[amo-ik] wrote {path}")
        return path


class AmoBackend(Sim2RealBackend):
    """AMO whole-body policy + PICO 3-point arm IK."""

    name = "amo"
    # TorchScript, not ONNX -- see AmoBridgeTracking / sim2real's amo.py.
    POLICY_CONFIGS = {"amo": get_family("amo").policy_config}
    controller_json = "g1_generic.json"
    #: Stays False even under amo_arm_source="gmr". This flag means "the GMR
    #: *publisher* owns the PICO connection", which drives the PICO-source swap
    #: to GMRButtonRelay -- and that would cost AMO its joysticks. The arm
    #: worker is a different thing entirely: it touches no hardware, so this
    #: process keeps its own PICO source either way. See gmr_utils/arm_worker.py.
    uses_gmr = False

    def __init__(self):
        self._amo_arm_sim_indices = None   # 8 indices into G1_JOINT_NAMES
        self._amo_wrist_sim_indices = None  # 6 indices into G1_JOINT_NAMES
        self._amo_commands = np.zeros(8, dtype=np.float32)
        self._amo_vyaw = 0.0
        #: Heading the operator has steered to, RELATIVE to the heading the
        #: episode started in (right-stick vyaw integrated). commands[1] is the
        #: absolute world heading the policy tracks, i.e. this plus
        #: _amo_yaw_anchor -- see _ensure_yaw_anchor.
        self._amo_heading_rel = 0.0
        #: World yaw the robot settled into this episode; None until the first
        #: post-settling tick seeds it.
        self._amo_yaw_anchor = None
        self._amo_ik = None
        self._amo_body_ik = None  # PinkIKSolver: head+2-wrist -> torso h/rpy
        #: GMRArmRelay when amo_arm_source == "gmr", else None. Built in
        #: build_tracking (once per process); on_load picks it up off res.extra.
        self._gmr_arm_relay = None
        self._gmr_stale_since = None
        self._gmr_warned_no_body = False
        #: (q_anchor14, t0, duration_s) while ramping onto the retargeted pose.
        self._gmr_blend = None
        self._amo_arm_engaged = False
        self._amo_prev_grip_pressed = False
        self._amo_eng_captured = False
        self._amo_eng_L = self._amo_eng_R = None
        self._amo_eng_3pt_L = self._amo_eng_3pt_R = None
        self._amo_eng_head = self._amo_eng_3pt_head = None
        self._amo_warned_no_3pt = False
        self._amo_arm_action = _AMO_DEFAULT_DOF_POS[15:].copy()
        self._amo_wrist_targets = np.zeros(len(_AMO_WRIST_NAMES), dtype=np.float32)
        # Last command actually issued, in IK-14 order -- the rate limiter's
        # reference, and what both guards hold on to when they reject a solve.
        # None until the first engaged tick.
        self._amo_prev_cmd_q14 = None
        # Previous RAW solution from whichever arm source is active -- the jump
        # guard's own reference, deliberately NOT the command. See
        # _commit_arm_q14. None until the first solve of an engagement.
        self._amo_prev_raw_q14 = None
        self._amo_jump_streak = 0
        # Shoulder-pitch origins in the IK model's pelvis frame, read from the
        # model in _init_amo_arm_ik so they can never drift from the URDF.
        self._amo_shoulder_l = None
        self._amo_shoulder_r = None
        self._amo_ik_log = _ArmIkLog() if _AMO_IK_DEBUG else None
        # ---- absolute wrist teleop ----
        self._vr = dict(_AMO_VR_DEFAULTS)
        self._calib = None  # _WristCalibration, built in on_load
        self._calib_chord_t0 = None  # menu-chord press time, or None
        self._calib_frames = None  # side -> [4x4] while capturing
        self._calib_chord_done = False  # one capture per press
        #: Engage/recovery blend: (T_hold, t_start, duration). The single
        #: mechanism for every target discontinuity in this backend.
        self._blend = None
        self._vr_invalid_since = None  # sim time tracking went bad, or None
        self._vr_targets_held = None  # last good (T_L, T_R) during a dropout
        self._vr_warned_invalid = False
        self._amo_torso_h = 0.75
        self._amo_torso_rpy = np.zeros(3, dtype=np.float32)
        self._amo_torso_control_mode = "joystick"
        self._amo_torso_pitch_cmd = 0.0
        self._amo_torso_roll_cmd = 0.0
        self._amo_torso_yaw_cmd = 0.0
        # Height is a held offset from the 0.75 baseline, same shape as the
        # three rpy nudges above -- so like them it MUST be cleared per episode
        # (see _reset_amo), or a height dialled in during one episode silently
        # carries into the next.
        self._amo_torso_h_cmd = 0.0
        self._amo_turning = False  # Schmitt state for the height suppressor
        # The torso command as it stood at the PREVIOUS tick -- recorded as the
        # rpy/height STATE (see _amo_teleop_data_sample): Psi0's G1 wholebody
        # datasets define those state entries as "the previous frame's command"
        # (raw_to_lerobot.py build_obs: prev_rpy_height), which is what a policy
        # trained on them expects, and which every host can answer.
        self._amo_torso_state_rpy = np.zeros(3, dtype=np.float32)
        self._amo_torso_state_h = _AMO_TORSO_HEIGHT_DEFAULT

    # ------------------------------------------------------------------ #
    # Construction (cache-key-guarded: about once per process)
    # ------------------------------------------------------------------ #
    def build_tracking(self, cfg, policy_config_path, inference_backend):
        # No GMR/motion reference at all -- AMO is command-velocity locomotion
        # (PICO joysticks), not motion-tracking. motion_mode="push" is only
        # used here to satisfy BridgeTracking's own requirement that *some*
        # valid motion_backend be configured; sim2real.rl_policy.observations
        # .amo never reads motion_data.
        from g1_bridge.bridge_runtime import make_bridge_tracking

        tracking, io = make_bridge_tracking(
            policy_config_path, motion_mode="push", inference_backend=inference_backend
        )
        # The worker is a subprocess, so it belongs here (once per process) and
        # not in on_load (once per episode). It owns no hardware -- this
        # process keeps the only PICO connection -- so it is safe to leave
        # running across resets. See gmr_utils/arm_worker.py.
        extra = {}
        arm_source = cfg.get("amo_arm_source", _AMO_VR_DEFAULTS["amo_arm_source"])
        if arm_source not in _AMO_ARM_SOURCES:
            raise ValueError(
                f"amo_arm_source={arm_source!r} is not one of {_AMO_ARM_SOURCES}"
            )
        if arm_source == "gmr":
            from g1_bridge.gmr_utils.arm_relay import GMRArmRelay

            relay = GMRArmRelay(
                actual_human_height=float(
                    cfg.get(
                        "amo_gmr_human_height",
                        _AMO_VR_DEFAULTS["amo_gmr_human_height"],
                    )
                ),
                viewer=bool(cfg.get("amo_gmr_viewer", _AMO_GMR_VIEWER_DEFAULT)),
                viewer_port=_AMO_GMR_VIEWER_PORT_DEFAULT,
            )
            relay.start(warmup_s=float(cfg.get("gmr_warmup_s", 3.0)))
            extra["gmr_arm_relay"] = relay
        arms = (
            "PICO body skeleton -> GMR retarget"
            if arm_source == "gmr"
            else "PICO 3-point IK"
        )
        print(
            f"[g1_bridge] G1_BRIDGE armed, backend=amo "
            f"(policy_config={policy_config_path}). Legs+waist from "
            f"amo_jit.pt; shoulder/elbow+wrist from {arms} "
            "teleop (toggle on right-stick click)."
        )
        return BackendResources(tracking=tracking, io=io, extra=extra)

    # ------------------------------------------------------------------ #
    # Per env.reset() -- UNGUARDED, must rebuild the sim-dependent maps and
    # the IK solvers every episode (sim.model is a new object each reset).
    # ------------------------------------------------------------------ #
    #: Rows shared by both torso-control modes. AMO has no engage state machine
    #: of its own -- the legs balance from the first tick and the arms toggle on
    #: their own button -- so the A+B+X+Y chord is listed for what it actually
    #: does here: gate the recorder (see g1_data_collection's ``engaged``).
    _CHEATSHEET_HEAD = (
        ("", "-- AMO: legs always live --"),
        ("L-stick", "walk fwd/back + strafe"),
        ("R-stick X", "turn"),
    )
    _CHEATSHEET_TAIL = (
        ("triggers", "close hands"),
        ("A+B+X+Y", "engage chord: recording gate only"),
        ("L+R menu 1s", "calibrate wrists (palms facing)"),
    )

    @property
    def hud_cheatsheet(self):
        """AMO's on-screen control map -- the reason this hook exists. Its
        bindings are spread over both sticks, a stick click, all four face
        buttons and both grips, and (unlike the other backends) none of them is
        the chord an operator already knows from SONIC.

        The torso half depends on ``amo_torso_control``: in "head_ik" the face
        buttons and grips are dead and the torso rides the same engage toggle as
        the arms, so listing them would be wrong."""
        if self._amo_torso_control_mode == "head_ik":
            return (
                self._CHEATSHEET_HEAD
                + (("R-click", "arm+torso teleop ON/OFF (VR 3-pt)"),)
                + self._CHEATSHEET_TAIL
                + (("", "-- torso follows your head --"),)
            )
        return (
            self._CHEATSHEET_HEAD
            + (("R-click", "arm teleop ON/OFF (VR 3-point)"),)
            + self._CHEATSHEET_TAIL
            + (
                ("", "-- TORSO (nudge, holds on release) --"),
                ("R-stick Y", "height (muted while turning)"),
                ("A / B", "roll  left / right"),
                ("X / Y", "pitch up / down"),
                ("L / R grip", "yaw left / right"),
            )
        )

    def on_load(self, ctx: Sim2RealContext, cfg: dict, res: BackendResources) -> None:
        # "joystick" (default) drives commands[3:7] from the handset directly
        # in _tick_amo_commands; "head_ik" uses _tick_amo_arms' head-tracking
        # multi-target IK instead.
        self._amo_torso_control_mode = cfg.get("amo_torso_control", "joystick")
        for key, default in _AMO_VR_DEFAULTS.items():
            self._vr[key] = cfg.get(key, default)
        self._gmr_arm_relay = (res.extra or {}).get("gmr_arm_relay")
        self._build_amo_joint_mapping()
        self._init_amo_arm_ik()
        self._init_wrist_calibration()

    def _init_wrist_calibration(self):
        """Build the controller->EE mapping, from disk if the operator has ever
        calibrated on this machine. Rebuilt every episode alongside the IK, so
        the FK reference always comes from the live model."""
        if self._amo_ik is None:
            self._calib = None
            return
        import pinocchio as pin

        model, data = self._amo_ik.model, self._amo_ik.data
        pin.framesForwardKinematics(model, data, np.zeros(self._amo_ik.nq))
        ee_ref = {
            "left": data.oMf[self._amo_ik.L_id].homogeneous.copy(),
            "right": data.oMf[self._amo_ik.R_id].homogeneous.copy(),
        }
        self._calib = _WristCalibration(
            ee_ref, pos_scale=float(self._vr["amo_vr_arm_pos_scale"])
        )
        path = self._vr["amo_vr_calib_path"] or default_calib_path()
        if self._calib.load(path):
            print(
                f"[amo-vr] wrist calibration loaded from {path} "
                f"(residual {self._calib.residual_deg:.2f} deg / "
                f"{self._calib.residual_m*100:.2f} cm)"
            )
        elif self._vr["amo_vr_arm_ori_mode"] == "absolute":
            print(
                "[amo-vr] no wrist calibration found -- using the DEFAULT "
                "controller->EE remap. That is a prior borrowed from Psi0's "
                "hand-tracking constants, NOT a measurement for a held "
                "controller, so expect a systematic wrist offset. To fix it: "
                "stand upright looking level, upper arms down, elbows at 90 "
                "deg, forearms forward, PALMS FACING EACH OTHER, then hold "
                "both menu buttons for 1 s."
            )

    def reset(self, ctx: Sim2RealContext) -> None:
        self._reset_amo(ctx)

    # ------------------------------------------------------------------ #
    # Per tick
    # ------------------------------------------------------------------ #
    def _ensure_yaw_anchor(self, ctx) -> float:
        """The world heading this episode's ``target_yaw`` is measured from,
        seeded on the first tick that needs it.

        ``commands[1]`` is an ABSOLUTE world heading: the policy's observation
        is ``dyaw = wrap(pelvis_world_yaw - commands[1])`` (sim2real
        ``observations/amo.py``), so a zero there means "face world +x", not
        "keep facing whichever way you are". On hardware that distinction is
        invisible -- the IMU yaw starts at ~0 where the robot is switched on,
        which is exactly what AMO's own ``play_amo.py`` assumes -- but robocasa
        spawns the G1 at a per-scene calibrated heading
        (``ithor_base.robot_base_ori``, measured 2.81 rad in FloorPlan329),
        so a zeroed command asked the robot to turn that whole angle out.

        It did not show up as "the robot spawns facing wrong" because the
        observation suppresses ``dyaw`` entirely while standing in place
        (``_in_place_stand``): the heading error only went live on the first
        tick the operator pushed the stick, so the robot set off, then curved
        through ~160 deg mid-walk and carried on in the new direction.

        Anchoring here (first post-settling tick, not ``reset``) is what makes
        it the *landed* heading: ``reset`` runs before robocasa teleports the
        robot into the scene, and the settling window's landing rotates it a
        further few degrees.
        """
        if self._amo_yaw_anchor is None:
            self._amo_yaw_anchor = _amo_world_yaw(ctx.read_pelvis_state()[0])
            self._amo_commands[1] = self._amo_yaw_anchor + self._amo_heading_rel
        return self._amo_yaw_anchor

    def _apply_heading(self) -> None:
        """Push the operator/policy heading into the absolute command."""
        anchor = 0.0 if self._amo_yaw_anchor is None else self._amo_yaw_anchor
        self._amo_commands[1] = anchor + self._amo_heading_rel

    def step(self, ctx: Sim2RealContext, tick: TickInput):
        """Command-velocity locomotion, not motion-tracking -- runs in every
        mode (no reference to gate on); ModeManager only gates the hand
        trigger and data recording, same as the other backends."""
        self._ensure_yaw_anchor(ctx)
        self._tick_amo_commands(ctx, tick.buttons)
        self._tick_amo_arms(tick.qpos, tick.buttons, tick.sample, tick.sim_time)
        return self._step_amo(ctx, tick.qpos)

    # ------------------------------------------------------------------ #
    # VLA-driven operation
    # ------------------------------------------------------------------ #
    #: AMO takes commands, not reference motion, so it shares no command space
    #: with the tracking families -- a policy for it must be trained on AMO
    #: data. Keys mirror _amo_teleop_data_sample() exactly.
    vla_action_space = "velocity_cmd"

    def vla_sink(self, ctx):
        """The commands live here, not on the policy runtime."""
        return self

    def set_velocity_command(self, cmd: dict) -> None:
        """Fill this tick's locomotion/torso command from policy output.

        ``target_yaw`` is a heading, not a rate, and that is deliberate: teleop
        integrates the right-stick rate into a heading, so the heading -- not
        the rate -- is what got recorded and what the policy learned to
        predict. Feeding the rate back through the integrator here would
        double-integrate. It is also the heading *relative to where the episode
        started*, exactly as recorded (see _amo_teleop_data_sample); the
        episode's own anchor turns it into the world heading the policy tracks,
        so a policy trained in one scene is not pinned to that scene's spawn
        yaw.
        """
        self._latch_torso_state()
        c = self._amo_commands
        c[0] = float(cmd["torso_vx"][0])
        c[2] = float(cmd["torso_vy"][0])
        self._amo_vyaw = float(cmd["torso_vyaw"][0])
        if "target_yaw" in cmd:
            self._amo_heading_rel = float(cmd["target_yaw"][0])
        else:  # rate-only policy: integrate, as the operator path does
            self._amo_heading_rel += self._amo_vyaw * _POLICY_DT
        self._apply_heading()
        if "height" in cmd:
            self._amo_torso_h = float(cmd["height"][0])
            c[3] = self._amo_torso_h - _AMO_TORSO_HEIGHT_DEFAULT
        if "rpy" in cmd:
            rpy = np.asarray(cmd["rpy"], dtype=np.float32).reshape(-1)
            self._amo_torso_rpy = rpy
            c[4], c[5], c[6] = float(rpy[2]), float(rpy[1]), float(rpy[0])
        # Arm targets are recorded as slices of the canonical 29-order target_q
        # (see _amo_teleop_data_sample), so the policy predicts joint angles
        # directly and the IK the operator path needs has nothing to do here.
        self._vla_arm_targets = None
        if "left_arm" in cmd and "right_arm" in cmd:
            self._vla_arm_targets = (
                np.asarray(cmd["left_arm"], dtype=np.float32).reshape(-1),
                np.asarray(cmd["right_arm"], dtype=np.float32).reshape(-1),
            )

    def vla_step(self, ctx, tick, action):
        """Commands from the policy instead of the joysticks; no arm IK."""
        # Before apply: set_velocity_command has no ctx of its own, and it needs
        # the anchor to turn the policy's relative heading into a world one.
        self._ensure_yaw_anchor(ctx)
        self.apply_vla_action(ctx, action)
        ctx.tracking.amo_commands = self._amo_commands
        ctx.tracking.amo_vyaw = self._amo_vyaw
        target_q, kps, kds = ctx.step_policy(tick.qpos)
        arms = getattr(self, "_vla_arm_targets", None)
        if arms is not None and _runtime_owns_the_robot(ctx):
            target_q = np.asarray(target_q, dtype=np.float32).copy()
            target_q[15:22], target_q[22:29] = arms
        return target_q, kps, kds

    @staticmethod
    def read_triggers(buttons, engaged):
        """AMO never uses ModeManager engagement for its own control (legs run
        continuously, arms toggle independently on right grip -- see
        _tick_amo_arms), so gating hands on `engaged` would mean they silently
        never respond unless the operator also happened to do an unrelated
        A+B+X+Y chord. AMOG1Composite gated hands on nothing at all; match it."""
        if not buttons:
            return 0.0, 0.0
        return (
            float(buttons.get("left_trigger", 0.0)),
            float(buttons.get("right_trigger", 0.0)),
        )

    def teleop_sample(self, ctx: Sim2RealContext) -> dict:
        return self._amo_teleop_data_sample(ctx)

    def _init_amo_arm_ik(self):
        try:
            import pinocchio as pin
            from g1_bridge.vendored.amo.g1_arm_ik import G1ArmIK

            self._amo_ik = G1ArmIK()
            # Shoulder-pitch joint origins at q=0, for the reach clamp. Read
            # from the model rather than hard-coded so they track the URDF.
            model, data = self._amo_ik.model, self._amo_ik.data
            pin.framesForwardKinematics(model, data, np.zeros(self._amo_ik.nq))
            self._amo_shoulder_l = data.oMi[
                model.getJointId("left_shoulder_pitch_joint")
            ].translation.copy()
            self._amo_shoulder_r = data.oMi[
                model.getJointId("right_shoulder_pitch_joint")
            ].translation.copy()
            if self._vr.get("amo_arm_source") == "gmr":
                # Still built: the head_ik torso mode and _amo_wrist_fk use this
                # model even when the arms come from the retargeter. Say so,
                # rather than announcing a solver that is not driving anything.
                print(
                    "[g1_bridge] amo arm teleop ready (PICO body skeleton -> "
                    f"GMR retarget; the 3-point IK is loaded but idle. "
                    f"rate limit {_AMO_ARM_VEL_LIMIT:.1f} rad/s)."
                )
            else:
                print(
                    "[g1_bridge] amo arm teleop ready (PICO 3-point -> DLS IK, "
                    f"reach clamp {_AMO_ARM_REACH_MAX:.2f} m, "
                    f"rate limit {_AMO_ARM_VEL_LIMIT:.1f} rad/s)."
                )
        except Exception as e:
            print(f"[g1_bridge] amo arm teleop disabled ({type(e).__name__}: {e}); arms hold default.")
            self._amo_ik = None

        try:
            import os

            import g1_bridge.vendored.amo as _amo_vendored_pkg
            from g1_bridge.vendored.amo.multi_target_ik import PinkIKSolver

            vendored_dir = os.path.dirname(_amo_vendored_pkg.__file__)
            urdf_path = os.path.join(vendored_dir, "g1_body29_hand14_virtual.urdf")
            self._amo_body_ik = PinkIKSolver(urdf_path, _AMO_BODY_IK_FRAME_NAMES, _AMO_IK14_JOINT_NAMES)
            print("[g1_bridge] amo torso teleop ready (PICO head+3-point -> multi-target IK).")
        except Exception as e:
            print(f"[g1_bridge] amo torso teleop disabled ({type(e).__name__}: {e}); torso holds default.")
            self._amo_body_ik = None

    def _build_amo_joint_mapping(self):
        self._amo_arm_sim_indices = np.array(
            [G1_JOINT_NAMES.index(name) for name in _AMO_ARM_NAMES], dtype=int
        )
        self._amo_wrist_sim_indices = np.array(
            [G1_JOINT_NAMES.index(name) for name in _AMO_WRIST_NAMES], dtype=int
        )

    def _latch_torso_state(self) -> None:
        """Snapshot the torso command before this tick changes it.

        Called at the top of every per-tick command update (joystick path and
        VLA path alike), so ``_amo_torso_state_*`` always hold the command that
        was in force during the previous tick -- the ``rpy``/``height`` STATE
        the recorder writes (Psi0 convention, see _amo_teleop_data_sample).
        """
        c = self._amo_commands
        self._amo_torso_state_rpy = np.array([c[6], c[5], c[4]], dtype=np.float32)
        self._amo_torso_state_h = float(_AMO_TORSO_HEIGHT_DEFAULT + c[3])

    def torso_command_state(self) -> tuple[np.ndarray, float]:
        """``(rpy, height)`` of the torso command in force RIGHT NOW -- for a
        deploy/eval client building the policy's ``state.rpy``/``state.height``.

        Called before this tick's action is applied, it is "the command in
        force during the previous tick", i.e. exactly what the recorder writes
        as the rpy/height STATE (``_amo_teleop_data_sample`` reads the latched
        copy because it samples AFTER the tick's update). Same definition as
        Psi0's G1 wholebody datasets.
        """
        c = self._amo_commands
        return (
            np.array([c[6], c[5], c[4]], dtype=np.float32),
            float(_AMO_TORSO_HEIGHT_DEFAULT + c[3]),
        )

    def on_episode_start(self) -> None:
        """Per-recorded-episode state, independent of ``reset()``.

        ``reset()`` runs on ``env.reset()`` -- per episode in sim, but ONCE per
        hardware session (RealRobotRunner.arm). Everything the dataset defines
        "relative to this episode's start" therefore has to be re-taken here or
        hardware episode N silently inherits episode N-1's values:

        * ``target_yaw`` is recorded as the heading relative to the episode's
          start (``_amo_heading_rel``) -- re-anchor, so the next tick's
          ``_ensure_yaw_anchor`` takes the robot's current heading as zero.
        * the torso height/rpy integrators (``_amo_torso_*_cmd``, the
          ``_amo_torso_h``/``_amo_torso_rpy`` they feed, and commands[3:7]) go
          back to the standing defaults, as Psi0's collector does on episode
          start -- a height dialled in during one take must not be the opening
          command of the next.
        * the recorded rpy/height STATE ("previous tick's command") restarts at
          the defaults for the same reason.

        In sim this runs right after ``_reset_amo`` and is a no-op on top of it.
        """
        self._amo_heading_rel = 0.0
        self._amo_yaw_anchor = None
        self._amo_torso_h = _AMO_TORSO_HEIGHT_DEFAULT
        self._amo_torso_rpy = np.zeros(3, dtype=np.float32)
        self._amo_torso_h_cmd = 0.0
        self._amo_torso_pitch_cmd = 0.0
        self._amo_torso_roll_cmd = 0.0
        self._amo_torso_yaw_cmd = 0.0
        self._amo_commands[3:7] = 0.0
        self._amo_torso_state_rpy = np.zeros(3, dtype=np.float32)
        self._amo_torso_state_h = _AMO_TORSO_HEIGHT_DEFAULT

    def _reset_amo(self, ctx):
        """Matches AMOG1Composite.reset(): clear command/arm-teleop state and
        write AMO's own trained default pose into sim (legs+waist+arms) --
        AMO's balance policy assumes starting from this exact standing
        configuration, unlike the frozen-reference sonic path which doesn't
        do live closed-loop balance in this setup."""
        self._amo_commands[:] = 0.0
        self._amo_vyaw = 0.0
        # Not seeded here on purpose: robosuite runs reset() BEFORE the scene
        # teleports the robot to its spawn pose, so any heading read now is the
        # previous episode's. _ensure_yaw_anchor takes it on the first
        # post-settling tick instead.
        self._amo_heading_rel = 0.0
        self._amo_yaw_anchor = None
        self._amo_arm_engaged = False
        self._amo_prev_grip_pressed = False
        self._amo_eng_captured = False
        self._amo_wrist_targets[:] = 0.0
        self._amo_arm_action = _AMO_DEFAULT_DOF_POS[15:].copy()
        # The IK object is built once in on_load and SURVIVES env.reset(), so
        # without this the output filter carries the previous episode's last
        # solutions into the first solve of the next one.
        self._amo_prev_cmd_q14 = None
        self._amo_prev_raw_q14 = None
        self._amo_jump_streak = 0
        self._blend = None
        self._vr_invalid_since = None
        self._vr_targets_held = None
        self._calib_chord_t0 = None
        self._calib_frames = None
        self._calib_chord_done = False
        if self._amo_ik is not None:
            self._amo_ik.prime_filter(_AMO_DEFAULT_Q14)
        self._amo_torso_h = _AMO_TORSO_HEIGHT_DEFAULT
        self._amo_torso_rpy = np.zeros(3, dtype=np.float32)
        self._amo_torso_state_rpy = np.zeros(3, dtype=np.float32)
        self._amo_torso_state_h = _AMO_TORSO_HEIGHT_DEFAULT
        self._amo_eng_head = self._amo_eng_3pt_head = None
        self._amo_torso_pitch_cmd = 0.0
        self._amo_torso_roll_cmd = 0.0
        self._amo_torso_yaw_cmd = 0.0
        self._amo_torso_h_cmd = 0.0
        self._amo_turning = False
        if getattr(ctx, "sim", None) is None:
            # Hardware: there is nothing to teleport. The physical equivalent of
            # this reset is the operator ramping the robot to the default pose
            # before arming (the real host's INIT stage), which puts the joints
            # where these lines would have written them.
            print(
                colored(
                    "[amo] no simulator: skipping the joint teleport. Ramp the "
                    "robot to the default pose ('y' at startup) before arming.",
                    "yellow",
                ),
                flush=True,
            )
            return
        for name, default_val in zip(_AMO_JOINT_NAMES, _AMO_DEFAULT_DOF_POS):
            i = G1_JOINT_NAMES.index(name)
            qp, qv = ctx.qpos_addrs[i], ctx.qvel_addrs[i]
            if qp is not None:
                ctx.sim.data.qpos[qp] = float(default_val)
            if qv is not None:
                ctx.sim.data.qvel[qv] = 0.0
        for i in self._amo_wrist_sim_indices:
            qp, qv = ctx.qpos_addrs[i], ctx.qvel_addrs[i]
            if qp is not None:
                ctx.sim.data.qpos[qp] = 0.0
            if qv is not None:
                ctx.sim.data.qvel[qv] = 0.0

    def _tick_amo_commands(self, ctx, buttons):
        """Left stick: ly -> vx, -lx -> vy. Right stick: -rx -> vyaw (rate),
        integrated into the target heading (commands[1], absolute -- see
        _ensure_yaw_anchor for what it is measured from); ry -> torso height
        (proportional, snaps back to the 0.75 baseline when centered -- same
        feel as vx/vy). Only in amo_torso_control="joystick" mode (the
        default): B/A -> roll nudge right/left, Y/X -> pitch nudge up/down,
        left/right grip -> yaw nudge left/right -- these three persist (hold
        to nudge, release to hold the last value) since there's no natural
        "centered" torso lean the way there's a natural "walking speed 0".
        Grip is free for this now that arm-teleop engage is right-stick-click
        only (see _tick_amo_arms) -- otherwise squeezing right grip to yaw
        would also toggle arm engagement. Only real PICO joysticks drive any
        of this; mock/no source leaves commands at zero (stand still,
        upright)."""
        self._latch_torso_state()
        is_real = buttons is not None and type(ctx.pico_source).__name__ in (
            "RealPicoVRSource",  # legacy in-process SDK (PICO_DIRECT_XRT=1)
            "ProcessPicoVRSource",  # default for pico_source="real" now
        )
        if is_real:
            lx, ly = buttons.get("left_axis", (0.0, 0.0))
            rx, ry = buttons.get("right_axis", (0.0, 0.0))
            self._amo_commands[0] = _amo_scale_vx(ly)
            self._amo_commands[2] = _amo_scale_vy(-lx)
            self._amo_vyaw = _amo_scale_vyaw(-rx)
            if self._amo_torso_control_mode == "joystick":
                # Turning wins. Both channels live on the right stick, and a
                # thumb pushing sideways to turn unavoidably carries a vertical
                # component -- which used to cross the height deadzone (0.15)
                # BEFORE the turn deadzone (0.2), so the torso lurched on
                # essentially every turn. Suppressing height while turning is
                # only tolerable because height is an integrator (below): it
                # means "skip this tick's integration", so the height the
                # operator already dialled in is HELD. With the old
                # proportional height, suppression would have meant "output 0",
                # snapping the torso back to 0.75 on every turn -- worse than
                # the bug.
                if abs(rx) >= _AMO_YAW_DEADZONE:
                    self._amo_turning = True
                elif abs(rx) < _AMO_YAW_RELEASE:
                    self._amo_turning = False
                if not self._amo_turning:
                    self._amo_torso_h_cmd += (
                        _AMO_TORSO_HEIGHT_RATE * _amo_height_rate(ry) * _POLICY_DT
                    )
                    self._amo_torso_h_cmd = float(
                        np.clip(
                            self._amo_torso_h_cmd,
                            -_AMO_TORSO_HEIGHT_RANGE,
                            _AMO_TORSO_HEIGHT_RANGE,
                        )
                    )
                self._amo_commands[3] = self._amo_torso_h_cmd
                # One axis per hand, upper button positive: B/A (right hand) are
                # roll, Y/X (left hand) are pitch. The pairs used to be split
                # across the two handsets (Y/A pitch, X/B roll), which is what
                # made the mapping unlearnable.
                if buttons.get("b"):
                    self._amo_torso_roll_cmd += _AMO_TORSO_RPY_RATE * _POLICY_DT
                if buttons.get("a"):
                    self._amo_torso_roll_cmd -= _AMO_TORSO_RPY_RATE * _POLICY_DT
                if buttons.get("y"):
                    self._amo_torso_pitch_cmd += _AMO_TORSO_RPY_RATE * _POLICY_DT
                if buttons.get("x"):
                    self._amo_torso_pitch_cmd -= _AMO_TORSO_RPY_RATE * _POLICY_DT
                # LEFT grip is +yaw. commands[4] feeds the adapter's torso-yaw
                # input, which is z-up right-handed (positive = turn left), so
                # the original right-grip-positive wiring turned the robot the
                # wrong way for whichever grip the operator squeezed.
                if float(buttons.get("left_grip", 0.0)) >= _AMO_TORSO_YAW_GRIP_THRESHOLD:
                    self._amo_torso_yaw_cmd += _AMO_TORSO_RPY_RATE * _POLICY_DT
                if float(buttons.get("right_grip", 0.0)) >= _AMO_TORSO_YAW_GRIP_THRESHOLD:
                    self._amo_torso_yaw_cmd -= _AMO_TORSO_RPY_RATE * _POLICY_DT
                self._amo_torso_pitch_cmd = float(np.clip(self._amo_torso_pitch_cmd, -_AMO_TORSO_RPY_MAX, _AMO_TORSO_RPY_MAX))
                self._amo_torso_roll_cmd = float(np.clip(self._amo_torso_roll_cmd, -_AMO_TORSO_RPY_MAX, _AMO_TORSO_RPY_MAX))
                self._amo_torso_yaw_cmd = float(np.clip(self._amo_torso_yaw_cmd, -_AMO_TORSO_RPY_MAX, _AMO_TORSO_RPY_MAX))
                self._amo_commands[4] = self._amo_torso_yaw_cmd
                self._amo_commands[5] = self._amo_torso_pitch_cmd
                self._amo_commands[6] = self._amo_torso_roll_cmd
        # Relative first, absolute second: the operator steers a heading
        # relative to where the episode started (that is what gets recorded),
        # and the policy is handed that heading in world terms.
        self._amo_heading_rel += self._amo_vyaw * _POLICY_DT
        self._apply_heading()

    def _read_amo_3pt_pose(self, sample):
        """SONIC-calibrated L/R wrist + neck(head) poses as 4x4 SE3 in the
        robot pelvis frame, from the latest PICO sample, or None when
        unavailable. Row 2 (neck) drives the torso-IK head target."""
        if not sample:
            return None
        pos = sample.get("vr_3point_local_target")
        orn = sample.get("vr_3point_local_orn_target")
        if pos is None or orn is None:
            return None
        from scipy.spatial.transform import Rotation as R

        pos = np.asarray(pos, dtype=float).reshape(3, 3)
        orn = np.asarray(orn, dtype=float).reshape(3, 4)  # wxyz

        def se3(p, q):
            T = np.eye(4)
            T[:3, 3] = p
            T[:3, :3] = R.from_quat(q, scalar_first=True).as_matrix()
            return T

        return se3(pos[0], orn[0]), se3(pos[1], orn[1]), se3(pos[2], orn[2])

    def _amo_head_fk(self, qpos):
        """head_site FK at the current torso h/rpy + joint config, used only
        once at arm-teleop engage to anchor the incremental head target."""
        import pinocchio as pin

        q = self._amo_body_ik.assemble_q(
            qpos,
            base_xyz=[0.0, 0.0, self._amo_torso_h],
            base_quat_wxyz=_amo_rpy_to_quat_wxyz(self._amo_torso_rpy),
        )
        pin.framesForwardKinematics(self._amo_body_ik.model, self._amo_body_ik.data, q)
        fid = self._amo_body_ik.model.getFrameId("head_site")
        return self._amo_body_ik.data.oMf[fid].homogeneous.copy()

    def _amo_wrist_fk(self, q14):
        import pinocchio as pin

        pin.framesForwardKinematics(self._amo_ik.model, self._amo_ik.data, q14)
        return (
            self._amo_ik.data.oMf[self._amo_ik.L_id].homogeneous.copy(),
            self._amo_ik.data.oMf[self._amo_ik.R_id].homogeneous.copy(),
        )

    def _read_amo_arm_q14(self, qpos):
        """Current 14 arm joints (IK reduced-model order) from the full
        29-joint qpos vector (G1_JOINT_NAMES order)."""
        arm8 = qpos[self._amo_arm_sim_indices]
        wr6 = qpos[self._amo_wrist_sim_indices]
        return np.array(
            list(arm8[:4]) + list(wr6[:3]) + list(arm8[4:]) + list(wr6[3:]),
            dtype=float,
        )

    def _rel_target(self, eng_amo, eng_3pt, cur_3pt):
        """Legacy engage-anchored increment (``amo_vr_arm_*_mode="relative"``).

        Kept verbatim as a one-flag rollback. Note what it actually does: the
        3-point poses it differences are themselves expressed relative to the
        operator's ThreePointPose calibration stance, so ``delta_R`` is the
        world-frame hand rotation conjugated by that stance and then applied in
        the robot's pelvis frame. Measured, that makes the same physical hand
        motion command a differently-rotated robot motion on every session.
        """
        T = np.eye(4)
        T[:3, 3] = eng_amo[:3, 3] + _AMO_ARM_POS_SCALE * (cur_3pt[:3, 3] - eng_3pt[:3, 3])
        delta_R = cur_3pt[:3, :3] @ eng_3pt[:3, :3].T
        T[:3, :3] = delta_R @ eng_amo[:3, :3]
        return T

    # ------------------------------------------------------------------ #
    # Absolute path: device poses -> EE targets
    # ------------------------------------------------------------------ #
    def _head_relative(self, sample, side):
        """Head-relative controller pose (4x4), or None if either the head or
        that controller is not tracking."""
        if not sample:
            return None
        valid = sample.get("vr_pose_valid")
        if valid is None or len(valid) < 3 or not valid[0]:
            return None
        idx, key = (
            (1, "vr_left_ctrl_pose") if side == "left" else (2, "vr_right_ctrl_pose")
        )
        if not valid[idx]:
            return None
        head = sample.get("vr_head_pose")
        ctrl = sample.get(key)
        if head is None or ctrl is None:
            return None
        return head_relative_pose(
            head, ctrl, yaw_only=(self._vr["amo_vr_ori_ref"] == "head_yaw")
        )

    def _absolute_targets(self, sample):
        """Absolute (T_left, T_right) EE targets, or None if not trackable."""
        if self._calib is None:
            return None
        out = []
        for side in ("left", "right"):
            rel = self._head_relative(sample, side)
            if rel is None:
                return None
            out.append(self._calib.target(side, rel))
        return tuple(out)

    def _begin_blend(self, q14, sim_time, duration_s):
        """Start ramping the commanded target from the arm's CURRENT EE pose to
        whatever the operator is asking for, over ``duration_s``.

        The single mechanism for every target discontinuity here (engage, and
        recovery from a tracking dropout). Sized so that even a worst-case 180
        deg error stays inside the rate limiter's budget: pi / (1.5 s / 0.02 s)
        = 0.042 rad/tick against a 0.06 rad/tick budget.
        """
        duration = float(duration_s)
        if duration <= 0.0:
            self._blend = None
            return
        hold_l, hold_r = self._amo_wrist_fk(q14)
        self._blend = (hold_l, hold_r, float(sim_time), duration)

    def _apply_blend(self, tL, tR, sim_time):
        if self._blend is None:
            return tL, tR
        hold_l, hold_r, t0, dur = self._blend
        alpha = (float(sim_time) - t0) / dur
        if alpha >= 1.0:
            self._blend = None
            return tL, tR
        return se3_interp(hold_l, tL, alpha), se3_interp(hold_r, tR, alpha)

    def _commit_arm_q14(self, q_new, qpos, *, solver_failed, source_label):
        """Guards 2 and 3, then publish. Shared by both arm sources.

        Returns ``(prev_cmd, q_raw, q_sol, jump_rejected, rate_limited)``, or
        ``None`` when a guard disengaged the arms -- in which case the caller
        must return immediately without touching the command fields.
        ``prev_cmd`` is handed back because it is the reference both diagnostic
        deltas are measured against, and this method overwrites it.
        """
        # Normally set by the engage-capture block; re-derived here for the
        # paths that clear it (episode reset, auto-disengage) racing an
        # already-captured engagement.
        if self._amo_prev_cmd_q14 is None:
            self._amo_prev_cmd_q14 = self._read_amo_arm_q14(qpos)
        prev_cmd = self._amo_prev_cmd_q14
        q_sol = (
            prev_cmd.copy()
            if solver_failed
            else np.asarray(q_new, dtype=float).copy()
        )
        q_raw = q_sol.copy()
        jump_rejected = False
        # Measured against the PREVIOUS RAW SOLUTION, not against the command.
        #
        # Against the command it does not measure a jump at all, it measures
        # how far the rate limiter has fallen behind -- and then latches on it.
        # The limiter caps the command at _AMO_ARM_VEL_LIMIT, so any input
        # moving faster than that opens a gap that grows every tick; once the
        # gap passes the threshold the guard rejects, a rejection freezes the
        # command, and a frozen command can never close the gap. Simulated
        # against these exact constants: a sustained 3.5 rad/s input
        # auto-disengages after 2.4 s, 6.9 rad/s after 0.7 s -- and 6.9 rad/s
        # is the *measured p99* of a real session. So a fast reach was enough
        # to drop the operator, which is the opposite of what a safety guard
        # should do.
        #
        # A jump is a discontinuity in the INPUT, so that is what gets
        # measured. "The command cannot keep up" is the rate limiter's job and
        # it already handles it, by ramping. Note the raw reference advances
        # even on a rejected tick: the guard tracks the input stream, so a real
        # one-frame spike costs two rejections (in and out) and then clears,
        # instead of poisoning every later comparison.
        #
        # This matters much more for GMR than for the IK. G1ArmIK warm-starts
        # from the measured pose and carries a smoothness term, so its output
        # stays partly pinned to the robot; GMR's q14 is a pure function of the
        # operator's body and does not know where the arm currently is.
        prev_raw = self._amo_prev_raw_q14
        if not solver_failed:
            self._amo_prev_raw_q14 = q_raw.copy()
        if (
            not solver_failed
            and prev_raw is not None
            and np.any(np.abs(q_sol - prev_raw) > _AMO_JUMP_THRESH)
        ):
            # Hold the last COMMAND, not the measured pose: measured would
            # inject the whole PD tracking error as a step, which is the very
            # thing this guard exists to prevent.
            jump_rejected = True
            self._amo_jump_streak += 1
            q_sol = prev_cmd.copy()
            if self._amo_jump_streak >= _AMO_JUMP_REJECT_MAX:
                # Psi0 holds forever here, which freezes the arms with no
                # operator recourse. Drop out instead; re-engaging re-anchors.
                self._disengage_arms(
                    qpos,
                    reason=f"{source_label} handed over {_AMO_JUMP_REJECT_MAX} "
                    "discontinuous solutions in a row",
                )
                return None
            print(f"[g1_bridge] amo arm {source_label} jump rejected; holding last command.")
        elif not solver_failed:
            self._amo_jump_streak = 0

        q_sol, rate_limited = rate_limit(
            q_sol, prev_cmd, _AMO_ARM_VEL_LIMIT * _POLICY_DT
        )
        self._amo_prev_cmd_q14 = q_sol.copy()
        self._amo_arm_action = np.array(
            [q_sol[i] for i in _IK_TO_AMO_ARM], dtype=np.float32
        )
        self._amo_wrist_targets = np.array(
            [q_sol[i] for i in _IK_TO_WRIST], dtype=np.float32
        )
        return prev_cmd, q_raw, q_sol, jump_rejected, rate_limited

    def _gmr_begin_blend(self, sim_time, duration_s):
        """Ramp the retargeted target in from wherever the arm is commanded now.

        Anchored at the last COMMAND rather than the measured pose, for the
        same reason the jump guard holds the command: measured would fold the
        PD tracking error into the ramp's starting point.
        """
        self._gmr_blend = (
            np.asarray(self._amo_prev_cmd_q14, dtype=float).copy(),
            float(sim_time),
            max(float(duration_s), 1e-6),
        )

    def _gmr_apply_blend(self, q_target, sim_time):
        if self._gmr_blend is None:
            return q_target
        q0, t0, duration = self._gmr_blend
        alpha = (sim_time - t0) / duration
        if alpha >= 1.0:
            self._gmr_blend = None
            return q_target
        q_target = np.asarray(q_target, dtype=float)
        return q0 + (q_target - q0) * max(alpha, 0.0)

    def _tick_amo_arms_gmr(self, qpos, sample, sim_time):
        """Arms from the operator's body skeleton, retargeted by GMR.

        Structurally simpler than the 3-point path because there is no target
        to build: the worker already returns joint angles in the order this
        backend wants. What it still needs is everything around the solve --
        engagement, an anchored start, staleness, and the two command guards.

        Engage needs the same blend the absolute path needs, for exactly the
        same reason -- and it is worth stating plainly, because the rate
        limiter looks like it should make one unnecessary and does not. The
        retargeted posture is generally nowhere near the arm's current pose,
        and the **jump guard sits upstream of the limiter**: it sees the raw
        step, vetoes it, and 25 vetoes later auto-disengages the operator.
        Measured, that is exactly what happens -- every tick rejected, dropped
        out in half a second. So the ramp has to be applied to the *target*,
        in joint space here rather than SE3, which makes the first tick's step
        exactly zero while leaving the guard armed to catch a genuine glitch.
        """
        raw = sample.get("raw_smpl") if sample else None
        if self._gmr_arm_relay is not None:
            # Submit even while disengaged: the worker's solve is stateful
            # (warm-started from the previous frame), so keeping it fed means
            # engagement lands on a converged solution rather than a cold one.
            self._gmr_arm_relay.submit(raw)

        if not self._amo_arm_engaged:
            return
        if self._gmr_arm_relay is None:
            if not self._gmr_warned_no_body:
                print(
                    '[amo-gmr] amo_arm_source="gmr" but no GMR arm worker was '
                    "built -- the arms will hold. This is a startup failure; "
                    "see the launch output above."
                )
                self._gmr_warned_no_body = True
            return

        if not self._amo_eng_captured:
            self._amo_eng_captured = True
            self._amo_prev_cmd_q14 = self._read_amo_arm_q14(qpos)
            self._amo_prev_raw_q14 = None
            self._amo_jump_streak = 0
            self._gmr_stale_since = None
            self._gmr_begin_blend(sim_time, self._vr["amo_vr_engage_blend_s"])

        got = self._gmr_arm_relay.latest()
        stale_s = float(self._vr["amo_vr_pose_stale_s"])
        # Wall clock on BOTH sides of this window, not sim_time. The age comes
        # from the relay and measures whether a separate OS process is still
        # answering -- a real-time property. Holding it against a sim-time
        # budget silently changes the timeout whenever the sim is not running
        # at 1x, and errs toward spurious disengages on a loaded box, which is
        # exactly when you least want the arms to drop out.
        now = time.monotonic()
        if got is None or got[1] > stale_s:
            # Gate on age, not on the values: a worker that has gone quiet and
            # an operator standing still are indistinguishable in the numbers.
            if self._gmr_stale_since is None:
                self._gmr_stale_since = now
            if not self._gmr_warned_no_body and got is None:
                print(
                    "[amo-gmr] arm teleop engaged but the GMR worker has "
                    "produced nothing yet -- needs the real PICO source with "
                    "body tracking active (xrt.is_body_data_available())."
                )
                self._gmr_warned_no_body = True
            if now - self._gmr_stale_since > float(self._vr["amo_vr_pose_hold_s"]):
                self._disengage_arms(qpos, reason="GMR arm stream went stale")
            # Hold the last command meanwhile -- _amo_arm_action is untouched.
            return
        if self._gmr_stale_since is not None:
            # The operator kept moving through the outage, so resuming is a
            # step -- route it through the same ramp as engage.
            self._gmr_begin_blend(sim_time, self._vr["amo_vr_recover_blend_s"])
            self._gmr_stale_since = None

        committed = self._commit_arm_q14(
            self._gmr_apply_blend(got[0], sim_time),
            qpos,
            solver_failed=False,
            source_label="GMR",
        )
        if committed is None:
            return
        prev_cmd, q_raw, q_sol, jump_rejected, rate_limited = committed

        if self._amo_ik_log is not None:
            q14 = self._read_amo_arm_q14(qpos)
            self._amo_ik_log.add(
                (
                    sim_time,
                    0.0, 0.0, 0.0, 0.0,   # reach/clamp: no EE target on this path
                    float(np.max(np.abs(q_raw - prev_cmd))),
                    float(np.max(np.abs(q_sol - prev_cmd))),
                    0.0, 0.0,             # iters/grad: the solve happened offboard
                    float(got[1]),        # err_norm slot reused: stream age (s)
                    float(jump_rejected),
                    float(rate_limited),
                    0.0,
                ),
                q_raw,
                q_sol,
                q14,
                sample,
            )
            self._amo_ik_log.maybe_report(sim_time)

    def _disengage_arms(self, qpos, reason: str):
        """Drop out of arm teleop and park the arms, from any cause.

        Shared by the jump-guard streak and the pose-dropout timeout so both
        leave identical state -- the operator's way back is always the same
        (re-click the right stick), and re-engaging re-anchors everything.
        """
        self._amo_arm_engaged = False
        self._amo_eng_captured = False
        self._amo_arm_action = _AMO_DEFAULT_DOF_POS[15:].copy()
        self._amo_wrist_targets[:] = 0.0
        self._amo_prev_cmd_q14 = None
        self._amo_prev_raw_q14 = None
        self._amo_jump_streak = 0
        self._blend = None
        self._vr_invalid_since = None
        self._vr_targets_held = None
        self._gmr_stale_since = None
        self._gmr_blend = None
        if self._amo_ik is not None:
            self._amo_ik.prime_filter(self._read_amo_arm_q14(qpos))
        print(
            f"[g1_bridge] amo arm teleop disengaged: {reason}. "
            "Re-click the right stick to re-engage."
        )

    def _tick_wrist_calibration(self, buttons, sample, sim_time):
        """Both menu buttons held for _AMO_CALIB_HOLD_S captures the gesture.

        Frames are collected DURING the hold, so by the time the chord matures
        the operator has already been still for a second -- which is exactly
        the sample we want, and it makes the jitter gate meaningful.
        """
        if self._calib is None or self._vr["amo_vr_arm_ori_mode"] != "absolute":
            return
        held = bool(buttons and buttons.get("left_menu") and buttons.get("right_menu"))
        if not held:
            if self._calib_chord_t0 is not None:
                self._calib_chord_t0 = None
                self._calib_frames = None
            self._calib_chord_done = False
            return
        if self._calib_chord_done:
            # One capture per press. Without this, keeping the chord held after
            # a successful capture immediately re-arms and re-captures, once per
            # hold period, spamming the operator.
            return
        if self._calib_chord_t0 is None:
            self._calib_chord_t0 = sim_time
            self._calib_frames = {"left": [], "right": []}
            print("[amo-vr] hold the reference gesture...")
        for side in ("left", "right"):
            rel = self._head_relative(sample, side)
            if rel is not None and len(self._calib_frames[side]) < _AMO_CALIB_FRAMES:
                self._calib_frames[side].append(rel)
        if sim_time - self._calib_chord_t0 < _AMO_CALIB_HOLD_S:
            return
        ok, msg = self._calib.solve_from(
            self._calib_frames,
            float(self._vr["amo_vr_calib_max_jitter_deg"]),
            float(self._vr["amo_vr_calib_max_jitter_m"]),
        )
        self._calib_chord_t0 = None
        self._calib_frames = None
        self._calib_chord_done = True
        if not ok:
            print(f"[amo-vr] calibration REJECTED: {msg}")
            return
        path = self._vr["amo_vr_calib_path"] or default_calib_path()
        saved = self._calib.save(path)
        print(
            f"[amo-vr] wrist calibration captured ({msg})"
            + (f"; saved to {path}" if saved else "; NOT saved")
        )

    def _tick_amo_arms(self, qpos, buttons, sample, sim_time=0.0):
        """Engagement is a TOGGLE on the right stick click (tap to start, tap
        to stop) -- NOT right grip, which now drives torso yaw instead (see
        _tick_amo_commands) and would otherwise double-trigger this toggle
        while the operator is just adjusting height on the same stick. Sets
        self._amo_arm_action (8) + self._amo_wrist_targets (6).

        Between the raw IK solution and those two fields sit three guards, in
        this order, each bounding a different failure:

        1. reach clamp on the *target* (before the solve) -- keeps the solver
           out of the near-singular regime it cannot be trusted in;
        2. jump guard on the *solution* -- rejects a solve that moved a joint
           implausibly far and holds the last command instead (Psi0's
           safelySetMotor);
        3. rate limit on the *command* -- bounds what a surviving solution can
           do in one tick, and turns the disengage snap into a ramp.

        Under ``amo_arm_source="gmr"`` the solve is replaced (see
        ``_tick_amo_arms_gmr``) but guards 2 and 3 are not: they bound the
        *command*, and a retargeter can hand over a bad frame just as an IK
        can. Guard 1 has nothing to act on there -- it clamps an EE target,
        and the GMR path never forms one.
        """
        gmr = self._vr.get("amo_arm_source") == "gmr"
        if self._amo_ik is None and not gmr:
            return
        click = bool(buttons.get("right_axis_click", False)) if buttons else False
        pressed = click
        if pressed and not self._amo_prev_grip_pressed:
            self._amo_arm_engaged = not self._amo_arm_engaged
            self._amo_eng_captured = False
            self._amo_warned_no_3pt = False
            print(f"[g1_bridge] amo arm teleop {'ON (re-zeroed to current pose)' if self._amo_arm_engaged else 'OFF'}")
            if not self._amo_arm_engaged:
                self._amo_arm_action = _AMO_DEFAULT_DOF_POS[15:].copy()
                self._amo_wrist_targets[:] = 0.0
                # Re-seed the filter and the limiter reference at the pose the
                # arm is actually in, so a later re-engage cannot blend in
                # anything from this engagement.
                if self._amo_ik is not None:
                    self._amo_ik.prime_filter(self._read_amo_arm_q14(qpos))
                self._amo_prev_cmd_q14 = None
                self._amo_prev_raw_q14 = None
                self._amo_jump_streak = 0
                self._blend = None
                self._vr_invalid_since = None
                self._vr_targets_held = None
                self._gmr_stale_since = None
                self._gmr_blend = None
                if self._amo_ik_log is not None:
                    self._amo_ik_log.dump()
                # head_ik mode ties torso to the same engage toggle as arms
                # -- snap back to default and stop steering commands[3:7]
                # once disengaged. joystick mode (the default) drives torso
                # independently of arm engagement, so leave it alone here.
                if self._amo_torso_control_mode == "head_ik":
                    self._amo_torso_h = _AMO_TORSO_HEIGHT_DEFAULT
                    self._amo_torso_rpy = np.zeros(3, dtype=np.float32)
                    self._amo_commands[3:7] = 0.0
                    self._amo_torso_h_cmd = 0.0
        self._amo_prev_grip_pressed = pressed

        self._tick_wrist_calibration(buttons, sample, sim_time)

        if gmr:
            self._tick_amo_arms_gmr(qpos, sample, sim_time)
            return

        absolute = self._vr["amo_vr_arm_ori_mode"] == "absolute"
        # The 3-point set is still needed by the head_ik torso mode (its neck
        # row), and by the legacy relative arm mode. Absolute arm teleop does
        # not touch it -- so it no longer needs PICO body tracking at all.
        pose = self._read_amo_3pt_pose(sample)
        if not self._amo_arm_engaged:
            return
        if not absolute and pose is None:
            if not self._amo_warned_no_3pt:
                print("[g1_bridge] amo arm teleop engaged but no VR 3-point "
                      "available (needs the real PICO source with body tracking).")
                self._amo_warned_no_3pt = True
            return
        cHead = pose[2] if pose is not None else None

        q14 = self._read_amo_arm_q14(qpos)
        if not self._amo_eng_captured:
            self._amo_eng_L, self._amo_eng_R = self._amo_wrist_fk(q14)
            if pose is not None:
                self._amo_eng_3pt_L = pose[0].copy()
                self._amo_eng_3pt_R = pose[1].copy()
            if self._amo_body_ik is not None and cHead is not None:
                self._amo_eng_head = self._amo_head_fk(qpos)
                self._amo_eng_3pt_head = cHead.copy()
            self._amo_eng_captured = True
            # Engage is continuous by construction only if the filter and the
            # limiter both start from the pose the arm is in right now.
            self._amo_ik.prime_filter(q14)
            self._amo_prev_cmd_q14 = q14.copy()
            self._amo_prev_raw_q14 = None
            self._amo_jump_streak = 0
            self._vr_invalid_since = None
            if absolute:
                # An absolute target generally does NOT equal the arm's current
                # pose, and that step must be removed here rather than absorbed
                # downstream: the jump guard would veto it and auto-disengage
                # after 0.5 s. Blending the TARGET makes the first tick's step
                # exactly zero while leaving the guard armed throughout.
                self._begin_blend(q14, sim_time, self._vr["amo_vr_engage_blend_s"])

        if absolute:
            targets = self._absolute_targets(sample)
            if targets is not None:
                if self._vr_invalid_since is not None:
                    # The operator moved during the outage, so resuming is a
                    # step -- route it through the same blend as engage.
                    self._begin_blend(q14, sim_time, self._vr["amo_vr_recover_blend_s"])
                    self._vr_invalid_since = None
                self._vr_targets_held = targets
            else:
                if self._vr_invalid_since is None:
                    self._vr_invalid_since = sim_time
                if self._vr_targets_held is None:
                    if not self._vr_warned_invalid:
                        print(
                            "[amo-vr] arm teleop engaged but this PICO source "
                            "reports no headset/controller pose (vr_pose_valid is "
                            "all zero) -- absolute mode needs a source that "
                            'tracks them. Set amo_vr_arm_ori_mode="relative" to '
                            "fall back to the body-tracking path."
                        )
                        self._vr_warned_invalid = True
                    return
                if sim_time - self._vr_invalid_since > self._vr["amo_vr_pose_hold_s"]:
                    self._disengage_arms(qpos, reason="controller pose lost")
                    return
                # Freeze the TARGET, not the command: a frozen target is a
                # fixed point of the solver, so dq -> 0 and the jump guard stays
                # naturally inert. Freezing the command instead would pin the
                # PD tracking error in place.
                targets = self._vr_targets_held
            tL, tR = (targets[0].copy(), targets[1].copy())
        else:
            tL = self._rel_target(self._amo_eng_L, self._amo_eng_3pt_L, pose[0])
            tR = self._rel_target(self._amo_eng_R, self._amo_eng_3pt_R, pose[1])

        tL, tR = self._apply_blend(tL, tR, sim_time)
        tL[:3, 3], clamped_l = clamp_reach(tL[:3, 3], self._amo_shoulder_l)
        tR[:3, 3], clamped_r = clamp_reach(tR[:3, 3], self._amo_shoulder_r)

        ik_failed = False
        try:
            q_solved, _tau = self._amo_ik.solve_ik(tL, tR, current_q=q14)
            if not np.all(np.isfinite(q_solved)):
                raise ValueError("IK returned non-finite joint angles")
        except Exception as e:
            ik_failed = True
            q_solved = None
            print(
                f"[g1_bridge] amo arm IK failed ({type(e).__name__}: {e}); holding."
            )

        committed = self._commit_arm_q14(
            q_solved, qpos, solver_failed=ik_failed, source_label="IK"
        )
        if committed is None:
            return
        prev_cmd, q_raw, q_sol, jump_rejected, rate_limited = committed

        if self._amo_ik_log is not None:
            self._amo_ik_log.add(
                (
                    sim_time,
                    float(np.linalg.norm(tL[:3, 3] - self._amo_shoulder_l)),
                    float(np.linalg.norm(tR[:3, 3] - self._amo_shoulder_r)),
                    float(clamped_l),
                    float(clamped_r),
                    float(np.max(np.abs(q_raw - prev_cmd))),
                    float(np.max(np.abs(q_sol - prev_cmd))),
                    float(self._amo_ik.last_iters),
                    float(self._amo_ik.last_grad_inf),
                    float(self._amo_ik.last_err_norm),
                    float(jump_rejected),
                    float(rate_limited),
                    float(ik_failed),
                ),
                q_raw,
                q_sol,
                q14,
                sample,
            )
            self._amo_ik_log.maybe_report(sim_time)

        if (
            self._amo_torso_control_mode == "head_ik"
            and self._amo_body_ik is not None
            and self._amo_eng_head is not None
        ):
            tHead = self._rel_target(self._amo_eng_head, self._amo_eng_3pt_head, cHead)
            q_init = self._amo_body_ik.assemble_q(
                qpos,
                base_xyz=[0.0, 0.0, self._amo_torso_h],
                base_quat_wxyz=_amo_rpy_to_quat_wxyz(self._amo_torso_rpy),
            )
            # tL/tR come from G1ArmIK's own model, whose "pelvis" is the fixed
            # kinematic root at [0,0,0] (verified: wrist FK there sits ~0.095m
            # above that origin). PinkIKSolver's "pelvis" instead FLOATS at
            # world height self._amo_torso_h (its virtual ik_pelvis_z_joint,
            # via assemble_q's base_xyz above) -- same wrist FK comes out
            # ~0.095m above THAT. Feeding tL/tR straight in as-is put every
            # arm target ~torso_h too low in Pink's frame, which the solver
            # "fixed" by dragging the whole pelvis down to close the gap --
            # this was the actual cause of the squat-on-engage. tHead needs
            # no such correction: _amo_head_fk already builds eng_head via
            # this same PinkIKSolver model, so it's already frame-consistent.
            pelvis_in_pink_world = np.eye(4)
            from scipy.spatial.transform import Rotation as _R

            pelvis_in_pink_world[:3, :3] = _R.from_euler(
                "xyz", list(self._amo_torso_rpy)
            ).as_matrix()
            pelvis_in_pink_world[2, 3] = self._amo_torso_h
            tL_pink = pelvis_in_pink_world @ tL
            tR_pink = pelvis_in_pink_world @ tR
            targets = {
                "left": (tL_pink[:3, 3], tL_pink[:3, :3]),
                "right": (tR_pink[:3, 3], tR_pink[:3, :3]),
                "head": (tHead[:3, 3], tHead[:3, :3]),
            }
            _, h_new, rpy_new = self._amo_body_ik.solve(
                q_init, targets, substeps=5, h0=self._amo_torso_h, rpy0=self._amo_torso_rpy,
            )
            self._amo_torso_h = float(h_new)
            self._amo_torso_rpy = np.asarray(rpy_new, dtype=np.float32)
            # commands layout: [vx, target_yaw, vy, height_delta, torso_yaw,
            # torso_pitch, torso_roll, arm_toggle] -- rpy_new is [roll,pitch,yaw].
            self._amo_commands[3] = self._amo_torso_h - _AMO_TORSO_HEIGHT_DEFAULT
            self._amo_commands[4] = self._amo_torso_rpy[2]
            self._amo_commands[5] = self._amo_torso_rpy[1]
            self._amo_commands[6] = self._amo_torso_rpy[0]

    def _step_amo(self, ctx, qpos):
        """Legs+waist from the sim2real-hosted AMO policy (AmoBridgeTracking);
        shoulder/elbow+wrist overwritten with the IK-driven targets computed
        above -- exactly mirroring how AMOG1Composite builds one pd_target in
        a single place."""
        ctx.tracking.amo_commands = self._amo_commands
        ctx.tracking.amo_vyaw = self._amo_vyaw
        target_q, kps, kds = ctx.step_policy(qpos)
        target_q = np.asarray(target_q, dtype=np.float32).copy()
        # No blend here: _tick_amo_arms rate-limits all 14 joints before they
        # reach these fields. (There used to be a `_amo_arm_blend` ramp, but it
        # was pinned to 1.0 on every engaged tick so it was always a no-op, and
        # it only ever covered the 8 shoulder/elbow joints -- the wrists snapped
        # regardless. The limiter subsumes its intent and covers all 14.)
        if _runtime_owns_the_robot(ctx):
            target_q[self._amo_arm_sim_indices] = self._amo_arm_action
            target_q[self._amo_wrist_sim_indices] = self._amo_wrist_targets
        return target_q, kps, kds

    def _amo_teleop_data_sample(self, ctx) -> dict:
        """AMO's own left/right hand(7 each)/arm(7 each) + torso rpy/height +
        locomotion-command schema -- keyed to match
        robocasa.utils.g1_data_collection.AMO_MODALITY_BLOBS exactly (see
        AMOG1Composite.get_teleop_data_sample(), which this mirrors, sourcing
        from this composite's own amo state instead). ``ctx.cached_target_q``
        is already in the canonical SONIC 29-order (12 legs + 3 waist + 7 left
        arm + 7 right arm, see G1_JOINT_NAMES) for every backend including amo,
        so leg/arm slices don't need AMOG1Composite's own separate pd/wrist
        reconstruction."""
        if ctx.cached_target_q is None:
            return {"left_arm_action": None}

        target_q = ctx.cached_target_q
        left_arm_action = np.asarray(target_q[15:22], dtype=np.float32)
        right_arm_action = np.asarray(target_q[22:29], dtype=np.float32)
        state_qpos, _state_qvel = ctx.read_joint_state()
        left_arm_state = np.asarray(state_qpos[15:22], dtype=np.float32)
        right_arm_state = np.asarray(state_qpos[22:29], dtype=np.float32)

        hands = read_hand_state(ctx)
        zeros7 = np.zeros(7, dtype=np.float32)
        left_hand_state = np.asarray(hands.get("left", zeros7), dtype=np.float32)
        right_hand_state = np.asarray(hands.get("right", zeros7), dtype=np.float32)
        left_hand_action = grip_to_joint_targets(
            float(np.clip(ctx.left_trigger, 0.0, 1.0)), "left"
        ).astype(np.float32)
        right_hand_action = grip_to_joint_targets(
            float(np.clip(ctx.right_trigger, 0.0, 1.0)), "right"
        ).astype(np.float32)

        # commands[4:7] = [torso_yaw, torso_pitch, torso_roll] -> reorder to
        # the conventional (roll, pitch, yaw); commands[3] is a height DELTA
        # from the 0.75 baseline (see _amo_scale_height / _tick_amo_commands).
        rpy_action = np.array(
            [self._amo_commands[6], self._amo_commands[5], self._amo_commands[4]],
            dtype=np.float32,
        )
        height_action = np.array(
            [_AMO_TORSO_HEIGHT_DEFAULT + self._amo_commands[3]], dtype=np.float32
        )

        # ...and the rpy/height STATE: the command that was in force during the
        # PREVIOUS tick (schema_version >= 7). This is Psi0's definition for its
        # G1 wholebody datasets (raw_to_lerobot.py build_obs: the state's
        # torso_rpy/torso_height are prev_rpy_height, seeded 0/0.75), so a
        # policy trained on either reads the same thing: "the posture the torso
        # is currently being told to hold". Same on both hosts -- a G1 has no
        # odometry, so there is no measured torso height on hardware, and this
        # is what lets the real layout keep the column.
        #
        # History, so nobody reintroduces either: until 2026-08-07 both fields
        # were the CURRENT command (state == action, bit-identical -- a policy
        # learns to echo itself); from 2026-08-07 to schema v6 they were the
        # measured waist joints and read_root_pos()[2] -- but that z is
        # robosuite's root frame, which sits 0.793 m below the physical pelvis
        # (robot.xml: pelvis body at z=0.793), so the column read ~-0.04 m
        # against a 0.75 m action, and the waist joints are torso-vs-pelvis
        # while the command is torso attitude in the world. The measured
        # attitude now goes into observation.root_orientation /
        # projected_gravity (modality.AMO_MODALITY_BLOBS) instead.
        rpy_state = np.asarray(self._amo_torso_state_rpy, dtype=np.float32).reshape(3)
        height_state = np.array([self._amo_torso_state_h], dtype=np.float32)

        return {
            "left_hand_state": left_hand_state,
            "right_hand_state": right_hand_state,
            "left_arm_state": left_arm_state,
            "right_arm_state": right_arm_state,
            "rpy_state": rpy_state,
            "height_state": height_state,
            "left_hand_action": left_hand_action,
            "right_hand_action": right_hand_action,
            "left_arm_action": left_arm_action,
            "right_arm_action": right_arm_action,
            "rpy_action": rpy_action,
            "height_action": height_action,
            "torso_vx": np.array([self._amo_commands[0]], dtype=np.float32),
            "torso_vy": np.array([self._amo_commands[2]], dtype=np.float32),
            "torso_vyaw": np.array([self._amo_vyaw], dtype=np.float32),
            # The heading RELATIVE to where this episode started, not
            # commands[1]'s world heading: the images a policy learns from carry
            # no world frame, so an absolute yaw would make the same walk a
            # different label in every scene (spawn yaw varies per scene, e.g.
            # 2.81 rad in FloorPlan329). set_velocity_command re-anchors it.
            "target_yaw": np.array([self._amo_heading_rel], dtype=np.float32),
        }
