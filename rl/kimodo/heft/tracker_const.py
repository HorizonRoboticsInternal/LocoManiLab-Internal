"""HEFT tracker constants, and the arithmetic that proves the transcription.

Ported from ``heft_work/kimodo_rl/tracker.py``. Two deliberate differences:

  * **numpy, not torch.** These helpers run inside the robosuite worker, which
    must never import torch (see ``vec_task_env`` and DEBUG_LOG #11: a CUDA
    context in the same process as an EGL context deadlocks on shared nodes).
    The tracker itself is an onnxruntime CPU session there.
  * Constants remain transcribed from the original HEFT training setup. The
    original training checkout is not required. ``check_against_deploy`` can
    optionally compare a separately supplied compatible deployment YAML.

The observation layout, transcribed from HEFT's deploy runtime and confirmed
term-by-term against ``heft/cfg/task/common/tracking_base.yaml``'s
``observation.policy`` block, sums to exactly 1729:

    BootIndicator                1     boot_indicator_state_obs
    TrackingCommandObsRaw      114     (13-1)*3 + 13*6      command_obs
    TargetJointPosObs          754     13 * 29 * 2          horizon: student
    TargetRootZObs              13                          horizon: student
    TargetProjectedGravityB     39     13 * 3               horizon: student
    RootAngVelBHistory          27     9 * 3
    ProjectedGravityBHistory    27     9 * 3
    JointPos                   261     9 * 29
    JointVel                   261     9 * 29
    PrevActions                232     29 * 8               steps: 8
                              ----
                              1729

``root_linacc_b_history`` is commented out in that block; if it were enabled the
sum would be 1756 and nothing here would fit.

Three traps that produce silently wrong behaviour rather than an error:

  * ``ACTION_JOINT_NAMES`` is NOT ``DATASET_JOINT_NAMES``. 1276 of the 1729 dims
    are in the former; motion files and Kimodo are in the latter. Getting this
    wrong topples the robot slowly, which reads like a modelling problem.
  * ``ACTION_SCALE`` is per-joint and in the ACTION order, so it must be applied
    *before* any permutation.
  * The config carries both ``future_steps`` (12 entries, teacher) and
    ``student_future_steps`` (13 entries). The deployed policy is the student.
    Checking against the teacher's list would "pass" against the wrong contract.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

FUTURE_STEPS = (0, 1, 2, 3, 4, 5, 6, -1, -2, -4, -8, -12, -16)
HISTORY_STEPS = (0, 1, 2, 3, 4, 8, 12, 16, 20)
PREV_ACTION_STEPS = 8
BOOT_INDICATOR_MAX = 25
ACTION_CLIP = 10.0
OBS_DIM = 1729
N_JOINTS = 29

# Declared robot mass from the original HEFT G1 training profile.
# The retained tracker uses the same physical reference.
HEFT_ROBOT_MASS = 33.341142

# The order MOTION FILES and Kimodo use. Also the order of g1_5.xml's joints.
DATASET_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)

# The policy's action order, which is ALSO the canonical order every joint
# observation is built in. Deliberately different from the dataset order above.
ACTION_JOINT_NAMES = (
    "left_hip_pitch_joint", "right_hip_pitch_joint", "waist_yaw_joint",
    "left_hip_roll_joint", "right_hip_roll_joint", "waist_roll_joint",
    "left_hip_yaw_joint", "right_hip_yaw_joint", "waist_pitch_joint",
    "left_knee_joint", "right_knee_joint",
    "left_shoulder_pitch_joint", "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint", "right_ankle_pitch_joint",
    "left_shoulder_roll_joint", "right_shoulder_roll_joint",
    "left_ankle_roll_joint", "right_ankle_roll_joint",
    "left_shoulder_yaw_joint", "right_shoulder_yaw_joint",
    "left_elbow_joint", "right_elbow_joint",
    "left_wrist_roll_joint", "right_wrist_roll_joint",
    "left_wrist_pitch_joint", "right_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_wrist_yaw_joint",
)

CANONICAL_JOINT_NAMES = ACTION_JOINT_NAMES  # same list, named for its other role

ACTION_SCALE = np.array(
    [0.5] * 11 + [1.0, 1.0] + [0.5, 0.5] + [1.0, 1.0] + [0.5, 0.5] + [1.0] * 10,
    dtype=np.float32,
)

DEFAULT_JOINT_POS = np.array([
    -0.28, 0.0, 0.0, 0.5, -0.23, 0.0, -0.28, 0.0, 0.0, 0.5, -0.23, 0.0,
    0.0, 0.0, 0.0, 0.35, 0.16, 0.0, 0.87, 0.0, 0.0, 0.0,
    0.35, -0.16, 0.0, 0.87, 0.0, 0.0, 0.0,
], dtype=np.float32)   # DATASET order; root at [0, 0, 0.78], quat identity


# --------------------------------------------------------------------------- #
# Name-resolved permutations. Never index by position: that is how 1276 of the
# 1729 dims ended up misaligned in heft_work E6.
# --------------------------------------------------------------------------- #
def permutation(src: tuple[str, ...], dst: tuple[str, ...]) -> np.ndarray:
    """Index array ``p`` such that ``src_array[p]`` is ordered like ``dst``."""
    if sorted(src) != sorted(dst):
        missing = set(dst) - set(src)
        extra = set(src) - set(dst)
        raise ValueError(f"joint name sets differ: missing={sorted(missing)} extra={sorted(extra)}")
    pos = {n: i for i, n in enumerate(src)}
    return np.array([pos[n] for n in dst], dtype=np.int64)


DS_TO_CANON = permutation(DATASET_JOINT_NAMES, CANONICAL_JOINT_NAMES)
CANON_TO_DS = permutation(CANONICAL_JOINT_NAMES, DATASET_JOINT_NAMES)

# The action offset the policy's delta is added to, expressed in canonical order.
DEFAULT_JOINT_POS_CANON = DEFAULT_JOINT_POS[DS_TO_CANON]


# --------------------------------------------------------------------------- #
# Quaternion helpers (numpy; wxyz throughout, which is MuJoCo's xquat order)
# --------------------------------------------------------------------------- #
def quat_to_mat(q_wxyz: np.ndarray) -> np.ndarray:
    """[..., 4] wxyz -> [..., 3, 3]."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.stack([
        1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
        2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
        2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
    ], axis=-1)
    return R.reshape(*q.shape[:-1], 3, 3)


def quat_apply_inv(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` [..., 3] into the frame given by ``q`` [..., 4], i.e. R^T v."""
    R = quat_to_mat(q_wxyz)
    return np.einsum("...ij,...i->...j", R, np.asarray(v, dtype=np.float64))


def quat_to_rot6d_rel(q_cur: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
    """rot6d of ``R_cur^T R_ref``, in the runtime's first-two-COLUMNS layout.

    The runtime builds it as ``rel[:, :, :2].transpose(0, 2, 1).reshape(-1)`` --
    the first two columns read out column-major, not the first two rows. Rows
    would also be a valid 6-D rotation encoding and would also "work", silently,
    at a different answer.
    """
    rel = np.swapaxes(quat_to_mat(q_cur), -1, -2) @ quat_to_mat(q_ref)
    return np.swapaxes(rel[..., :, :2], -1, -2).reshape(*rel.shape[:-2], 6)


def projected_gravity_b(q_wxyz: np.ndarray) -> np.ndarray:
    """Gravity direction expressed in the body frame: R^T @ [0, 0, -1]."""
    return quat_apply_inv(q_wxyz, np.array([0.0, 0.0, -1.0]))


# --------------------------------------------------------------------------- #
# Observation width, derived rather than asserted
# --------------------------------------------------------------------------- #
def obs_blocks() -> list[tuple[str, int]]:
    """The 10 observation blocks, in the order the runtime concatenates them.

    Returned as a list so the equivalence test (h2) can slice the 1729-vector by
    block and report max|delta| per block. A single scalar over all 1729 dims
    tells you something is wrong; the per-block table tells you *what*.
    """
    n_f, n_h = len(FUTURE_STEPS), len(HISTORY_STEPS)
    return [
        ("BootIndicator", 1),
        ("TrackingCommandObsRaw", (n_f - 1) * 3 + n_f * 6),
        ("TargetJointPos", n_f * N_JOINTS * 2),
        ("TargetRootZ", n_f),
        ("TargetProjectedGravityB", n_f * 3),
        ("RootAngVelBHistory", n_h * 3),
        ("ProjectedGravityBHistory", n_h * 3),
        ("JointPos", n_h * N_JOINTS),
        ("JointVel", n_h * N_JOINTS),
        ("PrevActions", N_JOINTS * PREV_ACTION_STEPS),
    ]


def obs_slices() -> dict[str, slice]:
    out, off = {}, 0
    for name, width in obs_blocks():
        out[name] = slice(off, off + width)
        off += width
    return out


def obs_width() -> int:
    return sum(w for _, w in obs_blocks())


# --------------------------------------------------------------------------- #
# Optional comparison with an explicitly selected deployment config
# --------------------------------------------------------------------------- #

# Optional HEFT deployment YAML selected by the caller. It states both joint
# orders, action_clip and an explicit 29-entry action_scale.
DEFAULT_DEPLOY_CFG = (
    Path(os.environ["HEFT_DEPLOY_CFG"]).expanduser()
    if os.environ.get("HEFT_DEPLOY_CFG") else None
)


def check_against_deploy(deploy_cfg: str | Path | None = DEFAULT_DEPLOY_CFG,
                         verbose: bool = True) -> bool:
    """Verify against HEFT's sim2real deploy YAML. Returns False if absent.

    The selected file states both joint orders and the 29-entry action_scale
    literally. This optional check does not require the original HEFT training
    checkout.
    """
    if deploy_cfg is None:
        if verbose:
            print("[tracker_const] no deploy config selected; set HEFT_DEPLOY_CFG "
                  "or pass deploy_cfg to check it; skipped")
        return False

    import yaml

    deploy_cfg = Path(deploy_cfg).expanduser()
    if not deploy_cfg.exists():
        if verbose:
            print(f"[tracker_const] deploy config not found at {deploy_cfg}; skipped")
        return False

    with open(deploy_cfg) as f:
        cfg = yaml.safe_load(f)

    checks = {
        "future_steps": (list(FUTURE_STEPS), cfg["future_steps"]),
        "joint_pos_history_steps": (list(HISTORY_STEPS), cfg["joint_pos_history_steps"]),
        "joint_vel_history_steps": (list(HISTORY_STEPS), cfg["joint_vel_history_steps"]),
        "root_angvel_history_steps": (list(HISTORY_STEPS), cfg["root_angvel_history_steps"]),
        "projected_gravity_history_steps": (
            list(HISTORY_STEPS), cfg["projected_gravity_history_steps"]),
        "prev_action_steps": (PREV_ACTION_STEPS, cfg["prev_action_steps"]),
        "action_clip": (ACTION_CLIP, cfg["action_clip"]),
        "boot_indicator_max": (BOOT_INDICATOR_MAX, cfg["boot_indicator_max"]),
        "action_joint_names": (list(ACTION_JOINT_NAMES), cfg["action_joint_names"]),
        "dataset_joint_names": (list(DATASET_JOINT_NAMES), cfg["dataset_joint_names"]),
        "action_scale": (ACTION_SCALE.tolist(), cfg["action_scale"]),
    }
    bad = {k: (a, b) for k, (a, b) in checks.items() if a != b}
    if bad:
        lines = "\n".join(f"    {k}:\n      here   = {a}\n      deploy = {b}"
                          for k, (a, b) in bad.items())
        raise SystemExit(f"[tracker_const] constants disagree with {deploy_cfg}:\n{lines}")

    # Would widen the observation past 1729 if switched on.
    if cfg.get("use_compliance_flag_obs", False):
        raise SystemExit("[tracker_const] use_compliance_flag_obs is True in the deploy "
                         "config; the observation is wider than 1729")

    # NOTE: root_linacc_history_steps is PRESENT in this file (9 steps) but the
    # term is disabled upstream -- it is commented out in tracking_base.yaml's
    # observation.policy block. If it were live the input would be 1756, not
    # 1729. So its presence here must NOT be treated as evidence either way;
    # policy.json's in_shapes is what settles it, and h3 checks that.

    if verbose:
        print(f"[tracker_const] constants match the DEPLOY config {deploy_cfg.name} "
              f"(both joint orders and action_scale stated literally)")
    return True
