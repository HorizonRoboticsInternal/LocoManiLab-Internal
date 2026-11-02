"""Stateless sim-layer primitives shared by every G1 composite controller.

Two composite controllers drive the G1 in this repo -- ``WholeBodyG1Composite``
(``WHOLE_BODY_G1``) and ``G1BridgeComposite`` (``G1_BRIDGE``), both siblings
of this module in ``g1_bridge/``. They differ entirely in *what* they
compute for a given tick; they were identical in *how* they talk to MuJoCo, and
each carried its own copy of that code (difflib similarity 0.85-0.999 on the
matched methods). This module is the single copy.

Everything here is a free function over ``(sim, ...)`` -- no controller state, no
``self`` -- so the composites keep owning their own fields and just delegate.

Checkpoints are NEVER kept in this repo: ``resolve_ckpt_path`` resolves relative
paths against the **g1_backend repo root** (the directory holding the ``sim2real``
package -- the distribution is named ``g1_backend`` but imports as ``sim2real`` --
located via ``find_spec``, override with ``$SIM2REAL_ROOT``),
so a config value looks like ``checkpoints/sonic/low_latency/g1/model_encoder.onnx``
-- the same convention ``g1_bridge`` already uses for its ``policy.yaml``
paths. This file's own location no longer matters for path resolution.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Optional, Tuple

import numpy as np

# Root of the sim2real repo, i.e. the directory that contains ``checkpoints/``.
# Resolved lazily and cached: find_spec locates the package without importing it
# (importing sim2real pulls in torch/onnxruntime, which this module must not do).
_SIM2REAL_ROOT_ENV = "SIM2REAL_ROOT"
_sim2real_root_cache: Optional[str] = None
_sim2real_root_resolved = False


def sim2real_root():
    """Absolute path of the sim2real repo root, or None if it can't be found."""
    global _sim2real_root_cache, _sim2real_root_resolved
    if _sim2real_root_resolved:
        return _sim2real_root_cache
    root = os.environ.get(_SIM2REAL_ROOT_ENV)
    if not root:
        try:
            spec = importlib.util.find_spec("sim2real")
        except (ImportError, ValueError):
            spec = None
        if spec is not None and spec.origin:
            # <root>/sim2real/__init__.py -> <root>
            root = os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))
    _sim2real_root_cache = os.path.abspath(root) if root else None
    _sim2real_root_resolved = True
    return _sim2real_root_cache


# The G1's 29-DoF MuJoCo joint order (matches whole_body_teleop/utils/
# bydmimic_utils.py, and -- asserted at import in g1_bridge -- sim2real's
# own ``G1_CFG.joint_names``).
G1_JOINT_NAMES: Tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

NUM_G1_JOINTS = len(G1_JOINT_NAMES)

# Identity of the "real hardware" PICO sources, as opposed to mock/keyboard.
# Kept as a name tuple (rather than an isinstance check) so this module does not
# have to import pico_vr_source -- see is_real_pico_source.
_REAL_PICO_SOURCE_NAMES = ("RealPicoVRSource", "ProcessPicoVRSource")


# --------------------------------------------------------------------------- #
# Checkpoint paths
# --------------------------------------------------------------------------- #
def resolve_ckpt_path(path):
    """Resolve a checkpoint path.

    Absolute paths are returned unchanged. A relative path is resolved against
    the sim2real repo root (see ``sim2real_root``) -- checkpoints live there
    (``checkpoints/sonic/low_latency/g1/...``), never inside this repo. If it
    does not exist there, the original string is returned so the caller can
    raise a clear FileNotFoundError.
    """
    if not path:
        return path
    if os.path.isabs(path):
        return path
    root = sim2real_root()
    if root:
        candidate = os.path.join(root, path)
        if os.path.exists(candidate):
            return candidate
    return path


# --------------------------------------------------------------------------- #
# Quaternion helpers (wxyz throughout, float32 out)
# --------------------------------------------------------------------------- #
def quat_mul(a, b):
    """Hamilton product, a*b, both wxyz."""
    aw, ax, ay, az = float(a[0]), float(a[1]), float(a[2]), float(a[3])
    bw, bx, by, bz = float(b[0]), float(b[1]), float(b[2]), float(b[3])
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float32,
    )


def quat_inv(q):
    """Inverse of a unit wxyz quaternion = conjugate."""
    return np.array([float(q[0]), -float(q[1]), -float(q[2]), -float(q[3])], dtype=np.float32)


def _yaw_only_quat(quat_wxyz):
    """The yaw component of ``quat_wxyz``, as a wxyz quaternion.

    Must stay bit-compatible with ``sim2real.utils.math.yaw_quat`` (itself
    IsaacLab's ``yaw_quat``), because that is the convention the heading-
    normalized SONIC checkpoints were trained against. Deliberately NOT
    ``heading_yaw_from_quat`` below: that one projects the +x body axis into the
    xy plane, which agrees with this only while the torso is level and diverges
    exactly when the operator leans -- i.e. in every frame where the heading
    frame is doing any work at all.

    Reimplemented rather than imported because this module must not import
    sim2real (that pulls in torch/onnxruntime -- see the module header).
    ``tests/test_math_helpers.py`` pins the two against each other.
    """
    w, x, y, z = (
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array(
        [np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)], dtype=np.float32
    )


def relative_anchor6(robot_quat_wxyz, ref_quat_wxyz, *, frame: str = "body"):
    """Return first-two-columns of the rotation matrix for ``q_robot^-1 * q_ref``,
    flattened to 6D (matches the anchor_orientation rep used by Sonic).

    ``frame="heading"`` divides out only the robot's yaw instead of its full
    orientation, which is what the sonic_v1_1 checkpoint's
    ``motion_anchor_orientation_heading`` / ``smpl_anchor_orientation_heading``
    slots expect. The two agree whenever the robot is level, so a checkpoint fed
    the wrong one looks perfectly fine standing still and only misbehaves under
    pitch/roll -- pass the frame its manifest asks for, don't sample it.
    """
    if frame == "heading":
        robot_quat_wxyz = _yaw_only_quat(robot_quat_wxyz)
    elif frame != "body":
        raise ValueError(f"frame must be 'body' or 'heading', got {frame!r}")
    rel = quat_mul(quat_inv(robot_quat_wxyz), ref_quat_wxyz)
    w, x, y, z = float(rel[0]), float(rel[1]), float(rel[2]), float(rel[3])
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + w * z)
    r20 = 2 * (x * z - w * y)
    r01 = 2 * (x * y - w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + w * x)
    # Row-major flatten of [[r00,r01],[r10,r11],[r20,r21]].
    return np.array([r00, r01, r10, r11, r20, r21], dtype=np.float32)


def heading_yaw_from_quat(quat_wxyz) -> float:
    """Yaw (rad) of a body's heading, from its wxyz quaternion.

    The +x body axis rotated into world, projected to the xy plane:
    ``yaw = atan2(fwd_y, fwd_x)``. Matches the convention used for
    ``facing_direction = [cos(yaw), sin(yaw), 0]``.
    """
    w, x, y, z = (
        float(quat_wxyz[0]),
        float(quat_wxyz[1]),
        float(quat_wxyz[2]),
        float(quat_wxyz[3]),
    )
    fwd_x = 1.0 - 2.0 * (y * y + z * z)
    fwd_y = 2.0 * (x * y + w * z)
    return float(np.arctan2(fwd_y, fwd_x))


def rotate_in_xy(local_vec, facing_dir):
    """Rotate a body-frame WASD vector into world frame using facing_dir."""
    fx, fy = float(facing_dir[0]), float(facing_dir[1])
    n = (fx * fx + fy * fy) ** 0.5
    if n < 1e-6:
        return local_vec.copy()
    c, s = fx / n, fy / n
    x, y = float(local_vec[0]), float(local_vec[1])
    return np.array([c * x - s * y, s * x + c * y, 0.0], dtype=np.float32)


def rotate_yaw(vec, dtheta):
    x, y = float(vec[0]), float(vec[1])
    c, s = np.cos(dtheta), np.sin(dtheta)
    return np.array([c * x - s * y, s * x + c * y, float(vec[2])], dtype=np.float32)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #
def is_real_pico_source(source) -> bool:
    """True for the PICO sources backed by actual hardware.

    Class-name based rather than isinstance so callers outside the robosuite
    package (and this module itself) need not import pico_vr_source.
    """
    return type(source).__name__ in _REAL_PICO_SOURCE_NAMES


def policy_due(sim_time: float, last_policy_time: float, dt: float) -> bool:
    """Whether the throttled policy should run again on this tick.

    The ``sim_time < last_policy_time`` clause is load-bearing: MuJoCo
    recovering from a divergence (or any external ``sim.reset()``) restarts the
    clock, which leaves ``last_policy_time`` permanently in the "future".
    Without it the ``>=`` comparison never fires again and the policy SILENTLY
    STOPS -- the caller keeps re-applying its last cached PD target forever.
    That failure mode is nearly invisible: the robot looks quietly stable
    rather than obviously broken.
    """
    return sim_time < last_policy_time or (sim_time - last_policy_time) >= dt - 1e-6
