"""PICO VR data source for SONIC teleop mode.

Provides an abstract ``PicoVRSource`` interface and two implementations:

  - ``MockPicoVRSource`` — returns hardcoded "T-pose / hands at sides" 3-point
    data.  Cross-platform.  Used for plumbing tests and Mac development since
    the real PICO SDK is Linux-only.

  - ``ProcessPicoVRSource`` — default real PICO source. Runs
    ``xrobotoolkit_sdk`` in an isolated child process so native SDK crashes do
    not take down MuJoCo / ONNX / viewer in the data-collection process.

  - ``RealPicoVRSource`` — legacy in-process wrapper around
    ``xrobotoolkit_sdk`` (Linux only), kept for debugging with
    ``PICO_DIRECT_XRT=1``.

Also vendored from ``gear_sonic/scripts/pico_manager_thread_server.py``:

  - ``_process_3pt_pose(smpl_pose_np)`` — pure scipy/numpy transform from
    24-joint SMPL body poses to 3-point VR pose (L-wrist, R-wrist, Neck)
    in robot-relative frame.  Cross-platform.

  - ``ThreePointPose`` — calibration logic (captures initial neck orientation
    and aligns wrist positions to G1 FK reference).  Cross-platform.
"""

from __future__ import annotations

import abc
import atexit
import multiprocessing as mp
import os
import pathlib
import queue
import shutil
import subprocess
import threading
import time
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as sRot


# =============================================================================
# Constants — match SONIC's encoder layout for VR3Point inputs.
# =============================================================================
VR_3POINT_POSITIONS_DIM = 9  # 3 points × 3D position
VR_3POINT_ORIENTATIONS_DIM = 12  # 3 points × 4D quaternion (wxyz)

# Per-keypoint rotation offsets, copied from
# gear_sonic/scripts/pico_manager_thread_server.py — align SMPL joint frames
# with the robot/visualization coordinate convention.  Index mapping:
#   OFFSETS[0]: Root/Pelvis (SMPL joint 0)
#   OFFSETS[1]: Left Wrist  (SMPL joint 22)
#   OFFSETS[2]: Right Wrist (SMPL joint 23)
#   OFFSETS[3]: Neck        (SMPL joint 12)
_KEYPOINT_ROTATION_OFFSETS = [
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),
    sRot.from_euler("xyz", [90, 0, 0], degrees=True),
    sRot.from_euler("xyz", [-90, 0, 180], degrees=True),
    sRot.from_euler("xyz", [0, 0, -90], degrees=True),
]

# Unity (Y-up, left-handed) → Robot (Z-up, right-handed) conversion matrix.
# Unity [x, y, z] -> Robot [-x, z, y]
#
# ONLY valid as half of a package with _KEYPOINT_ROTATION_OFFSETS — see
# _Q_YUP2ZUP below before reusing it for anything else.
_Q_UNITY_TO_ROBOT = np.array([[-1, 0, 0], [0, 0, 1], [0, 1, 0.0]])


# =============================================================================
# Direct 6-DoF pose frames (headset + controllers)
#
# Separate from the SMPL path above, deliberately. The SMPL 3-point output is
# SONIC-encoder data: calibration-relative, anchored to nothing on the robot.
# These helpers are the metric path — head-relative device poses in robot axes,
# suitable as IK targets.
#
# Frames, named once so every transform below can say what it maps:
#   A  SDK/OpenXR world:  right-handed, +Y up, -Z forward, +X right
#   R  robot:             right-handed, +X forward, +Y left, +Z up
#   H  operator head (or its yaw-only projection)
#   D  device body frame, re-expressed in robot axes (see pose7_to_robot)
# =============================================================================

# A-coords -> R-coords:  v_R = _Q_YUP2ZUP @ v_A.  det = +1.
#
# NOT the same matrix as _Q_UNITY_TO_ROBOT, and the difference is load-bearing:
# measured, Rz(+90°) @ _Q_UNITY_TO_ROBOT == _Q_YUP2ZUP == Psi0's
# grd_yup2grd_zup. The SMPL path's 90° yaw is supplied by
# _KEYPOINT_ROTATION_OFFSETS[0] = Rz(-90°) on the root and cancels during the
# root-relative step, so _Q_UNITY_TO_ROBOT is only correct bundled with those
# offsets. Using it on a headset/controller pose silently yaws everything 90°.
# tests/test_amo_vr_frames.py asserts the identity so the two cannot drift.
_Q_YUP2ZUP = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0.0]])


def pose7_to_robot(pose7) -> np.ndarray:
    """SDK pose -> robot axes.

    Args:
        pose7: (7,) ``[x, y, z, qx, qy, qz, qw]`` in frame A (scalar-LAST, the
            layout every ``xrt.get_*_pose()`` returns).

    Returns:
        (7,) ``[x, y, z, qw, qx, qy, qz]`` in frame R (scalar-FIRST, matching
        ``vr_3point_local_orn_target``).

    The rotation is converted as a **similarity transform** ``Q R Qᵀ``, not
    ``Q R``. That is what makes the result a rotation between two frames that
    have BOTH been re-expressed in robot axes, so the device body frame
    downstream is ``Q·(grip frame)``. The wrist calibration matrix is measured
    against exactly that quantity, so the two are consistent by construction —
    "simplifying" this to a single multiply breaks the calibration silently.
    """
    p = np.asarray(pose7, dtype=float).reshape(7)
    Q = _Q_YUP2ZUP
    pos = Q @ p[:3]
    rot = Q @ sRot.from_quat(p[3:], scalar_first=False).as_matrix() @ Q.T
    quat = sRot.from_matrix(rot).as_quat(scalar_first=True)
    return np.concatenate([pos, quat]).astype(np.float32)


def robot_pose7_to_mat(pose7_wxyz) -> np.ndarray:
    """(7,) ``[xyz, qw,qx,qy,qz]`` in frame R -> 4x4 SE3."""
    p = np.asarray(pose7_wxyz, dtype=float).reshape(7)
    T = np.eye(4)
    T[:3, :3] = sRot.from_quat(p[3:], scalar_first=True).as_matrix()
    T[:3, 3] = p[:3]
    return T


def head_relative_pose(head7_wxyz, device7_wxyz, yaw_only: bool = False) -> np.ndarray:
    """Device pose expressed in the operator's head frame. Both inputs are
    robot-axes poses from :func:`pose7_to_robot`; the result is 4x4 SE3.

    ``R_rel = R_headᵀ R_dev`` and ``p_rel = R_headᵀ (p_dev - p_head)`` — a
    LEFT multiplication, i.e. a change of *reference* frame (A -> H), not a
    change of body frame. Equivalent to Psi0's
    ``fast_mat_inv(head_mat) @ wrist_mat`` (vr_pico.py:157-162).

    This is the invariant the whole absolute-teleop design rests on: the
    operator can stand anywhere and face any direction and get the same result
    for the same body-relative gesture.

    Args:
        yaw_only: reference against the head's yaw only, dropping its pitch and
            roll. With the full head frame (Psi0 parity) the operator looking
            down rotates the arm targets with them; with yaw_only it does not.
    """
    head = robot_pose7_to_mat(head7_wxyz)
    dev = robot_pose7_to_mat(device7_wxyz)
    R_ref = head[:3, :3]
    if yaw_only:
        yaw = np.arctan2(R_ref[1, 0], R_ref[0, 0])
        c, s = np.cos(yaw), np.sin(yaw)
        R_ref = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.eye(4)
    T[:3, :3] = R_ref.T @ dev[:3, :3]
    T[:3, 3] = R_ref.T @ (dev[:3, 3] - head[:3, 3])
    return T


def pose_is_valid(pose7) -> bool:
    """Whether an SDK pose carries real tracking data.

    Guards the three ways the SDK reports "no data": an all-zero quaternion
    (also what Psi0 checks, via a determinant test), non-finite components, and
    an exactly-zero position. Feeding any of those to the IK is an instant
    full-arm slam.
    """
    p = np.asarray(pose7, dtype=float).reshape(-1)
    if p.size < 7 or not np.all(np.isfinite(p)):
        return False
    if np.linalg.norm(p[3:7]) < 1e-6:
        return False
    return bool(np.any(np.abs(p[:3]) > 1e-9))


# =============================================================================
# SMPL skeleton tree — parent of each of the 24 SMPL joints.
# Verbatim from ``gear_sonic/scripts/pico_manager_thread_server.py:1200-1226``
# (the [:24] slice).  parent_indices[i] is the SMPL joint index of joint i's
# parent, or -1 for the root.  Used for global -> local rotation chaining.
# =============================================================================
_SMPL_PARENT_INDICES = [
    -1,
    0,
    0,
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    9,
    9,
    12,
    13,
    14,
    16,
    17,
    18,
    19,
    20,
    22,
]

# SMPL joint indices referenced by the wrist retargeting (POSE mode, mode_id=2).
# Convention matches pico_manager_thread_server.py:1349-1362.
_SMPL_L_ELBOW_IDX = 17  # body_pose row index (= SMPL joint 18 with pelvis excluded)
_SMPL_R_ELBOW_IDX = 18  # body_pose row index (= SMPL joint 19)
_SMPL_L_WRIST_IDX = 19  # body_pose row index (= SMPL joint 20)
_SMPL_R_WRIST_IDX = 20  # body_pose row index (= SMPL joint 21)

# =============================================================================
# SMPL forward-kinematics skeleton — vendored from
# ``gear_sonic/data/human/human_joints_info.pkl`` as an npz (J rest joints +
# parent chain).  Used to reproduce ``compute_human_joints`` (SMPL FK) in pure
# numpy, so the POSE-mode ``smpl_joints_local`` matches what the SONIC policy
# was trained on — the *canonical* SMPL skeleton joints, NOT the raw
# operator-tracked PICO positions.
# =============================================================================
# Package-root relative: this module lives in pico/, the asset in vendored/wbt/.
_SMPL_INFO_NPZ = (
    pathlib.Path(__file__).resolve().parent.parent
    / "vendored"
    / "wbt"
    / "data"
    / "smpl_human_joints_info.npz"
)
_SMPL_J: Optional[np.ndarray] = None  # (55, 3) rest joint positions
_SMPL_FK_PARENTS: Optional[list] = None  # (55,) full SMPL+hand parent chain

# 90° about X maps SMPL's Y-up root frame to the robot's Z-up frame
# (gear_sonic/isaac_utils/rotations.py:smpl_root_ytoz_up).
_SMPL_YTOZ = sRot.from_rotvec([np.pi / 2, 0.0, 0.0])
# SMPL's default rest orientation [0.5,0.5,0.5,0.5] (120° about [1,1,1]); the
# pipeline conjugates it out (remove_smpl_base_rot).
_SMPL_BASE_ROT = sRot.from_quat([0.5, 0.5, 0.5, 0.5], scalar_first=True)


def _load_smpl_skeleton() -> None:
    """Lazy-load the vendored SMPL rest skeleton (J + parents)."""
    global _SMPL_J, _SMPL_FK_PARENTS
    if _SMPL_J is not None:
        return
    data = np.load(str(_SMPL_INFO_NPZ))
    _SMPL_J = data["J"].astype(np.float64)
    _SMPL_FK_PARENTS = data["parents_list"].tolist()


def _smpl_fk_joints(
    global_orient_aa: np.ndarray, body_pose_63: np.ndarray
) -> np.ndarray:
    """Pure-numpy port of ``compute_human_joints`` (SMPL forward kinematics).

    Args:
        global_orient_aa: (3,) root orientation, axis-angle.
        body_pose_63:     (63,) body joints 1..21 local rotations, axis-angle.

    Returns:
        (24, 3) joint positions — SMPL joints 0..21 plus the two thumb tips
        (indices 39, 54), matching ``compute_human_joints``'s output order.
    """
    _load_smpl_skeleton()
    full = np.zeros((55, 3), dtype=np.float64)
    full[0] = global_orient_aa
    full[1:22] = np.asarray(body_pose_63, dtype=np.float64).reshape(21, 3)
    rot_mats = sRot.from_rotvec(full).as_matrix()  # (55,3,3)

    rel = _SMPL_J.copy()
    rel[1:] -= _SMPL_J[[_SMPL_FK_PARENTS[i] for i in range(1, 55)]]

    T = np.zeros((55, 4, 4), dtype=np.float64)
    T[:, :3, :3] = rot_mats
    T[:, :3, 3] = rel
    T[:, 3, 3] = 1.0

    chain = [None] * 55
    chain[0] = T[0]
    for i in range(1, 55):
        chain[i] = chain[_SMPL_FK_PARENTS[i]] @ T[i]
    joints = np.stack([chain[i][:3, 3] for i in range(55)])  # (55,3)
    out_idx = list(range(22)) + [39, 54]
    return joints[out_idx]


# =============================================================================
# SMPL pose -> 3-point processing (vendored from pico_manager_thread_server.py)
# =============================================================================


def _compute_rel_transform(pose, world_frame, scalar_first=True):
    """Transform a single pose from Unity frame to robot frame, relative to
    ``world_frame``.

    Args:
        pose: (7,) [x, y, z, qx, qy, qz, qw] in Unity frame.
        world_frame: (7,) reference frame to compute relative transform.
        scalar_first: True if quaternion is [w, x, y, z].

    Returns:
        (rel_pos[3], rel_quat[4]) — relative pose in robot frame (quat is wxyz).
    """
    world_frame = world_frame.copy()
    Q = _Q_UNITY_TO_ROBOT
    pose[:3] = Q @ pose[:3]
    world_frame[:3] = Q @ world_frame[:3]
    rot_base = sRot.from_quat(world_frame[3:], scalar_first=scalar_first).as_matrix()
    rot = sRot.from_quat(pose[3:], scalar_first=scalar_first).as_matrix()
    rel_rot = sRot.from_matrix(Q @ (rot_base.T @ rot) @ Q.T)
    rel_pos = sRot.from_matrix(Q @ rot_base.T @ Q.T).apply(pose[:3] - world_frame[:3])
    return rel_pos, rel_rot.as_quat(scalar_first=True)


def process_3pt_pose(smpl_pose_np):
    """Extract 3-point VR pose (L-Wrist, R-Wrist, Neck) from full 24-joint SMPL.

    Args:
        smpl_pose_np: (24, 7) array — 24 SMPL joints, each
            [x, y, z, qx, qy, qz, qw] in Unity frame (scalar-last quat).

    Returns:
        vr_3pt_pose: (3, 7) — 3 keypoints in robot frame, each row
            [x, y, z, qw, qx, qy, qz] (scalar-FIRST), RELATIVE TO ROOT.
            Row 0: Left Wrist, Row 1: Right Wrist, Row 2: Neck.

    Note: Position calibration (wrist offsets, neck kinematic chain) is done
    in :class:`ThreePointPose.apply_calibration`.
    """
    smpl_pose_np = smpl_pose_np.copy()  # defensive: _compute_rel_transform mutates

    # Step 1: Unity -> robot frame for all 24 joints.
    body_poses = np.zeros((smpl_pose_np.shape[0], 7), dtype=np.float32)
    for i in range(smpl_pose_np.shape[0]):
        pos, orn = _compute_rel_transform(
            smpl_pose_np[i], [0, 0, 0, 0, 0, 0, 1], scalar_first=False
        )
        body_poses[i, :3] = pos
        body_poses[i, 3:] = orn

    # Step 2 & 3: Extract 4 keypoints (Root, L-Wrist, R-Wrist, Neck), apply offsets.
    kp_indices = [0, 22, 23, 12]
    kp_poses = np.zeros((4, 7), dtype=np.float32)
    for i, pose in enumerate(body_poses):
        if i not in kp_indices:
            continue
        rel_i = kp_indices.index(i)
        quat_wxyz = pose[3:7].copy()
        rot_xyzw = (
            sRot.from_quat(quat_wxyz, scalar_first=True)
            * _KEYPOINT_ROTATION_OFFSETS[rel_i]
        ).as_quat(scalar_first=False)
        kp_poses[rel_i, 3:] = rot_xyzw  # temporarily scalar-last for scipy
        kp_poses[rel_i, :3] = pose[:3]

    # Step 4: Make positions and orientations relative to root.
    root_pos = kp_poses[0, :3].copy()
    root_quat = kp_poses[0, 3:].copy()  # still scalar-last
    for i in range(1, 4):
        kp_poses[i, :3] = (
            sRot.from_quat(root_quat).inv().apply(kp_poses[i, :3] - root_pos)
        )
        kp_poses[i, 3:] = (
            sRot.from_quat(root_quat).inv() * sRot.from_quat(kp_poses[i, 3:])
        ).as_quat(scalar_first=True)

    # Step 5: Drop root, return L-Wrist, R-Wrist, Neck.
    return kp_poses[1:]


# =============================================================================
# Full-body SMPL pose -> SONIC POSE-mode encoder inputs (mode_id=2)
# =============================================================================


def _decompose_rotation_aa(rotation_aa: np.ndarray, twist_axis: np.ndarray):
    """Swing-twist decomposition of an axis-angle rotation.

    Verbatim port of ``decompose_rotation_aa`` from
    ``gear_sonic/trl/utils/rotation_conversion.py:605``.

    Args:
        rotation_aa: (N, 3) axis-angle rotations.
        twist_axis: (3,) unit vector — the joint's rotation axis.

    Returns:
        (q_twist, q_swing) each (N, 4) wxyz.  q = q_twist * q_swing.
    """
    angle = np.linalg.norm(rotation_aa, axis=1)[:, None]
    angle = np.where(angle == 0, 1.0, angle)  # avoid /0; v will be 0 anyway
    w = np.cos(angle / 2)
    v = np.sin(angle / 2) * rotation_aa / angle
    v_twist = np.dot(v, twist_axis)[:, None] * twist_axis
    q_twist = np.concatenate([w, v_twist], axis=1)
    q_twist /= np.linalg.norm(q_twist, axis=1, keepdims=True)
    # q_swing = q_twist^-1 * q_full  (Hamilton product, both scalar-first wxyz)
    q_twist_inv = q_twist * np.array([1.0, -1.0, -1.0, -1.0])
    aw, ax, ay, az = (
        q_twist_inv[:, 0:1],
        q_twist_inv[:, 1:2],
        q_twist_inv[:, 2:3],
        q_twist_inv[:, 3:4],
    )
    bw, bx, by, bz = w, v[:, 0:1], v[:, 1:2], v[:, 2:3]
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    q_swing = np.concatenate([ow, ox, oy, oz], axis=1)
    return q_twist, q_swing


def process_smpl_pose_full(smpl_pose_np: np.ndarray) -> dict:
    """Extract POSE-mode (encoder mode_id=2) inputs from raw PICO SMPL data.

    Faithful 1:1 numpy port of SONIC's pipeline
    (``compute_from_body_poses`` -> ``process_smpl_joints`` -> the wrist block
    of ``PoseStreamer.run_once``).  Verified to match the real torch
    implementation to <1e-5 on random input.

    Crucially, ``smpl_joints_local`` is produced by running SMPL *forward
    kinematics* on the pose parameters (the canonical neutral-body skeleton the
    policy was trained on), NOT by transforming the raw PICO joint positions —
    the latter is out-of-distribution and makes the policy thrash.

    Args:
        smpl_pose_np: (24, 7) — 24 SMPL joints in Unity frame, each
            [x, y, z, qx, qy, qz, qw] (scalar-last quaternion), exactly as
            ``xrobotoolkit_sdk.get_body_joints_pose()`` returns.

    Returns:
        dict with:
          - ``smpl_joints_local``: (24, 3) — FK joints de-rotated by the root
            orientation (root frame).  Fed directly into the encoder.
          - ``smpl_root_quat``: (4,) wxyz — the SMPL root orientation
            (ytoz-up + base-rot-removed).  The composite turns this into the
            6D anchor *live*, relative to the robot's current base quaternion,
            mirroring the deploy's ``GatherMotionAnchorOrientationMutiFrame``.
          - ``anchor_6d``: (6,) — first-two-columns row-major of the root
            rotation matrix.  Kept for debugging / parity tests only; the
            composite does NOT use it for the encoder (it computes a
            robot-relative anchor instead).
          - ``wrist_joints``: (6,) — G1 wrist joint angles in IsaacLab wrist
            order: [L_roll, R_roll, L_pitch, R_pitch, L_yaw, R_yaw].
    """
    smpl_pose_np = np.asarray(smpl_pose_np, dtype=np.float64)

    # ---- compute_from_body_poses: global rotations -> local axis-angles ----
    positions = smpl_pose_np[:, :3]
    global_quats_wxyz = smpl_pose_np[:, [6, 3, 4, 5]]  # xyzw -> wxyz
    global_rots = sRot.from_quat(
        global_quats_wxyz, scalar_first=True
    ) * sRot.from_euler("y", 180, degrees=True)
    pose_aa = np.zeros((24, 3), dtype=np.float64)
    for i in range(24):
        p = _SMPL_PARENT_INDICES[i]
        if p == -1:
            pose_aa[i] = global_rots[i].as_rotvec()
        else:
            pose_aa[i] = (global_rots[p].inv() * global_rots[i]).as_rotvec()
    body_pose = pose_aa[1:].flatten()  # (69,) joints 1..23
    global_orient = pose_aa[0]  # (3,)

    # ---- process_smpl_joints: SMPL FK + de-rotate by global orient ----
    go = sRot.from_rotvec(global_orient)
    go_ytoz = _SMPL_YTOZ * go  # smpl_root_ytoz_up
    joints = _smpl_fk_joints(go_ytoz.as_rotvec(), body_pose[:63])  # (24,3)
    go_final = go_ytoz * _SMPL_BASE_ROT.inv()  # remove_smpl_base_rot
    smpl_joints_local = go_final.inv().apply(joints).astype(np.float32)
    smpl_root_quat = go_final.as_quat(scalar_first=True).astype(np.float32)
    M = go_final.as_matrix()
    anchor_6d = np.array(
        [M[0, 0], M[0, 1], M[1, 0], M[1, 1], M[2, 0], M[2, 1]], dtype=np.float32
    )

    # ---- Wrist retargeting — SONIC PoseStreamer.run_once lines 1346-1403 ----
    # body_pose here is in the (180°-Y-flipped) global LOCAL frame, with NO Q
    # conjugation — exactly what compute_from_body_poses produces.
    bp = body_pose[:63].reshape(21, 3)
    g1_axis_y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    _, q_l_elbow_swing = _decompose_rotation_aa(
        bp[_SMPL_L_ELBOW_IDX : _SMPL_L_ELBOW_IDX + 1], g1_axis_y
    )
    _, q_r_elbow_swing = _decompose_rotation_aa(
        bp[_SMPL_R_ELBOW_IDX : _SMPL_R_ELBOW_IDX + 1], g1_axis_y
    )

    # q_*_swing is wxyz; scipy expects xyzw — reindex.
    l_swing_euler = sRot.from_quat(q_l_elbow_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    r_swing_euler = sRot.from_quat(q_r_elbow_swing[:, [1, 2, 3, 0]]).as_euler(
        "XYZ", degrees=False
    )
    l_wrist_euler = sRot.from_rotvec(
        bp[_SMPL_L_WRIST_IDX : _SMPL_L_WRIST_IDX + 1]
    ).as_euler("XYZ", degrees=False)
    r_wrist_euler = sRot.from_rotvec(
        bp[_SMPL_R_WRIST_IDX : _SMPL_R_WRIST_IDX + 1]
    ).as_euler("XYZ", degrees=False)

    g1_l_wrist_roll = l_swing_euler[0, 0] + l_wrist_euler[0, 0]
    g1_l_wrist_pitch = l_wrist_euler[0, 1]  # -(-wrist_y) = +wrist_y
    g1_l_wrist_yaw = l_swing_euler[0, 2] + l_wrist_euler[0, 2]
    g1_r_wrist_roll = -(r_swing_euler[0, 0] + r_wrist_euler[0, 0])
    g1_r_wrist_pitch = -r_wrist_euler[0, 1]
    g1_r_wrist_yaw = r_swing_euler[0, 2] + r_wrist_euler[0, 2]

    # Order matches isaaclab wrist indices {23,24,25,26,27,28} =
    # [L_roll, R_roll, L_pitch, R_pitch, L_yaw, R_yaw].
    wrist_joints = np.array(
        [
            g1_l_wrist_roll,
            g1_r_wrist_roll,
            g1_l_wrist_pitch,
            g1_r_wrist_pitch,
            g1_l_wrist_yaw,
            g1_r_wrist_yaw,
        ],
        dtype=np.float32,
    )

    return {
        "smpl_joints_local": smpl_joints_local,  # (24, 3)
        "smpl_root_quat": smpl_root_quat,  # (4,) wxyz
        "anchor_6d": anchor_6d,  # (6,) debug/parity only
        "wrist_joints": wrist_joints,  # (6,)
    }


# =============================================================================
# ThreePointPose — calibration helper (vendored & simplified)
# =============================================================================


class ThreePointPose:
    """Calibration logic for VR 3-point pose.

    On first valid frame after :meth:`reset` (or :meth:`calibrate_now`), this
    captures the operator's neck orientation and wrist position offsets so
    that subsequent VR poses are aligned with the G1 robot's default standing
    posture.

    The full implementation in
    ``gear_sonic/scripts/pico_manager_thread_server.py`` uses G1 forward
    kinematics to find the reference wrist positions; here we use hardcoded
    nominal positions for the G1 default standing pose, which avoids pulling
    in the gear_sonic FK dependency.  Refinement to use real FK happens in
    Phase 2 if needed.
    """

    # Nominal G1 wrist positions in pelvis-relative coordinates at default
    # standing pose.  Computed by FK on the SONIC default joint angles —
    # TODO(phase2): replace with live FK from robosuite sim.data on Ubuntu.
    _G1_NOMINAL_LWRIST_POS = np.array([0.214, 0.183, 0.020], dtype=np.float32)
    _G1_NOMINAL_RWRIST_POS = np.array([0.214, -0.183, 0.020], dtype=np.float32)

    TORSO_LINK_OFFSET_Z = 0.05
    NECK_LINK_LENGTH = 0.35

    def __init__(self, log_prefix: str = "ThreePointPose", verbose: bool = True):
        self.log_prefix = log_prefix
        self.verbose = verbose
        self._calibration_pending = False
        self._calibration_neck_quat_inv: Optional[np.ndarray] = None
        self._calibration_lwrist_offset: Optional[np.ndarray] = None
        self._calibration_rwrist_offset: Optional[np.ndarray] = None
        self._calibration_lwrist_rot_offset: Optional[sRot] = None
        self._calibration_rwrist_rot_offset: Optional[sRot] = None

    @property
    def is_calibrated(self) -> bool:
        return self._calibration_neck_quat_inv is not None

    @property
    def is_pending(self) -> bool:
        return self._calibration_pending

    def reset(self) -> None:
        """Clear calibration; next process_smpl_pose() will recalibrate."""
        self._calibration_neck_quat_inv = None
        self._calibration_lwrist_offset = None
        self._calibration_rwrist_offset = None
        self._calibration_lwrist_rot_offset = None
        self._calibration_rwrist_rot_offset = None
        self._calibration_pending = True
        if self.verbose:
            print(f"[{self.log_prefix}] calibration reset")

    def calibrate_now(self, smpl_pose_np: np.ndarray) -> bool:
        """Calibrate using the current SMPL frame as the zero-reference pose."""
        try:
            vr_3pt = process_3pt_pose(smpl_pose_np)
            self._capture_calibration(vr_3pt)
            return True
        except Exception as e:
            if self.verbose:
                print(f"[{self.log_prefix}] calibrate_now failed: {e}")
            return False

    def process_smpl_pose(self, smpl_pose_np: np.ndarray) -> np.ndarray:
        """Process SMPL pose to extract and calibrate 3-point VR pose."""
        vr_3pt_raw = process_3pt_pose(smpl_pose_np)
        if self._calibration_pending:
            self._capture_calibration(vr_3pt_raw)
        return self._apply_calibration(vr_3pt_raw)

    def _capture_calibration(self, vr_3pt_pose: np.ndarray) -> None:
        neck_quat_wxyz = vr_3pt_pose[2, 3:].copy()
        neck_rot = sRot.from_quat(neck_quat_wxyz, scalar_first=True)
        self._calibration_neck_quat_inv = neck_rot.inv().as_quat(scalar_first=True)
        calib_inv_rot = neck_rot.inv()

        lwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[0, :3].copy())
        rwrist_pos_corrected = calib_inv_rot.apply(vr_3pt_pose[1, :3].copy())
        lwrist_rot_corrected = calib_inv_rot * sRot.from_quat(
            vr_3pt_pose[0, 3:], scalar_first=True
        )
        rwrist_rot_corrected = calib_inv_rot * sRot.from_quat(
            vr_3pt_pose[1, 3:], scalar_first=True
        )

        self._calibration_lwrist_offset = (
            lwrist_pos_corrected - self._G1_NOMINAL_LWRIST_POS
        )
        self._calibration_rwrist_offset = (
            rwrist_pos_corrected - self._G1_NOMINAL_RWRIST_POS
        )

        # Identity wrist orientation as reference (G1 default standing has
        # wrists in a roughly forward-facing orientation, approximated as
        # identity here for now). TODO(phase2): use real G1 FK on Ubuntu.
        ident = sRot.identity()
        self._calibration_lwrist_rot_offset = ident * lwrist_rot_corrected.inv()
        self._calibration_rwrist_rot_offset = ident * rwrist_rot_corrected.inv()

        self._calibration_pending = False
        if self.verbose:
            print(
                f"[{self.log_prefix}] calibrated — L-wrist offset "
                f"{self._calibration_lwrist_offset.round(3).tolist()} | R-wrist offset "
                f"{self._calibration_rwrist_offset.round(3).tolist()}"
            )

    def _apply_calibration(self, vr_3pt_pose: np.ndarray) -> np.ndarray:
        if self._calibration_neck_quat_inv is None:
            return vr_3pt_pose

        calibrated = vr_3pt_pose.copy()
        calib_inv_rot = sRot.from_quat(
            self._calibration_neck_quat_inv, scalar_first=True
        )

        # Neck orientation: calibrated = inv(initial) * current
        neck_rot = sRot.from_quat(vr_3pt_pose[2, 3:], scalar_first=True)
        calibrated[2, 3:] = (calib_inv_rot * neck_rot).as_quat(scalar_first=True)

        # Wrist positions: rotate by neck inverse, then subtract offset
        calibrated[0, :3] = (
            calib_inv_rot.apply(vr_3pt_pose[0, :3]) - self._calibration_lwrist_offset
        )
        calibrated[1, :3] = (
            calib_inv_rot.apply(vr_3pt_pose[1, :3]) - self._calibration_rwrist_offset
        )

        # Wrist orientations: rot_offset * (neck_inv * current)
        lw = calib_inv_rot * sRot.from_quat(vr_3pt_pose[0, 3:], scalar_first=True)
        rw = calib_inv_rot * sRot.from_quat(vr_3pt_pose[1, 3:], scalar_first=True)
        calibrated[0, 3:] = (self._calibration_lwrist_rot_offset * lw).as_quat(
            scalar_first=True
        )
        calibrated[1, 3:] = (self._calibration_rwrist_rot_offset * rw).as_quat(
            scalar_first=True
        )

        # Neck position via kinematic chain: root → torso → neck (along neck Z)
        neck_z = sRot.from_quat(calibrated[2, 3:], scalar_first=True).apply([0, 0, 1])
        calibrated[2, :3] = (
            np.array([0, 0, self.TORSO_LINK_OFFSET_Z]) + self.NECK_LINK_LENGTH * neck_z
        ).astype(np.float32)

        return calibrated


# =============================================================================
# PICO button/axis polling helpers
# =============================================================================

_BUTTON_KEYS = [
    "a",
    "b",
    "x",
    "y",
    "left_axis_click",
    "right_axis_click",
    "left_menu",
    "right_menu",
]
_AXIS_KEYS = ["left_axis", "right_axis"]
_TRIGGER_KEYS = ["left_trigger", "right_trigger", "left_grip", "right_grip"]


def empty_buttons() -> dict:
    """All buttons released / sticks centred — used by MockPicoVRSource, as the
    safe default when xrt button polling fails, and by out-of-tree PicoVRSource
    implementations (g1_bridge.gmr_utils) building a sample."""
    out = {k: False for k in _BUTTON_KEYS}
    for k in _TRIGGER_KEYS:
        out[k] = 0.0
    out["left_axis"] = (0.0, 0.0)
    out["right_axis"] = (0.0, 0.0)
    return out


# Original private name kept for this module's own call sites.
_empty_buttons = empty_buttons


def _poll_xrt_buttons(xrt) -> dict:
    """Snapshot every controller input from xrobotoolkit_sdk in one call.

    All xrt button accessors are wrapped in try/except — older firmware /
    binding versions are missing some methods, and we'd rather lose a button
    than crash the poll thread.
    """
    out = _empty_buttons()
    button_map = {
        "a": "get_A_button",
        "b": "get_B_button",
        "x": "get_X_button",
        "y": "get_Y_button",
        "left_axis_click": "get_left_axis_click",
        "right_axis_click": "get_right_axis_click",
        "left_menu": "get_left_menu_button",
        "right_menu": "get_right_menu_button",
    }
    for key, fn_name in button_map.items():
        fn = getattr(xrt, fn_name, None)
        if fn is None:
            continue
        try:
            out[key] = bool(fn())
        except Exception:
            pass

    for key, fn_name in (
        ("left_trigger", "get_left_trigger"),
        ("right_trigger", "get_right_trigger"),
        ("left_grip", "get_left_grip"),
        ("right_grip", "get_right_grip"),
    ):
        fn = getattr(xrt, fn_name, None)
        if fn is None:
            continue
        try:
            out[key] = float(fn())
        except Exception:
            pass

    for key, fn_name in (
        ("left_axis", "get_left_axis"),
        ("right_axis", "get_right_axis"),
    ):
        fn = getattr(xrt, fn_name, None)
        if fn is None:
            continue
        try:
            axis = fn()
            ax = float(axis[0]) if len(axis) >= 1 else 0.0
            ay = float(axis[1]) if len(axis) >= 2 else 0.0
            out[key] = (ax, ay)
        except Exception:
            pass

    return out


#: Sample keys carrying the direct 6-DoF device poses. Named once so every
#: source and the tests agree; see empty_device_poses for the schema.
DEVICE_POSE_KEYS = (
    "vr_head_pose",
    "vr_left_ctrl_pose",
    "vr_right_ctrl_pose",
    "vr_pose_valid",
    "vr_pose_stamp_ns",
)

_IDENTITY_POSE7 = np.array([0, 0, 0, 1, 0, 0, 0], dtype=np.float32)


def empty_device_poses() -> dict:
    """No-tracking-data device poses: identity, all flagged invalid.

    The safe default for every source that has no headset (mock without a
    gesture, the GMR button relay) and the fallback when polling raises.
    Consumers gate on ``vr_pose_valid``, never on the poses themselves.
    """
    return {
        "vr_head_pose": _IDENTITY_POSE7.copy(),
        "vr_left_ctrl_pose": _IDENTITY_POSE7.copy(),
        "vr_right_ctrl_pose": _IDENTITY_POSE7.copy(),
        "vr_pose_valid": np.zeros(3, dtype=np.uint8),
        "vr_pose_stamp_ns": 0,
    }


# --------------------------------------------------------------------------- #
# The reference gesture
#
# One physical operator pose, described once, used for three things: what the
# operator is told to assume when calibrating, what the mock/keyboard sources
# synthesize, and (via amo.py, which derives its uncalibrated default from
# these constants) the built-in fallback calibration. Keeping them the same
# object is what makes `--pico-source mock` a real regression test of the frame
# chain rather than a plumbing check.
#
# The gesture, which corresponds to the G1's arms at q14 = 0:
#   stand upright, look level, upper arms hanging vertically at your sides,
#   elbows bent 90 deg, forearms horizontal pointing straight forward,
#   PALMS FACING EACH OTHER (thumbs up, as if holding a box between your hands).
#
# Palms facing each other, NOT palms down: measured on g1_body29_hand14.urdf at
# q=0, the two gripper fingers are stacked VERTICALLY and both thumbs (the
# opposing digit, i.e. the palm normal) point toward the midline. Getting this
# wrong bakes a 90 deg error into the calibration, which is the exact class of
# bug this path exists to fix. tests/test_amo_vr_frames.py asserts it against
# the URDF so the instruction cannot drift from the robot.
# --------------------------------------------------------------------------- #

#: Controller position in the head frame for the reference gesture, in metres:
#: hands forward, shoulder-width apart, at roughly belly height below the eyes.
REF_GESTURE_CTRL_POS = {
    "left": np.array([0.20, 0.15, -0.45]),
    "right": np.array([0.20, -0.15, -0.45]),
}

#: Controller ROTATION in the head frame for the reference gesture. This is the
#: prior for how a held controller's grip frame sits relative to the robot's EE
#: frame; it is the transpose of Psi0's hand2inspire_{l,r}_arm (Rx(+/-90 deg),
#: constants_vuer.py:5,7), which is the closest known-good analogue. It is a
#: PRIOR, not a measurement -- Psi0's constants were derived for the OpenXR
#: hand-tracking wrist frame, and a held controller's grip frame is a different
#: frame. Real hardware should overwrite it via the calibration gesture.
REF_GESTURE_CTRL_ROT = {
    "left": sRot.from_euler("x", -90, degrees=True).as_matrix(),
    "right": sRot.from_euler("x", 90, degrees=True).as_matrix(),
}

#: Head pose the synthetic sources report: standing, facing +x, eyes at 1.6 m.
REF_GESTURE_HEAD_POS = np.array([0.0, 0.0, 1.60])


def synthetic_device_poses(left_offset=None, right_offset=None) -> dict:
    """Device poses for an operator holding the reference gesture.

    Used by the mock and keyboard sources so the whole absolute-teleop chain
    runs headless and lands on a known answer (the robot's q14 = 0 pose).

    Args:
        left_offset / right_offset: optional (3,) metres added to the
            controller position in the HEAD frame, so the keyboard source can
            drive the arms around without re-deriving the gesture.
    """
    out = empty_device_poses()
    head = np.eye(4)
    head[:3, 3] = REF_GESTURE_HEAD_POS
    out["vr_head_pose"] = np.concatenate(
        [head[:3, 3], sRot.from_matrix(head[:3, :3]).as_quat(scalar_first=True)]
    ).astype(np.float32)
    for side, key, off in (
        ("left", "vr_left_ctrl_pose", left_offset),
        ("right", "vr_right_ctrl_pose", right_offset),
    ):
        p = REF_GESTURE_CTRL_POS[side].copy()
        if off is not None:
            p = p + np.asarray(off, dtype=float).reshape(3)
        # Head rotation is identity here, so head-frame == robot-frame and the
        # composition is a plain translation.
        pos = head[:3, 3] + p
        quat = sRot.from_matrix(REF_GESTURE_CTRL_ROT[side]).as_quat(scalar_first=True)
        out[key] = np.concatenate([pos, quat]).astype(np.float32)
    out["vr_pose_valid"] = np.ones(3, dtype=np.uint8)
    out["vr_pose_stamp_ns"] = int(time.monotonic_ns())
    return out


def _poll_xrt_device_poses(xrt) -> dict:
    """Snapshot headset + both controller poses, converted to robot axes.

    Deliberately independent of body tracking: these come from the controllers'
    own 6-DoF tracking, which keeps streaming when ``is_body_data_available()``
    goes false (e.g. while a competing PICO connection is being served the body
    frames). Callers must therefore poll this on the stale-body path too, or
    the arm targets freeze exactly when they should not.

    Same defensive style as _poll_xrt_buttons: a missing getter or a raising
    one costs that device its validity flag, never the poll thread.
    """
    out = empty_device_poses()
    valid = np.zeros(3, dtype=np.uint8)
    for i, (key, fn_name) in enumerate(
        (
            ("vr_head_pose", "get_headset_pose"),
            ("vr_left_ctrl_pose", "get_left_controller_pose"),
            ("vr_right_ctrl_pose", "get_right_controller_pose"),
        )
    ):
        fn = getattr(xrt, fn_name, None)
        if fn is None:
            continue
        try:
            raw = np.asarray(fn(), dtype=float).reshape(-1)
            if not pose_is_valid(raw):
                continue
            out[key] = pose7_to_robot(raw)
            valid[i] = 1
        except Exception:
            pass
    out["vr_pose_valid"] = valid
    stamp_fn = getattr(xrt, "get_time_stamp_ns", None)
    if stamp_fn is not None:
        try:
            out["vr_pose_stamp_ns"] = int(stamp_fn())
        except Exception:
            pass
    return out


# =============================================================================
# PicoVRSource — abstract base + Mock + Real (stub)
# =============================================================================


class PicoVRSource(abc.ABC):
    """Abstract PICO VR data source.

    Returns calibrated VR 3-point data in the format SONIC's teleop encoder
    expects:
        vr_3point_local_target:     (9,)  float32 — 3 points × xyz
        vr_3point_local_orn_target: (12,) float32 — 3 points × quat (wxyz)
    """

    @abc.abstractmethod
    def start(self) -> None:
        ...

    @abc.abstractmethod
    def stop(self) -> None:
        ...

    @abc.abstractmethod
    def get_latest(self) -> Optional[dict]:
        """Return latest sample or None if no data yet.

        Returns:
            dict with keys:
              "vr_3point_local_target":     (9,) float32
              "vr_3point_local_orn_target": (12,) float32
              "calibrated":  bool
              "timestamp":   float (monotonic seconds)
              "raw_smpl":    (24, 7) float32  (only set by Real source)
              "buttons":     dict with PICO controller button/axis states
                             — keys: a, b, x, y, left_axis_click,
                                     right_axis_click, left_menu, right_menu,
                                     left_trigger, right_trigger,
                                     left_axis (2,), right_axis (2,).
                             Mock source returns all False/zero.
              "vr_head_pose":       (7,) float32 [xyz, qw,qx,qy,qz], robot axes
              "vr_left_ctrl_pose":  (7,) float32, same layout
              "vr_right_ctrl_pose": (7,) float32, same layout
              "vr_pose_valid":      (3,) uint8 [head, left, right]
              "vr_pose_stamp_ns":   int

        The ``vr_3point_*`` entries and the ``vr_*_pose`` entries are two
        DIFFERENT kinds of data and are not interchangeable. The 3-point set is
        SONIC-encoder input: derived from body tracking and expressed relative
        to whatever pose the operator held at ``ThreePointPose`` calibration, so
        it is not anchored to any robot link frame. The device poses are metric
        6-DoF in robot axes, suitable as IK targets. Use the former for the
        policy, the latter for kinematics.
        """

    @abc.abstractmethod
    def calibrate(self) -> bool:
        """Trigger calibration on next valid frame."""

    def inject_key(self, keycode: int) -> None:
        """Optional: inject a GLFW keycode for keyboard-simulated PICO input.

        Base implementation is a no-op.  Only ``KeyboardPicoVRSource`` handles it.
        This lets the composite controller route unhandled keyboard events to the
        PICO source unconditionally (no ``hasattr`` dance).
        """
        pass


class MockPicoVRSource(PicoVRSource):
    """Mock PICO source returning fixed 3-point VR data — hands at G1 default
    standing pose, neck identity.

    Useful for:
      - Mac development (no real PICO SDK / hardware available)
      - Unit testing the encoder plumbing
      - Smoke-checking that nothing crashes when teleop mode is enabled

    Does NOT produce meaningful teleop motion — robot will just try to hold
    its default standing pose (similar to ``ctrl_type="sonic_planner"``
    in IDLE mode).
    """

    def __init__(self, with_jitter: bool = False, jitter_amplitude: float = 0.0):
        # Default 3-point: hands at sides, neck identity (matches what
        # ThreePointPose would produce post-calibration for a T-pose operator).
        self._default_target = np.concatenate(
            [
                ThreePointPose._G1_NOMINAL_LWRIST_POS,
                ThreePointPose._G1_NOMINAL_RWRIST_POS,
                np.array(
                    [
                        0.0,
                        0.0,
                        ThreePointPose.TORSO_LINK_OFFSET_Z
                        + ThreePointPose.NECK_LINK_LENGTH,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        # Identity quat (wxyz) for all 3 points
        ident_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._default_orn = np.tile(ident_wxyz, 3).astype(np.float32)

        self.with_jitter = with_jitter
        self.jitter_amplitude = jitter_amplitude
        self._t0 = None
        self._rng = np.random.default_rng(seed=0)

    def start(self) -> None:
        self._t0 = time.monotonic()
        print("[MockPicoVRSource] started")

    def stop(self) -> None:
        print("[MockPicoVRSource] stopped")

    def calibrate(self) -> bool:
        # Mock source is always "calibrated" (returns identity-relative data)
        print("[MockPicoVRSource] calibrate() — no-op (mock is always calibrated)")
        return True

    def get_latest(self) -> Optional[dict]:
        if self._t0 is None:
            return None
        target = self._default_target.copy()
        if self.with_jitter and self.jitter_amplitude > 0:
            target += (
                self._rng.standard_normal(9).astype(np.float32) * self.jitter_amplitude
            )
        # Mock POSE data: 24 joints at SMPL T-pose-ish defaults — close enough
        # for plumbing tests.  Real values come from PICO body tracking.
        mock_joints = np.zeros((24, 3), dtype=np.float32)
        mock_anchor = np.array(
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32
        )  # identity 6D
        mock_root_quat = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )  # identity wxyz
        mock_wrist = np.zeros(6, dtype=np.float32)
        return {
            # Reference-gesture device poses: a valid, deterministic operator
            # holding the calibration pose, so --pico-source mock exercises the
            # absolute-teleop frame chain end to end instead of skipping it.
            **synthetic_device_poses(),
            "vr_3point_local_target": target,
            "vr_3point_local_orn_target": self._default_orn.copy(),
            "calibrated": True,
            "timestamp": time.monotonic(),
            # Synthetic sources have no separate body stream; the two clocks
            # coincide so the freshness gate is a no-op for them.
            "body_timestamp": time.monotonic(),
            "buttons": _empty_buttons(),
            "pose_smpl_joints_local": mock_joints,
            "pose_smpl_root_quat": mock_root_quat,
            "pose_anchor_6d": mock_anchor,
            "pose_wrist_joints": mock_wrist,
        }


class KeyboardPicoVRSource(PicoVRSource):
    """PICO source driven entirely by the keyboard — no real PICO hardware.

    Starts directly in VR_3PT mode.  Wrist-position controls::

        left arm                     right arm                PICO buttons / other
        W/S   — front / back         I/K   — front / back     Num 8/2/4/6 — Y/A/X/B
        A/D   — left / right         J/L   — left / right     Z/X   — gripper toggle
        R/F   — up / down            Y/H   — up / down        \\     — reset all
    """

    # PICO buttons (momentary).  Numpad 8/2/4/6 map to Y/A/X/B matching the
    # physical PICO controller button layout (up/down/left/right).
    _KEY_MAP: dict[int, str] = {
        328: "y",  # KP_8  (↑)  → Y (top)
        322: "a",  # KP_2  (↓)  → A (bottom)
        324: "x",  # KP_4  (←)  → X (left)
        326: "b",  # KP_6  (→)  → B (right)
    }
    # PICO analog triggers (persistent toggle).
    _TRIGGER_TOGGLE_MAP: dict[int, str] = {
        ord("Z"): "left_trigger",
        ord("X"): "right_trigger",
    }

    _ARM_STEP = 0.03  # metres per keypress

    # (GLFW keycode → (is_right_wrist, axis_index, delta))
    # Left arm:  W/S=front/back(Z)  A/D=left/right(X)  R/F=up/down(Y)
    # Right arm: I/K=front/back(Z)  J/L=left/right(X)  Y/H=up/down(Y)
    _ARM_MAP: dict[int, tuple[bool, int, float]] = {
        # --- left wrist (WASD) ---
        ord("W"): (False, 2, +_ARM_STEP),  # Z forward
        ord("S"): (False, 2, -_ARM_STEP),  # Z back
        ord("A"): (False, 0, -_ARM_STEP),  # X left
        ord("D"): (False, 0, +_ARM_STEP),  # X right
        ord("R"): (False, 1, +_ARM_STEP),  # Y up
        ord("F"): (False, 1, -_ARM_STEP),  # Y down
        # --- right wrist (IJKL) ---
        ord("I"): (True, 2, +_ARM_STEP),  # Z forward
        ord("K"): (True, 2, -_ARM_STEP),  # Z back
        ord("J"): (True, 0, -_ARM_STEP),  # X left
        ord("L"): (True, 0, +_ARM_STEP),  # X right
        ord("Y"): (True, 1, +_ARM_STEP),  # Y up
        ord("H"): (True, 1, -_ARM_STEP),  # Y down
    }
    _ARM_RESET_KEY = ord("\\")

    def __init__(self):
        self._default_target = np.concatenate(
            [
                ThreePointPose._G1_NOMINAL_LWRIST_POS,
                ThreePointPose._G1_NOMINAL_RWRIST_POS,
                np.array(
                    [
                        0.0,
                        0.0,
                        ThreePointPose.TORSO_LINK_OFFSET_Z
                        + ThreePointPose.NECK_LINK_LENGTH,
                    ],
                    dtype=np.float32,
                ),
            ]
        ).astype(np.float32)
        ident_wxyz = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._default_orn = np.tile(ident_wxyz, 3).astype(np.float32)

        self._pending_buttons: set[str] = set()
        self._trigger_state: dict[str, float] = {
            "left_trigger": 0.0,
            "right_trigger": 0.0,
        }
        self._toggle_cooldown: dict[int, float] = {}  # keycode → last toggle time
        self._lwrist_offset = np.zeros(3, dtype=np.float32)
        self._rwrist_offset = np.zeros(3, dtype=np.float32)
        self._mock_joints = np.zeros((24, 3), dtype=np.float32)
        self._mock_anchor = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
        self._mock_root_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._mock_wrist = np.zeros(6, dtype=np.float32)
        self._t0: float | None = None

    def start(self) -> None:
        self._t0 = time.monotonic()
        print(
            "[KeyboardPicoVRSource] started (VR_3PT)\n"
            "  left arm:  W/S=front/back  A/D=left/right  R/F=up/down\n"
            "  right arm: I/K=front/back  J/L=left/right  Y/H=up/down\n"
            "  buttons:   Num8/2/4/6=Y/A/X/B  Z/X=grip  \\=reset"
        )

    def stop(self) -> None:
        print("[KeyboardPicoVRSource] stopped")

    def calibrate(self) -> bool:
        return True

    _TOGGLE_COOLDOWN = 0.4  # seconds — suppress GLFW key-repeat flapping

    def inject_key(self, keycode: int) -> None:
        if keycode in self._KEY_MAP:
            self._pending_buttons.add(self._KEY_MAP[keycode])
        elif keycode in self._TRIGGER_TOGGLE_MAP:
            # Toggle gripper state.  Cooldown suppresses GLFW key-repeat flapping
            # (~30 Hz); a 0.15 s window absorbs ~5 repeats after the initial press.
            now = time.monotonic()
            last = self._toggle_cooldown.get(keycode, -999.0)
            if now - last > 0.15:
                self._toggle_cooldown[keycode] = now
                field = self._TRIGGER_TOGGLE_MAP[keycode]
                new_val = 1.0 if self._trigger_state[field] < 0.5 else 0.0
                self._trigger_state[field] = new_val
                print(f"[grip] key={chr(keycode)} {field} -> {new_val:.0f}")
        elif keycode in self._ARM_MAP:
            is_right, axis, delta = self._ARM_MAP[keycode]
            off = self._rwrist_offset if is_right else self._lwrist_offset
            off[axis] += delta
            side = "R" if is_right else "L"
            print(f"[arm] {side} wrist: {off}")
        elif keycode == self._ARM_RESET_KEY:
            self._lwrist_offset[:] = 0.0
            self._rwrist_offset[:] = 0.0
            print("[arm] wrists reset")

    def get_latest(self) -> Optional[dict]:
        if self._t0 is None:
            return None
        buttons = _empty_buttons()
        for key in self._pending_buttons:
            buttons[key] = True
        self._pending_buttons.clear()
        for field, val in self._trigger_state.items():
            buttons[field] = val
        target = self._default_target.copy()
        target[0:3] += self._lwrist_offset
        target[3:6] += self._rwrist_offset
        return {
            # Same WASD/IJKL offsets, re-expressed in head-frame robot axes, so
            # the keys drive the arms in absolute mode too. _ARM_MAP's axes are
            # (0 = lateral, 1 = up, 2 = forward) with +0 meaning the operator's
            # right; robot head-frame axes are (x fwd, y left, z up).
            **synthetic_device_poses(
                left_offset=self._head_frame_offset(self._lwrist_offset),
                right_offset=self._head_frame_offset(self._rwrist_offset),
            ),
            "vr_3point_local_target": target,
            "vr_3point_local_orn_target": self._default_orn.copy(),
            "calibrated": True,
            "timestamp": time.monotonic(),
            # Synthetic sources have no separate body stream; the two clocks
            # coincide so the freshness gate is a no-op for them.
            "body_timestamp": time.monotonic(),
            "buttons": buttons,
            "pose_smpl_joints_local": self._mock_joints.copy(),
            "pose_smpl_root_quat": self._mock_root_quat.copy(),
            "pose_anchor_6d": self._mock_anchor.copy(),
            "pose_wrist_joints": self._mock_wrist.copy(),
        }

    @staticmethod
    def _head_frame_offset(off) -> np.ndarray:
        off = np.asarray(off, dtype=float).reshape(3)
        return np.array([off[2], -off[0], off[1]])


def _is_proc_running(proc_name: str) -> bool:
    if shutil.which("pgrep") is None:
        return False
    try:
        subprocess.check_output(["pgrep", "-f", proc_name])
        return True
    except subprocess.CalledProcessError:
        return False


def _maybe_launch_xrt_service(service_script: str, log_prefix: str, verbose: bool):
    if not os.path.exists(service_script):
        print(
            f"[{log_prefix}] WARNING: service script not found at "
            f"{service_script}; skipping auto-launch. Make sure the daemon is "
            "started manually.",
            flush=True,
        )
        return None
    script_name = os.path.basename(service_script)
    for proc_name in (script_name, "RoboticsServiceProcess"):
        if _is_proc_running(proc_name):
            if verbose:
                print(
                    f"[{log_prefix}] {proc_name} already running — not relaunching",
                    flush=True,
                )
            return None

    if verbose:
        print(f"[{log_prefix}] launching {service_script}", flush=True)
    return subprocess.Popen(
        ["bash", service_script],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _queue_put_latest(q, item) -> None:
    while True:
        try:
            q.put_nowait(item)
            return
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                return


def _xrt_worker_main(
    data_queue,
    status_queue,
    command_queue,
    stop_event,
    service_script,
    init_timeout_s,
    verbose,
    enable_pose_data,
):
    log_prefix = "RealPicoVRWorker"
    service_proc = None
    xrt = None
    three_point = ThreePointPose(log_prefix=log_prefix, verbose=verbose)
    three_point.reset()
    last_stamp_ns = None
    last_sample = None  # most recent full sample, for buttons-only refreshes
    fps_ema = 0.0
    sample_count = 0
    last_report = time.monotonic()
    try:
        try:
            import xrobotoolkit_sdk as xrt_mod
        except ImportError as e:
            status_queue.put(("error", f"xrobotoolkit_sdk not importable: {e}"))
            return
        xrt = xrt_mod

        if service_script is not None:
            service_proc = _maybe_launch_xrt_service(
                service_script, log_prefix, verbose
            )

        time.sleep(0.5)
        if verbose:
            print(f"[{log_prefix}] calling xrt.init()...", flush=True)
        xrt.init()

        deadline = time.monotonic() + init_timeout_s
        if verbose:
            print(f"[{log_prefix}] waiting for body data...", flush=True)
        while time.monotonic() < deadline and not stop_event.is_set():
            if xrt.is_body_data_available():
                status_queue.put(("ready", None))
                if verbose:
                    print(f"[{log_prefix}] body data available — ready", flush=True)
                break
            time.sleep(0.25)
        else:
            status_queue.put(
                (
                    "error",
                    f"Timed out after {init_timeout_s}s waiting for "
                    "xrt.is_body_data_available().",
                )
            )
            return

        while not stop_event.is_set():
            try:
                while True:
                    try:
                        cmd = command_queue.get_nowait()
                    except queue.Empty:
                        break
                    if cmd == "calibrate":
                        three_point.reset()
                    elif cmd == "stop":
                        stop_event.set()
                        break

                # Buttons/triggers are logically independent of body
                # tracking -- poll them every tick regardless of whether a
                # fresh body-pose frame is available on THIS connection, and
                # push a refreshed sample (reusing the last known body pose)
                # so hand-trigger/grip control doesn't freeze in lockstep
                # with body-tracking outages here -- e.g. while a competing
                # PICO connection (such as the GMR full-body retargeting
                # subprocess) is the one currently being served fresh body
                # frames by the underlying XRoboToolkit service.
                buttons = _poll_xrt_buttons(xrt)
                # Controller/headset 6-DoF, same reasoning as buttons: it is
                # independent of body tracking, so it must be refreshed on the
                # stale-body path below or absolute arm teleop freezes whenever
                # a competing connection owns the body stream.
                device_poses = _poll_xrt_device_poses(xrt)

                body_available = xrt.is_body_data_available()
                stamp_ns = xrt.get_time_stamp_ns() if body_available else None
                has_new_body_frame = body_available and (
                    last_stamp_ns is None or stamp_ns != last_stamp_ns
                )
                if not has_new_body_frame:
                    if last_sample is not None:
                        refreshed = dict(last_sample)
                        refreshed["buttons"] = buttons
                        refreshed.update(device_poses)
                        refreshed["timestamp"] = time.monotonic()
                        _queue_put_latest(data_queue, refreshed)
                        last_sample = refreshed
                    time.sleep(0.001 if not body_available else 0.0005)
                    continue

                if last_stamp_ns is not None:
                    dt = (stamp_ns - last_stamp_ns) * 1e-9
                    if dt > 0:
                        inst = 1.0 / dt
                        fps_ema = inst if fps_ema == 0.0 else 0.9 * fps_ema + 0.1 * inst
                last_stamp_ns = stamp_ns

                body_poses_np = np.array(xrt.get_body_joints_pose())
                calibrated = three_point.process_smpl_pose(body_poses_np)

                pose_data = None
                if enable_pose_data:
                    try:
                        pose_data = process_smpl_pose_full(body_poses_np)
                    except Exception as e:
                        if verbose:
                            print(
                                f"[{log_prefix}] process_smpl_pose_full failed: {e}",
                                flush=True,
                            )

                sample = {
                    "vr_3point_local_target": calibrated[:, :3]
                    .flatten()
                    .astype(np.float32),
                    "vr_3point_local_orn_target": calibrated[:, 3:]
                    .flatten()
                    .astype(np.float32),
                    "calibrated": three_point.is_calibrated,
                    "timestamp": time.monotonic(),
                    # When this BODY frame arrived, as opposed to when the
                    # sample dict was last refreshed. The stale-body path below
                    # re-stamps ``timestamp`` every loop so buttons and device
                    # poses keep flowing during a body outage; carrying the
                    # body clock separately is what lets a consumer tell that
                    # the SMPL in this sample is old. See
                    # backends.sonic._smpl_is_fresh.
                    "body_timestamp": time.monotonic(),
                    "raw_smpl": body_poses_np.astype(np.float32),
                    "buttons": buttons,
                    "pose_smpl_joints_local": (
                        pose_data["smpl_joints_local"] if pose_data else None
                    ),
                    "pose_smpl_root_quat": (
                        pose_data["smpl_root_quat"] if pose_data else None
                    ),
                    "pose_anchor_6d": pose_data["anchor_6d"] if pose_data else None,
                    "pose_wrist_joints": (
                        pose_data["wrist_joints"] if pose_data else None
                    ),
                    **device_poses,
                }
                last_sample = sample
                _queue_put_latest(data_queue, sample)
                sample_count += 1

                if verbose and time.monotonic() - last_report >= 5.0:
                    target = sample["vr_3point_local_target"].reshape(3, 3)
                    print(
                        f"[{log_prefix}] {sample_count} samples in 5s | "
                        f"fps={fps_ema:.1f} | "
                        f"L-wrist={target[0].round(3).tolist()} | "
                        f"R-wrist={target[1].round(3).tolist()} | "
                        f"calibrated={three_point.is_calibrated}",
                        flush=True,
                    )
                    last_report = time.monotonic()
                    sample_count = 0
            except Exception as e:
                print(f"[{log_prefix}] poll error: {e}", flush=True)
                time.sleep(0.05)
    except BaseException as e:
        try:
            status_queue.put(("error", f"{type(e).__name__}: {e}"))
        except Exception:
            pass
        raise
    finally:
        if xrt is not None:
            try:
                xrt.close()
            except Exception as e:
                print(f"[{log_prefix}] xrt.close() failed: {e}", flush=True)
        if service_proc is not None and service_proc.poll() is None:
            service_proc.terminate()
            try:
                service_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                service_proc.kill()


class RealPicoVRSource(PicoVRSource):
    """Legacy in-process PICO VR source — wraps ``xrobotoolkit_sdk`` directly.

    This path is kept for debugging with ``PICO_DIRECT_XRT=1``.  Normal data
    collection should use :class:`ProcessPicoVRSource` via ``pico_source='real'``.

    Pipeline (mirrors ``gear_sonic/scripts/pico_manager_thread_server.py``):
      1. Spawn ``/opt/apps/roboticsservice/runService.sh`` subprocess if it
         isn't already running.
      2. ``xrt.init()`` + block until body data is available.
      3. Background thread polls ``xrt.get_body_joints_pose()`` whenever the
         device timestamp advances; pipes through
         :func:`process_3pt_pose` + :meth:`ThreePointPose.process_smpl_pose`.
      4. :meth:`get_latest` returns the most recent calibrated sample.
      5. :meth:`calibrate` triggers :meth:`ThreePointPose.reset` — the next
         valid frame becomes the new zero-reference.

    Args:
        service_script: Path to ``runService.sh``.  Default
            ``/opt/apps/roboticsservice/runService.sh``; pass ``None`` to skip
            auto-launch (assume the service is already running).
        init_timeout_s: Seconds to wait for ``xrt.is_body_data_available()``
            after ``init()`` before giving up.
        verbose: If True, print debug info (FPS, raw shapes) every 5 s.
    """

    _DEFAULT_SERVICE_SCRIPT = "/opt/apps/roboticsservice/runService.sh"

    def __init__(
        self,
        service_script: Optional[str] = _DEFAULT_SERVICE_SCRIPT,
        init_timeout_s: float = 15.0,
        verbose: bool = True,
    ):
        try:
            import xrobotoolkit_sdk as xrt
        except ImportError as e:
            raise ImportError(
                "xrobotoolkit_sdk not importable — RealPicoVRSource only works "
                "on Linux with XRoboToolkit PC Service installed.  Run "
                "scripts/setup_xrobotoolkit_ubuntu.sh on Ubuntu first.  "
                f"Underlying error: {e}"
            ) from e

        self._xrt = xrt
        self._verbose = verbose
        self._service_proc: Optional[subprocess.Popen] = None
        self._three_point = ThreePointPose(
            log_prefix="RealPicoVRSource", verbose=verbose
        )
        self._three_point.reset()  # next frame triggers initial calibration

        self._latest: Optional[dict] = None
        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._last_stamp_ns: Optional[int] = None
        self._fps_ema: float = 0.0
        self._closed = False
        self._enable_pose_data = os.environ.get(
            "PICO_ENABLE_BACKGROUND_POSE_DATA", "0"
        ) in ("1", "true", "True",) and os.environ.get(
            "PICO_DISABLE_POSE_DATA", "0"
        ) not in (
            "1",
            "true",
            "True",
        )
        self._sync_poll = os.environ.get("PICO_ASYNC_POLL", "0") not in (
            "1",
            "true",
            "True",
        )
        # The xrt SDK spins its own C++ threads ("watch server feedback");
        # if the process exits without calling xrt.close() they're destroyed
        # while joinable -> "terminate called without an active exception" +
        # core dump.  Register an atexit fallback so even paths that never call
        # stop() (e.g. the headless --check, or a Ctrl-C before the finally)
        # shut the SDK down cleanly.
        atexit.register(self._close_sdk)

        # 1. Start the PC Service if asked
        if service_script is not None:
            self._maybe_launch_service(service_script)

        # 2. xrt.init() — process-global, guarded against double-init.
        global _XRT_INITED
        time.sleep(0.5)
        if not _XRT_INITED:
            print("[RealPicoVRSource] calling xrt.init()...")
            xrt.init()
            _XRT_INITED = True
        else:
            print(
                "[RealPicoVRSource] xrt already initialised this process — " "reusing"
            )

        # 3. Wait for body data
        deadline = time.monotonic() + init_timeout_s
        print("[RealPicoVRSource] waiting for body data...")
        while time.monotonic() < deadline:
            if xrt.is_body_data_available():
                print("[RealPicoVRSource] body data available — ready")
                return
            time.sleep(0.25)
        # Constructor failure means stop() will not be called by the composite
        # controller. Close the native SDK here so its C++ threads are not left
        # alive while the caller handles the startup error.
        self._close_sdk()
        raise RuntimeError(
            f"Timed out after {init_timeout_s}s waiting for "
            "xrt.is_body_data_available().  Check that the PICO headset is "
            "connected, Body Tracking is enabled in the XRoboToolkit app on "
            "the headset, and runService.sh is running."
        )

    # ------------------------------------------------------------------ #
    def _maybe_launch_service(self, service_script: str) -> None:
        if not os.path.exists(service_script):
            print(
                f"[RealPicoVRSource] WARNING: service script not found at "
                f"{service_script}; skipping auto-launch.  Make sure the "
                f"daemon is started manually."
            )
            return
        # Avoid double-launching: if either the wrapper script or the binary it
        # shells into is already running, skip.
        script_name = os.path.basename(service_script)
        if shutil.which("pgrep") is not None:
            for proc_name in (script_name, "RoboticsServiceProcess"):
                try:
                    subprocess.check_output(["pgrep", "-f", proc_name])
                    print(
                        f"[RealPicoVRSource] {proc_name} already running — "
                        "not relaunching"
                    )
                    return
                except subprocess.CalledProcessError:
                    pass  # not running, proceed

        print(f"[RealPicoVRSource] launching {service_script}")
        self._service_proc = subprocess.Popen(
            ["bash", service_script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._closed:
            raise RuntimeError("RealPicoVRSource has already been stopped/closed")
        if self._sync_poll:
            print("[RealPicoVRSource] sync polling enabled — no Python poll thread")
            return
        if not self._thread.is_alive():
            self._thread.start()
            print("[RealPicoVRSource] poll thread started")

    def stop(self) -> None:
        self._stop_poll_thread()
        # Shut the SDK down before tearing down the service so its threads stop
        # touching the (about-to-die) connection.
        self._close_sdk()
        if self._service_proc is not None and self._service_proc.poll() is None:
            self._service_proc.terminate()
            try:
                self._service_proc.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._service_proc.kill()
        print("[RealPicoVRSource] stopped")

    def _stop_poll_thread(self) -> None:
        self._stop_evt.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def _close_sdk(self) -> None:
        """Call ``xrt.close()`` exactly once to join the SDK's C++ threads and
        avoid an abort at process exit.  Safe to call from both stop() and the
        atexit fallback (guarded by ``self._closed``)."""
        global _XRT_INITED, _REAL_SOURCE_SINGLETON
        if self._closed:
            return
        self._closed = True
        self._stop_poll_thread()
        try:
            self._xrt.close()
        except Exception as e:
            print(f"[RealPicoVRSource] xrt.close() failed: {e}")
        finally:
            _XRT_INITED = False
            if _REAL_SOURCE_SINGLETON is self:
                _REAL_SOURCE_SINGLETON = None

    def get_latest(self) -> Optional[dict]:
        if self._sync_poll and not self._closed:
            self._poll_once()
        with self._lock:
            return None if self._latest is None else dict(self._latest)

    def calibrate(self) -> bool:
        self._three_point.reset()
        print("[RealPicoVRSource] calibrate(): next frame will re-zero")
        return True

    # ------------------------------------------------------------------ #
    def _poll_once(self) -> bool:
        # Buttons/triggers are logically independent of body tracking --
        # refresh them every tick regardless of whether a fresh body-pose
        # frame is available.  Previously this method early-returned
        # (leaving ``self._latest`` -- and its buttons -- completely
        # untouched) whenever body data was stale, which silently froze
        # hand-trigger/grip control in lockstep with body-tracking outages on
        # THIS connection -- e.g. while a competing PICO connection (such as
        # the GMR full-body retargeting subprocess) is the one currently
        # being served fresh body frames by the underlying XRoboToolkit
        # service.
        buttons = _poll_xrt_buttons(self._xrt)
        # Same argument for the device 6-DoF poses -- they come from controller
        # tracking, not body tracking, so they must survive a body outage or
        # absolute arm teleop freezes with them.
        device_poses = _poll_xrt_device_poses(self._xrt)
        with self._lock:
            if self._latest is not None:
                self._latest["buttons"] = buttons
                self._latest.update(device_poses)

        if not self._xrt.is_body_data_available():
            return False
        stamp_ns = self._xrt.get_time_stamp_ns()
        if self._last_stamp_ns is not None and stamp_ns == self._last_stamp_ns:
            return False

        # Update EMA fps for debug
        if self._last_stamp_ns is not None:
            dt = (stamp_ns - self._last_stamp_ns) * 1e-9
            if dt > 0:
                inst = 1.0 / dt
                self._fps_ema = (
                    inst if self._fps_ema == 0.0 else 0.9 * self._fps_ema + 0.1 * inst
                )
        self._last_stamp_ns = stamp_ns

        body_poses_np = np.array(self._xrt.get_body_joints_pose())
        calibrated = self._three_point.process_smpl_pose(body_poses_np)
        # Full-body pipeline for POSE mode (encoder mode_id=2). Wrap in
        # try/except so any retargeting glitch doesn't kill VR_3PT data.
        pose_data = None
        if self._enable_pose_data:
            try:
                pose_data = process_smpl_pose_full(body_poses_np)
            except Exception as e:
                if self._verbose:
                    print(f"[RealPicoVRSource] process_smpl_pose_full failed: {e}")

        sample = {
            "vr_3point_local_target": calibrated[:, :3].flatten().astype(np.float32),
            "vr_3point_local_orn_target": calibrated[:, 3:]
            .flatten()
            .astype(np.float32),
            "calibrated": self._three_point.is_calibrated,
            "timestamp": time.monotonic(),
            #: Body-frame clock -- see the isolated worker's copy above.
            "body_timestamp": time.monotonic(),
            "raw_smpl": body_poses_np.astype(np.float32),
            "buttons": buttons,
            "pose_smpl_joints_local": (
                pose_data["smpl_joints_local"] if pose_data else None
            ),
            "pose_smpl_root_quat": pose_data["smpl_root_quat"] if pose_data else None,
            "pose_anchor_6d": pose_data["anchor_6d"] if pose_data else None,
            "pose_wrist_joints": pose_data["wrist_joints"] if pose_data else None,
            **device_poses,
        }
        with self._lock:
            self._latest = sample
        return True

    def _poll_loop(self) -> None:
        last_report = time.monotonic()
        sample_count = 0
        while not self._stop_evt.is_set():
            try:
                if not self._poll_once():
                    time.sleep(0.0005)
                    continue
                sample_count += 1

                if self._verbose and time.monotonic() - last_report >= 5.0:
                    latest = self.get_latest()
                    target = (
                        latest["vr_3point_local_target"].reshape(3, 3)
                        if latest is not None
                        else np.zeros((3, 3), dtype=np.float32)
                    )
                    print(
                        f"[RealPicoVRSource] {sample_count} samples in 5s | "
                        f"fps={self._fps_ema:.1f} | "
                        f"L-wrist={target[0].round(3).tolist()} | "
                        f"R-wrist={target[1].round(3).tolist()} | "
                        f"calibrated={self._three_point.is_calibrated}"
                    )
                    last_report = time.monotonic()
                    sample_count = 0
            except Exception as e:
                print(f"[RealPicoVRSource] poll error: {e}")
                time.sleep(0.05)


class ProcessPicoVRSource(PicoVRSource):
    """Real PICO source with the XRoboToolkit SDK isolated in a child process.

    This is the default for ``pico_source='real'`` because the SDK starts native
    C++ streaming threads. Keeping those threads out of the MuJoCo/ONNX/viewer
    process prevents SDK aborts or segfaults from taking down data collection.
    """

    _DEFAULT_SERVICE_SCRIPT = RealPicoVRSource._DEFAULT_SERVICE_SCRIPT

    def __init__(
        self,
        service_script: Optional[str] = _DEFAULT_SERVICE_SCRIPT,
        init_timeout_s: float = 15.0,
        verbose: bool = False,
    ):
        self._verbose = verbose
        self._latest: Optional[dict] = None
        self._closed = False
        self._ctx = mp.get_context("spawn")
        self._data_queue = self._ctx.Queue(maxsize=8)
        self._status_queue = self._ctx.Queue(maxsize=4)
        self._command_queue = self._ctx.Queue(maxsize=8)
        self._stop_event = self._ctx.Event()
        enable_pose_data = os.environ.get("PICO_ENABLE_BACKGROUND_POSE_DATA", "0") in (
            "1",
            "true",
            "True",
        ) and os.environ.get("PICO_DISABLE_POSE_DATA", "0") not in (
            "1",
            "true",
            "True",
        )
        self._proc = self._ctx.Process(
            target=_xrt_worker_main,
            name="RealPicoVRWorker",
            args=(
                self._data_queue,
                self._status_queue,
                self._command_queue,
                self._stop_event,
                service_script,
                init_timeout_s,
                verbose,
                enable_pose_data,
            ),
            daemon=True,
        )
        if self._verbose:
            print(
                "[RealPicoVRSource] launching isolated XRT worker process", flush=True
            )
        self._proc.start()
        atexit.register(self.stop)
        self._wait_ready(init_timeout_s + 5.0)

    def _wait_ready(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if not self._proc.is_alive():
                code = self._proc.exitcode
                raise RuntimeError(f"isolated XRT worker exited early (code={code})")
            try:
                status, payload = self._status_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if status == "ready":
                if self._verbose:
                    print("[RealPicoVRSource] isolated XRT worker ready", flush=True)
                return
            if status == "error":
                self.stop()
                raise RuntimeError(payload)
        self.stop()
        raise RuntimeError(
            f"Timed out after {timeout_s:.1f}s waiting for isolated XRT worker."
        )

    def start(self) -> None:
        if self._closed:
            raise RuntimeError("ProcessPicoVRSource has already been stopped/closed")
        if self._verbose:
            print("[RealPicoVRSource] using isolated XRT worker process", flush=True)

    def stop(self) -> None:
        global _PROCESS_SOURCE_SINGLETON
        if self._closed:
            return
        self._closed = True
        try:
            self._command_queue.put_nowait("stop")
        except Exception:
            pass
        self._stop_event.set()
        if self._proc.is_alive():
            self._proc.join(timeout=3.0)
        if self._proc.is_alive():
            self._proc.terminate()
            self._proc.join(timeout=1.0)
        if self._proc.is_alive():
            self._proc.kill()
            self._proc.join(timeout=1.0)
        if _PROCESS_SOURCE_SINGLETON is self:
            _PROCESS_SOURCE_SINGLETON = None
        if self._proc.is_alive():
            print(
                "[RealPicoVRSource] WARNING: worker process still alive after SIGKILL",
                flush=True,
            )
        elif self._verbose:
            print("[RealPicoVRSource] isolated worker stopped", flush=True)

    def get_latest(self) -> Optional[dict]:
        if self._closed:
            return None if self._latest is None else dict(self._latest)
        while True:
            try:
                self._latest = self._data_queue.get_nowait()
            except queue.Empty:
                break
        if self._latest is None and not self._proc.is_alive():
            raise RuntimeError(
                f"isolated XRT worker died unexpectedly (code={self._proc.exitcode})"
            )
        return None if self._latest is None else dict(self._latest)

    def calibrate(self) -> bool:
        if self._closed:
            return False
        try:
            self._command_queue.put_nowait("calibrate")
        except queue.Full:
            return False
        print("[RealPicoVRSource] calibrate(): worker will re-zero")
        return True


# =============================================================================
# Factory
# =============================================================================

# Process-global real PICO sources.  ``xrobotoolkit_sdk.init()`` is a native
# process-level singleton, so direct mode still needs a singleton.  The default
# isolated-process mode also uses one worker per parent process.
_REAL_SOURCE_SINGLETON: Optional["RealPicoVRSource"] = None
_PROCESS_SOURCE_SINGLETON: Optional["ProcessPicoVRSource"] = None

# Set once xrt.init() has run in this process (see RealPicoVRSource.__init__);
# guards against a second init() aborting the process.
_XRT_INITED: bool = False


def make_pico_source(source_type: str, **kwargs) -> PicoVRSource:
    """Construct a PICO VR source by name.

    Args:
        source_type: ``"mock"`` (Mac/Linux), ``"keyboard"`` (any platform,
                     simulated PICO buttons via keyboard), or ``"real"``
                     (Linux + xrobotoolkit_sdk only).
        **kwargs:    passed through to the chosen constructor.

    The ``"real"`` source defaults to an isolated worker process. Set
    ``PICO_DIRECT_XRT=1`` to use the legacy in-process SDK path for debugging.
    """
    global _REAL_SOURCE_SINGLETON, _PROCESS_SOURCE_SINGLETON
    if source_type == "mock":
        return MockPicoVRSource(**kwargs)
    if source_type == "keyboard":
        return KeyboardPicoVRSource()
    if source_type == "real":
        direct = os.environ.get("PICO_DIRECT_XRT", "0") in ("1", "true", "True")
        if direct:
            if _REAL_SOURCE_SINGLETON is None:
                _REAL_SOURCE_SINGLETON = RealPicoVRSource(**kwargs)
            else:
                print(
                    "[RealPicoVRSource] reusing existing process-global source "
                    "(skipping a second xrt.init() — e.g. across env.reset())"
                )
            return _REAL_SOURCE_SINGLETON
        if _PROCESS_SOURCE_SINGLETON is None:
            _PROCESS_SOURCE_SINGLETON = ProcessPicoVRSource(**kwargs)
        else:
            if getattr(_PROCESS_SOURCE_SINGLETON, "_verbose", False):
                print("[RealPicoVRSource] reusing isolated XRT worker process")
        return _PROCESS_SOURCE_SINGLETON
    raise ValueError(f"Unknown pico_source: {source_type!r}")
