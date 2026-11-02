"""Flavor A (SONIC-via-sim2real, POSE mode) reference-motion payload builder.

Converts tele's own PICO SMPL sample straight into the payload dict
``RealtimeSmplMotionBuffer.push_payload`` expects -- no GMR retargeting
involved. This is the genuinely GMR-free path: sim2real's ``smpl/policy.yaml``
SONIC checkpoint reads only ``smpl_body_pose_aa``/``smpl_joint_pos_root``/
``smpl_root_quat_w`` (all three built by sim2real's own
``build_smpl_frame_from_xrobot_raw`` from the raw (24,7) PICO array) plus a
handful of wrist joint angles spliced into a 29-DOF ``joint_pos`` vector
(everything else in that vector is unused by the smpl-mode observations, per
sim2real/rl_policy/observations/sonic.py, but the buffer's shape validation
still requires the full vector).

``raw_smpl`` here is exactly the ``(24, 7)`` ``[x, y, z, qx, qy, qz, qw]``
array tele_humanoid_benchmark's ``pico_vr_source.PicoVRSource.get_latest()``
already exposes as ``"raw_smpl"`` -- verified byte-identical in layout to
what sim2real's own ``XRobotStreamer``-based pipeline consumes.
"""

from __future__ import annotations

import numpy as np

from sim2real.config.robots.g1 import G1_CFG
from sim2real.teleop.smpl_stream import build_smpl_frame_from_xrobot_raw

# [L_roll, R_roll, L_pitch, R_pitch, L_yaw, R_yaw] -- matches the order
# tele_humanoid_benchmark's pico_vr_source.process_smpl_pose_full() returns
# as "wrist_joints" (IsaacLab wrist-joint convention).
_WRIST_JOINT_NAMES_IN_ORDER = (
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
)
_WRIST_INDICES_IN_G1_CFG_ORDER = tuple(
    G1_CFG.joint_names.index(name) for name in _WRIST_JOINT_NAMES_IN_ORDER
)


def build_payload(
    raw_smpl: np.ndarray,
    wrist_joints: np.ndarray,
    *,
    human_joints_info_path: str,
    waist_yaw_offset_deg: float = 0.0,
) -> dict:
    """Build one ``RealtimeSmplMotionBuffer.push_payload`` frame.

    Args:
        raw_smpl: ``(24, 7)`` raw PICO SMPL body-tracking sample
            (``PicoVRSource.get_latest()["raw_smpl"]``).
        wrist_joints: ``(6,)`` ``[L_roll, R_roll, L_pitch, R_pitch, L_yaw,
            R_yaw]`` wrist joint angles (``pico_vr_source.
            process_smpl_pose_full(raw_smpl)["wrist_joints"]``, or the
            equivalent precomputed ``pose_wrist_joints`` field a real PICO
            source already exposes in the background).
        human_joints_info_path: absolute path to the ``human_joints_info.pkl``
            shipped alongside the loaded SONIC ``smpl/policy.yaml`` checkpoint.

    Returns:
        dict with ``smpl_body_pose_aa``, ``smpl_joint_pos_root``,
        ``smpl_root_quat_w``, ``joint_pos`` -- the full set
        ``RealtimeSmplMotionBuffer`` requires.
    """
    frame = build_smpl_frame_from_xrobot_raw(
        np.asarray(raw_smpl, dtype=np.float32),
        joint_names=(),  # only consulted when waist_yaw_offset_deg != 0
        waist_yaw_offset_deg=waist_yaw_offset_deg,
        human_joints_info_path=human_joints_info_path,
    )

    joint_pos = np.zeros(len(G1_CFG.joint_names), dtype=np.float32)
    wrist_joints = np.asarray(wrist_joints, dtype=np.float32).reshape(-1)
    joint_pos[list(_WRIST_INDICES_IN_G1_CFG_ORDER)] = wrist_joints

    frame["joint_pos"] = joint_pos
    return frame
