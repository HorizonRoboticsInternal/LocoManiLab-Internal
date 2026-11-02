"""Forward kinematics of the G1 waist + arm chains, numpy/scipy only.

Exists for ONE column: the SONIC schema's ``observation.eef_state`` (14) --
``[left_pos(3), left_quat_wxyz(4), right_pos(3), right_quat_wxyz(4)]`` of the
``{left,right}_wrist_yaw_link`` frames, expressed in the **pelvis** frame. That
is exactly how NVlabs' own collector defines the column
(``gear_sonic/scripts/run_data_exporter.py``: pinocchio FK of the 43-entry
configuration with the floating base left at identity, ``hand_frame_names`` =
the wrist-yaw links, quaternion ``scalar_first``), and the definition is a pure
function of ``observation.state`` -- so both hosts compute it the same way, a
hardware run gets the column too, and a dataset repair can recompute it from
the state it already carries.

Why not pinocchio: this package is what a robot host imports, and the recording
layer is deliberately dependency-free (see ``schema.py``). The chain is ten
single-DoF revolute joints per side; parsing the URDF with ``xml.etree`` and
composing the transforms with ``scipy.spatial.transform.Rotation`` is ~100
lines and vectorises over frames, which the offline repair needs (hundreds of
thousands of rows).

Verified against pinocchio (``RobotModel.frame_placement`` in
GR00T-WholeBodyControl) on random configurations inside that model's joint
limits: position and orientation agree to ~1e-15 -- ``tests/test_sonic_schema_v6.py``
pins golden values from that run. One deliberate difference: the upstream
collector CLIPS the configuration to its teleop joint limits before FK
(``RobotModel.cache_forward_kinematics(auto_clip=True)``; e.g. shoulder roll is
held >= 0.19 rad), this function does not -- it reports where the measured
joints actually put the wrist.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET
from functools import lru_cache

import numpy as np
from scipy.spatial.transform import Rotation as sRot

#: The URDF the chains are read from: Unitree's ``g1_29dof_with_hand.urdf`` as
#: shipped in NVlabs/GR00T-WholeBodyControl (``decoupled_wbc/control/robot_model/
#: model_data/g1/``), vendored verbatim so both hosts and the offline repair read
#: one file. NOT the amo-vendored ``g1_body29_hand14.urdf``: that one splits the
#: waist offsets differently (roll at z=0.044, pitch at 0 vs 0.035/0.019 here),
#: identical at zero waist but up to 7 mm apart at the waist limits -- and the
#: column has to match what the upstream collector/inference compute.
DEFAULT_URDF = pathlib.Path(__file__).resolve().parent / "assets" / "g1_29dof_with_hand.urdf"

ROOT_LINK = "pelvis"
EEF_LINKS = {"left": "left_wrist_yaw_link", "right": "right_wrist_yaw_link"}

#: Joint names of the 43-entry SONIC ``observation.state``, slot by slot
#: (see ``frames.sonic_state_43``): canonical 29-order body joints with the 7+7
#: Dex3 fingers interleaved after each arm. Only the waist and arm names matter
#: to the FK; the rest are listed so a name lookup is total.
STATE43_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
)
assert len(STATE43_JOINT_NAMES) == 43


def _vec(text, default):
    return np.array([float(x) for x in text.split()], dtype=np.float64) if text else np.array(default, dtype=np.float64)


class _Joint:
    __slots__ = ("name", "kind", "parent", "child", "origin_xyz", "origin_rot", "axis", "state_index")

    def __init__(self, name, kind, parent, child, origin_xyz, origin_rpy, axis):
        self.name = name
        self.kind = kind
        self.parent = parent
        self.child = child
        self.origin_xyz = origin_xyz
        # URDF rpy: fixed-axis roll, pitch, yaw -> R = Rz(y) Ry(p) Rx(r) = extrinsic xyz.
        self.origin_rot = sRot.from_euler("xyz", origin_rpy)
        self.axis = axis / np.linalg.norm(axis) if np.linalg.norm(axis) > 0 else axis
        self.state_index = STATE43_JOINT_NAMES.index(name) if name in STATE43_JOINT_NAMES else None


def _parse_chain(urdf_path: pathlib.Path, eef_link: str) -> list[_Joint]:
    root = ET.parse(str(urdf_path)).getroot()
    by_child = {}
    for j in root.findall("joint"):
        origin = j.find("origin")
        axis = j.find("axis")
        by_child[j.find("child").get("link")] = _Joint(
            j.get("name"),
            j.get("type"),
            j.find("parent").get("link"),
            j.find("child").get("link"),
            _vec(origin.get("xyz") if origin is not None else None, [0, 0, 0]),
            _vec(origin.get("rpy") if origin is not None else None, [0, 0, 0]),
            _vec(axis.get("xyz") if axis is not None else None, [1, 0, 0]),
        )
    chain = []
    link = eef_link
    while link != ROOT_LINK:
        if link not in by_child:
            raise ValueError(f"{urdf_path}: no path from {eef_link} back to {ROOT_LINK} (stuck at {link})")
        j = by_child[link]
        chain.append(j)
        link = j.parent
    chain.reverse()
    for j in chain:
        if j.kind not in ("revolute", "continuous", "fixed"):
            raise ValueError(f"{urdf_path}: unsupported joint type {j.kind!r} on {j.name}")
        if j.kind != "fixed" and j.state_index is None:
            raise ValueError(f"{urdf_path}: moving joint {j.name} is not in the 43-entry state")
    return chain


@lru_cache(maxsize=4)
def _chains(urdf_path: str) -> dict:
    p = pathlib.Path(urdf_path)
    return {side: _parse_chain(p, link) for side, link in EEF_LINKS.items()}


def _fk_chain(chain: list[_Joint], state: np.ndarray) -> tuple[np.ndarray, sRot]:
    """``(pos (N,3), rot)`` of the chain's last link in the root frame."""
    n = state.shape[0]
    pos = np.zeros((n, 3), dtype=np.float64)
    rot = sRot.identity(n)
    for j in chain:
        pos = pos + rot.apply(np.broadcast_to(j.origin_xyz, (n, 3)))
        rot = rot * j.origin_rot
        if j.kind != "fixed":
            q = state[:, j.state_index]
            rot = rot * sRot.from_rotvec(q[:, None] * j.axis[None, :])
    return pos, rot


def eef_state_from_state43(state43, urdf_path=DEFAULT_URDF) -> np.ndarray:
    """``observation.eef_state`` for one state (43,) -> (14,), or a batch (N,43) -> (N,14).

    Pelvis frame (floating base at identity), quaternions wxyz, float64. Only
    the waist (12:15) and arm (15:22, 29:36) entries are read.
    """
    s = np.asarray(state43, dtype=np.float64)
    single = s.ndim == 1
    s = s.reshape(-1, 43)
    chains = _chains(str(urdf_path))
    parts = []
    for side in ("left", "right"):
        pos, rot = _fk_chain(chains[side], s)
        quat = rot.as_quat(canonical=False)[:, [3, 0, 1, 2]]  # xyzw -> wxyz
        parts.append(np.concatenate([pos, quat], axis=1))
    out = np.concatenate(parts, axis=1)
    return out[0] if single else out
