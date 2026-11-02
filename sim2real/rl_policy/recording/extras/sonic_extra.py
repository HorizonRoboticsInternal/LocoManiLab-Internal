"""sonic PICO/SMPL teleop metadata -- the ``teleop.*`` fields.

Ported from LocoManiLab's GR00T-WholeBodyControl fork
(``external/GR00T-WholeBodyControl/gear_sonic/scripts/run_data_exporter.py``'s
``_handle_pose_message``/``_handle_planner_message``/``_handle_manager_state``/
``_add_sonic_pose_features``): sonic records raw SMPL retargeting output and PICO
planner/VR-tracking metadata alongside the generic body state, for downstream analysis
of what the human operator was doing (not fed into the RL policy itself).

Wire format: ``pose``/``planner``/``manager_state`` topics on a plain ZMQ SUB socket,
each message being ``topic_bytes + 1280_byte_json_header + concatenated_binary_fields``
(``unpack_pose_message`` below is a verbatim port of the original). These are published
by ``pico_manager_thread_server``, a component of ``gear_sonic`` itself -- NOT sim2real's
own ``teleop/pico_retarget_pub.py``, which speaks a different, simpler protocol. This
extra only produces real data when that sonic-specific server is running and pointed at
the configured host/port; otherwise every field stays at its zero/default fallback,
exactly like the original exporter's own fallback path when no message has arrived yet.

Deliberately NOT ported (out of scope for this extra, would need infrastructure this
repo doesn't have):
  - ``action.motion_token`` / ``action.wbc`` -- come from the sonic C++ controller's own
    ``g1_debug`` proprio topic and (for ``wbc``) the sonic RL policy's own
    actuated-joint reconstruction; neither concept exists in sim2real's ``BasePolicy``.
  - ``observation.eef_state`` -- needs a forward-kinematics robot model
    (``Gr00tRobotModel``/``robot_model.frame_placement``); sim2real has no equivalent.
  - ``observation.cpp_rotation_offset``/``observation.init_base_quat``/
    ``teleop.delta_heading`` -- also sourced from the C++ ``g1_debug`` proprio topic
    (same category as heft's hand state in ``extras/heft_extra.py``, but not implemented here
    since it needs its own wire-format port once that topic's schema is confirmed).
    ``teleop.target_body_orientation`` below reuses the original's yaw-zeroing formula
    with ``delta_heading`` pinned to 0.0 (i.e. no C++ heading correction applied) --
    documented inline where it matters.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import zmq
from loguru import logger
from scipy.spatial.transform import Rotation as R

from sim2real.rl_policy.recording.extras.base import RecordingExtra

DEFAULT_SONIC_TELEOP_PORT = 5556
HEADER_SIZE = 1280

_DTYPE_MAP = {"f32": np.float32, "f64": np.float64, "i32": np.int32, "i64": np.int64, "bool": bool}

G1_L_WRIST_ROLL_IDX = 23
G1_L_WRIST_PITCH_IDX = 25
G1_L_WRIST_YAW_IDX = 27
G1_R_WRIST_ROLL_IDX = 24
G1_R_WRIST_PITCH_IDX = 26
G1_R_WRIST_YAW_IDX = 28


def unpack_pose_message(packed_data: bytes, topic: str) -> dict:
    """Verbatim port of gear_sonic's wire format: [topic][1280B JSON header][fields]."""
    topic_bytes = topic.encode("utf-8")
    if not packed_data.startswith(topic_bytes):
        raise ValueError(f"Message does not start with expected topic '{topic}'")

    offset = len(topic_bytes)
    if len(packed_data) < offset + HEADER_SIZE:
        raise ValueError(f"Packed data too small: {len(packed_data)} < {offset + HEADER_SIZE}")

    header_bytes = packed_data[offset:offset + HEADER_SIZE]
    null_idx = header_bytes.find(b"\x00")
    if null_idx > 0:
        header_bytes = header_bytes[:null_idx]
    header = json.loads(header_bytes.decode("utf-8"))

    result: dict[str, Any] = {"version": header.get("v", 0), "endian": header.get("endian", "le")}
    current_offset = offset + HEADER_SIZE
    for field in header.get("fields", []):
        dtype = _DTYPE_MAP.get(field["dtype"], np.float32)
        shape = tuple(field["shape"])
        n_bytes = int(np.prod(shape)) * np.dtype(dtype).itemsize
        result[field["name"]] = (
            np.frombuffer(packed_data[current_offset:current_offset + n_bytes], dtype=dtype)
            .reshape(shape)
            .copy()
        )
        current_offset += n_bytes
    return result


def _extract_hand_joints(data: dict, key: str) -> np.ndarray:
    arr = data.get(key)
    if arr is None:
        return np.zeros(7, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[0]
    return arr.astype(np.float32)


def _quat_to_rot6d(quat_wxyz: np.ndarray) -> np.ndarray:
    """Standard continuous 6D rotation representation (Zhou et al. 2019): the first
    two columns of the rotation matrix, flattened.

    NOTE: this is a reimplementation from the standard definition, not a verified
    byte-for-byte port of gear_sonic's own ``quat_to_rot6d``
    (``gear_sonic/utils/data_collection/transforms.py``) -- that file wasn't read
    this session, so its exact per-quaternion layout/convention for the 3-point VR
    orientation (``teleop.vr_3pt_orientation``, 12 floats in -> 18 out) is assumed,
    not confirmed. ``teleop.target_body_orientation`` uses this same helper on a
    single quaternion, where the wxyz/scalar-first convention IS directly confirmed
    from ``run_data_exporter.py``'s own ``R.from_quat(body_quat_w, scalar_first=True)``.
    """
    rot_matrix = R.from_quat(quat_wxyz, scalar_first=True).as_matrix()
    return rot_matrix[:, :2].reshape(-1).astype(np.float32)


class SonicTeleopRecordingExtra(RecordingExtra, extra_name="sonic-teleop"):
    def __init__(
        self,
        policy: Any,
        host: str = "localhost",
        port: int = DEFAULT_SONIC_TELEOP_PORT,
    ) -> None:
        super().__init__(policy)
        self._stream_mode = 0
        self._smpl_msg: dict | None = None
        self._planner_msg: dict | None = None
        self._initial_yaw: float | None = None

        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.setsockopt(zmq.CONFLATE, 0)
        self._socket.setsockopt(zmq.RCVHWM, 20)
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "pose")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "planner")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "manager_state")
        try:
            self._socket.connect(f"tcp://{host}:{port}")
            logger.info("SonicTeleopRecordingExtra connected to tcp://{}:{}", host, port)
        except Exception as exc:
            logger.warning(
                "SonicTeleopRecordingExtra ZMQ connect failed, teleop fields will be "
                "defaults: {}", exc,
            )

    def _poll(self) -> None:
        for _ in range(20):
            try:
                raw = self._socket.recv(zmq.NOBLOCK)
            except zmq.Again:
                return
            try:
                if raw.startswith(b"manager_state"):
                    data = unpack_pose_message(raw, topic="manager_state")
                    if "stream_mode" in data:
                        self._stream_mode = int(data["stream_mode"].flat[0])
                elif raw.startswith(b"planner"):
                    self._planner_msg = unpack_pose_message(raw, topic="planner")
                elif raw.startswith(b"pose"):
                    self._smpl_msg = unpack_pose_message(raw, topic="pose")
            except Exception as exc:
                logger.debug("SonicTeleopRecordingExtra decode error: {}", exc)

    def features(self) -> dict[str, dict]:
        return {
            "teleop.stream_mode": {"dtype": "int32", "shape": (1,), "names": None},
            "teleop.smpl_joints": {"dtype": "float32", "shape": (72,), "names": None},
            "teleop.smpl_pose": {"dtype": "float32", "shape": (63,), "names": None},
            "teleop.body_quat_w": {"dtype": "float32", "shape": (4,), "names": ["w", "x", "y", "z"]},
            "teleop.target_body_orientation": {"dtype": "float32", "shape": (6,), "names": None},
            "teleop.left_wrist_joints": {"dtype": "float32", "shape": (3,), "names": None},
            "teleop.right_wrist_joints": {"dtype": "float32", "shape": (3,), "names": None},
            "teleop.smpl_frame_index": {"dtype": "int64", "shape": (1,), "names": None},
            "teleop.left_hand_joints": {"dtype": "float32", "shape": (7,), "names": None},
            "teleop.right_hand_joints": {"dtype": "float32", "shape": (7,), "names": None},
            "teleop.planner_mode": {"dtype": "int32", "shape": (1,), "names": None},
            "teleop.planner_movement": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "teleop.planner_facing": {"dtype": "float32", "shape": (3,), "names": ["x", "y", "z"]},
            "teleop.planner_speed": {"dtype": "float32", "shape": (1,), "names": None},
            "teleop.planner_height": {"dtype": "float32", "shape": (1,), "names": None},
            "teleop.vr_3pt_position": {"dtype": "float32", "shape": (9,), "names": None},
            "teleop.vr_3pt_orientation": {"dtype": "float32", "shape": (18,), "names": None},
        }

    def compute(self, frame: dict[str, Any]) -> dict[str, Any]:
        self._poll()

        smpl_msg = self._smpl_msg
        use_smpl = self._stream_mode in (1, 4) and smpl_msg is not None

        planner_msg = self._planner_msg
        use_planner = self._stream_mode == 5 and planner_msg is not None

        out: dict[str, Any] = {"teleop.stream_mode": np.array([self._stream_mode], dtype=np.int32)}

        if use_smpl and smpl_msg.get("smpl_joints") is not None:
            joints = np.asarray(smpl_msg["smpl_joints"], dtype=np.float32)
            out["teleop.smpl_joints"] = joints.reshape(-1) if joints.ndim == 2 else joints
        else:
            out["teleop.smpl_joints"] = np.zeros(72, dtype=np.float32)

        if use_smpl and smpl_msg.get("smpl_pose") is not None:
            pose = np.asarray(smpl_msg["smpl_pose"], dtype=np.float32)
            out["teleop.smpl_pose"] = pose.reshape(-1) if pose.ndim > 1 else pose
        else:
            out["teleop.smpl_pose"] = np.zeros(63, dtype=np.float32)

        body_quat_w = (
            smpl_msg["body_quat_w"].astype(np.float32)
            if use_smpl and smpl_msg.get("body_quat_w") is not None
            else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        )
        out["teleop.body_quat_w"] = body_quat_w
        # delta_heading would come from sonic's C++ g1_debug proprio topic (not wired
        # up here, see module docstring) -- pinned to 0.0, so this only applies the
        # yaw-zeroing (first-frame-relative) part of the original computation.
        out["teleop.target_body_orientation"] = self._compute_target_body_orientation(
            body_quat_w, delta_heading=0.0
        )

        joint_pos = None
        if use_smpl and smpl_msg.get("joint_pos") is not None and smpl_msg["joint_pos"].ndim == 2:
            joint_pos = smpl_msg["joint_pos"][0]
        if joint_pos is not None:
            out["teleop.left_wrist_joints"] = np.array(
                [joint_pos[G1_L_WRIST_ROLL_IDX], joint_pos[G1_L_WRIST_PITCH_IDX],
                 joint_pos[G1_L_WRIST_YAW_IDX]], dtype=np.float32)
            out["teleop.right_wrist_joints"] = np.array(
                [joint_pos[G1_R_WRIST_ROLL_IDX], joint_pos[G1_R_WRIST_PITCH_IDX],
                 joint_pos[G1_R_WRIST_YAW_IDX]], dtype=np.float32)
        else:
            out["teleop.left_wrist_joints"] = np.zeros(3, dtype=np.float32)
            out["teleop.right_wrist_joints"] = np.zeros(3, dtype=np.float32)

        out["teleop.smpl_frame_index"] = (
            smpl_msg["frame_index"].astype(np.int64).reshape(1)
            if use_smpl and smpl_msg is not None and smpl_msg.get("frame_index") is not None
            else np.array([0], dtype=np.int64)
        )

        hand_msg = smpl_msg if use_smpl else (planner_msg if planner_msg is not None else smpl_msg)
        out["teleop.left_hand_joints"] = _extract_hand_joints(hand_msg or {}, "left_hand_joints")
        out["teleop.right_hand_joints"] = _extract_hand_joints(hand_msg or {}, "right_hand_joints")

        out["teleop.planner_mode"] = np.array(
            [int(planner_msg["mode"].flat[0])] if use_planner and "mode" in planner_msg else [0],
            dtype=np.int32,
        )
        out["teleop.planner_movement"] = (
            planner_msg["movement"].flatten().astype(np.float32)
            if use_planner and planner_msg.get("movement") is not None
            and planner_msg["movement"].size == 3
            else np.zeros(3, dtype=np.float32)
        )
        out["teleop.planner_facing"] = (
            planner_msg["facing"].flatten().astype(np.float32)
            if use_planner and planner_msg.get("facing") is not None
            and planner_msg["facing"].size == 3
            else np.array([1.0, 0.0, 0.0], dtype=np.float32)
        )
        out["teleop.planner_speed"] = np.array(
            [float(planner_msg["speed"].flat[0])] if use_planner and "speed" in planner_msg else [-1.0],
            dtype=np.float32,
        )
        out["teleop.planner_height"] = np.array(
            [float(planner_msg["height"].flat[0])] if use_planner and "height" in planner_msg else [-1.0],
            dtype=np.float32,
        )

        if use_planner and planner_msg.get("vr_position") is not None and planner_msg["vr_position"].size == 9:
            out["teleop.vr_3pt_position"] = planner_msg["vr_position"].flatten().astype(np.float32)
        else:
            out["teleop.vr_3pt_position"] = np.zeros(9, dtype=np.float32)

        if use_planner and planner_msg.get("vr_orientation") is not None and planner_msg["vr_orientation"].size == 12:
            quats = planner_msg["vr_orientation"].flatten().astype(np.float32).reshape(3, 4)
            out["teleop.vr_3pt_orientation"] = np.concatenate(
                [_quat_to_rot6d(q) for q in quats]
            ).astype(np.float32)
        else:
            out["teleop.vr_3pt_orientation"] = np.zeros(18, dtype=np.float32)

        return out

    def _compute_target_body_orientation(self, body_quat_w: np.ndarray, delta_heading: float) -> np.ndarray:
        body_rot = R.from_quat(body_quat_w, scalar_first=True)
        target_rot = R.from_euler("z", delta_heading, degrees=False) * body_rot
        euler = target_rot.as_euler("ZYX", degrees=False)
        current_yaw = euler[0]

        if self._initial_yaw is None:
            self._initial_yaw = current_yaw

        normalised_euler = np.array([current_yaw - self._initial_yaw, euler[1], euler[2]])
        target_quat = R.from_euler("ZYX", normalised_euler, degrees=False).as_quat(scalar_first=True)
        return _quat_to_rot6d(target_quat.astype(np.float32))

    def close(self) -> None:
        try:
            self._socket.close(0)
        except Exception as exc:
            logger.debug("Failed to stop SonicTeleopRecordingExtra socket cleanly: {}", exc)
