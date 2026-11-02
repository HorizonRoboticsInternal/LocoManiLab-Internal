from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from sim2real.rl_policy.recording.extras.sonic_extra import unpack_pose_message

HEADER_SIZE = 1280


def _pack_message(topic: str, fields: dict[str, np.ndarray]) -> bytes:
    """Build a wire-format message matching gear_sonic's own encoder, for testing
    the decoder (``unpack_pose_message``) independently of a live ZMQ publisher."""
    header = {
        "v": 1,
        "endian": "le",
        "fields": [
            {"name": name, "dtype": "f32", "shape": list(arr.shape)}
            for name, arr in fields.items()
        ],
    }
    header_bytes = json.dumps(header).encode("utf-8")
    if len(header_bytes) > HEADER_SIZE:
        raise ValueError("test header too large")
    header_bytes = header_bytes + b"\x00" * (HEADER_SIZE - len(header_bytes))

    payload = b"".join(arr.astype(np.float32).tobytes() for arr in fields.values())
    return topic.encode("utf-8") + header_bytes + payload


def test_unpack_pose_message_round_trips_fields() -> None:
    smpl_joints = np.arange(24 * 3, dtype=np.float32).reshape(1, 24, 3)
    frame_index = np.array([[42]], dtype=np.float32)
    raw = _pack_message("pose", {"smpl_joints": smpl_joints, "frame_index": frame_index})

    decoded = unpack_pose_message(raw, topic="pose")

    assert decoded["version"] == 1
    np.testing.assert_array_equal(decoded["smpl_joints"], smpl_joints)
    np.testing.assert_array_equal(decoded["frame_index"], frame_index)


def test_unpack_pose_message_rejects_wrong_topic() -> None:
    raw = _pack_message("pose", {"x": np.zeros((1,), dtype=np.float32)})
    with pytest.raises(ValueError, match="does not start with expected topic"):
        unpack_pose_message(raw, topic="planner")


def test_unpack_pose_message_rejects_truncated_data() -> None:
    with pytest.raises(ValueError, match="too small"):
        unpack_pose_message(b"pose" + b"\x00" * 10, topic="pose")
