from __future__ import annotations

import numpy as np

from sim2real.camera.messages import decode_camera_frame, encode_camera_frame


def test_camera_frame_round_trip_preserves_timestamps_and_image_shape() -> None:
    image = (np.random.rand(48, 64, 3) * 255).astype(np.uint8)
    raw = encode_camera_frame(
        images={"ego_view": image},
        timestamps={"ego_view": 123.456},
    )

    decoded = decode_camera_frame(raw)

    assert decoded["timestamps"] == {"ego_view": 123.456}
    assert set(decoded["images"]) == {"ego_view"}
    assert decoded["images"]["ego_view"].shape == (48, 64, 3)
    assert decoded["images"]["ego_view"].dtype == np.uint8


def test_camera_frame_round_trip_supports_multiple_named_cameras() -> None:
    images = {
        "ego_view": (np.random.rand(32, 32, 3) * 255).astype(np.uint8),
        "wrist_view": (np.random.rand(32, 32, 3) * 255).astype(np.uint8),
    }
    timestamps = {"ego_view": 1.0, "wrist_view": 1.0001}

    decoded = decode_camera_frame(encode_camera_frame(images, timestamps))

    assert set(decoded["images"]) == {"ego_view", "wrist_view"}
    assert decoded["timestamps"] == timestamps


def test_camera_frame_jpeg_round_trip_is_lossy_but_close() -> None:
    # Flat-color image so JPEG quantization doesn't introduce visible artifacts.
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    decoded = decode_camera_frame(encode_camera_frame({"cam": image}, {"cam": 0.0}))

    assert np.abs(decoded["images"]["cam"].astype(int) - 128).max() <= 5
