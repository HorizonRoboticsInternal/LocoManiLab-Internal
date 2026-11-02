"""Regression checks for lossless simulator ego-depth sidecars."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from robocasa.utils.g1_data_collection import (
    DEPTH_FEATURE,
    DEPTH_MAX_MM,
    EGO_CAMERA,
    EGO_H,
    EGO_W,
    DepthSequenceWriter,
    _depth_millimeters,
    _depth_rgb_image_from_millimeters,
    build_collection_info,
)


def _obs(depth):
    depth = np.asarray(depth)
    return {
        f"{EGO_CAMERA}_depth": depth,
        f"{EGO_CAMERA}_image": np.zeros((*depth.shape[:2], 3), dtype=np.uint8),
    }


def test_depth_is_rounded_to_uint16_millimeters(monkeypatch):
    # The fake treats the vertically corrected MuJoCo buffer as metric depth so
    # this test isolates orientation, quantization, invalid values and overflow.
    metric_depth = np.array(
        [
            [0.0, 0.0004, 0.0015, 1.234],
            [65.535, 70.0, np.nan, -1.0],
        ],
        dtype=np.float64,
    )
    stored_buffer = metric_depth[::-1]
    monkeypatch.setattr(
        "robosuite.utils.camera_utils.get_real_depth_map",
        lambda _sim, vertically_corrected: vertically_corrected,
    )

    depth_mm = _depth_millimeters(_obs(stored_buffer), sim=object())

    assert depth_mm.dtype == np.uint16
    assert depth_mm.flags.c_contiguous
    np.testing.assert_array_equal(
        depth_mm,
        [
            [0, 1, 2, 1234],
            [65535, DEPTH_MAX_MM, 0, 0],
        ],
    )


def test_depth_resolution_must_match_rgb(monkeypatch):
    monkeypatch.setattr(
        "robosuite.utils.camera_utils.get_real_depth_map",
        lambda _sim, depth: depth,
    )
    obs = _obs(np.zeros((2, 3), dtype=np.float32))
    obs[f"{EGO_CAMERA}_image"] = np.zeros((3, 2, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="does not match RGB"):
        _depth_millimeters(obs, sim=object())


def test_pseudocolor_png_keeps_depth_resolution_and_marks_invalid_black():
    depth_mm = np.array([[0, 250], [1000, 5000]], dtype=np.uint16)
    image = _depth_rgb_image_from_millimeters(depth_mm)

    assert image.mode == "RGB"
    assert image.size == (2, 2)
    assert np.asarray(image)[0, 0].tolist() == [0, 0, 0]


def test_writer_commits_png_and_raw_uint16_depth(tmp_path):
    depth_mm = np.arange(EGO_H * EGO_W, dtype=np.uint16).reshape(EGO_H, EGO_W)
    writer = DepthSequenceWriter(tmp_path, workers=1)
    writer.add(depth_mm, episode_index=0, frame_index=0)
    writer.commit(0)
    writer.close()

    episode_dir = tmp_path / "depth" / DEPTH_FEATURE / "episode_000000"
    png_path = episode_dir / "frame_000000.png"
    npy_path = episode_dir / "frame_000000.npy"
    with Image.open(png_path) as image:
        assert image.mode == "RGB"
        assert image.size == (EGO_W, EGO_H)
    saved = np.load(npy_path, allow_pickle=False)
    assert saved.dtype == np.uint16
    np.testing.assert_array_equal(saved, depth_mm)


def test_writer_rejects_non_uint16_or_non_rgb_resolution(tmp_path):
    writer = DepthSequenceWriter(tmp_path, workers=1)
    with pytest.raises(TypeError, match="must be uint16"):
        writer.add(
            np.zeros((EGO_H, EGO_W), dtype=np.float32),
            episode_index=0,
            frame_index=0,
        )
    with pytest.raises(ValueError, match="must match RGB video"):
        writer.add(
            np.zeros((EGO_H - 1, EGO_W), dtype=np.uint16),
            episode_index=0,
            frame_index=0,
        )
    writer.close()


def test_collection_metadata_describes_uint16_millimeter_sidecars():
    args = SimpleNamespace(
        backend="sonic",
        pico_source="mock",
    )
    info = build_collection_info(args, {"composite_controller_specific_configs": {}})
    depth = info["depth"]

    assert depth["key"] == "observation.images.ego_depth_view"
    assert depth["npy_dtype"] == "uint16"
    assert depth["npy_unit"] == "millimeter"
    assert depth["shape"] == [EGO_H, EGO_W]
    assert depth["matches_rgb_feature"] == "observation.images.ego_view"
