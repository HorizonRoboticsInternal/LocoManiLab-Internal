"""Pure-numpy tests for the image-space lens randomization."""

import numpy as np

from robocasa.utils.scene_randomization import LensWarp


def _checker(h=240, w=320):
    yy, xx = np.mgrid[0:h, 0:w]
    img = (((xx // 20) + (yy // 20)) % 2 * 255).astype(np.uint8)
    return np.stack([img, img, img], axis=-1)


def test_identity_at_zero_scale():
    img = _checker()
    warp = LensWarp(240, 320, np.random.default_rng(0), scale=0.0)
    assert warp(img) is img
    assert warp.describe()["zoom"] == 1.0


def test_shape_dtype_and_actually_warps():
    img = _checker()
    warp = LensWarp(240, 320, np.random.default_rng(1), scale=1.0)
    out = warp(img)
    assert out.shape == img.shape and out.dtype == img.dtype
    assert np.abs(out.astype(int) - img.astype(int)).mean() > 5
    d = warp.describe()
    assert -0.25 <= d["k1"] <= 0.25 and 1.0 <= d["zoom"] <= 1.12
    assert all(abs(v) <= 8.0 for v in d["principal_shift_px"])


def test_deterministic_per_seed():
    img = _checker()
    a = LensWarp(240, 320, np.random.default_rng(7), scale=1.0)(img)
    b = LensWarp(240, 320, np.random.default_rng(7), scale=1.0)(img)
    c = LensWarp(240, 320, np.random.default_rng(8), scale=1.0)(img)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
