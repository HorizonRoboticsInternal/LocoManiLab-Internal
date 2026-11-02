"""Resources needed by an installed g1_backend wheel, without model weights.

Run this file from outside the source tree after installing the wheel to catch
resources that work in an editable checkout but are missing from distributions.
"""

from __future__ import annotations

import numpy as np

from g1_bridge.recording.g1_fk import DEFAULT_URDF, eef_state_from_state43


def test_recording_fk_uses_the_packaged_urdf() -> None:
    assert DEFAULT_URDF.is_file(), f"Recording FK resource is missing: {DEFAULT_URDF}"
    # Reading the XML and traversing both chains exercises the actual resource,
    # rather than merely checking its presence in the wheel archive.
    result = eef_state_from_state43(np.zeros(43))
    assert result.shape == (14,)
    assert np.isfinite(result).all()
    np.testing.assert_allclose(np.linalg.norm(result[3:7]), 1.0)
    np.testing.assert_allclose(np.linalg.norm(result[10:14]), 1.0)
