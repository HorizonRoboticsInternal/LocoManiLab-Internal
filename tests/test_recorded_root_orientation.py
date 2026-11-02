"""The recorded attitude columns must be the robot's actual attitude.

``observation.root_orientation`` and ``observation.projected_gravity`` are the
only two recorded columns that go through a quaternion convention boundary:
robosuite's ``robot0_base_quat`` sensor is ``T.mat2quat(...)``, i.e. **xyzw**,
while ``g1_bridge.recording.frames`` documents and assumes **wxyz** (which is
what the real-robot host's ``read_pelvis_state()[0]`` gives it).

Handing the xyzw straight over produced a reordered quaternion -- still unit
norm, still plausible-looking, and wrong: a standing G1 recorded 30.8 deg of
tilt where its pelvis was at 0.0. Nothing downstream could tell, which is why
this is pinned here rather than left to a visual check.

Run: pytest tests/test_recorded_root_orientation.py
"""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as sRot

from robocasa.utils.g1_data_collection import _base_quat_wxyz
from g1_bridge.recording.frames import projected_gravity


def _obs(quat_xyzw):
    return {"robot0_base_quat": np.asarray(quat_xyzw, dtype=np.float64)}


def test_conversion_reorders_xyzw_into_wxyz():
    assert _base_quat_wxyz(_obs([0.1, 0.2, 0.3, 0.9])) == pytest.approx(
        [0.9, 0.1, 0.2, 0.3]
    )


def test_upright_robot_records_gravity_straight_down():
    """Identity attitude -> [0, 0, -1]. The unconverted path gives [0, 0, +1]."""
    upright_xyzw = [0.0, 0.0, 0.0, 1.0]
    g = projected_gravity(_base_quat_wxyz(_obs(upright_xyzw)))
    np.testing.assert_allclose(g, [0.0, 0.0, -1.0], atol=1e-9)


@pytest.mark.parametrize(
    "axis,angle_deg", [("y", 20.0), ("x", -15.0), ("z", 90.0), ("y", 45.0)]
)
def test_recorded_gravity_matches_the_attitude_it_came_from(axis, angle_deg):
    """For any attitude, the recorded vector is gravity in the base frame.

    Built independently of the production path: take a rotation, express it the
    way robosuite's sensor would, and check the column equals ``R^T @ [0,0,-1]``.
    A yaw-only case is included because yaw must NOT change the vector -- the
    buggy path changed it for every rotation, including pure yaw.
    """
    rot = sRot.from_euler(axis, angle_deg, degrees=True)
    obs = _obs(rot.as_quat())  # scipy returns xyzw, same as robosuite's sensor
    expected = rot.as_matrix().T @ np.array([0.0, 0.0, -1.0])

    np.testing.assert_allclose(
        projected_gravity(_base_quat_wxyz(obs)), expected, atol=1e-9
    )


def test_pure_yaw_leaves_gravity_untouched():
    for yaw in (0.0, 30.0, 179.0, -95.0):
        obs = _obs(sRot.from_euler("z", yaw, degrees=True).as_quat())
        np.testing.assert_allclose(
            projected_gravity(_base_quat_wxyz(obs)), [0.0, 0.0, -1.0], atol=1e-9
        )


def test_the_unconverted_path_is_what_it_used_to_record():
    """Pin the old behaviour so the regression is unmistakable if it returns.

    A robot yawed 33 deg but perfectly upright used to record 30.8 deg of tilt.
    """
    quat_xyzw = sRot.from_euler("z", 33.0, degrees=True).as_quat()
    buggy = projected_gravity(quat_xyzw)  # xyzw fed to a wxyz-expecting helper
    fixed = projected_gravity(_base_quat_wxyz(_obs(quat_xyzw)))

    def tilt(g):
        return np.degrees(np.arccos(np.clip(-g[2], -1.0, 1.0)))

    assert tilt(fixed) == pytest.approx(0.0, abs=1e-6)
    assert tilt(buggy) > 25.0
