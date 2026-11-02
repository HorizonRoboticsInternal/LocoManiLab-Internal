"""Keep the scene 329 open-drawer calibration isolated from other tasks."""

from unittest.mock import Mock

import numpy as np
import pytest

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_base import IThorRoomEnv
from robocasa.environments.ithor_bedroom.tasks.drawer import (
    IThorBedroomCloseDrawer,
    IThorBedroomOpenDrawer,
)


def test_scene329_uses_fixed_pose_without_search(monkeypatch):
    search = Mock(side_effect=AssertionError("fixed pose must bypass search"))
    monkeypatch.setattr(
        ExactSceneArticulation, "_finalize_articulation_robot_pose", search
    )
    task = object.__new__(IThorBedroomOpenDrawer)
    task.scene_name = "FloorPlan329"
    task._set_robot_free_pose = Mock(return_value=True)
    task._calibrate_robot_free_joint_height = Mock(return_value=True)
    task.count_body_collisions = Mock(return_value=0)

    assert task._finalize_articulation_robot_pose(0.45)
    search.assert_not_called()
    task._set_robot_free_pose.assert_called_once_with(
        task._SCENE_329_ROBOT_POS[:2], task._SCENE_329_ROBOT_YAW
    )
    np.testing.assert_array_equal(task.robot_base_pos, task._SCENE_329_ROBOT_POS)
    assert task.robot_base_ori[2] == task._SCENE_329_ROBOT_YAW
    assert task.INITIAL_ROBOT_JITTER_RADIUS == 0.0


def test_other_scenes_keep_search_and_restore_jitter(monkeypatch):
    search = Mock(return_value=True)
    monkeypatch.setattr(
        ExactSceneArticulation, "_finalize_articulation_robot_pose", search
    )
    task = object.__new__(IThorBedroomOpenDrawer)
    task.scene_name = "FloorPlan301"
    task.INITIAL_ROBOT_JITTER_RADIUS = 0.0  # A preceding scene 329 reset.

    assert task._finalize_articulation_robot_pose(0.45)
    search.assert_called_once_with(0.45)
    assert task.INITIAL_ROBOT_JITTER_RADIUS == pytest.approx(0.10)
    # No simulation is attached: another scene must not touch the robot pose.
    task._constrain_initial_jitter_candidate()


def test_close_drawer_keeps_original_initialization():
    assert (
        IThorBedroomCloseDrawer._finalize_articulation_robot_pose
        is ExactSceneArticulation._finalize_articulation_robot_pose
    )
    assert (
        IThorBedroomCloseDrawer._constrain_initial_jitter_candidate
        is IThorRoomEnv._constrain_initial_jitter_candidate
    )
    assert IThorBedroomCloseDrawer.INITIAL_ROBOT_JITTER_RADIUS == pytest.approx(0.10)
