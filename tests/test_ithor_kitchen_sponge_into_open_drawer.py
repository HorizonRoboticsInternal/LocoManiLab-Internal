"""Fast geometry, registration and hold checks for Scene 8's sponge task."""

import numpy as np
import pytest

from robocasa.environments.ithor_kitchen import (
    IThorKitchenSpongeIntoOpenDrawer as Task,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registered_scene8_only():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_KITCHEN_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (8,)


def test_whole_sponge_containment_rejects_rim_and_wrong_drawer():
    points = np.array([[-0.07, -0.013, -0.04], [0.07, 0.013, 0.04]])
    center, rotation, half = np.zeros(3), np.eye(3), np.array([0.112, 0.033, 0.130])
    assert Task._inside_drawer(points, center, rotation, half)
    assert not Task._inside_drawer(points + [0.3, 0, 0], center, rotation, half)
    assert not Task._inside_drawer(points + [0.06, 0, 0], center, rotation, half)
    assert not Task._inside_drawer(points + [0, -0.05, 0], center, rotation, half)


def test_region_follows_drawer_pose():
    points = np.array([[-0.07, -0.013, -0.04], [0.07, 0.013, 0.04]])
    rotation = np.array([[-1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, -1.0, 0.0]])
    center = np.array([-0.37, -1.33, 0.796])
    half = np.array([0.112, 0.033, 0.130])
    assert Task._inside_drawer(points @ rotation.T + center, center, rotation, half)


@pytest.mark.parametrize("freq", [20, 30])
def test_continuous_one_second(freq):
    task = object.__new__(Task)
    task.control_freq = freq
    task._get_task_spec = lambda: {"hold_seconds": 1.0}
    task.timestep, task.horizon, task.ignore_done = 0, 1000, True
    task._raw_success = lambda: True
    task._reset_task_progress()
    for _ in range(freq - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    task._raw_success = lambda: False
    task._post_action(None)
    assert not task._check_success() and task._success_hold_steps == 0
