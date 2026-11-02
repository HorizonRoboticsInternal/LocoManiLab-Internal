"""Focused registration, containment and hold checks for L05."""

import numpy as np
import pytest

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomStatueIntoBox as Task,
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registered_scene209_only():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_LIVING_ROOM_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (209,)
    assert Task.STATUE_XML.endswith("Stone_Statue_4/Stone_Statue_4.xml")
    assert Task.STATUE_SCALE == 0.5
    assert Task.NATIVE_STATUE_BODY.startswith("statue_")


def test_statue_footprint_and_bottom_must_be_inside_box():
    lower, upper = np.array([-0.14, -0.10]), np.array([0.14, 0.10])
    points = np.array([[-0.08, -0.08, 0.01], [0.08, 0.08, 0.42]])
    check = Task._inside_open_box
    assert check(points, lower, upper, 0.0, 0.28, 0.005, 0.02)
    assert not check(points + [0.08, 0.0, 0.0], lower, upper, 0.0, 0.28, 0.005, 0.02)
    assert not check(points - [0.0, 0.0, 0.05], lower, upper, 0.0, 0.28, 0.005, 0.02)
    assert not check(points + [0.0, 0.0, 0.31], lower, upper, 0.0, 0.28, 0.005, 0.02)


@pytest.mark.parametrize("freq", [20, 50])
def test_continuous_one_second(freq):
    task = object.__new__(Task)
    task.control_freq = freq
    task._get_task_spec = lambda: {"hold_seconds": 1.0}
    task.timestep, task.horizon, task.ignore_done = 0, 1000, True
    task._raw_success = lambda: True
    task._success_hold_steps = 0
    for _ in range(freq - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    task._raw_success = lambda: False
    task._post_action(None)
    assert not task._check_success()
    assert task._success_hold_steps == 0
