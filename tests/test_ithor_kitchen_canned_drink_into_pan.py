"""Registration, containment and hold checks for redesigned K05."""

import numpy as np
import pytest

from robocasa.environments.ithor_kitchen import (
    IThorKitchenCannedDrinkIntoPan as Task,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registered_scene13_only_and_layout():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_KITCHEN_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (13,)
    spec = get_ithor_kitchen_task_spec("FloorPlan13", "canned_drink_into_pan")
    assert spec["can_pos"][:2] == [-1.95, 2.02]
    assert spec["robot_pose"]["pos"][:2] == [-2.12, 2.55]
    assert spec["can_quat"] == [1.0, 0.0, 0.0, 0.0]
    assert spec["hold_seconds"] == 1.0


def test_point_inside_translated_rotated_pan_region():
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    center = np.array([2.0, 3.0, 1.0])
    half = np.array([0.06, 0.05, 0.09])
    assert Task._point_inside_region(center, center, rotation, half, 0.015)
    assert Task._point_inside_region(
        center + [0.0, 0.07, 0.0], center, rotation, half, 0.015
    )
    assert not Task._point_inside_region(
        center + [0.0, 0.08, 0.0], center, rotation, half, 0.015
    )


@pytest.mark.parametrize("freq", [20, 50])
def test_continuous_one_second_and_interruption(freq):
    task = object.__new__(Task)
    task.control_freq = freq
    task._get_task_spec = lambda: {"hold_seconds": 1.0}
    task.timestep, task.horizon, task.ignore_done = 0, 1000, True
    task._canned_drink_in_pan = lambda: True
    task._success_hold_steps = 0
    for _ in range(freq - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    task._canned_drink_in_pan = lambda: False
    task._post_action(None)
    assert not task._check_success()
    assert task._success_hold_steps == 0
