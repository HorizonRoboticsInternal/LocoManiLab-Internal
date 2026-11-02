"""Registration, layout and success checks for the redesigned K02 task."""

import numpy as np
import pytest

from robocasa.environments.ithor_kitchen import (
    IThorKitchenWineBottleOntoStool as Task,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registration_scene_scope_and_layout():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_KITCHEN_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (3,)
    spec = get_ithor_kitchen_task_spec("FloorPlan3", "wine_bottle_onto_stool")
    assert spec["stool_xy"] == [-0.15, -1.70]
    assert spec["robot_pose"]["pos"][:2] == [-0.15, -1.0889]
    assert spec["bottle_xy"] == [0.72, -1.0141]
    assert spec["stool_xy"][1] < spec["robot_pose"]["pos"][1]
    assert spec["bottle_xy"][0] > spec["robot_pose"]["pos"][0]
    assert Task.REMOVED_BODY_NAMES == (Task.CUP_BODY,)


def test_bottle_footprint_must_fit_on_seat_with_small_tolerance():
    seat = np.array([[-0.2, -0.3, 0.9], [0.2, 0.3, 0.96]])
    bottle = np.array([[-0.04, -0.04, 0.96], [0.04, 0.04, 1.25]])
    assert Task._footprint_on_seat(bottle, seat, 0.02)
    assert not Task._footprint_on_seat(bottle + [0.30, 0.0, 0.0], seat, 0.02)


def test_stability_rejects_moving_bottle():
    assert Task._stable(np.zeros(6), 0.04, 0.20)
    assert not Task._stable(np.array([0.05, 0, 0, 0, 0, 0]), 0.04, 0.20)
    assert not Task._stable(np.array([0, 0, 0, 0, 0, 0.21]), 0.04, 0.20)


@pytest.mark.parametrize("freq", [20, 50])
def test_one_second_hold_and_interruption(freq):
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
    assert not task._check_success() and task._success_hold_steps == 0
