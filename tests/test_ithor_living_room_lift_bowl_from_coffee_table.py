"""Focused registration, height and hold checks for L02."""

import pytest

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomLiftBowlFromCoffeeTable as Task,
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registered_scene201_only():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_LIVING_ROOM_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (201,)


def test_lift_uses_initial_height():
    task = object.__new__(Task)
    task._bowl_initial_z = 0.64
    task._get_task_spec = lambda: {"lift_height": 0.30}
    height = {"value": 0.939}
    task._bowl_root_z = lambda: height["value"]
    assert not task._raw_success()
    height["value"] = 0.94
    assert task._raw_success()


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
