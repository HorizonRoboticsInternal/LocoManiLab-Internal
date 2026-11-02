"""Contracts for redesigned Scene 205 L03."""

import pytest

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomBoxFromSofaToFloor as Task,
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registered_scene205_only_and_layout():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_LIVING_ROOM_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (205,)
    spec = get_ithor_livingroom_task_spec("FloorPlan205", "box_from_sofa_to_floor")
    assert spec["robot_pose"]["pos"][:2] == [-3.60, 5.10]
    assert spec["box_xy"] == [-4.20, 5.10]
    assert spec["hold_seconds"] == 1.0
    assert Task.BOX_BODY.startswith("box_")
    assert Task.SOFA_BODY.startswith("sofa_")


@pytest.mark.parametrize("freq", [20, 50])
def test_continuous_one_second_and_interruption(freq):
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
