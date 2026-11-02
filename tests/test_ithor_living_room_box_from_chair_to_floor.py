"""Contracts for redesigned Scene 201 L01."""

import pytest

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomBoxFromChairToFloor as Task,
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registered_scene201_only_and_layout():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_LIVING_ROOM_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (201,)
    spec = get_ithor_livingroom_task_spec("FloorPlan201", "box_from_chair_to_floor")
    assert spec["robot_pose"]["pos"][:2] == [-1.248, 2.507]
    assert spec["chair_xy"] == [-1.8592, 2.31]
    assert spec["chair_quat"] == [
        0.032031399914441316,
        0.02856371822958566,
        0.7060279544121009,
        0.7068823317986427,
    ]
    assert spec["hold_seconds"] == 1.0
    assert Task.BOX_XML.endswith("manual/open_cardboard_box/model.xml")


def test_success_requires_floor_contact_release_and_stability():
    task = object.__new__(Task)
    state = {"floor": True, "released": True, "stable": True}
    task._box_contacts_floor = lambda: state["floor"]
    task.object_released = lambda name, th: state["released"]
    task._box_is_stable = lambda: state["stable"]
    task._get_task_spec = lambda: {"release_distance": 0.2}
    assert task._raw_success()
    for key in state:
        state[key] = False
        assert not task._raw_success()
        state[key] = True


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
