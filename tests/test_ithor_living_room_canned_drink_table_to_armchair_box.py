"""Contracts for the Scene 226 canned-drink-to-armchair-box task."""

import math

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomCannedDrinkFromTableIntoBoxOnArmchair,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robocasa.environments.ithor_living_room.tasks.composite import __all__
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


Task = IThorLivingRoomCannedDrinkFromTableIntoBoxOnArmchair


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (226,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_LIVING_ROOM_ENVS[Task.__name__] is Task


def test_layout_uses_table_can_and_open_armchair_box():
    spec = get_ithor_livingroom_task_spec(
        "FloorPlan226", "canned_drink_from_table_into_box_on_armchair"
    )
    assert Task.CAN_XML.endswith("can/can_0/model.xml")
    assert spec["robot_pose"] == {
        "pos": [-0.8178, -1.15, 0.05],
        "yaw": 1.406,
    }
    assert spec["can_xy"] == [-0.712, -0.51]
    assert spec["box_xy"] == [1.356, -0.581]
    assert all(value == 0.0 for value in spec["flap_qpos"].values())
    assert np.isclose(np.linalg.norm(spec["box_quat"]), 1.0)
    assert spec["hold_seconds"] == 1.0


def test_stages_are_ordered_and_final_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan226"
    task.control_freq = 20
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert [stage.name for stage in task._atomic_tasks] == [
        "take_canned_drink_from_table",
        "put_canned_drink_into_box_on_armchair",
    ]
    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)


def test_first_stage_finishes_only_after_can_leaves_table():
    task = object.__new__(Task)
    task._can_on_table = lambda: True
    assert not task._can_removed_from_table()
    task._can_on_table = lambda: False
    assert task._can_removed_from_table()


def test_final_stage_requires_box_support_containment_and_release():
    task = object.__new__(Task)
    state = {"inside": True, "armchair": True, "released": True}
    task._box_contacts_armchair = lambda: state["armchair"]
    task._can_inside_box = lambda: state["inside"]
    task.object_released = lambda name: state["released"] and name == task.CAN_NAME
    assert task._can_in_box_on_armchair()

    state["inside"] = False
    assert not task._can_in_box_on_armchair()
    state.update(inside=True, armchair=False)
    assert not task._can_in_box_on_armchair()
    state.update(armchair=True, released=False)
    assert not task._can_in_box_on_armchair()


def test_no_visual_success_region():
    assert object.__new__(Task)._success_region_bounds() == []
