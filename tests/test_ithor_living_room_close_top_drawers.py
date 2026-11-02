"""Contracts for Scene 205's ordered top-drawer closing task."""

import math

from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomCloseTopRightThenTopLeftDrawer,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robocasa.environments.ithor_living_room.tasks.composite import __all__
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


Task = IThorLivingRoomCloseTopRightThenTopLeftDrawer


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (205,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_LIVING_ROOM_ENVS[Task.__name__] is Task


def test_layout_opens_only_the_two_top_drawers():
    spec = get_ithor_livingroom_task_spec(
        "FloorPlan205", "close_top_right_then_top_left_drawer"
    )
    assert spec["robot_pose"]["pos"] == [-2.07, 5.95, 0.05]
    assert spec["robot_pose"]["yaw"] == 2.40
    assert spec["init_open_fraction"] == 1.0
    assert spec["success_open_fraction"] == 0.8
    assert spec["success_close_fraction"] == 0.1
    assert len(Task.BOTTOM_DRAWER_JOINTS) == 2
    assert "_1_8_0_" in Task.TOP_RIGHT_DRAWER_JOINT
    assert "_1_2_0_" in Task.TOP_LEFT_DRAWER_JOINT


def test_stages_are_ordered_and_final_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan205"
    task.control_freq = 20
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())
    assert [stage.name for stage in task._atomic_tasks] == [
        "close_top_right_drawer",
        "walk_left_and_close_top_left_drawer",
    ]
    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)


def test_first_stage_requires_right_closed_while_left_stays_open():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan205"
    state = {"right": 0.1, "left": 1.0}
    task._top_right_drawer_fraction = lambda: state["right"]
    task._top_left_drawer_fraction = lambda: state["left"]
    assert task._right_closed_while_left_open()

    state["right"] = 0.2
    assert not task._right_closed_while_left_open()
    state["right"] = 0.1
    state["left"] = 0.7
    assert not task._right_closed_while_left_open()


def test_final_stage_requires_both_top_drawers_closed():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan205"
    state = {"right": 0.1, "left": 0.1}
    task._top_right_drawer_fraction = lambda: state["right"]
    task._top_left_drawer_fraction = lambda: state["left"]
    assert task._both_top_drawers_closed()

    state["right"] = 0.2
    assert not task._both_top_drawers_closed()
    state["right"] = 0.1
    state["left"] = 0.2
    assert not task._both_top_drawers_closed()


def test_no_visual_success_region():
    task = object.__new__(Task)
    assert task._success_region_bounds() == []
