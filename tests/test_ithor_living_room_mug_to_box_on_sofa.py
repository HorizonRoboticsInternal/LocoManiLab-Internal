"""Contracts for carrying the Scene 201 mug into a sofa-mounted box."""

import math

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomMugFromCoffeeTableToBoxOnSofa,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robocasa.environments.ithor_living_room.tasks.composite import __all__
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


Task = IThorLivingRoomMugFromCoffeeTableToBoxOnSofa


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (201,)
    assert Task.BOX_SCALE == 1.2
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_LIVING_ROOM_ENVS[Task.__name__] is Task


def test_layout_starts_robot_at_coffee_table_and_box_on_sofa():
    spec = get_ithor_livingroom_task_spec(
        "FloorPlan201", "mug_from_coffee_table_to_box_on_sofa"
    )
    assert spec["robot_pose"]["pos"] == [-2.85, 4.79, 0.05]
    assert spec["robot_pose"]["yaw"] == -0.007386
    assert spec["mug_xy"] == [-2.448, 4.787]
    assert spec["box_xy"] == [-2.4, 3.65]
    assert spec["sofa_surface_z"] == 0.553
    assert spec["box_inner_half_xy"] == [0.114, 0.138]
    assert spec["hold_seconds"] == 1.0


def test_stages_are_ordered_and_final_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan201"
    task.control_freq = 20
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert [stage.name for stage in task._atomic_tasks] == [
        "take_mug_from_coffee_table",
        "put_mug_in_box_on_sofa",
    ]
    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)


def test_box_predicate_requires_the_whole_released_mug_inside():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan201"
    inside = np.array(
        [
            [-0.08, -0.08, 0.03],
            [0.08, 0.08, 0.16],
        ]
    )
    task._mug_points_in_box_frame = lambda: inside
    task.object_released = lambda name: name == Task.MUG_NAME
    assert task._mug_in_box()

    task.object_released = lambda name: False
    assert not task._mug_in_box()

    task.object_released = lambda name: True
    task._mug_points_in_box_frame = lambda: inside + [0.2, 0.0, 0.0]
    assert not task._mug_in_box()

    task._mug_points_in_box_frame = lambda: inside + [0.0, 0.0, 0.1]
    assert not task._mug_in_box()


def test_first_stage_finishes_only_after_mug_leaves_table():
    task = object.__new__(Task)
    task._mug_on_coffee_table = lambda: True
    assert not task._mug_removed_from_coffee_table()

    task._mug_on_coffee_table = lambda: False
    assert task._mug_removed_from_coffee_table()
