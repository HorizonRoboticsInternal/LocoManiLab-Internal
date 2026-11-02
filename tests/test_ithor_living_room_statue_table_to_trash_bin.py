"""Contracts for the Scene 209 statue-to-trash-bin composite task."""

import math

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomStatueFromTableToTrashBin,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robocasa.environments.ithor_living_room.tasks.composite import __all__
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


Task = IThorLivingRoomStatueFromTableToTrashBin


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (209,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_LIVING_ROOM_ENVS[Task.__name__] is Task


def test_layout_uses_stone_statue_on_bin_side_of_table():
    spec = get_ithor_livingroom_task_spec(
        "FloorPlan209", "statue_from_table_to_trash_bin"
    )
    assert Task.STATUE_XML.endswith("Stone_Statue_4/Stone_Statue_4.xml")
    assert Task.STATUE_SCALE == 0.5
    assert spec["statue_xy"] == [-2.25, -2.98]
    assert spec["robot_pose"] == {"pos": [-1.77, -3.74, 0.05], "yaw": 2.133}
    assert spec["bin_xy_half_tolerance"] == [0.04, 0.05]
    assert spec["bin_above_rim_tolerance"] == 0.10
    assert spec["hold_seconds"] == 1.0


def test_stages_are_ordered_and_final_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan209"
    task.control_freq = 20
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert [stage.name for stage in task._atomic_tasks] == [
        "take_statue_from_table",
        "put_statue_in_trash_bin",
    ]
    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)


def test_first_stage_finishes_only_after_statue_leaves_table():
    task = object.__new__(Task)
    task._statue_on_table = lambda: True
    assert not task._statue_removed_from_table()
    task._statue_on_table = lambda: False
    assert task._statue_removed_from_table()


def test_relaxed_bin_region_allows_statue_to_protrude_or_rest_on_rim():
    task = object.__new__(Task)
    task._statue_geom_ids = [0]
    task._bin_world_bounds = lambda: (
        np.array([-0.17, -0.12, 0.25]),
        np.array([0.17, 0.12, 0.63]),
    )
    task._get_task_spec = lambda: {
        "bin_xy_half_tolerance": [0.04, 0.05],
        "bin_above_rim_tolerance": 0.10,
        "bin_below_floor_tolerance": 0.05,
    }
    task.object_released = lambda name: name == task.STATUE_NAME
    center = np.array([0.20, 0.15, 0.70])
    task._geom_points = lambda *_: np.array([center - 0.05, center + 0.05])
    assert task._statue_in_relaxed_bin_region()

    center[:] = [0.22, 0.15, 0.70]
    assert not task._statue_in_relaxed_bin_region()
    center[:] = [0.20, 0.15, 0.74]
    assert not task._statue_in_relaxed_bin_region()

    center[:] = [0.20, 0.15, 0.70]
    task.object_released = lambda *_: False
    assert not task._statue_in_relaxed_bin_region()


def test_no_visual_success_region():
    assert object.__new__(Task)._success_region_bounds() == []
