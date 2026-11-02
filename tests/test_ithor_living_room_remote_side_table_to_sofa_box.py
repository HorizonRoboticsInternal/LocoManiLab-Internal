"""Contracts for the Scene 205 remote-to-sofa-box composite task."""

import math

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomRemoteFromSideTableIntoBoxOnSofa,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robocasa.environments.ithor_living_room.tasks.composite import __all__
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


Task = IThorLivingRoomRemoteFromSideTableIntoBoxOnSofa


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (205,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_LIVING_ROOM_ENVS[Task.__name__] is Task


def test_layout_starts_with_upright_remote_and_open_sofa_box():
    spec = get_ithor_livingroom_task_spec(
        "FloorPlan205", "remote_from_side_table_into_box_on_sofa"
    )
    assert spec["robot_pose"] == {"pos": [-2.07, 5.95, 0.05], "yaw": 1.5708}
    assert spec["remote_xy"] == [-2.071, 6.699]
    assert spec["remote_quat"] == [1.0, 0.0, 0.0, 0.0]
    assert spec["box_pos"] == [-4.2, 5.1, 0.60486]
    assert all(value == 0.0 for value in spec["flap_qpos"].values())
    assert spec["horizontal_tolerance"] == 0.06
    assert spec["height_tolerance"] == 0.08
    assert spec["hold_seconds"] == 1.0


def test_stages_are_ordered_and_final_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan205"
    task.control_freq = 20
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert [stage.name for stage in task._atomic_tasks] == [
        "take_remote_from_side_table",
        "put_remote_into_box_on_sofa",
    ]
    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)


def test_first_stage_finishes_only_after_remote_leaves_side_table():
    task = object.__new__(Task)
    task._remote_on_side_table = lambda: True
    assert not task._remote_removed_from_side_table()
    task._remote_on_side_table = lambda: False
    assert task._remote_removed_from_side_table()


def test_expanded_box_accepts_upright_remote_above_rim():
    # The remote's lower end is inside the box and its center is within the
    # expanded horizontal footprint; its top may remain well above the rim.
    upright_remote = np.array(
        [
            [-0.02, -0.01, 0.01],
            [0.02, 0.01, 0.24],
        ]
    )
    assert Task._remote_within_expanded_box(
        upright_remote,
        lower=np.array([-0.10, -0.08]),
        upper=np.array([0.10, 0.08]),
        base_top=0.0,
        rim_top=0.10,
        horizontal_tolerance=0.06,
        height_tolerance=0.08,
    )

    outside = upright_remote + np.array([0.20, 0.0, 0.0])
    assert not Task._remote_within_expanded_box(
        outside,
        lower=np.array([-0.10, -0.08]),
        upper=np.array([0.10, 0.08]),
        base_top=0.0,
        rim_top=0.10,
        horizontal_tolerance=0.06,
        height_tolerance=0.08,
    )


def test_final_stage_requires_remote_in_box_and_box_on_sofa():
    task = object.__new__(Task)
    state = {"inside": True, "sofa": True}
    task._remote_inside_box = lambda: state["inside"]
    task._box_contacts_sofa = lambda: state["sofa"]
    assert task._remote_in_box_on_sofa()
    state["inside"] = False
    assert not task._remote_in_box_on_sofa()
    state.update(inside=True, sofa=False)
    assert not task._remote_in_box_on_sofa()


def test_box_quaternion_is_normalized_and_no_visual_success_region():
    spec = get_ithor_livingroom_task_spec(
        "FloorPlan205", "remote_from_side_table_into_box_on_sofa"
    )
    assert np.isclose(np.linalg.norm(spec["box_quat"]), 1.0)
    assert object.__new__(Task)._success_region_bounds() == []
