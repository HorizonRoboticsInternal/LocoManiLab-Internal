"""Contracts for carrying the Scene 201 mug from sofa to coffee table."""

import math

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_atomic.task_assets import MUG_MODEL, MUG_SCALE
from robocasa.environments.ithor_living_room import (
    IThorLivingRoomMugFromSofaToCoffeeTable,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robocasa.environments.ithor_living_room.tasks.composite import __all__
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


Task = IThorLivingRoomMugFromSofaToCoffeeTable


def test_task_is_public_registered_scene_scoped_and_uses_canonical_mug():
    assert Task.SUPPORTED_SCENE_IDS == (201,)
    assert Task.MUG_XML == MUG_MODEL
    assert Task.MUG_SCALE == MUG_SCALE
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_LIVING_ROOM_ENVS[Task.__name__] is Task


def test_layout_starts_with_mug_on_sofa_and_robot_in_front():
    spec = get_ithor_livingroom_task_spec(
        "FloorPlan201", "mug_from_sofa_to_coffee_table"
    )
    assert spec["robot_pose"]["pos"] == [-2.4, 4.25, 0.05]
    assert spec["robot_pose"]["yaw"] == -1.5708
    assert spec["mug_xy"] == [-2.6, 3.65]
    assert spec["mug_yaw"] == 0.0
    assert spec["sofa_surface_z"] == 0.568
    assert spec["hold_seconds"] == 1.0


def test_stages_are_ordered_and_final_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan201"
    task.control_freq = 20
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert [stage.name for stage in task._atomic_tasks] == [
        "take_mug_from_sofa",
        "put_mug_on_coffee_table",
    ]
    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)


def test_final_stage_requires_table_support_stability_and_release():
    task = object.__new__(Task)
    state = {"table": True, "stable": True, "released": True}
    task._mug_on_coffee_table = lambda: state["table"]
    task._mug_is_stable = lambda: state["stable"]
    task.object_released = lambda name: name == Task.MUG_NAME and state["released"]

    assert task._mug_set_down_on_coffee_table()
    for key in state:
        state[key] = False
        assert not task._mug_set_down_on_coffee_table()
        state[key] = True


def test_stability_uses_configured_linear_and_angular_limits():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan201"
    task._mug_joint = "mug_joint"
    velocity = np.array([0.08, 0.0, 0.0, 0.0, 0.0, 0.40])
    task.sim = type(
        "Sim",
        (),
        {
            "model": type("Model", (), {"get_joint_qvel_addr": lambda *_: (0, 6)})(),
            "data": type("Data", (), {"qvel": velocity})(),
        },
    )()
    assert task._mug_is_stable()

    velocity[0] = 0.11
    assert not task._mug_is_stable()
