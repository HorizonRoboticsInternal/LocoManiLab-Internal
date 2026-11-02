"""Contracts for the Scene 13 canned-drink drawer composite task."""

import math

from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen import (
    IThorKitchenOpenDrawerThenPutCannedDrinkInside,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.composite import __all__
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


Task = IThorKitchenOpenDrawerThenPutCannedDrinkInside


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (13,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_KITCHEN_ENVS[Task.__name__] is Task


def test_calibration_uses_scene_13_counter_and_drawer():
    spec = get_ithor_kitchen_task_spec("FloorPlan13", "canned_drink_drawer_cycle")

    assert spec["target_fixture"] == "FloorPlan13:Drawer_7f6923d8_17"
    assert spec["drawer_receptacle_region"].endswith(":receptacle:0")
    assert spec["init_open_fraction"] == 0.0
    assert spec["success_open_fraction"] == 0.80
    assert spec["success_close_fraction"] == 0.20
    assert spec["can_pos"][:2] == [-3.35, 5.15]
    assert spec["support_region"]["body_name"].startswith("cube_28bb627")
    assert spec["can_quat"] == [1.0, 0.0, 0.0, 0.0]


def test_task_has_the_required_ordered_stages():
    assert Task.COMPOSITE_STAGES == (
        "open_drawer",
        "put_canned_drink_in_drawer",
        "close_drawer",
    )


def test_final_state_requires_open_stage_and_can_inside():
    task = object.__new__(Task)
    task._drawer_opened = False
    task._canned_drink_placed = False
    task._canned_drink_inside_drawer = lambda: True
    task._get_task_spec = lambda: {"success_close_fraction": 0.20}
    task.target_semantic_joint = type(
        "Joint", (), {"fraction": lambda self, sim: 0.0}
    )()
    task.sim = object()
    assert not task._task_success_condition()
    task._drawer_opened = True
    assert not task._task_success_condition()
    task._canned_drink_placed = True
    assert task._task_success_condition()
    task._canned_drink_inside_drawer = lambda: False
    assert not task._task_success_condition()


def test_final_state_holds_for_one_second():
    task = object.__new__(Task)
    task.success_hold_seconds = 1.0
    task.control_freq = 20
    assert task.required_success_hold_steps == math.ceil(20)
