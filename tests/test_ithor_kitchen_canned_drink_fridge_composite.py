"""Contracts for the Scene 8 canned-drink-to-fridge composite task."""

import math

from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen import (
    IThorKitchenCannedDrinkIntoFridgeThenCloseDoor,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.composite import __all__
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


Task = IThorKitchenCannedDrinkIntoFridgeThenCloseDoor


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (8,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_KITCHEN_ENVS[Task.__name__] is Task


def test_calibration_targets_the_open_lower_fridge_door():
    spec = get_ithor_kitchen_task_spec(
        "FloorPlan8", "canned_drink_into_fridge_then_close_door"
    )

    assert spec["robot_pose"]["pos"][:2] == [-1.68, -1.05]
    assert spec["can_xy"] == [-1.68, -1.39]
    assert spec["target_fixture"] == "FloorPlan8:Fridge_8_13"
    assert spec["target_joint"].endswith("_1_6_0_fridgefreezerdoor14b5bfef821_4")
    assert {
        region.rsplit(":", 1)[-1] for region in spec["fridge_receptacle_regions"]
    } == {"7", "8", "9", "10", "11", "12"}
    assert spec["init_open_fraction"] == 1.00
    assert spec["success_close_fraction"] == 0.10
    assert spec["can_region_center_tolerance"] == [0.06, 0.06, 0.06]


def test_task_has_the_required_ordered_stages():
    task = object.__new__(Task)
    names = [stage.name for stage in task._get_atomic_tasks()]
    assert names == [
        "place_canned_drink_in_fridge",
        "close_lower_fridge_door",
    ]


def test_place_and_close_stages_require_fresh_transitions_and_retention():
    task = object.__new__(Task)
    state = {"can": True, "open": True, "closed": True}

    task._canned_drink_placement_armed = False
    task._canned_drink_in_fridge = lambda: state["can"]
    task._door_open = lambda: state["open"]
    assert not task._fresh_canned_drink_placement()
    state["can"] = False
    assert not task._fresh_canned_drink_placement()
    state["can"] = True
    assert task._fresh_canned_drink_placement()

    task._door_close_armed = False
    task._door_closed = lambda: state["closed"]
    assert not task._closed_with_canned_drink_retained()
    state["closed"] = False
    assert not task._closed_with_canned_drink_retained()
    state["closed"] = True
    assert task._closed_with_canned_drink_retained()
    state["can"] = False
    assert not task._closed_with_canned_drink_retained()


def test_final_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan8"
    task.control_freq = 20
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 1
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)
