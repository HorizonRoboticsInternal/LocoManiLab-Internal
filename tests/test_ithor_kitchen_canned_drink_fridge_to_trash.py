"""Contracts for Scene 25's canned-drink-from-fridge-to-trash task."""

import math
from pathlib import Path
from types import SimpleNamespace
from xml.etree import ElementTree

import numpy as np

from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen import (
    IThorKitchenCannedDrinkFromFridgeToTrash,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.composite import __all__
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


Task = IThorKitchenCannedDrinkFromFridgeToTrash


def test_task_is_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (25,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_KITCHEN_ENVS[Task.__name__] is Task


def test_scene_25_layout_contract():
    spec = get_ithor_kitchen_task_spec(
        "FloorPlan25", "canned_drink_from_fridge_to_trash"
    )
    assert spec["target_fixture"] == "FloorPlan25:Fridge_25_16"
    assert spec["robot_pose"] == {
        "pos": [-1.85, 2.50, 0.0],
        "yaw": -2.74,
    }
    assert spec["init_open_fraction"] == 1.0
    assert spec["can_xy"] == [-2.50, 2.22]
    assert spec["fridge_receptacle_region"].endswith(":receptacle:0")
    assert spec["trash_fixture"] == "FloorPlan25:bin_25_13"
    assert spec["trash_region"].endswith(":receptacle:0")
    assert spec["stool_xy"] == [-1.46, 0.92]


def test_task_has_two_ordered_actions_without_door_or_stool_stage():
    task = object.__new__(Task)
    stages = task._normalize_atomic_tasks(task._get_atomic_tasks())
    assert [stage.name for stage in stages] == [
        "take_canned_drink_from_fridge",
        "put_canned_drink_in_trash",
    ]


def test_task_reuses_verified_canned_drink_asset():
    task = object.__new__(Task)
    path = Path(task._get_obj_cfgs()[0]["obj_groups"])
    assert path.parts[-7:] == (
        "models",
        "assets",
        "objects",
        "objaverse",
        "can",
        "can_0",
        "model.xml",
    )
    assert path.is_file()
    referenced_files = [
        element.get("file")
        for element in ElementTree.parse(path).getroot().findall(".//*[@file]")
    ]
    assert referenced_files
    assert all(
        (path.parent / relative_path).is_file() for relative_path in referenced_files
    )


def test_canned_drink_fridge_stage_uses_center_instead_of_full_mesh_or_contact():
    task = object.__new__(Task)
    task.sim = None
    task.fridge_receptacle_region = SimpleNamespace(
        world_pose=lambda sim: (np.array([1.0, 2.0, 0.50]), 0.0),
        half_size=np.array([0.20, 0.30, 0.15]),
    )
    position = np.array([1.18, 2.0, 0.50])
    task.object_pos = lambda name: position

    assert task._canned_drink_in_fridge()
    position[0] = 1.25
    assert not task._canned_drink_in_fridge()


def test_only_final_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan25"
    task.control_freq = 20
    task.task_spec = {
        "hold_seconds": 1.0,
    }
    task._multi_step_spec_key = (task.scene_name, task.SPEC_TASK)
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())
    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)


def test_trash_success_uses_center_and_requires_release():
    task = object.__new__(Task)
    task.sim = None
    task.trash_region = SimpleNamespace(
        world_pose=lambda sim: (np.array([1.0, 2.0, 0.30]), 0.0),
        half_size=np.array([0.13, 0.14, 0.24]),
    )
    position = np.array([1.0, 2.0, 0.10])
    state = {"grasped": False}
    task.object_pos = lambda name: position
    task.g1_hand_grasping_object = lambda name: state["grasped"]
    assert task._canned_drink_in_trash()
    state["grasped"] = True
    assert not task._canned_drink_in_trash()
    state["grasped"] = False
    position[0] = 1.18
    assert not task._canned_drink_in_trash()
    position[:] = [1.0, 2.0, 0.60]
    assert not task._canned_drink_in_trash()
