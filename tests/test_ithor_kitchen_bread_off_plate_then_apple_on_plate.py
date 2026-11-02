"""Contracts for the Scene 3 bread-then-apple composite task."""

import math
from pathlib import Path

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen import (
    IThorKitchenBreadOffPlateThenAppleOnPlate,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.composite import __all__
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


Task = IThorKitchenBreadOffPlateThenAppleOnPlate


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (3,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_KITCHEN_ENVS[Task.__name__] is Task


def test_task_uses_packaged_apple_asset():
    task = object.__new__(Task)
    path = Path(task._get_obj_cfgs()[0]["obj_groups"])
    assert path.parts[-6:] == (
        "environments",
        "ithor_kitchen",
        "tasks",
        "assets",
        "Apple_25",
        "model.xml",
    )
    assert path.is_file()


def test_calibration_places_apple_left_and_covers_right_counter():
    spec = get_ithor_kitchen_task_spec(
        "FloorPlan3", "bread_off_plate_then_apple_on_plate"
    )
    plate_xy = np.asarray(spec["plate_xy"])
    apple_xy = np.asarray(spec["apple_xy"])
    bread_target = spec["bread_target"]
    bread_xy = np.asarray(bread_target["center"][:2])
    bread_half = np.asarray(bread_target["half_size"][:2])

    # The apple sits closer to the robot and offset toward the robot's left hand.
    assert apple_xy[0] > plate_xy[0] + 0.1
    assert apple_xy[1] < plate_xy[1] - 0.1
    assert bread_xy[1] > plate_xy[1] + 0.3
    assert np.allclose(bread_xy - bread_half, [-2.314, -0.221])
    assert np.allclose(bread_xy + bread_half, [-1.329, 2.397])
    assert spec["bread_support_z_tolerance"] == 0.015
    assert spec["apple_plate_xy_tolerance"] == 0.14
    assert spec["hold_seconds"] == 1.0


def test_volume_requires_the_whole_object():
    volume = {
        "center": [0.0, 0.0, 0.0],
        "half_size": [0.2, 0.2, 0.2],
        "tolerance": [0.01, 0.01, 0.01],
    }
    points = np.array([[-0.2, -0.1, -0.1], [0.2, 0.1, 0.1]])
    assert Task._points_in_volume(points, volume)
    assert not Task._points_in_volume(points + [0.02, 0.0, 0.0], volume)


def test_final_stage_requires_apple_on_plate_and_bread_retained():
    task = object.__new__(Task)
    state = {"apple": True, "bread": True}
    task._apple_on_plate = lambda: state["apple"]
    task._bread_on_right_counter = lambda: state["bread"]

    assert task._apple_on_plate_with_bread_retained()
    state["apple"] = False
    assert not task._apple_on_plate_with_bread_retained()
    state["apple"] = True
    state["bread"] = False
    assert not task._apple_on_plate_with_bread_retained()


def test_final_stage_holds_for_one_second_at_control_frequency():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan3"
    task.control_freq = 20
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)
