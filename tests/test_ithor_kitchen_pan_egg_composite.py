"""Contracts for the Scene 8 pan-then-egg composite task."""

import math

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen import (
    IThorKitchenPanToRightBurnerThenEggIntoPan,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.pan_left_to_right_burner import (
    IThorKitchenPanLeftToRightBurner,
)
from robocasa.environments.ithor_kitchen.tasks.composite import __all__
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


Task = IThorKitchenPanToRightBurnerThenEggIntoPan


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (8,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_KITCHEN_ENVS[Task.__name__] is Task


def test_calibration_places_egg_at_front_right_of_right_burner():
    spec = get_ithor_kitchen_task_spec(
        "FloorPlan8", "pan_to_right_burner_then_egg_into_pan"
    )
    egg_xy = np.asarray(spec["egg_xy"])
    right_burner_center = np.array([-1.434, -1.526])

    assert egg_xy[0] < right_burner_center[0]
    assert egg_xy[1] > right_burner_center[1]
    assert np.linalg.norm(egg_xy - right_burner_center) < 0.35
    assert spec["robot_pose"]["pos"][:2] == [-1.32, -1.05]
    assert spec["min_overlap_ratio"] == 0.70
    assert spec["hold_seconds"] == 1.0


def test_pan_stage_reuses_the_seventy_percent_footprint_rule():
    task = object.__new__(Task)
    task._pan_geom_ids = ("pan",)
    task._geom_points = lambda _ids: np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    task._target_bounds_xy = lambda: (np.array([0.3, 0.0]), np.array([1.3, 1.0]))
    task._get_task_spec = lambda: {"min_overlap_ratio": 0.70}

    assert task._pan_on_target_burner()
    task._target_bounds_xy = lambda: (
        np.array([0.31, 0.0]),
        np.array([1.31, 1.0]),
    )
    assert not task._pan_on_target_burner()
    assert np.isclose(
        IThorKitchenPanLeftToRightBurner._overlap_ratio(
            np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
            [0.3, 0],
            [1.3, 1],
        ),
        0.7,
    )


def test_final_stage_requires_fresh_egg_placement_and_retains_pan():
    task = object.__new__(Task)
    state = {"egg": True, "pan": True}
    task._egg_stage_armed = False
    task._egg_in_pan = lambda: state["egg"]
    task._pan_on_target_burner = lambda: state["pan"]

    assert not task._egg_in_pan_with_pan_retained()
    state["egg"] = False
    assert not task._egg_in_pan_with_pan_retained()
    state["egg"] = True
    assert task._egg_in_pan_with_pan_retained()
    state["pan"] = False
    assert not task._egg_in_pan_with_pan_retained()


def test_final_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.scene_name = "FloorPlan8"
    task.control_freq = 20
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)
