"""Contracts for the Scene 3 cup-from-stool-to-sink composite task."""

import math

import numpy as np
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen import IThorKitchenCupFromStoolToSink
from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.composite import __all__
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


Task = IThorKitchenCupFromStoolToSink


def test_task_is_public_registered_and_scene_scoped():
    assert Task.SUPPORTED_SCENE_IDS == (3,)
    assert Task.__name__ in __all__
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert REGISTERED_ITHOR_KITCHEN_ENVS[Task.__name__] is Task


def test_layout_and_sink_volume_are_configured():
    spec = get_ithor_kitchen_task_spec("FloorPlan3", "cup_from_stool_to_sink")
    assert spec["robot_pose"]["pos"][:2] == [-0.35, -1.0889]
    assert spec["robot_pose"]["yaw"] == 0.0
    assert spec["stool_xy"] == [0.25, -1.0889]
    assert "lift_height" not in spec
    assert spec["sink_volume"]["center"] == [-1.825, -0.534, 1.18]
    assert spec["sink_volume"]["half_size"] == [0.34, 0.28, 0.22]
    assert spec["hold_seconds"] == 1.0


def test_ordered_stages_take_then_sink():
    task = object.__new__(Task)
    stages = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert [stage.name for stage in stages] == [
        "take_cup_from_stool",
        "put_cup_in_sink",
    ]
    assert stages[0].task_cls is None
    assert stages[1].task_cls == "IThorKitchenCupToSink"


def test_stool_removal_and_sink_predicates():
    task = object.__new__(Task)
    state = {
        "cup": np.array([[0.20, -1.14, 0.70], [0.30, -1.04, 0.82]]),
        "sink_center": np.array([-1.825, -0.534, 1.18]),
    }
    task._cup_geom_ids = ("cup",)
    task._seat_geom_id = "seat"
    task._geom_points = lambda _ids: state["cup"]
    task._single_geom_points = lambda _geom: np.array(
        [[0.10, -1.20, 0.65], [0.40, -0.90, 0.70]]
    )
    task._get_task_spec = lambda: {
        "sink_volume": {
            "center": [-1.825, -0.534, 1.18],
            "half_size": [0.34, 0.28, 0.22],
            "tolerance": [0.02, 0.02, 0.02],
        },
    }

    assert task._cup_on_stool()
    assert not task._cup_removed_from_stool()
    state["cup"] = state["cup"] + np.array([0.50, 0.0, 0.0])
    assert task._cup_removed_from_stool()
    state["cup"] = np.vstack((state["sink_center"] - 0.02, state["sink_center"] + 0.02))
    assert task._cup_in_sink()
    state["cup"] = state["cup"] + np.array([0.60, 0.0, 0.0])
    assert not task._cup_in_sink()


def test_sink_stage_holds_for_one_second():
    task = object.__new__(Task)
    task.control_freq = 20
    task._get_task_spec = lambda: {"hold_seconds": 1.0}
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())

    assert task._required_stage_hold_steps(task._atomic_tasks[0]) == 5
    assert task._required_stage_hold_steps(task._atomic_tasks[1]) == math.ceil(20)
