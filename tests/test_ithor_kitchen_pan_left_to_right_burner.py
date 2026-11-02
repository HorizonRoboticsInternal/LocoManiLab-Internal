"""Target-burner geometry, registry and continuous hold checks."""
import numpy as np
import pytest
from robocasa.environments.ithor_kitchen import (
    IThorKitchenPanLeftToRightBurner as Task,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registered_scene8_only():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_KITCHEN_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (8,)


def test_overlap_uses_fraction_of_pan_area():
    pan = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    assert Task._overlap_ratio(pan, [0, 0], [1, 1]) == pytest.approx(1.0)
    assert Task._overlap_ratio(pan, [0.3, 0], [1.3, 1]) == pytest.approx(0.7)
    assert Task._overlap_ratio(pan, [0.31, 0], [1.31, 1]) == pytest.approx(0.69)
    assert Task._overlap_ratio(pan, [2, 0], [3, 1]) == 0.0
    # A larger burner containing the whole pan must count as 100%, not IoU.
    assert Task._overlap_ratio(pan, [-2, -2], [2, 2]) == pytest.approx(1.0)
    # Rotated outline: its bounding box overestimates overlap in this corner.
    diamond = np.array([[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    assert Task._overlap_ratio(diamond, [0.5, 0.5], [1.0, 1.0]) == pytest.approx(0.0)


def test_raw_success_threshold_inclusive():
    task = object.__new__(Task)
    task._get_task_spec = lambda: {"min_overlap_ratio": 0.7}
    task._pan_points = lambda: np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    )
    task._target_bounds_xy = lambda: (np.array([0.3, 0]), np.array([1.3, 1]))
    assert task._raw_success()
    task._target_bounds_xy = lambda: (np.array([0.31, 0]), np.array([1.31, 1]))
    assert not task._raw_success()


@pytest.mark.parametrize("freq", [20, 30])
def test_continuous_one_second(freq):
    task = object.__new__(Task)
    task.control_freq = freq
    task._get_task_spec = lambda: {"hold_seconds": 1.0}
    task.timestep, task.horizon, task.ignore_done = 0, 1000, True
    task._raw_success = lambda: True
    task._reset_task_progress()
    for _ in range(freq - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    task._raw_success = lambda: False
    task._post_action(None)
    assert not task._check_success() and task._success_hold_steps == 0
    task._raw_success = lambda: True
    task._post_action(None)
    assert not task._check_success()
