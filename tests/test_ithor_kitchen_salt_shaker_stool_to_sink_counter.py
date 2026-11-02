"""Focused registration, placement-region and hold checks for K06."""

import numpy as np
import pytest

from robocasa.environments.ithor_kitchen import (
    IThorKitchenSaltShakerStoolToSinkCounter as Task,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registered_scene25_only():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_KITCHEN_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (25,)


def test_whole_salt_shaker_must_be_on_counter_height():
    points = np.array([[-0.03, -0.03, 0.81], [0.03, 0.03, 1.01]], dtype=float)
    lower, upper = np.array([-0.5, -0.4]), np.array([0.5, 0.4])
    assert Task._on_counter(points, lower, upper, 0.812, 0.03)
    assert not Task._on_counter(points + [0.51, 0.0, 0.0], lower, upper, 0.812, 0.03)
    assert not Task._on_counter(points + [0.0, 0.0, 0.1], lower, upper, 0.812, 0.03)


@pytest.mark.parametrize("freq", [20, 30])
def test_continuous_one_second(freq):
    task = object.__new__(Task)
    task.control_freq = freq
    task._get_task_spec = lambda: {"hold_seconds": 1.0}
    task.timestep, task.horizon, task.ignore_done = 0, 1000, True
    task._raw_success = lambda: True
    task._success_hold_steps = 0
    for _ in range(freq - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    task._raw_success = lambda: False
    task._post_action(None)
    assert not task._check_success()
    assert task._success_hold_steps == 0
