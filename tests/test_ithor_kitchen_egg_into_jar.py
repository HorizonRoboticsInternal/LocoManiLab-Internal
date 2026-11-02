"""Fast geometric and timing checks; no scene instances are created."""

import math

import numpy as np
import pytest

from robocasa.environments.ithor_kitchen import (
    IThorKitchenEggIntoJar,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robosuite.environments.base import REGISTERED_ENVS


def test_registered_once():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_KITCHEN_ENVS):
        assert registry["IThorKitchenEggIntoJar"] is IThorKitchenEggIntoJar
    assert IThorKitchenEggIntoJar.SUPPORTED_SCENE_IDS == (3,)


def test_whole_shape_inside_and_rim_floor_rejection():
    contains = IThorKitchenEggIntoJar._contained
    half = np.array([0.055, 0.08, 0.09])
    points = np.array([[-0.035, -0.03, -0.04], [0.035, 0.03, 0.04]])
    assert contains(points, np.zeros(3), np.eye(3), half)
    for offset in ([0.08, 0, 0], [0, -0.06, 0], [0, 0.06, 0]):
        assert not contains(points + offset, np.zeros(3), np.eye(3), half)
    # The volume follows a translated/rotated pot, rather than world axes.
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    center = np.array([1.0, 2.0, 3.0])
    assert contains(points @ rotation.T + center, center, rotation, half)


@pytest.mark.parametrize("control_freq", [20, 30, 19.5])
def test_continuous_one_second_and_interruption(control_freq):
    task = object.__new__(IThorKitchenEggIntoJar)
    task.control_freq = control_freq
    task.timestep = 0
    task.horizon = 1000
    task.ignore_done = True
    task._get_task_spec = lambda: {"hold_seconds": 1.0}
    task._raw_success = lambda: True
    task._reset_task_progress()
    required = math.ceil(control_freq)
    for _ in range(required - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    # Reading success repeatedly must not advance the timer.
    assert task._check_success() and task._success_hold_steps == required
    task._raw_success = lambda: False
    task._post_action(None)
    assert task._success_hold_steps == 0 and not task._check_success()


def test_off_center_egg_uses_union_of_native_volumes():
    task = IThorKitchenEggIntoJar
    points = np.array([[-0.01, -0.03, -0.04], [0.06, 0.03, 0.04]])
    rotation = np.array([[0.0, 0.0, -1.0], [0.0, -1.0, 0.0], [-1.0, 0.0, 0.0]])
    volumes = [
        (np.zeros(3), np.eye(3), np.array([0.055, 0.08, 0.09])),
        (np.zeros(3), rotation, np.array([0.055, 0.08, 0.09])),
    ]
    assert not task._contained(points, *volumes[0])
    assert task._inside_receptacle(points, volumes)
    assert not task._inside_receptacle(points + [0.2, 0, 0], volumes)
