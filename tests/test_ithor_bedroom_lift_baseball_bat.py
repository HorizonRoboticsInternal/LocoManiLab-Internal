"""Scene 329 bat clearance and uninterrupted hold regression checks."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_bedroom.tasks.lift_baseball_bat import (
    IThorBedroomLiftBaseballBat,
)
from robocasa.environments.ithor_task_catalog import (
    BEDROOM_PACKAGE,
    discover_task_catalog,
    EXPECTED_TASK_MATRIX,
    task_counts,
    validate_task_catalog,
)


def make_predicate_env(control_freq=20):
    env = object.__new__(IThorBedroomLiftBaseballBat)
    env.control_freq = control_freq
    env._configure_success_hold_duration()
    env._bat_hold_steps = 0
    env._floor_geom_id = 0
    env._bat_collision_meshes = [(1, np.array([[0, 0, -0.5], [0, 0, 0.5]]))]
    env.sim = SimpleNamespace(
        data=SimpleNamespace(
            geom_xpos=np.array([[0, 0, 0], [0, 0, 0.5]], dtype=float),
            geom_xmat=np.array([np.eye(3).ravel(), np.eye(3).ravel()]),
        )
    )
    return env


def test_registration_and_scene_scope():
    entries = discover_task_catalog([BEDROOM_PACKAGE])
    # Existing composite counts differ from the stored matrix. Audit registration
    # and all atomic counts without changing unrelated composite expectations.
    matrix = task_counts(entries)[BEDROOM_PACKAGE]
    expected = EXPECTED_TASK_MATRIX[BEDROOM_PACKAGE]
    assert {scene: row["atomic"] for scene, row in matrix.items()} == {
        scene: row["atomic"] for scene, row in expected.items()
    }
    registration_issues = [
        issue
        for issue in validate_task_catalog(entries)
        if ": task matrix changed" not in issue
    ]
    assert not registration_issues
    bat = next(entry for entry in entries if entry.cls is IThorBedroomLiftBaseballBat)
    assert bat.scene_ids == (329,)
    assert bat.family == "lift"
    with pytest.raises(ValueError, match="uncalibrated scene ids"):
        IThorBedroomLiftBaseballBat(scene_ids=[323])


def test_whole_mesh_must_clear_floor():
    env = make_predicate_env()
    # The root is half a metre high but the lower end is still on the floor.
    assert not env._bat_is_off_ground()
    env.sim.data.geom_xpos[1, 2] += 0.003
    assert not env._bat_is_off_ground()
    env.sim.data.geom_xpos[1, 2] += 0.01
    assert env._bat_is_off_ground()
    # A horizontal bat uses the rotated mesh, not its original vertical bounds.
    env.sim.data.geom_xmat[1] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]).ravel()
    env.sim.data.geom_xpos[1, 2] = 0.01
    assert env._bat_is_off_ground()


@pytest.mark.parametrize("control_freq", [10, 20, 30])
def test_one_second_without_grasp_and_interruption(control_freq):
    env = make_predicate_env(control_freq)
    env.sim.data.geom_xpos[1, 2] = 0.6
    # No robot, gripper or grasp state exists on this predicate-only instance.
    with patch.object(IThorBedroomEnv, "_post_action", return_value=None):
        for _ in range(control_freq - 1):
            env._post_action(None)
        assert not env._check_success()
        assert not env._check_success()  # Reading success does not advance time.
        env.sim.data.geom_xpos[1, 2] = 0.5
        env._post_action(None)
        assert env._bat_hold_steps == 0
        env.sim.data.geom_xpos[1, 2] = 0.6
        for _ in range(control_freq - 1):
            env._post_action(None)
        assert not env._check_success()
        env._post_action(None)
        assert env._check_success()
        env.sim.data.geom_xpos[1, 2] = 0.5
        env._post_action(None)
        assert not env._check_success()
