"""The close-safe composite must spawn clear of the bed during calibration."""

import json

import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_bedroom.tasks import (
    IThorBedroomApproachStorageAndCloseSafeDoor,
    IThorBedroomApproachStorageAndOpenSafeDoor,
    IThorBedroomCloseSafeDoor,
    IThorBedroomNavigateToStorage,
    IThorBedroomOpenSafeDoor,
)
from validate_ithor_tasks import _load_reset_controller


Task = IThorBedroomApproachStorageAndCloseSafeDoor


def test_close_safe_only_overrides_its_navigation_spawn_and_obstruction():
    spec = Task.SCENE_SPECS["FloorPlan302"]
    atomic_operation = IThorBedroomCloseSafeDoor.SCENE_SPECS["FloorPlan302"]
    assert {k: v for k, v in spec.items() if k not in {"lang", "navigation"}} == {
        k: v for k, v in atomic_operation.items() if k != "lang"
    }
    atomic_navigation = IThorBedroomNavigateToStorage.SCENE_SPECS["FloorPlan302"]
    assert spec["navigation"] == {
        **atomic_navigation,
        "start_pose": {"xy": [0.40, -0.80], "yaw": 2.018316},
        "minimum_start_distance": 0.60,
        "remove_fixtures": ["FloorPlan302:Tennis_Racquet_1_13"],
    }
    assert spec["navigation"] is not atomic_navigation
    assert spec["navigation"]["start_pose"] is not atomic_navigation["start_pose"]
    assert atomic_navigation["start_pose"]["xy"] == [-0.45, -0.60]
    assert atomic_navigation["minimum_start_distance"] == 1.50
    assert atomic_navigation["remove_fixtures"] == []
    assert (
        IThorBedroomApproachStorageAndOpenSafeDoor.SCENE_SPECS["FloorPlan302"][
            "navigation"
        ]["remove_fixtures"]
        == []
    )


@pytest.mark.parametrize(
    ("task", "atomic", "yaw", "minimum_start_distance"),
    (
        (
            IThorBedroomApproachStorageAndOpenSafeDoor,
            IThorBedroomOpenSafeDoor,
            2.400676,
            2.75,
        ),
        (
            IThorBedroomApproachStorageAndCloseSafeDoor,
            IThorBedroomCloseSafeDoor,
            1.368820,
            3.70,
        ),
    ),
)
def test_safe_composites_323_approach_the_relocated_safe(
    task, atomic, yaw, minimum_start_distance
):
    spec = task.SCENE_SPECS["FloorPlan323"]
    atomic_operation = atomic.SCENE_SPECS["FloorPlan323"]
    generic_storage = IThorBedroomNavigateToStorage.SCENE_SPECS["FloorPlan323"]

    assert {k: v for k, v in spec.items() if k not in {"lang", "navigation"}} == {
        k: v for k, v in atomic_operation.items() if k != "lang"
    }
    assert spec["navigation"] is not generic_storage
    assert spec["navigation"]["fixture"] == atomic_operation["fixture"]
    assert spec["navigation"]["fixture"] != generic_storage["fixture"]
    assert spec["navigation"]["start_pose"]["xy"] == [0.30, -2.85]
    assert spec["navigation"]["start_pose"]["yaw"] == pytest.approx(yaw)
    assert spec["navigation"]["minimum_start_distance"] == pytest.approx(
        minimum_start_distance
    )
    assert spec["navigation"]["remove_fixtures"] == []
    assert spec["navigation"]["lang"] == "walk to the safe"


def test_close_safe_302_reset_and_navigation_then_close():
    env = robosuite.make(
        env_name=Task.__name__,
        robots="G1",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        control_freq=20,
        renderer="mujoco",
        scene_ids=[302],
        seed=0,
    )
    try:
        env.reset()
        data, model = env.sim.data, env.sim.model
        joint = env.target_semantic_joint
        np.testing.assert_allclose(env._robot_base_xy(), [0.40, -0.80])
        assert env._robot_base_yaw() == pytest.approx(2.018316)
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._approach_start_distance >= 0.60
        assert joint.fraction(env.sim) == pytest.approx(0.80)
        assert env._initial_articulation_intact()
        assert not env._near_articulation()
        assert not env._check_success()
        assert env.target_fixture.stable_id == "FloorPlan302:Safe_18_1_16"
        np.testing.assert_allclose(
            env.target_fixture.world_center(env.sim)[:2], [-0.20, 0.45]
        )
        assert env.removed_approach_fixture_ids == ["FloorPlan302:Tennis_Racquet_1_13"]
        racquet_prefix = "tennisracket_e0826620cab4913787acdae1480d265b_1_"
        assert not any(name.startswith(racquet_prefix) for name in model.body_names)
        assert not any(name.startswith(racquet_prefix) for name in model.geom_names)

        robot_joint = env._get_robot_free_joint()
        robot_joint_id = model.joint_name2id(robot_joint)
        robot_dof = int(model.jnt_dofadr[robot_joint_id])
        np.testing.assert_allclose(data.qvel[robot_dof : robot_dof + 6], 0.0)
        valid_pose = data.get_joint_qpos(robot_joint).copy()
        bed = next(
            f
            for f in env.semantic_scene.fixtures.values()
            if f.category.lower() == "bed"
        )
        bed_clearance = env.robot_fixture_collision_distance(bed, max_distance=1.0)
        assert bed_clearance > 0.0

        assert env._finalize_supported_articulation_robot_pose(0.45)
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._near_articulation()
        zero_action = np.zeros(env.action_dim)

        def set_fraction(fraction):
            data.set_joint_qpos(
                joint.name,
                joint.closed_qpos + fraction * (joint.open_qpos - joint.closed_qpos),
            )
            env.sim.forward()

        # Predicate checks use explicit joint states, not a teleop rollout.
        set_fraction(0.10)
        for _ in range(env.HOLD_STEPS):
            env._post_action(zero_action)
        assert env._approach_stage == 0
        assert not env._check_success()
        set_fraction(0.80)
        for step in range(env.APPROACH_HOLD_STEPS):
            env._post_action(zero_action)
            assert env._approach_stage == (0 if step < 4 else 1)
        set_fraction(0.10)
        assert env.HOLD_STEPS == 20
        for step in range(env.HOLD_STEPS):
            env._post_action(zero_action)
            assert env._check_success() == (step == env.HOLD_STEPS - 1)
        assert env.count_body_collisions() == 0
        print(
            json.dumps(
                {
                    "scene": 302,
                    "spawn_xy": valid_pose[:2].tolist(),
                    "spawn_free_joint_z": float(valid_pose[2]),
                    "spawn_bed_clearance": float(bed_clearance),
                    "spawn_safe_distance": float(env._approach_start_distance),
                    "success_hold_steps": env._operation_hold,
                    "success": env._check_success(),
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()
