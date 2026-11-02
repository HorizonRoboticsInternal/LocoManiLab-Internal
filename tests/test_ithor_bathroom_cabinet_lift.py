"""The new 405 cabinet lift is isolated and depends only on bottle height."""

import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_bathroom import tasks
from robocasa.environments.ithor_bathroom.tasks.cabinet_lift import (
    IThorBathroomLiftSoapBottleFromCabinet as Task,
)
from robocasa.environments.ithor_bathroom.tasks.object_manipulation import (
    _BathroomNativeObjectTask,
)
from robocasa.environments.ithor_task_catalog import (
    BATHROOM_PACKAGE,
    discover_task_catalog,
    validate_task_catalog,
)
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_spec,
)
from robocasa.utils import env_utils
from validate_ithor_tasks import (
    _assert_body_contact_state,
    _load_reset_controller,
    _settle_scene_with_frozen_robot,
    _task_body_contact_contracts,
)


def test_cabinet_task_registration_and_existing_lift_are_isolated():
    assert Task.SUPPORTED_SCENE_IDS == (405,)
    assert Task in tasks.ATOMIC_TASK_CLASSES
    assert Task.__name__ in tasks.__all__
    entries = discover_task_catalog((BATHROOM_PACKAGE,))
    assert validate_task_catalog(entries) == []
    spec = Task.SCENE_SPECS["FloorPlan405"]
    old_spec = get_ithor_bathroom_task_spec("FloorPlan405", "lift_soap_bottle")
    assert tasks.IThorBathroomLiftSoapBottle.SCENE_SPECS["FloorPlan405"] == old_spec
    assert old_spec["surface_region"].endswith(":receptacle:7")
    assert old_spec["source_xy"] == [-2.87, -0.444552]
    assert spec["source_fixture"] == old_spec["source_fixture"]
    assert spec["surface_fixture"] != old_spec["surface_fixture"]
    assert spec["remove_fixtures"] == []
    assert spec["cabinet_open_fraction"] == 1.0
    assert tasks.IThorBathroomLiftSoapBottle._raw_success is not Task._raw_success


@pytest.mark.parametrize("control_freq", (10, 20))
def test_height_only_success_requires_one_second_and_resets_on_drop(
    monkeypatch, control_freq
):
    task = object.__new__(Task)
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    task._initial_source_z = 0.56
    task._success_hold = 0
    height = [0.56]
    task._source_root_z = lambda: height[0]

    def unexpected_grasp(*args):
        raise AssertionError("height-only success must not inspect hand state")

    task._g1_hand_grasping_fixture = unexpected_grasp
    monkeypatch.setattr(
        _BathroomNativeObjectTask,
        "_post_action",
        lambda self, action: None,
        raising=False,
    )
    assert not task._raw_success()
    height[0] += Task.MIN_LIFT_DELTA + 0.001
    for _ in range(control_freq - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    height[0] = task._initial_source_z + Task.MIN_LIFT_DELTA - 0.001
    task._post_action(None)
    assert task._success_hold == 0
    assert not task._check_success()


def test_405_reset_support_stability_and_physical_lift(monkeypatch):
    def apply_sonic_spawn_pose(env):
        # Reproduce G1_BRIDGE's post-setup standing pose without loading policies.
        # FloatingBody starts with lowered arms; sonic bends both elbows forward.
        angles = {
            "hip_pitch": -0.312,
            "hip_roll": 0.0,
            "hip_yaw": 0.0,
            "knee": 0.669,
            "ankle_pitch": -0.363,
            "ankle_roll": 0.0,
            "shoulder_pitch": 0.2,
            "shoulder_yaw": 0.0,
            "elbow": 0.6,
            "wrist_roll": 0.0,
            "wrist_pitch": 0.0,
            "wrist_yaw": 0.0,
        }
        targets = {f"waist_{axis}": 0.0 for axis in ("yaw", "roll", "pitch")}
        for side, roll in (("left", 0.2), ("right", -0.2)):
            targets.update({f"{side}_{name}": value for name, value in angles.items()})
            targets[f"{side}_shoulder_roll"] = roll
        for name, value in targets.items():
            joint = f"robot0_{name}_joint"
            env.sim.data.set_joint_qpos(joint, value)
            env.sim.data.set_joint_qvel(joint, 0.0)
        env.sim.forward()
        return len(targets)

    monkeypatch.setattr(
        env_utils, "_apply_controller_spawn_pose", apply_sonic_spawn_pose
    )
    env = robosuite.make(
        env_name=Task.__name__,
        robots="G1FloatingBody",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        control_freq=20,
        renderer="mujoco",
        scene_ids=[405],
        seed=0,
    )
    try:
        env.reset()
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert not env._raw_success()
        assert not env._check_success()
        assert env.sim.data.get_joint_qpos(env.cabinet_joint.name) == pytest.approx(
            env.cabinet_joint.open_qpos
        )
        assert env._fixture_bbox_inside_region_xy(
            env.source_fixture, env.work_surface_region
        )
        support_z = env.work_surface_region.world_pose(env.sim)[0][2]
        # The candle's Stool006 at scale 0.55 has a 0.4493 m top height.
        assert support_z == pytest.approx(0.4493, abs=0.01)
        center = env.source_fixture.world_center(env.sim)
        spec = env.scene_task_spec()
        table_center = env.work_surface_region.world_pose(env.sim)[0]
        np.testing.assert_allclose(table_center[:2], spec["table_xy"], atol=1e-4)
        np.testing.assert_allclose(center[:2], spec["source_xy"], atol=1e-4)
        assert env._robot_base_xy()[0] > table_center[0]
        assert center[0] - table_center[0] > 0.07
        near_edge_x = table_center[0] + env.work_surface_region.half_size[0]
        edge_clearance = (
            near_edge_x - env._fixture_bbox_points(env.source_fixture)[:, 0].max()
        )
        assert 0.005 < edge_clearance < 0.03
        region_center, _ = env.cabinet_region.world_pose(env.sim)
        assert np.all(np.abs(center - region_center) < env.cabinet_region.half_size)
        assert 0.49 < env._initial_source_z < 0.62
        assert env._source_horizontal_distance < 0.50
        for name, subjects, supports in _task_body_contact_contracts(env):
            _assert_body_contact_state(env, name, subjects, supports, "reset")
        initial_center = center.copy()
        _settle_scene_with_frozen_robot(env, int(0.5 / env.sim.model.opt.timestep))
        settle_drift = np.linalg.norm(
            env.source_fixture.world_center(env.sim) - initial_center
        )
        assert settle_drift < 0.01
        assert env.count_body_collisions() == 0
        assert not env._raw_success()
        for name, subjects, supports in _task_body_contact_contracts(env):
            _assert_body_contact_state(env, name, subjects, supports, "after settle")
        assert (
            abs(
                env.sim.data.get_joint_qpos(env.cabinet_joint.name)
                - env.cabinet_joint.open_qpos
            )
            < 0.01
        )

        joint = env._fixture_free_joint_name(env.source_fixture)
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        # Pull out of the opening before lifting past the cabinet ceiling.
        qpos[0] += 0.30
        qpos[2] += Task.MIN_LIFT_DELTA + 0.01
        env.sim.data.set_joint_qpos(joint, qpos)
        env._zero_fixture_velocity(env.source_fixture)
        env.sim.forward()
        assert not env._g1_hand_grasping_fixture(env.source_fixture)
        assert not env._fixture_has_illegal_penetration(env.source_fixture, ())
        assert env._raw_success()
        qpos[2] -= 0.02
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        assert not env._raw_success()
        meta = env.get_ep_meta()
        print(
            {
                "initial_root_z": env._initial_source_z,
                "support_z": support_z,
                "source_distance": env._source_horizontal_distance,
                "edge_clearance": float(edge_clearance),
                "success_hold_steps": env.HOLD_STEPS,
                "settle_drift": float(settle_drift),
            }
        )
        assert "independent of grasp" in meta["success_definition"]
    finally:
        env.close()
