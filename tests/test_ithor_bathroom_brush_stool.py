"""Regressions for native scrub brushes on fixed bathroom stools."""

import json

import mujoco
import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_bathroom.tasks import (
    IThorBathroomApproachSinkAndLiftScrubBrush,
    IThorBathroomLiftScrubBrush,
    IThorBathroomNavigateToBathingAreaAndLiftScrubBrush,
)
from robocasa.environments.ithor_bathroom.tasks.brush_support import (
    _BathroomScrubBrushSupportMixin,
)
from robocasa.environments.ithor_bathroom.tasks.composite.brush_support import (
    _BathroomScrubBrushLiftTask,
)
from robocasa.environments.ithor_bathroom.tasks.lift import (
    _BathroomNativeLiftTask,
)
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_spec,
)
from validate_ithor_tasks import (
    _assert_body_contact_state,
    _capture_robot_joint_state,
    _load_reset_controller,
    _task_body_contact_contracts,
)


def test_atomic_brush_task_uses_scene_local_stability_layouts():
    atomic = IThorBathroomLiftScrubBrush
    assert atomic.__bases__[0] is _BathroomScrubBrushSupportMixin
    assert atomic.__bases__[1] is _BathroomNativeLiftTask
    assert _BathroomScrubBrushLiftTask not in atomic.__mro__

    stool_layout = {
        "model": "/objects/lightwheel/stool/Stool005/model.xml",
        "scale": [0.90, 1.90, 0.60],
        "yaw": 0.0,
        "top_half_xy": [0.13, 0.28],
        "floor_probe_above": 0.08,
        "floor_probe_below": 0.01,
        "floor_probe_step": 0.0005,
    }
    for scene, xy, robot_xy in (
        ("FloorPlan405", [-1.49, 0.83], [-1.89, 0.83]),
        ("FloorPlan407", [0.10, -0.70], [-0.30, -0.70]),
        ("FloorPlan415", [-1.587, -3.322], [-1.987, -3.322]),
    ):
        spec = atomic.SCENE_SPECS[scene]
        assert spec["source_xy"] == xy
        assert spec["robot_pose"] == {"xy": robot_xy, "yaw": 0.0}
        assert spec["source_quat"] == [1.0, 0.0, 0.0, 0.0]
        assert spec["surface_fixture"] == f"{scene}:spawned:brush_stool"
        assert spec["surface_region"] == f"{scene}:spawned:brush_stool:region:support"
        assert spec["atomic_source_stool"] == {
            **stool_layout,
            "xy": xy,
            "robot_pose": {"xy": robot_xy, "yaw": 0.0},
            "remove_fixtures": spec["remove_fixtures"],
        }
        assert "source_stool" not in spec

    assert atomic.SCENE_SPECS["FloorPlan415"]["remove_fixtures"] == [
        "FloorPlan415:Candle_3_9"
    ]


def test_brush_stool_layout_is_exclusive_to_bathing_area_composite_in_407():
    atomic = IThorBathroomLiftScrubBrush
    composite = IThorBathroomNavigateToBathingAreaAndLiftScrubBrush
    spec = composite.SCENE_SPECS["FloorPlan407"]
    assert atomic.SUPPORTED_SCENE_IDS == (401, 405, 407, 415, 418)
    assert composite.SUPPORTED_SCENE_IDS == (405, 407, 415)
    assert atomic not in composite.__mro__
    assert issubclass(composite, _BathroomScrubBrushLiftTask)
    assert not issubclass(
        IThorBathroomApproachSinkAndLiftScrubBrush, _BathroomScrubBrushLiftTask
    )
    for task in (atomic, composite):
        assert task._raw_success is _BathroomNativeLiftTask._raw_success
        assert task.MIN_LIFT_DELTA == 0.30
        assert task.GRASP_HEIGHT_ALLOWANCE == 0.10
    assert composite.APPROACH_HOLD_STEPS == 5
    assert composite.SUCCESS_DISTANCE == 0.35
    overrides = get_ithor_bathroom_task_spec("FloorPlan407", composite.SPEC_TASK)
    for field, value in atomic.SCENE_SPECS["FloorPlan407"].items():
        if field not in overrides and field != "lang":
            assert spec[field] == value
    for field, value in overrides.items():
        assert spec[field] == value
    assert spec["source_stool"] is not overrides["source_stool"]
    assert spec["navigation"] == get_ithor_bathroom_task_spec(
        "FloorPlan407", "navigate_to_bathing_area"
    )
    assert spec["source_fixture"] == "FloorPlan407:Scrub_Brush_1_12"
    assert spec["source_quat"] == [1.0, 0.0, 0.0, 0.0]
    assert spec["source_xy"] == [0.10, -0.70]
    assert spec["robot_pose"] == {"xy": [-0.30, -0.70], "yaw": 0.0}
    assert spec["source_stool"] == {
        "model": "/objects/lightwheel/stool/Stool005/model.xml",
        "scale": [0.90, 1.90, 0.60],
        "xy": [0.10, -0.70],
        "yaw": 0.0,
        "top_half_xy": [0.13, 0.28],
        "floor_probe_above": 0.08,
        "floor_probe_below": 0.01,
        "floor_probe_step": 0.0005,
    }
    distance = np.linalg.norm(np.asarray(spec["source_xy"]) - spec["robot_pose"]["xy"])
    assert distance == pytest.approx(spec["source_horizontal_distance"])
    assert distance == pytest.approx(0.40)
    assert spec["lang"] == "walk to the bathing area, then pick up the scrub brush"
    for task in (composite, IThorBathroomApproachSinkAndLiftScrubBrush):
        for scene, other_spec in task.SCENE_SPECS.items():
            if (
                scene not in ("FloorPlan405", "FloorPlan407", "FloorPlan415")
                or task is not composite
            ):
                assert "source_stool" not in other_spec
                shared_spec = get_ithor_bathroom_task_spec(scene, "lift_scrub_brush")
                expected = dict(shared_spec)
                if task is IThorBathroomApproachSinkAndLiftScrubBrush:
                    assert scene == "FloorPlan418"
                    assert shared_spec["source_xy"] == [-2.35, -3.65]
                    assert shared_spec["robot_pose"] == {
                        "xy": [-2.0, -3.65],
                        "yaw": 3.140581,
                    }
                    assert atomic.SCENE_SPECS[scene] == shared_spec
                    expected["source_xy"] = [-2.49, -3.65]
                    expected["robot_pose"] = {
                        "xy": [-1.80, -3.65],
                        "yaw": 3.140581,
                    }
                    for field in ("source_xy", "robot_pose"):
                        assert other_spec[field] is not atomic.SCENE_SPECS[scene][field]
                for field, value in expected.items():
                    if field != "lang":
                        assert other_spec[field] == value


def test_405_brush_composite_matches_atomic_stool_without_sharing_spec():
    composite = IThorBathroomNavigateToBathingAreaAndLiftScrubBrush
    atomic = IThorBathroomLiftScrubBrush.SCENE_SPECS["FloorPlan405"]
    spec = composite.SCENE_SPECS["FloorPlan405"]
    override = get_ithor_bathroom_task_spec("FloorPlan405", composite.SPEC_TASK)
    for field in (
        "source_fixture",
        "surface_fixture",
        "surface_region",
        "source_xy",
        "source_quat",
        "robot_pose",
        "remove_fixtures",
        "auxiliary_placements",
    ):
        assert spec[field] == atomic[field]
    assert spec["source_xy"] == [-1.49, 0.83]
    assert spec["robot_pose"] == {"xy": [-1.89, 0.83], "yaw": 0.0}
    assert spec["source_stool"] == override["source_stool"]
    assert spec["source_stool"] is not override["source_stool"]
    assert spec["source_stool"] == {
        key: value
        for key, value in atomic["atomic_source_stool"].items()
        if key not in ("robot_pose", "remove_fixtures")
    }
    assert spec["navigation"] == get_ithor_bathroom_task_spec(
        "FloorPlan405", "navigate_to_bathing_area"
    )
    assert spec["source_horizontal_distance"] == pytest.approx(0.40)
    assert "source_stool" not in atomic
    assert composite.SCENE_SPECS["FloorPlan407"]["source_xy"] == [0.10, -0.70]
    assert composite.SCENE_SPECS["FloorPlan415"]["source_xy"] == [-1.17, -0.83]


def test_415_brush_composite_has_its_own_reachable_layout():
    composite = IThorBathroomNavigateToBathingAreaAndLiftScrubBrush
    atomic = IThorBathroomLiftScrubBrush.SCENE_SPECS["FloorPlan415"]
    spec = composite.SCENE_SPECS["FloorPlan415"]
    override = get_ithor_bathroom_task_spec("FloorPlan415", composite.SPEC_TASK)
    assert spec["source_stool"] == override["source_stool"]
    assert spec["source_stool"] is not override["source_stool"]
    assert spec["surface_fixture"] == "FloorPlan415:spawned:brush_stool"
    assert spec["source_xy"] == [-1.17, -0.83]
    assert spec["robot_pose"] == {"xy": [-1.57, -0.83], "yaw": 0.0}
    assert np.linalg.norm(
        np.asarray(spec["source_xy"]) - spec["robot_pose"]["xy"]
    ) == pytest.approx(0.40)
    assert spec["navigation"] == get_ithor_bathroom_task_spec(
        "FloorPlan415", "navigate_to_bathing_area"
    )
    assert atomic["surface_fixture"] == "FloorPlan415:spawned:brush_stool"
    assert atomic["source_xy"] == [-1.587, -3.322]
    assert "source_stool" not in atomic
    assert atomic["atomic_source_stool"]["xy"] == [-1.587, -3.322]


def test_405_brush_composite_stool_initial_state():
    env = robosuite.make(
        env_name=IThorBathroomNavigateToBathingAreaAndLiftScrubBrush.__name__,
        robots="G1",
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
        assert env._approach_stage == 0
        assert not env._near_approach_fixture()
        assert not env._check_success()
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._source_horizontal_distance == pytest.approx(0.40)
        _assert_support_contacts(env, "405 reset")
        assert env._operation_initial_state_intact()
        env._set_exact_robot_pose(env.scene_task_spec()["robot_pose"], "405 brush")
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        _assert_support_contacts(env, "405 brush stance")
    finally:
        env.close()


def test_415_brush_composite_can_approach_and_finish_after_lift(monkeypatch):
    # One representative reset; emulate only the grasp signal after verifying
    # that the actual staged brush and navigation stance are collision-free.
    env = robosuite.make(
        env_name=IThorBathroomNavigateToBathingAreaAndLiftScrubBrush.__name__,
        robots="G1",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        control_freq=20,
        renderer="mujoco",
        scene_ids=[415],
        seed=0,
    )
    try:
        env.reset()
        assert env._approach_stage == 0
        assert not env._near_approach_fixture()
        assert not env._check_success()
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._source_horizontal_distance == pytest.approx(0.40)
        _assert_support_contacts(env, "415 reset")

        # Facing the nearby brush alone must not bypass the bathtub gate.
        env._set_exact_robot_pose(env.scene_task_spec()["robot_pose"], "415 brush")
        assert env.count_body_collisions() == 0
        assert not env._near_approach_fixture()
        env._set_exact_robot_pose(
            {"xy": [-1.57, -0.93], "yaw": 2.988943}, "415 bathing area"
        )
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._near_approach_fixture()
        assert env._operation_initial_state_intact()
        monkeypatch.setattr(
            _BathroomNativeLiftTask, "_post_action", lambda self, action: None
        )
        for _ in range(env.APPROACH_HOLD_STEPS):
            env._post_action(None)
        assert env._approach_stage == 1
        assert not env._check_success()
        env._set_exact_robot_pose(env.scene_task_spec()["robot_pose"], "415 brush")
        assert env.count_body_collisions() == 0

        grasp = {"held": True}
        monkeypatch.setattr(
            env, "_g1_hand_grasping_fixture", lambda fixture: grasp["held"]
        )
        assert not env._raw_success()  # Contact before lifting must be registered.
        joint = env._fixture_free_joint_name(env.source_fixture)
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        qpos[2] += env.MIN_LIFT_DELTA + 0.01
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        for _ in range(env.HOLD_STEPS - 1):
            env._post_action(None)
        assert not env._check_success()
        env._post_action(None)
        assert env._check_success()
        grasp["held"] = False
        env._post_action(None)
        assert not env._check_success()
    finally:
        env.close()


@pytest.mark.parametrize(
    ("scene_id", "max_drift"),
    ((405, 0.03), (407, 0.06), (415, 0.03)),
)
def test_atomic_brush_initial_state_stays_supported(scene_id, max_drift):
    env = robosuite.make(
        env_name=IThorBathroomLiftScrubBrush.__name__,
        robots="G1",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        control_freq=20,
        renderer="mujoco",
        scene_ids=[scene_id],
        seed=0,
    )
    try:
        env.reset()
        source = env.source_fixture
        initial = source.world_center(env.sim).copy()
        stool_initial = env.task_stool_fixture.world_center(env.sim).copy()
        assert not env._check_success()
        assert not env._raw_success()
        assert env.count_body_collisions() == 0
        _assert_support_contacts(env, "reset")
        robot_state = _capture_robot_joint_state(env)
        initial_time = env.sim.data.time
        max_observed_drift = 0.0
        while env.sim.data.time - initial_time < 5.0:
            env.sim.step()
            for qpos_slice, qpos, qvel_slice, qvel in robot_state:
                env.sim.data.qpos[qpos_slice] = qpos
                env.sim.data.qvel[qvel_slice] = qvel
            env.sim.forward()
            max_observed_drift = max(
                max_observed_drift,
                float(np.linalg.norm(source.world_center(env.sim) - initial)),
            )

        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._fixture_bbox_inside_region_xy(source, env.work_surface_region)
        assert max_observed_drift < max_drift
        _assert_support_contacts(env, "after 5 seconds")
        assert not env._check_success()
        assert not env._raw_success()
        assert (
            np.linalg.norm(env.task_stool_fixture.world_center(env.sim) - stool_initial)
            < 0.001
        )
        assert env.sim.data.eq_active[env._stool_base_weld_id()]
    finally:
        env.close()


def _assert_support_contacts(env, phase):
    for name, subjects, supports in _task_body_contact_contracts(env):
        _assert_body_contact_state(env, name, subjects, supports, phase)
    assert env._fixture_bbox_inside_region_xy(
        env.source_fixture, env.work_surface_region
    )


def test_407_brush_stays_on_stool_for_twenty_simulated_seconds():
    # Pin only the uncontrolled robot. Both brush and stool participate in
    # physics; only the stool has a world weld. This is not a teleop test.
    env = robosuite.make(
        env_name=IThorBathroomNavigateToBathingAreaAndLiftScrubBrush.__name__,
        robots="G1",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        control_freq=20,
        renderer="mujoco",
        scene_ids=[407],
        seed=0,
    )
    try:
        env.reset()
        spec = env.scene_task_spec()
        source = env.source_fixture
        initial = source.world_center(env.sim).copy()
        stool_initial = env.task_stool_fixture.world_center(env.sim).copy()
        assert env._source_horizontal_distance == pytest.approx(0.40)
        assert env.HOLD_STEPS / env.control_freq == pytest.approx(1.0)
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        np.testing.assert_allclose(
            env._robot_base_xy(), spec["navigation"]["start_pose"]["xy"]
        )
        assert env._approach_stage == 0
        assert not env._near_approach_fixture()
        assert not env._check_success()
        _assert_support_contacts(env, "reset")
        source_joint = env._fixture_free_joint_name(source)
        assert source_joint is not None
        joint_id = env.sim.model.joint_name2id(source_joint)
        assert env.sim.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
        source_bodies = env._fixture_body_ids(source, include_siblings=True)
        for equality_id in range(env.sim.model.neq):
            if env.sim.model.eq_type[equality_id] == mujoco.mjtEq.mjEQ_WELD:
                assert env.sim.model.eq_obj1id[equality_id] not in source_bodies
                assert env.sim.model.eq_obj2id[equality_id] not in source_bodies
        assert env.sim.data.eq_active[env._stool_base_weld_id()]
        robot_state = _capture_robot_joint_state(env)
        initial_time = env.sim.data.time
        max_drift = 0.0
        physics_steps = 0
        while env.sim.data.time - initial_time < 20.0:
            env.sim.step()
            for qpos_slice, qpos, qvel_slice, qvel in robot_state:
                env.sim.data.qpos[qpos_slice] = qpos
                env.sim.data.qvel[qvel_slice] = qvel
            env.sim.forward()
            drift = float(np.linalg.norm(source.world_center(env.sim) - initial))
            max_drift = max(max_drift, drift)
            assert drift < 0.03, "brush left its reachable navigation initial state"
            physics_steps += 1
            if physics_steps % 25 == 0:
                assert env._fixture_bbox_inside_region_xy(
                    source, env.work_surface_region
                )
                assert env._operation_initial_state_intact()
                assert not env._raw_success()
        _assert_support_contacts(env, "after 20 seconds")
        assert env._operation_initial_state_intact()
        assert env.count_body_collisions() == 0
        assert not env._check_success()
        assert (
            np.linalg.norm(env.task_stool_fixture.world_center(env.sim) - stool_initial)
            < 0.001
        )

        env._set_exact_robot_pose(spec["robot_pose"], "brush operation regression")
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        bath_distance = env.robot_fixture_collision_distance(
            env.approach_fixture, max_distance=1.0
        )
        assert 0.0 < bath_distance <= env.SUCCESS_DISTANCE
        assert env._near_approach_fixture()
        assert env._operation_initial_state_intact()
        assert not env._check_success()
        _assert_support_contacts(env, "near manipulation pose")
        print(
            json.dumps(
                {
                    "scene": 407,
                    "simulated_seconds": env.sim.data.time - initial_time,
                    "max_brush_displacement_m": max_drift,
                    "brush_initial_center": initial.tolist(),
                    "brush_final_center": source.world_center(env.sim).tolist(),
                    "near_bathtub_collision_distance_m": float(bath_distance),
                    "near_brush_horizontal_distance_m": float(
                        np.linalg.norm(
                            source.world_center(env.sim)[:2] - env._robot_base_xy()
                        )
                    ),
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()
