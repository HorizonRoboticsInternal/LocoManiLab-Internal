"""Opt-in headless evidence for the six distinct routine initialization paths.

TELE_BEDROOM_ROUTINE_PHYSICS=1 MUJOCO_GL=egl python -m pytest -q \
    tests/test_ithor_bedroom_routine_physics.py

Diagnostic state placement proves physical capacity and predicates, not teleop.
"""

import os
from pathlib import Path
import re

import mujoco
import numpy as np
import pytest
import robosuite
import yaml

from robocasa.environments.ithor_bedroom.tasks.routine_composites import (
    TASK_CLASSES,
    IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop,
)
from validate_ithor_tasks import _capture_robot_joint_state, _load_reset_controller


pytestmark = pytest.mark.skipif(
    os.environ.get("TELE_BEDROOM_ROUTINE_PHYSICS") != "1",
    reason="explicit opt-in: one headless construction per changed initialization path",
)


def _advance(env, robot_state, seconds):
    start = env.sim.data.time
    while env.sim.data.time - start < seconds - 1e-8:
        env.sim.step()
        for qs, qpos, vs, qvel in robot_state:
            env.sim.data.qpos[qs] = qpos
            env.sim.data.qvel[vs] = qvel
        env.sim.forward()


def _assert_feet_on_physical_floor(env):
    # A zero collision count and an indoor XY footprint can still hide a
    # robot suspended above the floor because its hand stopped the ground snap.
    from robocasa.g1_host.mujoco_io import build_foot_geom_ids

    feet = build_foot_geom_ids(env.sim, env.robots[0].robot_model)
    collidable = (
        lambda g: env.sim.model.geom_contype[g] or env.sim.model.geom_conaffinity[g]
    )
    floors = [
        g for g in range(env.sim.model.ngeom) if collidable(g) and env._is_floor_geom(g)
    ]
    assert feet and floors
    for side in ("left", "right"):
        foot = [
            g
            for g in feet
            if collidable(g)
            and side in env.sim.model.body_id2name(int(env.sim.model.geom_bodyid[g]))
        ]
        assert foot
        distance = min(
            mujoco.mj_geomDistance(
                env.sim.model._model, env.sim.data._data, g, floor, 0.1, None
            )
            for g in foot
            for floor in floors
        )
        assert -0.002 <= distance <= 0.0001, (side, distance)


def _center_at(env, name, xy, bottom):
    env._set_object_bottom_at(name, xy, bottom)
    points = env.object_bbox_points(name)
    joint = env.objects[name].joints[0]
    qpos = env.sim.data.get_joint_qpos(joint).copy()
    qpos[:2] += xy - (points.min(axis=0)[:2] + points.max(axis=0)[:2]) / 2
    env.sim.data.set_joint_qpos(joint, qpos)
    env._zero_free_joint_velocity(env.objects[name])
    env.sim.forward()


def _check_packing(env, robot_state):
    """Capacity positive examples, then first-item-loss and passive negatives."""
    points = env.object_bbox_points("basket")
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    bottom = points[:, 2].min()
    if len(env.ITEM_NAMES) == 3:
        # A horizontal can leaves clearance for the mug beside/above it.
        joint = env.objects["can"].joints[0]
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        w, x, y, z = qpos[3:]
        c = np.sqrt(0.5)
        qpos[3:] = [c * w - c * y, c * x + c * z, c * w + c * y, c * z - c * x]
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        _center_at(env, "can", center[:2] + [0, -0.032], bottom + 0.035)
        _center_at(env, "boxed_food", center[:2] + [0, 0.034], bottom + 0.035)
        _advance(env, robot_state, 2)
        assert env._supported_basket_items() == {"can", "boxed_food"}
        food = env.object_bbox_points("boxed_food")
        food_center = (food.min(axis=0) + food.max(axis=0)) / 2
        _center_at(env, "mug", food_center[:2] + [0, -0.008], food[:, 2].max() + 0.005)
        _advance(env, robot_state, 3)
        assert env._load_bearing_contact("mug", "boxed_food")
    elif len(env.ITEM_NAMES) == 2:
        _center_at(env, "can", center[:2] + [0, -0.037], bottom + 0.02)
        _center_at(env, "boxed_food", center[:2] + [0, 0.034], bottom + 0.02)
        _advance(env, robot_state, 2)
    else:
        env._set_object_bottom_at("mug", env.object_pos("basket")[:2], bottom + 0.035)
        _advance(env, robot_state, 2)
        joint = env.target_semantic_joint
        # Verify lid sweep does not collide with either tabletop object.
        for fraction in np.linspace(0.85, 0, 18):
            env.sim.data.set_joint_qpos(
                joint.name,
                joint.closed_qpos + fraction * (joint.open_qpos - joint.closed_qpos),
            )
            env.sim.forward()
            assert env._laptop_support_contact_is_valid()
            geoms = env._geom_names_for_bodies(
                env._fixture_body_ids(env.target_fixture)
            )
            assert all(
                not env.check_contact(geoms, obj) for obj in env.objects.values()
            )
    for _ in range(20):
        _advance(env, robot_state, 0.05)
        assert env._supported_basket_items() == set(env.ITEM_NAMES)
        if "close_laptop" in env.COMPOSITE_ORDERING:
            assert env._routine_goal("close_laptop")
    # A final-looking state with no observed manipulation cannot earn credit.
    env._reset_task_progress()
    for _ in range(25):
        env._advance_routine()
    assert not env._check_success()
    if "can" in env.ITEM_NAMES:
        _center_at(env, "can", center[:2] + [0.4, 0], 0.618)
        assert "can" not in env._supported_basket_items()


@pytest.mark.parametrize("scene_id", (306, 323))
def test_can_then_boxed_food_new_scene_support(scene_id):
    """Verify both new desk layouts and the physical two-item goal."""
    from robocasa.environments.ithor_bedroom.tasks.routine_composites import (
        IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket as Task,
    )
    from robocasa.models.scenes.ithor_bedroom_task_specs import (
        get_ithor_bedroom_task_spec,
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
        scene_ids=[scene_id],
        seed=0,
    )
    try:
        env.reset()
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert not env._check_success()
        expected_pose = Task.SCENE_SPECS[env.scene_name]["routine_start"]
        pose = expected_pose.get("start_pose", expected_pose.get("far_start_pose"))
        np.testing.assert_allclose(env._robot_base_xy(), pose["xy"], atol=1e-8)
        if scene_id == 323:
            assert (
                pose
                == get_ithor_bedroom_task_spec(env.scene_name, "approach_desk")[
                    "start_pose"
                ]
            )
        initial = {name: env.object_pos(name).copy() for name in env.objects}
        robot_state = _capture_robot_joint_state(env)
        _advance(env, robot_state, 20)
        for name, position in initial.items():
            assert np.linalg.norm(env.object_pos(name) - position) < 0.003
            assert env._object_contacts_fixture(name, env.work_surface_fixture)
            assert not env._object_has_illegal_penetration(
                name, env.work_surface_region, tolerance=0.003
            )
        _check_packing(env, robot_state)
    finally:
        env.close()


def _check_large_basket_layout(env):
    # Only the three-item routine uses the enlarged Basket034 variant.
    np.testing.assert_allclose(
        np.ptp(env.object_bbox_points("basket"), axis=0),
        np.array([0.2186197875, 0.175494825, 0.126461487]) * 1.2,
        atol=0.001,
    )
    assert env.get_ep_meta()["basket_variant"] == "three_item_large"
    for name in env.objects:
        assert env.object_on_region(
            name, env.work_surface_region, z_tol=0.003, require_contact=True
        )
        assert not env._object_has_illegal_penetration(
            name, env.work_surface_region, tolerance=0.002
        )


def _check_hamper_then_desk(env):
    op = env.scene_task_spec()["hamper_robot_pose"]
    joint = env.target_semantic_joint
    env.sim.data.set_joint_qpos(
        joint.name, joint.closed_qpos + 0.1 * (joint.open_qpos - joint.closed_qpos)
    )
    dof = int(env.sim.model.jnt_dofadr[env.sim.model.joint_name2id(joint.name)])
    env.sim.data.qvel[dof] = 0.0
    assert env._set_robot_free_pose(op["xy"], op["yaw"])
    env._routine_stage = 2  # diagnostic completion of the first two stages
    env._routine_success = False
    env._enter_routine_stage()
    steps = int(np.ceil(env.control_freq))
    for _ in range(steps):
        env._advance_routine()
    assert not env._check_success()  # the desk is already within 0.35 m here

    # Collision-checked route around the chair. These poses are test evidence,
    # never stage-switch teleports or required waypoints in the actual task.
    waypoints = [
        (*op["xy"], op["yaw"]),
        (0.9, -0.1, op["yaw"]),
        (0.9, -0.1, np.pi),
        (-0.85, -0.1, np.pi),
        (-0.85, -0.6, np.pi),
    ]
    for start, end in zip(waypoints, waypoints[1:]):
        start, end = np.array(start), np.array(end)
        count = max(
            2,
            int(np.ceil(np.linalg.norm(end[:2] - start[:2]) / 0.05)) + 1,
            int(np.ceil(abs(end[2] - start[2]) / 0.1)) + 1,
        )
        for pose in np.linspace(start, end, count):
            assert env._set_robot_free_pose(pose[:2], pose[2])
            assert env.robot_footprint_is_on_floor(pose[:2], radius=0.25)
            assert env.count_body_collisions() == 0
    _assert_feet_on_physical_floor(env)
    assert env._routine_geometry("approach_desk")
    assert env.get_ep_meta()["final_navigation_min_displacement"] == 1.0
    robot_state = _capture_robot_joint_state(env)
    for _ in range(steps - 1):
        _advance(env, robot_state, 1.0 / env.control_freq)
        env._advance_routine()
    assert not env._check_success()
    _advance(env, robot_state, 1.0 / env.control_freq)
    env._advance_routine()
    assert env._check_success()
    env.sim.data.set_joint_qpos(
        joint.name, joint.closed_qpos + 0.85 * (joint.open_qpos - joint.closed_qpos)
    )
    env.sim.forward()
    env._advance_routine()
    assert not env._check_success() and env._routine_hold == 0


def _check_joints_and_navigation(env):
    if env.scene_name == "FloorPlan306":
        _check_hamper_then_desk(env)
        return
    for stage, (joint, threshold) in env._routine_joints.items():
        others = {
            key: float(env.sim.data.get_joint_qpos(other.name))
            for key, (other, _) in env._routine_joints.items()
            if key != stage
        }
        env.sim.data.set_joint_qpos(
            joint.name, joint.closed_qpos + 0.10 * (joint.open_qpos - joint.closed_qpos)
        )
        env.sim.forward()
        assert env._routine_goal(stage)
        assert all(
            float(env.sim.data.get_joint_qpos(env._routine_joints[key][0].name)) == qpos
            for key, qpos in others.items()
        )
    env._routine_entry_xy = env._robot_base_xy().copy()
    xy, yaw = [0.9, -0.5], -2.3392424061524397
    if hasattr(env, "_robot_free_joint_z"):
        del env._robot_free_joint_z
    assert (
        env._set_robot_free_pose(xy, yaw) and env._calibrate_robot_free_joint_height()
    )
    assert env.robot_footprint_is_on_floor(xy, radius=0.25)
    assert env.count_body_collisions() == 0 and env._routine_geometry("approach_bed")


def _inject_sonic_spawn_stance(monkeypatch):
    """Exercise the bridge's spawn-stance protocol without loading a policy/VR.

    Pose-only reset audits otherwise retain the MJCF stance, missing the
    SONIC arm/desk or arm/basket penetration that prevented these routines
    from loading.
    """
    from g1_bridge.g1_runtime import sim2real_root
    from robocasa.g1_host.wbt import WholeBodyG1Composite

    policy = Path(sim2real_root()) / "checkpoints/sonic/v1_1/g1/policy.yaml"
    defaults = yaml.safe_load(policy.read_text())["default_joint_pos"]
    angles = np.array(
        [
            next(
                (
                    value
                    for pattern, value in defaults.items()
                    if re.fullmatch(pattern, name)
                ),
                0.0,
            )
            for name in WholeBodyG1Composite.JOINT_NAMES
        ]
    )

    def spawn_joint_targets(controller):
        return tuple(
            (qa, qv, float(angle))
            for qa, qv, angle in zip(
                controller.qpos_addrs, controller.qvel_addrs, angles
            )
            if qa is not None and qv is not None
        )

    monkeypatch.setattr(
        WholeBodyG1Composite, "spawn_joint_targets", spawn_joint_targets, raising=False
    )
    return angles


def _observe_mug_laptop_sonic_layout(monkeypatch, angles):
    cls = IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop
    original = cls._layout_basket_row
    layouts = []

    def observe_layout(env, *, laptop=False):
        # Both construction and reset must clear the robot before any probe.
        pose = env.scene_task_spec()["routine_start"]["far_start_pose"]
        np.testing.assert_allclose(env._robot_base_xy(), pose["xy"], atol=1e-8)
        controller = env.robots[0].composite_controller
        np.testing.assert_allclose(
            env.sim.data.qpos[controller.qpos_addrs], angles, atol=1e-8
        )
        original(env, laptop=laptop)
        positions = {name: env.object_pos(name).copy() for name in env.objects}
        # Calibrated before this fix: advancing the robot spawn must not move
        # the original row, change its operating side, or weaken support.
        np.testing.assert_allclose(
            positions["basket"][:2], [-0.473112579, 0.682061737], atol=1e-6
        )
        np.testing.assert_allclose(
            positions["mug"][:2], [-0.278910819, 0.682061737], atol=1e-6
        )
        assert all(
            env.object_on_region(
                name, env.work_surface_region, z_tol=0.003, require_contact=True
            )
            and not env._object_has_illegal_penetration(
                name, env.work_surface_region, tolerance=0.002
            )
            for name in env.objects
        )
        layouts.append(positions)

    monkeypatch.setattr(cls, "_layout_basket_row", observe_layout)
    return layouts


@pytest.mark.parametrize("cls", TASK_CLASSES, ids=lambda cls: cls.__name__)
def test_routine_initialization_and_physical_goals(cls, monkeypatch):
    angles = _inject_sonic_spawn_stance(monkeypatch)
    layouts = (
        _observe_mug_laptop_sonic_layout(monkeypatch, angles)
        if cls is IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop
        else None
    )
    env = robosuite.make(
        env_name=cls.__name__,
        robots="G1FloatingBody",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        control_freq=20,
        renderer="mujoco",
        # This regression preserves the original calibrated 329 layouts; the
        # hamper routine has only its original Scene 306.
        scene_ids=[306 if cls.SUPPORTED_SCENE_IDS == (306,) else 329],
        seed=0,
    )
    try:
        env.reset()
        controller = env.robots[0].composite_controller
        np.testing.assert_allclose(
            env.sim.data.qpos[controller.qpos_addrs], angles, atol=1e-8
        )
        if layouts is not None:
            assert len(layouts) == 2
            for name in env.objects:
                np.testing.assert_allclose(
                    layouts[0][name], layouts[1][name], atol=1e-6
                )
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        _assert_feet_on_physical_floor(env)
        # XY footprint and floor-exempt collision counts alone cannot detect
        # an obsolete robot configuration placed below the physical floor.
        assert all(position[2] > 0.4 for position in env.eef_positions())
        assert not env._check_success()
        initial = {name: env.object_pos(name).copy() for name in env.objects}
        assert all(
            env._object_contacts_fixture(name, env.work_surface_fixture)
            for name in env.objects
        )
        if len(env.ITEM_NAMES) == 3:
            _check_large_basket_layout(env)
        fractions = {
            key: joint.fraction(env.sim)
            for key, (joint, _) in getattr(env, "_routine_joints", {}).items()
        }
        for key, fraction in fractions.items():
            assert fraction == pytest.approx(
                {"close_laptop": 0.85, "close_top_drawer": 0.80, "close_hamper": 1.0}[
                    key
                ]
            )
        robot_state = _capture_robot_joint_state(env)
        _advance(env, robot_state, 20)
        for name, pos in initial.items():
            assert np.linalg.norm(env.object_pos(name) - pos) < 0.003
            assert env._object_contacts_fixture(name, env.work_surface_fixture)
        for key, fraction in fractions.items():
            assert env._routine_joints[key][0].fraction(env.sim) == pytest.approx(
                fraction, abs=0.001
            )
        if env.ITEM_NAMES:
            _check_packing(env, robot_state)
        else:
            _check_joints_and_navigation(env)
    finally:
        env.close()


@pytest.mark.parametrize(
    "task_name,scene_id",
    (
        ("IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop", 302),
        ("IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop", 323),
        ("IThorBedroomCloseLaptopThenApproachBedAndCloseTopDrawer", 310),
        ("IThorBedroomApproachDeskAndCloseLaptopThenApproachBed", 302),
        ("IThorBedroomApproachDeskAndCloseLaptopThenApproachBed", 323),
        ("IThorBedroomPutCanThenBoxedFoodThenMugInBasket", 302),
        ("IThorBedroomPutCanThenBoxedFoodThenMugInBasket", 306),
        ("IThorBedroomPutCanThenBoxedFoodThenMugInBasket", 323),
    ),
)
def test_new_routine_scene_layout_and_goals(task_name, scene_id, monkeypatch):
    """One SONIC-stance construction for each newly supported initialization."""
    from robocasa.environments.ithor_bedroom.tasks import routine_composites
    from robocasa.models.scenes.ithor_bedroom_task_specs import (
        get_ithor_bedroom_task_spec,
    )

    _inject_sonic_spawn_stance(monkeypatch)
    cls = getattr(routine_composites, task_name)
    env = robosuite.make(
        env_name=task_name,
        robots="G1FloatingBody",
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
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        _assert_feet_on_physical_floor(env)
        assert not env._check_success()
        if (
            task_name == "IThorBedroomPutCanThenBoxedFoodThenMugInBasket"
            and scene_id == 302
        ):
            workspace = get_ithor_bedroom_task_spec(env.scene_name, "grasp_workspace")
            center, outward, _ = env._surface_layout_frame(
                env.work_surface_region,
                operator_xy=(
                    env.work_surface_region.world_pose(env.sim)[0][:2]
                    + 2.0 * np.asarray(workspace["approach_direction"])
                ),
            )
            offset = env._robot_base_xy() - center[:2]
            # The robot must be centered on the object row's normal, on its
            # front side; the default north-side desk approach is invalid.
            assert abs(np.cross(outward, offset)) < 1e-8
            assert np.dot(outward, offset) > env.support_region_radial_extent(
                env.work_surface_region, outward, sim=env.sim
            )
            row_xy = np.mean(
                [env.object_pos(name)[:2] for name in ("basket", *env.ITEM_NAMES)],
                axis=0,
            )
            robot_xy = env._robot_base_xy()
            expected_yaw = np.arctan2(row_xy[1] - robot_xy[1], row_xy[0] - robot_xy[0])
            yaw_error = np.arctan2(
                np.sin(env.robot_base_ori[2] - expected_yaw),
                np.cos(env.robot_base_ori[2] - expected_yaw),
            )
            assert abs(yaw_error) < 1e-8
        if scene_id == 323 and task_name.startswith("IThorBedroomApproachDesk"):
            pose = get_ithor_bedroom_task_spec(env.scene_name, "approach_desk")[
                "start_pose"
            ]
            np.testing.assert_allclose(env._robot_base_xy(), pose["xy"], atol=1e-8)
        if (
            scene_id in (302, 323)
            and task_name.startswith("IThorBedroomApproachDesk")
            and "CloseLaptop" in task_name
        ):
            # The native laptop atomic stages on a different bedside table;
            # these routines must instead operate on their named main desk.
            desk = get_ithor_bedroom_task_spec(env.scene_name, "grasp_workspace")
            assert env.laptop_work_surface_region.stable_id == desk["region"]
            assert env._laptop_support_contact_is_valid()
        initial = {name: env.object_pos(name).copy() for name in env.objects}
        robot_state = _capture_robot_joint_state(env)
        _advance(env, robot_state, 2)
        for name, pos in initial.items():
            assert np.linalg.norm(env.object_pos(name) - pos) < 0.003
            # The original iTHOR desk supports have different physical
            # surface offsets (302 settles 8 mm low; 323 rests 5 mm high).
            # Demand real contact and a stable full footprint, not a uniform
            # zero-offset support plane shared by every scene.
            _, local, bottom, _ = env._object_region_bounds(
                name, env.work_surface_region
            )
            assert np.all(np.abs(local) <= env.work_surface_region.half_size[:2] + 1e-4)
            assert (
                abs(bottom - env.work_surface_region.world_pose(env.sim)[0][2]) < 0.01
            )
            assert env._object_contacts_fixture(name, env.work_surface_fixture)
            assert not env._object_has_illegal_penetration(
                name, env.work_surface_region, tolerance=0.003
            )
        if env.ITEM_NAMES:
            if len(env.ITEM_NAMES) == 3:
                basket_cfg = next(
                    cfg for cfg in env._get_obj_cfgs() if cfg["name"] == "basket"
                )
                np.testing.assert_allclose(
                    basket_cfg["object_scale"], cls.BASKET_ASSET[1]
                )
                assert env.get_ep_meta()["basket_variant"] == "three_item_large"
            _check_packing(env, robot_state)
        if "close_top_drawer" in env.COMPOSITE_ORDERING:
            spec = env.scene_task_spec()["routine_targets"]["close_top_drawer"]
            assert (
                spec["joint"]
                == get_ithor_bedroom_task_spec(env.scene_name, "drawer")["joint"]
            )
            joint = env._routine_joints["close_top_drawer"][0]
            assert joint.fraction(env.sim) == pytest.approx(0.8, abs=0.001)
            assert not env._routine_geometry("close_top_drawer")
            initial_qpos = env.sim.data.get_joint_qpos(joint.name)
            env.sim.data.set_joint_qpos(
                joint.name,
                joint.closed_qpos + 0.1 * (joint.open_qpos - joint.closed_qpos),
            )
            env.sim.forward()
            assert env._routine_goal("close_top_drawer")
            env.sim.data.set_joint_qpos(joint.name, initial_qpos)
            env.sim.forward()
        for stage in env.COMPOSITE_ORDERING:
            if stage == "close_laptop":
                joint = env._routine_joints[stage][0]
                env.sim.data.set_joint_qpos(
                    joint.name,
                    joint.closed_qpos + 0.1 * (joint.open_qpos - joint.closed_qpos),
                )
                env.sim.forward()
                assert env._routine_goal(stage)
    finally:
        env.close()
