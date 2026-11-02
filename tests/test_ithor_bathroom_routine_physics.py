"""Opt-in headless layout evidence, not a teleop or grasp-success claim.

TELE_BATHROOM_ROUTINE_PHYSICS=1 MUJOCO_GL=egl python -m pytest -q -s \
    tests/test_ithor_bathroom_routine_physics.py
"""

import json
import os
from pathlib import Path
import re

import mujoco
import numpy as np
import pytest
import robosuite
import yaml

from robocasa.environments.ithor_bathroom.tasks.composite.routines import (
    TASK_CLASSES,
    IThorBathroomOpenUpperDrawerThenLiftToiletPaperRollAndCloseUpperDrawer as DRAWER_TASK,
)
from validate_ithor_tasks import (
    _assert_body_contact_state, _capture_robot_joint_state, _load_reset_controller,
    _task_body_contact_contracts,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("TELE_BATHROOM_ROUTINE_PHYSICS") != "1",
    reason="explicit opt-in: new routine initialization and physics",
)


def _centres(env):
    positions = {name: env.object_pos(name).copy() for name in env.objects}
    positions.update({name: skill.source_fixture.world_center(env.sim).copy()
                      for name, skill in env._skills.items() if "source_fixture" in skill.__dict__})
    return positions


def _advance(env, seconds, robot, joint_fraction=None):
    start = env.sim.data.time
    initial = env._routine_joint.fraction(env.sim) if joint_fraction is not None else None
    jid = env.sim.model.joint_name2id(env._routine_joint.name) if initial is not None else None
    dof = int(env.sim.model.jnt_dofadr[jid]) if jid is not None else None
    while env.sim.data.time - start < seconds - 1e-8:
        if joint_fraction is not None:
            joint = env._routine_joint
            fraction = initial + (joint_fraction - initial) * min(1., (env.sim.data.time - start) / seconds)
            env.sim.data.set_joint_qpos(joint.name, joint.closed_qpos + fraction * (joint.open_qpos - joint.closed_qpos))
            env.sim.data.qvel[dof] = (joint_fraction - initial) * (joint.open_qpos - joint.closed_qpos) / seconds
        env.sim.step()
        for qs, q, vs, v in robot:
            env.sim.data.qpos[qs] = q
            env.sim.data.qvel[vs] = v
        env.sim.forward()


def _assert_contacts(env, phase):
    for name, subjects, supports in _task_body_contact_contracts(env):
        _assert_body_contact_state(env, name, subjects, supports, phase)


def _sonic_spawn_stance(monkeypatch):
    """Use the real policy defaults through the bridge's spawn protocol.

    No policy inference or VR is needed. Default WHOLE_BODY_G1/MJCF angles
    alone missed the shelf / index-finger collisions in the collector stance.
    """
    from g1_bridge.g1_runtime import sim2real_root
    from robocasa.g1_host.wbt import WholeBodyG1Composite

    policy = Path(sim2real_root()) / "checkpoints/sonic/v1_1/g1/policy.yaml"
    defaults = yaml.safe_load(policy.read_text())["default_joint_pos"]
    angles = np.array([
        next((value for pattern, value in defaults.items() if re.fullmatch(pattern, name)), 0.)
        for name in WholeBodyG1Composite.JOINT_NAMES
    ])

    def targets(controller):
        return tuple((qa, qv, float(angle)) for qa, qv, angle in zip(
            controller.qpos_addrs, controller.qvel_addrs, angles
        ) if qa is not None and qv is not None)

    monkeypatch.setattr(WholeBodyG1Composite, "spawn_joint_targets", targets, raising=False)
    return angles


def _assert_feet_touch_floor(env):
    # Floor-exempt collision counts and an XY footprint cannot rule out a
    # suspended robot whose hand (instead of a foot) stopped the ground snap.
    from robocasa.g1_host.mujoco_io import build_foot_geom_ids

    feet = build_foot_geom_ids(env.sim, env.robots[0].robot_model)
    collidable = lambda g: env.sim.model.geom_contype[g] or env.sim.model.geom_conaffinity[g]
    floors = [g for g in range(env.sim.model.ngeom) if collidable(g) and env._is_floor_geom(g)]
    assert feet and floors
    for side in ("left", "right"):
        foot = [g for g in feet if collidable(g) and
                side in env.sim.model.body_id2name(int(env.sim.model.geom_bodyid[g]))]
        assert foot
        distance = min(mujoco.mj_geomDistance(env.sim.model._model, env.sim.data._data,
                       g, floor, .1, None) for g in foot for floor in floors)
        assert -.002 <= distance <= .0001, (side, distance)


@pytest.mark.parametrize("cls", TASK_CLASSES, ids=lambda c: c.__name__)
def test_new_routine_sonic_spawn(cls, monkeypatch):
    angles = _sonic_spawn_stance(monkeypatch)
    env = robosuite.make(env_name=cls.__name__, robots="G1FloatingBody",
        controller_configs=_load_reset_controller(), has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True,
        hard_reset=False, control_freq=20, renderer="mujoco",
        scene_ids=cls.SUPPORTED_SCENE_IDS, seed=0)
    try:
        initial = _centres(env)
        # Repeat only the formerly colliding path; unchanged paths need one reset.
        for _ in range(3 if cls is TASK_CLASSES[1] else 1):
            env.reset()
            controller = env.robots[0].composite_controller
            np.testing.assert_allclose(env.sim.data.qpos[controller.qpos_addrs], angles, atol=1e-8)
            assert env.count_body_collisions() == 0
            assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=.25)
            _assert_feet_touch_floor(env)
            assert env._routine_stage == 0 and not env._check_success()
            _assert_contacts(env, "SONIC reset")
            for name, pos in _centres(env).items():
                np.testing.assert_allclose(pos, initial[name], atol=1e-6)
        if cls is TASK_CLASSES[1]:
            np.testing.assert_allclose(env._robot_base_xy(), [-2.57432, 3.38], atol=1e-8)
            shelf = env._skills["soap"].multilevel_fixture
            assert env.robot_fixture_collision_distance(shelf, max_distance=5.) > .18
            # Original item placements, not objects relocated to hide a collision.
            for name, xy in (("bar_soap", [-2.57432, 3.942]), ("sponge", [-.689, 1.895])):
                np.testing.assert_allclose(env.object_pos(name)[:2], xy, atol=1e-6)
            np.testing.assert_allclose(env._skills["bottle"].source_fixture.world_center(env.sim)[:2],
                                       [-.78, .70], atol=1e-6)
            robot = _capture_robot_joint_state(env)
            drift = dict.fromkeys(initial, 0.)
            for _ in range(100):
                _advance(env, .1, robot)
                env._advance_routine()
                for name, pos in _centres(env).items():
                    drift[name] = max(drift[name], float(np.linalg.norm(pos - initial[name])))
            assert max(drift.values()) < .03, drift
            assert env._routine_stage == 0 and not env._check_success()
            _assert_contacts(env, "SONIC shelf routine after 10 s")
            print("SONIC_SHELF_DRIFT " + json.dumps(drift, sort_keys=True))
    finally:
        env.close()


def _drawer_extraction_geometry(env):
    """Sweep the real source collision model, not a simulated robot grasp."""
    source = env._skills["source"].source_fixture
    start = source.world_center(env.sim)[:2].copy()
    bottom = float(env._fixture_bbox_points(source)[:, 2].min())
    end = np.array([-.68, start[1]])
    # Raise the roll above the drawer front, pull it out below the upper
    # drawer handle, THEN finish the 30 cm lift. Straight up hits that handle.
    clearance = env._interior.world_pose(env.sim)[0][2] - env._interior.half_size[2] + .24
    segments = [(start, clearance), (end, clearance), (end, bottom + .31)]
    current_xy, current_z = start, bottom
    robot_bodies = env._body_tree_ids([env.sim.model.body_name2id("robot0_pelvis")])
    source_bodies = env._fixture_body_ids(source, include_siblings=True)
    for xy, z in segments:
        for fraction in np.linspace(0., 1., 41):
            env._set_fixture_center_xy_and_bbox_bottom(source,
                current_xy + fraction * (xy - current_xy), current_z + fraction * (z - current_z))
            for index in range(env.sim.data.ncon):
                contact = env.sim.data.contact[index]
                first, second = (int(env.sim.model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                if (first in source_bodies) == (second in source_bodies):
                    continue
                other = second if first in source_bodies else first
                if other not in robot_bodies:
                    assert contact.dist >= -.003, (fraction, contact.dist,
                        env.sim.model.geom_id2name(contact.geom1), env.sim.model.geom_id2name(contact.geom2))
        current_xy, current_z = xy, z
    assert env._outside_storage()


def _drawer_geometric_sequence(env, monkeypatch):
    """Real geometry and predicates, injected hand sensing; NOT a teleop grasp."""
    sensors = dict(touch=True, grasp=False)
    monkeypatch.setattr(env, "_stage_contact", lambda stage:
                        sensors["grasp"] if stage.startswith("lift_") else sensors["touch"])
    monkeypatch.setattr(env, "_source_grasped", lambda: sensors["grasp"])
    for _ in range(5):
        env._advance_routine()
    assert env._routine_stage == 1
    source = env._skills["source"].source_fixture
    points = env._fixture_bbox_points(source)
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    # Grasp near the front: part of the bbox leaves the semantic patch, but the
    # centre stays inside and the real collision geometry is clear.
    env._set_fixture_center_xy_and_bbox_bottom(source, center[:2] + [-.045, 0.],
                                              points[:, 2].min() + .015)
    assert not env._fixture_bbox_inside_region_xy(source, env._interior)  # old false negative
    assert env._source_grasp_is_at_start() and env._source_clear()
    env._advance_routine()
    assert not env._grasped_at_source  # moving alone is not a grasp
    sensors["grasp"] = True
    env._advance_routine()
    assert env._grasped_at_source
    _drawer_extraction_geometry(env)
    for _ in range(5):
        env._advance_routine()
    assert env._routine_stage == 2 and not env._check_success()
    sensors["touch"] = False
    joint = env._routine_joint
    for fraction in np.linspace(1., .19, 30):
        env.sim.data.set_joint_qpos(joint.name, joint.closed_qpos + fraction * (joint.open_qpos - joint.closed_qpos))
        env.sim.forward()
        assert env.count_body_collisions() == 0
        assert env._source_clear()
        env._advance_routine()
    assert not env._check_success()  # no closing interaction yet
    # It is fine to lower the held roll after completing its lift stage.
    env._set_fixture_center_xy_and_bbox_bottom(source, [-.68, center[1]], .60)
    assert not env._stage_geometry("lift_toilet_paper_roll")
    sensors["touch"] = True
    for _ in range(env.control_freq - 1):
        env._advance_routine()
    assert not env._check_success()
    sensors["grasp"] = False
    env._advance_routine()
    assert env._routine_hold == 0
    sensors["grasp"] = True
    for _ in range(env.control_freq):
        env._advance_routine()
    assert env._check_success()
    # Returned to the closed drawer is not an acceptable final state.
    inside = env._interior.world_pose(env.sim)[0]
    env._set_fixture_center_xy_and_bbox_bottom(source, inside[:2], inside[2] - env._interior.half_size[2])
    env._advance_routine()
    assert not env._check_success()


@pytest.mark.parametrize("cls", TASK_CLASSES, ids=lambda c: c.__name__)
def test_new_routine_initialization_stability_and_operating_poses(cls, monkeypatch):
    if cls is DRAWER_TASK:
        _sonic_spawn_stance(monkeypatch)
    env = robosuite.make(env_name=cls.__name__, robots="G1FloatingBody" if cls is DRAWER_TASK else "G1",
        controller_configs=_load_reset_controller(), has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False, ignore_done=True,
        hard_reset=False, control_freq=20, renderer="mujoco",
        scene_ids=cls.SUPPORTED_SCENE_IDS, seed=0)
    try:
        env.reset()
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=.25)
        assert env._routine_stage == 0 and not env._check_success()
        _assert_contacts(env, "routine reset")
        if cls is DRAWER_TASK:
            _assert_feet_touch_floor(env)
            assert env._skills["source"].work_surface_region.parent_body == env._interior.parent_body
            points = env._fixture_bbox_points(env._skills["source"].source_fixture)
            assert .60 < points[:, 2].min() < .62
            from test_ithor_bathroom_articulation_cycle_reset import _load_fixed_reset
            _load_fixed_reset()(env)
        robot = _capture_robot_joint_state(env)
        initial = _centres(env)
        initial_joint = env._routine_joint.fraction(env.sim) if env._routine_joint else None
        drift = {name: 0. for name in initial}
        # Ten seconds cover navigation waiting; each object remains fully free.
        for _ in range(100):
            _advance(env, .1, robot)
            env._advance_routine()
            for name, value in _centres(env).items():
                drift[name] = max(drift[name], float(np.linalg.norm(value - initial[name])))
            assert not env._check_success()
        assert max(drift.values()) < .03, drift
        assert env._routine_stage == 0
        waited_joint = env._routine_joint.fraction(env.sim) if env._routine_joint else None
        if initial_joint is not None:
            assert abs(waited_joint - initial_joint) < .01
        _assert_contacts(env, "after navigation wait")
        measurements = {}
        for name, skill in env._skills.items():
            pose = skill.spec.get("robot_pose")
            if pose:
                if "operating_open_fraction" in skill.spec:
                    joint = env._routine_joint
                    env.sim.data.set_joint_qpos(joint.name, joint.closed_qpos +
                        skill.spec["operating_open_fraction"] * (joint.open_qpos - joint.closed_qpos))
                    env.sim.forward()
                env._set_exact_robot_pose(pose, f"routine {name} operating pose")
                assert env.count_body_collisions() == 0
                if "source_fixture" in skill.__dict__:
                    measurements[name] = float(np.linalg.norm(skill.source_fixture.world_center(env.sim)[:2] - env._robot_base_xy()))
                if env._navigation_fixture is not None:
                    measurements[name + "_navigation_distance"] = float(env.robot_fixture_collision_distance(env._navigation_fixture, max_distance=5.))
        if env._routine_joint is not None:
            joint = env._routine_joint
            art = env.scene_task_spec()["articulation"]
            pose = art.get("robot_pose", art.get("close_robot_pose"))
            env._set_exact_robot_pose(pose, "routine joint operating pose")
            for fraction in np.linspace(.2, .8, 13):
                env.sim.data.set_joint_qpos(joint.name, joint.closed_qpos + fraction * (joint.open_qpos - joint.closed_qpos))
                env.sim.forward()
                assert env.count_body_collisions() == 0
            env.sim.data.set_joint_qpos(joint.name, joint.closed_qpos + art["init_open_fraction"] * (joint.open_qpos - joint.closed_qpos))
            env.sim.forward()
            if cls is DRAWER_TASK:
                # Prescribed drawer trajectory, freely simulated paper. This
                # checks support motion, NOT a robot-generated opening action.
                robot = _capture_robot_joint_state(env)
                _advance(env, 2., robot, joint_fraction=1.0)
                source = env._skills["source"].source_fixture
                assert env._fixture_bbox_inside_region_xy(source, env._interior)
                assert env._fixture_contacts_fixture(source, env._joint_fixture)
                parent = env.sim.model.body_name2id(env._interior.parent_body)
                assert env._fixture_contact_distances(source, env._body_tree_ids([parent]))
                assert not env._outside_storage()
                _drawer_geometric_sequence(env, monkeypatch)
                env.reset()  # actual collector closure restores geometry and progress
                assert env._routine_stage == env._routine_hold == 0
                assert not env._grasped_at_source and not env._check_success()
                assert env._routine_joint.fraction(env.sim) == pytest.approx(.2)
                _assert_contacts(env, "upper drawer fixed reset")
                _assert_feet_touch_floor(env)
        print("ROUTINE_PHYSICS " + json.dumps(dict(task=cls.__name__, max_drift_m=drift,
              operating_distances_m=measurements,
              initial_joint_fraction=initial_joint, waited_joint_fraction=waited_joint), sort_keys=True))
    finally:
        env.close()
