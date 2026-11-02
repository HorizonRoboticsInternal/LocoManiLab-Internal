"""Start the navigation close task fully open while preserving hamper physics."""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import robosuite

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_bedroom.tasks import (
    IThorBedroomApproachStorageAndCloseHamperLid,
)
from validate_ithor_tasks import _capture_robot_joint_state, _load_reset_controller


Task = IThorBedroomApproachStorageAndCloseHamperLid


def test_close_hamper_requires_navigation_before_closing_and_a_full_hold(monkeypatch):
    monkeypatch.setattr(ExactSceneArticulation, "_post_action", lambda self, action: None)
    monkeypatch.setattr(ExactSceneArticulation, "_reset_internal", lambda self: None)
    task = object.__new__(Task)
    task.control_freq = 20
    task._configure_success_hold_duration()
    assert task.HOLD_STEPS == 20
    state = {"fraction": 1.0, "distance": 0.10}
    task.sim = object()
    task.target_fixture = object()
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task.robot_fixture_collision_distance = lambda *args, **kwargs: state["distance"]
    task._report_stage_complete = lambda *args, **kwargs: None
    task._reset_internal()

    # Closing before arrival must not skip the navigation stage.
    state["fraction"] = 0.10
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 0
    assert not task._check_success()
    state["fraction"] = 1.0
    for _ in range(task.APPROACH_HOLD_STEPS - 1):
        task._post_action(None)
    assert task._approach_stage == 0
    state["distance"] = 0.50
    task._post_action(None)
    assert task._approach_hold == 0
    state["distance"] = 0.10
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 1

    state["fraction"] = 0.10
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    state["fraction"] = 0.13
    task._post_action(None)
    assert task._operation_hold == 0
    state["fraction"] = 0.10
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()
    state["distance"] = 0.50
    task._post_action(None)
    assert task._operation_hold == 0
    assert not task._check_success()
    state["fraction"] = 1.0
    task._reset_internal()
    assert task._approach_stage == task._approach_hold == task._operation_hold == 0


def test_306_hamper_waits_fully_open_then_closes_under_tangential_force():
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
        scene_ids=[306],
        seed=0,
    )
    try:
        env.reset()
        model, data = env.sim.model, env.sim.data
        joint = env.target_semantic_joint
        joint_id = model.joint_name2id(env.target_joint)
        dof = int(model.jnt_dofadr[joint_id])
        lid_body = int(model.jnt_bodyid[joint_id])
        assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        assert model.jnt_stiffness[joint_id] == 0.0
        np.testing.assert_allclose(model.dof_damping[dof], 0.1)
        np.testing.assert_allclose(model.dof_frictionloss[dof], 1.0)
        np.testing.assert_allclose(model.jnt_range[joint_id], [0.0, 1.5708])
        np.testing.assert_allclose(model.dof_solref[dof], [0.02, 1.0])
        np.testing.assert_allclose(model.dof_solimp[dof], [0.9, 0.95, 0.001, 0.5, 2.0])
        np.testing.assert_allclose(data.get_joint_qpos(env.target_joint), joint.open_qpos)
        assert data.qvel[dof] == 0.0
        assert env._approach_initial_fraction == 1.0
        np.testing.assert_allclose(env._robot_base_xy(), [2.85, 0.30])
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._approach_start_distance >= 2.40
        assert not env._near_articulation()
        assert not env._raw_success()
        assert not env._check_success()
        robot_state = _capture_robot_joint_state(env)

        def advance(seconds, closing_force=0.0, stop_fraction=None):
            # Freeze only the uncontrolled robot. Apply the diagnostic force
            # tangentially at the lid COM, starting horizontally at full open.
            start = data.time
            initial_qpos = float(data.get_joint_qpos(env.target_joint))
            max_qpos_drift = 0.0
            try:
                while data.time - start < seconds:
                    axis = np.asarray(data._data.xaxis[joint_id])
                    radius = (
                        np.asarray(data._data.xipos[lid_body])
                        - np.asarray(data._data.xanchor[joint_id])
                    )
                    tangent = -np.cross(axis, radius)
                    data.xfrc_applied[lid_body, :3] = (
                        closing_force * tangent / np.linalg.norm(tangent)
                    )
                    env.sim.step()
                    for qs, qpos, vs, qvel in robot_state:
                        data.qpos[qs] = qpos
                        data.qvel[vs] = qvel
                    env.sim.forward()
                    max_qpos_drift = max(
                        max_qpos_drift,
                        abs(float(data.get_joint_qpos(env.target_joint)) - initial_qpos),
                    )
                    if stop_fraction is not None and joint.fraction(env.sim) <= stop_fraction:
                        break
            finally:
                data.xfrc_applied[lid_body, :3] = 0.0
            return float(data.time - start), float(max_qpos_drift)

        wait_seconds, max_qpos_drift = advance(20.0)
        waiting_fraction = float(joint.fraction(env.sim))
        # Check unclipped qpos as well, so the semantic fraction cannot mask
        # drift beyond the fully-open joint limit.
        assert max_qpos_drift / abs(joint.open_qpos - joint.closed_qpos) < 0.01
        assert waiting_fraction >= 0.99
        assert env._initial_articulation_intact()
        assert not env._check_success()
        assert env.count_body_collisions() == 0
        assert env._finalize_supported_articulation_robot_pose(0.45)
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._near_articulation()
        robot_state = _capture_robot_joint_state(env)
        for step in range(env.APPROACH_HOLD_STEPS):
            advance(1.0 / env.control_freq)
            env._post_action(np.zeros(env.action_dim))
            assert env._approach_stage == (0 if step < 4 else 1)

        force_seconds, _ = advance(2.0, closing_force=15.0, stop_fraction=0.10)
        closed_fraction = float(joint.fraction(env.sim))
        assert env._raw_success()
        assert env.HOLD_STEPS == 20
        for step in range(env.HOLD_STEPS):
            advance(1.0 / env.control_freq)
            env._post_action(np.zeros(env.action_dim))
            assert env._check_success() == (step == env.HOLD_STEPS - 1)
        assert env.count_body_collisions() == 0
        print(json.dumps({
            "scene": 306,
            "wait_simulated_seconds": wait_seconds,
            "max_wait_qpos_drift_radians": max_qpos_drift,
            "waiting_fraction": waiting_fraction,
            "force_seconds": force_seconds,
            "closed_fraction": closed_fraction,
            "final_fraction": float(joint.fraction(env.sim)),
            "success_hold_steps": env._operation_hold,
            "control_freq": env.control_freq,
            "success": env._check_success(),
        }, sort_keys=True))
    finally:
        env.close()
