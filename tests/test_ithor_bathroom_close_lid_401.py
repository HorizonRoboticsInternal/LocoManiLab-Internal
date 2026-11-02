"""Keep the 401 navigation lid stable but pressable without changing atomics."""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_bathroom.tasks import (
    IThorBathroomApproachToiletAndCloseToiletLid,
    IThorBathroomApproachToiletAndOpenToiletLid,
    IThorBathroomCloseThenOpenToiletLid,
    IThorBathroomCloseToiletLid,
    IThorBathroomOpenToiletLid,
)
from robocasa.environments.ithor_bathroom.tasks.articulation import (
    _BathroomToiletLid,
)
from robocasa.environments.ithor_bathroom.tasks.composite.approach_manipulation import (
    _ApproachThenArticulationMixin,
)
from validate_ithor_tasks import _capture_robot_joint_state, _load_reset_controller


Task = IThorBathroomApproachToiletAndCloseToiletLid


@pytest.mark.parametrize("scene_id", (401, 405, 407, 415, 418))
def test_close_lid_calibration_only_changes_401_target_dof(monkeypatch, scene_id):
    task = object.__new__(Task)
    task.scene_name = f"FloorPlan{scene_id}"
    task.target_joint = "lid"
    # A previous 401 episode must not leak its instance setting into a new scene.
    task.CLOSE_START_FRACTION = 0.80
    default_ref = np.tile([0.02, 1.0], (3, 1))
    default_imp = np.tile([0.9, 0.95, 0.001, 0.5, 2.0], (3, 1))
    model = SimpleNamespace(
        joint_name2id=lambda name: 1 if name == "lid" else -1,
        jnt_dofadr=np.arange(3),
        dof_solref=default_ref.copy(),
        dof_solimp=default_imp.copy(),
    )
    task.sim = SimpleNamespace(model=model, forward=lambda: None)
    initial_qpos = []

    def parent_setup(self):
        joint = SimpleNamespace(closed_qpos=0.0, open_qpos=-1.51844)
        initial_qpos.append(self._initial_joint_qpos(joint))

    monkeypatch.setattr(_ApproachThenArticulationMixin, "_setup_scene", parent_setup)
    task._setup_scene()
    expected_fraction = 0.80 if scene_id == 401 else 1.0
    assert task.CLOSE_START_FRACTION == expected_fraction
    assert initial_qpos == pytest.approx([-1.51844 * expected_fraction])
    if scene_id == 401:
        default_ref[1] = [0.002, 1.0]
        default_imp[1] = [0.9999, 0.9999, 0.001, 0.5, 2.0]
    np.testing.assert_array_equal(model.dof_solref, default_ref)
    np.testing.assert_array_equal(model.dof_solimp, default_imp)
    assert Task.CLOSE_START_FRACTION == 1.0


@pytest.mark.parametrize(
    "task_class, initial_fraction",
    (
        (IThorBathroomOpenToiletLid, 0.20),
        (IThorBathroomCloseToiletLid, 0.80),
        (IThorBathroomApproachToiletAndOpenToiletLid, 0.20),
        (IThorBathroomCloseThenOpenToiletLid, 1.00),
    ),
)
def test_other_toilet_lid_tasks_keep_their_initialization(task_class, initial_fraction):
    task = object.__new__(task_class)
    task.scene_name = "FloorPlan401"
    joint = SimpleNamespace(closed_qpos=0.0, open_qpos=-1.51844)
    assert task._initial_joint_qpos(joint) == pytest.approx(-1.51844 * initial_fraction)
    assert task_class._setup_scene is not Task._setup_scene
    assert task_class._raw_success is Task._raw_success
    assert task_class.SUCCESS_CLOSE_FRACTION == 0.20
    assert task_class.SCENE_SPECS["FloorPlan401"]["close_robot_pose"] == {
        "xy": [0.116, 2.494], "yaw": 1.570796
    }


def test_close_lid_cannot_skip_navigation_or_interrupted_final_hold(monkeypatch):
    monkeypatch.setattr(_BathroomToiletLid, "_post_action", lambda self, action: None)
    monkeypatch.setattr(_BathroomToiletLid, "_reset_internal", lambda self: None)
    task = object.__new__(Task)
    task.HOLD_STEPS = 20
    task._capture_operation_initial_state = lambda: 0.80
    task._report_stage_complete = lambda *args, **kwargs: None
    task._reset_internal()
    state = {"near": True, "intact": False, "closed": True}
    task._near_approach_fixture = lambda: state["near"]
    task._operation_initial_state_intact = lambda: state["intact"]
    task._raw_success = lambda: state["closed"]

    # Closing first must not bypass navigation, even if held for a full second.
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 0
    assert not task._check_success()
    state.update(near=True, intact=True, closed=False)
    for _ in range(4):
        task._post_action(None)
    assert task._approach_stage == 0
    state["near"] = False
    task._post_action(None)
    assert task._approach_hold == 0
    state["near"] = True
    for _ in range(5):
        task._post_action(None)
    assert task._approach_stage == 1

    state.update(near=False, closed=True)
    for _ in range(19):
        task._post_action(None)
    assert not task._check_success()
    state["closed"] = False
    task._post_action(None)
    assert task._composite_operation_hold == 0
    state["closed"] = True
    for _ in range(20):
        task._post_action(None)
    assert task._check_success()
    task._reset_internal()
    assert task._approach_stage == task._approach_hold == 0
    assert task._composite_operation_hold == 0
    assert not task._check_success()


def test_401_lid_waits_twenty_seconds_then_closes_under_downward_force():
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
        scene_ids=[401],
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
        assert model.dof_damping[dof] == pytest.approx(0.5)
        assert model.dof_frictionloss[dof] == pytest.approx(0.6)
        assert joint.fraction(env.sim) == pytest.approx(0.80)
        assert env.get_ep_meta()["initial_open_fraction"] == 0.80
        np.testing.assert_allclose(model.jnt_range[joint_id], [-1.51844, -0.303688])
        np.testing.assert_allclose(model.dof_solref[dof], [0.002, 1.0])
        np.testing.assert_allclose(model.dof_solimp[dof], [0.9999, 0.9999, 0.001, 0.5, 2.0])
        np.testing.assert_allclose(env._robot_base_xy(), [-2.432, 0.135])
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert not env._near_approach_fixture()
        assert not env._check_success()
        assert not env._raw_success()
        robot_state = _capture_robot_joint_state(env)

        def advance(seconds, downforce=0.0):
            # Freeze only the uncontrolled robot, not the lid. The synthetic
            # force acts downward at the lid COM; it is not a teleop grasp.
            start = data.time
            initial = float(joint.fraction(env.sim))
            max_change = 0.0
            try:
                while data.time - start < seconds:
                    data.xfrc_applied[lid_body, :3] = [0.0, 0.0, -downforce]
                    env.sim.step()
                    for qs, qpos, vs, qvel in robot_state:
                        data.qpos[qs] = qpos
                        data.qvel[vs] = qvel
                    env.sim.forward()
                    max_change = max(max_change, abs(joint.fraction(env.sim) - initial))
            finally:
                data.xfrc_applied[lid_body, :3] = 0.0
            return float(data.time - start), float(max_change)

        elapsed, max_drift = advance(20.0)
        waiting_fraction = float(joint.fraction(env.sim))
        assert max_drift < env.INITIAL_STATE_TOLERANCE
        assert env._operation_initial_state_intact()
        assert not env._raw_success()
        assert env.count_body_collisions() == 0
        assert env._finalize_articulation_robot_pose(0.45)
        np.testing.assert_allclose(env._robot_base_xy(), [0.116, 2.494])
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._robot_clears_lid_sweep()
        assert env._near_approach_fixture()
        robot_state = _capture_robot_joint_state(env)
        for step in range(5):
            advance(1.0 / env.control_freq)
            env._post_action(np.zeros(env.action_dim))
            assert env._approach_stage == (0 if step < 4 else 1)

        advance(2.0, downforce=10.0)
        pressed_fraction = float(joint.fraction(env.sim))
        assert env._raw_success()
        assert env.HOLD_STEPS == 20
        assert env.HOLD_STEPS / env.control_freq == 1.0
        for step in range(env.HOLD_STEPS):
            advance(1.0 / env.control_freq)
            env._post_action(np.zeros(env.action_dim))
            assert env._check_success() == (step == env.HOLD_STEPS - 1)
        assert env.count_body_collisions() == 0
        print(json.dumps({
            "scene": 401,
            "wait_simulated_seconds": elapsed,
            "max_wait_fraction_drift": max_drift,
            "waiting_fraction": waiting_fraction,
            "pressed_fraction": pressed_fraction,
            "final_fraction": float(joint.fraction(env.sim)),
            "success_hold_steps": env._composite_operation_hold,
            "control_freq": env.control_freq,
            "success": env._check_success(),
        }, sort_keys=True))
    finally:
        env.close()
