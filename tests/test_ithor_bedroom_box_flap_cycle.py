"""A box-flap cycle needs closing travel as well as the atomic opening travel."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_bedroom.tasks import (
    IThorBedroomApproachStorageAndCloseBoxFlap,
    IThorBedroomApproachStorageAndOpenBoxFlap,
    IThorBedroomCloseBoxFlap,
    IThorBedroomOpenBoxFlap,
    IThorBedroomOpenThenCloseBoxFlap,
)
from validate_ithor_tasks import _capture_robot_joint_state, _load_reset_controller


Task = IThorBedroomOpenThenCloseBoxFlap


@pytest.mark.parametrize(
    "scene_name",
    ("FloorPlan302", "FloorPlan306", "FloorPlan310", "FloorPlan323", "FloorPlan329"),
)
def test_full_closing_travel_is_isolated_to_the_cycle(scene_name):
    protected = (
        IThorBedroomOpenBoxFlap,
        IThorBedroomCloseBoxFlap,
        IThorBedroomApproachStorageAndOpenBoxFlap,
        IThorBedroomApproachStorageAndCloseBoxFlap,
    )
    for task_class in (Task, *protected):
        if scene_name not in task_class.SCENE_SPECS:
            continue
        task = object.__new__(task_class)
        task.scene_name = scene_name
        spec = task.scene_task_spec()
        limits = np.array(
            [[-3.0, 1.0], [spec["closed_qpos"], spec["open_qpos"]], [-2.0, 2.0]]
        )
        original = limits.copy()
        task.sim = SimpleNamespace(
            model=SimpleNamespace(
                joint_name2id=lambda name: 1,
                jnt_limited=np.ones(3, dtype=int),
                jnt_range=limits,
            )
        )
        joint = SimpleNamespace(
            name=spec["joint"],
            range=tuple(original[1]),
            closed_qpos=0.0,
            open_qpos=spec["closed_qpos"],
        )
        task._configure_articulation_joint(joint)
        if task_class is not Task and task.BEHAVIOR == "open":
            original[1, 0] = spec["open_start_qpos"]
        np.testing.assert_array_equal(limits, original)
        assert joint.range == tuple((spec["closed_qpos"], spec["open_qpos"]))
        assert joint.closed_qpos == spec["closed_qpos"]
        assert joint.open_qpos == spec["open_qpos"]
        if task.BEHAVIOR == "open":
            assert task._initial_joint_qpos(joint) == spec["open_start_qpos"]
        if task_class is Task:
            assert task.SUCCESS_OPEN_FRACTION == 0.80
            assert task.SUCCESS_CLOSE_FRACTION == 0.12
            closed_threshold = joint.closed_qpos + 0.12 * (
                joint.open_qpos - joint.closed_qpos
            )
            assert limits[1, 0] <= closed_threshold <= limits[1, 1]
            assert closed_threshold < spec["open_start_qpos"]


@pytest.mark.parametrize("control_freq", (10, 20))
def test_box_cycle_order_hold_interruptions_and_reset(monkeypatch, control_freq):
    monkeypatch.setattr(
        ExactSceneArticulation, "_post_action", lambda self, action: None
    )
    monkeypatch.setattr(
        ExactSceneArticulation,
        "_reset_internal",
        lambda self: setattr(self, "_success_hold", 0),
    )
    task = object.__new__(Task)
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    state = {"fraction": 0.0}
    task.sim = SimpleNamespace()
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task._reset_internal()
    assert task.HOLD_STEPS == control_freq

    def advance(fraction, steps):
        state["fraction"] = fraction
        for _ in range(steps):
            task._post_action(None)

    advance(0.10, task.HOLD_STEPS)
    assert task._cycle_stage == 0
    assert not task._check_success()
    advance(0.85, task.FIRST_HOLD_STEPS - 1)
    assert task._cycle_stage == 0
    advance(0.50, 1)
    assert task._cycle_open_hold == 0
    advance(0.85, task.FIRST_HOLD_STEPS)
    assert task._cycle_stage == 1
    advance(0.10, task.HOLD_STEPS - 1)
    assert not task._check_success()
    advance(0.50, 1)
    assert task._cycle_close_hold == 0
    advance(0.10, task.HOLD_STEPS)
    assert task._check_success()

    for reset in (task._reset_task_progress, task._reset_internal):
        task._success_hold = task._cycle_close_hold = task.HOLD_STEPS
        task._cycle_stage = 1
        task._cycle_open_hold = task.FIRST_HOLD_STEPS
        reset()
        assert task._success_hold == task._cycle_stage == 0
        assert task._cycle_open_hold == task._cycle_close_hold == 0
        advance(0.10, task.HOLD_STEPS)
        assert not task._check_success()


def test_box_cycle_329_physically_opens_then_closes():
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
        scene_ids=[329],
        seed=0,
    )
    try:
        env.reset()
        model, data = env.sim.model, env.sim.data
        joint = env.target_semantic_joint
        joint_id = model.joint_name2id(joint.name)
        dof = int(model.jnt_dofadr[joint_id])
        flap_body = int(model.jnt_bodyid[joint_id])
        np.testing.assert_allclose(model.jnt_range[joint_id], [-2.74176, 0.0])
        assert data.get_joint_qpos(joint.name) == pytest.approx(-1.1691420089)
        assert model.dof_frictionloss[dof] == pytest.approx(0.01)
        assert model.dof_damping[dof] == pytest.approx(0.01)
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert not env._check_success()
        assert data.eq_active[env._box_base_weld_id()]
        initial = env.sim.get_state()
        robot_state = _capture_robot_joint_state(env)

        def advance(torque=0.0):
            # Diagnostic hinge torque only; the robot is held at reset and
            # the flap and remaining box bodies participate in physics.
            try:
                for _ in range(round(1.0 / env.control_freq / model.opt.timestep)):
                    data.xfrc_applied[flap_body, 3:] = torque * np.asarray(
                        data._data.xaxis[joint_id]
                    )
                    env.sim.step()
                    for qs, qpos, vs, qvel in robot_state:
                        data.qpos[qs] = qpos
                        data.qvel[vs] = qvel
                    env.sim.forward()
            finally:
                data.xfrc_applied[flap_body, 3:] = 0.0
            env._post_action(np.zeros(env.action_dim))

        for _ in range(3 * env.control_freq):
            advance()
            assert env._cycle_stage == 0
            assert not env._check_success()
        waiting_fraction = float(joint.fraction(env.sim))
        assert abs(waiting_fraction - 0.573579741151669) < 0.01

        for _ in range(3 * env.control_freq):
            advance(torque=0.10)
            if joint.fraction(env.sim) >= 0.85:
                break
        assert joint.fraction(env.sim) >= 0.85
        for _ in range(env.control_freq):
            advance()
            assert not env._check_success()
        assert env._cycle_stage == 1
        opened_fraction = float(joint.fraction(env.sim))

        for _ in range(3 * env.control_freq):
            advance(torque=-0.10)
            if joint.fraction(env.sim) <= 0.10:
                break
        assert joint.fraction(env.sim) <= 0.10
        for _ in range(env.control_freq):
            advance()
            assert env._joint_matches("close")
        assert env._check_success()
        assert env.count_body_collisions() == 0
        closed_fraction = float(joint.fraction(env.sim))
        np.testing.assert_allclose(
            data.body_xpos[model.body_name2id(env.target_fixture.body_name)],
            env._fixed_box_base_position,
            atol=0.0025,
        )

        # Fixed-scene tools restore data without rebuilding the model.
        env.sim.set_state(initial)
        env.sim.forward()
        env._reset_task_progress()
        assert not env._check_success()
        assert env._cycle_stage == env._cycle_open_hold == env._cycle_close_hold == 0
        np.testing.assert_allclose(model.jnt_range[joint_id], [-2.74176, 0.0])
        print(
            json.dumps(
                {
                    "scene": 329,
                    "waiting_fraction": waiting_fraction,
                    "opened_fraction": opened_fraction,
                    "closed_fraction": closed_fraction,
                    "diagnostic_torque_nm": 0.10,
                    "close_then_hold_success": True,
                    "success_after_fixed_restore": env._check_success(),
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()
