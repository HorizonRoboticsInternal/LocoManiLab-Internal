"""Ordered Bedroom cycles must restart after restoring a fixed scene."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace
import types

import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_bedroom.tasks import (
    IThorBedroomOpenThenCloseBoxFlap,
    IThorBedroomOpenThenCloseDrawer,
    IThorBedroomOpenThenCloseHamperLid,
    IThorBedroomOpenThenCloseLaptop,
    IThorBedroomOpenThenCloseSafeDoor,
    IThorBedroomOpenThenCloseTopDrawer,
    IThorBedroomOpenHamperLid,
)
from robocasa.environments.ithor_bedroom.tasks.articulation_composites import (
    _OpenThenCloseArticulationMixin,
    _ResettableOpenThenCloseArticulationMixin,
)
from robocasa.environments.ithor_bedroom.tasks.hamper_lid import _FloorHamperMixin
from validate_ithor_tasks import _capture_robot_joint_state, _load_reset_controller


TASKS = (
    IThorBedroomOpenThenCloseDrawer,
    IThorBedroomOpenThenCloseTopDrawer,
    IThorBedroomOpenThenCloseSafeDoor,
    IThorBedroomOpenThenCloseHamperLid,
)


def _load_fixed_reset():
    # Exercise the collector's actual reset closure without importing its GUI
    # and hardware backend. Controller/observation refresh is outside this test.
    path = Path(__file__).parents[1] / "robocasa/scripts/collect_fixed_task.py"
    module = ast.parse(path.read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "freeze_resets"
    )
    namespace = {
        "np": np,
        "types": types,
        "colored": lambda message, color: message,
        "_reset_controllers": lambda env: None,
        "_refresh_after_restore": lambda env: None,
        "apply_reset_pose_jitter": lambda *args, **kwargs: {
            "g1": np.zeros(2),
            "g1_ok": True,
        },
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["freeze_resets"]


def _make_task(monkeypatch, task_class, control_freq):
    monkeypatch.setattr(
        ExactSceneArticulation, "_post_action", lambda self, action: None
    )
    monkeypatch.setattr(
        ExactSceneArticulation,
        "_reset_internal",
        lambda self: setattr(self, "_success_hold", 0),
    )
    task = object.__new__(task_class)
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    joint = SimpleNamespace(closed_qpos=0.0, open_qpos=1.5708)
    initial = task._initial_joint_qpos(joint) / joint.open_qpos
    state = {"fraction": initial}
    task.sim = SimpleNamespace(
        get_state=lambda: state["fraction"],
        set_state=lambda fraction: state.update(fraction=fraction),
        forward=lambda: None,
    )
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task.robot_base_pos = np.zeros(3)
    task.robot_base_ori = np.zeros(3)
    task.reset = lambda: None
    task._get_observations = lambda **kwargs: {"success": task._check_success()}
    task._reset_internal()
    return task, state, initial


def _open(task, state):
    state["fraction"] = 0.90
    for _ in range(task.FIRST_HOLD_STEPS - 1):
        task._post_action(None)
    assert task._cycle_stage == 0
    task._post_action(None)
    assert task._cycle_stage == 1
    assert not task._check_success()


def _close(task, state):
    state["fraction"] = 0.10
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()


@pytest.mark.parametrize("task_class", TASKS)
@pytest.mark.parametrize("phase", ("opened", "completed"))
@pytest.mark.parametrize("control_freq", (10, 20))
def test_fixed_reset_clears_cycle_and_requires_a_new_open(
    monkeypatch, capsys, task_class, phase, control_freq
):
    task, state, initial = _make_task(monkeypatch, task_class, control_freq)
    _load_fixed_reset()(task)
    for _ in range(2):
        _open(task, state)
        assert capsys.readouterr().out.count("stage 1/2") == 1
        if phase == "completed":
            _close(task, state)
        task._success_hold = task.HOLD_STEPS
        observations = task.reset()
        assert state["fraction"] == initial
        assert observations == {"success": False}
        assert task._success_hold == task._cycle_stage == 0
        assert task._cycle_open_hold == task._cycle_close_hold == 0

        # Returning to a closed initial state must not complete an old cycle.
        state["fraction"] = 0.10
        for _ in range(task.HOLD_STEPS + 1):
            task._post_action(None)
            assert not task._check_success()
        assert task._cycle_stage == 0
        capsys.readouterr()


@pytest.mark.parametrize("task_class", TASKS)
def test_normal_reset_and_interrupted_final_hold(monkeypatch, task_class):
    task, state, initial = _make_task(monkeypatch, task_class, 20)
    _open(task, state)
    state["fraction"] = 0.10
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    state["fraction"] = 0.50
    task._post_action(None)
    assert task._cycle_close_hold == 0
    assert not task._check_success()
    _close(task, state)
    task._success_hold = task.HOLD_STEPS
    state["fraction"] = initial
    task._reset_internal()
    assert task._success_hold == task._cycle_stage == 0
    assert task._cycle_open_hold == task._cycle_close_hold == 0
    assert not task._check_success()


def test_fixed_reset_hook_is_limited_to_requested_cycles():
    for task_class in (*TASKS, IThorBedroomOpenThenCloseBoxFlap):
        assert issubclass(task_class, _ResettableOpenThenCloseArticulationMixin)
        assert (
            task_class._check_success is _OpenThenCloseArticulationMixin._check_success
        )
    for task_class in (IThorBedroomOpenThenCloseLaptop,):
        assert not issubclass(task_class, _ResettableOpenThenCloseArticulationMixin)
        assert not hasattr(task_class, "_reset_task_progress")


def test_hamper_cycle_stabilization_only_changes_its_target_dof(monkeypatch):
    task = object.__new__(IThorBedroomOpenThenCloseHamperLid)
    task.scene_name = "FloorPlan306"
    monkeypatch.setattr(_FloorHamperMixin, "_setup_scene", lambda self: None)
    task.target_joint = "lid"
    expected_ref = np.tile([0.02, 1.0], (4, 1))
    expected_imp = np.tile([0.9, 0.95, 0.001, 0.5, 2.0], (4, 1))
    expected_velocity = np.ones(4)
    task.sim = SimpleNamespace(
        model=SimpleNamespace(
            joint_name2id=lambda name: 1 if name == "lid" else -1,
            jnt_dofadr=np.array([0, 2]),
            dof_solref=expected_ref.copy(),
            dof_solimp=expected_imp.copy(),
        ),
        data=SimpleNamespace(qvel=expected_velocity.copy()),
        forward=lambda: None,
    )
    task._setup_scene()
    expected_ref[2] = [0.002, 1.0]
    expected_imp[2] = [0.9999, 0.9999, 0.001, 0.5, 2.0]
    expected_velocity[2] = 0.0
    np.testing.assert_array_equal(task.sim.model.dof_solref, expected_ref)
    np.testing.assert_array_equal(task.sim.model.dof_solimp, expected_imp)
    np.testing.assert_array_equal(task.sim.data.qvel, expected_velocity)


@pytest.mark.parametrize("scene_id", (302, 310, 323, 329))
def test_spawned_hamper_cycle_uses_atomic_initial_configuration(scene_id):
    scene = f"FloorPlan{scene_id}"
    task = object.__new__(IThorBedroomOpenThenCloseHamperLid)
    task.scene_name = scene
    atomic = IThorBedroomOpenHamperLid.SCENE_SPECS[scene]
    cycle = task.scene_task_spec()
    assert {key: value for key, value in cycle.items() if key != "lang"} == {
        key: value for key, value in atomic.items() if key != "lang"
    }
    assert cycle["fixture_mode"] == "spawn_asset"
    assert cycle["hamper_robot_pose"] == atomic["hamper_robot_pose"]
    joint = SimpleNamespace(closed_qpos=0.0, open_qpos=np.pi / 2)
    assert task._initial_joint_qpos(joint) == pytest.approx(
        IThorBedroomOpenHamperLid._initial_joint_qpos(task, joint)
    )


@pytest.mark.parametrize("scene_id", (302, 310, 323, 329))
def test_spawned_hamper_cycle_preserves_atomic_lid_physics(monkeypatch, scene_id):
    task = object.__new__(IThorBedroomOpenThenCloseHamperLid)
    task.scene_name = f"FloorPlan{scene_id}"
    calls = []
    monkeypatch.setattr(
        _FloorHamperMixin, "_setup_scene", lambda self: calls.append("setup")
    )
    task.sim = SimpleNamespace(
        model=SimpleNamespace(
            joint_name2id=lambda name: pytest.fail("spawned lid joint was recalibrated")
        ),
        forward=lambda: pytest.fail("spawned lid physics was modified"),
    )
    task._setup_scene()
    assert calls == ["setup"]


def test_hamper_cycle_does_not_close_itself_and_restarts_after_success():
    env = robosuite.make(
        env_name=IThorBedroomOpenThenCloseHamperLid.__name__,
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
        joint_id = model.joint_name2id(joint.name)
        dof = int(model.jnt_dofadr[joint_id])
        lid_body = int(model.jnt_bodyid[joint_id])
        assert data.get_joint_qpos(joint.name) == pytest.approx(np.pi / 6)
        assert model.dof_frictionloss[dof] == pytest.approx(1.0)
        assert model.dof_damping[dof] == pytest.approx(0.1)
        assert model.jnt_stiffness[joint_id] == 0.0
        np.testing.assert_allclose(model.jnt_range[joint_id], [0.0, 1.5708])
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert not env._check_success()
        _load_fixed_reset()(env)
        robot_state = _capture_robot_joint_state(env)

        def advance_step(closing_torque=0.0):
            # Hold only the uncontrolled robot, leaving the lid free to move.
            try:
                for _ in range(round(1.0 / env.control_freq / model.opt.timestep)):
                    data.xfrc_applied[lid_body, 3:] = -closing_torque * np.asarray(
                        data._data.xaxis[joint_id]
                    )
                    env.sim.step()
                    for qs, qpos, vs, qvel in robot_state:
                        data.qpos[qs] = qpos
                        data.qvel[vs] = qvel
                    env.sim.forward()
            finally:
                data.xfrc_applied[lid_body, 3:] = 0.0
            env._post_action(np.zeros(env.action_dim))

        for _ in range(env.control_freq):
            advance_step()
            assert env._cycle_stage == 0
            assert not env._check_success()

        # Reproduce releasing the lid at 85%, where the original task drifted
        # into success after 38.65 seconds without a closing interaction.
        data.set_joint_qpos(
            joint.name, joint.closed_qpos + 0.85 * (joint.open_qpos - joint.closed_qpos)
        )
        data.qvel[dof] = 0.0
        env.sim.forward()
        for step in range(env.FIRST_HOLD_STEPS):
            advance_step()
            assert env._cycle_stage == (0 if step < 4 else 1)
        start = data.time
        max_drift = 0.0
        for _ in range(40 * env.control_freq):
            advance_step()
            max_drift = max(max_drift, abs(joint.fraction(env.sim) - 0.85))
            assert env._cycle_stage == 1
            assert env._cycle_close_hold == 0
            assert not env._check_success()
        wait_seconds = float(data.time - start)
        assert max_drift < 0.001
        waiting_fraction = float(joint.fraction(env.sim))

        for _ in range(2 * env.control_freq):
            advance_step(closing_torque=2.0)
            if env._joint_matches("close"):
                break
        assert env._joint_matches("close")
        assert env.HOLD_STEPS == 20
        for _ in range(env.HOLD_STEPS - env._cycle_close_hold):
            assert not env._check_success()
            advance_step()
        assert env._check_success()
        assert env.count_body_collisions() == 0

        # Restore the actual initial MuJoCo state without rebuilding the model.
        env.reset()
        assert data.get_joint_qpos(joint.name) == pytest.approx(np.pi / 6)
        assert env._cycle_stage == env._cycle_open_hold == env._cycle_close_hold == 0
        assert env._success_hold == 0
        assert not env._check_success()
        print(
            json.dumps(
                {
                    "scene": 306,
                    "wait_simulated_seconds": wait_seconds,
                    "waiting_fraction": waiting_fraction,
                    "max_wait_fraction_drift": max_drift,
                    "close_then_hold_success": True,
                    "success_after_fixed_reset": env._check_success(),
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()
