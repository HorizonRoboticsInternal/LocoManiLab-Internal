"""Bathroom cycles must not inherit success when a fixed episode is restored."""

import ast
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import types

import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_bathroom import tasks as bathroom_tasks
from robocasa.environments.ithor_bathroom.tasks.articulation import _BathroomToiletLid
from robocasa.environments.ithor_bathroom.tasks.composite.articulation_cycles import (
    _BathroomArticulationCycleMixin,
)
from robocasa.environments.ithor_bathroom.tasks.storage_articulation import (
    _BathroomStorageArticulation,
)
from robosuite.environments.base import REGISTERED_ENVS
from validate_ithor_tasks import _capture_robot_joint_state, _load_reset_controller


SCENES_BY_TARGET = {
    "ToiletLid": (401, 405, 407, 415, 418),
    "ShowerDoor": (407, 418),
    "CabinetDoor": (418,),
    "LowerDrawer": (415,),
}
TASKS = tuple(
    getattr(bathroom_tasks, f"IThorBathroom{order}{target}")
    for target in SCENES_BY_TARGET
    for order in ("OpenThenClose", "CloseThenOpen")
)
PROGRESS_FIELDS = (
    "_cycle_stage", "_first_state_hold", "_second_state_hold", "_success_hold",
)


def _load_fixed_reset():
    # Execute the collector's actual closure without importing its GUI/hardware
    # backend. Controller, observation refresh and XY jitter are out of scope.
    path = Path(__file__).parents[1] / "robocasa/scripts/collect_fixed_task.py"
    module = ast.parse(path.read_text())
    function = next(
        node for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "freeze_resets"
    )
    namespace = {
        "np": np,
        "types": types,
        "colored": lambda message, color: message,
        "_reset_controllers": lambda env: None,
        "_refresh_after_restore": lambda env: None,
        "apply_reset_pose_jitter": lambda *args, **kwargs: {
            "g1": np.zeros(2), "g1_ok": True,
        },
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace["freeze_resets"]


def _make_task(monkeypatch, task_class, control_freq):
    monkeypatch.setattr(
        ExactSceneArticulation, "_post_action", lambda self, action: None,
    )
    monkeypatch.setattr(
        ExactSceneArticulation, "_reset_internal",
        lambda self: setattr(self, "_success_hold", 0),
    )
    task = object.__new__(task_class)
    task.scene_name = f"FloorPlan{task_class.SUPPORTED_SCENE_IDS[0]}"
    spec = task.scene_task_spec()
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    if isinstance(task, _BathroomToiletLid):
        initial = (
            task.OPEN_START_FRACTION if task.FIRST_BEHAVIOR == "open"
            else task.CLOSE_START_FRACTION
        )
    else:
        initial = float(spec["init_open_fraction"])
        task.SUCCESS_OPEN_FRACTION = float(spec["success_open_fraction"])
        task.SUCCESS_CLOSE_FRACTION = float(spec["success_close_fraction"])
    state = {"fraction": initial}
    for cls, method in (
        (_BathroomToiletLid, "_reset_toilet_lid_pose"),
        (_BathroomStorageArticulation, "_reset_target_joint_pose"),
    ):
        monkeypatch.setattr(cls, method, lambda self: state.update(fraction=initial))
    task.sim = SimpleNamespace(
        get_state=lambda: state["fraction"],
        set_state=lambda fraction: state.update(fraction=fraction),
        forward=lambda: None,
    )
    joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task._success_joints = lambda: [joint]
    task.robot_base_pos = np.zeros(3)
    task.robot_base_ori = np.zeros(3)
    task.reset = task._reset_internal
    task._get_observations = lambda **kwargs: {"success": task._check_success()}
    task._reset_internal()
    return task, state, initial


def _fraction_for(task, behavior):
    if behavior == "open":
        return min(1.0, task.SUCCESS_OPEN_FRACTION + 0.01)
    return max(0.0, task.SUCCESS_CLOSE_FRACTION - 0.01)


def _step_at(task, state, behavior, steps):
    state["fraction"] = _fraction_for(task, behavior)
    for _ in range(steps):
        task._post_action(None)


def _finish_first(task, state):
    _step_at(task, state, task.FIRST_BEHAVIOR, task.FIRST_HOLD_STEPS - 1)
    assert task._cycle_stage == 0
    task._post_action(None)
    assert task._cycle_stage == 1
    assert task._second_state_hold == 0
    assert not task._check_success()


def _finish_second(task, state):
    _step_at(task, state, task.SECOND_BEHAVIOR, task.HOLD_STEPS - 1)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()


def _assert_clean(task):
    assert all(getattr(task, name) == 0 for name in PROGRESS_FIELDS)
    assert not task._check_success()


@pytest.mark.parametrize("task_class", TASKS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("control_freq", (20, 50))
@pytest.mark.parametrize(
    "phase", ("first_partial", "first_complete", "second_partial", "complete"),
)
def test_fixed_reset_clears_every_phase_and_requires_a_new_cycle(
    monkeypatch, capsys, task_class, control_freq, phase,
):
    task, state, initial = _make_task(monkeypatch, task_class, control_freq)
    assert task.HOLD_STEPS / control_freq == 1.0
    _load_fixed_reset()(task)
    for _ in range(2):
        capsys.readouterr()
        if phase == "first_partial":
            _step_at(task, state, task.FIRST_BEHAVIOR, task.FIRST_HOLD_STEPS - 1)
        else:
            _finish_first(task, state)
            assert capsys.readouterr().out.count("stage 1/2") == 1
        if phase == "second_partial":
            _step_at(task, state, task.SECOND_BEHAVIOR, task.HOLD_STEPS - 1)
        elif phase == "complete":
            _finish_second(task, state)
        assert task._check_success() == (phase == "complete")
        task._success_hold = task.HOLD_STEPS
        assert task.reset() == {"success": False}
        assert state["fraction"] == initial
        _assert_clean(task)

        # Simulate the next episode's warm-up at the untouched reset pose.
        for _ in range(2 * control_freq):
            task._post_action(None)
            assert task._cycle_stage == 0
            assert not task._check_success()
        _step_at(task, state, task.SECOND_BEHAVIOR, task.HOLD_STEPS + 1)
        assert task._cycle_stage == 0
        assert not task._check_success()

        capsys.readouterr()
        _finish_first(task, state)
        assert capsys.readouterr().out.count("stage 1/2") == 1
        _finish_second(task, state)
        assert task.reset() == {"success": False}
        _assert_clean(task)


@pytest.mark.parametrize("task_class", TASKS, ids=lambda cls: cls.__name__)
@pytest.mark.parametrize("control_freq", (20, 50))
def test_interrupted_holds_and_normal_reset(monkeypatch, task_class, control_freq):
    task, state, initial = _make_task(monkeypatch, task_class, control_freq)
    between = (task.SUCCESS_OPEN_FRACTION + task.SUCCESS_CLOSE_FRACTION) / 2
    _step_at(task, state, task.FIRST_BEHAVIOR, task.FIRST_HOLD_STEPS - 1)
    state["fraction"] = between
    task._post_action(None)
    assert task._first_state_hold == 0
    _finish_first(task, state)
    _step_at(task, state, task.SECOND_BEHAVIOR, task.HOLD_STEPS - 1)
    state["fraction"] = between
    task._post_action(None)
    assert task._second_state_hold == 0
    assert not task._check_success()
    _finish_second(task, state)
    task._reset_internal()
    _assert_clean(task)
    # Shower state restoration belongs to the mocked scene reset, not this hook.
    state["fraction"] = initial
    _finish_first(task, state)
    _finish_second(task, state)


@pytest.mark.parametrize("task_class", TASKS, ids=lambda cls: cls.__name__)
def test_progress_hook_only_writes_four_progress_fields(task_class):
    task = object.__new__(task_class)
    for name in PROGRESS_FIELDS:
        setattr(task, name, 100)
    task.sim = object()
    task.scene_name = f"FloorPlan{task_class.SUPPORTED_SCENE_IDS[0]}"
    task.robot_base_pos = np.array([1.0, 2.0, 3.0])
    task.unrelated_progress = object()
    before = vars(task).copy()
    specs = copy.deepcopy(task_class.SCENE_SPECS)
    task._reset_task_progress()
    _assert_clean(task)
    assert vars(task).keys() == before.keys()
    for name in before.keys() - set(PROGRESS_FIELDS):
        assert getattr(task, name) is before[name]
    assert task_class.SCENE_SPECS == specs


def test_reset_hook_consumers_are_exactly_the_eight_registered_cycles():
    consumers = {
        cls for cls in REGISTERED_ENVS.values()
        if issubclass(cls, _BathroomArticulationCycleMixin)
    }
    assert consumers == set(TASKS)
    assert _BathroomArticulationCycleMixin.__name__ not in REGISTERED_ENVS
    hook = _BathroomArticulationCycleMixin._reset_task_progress
    for cls in bathroom_tasks.TASK_CLASSES:
        assert (getattr(cls, "_reset_task_progress", None) is hook) == (cls in TASKS)
    for cls in TASKS:
        assert REGISTERED_ENVS[cls.__name__] is cls
        target = next(name for name in SCENES_BY_TARGET if cls.__name__.endswith(name))
        assert cls.SUPPORTED_SCENE_IDS == SCENES_BY_TARGET[target]
        assert set(cls.SCENE_SPECS) == {
            f"FloorPlan{scene}" for scene in SCENES_BY_TARGET[target]
        }


@pytest.mark.parametrize("task_class", TASKS[:2], ids=lambda cls: cls.__name__)
def test_401_fixed_reset_restores_scene_waits_and_allows_another_cycle(task_class):
    env = robosuite.make(
        env_name=task_class.__name__,
        robots="G1",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        control_freq=50,
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
        initial_fraction = float(joint.fraction(env.sim))
        expected_initial = 0.20 if env.FIRST_BEHAVIOR == "open" else 1.0
        assert initial_fraction == pytest.approx(expected_initial)
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        _assert_clean(env)
        initial_qpos, initial_qvel = data.qpos.copy(), data.qvel.copy()
        parameters = {
            name: getattr(model, name).copy() for name in (
                "jnt_range", "dof_damping", "dof_frictionloss",
                "dof_solref", "dof_solimp",
            )
        }
        _load_fixed_reset()(env)
        robot_state = _capture_robot_joint_state(env)
        zero_action = np.zeros(env.action_dim)
        substeps = round(1.0 / env.control_freq / model.opt.timestep)

        def control_step(prescribed_fraction=None):
            # Hold only the uncontrolled robot while waiting. Positive cycles
            # prescribe lid poses to exercise the real joint predicate; they
            # do not claim hand contact, force-driven operation or teleop.
            for _ in range(substeps):
                env.sim.step()
                for qs, qpos, vs, qvel in robot_state:
                    data.qpos[qs] = qpos
                    data.qvel[vs] = qvel
                if prescribed_fraction is not None:
                    data.set_joint_qpos(
                        joint.name,
                        joint.closed_qpos + prescribed_fraction
                        * (joint.open_qpos - joint.closed_qpos),
                    )
                    data.qvel[dof] = 0.0
                env.sim.forward()
            env._post_action(zero_action)

        def complete_cycle():
            for step in range(env.FIRST_HOLD_STEPS):
                control_step(_fraction_for(env, env.FIRST_BEHAVIOR))
                assert env._cycle_stage == (0 if step < 4 else 1)
                assert not env._check_success()
            assert env.HOLD_STEPS / env.control_freq == 1.0
            for step in range(env.HOLD_STEPS):
                control_step(_fraction_for(env, env.SECOND_BEHAVIOR))
                assert env._check_success() == (step == env.HOLD_STEPS - 1)

        complete_cycle()
        env.reset()
        _assert_clean(env)
        np.testing.assert_array_equal(data.qpos, initial_qpos)
        np.testing.assert_array_equal(data.qvel, initial_qvel)
        assert joint.fraction(env.sim) == pytest.approx(initial_fraction)
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        for name, value in parameters.items():
            np.testing.assert_array_equal(getattr(model, name), value)

        start = float(data.time)
        for _ in range(2 * env.control_freq):
            control_step()
            assert env._cycle_stage == 0
            assert not env._check_success()
        elapsed = float(data.time - start)
        assert elapsed >= 2.0 - 1e-8
        waiting_fraction = float(joint.fraction(env.sim))
        complete_cycle()
        env.reset()
        _assert_clean(env)
        print(json.dumps({
            "task": task_class.__name__,
            "scene": 401,
            "initial_fraction": initial_fraction,
            "waiting_fraction": waiting_fraction,
            "second_episode_wait_seconds": elapsed,
            "second_cycle_success": True,
            "success_after_next_reset": env._check_success(),
        }, sort_keys=True))
    finally:
        env.close()
