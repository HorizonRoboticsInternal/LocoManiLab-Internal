"""Keep the navigation hamper lid stable without changing its atomic tasks."""

import json
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_bedroom.tasks import (
    IThorBedroomApproachStorageAndCloseHamperLid,
    IThorBedroomApproachStorageAndOpenHamperLid,
    IThorBedroomCloseHamperLid,
    IThorBedroomOpenHamperLid,
)
from robocasa.environments.ithor_bedroom.tasks.articulation_composites import (
    _ApproachThenArticulationMixin,
)
from robocasa.environments.ithor_bedroom.tasks.hamper_lid import _FloorHamperMixin
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
)
from validate_ithor_tasks import _capture_robot_joint_state, _load_reset_controller


Task = IThorBedroomApproachStorageAndOpenHamperLid


def test_open_hamper_calibration_only_changes_target_dof(monkeypatch):
    task = object.__new__(Task)
    task.scene_name = "FloorPlan306"
    default_ref = np.tile([0.02, 1.0], (4, 1))
    default_imp = np.tile([0.9, 0.95, 0.001, 0.5, 2.0], (4, 1))
    velocities = np.arange(4, dtype=float) + 0.5
    model = SimpleNamespace(
        joint_name2id=lambda name: 1 if name == "lid" else -1,
        jnt_dofadr=np.array([0, 2]),
        dof_solref=default_ref.copy(),
        dof_solimp=default_imp.copy(),
    )
    data = SimpleNamespace(qvel=velocities.copy())
    calls = []
    task.sim = SimpleNamespace(
        model=model, data=data, forward=lambda: calls.append("forward")
    )

    def parent_setup(self):
        calls.append("setup")
        self.target_joint = "lid"

    monkeypatch.setattr(_ApproachThenArticulationMixin, "_setup_scene", parent_setup)
    task._setup_scene()
    default_ref[2] = [0.002, 1.0]
    default_imp[2] = [0.9999, 0.9999, 0.001, 0.5, 2.0]
    velocities[2] = 0.0
    np.testing.assert_array_equal(model.dof_solref, default_ref)
    np.testing.assert_array_equal(model.dof_solimp, default_imp)
    np.testing.assert_array_equal(data.qvel, velocities)
    assert calls == ["setup", "forward"]


@pytest.mark.parametrize(
    "task_class, setup_method, initial_qpos",
    (
        (IThorBedroomOpenHamperLid, _FloorHamperMixin._setup_scene, np.pi / 6),
        (IThorBedroomCloseHamperLid, _FloorHamperMixin._setup_scene, 0.85 * 1.5708),
        (
            IThorBedroomApproachStorageAndCloseHamperLid,
            _ApproachThenArticulationMixin._setup_scene,
            1.5708,
        ),
    ),
)
def test_other_hamper_tasks_keep_initialization_and_specs(
    task_class, setup_method, initial_qpos
):
    task = object.__new__(task_class)
    task.scene_name = "FloorPlan306"
    joint = SimpleNamespace(closed_qpos=0.0, open_qpos=1.5708)
    assert task._initial_joint_qpos(joint) == pytest.approx(initial_qpos)
    assert task_class._setup_scene is setup_method
    assert task_class._setup_scene is not Task._setup_scene
    assert (
        task_class.SUPPORTED_SCENE_IDS
        == Task.SUPPORTED_SCENE_IDS
        == (302, 306, 310, 323, 329)
    )
    assert task_class.SUCCESS_OPEN_FRACTION == Task.SUCCESS_OPEN_FRACTION == 0.80
    assert task_class.SUCCESS_CLOSE_FRACTION == Task.SUCCESS_CLOSE_FRACTION == 0.12
    assert task_class.HAMPER_BASE_WELD_NAME == Task.HAMPER_BASE_WELD_NAME
    operation_spec = {
        key: value
        for key, value in task_class.SCENE_SPECS["FloorPlan306"].items()
        if key not in {"lang", "navigation"}
    }
    assert operation_spec == {
        key: value
        for key, value in Task.SCENE_SPECS["FloorPlan306"].items()
        if key not in {"lang", "navigation"}
    }


@pytest.mark.parametrize(
    "task_class",
    (
        IThorBedroomApproachStorageAndOpenHamperLid,
        IThorBedroomApproachStorageAndCloseHamperLid,
    ),
)
@pytest.mark.parametrize("scene_id", (302, 306, 310, 323, 329))
def test_approach_hamper_uses_exact_atomic_fixture_and_private_start(
    task_class, scene_id
):
    scene = f"FloorPlan{scene_id}"
    atomic = IThorBedroomOpenHamperLid.SCENE_SPECS[scene]
    spec = task_class.SCENE_SPECS[scene]
    nav = spec["navigation"]
    assert nav["fixture"] == atomic["fixture"]
    assert nav["remove_fixtures"] == []
    assert nav["minimum_start_distance"] > task_class.SUCCESS_DISTANCE + 0.15
    assert nav["start_pose"]["xy"] != atomic["hamper_robot_pose"]["xy"]
    operation_spec = {
        key: value for key, value in spec.items() if key not in ("lang", "navigation")
    }
    atomic_spec = {key: value for key, value in atomic.items() if key != "lang"}
    assert operation_spec == atomic_spec
    if scene_id == 306:
        original = get_ithor_bedroom_task_spec(scene, "approach_storage")
        assert all(nav[key] == original[key] for key in nav)


@pytest.mark.parametrize(
    "task_class",
    (
        IThorBedroomApproachStorageAndOpenHamperLid,
        IThorBedroomApproachStorageAndCloseHamperLid,
    ),
)
@pytest.mark.parametrize("scene_id", (302,))
def test_spawned_hamper_navigation_requires_operable_side(
    monkeypatch, task_class, scene_id
):
    scene = f"FloorPlan{scene_id}"
    task = object.__new__(task_class)
    task.scene_name = scene
    task._approach_stage = 0
    pose = task_class.SCENE_SPECS[scene]["navigation"]["operation_pose"]
    xy = np.asarray(pose["xy"])
    state = {"xy": xy.copy(), "near": True}
    task._robot_base_xy = lambda: state["xy"]
    monkeypatch.setattr(
        _ApproachThenArticulationMixin,
        "_near_articulation",
        lambda self: state["near"],
    )
    assert task._near_articulation()
    state["xy"] = xy + (task.HAMPER_OPERATION_RADIUS + 0.01) * np.array([1.0, 0.0])
    assert not task._near_articulation()
    state["xy"] = xy
    state["near"] = False
    assert not task._near_articulation()


@pytest.mark.parametrize(
    "task_class",
    (
        IThorBedroomApproachStorageAndOpenHamperLid,
        IThorBedroomApproachStorageAndCloseHamperLid,
    ),
)
@pytest.mark.parametrize(
    "scene_id, arrival_xy, operation_xy",
    (
        (323, [0.15, -0.75], [0.15, -1.95]),
        (329, [1.20, -0.60], [1.20, 0.60]),
    ),
)
def test_hamper_arrival_does_not_require_the_far_side_operation_pose(
    monkeypatch, task_class, scene_id, arrival_xy, operation_xy
):
    task = object.__new__(task_class)
    task.scene_name = f"FloorPlan{scene_id}"
    state = {"near": True, "xy": np.array(arrival_xy)}
    task._robot_base_xy = lambda: state["xy"]
    monkeypatch.setattr(
        _ApproachThenArticulationMixin,
        "_near_articulation",
        lambda self: state["near"],
    )

    task._approach_stage = 0
    assert task._near_articulation()  # Approached on the near side.
    state["near"] = False
    assert not task._near_articulation()  # Merely crossing the XY is insufficient.
    task._approach_stage = 1
    # After navigation completes, the lid state alone gates the second stage.
    # Neither the arrival side nor the collision-distance predicate may block it.
    assert task._near_articulation()
    state["near"] = False
    state["xy"] = np.array(operation_xy)
    assert task._near_articulation()


@pytest.mark.parametrize(
    "task_class",
    (
        IThorBedroomApproachStorageAndOpenHamperLid,
        IThorBedroomApproachStorageAndCloseHamperLid,
    ),
)
@pytest.mark.parametrize("scene_id", (302, 310, 323, 329))
def test_hamper_second_stage_success_ignores_robot_distance(
    monkeypatch, task_class, scene_id
):
    monkeypatch.setattr(
        ExactSceneArticulation, "_post_action", lambda self, action: None
    )
    task = object.__new__(task_class)
    task.scene_name = f"FloorPlan{scene_id}"
    task._approach_stage = 1
    task._operation_hold = 0
    task.HOLD_STEPS = 3
    # Use a location that is neither the arrival nor operation pose. The
    # second stage must ignore it once navigation has been completed.
    task._robot_base_xy = lambda: np.array([99.0, 99.0])
    task._raw_success = lambda: True

    # The underlying collision predicate is false, but an opened/closed target
    # must still accumulate the composite success hold in these four scenes.
    monkeypatch.setattr(
        _ApproachThenArticulationMixin,
        "_near_articulation",
        lambda self: False,
    )
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._operation_hold == task.HOLD_STEPS
    assert task._check_success()


def test_native_306_hamper_keeps_collision_distance_navigation_gate(monkeypatch):
    task = object.__new__(Task)
    task.scene_name = "FloorPlan306"
    monkeypatch.setattr(
        _ApproachThenArticulationMixin, "_near_articulation", lambda self: True
    )
    assert task._near_articulation()


@pytest.mark.parametrize("control_freq", (10, 20, 30))
def test_open_hamper_requires_order_and_uninterrupted_holds(monkeypatch, control_freq):
    monkeypatch.setattr(
        ExactSceneArticulation, "_post_action", lambda self, action: None
    )
    monkeypatch.setattr(ExactSceneArticulation, "_reset_internal", lambda self: None)
    task = object.__new__(Task)
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    assert task.HOLD_STEPS == control_freq
    state = {"fraction": np.pi / 6 / 1.5708, "distance": 0.10}
    task.sim = object()
    task.target_fixture = object()
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task.robot_fixture_collision_distance = lambda *args, **kwargs: state["distance"]
    task._report_stage_complete = lambda *args, **kwargs: None
    task._reset_internal()
    initial_fraction = task._approach_initial_fraction

    state["fraction"] = 0.85
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 0
    assert not task._check_success()

    state.update(fraction=initial_fraction, distance=0.50)
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._approach_hold == 0
    state["distance"] = 0.10
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

    state["fraction"] = 0.85
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    state["fraction"] = 0.79
    task._post_action(None)
    assert task._operation_hold == 0
    state["fraction"] = 0.85
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()
    state["distance"] = 0.50
    task._post_action(None)
    assert task._operation_hold == 0
    assert not task._check_success()

    state["fraction"] = initial_fraction
    task._reset_internal()
    assert task._approach_stage == task._approach_hold == task._operation_hold == 0
    assert not task._check_success()


def test_306_hamper_waits_twenty_seconds_then_opens_under_torque():
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
        assert model.dof_damping[dof] == pytest.approx(0.1)
        assert model.dof_frictionloss[dof] == pytest.approx(1.0)
        assert model.jnt_stiffness[joint_id] == 0.0
        assert data.qvel[dof] == 0.0
        assert data.get_joint_qpos(env.target_joint) == pytest.approx(np.pi / 6)
        np.testing.assert_allclose(model.jnt_range[joint_id], [0.0, 1.5708])
        np.testing.assert_allclose(model.dof_solref[dof], [0.002, 1.0])
        np.testing.assert_allclose(
            model.dof_solimp[dof], [0.9999, 0.9999, 0.001, 0.5, 2.0]
        )
        np.testing.assert_allclose(env._robot_base_xy(), [2.85, 0.30])
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._approach_start_distance >= 2.40
        assert not env._near_articulation()
        assert not env._raw_success()
        assert not env._check_success()
        robot_state = _capture_robot_joint_state(env)

        def advance(seconds, opening_torque=0.0, stop_fraction=None):
            # Hold only the uncontrolled robot. The lid remains dynamic; the
            # diagnostic torque along its hinge is not a teleop grasp.
            start = data.time
            initial_fraction = float(joint.fraction(env.sim))
            max_drift = 0.0
            try:
                while data.time - start < seconds:
                    data.xfrc_applied[lid_body, 3:] = opening_torque * np.asarray(
                        data._data.xaxis[joint_id]
                    )
                    env.sim.step()
                    for qs, qpos, vs, qvel in robot_state:
                        data.qpos[qs] = qpos
                        data.qvel[vs] = qvel
                    env.sim.forward()
                    fraction = float(joint.fraction(env.sim))
                    max_drift = max(max_drift, abs(fraction - initial_fraction))
                    if stop_fraction is not None and fraction >= stop_fraction:
                        break
            finally:
                data.xfrc_applied[lid_body, 3:] = 0.0
            return float(data.time - start), float(max_drift)

        wait_seconds, max_drift = advance(20.0)
        waiting_fraction = float(joint.fraction(env.sim))
        assert max_drift < env.INITIAL_JOINT_FRACTION_TOLERANCE
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

        torque_seconds, _ = advance(2.0, opening_torque=2.0, stop_fraction=0.85)
        opened_fraction = float(joint.fraction(env.sim))
        assert env._raw_success()
        assert env.HOLD_STEPS == 20
        for step in range(env.HOLD_STEPS):
            advance(1.0 / env.control_freq)
            env._post_action(np.zeros(env.action_dim))
            assert env._check_success() == (step == env.HOLD_STEPS - 1)
        assert env.count_body_collisions() == 0
        print(
            json.dumps(
                {
                    "scene": 306,
                    "wait_simulated_seconds": wait_seconds,
                    "max_wait_fraction_drift": max_drift,
                    "waiting_fraction": waiting_fraction,
                    "torque_seconds": torque_seconds,
                    "opened_fraction": opened_fraction,
                    "final_fraction": float(joint.fraction(env.sim)),
                    "success_hold_steps": env._operation_hold,
                    "control_freq": env.control_freq,
                    "success": env._check_success(),
                },
                sort_keys=True,
            )
        )
    finally:
        env.close()


@pytest.mark.parametrize(
    "task_class",
    (
        IThorBedroomApproachStorageAndOpenHamperLid,
        IThorBedroomApproachStorageAndCloseHamperLid,
    ),
)
def test_composite_hamper_uses_atomic_lid_colliders_only_when_spawned(task_class):
    scene = "FloorPlan302"
    assert (
        task_class._create_spawned_hamper_fixture
        is _FloorHamperMixin._create_spawned_hamper_fixture
    )
    assert (
        task_class.SCENE_SPECS[scene]["asset"]
        == IThorBedroomOpenHamperLid.SCENE_SPECS[scene]["asset"]
    )


def test_spawned_hamper_composites_keep_atomic_lid_and_operable_side(monkeypatch):
    open_task = object.__new__(Task)
    open_task.scene_name = "FloorPlan302"
    calls = []
    monkeypatch.setattr(
        _ApproachThenArticulationMixin,
        "_setup_scene",
        lambda self: calls.append("atomic joint setup"),
    )
    open_task._setup_scene()
    assert calls == ["atomic joint setup"]
    for task_class in (Task, IThorBedroomApproachStorageAndCloseHamperLid):
        assert (
            task_class._create_spawned_hamper_fixture
            is _FloorHamperMixin._create_spawned_hamper_fixture
        )

    for scene_id in (302, 310, 323, 329):
        scene = f"FloorPlan{scene_id}"
        atomic_spec = IThorBedroomOpenHamperLid.SCENE_SPECS[scene]
        close_task = object.__new__(IThorBedroomApproachStorageAndCloseHamperLid)
        close_task.scene_name = scene
        joint = SimpleNamespace(closed_qpos=0.0, open_qpos=1.5708)
        assert close_task._initial_joint_qpos(joint) == pytest.approx(0.85 * 1.5708)
        navigation = Task.SCENE_SPECS[scene]["navigation"]
        if "operation_pose" in navigation:
            assert navigation["operation_pose"] == atomic_spec["hamper_robot_pose"]
