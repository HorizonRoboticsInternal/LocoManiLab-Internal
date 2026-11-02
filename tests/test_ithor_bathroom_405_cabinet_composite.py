"""Isolated 405 registration, four-stage ordering and one physical reset."""

from copy import deepcopy
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.ithor_bathroom import tasks
from robocasa.environments.ithor_bathroom.tasks.cabinet_lift import (
    IThorBathroomLiftSoapBottleFromCabinet as Atomic,
)
from robocasa.environments.ithor_bathroom.tasks.composite.cabinet_soap_routine import (
    IThorBathroomOpenCabinetThenPlaceSoapBottleOnCounterAndCloseCabinet as Task,
)
from robocasa.environments.ithor_task_catalog import (
    BATHROOM_PACKAGE,
    discover_task_catalog,
    validate_task_catalog,
)


def test_registration_and_isolated_atomic_layout():
    from robosuite.environments.base import REGISTERED_ENVS

    assert Task.SUPPORTED_SCENE_IDS == (405,)
    assert set(Task.SCENE_SPECS) == {"FloorPlan405"}
    assert Task in tasks.COMPOSITE_TASK_CLASSES
    assert Task not in tasks.ATOMIC_TASK_CLASSES
    assert getattr(tasks, Task.__name__) is REGISTERED_ENVS[Task.__name__] is Task
    original = deepcopy(Atomic.SCENE_SPECS)
    derived = Task.SCENE_SPECS["FloorPlan405"]
    assert derived["cabinet_open_fraction"] == 0.2
    for key, value in original["FloorPlan405"].items():
        if key not in ("lang", "cabinet_open_fraction", "robot_pose"):
            assert derived[key] == value
    saved = derived["robot_pose"]["xy"][0]
    try:
        derived["robot_pose"]["xy"][0] = 123
        assert Atomic.SCENE_SPECS == original
    finally:
        derived["robot_pose"]["xy"][0] = saved
    with pytest.raises(ValueError, match="uncalibrated scene ids"):
        Task(scene_ids=[418])
    assert validate_task_catalog(discover_task_catalog((BATHROOM_PACKAGE,))) == []


def test_countertop_collision_replacement_is_local_to_composite():
    from pathlib import Path
    from robocasa.models import assets_root

    path = Path(assets_root) / "scenes/ithor/FloorPlan405_physics.robocasa.xml"
    original = ET.parse(path).getroot()
    snapshot = ET.tostring(original)
    arena = deepcopy(original)
    task = object.__new__(Task)
    task.mujoco_arena = SimpleNamespace(
        worldbody=arena.find("worldbody"), asset=arena.find("asset")
    )
    spec = Task.SCENE_SPECS["FloorPlan405"]
    task.scene_task_spec = lambda: spec
    body_name = spec["counter_top_geom"].rsplit("_collision_", 1)[0]
    original_body = original.find(f".//body[@name='{body_name}']")
    task._replace_countertop_collision()
    patched_body = arena.find(f".//body[@name='{body_name}']")
    removed = {spec["counter_top_geom"], spec["counter_underlay_geom"]}
    assert len(task.counter_top_geom_names) == 40
    assert all(patched_body.find(f"./geom[@name='{name}']") is None for name in removed)
    for element in original_body:
        if element.get("name") not in removed:
            assert ET.tostring(element) == ET.tostring(
                patched_body.find(f"./{element.tag}[@name='{element.get('name')}']")
            )
    assert ET.tostring(original) == snapshot
    assert "_replace_countertop_collision" not in Atomic.__dict__
    assert not hasattr(Atomic, "_replace_countertop_collision")


def _stub_task(control_freq=20):
    task = object.__new__(Task)
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    task.PRINT_STAGE_MESSAGES = False
    state = dict(fraction=0.2, height=0.5, low=True, counter=False)
    task.sim = None
    task.source_fixture = object()
    task.work_surface_fixture = object()
    task.work_surface_region = object()
    task.cabinet_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task._source_root_z = lambda: state["height"]
    task._g1_hand_grasping_fixture = lambda *args: pytest.fail(
        "Lift must not check grasp"
    )
    task._fixture_released = lambda *args: pytest.fail(
        "Lift must not check hand release"
    )
    task._bottle_is_on_surface = lambda *args: state["low"]
    task._bottle_is_on_counter = lambda: state["counter"]
    task._fixture_has_illegal_penetration = lambda *args: False
    task._reset_task_progress()
    return task, state


def _tick(task, count):
    for _ in range(count):
        task._advance_cabinet_routine()


def _open_and_lift(task, state):
    state["fraction"] = 0.8
    _tick(task, 5)
    assert task._routine_stage == 1
    # Lift needs the height threshold, without any grasp or hand-state evidence.
    state.update(height=0.79, low=False)
    _tick(task, 5)
    assert task._routine_stage == 1
    state["height"] = task._initial_source_z + task.MIN_LIFT_DELTA
    _tick(task, 5)
    assert task._routine_stage == 2


@pytest.mark.parametrize(
    "lift_delta,fraction,penetration,expected",
    (
        (0.299, 0.8, False, False),
        (0.30, 0.8, False, True),
        (0.31, 0.8, False, True),
        (0.31, 0.79, False, False),
        (0.31, 0.8, True, False),
    ),
)
def test_lift_height_door_and_penetration_without_hand_checks(
    lift_delta, fraction, penetration, expected
):
    task, state = _stub_task()
    state.update(height=task._initial_source_z + lift_delta, fraction=fraction)
    task._fixture_has_illegal_penetration = lambda *args: penetration
    # No grasp history is initialized by this stub; hand queries fail immediately.
    assert task._stage_goal(1) is expected


@pytest.mark.parametrize("control_freq", (10, 20))
def test_order_final_hold_interruption_and_fixed_reset(control_freq):
    task, state = _stub_task(control_freq)
    # Placing and closing first cannot skip opening or lifting.
    state.update(counter=True, low=False)
    _tick(task, control_freq + 5)
    assert task._routine_stage == 0
    assert not task._check_success()
    state.update(counter=False, low=True)
    _open_and_lift(task, state)
    # Placement depends only on countertop contact and release, for one second.
    state.update(counter=True, fraction=0.4)
    _tick(task, control_freq - 1)
    assert task._routine_stage == 2
    state["counter"] = False
    _tick(task, 1)
    assert task._routine_hold == 0
    state["counter"] = True
    _tick(task, control_freq - 1)
    assert task._routine_stage == 2
    _tick(task, 1)
    assert task._routine_stage == 3
    state["fraction"] = 0.1
    _tick(task, control_freq - 1)
    assert not task._check_success()
    _tick(task, 1)
    assert task._check_success()
    state["counter"] = False
    assert not task._check_success()
    _tick(task, 1)
    assert task._routine_hold == 0
    state["counter"] = True
    _tick(task, control_freq - 1)
    assert not task._check_success()
    _tick(task, 1)
    assert task._check_success()
    state.update(fraction=0.2, height=0.5, low=True, counter=False)
    task._reset_task_progress()
    assert task._routine_stage == task._routine_hold == task._success_hold == 0
    assert not task._check_success()


@pytest.mark.parametrize("control_freq", (10, 20))
def test_precompleted_placement_counts_without_repeating(control_freq):
    task, state = _stub_task(control_freq)
    state["counter"] = True
    _open_and_lift(task, state)
    state["fraction"] = 0.0
    _tick(task, control_freq - 1)
    assert task._routine_stage == 2
    _tick(task, 1)
    assert task._routine_stage == 3


@pytest.mark.parametrize(
    "case,expected",
    (
        ("main", True),
        ("narrow_strip", True),
        ("overhang_and_deep_contact", True),
        ("reversed_contact", True),
        ("left_hand", False),
        ("right_hand", False),
        ("no_contact", False),
        ("cabinet_side", False),
        ("other_fixture", False),
    ),
)
def test_counter_requires_only_top_contact_and_both_hands_released(case, expected):
    task = object.__new__(Task)
    task.source_fixture = object()
    task.counter_top_geom_names = ("counter_top",)
    task.counter_region = SimpleNamespace(
        world_pose=lambda sim: (np.array([0, 0, 1.0]), 0)
    )
    point = [0, 0.6 if case == "narrow_strip" else 0, 1.0]
    if case == "cabinet_side":
        point[2] = 0.8
    geoms = (2, 0) if case == "reversed_contact" else (0, 2)
    if case == "other_fixture":
        geoms = (1, 2)
    contact = SimpleNamespace(geom1=geoms[0], geom2=geoms[1], pos=point, dist=-0.01)
    task.sim = SimpleNamespace(
        model=SimpleNamespace(
            geom_name2id=lambda name: 0, geom_bodyid=np.array([10, 11, 20])
        ),
        data=SimpleNamespace(ncon=0 if case == "no_contact" else 1, contact=[contact]),
    )
    task._fixture_body_ids = lambda *args, **kw: {20}
    task._g1_hand_contacts_fixture = lambda side, fixture: case == f"{side}_hand"
    task._fixture_bbox_inside_region_xy = lambda *args: pytest.fail(
        "Placement must not require full footprint containment"
    )
    task._fixture_bbox_points = lambda *args: pytest.fail(
        "Placement must not require a bottle-bottom height"
    )
    task._fixture_has_illegal_penetration = lambda *args: pytest.fail(
        "Placement must not add penetration conditions"
    )
    assert task._bottle_is_on_counter() is expected


def test_headless_405_reset_support_door_sweep_and_counter(monkeypatch):
    import robosuite
    from robocasa.utils import env_utils
    from validate_ithor_tasks import (
        _assert_body_contact_state,
        _load_reset_controller,
        _settle_scene_with_frozen_robot,
        _task_body_contact_contracts,
    )

    def apply_sonic_spawn_pose(env):
        # Match G1_BRIDGE's standing arms without loading locomotion policies.
        angles = dict(
            hip_pitch=-0.312,
            hip_roll=0.0,
            hip_yaw=0.0,
            knee=0.669,
            ankle_pitch=-0.363,
            ankle_roll=0.0,
            shoulder_pitch=0.2,
            shoulder_yaw=0.0,
            elbow=0.6,
            wrist_roll=0.0,
            wrist_pitch=0.0,
            wrist_yaw=0.0,
        )
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
        assert env.cabinet_joint.fraction(env.sim) == pytest.approx(0.2, abs=0.01)
        assert not env._raw_success()
        assert not env._check_success()
        assert env._bottle_is_on_surface(
            env.work_surface_fixture, env.work_surface_region
        )
        initial = env.source_fixture.world_center(env.sim).copy()
        for name, subjects, supports in _task_body_contact_contracts(env):
            _assert_body_contact_state(env, name, subjects, supports, "reset")
        _settle_scene_with_frozen_robot(env, int(0.5 / env.sim.model.opt.timestep))
        drift = np.linalg.norm(env.source_fixture.world_center(env.sim) - initial)
        assert drift < 0.01
        assert env.cabinet_joint.fraction(env.sim) == pytest.approx(0.2, abs=0.01)
        assert env._bottle_is_on_surface(
            env.work_surface_fixture, env.work_surface_region
        )

        def set_fraction(fraction):
            joint = env.cabinet_joint
            env.sim.data.set_joint_qpos(
                joint.name,
                joint.closed_qpos + fraction * (joint.open_qpos - joint.closed_qpos),
            )
            env.sim.data.set_joint_qvel(joint.name, 0.0)
            env.sim.forward()

        for fraction in np.linspace(0.2, 1.0, 9):
            set_fraction(fraction)
            assert env.count_body_collisions() == 0
            assert not env._fixture_has_illegal_penetration(
                env.source_fixture, env.work_surface_fixture.body_names
            )
        assert env._stage_goal(0)
        _tick(env, 5)
        assert env._routine_stage == 1
        source_joint = env._fixture_free_joint_name(env.source_fixture)
        qpos = env.sim.data.get_joint_qpos(source_joint).copy()
        qpos[0] += 0.30  # Pull out of the opening before lifting above the shelf.
        qpos[2] += env.MIN_LIFT_DELTA + 0.01
        env.sim.data.set_joint_qpos(source_joint, qpos)
        env._zero_fixture_velocity(env.source_fixture)
        env.sim.forward()
        assert env._stage_goal(1)
        _tick(env, 5)
        assert env._routine_stage == 2
        counter_center = env.counter_region.world_pose(env.sim)[0]
        print(
            dict(
                counter_center=counter_center.tolist(),
                counter_half_size=env.counter_region.half_size.tolist(),
            )
        )
        assert env._place_fixture_on_support(
            env.source_fixture,
            env.counter_region,
            env.counter_fixture,
            [-2.84, -0.22],
            contract_name="test_counter_support",
        )
        assert env._bottle_is_on_counter()
        # There must be a physical hole over the basin, not just a failed goal.
        env._set_fixture_center_xy_and_bbox_bottom(
            env.source_fixture, [-3.018, 0.207], counter_center[2]
        )
        assert not env._bottle_is_on_counter()
        _tick(env, env.HOLD_STEPS)
        assert env._routine_stage == 2
        assert env._routine_hold == 0
        _settle_scene_with_frozen_robot(env, int(0.25 / env.sim.model.opt.timestep))
        bottom = float(np.min(env._fixture_bbox_points(env.source_fixture)[:, 2]))
        assert bottom < counter_center[2] - 0.05
        assert not env._bottle_is_on_counter()
        print(dict(basin_bottle_bottom=bottom, counter_height=counter_center[2]))
        # Move onto the narrow strip beside the sink, outside the old region.
        strip = env.counter_fixture.regions[
            "FloorPlan405:CabinetBodyMesh_45407f37_25:region:receptacle:5"
        ]
        strip_center = strip.world_pose(env.sim)[0]
        env._set_fixture_center_xy_and_bbox_bottom(
            env.source_fixture, strip_center[:2], counter_center[2]
        )
        assert not env._fixture_bbox_inside_region_xy(
            env.source_fixture, env.counter_region
        )
        assert env._bottle_is_on_counter()
        _tick(env, env.HOLD_STEPS - 1)
        assert env._routine_stage == 2
        _tick(env, 1)
        assert env._routine_stage == 3
        set_fraction(0.0)
        _tick(env, env.HOLD_STEPS - 1)
        assert not env._check_success()
        _tick(env, 1)
        assert env._check_success()
        # A counter-height bottle floating in midair must fail immediately.
        qpos = env.sim.data.get_joint_qpos(source_joint).copy()
        qpos[2] += 0.08
        env.sim.data.set_joint_qpos(source_joint, qpos)
        env.sim.forward()
        assert not env._bottle_is_on_counter()
        assert not env._check_success()
        meta = env.get_ep_meta()
        assert meta["stage_order"] == list(Task.COMPOSITE_ORDERING)
        assert meta["cabinet_initial_open_fraction"] == 0.2
        assert meta["placement_hold_steps"] == env.control_freq
        assert meta["placement_contact_geoms"] == list(env.counter_top_geom_names)
        print(
            dict(
                initial_source_z=env._initial_source_z,
                support_z=env.work_surface_region.world_pose(env.sim)[0][2],
                drift=float(drift),
                final_hold_steps=env.HOLD_STEPS,
            )
        )
    finally:
        env.close()
