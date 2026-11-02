"""Scene-410-only registration, ordered progress and physical staging checks."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.ithor_bathroom import tasks
from robocasa.environments.ithor_bathroom.tasks.composite.drawer_routines import (
    IThorBathroomOpenDrawerThenPutToiletPaperRollAndCloseDrawer as PutClose,
    IThorBathroomTakeSoapBottleFromDrawerThenStandOnSinkAndApproachToilet as TakeStandWalk,
)
from robocasa.environments.ithor_bathroom.tasks.drawer_pnp import (
    IThorBathroomTakeSoapBottleFromDrawer as Take,
    IThorBathroomToiletPaperRollIntoDrawer as Put,
)
from robocasa.environments.ithor_bathroom.tasks.object_manipulation import (
    _BathroomWorkspaceMixin,
)
from robocasa.environments.ithor_task_catalog import (
    BATHROOM_PACKAGE,
    discover_task_catalog,
    validate_task_catalog,
)


def test_registration_and_isolated_atomic_specs():
    from robosuite.environments.base import REGISTERED_ENVS

    for cls, atom in ((PutClose, Put), (TakeStandWalk, Take)):
        assert cls.SUPPORTED_SCENE_IDS == (410,)
        assert set(cls.SCENE_SPECS) == {"FloorPlan410"}
        assert cls in tasks.COMPOSITE_TASK_CLASSES
        assert cls not in tasks.ATOMIC_TASK_CLASSES
        assert getattr(tasks, cls.__name__) is cls
        assert REGISTERED_ENVS[cls.__name__] is cls
        original = deepcopy(atom.SCENE_SPECS)
        derived = cls.SCENE_SPECS["FloorPlan410"]
        for key, value in original["FloorPlan410"].items():
            if key != "lang" and not (cls is TakeStandWalk and key == "surface_region"):
                assert derived[key] == value
        pose = derived["robot_pose"]["xy"]
        saved = pose[0]
        try:
            pose[0] = 123
            assert atom.SCENE_SPECS == original
        finally:
            pose[0] = saved
        with pytest.raises(ValueError, match="uncalibrated scene ids"):
            cls(scene_ids=[415])
    assert validate_task_catalog(discover_task_catalog((BATHROOM_PACKAGE,))) == []
    assert "_OrderedDrawerRoutineMixin" not in REGISTERED_ENVS


def _stub_task(cls, control_freq=20):
    task = object.__new__(cls)
    task.PRINT_STAGE_MESSAGES = False
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    state = dict(fraction=0.0, bottom=False, height=0.5, counter=False, distance=1.0)
    task.sim = None
    task.drawer_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task._source_contacts_drawer_bottom = lambda: state["bottom"]
    task._source_root_z = lambda: state["height"]
    task._bottle_is_on_counter = lambda: state["counter"]
    task.robot_fixture_collision_distance = lambda *args, **kw: state["distance"]
    task.target_fixture = object()
    task._navigation_success_distance = 0.35
    task._reset_task_progress()
    return task, state


def _tick(task, count):
    for _ in range(count):
        task._post_action(None)


@pytest.mark.parametrize("cls", (PutClose, TakeStandWalk))
@pytest.mark.parametrize("control_freq", (10, 20))
def test_order_hold_interruption_and_fixed_reset(monkeypatch, cls, control_freq):
    monkeypatch.setattr(
        _BathroomWorkspaceMixin, "_post_action", lambda *args: None, raising=False
    )
    task, state = _stub_task(cls, control_freq)
    assert not task._check_success()
    # A valid final scene cannot skip either preceding skill.
    if cls is PutClose:
        state["bottom"] = True
    else:
        state.update(counter=True, distance=0.2)
    _tick(task, control_freq + 5)
    assert task._routine_stage == 0
    assert not task._check_success()
    # Complete the first stage; an already finished second stage gets no credit.
    if cls is PutClose:
        state["fraction"] = 1.0
    else:
        state["height"] = 0.81
    _tick(task, 5)
    assert task._routine_stage == 1
    _tick(task, 10)
    assert task._routine_stage == 1
    if cls is PutClose:
        state["bottom"] = False
    else:
        state["counter"] = False
    _tick(task, 1)
    if cls is PutClose:
        state["bottom"] = True
    else:
        state["counter"] = True
    _tick(task, 4)
    assert task._routine_stage == 1
    _tick(task, 1)
    assert task._routine_stage == 2
    if cls is PutClose:
        state["fraction"] = 0.0
    else:
        # Being near the toilet before completing placement must be undone.
        _tick(task, control_freq + 1)
        assert not task._check_success()
        state["distance"] = 0.8
        _tick(task, 1)
        state["distance"] = 0.35
    _tick(task, control_freq - 1)
    assert not task._check_success()
    _tick(task, 1)
    assert task._check_success()
    # Losing the placement invalidates success even before the next action.
    key = "bottom" if cls is PutClose else "counter"
    state[key] = False
    assert not task._check_success()
    _tick(task, 1)
    assert task._routine_hold == 0
    state[key] = True
    _tick(task, control_freq - 1)
    assert not task._check_success()
    _tick(task, 1)
    assert task._check_success()
    # Fixed collection restores initial physics before invoking this hook.
    state.update(fraction=0.0, bottom=False, height=0.5, counter=False, distance=1.0)
    task._reset_task_progress()
    assert task._routine_stage == task._routine_hold == task._success_hold == 0
    assert not task._check_success()
    if cls is TakeStandWalk:
        assert task._initial_source_z == 0.5


def test_drawer_requires_open_insertion_then_closed_with_bottom_contact():
    task, state = _stub_task(PutClose)
    state.update(bottom=True, fraction=0.79)
    assert not task._stage_goal(0)
    assert not task._stage_goal(1)
    state["fraction"] = 0.8
    assert task._stage_goal(0) and task._stage_goal(1)
    assert not task._raw_success()
    state["fraction"] = 0.2
    assert task._raw_success()
    state["bottom"] = False
    assert not task._raw_success()


def test_soap_reuses_lift_and_collision_distance():
    task, state = _stub_task(TakeStandWalk)
    state["height"] = 0.799
    assert not task._stage_goal(0)
    state["height"] = 0.801
    assert task._stage_goal(0)
    state["counter"] = True
    for distance, expected in (
        (0.35, True),
        (0.351, False),
        (np.inf, False),
        (np.nan, False),
    ):
        state["distance"] = distance
        assert task._raw_success() is expected
    state.update(distance=0.1, counter=False)
    assert not task._raw_success()


@pytest.mark.parametrize(
    "failure",
    (None, "sideways", "outside", "hover", "no_contact", "penetration", "held"),
)
def test_counter_requires_upright_supported_released_bottle(failure):
    task = object.__new__(TakeStandWalk)
    task.sim = SimpleNamespace(
        model=SimpleNamespace(body_name2id=lambda name: 0),
        data=SimpleNamespace(body_xmat=np.array([[1, 0, 0, 0, 0, -1, 0, 1, 0]])),
    )
    task.source_fixture = SimpleNamespace(
        body_name="bottle", world_center=lambda sim: np.array([0, 0, 1.0])
    )
    task.work_surface_fixture = SimpleNamespace(body_names=("counter",))
    task.work_surface_region = SimpleNamespace(
        world_pose=lambda sim: (np.array([0, 0, 1.0]), 0)
    )
    task._target_upright_axis = (1, 1)
    if failure == "sideways":
        task.sim.data.body_xmat[0] = np.eye(3).ravel()
    task._fixture_body_ids = lambda *a, **kw: {1}
    task._fixture_contact_distances = (
        lambda *a: [] if failure == "no_contact" else [0.0]
    )
    task.counter_support_regions = (
        SimpleNamespace(contains_xy=lambda *a: failure != "outside"),
    )
    task._fixture_bbox_points = lambda *a: np.array(
        [[0, 0, 1.1 if failure == "hover" else 1.0]]
    )
    task._fixture_has_illegal_penetration = lambda *a: failure == "penetration"
    task._fixture_released = lambda *a: failure != "held"
    assert task._bottle_is_on_counter() is (failure is None)


@pytest.mark.parametrize("index", range(4))
def test_counter_patch_union_includes_rim_and_excludes_sink_gap(index):
    from robocasa.models.scenes.thor_semantics import SemanticRegion

    task = object.__new__(TakeStandWalk)
    task.sim = None
    centers = ((-2.192, 1.757), (-3.433, 1.757), (-0.967, 1.757), (-2.205, 2.117))
    halves = ((0.360, 0.424), (0.076, 0.424), (0.062, 0.424), (1.300, 0.064))
    task.counter_support_regions = tuple(
        SemanticRegion(
            stable_id=stable_id,
            fixture_id="counter",
            kind="support",
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.array([*half, 0.0]),
            _world_center=np.array([*center, 0.806]),
        )
        for stable_id, center, half in zip(
            TakeStandWalk.COUNTER_REGION_IDS, centers, halves
        )
    )
    position = np.array([*centers[index], 1.0])
    task.source_fixture = SimpleNamespace(world_center=lambda sim: position)
    task.work_surface_fixture = SimpleNamespace(body_names=("counter",))
    task._target_upright_axis = (1, 1)
    task._fixture_body_ids = lambda *a, **kw: {1}
    task._fixture_contact_distances = lambda *a: [0.0]
    task._fixture_is_upright = lambda *a: True
    task._fixture_bottom_is_on_support_plane = lambda: True
    task._fixture_has_illegal_penetration = lambda *a: False
    task._fixture_released = lambda *a: True
    assert task._bottle_is_on_counter()
    # This lies within the overall counter bounds, but in the sink gap.
    position[:2] = [-2.9, 1.757]
    assert not task._bottle_is_on_counter()
    position[:2] = [-2.9, 2.3]
    assert not task._bottle_is_on_counter()


@pytest.mark.parametrize("cls", (PutClose, TakeStandWalk))
def test_headless_410_initialization_and_physical_goals(cls):
    import robosuite
    from robocasa.g1_host.bridge import load_sim2real_controller_config
    from validate_ithor_tasks import (
        _assert_body_contact_state,
        _settle_scene_with_frozen_robot,
        _task_body_contact_contracts,
    )

    env = robosuite.make(
        env_name=cls.__name__,
        robots="G1FloatingBody",
        controller_configs=load_sim2real_controller_config("sonic", "mock"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=True,
        control_freq=20,
        renderer="mujoco",
        scene_ids=[410],
        seed=0,
    )
    try:
        env.reset()
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert not env._raw_success() and not env._check_success()
        assert env.HOLD_STEPS == 20
        assert env._routine_stage == env._routine_hold == 0
        assert env._robot_base_xy() == pytest.approx([-2.0, 2.57])
        expected_fraction = 0.2 if cls is PutClose else 1.0
        assert env.drawer_joint.fraction(env.sim) == pytest.approx(expected_fraction)
        if cls is TakeStandWalk:
            assert env.target_fixture.stable_id == "FloorPlan410:Toilet_1_4"
            assert env._initial_toilet_distance >= 0.5
        initial = env.source_fixture.world_center(env.sim).copy()
        _settle_scene_with_frozen_robot(env, int(0.5 / env.sim.model.opt.timestep))
        assert np.linalg.norm(env.source_fixture.world_center(env.sim) - initial) < 0.01
        assert env.count_body_collisions() == 0
        for name, subjects, supports in _task_body_contact_contracts(env):
            _assert_body_contact_state(env, name, subjects, supports, "after settle")
        if cls is PutClose:
            # Check that a sideways roll physically fits the closed drawer.
            env._set_fixture_quaternion(env.source_fixture, [1, 0, 0, 0])
            target_xy = env.drawer_support_region.world_pose(env.sim)[0][:2]
            assert env._place_fixture_on_support(
                env.source_fixture,
                env.drawer_support_region,
                env.drawer_fixture,
                target_xy,
                contract_name="composite_roll_in_closed_drawer",
            )
            assert env._raw_success()
            assert (
                not env._check_success()
            )  # Physical final state alone cannot skip stages.
        else:
            env._set_fixture_quaternion(
                env.source_fixture, env.scene_task_spec()["source_quat"]
            )
            target_xy = env.work_surface_region.world_pose(env.sim)[0][:2]
            assert env._place_fixture_on_support(
                env.source_fixture,
                env.work_surface_region,
                env.work_surface_fixture,
                target_xy,
                contract_name="composite_bottle_on_counter",
            )
            assert env._bottle_is_on_counter()
            assert not env._raw_success()  # Robot is still at the drawer stance.
            _settle_scene_with_frozen_robot(env, int(0.5 / env.sim.model.opt.timestep))
            assert env._bottle_is_on_counter()
            # The front strip is narrower than the bottle bbox, but supports it.
            rim = env.counter_support_regions[3]
            rim_xy = [-2.9, rim.world_pose(env.sim)[0][1]]
            probe_region = replace(
                rim, half_size=np.maximum(rim.half_size, [0.2, 0.2, 0.0])
            )
            assert env._place_fixture_on_support(
                env.source_fixture,
                probe_region,
                env.work_surface_fixture,
                rim_xy,
                contract_name="composite_bottle_on_narrow_counter_rim",
            )
            assert not env._fixture_bbox_inside_region_xy(env.source_fixture, rim)
            assert rim.contains_xy(env.source_fixture.world_center(env.sim), env.sim)
            assert env._bottle_is_on_counter()
            _settle_scene_with_frozen_robot(env, int(0.5 / env.sim.model.opt.timestep))
            assert env._bottle_is_on_counter()
            assert env.get_ep_meta()["placement_regions"] == list(
                cls.COUNTER_REGION_IDS
            )
            # Inject a valid nearby pose to test geometry, not a walking policy.
            assert env._set_robot_free_pose([-1.8, 3.9], 0.7)
            assert env.count_body_collisions() == 0
            assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
            assert env._raw_success()
            assert not env._check_success()  # Still cannot skip the ordered skills.
        joint = env._fixture_free_joint_name(env.source_fixture)
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        qpos[2] += 0.05
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        assert not env._raw_success()
        if cls is TakeStandWalk:
            assert not env._bottle_is_on_counter()
    finally:
        env.close()
