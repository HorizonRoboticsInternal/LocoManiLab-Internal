"""Order, recovery and fixed-episode isolation for the five new routines."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.environments.ithor_bathroom.tasks import ATOMIC_TASK_CLASSES
from robocasa.environments.ithor_bathroom.tasks.composite.routines import (
    TASK_CLASSES, _BathroomRoutineMixin, _Bottle, _ShelfSoap, _Sponge,
    IThorBathroomApproachToiletAndOpenToiletLidThenLiftScrubBrush as BRUSH_TASK,
    IThorBathroomCloseToiletLidThenApproachSinkAndLiftSoapBottle as SOAP_TASK,
    IThorBathroomOpenUpperDrawerThenLiftToiletPaperRollAndCloseUpperDrawer as DRAWER_TASK,
)
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_spec,
)
from robocasa.environments.ithor_bathroom.tasks.reorientation import _BathroomNativeUprightTask
from robocasa.environments.ithor_bathroom.tasks.shelf_pnp import _BathroomShelfPnPMixin
from robocasa.environments.ithor_bathroom.tasks.bathtub_tasks import _BathroomSpongeInBathtubTask
from robocasa.environments import ithor_task_catalog as catalog
from robosuite.environments.base import REGISTERED_ENVS
from test_ithor_bathroom_articulation_cycle_reset import _load_fixed_reset


def test_shelf_routine_spawn_override_preserves_atomic_items_and_other_routine():
    from robocasa.environments.ithor_bathroom.tasks.composite.routine_specs import (
        _atomic, tidy_specs,
    )

    shelf = tidy_specs(shelf=True)["FloorPlan401"]
    navigation = tidy_specs()["FloorPlan401"]
    assert shelf["start_pose"] == dict(xy=[-2.57432, 3.38], yaw=np.pi / 2)
    assert shelf["navigation"] is None
    assert navigation["start_pose"] == _atomic(401, "navigate_to_bathing_area")["start_pose"]
    for name, key in (("soap", "soap_to_lower_shelf"),
                      ("bottle", "stand_soap_bottle_upright"),
                      ("sponge", "put_sponge_in_bathtub")):
        assert shelf["items"][name] == _atomic(401, key)
        if name != "soap":
            assert shelf["items"][name] == navigation["items"][name]
            assert shelf["items"][name] is not navigation["items"][name]
    shelf["start_pose"]["xy"][1] = 99.
    shelf["items"]["bottle"]["source_xy"][0] = 99.
    assert tidy_specs(shelf=True)["FloorPlan401"]["start_pose"]["xy"][1] == 3.38
    assert _atomic(401, "stand_soap_bottle_upright")["source_xy"][0] != 99.


def _machine(monkeypatch, cls, frequency):
    monkeypatch.setattr(IThorBathroomEnv, "_post_action", lambda self, action: None)
    task = object.__new__(cls)
    task.scene_name = f"FloorPlan{cls.SUPPORTED_SCENE_IDS[0]}"
    task.control_freq = frequency
    task.PRINT_STAGE_MESSAGES = False
    state = dict(goals={s: False for s in cls.COMPOSITE_ORDERING},
                 contact=True, height=0., invariants=True)
    source = SimpleNamespace(source_fixture=SimpleNamespace(body_name="source"))
    task._skills = {"source": source}
    task._interior = None
    task._source_height = lambda: state["height"]
    task._source_grasped = lambda: state["contact"]
    task._source_grasp_is_at_start = lambda: True
    task._stage_geometry = lambda stage: state["goals"][stage]
    task._stage_goal = lambda stage: state["goals"][stage]
    task._stage_contact = lambda stage: state["contact"]
    task._final_invariants = lambda: state["invariants"]
    task.sim = SimpleNamespace(
        model=SimpleNamespace(body_name2id=lambda _: 0),
        data=SimpleNamespace(body_xpos=np.zeros((1, 3))),
        get_state=lambda: deepcopy(state),
        set_state=lambda saved: (state.clear(), state.update(deepcopy(saved))),
        forward=lambda: None,
    )
    task.robot_base_pos = np.zeros(3)
    task.robot_base_ori = np.zeros(3)
    task.reset = task._reset_task_progress
    task._get_observations = lambda **kwargs: dict(success=task._check_success())
    task._reset_task_progress()
    return task, state


def _steps(task, number):
    for _ in range(number):
        task._post_action(None)


def _finish_stage(task, state, *, steps=None):
    stage = task.COMPOSITE_ORDERING[task._routine_stage]
    if stage.startswith("lift_"):
        state["height"] = 0.
        _steps(task, 1)
        state["height"] = .31
    state["goals"][stage] = True
    _steps(task, steps if steps is not None else
           (task.control_freq if task._routine_stage == 2 else 5))


@pytest.mark.parametrize("cls", TASK_CLASSES)
@pytest.mark.parametrize("frequency", (20, 50))
def test_order_holds_interruption_and_queries(monkeypatch, cls, frequency):
    task, state = _machine(monkeypatch, cls, frequency)
    # Waiting cannot complete an action; queries do not advance a control tick.
    _steps(task, frequency * 2)
    assert task._routine_stage == 0 and not task._check_success()
    for expected in (1, 2):
        _finish_stage(task, state, steps=4)
        assert task._routine_stage == expected - 1
        for _ in range(100):
            assert not task._check_success()
        stage = task.COMPOSITE_ORDERING[expected - 1]
        state["goals"][stage] = False
        _steps(task, 1)
        assert task._routine_hold == 0
        _finish_stage(task, state)
        assert task._routine_stage == expected and task._routine_hold == 0
    _finish_stage(task, state, steps=frequency - 1)
    assert not task._check_success()
    state["invariants"] = False
    _steps(task, 1)
    assert task._routine_hold == 0
    state["invariants"] = True
    _steps(task, frequency)
    assert task._check_success()
    state["goals"][cls.COMPOSITE_ORDERING[-1]] = False
    _steps(task, 1)
    assert not task._check_success()


@pytest.mark.parametrize("cls", TASK_CLASSES)
@pytest.mark.parametrize("frequency", (20, 50))
@pytest.mark.parametrize("phase", (0, 1, 2, 3))
def test_actual_fixed_reset_clears_every_stage(monkeypatch, cls, frequency, phase):
    task, state = _machine(monkeypatch, cls, frequency)
    _load_fixed_reset()(task)
    for _ in range(2):
        for _ in range(min(phase, 2)):
            _finish_stage(task, state)
        _finish_stage(task, state, steps=frequency if phase == 3 else 2)
        task.reset()
        assert task._routine_stage == task._routine_hold == 0
        assert not task._stage_touched and not task._grasped_at_source
        assert not task._check_success()
        _steps(task, frequency * 2)
        assert task._routine_stage == 0 and not task._check_success()
    for _ in range(3):
        _finish_stage(task, state)
    assert task._check_success()


@pytest.mark.parametrize("cls", TASK_CLASSES)
def test_precompleted_future_operation_requires_recovery(monkeypatch, cls):
    task, state = _machine(monkeypatch, cls, 20)
    final = cls.COMPOSITE_ORDERING[-1]
    state["goals"][final] = True
    _finish_stage(task, state)
    _finish_stage(task, state)
    assert not task._stage_armed
    _steps(task, 40)
    assert not task._check_success()
    state["goals"][final] = False
    _steps(task, 1)
    _finish_stage(task, state)
    assert task._check_success()


@pytest.mark.parametrize("cls", TASK_CLASSES[:-1])
def test_gravity_without_hand_interaction_cannot_complete_operations(monkeypatch, cls):
    task, state = _machine(monkeypatch, cls, 20)
    if cls.COMPOSITE_ORDERING[0].startswith("approach_"):
        _finish_stage(task, state)
    state["contact"] = False
    state["goals"][cls.COMPOSITE_ORDERING[task._routine_stage]] = True
    stage = task._routine_stage
    _steps(task, 40)
    assert task._routine_stage == stage and not task._check_success()


def test_upper_drawer_stages_use_target_state_without_contact():
    task = object.__new__(DRAWER_TASK)
    assert DRAWER_TASK._stage_contact(task, "open_upper_drawer")
    assert DRAWER_TASK._stage_contact(task, "lift_toilet_paper_roll")
    assert DRAWER_TASK._stage_contact(task, "close_upper_drawer")


def test_five_registrations_and_unchanged_predicate_reuse():
    assert len(TASK_CLASSES) == 5
    for cls in TASK_CLASSES:
        assert cls.SUPPORTED_SCENE_IDS in ((401,), (401, 405, 407, 415, 418), (415,))
        assert REGISTERED_ENVS[cls.__name__] is cls
        assert not any(atomic in cls.__mro__ for atomic in ATOMIC_TASK_CLASSES)
        assert len(cls.COMPOSITE_ORDERING) == 3
    assert _Bottle._raw_success is _BathroomNativeUprightTask._raw_success
    assert _Sponge._raw_success is _BathroomSpongeInBathtubTask._raw_success
    assert _ShelfSoap._object_is_on_target_shelf_and_released is _BathroomShelfPnPMixin._object_is_on_target_shelf_and_released
    assert all(REGISTERED_ENVS.get(name) is None for name in
               ("_BathroomRoutineMixin", "_Bottle", "_Sponge", "_ShelfSoap", "_StoolLift"))
    entries = catalog.discover_task_catalog((catalog.BATHROOM_PACKAGE,))
    assert len(entries) == 61
    assert sum(e.family == "composite" for e in entries) == 40
    signatures = [catalog._task_signature(e, scene) for e in entries for scene in e.scene_ids]
    assert len(signatures) == len(set(signatures))
    for removed in ("IThorBathroomApproachSinkAndOpenCabinetDoorThenLiftSprayBottle",
                    "IThorBathroomOpenLowerDrawerThenLiftToiletPaperRollAndCloseLowerDrawer"):
        assert removed not in REGISTERED_ENVS
        assert all(e.name != removed for e in entries)



def test_toilet_routines_have_independent_five_scene_stool_layouts():
    scenes = (401, 405, 407, 415, 418)
    expected_layouts = {
        SOAP_TASK: {
            401: ([-2.70, 2.20], [-2.25, 2.15], 3.030935),
            405: ([-2.55, -0.25], [-2.05, -0.25], 3.141593),
            407: ([-0.90, -0.90], [-0.35, -0.90], 3.141593),
            415: ([-0.75, -2.55], [-1.35, -2.55], 0.0),
            418: ([-1.80, -3.65], [-1.30, -3.65], 3.141593),
        },
        BRUSH_TASK: {
            401: ([-0.90, 2.80], [-1.40, 2.80], 0.0),
            405: ([-2.50, 0.85], [-1.90, 0.85], 0.0),
            407: ([-1.15, 0.20], [-0.55, 0.20], 0.0),
            415: ([-1.65, -0.85], [-1.05, -0.85], 0.0),
            418: ([-0.90, -3.55], [-1.50, -3.55], 0.0),
        },
    }
    soap_start_poses = {
        405: {"xy": [-2.393, 1.645], "yaw": -2.487094},
        418: {"xy": [-0.449, -2.495], "yaw": -1.832596},
    }
    for task, layouts in expected_layouts.items():
        assert task.SUPPORTED_SCENE_IDS == scenes
        assert set(task.SCENE_SPECS) == {f"FloorPlan{scene}" for scene in scenes}
        item_key = "lift_soap_bottle" if task is SOAP_TASK else "lift_scrub_brush"
        nav_key = "approach_sink" if task is SOAP_TASK else "approach_toilet"
        for scene, (xy, robot_xy, yaw) in layouts.items():
            scene_name = f"FloorPlan{scene}"
            spec = task.SCENE_SPECS[scene_name]
            item = spec["items"]["source"]
            atomic_item = get_ithor_bathroom_task_spec(scene_name, item_key)
            joint = get_ithor_bathroom_task_spec(scene_name, "toilet_lid")
            navigation = get_ithor_bathroom_task_spec(scene_name, nav_key)
            assert item["source_fixture"] == atomic_item["source_fixture"]
            assert item["source_xy"] == xy
            assert item["robot_pose"] == {"xy": robot_xy, "yaw": yaw}
            assert item["surface_fixture"] == f"{scene_name}:routine_stool"
            assert item["surface_region"] == f"{scene_name}:routine_stool:region:support"
            assert item["auxiliary_placements"] == []
            assert item["source_stool"]["xy"] == xy
            assert item["source_stool"]["top_half_xy"] == (
                [0.13, 0.13] if task is SOAP_TASK else [0.13, 0.28]
            )
            assert spec["articulation"]["fixture"] == joint["fixture"]
            assert spec["navigation"] == navigation
            expected_start_pose = (
                soap_start_poses.get(scene, joint["close_robot_pose"])
                if task is SOAP_TASK else navigation["start_pose"]
            )
            assert spec["start_pose"] == expected_start_pose
            if task is SOAP_TASK and scene in soap_start_poses:
                assert spec["start_pose"] != joint["close_robot_pose"]
            elif task is SOAP_TASK:
                assert spec["start_pose"] == joint["close_robot_pose"]
            assert atomic_item["surface_fixture"] != item["surface_fixture"]


def test_upper_drawer_spec_does_not_change_collected_lower_drawer():
    from robocasa.environments.ithor_bathroom.tasks.composite.routine_specs import _atomic

    original = _atomic(415, "open_lower_drawer")
    spec = DRAWER_TASK.SCENE_SPECS["FloorPlan415"]
    assert original["joint"].endswith("_1_6_0_DrawerMesh031_3")
    assert spec["articulation"]["joint"].endswith("_1_4_0_DrawerMesh021_2")
    assert spec["items"]["source"]["interior_region"].endswith(":receptacle:6")
    assert spec["items"]["source"]["source_fixture"] == _atomic(415, "lift_toilet_paper_roll")["source_fixture"]
    for key in ("init_open_fraction", "success_open_fraction", "success_close_fraction"):
        assert spec["articulation"][key] == original[key]


@pytest.mark.parametrize("center_x, height, expected", ((-.06, .015, True),
    (0., 0., True), (-.20, .015, False), (0., .11, False), (0., -.11, False)))
def test_drawer_grasp_accepts_partial_overhang_but_not_outside_or_wrong_height(center_x, height, expected):
    task = object.__new__(DRAWER_TASK)
    task.sim = None
    task._skills = {"source": SimpleNamespace(source_fixture=object())}
    task._initial_heights = {"source": 0.}
    task._source_height = lambda: height
    task._interior = SimpleNamespace(contains_xy=lambda xy, sim: abs(xy[0]) <= .095 and abs(xy[1]) <= .289)
    # A tilted / slightly pulled roll can overhang while its centre is inside.
    task._fixture_bbox_points = lambda _: np.array([
        [center_x - .08, -.06, height - .06], [center_x + .08, .06, height + .06]])
    assert task._source_grasp_is_at_start() is expected


def test_specs_are_independent_of_atomics_and_other_routines():
    atomic_before = {c: deepcopy(c.SCENE_SPECS) for c in ATOMIC_TASK_CLASSES}
    second_before = deepcopy(TASK_CLASSES[1].SCENE_SPECS)
    task = object.__new__(TASK_CLASSES[0])
    task.scene_name = "FloorPlan401"
    spec = task.scene_task_spec()
    spec["items"]["bottle"]["source_xy"][0] = 100
    assert TASK_CLASSES[1].SCENE_SPECS == second_before
    assert all(c.SCENE_SPECS == s for c, s in atomic_before.items())
    assert task.scene_task_spec()["items"]["bottle"]["source_xy"][0] != 100


@pytest.mark.parametrize("fraction, opened, closed", ((.8, True, False), (.2, False, True), (.5, False, False)))
def test_toilet_thresholds(fraction, opened, closed):
    task = object.__new__(TASK_CLASSES[3])
    task.scene_name = "FloorPlan401"
    task._routine_joint = SimpleNamespace(fraction=lambda _: fraction)
    task.sim = None
    assert task._stage_goal("open_toilet_lid") == opened
    assert task._stage_goal("close_toilet_lid") == closed


def test_lift_requires_low_grasp_height_and_storage_exit(monkeypatch):
    task, state = _machine(monkeypatch, DRAWER_TASK, 20)
    task._stage_geometry = lambda _: state["height"] >= .30
    task._outside_storage = lambda: state.get("outside", True)
    task._source_clear = lambda: state.get("clear", True)
    goal = lambda: _BathroomRoutineMixin._stage_goal(task, "lift_toilet_paper_roll")
    state["height"] = .30
    assert not goal()
    task._grasped_at_source = True
    assert goal()
    for key in ("outside", "clear", "contact"):
        state[key] = False
        assert not goal()
        state[key] = True
    state["height"] = .299
    assert not goal()


def test_upper_drawer_lift_stage_uses_height_only_goal():
    task = object.__new__(DRAWER_TASK)
    task._stage_geometry = lambda stage: stage == "lift_toilet_paper_roll"
    assert DRAWER_TASK._stage_goal(task, "lift_toilet_paper_roll")
    task._stage_geometry = lambda stage: False
    assert not DRAWER_TASK._stage_goal(task, "lift_toilet_paper_roll")


def test_other_routines_keep_shared_lift_goal():
    task = object.__new__(TASK_CLASSES[2])
    task._stage_geometry = lambda stage: True
    task._grasped_at_source = False
    task._source_grasped = lambda: True
    task._outside_storage = lambda: True
    task._source_clear = lambda: True
    assert not TASK_CLASSES[2]._stage_goal(task, "lift_soap_bottle")


def test_drawer_final_allows_lowering_release_and_hand_change_but_not_return(monkeypatch):
    task, state = _machine(monkeypatch, DRAWER_TASK, 20)
    task._outside_storage = lambda: state.get("outside", True)
    task._source_clear = lambda: True
    final = lambda: DRAWER_TASK._final_invariants(task)
    state["height"] = -.1  # Earlier lift is latched; final need not stay raised.
    assert final()
    state["contact"] = False
    assert final()
    state["contact"] = True  # Either hand is accepted by the fixture primitive.
    assert final()
    state["outside"] = False
    assert not final()


def test_native_grasp_requires_closed_hand_contact_and_accepts_either_hand():
    task = object.__new__(DRAWER_TASK)
    task._skills = {"source": SimpleNamespace(source_fixture=object())}
    closed = {"left": False, "right": False}
    touching = {"left": True, "right": False}
    task.g1_hand_closed = lambda side: closed[side]
    task._g1_hand_contacts_fixture = lambda side, fixture: touching[side]
    assert not task._source_grasped()
    closed["left"] = True
    assert task._source_grasped()
    closed["left"] = touching["left"] = False
    closed["right"] = touching["right"] = True
    assert task._source_grasped()
    touching["right"] = False
    assert not task._source_grasped()


@pytest.mark.parametrize("cls", TASK_CLASSES)
def test_normal_reset_uses_the_same_progress_cleanup(monkeypatch, cls):
    task, state = _machine(monkeypatch, cls, 20)
    def physical_reset(self):
        state["goals"] = {s: False for s in cls.COMPOSITE_ORDERING}
        state["height"] = 0.
    monkeypatch.setattr(IThorBathroomEnv, "_reset_internal", physical_reset)
    task._skills["source"]._reset_internal = lambda: None
    task.count_body_collisions = lambda: 0
    task.robot_footprint_is_on_floor = lambda *a, **kw: True
    task._robot_base_xy = lambda: np.zeros(2)
    task._get_robot_free_joint = lambda: "base"
    task.sim.data.get_joint_qpos = lambda _: np.zeros(7)
    task._navigation_fixture = None
    task.robot_fixture_collision_distance = lambda *a, **kw: 3.
    for _ in range(3):
        _finish_stage(task, state)
    assert task._check_success()
    task._reset_internal()
    assert not task._check_success()
    assert task._routine_stage == task._routine_hold == 0
    assert not task._stage_touched and not task._grasped_at_source


@pytest.mark.parametrize("cls", TASK_CLASSES)
def test_metadata_describes_final_height_and_storage_contract(monkeypatch, cls):
    task, state = _machine(monkeypatch, cls, 20)
    monkeypatch.setattr(IThorBathroomEnv, "get_ep_meta", lambda self: {})
    if cls in TASK_CLASSES[:2]:
        task._skills = {}
    if cls in TASK_CLASSES[4:]:
        task._interior = object()
    meta = task.get_ep_meta()
    assert meta["composite_ordering"] == list(cls.COMPOSITE_ORDERING)
    assert meta["final_hold_seconds"] == 1.
    assert meta["intermediate_hold_steps"] == 5
    assert meta["final_grasp_required"] == (cls in TASK_CLASSES[2:4])
    assert meta["final_lift_height_required"] == (cls in TASK_CLASSES[2:4])
    assert meta["storage_exit_required"] == (cls in TASK_CLASSES[4:])
