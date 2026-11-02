"""Fast sequence and physical-predicate contracts; no environment construction."""

import ast
from pathlib import Path
import types
from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.ithor_bedroom.tasks import IThorBedroomCloseHamperLid
from robocasa.environments.ithor_bedroom.tasks.routine_composites import (
    TASK_CLASSES,
    _ThreeStageRoutineMixin,
    _RoutineBasketMixin,
    _RoutineMugLaptopMixin,
    IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop,
    IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk,
    IThorBedroomPutCanThenBoxedFoodThenMugInBasket,
)
from robocasa.models.scenes.ithor_bedroom_task_specs import get_ithor_bedroom_task_spec


class _Terminal:
    def _post_action(self, action):
        return self._check_success(), self._check_success(), {}

    def _reset_internal(self):
        self.geometry = dict.fromkeys(self.COMPOSITE_ORDERING, False)


class Routine(_ThreeStageRoutineMixin, _Terminal):
    PRINT_STAGE_MESSAGES = False

    def __init__(self, cls, frequency=20):
        self.COMPOSITE_ORDERING = cls.COMPOSITE_ORDERING
        self.geometry = dict.fromkeys(self.COMPOSITE_ORDERING, False)
        self.support = dict.fromkeys(self.COMPOSITE_ORDERING, True)
        self.contacts = set()
        self.control_freq = frequency
        self.xy = np.zeros(2)
        self.sim = SimpleNamespace(
            forward=lambda: None,
            data=SimpleNamespace(get_joint_qpos=lambda _: np.zeros(7)),
        )
        self._reset_task_progress()

    def scene_task_spec(self):
        return {"routine_start": None}

    def _initialize_routine_layout(self):
        pass

    def _get_robot_free_joint(self):
        return "robot_root"

    def count_body_collisions(self):
        return 0

    def robot_footprint_is_on_floor(self, *args, **kwargs):
        return True

    def _robot_base_xy(self):
        return self.xy.copy()

    def _routine_geometry(self, stage):
        return self.geometry[stage]

    def _routine_goal(self, stage):
        return self.geometry[stage] and self.support[stage]

    def _routine_contact(self, stage):
        return stage in self.contacts

    def step(self, n=1):
        result = None
        for _ in range(n):
            result = self._post_action(None)
        return result

    def finish_stage(self):
        stage = self.COMPOSITE_ORDERING[self._routine_stage]
        self.contacts = {stage}
        self.step()
        self.geometry[stage] = True
        self.contacts.clear()
        return self.step(self.control_freq if self._routine_stage == 2 else 5)


@pytest.mark.parametrize("cls", TASK_CLASSES)
@pytest.mark.parametrize("frequency", [10, 20])
def test_all_six_sequences_duration_and_observation_purity(cls, frequency):
    task = Routine(cls, frequency)
    assert not task._check_success()
    for index in range(2):
        task.finish_stage()
        assert task._routine_stage == index + 1
        assert task._routine_hold == 0
    stage = task.COMPOSITE_ORDERING[2]
    task.contacts = {stage}
    task.step()
    task.geometry[stage] = True
    task.contacts.clear()
    task.step(frequency - 1)
    for _ in range(50):
        assert not task._check_success()
    reward, done, _ = task.step()
    assert reward and done and task._check_success()
    task.geometry[stage] = False
    task.step()
    assert not task._check_success() and task._routine_hold == 0


@pytest.mark.parametrize("cls", TASK_CLASSES)
def test_early_operations_require_restore_and_fresh_correct_contact(cls):
    task = Routine(cls)
    for index, stage in enumerate(task.COMPOSITE_ORDERING):
        if stage.startswith("approach_"):
            task.finish_stage()
            continue
        # Simulate an operation completed before this stage was activated.
        task.geometry[stage] = True
        task._enter_routine_stage()
        task.contacts = {stage}
        task.step(25)
        assert task._routine_stage == index and not task._check_success()
        task.geometry[stage] = False
        task.contacts = {"wrong_target"}
        task.step()
        task.geometry[stage] = True
        task.step(25)
        assert task._routine_stage == index and not task._check_success()
        task.geometry[stage] = False
        task.finish_stage()
    assert task._check_success()


@pytest.mark.parametrize("cls", TASK_CLASSES)
def test_interruption_retained_results_and_repair(cls):
    task = Routine(cls)
    first = task.COMPOSITE_ORDERING[0]
    task.contacts = {first}
    task.step()
    task.geometry[first] = True
    task.step(4)
    task.geometry[first] = False
    task.step()
    assert task._routine_hold == 0
    task.finish_stage()
    task.finish_stage()
    retained = next(
        s for s in task.COMPOSITE_ORDERING[:2] if not s.startswith("approach_")
    )
    task.geometry[retained] = False
    task.finish_stage()
    assert not task._check_success() and task._routine_stage == 2
    task.geometry[retained] = True
    task.step(19)
    assert not task._check_success()
    task.step()
    assert task._check_success()
    task.support[retained] = False
    task.step()
    assert not task._check_success()


@pytest.mark.parametrize("cls", TASK_CLASSES)
@pytest.mark.parametrize("reset", ["_reset_internal", "_reset_task_progress"])
def test_normal_and_fixed_restore_clear_all_stage_state(cls, reset):
    task = Routine(cls)
    for _ in range(3):
        task.finish_stage()
    task.xy = np.array([1.0, 2.0])
    task.geometry = dict.fromkeys(task.COMPOSITE_ORDERING, False)
    getattr(task, reset)()
    assert task._routine_stage == task._routine_hold == 0
    assert not task._routine_touched and not task._routine_success
    np.testing.assert_array_equal(task._routine_entry_xy, task.xy)


def test_bed_navigation_requires_displacement_and_collision_distance():
    task = Routine(TASK_CLASSES[3])
    task._routine_geometry = _ThreeStageRoutineMixin._routine_geometry.__get__(task)
    task.scene_task_spec = lambda: {
        "routine_targets": {"approach_bed": {"fixture": "exact_bed"}}
    }
    task.get_fixture = lambda **kwargs: kwargs["id"]
    distances = {"exact_bed": 0.34}
    task.robot_fixture_collision_distance = lambda fixture, **kwargs: distances[fixture]
    assert not task._routine_geometry("approach_bed")
    task.xy[0] = 0.31
    assert task._routine_geometry("approach_bed")
    distances["exact_bed"] = 0.36
    assert not task._routine_geometry("approach_bed")


@pytest.mark.parametrize("scene_id", (302, 310, 323, 329))
def test_hamper_desk_new_scene_initial_setup_uses_atomic_targets(scene_id):
    scene = f"FloorPlan{scene_id}"
    task_class = IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk
    task = object.__new__(task_class)
    task.scene_name = scene
    spec = task_class.SCENE_SPECS[scene]
    navigation = get_ithor_bedroom_task_spec(scene, "approach_hamper_lid")
    desk = get_ithor_bedroom_task_spec(scene, "approach_desk")
    atomic = IThorBedroomCloseHamperLid.SCENE_SPECS[scene]
    assert (
        spec["routine_start"]
        == spec["routine_targets"]["approach_storage"]
        == navigation
    )
    assert spec["routine_targets"]["approach_desk"] == desk
    assert (
        spec["fixture"]
        == spec["routine_targets"]["close_hamper"]["fixture"]
        == atomic["fixture"]
    )
    assert (
        spec["joint"]
        == spec["routine_targets"]["close_hamper"]["joint"]
        == atomic["joint"]
    )
    assert spec["hamper_robot_pose"] == atomic["hamper_robot_pose"]
    joint = SimpleNamespace(closed_qpos=0.0, open_qpos=1.5708)
    assert task._initial_joint_qpos(joint) == pytest.approx(0.85 * joint.open_qpos)


@pytest.mark.parametrize("scene_id", (302, 310, 323, 329))
def test_hamper_desk_close_stage_uses_lid_state_only(scene_id):
    cls = IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk
    task = object.__new__(cls)
    task.scene_name = f"FloorPlan{scene_id}"
    task.COMPOSITE_ORDERING = cls.COMPOSITE_ORDERING
    task.control_freq = 20
    task.sim = object()
    task._routine_stage = 1
    task._routine_hold = 0
    task._routine_success = False
    task._routine_armed = True
    task._routine_touched = False
    task._routine_entry_xy = np.zeros(2)
    task._robot_base_xy = lambda: np.zeros(2)
    state = {"fraction": 0.10}
    task._routine_joints = {
        "close_hamper": (
            SimpleNamespace(fraction=lambda sim: state["fraction"]),
            0.12,
        )
    }
    task._routine_contact = lambda stage: False
    task._report_stage_complete = lambda *args, **kwargs: None
    task._enter_routine_stage = lambda: None

    # No hand contact and no robot-distance query are needed once navigation
    # has completed; the closed-lid state alone advances this stage.
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._advance_routine()
    assert task._routine_stage == 2


def test_hamper_desk_native_scene_keeps_original_start_and_aperture():
    cls = IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk
    task = object.__new__(cls)
    task.scene_name = "FloorPlan306"
    spec = cls.SCENE_SPECS[task.scene_name]
    assert spec["routine_start"] == get_ithor_bedroom_task_spec(
        task.scene_name, "approach_storage"
    )
    assert spec["routine_targets"]["approach_desk"] == get_ithor_bedroom_task_spec(
        task.scene_name, "approach_desk"
    )
    assert task._initial_joint_qpos(
        SimpleNamespace(closed_qpos=0.0, open_qpos=1.5708)
    ) == pytest.approx(1.5708)


@pytest.mark.parametrize("scene_id", [302, 306, 310, 323, 329])
@pytest.mark.parametrize("frequency", [10, 20])
def test_hamper_desk_final_displacement_is_scene_specific(scene_id, frequency):
    cls = IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk
    task = object.__new__(cls)
    task.PRINT_STAGE_MESSAGES = False
    task.control_freq = frequency
    task.sim = object()
    task.scene_name = f"FloorPlan{scene_id}"
    spec = cls.SCENE_SPECS[task.scene_name]
    task.scene_task_spec = lambda: spec
    task.xy = np.array([-2.0, 0.0])
    task._robot_base_xy = lambda: task.xy.copy()
    task.get_fixture = lambda **kwargs: kwargs["id"]
    desk = spec["routine_targets"]["approach_desk"]["fixture"]
    distances = {
        desk: 0.2,
        spec["routine_targets"]["approach_storage"]["fixture"]: 0.2,
    }
    task.robot_fixture_collision_distance = lambda fixture, **kwargs: distances[fixture]
    state = dict(fraction=1.0, touching=False)
    task._routine_joints = {
        "close_hamper": (
            SimpleNamespace(fraction=lambda sim: state["fraction"]),
            0.12,
        )
    }
    task._routine_contact = lambda stage: state["touching"]
    task._reset_task_progress()

    def advance(count):
        for _ in range(count):
            task._advance_routine()

    advance(5)
    assert task._routine_stage == 1
    task.xy = np.zeros(2)
    state.update(touching=True, fraction=0.1)
    advance(5)
    state["touching"] = False
    assert task._routine_stage == 2

    np.testing.assert_array_equal(task._routine_entry_xy, task.xy)
    if scene_id == 306:
        # Earlier navigation cannot count toward the final-stage displacement.
        advance(frequency)
        assert not task._check_success()
        task.xy = np.array([0.99, 0.0])
        advance(frequency)
        assert not task._check_success()
        task.xy = np.array([1.0, 0.0])
        advance(frequency - 1)
        assert not task._check_success()
        # Both desktop proximity and the closed lid remain necessary.
        distances[desk] = 0.36
        advance(1)
        assert task._routine_hold == 0
        distances[desk] = 0.2
        advance(frequency - 1)
        state["fraction"] = 1.0
        advance(1)
        assert task._routine_hold == 0
        state["fraction"] = 0.1

    # Other scenes retain the existing one-second hold without displacement.
    advance(frequency - 1)
    assert not task._check_success()
    advance(1)
    assert task._check_success()


@pytest.mark.parametrize("scene_id", [302, 306, 310, 323, 329])
def test_hamper_desk_displacement_metadata_is_scene_specific(monkeypatch, scene_id):
    monkeypatch.setattr(_ThreeStageRoutineMixin, "get_ep_meta", lambda self: {})
    task = object.__new__(IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk)
    task.scene_name = f"FloorPlan{scene_id}"
    meta = task.get_ep_meta()
    if scene_id == 306:
        assert meta["final_navigation_min_displacement"] == 1.0
    else:
        assert "final_navigation_min_displacement" not in meta


def test_larger_basket_is_isolated_and_matches_recorded_asset():
    from robocasa.environments.ithor_bedroom.tasks.routine_composites import _ASSETS

    enlarged = IThorBedroomPutCanThenBoxedFoodThenMugInBasket
    for cls in TASK_CLASSES:
        if not cls.ITEM_NAMES:
            continue
        task = object.__new__(cls)
        configs = {cfg["name"]: cfg for cfg in task._get_obj_cfgs()}
        expected = _ASSETS["basket"][1] * (1.2 if cls is enlarged else 1.0)
        assert configs["basket"]["object_scale"] == pytest.approx(expected)
        for item in cls.ITEM_NAMES:
            assert configs[item]["object_scale"] == _ASSETS[item][1]
            target = cls.SCENE_SPECS["FloorPlan329"]["routine_targets"][f"put_{item}"]
            assert target["basket"][0] == _ASSETS["basket"][0]
            assert target["basket"][1] == pytest.approx(expected)
        configs["basket"]["object_scale"] = 999.0
        assert task._get_obj_cfgs()[0]["object_scale"] == pytest.approx(expected)


@pytest.mark.parametrize("can_move", [True, False])
def test_mug_laptop_layout_requires_robot_clear_before_support_probe(can_move):
    cls = IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop
    task = object.__new__(_RoutineMugLaptopMixin)
    task.COMPOSITE_ORDERING = cls.COMPOSITE_ORDERING
    task.scene_name = "FloorPlan329"
    spec = cls.SCENE_SPECS[task.scene_name]
    task.scene_task_spec = lambda: spec
    task.target_semantic_joint = object()
    task.laptop_work_surface_region = object()
    task.laptop_work_surface_fixture = object()
    task.sim = object()
    task.target_fixture = SimpleNamespace(
        world_center=lambda sim: np.array([1.0, 2.0, 3.0])
    )
    task._joint_interaction_pos = task._joint_target_pos = lambda joint: np.zeros(3)
    moves, probes = [], []

    def move(xy, yaw):
        moves.append((xy, yaw))
        return can_move

    def probe(*, laptop):
        assert can_move and moves
        pose = spec["routine_start"]["far_start_pose"]
        np.testing.assert_array_equal(moves[-1][0], pose["xy"])
        assert moves[-1][1] == pose["yaw"]
        assert task.work_surface_region is task.laptop_work_surface_region
        assert task.work_surface_fixture is task.laptop_work_surface_fixture
        probes.append(laptop)

    task._set_robot_free_pose = move
    task._layout_basket_row = probe
    if can_move:
        task._initialize_routine_layout()
        assert probes == [True]
        assert task._routine_joints == {
            "close_laptop": (task.target_semantic_joint, 0.12)
        }
    else:
        with pytest.raises(RuntimeError, match="FloorPlan329: cannot move robot clear"):
            task._initialize_routine_layout()
        assert not probes


def test_stack_support_chain_requires_released_contained_anchored_items():
    task = object.__new__(_RoutineBasketMixin)
    task.ITEM_NAMES = ("can", "boxed_food", "mug")
    task.work_surface_region = object()
    valid, held = set(task.ITEM_NAMES), set()
    task._receptacle_upright_released_on_region = lambda *args: True
    task._inside_routine_basket = lambda name: name in valid
    task._composite_object_released = lambda name: name not in held
    task._routine_no_penetration = lambda name: True
    edges = {("can", "basket"), ("boxed_food", "basket"), ("mug", "boxed_food")}
    task._load_bearing_contact = lambda upper, lower: (upper, lower) in edges
    assert task._supported_basket_items() == valid
    held.add("boxed_food")
    assert task._supported_basket_items() == {"can"}
    held.clear()
    valid.remove("boxed_food")
    assert task._supported_basket_items() == {"can"}
    valid.add("boxed_food")
    edges.remove(("boxed_food", "basket"))
    edges.add(("boxed_food", "mug"))
    assert task._supported_basket_items() == {"can"}  # unanchored contact cycle


def test_release_allows_nearby_hands_but_not_contact_or_grasp():
    task = object.__new__(_RoutineBasketMixin)
    task._hands_touch_routine_object = lambda name: False
    task.g1_hand_grasping_object = lambda name: False
    task.object_released = lambda *args, **kwargs: pytest.fail(
        "distance release must not be used"
    )
    assert task._composite_object_released("mug")
    task._hands_touch_routine_object = lambda name: True
    assert not task._composite_object_released("mug")


@pytest.mark.parametrize(
    "normal,force,distance,swapped,expected",
    [
        ((0, 0, 1), 1.0, -0.001, False, True),
        ((0, 0, -1), 1.0, -0.001, True, True),
        ((1, 0, 0), 1.0, -0.001, False, False),
        ((0, 0, 1), 0.0, -0.001, False, False),
        ((0, 0, 1), 1.0, -0.004, False, False),
    ],
)
def test_contact_chain_rejects_side_touch_unloaded_and_deep_contacts(
    monkeypatch, normal, force, distance, swapped, expected
):
    import robocasa.environments.ithor_bedroom.tasks.routine_composites as module

    task = object.__new__(_RoutineBasketMixin)
    task.obj_body_id = {"upper": 1, "lower": 2}
    task._body_tree_ids = set
    contact = SimpleNamespace(geom1=0, geom2=1, frame=normal, dist=distance)
    task.sim = SimpleNamespace(
        model=SimpleNamespace(geom_bodyid=[1, 2] if swapped else [2, 1], _model=None),
        data=SimpleNamespace(ncon=1, contact=[contact], _data=None),
    )
    monkeypatch.setattr(
        module.mujoco,
        "mj_contactForce",
        lambda model, data, index, result: result.__setitem__(0, force),
    )
    assert task._load_bearing_contact("upper", "lower") is expected


@pytest.mark.parametrize(
    "xy,bottom,name,expected",
    [
        ((0, 0), 0.02, "can", True),
        ((0.10, 0), 0.02, "can", False),  # outside wall
        ((0, 0), 0.119, "can", False),  # suspended above the rim
        ((0, 0), -0.01, "can", False),  # below the basket floor
        ((0, 0), -0.028, "mug", True),  # canonical mug bottom tolerance
        ((0, 0), 0.08, "mug", True),  # stacked, top protruding above rim
    ],
)
def test_basket_geometry_keeps_atomic_tolerances_and_allows_stacking(
    xy, bottom, name, expected
):
    task = object.__new__(_RoutineBasketMixin)
    task.objects = {"basket": SimpleNamespace(root_body="basket")}
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            get_body_xpos=lambda _: np.zeros(3), get_body_xmat=lambda _: np.eye(3)
        )
    )
    bounds = np.array([[-0.10, -0.08, 0], [0.10, 0.08, 0.12]])
    source = np.array(
        [
            [xy[0] - 0.02, xy[1] - 0.02, bottom],
            [xy[0] + 0.02, xy[1] + 0.02, bottom + 0.10],
        ]
    )
    task.object_bbox_points = lambda item: bounds if item == "basket" else source
    assert task._inside_routine_basket(name) is expected


@pytest.mark.parametrize("cls", TASK_CLASSES)
def test_fixed_restore_invokes_real_progress_hook(cls):
    # Read just the actual restore function: importing the GUI collection
    # entrypoint requires optional viewer dependencies unrelated to this task.
    path = (
        Path(__file__).resolve().parents[1] / "robocasa/scripts/collect_fixed_task.py"
    )
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef) and node.name == "freeze_resets"
    )
    namespace = dict(
        np=np,
        types=types,
        colored=lambda message, color: message,
        _reset_controllers=lambda env: None,
        _refresh_after_restore=lambda env: None,
        apply_reset_pose_jitter=lambda *args, **kwargs: {
            "g1": np.zeros(2),
            "g1_ok": True,
        },
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    task = Routine(cls)
    task.reset = task._reset_internal
    task._get_observations = lambda **kwargs: {}
    task.sim.get_state = lambda: {key: False for key in task.geometry}
    task.sim.set_state = lambda state: task.geometry.update(state)
    namespace["freeze_resets"](task)
    for _ in range(3):
        task.finish_stage()
    task.reset()
    assert task._routine_stage == task._routine_hold == 0
    assert (
        task._routine_armed and not task._routine_touched and not task._routine_success
    )


def test_catalog_and_exact_single_scene_specifications():
    from robocasa.environments.ithor_task_catalog import (
        BEDROOM_PACKAGE,
        discover_task_catalog,
        validate_task_catalog,
    )
    from robocasa.environments.ithor_bedroom.tasks import assets
    from robocasa.environments.ithor_bedroom.tasks.routine_composites import _ASSETS

    entries = discover_task_catalog((BEDROOM_PACKAGE,))
    assert not validate_task_catalog(entries)
    assert len(entries) == 52
    assert sum(entry.family == "composite" for entry in entries) == 32
    from robosuite.environments.base import REGISTERED_ENVS

    assert (
        "IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk"
        in REGISTERED_ENVS
    )
    assert (
        "IThorBedroomApproachStorageAndCloseHamperLidThenApproachBed"
        not in REGISTERED_ENVS
    )
    assert [cls.SUPPORTED_SCENE_IDS for cls in TASK_CLASSES] == [
        (302, 306, 310, 323, 329),
        (302, 306, 310, 323, 329),
        (302, 306, 310, 323, 329),
        (302, 306, 310, 323, 329),
        (302, 306, 310, 323, 329),
        (302, 306, 310, 323, 329),
    ]
    for name in ("basket", "can", "boxed_food", "mug"):
        assert _ASSETS[name] == (
            getattr(assets, name.upper() + "_MODEL"),
            getattr(assets, name.upper() + "_SCALE"),
        )
    for cls in TASK_CLASSES:
        assert len(cls.COMPOSITE_ORDERING) == 3
        # Every concrete routine resolves a real layout initializer.
        assert (
            cls._initialize_routine_layout.__qualname__
            != "_ThreeStageRoutineMixin._initialize_routine_layout"
        )
