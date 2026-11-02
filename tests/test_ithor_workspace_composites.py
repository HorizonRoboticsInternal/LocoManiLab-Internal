"""Focused contracts for the ergonomic Bedroom/Bathroom composites."""

import numpy as np
import pytest

import robocasa.environments.ithor_bedroom.tasks.assets as bedroom_assets
import robocasa.models.scenes.ithor_bedroom_task_specs as bedroom_specs

from robocasa.environments.ithor_atomic.articulation import ExactSceneArticulation
from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_base import IThorBathroomEnv, IThorBedroomEnv
from robocasa.environments.ithor_bathroom.tasks import (
    IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin,
    IThorBathroomApproachSinkAndCloseCabinetDoor,
    IThorBathroomApproachSinkAndLiftSoapBottle,
    IThorBathroomApproachSinkAndOpenCabinetDoor,
    IThorBathroomApproachToiletAndOpenToiletLid,
    IThorBathroomCloseCabinetDoor,
    IThorBathroomLiftSoapBottle,
    IThorBathroomOpenCabinetDoor,
    IThorBathroomNavigateToBathingAreaAndLiftScrubBrush,
    IThorBathroomNavigateToBathingAreaAndLiftSoapBottle,
    IThorBathroomNavigateToBathingAreaAndStandSoapBottleUpright,
    IThorBathroomNavigateToSinkThenToilet,
    IThorBathroomCloseThenOpenCabinetDoor,
    IThorBathroomOpenThenCloseCabinetDoor,
    IThorBathroomOpenThenCloseToiletLid,
    IThorBathroomStandSoapBottleUpright,
    IThorBathroomToiletPaperRollIntoBin,
)
from robocasa.environments.ithor_bathroom.tasks.composite import (
    TASK_CLASSES as BATHROOM_COMPOSITES,
)
from robocasa.environments.ithor_bathroom.tasks.composite.approach_manipulation import (
    _ApproachThenOperationMixin,
)
from robocasa.environments.ithor_bathroom.tasks.composite.articulation_cycles import (
    _BathroomArticulationCycleMixin,
)
from robocasa.environments.ithor_bathroom.tasks.composite.navigation_routes import (
    _BathroomNavigationRouteMixin,
)
from robocasa.environments.ithor_bathroom.tasks.lift import (
    _BathroomNativeLiftTask,
)
from robocasa.environments.ithor_bedroom.tasks import (
    IThorBedroomApproachDeskAndCloseLaptop,
    IThorBedroomApproachDeskAndLiftBasket,
    IThorBedroomApproachDeskAndOpenLaptop,
    IThorBedroomApproachDeskAndPutBoxedFoodInBasket,
    IThorBedroomApproachDeskAndPutCanInBasket,
    IThorBedroomApproachDeskAndPutMugInBasket,
    IThorBedroomApproachStorageAndOpenBoxFlap,
    IThorBedroomApproachStorageAndCloseBoxFlap,
    IThorBedroomApproachStorageAndCloseDrawer,
    IThorBedroomApproachStorageAndCloseHamperLid,
    IThorBedroomApproachStorageAndCloseSafeDoor,
    IThorBedroomApproachStorageAndOpenDrawer,
    IThorBedroomApproachStorageAndOpenHamperLid,
    IThorBedroomApproachStorageAndOpenSafeDoor,
    IThorBedroomBedThenDesk,
    IThorBedroomBedThenStorage,
    IThorBedroomDeskThenBed,
    IThorBedroomDeskThenStorage,
    IThorBedroomOpenThenCloseBoxFlap,
    IThorBedroomOpenThenCloseDrawer,
    IThorBedroomOpenThenCloseHamperLid,
    IThorBedroomOpenThenCloseLaptop,
    IThorBedroomOpenThenCloseSafeDoor,
    IThorBedroomOpenThenCloseTopDrawer,
    IThorBedroomStorageThenBed,
    IThorBedroomStorageThenDesk,
)
from robocasa.environments.ithor_bedroom.tasks.safe_door import (
    _BedsideSafeMixin,
    _RaisedSafeMixin,
)
from robocasa.environments.ithor_bedroom.tasks.workspace_composites import (
    _BedroomWalkGateMixin,
)
from robocasa.environments.ithor_bedroom.tasks.routine_composites import (
    TASK_CLASSES as BEDROOM_ROUTINES,
)


BEDROOM_PRIMARY_PNP_TASKS = (
    IThorBedroomApproachDeskAndPutCanInBasket,
    IThorBedroomApproachDeskAndPutMugInBasket,
    IThorBedroomApproachDeskAndPutBoxedFoodInBasket,
)

BEDROOM_NAVIGATION_COMPOSITES = (
    IThorBedroomBedThenDesk,
    IThorBedroomDeskThenBed,
    IThorBedroomBedThenStorage,
    IThorBedroomStorageThenBed,
    IThorBedroomDeskThenStorage,
    IThorBedroomStorageThenDesk,
)

BATHROOM_NAVIGATION_ROUTES = tuple(
    task_class
    for task_class in BATHROOM_COMPOSITES
    if issubclass(task_class, _BathroomNavigationRouteMixin)
)
BATHROOM_APPROACH_OPERATION_TASKS = tuple(
    task_class
    for task_class in BATHROOM_COMPOSITES
    if issubclass(task_class, _ApproachThenOperationMixin)
)

BEDROOM_APPROACH_ARTICULATION_COMPOSITES = (
    IThorBedroomApproachDeskAndOpenLaptop,
    IThorBedroomApproachDeskAndCloseLaptop,
    IThorBedroomApproachStorageAndOpenSafeDoor,
    IThorBedroomApproachStorageAndCloseSafeDoor,
    IThorBedroomApproachStorageAndOpenHamperLid,
    IThorBedroomApproachStorageAndCloseHamperLid,
    IThorBedroomApproachStorageAndOpenBoxFlap,
    IThorBedroomApproachStorageAndCloseBoxFlap,
    IThorBedroomApproachStorageAndOpenDrawer,
    IThorBedroomApproachStorageAndCloseDrawer,
)

BEDROOM_ARTICULATION_CYCLES = (
    IThorBedroomOpenThenCloseDrawer,
    IThorBedroomOpenThenCloseTopDrawer,
    IThorBedroomOpenThenCloseSafeDoor,
    IThorBedroomOpenThenCloseHamperLid,
    IThorBedroomOpenThenCloseBoxFlap,
    IThorBedroomOpenThenCloseLaptop,
)

BEDROOM_WORKSPACE_COMPOSITES = (
    IThorBedroomApproachDeskAndLiftBasket,
    *BEDROOM_PRIMARY_PNP_TASKS,
)

BEDROOM_COMPOSITES = (
    *BEDROOM_NAVIGATION_COMPOSITES,
    *BEDROOM_APPROACH_ARTICULATION_COMPOSITES,
    *BEDROOM_ARTICULATION_CYCLES,
    *BEDROOM_WORKSPACE_COMPOSITES,
    *BEDROOM_ROUTINES,
)


def _exercise_walk_gate(task, gate_type):
    task._walk_stage = 0
    task._walk_approach_hold = 0
    task._walk_operation_hold = 0
    state = {"near": True, "initial": True, "operation": False}
    task._robot_near_work_surface = lambda: state["near"]
    task._walk_initial_state_intact = lambda: state["initial"]
    task._raw_operation_success = lambda: state["operation"]
    # Reaching the final state before navigation invalidates the first stage.
    state.update(initial=False, operation=True)
    for _ in range(task.APPROACH_HOLD_STEPS + 1):
        gate_type._advance_walk_progress(task)
    assert task._walk_stage == 0

    state.update(initial=True, operation=False)
    for _ in range(task.APPROACH_HOLD_STEPS - 1):
        gate_type._advance_walk_progress(task)
    assert task._walk_stage == 0
    gate_type._advance_walk_progress(task)
    assert task._walk_stage == 1

    state["operation"] = True
    for _ in range(task.HOLD_STEPS - 1):
        gate_type._advance_walk_progress(task)
    assert not gate_type._check_success(task)
    state["near"] = False
    gate_type._advance_walk_progress(task)
    assert task._walk_operation_hold == 0
    state["near"] = True
    for _ in range(task.HOLD_STEPS):
        gate_type._advance_walk_progress(task)
    assert gate_type._check_success(task)


def test_bedroom_walk_gate_enforces_order_and_both_holds():
    task = object.__new__(IThorBedroomApproachDeskAndPutCanInBasket)
    _exercise_walk_gate(task, _BedroomWalkGateMixin)


@pytest.mark.parametrize(
    "task_class",
    (
        IThorBathroomApproachSinkAndLiftSoapBottle,
        IThorBathroomNavigateToBathingAreaAndLiftScrubBrush,
    ),
)
def test_bathroom_approach_operation_enforces_order_and_both_holds(
    monkeypatch, task_class
):
    monkeypatch.setattr(
        _BathroomNativeLiftTask,
        "_post_action",
        lambda self, action: None,
    )
    task = object.__new__(task_class)
    task._approach_stage = 0
    task._approach_hold = 0
    task._composite_operation_hold = 0
    task._grasped_near_source = True
    task._success_hold = 4
    task._report_stage_complete = lambda *args, **kwargs: None
    state = {"near": True, "intact": False, "success": True}
    task._near_approach_fixture = lambda: state["near"]
    task._operation_initial_state_intact = lambda: state["intact"]
    task._raw_success = lambda: state["success"]

    # Completing the manipulation before navigation cannot skip stage one.
    for _ in range(task.APPROACH_HOLD_STEPS + 1):
        task._post_action(None)
    assert task._approach_stage == 0
    assert not task._check_success()

    state.update(intact=True, success=False)
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 1
    assert not task._grasped_near_source
    assert task._success_hold == 0

    state["success"] = True
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()


def test_approach_desk_and_lift_basket_requires_low_grasp_after_navigation(
    monkeypatch,
):
    monkeypatch.setattr(RoomLiftMixin, "_post_action", lambda self, action: None)
    task = object.__new__(IThorBedroomApproachDeskAndLiftBasket)
    task._walk_stage = 0
    task._walk_approach_hold = 0
    task._walk_lift_hold = 0
    task._lift_initial_z = 0.60
    task._grasped_at_source_height = False
    state = {"near": True, "intact": True, "height": 0.91, "grasped": True}
    task._robot_near_lift_surface = lambda: state["near"]
    task._basket_initial_state_intact = lambda: state["intact"]
    task._target_root_z = lambda: state["height"]
    task.g1_hand_grasping_object = lambda _name: state["grasped"]

    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._walk_stage == 1

    # A basket first grasped after it is already high cannot complete the lift.
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert not task._check_success()
    state["height"] = 0.65
    task._post_action(None)
    state["height"] = 0.90
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()


def test_all_composite_lift_stages_require_thirty_centimeters():
    for task_class in (
        IThorBedroomApproachDeskAndLiftBasket,
        IThorBathroomNavigateToBathingAreaAndLiftSoapBottle,
        IThorBathroomApproachSinkAndLiftSoapBottle,
    ):
        assert task_class.MIN_LIFT_DELTA == pytest.approx(0.30)


def test_bedroom_navigation_route_enforces_order_travel_and_both_holds(
    monkeypatch,
):
    monkeypatch.setattr(IThorBedroomEnv, "_post_action", lambda self, action: None)
    task = object.__new__(IThorBedroomBedThenDesk)
    task._route_stage = 0
    task._first_hold = 0
    task._second_hold = 0
    task._first_arrival_xy = None
    task.first_fixture = object()
    task.second_fixture = object()
    task._report_stage_complete = lambda *args, **kwargs: None
    state = {
        "first": False,
        "second": True,
        "xy": np.zeros(2),
    }
    task._near_fixture = lambda fixture: (
        state["first"] if fixture is task.first_fixture else state["second"]
    )
    task._robot_base_xy = lambda: state["xy"]
    task.scene_task_spec = lambda: {"minimum_stage_travel": 0.30}

    # Visiting the second target first cannot advance or finish the route.
    for _ in range(task.HOLD_STEPS + 1):
        task._post_action(None)
    assert task._route_stage == 0
    assert not task._check_success()

    state.update(first=True, second=False)
    for _ in range(task.FIRST_HOLD_STEPS):
        task._post_action(None)
    assert task._route_stage == 1

    # Being at the second target without leaving the first arrival is invalid.
    state.update(first=False, second=True)
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert not task._check_success()
    state["xy"] = np.array([0.30, 0.0])
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()


def test_bathroom_navigation_route_enforces_order_travel_and_both_holds(
    monkeypatch,
):
    monkeypatch.setattr(IThorBathroomEnv, "_post_action", lambda self, action: None)
    task = object.__new__(IThorBathroomNavigateToSinkThenToilet)
    task._route_stage = 0
    task._first_hold = 0
    task._second_hold = 0
    task._first_arrival_xy = None
    task.first_fixture = object()
    task.second_fixture = object()
    task._report_stage_complete = lambda *args, **kwargs: None
    state = {"first": False, "second": True, "xy": np.zeros(2)}
    task._near_fixture = lambda fixture: (
        state["first"] if fixture is task.first_fixture else state["second"]
    )
    task._robot_base_xy = lambda: state["xy"]
    task.scene_task_spec = lambda: {"minimum_stage_travel": 0.30}

    for _ in range(task.HOLD_STEPS + 1):
        _BathroomNavigationRouteMixin._post_action(task, None)
    assert task._route_stage == 0

    state.update(first=True, second=False)
    for _ in range(task.FIRST_HOLD_STEPS):
        _BathroomNavigationRouteMixin._post_action(task, None)
    assert task._route_stage == 1

    state.update(first=False, second=True)
    for _ in range(task.HOLD_STEPS):
        _BathroomNavigationRouteMixin._post_action(task, None)
    assert not task._check_success()
    state["xy"] = np.array([0.30, 0.0])
    for _ in range(task.HOLD_STEPS):
        _BathroomNavigationRouteMixin._post_action(task, None)
    assert task._check_success()


@pytest.mark.parametrize(
    "task_class",
    BATHROOM_NAVIGATION_ROUTES,
    ids=lambda task_class: task_class.__name__,
)
def test_bathroom_route_fixed_reset_clears_completed_route(task_class):
    task = object.__new__(task_class)
    task._route_stage = 1
    task._first_hold = task.FIRST_HOLD_STEPS
    task._second_hold = task.HOLD_STEPS
    task._first_arrival_xy = np.array([1.0, 2.0])
    assert task._check_success()

    task._reset_task_progress()

    assert task._route_stage == 0
    assert task._first_hold == 0
    assert task._second_hold == 0
    assert task._first_arrival_xy is None
    assert not task._check_success()


@pytest.mark.parametrize(
    "task_class",
    BATHROOM_APPROACH_OPERATION_TASKS,
    ids=lambda task_class: task_class.__name__,
)
def test_bathroom_approach_operation_fixed_reset_clears_completion(task_class):
    task = object.__new__(task_class)
    task._approach_stage = 1
    task._approach_hold = task.APPROACH_HOLD_STEPS
    task._composite_operation_hold = task.HOLD_STEPS
    assert task._check_success()

    task._reset_task_progress()

    assert task._approach_stage == 0
    assert task._approach_hold == 0
    assert task._composite_operation_hold == 0
    assert not task._check_success()


def test_bedroom_approach_articulation_enforces_navigation_before_operation(
    monkeypatch,
):
    monkeypatch.setattr(
        ExactSceneArticulation,
        "_post_action",
        lambda self, action: None,
    )
    task = object.__new__(IThorBedroomApproachDeskAndOpenLaptop)
    task._approach_stage = 0
    task._approach_hold = 0
    task._operation_hold = 0
    task._report_stage_complete = lambda *args, **kwargs: None
    state = {"near": True, "initial": False, "success": True}
    task._near_articulation = lambda: state["near"]
    task._initial_articulation_intact = lambda: state["initial"]
    task._raw_success = lambda: state["success"]

    # An already-operated joint cannot satisfy the approach stage.
    for _ in range(task.APPROACH_HOLD_STEPS + 1):
        task._post_action(None)
    assert task._approach_stage == 0
    assert not task._check_success()

    state.update(initial=True, success=False)
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 1

    state["success"] = True
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    state["near"] = False
    task._post_action(None)
    assert task._operation_hold == 0
    state["near"] = True
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()


def test_bedroom_310_approach_close_box_flap_allows_only_opening_drift(monkeypatch):
    monkeypatch.setattr(
        ExactSceneArticulation, "_post_action", lambda self, action: None
    )
    task = object.__new__(IThorBedroomApproachStorageAndCloseBoxFlap)
    task.scene_name = "FloorPlan310"
    task._approach_initial_fraction = 0.85
    task._approach_stage = task._approach_hold = task._operation_hold = 0
    task.HOLD_STEPS = 20
    task._report_stage_complete = lambda *args, **kwargs: None
    state = {"fraction": 0.85, "near": False}
    task.sim = None
    task.target_semantic_joint = type(
        "Joint", (), {"fraction": lambda self, sim: state["fraction"]}
    )()
    task._near_articulation = lambda: state["near"]
    task._raw_success = lambda: state["fraction"] <= task.SUCCESS_CLOSE_FRACTION

    # Closing before reaching the box must not count as navigation.
    state.update(fraction=0.10, near=True)
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 0
    assert not task._check_success()

    # Passive opening drift is allowed only once the robot reaches the box.
    state.update(fraction=1.0, near=False)
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 0
    state["near"] = True
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 1

    state["fraction"] = 0.10
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    state["near"] = False
    task._post_action(None)
    assert task._operation_hold == 0
    state["near"] = True
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()

    # Other scenes retain the original symmetric reset tolerance.
    task.scene_name = "FloorPlan329"
    state["fraction"] = 1.0
    assert not task._initial_articulation_intact()


def test_bedroom_articulation_cycle_requires_open_then_close(monkeypatch):
    monkeypatch.setattr(
        ExactSceneArticulation,
        "_post_action",
        lambda self, action: None,
    )
    task = object.__new__(IThorBedroomOpenThenCloseDrawer)
    task._cycle_stage = 0
    task._cycle_open_hold = 0
    task._cycle_close_hold = 0
    task._report_stage_complete = lambda *args, **kwargs: None
    state = {"open": False, "close": True}
    task._joint_matches = lambda behavior: state[behavior]

    # A closed joint at reset cannot skip the opening stage.
    for _ in range(task.HOLD_STEPS + 1):
        task._post_action(None)
    assert task._cycle_stage == 0
    assert not task._check_success()

    state.update(open=True, close=False)
    for _ in range(task.FIRST_HOLD_STEPS):
        task._post_action(None)
    assert task._cycle_stage == 1
    state.update(open=False, close=True)
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()


def test_bathroom_articulation_cycle_requires_both_states_in_order(monkeypatch):
    monkeypatch.setattr(
        ExactSceneArticulation,
        "_post_action",
        lambda self, action: None,
    )
    task = object.__new__(IThorBathroomOpenThenCloseToiletLid)
    task._cycle_stage = 0
    task._first_state_hold = 0
    task._second_state_hold = 0
    task._success_hold = 0
    task._report_stage_complete = lambda *args, **kwargs: None
    state = {"open": False, "close": True}
    task._joint_matches = lambda behavior: state[behavior]

    for _ in range(task.HOLD_STEPS + 1):
        task._post_action(None)
    assert task._cycle_stage == 0
    assert not task._check_success()

    state.update(open=True, close=False)
    for _ in range(task.FIRST_HOLD_STEPS):
        task._post_action(None)
    assert task._cycle_stage == 1
    state.update(open=False, close=True)
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()


def test_new_bedroom_composite_catalog_has_exactly_thirty_two_classes():
    assert len(BEDROOM_COMPOSITES) == 32
    assert len({task_class.__name__ for task_class in BEDROOM_COMPOSITES}) == 32
    assert all(
        task_class.TASK_FAMILY == "composite" for task_class in BEDROOM_COMPOSITES
    )

    all_scenes = (302, 306, 310, 323, 329)
    for task_class in (
        *BEDROOM_NAVIGATION_COMPOSITES,
        IThorBedroomApproachDeskAndOpenLaptop,
        IThorBedroomApproachDeskAndCloseLaptop,
        IThorBedroomApproachDeskAndLiftBasket,
        IThorBedroomOpenThenCloseDrawer,
        IThorBedroomOpenThenCloseLaptop,
    ):
        assert task_class.SUPPORTED_SCENE_IDS == all_scenes
    for task_class in BEDROOM_PRIMARY_PNP_TASKS:
        assert task_class.SUPPORTED_SCENE_IDS == all_scenes
        for scene_id in all_scenes:
            name = f"FloorPlan{scene_id}"
            walk = bedroom_specs.get_ithor_bedroom_task_spec(
                name, "secondary_workspace"
            )
            grasp = bedroom_specs.get_ithor_bedroom_task_spec(name, "grasp_workspace")
            expected_start = (
                bedroom_specs.get_ithor_bedroom_task_spec(name, "approach_desk")[
                    "start_pose"
                ]
                if scene_id == 323
                else walk["far_start_pose"]
            )
            assert task_class.SCENE_SPECS[name]["far_start_pose"] == expected_start
            assert (
                task_class.SCENE_SPECS[name]["fixture"],
                task_class.SCENE_SPECS[name]["region"],
            ) == (grasp["fixture"], grasp["region"])
    for scene_id in all_scenes:
        name = f"FloorPlan{scene_id}"
        walk = bedroom_specs.get_ithor_bedroom_task_spec(name, "secondary_workspace")
        expected_start = (
            bedroom_specs.get_ithor_bedroom_task_spec(name, "approach_desk")[
                "start_pose"
            ]
            if scene_id == 323
            else walk["far_start_pose"]
        )
        assert (
            IThorBedroomApproachDeskAndLiftBasket.SCENE_SPECS[name]["far_start_pose"]
            == expected_start
        )
    for task_class in (
        IThorBedroomApproachStorageAndOpenSafeDoor,
        IThorBedroomApproachStorageAndCloseSafeDoor,
    ):
        assert task_class.SUPPORTED_SCENE_IDS == (302, 323)
    for task_class in (
        IThorBedroomApproachStorageAndOpenHamperLid,
        IThorBedroomApproachStorageAndCloseHamperLid,
    ):
        assert task_class.SUPPORTED_SCENE_IDS == (302, 306, 310, 323, 329)
    assert IThorBedroomOpenThenCloseHamperLid.SUPPORTED_SCENE_IDS == (
        302,
        306,
        310,
        323,
        329,
    )
    for task_class in (
        IThorBedroomApproachStorageAndOpenDrawer,
        IThorBedroomApproachStorageAndCloseDrawer,
    ):
        assert task_class.SUPPORTED_SCENE_IDS == (310, 323)
    box_flap_start_tasks = {
        302: "approach_hamper_lid",
        306: "approach_bed",
        310: "approach_bed",
        323: "approach_hamper_lid",
        329: "approach_storage",
    }
    for task_class in (
        IThorBedroomApproachStorageAndOpenBoxFlap,
        IThorBedroomApproachStorageAndCloseBoxFlap,
    ):
        assert task_class.SUPPORTED_SCENE_IDS == all_scenes
        for scene_id in all_scenes:
            scene_name = f"FloorPlan{scene_id}"
            navigation = task_class.SCENE_SPECS[scene_name]["navigation"]
            start = bedroom_specs.get_ithor_bedroom_task_spec(
                scene_name, box_flap_start_tasks[scene_id]
            )
            assert (
                navigation["fixture"] == task_class.SCENE_SPECS[scene_name]["fixture"]
            )
            assert navigation["start_pose"]["xy"] == start["start_pose"]["xy"]
            if scene_id == 310:
                box_xy = np.asarray(
                    task_class.SCENE_SPECS[scene_name]["box_floor_xy"],
                    dtype=float,
                )
                start_xy = np.asarray(navigation["start_pose"]["xy"], dtype=float)
                expected_yaw = np.arctan2(
                    box_xy[1] - start_xy[1], box_xy[0] - start_xy[0]
                )
                assert navigation["start_pose"]["yaw"] == pytest.approx(expected_yaw)
            else:
                assert navigation["start_pose"]["yaw"] == start["start_pose"]["yaw"]
            assert (
                navigation["minimum_start_distance"] == start["minimum_start_distance"]
            )
            assert navigation["remove_fixtures"] == []
    assert IThorBedroomOpenThenCloseTopDrawer.SUPPORTED_SCENE_IDS == (
        302,
        306,
        323,
        329,
    )
    assert IThorBedroomOpenThenCloseSafeDoor.SUPPORTED_SCENE_IDS == (302, 323)
    assert IThorBedroomOpenThenCloseBoxFlap.SUPPORTED_SCENE_IDS == all_scenes

    for task_class in BEDROOM_NAVIGATION_COMPOSITES:
        assert tuple(task_class.SCENE_SPECS) == tuple(
            f"FloorPlan{scene_id}" for scene_id in all_scenes
        )
        for spec in task_class.SCENE_SPECS.values():
            assert spec["first_fixture"] != spec["second_fixture"]
            assert spec["success_distance"] == pytest.approx(0.35)
            assert spec["minimum_stage_travel"] == pytest.approx(0.30)


def test_bedroom_approach_close_safe_uses_the_close_safe_layout_only():
    close_spec = IThorBedroomApproachStorageAndCloseSafeDoor.SCENE_SPECS["FloorPlan302"]
    open_spec = IThorBedroomApproachStorageAndOpenSafeDoor.SCENE_SPECS["FloorPlan302"]

    assert IThorBedroomApproachStorageAndCloseSafeDoor.SPEC_TASK == "safe_door"
    assert issubclass(IThorBedroomApproachStorageAndCloseSafeDoor, _RaisedSafeMixin)
    assert not issubclass(
        IThorBedroomApproachStorageAndCloseSafeDoor, _BedsideSafeMixin
    )
    assert close_spec["safe_xy"] == pytest.approx([-0.20, 0.45])
    assert close_spec["safe_robot_xy"] == pytest.approx([-1.00, 0.45])
    assert close_spec["safe_z_offset"] == pytest.approx(0.425)
    assert "safe_yaw_delta" not in close_spec
    assert "safe_support_fixture" not in close_spec
    assert close_spec["fixture"] == open_spec["fixture"]
    assert close_spec["joint"] == open_spec["joint"]
    assert close_spec["navigation"]["start_pose"] == {
        "xy": [0.40, -0.80],
        "yaw": pytest.approx(2.018316),
    }
    assert close_spec["navigation"]["minimum_start_distance"] == pytest.approx(0.60)


def test_new_bathroom_composite_catalog_has_exactly_forty_classes():
    assert len(BATHROOM_COMPOSITES) == 40
    assert len({task_class.__name__ for task_class in BATHROOM_COMPOSITES}) == 40
    assert all(
        task_class.TASK_FAMILY == "composite" for task_class in BATHROOM_COMPOSITES
    )
    assert (
        sum(
            issubclass(task_class, _BathroomNavigationRouteMixin)
            for task_class in BATHROOM_COMPOSITES
        )
        == 12
    )
    assert (
        sum(
            issubclass(task_class, _ApproachThenOperationMixin)
            for task_class in BATHROOM_COMPOSITES
        )
        == 15
    )
    assert (
        sum(
            issubclass(task_class, _BathroomArticulationCycleMixin)
            for task_class in BATHROOM_COMPOSITES
        )
        == 8
    )

    expected_by_scene = {401: 24, 405: 26, 407: 30, 415: 22, 418: 30}
    assert {
        scene_id: sum(
            scene_id in task_class.SUPPORTED_SCENE_IDS
            for task_class in BATHROOM_COMPOSITES
        )
        for scene_id in expected_by_scene
    } == expected_by_scene


def _without_language_and_navigation(spec):
    return {
        key: value for key, value in spec.items() if key not in {"lang", "navigation"}
    }


def test_bathroom_approach_composites_reuse_atomic_operation_layouts():
    cases = (
        (
            "FloorPlan401",
            IThorBathroomLiftSoapBottle,
            IThorBathroomNavigateToBathingAreaAndLiftSoapBottle,
        ),
        (
            "FloorPlan407",
            IThorBathroomLiftSoapBottle,
            IThorBathroomApproachSinkAndLiftSoapBottle,
        ),
        (
            "FloorPlan407",
            IThorBathroomOpenCabinetDoor,
            IThorBathroomApproachSinkAndOpenCabinetDoor,
        ),
        (
            "FloorPlan405",
            IThorBathroomOpenCabinetDoor,
            IThorBathroomOpenThenCloseCabinetDoor,
        ),
        (
            "FloorPlan405",
            IThorBathroomCloseCabinetDoor,
            IThorBathroomCloseThenOpenCabinetDoor,
        ),
        (
            "FloorPlan407",
            IThorBathroomOpenCabinetDoor,
            IThorBathroomOpenThenCloseCabinetDoor,
        ),
        (
            "FloorPlan407",
            IThorBathroomCloseCabinetDoor,
            IThorBathroomCloseThenOpenCabinetDoor,
        ),
        (
            "FloorPlan401",
            IThorBathroomStandSoapBottleUpright,
            IThorBathroomNavigateToBathingAreaAndStandSoapBottleUpright,
        ),
        (
            "FloorPlan401",
            IThorBathroomToiletPaperRollIntoBin,
            IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin,
        ),
    )
    for scene, atomic, composite in cases:
        assert _without_language_and_navigation(
            composite.SCENE_SPECS[scene]
        ) == _without_language_and_navigation(atomic.SCENE_SPECS[scene])
        navigation = composite.SCENE_SPECS[scene].get("navigation")
        if navigation is not None:
            assert set(navigation) == {
                "fixture",
                "start_pose",
                "minimum_start_distance",
                "remove_fixtures",
                "lang",
            }


def test_sink_cabinet_composites_keep_atomic_target_and_safe_pose():
    cases = (
        (
            "FloorPlan405",
            IThorBathroomOpenCabinetDoor,
            IThorBathroomApproachSinkAndOpenCabinetDoor,
            {"xy": [-1.569, 1.503], "yaw": pytest.approx(-2.323868)},
        ),
        (
            "FloorPlan405",
            IThorBathroomCloseCabinetDoor,
            IThorBathroomApproachSinkAndCloseCabinetDoor,
            {"xy": [-1.569, 1.503], "yaw": pytest.approx(-2.323868)},
        ),
        (
            "FloorPlan407",
            IThorBathroomCloseCabinetDoor,
            IThorBathroomApproachSinkAndCloseCabinetDoor,
            {"xy": [0.007, 0.162], "yaw": pytest.approx(-2.295108)},
        ),
    )
    for scene, atomic, composite, expected_pose in cases:
        atomic_spec = atomic.SCENE_SPECS[scene]
        composite_spec = composite.SCENE_SPECS[scene]
        for field in atomic_spec:
            if field not in {"lang", "robot_pose"}:
                assert composite_spec[field] == atomic_spec[field]
        assert composite_spec["robot_pose"] == expected_pose


def test_bedroom_walk_tasks_share_the_calibrated_main_desk_workspace():
    scene_name = "FloorPlan329"
    fixture_id = "FloorPlan329:Desk_329_1_16"
    region_id = "FloorPlan329:Desk_329_1_16:region:receptacle:0"
    grasp = bedroom_specs.get_ithor_bedroom_task_spec(
        scene_name, "grasp_workspace", required=("fixture", "region")
    )
    secondary = bedroom_specs.get_ithor_bedroom_task_spec(
        scene_name,
        "secondary_workspace",
        required=("fixture", "region"),
    )
    assert (grasp["fixture"], grasp["region"]) == (fixture_id, region_id)
    assert (secondary["fixture"], secondary["region"]) == (fixture_id, region_id)
    assert grasp["operation_normal_distance"] == pytest.approx(0.40)
    assert secondary["operation_normal_distance"] == pytest.approx(0.40)
    for task_class in BEDROOM_WORKSPACE_COMPOSITES:
        spec = task_class.SCENE_SPECS[scene_name]
        assert (spec["fixture"], spec["region"]) == (fixture_id, region_id)


def test_bedroom_walk_assets_reuse_the_canonical_atomic_scales():
    assert IThorBedroomApproachDeskAndPutCanInBasket.SOURCE_SCALE == pytest.approx(
        bedroom_assets.CAN_SCALE
    )
    assert (
        IThorBedroomApproachDeskAndPutBoxedFoodInBasket.SOURCE_SCALE
        == pytest.approx(bedroom_assets.BOXED_FOOD_SCALE)
    )
    assert IThorBedroomApproachDeskAndPutMugInBasket.SOURCE_SCALE == pytest.approx(
        bedroom_assets.MUG_SCALE
    )
    assert IThorBedroomApproachDeskAndLiftBasket.TARGET_SCALE == pytest.approx(
        bedroom_assets.LIFT_BASKET_SCALE
    )

    # FloorPlan407's bathtub rim is curved collision geometry: the native
    # SoapBottle slides into the tub after settling. Keep this one calibrated
    # exception on the exact, stable countertop used by the existing local
    # manipulation task.
    soap_407 = IThorBathroomApproachSinkAndLiftSoapBottle.SCENE_SPECS["FloorPlan407"]
    assert soap_407["surface_fixture"] == ("FloorPlan407:Countertopmesh_ebfc63b6_32")
    assert soap_407["surface_region"] == (
        "FloorPlan407:Countertopmesh_ebfc63b6_32:region:support:5"
    )
    assert soap_407["source_xy"] == pytest.approx([-1.09584, -1.44])
