"""Unit contracts for the ordered iTHOR kitchen composite base."""

from types import SimpleNamespace

import numpy as np
import pytest

from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    IThorKitchenEnv,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.composite.base import (
    IThorKitchenAtomicTask,
    IThorKitchenMultiStageComposite,
)
from robocasa.environments.ithor_kitchen.tasks.composite.drawer_pnp import (
    IThorKitchenDrawerPnP,
)
from robocasa.environments.ithor_kitchen.tasks.composite.open_drawer_take_apple import (
    IThorKitchenOpenDrawerTakeApple,
)
from robocasa.environments.ithor_kitchen.tasks.composite.multi_stage_navigation import (
    IThorKitchenDishwasherThenFridge,
    IThorKitchenDishwasherThenOven,
    IThorKitchenFridgeThenDishwasher,
    IThorKitchenFridgeThenOven,
    IThorKitchenOvenThenDishwasher,
    IThorKitchenOvenThenFridge,
)
from robocasa.environments.ithor_composite.base import (
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_composite.navigation import (
    MultiStageNavigationMixin,
)
from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation


class _RouteParent:
    def __init__(self, *args, scene_ids=None, control_freq=2, **kwargs):
        self.requested_scene_ids = scene_ids
        self.control_freq = control_freq
        self.sim = object()

    def _post_action(self, _action):
        return 0.0, False, {}

    def _robot_base_xy(self):
        return self.base_xy


class _ThreeStageRoute(
    MultiStageNavigationMixin,
    IThorMultiStageCompositeMixin,
    _RouteParent,
):
    NAVIGATION_HOLD_SECONDS = 1.0
    STAGE_NAMES = ("oven", "fridge", "sink")
    SUPPORTED_SCENE_IDS = (1,)
    SCENE_SPECS = {
        "FloorPlan1": {
            "stages": ("oven_fixture", "fridge_fixture", "sink_fixture"),
            "robot_pose": {"pos": (1.0, 2.0, 0.0), "yaw": 0.5},
            "lang": "walk to the oven, fridge, and sink in order",
        }
    }


class _PlanarFixture:
    def __init__(self, center, half_size, yaw=0.0):
        self._center = np.asarray(center, dtype=float)
        self.half_size = np.asarray(half_size, dtype=float)
        self.local_half_size = self.half_size.copy()
        self.yaw = float(yaw)

    def world_center(self, _sim):
        return self._center.copy()


class _RepeatedGoalRoute(_ThreeStageRoute):
    STAGE_NAMES = ("oven", "fridge", "oven")


def test_drawer_pnp_success_uses_configured_close_fraction():
    task = object.__new__(IThorKitchenDrawerPnP)
    task.scene_name = "FloorPlan6"
    task.sim = object()
    task._drawer_opened = True
    task._apple_placed = True
    task._apple_inside_drawer = lambda: True
    fraction = {"value": 0.20}
    task.target_semantic_joint = SimpleNamespace(
        fraction=lambda _sim: fraction["value"]
    )

    assert task._task_success_condition()
    fraction["value"] = 0.21
    assert not task._task_success_condition()

    task._get_task_spec()["success_close_fraction"] = 0.25
    assert task._task_success_condition()
    task._drawer_opened = False
    assert not task._task_success_condition()
    task._drawer_opened = True
    task._apple_inside_drawer = lambda: False
    assert not task._task_success_condition()


def test_drawer_pnp_reports_ordered_placement_and_held_close(monkeypatch, capsys):
    task = object.__new__(IThorKitchenDrawerPnP)
    task.scene_name = "FloorPlan13"
    task.sim = object()
    task.control_freq = 2
    task.success_hold_seconds = 1.0
    task.SUCCESS_OPEN_FRACTION = 0.8
    state = {"fraction": 0.2, "inside": True}
    task.target_semantic_joint = SimpleNamespace(
        fraction=lambda _sim: state["fraction"]
    )
    task._apple_inside_drawer = lambda: state["inside"]
    monkeypatch.setattr(
        RoomOpenArticulation,
        "_post_action",
        lambda self, action: (self.reward(action), False, {}),
    )
    task._reset_task_progress()
    task._post_action(None)
    assert not task._apple_placed
    assert not task._check_success()

    state.update(fraction=0.8, inside=False)
    task._post_action(None)
    assert task._drawer_opened
    task._post_action(None)
    assert not task._apple_placed

    # Closing before observing placement must not skip the middle stage.
    state.update(fraction=0.2, inside=True)
    task._post_action(None)
    assert not task._apple_placed
    assert not task._check_success()

    state["fraction"] = 0.8
    task._post_action(None)
    assert task._apple_placed
    task._post_action(None)
    state["fraction"] = 0.2
    assert task._post_action(None)[0] == 0.0
    state["inside"] = False
    task._post_action(None)
    assert task._task_success_hold_steps == 0
    state["inside"] = True
    assert task._post_action(None)[0] == 0.0
    assert task._post_action(None)[0] == 1.0
    task._post_action(None)
    output = capsys.readouterr().out
    for index in (1, 2, 3):
        assert output.count(f"stage {index}/3") == 1
    assert "next stage: close_drawer" in output

    task._reset_task_progress()
    assert not task._drawer_opened
    assert not task._apple_placed
    assert not task._final_stage_reported
    assert not task._check_success()


def test_drawer_pnp_region_follows_full_site_pose_and_requires_release():
    task = object.__new__(IThorKitchenDrawerPnP)
    task.scene_name = "FloorPlan13"
    task._get_task_spec = lambda: {"show_target_region": True}
    angle = 0.6
    c, s = np.cos(angle), np.sin(angle)
    # Native Y is vertical and the drawer is also rotated in world XY.
    rotation = np.array([[c, 0, -s], [s, 0, c], [0, -1, 0]])
    half = np.array([0.14, 0.066, 0.118])
    task._drawer_receptacle_site_id = 0
    task.sim = SimpleNamespace(
        model=SimpleNamespace(site_size=np.array([half])),
        data=SimpleNamespace(
            site_xpos=np.array([[-3.19, 5.01, 0.80]]),
            site_xmat=np.array([rotation.ravel()]),
        ),
    )
    signs = np.array(np.meshgrid(*[[-1, 1]] * 3)).T.reshape(-1, 3)
    points = {"value": task.sim.data.site_xpos[0] + (signs * 0.03) @ rotation.T}
    task.object_bbox_points = lambda _name: points["value"]
    task.objects = {"apple": object()}
    task._drawer_contact_geoms = ["drawer_floor"]
    state = {"supported": True, "held": False}
    task.check_contact = lambda obj, geoms: state["supported"]
    task.g1_hand_object_contact = lambda side, name: state["held"]
    assert task._apple_inside_drawer()

    center, shown_half, yaw = task._drawer_receptacle_bounds()
    np.testing.assert_allclose(center, task.sim.data.site_xpos[0])
    np.testing.assert_allclose(shown_half, half[[0, 2, 1]])
    assert yaw == pytest.approx(angle)
    # Sliding must move both the predicate and overlay with the actual site.
    task.sim.data.site_xpos[0] += [0.0, -0.3, 0.0]
    assert not task._apple_inside_drawer()
    points["value"] += [0.0, -0.3, 0.0]
    assert task._apple_inside_drawer()
    np.testing.assert_allclose(
        task._drawer_receptacle_bounds()[0], task.sim.data.site_xpos[0]
    )
    state["held"] = True
    assert not task._apple_inside_drawer()
    state.update(held=False, supported=False)
    assert not task._apple_inside_drawer()
    state["supported"] = True
    points["value"] += rotation[:, 0] * half[0]
    assert not task._apple_inside_drawer()


def test_open_drawer_take_apple_has_no_insert_stage():
    task = object.__new__(IThorKitchenOpenDrawerTakeApple)
    task._reset_task_progress()
    task._drawer_opened = True
    task._update_placement_progress()
    assert not task._apple_placed


def test_open_drawer_take_apple_requires_no_hand_contact():
    task = object.__new__(IThorKitchenOpenDrawerTakeApple)
    task.scene_name = "FloorPlan25"
    task.sim = object()
    task._drawer_opened = True
    task._apple_on_table = lambda: True
    fraction = {"value": task.SUCCESS_OPEN_FRACTION}
    task.target_semantic_joint = SimpleNamespace(
        fraction=lambda _sim: fraction["value"]
    )
    contacts = {"left": False, "right": False}
    task.g1_hand_object_contact = lambda side, name: name == "apple" and contacts[side]
    task.object_released = lambda *_args, **_kwargs: pytest.fail(
        "success must not depend on hand distance"
    )

    assert task._task_success_condition()
    for side in contacts:
        contacts[side] = True
        assert not task._task_success_condition()
        contacts[side] = False
    assert task._task_success_condition()

    task._drawer_opened = False
    assert not task._task_success_condition()
    task._drawer_opened = True
    fraction["value"] -= 0.01
    assert not task._task_success_condition()
    fraction["value"] = task.SUCCESS_OPEN_FRACTION
    task._apple_on_table = lambda: False
    assert not task._task_success_condition()


def test_composite_base_is_not_a_runnable_registered_environment():
    name = IThorKitchenMultiStageComposite.__name__

    assert name not in REGISTERED_ITHOR_KITCHEN_ENVS
    assert name not in REGISTERED_ENVS


def test_atomic_task_mappings_are_normalized_and_validated():
    tasks = IThorKitchenMultiStageComposite._normalize_atomic_tasks(
        (
            {
                "name": "reach_counter",
                "success_condition": lambda _env: True,
            },
        )
    )

    assert len(tasks) == 1
    assert isinstance(tasks[0], IThorKitchenAtomicTask)
    assert tasks[0].name == "reach_counter"
    IThorKitchenMultiStageComposite._validate_atomic_tasks(tasks)

    with pytest.raises(ValueError, match="must declare ATOMIC_TASKS"):
        IThorKitchenMultiStageComposite._validate_atomic_tasks(())
    with pytest.raises(ValueError, match="must be unique"):
        IThorKitchenMultiStageComposite._validate_atomic_tasks(
            (
                IThorKitchenAtomicTask("duplicate", lambda _env: True),
                IThorKitchenAtomicTask("duplicate", lambda _env: True),
            )
        )


def test_stages_advance_in_order_and_print_completion_messages(monkeypatch, capsys):
    monkeypatch.setattr(
        IThorKitchenEnv,
        "_post_action",
        lambda _self, _action: (0.0, False, {}),
    )
    state = {"second_complete": False}
    task = object.__new__(IThorKitchenMultiStageComposite)
    task._atomic_tasks = (
        IThorKitchenAtomicTask("reach_counter", lambda _env: True, hold_steps=2),
        IThorKitchenAtomicTask(
            "place_mug",
            lambda _env: state["second_complete"],
            hold_steps=1,
        ),
    )
    task._stage_index = 0
    task._stage_hold_steps = 0
    task.PRINT_STAGE_MESSAGES = True

    reward, _, info = task._post_action(None)
    assert reward == 0.0
    assert info["active_atomic_task"] == "reach_counter"
    assert capsys.readouterr().out == ""

    reward, _, info = task._post_action(None)
    output = capsys.readouterr().out
    assert reward == 0.0
    assert task.stage_index == 1
    assert info["completed_atomic_tasks"] == ["reach_counter"]
    assert "stage 1/2 complete: reach_counter" in output
    assert "Next stage: place_mug" in output

    state["second_complete"] = True
    reward, _, info = task._post_action(None)
    output = capsys.readouterr().out
    assert reward == 1.0
    assert info["task_success"] is True
    assert info["success"] is True
    assert info["active_atomic_task"] is None
    assert info["completed_atomic_tasks"] == ["reach_counter", "place_mug"]
    assert "stage 2/2 complete: place_mug" in output
    assert "Task succeeded" in output


def test_reset_progress_clears_stage_state():
    task = object.__new__(IThorKitchenMultiStageComposite)
    task._stage_index = 2
    task._stage_hold_steps = 7

    task._reset_task_progress()

    assert task.stage_index == 0
    assert task._stage_hold_steps == 0


def test_navigation_reset_task_progress_clears_route_state():
    task = object.__new__(_ThreeStageRoute)
    task._stage_index = 3
    task._stage_hold_steps = 7
    task._previous_stage_reached_xy = np.array([1.0, 2.0])

    task._reset_task_progress()

    assert task.stage_index == 0
    assert task._stage_hold_steps == 0
    assert task._previous_stage_reached_xy is None


def test_reset_clears_completed_success_before_parent_scene_setup(monkeypatch):
    parent_states = []

    def parent_reset(task):
        parent_states.append((task.stage_index, task._check_success()))

    monkeypatch.setattr(
        IThorKitchenEnv,
        "_reset_internal",
        parent_reset,
    )
    state = {"complete": False}
    task = object.__new__(IThorKitchenMultiStageComposite)
    task._atomic_tasks = (
        IThorKitchenAtomicTask(
            "reach_oven",
            lambda _env: state["complete"],
            hold_steps=1,
        ),
    )
    task._stage_index = len(task._atomic_tasks)
    task._stage_hold_steps = 9
    task.PRINT_STAGE_MESSAGES = False

    assert task._check_success()

    task._reset_internal()

    assert parent_states == [(0, False)]
    assert task.stage_index == 0
    assert task._stage_hold_steps == 0
    assert not task._check_success()


def test_multistage_navigation_supports_three_ordered_goals(capsys):
    task = _ThreeStageRoute()
    fixtures = [object(), object(), object()]
    task.route_fixtures = fixtures
    task.base_xy = np.zeros(2)
    task._previous_stage_reached_xy = None
    near = {fixture: False for fixture in fixtures}
    task._fixture_is_near = lambda fixture: near[fixture]

    assert len(task._atomic_tasks) == 3
    assert [stage.name for stage in task._atomic_tasks] == [
        "navigate_to_oven",
        "navigate_to_fridge",
        "navigate_to_sink",
    ]
    required_steps = task._required_stage_hold_steps(task.current_atomic_task)
    assert required_steps == 2

    # A later goal cannot unlock before the active stage is complete.
    near[fixtures[1]] = True
    for _ in range(required_steps + 1):
        task._post_action(None)
    assert task.stage_index == 0

    near[fixtures[1]] = False
    near[fixtures[0]] = True
    for _ in range(required_steps):
        task._post_action(None)
    assert task.stage_index == 1

    # Consecutive stages must be separated by real base motion.
    near[fixtures[0]] = False
    near[fixtures[1]] = True
    task._post_action(None)
    assert task.stage_index == 1
    task.base_xy = np.array([task.MIN_STAGE_TRAVEL + 0.1, 0.0])
    for _ in range(required_steps):
        task._post_action(None)
    assert task.stage_index == 2

    near[fixtures[1]] = False
    near[fixtures[2]] = True
    task.base_xy = np.array([2 * (task.MIN_STAGE_TRAVEL + 0.1), 0.0])
    result = None
    for _ in range(required_steps):
        result = task._post_action(None)

    reward, _, info = result
    assert reward == 1.0
    assert info["task_success"] is True
    assert task._check_success()
    output = capsys.readouterr().out
    assert output.count("stage 1/3 complete") == 1
    assert output.count("stage 2/3 complete") == 1
    assert output.count("stage 3/3 complete") == 1


def test_multistage_navigation_requires_one_second_in_range():
    task = _ThreeStageRoute(control_freq=4)
    fixtures = [object(), object(), object()]
    task.route_fixtures = fixtures
    task.base_xy = np.zeros(2)
    task._previous_stage_reached_xy = None
    task._fixture_is_near = lambda fixture: fixture is fixtures[0]

    for _ in range(3):
        task._post_action(None)

    assert task.stage_index == 0
    assert task._stage_hold_steps == 3

    task._post_action(None)

    assert task.stage_index == 1
    assert task._stage_hold_steps == 0


def test_multistage_navigation_uses_only_root_xy_and_fixture_footprint():
    task = _ThreeStageRoute()
    fixture = _PlanarFixture(
        center=[0.0, 0.0, 100.0],
        half_size=[0.6, 0.2, 50.0],
        yaw=np.pi / 2,
    )
    task.robot_fixture_collision_distance = lambda *_args, **_kwargs: pytest.fail(
        "navigation success must not inspect robot collision geometry"
    )

    task.base_xy = np.array([0.0, 0.94])
    assert task._fixture_is_near(fixture)

    task.base_xy = np.array([0.0, 0.96])
    assert not task._fixture_is_near(fixture)


def test_multistage_navigation_allows_revisiting_a_named_goal():
    task = _RepeatedGoalRoute()

    assert [stage.name for stage in task._atomic_tasks] == [
        "navigate_to_oven_1",
        "navigate_to_fridge",
        "navigate_to_oven_3",
    ]


def test_multistage_navigation_reads_exact_robot_pose_from_scene_spec():
    task = _ThreeStageRoute()
    task.scene_name = "FloorPlan1"

    pos, yaw = task._configured_robot_pose()

    assert pos.tolist() == [1.0, 2.0, 0.0]
    assert yaw == pytest.approx(0.5)
    assert task.INITIAL_POSE_JITTER is False


def test_kitchen_multistage_navigation_uses_then_names_only():
    task_classes = (
        IThorKitchenOvenThenFridge,
        IThorKitchenFridgeThenOven,
        IThorKitchenFridgeThenDishwasher,
        IThorKitchenDishwasherThenFridge,
        IThorKitchenOvenThenDishwasher,
        IThorKitchenDishwasherThenOven,
    )
    old_names = (
        "IThorKitchenNavigateOvenToFridge",
        "IThorKitchenNavigateFridgeToOven",
        "IThorKitchenNavigateFridgeToDishwasher",
        "IThorKitchenNavigateDishwasherToFridge",
        "IThorKitchenNavigateOvenToDishwasher",
        "IThorKitchenNavigateDishwasherToOven",
    )

    assert all(task_class.__name__ in REGISTERED_ENVS for task_class in task_classes)
    assert all(
        task_class.__name__ in REGISTERED_ITHOR_KITCHEN_ENVS
        for task_class in task_classes
    )
    assert all(old_name not in REGISTERED_ENVS for old_name in old_names)
    assert all(old_name not in REGISTERED_ITHOR_KITCHEN_ENVS for old_name in old_names)
