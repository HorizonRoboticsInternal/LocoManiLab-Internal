"""Contracts for Kitchen navigation-to-table manipulation composites."""

import pytest
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen import (
    REGISTERED_ITHOR_KITCHEN_ENVS,
    IThorKitchenApproachTableAndLiftApple,
    IThorKitchenApproachTableAndLiftBowl,
    IThorKitchenApproachTableAndLiftSaltShaker,
    IThorKitchenApproachTableAndPutAppleOnPlate,
    IThorKitchenApproachTableAndPutBreadOnPlate,
    IThorKitchenApproachTableAndPutCheeseOnPlate,
    IThorKitchenApproachTableAndPutMugOnPlate,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.apple_on_plate import (
    IThorKitchenAppleOnPlate,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.lift_objects import (
    KitchenLiftObjectMixin,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


COMPOSITES = (
    IThorKitchenApproachTableAndLiftApple,
    IThorKitchenApproachTableAndLiftSaltShaker,
    IThorKitchenApproachTableAndLiftBowl,
    IThorKitchenApproachTableAndPutAppleOnPlate,
    IThorKitchenApproachTableAndPutBreadOnPlate,
    IThorKitchenApproachTableAndPutCheeseOnPlate,
    IThorKitchenApproachTableAndPutMugOnPlate,
)
LIFT_COMPOSITES = COMPOSITES[:3]
PLATE_COMPOSITES = COMPOSITES[3:]
SCENE_IDS = (6, 7, 13, 24, 25)


def test_seven_approach_table_composites_are_public_and_registered():
    assert len(COMPOSITES) == 7
    assert len({task.__name__ for task in COMPOSITES}) == 7
    for task in COMPOSITES:
        assert task.SUPPORTED_SCENE_IDS == SCENE_IDS
        assert task.TASK_FAMILY == "composite"
        assert REGISTERED_ITHOR_KITCHEN_ENVS[task.__name__] is task
        assert REGISTERED_ENVS[task.__name__] is task
    assert "IThorKitchenApproachTableAndLiftPlate" not in REGISTERED_ITHOR_KITCHEN_ENVS
    assert "IThorKitchenApproachTableAndLiftPlate" not in REGISTERED_ENVS
    assert not any(
        name.startswith("_KitchenApproach") for name in REGISTERED_ITHOR_KITCHEN_ENVS
    )


@pytest.mark.parametrize("scene_id", SCENE_IDS)
def test_composites_have_explicit_yaml_initial_poses_and_layouts(scene_id):
    scene_name = f"FloorPlan{scene_id}"
    lift_plate = get_ithor_kitchen_task_spec(scene_name, "lift_plate")
    lift_salt = get_ithor_kitchen_task_spec(scene_name, "lift_salt_shaker")
    on_plate = get_ithor_kitchen_task_spec(scene_name, "apple_on_plate")
    navigation = get_ithor_kitchen_task_spec(scene_name, "approach_table")
    expected_pose = (
        {"pos": [-1.25, 2.45, 0.0], "yaw": -1.4403531274106511}
        if scene_name == "FloorPlan25"
        else navigation["robot_pose"]
    )

    for task in COMPOSITES:
        spec = get_ithor_kitchen_task_spec(scene_name, task.SPEC_TASK)
        assert task.SCENE_SPECS[scene_name] == spec
        assert spec["robot_pose"] == expected_pose
        assert spec["robot_pose"] != on_plate["robot_pose"]
        assert spec["target_fixture"] == navigation["target_fixture"]
        assert spec["support_body"] == navigation["support_body"]
        assert spec["approach_distance"] == pytest.approx(0.35)
        assert spec["minimum_start_distance"] >= 0.50
        assert "lang" not in spec
        assert task.TASK_LANG

    for task in (
        IThorKitchenApproachTableAndLiftApple,
        IThorKitchenApproachTableAndLiftBowl,
    ):
        spec = task.SCENE_SPECS[scene_name]
        for field in (
            "hold_seconds",
            "lift_height",
            "support_region",
            "object_pos",
            "object_quat",
        ):
            assert spec[field] == lift_plate[field]
        assert spec["lift_height"] == pytest.approx(0.3)
    salt_spec = IThorKitchenApproachTableAndLiftSaltShaker.SCENE_SPECS[scene_name]
    for field in (
        "hold_seconds",
        "lift_height",
        "support_region",
        "object_pos",
        "object_quat",
    ):
        assert salt_spec[field] == lift_salt[field]
    for task in PLATE_COMPOSITES:
        spec = task.SCENE_SPECS[scene_name]
        assert spec["hold_seconds"] == on_plate["hold_seconds"]
        assert spec["plate_pos"] == on_plate["plate_pos"]
        assert spec["source_pos"] == on_plate["apple_pos"]


def test_scene25_approach_table_composites_face_the_table_operation_area():
    expected_pose = {
        "pos": [-1.25, 2.45, 0.0],
        "yaw": -1.4403531274106511,
    }
    for task in COMPOSITES:
        spec = task.SCENE_SPECS["FloorPlan25"]
        assert spec["robot_pose"] == expected_pose
        assert spec["approach_distance"] == pytest.approx(0.35)
        assert spec["minimum_start_distance"] == pytest.approx(0.50)


def test_scene13_approach_table_tasks_share_shifted_robot_pose():
    expected_pose = {"pos": [-1.95, 3.20, 0.0], "yaw": -2.2}
    navigation = get_ithor_kitchen_task_spec("FloorPlan13", "approach_table")
    assert navigation["robot_pose"] == expected_pose
    assert navigation["minimum_start_distance"] == pytest.approx(0.75)

    for task in COMPOSITES:
        spec = get_ithor_kitchen_task_spec("FloorPlan13", task.SPEC_TASK)
        assert spec["robot_pose"] == expected_pose
        assert spec["minimum_start_distance"] == pytest.approx(0.75)


def test_lift_composite_requires_navigation_then_atomic_height_hold(monkeypatch):
    monkeypatch.setattr(
        KitchenLiftObjectMixin,
        "_post_action",
        lambda self, action: None,
    )
    task = object.__new__(IThorKitchenApproachTableAndLiftBowl)
    task._approach_stage = 0
    task._approach_hold = 0
    task._operation_hold = 0
    task._operation_hold_steps = 3
    task._lift_initial_z = 0.90
    task.min_lift_delta = 0.30
    task._report_stage_complete = lambda *args, **kwargs: None
    state = {"near": False, "height": 1.25}
    task._robot_near_table = lambda: state["near"]
    task._target_root_z = lambda: state["height"]

    # Lifting first cannot satisfy or advance the navigation stage.
    for _ in range(task.APPROACH_HOLD_STEPS + 1):
        task._post_action(None)
    assert task._approach_stage == 0
    assert not task._check_success()

    state.update(near=True, height=0.90)
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 1

    # After navigation, only the atomic 0.30 m height predicate is required.
    state["height"] = 1.20
    for _ in range(task._operation_hold_steps):
        task._post_action(None)
    assert task._check_success()


def test_plate_composite_requires_navigation_then_atomic_placement_hold(monkeypatch):
    monkeypatch.setattr(
        IThorKitchenAppleOnPlate,
        "_post_action",
        lambda self, action: None,
    )
    task = object.__new__(IThorKitchenApproachTableAndPutAppleOnPlate)
    task._approach_stage = 0
    task._approach_hold = 0
    task._operation_hold = 0
    task._operation_hold_steps = 3
    task._report_stage_complete = lambda *args, **kwargs: None
    state = {"near": False, "success": True}
    task._robot_near_table = lambda: state["near"]
    task._raw_operation_success = lambda: state["success"]

    for _ in range(task.APPROACH_HOLD_STEPS + 1):
        task._post_action(None)
    assert task._approach_stage == 0

    state.update(near=True, success=False)
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert task._approach_stage == 1

    # Leaving the table after navigation does not invalidate the atomic result.
    state["success"] = True
    state["near"] = False
    for _ in range(task._operation_hold_steps):
        task._post_action(None)
    assert task._check_success()


def test_plate_composite_delegates_to_atomic_placement_predicate(monkeypatch):
    state = {"success": False}
    monkeypatch.setattr(
        IThorKitchenAppleOnPlate,
        "_raw_success",
        lambda self: state["success"],
    )
    task = object.__new__(IThorKitchenApproachTableAndPutAppleOnPlate)

    assert not task._raw_operation_success()
    state["success"] = True
    assert task._raw_operation_success()


@pytest.mark.parametrize("task_class", COMPOSITES)
def test_fixed_reset_hook_clears_composite_and_atomic_success_state(task_class):
    task = object.__new__(task_class)
    task._approach_stage = 1
    task._approach_hold = 5
    task._operation_hold = 20
    task._lift_hold_steps = 20
    task._success_hold_steps = 20

    task._reset_task_progress()

    assert task._approach_stage == 0
    assert task._approach_hold == 0
    assert task._operation_hold == 0
    assert task._lift_hold_steps == 0
    assert task._success_hold_steps == 0


def test_lift_and_plate_variants_keep_the_requested_object_identities():
    assert [task.TARGET_NAME for task in LIFT_COMPOSITES] == [
        "apple",
        "salt_shaker",
        "bowl",
    ]
    bowl = object.__new__(IThorKitchenApproachTableAndLiftBowl)
    assert bowl._get_obj_cfgs() == [dict(name="bowl", obj_groups="bowl")]
    assert bowl.TARGET_LANG == "bowl"
    assert bowl.TASK_LANG == "walk to the table, then lift the bowl"
    assert bowl.COMPOSITE_ORDERING == ("approach_table", "lift_bowl")
    assert [task.SOURCE_NAME for task in PLATE_COMPOSITES] == [
        "apple",
        "bread",
        "cheese",
        "mug",
    ]
    assert all(task.INITIAL_ROBOT_JITTER_RADIUS == 0.0 for task in COMPOSITES)
    assert all(task.INITIAL_OBJECT_JITTER_RADIUS == 0.0 for task in COMPOSITES)
