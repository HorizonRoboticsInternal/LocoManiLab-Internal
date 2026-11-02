"""Focused checks for the scene-329 carry-and-fill basket task."""

from copy import deepcopy
from unittest.mock import Mock, patch

import numpy as np
import pytest

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_bedroom.tasks import (
    IThorBedroomLiftBasketFromFloorThenPlaceOnDeskAndPutBoxedFoodInBasket as Task,
)
from robocasa.environments.ithor_bedroom.tasks.assets import (
    BOXED_FOOD_MODEL,
    BOXED_FOOD_SCALE,
    LIFT_BASKET_MODEL,
    LIFT_BASKET_SCALE,
    asset_xml_path,
)
from robocasa.environments.ithor_bedroom.tasks.lift_basket import (
    IThorBedroomLiftBasketFromFloor,
)
from robocasa.environments.ithor_bedroom.tasks.object_in_receptacle import (
    IThorBedroomBoxedFoodIntoBasket,
)
from robocasa.environments.ithor_task_catalog import (
    BEDROOM_PACKAGE,
    discover_task_catalog,
    validate_task_catalog,
)


def make_task(control_freq=20):
    env = object.__new__(Task)
    env.control_freq = control_freq
    env._configure_success_hold_duration()
    env.EMIT_STAGE_MESSAGES = False
    env._reset_task_progress()
    env._lift_initial_z = 0.0
    env._target_root_z = Mock(return_value=0.0)
    env.g1_hand_grasping_object = Mock(
        side_effect=AssertionError("grasp must not gate success")
    )
    env.object_released = Mock(
        side_effect=AssertionError("release must not gate success")
    )
    env._basket_is_on_desk = Mock(return_value=False)
    env._food_waits_on_desk = Mock(return_value=True)
    env._raw_place_success = Mock(return_value=False)
    return env


def advance(env, steps):
    for _ in range(steps):
        env._advance_composite_progress()


def complete_lift(env):
    env._target_root_z.return_value = 0.31
    advance(env, env.INTERMEDIATE_HOLD_STEPS)
    assert env._completed_stage == 1


def complete_desk_placement(env):
    env._basket_is_on_desk.return_value = True
    advance(env, env.INTERMEDIATE_HOLD_STEPS)
    assert env._completed_stage == 2


def test_registration_scope_and_atomic_assets():
    entries = discover_task_catalog([BEDROOM_PACKAGE])
    entry = next(entry for entry in entries if entry.cls is Task)
    assert entry.scene_ids == (329,)
    assert entry.family == "composite"
    # Existing composite matrix drift is unrelated to this addition.
    assert not [
        i for i in validate_task_catalog(entries) if ": task matrix changed" not in i
    ]
    with pytest.raises(ValueError, match="uncalibrated scene ids"):
        Task(scene_ids=[323])
    assert Task.SOURCE_MJCF_PATH == asset_xml_path(BOXED_FOOD_MODEL)
    assert Task.SOURCE_SCALE == BOXED_FOOD_SCALE
    assert Task.RECEPTACLE_MJCF_PATH == asset_xml_path(LIFT_BASKET_MODEL)
    assert Task.RECEPTACLE_SCALE == LIFT_BASKET_SCALE
    atomic_before = deepcopy(IThorBedroomBoxedFoodIntoBasket.SCENE_SPECS)
    env = make_task()
    env.scene_name = "FloorPlan329"
    spec = env.scene_task_spec()
    atomic_layout = deepcopy(
        IThorBedroomLiftBasketFromFloor.SCENE_SPECS["FloorPlan329"]
    )
    for key in (
        "stool_model",
        "stool_scale",
        "stool_top_half_xy",
        "source_horizontal_distance",
    ):
        assert spec[key] == atomic_layout[key]
    # Preserve the atomic relative layout under a clockwise 90-degree rotation.
    rotation = np.array([[0.0, 1.0], [-1.0, 0.0]])
    atomic_robot_xy = np.array(atomic_layout["robot_pose"]["xy"])
    robot_xy = np.array([-0.2765, 0.2504])
    for key in ("basket_xy", "stool_xy"):
        np.testing.assert_allclose(
            spec[key], robot_xy + rotation @ (atomic_layout[key] - atomic_robot_xy)
        )
    for key in ("robot_pose", "staging_robot_pose"):
        np.testing.assert_allclose(
            spec[key]["xy"],
            robot_xy + rotation @ (atomic_layout[key]["xy"] - atomic_robot_xy),
        )
        yaw_delta = spec[key]["yaw"] - atomic_layout[key]["yaw"]
        assert np.cos(yaw_delta) == pytest.approx(0.0, abs=1e-9)
        assert np.sin(yaw_delta) == pytest.approx(-1.0)
    assert spec["stool_yaw"] == pytest.approx(atomic_layout["stool_yaw"] - np.pi / 2)
    np.testing.assert_allclose(
        spec["basket_quat"], [np.sqrt(0.5), 0.0, 0.0, -np.sqrt(0.5)]
    )
    # The desk is along +Y; a right turn from the basket heading faces it.
    desk_yaw = spec["robot_pose"]["yaw"] - np.pi / 2
    np.testing.assert_allclose([np.cos(desk_yaw), np.sin(desk_yaw)], [0, 1], atol=1e-9)
    spec["robot_pose"]["xy"][0] = 99
    assert env.scene_task_spec()["robot_pose"]["xy"] == [-0.2765, 0.2504]
    assert IThorBedroomLiftBasketFromFloor.SCENE_SPECS["FloorPlan329"] == atomic_layout
    assert IThorBedroomBoxedFoodIntoBasket.SCENE_SPECS == atomic_before
    assert IThorBedroomBoxedFoodIntoBasket not in Task.__mro__


def test_initial_state_and_filled_basket_cannot_skip_stages():
    env = make_task()
    advance(env, 30)
    assert env._completed_stage == 0
    assert not env._check_success()
    # A filled basket on the desk still lacks the empty-basket placement stage.
    env._basket_is_on_desk.return_value = True
    env._food_waits_on_desk.return_value = False
    env._raw_place_success.return_value = True
    env._target_root_z.return_value = 0.8
    advance(env, 30)
    assert env._completed_stage == 1
    assert not env._check_success()


def test_lift_requires_continuous_height_without_grasp_detection():
    env = make_task()
    env._target_root_z.return_value = 0.29
    advance(env, 30)
    assert env._completed_stage == 0
    env._target_root_z.return_value = 0.31
    advance(env, env.INTERMEDIATE_HOLD_STEPS - 1)
    assert env._completed_stage == 0
    env._target_root_z.return_value = 0.29
    advance(env, 1)
    assert env._bedroom_lift_hold == 0
    env._target_root_z.return_value = 0.31
    advance(env, env.INTERMEDIATE_HOLD_STEPS - 1)
    assert env._completed_stage == 0
    advance(env, 1)
    assert env._completed_stage == 1
    env.g1_hand_grasping_object.assert_not_called()
    env.object_released.assert_not_called()


def test_filling_before_desk_placement_cannot_skip_stage():
    env = make_task()
    complete_lift(env)
    env._basket_is_on_desk.return_value = True
    env._food_waits_on_desk.return_value = False
    env._raw_place_success.return_value = True
    advance(env, 30)
    assert env._completed_stage == 1
    assert not env._check_success()
    env._food_waits_on_desk.return_value = True
    advance(env, env.INTERMEDIATE_HOLD_STEPS - 1)
    env._basket_is_on_desk.return_value = False
    advance(env, 1)
    assert env._desk_hold == 0
    complete_desk_placement(env)


@pytest.mark.parametrize("control_freq", [10, 20, 30])
def test_ordered_completion_holds_one_second_and_resets(control_freq):
    env = make_task(control_freq)
    complete_lift(env)
    complete_desk_placement(env)
    env._raw_place_success.return_value = True
    with patch.object(IThorBedroomEnv, "_post_action", return_value=None):
        for _ in range(control_freq - 1):
            env._post_action(None)
        assert not env._check_success()
        assert not env._check_success()  # Queries cannot accumulate hold time.
        env._raw_place_success.return_value = False
        env._post_action(None)
        assert env._success_hold == 0
        env._raw_place_success.return_value = True
        for _ in range(control_freq):
            env._post_action(None)
        assert env._check_success()
        env._raw_place_success.return_value = False
        env._post_action(None)
        assert not env._check_success()


def test_food_contact_containment_and_basket_desk_are_required_without_release():
    env = make_task()
    del env._raw_place_success  # Exercise the real composed predicate.
    env.objects = {env.SOURCE_NAME: object(), env.RECEPTACLE_NAME: object()}
    env._basket_is_on_desk.return_value = True
    env.check_contact = Mock(return_value=True)
    env._source_bbox_inside_receptacle = Mock(return_value=True)
    assert env._raw_place_success()
    for predicate in (
        env._basket_is_on_desk,
        env.check_contact,
        env._source_bbox_inside_receptacle,
    ):
        predicate.return_value = False
        assert not env._raw_place_success()
        predicate.return_value = True


def configure_support_geometry(env):
    env.objects = {
        env.SOURCE_NAME: Mock(root_body="food_body"),
        env.RECEPTACLE_NAME: Mock(root_body="basket_body"),
    }
    env.work_surface_region = object()
    env.work_surface_fixture = object()
    env.sim = Mock()
    env.sim.data.get_body_xmat.return_value = np.eye(3)
    env.object_on_region = Mock(return_value=True)
    env._object_fixture_contact_is_shallow = Mock(return_value=True)
    env._object_has_illegal_penetration = Mock(return_value=False)


def test_basket_support_geometry_stays_required_without_release():
    env = make_task()
    del env._basket_is_on_desk
    configure_support_geometry(env)
    assert env._basket_is_on_desk()
    for predicate, invalid in (
        (env.object_on_region, False),
        (env._object_fixture_contact_is_shallow, False),
        (env._object_has_illegal_penetration, True),
    ):
        predicate.return_value = invalid
        assert not env._basket_is_on_desk()
        predicate.return_value = not invalid
    env.sim.data.get_body_xmat.return_value = np.diag([1, -1, -1])
    assert not env._basket_is_on_desk()
    env.object_released.assert_not_called()


@pytest.mark.parametrize("grasped", [False, True])
def test_real_predicates_complete_in_order_independent_of_hand_state(grasped):
    env = make_task()
    del env._basket_is_on_desk, env._food_waits_on_desk, env._raw_place_success
    configure_support_geometry(env)
    env.g1_hand_grasping_object = Mock(return_value=grasped)
    env.object_released = Mock(return_value=not grasped)
    env.check_contact = Mock(return_value=True)
    env._source_bbox_inside_receptacle = Mock(return_value=False)
    env.object_on_region.side_effect = lambda name, *a, **kw: name == env.SOURCE_NAME
    complete_lift(env)
    env.object_on_region.side_effect = None
    advance(env, env.INTERMEDIATE_HOLD_STEPS)
    assert env._completed_stage == 2
    advance(env, env.HOLD_STEPS)
    assert not env._check_success()  # Food outside the basket cannot finish.
    env._source_bbox_inside_receptacle.return_value = True
    advance(env, env.HOLD_STEPS - 1)
    assert not env._check_success()
    advance(env, 1)
    assert env._check_success()
    env.g1_hand_grasping_object.assert_not_called()
    env.object_released.assert_not_called()
