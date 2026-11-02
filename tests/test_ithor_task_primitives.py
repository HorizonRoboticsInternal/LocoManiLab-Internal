"""Unit contracts for shared scene-calibrated iTHOR task primitives."""

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import robocasa.environments.ithor_bedroom.tasks.articulation as bedroom_articulation
import robocasa.environments.ithor_bedroom.tasks.assets as bedroom_assets
import robocasa.environments.ithor_atomic.task_assets as task_assets
import robocasa.environments.ithor_task_catalog as task_catalog
import validate_ithor_tasks as task_validator
from robocasa.environments.ithor_atomic.articulation import (
    ExactSceneArticulation,
    RoomCloseLaptop,
    RoomOpenArticulation,
)
from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_atomic.navigation import (
    ExactSceneNavigation,
    RoomNavMixin,
)
from robocasa.environments.ithor_atomic.pick_place import (
    SemanticSurfaceTask,
    SupportContactPlacementMixin,
    SurfaceObjectInReceptacle,
    resolve_task_support_region,
)
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_base import (
    IThorBathroomEnv,
    IThorBedroomEnv,
    IThorKitchenEnv,
    IThorLivingRoomEnv,
    IThorRoomEnv,
    control_steps_for_duration,
)
from robocasa.environments.ithor_bathroom.tasks import (
    IThorBathroomApproachToiletAndCloseToiletLid,
    IThorBathroomApproachToiletAndOpenToiletLid,
    IThorBathroomCloseCabinetDoor,
    IThorBathroomCloseLowerDrawer,
    IThorBathroomCloseShowerDoor,
    IThorBathroomCloseThenOpenToiletLid,
    IThorBathroomCloseToiletLid,
    IThorBathroomLiftCandleFromFloor,
    IThorBathroomLiftScrubBrush,
    IThorBathroomLiftSoapBottle,
    IThorBathroomLiftSprayBottle,
    IThorBathroomLiftToiletPaperRoll,
    IThorBathroomNavigateToBathingAreaAndCloseShowerDoor,
    IThorBathroomNavigateToBathingAreaAndOpenShowerDoor,
    IThorBathroomNavigateToSinkThenToilet,
    IThorBathroomOpenCabinetDoor,
    IThorBathroomOpenLowerDrawer,
    IThorBathroomOpenShowerDoor,
    IThorBathroomOpenToiletLid,
    IThorBathroomPutSpongeInBathtub,
    IThorBathroomSoapToLowerShelf,
    IThorBathroomStandSoapBottleUpright,
    IThorBathroomToiletPaperRollIntoBin,
)
from robocasa.environments.ithor_bathroom.tasks.pick_place import (
    _BathroomNativePickPlaceTask,
)
from robocasa.environments.ithor_bathroom.tasks.object_manipulation import (
    _BathroomWorkspaceMixin,
)
from robocasa.environments.ithor_bedroom.tasks import (
    IThorBedroomApproachBed,
    IThorBedroomBoxedFoodIntoBasket,
    IThorBedroomCanIntoBasket,
    IThorBedroomCloseBoxFlap,
    IThorBedroomCloseDrawer,
    IThorBedroomCloseHamperLid,
    IThorBedroomCloseSafeDoor,
    IThorBedroomCloseTopDrawer,
    IThorBedroomLiftBasket,
    IThorBedroomLiftBasketFromFloor,
    IThorBedroomMugIntoBasket,
    IThorBedroomOpenBoxFlap,
    IThorBedroomOpenDrawer,
    IThorBedroomOpenHamperLid,
    IThorBedroomOpenLaptop,
    IThorBedroomOpenSafeDoor,
    IThorBedroomOpenTopDrawer,
)
from robocasa.environments.ithor_bedroom.tasks.surface_placement import (
    _square_operation_pose,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.objects_into_box import (
    IThorKitchenAppleIntoBox,
    IThorKitchenCupIntoBox,
    IThorKitchenStatueIntoBox,
    SingleObjectIntoBoxMixin,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.coffee_drink_to_trash import (
    IThorKitchenCoffeeDrinkToTrash,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.cup_stack import (
    IThorKitchenCupStack,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.cup_to_sink import (
    IThorKitchenCupToSink,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.pack_produce_into_box import (
    IThorKitchenPackProduceIntoBox,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.pan_table_to_stove import (
    IThorKitchenPanTableToStove,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.bowl_to_sink import (
    IThorKitchenBowlToSink,
)
from robocasa.environments.ithor_kitchen.tasks.composite.multi_stage_navigation import (
    IThorKitchenOvenThenFridge,
)
from robocasa.environments.ithor_living_room.tasks.atomic.pour_block_from_basket import (
    IThorLivingRoomPourBlockFromBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.pour_ice_from_cup import (
    IThorLivingRoomPourIceFromCup,
)
from robocasa.environments.holodeck.tasks.take_tissue_box_from_open_drawer import (
    HolodeckTakeTissueBoxFromOpenDrawer,
)
from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckApproachDeskThenOpenLaptop,
    HolodeckOpenDrawerThenTakeTissueBox,
    HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
)
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckAtomicTask,
    HolodeckMultiHopComposite,
)
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.holodeck.tasks.place_cup_on_placemat import (
    HolodeckPlaceCupOnPlacemat,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion
import robocasa.models.scenes.ithor_bathroom_task_specs as bathroom_specs
import robocasa.models.scenes.ithor_bedroom_task_specs as bedroom_specs


def _region(kind="support", *, half_z=0.02, parent_body=None):
    return SemanticRegion(
        stable_id=f"test:region:{kind}",
        fixture_id="test:fixture",
        kind=kind,
        parent_body=parent_body,
        local_center=np.array([1.0, 2.0, 3.0]),
        half_size=np.array([0.4, 0.3, half_z]),
        _world_center=np.array([4.0, 5.0, 6.0]),
        _world_yaw=0.25,
    )


@pytest.mark.parametrize(
    ("control_freq", "expected_steps"),
    ((10, 10), (20, 20), (30, 30)),
)
def test_success_duration_is_converted_to_control_steps(
    control_freq,
    expected_steps,
):
    assert control_steps_for_duration(1.0, control_freq) == expected_steps


@pytest.mark.parametrize(
    ("duration_seconds", "control_freq"),
    ((0.0, 20), (-1.0, 20), (np.nan, 20), (1.0, 0), (1.0, np.inf)),
)
def test_success_duration_rejects_invalid_timing(duration_seconds, control_freq):
    with pytest.raises(ValueError, match="finite and positive"):
        control_steps_for_duration(duration_seconds, control_freq)


@pytest.mark.parametrize("env_class", (IThorBedroomEnv, IThorBathroomEnv))
@pytest.mark.parametrize("control_freq", (10, 20, 30))
def test_bedroom_and_bathroom_configure_one_second_final_hold(
    env_class,
    control_freq,
):
    env = object.__new__(env_class)
    env.control_freq = control_freq
    env.HOLD_STEPS = 5
    env.SUCCESS_HOLD_STEPS = 5

    env._configure_success_hold_duration()

    assert env.SUCCESS_HOLD_SECONDS == pytest.approx(1.0)
    assert env.HOLD_STEPS == control_freq
    assert env.SUCCESS_HOLD_STEPS == control_freq


@pytest.mark.parametrize("env_class", (IThorKitchenEnv, IThorLivingRoomEnv))
def test_room_hold_policy_does_not_override_kitchen_or_living_room(env_class):
    env = object.__new__(env_class)
    env.control_freq = 20
    env.HOLD_STEPS = 7
    env.SUCCESS_HOLD_STEPS = 9

    env._configure_success_hold_duration()

    assert env.SUCCESS_HOLD_SECONDS is None
    assert env.HOLD_STEPS == 7
    assert env.SUCCESS_HOLD_STEPS == 9


def test_bathroom_route_only_extends_the_final_navigation_hold():
    task = object.__new__(IThorBathroomNavigateToSinkThenToilet)
    task.control_freq = 20
    task._configure_success_hold_duration()

    assert task.FIRST_HOLD_STEPS == 5
    assert task.HOLD_STEPS == 20


def test_kitchen_route_only_extends_the_final_navigation_hold():
    task = object.__new__(IThorKitchenOvenThenFridge)
    task.control_freq = 20
    task._configure_success_hold_duration()
    stages = task._get_atomic_tasks()

    assert [stage.hold_steps for stage in stages] == [5, None]
    assert [task._required_stage_hold_steps(stage) for stage in stages] == [5, 20]


def test_one_second_final_hold_must_be_uninterrupted(monkeypatch):
    monkeypatch.setattr(
        IThorRoomEnv,
        "_post_action",
        lambda _self, action: action,
    )
    task = object.__new__(IThorBedroomApproachBed)
    task.control_freq = 20
    task._configure_success_hold_duration()
    task._in_range_steps = 0
    near = {"value": True}
    task._robot_is_near_target = lambda: near["value"]

    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()

    near["value"] = False
    task._post_action(None)
    assert task._in_range_steps == 0

    near["value"] = True
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()


def test_cup_stack_success_only_checks_the_two_cups():
    task = object.__new__(IThorKitchenCupStack)
    task.objects = {"cup_1": object(), "cup_2": object()}
    task._cup_bounds = lambda name: (
        (np.array([0.0, 0.0, 0.9]), np.array([0.1, 0.1, 0.1]))
        if name == "cup_1"
        else (np.array([0.0, 0.0, 0.98]), np.array([0.1, 0.1, 0.1]))
    )
    task._axis_up = lambda _name: 1.0
    contact = {"value": True}
    task.check_contact = lambda *_args: contact["value"]
    task.object_released = lambda *_args, **_kwargs: pytest.fail(
        "cup-stack success must not query hand release distance"
    )
    task.object_on_region = lambda *_args, **_kwargs: pytest.fail(
        "cup-stack success must not query table contact or bounds"
    )
    task._object_region_bounds = lambda *_args, **_kwargs: pytest.fail(
        "cup-stack success must not query table geometry"
    )

    assert task._cups_are_stacked()
    contact["value"] = False
    assert not task._cups_are_stacked()


def test_true_support_is_returned_as_an_independent_region():
    source = _region()
    resolved = resolve_task_support_region(source)

    assert resolved is not source
    assert resolved.kind == "support"
    np.testing.assert_allclose(resolved.local_center, source.local_center)
    np.testing.assert_allclose(resolved._world_center, source._world_center)
    resolved.local_center[2] = -1.0
    resolved.half_size[0] = -1.0
    assert source.local_center[2] == pytest.approx(3.0)
    assert source.half_size[0] == pytest.approx(0.4)


def test_legacy_shared_room_contracts_remain_available_outside_scope():
    assert RoomNavMixin.SHOW_TARGET_REGION is False
    assert RoomNavMixin.TARGET_REGION_SITE_NAME == "navigation_target_region"
    assert callable(RoomLiftMixin._setup_lift_support)
    assert callable(RoomCloseLaptop._clear_work_surface)


def test_floor_footprint_does_not_treat_floor_lamps_as_floor():
    env = object.__new__(IThorRoomEnv)

    def fixture(category, center=(0.0, 0.0), half=(1.0, 1.0)):
        return SimpleNamespace(
            category=category,
            center=np.asarray((*center, 0.0), dtype=float),
            half_size=np.asarray((*half, 0.01), dtype=float),
            yaw=0.0,
        )

    env.semantic_scene = SimpleNamespace(fixtures={"lamp": fixture("floor_lamp")})
    assert not env.xy_is_on_floor((0.0, 0.0))

    env.semantic_scene.fixtures["floor"] = fixture("floor_c6579873")
    assert env.xy_is_on_floor((0.0, 0.0))
    assert env.robot_footprint_is_on_floor((0.0, 0.0), radius=0.25)
    assert not env.robot_footprint_is_on_floor((0.85, 0.0), radius=0.25)


def test_legacy_room_nav_still_builds_an_opt_in_target_site():
    class _Parent:
        def _setup_task_fixtures(self):
            return None

        def register_fixture_ref(self, _name, _query):
            return SimpleNamespace(center=np.array([1.0, 2.0, 0.0]))

    class _LegacyNavigation(RoomNavMixin, _Parent):
        pass

    task = _LegacyNavigation()
    task.show_target_region = True
    task.target_xy = np.array([3.0, 4.0])
    task.target_tolerance = np.array([0.2, 0.3])
    task.mujoco_arena = SimpleNamespace(worldbody=[])

    task._setup_task_fixtures()

    assert len(task.mujoco_arena.worldbody) == 1
    site = task.mujoco_arena.worldbody[0]
    assert site.get("name") == RoomNavMixin.TARGET_REGION_SITE_NAME
    assert site.get("type") == "box"
    np.testing.assert_allclose(
        np.fromstring(site.get("pos"), sep=" "), [3.0, 4.0, 0.012]
    )
    np.testing.assert_allclose(
        np.fromstring(site.get("size"), sep=" "), [0.2, 0.3, 0.01]
    )


def test_pan_table_to_stove_rejects_a_pan_still_grasped_by_g1():
    task = object.__new__(IThorKitchenPanTableToStove)
    task._pan_base_in_target_region = lambda: True
    grasping = {"value": True}
    task.g1_hand_grasping_object = lambda name: (name == "pan" and grasping["value"])

    assert not task._raw_success()
    grasping["value"] = False
    assert task._raw_success()


def test_statue_and_pour_block_do_not_define_success_region_visualizers():
    assert not hasattr(SingleObjectIntoBoxMixin, "_visualize_box_success_region")
    assert not hasattr(SingleObjectIntoBoxMixin, "BOX_SUCCESS_SITE")
    assert not hasattr(IThorLivingRoomPourBlockFromBasket, "SUCCESS_REGION_SITE_NAME")


def test_pour_ice_requires_the_ice_outside_for_two_seconds(monkeypatch):
    task = object.__new__(IThorLivingRoomPourIceFromCup)
    task._success_hold_steps = 0
    task._hold_steps_required = 40
    state = {"ice_out": False}
    task._ice_fully_outside_cup = lambda: state["ice_out"]
    monkeypatch.setattr(
        IThorLivingRoomPourIceFromCup.__mro__[1],
        "_post_action",
        lambda self, action: None,
        raising=False,
    )

    state["ice_out"] = True
    for _ in range(task._hold_steps_required - 1):
        task._post_action(None)
    assert not task._check_success()
    state["ice_out"] = False
    task._post_action(None)
    assert task._success_hold_steps == 0

    state["ice_out"] = True
    for _ in range(task._hold_steps_required):
        task._post_action(None)
    assert task._check_success()


def test_shallow_receptacle_resolves_to_its_lower_face():
    source = _region("receptacle", half_z=0.05, parent_body="desk")
    resolved = resolve_task_support_region(source)

    assert resolved.kind == "support"
    assert resolved.stable_id == source.stable_id
    np.testing.assert_allclose(resolved.local_center, source.local_center)
    np.testing.assert_allclose(resolved._world_center, source._world_center)
    np.testing.assert_allclose(resolved._world_offset, [0.0, 0.0, -0.05])
    np.testing.assert_allclose(resolved.world_pose()[0], [4.0, 5.0, 5.95])
    assert source.local_center[2] == pytest.approx(3.0)
    assert source._world_center[2] == pytest.approx(6.0)
    np.testing.assert_allclose(source._world_offset, np.zeros(3))


def test_shallow_receptacle_lower_face_stays_world_vertical_for_rotated_parent():
    source = _region("receptacle", half_z=0.05, parent_body="desk")
    resolved = resolve_task_support_region(source)
    body_rotation = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ]
    )
    sim = SimpleNamespace(
        model=SimpleNamespace(body_name2id=lambda name: 0),
        data=SimpleNamespace(
            body_xpos=np.array([[10.0, 20.0, 30.0]]),
            body_xmat=np.array([body_rotation.reshape(-1)]),
        ),
    )

    source_center, _ = source.world_pose(sim)
    resolved_center, _ = resolved.world_pose(sim)
    np.testing.assert_allclose(resolved_center[:2], source_center[:2])
    assert resolved_center[2] == pytest.approx(source_center[2] - 0.05)


def test_deep_receptacle_is_not_misclassified_as_a_support():
    with pytest.raises(ValueError, match="too deep"):
        resolve_task_support_region(_region("receptacle", half_z=0.081))


def _bbox_corners(low, high):
    return np.asarray(
        [
            [x, y, z]
            for x in (low[0], high[0])
            for y in (low[1], high[1])
            for z in (low[2], high[2])
        ],
        dtype=float,
    )


def _receptacle_contract_task(source_points, task_class=SurfaceObjectInReceptacle):
    task = object.__new__(task_class)
    source = SimpleNamespace(root_body="source")
    receptacle = SimpleNamespace(root_body="receptacle")
    task.objects = {task.SOURCE_NAME: source, task.RECEPTACLE_NAME: receptacle}
    task.work_surface_region = object()
    task.work_surface_fixture = object()
    positions = {"source": np.zeros(3), "receptacle": np.zeros(3)}
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            get_body_xpos=lambda name: positions[name],
            get_body_xmat=lambda _name: np.eye(3).reshape(-1),
        )
    )
    receptacle_points = _bbox_corners((-1.0, -1.0, 0.0), (1.0, 1.0, 1.0))
    task.object_bbox_points = lambda name: (
        source_points if name == task.SOURCE_NAME else receptacle_points
    )
    task._object_fixture_contact_is_shallow = lambda *args, **kwargs: True
    task._object_has_illegal_penetration = lambda *args, **kwargs: False
    return task


def test_receptacle_contract_rejects_bbox_overhang_and_vertical_false_positives():
    inside = _bbox_corners((-0.2, -0.2, 0.1), (0.2, 0.2, 0.4))
    task = _receptacle_contract_task(inside)
    assert task._source_bbox_inside_receptacle()

    task = _receptacle_contract_task(_bbox_corners((0.75, -0.1, 0.1), (0.9, 0.1, 0.4)))
    assert not task._source_bbox_inside_receptacle()

    task = _receptacle_contract_task(_bbox_corners((-0.1, -0.1, -0.1), (0.1, 0.1, 0.2)))
    assert not task._source_bbox_inside_receptacle()


def test_center_containment_allows_tall_sources_above_the_rim():
    tall_source = _bbox_corners((-0.95, -0.2, 0.1), (0.95, 0.2, 2.0))
    task = _receptacle_contract_task(tall_source)
    task.RECEPTACLE_XY_CONTAINMENT = "center"
    assert task._source_bbox_inside_receptacle()

    outside_center = _bbox_corners((0.85, -0.1, 0.1), (1.05, 0.1, 2.0))
    task = _receptacle_contract_task(outside_center)
    task.RECEPTACLE_XY_CONTAINMENT = "center"
    assert not task._source_bbox_inside_receptacle()


@pytest.mark.parametrize(
    "task_class",
    (
        IThorBedroomMugIntoBasket,
        IThorBedroomBoxedFoodIntoBasket,
        IThorBedroomCanIntoBasket,
    ),
)
def test_simple_bedroom_receptacle_tasks_match_bathroom_contract(task_class):
    inside = _bbox_corners((-0.95, -0.2, 0.1), (0.95, 0.2, 2.0))
    task = _receptacle_contract_task(inside, task_class)
    state = {"contact": True, "released": True}
    task.check_contact = lambda _source, _receptacle: state["contact"]
    task.object_released = lambda _name, th: state["released"]
    task._receptacle_is_upright_on_support = lambda: False

    assert task._raw_place_success()
    state["contact"] = False
    assert not task._raw_place_success()
    state["contact"] = True
    state["released"] = False
    assert not task._raw_place_success()

    outside = _receptacle_contract_task(
        _bbox_corners((0.85, -0.1, 0.1), (1.05, 0.1, 0.4)),
        task_class,
    )
    outside.check_contact = lambda _source, _receptacle: True
    outside.object_released = lambda _name, th: True
    assert not outside._raw_place_success()


def test_bathroom_soap_tasks_share_one_canonical_model_and_scale():
    assert IThorBathroomSoapToLowerShelf.SOURCE_GROUP == (
        task_assets.asset_xml_path(task_assets.SOAP_MODEL)
    )
    assert IThorBathroomSoapToLowerShelf.SOURCE_SCALE == pytest.approx(
        task_assets.SOAP_SCALE
    )


def test_bathroom_lower_shelf_accepts_physical_edge_but_keeps_level_guards():
    task = object.__new__(IThorBathroomSoapToLowerShelf)
    task.target_region = SimpleNamespace(half_size=np.array([0.374525, 0.121549, 0.02]))
    task.scene_task_spec = lambda: {"target_center_xy_tolerance": [0.022, 0.03]}
    state = {
        # The center is on the collidable shelf extension even though one end
        # of the soap lies beyond the smaller exported semantic patch.
        "local": np.array([[0.34, -0.04], [0.43, -0.04], [0.34, 0.04], [0.43, 0.04]]),
        "bottom": 0.779815,
        "contact": True,
        "released": True,
    }
    task._object_region_bounds = lambda _name, _region: (
        np.array([0.0, 0.0, 0.779815]),
        state["local"],
        state["bottom"],
        state["bottom"] + 0.04,
    )
    task._object_contacts_shelf = lambda: state["contact"]
    task.object_released = lambda _name, th: state["released"]

    assert not np.all(np.abs(state["local"]) <= task.target_region.half_size[:2] + 1e-4)
    assert task._object_is_on_target_shelf_and_released()

    state["local"][:, 0] += 0.012
    assert not task._object_is_on_target_shelf_and_released()
    state["local"][:, 0] -= 0.012
    state["bottom"] += task.TARGET_LEVEL_Z_TOLERANCE + 0.001
    assert not task._object_is_on_target_shelf_and_released()
    state["bottom"] = 0.779815
    state["contact"] = False
    assert not task._object_is_on_target_shelf_and_released()
    state.update(contact=True, released=False)
    assert not task._object_is_on_target_shelf_and_released()


def test_bathroom_pnp_and_reorientation_tasks_are_floorplan401_only():
    for task_class in (
        IThorBathroomSoapToLowerShelf,
        IThorBathroomStandSoapBottleUpright,
    ):
        assert task_class.SUPPORTED_SCENE_IDS == (401,)
        assert tuple(task_class.SCENE_SPECS) == ("FloorPlan401",)

    bin_task = IThorBathroomToiletPaperRollIntoBin
    assert bin_task.SUPPORTED_SCENE_IDS == (401, 405, 407)
    assert tuple(bin_task.SCENE_SPECS) == (
        "FloorPlan401",
        "FloorPlan405",
        "FloorPlan407",
    )

    shelf_spec = IThorBathroomSoapToLowerShelf.SCENE_SPECS["FloorPlan401"]
    assert shelf_spec["target_region"].endswith(":region:support:5")
    assert shelf_spec["target_center_xy_tolerance"] == pytest.approx((0.022, 0.03))


def test_canonical_handheld_assets_are_over_seven_and_at_most_eight_centimeters():
    assert not hasattr(task_assets, "INITIAL_SOURCE_HORIZONTAL_DISTANCE")
    assert not hasattr(task_assets, "INITIAL_SOURCE_HORIZONTAL_DISTANCE_TOLERANCE")
    assert task_assets.TARGET_GRASP_SPAN == pytest.approx(0.075)
    assert set(task_assets.NATIVE_GRASP_SPANS) == {
        task_assets.CAN_MODEL,
        task_assets.CANDLE_MODEL,
        task_assets.BOXED_FOOD_MODEL,
        task_assets.MUG_MODEL,
        task_assets.SOAP_MODEL,
        task_assets.SPONGE_MODEL,
    }
    for model, native_span in task_assets.NATIVE_GRASP_SPANS.items():
        scale = task_assets.CANONICAL_ASSET_SCALES[model]
        calibrated_span = native_span * scale
        assert task_assets.MIN_GRASP_SPAN < calibrated_span
        assert calibrated_span <= task_assets.MAX_GRASP_SPAN


def test_bedroom_source_distances_are_optional_scene_local_calibrations():
    grasp_workspace = bedroom_specs.get_ithor_bedroom_task_spec(
        "FloorPlan329", "grasp_workspace"
    )
    assert "source_horizontal_distance" not in grasp_workspace
    assert grasp_workspace["operation_normal_distance"] == pytest.approx(0.40)
    assert bedroom_specs.get_ithor_bedroom_task_spec("FloorPlan329", "lift_basket")[
        "source_horizontal_distance"
    ] == pytest.approx(0.40)


def test_bedroom_pnp_operation_pose_is_square_to_the_object_row():
    source_xy = np.array([0.12, 0.64])
    receptacle_xy = np.array([-0.12, 0.64])

    base_xy, yaw = _square_operation_pose(
        source_xy,
        receptacle_xy,
        outward_direction=[0.0, -1.0],
        normal_distance=0.40,
    )

    assert base_xy == pytest.approx([0.0, 0.24])
    assert yaw == pytest.approx(np.pi / 2.0)
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    right = np.array([forward[1], -forward[0]])
    assert np.dot(source_xy - base_xy, forward) == pytest.approx(0.40)
    assert np.dot(receptacle_xy - base_xy, forward) == pytest.approx(0.40)
    assert np.dot(source_xy - base_xy, right) > 0.0
    assert np.dot(receptacle_xy - base_xy, right) < 0.0


def test_all_repeated_small_assets_use_the_canonical_model_and_scale():
    sponge_task = object.__new__(IThorBathroomPutSpongeInBathtub)
    sponge_task.rim_region = object()
    assert sponge_task._get_obj_cfgs()[0]["obj_groups"] == (
        task_assets.asset_xml_path(task_assets.SPONGE_MODEL)
    )
    assert IThorBathroomPutSpongeInBathtub.SOURCE_SCALE == pytest.approx(
        task_assets.SPONGE_SCALE
    )
    assert IThorBathroomSoapToLowerShelf.SOURCE_GROUP == task_assets.asset_xml_path(
        task_assets.SOAP_MODEL
    )
    assert IThorBathroomSoapToLowerShelf.SOURCE_SCALE == pytest.approx(
        task_assets.SOAP_SCALE
    )


@pytest.mark.parametrize(
    ("task_class", "native_grasp_span", "expected_scale"),
    (
        (IThorBedroomCanIntoBasket, 0.0690627916, bedroom_assets.CAN_SCALE),
        (IThorBedroomMugIntoBasket, 0.0785635518, bedroom_assets.MUG_SCALE),
        (
            IThorBedroomBoxedFoodIntoBasket,
            0.0723618216,
            bedroom_assets.BOXED_FOOD_SCALE,
        ),
    ),
)
def test_bedroom_handheld_assets_use_seven_point_five_centimeter_grasp_span(
    task_class,
    native_grasp_span,
    expected_scale,
):
    assert task_class.SOURCE_SCALE == pytest.approx(expected_scale)
    assert native_grasp_span * task_class.SOURCE_SCALE == pytest.approx(
        bedroom_assets.TARGET_GRASP_SPAN
    )


@pytest.mark.parametrize(
    "task_class",
    (IThorBedroomCanIntoBasket,),
)
def test_bedroom_receptacle_tasks_use_right_hand_near_edge_layout(task_class):
    assert task_class.SOURCE_ON_OPERATOR_RIGHT is True
    assert task_class.MIN_SOURCE_RIGHT_SEPARATION == pytest.approx(0.05)


def test_mug_and_boxed_food_use_close_right_hand_layouts():
    assert IThorBedroomMugIntoBasket.RECEPTACLE_BOTTOM_TOLERANCE == pytest.approx(0.035)
    assert IThorBedroomMugIntoBasket.STAND_LATERAL_OFFSETS[-2:] == (0.40, -0.40)
    assert IThorBedroomMugIntoBasket.SOURCE_ON_OPERATOR_RIGHT is True
    assert IThorBedroomMugIntoBasket.MIN_SOURCE_RIGHT_SEPARATION == pytest.approx(0.05)
    assert IThorBedroomBoxedFoodIntoBasket.SOURCE_ON_OPERATOR_RIGHT is True


def test_bedroom_mug_accepts_settled_basket_depth_but_rejects_deeper_overlap():
    settled = _receptacle_contract_task(
        _bbox_corners((-0.1, -0.1, -0.034), (0.1, 0.1, 0.2)),
        IThorBedroomMugIntoBasket,
    )
    assert settled._source_bbox_inside_receptacle()

    too_deep = _receptacle_contract_task(
        _bbox_corners((-0.1, -0.1, -0.036), (0.1, 0.1, 0.2)),
        IThorBedroomMugIntoBasket,
    )
    assert not too_deep._source_bbox_inside_receptacle()


def test_bedroom_containers_are_sized_for_calibrated_handheld_objects():
    for task_class in (
        IThorBedroomCanIntoBasket,
        IThorBedroomMugIntoBasket,
        IThorBedroomBoxedFoodIntoBasket,
    ):
        assert task_class.RECEPTACLE_SCALE == pytest.approx(bedroom_assets.BASKET_SCALE)


def test_new_bathroom_object_tasks_have_complete_exact_scene_specs():
    expected_sources = {
        IThorBathroomLiftSoapBottle: (
            "Soap_Bottle_1_10",
            "Soap_Bottle_5_7",
            "Soap_Bottle_7_13",
            "Soap_Bottle_15_0",
            "Soap_Bottle_18_10",
        ),
        IThorBathroomLiftSprayBottle: (
            "Spray_Bottle_1_11",
            "Spray_Bottle_5_10",
            "Spray_Bottle_1_6",
            "Spray_Bottle_3_8",
            "Spray_Bottle_6_1",
        ),
        IThorBathroomLiftToiletPaperRoll: (
            "Toilet_Paper_5",
            "Toilet_Paper_2",
            "Toilet_Paper_8",
            "Toilet_Paper_6",
            "Toilet_Paper_9",
        ),
        IThorBathroomLiftScrubBrush: (
            "Scrub_Brush_1_0",
            "Scrub_Brush_1_5",
            "Scrub_Brush_1_12",
            "Scrub_Brush_1_1",
            "Scrub_Brush_1_5",
        ),
    }
    scenes = (401, 405, 407, 415, 418)
    for task_class, source_names in expected_sources.items():
        assert task_class.SUPPORTED_SCENE_IDS == scenes
        for scene_id, source_name in zip(scenes, source_names):
            spec = task_class.SCENE_SPECS[f"FloorPlan{scene_id}"]
            assert spec["source_fixture"] == f"FloorPlan{scene_id}:{source_name}"
            assert spec["surface_fixture"].startswith(f"FloorPlan{scene_id}:")
            assert spec["surface_region"].startswith(
                f"{spec['surface_fixture']}:region:"
            )
            assert set(spec["robot_pose"]) == {"xy", "yaw"}


def test_lift_soap_bottle_uses_exact_bathtub_401_and_accessible_407_layouts():
    specs = IThorBathroomLiftSoapBottle.SCENE_SPECS
    scene401 = specs["FloorPlan401"]
    assert scene401["surface_fixture"] == (
        "FloorPlan401:manual_support:scrub_brush_bathtub_rim"
    )
    assert scene401["surface_region"] == (
        "FloorPlan401:manual_support:scrub_brush_bathtub_rim:region:support"
    )
    assert scene401["source_xy"] == [-0.78, 0.70]
    assert scene401["source_quat"] == pytest.approx((0.70710678, 0.0, 0.70710678, 0.0))
    assert scene401["remove_fixtures"] == []
    assert scene401["auxiliary_placements"] == []
    assert scene401["robot_pose"] == {
        "xy": [-1.30, 0.70],
        "yaw": 0.0,
    }
    assert scene401["source_horizontal_distance"] == pytest.approx(0.52)

    scene407 = specs["FloorPlan407"]
    assert scene407["source_xy"] == [-1.09584, -1.44]
    assert scene407["remove_fixtures"] == []
    assert scene407["auxiliary_placements"] == [
        {
            "fixture": "FloorPlan407:Toilet_Paper_8",
            "xy": [-1.02, -1.60],
        }
    ]
    for scene_id in (405, 415, 418):
        spec = specs[f"FloorPlan{scene_id}"]
        assert spec["remove_fixtures"] == []
        assert spec["auxiliary_placements"] == []


def test_new_bathroom_object_task_layouts_stay_in_local_reach():
    scenes = (401, 405, 407, 415, 418)
    lift_tasks = (
        IThorBathroomLiftScrubBrush,
        IThorBathroomLiftSoapBottle,
        IThorBathroomLiftSprayBottle,
        IThorBathroomLiftToiletPaperRoll,
    )
    for task_class in lift_tasks:
        for scene_id in scenes:
            spec = task_class.SCENE_SPECS[f"FloorPlan{scene_id}"]
            robot_xy = np.asarray(spec["robot_pose"]["xy"], dtype=float)
            source_xy = np.asarray(spec["source_xy"], dtype=float)
            assert np.linalg.norm(robot_xy - source_xy) <= 0.70
            assert isinstance(spec["remove_fixtures"], list)
            assert isinstance(spec["auxiliary_placements"], list)

    used_spec = IThorBathroomToiletPaperRollIntoBin.SCENE_SPECS["FloorPlan401"]
    used_robot = np.asarray(used_spec["robot_pose"]["xy"], dtype=float)
    assert (
        np.linalg.norm(used_robot - np.asarray(used_spec["source_xy"], dtype=float))
        <= 0.70
    )
    assert (
        np.linalg.norm(used_robot - np.asarray(used_spec["bin_xy"], dtype=float))
        <= 0.70
    )

    spray401 = IThorBathroomLiftSprayBottle.SCENE_SPECS["FloorPlan401"]
    assert spray401["surface_fixture"] == "FloorPlan401:Side_Table_401_1_3"
    assert spray401["remove_fixtures"] == []

    # A reach distance is a per-task calibration, not a room-wide target.
    for task_class in lift_tasks:
        spec = task_class.SCENE_SPECS["FloorPlan401"]
        measured = np.linalg.norm(
            np.asarray(spec["robot_pose"]["xy"], dtype=float)
            - np.asarray(spec["source_xy"], dtype=float)
        )
        assert measured == pytest.approx(spec["source_horizontal_distance"], abs=1e-6)


def test_bathroom_source_distance_records_actual_near_pose_without_threshold():
    task = object.__new__(_BathroomWorkspaceMixin)
    task.sim = object()
    task.scene_name = "FloorPlanTest"
    spec = {
        "robot_pose": {"xy": [1.0, 2.0], "yaw": 0.0},
        "source_horizontal_distance": 0.60,
    }
    task.scene_task_spec = lambda: spec
    source = SimpleNamespace(world_center=lambda _sim: np.array([1.0, 2.60, 0.75]))

    assert task._record_source_horizontal_distance(source) == pytest.approx(0.60)
    assert task._source_horizontal_distance == pytest.approx(0.60)

    spec["source_horizontal_distance"] = 0.40
    assert task._record_source_horizontal_distance(source) == pytest.approx(0.60)


def test_native_bathroom_lift_requires_source_grasp_then_thirty_cm_lift():
    for task_class in (
        IThorBathroomLiftScrubBrush,
        IThorBathroomLiftSoapBottle,
        IThorBathroomLiftSprayBottle,
        IThorBathroomLiftToiletPaperRoll,
    ):
        assert task_class.MIN_LIFT_DELTA == pytest.approx(0.30)

    task = object.__new__(IThorBathroomLiftSoapBottle)
    task.source_fixture = object()
    task._initial_source_z = 1.0
    task._grasped_near_source = False
    state = {"grasped": False, "height": 1.0}
    task._g1_hand_grasping_fixture = lambda fixture: state["grasped"]
    task._source_root_z = lambda: state["height"]

    assert not task._raw_success()
    state.update(grasped=True, height=1.0)
    assert not task._raw_success()
    assert task._grasped_near_source
    state["height"] = 1.299
    assert not task._raw_success()
    state["height"] = 1.30
    assert task._raw_success()
    state["grasped"] = False
    assert not task._raw_success()


def test_new_bathroom_atomic_tasks_use_exact_primary_scene_calibrations():
    candle = IThorBathroomLiftCandleFromFloor.SCENE_SPECS["FloorPlan401"]
    assert IThorBathroomLiftCandleFromFloor.SUPPORTED_SCENE_IDS == (401,)
    assert candle["source_xy"] == pytest.approx((-1.60, 1.00))
    assert candle["stool_model"] == IThorBathroomLiftCandleFromFloor.STOOL_MODEL
    assert (
        IThorBathroomLiftCandleFromFloor.STOOL_MODEL
        == "/objects/lightwheel/stool/Stool006/model.xml"
    )
    assert candle["stool_scale"] == pytest.approx(
        IThorBathroomLiftCandleFromFloor.STOOL_SCALE
    )
    assert candle["stool_scale"] == pytest.approx(0.55)
    assert candle["stool_xy"] == pytest.approx((-1.62, 1.00))
    assert candle["stool_yaw"] == pytest.approx(0.0)
    assert candle["stool_top_half_xy"] == pytest.approx((0.075, 0.075))
    assert candle["stool_floor_probe_below"] == pytest.approx(0.01)
    assert candle["robot_pose"]["xy"] == pytest.approx((-1.15, 1.00))
    assert candle["robot_pose"]["yaw"] == pytest.approx(np.pi, abs=1e-6)
    assert candle["source_grasp_span"] == pytest.approx(0.075)
    assert candle["source_horizontal_distance"] == pytest.approx(0.45)
    assert candle["remove_fixtures"] == ["FloorPlan401:Candle_1_12"]
    assert task_assets.NATIVE_GRASP_SPANS[task_assets.CANDLE_MODEL] * (
        IThorBathroomLiftCandleFromFloor.TARGET_SCALE
    ) == pytest.approx(0.075)

    sponge = IThorBathroomPutSpongeInBathtub.SCENE_SPECS["FloorPlan401"]
    assert IThorBathroomPutSpongeInBathtub.SUPPORTED_SCENE_IDS == (
        401,
        405,
        407,
        415,
    )
    assert sponge["bathtub_fixture"] == "FloorPlan401:Bathtub_fc4e4aac_16"
    assert sponge["robot_pose"]["xy"] == pytest.approx((-1.189, 1.895))
    assert sponge["robot_pose"]["yaw"] == pytest.approx(-0.349066)
    assert sponge["source_horizontal_distance"] == pytest.approx(0.50)
    assert sponge["source_near_edge_distance"] == pytest.approx(0.40)
    assert task_assets.NATIVE_GRASP_SPANS[task_assets.SPONGE_MODEL] * (
        IThorBathroomPutSpongeInBathtub.SOURCE_SCALE
    ) == pytest.approx(0.075)

    expected_sponge_scenes = {
        405: {
            "rim_anchor_xy": (-1.08, 0.83),
            "remove_fixtures": ["FloorPlan405:Scrub_Brush_1_5"],
            "robot_pose": (-1.40, 0.83),
        },
        407: {
            "rim_anchor_xy": (0.54, -1.08),
            "remove_fixtures": ["FloorPlan407:Scrub_Brush_1_12"],
            "robot_pose": (-0.06, -1.08),
        },
        415: {
            "rim_anchor_xy": (-2.22, -0.83),
            "remove_fixtures": ["FloorPlan415:Scrub_Brush_1_1"],
            "robot_pose": (-1.57, -0.93),
        },
    }
    for scene_id, expected in expected_sponge_scenes.items():
        scene = f"FloorPlan{scene_id}"
        spec = IThorBathroomPutSpongeInBathtub.SCENE_SPECS[scene]
        assert spec["rim_fixture"] == (
            f"{scene}:manual_support:scrub_brush_bathtub_rim"
        )
        assert spec["rim_region"] == (
            f"{scene}:manual_support:scrub_brush_bathtub_rim:region:support"
        )
        assert spec["rim_anchor_xy"] == pytest.approx(expected["rim_anchor_xy"])
        assert spec["remove_fixtures"] == expected["remove_fixtures"]
        assert spec["robot_pose"]["xy"] == pytest.approx(expected["robot_pose"])
        assert len(spec["basin_center_xy"]) == 2
        assert all(value > 0 for value in spec["basin_half_xy"])
        assert spec["basin_z_range"][0] < spec["basin_z_range"][1]

    for task_class, scene_id, joint_type, initial, target in (
        (IThorBathroomOpenCabinetDoor, 405, "hinge", 0.20, 0.80),
        (IThorBathroomCloseCabinetDoor, 405, "hinge", 0.80, 0.20),
        (IThorBathroomOpenCabinetDoor, 407, "hinge", 0.20, 0.80),
        (IThorBathroomCloseCabinetDoor, 407, "hinge", 0.80, 0.20),
        (IThorBathroomOpenCabinetDoor, 418, "hinge", 0.10, 0.50),
        (IThorBathroomCloseCabinetDoor, 418, "hinge", 0.50, 0.10),
        (IThorBathroomOpenLowerDrawer, 415, "slide", 0.20, 0.80),
        (IThorBathroomCloseLowerDrawer, 415, "slide", 0.80, 0.20),
    ):
        expected_scenes = (
            (405, 407, 418)
            if task_class
            in (IThorBathroomOpenCabinetDoor, IThorBathroomCloseCabinetDoor)
            else (scene_id,)
        )
        assert task_class.SUPPORTED_SCENE_IDS == expected_scenes
        spec = task_class.SCENE_SPECS[f"FloorPlan{scene_id}"]
        assert spec["fixture"].startswith(f"FloorPlan{scene_id}:")
        assert spec["joint_type"] == joint_type
        assert spec["init_open_fraction"] == pytest.approx(initial)
        threshold = (
            spec["success_close_fraction"]
            if task_class.BEHAVIOR == "close"
            else spec["success_open_fraction"]
        )
        assert threshold == pytest.approx(target)
        assert set(spec["robot_pose"]) == {"xy", "yaw"}

    drawer = IThorBathroomOpenLowerDrawer.SCENE_SPECS["FloorPlan415"]
    assert drawer["fixture"] == "FloorPlan415:Dresser_413_1_12"
    assert "DrawerMesh031_3" in drawer["joint"]
    assert drawer["anchor_fixture"] is True
    for task_class in (
        IThorBathroomOpenLowerDrawer,
        IThorBathroomCloseLowerDrawer,
    ):
        drawer_pose = task_class.SCENE_SPECS["FloorPlan415"]["robot_pose"]
        assert drawer_pose["xy"] == pytest.approx((-0.7915, -3.5708))
        assert drawer_pose["yaw"] == pytest.approx(0.392699)

    open_cabinet = IThorBathroomOpenCabinetDoor.SCENE_SPECS["FloorPlan418"]
    close_cabinet = IThorBathroomCloseCabinetDoor.SCENE_SPECS["FloorPlan418"]
    for cabinet in (open_cabinet, close_cabinet):
        assert cabinet["robot_pose"]["xy"] == pytest.approx((-1.82, -2.60))
        assert cabinet["robot_pose"]["yaw"] == pytest.approx(np.pi, abs=1e-6)
    assert open_cabinet["robot_pose"] == close_cabinet["robot_pose"]

    cabinet_scene_specs = (
        (
            405,
            "FloorPlan405:Cabinet_cc651d88_3",
            "cabinet_4d8401876c23e5ef4325895368767cfe_1_1_0_joint_0",
            (-2.234461, -0.444552),
            np.pi,
            (-2.234461, -0.444552),
            np.pi,
        ),
        (
            407,
            "FloorPlan407:Cabinet_93cd2f94_2",
            "cabinet_e6db8a25b329231332368548293d3ba6_1_1_0_joint_0",
            (-1.745500, -0.927210),
            -0.523599,
            (-0.80, -0.90),
            -2.093995,
        ),
    )
    for (
        scene_id,
        fixture,
        joint,
        open_robot_xy,
        open_robot_yaw,
        close_robot_xy,
        close_robot_yaw,
    ) in cabinet_scene_specs:
        open_spec = IThorBathroomOpenCabinetDoor.SCENE_SPECS[f"FloorPlan{scene_id}"]
        close_spec = IThorBathroomCloseCabinetDoor.SCENE_SPECS[f"FloorPlan{scene_id}"]
        for spec, robot_xy, robot_yaw in (
            (open_spec, open_robot_xy, open_robot_yaw),
            (close_spec, close_robot_xy, close_robot_yaw),
        ):
            assert spec["fixture"] == fixture
            assert spec["joint"] == joint
            assert spec["joint_type"] == "hinge"
            assert spec["robot_pose"]["xy"] == pytest.approx(robot_xy)
            assert spec["robot_pose"]["yaw"] == pytest.approx(robot_yaw)
            assert spec["success_open_fraction"] == pytest.approx(0.80)
            assert spec["success_close_fraction"] == pytest.approx(0.20)
        assert open_spec["init_open_fraction"] == pytest.approx(0.20)
        assert close_spec["init_open_fraction"] == pytest.approx(0.80)


def test_stool_candle_lift_requires_grasp_at_reset_height_then_thirty_cm_lift():
    assert IThorBathroomLiftCandleFromFloor.MIN_LIFT_DELTA == pytest.approx(0.30)

    task = object.__new__(IThorBathroomLiftCandleFromFloor)
    task._initial_source_z = 0.527
    task._grasped_near_source = False
    state = {"grasped": False, "height": 0.527}
    task.g1_hand_grasping_object = lambda name: state["grasped"]
    task._source_root_z = lambda: state["height"]

    assert not task._raw_success()
    state["grasped"] = True
    assert not task._raw_success()
    assert task._grasped_near_source
    state["height"] = 0.826
    assert not task._raw_success()
    state["height"] = 0.827
    assert task._raw_success()
    state["grasped"] = False
    assert not task._raw_success()


def test_sponge_bathtub_predicate_relaxes_edge_fit_but_keeps_physical_guards():
    task = object.__new__(IThorBathroomPutSpongeInBathtub)
    task.basin_center_xy = np.array([-0.13, 0.92])
    task.basin_half_xy = np.array([0.48, 0.90])
    task.basin_z_range = np.array([0.02, 0.60])
    task.bathtub_fixture = object()
    state = {
        "points": _bbox_corners((-0.60, 0.80, 0.05), (0.40, 1.00, 0.55)),
        "contact": True,
        "penetration": False,
        "released": True,
    }
    task.object_bbox_points = lambda name: state["points"]
    task._source_has_shallow_bathtub_contact = lambda: state["contact"]
    task._source_has_illegal_penetration = lambda: state["penetration"]
    task._source_is_released = lambda: state["released"]

    # The center is safely in the basin even though this synthetic AABB is
    # wider than it. The real canonical sponge is only 7.5 cm across its grip.
    assert task._source_is_in_bathtub_and_released()
    state["contact"] = False
    assert not task._source_is_in_bathtub_and_released()
    state.update(contact=True, released=False)
    assert not task._source_is_in_bathtub_and_released()
    state.update(released=True, penetration=True)
    assert not task._source_is_in_bathtub_and_released()
    state.update(
        penetration=False,
        points=_bbox_corners((0.45, 0.80, 0.05), (0.65, 1.00, 0.55)),
    )
    assert not task._source_is_in_bathtub_and_released()
    state["points"] = _bbox_corners((-0.20, 0.80, 0.05), (0.00, 1.00, 0.64))
    assert not task._source_is_in_bathtub_and_released()


def test_sponge_bathtub_405_requires_full_footprint_and_safe_contact():
    task = object.__new__(IThorBathroomPutSpongeInBathtub)
    task.scene_name = "FloorPlan405"
    task.basin_center_xy = np.array([-0.54, 0.83])
    task.basin_half_xy = np.array([0.47, 0.90])
    task.basin_z_range = np.array([0.0, 0.53])
    state = {
        "points": _bbox_corners((-0.65, 0.78, 0.02), (-0.57, 0.89, 0.07)),
        "contact": True,
        "penetration": False,
    }
    task.object_bbox_points = lambda name: state["points"]
    task._source_has_shallow_bathtub_contact = lambda: state["contact"]
    task._source_has_illegal_penetration = lambda: state["penetration"]
    task._source_is_released = lambda: True
    assert task._source_is_in_bathtub_and_released()
    state["points"] = _bbox_corners((-1.06, 0.78, 0.02), (-0.98, 0.89, 0.07))
    assert not task._source_is_in_bathtub_and_released()  # overhangs near rim
    state["points"] = _bbox_corners((-0.65, 0.78, -0.09), (-0.57, 0.89, -0.04))
    assert not task._source_is_in_bathtub_and_released()  # under basin
    state["points"] = _bbox_corners((-0.65, 0.78, 0.02), (-0.57, 0.89, 0.07))
    state["penetration"] = True
    assert not task._source_is_in_bathtub_and_released()


@pytest.mark.parametrize(
    ("task_class", "failure_fraction", "success_fraction"),
    (
        (IThorBathroomOpenCabinetDoor, 0.49, 0.50),
        (IThorBathroomCloseCabinetDoor, 0.11, 0.10),
        (IThorBathroomOpenLowerDrawer, 0.79, 0.80),
        (IThorBathroomCloseLowerDrawer, 0.21, 0.20),
    ),
)
def test_bathroom_storage_success_uses_exact_joint_fraction_only(
    task_class,
    failure_fraction,
    success_fraction,
):
    task = object.__new__(task_class)
    task.sim = object()
    spec = next(iter(task_class.SCENE_SPECS.values()))
    task.SUCCESS_OPEN_FRACTION = float(spec["success_open_fraction"])
    task.SUCCESS_CLOSE_FRACTION = float(spec["success_close_fraction"])
    state = {"fraction": failure_fraction}
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    assert not task._raw_success()
    state["fraction"] = success_fraction
    assert task._raw_success()


def test_new_bathroom_scrub_brush_lift_and_soap_upright_specs_are_exact():
    scenes = (401, 405, 407, 415, 418)
    assert IThorBathroomLiftScrubBrush.SUPPORTED_SCENE_IDS == scenes
    for scene_id in scenes:
        scene = f"FloorPlan{scene_id}"
        lift = IThorBathroomLiftScrubBrush.SCENE_SPECS[scene]
        assert lift["source_fixture"].startswith(f"FloorPlan{scene_id}:Scrub_Brush_")

    assert IThorBathroomStandSoapBottleUpright.SUPPORTED_SCENE_IDS == scenes
    expected_surfaces = {
        401: "FloorPlan401:manual_support:scrub_brush_bathtub_rim",
        405: "FloorPlan405:CabinetBodyMesh_45407f37_25",
        407: "FloorPlan407:Countertopmesh_ebfc63b6_32",
        415: "FloorPlan415:TableMesh_31b19dd3_29",
        418: "FloorPlan418:FP418:BathroomSink2_5af95d26_18",
    }
    for scene_id in scenes:
        scene = f"FloorPlan{scene_id}"
        upright = IThorBathroomStandSoapBottleUpright.SCENE_SPECS[scene]
        assert upright["source_fixture"].startswith(f"{scene}:Soap_Bottle_")
        assert upright["surface_fixture"] == expected_surfaces[scene_id]
        assert upright["upright_axis"] == 1
        assert upright["upright_sign"] == pytest.approx(1.0)
        assert upright["source_quat"] == pytest.approx(
            (0.70710678, 0.0, 0.70710678, 0.0)
        )
        if scene_id == 407:
            assert upright["source_xy"] == pytest.approx((-1.00, -1.58))
            assert upright["remove_fixtures"] == [
                "FloorPlan407:Toilet_Paper_8",
            ]
            assert upright["auxiliary_placements"] == []
            assert upright["robot_pose"]["yaw"] == pytest.approx(-1.716862)
            assert upright["source_horizontal_distance"] == pytest.approx(0.708802)


def test_stand_soap_bottle_requires_upright_supported_released_pose():
    task = object.__new__(IThorBathroomStandSoapBottleUpright)
    task.source_fixture = object()
    task.work_surface_fixture = SimpleNamespace(body_names=("surface",))
    task.work_surface_region = object()
    task._target_upright_axis = (1, 1.0)
    state = {
        "upright": True,
        "inside": True,
        "on_plane": True,
        "contact": [0.0],
        "illegal": False,
        "released": True,
    }
    task._fixture_is_upright = lambda *args: state["upright"]
    task._fixture_bbox_inside_region_xy = lambda *args: state["inside"]
    task._fixture_bottom_is_on_support_plane = lambda: state["on_plane"]
    task._fixture_body_ids = lambda *args, **kwargs: {1}
    task._fixture_contact_distances = lambda *args: state["contact"]
    task._fixture_has_illegal_penetration = lambda *args: state["illegal"]
    task._fixture_released = lambda *args: state["released"]

    assert task._raw_success()
    for key in ("upright", "inside", "on_plane", "released"):
        state[key] = False
        assert not task._raw_success()
        state[key] = True
    state["contact"] = []
    assert not task._raw_success()
    state["contact"] = [-task.CONTACT_PENETRATION_TOLERANCE - 1e-4]
    assert not task._raw_success()
    state["contact"] = [0.0]
    state["illegal"] = True
    assert not task._raw_success()


def test_stand_soap_bottle_rejects_upside_down_and_excess_tilt():
    task = object.__new__(IThorBathroomStandSoapBottleUpright)
    fixture = SimpleNamespace(body_name="soap_bottle")
    matrix = np.eye(3)
    task.sim = SimpleNamespace(
        model=SimpleNamespace(body_name2id=lambda _name: 0),
        data=SimpleNamespace(body_xmat=np.asarray([matrix.reshape(-1)])),
    )
    target_axis = (1, 1.0)

    task.sim.data.body_xmat[0, 7] = 1.0
    assert task._fixture_is_upright(fixture, target_axis)
    task.sim.data.body_xmat[0, 7] = -1.0
    assert not task._fixture_is_upright(fixture, target_axis)
    task.sim.data.body_xmat[0, 7] = task.UPRIGHT_COSINE
    assert task._fixture_is_upright(fixture, target_axis)
    task.sim.data.body_xmat[0, 7] = task.UPRIGHT_COSINE - 1e-6
    assert not task._fixture_is_upright(fixture, target_axis)


def test_stand_soap_bottle_requires_bbox_bottom_near_support_plane():
    task = object.__new__(IThorBathroomStandSoapBottleUpright)
    task.sim = object()
    task.source_fixture = object()
    task.work_surface_region = SimpleNamespace(
        world_pose=lambda _sim: (np.array([0.0, 0.0, 1.0]), 0.0)
    )
    state = {"bottom": 1.0}
    task._fixture_bbox_points = lambda _fixture: np.array(
        [[0.0, 0.0, state["bottom"]], [0.1, 0.1, state["bottom"] + 0.2]]
    )

    assert task._fixture_bottom_is_on_support_plane()
    state["bottom"] = 1.0 - task.SUPPORT_PLANE_TOLERANCE + 1e-6
    assert task._fixture_bottom_is_on_support_plane()
    state["bottom"] -= 1e-4
    assert not task._fixture_bottom_is_on_support_plane()
    state["bottom"] = 1.0 + task.SUPPORT_PLANE_TOLERANCE + 1e-4
    assert not task._fixture_bottom_is_on_support_plane()


def test_full_roll_success_requires_containment_contact_release_and_fixed_bin():
    task = object.__new__(IThorBathroomToiletPaperRollIntoBin)
    task.source_fixture = object()
    task.bin_fixture = object()
    task._bin_upright_axis = object()
    state = {
        "inside": True,
        "contact": True,
        "released": True,
        "upright": True,
        "fixed": True,
    }
    task._source_inside_bin = lambda: state["inside"]
    task._fixture_contacts_fixture = lambda *args: state["contact"]
    task._fixture_released = lambda *args: state["released"]
    task._fixture_is_upright = lambda *args: state["upright"]
    task._bin_stays_at_reset_pose = lambda: state["fixed"]

    assert task._raw_success()
    for key in state:
        state[key] = False
        assert not task._raw_success()
        state[key] = True


def test_native_bathroom_release_uses_hand_contact_without_distance_threshold():
    task = object.__new__(IThorBathroomToiletPaperRollIntoBin)
    fixture = object()
    contacts = {"left": False, "right": False}
    task._g1_hand_contacts_fixture = (
        lambda side, target: target is fixture and contacts[side]
    )

    assert task._fixture_released(fixture)
    contacts["left"] = True
    assert not task._fixture_released(fixture)
    contacts["left"] = False
    contacts["right"] = True
    assert not task._fixture_released(fixture)


def test_bin_task_405_and_407_keep_roll_and_bin_in_one_stationary_workspace():
    expected = {
        405: {
            "source_xy": [-2.8, 0.55],
            "robot_xy": [-2.45, 0.65],
            "robot_yaw": 3.11,
        },
        407: {
            "source_xy": [-0.12, -1.35],
            "robot_xy": [0.10, -0.95],
            "robot_yaw": -1.46,
        },
    }
    bin_xy = {
        405: [-3.0808, 0.899995],
        407: [0.121763, -1.566798],
    }

    for scene_id, values in expected.items():
        spec = IThorBathroomToiletPaperRollIntoBin.SCENE_SPECS[f"FloorPlan{scene_id}"]
        assert spec["source_xy"] == values["source_xy"]
        if scene_id == 405:
            assert spec["surface_region"].endswith(
                "CabinetBodyMesh_45407f37_25:region:receptacle:4"
            )
        elif scene_id == 407:
            assert spec["surface_region"].endswith(
                "Countertopmesh_ebfc63b6_32:region:receptacle:1"
            )
        assert spec["robot_pose"]["xy"] == values["robot_xy"]
        assert spec["robot_pose"]["yaw"] == pytest.approx(values["robot_yaw"])
        robot_xy = np.asarray(values["robot_xy"], dtype=float)
        assert np.linalg.norm(robot_xy - np.asarray(values["source_xy"])) < 1.1
        assert np.linalg.norm(robot_xy - np.asarray(bin_xy[scene_id])) < 1.1


def test_bin_task_uses_native_full_roll_upright_in_floorplan401():
    assert IThorBathroomToiletPaperRollIntoBin.SOURCE_SCALE == pytest.approx(1.0)
    upright_quat = [0.0, 0.0, 0.70710678, 0.70710678]
    assert IThorBathroomToiletPaperRollIntoBin.SUPPORTED_SCENE_IDS == (
        401,
        405,
        407,
    )
    spec = IThorBathroomToiletPaperRollIntoBin.SCENE_SPECS["FloorPlan401"]
    assert spec["source_quat"] == upright_quat
    assert "Used_Up" not in spec["source_fixture"]
    assert spec["bin_region_z_offset"] == pytest.approx(-0.04)


@pytest.mark.parametrize(
    ("task_class", "closed_qpos", "open_qpos", "expected_qpos"),
    (
        (IThorBedroomOpenLaptop, np.pi / 2.0, 0.0, np.pi / 3.0),
        (IThorBedroomOpenHamperLid, 0.0, np.pi / 2.0, np.pi / 6.0),
    ),
)
def test_open_laptop_and_hamper_start_with_thirty_degree_gap(
    task_class,
    closed_qpos,
    open_qpos,
    expected_qpos,
):
    task = object.__new__(task_class)
    task.scene_name = "FloorPlanTest"
    joint = SimpleNamespace(
        closed_qpos=closed_qpos,
        open_qpos=open_qpos,
    )
    assert task._initial_joint_qpos(joint) == pytest.approx(expected_qpos)


@pytest.mark.parametrize(
    "task_class",
    (IThorBedroomOpenHamperLid, IThorBedroomCloseHamperLid),
)
def test_spawned_hamper_metadata_has_clearance_mapping(monkeypatch, task_class):
    monkeypatch.setattr(
        ExactSceneArticulation,
        "get_ep_meta",
        lambda _self: {},
    )
    task = object.__new__(task_class)

    meta = task.get_ep_meta()

    assert meta["hamper_robot_clearances"] == {}
    assert meta["hamper_base_fixed"] is True
    assert meta["hamper_on_floor"] is True


def test_open_and_close_hamper_share_floor_fixed_layout():
    open_spec = IThorBedroomOpenHamperLid.SCENE_SPECS["FloorPlan306"]
    close_spec = IThorBedroomCloseHamperLid.SCENE_SPECS["FloorPlan306"]
    assert close_spec == open_spec
    assert open_spec["hamper_floor_probe_below"] == pytest.approx(0.08)
    assert "hamper_z_offset" not in open_spec
    assert "hamper_pedestal_half_xy" not in open_spec
    assert open_spec["hamper_robot_pose"] == {
        "xy": [0.774725, -0.965884],
        "yaw": -2.233756,
    }
    assert open_spec["hamper_robot_clearances"] == {
        "FloorPlan306:Laundry_Hamper_1_1_14": 0.08,
        "FloorPlan306:AirConditioner_b71184a7_15": 0.25,
        "FloorPlan306:Bed_6_9": 0.10,
        "FloorPlan306:Desk_306_1_10": 0.10,
    }
    for task_class in (
        IThorBedroomOpenHamperLid,
        IThorBedroomCloseHamperLid,
    ):
        assert task_class.SUPPORTED_SCENE_IDS == (302, 306, 310, 323, 329)
        assert task_class.HAMPER_BASE_WELD_NAME == "task_hamper_fixed_base"
        for scene_id in (302, 310, 323, 329):
            spawned = task_class.SCENE_SPECS[f"FloorPlan{scene_id}"]
            assert spawned["fixture_mode"] == "spawn_asset"
            assert spawned["fixture"].startswith(f"FloorPlan{scene_id}:spawned:")
            assert spawned["spawn_quat"] == pytest.approx(
                [0.7071067812, 0.7071067812, 0.0, 0.0]
            )


def test_full_roll_bin_containment_uses_center_with_calibration_tolerance():
    task = object.__new__(IThorBathroomToiletPaperRollIntoBin)
    assert task.BIN_MAX_RESET_DISPLACEMENT == pytest.approx(0.10)
    assert task.BIN_CENTER_XY_TOLERANCE == pytest.approx(0.01)
    assert task.BIN_CENTER_Z_TOLERANCE == pytest.approx(0.01)
    source_center = np.array([0.0, 0.0, 0.20])
    task.source_fixture = SimpleNamespace(
        world_center=lambda _sim: source_center.copy()
    )
    task.sim = None
    task.bin_region = SimpleNamespace(
        half_size=np.array([0.10, 0.10, 0.20]),
        world_pose=lambda _sim: (np.array([0.0, 0.0, 0.20]), 0.0),
        contains_xy=lambda point, _sim, margin=0.0: bool(
            abs(float(point[0])) <= 0.10 - margin
            and abs(float(point[1])) <= 0.10 - margin
        ),
    )
    task._fixture_bbox_points = lambda _fixture: pytest.fail(
        "containment must not depend on the orientation-sensitive bbox"
    )

    assert task._source_inside_bin()

    source_center[:] = [0.109, 0.0, -0.009]
    assert task._source_inside_bin()

    source_center[0] = 0.011 + task.bin_region.half_size[0]
    assert not task._source_inside_bin()
    source_center[:] = [0.0, 0.0, -0.011]
    assert not task._source_inside_bin()
    source_center[:] = [0.0, 0.0, 0.411]
    assert not task._source_inside_bin()


def test_native_bathroom_pick_place_requires_one_second():
    class _Parent:
        def _post_action(self, action):
            return action

    class _Task(_BathroomNativePickPlaceTask, _Parent):
        pass

    task = object.__new__(_Task)
    valid = {"value": True}
    task._raw_success = lambda: valid["value"]
    task._success_hold = 0
    task.HOLD_STEPS = control_steps_for_duration(1.0, 20)
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    valid["value"] = False
    task._post_action(None)
    assert task._success_hold == 0
    assert not task._check_success()


def test_receptacle_success_requires_named_support_contact_and_release():
    task = _receptacle_contract_task(_bbox_corners((-0.2, -0.2, 0.1), (0.2, 0.2, 0.4)))
    task.check_contact = lambda _source, _receptacle: True
    released = {task.SOURCE_NAME: True, task.RECEPTACLE_NAME: True}
    task.object_released = lambda name, th: released[name]
    supported = {"value": True}
    task.object_on_region = lambda *args, **kwargs: supported["value"]

    assert task._raw_place_success()
    supported["value"] = False
    assert not task._raw_place_success()
    supported["value"] = True
    released[task.RECEPTACLE_NAME] = False
    assert not task._raw_place_success()


def test_shared_receptacle_rejects_deep_or_unrelated_support_penetration():
    task = _receptacle_contract_task(_bbox_corners((-0.2, -0.2, 0.1), (0.2, 0.2, 0.4)))
    task.object_on_region = lambda *args, **kwargs: True
    task.object_released = lambda *args, **kwargs: True
    state = {"shallow": True, "illegal": False}
    observed = {}

    def shallow(name, fixture, tolerance):
        observed["shallow"] = (name, fixture, tolerance)
        return state["shallow"]

    def illegal(name, region, tolerance):
        observed["illegal"] = (name, region, tolerance)
        return state["illegal"]

    task._object_fixture_contact_is_shallow = shallow
    task._object_has_illegal_penetration = illegal

    assert task._receptacle_is_upright_on_support()
    expected = task.RECEPTACLE_SUPPORT_PENETRATION_TOLERANCE
    assert observed["shallow"] == (
        task.RECEPTACLE_NAME,
        task.work_surface_fixture,
        expected,
    )
    assert observed["illegal"] == (
        task.RECEPTACLE_NAME,
        task.work_surface_region,
        expected,
    )

    state["shallow"] = False
    assert not task._receptacle_is_upright_on_support()
    state.update(shallow=True, illegal=True)
    assert not task._receptacle_is_upright_on_support()


def test_support_contact_probe_scans_down_to_first_shallow_contact():
    task = object.__new__(SupportContactPlacementMixin)
    task.objects = {"item": object()}
    task.sim = object()
    region = _region()
    fixture = SimpleNamespace(stable_id=region.fixture_id)
    state = {"z": None, "accepted": []}

    def set_bottom(_name, _xy, bottom_z):
        state["z"] = float(bottom_z)
        state["accepted"].append(float(bottom_z))

    task._set_task_object_bottom_at = set_bottom
    task.object_on_region = lambda *args, **kwargs: state["z"] <= 5.9501
    task._object_fixture_contact_is_shallow = lambda *args, **kwargs: True
    task._object_has_illegal_penetration = lambda *args, **kwargs: False
    task.object_bottom_z = lambda _name: state["z"]

    assert task.place_object_on_support_contact(
        "item",
        region,
        fixture,
        probe_above=0.01,
        probe_below=0.10,
        probe_step=0.01,
    )
    assert state["accepted"][0] == pytest.approx(6.01)
    assert state["z"] == pytest.approx(5.95)
    assert task._task_support_contracts == {"item": region}


def test_semantic_surface_selection_requires_both_exact_scene_mappings():
    task = object.__new__(SemanticSurfaceTask)
    task.scene_name = "FloorPlan1"
    task.work_surface_fixture = None
    task.work_surface_region = None
    task.WORK_SURFACE_FIXTURES = {"FloorPlan1": "fixture"}
    task.WORK_SURFACE_REGIONS = None
    task.get_fixture = lambda **_kwargs: pytest.fail(
        "fixture lookup must not run without both mappings"
    )
    task._select_work_surface()
    assert task.work_surface_fixture is None
    assert task.work_surface_region is None

    region = _region()
    fixture = SimpleNamespace(stable_id=region.fixture_id)
    task.WORK_SURFACE_FIXTURES = {"FloorPlan1": region.fixture_id}
    task.WORK_SURFACE_REGIONS = {"FloorPlan1": region.stable_id}
    task.get_fixture = lambda **_kwargs: fixture
    task.get_region = lambda **_kwargs: [region]
    task._support_matches_requirements = lambda *_args: True
    task._select_work_surface()
    assert task.work_surface_fixture is fixture
    assert task.work_surface_region is not region
    assert task.work_surface_region.stable_id == region.stable_id


@pytest.mark.parametrize(
    ("module", "getter", "scene_name", "task_key"),
    (
        (
            bedroom_specs,
            bedroom_specs.get_ithor_bedroom_task_spec,
            "FloorPlan302",
            "approach_bed",
        ),
        (
            bathroom_specs,
            bathroom_specs.get_ithor_bathroom_task_spec,
            "FloorPlan401",
            "approach_toilet",
        ),
    ),
)
def test_yaml_task_spec_loaders_are_strict_and_return_copies(
    module, getter, scene_name, task_key
):
    module._CACHE = None
    first = getter(scene_name, task_key, required=("fixture", "lang"))
    original_fixture = first["fixture"]
    first["fixture"] = "mutated"
    assert getter(scene_name, task_key)["fixture"] == original_fixture
    with pytest.raises(KeyError, match="No iTHOR"):
        getter("FloorPlan999", task_key)
    with pytest.raises(KeyError, match="No iTHOR"):
        getter(scene_name, "missing_task")
    with pytest.raises(ValueError, match="missing"):
        getter(scene_name, task_key, required=("missing_field",))


@pytest.mark.parametrize("module", (bedroom_specs, bathroom_specs))
@pytest.mark.parametrize(
    "malformed_document",
    (
        "scenes: []\n",
        "scenes: {}\n",
        "scenes:\n  FloorPlan1: []\n",
        "scenes:\n  FloorPlan1: {}\n",
        "scenes:\n  FloorPlan1:\n    task: {}\nextra: true\n",
    ),
)
def test_yaml_task_spec_loaders_require_a_scenes_catalog(
    module, malformed_document, tmp_path, monkeypatch
):
    malformed = tmp_path / "malformed.yaml"
    malformed.write_text(malformed_document)
    monkeypatch.setattr(module, "_SPEC_PATH", str(malformed))
    module._CACHE = None
    with pytest.raises(ValueError, match="Invalid iTHOR"):
        module._load_specs()


@pytest.mark.parametrize("module", (bedroom_specs, bathroom_specs))
def test_yaml_task_spec_loaders_reject_duplicate_keys(module, tmp_path, monkeypatch):
    duplicate = tmp_path / "duplicate.yaml"
    duplicate.write_text(
        "scenes:\n"
        "  FloorPlan1:\n"
        "    task:\n"
        "      fixture: first\n"
        "      fixture: second\n"
    )
    monkeypatch.setattr(module, "_SPEC_PATH", str(duplicate))
    module._CACHE = None
    with pytest.raises(ValueError, match="Duplicate YAML key 'fixture'"):
        module._load_specs()


class _DummyRoom:
    def __init__(self, *args, scene_ids=None, **kwargs):
        del args, kwargs
        self.scene_ids = tuple(scene_ids)
        self.scene_name = f"FloorPlan{self.scene_ids[0]}"


class _SpecTask(SceneSpecMixin, _DummyRoom):
    SUPPORTED_SCENE_IDS = (1, 2)
    SCENE_SPECS = {
        "FloorPlan1": {"nested": {"value": 1}},
        "FloorPlan2": {"nested": {"value": 2}},
    }


def test_scene_specs_restrict_scene_ids_and_return_defensive_copies():
    task = _SpecTask(scene_ids=[2])
    assert task.scene_ids == (2,)
    spec = task.scene_task_spec()
    spec["nested"]["value"] = 99
    assert task.scene_task_spec()["nested"]["value"] == 2

    with pytest.raises(ValueError, match="uncalibrated scene ids"):
        _SpecTask(scene_ids=[3])


def test_scene_specs_reject_catalog_drift():
    class _DriftedTask(SceneSpecMixin, _DummyRoom):
        SUPPORTED_SCENE_IDS = (1, 2)
        SCENE_SPECS = {"FloorPlan1": {}}

    with pytest.raises(ValueError, match="disagree"):
        _DriftedTask()


def test_catalog_is_derived_from_one_ordered_task_classes_tuple(monkeypatch):
    class First:
        SUPPORTED_SCENE_IDS = (1,)

    class Second:
        SUPPORTED_SCENE_IDS = (2,)

    tasks = SimpleNamespace(
        __name__="fake_room.tasks",
        TASK_CLASSES=(First, Second),
        __all__=["Second", "First"],
        First=First,
        Second=Second,
    )
    monkeypatch.setattr(task_catalog.importlib, "import_module", lambda _name: tasks)

    with pytest.raises(ValueError, match="exactly follow TASK_CLASSES"):
        task_catalog.discover_task_catalog(("fake_room",))

    tasks.__all__ = ["First", "Second"]
    entries = task_catalog.discover_task_catalog(("fake_room",))
    assert [entry.name for entry in entries] == ["First", "Second"]

    tasks.TASK_CLASSES = ()
    tasks.__all__ = []
    with pytest.raises(ValueError, match="TASK_CLASSES must not be empty"):
        task_catalog.discover_task_catalog(("fake_room",))


def test_catalog_validation_finds_registered_hidden_task(monkeypatch):
    package = "fake_room"

    class Public:
        SUPPORTED_SCENE_IDS = (1,)
        SCENE_SPECS = {"FloorPlan1": {"fixture": "fixture"}}
        TASK_FAMILY = "navigation"

    class Hidden:
        pass

    Public.__module__ = f"{package}.tasks.public"
    Hidden.__module__ = f"{package}.tasks.hidden"
    registry = {"Public": Public, "Hidden": Hidden}
    Public.REGISTRY = registry
    Hidden.REGISTRY = registry
    monkeypatch.setitem(task_catalog.REGISTERED_ENVS, "Public", Public)
    entry = task_catalog.TaskCatalogEntry(
        package=package,
        name="Public",
        cls=Public,
        scene_ids=(1,),
        family="navigation",
    )

    assert any(
        "Hidden: registered task is hidden" in issue
        for issue in task_catalog.validate_task_catalog([entry])
    )


def test_catalog_validation_reverse_scans_global_registry(monkeypatch):
    package = "fake_room"

    class Public:
        SUPPORTED_SCENE_IDS = (1,)
        SCENE_SPECS = {"FloorPlan1": {"fixture": "fixture"}}
        TASK_FAMILY = "navigation"

    class ObsoleteAlias:
        pass

    Public.__module__ = f"{package}.tasks.public"
    ObsoleteAlias.__module__ = f"{package}.tasks.legacy"
    Public.REGISTRY = {"Public": Public}
    monkeypatch.setitem(task_catalog.REGISTERED_ENVS, "Public", Public)
    monkeypatch.setitem(task_catalog.REGISTERED_ENVS, "ObsoleteAlias", ObsoleteAlias)
    entry = task_catalog.TaskCatalogEntry(
        package=package,
        name="Public",
        cls=Public,
        scene_ids=(1,),
        family="navigation",
    )

    assert any(
        "ObsoleteAlias: globally registered task is hidden" in issue
        for issue in task_catalog.validate_task_catalog([entry])
    )


def test_catalog_validation_rejects_duplicate_physical_signatures(monkeypatch):
    package = "fake_room"

    class First:
        SUPPORTED_SCENE_IDS = (1,)
        SCENE_SPECS = {
            "FloorPlan1": {"fixture": "same_fixture", "lang": "first wording"}
        }
        TASK_FAMILY = "navigation"

    class Second:
        SUPPORTED_SCENE_IDS = (1,)
        SCENE_SPECS = {
            "FloorPlan1": {"fixture": "same_fixture", "lang": "other wording"}
        }
        TASK_FAMILY = "navigation"

    for cls in (First, Second):
        cls.__module__ = f"{package}.tasks.public"
    registry = {"First": First, "Second": Second}
    First.REGISTRY = registry
    Second.REGISTRY = registry
    monkeypatch.setitem(task_catalog.REGISTERED_ENVS, "First", First)
    monkeypatch.setitem(task_catalog.REGISTERED_ENVS, "Second", Second)
    entries = [
        task_catalog.TaskCatalogEntry(
            package=package,
            name=cls.__name__,
            cls=cls,
            scene_ids=(1,),
            family="navigation",
        )
        for cls in (First, Second)
    ]

    assert any(
        "duplicate physical task" in issue
        for issue in task_catalog.validate_task_catalog(entries)
    )


def test_catalog_validation_rejects_final_task_inheriting_final_task(monkeypatch):
    package = "fake_room"

    class PublicBase:
        SUPPORTED_SCENE_IDS = (1,)
        TASK_FAMILY = "navigation"

    class PublicChild(PublicBase):
        SUPPORTED_SCENE_IDS = (2,)

    for cls in (PublicBase, PublicChild):
        cls.__module__ = f"{package}.tasks.public"
    registry = {cls.__name__: cls for cls in (PublicBase, PublicChild)}
    PublicBase.REGISTRY = registry
    PublicChild.REGISTRY = registry
    monkeypatch.setitem(task_catalog.REGISTERED_ENVS, "PublicBase", PublicBase)
    monkeypatch.setitem(task_catalog.REGISTERED_ENVS, "PublicChild", PublicChild)
    entries = [
        task_catalog.TaskCatalogEntry(
            package=package,
            name=cls.__name__,
            cls=cls,
            scene_ids=cls.SUPPORTED_SCENE_IDS,
            family=cls.TASK_FAMILY,
        )
        for cls in (PublicBase, PublicChild)
    ]

    assert any(
        "PublicChild: public task must not inherit public task PublicBase" in issue
        for issue in task_catalog.validate_task_catalog(entries)
    )


def test_catalog_validation_rejects_missing_physical_signature(monkeypatch):
    package = "fake_room"

    class LanguageOnly:
        SUPPORTED_SCENE_IDS = (1,)
        SCENE_SPECS = {"FloorPlan1": {"lang": "language is not geometry"}}
        TASK_FAMILY = "unknown_family"

    LanguageOnly.__module__ = f"{package}.tasks.public"
    LanguageOnly.REGISTRY = {"LanguageOnly": LanguageOnly}
    monkeypatch.setitem(task_catalog.REGISTERED_ENVS, "LanguageOnly", LanguageOnly)
    entry = task_catalog.TaskCatalogEntry(
        package=package,
        name="LanguageOnly",
        cls=LanguageOnly,
        scene_ids=(1,),
        family=LanguageOnly.TASK_FAMILY,
    )

    assert any(
        "LanguageOnly:1: missing physical task signature" in issue
        for issue in task_catalog.validate_task_catalog([entry])
    )


def test_bedroom_bathroom_catalog_counts_and_scene_matrix():
    assert task_catalog.EXPECTED_TASK_COUNTS == {
        task_catalog.BEDROOM_PACKAGE: 52,
        task_catalog.BATHROOM_PACKAGE: 61,
    }
    assert task_catalog.EXPECTED_TASK_MATRIX == {
        task_catalog.BEDROOM_PACKAGE: {
            302: {"atomic": 20, "composite": 15},
            306: {"atomic": 18, "composite": 16},
            310: {"atomic": 16, "composite": 14},
            323: {"atomic": 20, "composite": 15},
            329: {"atomic": 18, "composite": 23},
        },
        task_catalog.BATHROOM_PACKAGE: {
            401: {"atomic": 15, "composite": 24},
            405: {"atomic": 10, "composite": 14},
            407: {"atomic": 12, "composite": 18},
            415: {"atomic": 12, "composite": 14},
            418: {"atomic": 14, "composite": 22},
        },
    }
    entries = task_catalog.discover_task_catalog(
        (task_catalog.BEDROOM_PACKAGE, task_catalog.BATHROOM_PACKAGE)
    )

    assert task_catalog.validate_task_catalog(entries) == []
    counts = {
        package: sum(entry.package == package for entry in entries)
        for package in task_catalog.EXPECTED_TASK_COUNTS
    }
    assert counts == task_catalog.EXPECTED_TASK_COUNTS
    assert task_catalog.task_counts(entries) == task_catalog.EXPECTED_TASK_MATRIX
    assert all(
        task_catalog._task_signature(entry, scene_id) is not None
        for entry in entries
        for scene_id in entry.scene_ids
    )


def test_bathroom_catalog_family_counts():
    entries = task_catalog.discover_task_catalog((task_catalog.BATHROOM_PACKAGE,))
    counts = {}
    for entry in entries:
        counts[entry.family] = counts.get(entry.family, 0) + 1
    assert counts == {
        "articulation": 8,
        "composite": 40,
        "lift": 5,
        "navigation": 4,
        "pick_place": 3,
        "reorientation": 1,
    }


def test_every_bedroom_bathroom_catalog_task_uses_one_second_final_hold():
    entries = task_catalog.discover_task_catalog(
        (task_catalog.BEDROOM_PACKAGE, task_catalog.BATHROOM_PACKAGE)
    )

    assert len(entries) == 102
    for entry in entries:
        assert entry.cls.SUCCESS_HOLD_SECONDS == pytest.approx(1.0)
        assert hasattr(entry.cls, "HOLD_STEPS")


def test_removed_bathroom_tasks_have_no_catalog_registry_or_scene_specs():
    removed_names = {
        "IThorBathroomAppleIntoBowl",
        "IThorBathroomCanShelfPnP",
        "IThorBathroomCarryScrubBrushToCounter",
        "IThorBathroomPlungeToilet",
        "IThorBathroomScrubBrushInBathtub",
        "IThorBathroomShelfPnP",
        "IThorBathroomSpongeInBathtub",
        "IThorBathroomSpongeIntoSink",
        "IThorBathroomSpongeIntoBowl",
        "IThorBathroomSoapIntoBowl",
        "IThorBathroomSoapBottleOnCounter",
        "IThorBathroomStandSprayBottleUpright",
        "IThorBathroomUsedToiletPaperIntoBin",
        "IThorBathroomLiftAndStandSprayBottleUpright",
        "IThorBathroomLiftAndStandSoapBottleUpright",
        "IThorBathroomLiftSoapBottleThenStandSoapBottleUpright",
        "IThorBathroomLiftToiletPaperRollThenPutToiletPaperRollIntoBin",
        "IThorBathroomStandSoapBottleUprightThenLiftSoapBottle",
        "IThorBathroomBathingAreaThenToilet",
        "IThorBathroomDiscardBarSoap",
        "IThorBathroomSinkThenBathingArea",
        "IThorBathroomSinkThenToilet",
        "IThorBathroomToiletThenBathingArea",
        "IThorBathroomToiletThenSink",
        "IThorBathroomWalkAndCloseShowerDoor",
        "IThorBathroomWalkAndLiftSoapBottle",
        "IThorBathroomWalkAndLiftSprayBottle",
        "IThorBathroomWalkAndLiftToiletPaperRoll",
        "IThorBathroomWalkAndOpenShowerDoor",
        "IThorBathroomWalkAndOpenToiletLid",
        "IThorBathroomWalkAndStandSoapBottleUpright",
        "IThorBathroomWalkAndPutScrubBrushInBathtub",
        "IThorBathroomWalkAndPutScrubBrushInShower",
        "IThorBathroomWalkAndStandSprayBottleUpright",
    }
    entries = task_catalog.discover_task_catalog((task_catalog.BATHROOM_PACKAGE,))

    assert removed_names.isdisjoint(entry.name for entry in entries)
    assert removed_names.isdisjoint(task_catalog.REGISTERED_ENVS)
    for spec_name in (
        "apple_into_bowl",
        "can_shelf_pnp",
        "carry_scrub_brush_to_counter",
        "plunge_toilet",
        "scrub_brush_in_bathtub",
        "shelf_pnp",
        "soap_bottle_on_counter",
        "stand_spray_bottle_upright",
        "sponge_in_bathtub",
        "sponge_into_bowl",
        "sponge_into_sink",
        "soap_into_bowl",
        "used_toilet_paper_into_bin",
        "lift_and_stand_spray_bottle_upright",
        "lift_and_stand_soap_bottle_upright",
        "discard_bar_soap",
        "toilet_then_sink",
        "sink_then_toilet",
        "sink_then_bathing_area",
        "bathing_area_then_toilet",
        "toilet_then_bathing_area",
        "walk_and_lift_soap_bottle",
        "walk_and_lift_spray_bottle",
        "walk_and_lift_toilet_paper_roll",
        "walk_and_open_shower_door",
        "walk_and_close_shower_door",
        "walk_and_stand_soap_bottle_upright",
        "walk_and_put_scrub_brush_in_bathtub",
        "walk_and_put_scrub_brush_in_shower",
        "walk_and_stand_spray_bottle_upright",
    ):
        assert bathroom_specs.get_supported_scene_ids(spec_name) == ()


def test_removed_bedroom_tasks_have_no_catalog_registry_or_scene_specs():
    removed_names = {
        "IThorBedroomAppleInBowl",
        "IThorBedroomCarryBananaBasketToTable",
        "IThorBedroomCarryMugToDesk",
        "IThorBedroomCanToTrashBin",
        "IThorBedroomNestTwoCups",
        "IThorBedroomPackFruitIntoBasket",
        "IThorBedroomSoapIntoBowl",
        "IThorBedroomSpongeIntoBasket",
        "IThorBedroomWalkAndCloseLaptop",
        "IThorBedroomWalkAndLiftBasket",
        "IThorBedroomWalkAndPutBoxedFoodInBasket",
        "IThorBedroomWalkAndPutCanInBasket",
        "IThorBedroomWalkAndPutAppleInBowl",
        "IThorBedroomWalkAndPutSoapInBowl",
        "IThorBedroomWalkAndPutSpongeInBasket",
    }
    entries = task_catalog.discover_task_catalog((task_catalog.BEDROOM_PACKAGE,))

    assert removed_names.isdisjoint(entry.name for entry in entries)
    assert removed_names.isdisjoint(task_catalog.REGISTERED_ENVS)
    for spec_name in (
        "apple_in_bowl",
        "can_to_trash_bin",
        "carry_banana_basket_to_table",
        "carry_mug_to_desk",
        "nest_two_cups",
        "pack_fruit_into_basket",
        "soap_into_bowl",
        "sponge_into_basket",
        "walk_and_put_apple_in_bowl",
        "walk_and_put_soap_in_bowl",
        "walk_and_put_sponge_in_basket",
    ):
        assert bedroom_specs.get_supported_scene_ids(spec_name) == ()


def test_bedroom_pnp_scene_support_and_other_atomic_support_is_preserved():
    all_scenes = (302, 306, 310, 323, 329)
    for task_cls in (
        IThorBedroomOpenLaptop,
        IThorBedroomLiftBasket,
    ):
        assert task_cls.SUPPORTED_SCENE_IDS == all_scenes
        assert tuple(task_cls.SCENE_SPECS) == tuple(
            f"FloorPlan{scene_id}" for scene_id in all_scenes
        )

    for task_cls in (
        IThorBedroomCanIntoBasket,
        IThorBedroomMugIntoBasket,
        IThorBedroomBoxedFoodIntoBasket,
    ):
        assert task_cls.SUPPORTED_SCENE_IDS == (302, 306, 310, 323, 329)
        assert tuple(task_cls.SCENE_SPECS) == (
            "FloorPlan302",
            "FloorPlan306",
            "FloorPlan310",
            "FloorPlan323",
            "FloorPlan329",
        )
        for scene_name in task_cls.SCENE_SPECS:
            spec = bedroom_specs.get_ithor_bedroom_task_spec(
                scene_name, "grasp_workspace"
            )
            if scene_name in ("FloorPlan302", "FloorPlan323"):
                assert spec["fixture"] == (
                    scene_name + ":manual_support:bedroom_main_desk"
                )
                assert spec["region"] == spec["fixture"] + ":region:support"
            else:
                assert spec["fixture"].startswith(scene_name + ":Desk_")
                assert spec["region"].startswith(scene_name + ":Desk_")
            expected_distance = 0.60 if scene_name == "FloorPlan306" else 0.40
            assert spec["operation_normal_distance"] == pytest.approx(expected_distance)
            if scene_name == "FloorPlan323":
                assert spec["workspace_lateral_offset"] == pytest.approx(-0.20)

    laptop_329 = IThorBedroomOpenLaptop.SCENE_SPECS["FloorPlan329"]
    assert laptop_329["fixture_mode"] == "spawn_asset"
    assert laptop_329["asset_id"] == "Laptop_13"
    assert "open" in laptop_329["behaviors"]

    assert IThorBedroomMugIntoBasket.RECEPTACLE_XY_CONTAINMENT == "center"
    assert IThorBedroomBoxedFoodIntoBasket.RECEPTACLE_XY_CONTAINMENT == "center"


def test_bedroom_open_safe_uses_clear_bedside_tables_and_close_remains_raised():
    expected_offsets = {
        "FloorPlan302": 0.425,
        "FloorPlan323": -0.7025,
    }
    expected_open_supports = {
        "FloorPlan302": (
            "FloorPlan302:Side_Table_302_1_1_20",
            "FloorPlan302:Side_Table_302_1_1_20:region:support:1",
            ["FloorPlan302:Desk_Lamp_2_10"],
        ),
        "FloorPlan323": (
            "FloorPlan323:Side_Table_302_1_5_18",
            "FloorPlan323:Side_Table_302_1_5_18:region:support:1",
            [
                "FloorPlan323:Desk_Lamp_1_15",
                "FloorPlan323:Cellphone_5_2",
                "FloorPlan323:Mug_3_0",
            ],
        ),
    }
    assert IThorBedroomOpenSafeDoor.SUPPORTED_SCENE_IDS == (302, 323)
    assert set(IThorBedroomOpenSafeDoor.SCENE_SPECS) == set(expected_offsets)
    for scene_name, (support, region, removals) in expected_open_supports.items():
        spec = IThorBedroomOpenSafeDoor.SCENE_SPECS[scene_name]
        assert spec["safe_support_fixture"] == support
        assert spec["safe_support_region"] == region
        assert spec["safe_support_probe_below"] > 0.0
        assert spec["remove_fixtures"] == removals
        assert np.isfinite(spec["safe_yaw_delta"])
        assert "safe_z_offset" not in spec
        assert "safe_pedestal_half_xy" not in spec

    assert IThorBedroomCloseSafeDoor.SUPPORTED_SCENE_IDS == (302, 323)
    assert set(IThorBedroomCloseSafeDoor.SCENE_SPECS) == set(expected_offsets)
    for scene_name, expected_offset in expected_offsets.items():
        assert IThorBedroomCloseSafeDoor.SCENE_SPECS[scene_name][
            "safe_z_offset"
        ] == pytest.approx(expected_offset)

    close302 = IThorBedroomCloseSafeDoor.SCENE_SPECS["FloorPlan302"]
    assert close302["safe_xy"] == pytest.approx([-0.20, 0.45])
    assert close302["safe_robot_xy"] == pytest.approx([-1.00, 0.45])
    for task_cls in (IThorBedroomOpenSafeDoor, IThorBedroomCloseSafeDoor):
        assert task_cls.SAFE_BASE_WELD_NAME == "task_safe_fixed_base"


@pytest.mark.parametrize(
    ("task_class", "initial_fraction", "target_fraction"),
    (
        (IThorBedroomOpenDrawer, 0.20, 0.80),
        (IThorBedroomCloseDrawer, 0.80, 0.20),
        (IThorBedroomOpenTopDrawer, 0.20, 0.80),
        (IThorBedroomCloseTopDrawer, 0.80, 0.20),
        (IThorBedroomOpenSafeDoor, 0.20, 0.80),
        (IThorBedroomCloseSafeDoor, 0.80, 0.20),
    ),
)
def test_bedroom_drawers_and_small_safe_doors_use_twenty_eighty_stroke(
    task_class,
    initial_fraction,
    target_fraction,
):
    task = object.__new__(task_class)
    joint = SimpleNamespace(closed_qpos=-0.25, open_qpos=0.75)
    initial_qpos = task._initial_joint_qpos(joint)
    assert initial_qpos == pytest.approx(-0.25 + initial_fraction)
    threshold = (
        task.SUCCESS_OPEN_FRACTION
        if task.BEHAVIOR == "open"
        else task.SUCCESS_CLOSE_FRACTION
    )
    assert threshold == pytest.approx(target_fraction)


@pytest.mark.parametrize(
    "task_class", (IThorBedroomOpenBoxFlap, IThorBedroomCloseBoxFlap)
)
def test_bedroom_box_flap_uses_exact_scene_fixtures(task_class):
    expected = {
        "FloorPlan302": (
            "FloorPlan302:spawned:task_box",
            "task_box_Box_10_box_10_flap_outer_1_joint_0",
            -2.93938,
            0.0,
        ),
        "FloorPlan306": (
            "FloorPlan306:spawned:task_box",
            "task_box_Box_10_box_10_flap_outer_1_joint_0",
            -2.93938,
            0.0,
        ),
        "FloorPlan310": (
            "FloorPlan310:Box_10_2",
            "box_7c54a26cba93093ca8aceb6fbac82646_1_4_0_" "box10flapouter1joint0_3",
            -2.93938,
            0.0,
        ),
        "FloorPlan323": (
            "FloorPlan323:spawned:task_box",
            "task_box_Box_10_box_10_flap_outer_1_joint_0",
            -2.93938,
            0.0,
        ),
        "FloorPlan329": (
            "FloorPlan329:Box_29_2",
            "box_993495f45b10f36d15adc52172c4a7ec_1_1_0_" "box29flapouter1joint0_1",
            -2.74176,
            0.0,
        ),
    }
    assert task_class.SUPPORTED_SCENE_IDS == (302, 306, 310, 323, 329)
    assert task_class.BOX_BASE_WELD_NAME == "task_box_flap_fixed_base"
    assert tuple(task_class.SCENE_SPECS) == tuple(expected)
    vertical_starts = {
        "FloorPlan302": -1.3843677664,
        "FloorPlan306": -1.3843677664,
        "FloorPlan310": -1.3843677664,
        "FloorPlan323": -1.3843677664,
        "FloorPlan329": -1.1691420089,
    }
    for scene_name, target in expected.items():
        spec = task_class.SCENE_SPECS[scene_name]
        assert (
            spec["fixture"],
            spec["joint"],
            spec["closed_qpos"],
            spec["open_qpos"],
        ) == target
        assert (
            spec["box_floor_xy"]
            == {
                "FloorPlan302": [-0.25, -0.50],
                "FloorPlan306": [1.20, 0.00],
                "FloorPlan310": [-1.0, 0.15],
                "FloorPlan323": [0.00, -1.20],
                "FloorPlan329": [1.5, 0.0],
            }[scene_name]
        )
        assert spec["box_floor_probe_below"] == pytest.approx(0.08)
        assert "stand_top" not in spec
        assert spec["minimum_target_right_offset"] == pytest.approx(0.05)
        task = object.__new__(task_class)
        task.scene_name = scene_name
        task.sim = SimpleNamespace(
            model=SimpleNamespace(
                joint_name2id=lambda _name: 0,
                jnt_limited=np.zeros(1, dtype=int),
                jnt_range=np.array([[target[2], target[3]]], dtype=float),
            )
        )
        joint = SimpleNamespace(
            name=target[1],
            range=(target[2], target[3]),
            closed_qpos=0.0,
            open_qpos=target[2],
        )
        task._configure_articulation_joint(joint)
        assert joint.closed_qpos == pytest.approx(target[2])
        assert joint.open_qpos == pytest.approx(target[3])
        assert spec["open_start_qpos"] == pytest.approx(vertical_starts[scene_name])
        if task_class is IThorBedroomOpenBoxFlap:
            start_qpos = task._initial_joint_qpos(joint)
            start_fraction = (start_qpos - target[2]) / (target[3] - target[2])
            assert start_qpos == pytest.approx(vertical_starts[scene_name])
            assert 0.52 < start_fraction < 0.58
            assert task.sim.model.jnt_limited[0] == 1
            assert task.sim.model.jnt_range[0, 0] == pytest.approx(start_qpos)
    assert task_class.SCENE_SPECS["FloorPlan310"]["remove_fixtures"] == [
        "FloorPlan310:BaseballBat_2_14",
        "FloorPlan310:Tennis_Racquet_3_15",
    ]
    assert task_class.SCENE_SPECS["FloorPlan329"]["remove_fixtures"] == [
        "FloorPlan329:BaseballBat_1_13"
    ]
    for scene_name, distance in (("FloorPlan310", 0.55), ("FloorPlan329", 0.50)):
        spec = task_class.SCENE_SPECS[scene_name]
        assert spec["robot_outward_direction"] == [0.0, -1.0]
        assert spec["robot_center_distance"] == pytest.approx(distance)


@pytest.mark.parametrize(
    "task_cls", (IThorBedroomOpenTopDrawer, IThorBedroomCloseTopDrawer)
)
def test_bedroom_top_drawer_uses_exact_alternate_fixtures(task_cls):
    expected = {
        "FloorPlan302": (
            "FloorPlan302:Side_Table_302_2_1_17",
            "stand_ff4dc8c26a171f1ee525b510e1af7754_1_2_0_joint_1",
        ),
        "FloorPlan306": (
            "FloorPlan306:Side_Table_306_1_15",
            "stand_789e1bee5ebbeabb503337b46417ac81_1_2_0_joint_1",
        ),
        "FloorPlan323": (
            "FloorPlan323:Side_Table_302_1_5_22",
            "stand_ea63f84ad6a9306dd5bfefe095742a7d_2_2_0_joint_1",
        ),
        "FloorPlan329": (
            "FloorPlan329:Side_Table_329_1_15",
            "stand_97b1d5461f86f5f69c9681415d63ebad_1_2_0_joint_1",
        ),
    }

    assert task_cls.SUPPORTED_SCENE_IDS == (302, 306, 323, 329)
    assert set(task_cls.SCENE_SPECS) == set(expected)
    for scene_name, (fixture, joint) in expected.items():
        spec = task_cls.SCENE_SPECS[scene_name]
        assert (spec["fixture"], spec["joint"]) == (fixture, joint)
        primary = bedroom_specs.get_ithor_bedroom_task_spec(
            scene_name, "drawer", required=("fixture", "joint")
        )
        assert spec["fixture"] != primary["fixture"]
        assert spec["joint"] != primary["joint"]


@pytest.mark.parametrize("overlap_field", ("fixture", "joint"))
def test_bedroom_top_drawer_loader_rejects_drawer_overlap(monkeypatch, overlap_field):
    primary = {"fixture": "fixture-a", "joint": "joint-a"}
    alternate = {"fixture": "fixture-b", "joint": "joint-b"}
    alternate[overlap_field] = primary[overlap_field]
    alternate.update(target="top drawer", lang="{verb} the top drawer")
    monkeypatch.setattr(
        bedroom_articulation,
        "get_ithor_bedroom_task_specs",
        lambda *args, **kwargs: {"FloorPlan1": alternate},
    )
    monkeypatch.setattr(
        bedroom_articulation,
        "get_ithor_bedroom_task_spec",
        lambda *args, **kwargs: primary,
    )

    with pytest.raises(ValueError, match=f"different {overlap_field}"):
        bedroom_articulation._load_bedroom_top_drawer_scene_specs()


@pytest.mark.parametrize(
    ("task_cls", "threshold", "roundoff_value", "failing_value"),
    (
        (IThorBedroomOpenTopDrawer, 0.80, 0.80 - 5e-10, 0.80 - 2e-9),
        (IThorBedroomCloseTopDrawer, 0.20, 0.20 + 5e-10, 0.20 + 2e-9),
    ),
)
def test_bedroom_top_drawer_threshold_tolerates_only_float_roundoff(
    task_cls, threshold, roundoff_value, failing_value
):
    task = object.__new__(task_cls)
    task.sim = object()
    fraction = {"value": threshold}
    task.target_semantic_joint = SimpleNamespace(
        fraction=lambda _sim: fraction["value"]
    )

    assert task._raw_success()
    fraction["value"] = roundoff_value
    assert task._raw_success()
    fraction["value"] = failing_value
    assert not task._raw_success()


def test_legacy_room_task_registrations_do_not_return_on_fresh_import():
    script = """
import json
import robocasa
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.environments.ithor_base import (
    REGISTERED_ITHOR_BATHROOM_ENVS,
    REGISTERED_ITHOR_BEDROOM_ENVS,
)
print(json.dumps({
    "global": sorted(REGISTERED_ENVS),
    "bedroom": sorted(REGISTERED_ITHOR_BEDROOM_ENVS),
    "bathroom": sorted(REGISTERED_ITHOR_BATHROOM_ENVS),
}))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    registries = json.loads(result.stdout.strip().splitlines()[-1])
    denied_exact = {
        "NavigateToBed",
        "NavigateToDesk",
        "AppleIntoBowl",
        "CloseLaptopFiveScene",
        "StackTwoCups",
        "NavigateToToilet",
        "NavigateToSink",
        "IThorBedroomNavigateToBed",
        "IThorBedroomNavigateToDesk",
        "IThorBedroomAppleIntoBowl",
        "IThorBedroomCloseLaptopFiveScene",
        "IThorBedroomStackTwoCups",
        "IThorBathroomNavigateToToilet",
        "IThorBathroomNavigateToSink",
    }
    for registry_names in registries.values():
        assert denied_exact.isdisjoint(registry_names)
        assert not any(
            name.startswith("IThorBedroom")
            and ("FiveScene" in name or "Scene306" in name or "Scene329" in name)
            for name in registry_names
        )
        assert not any(
            name.startswith("IThorBathroom") and "FiveScene" in name
            for name in registry_names
        )


def test_legacy_aggregate_and_registration_modules_are_absent():
    root = Path(__file__).resolve().parents[1]
    forbidden = (
        "validate_new_suite.py",
        "robocasa/environments/ithor_atomic/room_task_suite.py",
        "robocasa/environments/ithor_bedroom/tasks/five_scene_suite.py",
        "robocasa/environments/ithor_bedroom/tasks/_five_scene_common.py",
        "robocasa/environments/ithor_bedroom/tasks/_registration.py",
        "robocasa/environments/ithor_bathroom/tasks/five_scene_suite.py",
        "robocasa/environments/ithor_bathroom/tasks/_five_scene_common.py",
        "robocasa/environments/ithor_bathroom/tasks/_registration.py",
        "robocasa/environments/ithor_bathroom/tasks/experimental.py",
        "robocasa/environments/ithor_bathroom/tasks/plunge_toilet.py",
    )
    assert not [relative for relative in forbidden if (root / relative).exists()]
    for room in ("ithor_bedroom", "ithor_bathroom"):
        task_dir = root / "robocasa" / "environments" / room / "tasks"
        assert all(
            "TASK_MODULES" not in source.read_text() for source in task_dir.glob("*.py")
        )


def test_reset_validator_reuses_one_single_scene_environment(monkeypatch):
    entry = task_catalog.TaskCatalogEntry(
        package="fake_room",
        name="FakeTask",
        cls=object,
        scene_ids=(7,),
        family="navigation",
    )

    class FakeEnvironment:
        def __init__(self):
            self.reset_count = 0
            self.close_count = 0

        def reset(self):
            self.reset_count += 1

        def close(self):
            self.close_count += 1

        def count_body_collisions(self):
            return 0

        def _check_success(self):
            return False

        def _robot_base_xy(self):
            return np.zeros(2)

        def robot_footprint_is_on_floor(self, _xy, radius):
            return radius == pytest.approx(0.25)

    environment = FakeEnvironment()
    make_kwargs = []
    monkeypatch.setattr(
        task_validator,
        "_load_reset_controller",
        lambda *_args: object(),
    )
    monkeypatch.setattr(
        task_validator.robosuite,
        "make",
        lambda **kwargs: make_kwargs.append(kwargs) or environment,
    )
    monkeypatch.setattr(
        task_validator, "_assert_raw_predicates_false", lambda *_args: None
    )
    monkeypatch.setattr(
        task_validator, "_validate_support_stability", lambda *_args: None
    )

    assert (
        task_validator._validate_resets(
            [entry], resets=5, task_filter="", scene_filter="", atomic_only=False
        )
        == []
    )
    assert environment.reset_count == 5
    assert environment.close_count == 1
    assert len(make_kwargs) == 1
    assert make_kwargs[0]["scene_ids"] == [7]
    assert make_kwargs[0]["hard_reset"] is False


def test_isolated_settle_freezes_only_robot_joint_state():
    qpos = np.arange(9, dtype=float)
    qvel = np.arange(8, dtype=float)

    class FakeSim:
        def __init__(self):
            self.model = SimpleNamespace(
                body_jntadr=np.array([-1, 0, 1]),
                body_jntnum=np.array([0, 1, 1]),
                jnt_qposadr=np.array([0, 7, 8]),
                jnt_dofadr=np.array([0, 6, 7]),
                njnt=3,
                nq=9,
                nv=8,
                body_name2id=lambda _name: 1,
            )
            self.data = SimpleNamespace(qpos=qpos.copy(), qvel=qvel.copy())
            self.forward_count = 0

        def step(self):
            self.data.qpos += 1.0
            self.data.qvel += 1.0

        def forward(self):
            self.forward_count += 1

    sim = FakeSim()
    env = SimpleNamespace(
        sim=sim,
        robots=[SimpleNamespace(robot_model=SimpleNamespace(root_body="robot"))],
        _body_tree_ids=lambda _roots: {1, 2},
    )
    task_validator._settle_scene_with_frozen_robot(env, 3)

    np.testing.assert_allclose(sim.data.qpos[:8], qpos[:8])
    np.testing.assert_allclose(sim.data.qvel[:7], qvel[:7])
    assert sim.data.qpos[8] == pytest.approx(qpos[8] + 3.0)
    assert sim.data.qvel[7] == pytest.approx(qvel[7] + 3.0)
    assert sim.forward_count == 3


def test_fixture_placement_rejects_nonfinite_minimum_clearance():
    env = object.__new__(IThorRoomEnv)
    env.sim = object()
    fixture = SimpleNamespace(
        world_center=lambda _sim: np.zeros(3),
        radial_extent=lambda _direction: 0.0,
    )
    env.get_fixture = lambda **_kwargs: fixture
    env._set_robot_free_pose = lambda *_args: True
    env._calibrate_robot_free_joint_height = lambda: True
    distance = {"value": np.nan}
    env.robot_fixture_collision_distance = lambda *_args, **_kwargs: distance["value"]
    kwargs = dict(
        distances=(0.5,),
        lateral_offsets=(0.0,),
        max_yaw_offset_deg=0,
        approach_directions=(np.array([1.0, 0.0]),),
        min_fixture_distance=0.25,
    )

    assert not env.place_robot_at_fixture(fixture, **kwargs)
    distance["value"] = 0.30
    assert env.place_robot_at_fixture(fixture, **kwargs)


def test_body_contact_contract_registration_is_neutral_and_strict():
    env = object.__new__(IThorRoomEnv)
    env.register_task_body_contact_contract(
        "laptop_table",
        "laptop_root",
        ("table_root", 0),
    )
    assert env._task_body_contact_contracts == {
        "laptop_table": {
            "subject_body_roots": ("laptop_root",),
            "support_body_roots": ("table_root", 0),
        }
    }
    with pytest.raises(ValueError, match="nonempty"):
        env.register_task_body_contact_contract("", "subject", "support")
    with pytest.raises(ValueError, match="must contain"):
        env.register_task_body_contact_contract("bad", (), "support")


def test_body_contact_contract_treats_world_as_one_support_body():
    model = SimpleNamespace(
        nbody=4,
        geom_bodyid=np.array([1, 0]),
        body_name2id=lambda name: {"subject": 1}[name],
    )
    contact = SimpleNamespace(geom1=0, geom2=1, dist=-0.001)
    env = SimpleNamespace(
        sim=SimpleNamespace(
            model=model,
            data=SimpleNamespace(ncon=1, contact=[contact]),
        ),
        # This deliberately mimics the normal tree helper's world expansion;
        # the contract resolver must special-case body zero before calling it.
        _body_tree_ids=lambda roots: ({0, 1, 2, 3} if 0 in roots else set(roots)),
        illegal_contact_pairs_for_bodies=lambda *_args, **_kwargs: set(),
    )

    _, world_bodies = task_validator._resolve_body_roots(
        env, (0,), "subject_floor", "support"
    )
    assert world_bodies == {0}
    assert task_validator._assert_body_contact_state(
        env,
        "subject_floor",
        ("subject",),
        (0,),
        "initial",
    ) == {1}
    contact.dist = -0.004
    with pytest.raises(AssertionError, match="penetrates support"):
        task_validator._assert_body_contact_state(
            env,
            "subject_floor",
            ("subject",),
            (0,),
            "initial",
        )


@pytest.mark.parametrize("predicate_name", ("_raw_success", "_raw_place_success"))
def test_reset_validator_rejects_initial_raw_success(predicate_name):
    env = SimpleNamespace(**{predicate_name: lambda: True})
    with pytest.raises(AssertionError, match=f"{predicate_name} is already satisfied"):
        task_validator._assert_raw_predicates_false(env, "initial")


class _StateMachineRoom(_DummyRoom):
    def _post_action(self, action):
        return action


class _ExactNavigationTask(ExactSceneNavigation, _StateMachineRoom):
    SUPPORTED_SCENE_IDS = (1,)
    SCENE_SPECS = {
        "FloorPlan1": {
            "fixture": "target",
            "lang": "approach target",
            "start_pose": {"xy": [1.0, 2.0], "yaw": 0.25},
            "minimum_start_distance": 0.75,
            "remove_fixtures": ["FloorPlan1:small_obstacle"],
        }
    }


def test_exact_navigation_requires_one_explicit_distant_start():
    task = _ExactNavigationTask()
    task._navigation_success_distance = 0.35

    xy, yaw, minimum, removals = task._navigation_start_config()
    np.testing.assert_allclose(xy, [1.0, 2.0])
    assert yaw == pytest.approx(0.25)
    assert minimum == pytest.approx(0.75)
    assert removals == ("FloorPlan1:small_obstacle",)
    assert not hasattr(task, "START_DISTANCES")

    valid = task.scene_task_spec()
    task.scene_task_spec = lambda: {**valid, "minimum_start_distance": 0.49}
    with pytest.raises(ValueError, match="at least 0.15 m"):
        task._navigation_start_config()

    task.scene_task_spec = lambda: {
        **valid,
        "remove_fixtures": ["FloorPlan1:small_obstacle"] * 2,
    }
    with pytest.raises(ValueError, match="duplicate"):
        task._navigation_start_config()


def test_exact_navigation_requires_an_unbroken_hold():
    task = _ExactNavigationTask()
    task._in_range_steps = 0
    near = {"value": True}
    task._robot_is_near_target = lambda: near["value"]

    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    near["value"] = False
    task._post_action(None)
    assert task._in_range_steps == 0
    near["value"] = True
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()
    near["value"] = False
    task._post_action(None)
    assert not task._check_success()


def test_exact_navigation_requires_fixture_id_and_collision_distance():
    task = _ExactNavigationTask()
    task.target_fixture = object()
    task._navigation_success_distance = 0.35
    distance = {"value": np.inf}
    task.robot_fixture_collision_distance = lambda *_args, **_kwargs: distance["value"]

    assert task._navigation_fixture_query() == {"id": "target"}
    assert not task._robot_is_near_target()
    distance["value"] = 0.34
    assert task._robot_is_near_target()

    class MissingFixtureTask(ExactSceneNavigation, _StateMachineRoom):
        SUPPORTED_SCENE_IDS = (1,)
        SCENE_SPECS = {"FloorPlan1": {"lang": "invalid navigation"}}

    with pytest.raises(KeyError, match="fixture"):
        MissingFixtureTask()._navigation_fixture_query()


class _ExactArticulationTask(ExactSceneArticulation, _StateMachineRoom):
    BEHAVIOR = "open"
    SUPPORTED_SCENE_IDS = (1,)
    SCENE_SPECS = {
        "FloorPlan1": {
            "fixture": "target",
            "joint": "target_joint",
            "target": "door",
            "lang": "open target",
        }
    }


def test_exact_articulation_requires_an_unbroken_hold():
    task = _ExactArticulationTask()
    task._success_hold = 0
    raw_success = {"value": True}
    task._raw_success = lambda: raw_success["value"]

    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    raw_success["value"] = False
    task._post_action(None)
    assert task._success_hold == 0
    raw_success["value"] = True
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()
    raw_success["value"] = False
    task._post_action(None)
    assert not task._check_success()


@pytest.mark.parametrize(
    "task_class,behavior,initial_fraction,open_threshold",
    (
        (IThorBathroomOpenShowerDoor, "open", 0.0, 1.0 / 3.0),
        (IThorBathroomCloseShowerDoor, "close", 1.0 / 3.0, 1.0 / 3.0),
        (
            IThorBathroomNavigateToBathingAreaAndOpenShowerDoor,
            "open",
            0.0,
            1.0 / 3.0,
        ),
        (
            IThorBathroomNavigateToBathingAreaAndCloseShowerDoor,
            "close",
            1.0 / 3.0,
            1.0 / 3.0,
        ),
    ),
)
def test_shower_door_tasks_bind_only_exact_hinge_specs(
    task_class, behavior, initial_fraction, open_threshold
):
    assert task_class.SUPPORTED_SCENE_IDS == (407, 418)
    assert task_class.BEHAVIOR == behavior
    assert task_class.TARGET_JOINT_TYPES == ("hinge",)
    for scene_name, spec in task_class.SCENE_SPECS.items():
        assert scene_name in {"FloorPlan407", "FloorPlan418"}
        assert spec["fixture"].startswith(f"{scene_name}:ShowerDoor_")
        assert spec["joint"].startswith("door_")
        assert spec["joint_type"] == "hinge"
        assert spec["joint_range"] == [0.0, 1.5708]
        assert spec["closed_qpos"] == pytest.approx(0.0)
        assert spec["open_qpos"] == pytest.approx(1.5708)
        assert spec["init_open_fraction"] == pytest.approx(initial_fraction)
        assert spec["success_open_fraction"] == pytest.approx(open_threshold)
        assert spec["success_close_fraction"] == pytest.approx(2.0 / 90.0)
        assert spec["target_open_angle_degrees"] == pytest.approx(30.0)
        assert spec["closed_tolerance_degrees"] == pytest.approx(2.0)
        assert "hold_steps" not in spec


def test_atomic_shower_door_open_target_is_close_start():
    for scene_name in ("FloorPlan407", "FloorPlan418"):
        open_spec = IThorBathroomOpenShowerDoor.SCENE_SPECS[scene_name]
        close_spec = IThorBathroomCloseShowerDoor.SCENE_SPECS[scene_name]
        assert open_spec["success_open_fraction"] == pytest.approx(1.0 / 3.0)
        assert close_spec["init_open_fraction"] == pytest.approx(
            open_spec["success_open_fraction"]
        )
        travel = open_spec["open_qpos"] - open_spec["closed_qpos"]
        assert np.rad2deg(travel * open_spec["success_open_fraction"]) == pytest.approx(
            30.0, abs=1e-3
        )


@pytest.mark.parametrize(
    ("task_class", "initial_fraction", "success_fraction"),
    (
        (IThorBathroomOpenToiletLid, 0.20, 0.80),
        (IThorBathroomCloseToiletLid, 0.80, 0.20),
        (IThorBathroomApproachToiletAndOpenToiletLid, 0.20, 0.80),
        (IThorBathroomApproachToiletAndCloseToiletLid, 1.00, 0.20),
        (IThorBathroomCloseThenOpenToiletLid, 1.00, 0.20),
    ),
)
def test_toilet_lid_atomic_close_keeps_eighty_percent_compatibility(
    task_class,
    initial_fraction,
    success_fraction,
):
    task = object.__new__(task_class)
    joint = SimpleNamespace(closed_qpos=0.0, open_qpos=-1.5)
    initial_qpos = task._initial_joint_qpos(joint)

    assert initial_qpos == pytest.approx(-1.5 * initial_fraction)
    assert task.OPEN_START_FRACTION == pytest.approx(0.20)
    if task.BEHAVIOR == "close":
        assert task.SUCCESS_CLOSE_FRACTION == pytest.approx(success_fraction)
        outside = success_fraction + 0.01
    else:
        assert task.SUCCESS_OPEN_FRACTION == pytest.approx(success_fraction)
        outside = success_fraction - 0.01

    fraction = {"value": success_fraction}
    success_joint = SimpleNamespace(fraction=lambda _sim: fraction["value"])
    task.sim = object()
    task._success_joints = lambda: [success_joint]
    assert task._raw_success()
    fraction["value"] = outside
    assert not task._raw_success()


def test_floorplan405_close_toilet_lid_starts_slightly_more_open():
    task = object.__new__(IThorBathroomCloseToiletLid)
    joint = SimpleNamespace(closed_qpos=0.0, open_qpos=-1.51844)

    task.scene_name = "FloorPlan405"
    assert task._initial_joint_qpos(joint) == pytest.approx(-1.51844 * 0.85)

    task.scene_name = "FloorPlan418"
    assert task._initial_joint_qpos(joint) == pytest.approx(-1.51844 * 0.80)


@pytest.mark.parametrize(
    ("scene_name", "expected_xy", "expected_yaw"),
    (
        ("FloorPlan401", (0.116, 2.494), 1.570796),
        ("FloorPlan405", (-2.493, 1.745), -2.487094),
        ("FloorPlan407", (-1.007, 0.363), -2.976444),
        ("FloorPlan415", (-0.258, -1.151), 1.735945),
        ("FloorPlan418", (-0.249, -2.495), -1.832596),
    ),
)
def test_close_toilet_lid_uses_exact_sweep_clear_pose(
    scene_name,
    expected_xy,
    expected_yaw,
):
    task = object.__new__(IThorBathroomCloseToiletLid)
    task.scene_name = scene_name

    xy, yaw = task._close_robot_pose()

    np.testing.assert_allclose(xy, expected_xy)
    assert yaw == pytest.approx(expected_yaw)
    assert task.INITIAL_ROBOT_JITTER_RADIUS == pytest.approx(0.0)
    assert task.LID_SWEEP_SAMPLES >= 10


@pytest.mark.parametrize(
    "task_class,boundary,outside",
    (
        (IThorBathroomOpenShowerDoor, 1.0 / 3.0, 0.332),
        (IThorBathroomCloseShowerDoor, 2.0 / 90.0, 0.023),
    ),
)
def test_atomic_shower_door_threshold_and_metadata(
    monkeypatch, task_class, boundary, outside
):
    monkeypatch.setattr(
        ExactSceneArticulation,
        "get_ep_meta",
        lambda _self: {},
    )
    task = object.__new__(task_class)
    task.sim = object()
    task.OPEN_INIT_FRACTION = 0.0 if task.BEHAVIOR == "open" else 1.0 / 3.0
    task.SUCCESS_OPEN_FRACTION = 1.0 / 3.0
    task.SUCCESS_CLOSE_FRACTION = 2.0 / 90.0
    task.HOLD_STEPS = 5
    fraction = {"value": boundary}
    joint = SimpleNamespace(
        closed_qpos=0.0,
        open_qpos=1.5708,
        fraction=lambda _sim: fraction["value"],
    )
    task.target_semantic_joint = joint
    task._success_joints = lambda: [joint]

    assert task._raw_success()
    fraction["value"] = outside
    assert not task._raw_success()

    meta = task.get_ep_meta()
    assert meta["mechanism"] == "shower door"
    assert meta["target_state"] == task.BEHAVIOR
    assert meta["success_fraction_threshold"] == pytest.approx(boundary)
    assert "success_open_fraction_threshold" not in meta


def _stub_task_parent_post_action(monkeypatch, task_class):
    monkeypatch.setattr(
        task_class.__mro__[1],
        "_post_action",
        lambda _self, action: action,
        raising=False,
    )


@pytest.mark.parametrize(
    "task_class",
    (
        IThorKitchenPackProduceIntoBox,
        IThorKitchenAppleIntoBox,
        IThorKitchenCupIntoBox,
        IThorKitchenStatueIntoBox,
        IThorKitchenCoffeeDrinkToTrash,
        IThorKitchenBowlToSink,
        IThorKitchenCupToSink,
    ),
)
def test_kitchen_success_hold_advances_only_on_post_action(monkeypatch, task_class):
    monkeypatch.setattr(IThorKitchenEnv, "_post_action", lambda _self, action: action)
    task = object.__new__(task_class)
    task._success_hold_steps = 0
    task.success_hold_seconds = 1.0
    task.control_freq = 4
    state = {"success": True}
    task._raw_success = lambda: state["success"]

    assert not task._check_success()
    assert not task._check_success()
    assert task._success_hold_steps == 0

    for _ in range(3):
        task._post_action(None)
        assert not task._check_success()
        assert not task._check_success()
    assert task._success_hold_steps == 3

    task._post_action(None)
    assert task._check_success()
    assert task._check_success()
    assert task._success_hold_steps == 4

    state["success"] = False
    task._post_action(None)
    assert task._success_hold_steps == 0
    assert not task._check_success()


def test_stand_soap_bottle_requires_uninterrupted_hold(monkeypatch):
    task_class = IThorBathroomStandSoapBottleUpright
    monkeypatch.setattr(
        task_class.__mro__[2],
        "_post_action",
        lambda _self, action: action,
        raising=False,
    )
    task = object.__new__(task_class)
    task._success_hold = 0
    state = {"success": True}
    task._raw_success = lambda: state["success"]

    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    state["success"] = False
    task._post_action(None)
    assert task._success_hold == 0

    state["success"] = True
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()


def test_lift_scrub_brush_requires_low_grasp_and_uninterrupted_hold(monkeypatch):
    task_class = IThorBathroomLiftScrubBrush
    monkeypatch.setattr(
        task_class.__mro__[2],
        "_post_action",
        lambda _self, action: action,
        raising=False,
    )
    task = object.__new__(task_class)
    task.source_fixture = object()
    task._initial_source_z = 1.0
    task._grasped_near_source = False
    task._success_hold = 0
    state = {"grasped": False, "height": 1.20}
    task._g1_hand_grasping_fixture = lambda _fixture: state["grasped"]
    task._source_root_z = lambda: state["height"]

    # Viewer-style displacement without a real hand grasp cannot succeed.
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert not task._check_success()
    state["grasped"] = True
    task._post_action(None)
    assert not task._grasped_near_source

    state["height"] = 1.05
    task._post_action(None)
    assert task._grasped_near_source
    state["height"] = 1.31
    for _ in range(task.HOLD_STEPS - 1):
        task._post_action(None)
    assert not task._check_success()
    state["grasped"] = False
    task._post_action(None)
    assert task._success_hold == 0

    state["grasped"] = True
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()


def test_holodeck_tissue_box_success_uses_height_only(monkeypatch):
    _stub_task_parent_post_action(monkeypatch, HolodeckTakeTissueBoxFromOpenDrawer)
    task = object.__new__(HolodeckTakeTissueBoxFromOpenDrawer)
    task._success_hold = 0
    task._tissue_box_removed_after_open_state = False
    task._invalid_pre_open_grasp = False
    task.target_semantic_joint = SimpleNamespace(fraction=lambda _sim: 1.0)
    task.sim = object()
    task.control_freq = 20
    task._tissue_box_initial_z = 1.0
    state = {"grasped": True, "inside": True, "height": 1.0}
    task.g1_hand_grasping_object = lambda _name: state["grasped"]
    task._tissue_box_in_drawer = lambda: state["inside"]
    task._tissue_box_root_z = lambda: state["height"]

    task._post_action(None)
    assert not task._check_success()

    state.update(inside=False, height=1.30)
    task._post_action(None)
    assert not task._check_success()

    state["height"] = 1.31
    for _ in range(task.hold_steps - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()


def test_open_drawer_tissue_composite_matches_atomic_height_predicate():
    task = object.__new__(HolodeckOpenDrawerThenTakeTissueBox)
    task._tissue_box_removed_after_open_state = True
    task._tissue_box_removed = False
    task._invalid_pre_open_lift = False
    task.g1_hand_grasping_object = lambda _name: (_ for _ in ()).throw(
        AssertionError("composite predicate must not inspect grasp state")
    )

    assert task._tissue_box_removed_after_open()
    assert task._tissue_box_removed


def test_open_drawer_tissue_composite_rejects_lift_before_second_stage():
    task = object.__new__(HolodeckOpenDrawerThenTakeTissueBox)
    task._tissue_box_removed_after_open_state = True
    task._tissue_box_removed = False

    task._begin_tissue_box_removal_stage()

    assert task._invalid_pre_open_lift
    assert not task._tissue_box_removed_after_open()


def test_tissue_box_close_stage_does_not_recheck_hand_distance():
    task = object.__new__(HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer)
    task.target_semantic_joint = SimpleNamespace(fraction=lambda _sim: 0.0)
    task.close_drawer_spec = {"success_close_fraction": 0.20}
    task._tissue_box_placed_before_close = True
    task._drawer_open_after_placement = True
    task.object_on_region = lambda *_args, **_kwargs: True
    task.g1_hand_grasping_object = lambda _name: False
    task._tissue_box_released_on_desk = lambda: False

    assert task._drawer_closed_after_placement()


def test_tissue_box_following_is_limited_to_open_drawer_stage():
    task = object.__new__(HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer)
    task._tissue_box_removed_after_open_state = False
    task._tissue_box_removed = False
    task._atomic_tasks = tuple(
        SimpleNamespace(name=name)
        for name in (
            "open_drawer",
            "take_tissue_box",
            "place_tissue_box_on_desk",
            "close_drawer",
        )
    )

    task._stage_index = 0
    assert task._tissue_box_following_allowed()

    for stage_index in (1, 2, 3):
        task._stage_index = stage_index
        assert not task._tissue_box_following_allowed()

    task._stage_index = 0
    task._tissue_box_removed = True
    assert not task._tissue_box_following_allowed()


def test_composite_tissue_removal_does_not_follow_after_open_stage():
    task = object.__new__(HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer)
    task._tissue_box_grasped_after_open = False
    task._tissue_box_removed = False
    task._invalid_pre_open_grasp = False
    task._atomic_tasks = tuple(
        SimpleNamespace(name=name)
        for name in (
            "open_drawer",
            "take_tissue_box",
            "place_tissue_box_on_desk",
            "close_drawer",
        )
    )
    task.target_semantic_joint = SimpleNamespace(fraction=lambda _sim: 0.0)
    task.sim = object()
    task.g1_hand_grasping_object = lambda _name: False
    called = []
    task._follow_tissue_box_with_drawer = lambda: called.append(True)

    for stage_index in (1, 2, 3):
        task._stage_index = stage_index
        assert not task._tissue_box_removed_after_open()
    assert not called


def test_tissue_box_placement_accepts_released_hand_near_box():
    task = object.__new__(HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer)
    task._tissue_box_removed_before_placement = True
    task._tissue_box_removed = True
    task._tissue_box_grasped_for_placement = True
    task.g1_hand_grasping_object = lambda _name: False
    task.object_on_region = lambda *_args, **_kwargs: True

    assert task._tissue_box_released_on_desk()


def test_cup_placement_uses_only_the_target_region(monkeypatch):
    _stub_task_parent_post_action(monkeypatch, HolodeckPlaceCupOnPlacemat)
    task = object.__new__(HolodeckPlaceCupOnPlacemat)
    task._success_hold = 0
    task.control_freq = 20
    task._cup_on_placemat = lambda: True

    task._post_action(None)
    assert task._success_hold == 1


def test_composite_reports_final_success_in_transition_step(monkeypatch):
    monkeypatch.setattr(
        HolodeckEnv,
        "_post_action",
        lambda _self, _action: (0.0, False, {}),
    )
    task = object.__new__(HolodeckMultiHopComposite)
    task._atomic_tasks = (
        HolodeckAtomicTask("finish", lambda _env: True, hold_steps=2),
    )
    task._stage_index = 0
    task._stage_hold_steps = 0

    _, _, first_info = task._post_action(None)
    assert not first_info["task_success"]

    reward, _, final_info = task._post_action(None)
    assert reward == 1.0
    assert final_info["task_success"] is True
    assert final_info["success"] is True
    assert final_info["completed_atomic_tasks"] == ["finish"]


def test_holodeck_composite_reset_clears_all_success_progress():
    task = object.__new__(HolodeckApproachDeskThenOpenLaptop)
    task._stage_index = 2
    task._stage_hold_steps = 7
    task._in_range_steps = 10
    task._success_hold = 10
    task._success_hold_steps = 10

    task._reset_composite_progress()

    assert task.stage_index == 0
    assert task._stage_hold_steps == 0
    assert task._in_range_steps == 0
    assert task._success_hold == 0
    assert task._success_hold_steps == 0


def test_articulation_reset_clears_success_hold():
    class Parent:
        def _reset_internal(self):
            self.parent_reset = True

    class StubArticulation(RoomOpenArticulation, Parent):
        pass

    task = StubArticulation()
    task._success_hold = 10

    task._reset_internal()

    assert task.parent_reset
    assert task._success_hold == 0


def test_lift_basket_preserves_legacy_asset_and_requires_low_grasp(monkeypatch):
    assert IThorBedroomLiftBasket.TARGET_XML.endswith(
        "/objects/lightwheel/basket/Basket039/model.xml"
    )
    assert IThorBedroomLiftBasket.TARGET_SCALE == pytest.approx(
        bedroom_assets.LIFT_BASKET_SCALE
    )
    assert IThorBedroomLiftBasket.MIN_LIFT_DELTA == pytest.approx(0.30)
    _stub_task_parent_post_action(monkeypatch, IThorBedroomLiftBasket)
    task = object.__new__(IThorBedroomLiftBasket)
    task._lift_initial_z = 0.60
    task._grasped_at_source_height = False
    task._bedroom_lift_hold = 0
    state = {"height": 0.91, "grasped": True}
    task._target_root_z = lambda: state["height"]
    task.g1_hand_grasping_object = lambda _name: state["grasped"]

    # Grasping only after the basket is already high cannot skip the source
    # grasp stage.
    for _ in range(task.HOLD_STEPS + 1):
        task._post_action(None)
    assert not task._check_success()

    state["height"] = 0.65
    task._post_action(None)
    assert task._grasped_at_source_height
    state["height"] = 0.90
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()

    state["grasped"] = False
    task._post_action(None)
    assert not task._check_success()


def test_floor_basket_lift_uses_clear_329_pose_and_requires_real_grasp(monkeypatch):
    assert IThorBedroomLiftBasketFromFloor.SUPPORTED_SCENE_IDS == (
        302,
        306,
        310,
        323,
        329,
    )
    expected = {
        302: ([-0.25, -0.50], [-0.25, -1.05], [-0.25, -1.20], 1.5707963268),
        306: ([1.20, 0.00], [1.20, 0.55], [1.20, 0.70], -1.5707963268),
        310: ([-1.00, 0.15], [-1.00, 0.70], [-1.00, 0.85], -1.5707963268),
        323: ([0.00, -1.20], [0.00, -0.65], [0.00, -0.50], -1.5707963268),
        329: ([1.20, 0.00], [1.20, 0.55], [1.20, 0.70], -1.5707963268),
    }
    for scene_id, (basket_xy, robot_xy, staging_xy, yaw) in expected.items():
        scene_spec = IThorBedroomLiftBasketFromFloor.SCENE_SPECS[f"FloorPlan{scene_id}"]
        assert scene_spec["basket_xy"] == pytest.approx(basket_xy)
        assert scene_spec["stool_xy"] == pytest.approx(basket_xy)
        assert scene_spec["robot_pose"]["xy"] == pytest.approx(robot_xy)
        assert scene_spec["robot_pose"]["yaw"] == pytest.approx(yaw)
        assert scene_spec["staging_robot_pose"]["xy"] == pytest.approx(staging_xy)
        assert scene_spec["staging_robot_pose"]["yaw"] == pytest.approx(yaw)
        assert scene_spec["source_horizontal_distance"] == pytest.approx(0.55)
        assert scene_spec["remove_fixtures"] == []
    assert IThorBedroomLiftBasketFromFloor.TARGET_XML == (
        IThorBedroomLiftBasket.TARGET_XML
    )
    assert IThorBedroomLiftBasketFromFloor.TARGET_SCALE == pytest.approx(
        IThorBedroomLiftBasket.TARGET_SCALE
    )
    spec = IThorBedroomLiftBasketFromFloor.SCENE_SPECS["FloorPlan329"]
    assert spec["basket_xy"] == pytest.approx([1.20, 0.00])
    assert spec["stool_model"] == IThorBedroomLiftBasketFromFloor.STOOL_MODEL
    assert spec["stool_model"].endswith("/objects/lightwheel/stool/Stool005/model.xml")
    assert spec["stool_scale"] == pytest.approx([1.15, 0.90, 0.30])
    assert spec["stool_xy"] == pytest.approx([1.20, 0.00])
    assert spec["stool_top_half_xy"] == pytest.approx([0.176, 0.137])
    assert spec["stool_floor_probe_below"] == pytest.approx(0.01)
    assert spec["robot_pose"]["xy"] == pytest.approx([1.20, 0.55])
    assert spec["staging_robot_pose"]["xy"] == pytest.approx([1.20, 0.70])
    assert spec["source_horizontal_distance"] == pytest.approx(0.55)
    assert spec["remove_fixtures"] == []

    _stub_task_parent_post_action(monkeypatch, IThorBedroomLiftBasketFromFloor)
    task = object.__new__(IThorBedroomLiftBasketFromFloor)
    task._lift_initial_z = 0.34
    task._grasped_at_source_height = False
    task._bedroom_lift_hold = 0
    state = {"height": 0.65, "grasped": False}
    task._target_root_z = lambda: state["height"]
    task.g1_hand_grasping_object = lambda _name: state["grasped"]

    # Viewer dragging alone cannot satisfy a lift task because the basket was
    # never grasped by a closed G1 hand at its reset height.
    for _ in range(task.HOLD_STEPS + 1):
        task._post_action(None)
    assert not task._check_success()

    state.update(height=0.37, grasped=True)
    task._post_action(None)
    state["height"] = 0.67
    for _ in range(task.HOLD_STEPS):
        task._post_action(None)
    assert task._check_success()

    state["grasped"] = False
    task._post_action(None)
    assert not task._check_success()
