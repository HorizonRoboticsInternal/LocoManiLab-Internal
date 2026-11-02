"""Restore evaluation predicates only for feedback items 12, 13, 15 and 17."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.holodeck.tasks.multi_hop.navigation import (
    HolodeckApproachDeskIceCubePlaceCupThenPickCup,
    HolodeckIceCubeThenPlaceCup,
)
from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat,
    HolodeckTakeTissueBoxThenPlaceCup,
)
from robocasa.environments.holodeck.tasks.place_cup_on_placemat import (
    HolodeckPlaceCupOnPlacemat as PlaceCup,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.apple_on_plate import (
    IThorKitchenAppleOnPlate,
)
from robocasa.environments.ithor_kitchen.tasks.composite.apple_banana_plate_swap import (
    IThorKitchenAppleBananaPlateSwap as Swap,
)
from robocasa.environments.ithor_kitchen.tasks.composite.open_drawer_take_apple import (
    IThorKitchenOpenDrawerTakeApple as TakeApple,
)
from robocasa.environments.ithor_living_room.tasks.composite.pack_candle_statue import (
    IThorLivingRoomPackCandleStatueIntoBasket as Pack,
)


def _bbox(half):
    return np.array(list(product((-1, 1), repeat=3))) * half


@pytest.mark.parametrize("scene", ("FloorPlan209", "FloorPlan202"))
def test_pack_candle_restores_contact_and_distance_only_in_scene209(scene):
    task = object.__new__(Pack)
    task.scene_name = scene
    task.success_xy_tolerance = 0.18
    task.objects = {
        "basket": SimpleNamespace(get_bbox_points=lambda: _bbox([0.3] * 3)),
        "candle": object(),
        "statue": object(),
    }
    task.obj_body_id = {"basket": 0, "candle": 1, "statue": 2}
    positions = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.1], [0.1, 0.0, 0.1]])
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            body_xpos=positions,
            body_xmat=np.tile(np.eye(3).ravel(), (3, 1)),
        )
    )
    task.object_bbox_points = lambda name: positions[task.obj_body_id[name]] + _bbox(
        [0.03] * 3
    )
    contact = {"value": True}
    task.check_contact = lambda *_args: contact["value"]

    assert task._item_is_packed("candle")
    assert task._item_is_packed("statue")
    contact["value"] = False
    assert task._item_is_packed("candle") == (scene != "FloorPlan209")
    assert not task._item_is_packed("statue")
    contact["value"] = True
    positions[1, 0] = 0.20  # Inside the basket bbox, beyond evaluation distance.
    assert task._item_is_packed("candle") == (scene != "FloorPlan209")
    positions[1, 0] = 0.40
    assert not task._item_is_packed("candle")


@pytest.mark.parametrize("scene", ("FloorPlan13", "FloorPlan6"))
def test_swap_restores_final_state_without_off_plate_check_only_in_scene13(
    monkeypatch,
    scene,
):
    task = object.__new__(Swap)
    task.scene_name = scene
    state = {"table": True, "banana": True, "apple_on_plate": True}
    task._apple_on_table = lambda: state["table"]
    task._banana_on_plate = lambda: state["banana"]
    monkeypatch.setattr(
        IThorKitchenAppleOnPlate,
        "_raw_success",
        lambda self: state["apple_on_plate"],
    )

    assert task._raw_success() == (scene == "FloorPlan13")
    state["apple_on_plate"] = False
    assert task._raw_success()
    for required in ("table", "banana"):
        state[required] = False
        assert not task._raw_success()
        state[required] = True


def test_take_apple_restores_object_released_only_in_scene13():
    task = object.__new__(TakeApple)
    task.scene_name = "FloorPlan13"
    task.sim = object()
    task._drawer_opened = True
    task._apple_on_table = lambda: True
    state = {"released": True, "fraction": task.SUCCESS_OPEN_FRACTION}
    task.target_semantic_joint = SimpleNamespace(
        fraction=lambda _sim: state["fraction"],
    )
    task.object_released = lambda name: name == "apple" and state["released"]
    task.g1_hand_object_contact = lambda *_args: pytest.fail(
        "scene13 must use object_released instead of querying hand contact directly"
    )

    assert task._task_success_condition()
    state["released"] = False
    assert not task._task_success_condition()
    state["released"] = True
    task._drawer_opened = False
    assert not task._task_success_condition()
    task._drawer_opened = True
    state["fraction"] -= 0.01
    assert not task._task_success_condition()
    state["fraction"] = task.SUCCESS_OPEN_FRACTION
    task._apple_on_table = lambda: False
    assert not task._task_success_condition()


@pytest.mark.parametrize(
    ("cls", "scene", "contact_required"),
    (
        (PlaceCup, "train_117", False),
        (PlaceCup, "train_1018", True),
        (HolodeckIceCubeThenPlaceCup, "train_117", True),
        (HolodeckApproachDeskIceCubePlaceCupThenPickCup, "train_117", True),
        (HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat, "train_117", True),
        (HolodeckTakeTissueBoxThenPlaceCup, "train_117", True),
    ),
)
def test_placemat_skips_contact_only_for_scene117_atomic_task(
    cls,
    scene,
    contact_required,
):
    task = object.__new__(cls)
    task.scene_name = scene
    task.objects = {"placemat": SimpleNamespace(root_body="placemat")}
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            get_body_xpos=lambda _body: np.zeros(3),
            get_body_xmat=lambda _body: np.eye(3).ravel(),
        )
    )
    cup_offset = np.zeros(3)
    task.object_bbox_points = lambda name: (
        _bbox([0.2, 0.15, 0.005])
        if name == "placemat"
        else _bbox([0.03, 0.03, 0.05]) + cup_offset
    )
    contact = {"value": False, "calls": 0}

    def check_contact():
        contact["calls"] += 1
        return contact["value"]

    task._cup_contacts_placemat = check_contact
    assert task._cup_on_placemat() == (not contact_required)
    assert contact["calls"] == int(contact_required)
    contact["value"] = True
    assert task._cup_on_placemat()
    cup_offset[0] = 0.5
    assert not task._cup_on_placemat()


@pytest.mark.parametrize("scene", ("train_117", "train_1018"))
def test_placemat_metadata_matches_scene_specific_success(monkeypatch, scene):
    monkeypatch.setattr(HolodeckEnv, "get_ep_meta", lambda self: {})
    task = object.__new__(PlaceCup)
    task.scene_name = scene
    definition = task.get_ep_meta()["success_definition"]
    assert ("contact the placemat" in definition) == (scene != "train_117")
    assert "for one second" in definition
