"""Focused asset and stage-order tests for the TeleLab cup-carry task."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

from robocasa.environments.telelab.carry_cup_from_shelf_to_desk import (
    TeleLabCarryCupFromShelfToDesk,
)
from robocasa.models.fixtures import Shelf
from robocasa.models.objects.objects import MJCFObject


_CUP_ASSET_DIR = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "models"
    / "assets"
    / "objects"
    / "manual"
    / "cup"
)
_SHELF_XML = (
    Path(__file__).resolve().parents[1]
    / "robocasa"
    / "models"
    / "assets"
    / "objects"
    / "manual"
    / "shelf"
    / "shelf.xml"
)


def test_cup_asset_has_no_handle_and_has_graspable_collision():
    root = ET.parse(_CUP_ASSET_DIR / "cup.xml").getroot()
    visual = root.find(".//geom[@name='cup_visual']")
    base = root.find(".//geom[@name='cup_base']")
    body = root.find(".//geom[@name='cup_body']")

    mesh = root.find("./asset/mesh")
    assert mesh is not None
    assert mesh.get("file") == "cup_hollow.obj"
    assert (_CUP_ASSET_DIR / mesh.get("file")).is_file()
    assert visual is not None
    assert visual.get("type") == "mesh"
    assert visual.get("mesh") == "cup_mesh"
    assert base.get("size") == "0.035 0.005"
    assert body.get("size") == "0.035 0.060"

    vertices = []
    for line in (_CUP_ASSET_DIR / mesh.get("file")).read_text().splitlines():
        if line.startswith("v "):
            vertices.append(tuple(map(float, line.split()[1:4])))
    bounds = [
        max(vertex[axis] for vertex in vertices)
        - min(vertex[axis] for vertex in vertices)
        for axis in range(3)
    ]
    assert bounds == pytest.approx((0.07, 0.07, 0.12))
    inner_radii = [
        (vertex[0] ** 2 + vertex[1] ** 2) ** 0.5
        for vertex in vertices
        if vertex[2] == pytest.approx(0.01)
        and (vertex[0] ** 2 + vertex[1] ** 2) ** 0.5 > 0.02
    ]
    assert min(inner_radii) == pytest.approx(0.029, abs=1e-6)


def test_cup_mjcf_exposes_contact_geometries():
    cup = MJCFObject(
        name="cup",
        mjcf_path=str(_CUP_ASSET_DIR / "cup.xml"),
        scale=1.0,
    )
    assert set(cup.contact_geoms) == {
        f"{cup.naming_prefix}cup_base",
        f"{cup.naming_prefix}cup_body",
    }


def test_task_uses_raw_merged_cup_geom_name():
    assert TeleLabCarryCupFromShelfToDesk.CUP_BOTTOM_GEOM == "cup_base"


def test_task_declares_pick_walk_place_in_that_order():
    assert TeleLabCarryCupFromShelfToDesk.COMPOSITE_KIND == "pick_navigate_place"
    assert TeleLabCarryCupFromShelfToDesk.COMPOSITE_ORDERING == (
        "pick_cup_from_shelf",
        "walk_to_desk",
        "place_cup_on_desk",
    )


def test_robot_starts_at_the_shelf_side_facing_the_cup():
    task = TeleLabCarryCupFromShelfToDesk
    source_direction = np.asarray(task.SOURCE_XY) - np.asarray(
        task.ROBOT_BASE_POS[:2]
    )
    expected_yaw = math.atan2(source_direction[1], source_direction[0])
    assert task.ROBOT_BASE_YAW == pytest.approx(expected_yaw)


def test_lift_stage_does_not_require_a_grasp():
    task = object.__new__(TeleLabCarryCupFromShelfToDesk)
    task._source_cup_root_z = 0.80
    task._cup_lifted_once = False
    task._cup_root_z = lambda: 0.90

    assert task._cup_is_picked()


def test_lift_stage_remains_complete_after_cup_drops():
    task = object.__new__(TeleLabCarryCupFromShelfToDesk)
    task._source_cup_root_z = 0.80
    task._cup_lifted_once = False
    cup_root_z = [0.90]
    task._cup_root_z = lambda: cup_root_z[0]

    assert task._cup_is_picked()
    cup_root_z[0] = 0.80
    assert task._cup_is_picked()


def test_navigation_stage_does_not_require_carry_distance():
    task = object.__new__(TeleLabCarryCupFromShelfToDesk)
    task._desk_approach_distance = lambda: 0.35

    assert task._cup_carried_to_desk()


def test_task_aligns_source_support_with_scanned_second_shelf():
    shelf = Shelf(xml=str(_SHELF_XML), name="telelab_shelf")
    task = object.__new__(TeleLabCarryCupFromShelfToDesk)
    task.fixtures = {"shelf": shelf}
    geom_name = f"{shelf.naming_prefix}{task.SOURCE_SHELF_BOARD}"
    geom = shelf.get_obj().find(f".//geom[@name='{geom_name}']")
    task._calibrate_source_shelf_collision()

    position = np.fromstring(geom.get("pos"), sep=" ")
    half_size = np.fromstring(geom.get("size"), sep=" ")
    assert position[1] + half_size[1] == pytest.approx(
        task.SOURCE_SHELF_SURFACE_HEIGHT
    )


def test_desk_goal_requires_bottom_contact_and_stability_only():
    task = object.__new__(TeleLabCarryCupFromShelfToDesk)
    task._cup_bottom_geom_name = "cup_base"
    task._target_desk_geom_name = "desk_top"
    contacts = set()
    task.check_contact = lambda geom_1, geom_2: (geom_1, geom_2) in contacts
    task._cup_bottom_supported_by_desk = lambda: False
    task._cup_is_stable = lambda: True

    assert not task._cup_on_desk()
    contacts.add(("cup_base", "desk_top"))
    assert task._cup_on_desk()

    task._cup_is_stable = lambda: False
    assert not task._cup_on_desk()


def test_desk_goal_accepts_geometric_support_without_contact_callback():
    task = object.__new__(TeleLabCarryCupFromShelfToDesk)
    task._cup_bottom_geom_name = "cup_base"
    task._target_desk_geom_name = "desk_top"
    task.check_contact = lambda geom_1, geom_2: False
    task._cup_bottom_supported_by_desk = lambda: True
    task._cup_is_stable = lambda: True

    assert task._cup_on_desk()


def _progress_only_task(hold_steps=20):
    task = object.__new__(TeleLabCarryCupFromShelfToDesk)
    task._stage = 0
    task._stage_hold = 0
    task._stage_hold_steps = hold_steps
    task._picked = False
    task._carried = False
    task._placed = False
    task._cup_is_picked = lambda: task._picked
    task._cup_carried_to_desk = lambda: task._carried
    task._final_stage_goal = lambda: task._placed
    task._robot_base_xy = lambda: np.array((0.0, 0.0))
    return task


def test_task_requires_pick_then_walk_before_the_placement_hold():
    task = _progress_only_task(hold_steps=2)

    task._placed = True
    task._advance_task_progress()
    assert task._stage == 0

    task._picked = True
    task._advance_task_progress()
    assert task._stage == 1

    task._advance_task_progress()
    assert task._stage == 1
    task._carried = True
    task._advance_task_progress()
    assert task._stage == 2

    task._advance_task_progress()
    assert task._stage == 2
    task._advance_task_progress()
    assert task._stage == 3
    assert task._check_success()


def test_navigation_stage_does_not_require_continuous_grasp():
    task = _progress_only_task()
    task._picked = True
    task._advance_task_progress()
    assert task._stage == 1

    task._picked = False
    task._carried = True
    task._advance_task_progress()
    assert task._stage == 2
