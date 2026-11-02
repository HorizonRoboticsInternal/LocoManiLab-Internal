"""Static regression checks for the requested Holodeck multi-hop tasks."""

import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robocasa.environments.holodeck.holodeck import (
    HolodeckEnv,
    REGISTERED_HOLODECK_ENVS,
)
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckMultiHopComposite,
)
from robocasa.environments.holodeck.tasks.multi_hop.navigation import (
    HolodeckApproachDeskThenPullChair,
    HolodeckIceCubeThenPlaceCup,
    HolodeckApproachDeskIceCubePlaceCupThenPickCup,
    HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan,
    HolodeckApproachDeskThenOpenLaptopThenApproachWindow,
    HolodeckApproachTrashCanThenApproachDeskThenPullChair,
    HolodeckApproachTrashCanThenThrowBottle,
    HolodeckApproachTrashCanThenThrowBottleThenApproachDesk,
    HolodeckApproachWindowThenDesk,
    HolodeckApproachWindowThenDeskThenPickCup,
)
from robocasa.environments.holodeck.tasks.ice_cube_in_glass import (
    HolodeckIceCubeInGlass,
)
from robocasa.environments.holodeck.tasks.move_cup import HolodeckMoveCupToCorner
from robocasa.environments.holodeck.tasks.pick_cup_from_desk import (
    HolodeckPickCupFromDesk,
)
from robocasa.environments.holodeck.tasks.pick_book_from_dresser import (
    HolodeckPickBookFromDresser,
)
from robocasa.environments.holodeck.tasks.place_box_on_floor import (
    HolodeckPlaceBoxOnFloor,
)
from robocasa.environments.holodeck.tasks.approach_desk import HolodeckApproachDesk
from robocasa.environments.holodeck.tasks.pull_chair_from_desk import (
    HolodeckPullChairFromDesk,
)
from robocasa.environments.holodeck.tasks.throw_bottle_into_trash_can import (
    HolodeckThrowBottleIntoTrashCan,
)
from robocasa.environments.holodeck.tasks.articulation import HolodeckCloseDrawer
from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckApproachDeskThenOpenLaptop,
    HolodeckApproachDeskPullChairThenOpenLaptop,
    HolodeckApproachDeskThenIceCubeInGlass,
    HolodeckCloseJarThenApproachDeskThenApproachTrashCan,
    HolodeckApproachDeskThenCloseLaptopMoveCup,
    HolodeckMoveCupToCenterThenOpenLaptop,
    HolodeckOpenDrawerTakeTissueBoxToDesk,
    HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
    HolodeckOpenDrawerThenTakeTissueBox,
    HolodeckPickBookFromSourceToDesk,
    HolodeckPickBookFromSourceToDeskThenApproachTrashCan,
    HolodeckRemoveFolderFromShelfToDesk,
    HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop,
    HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop,
    HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan,
    HolodeckTakeTissueBoxThenPlaceCup,
    HolodeckApproachTrashCanThenApproachDeskThenOpenJar,
    HolodeckApproachTrashCanThenPickBoxFromFloor,
    HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk,
    HolodeckCloseLaptopThenApproachTrashCanThenPickBox,
    HolodeckIceCubeThenApproachWindowThenPickBox,
    HolodeckApproachWindowThenPickBox,
    HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer,
    HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat,
    HolodeckPlaceBoxOnFloorThenPickBook,
    HolodeckPullChairThenApproachTrashCanThenPickBox,
    HolodeckSugarCubeThenCloseJar,
    HolodeckOpenJarAddSugarThenCloseJar,
)
from robocasa.environments.holodeck.tasks.multi_hop.desk_workflows import (
    HolodeckApproachDeskThenCloseDrawer,
    HolodeckApproachDeskThenMoveCupToCenter,
    HolodeckApproachDeskThenOpenJar,
    HolodeckApproachDeskThenPickCup,
)
from robocasa.environments.holodeck.tasks.mug_lid import (
    HolodeckCloseMugLid,
    HolodeckOpenMugLid,
)
from robocasa.environments.holodeck.tasks.sugar_cube_in_jar import (
    HolodeckSugarCubeInJar,
)
from robocasa.environments.holodeck.tasks.take_tissue_box_from_open_drawer import (
    _HolodeckDrawerTissueBoxMixin,
)


TISSUE_BOX_WORKFLOWS = (
    HolodeckTakeTissueBoxThenPlaceCup,
    HolodeckOpenDrawerThenTakeTissueBox,
    HolodeckOpenDrawerTakeTissueBoxToDesk,
    HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
    HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan,
)


HOLODECK_TABLETOP_TASKS = (
    HolodeckOpenMugLid,
    HolodeckCloseMugLid,
    HolodeckIceCubeInGlass,
    HolodeckSugarCubeInJar,
)


REQUESTED_TASKS = (
    HolodeckApproachDeskThenOpenLaptop,
    HolodeckApproachWindowThenDeskThenPickCup,
    HolodeckApproachDeskPullChairThenOpenLaptop,
    HolodeckMoveCupToCenterThenOpenLaptop,
    HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop,
    HolodeckTakeTissueBoxThenPlaceCup,
    HolodeckOpenDrawerThenTakeTissueBox,
    HolodeckOpenDrawerTakeTissueBoxToDesk,
    HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
    HolodeckApproachTrashCanThenThrowBottle,
    HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop,
    HolodeckApproachWindowThenDesk,
    HolodeckApproachTrashCanThenThrowBottleThenApproachDesk,
    HolodeckApproachDeskThenOpenLaptopThenApproachWindow,
    HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan,
    HolodeckApproachTrashCanThenApproachDeskThenPullChair,
    HolodeckPickBookFromSourceToDeskThenApproachTrashCan,
    HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan,
    HolodeckApproachDeskThenIceCubeInGlass,
    HolodeckCloseJarThenApproachDeskThenApproachTrashCan,
    HolodeckApproachTrashCanThenApproachDeskThenOpenJar,
    HolodeckApproachTrashCanThenPickBoxFromFloor,
    HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk,
    HolodeckIceCubeThenApproachWindowThenPickBox,
    HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer,
    HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat,
    HolodeckPlaceBoxOnFloorThenPickBook,
    HolodeckSugarCubeThenCloseJar,
    HolodeckOpenJarAddSugarThenCloseJar,
)


NEW_SCENE_COMPOSITES = (
    HolodeckApproachDeskThenOpenLaptop,
    HolodeckOpenDrawerThenTakeTissueBox,
    HolodeckOpenDrawerTakeTissueBoxToDesk,
    HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
    HolodeckSugarCubeThenCloseJar,
    HolodeckOpenJarAddSugarThenCloseJar,
    HolodeckApproachDeskThenCloseLaptopMoveCup,
    HolodeckRemoveFolderFromShelfToDesk,
    HolodeckIceCubeThenPlaceCup,
    HolodeckApproachDeskThenPullChair,
    HolodeckApproachWindowThenDeskThenPickCup,
    HolodeckApproachDeskPullChairThenOpenLaptop,
    HolodeckMoveCupToCenterThenOpenLaptop,
    HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop,
)


TRAIN_117_COMPOSITES = (
    (
        HolodeckIceCubeThenApproachWindowThenPickBox,
        ("put_ice_in_cup", "approach_rightmost_window", "pick_box_from_floor"),
    ),
    (
        HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer,
        ("approach_desk", "place_box_on_floor", "close_drawer"),
    ),
    (
        HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat,
        ("pick_cup", "approach_desk", "place_cup_on_placemat"),
    ),
    (
        HolodeckPlaceBoxOnFloorThenPickBook,
        ("place_box_on_floor", "place_box_on_desk"),
    ),
)


NEW_TRAIN_117_DESK_COMPOSITES = (
    (HolodeckApproachDeskThenPickCup, ("approach_desk", "pick_cup")),
    (
        HolodeckApproachDeskThenMoveCupToCenter,
        ("approach_desk", "move_cup_to_center"),
    ),
    (HolodeckApproachDeskThenCloseDrawer, ("approach_desk", "close_drawer")),
    (HolodeckApproachDeskThenOpenJar, ("approach_desk", "open_jar")),
)


REQUESTED_SCENE_EXPANSIONS = tuple(
    task_cls for task_cls, _stage_names in TRAIN_117_COMPOSITES
) + tuple(task_cls for task_cls, _stage_names in NEW_TRAIN_117_DESK_COMPOSITES)


@pytest.mark.parametrize("task_cls", REQUESTED_TASKS)
def test_requested_workflows_are_registered_and_concrete(task_cls):
    assert task_cls.__name__ in REGISTERED_HOLODECK_ENVS
    assert task_cls.ABSTRACT is False
    assert task_cls.TASK_FAMILY == "composite"
    assert len(task_cls.ATOMIC_TASKS) >= 2


@pytest.mark.parametrize("task_cls", NEW_SCENE_COMPOSITES)
def test_requested_holodeck_composites_support_new_scenes(task_cls):
    expected = (117, 1018, 1197, 1664, 2025)
    if task_cls is HolodeckRemoveFolderFromShelfToDesk:
        expected = (1018, 1197, 2025)
    elif task_cls is HolodeckApproachWindowThenDeskThenPickCup:
        expected = (117, 1018, 1664, 2025)
    assert task_cls.SUPPORTED_SCENE_IDS == expected


@pytest.mark.parametrize("task_cls", REQUESTED_SCENE_EXPANSIONS)
def test_requested_scene_expansions_support_all_calibrated_scenes(task_cls):
    expected = (117, 1018, 1197, 1664, 2025)
    if task_cls in (
        HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer,
        HolodeckApproachDeskThenCloseDrawer,
    ):
        expected = (117, 1197)
    elif task_cls is HolodeckIceCubeThenApproachWindowThenPickBox:
        expected = (117, 1018, 1664, 2025)
    elif task_cls is HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat:
        expected = (117, 1018, 1664, 2025)
    assert task_cls.SUPPORTED_SCENE_IDS == expected


def test_requested_composite_specs_are_scene_local():
    current_specs = yaml.safe_load(
        open("robocasa/models/assets/scenes/holodeck_task_specs.yaml")
    )["scenes"]
    previous_specs = yaml.safe_load(
        subprocess.check_output(
            [
                "git",
                "show",
                "HEAD:robocasa/models/assets/scenes/holodeck_task_specs.yaml",
            ]
        )
    )["scenes"]
    requested_added = {
        "approach_window_then_desk_then_pick_cup": set(),
        "approach_desk_pull_chair_then_open_laptop": {
            "train_1197",
            "train_1664",
            "train_2025",
        },
        "move_cup_to_center_then_open_laptop": {
            "train_1197",
            "train_1664",
            "train_2025",
        },
        "remove_folder_from_shelf_to_desk_then_open_laptop": {
            "train_1197",
            "train_2025",
        },
        "pick_book_to_desk": set(),
        "open_jar_add_sugar_then_close_jar": {"train_1197", "train_1664", "train_2025"},
        "approach_desk_then_close_laptop_move_cup": {
            "train_1197",
            "train_1664",
            "train_2025",
        },
        "remove_folder_from_shelf_to_desk": {"train_1197", "train_2025"},
        "ice_cube_then_place_cup": {"train_1197", "train_1664", "train_2025"},
        "approach_desk_then_pull_chair": {"train_1197", "train_1664", "train_2025"},
        "place_box_on_desk": {
            "train_1018",
            "train_1197",
            "train_1664",
            "train_2025",
        },
    }
    for task_name, expected_scenes in requested_added.items():
        assert {
            scene_name
            for scene_name, scene in current_specs.items()
            if task_name in scene
            and task_name not in previous_specs.get(scene_name, {})
        } == expected_scenes


def test_requested_scene_expansion_specs_are_scene_local():
    current_specs = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]
    previous_specs = yaml.safe_load(
        subprocess.check_output(
            [
                "git",
                "show",
                "HEAD:robocasa/models/assets/scenes/holodeck_task_specs.yaml",
            ]
        )
    )["scenes"]
    expected = {
        "ice_cube_then_approach_window_then_pick_box": {
            "train_1018",
            "train_1664",
            "train_2025",
        },
        "pick_cup_then_approach_desk_then_place_cup_on_placemat": {
            "train_1018",
            "train_1664",
            "train_2025",
        },
        "approach_desk_then_move_cup_to_center": {"train_1018"},
        "approach_desk_then_open_jar": {"train_1018"},
    }

    for task_name, expected_scenes in expected.items():
        assert {
            scene_name
            for scene_name, scene in current_specs.items()
            if task_name in scene
            and task_name not in previous_specs.get(scene_name, {})
        } == expected_scenes


def test_desk_laptop_trash_composite_keeps_atomic_trash_spawn_1197():
    specs = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]["train_1197"]

    atomic_pose = specs["approach_trash_can"]["robot_pose"]["pos"]
    composite_pose = specs["approach_desk_then_close_laptop_then_approach_trash_can"][
        "robot_pose"
    ]["pos"]

    assert atomic_pose == pytest.approx([4.00, 2.10, 0.0])
    assert composite_pose == pytest.approx([2.75, 2.10, 0.0])


@pytest.mark.parametrize("task_cls, stage_names", NEW_TRAIN_117_DESK_COMPOSITES)
def test_new_train_117_desk_composites_are_ordered_and_registered(
    task_cls, stage_names
):
    assert task_cls.__name__ in REGISTERED_HOLODECK_ENVS
    expected = (117, 1018, 1197, 1664, 2025)
    if task_cls is HolodeckApproachDeskThenCloseDrawer:
        expected = (117, 1197)
    assert task_cls.SUPPORTED_SCENE_IDS == expected
    assert tuple(stage.name for stage in task_cls.ATOMIC_TASKS) == stage_names
    assert all(stage.task_cls is not None for stage in task_cls.ATOMIC_TASKS)


@pytest.mark.parametrize("task_cls", HOLODECK_TABLETOP_TASKS)
def test_tabletop_atomic_tasks_support_requested_holodeck_scenes(task_cls):
    assert task_cls.__name__ in REGISTERED_HOLODECK_ENVS
    assert task_cls.SUPPORTED_SCENE_IDS == (117, 1018, 1197, 1664, 2025)


def test_ice_cube_task_uses_only_its_task_local_scene_layout():
    assert HolodeckIceCubeInGlass.USE_DEFAULT_ASSET_LAYOUT is False


def test_ice_window_box_composite_uses_desk_ice_layout_and_facing_spawn():
    specs = yaml.safe_load(
        open("robocasa/models/assets/scenes/holodeck_task_specs.yaml")
    )["scenes"]["train_117"]
    atomic_spec = specs["pick_box_from_floor"]
    spec = specs["ice_cube_then_approach_window_then_pick_box"]
    assert atomic_spec["box_pose"]["pos"] == [3.00, 3.00, 0.1580]
    assert spec["table_body"] == "desk_aeb6ab35aea76da67db0d3cd45d3e821_1_0_0"
    assert spec["ice_pose"]["pos"] == [5.45, 4.45, 0.72108]
    assert spec["cup_pose"]["pos"] == [5.45, 4.60, 0.7773]
    assert spec["box_pose"]["pos"] == [4.55, 0.60, 0.1580]
    assert spec["robot_pose"]["pos"] == [4.95, 4.45, 0.0]
    assert spec["robot_pose"]["yaw"] == pytest.approx(0.0)
    assert spec["remove_scene_bodies"] == [
        "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",
    ]
    assert HolodeckIceCubeThenApproachWindowThenPickBox.ICE_SPEC_TASK == (
        "ice_cube_then_approach_window_then_pick_box"
    )


def test_window_box_composite_has_local_box_poses_without_changing_atomic_poses():
    specs = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]
    expected_atomic = {
        "train_117": [3.00, 3.00, 0.1580],
        "train_1018": [5.50, 1.80, 0.1580],
        "train_1197": [3.45, 3.00, 0.1580],
        "train_1664": [3.45, 3.00, 0.1580],
        "train_2025": [3.45, 3.00, 0.1580],
    }
    expected_composite = {
        "train_117": [4.30, 0.60, 0.1580],
        "train_1018": [4.50, 0.70, 0.1580],
        "train_1197": [4.65, 2.85, 0.1580],
        "train_1664": [4.45, 5.15, 0.1580],
        "train_2025": [1.50, 1.00, 0.1580],
    }
    for scene_name, atomic_pos in expected_atomic.items():
        scene = specs[scene_name]
        assert scene["pick_box_from_floor"]["box_pose"]["pos"] == atomic_pos
        assert (
            scene["approach_window_then_pick_box"]["box_pose"]["pos"]
            == expected_composite[scene_name]
        )
    assert HolodeckApproachWindowThenPickBox.BOX_SPEC_TASK == (
        "approach_window_then_pick_box"
    )
    assert (
        specs["train_2025"]["approach_window_then_pick_box"][
            "ground_box_on_collision_geometry"
        ]
        is True
    )
    for scene_name in ("train_117", "train_1018", "train_1197", "train_1664"):
        assert (
            "ground_box_on_collision_geometry"
            not in specs[scene_name]["approach_window_then_pick_box"]
        )


def test_new_ice_window_box_layouts_reuse_calibrated_reachable_poses():
    scenes = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]
    task_name = "ice_cube_then_approach_window_then_pick_box"

    for scene_name in ("train_1018", "train_1664", "train_2025"):
        scene = scenes[scene_name]
        layout = scene[task_name]
        robot_xy = layout["robot_pose"]["pos"][:2]
        assert math.dist(robot_xy, layout["ice_pose"]["pos"][:2]) == pytest.approx(
            0.45, abs=1e-5
        )
        assert math.dist(robot_xy, layout["cup_pose"]["pos"][:2]) == pytest.approx(
            0.45, abs=1e-5
        )
        assert layout["box_pose"] == scene["approach_window_then_pick_box"]["box_pose"]

    assert scenes["train_2025"][task_name]["box_pose"] == {
        "pos": [1.50, 1.00, 0.1580],
        "quat": [0.707107, 0.707107, 0.0, 0.0],
    }
    assert scenes["train_2025"][task_name]["ground_box_on_collision_geometry"] is True
    for scene_name in ("train_117", "train_1018", "train_1664"):
        assert "ground_box_on_collision_geometry" not in scenes[scene_name][task_name]
    assert task_name not in scenes["train_1197"]


def test_box_floor_workflow_reuses_floor_contact_margin():
    assert (
        HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer.CONTACT_MARGIN
        == HolodeckPlaceBoxOnFloor.CONTACT_MARGIN
    )


def test_box_floor_workflow_closes_the_drawer():
    task_cls = HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer
    assert task_cls.BEHAVIOR == "close"
    assert task_cls.SPEC_TASK == "close_drawer"
    assert task_cls.ATOMIC_TASKS[-1].task_cls is HolodeckCloseDrawer
    assert task_cls.ATOMIC_TASKS[-1].success == "_drawer_closed"


@pytest.mark.parametrize(("task_cls", "stage_names"), TRAIN_117_COMPOSITES)
def test_train_117_composites_have_ordered_one_second_stages(task_cls, stage_names):
    expected = (117, 1018, 1197, 1664, 2025)
    if task_cls is HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer:
        expected = (117, 1197)
    elif task_cls is HolodeckIceCubeThenApproachWindowThenPickBox:
        expected = (117, 1018, 1664, 2025)
    elif task_cls is HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat:
        expected = (117, 1018, 1664, 2025)
    assert task_cls.SUPPORTED_SCENE_IDS == expected
    assert tuple(stage.name for stage in task_cls.ATOMIC_TASKS) == stage_names

    task = object.__new__(task_cls)
    task.control_freq = 20
    assert [
        task._required_stage_hold_steps(stage) for stage in task_cls.ATOMIC_TASKS
    ] == [20] * len(stage_names)


def test_cup_navigation_composite_starts_on_a_separate_source_table():
    specs = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]["train_117"]
    composite = specs["pick_cup_then_approach_desk_then_place_cup_on_placemat"]
    target = specs["place_cup_on_placemat"]

    assert composite["source_table_body"] != target["table_body"]
    assert composite["source_table_body"] == (
        "diningtable_5998373f8cc3a5cf95fd99d754406513_1_0_0"
    )
    assert composite["source_cup_pose"]["pos"] != target["cup_pose"]["pos"]
    assert composite["source_robot_pose"]["pos"] == [4.57380371, 1.45, 0.0]
    assert target["table_body"] == ("desk_aeb6ab35aea76da67db0d3cd45d3e821_1_0_0")


def test_new_cup_navigation_layouts_match_the_approach_desk_target():
    scenes = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]
    task_name = "pick_cup_then_approach_desk_then_place_cup_on_placemat"

    assert task_name not in scenes["train_1197"]

    for scene_name in ("train_1018", "train_1664", "train_2025"):
        layout = scenes[scene_name][task_name]
        assert layout["table_body"] == scenes[scene_name]["approach_desk"]["desk_body"]
        assert math.dist(
            layout["source_robot_pose"]["pos"][:2],
            layout["source_cup_pose"]["pos"][:2],
        ) == pytest.approx(0.45, abs=1e-5)
        assert layout["source_support_z"] > 0.0

    assert scenes["train_1018"][task_name]["source_table_body"] != (
        scenes["train_1018"][task_name]["table_body"]
    )
    assert scenes["train_1664"][task_name]["source_table_body"] == (
        "diningtable_ccf909fdcb2bcd7c1441fe7740cecd0b_1_0_0"
    )
    assert scenes["train_1664"][task_name]["source_robot_pose"] == {
        "pos": [1.50, 0.50, 0.0],
        "yaw": 1.5707963267948966,
    }
    assert scenes["train_1664"][task_name]["source_cup_pose"]["pos"] == [
        1.50,
        0.95,
        0.7566,
    ]
    assert scenes["train_2025"][task_name]["source_table_body"] == (
        "stand_dba5fb19566f0e1caaf78947145fe7fc_2_0_0"
    )
    assert scenes["train_2025"][task_name]["source_robot_pose"] == {
        "pos": [3.30, 0.90, 0.0],
        "yaw": -1.5707963267948966,
    }
    assert scenes["train_2025"][task_name]["source_cup_pose"]["pos"] == [
        3.30,
        0.45,
        0.73173,
    ]
    assert scenes["train_2025"][task_name]["source_support_z"] == pytest.approx(0.65333)


def test_1018_desk_workflow_overrides_keep_navigation_and_objects_together():
    scene = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]["train_1018"]
    desk_body = scene["approach_desk"]["desk_body"]

    assert scene["approach_desk_then_move_cup_to_center"]["desk_body"] == desk_body
    assert scene["approach_desk_then_open_jar"]["table_body"] == desk_body
    assert scene["approach_desk_then_open_jar"]["remove_scene_bodies"] == [
        "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",
    ]


def test_1018_open_jar_composite_removes_only_configured_laptop(monkeypatch):
    task = object.__new__(HolodeckApproachDeskThenOpenJar)
    task.scene_name = "train_1018"
    removed_bodies = []
    parent_calls = []

    def remove_scene_bodies(body_names):
        removed_bodies.extend(body_names)
        return set(body_names)

    task.remove_scene_bodies = remove_scene_bodies
    monkeypatch.setattr(
        HolodeckOpenMugLid,
        "_setup_baked_scene_fixtures",
        lambda self: parent_calls.append(self.scene_name),
    )

    task._setup_baked_scene_fixtures()

    assert removed_bodies == [
        "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",
    ]
    assert parent_calls == ["train_1018"]


def test_1664_open_jar_composite_has_task_local_collision_safe_spawn():
    scene = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]["train_1664"]
    approach_spec = scene["approach_desk"]

    assert approach_spec["robot_pose"]["pos"] == pytest.approx([4.00, 1.80, 0.0])
    assert "desk_composite_robot_pose" not in approach_spec
    assert approach_spec["open_jar_composite_robot_pose"]["pos"] == pytest.approx(
        [3.90, 1.80, 0.0]
    )
    assert HolodeckApproachDeskThenOpenJar.COMPOSITE_ROBOT_POSE_KEY == (
        "open_jar_composite_robot_pose"
    )


def test_1664_pick_cup_composite_has_task_local_collision_safe_spawn():
    scene = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]["train_1664"]
    approach_spec = scene["approach_desk"]

    assert approach_spec["robot_pose"]["pos"] == pytest.approx([4.00, 1.80, 0.0])
    assert "desk_composite_robot_pose" not in approach_spec
    assert approach_spec["pick_cup_composite_robot_pose"]["pos"] == pytest.approx(
        [3.90, 1.80, 0.0]
    )
    assert HolodeckApproachDeskThenPickCup.COMPOSITE_ROBOT_POSE_KEY == (
        "pick_cup_composite_robot_pose"
    )


def test_box_desk_height_uses_scene_support_calibration():
    task = object.__new__(HolodeckPlaceBoxOnFloorThenPickBook)
    task.objects = {"box": SimpleNamespace(root_body="box")}
    task.box_desk_body = "desk"
    task.box_task_spec = {"support_top_z": 0.7486}
    points = {
        "box": np.asarray([[5.60, 1.70, 0.7476], [5.80, 2.00, 1.10]], dtype=float),
        "desk": np.asarray([[5.55, 1.10, 0.0], [5.95, 2.75, 1.30]], dtype=float),
    }
    task._body_geom_points = lambda names: (points[names[0]], None)

    assert task._box_at_desk_height() is True


@pytest.mark.parametrize("task_cls", TISSUE_BOX_WORKFLOWS)
def test_tissue_box_workflows_share_lift_threshold(task_cls):
    take_stages = [
        task for task in task_cls.ATOMIC_TASKS if task.name == "take_tissue_box"
    ]
    assert len(take_stages) == 1
    assert issubclass(task_cls, _HolodeckDrawerTissueBoxMixin)
    assert task_cls.MIN_LIFT_DELTA == pytest.approx(0.30)


def test_tissue_box_destination_support_is_required_only_for_placement_workflows():
    assert HolodeckOpenDrawerThenTakeTissueBox.DESTINATION_REGION_REQUIRED is False
    assert HolodeckOpenDrawerTakeTissueBoxToDesk.DESTINATION_REGION_REQUIRED is True
    assert (
        HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer.DESTINATION_REGION_REQUIRED
        is True
    )


def test_tissue_box_to_desk_uses_full_desktop_contact_in_scene_1197():
    task = object.__new__(HolodeckOpenDrawerTakeTissueBoxToDesk)
    task._tissue_box_removed_before_placement = True
    task._tissue_box_removed = True
    task.destination_region = object()
    calls = []
    task.object_on_region = lambda *_args, **kwargs: calls.append(kwargs) or True

    task.scene_name = "train_1197"
    assert task._tissue_box_released_on_desk() is True
    assert calls[-1]["require_bounds"] is False
    assert calls[-1]["require_contact"] is True

    task.scene_name = "train_117"
    assert task._tissue_box_still_placed_on_desk() is True
    assert calls[-1]["require_bounds"] is True


def test_tissue_box_to_desk_still_requires_the_placement_stage():
    task = object.__new__(HolodeckOpenDrawerTakeTissueBoxToDesk)
    task._tissue_box_removed_before_placement = False
    task._tissue_box_removed = True
    task._tissue_box_on_destination_desk = lambda: pytest.fail(
        "desk contact must not be checked before the placement stage"
    )

    assert task._tissue_box_released_on_desk() is False


@pytest.mark.parametrize("frequency", [10, 20])
@pytest.mark.parametrize("scene_name", ["train_117", "train_1197"])
@pytest.mark.parametrize(
    "task_cls",
    [
        HolodeckOpenDrawerTakeTissueBoxToDesk,
        HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
    ],
)
def test_tissue_box_desk_placement_hold_is_task_and_scene_specific(
    task_cls, scene_name, frequency
):
    task = object.__new__(task_cls)
    task.scene_name = scene_name
    task.control_freq = frequency
    for stage in task_cls.ATOMIC_TASKS:
        assert stage.hold_steps is None
        expected = (
            5
            if task_cls is HolodeckOpenDrawerTakeTissueBoxToDesk
            and scene_name == "train_117"
            and stage.name == "place_tissue_box_on_desk"
            else frequency
        )
        assert task._required_stage_hold_steps(stage) == expected


@pytest.mark.parametrize("frequency", [10, 20])
def test_tissue_box_desk_placement_requires_five_consecutive_steps(
    monkeypatch, frequency
):
    monkeypatch.setattr(
        HolodeckEnv, "_post_action", lambda self, action: (0.0, False, {})
    )
    task = object.__new__(HolodeckOpenDrawerTakeTissueBoxToDesk)
    task.scene_name = "train_117"
    task.control_freq = frequency
    task.SUCCESS_DEBUG_ENABLED = False
    task._atomic_tasks = task.ATOMIC_TASKS
    task._stage_index = 2
    task._stage_hold_steps = 0
    state = {"placed": True}
    task._tissue_box_released_on_desk = lambda: state["placed"]

    for _ in range(4):
        assert (
            HolodeckMultiHopComposite._post_action(task, None)[2]["task_success"]
            is False
        )
    state["placed"] = False
    assert (
        HolodeckMultiHopComposite._post_action(task, None)[2]["task_success"] is False
    )
    assert task._stage_hold_steps == 0
    state["placed"] = True
    for _ in range(4):
        assert (
            HolodeckMultiHopComposite._post_action(task, None)[2]["task_success"]
            is False
        )
    reward, _, info = HolodeckMultiHopComposite._post_action(task, None)
    assert reward == 1.0
    assert info["task_success"] is True
    assert info["success"] is True
    assert task._check_success()


@pytest.mark.parametrize("scene_name", ["train_117", "train_1197"])
def test_tissue_box_desk_placement_metadata_matches_hold(monkeypatch, scene_name):
    monkeypatch.setattr(
        HolodeckOpenDrawerThenTakeTissueBox, "get_ep_meta", lambda self: {}
    )
    task = object.__new__(HolodeckOpenDrawerTakeTissueBoxToDesk)
    task.scene_name = scene_name
    hold = "5 control steps" if scene_name == "train_117" else "1.0 s"
    assert task.get_ep_meta()["success_definition"].endswith(f"for {hold}")


def test_tissue_box_placement_uses_the_approach_desk_calibration():
    assert (
        HolodeckOpenDrawerTakeTissueBoxToDesk.DESTINATION_TABLE_SPEC_TASK
        == HolodeckApproachDesk.SPEC_TASK
    )
    assert (
        HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer.DESTINATION_TABLE_SPEC_TASK
        == HolodeckApproachDesk.SPEC_TASK
    )
    assert (
        HolodeckOpenDrawerTakeTissueBoxToDesk.DESTINATION_SUPPORT_SPEC_TASK
        == "move_cup_to_center"
    )
    assert (
        HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer.DESTINATION_SUPPORT_SPEC_TASK
        == "move_cup_to_center"
    )
    assert HolodeckOpenDrawerTakeTissueBoxToDesk.LAPTOP_CLEARED_SCENES == (
        "train_1664",
        "train_2025",
    )
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        specs = yaml.safe_load(file)["scenes"]
    for scene_name in (
        "train_117",
        "train_1018",
        "train_1197",
        "train_1664",
        "train_2025",
    ):
        assert (
            specs[scene_name]["open_drawer_then_take_tissue_box"]["target_table_body"]
            == specs[scene_name]["approach_desk"]["desk_body"]
        )


def test_tissue_box_new_scene_support_uses_full_desk_bounds():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        specs = yaml.safe_load(file)["scenes"]
    for scene_name in ("train_1664", "train_2025"):
        desk = specs[scene_name]["approach_desk"]
        support = specs[scene_name]["move_cup_to_center"]
        bounds = np.asarray(support["tabletop_bounds_xy"], dtype=float)
        assert support["desk_body"] == desk["desk_body"]
        assert bounds.shape == (2, 2)
        assert np.all(bounds[1] > bounds[0])
        # The placement region must cover more than the old placemat-sized
        # patch along the desk's long axis.
        assert np.max(bounds[1] - bounds[0]) > 0.5


def test_pick_cup_tasks_reuse_the_ice_cube_glass_asset_and_registered_names():
    assert HolodeckPickCupFromDesk.CUP_ASSET == HolodeckIceCubeInGlass.CUP_ASSET
    assert HolodeckPickCupFromDesk.MIN_LIFT_DELTA == pytest.approx(0.30)
    assert HolodeckPickCupFromDesk.HOLD_SECONDS == pytest.approx(1.0)
    assert HolodeckPickCupFromDesk.NOISE_FREE_ROBOT_SPAWN is True
    assert HolodeckPickCupFromDesk.__name__ in REGISTERED_HOLODECK_ENVS
    for task_cls in (
        HolodeckApproachWindowThenDeskThenPickCup,
        HolodeckApproachDeskIceCubePlaceCupThenPickCup,
    ):
        assert task_cls.__name__ in REGISTERED_HOLODECK_ENVS
        assert task_cls.ATOMIC_TASKS[-1].name == "pick_cup"
        assert task_cls.ATOMIC_TASKS[-1].task_cls is HolodeckPickCupFromDesk
        assert task_cls.ATOMIC_TASKS[-1].hold_steps is None

    assert (
        HolodeckApproachDeskIceCubePlaceCupThenPickCup.PICK_CUP_LIFT_DELTA
        == HolodeckPickCupFromDesk.MIN_LIFT_DELTA
    )


def test_cup_composites_reuse_their_atomic_initial_layout_specs():
    """Composite cup resets must follow the matching atomic calibration."""
    assert HolodeckMoveCupToCenterThenOpenLaptop.CUP_SPEC_TASK == "move_cup_to_center"
    assert (
        HolodeckApproachDeskThenCloseLaptopMoveCup.CUP_SPEC_TASK == "move_cup_to_corner"
    )
    assert hasattr(HolodeckApproachDeskThenCloseLaptopMoveCup, "_desk_target_frame")
    assert not hasattr(HolodeckApproachDeskThenCloseLaptopMoveCup, "CUP_RELOCATIONS")

    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        scenes = yaml.safe_load(file)["scenes"]
    for scene_name in HolodeckApproachDeskThenCloseLaptopMoveCup.SUPPORTED_SCENE_IDS:
        scene = f"train_{scene_name}"
        corner = scenes[scene]["move_cup_to_corner"]
        assert corner["initial_cup_base_xy"] == corner["cup_pose"]["pos"][:2]


def test_ice_place_pick_workflow_reuses_the_cup_between_ice_and_placemat():
    task_cls = HolodeckApproachDeskIceCubePlaceCupThenPickCup

    assert task_cls.SUPPORTED_SCENE_IDS == (117, 1018, 1197, 1664, 2025)
    assert set(task_cls.COMPOSITE_LAYOUTS) == {
        "train_117",
        "train_1018",
        "train_1197",
        "train_1664",
        "train_2025",
    }
    assert all(
        "pick_cup_pose" not in layout for layout in task_cls.COMPOSITE_LAYOUTS.values()
    )
    assert task_cls.ATOMIC_TASKS[-1].metadata["object"] == "cup"
    assert task_cls.ATOMIC_TASKS[1].hold_steps == 20


def test_ice_place_pick_1018_uses_the_approach_desk_surface():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        spec = yaml.safe_load(file)["scenes"]["train_1018"]["ice_cube_then_place_cup"]
    assert spec["table_body"] == ("desk_0ecf411cb557754bf283627e911b16c6_1_0_0")
    layout = HolodeckApproachDeskIceCubePlaceCupThenPickCup.COMPOSITE_LAYOUTS[
        "train_1018"
    ]
    assert layout["cup_pose"]["pos"] == [7.55, 0.82, 0.8009]
    assert layout["ice_pose"]["pos"] == [7.55, 1.16, 0.7248]
    assert layout["placemat_pose"]["pos"] == [7.55, 0.48, 0.7248]
    assert spec["robot_pose"] == {"pos": [5.35, 1.25, 0.0], "yaw": 0.0}


def test_book_composite_has_only_pick_then_one_second_placement_stages():
    stages = HolodeckPickBookFromSourceToDesk.ATOMIC_TASKS

    assert tuple(stage.name for stage in stages) == (
        "pick_book",
        "place_book_on_desk",
    )
    assert stages[-1].hold_steps is None
    assert stages[-1].task_cls.HOLD_SECONDS == pytest.approx(1.0)


def test_book_composite_success_uses_only_lift_and_desk_contact(monkeypatch):
    monkeypatch.setattr(
        HolodeckEnv,
        "_post_action",
        lambda _self, _action: (0.0, False, {}),
    )
    monkeypatch.setattr(
        HolodeckPickBookFromDresser,
        "_post_action",
        lambda _self, _action: pytest.fail("composite must not require a grasp"),
    )

    task = object.__new__(HolodeckPickBookFromSourceToDesk)
    state = {"z": 1.31, "on_desk": False}
    task._atomic_tasks = HolodeckPickBookFromSourceToDesk.ATOMIC_TASKS
    task._stage_index = 0
    task._stage_hold_steps = 0
    task._book_initial_z = 1.0
    task._book_root_z = lambda: state["z"]
    task.destination_region = object()
    region_calls = []
    task._book_on_region = lambda *_args, **kwargs: (
        region_calls.append(kwargs) or state["on_desk"]
    )
    task.control_freq = 50
    task.scene_name = "train_117"
    task.sim = SimpleNamespace(data=SimpleNamespace(time=0.0))
    task._success_debug_next_log_time = 1e9

    for _ in range(50):
        task._post_action(None)
    assert task.stage_index == 1

    state["on_desk"] = True
    for _ in range(50):
        task._post_action(None)
    assert task._check_success() is True
    assert region_calls[-1]["require_contact"] is True
    assert region_calls[-1]["require_bounds"] is False


def test_book_success_diagnostics_log_every_five_seconds(capsys):
    task = object.__new__(HolodeckPickBookFromSourceToDesk)
    task.scene_name = "train_117"
    task.sim = SimpleNamespace(data=SimpleNamespace(time=4.99))
    task._success_debug_next_log_time = 5.0
    task._book_initial_z = 1.0
    task._book_root_z = lambda: 1.31
    task.destination_region = object()
    task._book_on_region = lambda *_args, **_kwargs: True
    task._atomic_tasks = (
        SimpleNamespace(name="pick_book"),
        SimpleNamespace(name="place_book_on_desk"),
    )
    task._stage_index = 1
    task._stage_hold_steps = 12
    task._required_stage_hold_steps = lambda _stage: 20

    task._log_success_conditions()
    assert capsys.readouterr().out == ""

    task.sim.data.time = 5.0
    task._log_success_conditions()
    output = capsys.readouterr().out
    assert "t=5.00s" in output
    assert "pick_book(" in output
    assert "place_book_on_desk(" in output
    assert "grasped" not in output
    assert "pick_complete" not in output
    assert task._success_debug_next_log_time == 10.0


def test_requested_tasks_keep_the_calibrated_10cm_robot_poses():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        scenes = yaml.safe_load(file)["scenes"]

    expected = {
        "train_117": {
            "pick_book_from_dresser": [5.19421348, 3.30, 0.0],
            "throw_bottle_into_trash_can": [1.09107319, 4.15392845, 0.0],
            "open_laptop": [4.99989971, 4.0661376, 0.0],
            "close_laptop": [4.99989971, 4.0661376, 0.0],
            "ice_cube_in_glass": [4.57380371, 1.45, 0.0],
            "open_mug_lid": [4.99989361, 4.8, 0.0],
            "close_mug_lid": [4.57380371, 1.45, 0.0],
            "sugar_cube_in_jar": [4.57380371, 1.45, 0.0],
            "place_cup_on_placemat": [4.99989361, 4.8, 0.0],
            "move_cup_to_center": [5.0, 4.4, 0.0],
            "pick_cup_from_desk": [4.99990582, 4.8, 0.0],
        },
        "train_1018": {
            "insert_folder_into_shelf": [3.81, 2.20, 0.0],
            "pick_book_from_dresser": [7.41427139, 2.00, 0.0],
            "open_laptop": [7.04657549, 0.89, 0.0],
            "close_laptop": [7.04658770, 0.89, 0.0],
            "ice_cube_in_glass": [3.81892754, 1.25, 0.0],
            "open_mug_lid": [3.81892754, 1.25, 0.0],
            "close_mug_lid": [3.81892754, 1.25, 0.0],
            "sugar_cube_in_jar": [3.81892754, 1.25, 0.0],
            "place_cup_on_placemat": [3.81892754, 1.25, 0.0],
            "move_cup_to_center": [3.7261299324, 1.4109423525, 0.0],
            "throw_bottle_into_trash_can": [4.39622440, 0.92098549, 0.0],
            "pick_cup_from_desk": [7.04658770, 0.89, 0.0],
        },
        "train_1197": {
            "pick_book_from_dresser": [4.80, 5.20, 0.0],
            "pick_cup_from_desk": [4.84698800, 2.17987800, 0.0],
            "throw_bottle_into_trash_can": [0.90, 1.58386200, 0.0],
        },
        "train_1664": {
            "pick_book_from_dresser": [3.045, 5.009, 0.0],
            "pick_cup_from_desk": [5.19, 1.85, 0.0],
            "throw_bottle_into_trash_can": [0.90, 4.18386200, 0.0],
        },
        "train_2025": {
            "pick_book_from_dresser": [1.45, 0.85, 0.0],
            "pick_cup_from_desk": [5.19, 2.15, 0.0],
            "throw_bottle_into_trash_can": [0.90, 4.18386200, 0.0],
        },
    }

    assert HolodeckMoveCupToCorner.CALIBRATION_SPEC_TASK == "move_cup_to_corner"
    for scene_name, tasks in expected.items():
        for task_name, pose in tasks.items():
            assert scenes[scene_name][task_name]["robot_pose"]["pos"] == pytest.approx(
                pose
            )


def test_requested_object_tasks_start_45cm_away_and_facing_the_source():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        scenes = yaml.safe_load(file)["scenes"]

    source_pose_keys = {
        "ice_cube_in_glass": "ice_pose",
        "place_cup_on_placemat": "cup_pose",
        "move_cup_to_center": "initial_cup_base_xy",
        "move_cup_to_corner": "initial_cup_base_xy",
        "throw_bottle_into_trash_can": "bottle_pose",
        "insert_folder_into_shelf": "folder_pose",
    }
    for scene_name in (
        "train_117",
        "train_1018",
    ):
        for task_name, source_key in source_pose_keys.items():
            if task_name not in scenes[scene_name]:
                continue
            spec = scenes[scene_name][task_name]
            robot_pose = spec["robot_pose"]
            source = spec[source_key]
            source_xy = (
                source if source_key == "initial_cup_base_xy" else source["pos"][:2]
            )
            delta = [
                source_xy[0] - robot_pose["pos"][0],
                source_xy[1] - robot_pose["pos"][1],
            ]
            expected_distance = (
                0.50
                if scene_name == "train_117"
                and task_name in {"move_cup_to_center", "move_cup_to_corner"}
                else 0.45
            )
            assert math.hypot(*delta) == pytest.approx(expected_distance, abs=1e-5)
            assert math.atan2(
                math.sin(robot_pose["yaw"] - math.atan2(delta[1], delta[0])),
                math.cos(robot_pose["yaw"] - math.atan2(delta[1], delta[0])),
            ) == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("scene_name", ("train_1197", "train_1664", "train_2025"))
def test_new_bottle_tasks_start_45cm_away_and_facing_the_source(scene_name):
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        spec = yaml.safe_load(file)["scenes"][scene_name]["throw_bottle_into_trash_can"]
    source_xy = spec["bottle_pose"]["pos"][:2]
    robot_pose = spec["robot_pose"]
    delta = np.asarray(source_xy) - np.asarray(robot_pose["pos"][:2])
    assert np.linalg.norm(delta) == pytest.approx(0.45, abs=1e-5)
    assert math.atan2(
        math.sin(robot_pose["yaw"] - math.atan2(delta[1], delta[0])),
        math.cos(robot_pose["yaw"] - math.atan2(delta[1], delta[0])),
    ) == pytest.approx(0.0, abs=1e-5)


def test_ice_cube_in_glass_cup_is_within_reach_of_the_robot():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        scenes = yaml.safe_load(file)["scenes"]

    for scene_name in ("train_117", "train_1018"):
        spec = scenes[scene_name]["ice_cube_in_glass"]
        robot_xy = spec["robot_pose"]["pos"][:2]
        cup_xy = spec["cup_pose"]["pos"][:2]
        ice_xy = spec["ice_pose"]["pos"][:2]

        cup_distance = math.hypot(cup_xy[0] - robot_xy[0], cup_xy[1] - robot_xy[1])
        ice_distance = math.hypot(ice_xy[0] - robot_xy[0], ice_xy[1] - robot_xy[1])
        assert cup_distance < 0.45 + 1e-5
        assert cup_distance == pytest.approx(0.45, abs=1e-5)
        assert ice_distance == pytest.approx(0.45, abs=1e-5)


@pytest.mark.parametrize(
    ("task_cls", "stage_names"),
    (
        (
            HolodeckTakeTissueBoxThenPlaceCup,
            (
                "take_tissue_box",
                "place_tissue_box_on_desk",
                "place_cup_on_placemat",
            ),
        ),
        (
            HolodeckOpenDrawerThenTakeTissueBox,
            ("open_drawer", "take_tissue_box"),
        ),
        (
            HolodeckOpenDrawerTakeTissueBoxToDesk,
            ("open_drawer", "take_tissue_box", "place_tissue_box_on_desk"),
        ),
        (
            HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
            (
                "open_drawer",
                "take_tissue_box",
                "place_tissue_box_on_desk",
                "close_drawer",
            ),
        ),
        (
            HolodeckApproachTrashCanThenThrowBottle,
            ("approach_trash_can", "throw_bottle"),
        ),
        (
            HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop,
            ("remove_folder_from_shelf", "place_folder_on_desk", "close_laptop"),
        ),
        (
            HolodeckApproachWindowThenDesk,
            ("approach_rightmost_window", "approach_desk"),
        ),
        (
            HolodeckApproachTrashCanThenThrowBottleThenApproachDesk,
            ("approach_trash_can", "throw_bottle", "approach_desk"),
        ),
        (
            HolodeckApproachDeskThenOpenLaptopThenApproachWindow,
            ("approach_desk", "open_laptop", "approach_rightmost_window"),
        ),
        (
            HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan,
            ("approach_desk", "close_laptop", "approach_trash_can"),
        ),
        (
            HolodeckApproachTrashCanThenApproachDeskThenPullChair,
            ("approach_trash_can", "approach_desk", "pull_chair_from_desk"),
        ),
        (
            HolodeckPickBookFromSourceToDeskThenApproachTrashCan,
            (
                "pick_book",
                "place_book_on_desk",
                "approach_trash_can",
            ),
        ),
        (
            HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan,
            (
                "take_tissue_box",
                "place_tissue_box_on_desk",
                "close_drawer",
                "approach_trash_can",
            ),
        ),
        (
            HolodeckApproachDeskThenIceCubeInGlass,
            ("approach_desk", "put_ice_in_cup"),
        ),
        (
            HolodeckCloseJarThenApproachDeskThenApproachTrashCan,
            ("close_jar", "approach_desk", "approach_trash_can"),
        ),
        (
            HolodeckApproachTrashCanThenApproachDeskThenOpenJar,
            ("approach_trash_can", "approach_desk", "open_jar"),
        ),
        (
            HolodeckApproachTrashCanThenPickBoxFromFloor,
            ("approach_trash_can", "pick_box_from_floor"),
        ),
    ),
)
def test_requested_workflows_preserve_declared_stage_order(task_cls, stage_names):
    assert tuple(task.name for task in task_cls.ATOMIC_TASKS) == stage_names


def test_trash_box_workflow_ends_with_floor_box_pick():
    assert tuple(
        task.name for task in HolodeckApproachTrashCanThenPickBoxFromFloor.ATOMIC_TASKS
    ) == ("approach_trash_can", "pick_box_from_floor")


def test_close_laptop_trash_box_workflow_omits_chair_stage():
    assert tuple(
        task.name
        for task in HolodeckCloseLaptopThenApproachTrashCanThenPickBox.ATOMIC_TASKS
    ) == ("close_laptop", "approach_trash_can", "pick_box_from_floor")
    assert all(
        task.name != "push_chair_to_desk"
        for task in HolodeckCloseLaptopThenApproachTrashCanThenPickBox.ATOMIC_TASKS
    )


def test_trash_box_desk_workflow_preserves_navigation_and_placement_order():
    task_cls = HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk
    assert task_cls.SUPPORTED_SCENE_IDS == (117, 1018, 1197, 1664, 2025)
    assert tuple(task.name for task in task_cls.ATOMIC_TASKS) == (
        "approach_trash_can",
        "approach_desk",
        "pick_box_from_floor",
        "place_box_on_desk",
    )


def test_1018_trash_box_desk_workflow_matches_atomic_trash_orientation():
    scene = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]["train_1018"]
    trash_spec = scene["approach_trash_can"]

    assert trash_spec["robot_pose"]["pos"] == pytest.approx([4.75, 2.25, 0.0])
    assert trash_spec["desk_composite_robot_pose"]["pos"] == pytest.approx(
        [4.75, 2.25, 0.0]
    )
    assert trash_spec["desk_composite_robot_pose"]["yaw"] == pytest.approx(
        -0.471415628820373
    )
    trash_first_pose = trash_spec["trash_first_composite_robot_pose"]
    assert trash_first_pose["pos"] == pytest.approx([4.73675, 2.34912, 0.0])
    assert trash_first_pose["yaw"] == pytest.approx(trash_spec["robot_pose"]["yaw"])
    assert (
        HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk.TRASH_FIRST_ROBOT_POSE_KEY
        == "trash_first_composite_robot_pose"
    )


def test_pull_trash_box_workflow_ends_with_box_lift():
    task_cls = HolodeckPullChairThenApproachTrashCanThenPickBox
    assert task_cls.SUPPORTED_SCENE_IDS == (117, 1018, 1197, 1664, 2025)
    assert tuple(task.name for task in task_cls.ATOMIC_TASKS) == (
        "pull_chair_from_desk",
        "approach_trash_can",
        "pick_box_from_floor",
    )
    assert task_cls.ATOMIC_TASKS[-1].success == "_box_lifted"
    assert task_cls.ATOMIC_TASKS[-1].metadata["skill"] == "lift"


def test_1197_pull_trash_box_workflow_uses_collision_safe_pull_pose():
    scene = yaml.safe_load(
        Path("robocasa/models/assets/scenes/holodeck_task_specs.yaml").read_text()
    )["scenes"]["train_1197"]
    pull_spec = scene["pull_chair_from_desk"]

    assert pull_spec["robot_pose"]["pos"] == pytest.approx([5.72, 2.10, 0.0])
    assert pull_spec["composite_robot_pose"] == {
        "pos": [3.90, 2.10, 0.0],
        "yaw": 0.0,
    }


def test_tissue_box_cup_workflow_exposes_composite_stage_metadata():
    assert HolodeckTakeTissueBoxThenPlaceCup.COMPOSITE_STAGES == (
        "grasp and remove the tissue box from the initially open drawer",
        "place the tissue box fully on the calibrated desk",
        "place the cup fully on the calibrated placemat",
    )


def test_tissue_box_cup_workflow_clears_1018_desk_laptops():
    assert HolodeckTakeTissueBoxThenPlaceCup.CALIBRATED_LAPTOP_BODIES == {
        "train_1018": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
    }


def test_tissue_box_cup_workflow_uses_calibrated_destination_support():
    task_cls = HolodeckTakeTissueBoxThenPlaceCup

    assert task_cls.DESTINATION_TABLE_SPEC_TASK == HolodeckApproachDesk.SPEC_TASK
    assert task_cls.DESTINATION_SUPPORT_SPEC_TASK == "move_cup_to_center"
    assert task_cls.LAPTOP_CLEARED_SCENES == ("train_1664", "train_2025")


def test_tissue_box_cup_workflow_uses_1018_composite_robot_pose():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        spec = yaml.safe_load(file)["scenes"]["train_1018"][
            "take_tissue_box_from_open_drawer"
        ]
    assert spec["composite_robot_pose"] == {
        "pos": [7.0, 2.25, 0.0],
        "yaw": 0.7073878461355123,
    }
    assert spec["composite_cup_layout"]["table_body"] == (
        "desk_0ecf411cb557754bf283627e911b16c6_1_0_0"
    )
    assert spec["composite_cup_layout"]["cup_pose"]["pos"] == [
        7.55,
        1.02,
        0.8009,
    ]
    assert spec["composite_cup_layout"]["placemat_pose"]["pos"] == [
        7.55,
        0.68,
        0.7248,
    ]


def test_requested_workflow_scene_boundaries_and_laptop_direction():
    assert HolodeckTakeTissueBoxThenPlaceCup.SUPPORTED_SCENE_IDS == (
        117,
        1018,
        1197,
        1664,
        2025,
    )
    assert HolodeckApproachTrashCanThenThrowBottle.SUPPORTED_SCENE_IDS == (117, 1018)
    assert HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop.SUPPORTED_SCENE_IDS == (
        1018,
    )
    assert HolodeckApproachWindowThenDesk.SUPPORTED_SCENE_IDS == (
        117,
        1018,
        1664,
        2025,
    )
    assert (
        HolodeckApproachTrashCanThenThrowBottleThenApproachDesk.SUPPORTED_SCENE_IDS
        == (117, 1018)
    )
    assert HolodeckApproachDeskThenOpenLaptopThenApproachWindow.SUPPORTED_SCENE_IDS == (
        117,
        1018,
    )
    assert (
        HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan.SUPPORTED_SCENE_IDS
        == (117, 1018, 1197)
    )
    assert HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop.BEHAVIOR == "close"
    assert (
        HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop.LAPTOP_SPEC_TASK
        == "close_laptop"
    )
    assert HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop.OPEN_INIT_FRACTION == 1.0


def test_folder_to_desk_close_laptop_stages_hold_one_second():
    task_cls = HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop
    stages = task_cls.ATOMIC_TASKS

    assert tuple(stage.hold_steps for stage in stages) == (None, None, None)
    assert all(
        stage.task_cls.HOLD_SECONDS == pytest.approx(1.0)
        for stage in stages
        if isinstance(stage.task_cls, type)
    )

    task = object.__new__(task_cls)
    task.control_freq = 20
    assert [task._required_stage_hold_steps(stage) for stage in stages] == [20, 20, 20]


def test_folder_to_desk_predicates_do_not_require_grasp_state(monkeypatch):
    task = object.__new__(HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop)
    task._folder_contacts_shelf = lambda: False
    task._folder_on_desk = lambda **kwargs: kwargs == {"require_released": False}

    assert task._folder_removed_from_shelf() is True
    assert task._folder_released_on_desk() is True


def test_folder_to_desk_placement_uses_contact_only_on_the_full_desk():
    task = object.__new__(HolodeckRemoveFolderFromShelfToDesk)
    task._folder_on_desk = lambda **kwargs: kwargs == {
        "require_released": False,
        "require_bounds": False,
    }

    assert task._folder_released_on_desk() is True


def test_folder_to_desk_workflow_reuses_atomic_folder_layout():
    assert HolodeckRemoveFolderFromShelfToDesk.FOLDER_SCALE_OVERRIDE is None


def test_folder_to_desk_open_laptop_workflow_reuses_atomic_folder_layout():
    assert (
        HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop.FOLDER_SCALE_OVERRIDE is None
    )


def test_new_requested_workflows_have_calibrated_scene_scope():
    assert (
        HolodeckApproachTrashCanThenApproachDeskThenPullChair.SUPPORTED_SCENE_IDS
        == (
            117,
            1018,
        )
    )
    assert HolodeckPickBookFromSourceToDeskThenApproachTrashCan.SUPPORTED_SCENE_IDS == (
        117,
        1018,
    )
    assert (
        HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan.SUPPORTED_SCENE_IDS
        == (
            117,
            1018,
        )
    )
    assert HolodeckApproachDeskThenIceCubeInGlass.SUPPORTED_SCENE_IDS == (
        117,
        1018,
        1197,
        1664,
        2025,
    )
    assert HolodeckCloseJarThenApproachDeskThenApproachTrashCan.SUPPORTED_SCENE_IDS == (
        117,
        1018,
        1664,
    )
    assert HolodeckApproachTrashCanThenApproachDeskThenOpenJar.SUPPORTED_SCENE_IDS == (
        117,
        1018,
    )


def test_close_jar_desk_trash_workflow_uses_only_the_relocated_1664_layout():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        specs = yaml.safe_load(file)["scenes"]

    task_name = "close_jar_then_approach_desk_then_approach_trash_can"
    assert task_name not in specs["train_1197"]
    assert task_name not in specs["train_2025"]

    layout = specs["train_1664"][task_name]
    dining_table = "diningtable_ccf909fdcb2bcd7c1441fe7740cecd0b_1_0_0"
    assert layout["table_body"] == dining_table
    assert layout["desk_body"] == dining_table
    assert layout["robot_pose"] == {
        "pos": [1.50, 0.50, 0.0],
        "yaw": pytest.approx(math.pi / 2),
    }
    assert layout["mug_pose"]["pos"] == pytest.approx([1.50, 0.80, 0.75140])
    assert layout["closed_lid_pose"]["pos"] == pytest.approx([1.50, 0.80, 0.84749])
    # The open lid is beside the jar: its relative offset is the old rearward
    # offset rotated 90 degrees around the vertical axis.
    assert layout["open_lid_pose"]["pos"] == pytest.approx([1.65, 0.82574, 0.69800])


def test_pull_chair_composites_match_atomic_displacement_threshold():
    assert HolodeckApproachDeskThenPullChair.MIN_PULL_DISPLACEMENT == pytest.approx(
        0.50
    )
    assert (
        HolodeckApproachDeskPullChairThenOpenLaptop.MIN_CHAIR_PULL_DISPLACEMENT
        == pytest.approx(HolodeckPullChairFromDesk.MIN_PULL_DISPLACEMENT)
    )
    assert (
        HolodeckApproachDeskPullChairThenOpenLaptop.MIN_CHAIR_PULL_DISPLACEMENT
        == pytest.approx(0.50)
    )

    for task_cls, stage_name in (
        (HolodeckApproachDeskThenPullChair, "pull_chair_from_desk"),
        (
            HolodeckApproachTrashCanThenApproachDeskThenPullChair,
            "pull_chair_from_desk",
        ),
        (HolodeckApproachDeskPullChairThenOpenLaptop, "pull_chair"),
        (
            HolodeckPullChairThenApproachTrashCanThenPickBox,
            "pull_chair_from_desk",
        ),
    ):
        stage = next(
            stage for stage in task_cls.ATOMIC_TASKS if stage.name == stage_name
        )
        assert stage.task_cls is HolodeckPullChairFromDesk
        assert stage.hold_steps is None


@pytest.mark.parametrize(
    ("task_cls", "predicate", "chair_attr"),
    (
        (
            HolodeckApproachDeskThenPullChair,
            "_chair_pulled_after_approach",
            "chair_body",
        ),
        (
            HolodeckApproachTrashCanThenApproachDeskThenPullChair,
            "_chair_pulled_after_approach",
            "chair_body",
        ),
        (
            HolodeckApproachDeskPullChairThenOpenLaptop,
            "_chair_pulled_after_approach",
            "chair_body",
        ),
        (
            HolodeckPullChairThenApproachTrashCanThenPickBox,
            "_chair_pulled",
            "pull_chair_body",
        ),
    ),
)
def test_pull_composite_predicates_use_the_atomic_reset_origin(
    task_cls, predicate, chair_attr
):
    task = object.__new__(task_cls)
    task._pull_origin_xy = np.zeros(2)
    setattr(task, chair_attr, "chair")

    class Model:
        def body_name2id(self, name):
            assert name == "chair"
            return 0

    class Data:
        body_xpos = np.asarray([(0.49, 0.0, 0.0)])

    class Sim:
        model = Model()
        data = Data()

    task.sim = Sim()
    assert getattr(task, predicate)() is False
    task.sim.data.body_xpos[0, 0] = 0.50
    assert getattr(task, predicate)() is True


def test_book_workflows_clear_the_destination_desk_laptops():
    expected = {
        "train_117": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
        "train_1018": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
    }
    assert HolodeckPickBookFromDresser.CALIBRATED_LAPTOP_BODIES == {
        "train_1018": expected["train_1018"],
    }
    assert HolodeckPickBookFromSourceToDesk.CALIBRATED_LAPTOP_BODIES == expected
    assert (
        HolodeckPickBookFromSourceToDeskThenApproachTrashCan.CALIBRATED_LAPTOP_BODIES
        == expected
    )


def test_ice_composite_uses_approach_desk_targets_and_removes_only_task_laptops():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        specs = yaml.safe_load(file)["scenes"]
    for scene_name in (
        "train_117",
        "train_1018",
        "train_1197",
        "train_1664",
        "train_2025",
    ):
        assert (
            HolodeckApproachDeskThenIceCubeInGlass.COMPOSITE_LAYOUTS[scene_name][
                "table_body"
            ]
            == specs[scene_name]["approach_desk"]["desk_body"]
        )
        if scene_name in ("train_117", "train_1018"):
            assert HolodeckApproachDeskThenIceCubeInGlass.CALIBRATED_LAPTOP_BODIES[
                scene_name
            ] == ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",)


def test_ice_composite_uses_remote_spawn_for_new_scene_desks():
    layouts = HolodeckApproachDeskThenIceCubeInGlass.COMPOSITE_LAYOUTS
    assert layouts["train_1197"]["robot_pose"]["pos"] == pytest.approx([3.5, 2.1, 0.0])
    assert layouts["train_1664"]["robot_pose"]["pos"] == pytest.approx([3.5, 1.8, 0.0])


def test_tissue_box_drawer_trash_workflow_requires_desk_placement():
    task_cls = HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan
    assert tuple(task.name for task in task_cls.ATOMIC_TASKS) == (
        "take_tissue_box",
        "place_tissue_box_on_desk",
        "close_drawer",
        "approach_trash_can",
    )
    assert task_cls.ATOMIC_TASKS[1].metadata == {
        "skill": "place",
        "object": "tissue_box",
        "target": "desk",
    }
    assert task_cls.COMPOSITE_STAGES[1] == (
        "place the tissue box fully on the calibrated desk"
    )


def test_close_jar_composite_uses_remote_trash_safe_spawn():
    specs = yaml.safe_load(
        open("robocasa/models/assets/scenes/holodeck_task_specs.yaml")
    )["scenes"]
    spec = specs["train_1018"]["close_mug_lid"]

    assert spec["robot_pose"] == {
        "pos": [3.81892754, 1.25, 0.0],
        "yaw": 3.141592653589793,
    }
    assert spec["composite_robot_pose"] == {
        "pos": [3.90, 1.45, 0.0],
        "yaw": -2.774418819771574,
    }


def test_open_jar_composite_uses_calibrated_jar_desk_and_scene_117_corner_spawn():
    specs = yaml.safe_load(
        open("robocasa/models/assets/scenes/holodeck_task_specs.yaml")
    )["scenes"]

    scene = specs["train_117"]
    assert scene["open_mug_lid"]["table_body"] == scene["approach_desk"]["desk_body"]

    scene = specs["train_1018"]
    open_spec = scene["open_mug_lid"]
    close_spec = scene["close_mug_lid"]
    assert open_spec["table_body"] == close_spec["table_body"]
    assert open_spec["robot_pose"] == close_spec["robot_pose"]
    assert open_spec["lid_pose"]["pos"][:2] == open_spec["mug_pose"]["pos"][:2]
    assert math.isclose(
        math.dist(open_spec["robot_pose"]["pos"][:2], open_spec["mug_pose"]["pos"][:2]),
        math.dist(
            specs["train_117"]["open_mug_lid"]["robot_pose"]["pos"][:2],
            specs["train_117"]["open_mug_lid"]["mug_pose"]["pos"][:2],
        ),
        abs_tol=1e-8,
    )

    scene = specs["train_117"]
    pose = scene["approach_trash_can"]["composite_robot_pose"]
    robot_xy = pose["pos"][:2]
    trash_xy = [0.35, 3.8]
    expected_yaw = math.atan2(trash_xy[1] - robot_xy[1], trash_xy[0] - robot_xy[0])
    assert math.isclose(pose["yaw"], expected_yaw, abs_tol=1e-6)
    assert math.dist(robot_xy, trash_xy) > 1.0 + 0.15


def test_open_jar_composite_uses_the_lid_target_state_only():
    task_cls = HolodeckApproachTrashCanThenApproachDeskThenOpenJar
    task = task_cls.__new__(task_cls)
    task._lid_is_open = lambda: False

    assert task._jar_opened() is False
    task._lid_is_open = lambda: True
    assert task._jar_opened() is True


def test_open_jar_composite_clears_the_1018_overlapping_laptop_only():
    assert (
        HolodeckApproachTrashCanThenApproachDeskThenOpenJar.CALIBRATED_LAPTOP_BODIES
        == {
            "train_1018": ("laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",),
        }
    )


def test_open_jar_composite_1018_places_jar_on_the_approach_desk():
    specs = yaml.safe_load(
        open("robocasa/models/assets/scenes/holodeck_task_specs.yaml")
    )["scenes"]
    layout = HolodeckApproachTrashCanThenApproachDeskThenOpenJar.COMPOSITE_LAYOUTS[
        "train_1018"
    ]

    assert layout["table_body"] == specs["train_1018"]["approach_desk"]["desk_body"]
    assert layout["mug_pose"]["pos"] == [7.55, 1.02, 0.7983895]
    assert layout["lid_pose"]["pos"][:2] == layout["mug_pose"]["pos"][:2]
    assert math.isclose(
        layout["lid_pose"]["pos"][2] - layout["mug_pose"]["pos"][2],
        0.09609,
        abs_tol=1e-8,
    )


@pytest.mark.parametrize(
    "task_cls, supported",
    [
        (HolodeckApproachWindowThenDesk, (117, 1018, 1664, 2025)),
        (HolodeckApproachDeskThenOpenLaptopThenApproachWindow, (117, 1018)),
    ],
)
def test_window_desk_composites_reject_explicit_scene_1197(task_cls, supported):
    assert task_cls.SUPPORTED_SCENE_IDS == supported
    for scene_id in (1197, "1197", "train_1197"):
        with pytest.raises(ValueError, match="train_1197"):
            task_cls(robots="G1FloatingBody", scene_ids=scene_id)


def test_window_desk_cup_composite_has_no_scene_1197_initialization():
    task_cls = HolodeckApproachWindowThenDeskThenPickCup
    assert task_cls.SUPPORTED_SCENE_IDS == (117, 1018, 1664, 2025)
    assert task_cls.COMPOSITE_SCENE_IDS == (1664, 2025)
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        spec = yaml.safe_load(file)["scenes"]["train_1197"]
    assert "approach_window_then_desk_then_pick_cup" not in spec
    assert "composite_robot_pose" not in spec["approach_window"]
    for scene_id in (1197, "1197", "train_1197"):
        with pytest.raises(ValueError, match="train_1197"):
            task_cls(robots="G1FloatingBody", scene_ids=scene_id)
