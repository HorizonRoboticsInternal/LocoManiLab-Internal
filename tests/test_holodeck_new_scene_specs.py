import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import pytest
import yaml

from robocasa.environments.holodeck.tasks.mug_lid import HolodeckCloseMugLid
from robocasa.environments.holodeck.tasks.ice_cube_in_glass import (
    HolodeckIceCubeInGlass,
)
from robocasa.environments.holodeck.tasks.move_cup import (
    HolodeckMoveCupToCenter,
    HolodeckMoveCupToCorner,
)
from robocasa.environments.holodeck.tasks.pick_book_from_dresser import (
    HolodeckPickBookFromDresser,
)
from robocasa.environments.holodeck.tasks.pick_cup_from_desk import (
    HolodeckPickCupFromDesk,
)
from robocasa.environments.holodeck.tasks.place_cup_on_placemat import (
    HolodeckPlaceCupOnPlacemat,
)
from robocasa.environments.holodeck.tasks.pull_chair_from_desk import (
    HolodeckPullChairFromDesk,
)
from robocasa.environments.holodeck.tasks.push_chair_to_desk import (
    HolodeckPushChairToDesk,
)
from robocasa.environments.holodeck.tasks.throw_bottle_into_trash_can import (
    HolodeckThrowBottleIntoTrashCan,
)
from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckOpenDrawerTakeTissueBoxToDesk,
    HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "robocasa/models/assets"
SPECS_PATH = REPO_ROOT / "robocasa/models/assets/scenes/holodeck_task_specs.yaml"
SCENES_DIR = REPO_ROOT / "robocasa/models/assets/scenes/holodeck-objaverse-train"
LAPTOP_26_XML = (
    REPO_ROOT
    / "robocasa/models/assets/objects/thor/Living Room and Bedroom Objects"
    / "Laptop_Full/Prefabs/Laptop_26/Laptop_26.xml"
)


EXPECTED_TASKS = {
    "train_1197": {
        "pick_box_from_floor",
        "approach_window",
        "approach_trash_can",
        "approach_desk",
        "open_laptop",
        "close_laptop",
        "approach_desk_pull_chair_then_open_laptop",
        "move_cup_to_center_then_open_laptop",
        "remove_folder_from_shelf_to_desk_then_open_laptop",
        "approach_desk_then_close_laptop_move_cup",
        "open_drawer",
        "open_drawer_then_take_tissue_box",
        "open_drawer_take_tissue_box_to_desk",
        "close_drawer",
        "open_mug_lid",
        "close_mug_lid",
        "sugar_cube_in_jar",
        "sugar_cube_then_close_jar",
        "open_jar_add_sugar_then_close_jar",
        "place_cup_on_placemat",
        "move_cup_to_center",
        "move_cup_to_corner",
        "push_chair_to_desk",
        "pull_chair_from_desk",
        "pick_book_from_dresser",
        "pick_cup_from_desk",
        "ice_cube_in_glass",
        "ice_cube_then_place_cup",
        "approach_desk_then_pull_chair",
        "throw_bottle_into_trash_can",
        "insert_folder_into_shelf",
        "remove_folder_from_shelf",
        "remove_folder_from_shelf_to_desk",
        "place_box_on_floor",
    },
    "train_1664": {
        "pick_box_from_floor",
        "approach_desk",
        "approach_trash_can",
        "approach_window",
        "open_laptop",
        "close_laptop",
        "approach_window_then_desk_then_pick_cup",
        "approach_desk_pull_chair_then_open_laptop",
        "move_cup_to_center_then_open_laptop",
        "approach_desk_then_close_laptop_move_cup",
        "open_drawer",
        "open_drawer_then_take_tissue_box",
        "close_drawer",
        "open_mug_lid",
        "close_mug_lid",
        "sugar_cube_in_jar",
        "sugar_cube_then_close_jar",
        "open_jar_add_sugar_then_close_jar",
        "place_cup_on_placemat",
        "move_cup_to_center",
        "move_cup_to_corner",
        "push_chair_to_desk",
        "pull_chair_from_desk",
        "pick_book_from_dresser",
        "pick_cup_from_desk",
        "ice_cube_in_glass",
        "ice_cube_then_place_cup",
        "approach_desk_then_pull_chair",
        "throw_bottle_into_trash_can",
        "place_box_on_floor",
    },
    "train_2025": {
        "pick_box_from_floor",
        "approach_desk",
        "approach_trash_can",
        "approach_window",
        "open_laptop",
        "close_laptop",
        "approach_window_then_desk_then_pick_cup",
        "approach_desk_pull_chair_then_open_laptop",
        "move_cup_to_center_then_open_laptop",
        "remove_folder_from_shelf_to_desk_then_open_laptop",
        "approach_desk_then_close_laptop_move_cup",
        "open_drawer",
        "open_drawer_then_take_tissue_box",
        "close_drawer",
        "open_mug_lid",
        "close_mug_lid",
        "sugar_cube_in_jar",
        "sugar_cube_then_close_jar",
        "open_jar_add_sugar_then_close_jar",
        "place_cup_on_placemat",
        "move_cup_to_center",
        "move_cup_to_corner",
        "push_chair_to_desk",
        "pull_chair_from_desk",
        "pick_book_from_dresser",
        "pick_cup_from_desk",
        "ice_cube_in_glass",
        "ice_cube_then_place_cup",
        "approach_desk_then_pull_chair",
        "throw_bottle_into_trash_can",
        "insert_folder_into_shelf",
        "remove_folder_from_shelf",
        "remove_folder_from_shelf_to_desk",
        "place_box_on_floor",
    },
}


def _scene_specs():
    return yaml.safe_load(SPECS_PATH.read_text())["scenes"]


@pytest.mark.parametrize("scene_name, expected", EXPECTED_TASKS.items())
def test_new_holodeck_scene_task_specs_are_complete(scene_name, expected):
    assert set(_scene_specs()[scene_name]) == expected


@pytest.mark.parametrize("scene_name", EXPECTED_TASKS)
def test_new_holodeck_spec_references_exist_in_scene_xml(scene_name):
    root = ET.parse(SCENES_DIR / f"{scene_name}.xml").getroot()
    body_names = {body.get("name") for body in root.iter("body")}
    joint_names = {joint.get("name") for joint in root.iter("joint")}

    for task, spec in _scene_specs()[scene_name].items():
        for key in (
            "source_body",
            "table_body",
            "target_body",
            "target_fixture",
            "window_body",
            "laptop_body",
            "trash_can_body",
        ):
            if key in spec:
                if (
                    key == "source_body"
                    and spec.get("source_table_pose") is not None
                    and spec[key] not in body_names
                ):
                    assert spec[key].startswith(f"holodeck{scene_name[6:]}_")
                    continue
                if spec.get("fixture_mode") == "spawn_asset" and key == "target_body":
                    assert spec[key] == "spawned_laptop_Laptop_26"
                else:
                    assert spec[key] in body_names, (scene_name, task, key)
        if spec.get("fixture_mode") == "spawn_asset" and "source_body" in spec:
            assert spec["source_body"] in body_names
        if "target_joint" in spec:
            if spec.get("fixture_mode") == "spawn_asset":
                assert spec["target_joint"] == (
                    "spawned_laptop_Laptop_26_laptop_lid_26_joint_0"
                )
            else:
                assert spec["target_joint"] in joint_names, (scene_name, task)


@pytest.mark.parametrize("scene_name", ("train_1197", "train_2025"))
def test_folder_specs_reference_the_desk_side_shelf(scene_name):
    specs = _scene_specs()[scene_name]
    insert = specs["insert_folder_into_shelf"]
    remove = specs["remove_folder_from_shelf"]
    expected = {
        "train_1197": (
            "shelf_cb210cde73055d16858af3c54bc507d0_1_0_0",
            "obja_45dadf1d294c461195e92c1cdeb8aabf_1_0_0",
        ),
        "train_2025": (
            "shelf_22fc3a3d40ecdc9abc7852ada2b32b02_2_0_0",
            "obja_42f1d3c373784c9996d35cbf08f2dbbb_1_0_0",
        ),
    }
    expected_shelf, expected_desk = expected[scene_name]
    assert insert["shelf_body"] == remove["shelf_body"]
    assert insert["shelf_body"] == expected_shelf
    assert insert["source_table_body"] == expected_desk
    root = ET.parse(SCENES_DIR / f"{scene_name}.xml").getroot()
    body_names = {body.get("name") for body in root.iter("body")}
    assert expected_shelf in body_names
    assert expected_desk in body_names
    for spec in (insert, remove):
        assert len(spec["region_center"]) == 3
        assert len(spec["region_half_size"]) == 3


@pytest.mark.parametrize("scene_name", ("train_1197", "train_1664", "train_2025"))
def test_new_holodeck_chair_specs_use_task_local_free_joint_asset(scene_name):
    specs = _scene_specs()[scene_name]
    for task_name in ("push_chair_to_desk", "pull_chair_from_desk"):
        spec = specs[task_name]
        assert spec["chair_body"].startswith("holodeck_task_chair_")
        assert spec["chair_joint"].startswith(spec["chair_body"] + "_")
        assert spec["chair_joint"].endswith("_free")
        assert spec["chair_asset"].endswith("Chair_313_1.xml") or spec[
            "chair_asset"
        ].endswith("RoboTHOR_office_chair_allak.xml")
        assert spec["desk_body"] in {
            "obja_45dadf1d294c461195e92c1cdeb8aabf_1_0_0",
            "obja_42f1d3c373784c9996d35cbf08f2dbbb_1_0_0",
        }
        assert len(spec["chair_start_pose"]["pos"]) == 3
        assert len(spec["chair_start_pose"]["quat"]) == 4
        expected_quat = (
            [0.5, 0.5, 0.5, 0.5] if scene_name == "train_1197" else [0.5, 0.5, 0.5, 0.5]
        )
        assert spec["chair_start_pose"]["quat"] == expected_quat


def test_train_117_pull_chair_uses_sliding_asset_and_open_side_calibration():
    spec = _scene_specs()["train_117"]["pull_chair_from_desk"]

    assert spec["fixture_mode"] == "spawn_asset"
    assert spec["chair_asset"].endswith("Chair_313_1.xml")
    assert spec["chair_body"] == "holodeck_task_chair_Chair_313_1"
    assert spec["chair_joint"] == ("holodeck_task_chair_Chair_313_1_free")
    assert spec["remove_scene_bodies"] == [
        "chair_d1ece0a380cf81a8d48fd1ffe684e8c7_1_0_0"
    ]
    assert spec["chair_start_pose"]["pos"] == pytest.approx([4.95, 4.7, 0.429])
    assert spec["robot_pose"]["pos"] == pytest.approx([4.28, 4.7, 0.0])
    assert spec["target_xy"] == pytest.approx([4.2, 4.7])
    assert spec["disable_chair_desk_collisions"] is False
    assert spec["disable_desk_drawer_collisions"] is False


def test_spawned_chair_specs_use_the_asset_free_joint_after_prefixing():
    specs = _scene_specs()
    for scene_specs in specs.values():
        for task_name, spec in scene_specs.items():
            if (
                spec.get("fixture_mode") != "spawn_asset"
                or "chair_asset" not in spec
                or "chair_body" not in spec
            ):
                continue
            root = ET.parse(ASSETS_DIR / spec["chair_asset"]).getroot()
            source_body = root.find("./worldbody/body")
            assert source_body is not None, task_name
            source_joint = source_body.find("./joint[@type='free']")
            assert source_joint is not None, task_name
            source_body_name = source_body.get("name")
            chair_body = spec["chair_body"]
            prefix = chair_body[: -(len(source_body_name) + 1)]
            expected_joint = f"{prefix}_{source_joint.get('name')}"
            assert spec["chair_joint"] == expected_joint, task_name


def test_train_117_place_box_uses_the_calibrated_desk_target():
    specs = _scene_specs()["train_117"]
    place = specs["place_box_on_desk"]

    assert place["table_body"] == ("desk_aeb6ab35aea76da67db0d3cd45d3e821_1_0_0")
    assert place["remove_scene_bodies"] == [
        "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0"
    ]


def test_train_1197_chair_specs_clear_the_desk_and_floor():
    specs = _scene_specs()["train_1197"]
    for task_name in (
        "approach_desk_then_pull_chair",
        "push_chair_to_desk",
        "pull_chair_from_desk",
        "approach_desk_pull_chair_then_open_laptop",
    ):
        assert specs[task_name]["chair_start_pose"]["pos"][2] == pytest.approx(0.429)
        assert specs[task_name]["chair_collision_local_y_offset"] == pytest.approx(
            -0.0036
        )
        assert specs[task_name]["chair_start_pose"]["pos"] == pytest.approx(
            [4.67, 2.10, 0.429]
        )
        assert specs[task_name]["disable_chair_desk_collisions"] is False
        assert specs[task_name]["disable_desk_drawer_collisions"] is False
        if task_name != "push_chair_to_desk":
            assert specs[task_name]["target_xy"] == pytest.approx([4.17, 2.10])


@pytest.mark.parametrize(
    "task_cls",
    (
        HolodeckPickBookFromDresser,
        HolodeckPickCupFromDesk,
        HolodeckPlaceCupOnPlacemat,
        HolodeckIceCubeInGlass,
        HolodeckMoveCupToCenter,
        HolodeckMoveCupToCorner,
        HolodeckPushChairToDesk,
        HolodeckPullChairFromDesk,
        HolodeckThrowBottleIntoTrashCan,
    ),
)
def test_tasks_support_the_three_new_holodeck_scenes(task_cls):
    assert task_cls.SUPPORTED_SCENE_IDS == (117, 1018, 1197, 1664, 2025)


def test_tissue_box_to_desk_workflows_are_limited_to_calibrated_scenes():
    assert HolodeckOpenDrawerTakeTissueBoxToDesk.SUPPORTED_SCENE_IDS == (117, 1197)
    assert HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer.SUPPORTED_SCENE_IDS == (
        117,
        1197,
    )

    specs = _scene_specs()
    for scene_name in ("train_117", "train_1197"):
        assert "open_drawer_take_tissue_box_to_desk" in specs[scene_name]
    for scene_name in ("train_1018", "train_1664", "train_2025"):
        assert "open_drawer_take_tissue_box_to_desk" not in specs[scene_name]


@pytest.mark.parametrize(
    ("scene_name", "expected_color"),
    (
        ("train_117", "black"),
        ("train_1018", "blue"),
        ("train_1197", "black"),
        ("train_1664", "blue"),
        ("train_2025", "blue"),
    ),
)
def test_pull_chair_language_uses_calibrated_scene_color(
    monkeypatch, scene_name, expected_color
):
    monkeypatch.setattr(HolodeckPushChairToDesk, "get_ep_meta", lambda self: {})
    task = object.__new__(HolodeckPullChairFromDesk)
    task.scene_name = scene_name

    assert task.get_ep_meta()["lang"] == (
        f"pull the {expected_color} office chair out from under the desk"
    )


@pytest.mark.parametrize(
    ("chair_xy", "expected"),
    (((0.49, 0.0), False), ((0.50, 0.0), True), ((0.0, 0.51), True)),
)
def test_pull_chair_success_starts_after_reset_displacement(chair_xy, expected):
    task = object.__new__(HolodeckPullChairFromDesk)
    task._pull_origin_xy = np.zeros(2)
    task.chair_body = "chair"

    class Model:
        def body_name2id(self, name):
            assert name == "chair"
            return 0

    class Data:
        body_xpos = np.asarray([(*chair_xy, 0.0)])

    class Sim:
        model = Model()
        data = Data()

    task.sim = Sim()
    assert task._success_position_condition() is expected


@pytest.mark.parametrize(
    ("scene_name", "book_distance", "cup_distance"),
    (
        ("train_1197", 0.6747743946683217, 0.46),
        ("train_1664", 0.5980852782003578, 0.45),
        ("train_2025", 0.4554119013580936, 0.45),
    ),
)
def test_new_holodeck_pick_specs_have_calibrated_source_and_standoff(
    scene_name, book_distance, cup_distance
):
    specs = _scene_specs()[scene_name]
    book = specs["pick_book_from_dresser"]
    cup = specs["pick_cup_from_desk"]

    assert book["source_body"]
    assert len(book["robot_pose"]["pos"]) == 3
    assert len(book["book_pose"]["pos"]) == 3
    assert cup["table_body"]
    if scene_name in ("train_1664", "train_2025"):
        assert cup["support_z"] > 0.0
        assert cup["support_top_z"] > cup["support_z"]

    for spec, source_key, expected_distance in (
        (book, "book_pose", book_distance),
        (cup, "cup_pose", cup_distance),
    ):
        robot_xy = np.asarray(spec["robot_pose"]["pos"][:2], dtype=float)
        source_xy = np.asarray(spec[source_key]["pos"][:2], dtype=float)
        delta = source_xy - robot_xy
        assert np.linalg.norm(delta) == pytest.approx(expected_distance, abs=1e-5)
        expected_yaw = math.atan2(delta[1], delta[0])
        yaw = float(spec["robot_pose"]["yaw"])
        yaw_error = math.atan2(
            math.sin(yaw - expected_yaw), math.cos(yaw - expected_yaw)
        )
        assert yaw_error == pytest.approx(0.0, abs=1e-5)


@pytest.mark.parametrize("scene_name", ("train_1664", "train_2025"))
def test_window_desk_cup_composite_reuses_canonical_support_height(scene_name):
    task = object.__new__(HolodeckPickCupFromDesk)
    task.scene_name = scene_name
    composite_spec = _scene_specs()[scene_name][
        "approach_window_then_desk_then_pick_cup"
    ]
    canonical_spec = _scene_specs()[scene_name]["pick_cup_from_desk"]

    assert "support_z" not in composite_spec
    assert task._support_top_z(composite_spec) == pytest.approx(
        canonical_spec["support_top_z"]
    )


def test_train_1664_window_cup_composite_uses_atomic_window_spawn():
    specs = _scene_specs()["train_1664"]
    assert (
        specs["approach_window_then_desk_then_pick_cup"]["robot_pose"]
        == specs["approach_window"]["robot_pose"]
    )


@pytest.mark.parametrize(
    ("scene_name", "book_z"),
    (("train_1664", 1.08650), ("train_2025", 0.93420)),
)
def test_dynamic_books_use_measured_dresser_contact_heights(scene_name, book_z):
    spec = _scene_specs()[scene_name]["pick_book_from_dresser"]
    assert spec["book_pose"]["pos"][2] == pytest.approx(book_z)


def test_train_2025_pick_book_keeps_the_side_cabinet_and_its_decor_supported():
    spec = _scene_specs()["train_2025"]["pick_book_from_dresser"]
    cabinet = "stand_dba5fb19566f0e1caaf78947145fe7fc_1_0_0"
    assert cabinet not in spec["remove_scene_bodies"]

    metadata = yaml.safe_load((SCENES_DIR / "train_2025_metadata.json").read_text())
    cabinet_metadata = metadata["objects"][cabinet]
    assert {
        "candle_c6e5d947796037dc5ccc452b1320f772_1_0_0",
        "candle_e556573454d609117d1a666932ebfcb2_1_0_0",
    }.issubset(cabinet_metadata["children"])


@pytest.mark.parametrize("scene_name", ("train_1197", "train_1664", "train_2025"))
def test_new_holodeck_books_are_upright_with_spines_toward_robot(scene_name):
    spec = _scene_specs()[scene_name]["pick_book_from_dresser"]
    w, x, y, z = np.asarray(spec["book_pose"]["quat"], dtype=float)
    rotation = np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)),
        )
    )
    assert rotation @ np.array([0.0, 0.0, 1.0]) == pytest.approx(
        [0.0, 0.0, -1.0], abs=1e-6
    )

    spine = rotation @ np.array([1.0, 0.0, 0.0])
    to_robot = np.asarray(spec["robot_pose"]["pos"][:2], dtype=float) - np.asarray(
        spec["book_pose"]["pos"][:2], dtype=float
    )
    to_robot /= np.linalg.norm(to_robot)
    assert np.dot(spine[:2], to_robot) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("scene_name", ("train_1197", "train_1664", "train_2025"))
def test_new_holodeck_cup_specs_share_a_valid_desk_layout(scene_name):
    specs = _scene_specs()[scene_name]
    place = specs["place_cup_on_placemat"]
    center = specs["move_cup_to_center"]
    assert place["table_body"] == center["desk_body"]
    assert place["cup_scale"] == pytest.approx(center["cup_scale"])
    assert place["support_z"] == pytest.approx(center["support_z"])
    assert center["target_base_xy"][0] > center["tabletop_bounds_xy"][0][0]
    assert center["target_base_xy"][0] < center["tabletop_bounds_xy"][1][0]
    assert center["target_base_xy"][1] > center["tabletop_bounds_xy"][0][1]
    assert center["target_base_xy"][1] < center["tabletop_bounds_xy"][1][1]
    target = center["target_base_xy"]
    initial = center.get("initial_cup_base_xy")
    if initial is None:
        source = center["cup_pose"]["pos"][:2]
        initial = [(s + t) / 2.0 for s, t in zip(source, target)]
    assert any(
        abs(value - expected) > tolerance
        for value, expected, tolerance in zip(
            initial, target, center["target_tolerance"]
        )
    )


@pytest.mark.parametrize("scene_name", ("train_1197", "train_1664", "train_2025"))
def test_new_holodeck_ice_specs_share_the_calibrated_desk_layout(scene_name):
    specs = _scene_specs()[scene_name]
    ice = specs["ice_cube_in_glass"]
    cup = specs["pick_cup_from_desk"]
    assert ice["table_body"] == cup["table_body"]
    assert ice["object_scale"] == pytest.approx(cup["object_scale"])
    assert ice["robot_pose"]["pos"]
    assert ice["cup_pose"]["pos"]
    assert ice["ice_pose"]["pos"]
    robot = np.asarray(ice["robot_pose"]["pos"][:2], dtype=float)
    for key in ("cup_pose", "ice_pose"):
        source = np.asarray(ice[key]["pos"][:2], dtype=float)
        assert np.linalg.norm(source - robot) == pytest.approx(0.45, abs=1e-5)


@pytest.mark.parametrize(
    ("scene_name", "expected_target", "expected_bounds"),
    (
        (
            "train_1197",
            [5.35, 2.10],
            [[5.19655, 1.12490], [5.71283, 3.06378]],
        ),
        (
            "train_1664",
            [5.76283, 1.85],
            [[5.61283, 1.12557], [5.91283, 2.75256]],
        ),
        (
            "train_2025",
            [5.73182, 2.17275],
            [[5.56364, 1.32818], [5.90000, 3.01733]],
        ),
    ),
)
def test_new_holodeck_cup_targets_match_measured_desk_support(
    scene_name, expected_target, expected_bounds
):
    center = _scene_specs()[scene_name]["move_cup_to_center"]
    assert center["target_base_xy"] == pytest.approx(expected_target)
    assert np.allclose(center["tabletop_bounds_xy"], expected_bounds)


@pytest.mark.parametrize(
    "scene_name",
    (
        "train_117",
        "train_1018",
        "train_1197",
        "train_1664",
        "train_2025",
    ),
)
def test_center_cup_starts_to_robot_right_of_target_with_10cm_clearance(scene_name):
    center = _scene_specs()[scene_name]["move_cup_to_center"]
    target = np.asarray(center["target_base_xy"], dtype=float)
    initial = np.asarray(center["initial_cup_base_xy"], dtype=float)
    tolerance = np.asarray(center["target_tolerance"], dtype=float)

    # Compare the source/target displacement in the robot's viewing frame:
    # the cup must be to the robot's right, not in front of or behind the
    # target region.
    yaw = float(center["robot_pose"]["yaw"])
    view_axis = np.array([math.cos(yaw), math.sin(yaw)])
    right_axis = np.array([view_axis[1], -view_axis[0]])
    separation = initial - target
    if scene_name == "train_117":
        assert initial == pytest.approx([5.50, 4.40], abs=1e-6)
        assert target[1] - tolerance[1] - initial[1] == pytest.approx(0.10, abs=1e-6)
    else:
        assert np.dot(separation, right_axis) > 0.0
        # The calibrated desk long axes are not all parallel to the robot
        # view axis. Match train_117's physical layout while allowing the
        # measured longitudinal offset (up to 20 cm).
        assert abs(np.dot(separation, view_axis)) <= 0.20 + 1e-6

    # ``target_tolerance`` is the half-size of the success rectangle. Measure
    # the clearance from the rectangle boundary, not just from its center.
    outside = np.maximum(np.abs(separation) - tolerance, 0.0)
    assert np.linalg.norm(outside) >= 0.10 - 1e-6

    # Keep the whole visible target region on the measured tabletop support.
    bounds = np.asarray(center["tabletop_bounds_xy"], dtype=float)
    assert np.all(target - tolerance >= bounds[0] - 1e-6)
    assert np.all(target + tolerance <= bounds[1] + 1e-6)


@pytest.mark.parametrize(
    "scene_name",
    (
        "train_117",
        "train_1018",
        "train_1197",
        "train_1664",
        "train_2025",
    ),
)
def test_center_cup_starts_45cm_away_and_facing_the_cup(scene_name):
    center = _scene_specs()[scene_name]["move_cup_to_center"]
    robot = np.asarray(center["robot_pose"]["pos"][:2], dtype=float)
    cup = np.asarray(center["initial_cup_base_xy"], dtype=float)
    delta = cup - robot
    if scene_name == "train_117":
        assert robot == pytest.approx([5.0, 4.4], abs=1e-6)
        assert float(center["robot_pose"]["yaw"]) == pytest.approx(0.0, abs=1e-6)
        assert delta[1] == pytest.approx(0.0, abs=1e-6)
        assert delta[0] > 0.0
        return
    distance_tolerance = 3e-4 if scene_name == "train_1664" else 1e-5
    assert np.linalg.norm(delta) == pytest.approx(0.45, abs=distance_tolerance)
    expected_yaw = math.atan2(delta[1], delta[0])
    yaw = float(center["robot_pose"]["yaw"])
    yaw_error = math.atan2(math.sin(yaw - expected_yaw), math.cos(yaw - expected_yaw))
    yaw_tolerance = 0.04 if scene_name == "train_1664" else 1e-5
    assert yaw_error == pytest.approx(0.0, abs=yaw_tolerance)


def test_corner_task_has_its_own_calibration_for_all_supported_scenes():
    assert HolodeckMoveCupToCorner.CALIBRATION_SPEC_TASK == "move_cup_to_corner"
    for scene_name in (
        "train_117",
        "train_1018",
        "train_1197",
        "train_1664",
        "train_2025",
    ):
        assert "move_cup_to_corner" in _scene_specs()[scene_name]


@pytest.mark.parametrize("scene_name", ("train_1664", "train_2025"))
def test_move_cup_desk_body_is_not_dependent_on_missing_category(scene_name):
    root = ET.parse(SCENES_DIR / f"{scene_name}.xml").getroot()
    desk_body = _scene_specs()[scene_name]["move_cup_to_center"]["desk_body"]
    body = root.find(f".//body[@name='{desk_body}']")
    assert body is not None

    metadata = yaml.safe_load((SCENES_DIR / f"{scene_name}_metadata.json").read_text())
    object_metadata = metadata["objects"][desk_body]
    assert object_metadata["category"] == ""


@pytest.mark.parametrize(
    ("scene_name", "expected_targets"),
    (
        (
            "train_1197",
            [[5.31655, 1.24490], [5.31655, 2.94378]],
        ),
        (
            "train_1664",
            [[5.66722, 0.91154], [5.66722, 2.71728]],
        ),
        (
            "train_2025",
            [[5.66709, 1.43340], [5.66709, 2.78908]],
        ),
    ),
)
def test_corner_targets_follow_each_desk_long_axis(scene_name, expected_targets):
    explicit_targets = _scene_specs()[scene_name]["move_cup_to_corner"].get(
        "corner_target_base_xys"
    )
    if explicit_targets is not None:
        assert np.allclose(explicit_targets, expected_targets)
        return

    corner = _scene_specs()[scene_name]["move_cup_to_corner"]
    bounds = np.asarray(corner["tabletop_bounds_xy"], dtype=float)
    tabletop_center = bounds.mean(axis=0)
    long_direction = np.array([0.0, -1.0])
    short_direction = np.array([1.0, 0.0])
    long_min, long_max = sorted(np.dot(bounds - tabletop_center, long_direction))
    short_min, short_max = sorted(np.dot(bounds - tabletop_center, short_direction))
    target_long_half = target_short_half = 0.12
    robot_side = np.sign(
        np.dot(
            np.asarray(corner["robot_pose"]["pos"][:2]) - tabletop_center,
            short_direction,
        )
    )
    near_short_center = (
        short_max - target_short_half
        if robot_side > 0
        else short_min + target_short_half
    )
    actual_targets = [
        tabletop_center
        + long_center * long_direction
        + near_short_center * short_direction
        for long_center in (long_max - target_long_half, long_min + target_long_half)
    ]
    assert np.allclose(actual_targets, expected_targets)


@pytest.mark.parametrize(
    "scene_name",
    (
        "train_117",
        "train_1018",
        "train_1197",
        "train_1664",
        "train_2025",
    ),
)
def test_corner_cup_starts_left_of_the_right_target_with_5cm_clearance(
    scene_name,
):
    corner = _scene_specs()[scene_name]["move_cup_to_corner"]
    robot = np.asarray(corner["robot_pose"]["pos"][:2], dtype=float)
    source = np.asarray(corner["initial_cup_base_xy"], dtype=float)
    targets = np.asarray(corner["corner_target_base_xys"], dtype=float)
    yaw = float(corner["robot_pose"]["yaw"])
    view_axis = np.array([math.cos(yaw), math.sin(yaw)])
    left_axis = np.array([-view_axis[1], view_axis[0]])

    if scene_name == "train_117":
        assert robot == pytest.approx([5.0, 4.1], abs=1e-6)
        assert source == pytest.approx([5.50, 4.1], abs=1e-6)
        assert float(corner["robot_pose"]["yaw"]) == pytest.approx(0.0, abs=1e-6)
        return

    # The target with the most negative left-axis offset is on the robot's
    # right. Keep the source clearly beyond that region's left boundary.
    right_target = targets[np.argmin((targets - robot) @ left_axis)]
    offset = source - right_target
    assert np.dot(offset, left_axis) >= 0.20 - 1e-6
    expected_distance = 0.50 if scene_name == "train_117" else 0.45
    assert np.linalg.norm(source - robot) >= expected_distance - 1e-6
    expected_yaw = math.atan2(*(source - robot)[::-1])
    yaw_error = math.atan2(math.sin(yaw - expected_yaw), math.cos(yaw - expected_yaw))
    assert yaw_error == pytest.approx(0.0, abs=1e-6)


def test_cup_target_region_visualizations_are_disabled():
    assert HolodeckMoveCupToCenter.SHOW_TARGET_REGION is False
    assert HolodeckMoveCupToCorner.SHOW_TARGET_REGION is False
    assert HolodeckMoveCupToCorner.SHOW_CORNER_TARGET_REGION is False
    assert (
        HolodeckMoveCupToCenter.TARGET_REGION_SITE_NAME
        != HolodeckMoveCupToCorner.TARGET_REGION_SITE_NAME
    )


@pytest.mark.parametrize("scene_name", EXPECTED_TASKS)
def test_new_holodeck_window_spawn_faces_target_and_starts_far_away(scene_name):
    root = ET.parse(SCENES_DIR / f"{scene_name}.xml").getroot()
    spec = _scene_specs()[scene_name]["approach_window"]
    target = root.find(f".//body[@name='{spec['window_body']}']")
    assert target is not None
    target_xy = tuple(float(value) for value in target.get("pos").split()[:2])
    robot_xy = tuple(float(value) for value in spec["robot_pose"]["pos"][:2])
    expected_yaw = math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0])
    yaw_error = math.atan2(
        math.sin(float(spec["robot_pose"]["yaw"]) - expected_yaw),
        math.cos(float(spec["robot_pose"]["yaw"]) - expected_yaw),
    )

    assert abs(yaw_error) < 1e-6
    assert math.dist(robot_xy, target_xy) > float(spec["success_radius"]) + 0.15


@pytest.mark.parametrize(
    ("scene_name", "desk_body", "expected_pose"),
    (
        (
            "train_1664",
            "obja_42f1d3c373784c9996d35cbf08f2dbbb_1_0_0",
            [4.00, 1.80, 0.0],
        ),
        (
            "train_2025",
            "obja_42f1d3c373784c9996d35cbf08f2dbbb_1_0_0",
            [3.50, 1.95, 0.0],
        ),
    ),
)
def test_new_holodeck_desk_specs_reference_baked_desk_and_start_far_away(
    scene_name, desk_body, expected_pose
):
    spec = _scene_specs()[scene_name]["approach_desk"]
    root = ET.parse(SCENES_DIR / f"{scene_name}.xml").getroot()
    desk = root.find(f".//body[@name='{desk_body}']")

    assert spec["desk_body"] == desk_body
    assert spec["robot_pose"]["pos"] == pytest.approx(expected_pose)
    assert desk is not None
    desk_xy = tuple(float(value) for value in desk.get("pos").split()[:2])
    expected_yaw = math.atan2(
        desk_xy[1] - spec["robot_pose"]["pos"][1],
        desk_xy[0] - spec["robot_pose"]["pos"][0],
    )
    assert spec["robot_pose"]["yaw"] == pytest.approx(expected_yaw, abs=1e-4)
    assert math.dist(spec["robot_pose"]["pos"][:2], desk_xy) > (
        spec["success_radius"] + 0.15
    )


@pytest.mark.parametrize("scene_name", EXPECTED_TASKS)
def test_holodeck_drawer_specs_use_target_state_and_one_second_hold(scene_name):
    scene = _scene_specs()[scene_name]
    for task_name in ("open_drawer", "close_drawer"):
        spec = scene[task_name]
        expected_init = (
            1.0
            if task_name == "close_drawer"
            else (0.0 if scene_name == "train_117" else 0.20)
        )
        assert spec["init_open_fraction"] == pytest.approx(expected_init)
        threshold_key = (
            "success_open_fraction"
            if task_name == "open_drawer"
            else "success_close_fraction"
        )
        expected_threshold = (
            (0.90 if scene_name == "train_117" else 0.80)
            if task_name == "open_drawer"
            else 0.10
        )
        assert spec[threshold_key] == pytest.approx(expected_threshold)
        assert spec["hold_steps"] == 10


@pytest.mark.parametrize(
    ("scene_name", "expected_joint", "expected_open_pose", "expected_close_pose"),
    (
        (
            "train_1197",
            "chestofdrawers_22b31e6c8819e16e5b2ab4f7bfe7202c_1_16_0_DrawerMesh041_7",
            {"pos": [4.65, 3.773, 0.0], "yaw": 0.0},
            {"pos": [4.65, 3.773, 0.0], "yaw": 0.0},
        ),
        (
            "train_1664",
            "chestofdrawers_c86848ac21624c499f4665989ac4f01a_1_20_0_DrawerMesh021_9",
            {"pos": [3.70, 4.65, 0.0], "yaw": 1.5707963267948966},
            {"pos": [3.70, 4.65, 0.0], "yaw": 1.5707963267948966},
        ),
        (
            "train_2025",
            "stand_dba5fb19566f0e1caaf78947145fe7fc_2_2_0_joint_0",
            {"pos": [3.30, 0.90, 0.0], "yaw": -1.5798708501714973},
            {"pos": [3.30, 1.20, 0.0], "yaw": -1.5816852989136816},
        ),
    ),
)
def test_new_holodeck_drawers_use_the_calibrated_ergonomic_candidates(
    scene_name, expected_joint, expected_open_pose, expected_close_pose
):
    scene = _scene_specs()[scene_name]
    assert scene["open_drawer"]["target_joint"] == expected_joint
    assert scene["close_drawer"]["target_joint"] == expected_joint
    assert scene["open_drawer"]["robot_pose"] == pytest.approx(expected_open_pose)
    assert scene["close_drawer"]["robot_pose"] == pytest.approx(expected_close_pose)


@pytest.mark.parametrize(
    "scene_name", ("train_117", "train_1018", "train_1197", "train_1664", "train_2025")
)
def test_holodeck_drawer_composite_keeps_its_own_success_threshold(scene_name):
    scene = _scene_specs()[scene_name]
    composite = scene["open_drawer_then_take_tissue_box"]
    atomic = scene["open_drawer"]
    assert composite["init_open_fraction"] == pytest.approx(
        0.0 if scene_name == "train_117" else 0.20
    )
    assert composite["init_open_fraction"] == pytest.approx(
        atomic["init_open_fraction"]
    )
    assert atomic["success_open_fraction"] == pytest.approx(
        0.90 if scene_name == "train_117" else 0.80
    )
    assert composite["success_open_fraction"] == pytest.approx(0.90)
    assert composite["hold_steps"] == 10


@pytest.mark.parametrize(
    ("scene_name", "expected_pose"),
    (
        ("train_1664", [5.70623, 1.65, 0.835000]),
        (
            "train_2025",
            [5.706100000000000, 1.950000000000000, 0.855529],
        ),
    ),
)
def test_new_holodeck_laptop_specs_have_task_local_collision_safe_pose(
    scene_name, expected_pose
):
    specs = _scene_specs()[scene_name]
    root = ET.parse(SCENES_DIR / f"{scene_name}.xml").getroot()
    laptop_spec = specs["open_laptop"]
    if laptop_spec.get("fixture_mode") == "spawn_asset":
        laptop_body = root.find(f".//body[@name='{laptop_spec['source_body']}']")
    else:
        laptop_body = root.find(f".//body[@name='{laptop_spec['target_body']}']")
    assert laptop_body is not None

    for task in ("open_laptop", "close_laptop"):
        pose = specs[task]["laptop_pose"]
        assert pose["pos"] == pytest.approx(expected_pose)
        if specs[task].get("fixture_mode") != "spawn_asset":
            assert pose["quat"] == pytest.approx(
                [float(value) for value in laptop_body.get("quat").split()]
            )

    if specs["open_laptop"].get("fixture_mode") != "spawn_asset":
        baked_pos = [float(value) for value in laptop_body.get("pos").split()]
        assert specs["open_laptop"]["laptop_pose"]["pos"] != pytest.approx(baked_pos)


def test_train_1664_move_cup_then_open_laptop_matches_atomic_center_layout():
    specs = _scene_specs()["train_1664"]
    composite = specs["move_cup_to_center_then_open_laptop"]
    center = specs["move_cup_to_center"]
    open_laptop = specs["open_laptop"]

    assert composite["robot_pose"] == center["robot_pose"]
    assert composite["cup_pose"] == center["cup_pose"]
    assert composite["initial_cup_base_xy"] == center["initial_cup_base_xy"]
    assert composite["target_base_xy"] == center["target_base_xy"]
    assert composite["target_tolerance"] == center["target_tolerance"]
    assert composite["tabletop_bounds_xy"] == center["tabletop_bounds_xy"]
    assert composite["support_z"] == center["support_z"]
    expected_laptop_pos = [
        open_laptop["laptop_pose"]["pos"][0],
        2.45,
        open_laptop["laptop_pose"]["pos"][2],
    ]
    assert composite["laptop_pose"]["pos"] == pytest.approx(expected_laptop_pos)
    assert composite["laptop_pose"]["quat"] == open_laptop["laptop_pose"]["quat"]
    assert composite["laptop_support_geom"]["pos"] == pytest.approx(
        [-0.65, 0.383, 0.05536]
    )
    assert (
        composite["laptop_support_geom"]["size"]
        == open_laptop["laptop_support_geom"]["size"]
    )


def test_train_2025_folder_laptop_composite_places_laptop_at_desk_center():
    specs = _scene_specs()["train_2025"]
    composite = specs["remove_folder_from_shelf_to_desk_then_open_laptop"]

    assert composite["laptop_pose"]["pos"] == pytest.approx(
        [5.73182, 2.17275, 0.855529]
    )
    assert composite["laptop_support_geom"]["pos"] == pytest.approx(
        [-0.07275, 0.383, 0.02964]
    )


@pytest.mark.parametrize(
    ("scene_name", "expected_body", "expected_pose"),
    (
        (
            "train_1664",
            "laptop_f09fcf086ebe9e3ab2d97bff50b833ef_1_0_0",
            [5.70623, 1.65, 0.835000],
        ),
        (
            "train_2025",
            "laptop_63fbcd680e0db93dfb0454951f2114f3_1_0_0",
            [5.706100000000000, 1.950000000000000, 0.855529],
        ),
    ),
)
def test_new_holodeck_approach_desk_reuses_collision_safe_laptop_pose(
    scene_name, expected_body, expected_pose
):
    spec = _scene_specs()[scene_name]["approach_desk"]
    assert spec["source_body"] == expected_body
    assert spec["fixture_mode"] == "spawn_asset"
    assert spec["laptop_pose"]["pos"] == pytest.approx(expected_pose)
    assert spec["laptop_pose"]["quat"] == pytest.approx(
        _scene_specs()[scene_name]["open_laptop"]["laptop_pose"]["quat"]
    )


@pytest.mark.parametrize(
    ("scene_name", "expected_robot_pose"),
    (
        ("train_117", {"pos": [4.99989971, 4.0661376, 0.0], "yaw": 0.0}),
        ("train_1018", {"pos": [7.04658770, 0.89, 0.0], "yaw": 0.0}),
        ("train_1197", {"pos": [4.76599120, 2.10, 0.0], "yaw": 0.0}),
        ("train_1664", {"pos": [5.18516845, 1.65, 0.0], "yaw": 0.0}),
        ("train_2025", {"pos": [5.18516846, 1.95, 0.0], "yaw": 0.0}),
    ),
)
def test_close_laptop_uses_the_calibrated_collision_standoff(
    scene_name, expected_robot_pose
):
    specs = _scene_specs()[scene_name]
    actual = specs["close_laptop"]["robot_pose"]
    assert actual["pos"] == pytest.approx(expected_robot_pose["pos"])
    assert actual["yaw"] == pytest.approx(expected_robot_pose["yaw"])


def test_train_2025_close_mug_lid_clears_laptop_and_levels_detached_lid():
    spec = _scene_specs()["train_2025"]["close_mug_lid"]

    assert "train_2025" in HolodeckCloseMugLid.LAPTOP_CLEARED_SCENES
    # Keep the lid on the robot-facing side at the same approximate reach.
    assert spec["open_lid_pose"]["pos"][1] < spec["mug_pose"]["pos"][1]
    robot_xy = np.asarray(spec["robot_pose"]["pos"][:2], dtype=float)
    lid_xy = np.asarray(spec["open_lid_pose"]["pos"][:2], dtype=float)
    assert math.dist(robot_xy, lid_xy) == pytest.approx(0.45, abs=0.01)


@pytest.mark.parametrize(
    "scene_name",
    (
        "train_117",
        "train_1018",
        "train_1197",
        "train_1664",
        "train_2025",
    ),
)
def test_close_mug_lid_starts_with_jar_and_lid_side_by_side(scene_name):
    spec = _scene_specs()[scene_name]["close_mug_lid"]
    robot_pose = spec["robot_pose"]
    forward = np.array(
        [
            math.cos(float(robot_pose["yaw"])),
            math.sin(float(robot_pose["yaw"])),
        ]
    )
    left = np.array([-forward[1], forward[0]])
    separation = np.asarray(spec["open_lid_pose"]["pos"][:2]) - np.asarray(
        spec["mug_pose"]["pos"][:2]
    )

    assert abs(np.dot(separation, left)) > 0.10
    assert abs(np.dot(separation, forward)) < 0.05


@pytest.mark.parametrize(
    ("scene_name", "expected_robot_pose"),
    (
        ("train_117", {"pos": [4.57380371, 1.45, 0.0], "yaw": 0.0}),
        ("train_1018", {"pos": [3.81892754, 1.25, 0.0], "yaw": math.pi}),
        ("train_1197", {"pos": [4.85, 2.10, 0.0], "yaw": 0.0}),
        ("train_1664", {"pos": [5.25, 1.45, 0.0], "yaw": 0.0}),
        ("train_2025", {"pos": [5.25, 1.45, 0.0], "yaw": 0.0}),
    ),
)
def test_close_mug_lid_keeps_robot_standoff_and_object_depth(
    scene_name, expected_robot_pose
):
    spec = _scene_specs()[scene_name]["close_mug_lid"]
    assert spec["robot_pose"] == pytest.approx(expected_robot_pose)

    robot_xy = np.asarray(spec["robot_pose"]["pos"][:2], dtype=float)
    forward = np.array(
        [
            math.cos(float(spec["robot_pose"]["yaw"])),
            math.sin(float(spec["robot_pose"]["yaw"])),
        ]
    )
    for pose_key in ("mug_pose", "open_lid_pose"):
        object_xy = np.asarray(spec[pose_key]["pos"][:2], dtype=float)
        assert math.dist(robot_xy, object_xy) == pytest.approx(0.45, abs=0.01)


@pytest.mark.parametrize(
    ("scene_name", "task", "expected_closed", "expected_open"),
    (
        ("train_1197", "open_drawer", 0.0, 0.172),
        ("train_1664", "open_drawer", 0.0, 0.466),
        ("train_2025", "open_drawer", 0.0, 0.381),
        ("train_1664", "open_laptop", 1.5708, 0.0),
        ("train_2025", "open_laptop", 1.5708, 0.0),
    ),
)
def test_new_holodeck_articulation_endpoints_match_xml(
    scene_name, task, expected_closed, expected_open
):
    spec = _scene_specs()[scene_name][task]
    if spec.get("fixture_mode") == "spawn_asset":
        root = ET.parse(LAPTOP_26_XML).getroot()
        joint_name = "Laptop_26_laptop_lid_26_joint_0"
    else:
        root = ET.parse(SCENES_DIR / f"{scene_name}.xml").getroot()
        joint_name = spec["target_joint"]
    joint = root.find(f".//joint[@name='{joint_name}']")
    assert joint is not None
    lower, upper = (float(value) for value in joint.get("range").split())

    assert spec["closed_qpos"] == pytest.approx(expected_closed)
    assert spec["open_qpos"] == pytest.approx(expected_open)
    assert lower <= spec["closed_qpos"] <= upper
    assert lower <= spec["open_qpos"] <= upper


def test_train_1197_spawned_laptop_is_near_the_robot_side_of_the_desk():
    spec = _scene_specs()["train_1197"]["open_laptop"]
    assert spec["fixture_mode"] == "spawn_asset"
    assert spec["laptop_pose"]["pos"] == pytest.approx([5.300, 2.100, 0.8491])
    assert spec["laptop_pose"]["quat"] == pytest.approx([0.5, 0.5, -0.5, -0.5])
    assert spec["laptop_front"] == pytest.approx([-1.0, 0.0])
    assert spec["desk_body"] == "obja_45dadf1d294c461195e92c1cdeb8aabf_1_0_0"
    assert spec["laptop_support_geom"]["size"] == pytest.approx([0.20, 0.005, 0.22])
    assert spec["laptop_support_geom"]["pos"] == pytest.approx([0.0, 0.3894, 0.1644])


def test_train_1197_desk_tasks_use_the_open_long_side():
    specs = _scene_specs()["train_1197"]
    assert specs["approach_desk"]["robot_pose"] == {
        "pos": [4.00, 2.10, 0.0],
        "yaw": 0.0,
    }
    for task in ("open_laptop", "close_laptop"):
        pose = specs[task]["robot_pose"]
        assert pose["pos"] == pytest.approx([4.76599120, 2.10, 0.0])
        assert pose["yaw"] == pytest.approx(0.0)


def test_train_1197_uses_the_articulated_desk_313_2_asset():
    metadata = yaml.safe_load((SCENES_DIR / "train_1197_metadata.json").read_text())
    desk = metadata["objects"]["obja_45dadf1d294c461195e92c1cdeb8aabf_1_0_0"]
    assert desk["asset_id"] == "Desk_313_2"
    assert desk["category"] == "Desk"

    model = mujoco.MjModel.from_xml_path(str(SCENES_DIR / "train_1197.xml"))
    drawer_joints = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if "desk_aeb6ab35aea76da67db0d3cd45d3e821_1_"
        in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "")
    ]
    assert len(drawer_joints) == 6
    for joint_name in drawer_joints:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert model.qpos0[model.jnt_qposadr[joint_id]] == pytest.approx(0.0)


def test_train_1197_dresser_articulations_compile_with_closed_source_zero_state():
    model = mujoco.MjModel.from_xml_path(str(SCENES_DIR / "train_1197.xml"))
    joint_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        for joint_id in range(model.njnt)
        if "chestofdrawers_22b31e6c8819e16e5b2ab4f7bfe7202c_"
        in (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "")
        and int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE)
    ]

    assert len(joint_names) == 8
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert model.qpos0[model.jnt_qposadr[joint_id]] == pytest.approx(0.0)
