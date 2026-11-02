import math
from pathlib import Path

import pytest
import yaml

from robocasa.environments.holodeck.holodeck import REGISTERED_HOLODECK_ENVS
from robocasa.environments.holodeck.tasks.place_box_on_desk import (
    HolodeckPlaceBoxOnDesk,
)


SPECS_PATH = Path(__file__).resolve().parents[1] / (
    "robocasa/models/assets/scenes/holodeck_task_specs.yaml"
)


def test_place_box_on_desk_registration_and_contract():
    assert HolodeckPlaceBoxOnDesk.SUPPORTED_SCENE_IDS == (
        117,
        1018,
        1197,
        1664,
        2025,
    )
    assert HolodeckPlaceBoxOnDesk.BOX_SCALE == pytest.approx(1.5)
    assert HolodeckPlaceBoxOnDesk.HOLD_SECONDS == pytest.approx(1.0)
    assert HolodeckPlaceBoxOnDesk.DESK_BODY == (
        "desk_aeb6ab35aea76da67db0d3cd45d3e821_1_0_0"
    )
    assert HolodeckPlaceBoxOnDesk.__name__ in REGISTERED_HOLODECK_ENVS


def test_place_box_on_desk_spec_uses_desk_and_floor_source():
    specs = yaml.safe_load(SPECS_PATH.read_text())["scenes"]["train_117"]
    spec = specs["place_box_on_desk"]
    assert spec["box_scale"] == pytest.approx(1.5)
    assert spec["table_body"] == HolodeckPlaceBoxOnDesk.DESK_BODY
    assert spec["remove_scene_bodies"] == [
        "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0"
    ]
    assert spec["box_pose"]["pos"][2] > 0.0
    assert spec["robot_pose"]["yaw"] == pytest.approx(-1.5707963267948966)
    assert spec["robot_pose"]["pos"][0] == pytest.approx(
        spec["box_pose"]["pos"][0]
    )
    assert spec["box_pose"]["pos"][1] < spec["robot_pose"]["pos"][1]
    assert spec["box_pose"]["quat"] == pytest.approx([0.5, 0.5, 0.5, 0.5])


def test_place_box_on_desk_success_predicate_only_requires_contact():
    task = object.__new__(HolodeckPlaceBoxOnDesk)
    task._box_contacts_desk = lambda: True
    assert task._box_on_desk() is True


@pytest.mark.parametrize(
    ("scene_name", "robot_pos", "box_pos"),
    (
        ("train_1018", [7.04658770, 0.89, 0.0], [7.09658770, 1.47, 0.1580]),
        ("train_1197", [4.59698800, 2.17987800, 0.0], [4.64698800, 2.75987800, 0.1580]),
        ("train_1664", [5.19, 1.85, 0.0], [5.24, 2.35, 0.1580]),
        ("train_2025", [5.19, 2.15, 0.0], [5.24, 2.65, 0.1580]),
    ),
)
def test_new_scene_box_starts_beside_robot_and_robot_faces_it(
    scene_name, robot_pos, box_pos
):
    spec = yaml.safe_load(SPECS_PATH.read_text())["scenes"][scene_name][
        "place_box_on_desk"
    ]

    assert spec["robot_pose"]["pos"] == pytest.approx(robot_pos)
    assert spec["box_pose"]["pos"] == pytest.approx(box_pos)
    delta_x = box_pos[0] - robot_pos[0]
    delta_y = box_pos[1] - robot_pos[1]
    expected_yaw = math.atan2(delta_y, delta_x)
    assert spec["robot_pose"]["yaw"] == pytest.approx(expected_yaw)
    assert 0.49 <= math.hypot(delta_x, delta_y) <= 0.59
