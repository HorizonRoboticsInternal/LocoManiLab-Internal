from types import SimpleNamespace

import pytest
import yaml

from robocasa.environments.holodeck.holodeck import REGISTERED_HOLODECK_ENVS
from robocasa.environments.holodeck.tasks.place_box_on_floor import (
    HolodeckPlaceBoxFromDeskToFloor,
    HolodeckPlaceBoxOnFloor,
)


def test_place_box_on_floor_contract_and_registration():
    assert HolodeckPlaceBoxOnFloor.BOX_SCALE == pytest.approx(2.0)
    assert HolodeckPlaceBoxOnFloor.HOLD_SECONDS == pytest.approx(1.0)
    assert HolodeckPlaceBoxOnFloor.SUPPORTED_SCENE_IDS == (117, 1018, 1197, 1664, 2025)
    assert HolodeckPlaceBoxOnFloor.__name__ in REGISTERED_HOLODECK_ENVS
    assert HolodeckPlaceBoxFromDeskToFloor is HolodeckPlaceBoxOnFloor


def test_place_box_success_requires_continuous_floor_contact(monkeypatch):
    monkeypatch.setattr(
        "robocasa.environments.holodeck.holodeck.HolodeckEnv._post_action",
        lambda _self, _action: (0.0, False, {}),
    )
    task = object.__new__(HolodeckPlaceBoxOnFloor)
    task.control_freq = 20
    task._success_hold = 0
    task._box_contacts_floor = lambda: True

    for _ in range(task.hold_steps - 1):
        reward, _, info = task._post_action(SimpleNamespace())
        assert reward == 0.0
        assert info["success"] is False
    reward, _, info = task._post_action(SimpleNamespace())
    assert reward == 1.0
    assert info["success"] is True

    task._box_contacts_floor = lambda: False
    reward, _, info = task._post_action(SimpleNamespace())
    assert reward == 0.0
    assert info["success_hold"] == 0


def test_place_box_scene_specs_are_complete():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        scenes = yaml.safe_load(file)["scenes"]
    for scene_id in HolodeckPlaceBoxOnFloor.SUPPORTED_SCENE_IDS:
        spec = scenes[f"train_{scene_id}"]["place_box_on_floor"]
        expected_scale = 1.5 if scene_id in (1664, 2025) else 2.0
        assert spec["box_scale"] == pytest.approx(expected_scale)
        assert spec["box_pose"]["quat"] == pytest.approx(
            [0.707107, 0.707107, 0.0, 0.0]
        )
        assert spec["table_body"]
        assert len(spec["robot_pose"]["pos"]) == 3

    # The larger box needs a wider initial robot standoff in these two scenes.
    assert scenes["train_117"]["place_box_on_floor"]["robot_pose"]["pos"][0] == pytest.approx(
        4.74990582
    )
    assert scenes["train_1197"]["place_box_on_floor"]["robot_pose"]["pos"][0] == pytest.approx(
        4.59698800
    )
    # Scene 1018 needs additional clearance for G1's torso and the box flap.
    assert scenes["train_1018"]["place_box_on_floor"]["robot_pose"]["pos"] == pytest.approx(
        [6.85, 0.89, 0.0]
    )
    assert scenes["train_1664"]["place_box_on_floor"]["box_pose"]["pos"] == pytest.approx(
        [5.73182, 1.85, 0.9586]
    )
    assert scenes["train_2025"]["place_box_on_floor"]["box_pose"]["pos"] == pytest.approx(
        [5.73182, 2.15, 0.9788]
    )
