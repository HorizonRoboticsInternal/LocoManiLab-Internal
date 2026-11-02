import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from robocasa.environments.holodeck.holodeck import REGISTERED_HOLODECK_ENVS
from robocasa.environments.holodeck.tasks.pick_box_from_floor import (
    HolodeckPickBoxFromFloor,
)
REPO_ROOT = Path(__file__).resolve().parents[1]
SPECS_PATH = REPO_ROOT / "robocasa/models/assets/scenes/holodeck_task_specs.yaml"


def test_pick_box_from_floor_contract_and_registration():
    assert "Box_Full/Prefabs/Box_19/Box_19.xml" in HolodeckPickBoxFromFloor.BOX_ASSET
    assert HolodeckPickBoxFromFloor.BOX_SCALE == pytest.approx(2.0)
    assert HolodeckPickBoxFromFloor.MIN_LIFT_DELTA == pytest.approx(0.30)
    assert HolodeckPickBoxFromFloor.HOLD_SECONDS == pytest.approx(1.0)
    assert HolodeckPickBoxFromFloor.SUCCESS_DEBUG_INTERVAL_SECONDS == pytest.approx(
        2.0
    )
    assert HolodeckPickBoxFromFloor.NOISE_FREE_ROBOT_SPAWN is True
    assert HolodeckPickBoxFromFloor.__name__ in REGISTERED_HOLODECK_ENVS


def test_pick_box_success_depends_only_on_height_hold(monkeypatch):
    monkeypatch.setattr(
        "robocasa.environments.holodeck.holodeck.HolodeckEnv._post_action",
        lambda _self, _action: (0.0, False, {}),
    )
    task = object.__new__(HolodeckPickBoxFromFloor)
    task._box_initial_z = 0.0
    task._success_hold = 0
    task.MIN_LIFT_DELTA = 0.30
    task._box_root_z = lambda: 0.31

    task._post_action(SimpleNamespace())
    assert task._success_hold == 1

    task._box_root_z = lambda: 0.05
    task._post_action(SimpleNamespace())
    assert task._success_hold == 0


def test_pick_box_logs_every_two_seconds_and_announces_success(
    monkeypatch, capsys
):
    monkeypatch.setattr(
        "robocasa.environments.holodeck.holodeck.HolodeckEnv._post_action",
        lambda _self, _action: (0.0, False, {}),
    )
    task = object.__new__(HolodeckPickBoxFromFloor)
    task.scene_name = "train_117"
    task.sim = SimpleNamespace(data=SimpleNamespace(time=0.0))
    task.control_freq = 20
    task._box_initial_z = 0.0
    task._success_hold = 0
    task._success_debug_next_log_time = 2.0
    task._success_debug_reported = False
    task._box_root_z = lambda: 0.31

    _, _, info = task._post_action(None)
    assert capsys.readouterr().out == ""
    assert info["success"] is False

    task.sim.data.time = 2.0
    _, _, info = task._post_action(None)
    periodic = capsys.readouterr().out
    assert "当前高度=0.3100m" in periodic
    assert "阈值高度=0.3000m" in periodic
    assert "任务=未成功" in periodic
    assert info["task_success"] is False

    task._success_hold = task.hold_steps - 1
    task.sim.data.time = 2.1
    reward, _, info = task._post_action(None)
    success = capsys.readouterr().out
    assert "任务=成功" in success
    assert reward == 1.0
    assert info["success"] is True


def test_pick_box_from_floor_supports_requested_scenes():
    assert HolodeckPickBoxFromFloor.SUPPORTED_SCENE_IDS == (
        117,
        1018,
        1197,
        1664,
        2025,
    )


@pytest.mark.parametrize(
    ("scene_name", "expected_room"),
    (
        ("train_117", 0),
        ("train_1018", 2),
        ("train_1197", 0),
        ("train_1664", 0),
        ("train_2025", 0),
    ),
)
def test_pick_box_from_floor_specs_are_grounded_and_supported(
    scene_name, expected_room
):
    scenes = yaml.safe_load(SPECS_PATH.read_text())["scenes"]
    spec = scenes[scene_name]["pick_box_from_floor"]

    assert spec["room_id"] == expected_room
    assert spec["box_scale"] == pytest.approx(2.0)
    assert spec["box_pose"]["quat"] == pytest.approx(
        [0.707107, 0.707107, 0.0, 0.0]
    )
    assert spec["box_pose"]["pos"][2] == pytest.approx(0.1580)
    robot_xy = spec["robot_pose"]["pos"][:2]
    box_xy = spec["box_pose"]["pos"][:2]
    assert math.dist(robot_xy, box_xy) == pytest.approx(0.70)
