"""Contracts for the Kitchen low-table bend-and-place task."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen import (
    IThorKitchenBendAndPutAppleOnPlate,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks import composite
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv


@pytest.mark.parametrize("scene_id", (6, 7, 13, 24, 25))
def test_low_table_task_registration_and_layout(scene_id):
    task = IThorKitchenBendAndPutAppleOnPlate
    assert REGISTERED_ENVS[task.__name__] is task
    assert REGISTERED_ITHOR_KITCHEN_ENVS[task.__name__] is task
    assert task.__name__ in composite.__all__
    assert task.TASK_FAMILY == "composite"
    assert task.SUPPORTED_SCENE_IDS == (6, 7, 13, 24, 25)
    assert set(task.SCENE_SPECS) == {
        f"FloorPlan{scene}" for scene in task.SUPPORTED_SCENE_IDS
    }
    spec = task.SCENE_SPECS[f"FloorPlan{scene_id}"]
    assert "lang" not in spec
    assert spec["hold_seconds"] == pytest.approx(1.0)
    assert spec["table_center"][2] == pytest.approx(0.45)
    for name in ("apple", "plate"):
        assert spec[f"{name}_pos"][2] == spec["table_center"][2]
    robot_xy = np.asarray(spec["robot_pose"]["pos"][:2])
    yaw = spec["robot_pose"]["yaw"]
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    apple_xy = np.asarray(spec["apple_pos"][:2])
    plate_xy = np.asarray(spec["plate_pos"][:2])
    assert np.dot(apple_xy - plate_xy, forward) == pytest.approx(0.0, abs=1e-8)
    assert np.linalg.norm(apple_xy - plate_xy) == pytest.approx(0.36)
    assert np.dot(apple_xy - robot_xy, forward) == pytest.approx(0.495)
    assert np.dot(plate_xy - robot_xy, forward) == pytest.approx(0.495)


def test_table_is_visible_when_viewer_hides_collision_group():
    task = object.__new__(IThorKitchenBendAndPutAppleOnPlate)
    task.scene_name = "FloorPlan6"
    task.mujoco_arena = SimpleNamespace(worldbody=ET.Element("worldbody"))
    task.semantic_scene = SimpleNamespace(add_fixture=lambda fixture: None)
    task.fixture_refs = {}
    task.region_refs = {}
    task._setup_task_fixtures()
    geoms = task.mujoco_arena.worldbody.findall("./body/geom")
    assert len(geoms) == 5
    for geom in geoms:
        assert geom.get("group") == "1"
        assert geom.get("contype", "1") == "1"
        assert geom.get("conaffinity", "1") == "1"
        assert float(geom.get("rgba").split()[3]) > 0.0


@pytest.mark.parametrize(
    "contact, horizontal_offset, expected",
    [(True, 0.0, True), (False, 0.0, False), (True, 0.20, False)],
)
def test_success_relation_only_depends_on_apple_on_plate(
    contact, horizontal_offset, expected
):
    task = object.__new__(IThorKitchenBendAndPutAppleOnPlate)
    task.objects = {"apple": object(), "plate": object()}
    task.obj_body_id = {"apple": 0, "plate": 1}
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            body_xpos=np.array(
                [
                    [horizontal_offset, 0.0, 0.51],
                    [0.0, 0.0, 0.46],
                ]
            )
        )
    )
    task.check_contact = lambda apple, plate: contact
    # No robot posture, navigation stage, release state, or timer is supplied.
    assert task._raw_success() is expected


@pytest.mark.parametrize("control_freq", (10, 20, 60))
def test_success_requires_one_continuous_second_and_reset_clears_it(
    monkeypatch, control_freq
):
    monkeypatch.setattr(IThorKitchenEnv, "_post_action", lambda self, action: None)
    task = object.__new__(IThorKitchenBendAndPutAppleOnPlate)
    task.control_freq = control_freq
    task.success_hold_seconds = 1.0
    task._reset_task_progress()
    state = {"on_plate": True}
    task._raw_success = lambda: state["on_plate"]
    for _ in range(control_freq - 1):
        task._post_action(None)
    assert not task._check_success()
    state["on_plate"] = False
    task._post_action(None)
    assert task._success_hold_steps == 0
    state["on_plate"] = True
    for _ in range(control_freq - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    task._reset_task_progress()
    assert not task._check_success()
