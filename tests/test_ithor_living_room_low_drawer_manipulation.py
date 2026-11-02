"""Scene and stage contracts for closing low drawers and apple storage."""

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from robosuite.environments.base import REGISTERED_ENVS

import robocasa.environments.ithor_living_room as room
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.tasks.composite.walk_and_close_low_drawer import (
    IThorLivingRoomWalkAndCloseLowDrawer as Close,
)
from robocasa.environments.ithor_living_room.tasks.composite.low_drawer_pnp import (
    IThorLivingRoomLowDrawerPnP as PnP,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
    get_ithor_livingroom_task_specs,
)


def make_task(cls, monkeypatch, control_freq=20):
    task = object.__new__(cls)
    task.scene_name = "FloorPlan209"
    task.control_freq = control_freq
    task.sim = object()
    task._atomic_tasks = task._get_atomic_tasks()
    task._reset_task_progress()
    task._success_hold = 0
    task._configure_articulation_task()
    task.SUCCESS_CLOSE_FRACTION = 0.2
    state = {"fraction": 0.8 if cls is Close else 0.2, "distance": 1.0, "inside": False}
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task._navigation_drawer = object()
    task.robot_fixture_collision_distance = lambda *args, **kwargs: state["distance"]
    task._apple_inside_drawer = lambda: state["inside"]
    monkeypatch.setattr(IThorLivingRoomEnv, "_post_action", lambda self, action: (0., False, {}))
    return task, state


def step(task, count):
    for _ in range(count):
        result = task._post_action(None)
    return result


@pytest.mark.parametrize("cls", [Close, PnP])
def test_three_scene_specs_registration_and_exact_targets(cls):
    assert REGISTERED_ENVS[cls.__name__] is cls
    assert room.REGISTERED_ITHOR_LIVING_ROOM_ENVS[cls.__name__] is cls
    assert cls.__name__ in room.__all__
    assert cls.SUPPORTED_SCENE_IDS == (209, 210, 226)
    specs = get_ithor_livingroom_task_specs(cls.SPEC_TASK)
    assert set(specs) == {"FloorPlan209", "FloorPlan210", "FloorPlan226"}
    for scene, spec in specs.items():
        opened = get_ithor_livingroom_task_spec(scene, "walk_and_open_low_drawer")
        assert spec["target_fixture"] == opened["target_fixture"]
        assert spec["target_joint"] == opened["target_joint"]
        assert spec["init_open_fraction"] == (0.8 if cls is Close else 0.2)
        assert spec["success_close_fraction"] == 0.2
        assert spec["hold_seconds"] == 1.0
        assert "lang" not in spec
        if cls is PnP:
            path = Path(__file__).resolve().parents[1] / "robocasa/models/assets/scenes/ithor" / f"{scene}_physics.xml"
            root = ET.parse(path).getroot()
            drawer = next(b for b in root.iter("body") if b.find(f"joint[@name='{spec['target_joint']}']") is not None)
            assert drawer.find(f"site[@name='{spec['drawer_receptacle_site']}']") is not None
            assert root.find(f".//geom[@name='{spec['apple_support_geom']}']") is not None
            assert spec["success_open_fraction"] == 0.8


def test_close_requires_navigation_then_one_second_and_reports_stages(monkeypatch, capsys):
    task, state = make_task(Close, monkeypatch)
    task._validate_initial_composite_state()
    state["distance"] = 0.35
    step(task, 4)
    assert task.stage_index == 0
    step(task, 1)
    assert task.stage_index == 1
    assert "stage 1/2, next stage: crouch_and_close_low_drawer" in capsys.readouterr().out
    state["fraction"] = 0.2
    step(task, 19)
    assert not task._check_success()
    state["fraction"] = 0.201
    step(task, 1)
    assert task._stage_hold_steps == 0
    state["fraction"] = 0.2
    reward, _, info = step(task, 20)
    assert reward == 1.0 and info["task_success"]
    assert "stage 2/2" in capsys.readouterr().out
    task._reset_task_progress()
    assert not task._check_success() and not task._close_stage_armed


def test_closing_before_arrival_cannot_skip_navigation(monkeypatch):
    task, state = make_task(Close, monkeypatch)
    state["fraction"] = 0.2
    step(task, 20)
    state["distance"] = 0.35
    step(task, 25)
    assert task.stage_index == 1 and not task._check_success()
    state["fraction"] = 0.8
    step(task, 1)
    state["fraction"] = 0.2
    step(task, 20)
    assert task._check_success()


@pytest.mark.parametrize("control_freq", [20, 30])
def test_pnp_open_five_steps_then_apple_inside_and_closed_together(monkeypatch, capsys, control_freq):
    task, state = make_task(PnP, monkeypatch, control_freq)
    task._validate_initial_composite_state()
    state["inside"] = True
    step(task, control_freq)
    assert task.stage_index == 0
    state["fraction"] = 0.8
    step(task, 4)
    state["fraction"] = 0.799
    step(task, 1)
    assert task._stage_hold_steps == 0
    state["fraction"] = 0.8
    step(task, 5)
    assert task.stage_index == 1
    assert "stage 1/2, next stage: put_apple_in_drawer_and_close" in capsys.readouterr().out
    step(task, control_freq)
    assert not task._check_success()  # Apple inside but drawer still open.
    state["fraction"] = 0.2
    step(task, control_freq - 1)
    state["inside"] = False
    step(task, 1)
    assert task._stage_hold_steps == 0  # Closing alone is insufficient.
    state["inside"] = True
    step(task, control_freq)
    assert task._check_success()
    assert "stage 2/2" in capsys.readouterr().out
    task._reset_task_progress()
    assert task.stage_index == 0 and not task._check_success()


def test_apple_receptacle_check_follows_drawer_translation_and_rotation():
    task = object.__new__(PnP)
    task._drawer_receptacle_site_id = 0
    rotation = np.array([[0., -1., 0.], [0., 0., 1.], [-1., 0., 0.]])
    center = np.array([[1., 2., 0.3]])
    task.sim = SimpleNamespace(
        data=SimpleNamespace(site_xpos=center, site_xmat=rotation.reshape(1, 9)),
        model=SimpleNamespace(site_size=np.array([[0.13, 0.45, 0.10]])),
    )
    apple = center[0].copy()
    task.object_bbox_points = lambda name: np.array([apple - 0.03, apple + 0.03])
    assert task._apple_inside_drawer()
    center[0, 0] += 1.0
    assert not task._apple_inside_drawer()
    apple[:] = center[0]
    assert task._apple_inside_drawer()
    apple[:] = center[0] + rotation @ np.array([0., 0., 0.11])
    assert not task._apple_inside_drawer()
