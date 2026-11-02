"""Contracts for the living-room two-stage low-drawer task."""

from types import SimpleNamespace

import pytest

import robocasa.environments.ithor_living_room as room
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.tasks.composite.walk_and_open_low_drawer import (
    IThorLivingRoomWalkAndOpenLowDrawer as LowDrawer,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
    get_ithor_livingroom_task_specs,
)


@pytest.fixture
def task_state(monkeypatch):
    task = object.__new__(LowDrawer)
    task.scene_name = "FloorPlan209"
    task.control_freq = 20
    task.sim = object()
    task._navigation_drawer = object()
    task._atomic_tasks = task._get_atomic_tasks()
    task._reset_task_progress()
    task._success_hold = 0
    task._configure_articulation_task()
    state = {"distance": 1.0, "fraction": 0.2}
    task.robot_fixture_collision_distance = lambda *args, **kwargs: state["distance"]
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    monkeypatch.setattr(IThorLivingRoomEnv, "_post_action", lambda self, action: (0., False, {}))
    return task, state


def step(task, count):
    for _ in range(count):
        result = task._post_action(None)
    return result


def test_registration_and_distinct_low_drawer_spec():
    assert REGISTERED_ENVS[LowDrawer.__name__] is LowDrawer
    assert room.REGISTERED_ITHOR_LIVING_ROOM_ENVS[LowDrawer.__name__] is LowDrawer
    assert LowDrawer.__name__ in room.__all__
    assert LowDrawer.SUPPORTED_SCENE_IDS == (209, 210, 226)
    specs = get_ithor_livingroom_task_specs(LowDrawer.SPEC_TASK)
    suffixes = {
        "FloorPlan209": "_1_6_0_DrawerMesh021_3",
        "FloorPlan210": "_1_8_0_DrawerMesh031_4",
        "FloorPlan226": "_1_3_0_DrawerMesh021_2",
    }
    assert set(specs) == set(suffixes)
    for scene, suffix in suffixes.items():
        spec = specs[scene]
        old = get_ithor_livingroom_task_spec(scene, "walk_and_open_drawer")
        assert spec["target_fixture"] == old["target_fixture"]
        assert spec["target_joint"] != old["target_joint"]
        assert spec["target_joint"].endswith(suffix)
        assert spec["init_open_fraction"] == 0.2
        assert spec["success_open_fraction"] == 0.8
        assert spec["hold_seconds"] == 1.0
        assert "lang" not in spec



@pytest.mark.parametrize("control_freq", [20, 30])
def test_navigation_five_steps_then_open_one_second(task_state, capsys, control_freq):
    task, state = task_state
    task.control_freq = control_freq
    task._validate_initial_composite_state()
    assert not task._check_success()
    state["distance"] = 0.35
    step(task, 4)
    assert task.stage_index == 0
    state["distance"] = 0.351
    step(task, 1)
    assert task._stage_hold_steps == 0
    state["distance"] = 0.35
    step(task, 5)
    assert task.stage_index == 1
    assert "stage 1/2, next stage: crouch_and_open_low_drawer" in capsys.readouterr().out
    state["fraction"] = 0.8
    step(task, control_freq - 1)
    assert not task._check_success()
    state["fraction"] = 0.799
    step(task, 1)
    assert task._stage_hold_steps == 0
    state["fraction"] = 0.8
    reward, _, info = step(task, control_freq)
    assert task._check_success() and reward == 1.0 and info["task_success"]
    assert "stage 2/2" in capsys.readouterr().out
    task._reset_task_progress()
    assert task.stage_index == 0 and not task._check_success()
    assert not task._open_stage_armed


def test_opening_before_navigation_cannot_complete(task_state):
    task, state = task_state
    state["fraction"] = 0.8
    step(task, 20)
    assert task.stage_index == 0
    state["distance"] = 0.35
    step(task, 5)
    step(task, 20)
    assert task.stage_index == 1 and not task._check_success()
    state["fraction"] = 0.2
    step(task, 1)
    state["fraction"] = 0.8
    step(task, 20)
    assert task._check_success()


def test_exact_eighty_percent_tolerates_float_roundoff(task_state):
    task, state = task_state
    state["fraction"] = 0.16 / 0.2
    assert task._raw_success()
    state["fraction"] = 0.159 / 0.2
    assert not task._raw_success()
