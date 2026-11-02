"""Focused contracts for kitchen navigation cycles and turn-to-trash tasks."""

from types import SimpleNamespace

import numpy as np
import pytest
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_kitchen import (
    IThorKitchenApproachCabinetAndOpenClose,
    IThorKitchenApproachDrawerAndOpenClose,
    IThorKitchenOpenDrawerAndTurnAppleToTrash,
    IThorKitchenPickAppleAndTurnToTrash,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_spec


CYCLES = (
    IThorKitchenApproachDrawerAndOpenClose,
    IThorKitchenApproachCabinetAndOpenClose,
)
TRASH = (IThorKitchenPickAppleAndTurnToTrash, IThorKitchenOpenDrawerAndTurnAppleToTrash)


@pytest.mark.parametrize("cls", CYCLES + TRASH)
def test_new_tasks_have_exact_registration_and_explicit_specs(cls):
    assert REGISTERED_ENVS[cls.__name__] is cls
    assert REGISTERED_ITHOR_KITCHEN_ENVS[cls.__name__] is cls
    assert set(cls.SCENE_SPECS) == {f"FloorPlan{i}" for i in cls.SUPPORTED_SCENE_IDS}
    for scene, spec in cls.SCENE_SPECS.items():
        assert set(cls.REQUIRED_SPEC_FIELDS) <= spec.keys()
        assert "lang" not in spec
        assert spec["hold_seconds"] == 1.0
        if cls in CYCLES:
            opened = get_ithor_kitchen_task_spec(scene, "open_" + cls.TARGET_NAME)
            closed = get_ithor_kitchen_task_spec(scene, "close_" + cls.TARGET_NAME)
            assert spec["navigation_goal_pose"] == opened["robot_pose"]
            for key in (
                "target_joint",
                "target_fixture",
                "init_open_fraction",
                "success_open_fraction",
            ):
                assert spec[key] == opened[key]
            assert spec["target_joint"] == closed["target_joint"]
            assert spec["success_close_fraction"] == closed["success_close_fraction"]
        else:
            pose = spec["robot_pose"]
            robot = np.asarray(pose["pos"])[:2]
            front = np.array([np.cos(pose["yaw"]), np.sin(pose["yaw"])])
            assert (np.asarray(spec["apple_pos"])[:2] - robot) @ front > 0.0
            assert (np.asarray(spec["trash_body_pos"])[:2] - robot) @ front < -0.5
            assert np.linalg.norm(np.asarray(spec["apple_pos"])[:2] - robot) < 0.75
            assert np.linalg.norm(np.asarray(spec["trash_body_pos"])[:2] - robot) < 0.65


def make_task(cls, control_freq=20):
    task = object.__new__(cls)
    task.scene_name = "FloorPlan13"
    task.control_freq = control_freq
    task._atomic_tasks = task._normalize_atomic_tasks(task._get_atomic_tasks())
    task._reset_task_progress()
    return task


@pytest.mark.parametrize("cls", CYCLES)
def test_cycles_require_navigation_open_close_and_reset_output(
    cls, monkeypatch, capsys
):
    monkeypatch.setattr(
        RoomOpenArticulation, "_post_action", lambda self, action: (0.0, False, {})
    )
    task = make_task(cls)
    spec = task._get_task_spec()
    goal = spec["navigation_goal_pose"]
    state = {"near": False, "fraction": spec["init_open_fraction"]}
    task.sim = object()
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task._robot_base_xy = lambda: np.asarray(goal["pos"])[:2] + (
        0.0 if state["near"] else 2.0
    )
    task._robot_base_yaw = lambda: goal["yaw"]
    task.target_fixture = object()
    task.robot_fixture_collision_distance = lambda *args, **kwargs: (
        0.2 if state["near"] else 2.0
    )

    def advance(steps=20):
        for _ in range(steps):
            task._post_action(None)

    for _ in range(2):
        task._reset_task_progress()
        state.update(near=False, fraction=spec["success_open_fraction"])
        advance()
        state["fraction"] = spec["success_close_fraction"]
        advance()
        assert task.stage_index == 0
        state.update(near=True, fraction=spec["success_open_fraction"])
        advance()
        assert task.stage_index == 1  # Early opening cannot skip the open stage.
        state["fraction"] = spec["init_open_fraction"]
        advance(1)
        state["fraction"] = spec["success_open_fraction"]
        advance()
        assert task.stage_index == 2
        advance(1)  # Observe the open state after entering the close stage.
        state["fraction"] = spec["success_close_fraction"]
        advance(19)
        assert not task._check_success()
        state["fraction"] = spec["success_open_fraction"]
        advance(1)
        assert task._stage_hold_steps == 0
        state["fraction"] = spec["success_close_fraction"]
        advance()
        assert task._check_success()
        output = capsys.readouterr().out
        for stage in (1, 2, 3):
            assert output.count(f"stage {stage}/3") == 1


@pytest.mark.parametrize("cls", CYCLES)
def test_cycle_navigation_uses_fixture_distance_without_pose_gate(
    cls, monkeypatch, capsys
):
    monkeypatch.setattr(
        RoomOpenArticulation, "_post_action", lambda self, action: (0.0, False, {})
    )
    task = make_task(cls)
    goal = task._get_task_spec()["navigation_goal_pose"]
    # Close to the fixture, but outside the former 15 cm / 20 degree pose gate.
    task._robot_base_xy = lambda: np.asarray(goal["pos"])[:2] + [0.0, -0.20]
    task._robot_base_yaw = lambda: goal["yaw"] + 0.4
    task.target_fixture = object()
    task.sim = object()
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: 0.2)
    state = {"distance": 0.351}

    def distance(fixture, max_distance):
        assert fixture is task.target_fixture
        return state["distance"]

    task.robot_fixture_collision_distance = distance
    for _ in range(5):
        task._post_action(None)
    assert task.stage_index == 0
    state["distance"] = 0.35
    for _ in range(4):
        task._post_action(None)
    assert task.stage_index == 0 and task._stage_hold_steps == 4
    state["distance"] = 0.351
    task._post_action(None)
    assert task._stage_hold_steps == 0
    state["distance"] = 0.35
    for _ in range(5):
        _, _, info = task._post_action(None)
    assert task.stage_index == 1
    assert info["active_atomic_task"] == "open_" + task.TARGET_NAME
    assert f"stage 1/3, next stage: open_{task.TARGET_NAME}" in capsys.readouterr().out


def make_trash_task(cls, monkeypatch, control_freq=20):
    monkeypatch.setattr(
        IThorKitchenEnv, "_post_action", lambda self, action: (0.0, False, {})
    )
    monkeypatch.setattr(
        RoomOpenArticulation, "_post_action", lambda self, action: (0.0, False, {})
    )
    task = make_task(cls, control_freq=control_freq)
    state = dict(inside=False, released=True, fraction=0.2)
    task.sim = object()
    task.object_in_region = lambda *args, **kwargs: state["inside"]
    task.object_released = lambda name: state["released"]
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task.trash_fixture = object()
    task.trash_region = object()
    task._fixture_contact = lambda name, fixture: False
    return task, state


@pytest.mark.parametrize(
    ("scene_name", "expected_support_z"),
    (
        ("FloorPlan13", None),
        ("FloorPlan24", 0.7051112278843826),
    ),
)
def test_trash_uses_scene_calibrated_apple_support_height(
    scene_name,
    expected_support_z,
):
    task = object.__new__(IThorKitchenPickAppleAndTurnToTrash)
    task.scene_name = scene_name
    spec = get_ithor_kitchen_task_spec(
        task.scene_name,
        task.SPEC_TASK,
    )
    task.source_fixture = SimpleNamespace(stable_id=spec["source_fixture"])
    task._get_task_spec = lambda: spec
    placement = {}

    def place(name, region, fixture, **kwargs):
        placement.update(name=name, region=region, fixture=fixture, kwargs=kwargs)
        return True

    task.place_object_on_support_contact = place
    task._before_object_placement()

    np.testing.assert_allclose(
        placement["region"].world_pose()[0],
        [
            *spec["apple_pos"][:2],
            spec["apple_pos"][2] if expected_support_z is None else expected_support_z,
        ],
    )
    assert placement["fixture"] is task.source_fixture


@pytest.mark.parametrize("cls", TRASH)
def test_trash_receptacle_requires_inside_and_release_without_contact(cls):
    task = make_task(cls)
    state = dict(xy=[0.0, 0.0], bottom=0.1, top=0.2, released=True)
    center = np.array([0.0, 0.0, 0.3])
    task.trash_region = SimpleNamespace(half_size=np.array([0.2, 0.2, 0.3]))
    task.trash_fixture = object()
    task._object_region_bounds = lambda name, region: (
        center,
        np.asarray(state["xy"]),
        state["bottom"],
        state["top"],
    )
    task.object_released = lambda name: state["released"]
    task._fixture_contact = lambda name, fixture: False

    assert task._apple_in_trash()  # Being inside does not require bin contact.
    state["released"] = False
    assert not task._apple_in_trash()
    state.update(released=True, xy=[0.23, 0.0], bottom=-0.03)
    assert task._apple_in_trash()  # Match the coffee-to-trash 4 cm tolerances.
    state["xy"] = [0.25, 0.0]
    assert not task._apple_in_trash()  # Outside the receptacle.
    state.update(xy=[0.0, 0.0], bottom=-0.05)
    assert not task._apple_in_trash()  # Below the receptacle.
    state.update(bottom=0.1, top=0.65)
    assert not task._apple_in_trash()  # Above the rim.


@pytest.mark.parametrize("cls", TRASH)
@pytest.mark.parametrize("control_freq", (10, 20, 30))
def test_trash_final_stage_requires_one_continuous_second(
    cls, control_freq, monkeypatch
):
    task, state = make_trash_task(cls, monkeypatch, control_freq=control_freq)
    expected_stages = ["put_apple_in_trash"]
    if cls is IThorKitchenOpenDrawerAndTurnAppleToTrash:
        expected_stages.insert(0, "open_drawer")
        state["fraction"] = 0.8
        for _ in range(5):
            task._post_action(None)
    assert [stage.name for stage in task._atomic_tasks] == expected_stages
    assert task.current_atomic_task.name == "put_apple_in_trash"
    assert task._required_stage_hold_steps(task.current_atomic_task) == control_freq

    state["inside"] = True
    for _ in range(control_freq - 1):
        task._post_action(None)
    assert not task._check_success()
    state["inside"] = False
    task._post_action(None)
    assert task._stage_hold_steps == 0
    state["inside"] = True
    for _ in range(control_freq - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    task._reset_task_progress()
    assert task.stage_index == 0 and task._stage_hold_steps == 0
    assert not task._check_success()


def test_drawer_trash_requires_five_open_steps_before_discard(monkeypatch):
    task, state = make_trash_task(
        IThorKitchenOpenDrawerAndTurnAppleToTrash, monkeypatch
    )
    state["inside"] = True
    for _ in range(20):
        task._post_action(None)
    assert task.stage_index == 0 and not task._check_success()

    state["fraction"] = 0.8
    for _ in range(4):
        task._post_action(None)
    assert task.stage_index == 0
    state["fraction"] = 0.79
    task._post_action(None)
    assert task._stage_hold_steps == 0
    state["fraction"] = 0.8
    for _ in range(5):
        task._post_action(None)
    assert task.stage_index == 1 and task._stage_hold_steps == 0
    for _ in range(19):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
