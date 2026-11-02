"""Focused contracts for the seven ordered table manipulation tasks."""

from types import SimpleNamespace

import numpy as np
import pytest

import robocasa.environments.ithor_living_room as room
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_living_room.tasks.atomic.remote_to_basket import (
    IThorLivingRoomRemoteToBasket,
)
from robocasa.environments.ithor_living_room.tasks.composite import (
    approach_table_manipulation as composite,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
    get_ithor_livingroom_task_specs,
)

TASKS = [getattr(composite, name) for name in composite.__all__]
ATOMIC_KEYS = (
    "lift_dumbbell",
    "lift_statue",
    "lift_watering_can",
    "remote_to_basket",
    "candle_to_basket",
    "tissue_box_to_basket",
    "book_into_basket",
)


def test_public_tasks_preserve_atomic_calibrations_and_navigation_poses():
    assert len(TASKS) == len({task.__name__ for task in TASKS}) == 7
    for task, atomic_key in zip(TASKS, ATOMIC_KEYS):
        assert task.SUPPORTED_SCENE_IDS == (202, 207, 209, 212, 226)
        assert getattr(room, task.__name__) is task
        assert REGISTERED_ENVS[task.__name__] is task
        assert room.REGISTERED_ITHOR_LIVING_ROOM_ENVS[task.__name__] is task
        assert task.TASK_FAMILY == "composite"
        assert "then" in task.TASK_LANG
        for scene_id in task.SUPPORTED_SCENE_IDS:
            scene = f"FloorPlan{scene_id}"
            atomic = get_ithor_livingroom_task_spec(scene, atomic_key)
            navigation = get_ithor_livingroom_task_spec(scene, "approach_table")
            spec = task.SCENE_SPECS[scene]
            overrides = get_ithor_livingroom_task_specs(task.SPEC_TASK).get(scene, {})
            assert spec["robot_pose"] == overrides.get(
                "robot_pose", navigation["robot_pose"]
            )
            assert spec["table_fixture"] == navigation["target_fixture"]
            assert spec["operation_robot_pose"] == overrides.get(
                "operation_robot_pose", atomic["robot_pose"]
            )
            for key, value in atomic.items():
                if key != "robot_pose":
                    assert spec[key] == overrides.get(key, value)
            if task in TASKS[3:]:
                source_key = "book_pos" if atomic_key == "book_into_basket" else "remote_pos"
                yaw = spec["operation_robot_pose"]["yaw"]
                left = np.array([-np.sin(yaw), np.cos(yaw)])
                delta = np.array(spec["basket_pos"][:2]) - spec[source_key][:2]
                assert delta @ left > 0.19, (task.__name__, scene)
            if overrides and scene_id == 207:
                source = spec[
                    "book_pos" if atomic_key == "book_into_basket" else "remote_pos"
                ]
                assert source[0] == spec["basket_pos"][0] > -0.1653
                assert spec["operation_robot_pose"]["pos"][0] > source[0]
                assert atomic["robot_pose"]["pos"][0] < -0.1653
            if overrides and scene_id == 209:
                spawn = np.array(spec["robot_pose"]["pos"][:2])
                assert np.linalg.norm(spawn - [-0.97, -5.06]) < 0.7
                to_table = np.array(navigation["target_xy"]) - spawn
                yaw = spec["robot_pose"]["yaw"]
                assert (
                    np.dot(
                        to_table / np.linalg.norm(to_table), [np.cos(yaw), np.sin(yaw)]
                    )
                    > 0.999
                )
                source_key = next(
                    key
                    for key in ("object_pos", "statue_pos", "remote_pos", "book_pos")
                    if key in spec
                )
                source = np.array(spec[source_key][:2])
                table_xy = np.array(navigation["target_xy"])
                assert source[0] > table_xy[0] > atomic[source_key][0]
                assert source[1] < table_xy[1]
                assert np.linalg.norm(source - spawn) < np.linalg.norm(
                    np.array(atomic[source_key][:2]) - spawn
                )
                assert spec["operation_robot_pose"]["pos"][0] > source[0]
                if "basket_pos" in spec:
                    assert spec["basket_pos"][0] == source[0]
        if "lift_height" in spec:
            assert spec["lift_height"] == 0.30
    for registry in (REGISTERED_ENVS, room.REGISTERED_ITHOR_LIVING_ROOM_ENVS):
        assert not any(name.startswith("_ApproachTable") for name in registry)


@pytest.mark.parametrize("task_class", TASKS)
def test_order_hold_messages_and_reset(task_class, monkeypatch, capsys):
    monkeypatch.setattr(RoomLiftMixin, "_post_action", lambda self, action: None)
    monkeypatch.setattr(
        IThorLivingRoomRemoteToBasket, "_post_action", lambda self, action: None
    )
    task = object.__new__(task_class)
    task._operation_hold_steps = 3
    task._reset_task_progress()
    state = {"near": False, "valid": True}
    task._robot_near_table = lambda: state["near"]
    task._raw_operation_success = lambda: state["valid"]
    for _ in range(6):
        task._post_action(None)
    assert not task._check_success()
    assert not capsys.readouterr().out

    state["near"] = True
    for _ in range(task.APPROACH_HOLD_STEPS - 1):
        task._post_action(None)
    state["near"] = False
    task._post_action(None)
    assert task._approach_hold == 0
    state["near"] = True
    for _ in range(task.APPROACH_HOLD_STEPS):
        task._post_action(None)
    assert "stage 1/2" in capsys.readouterr().out
    # Carrying an already completed result to the table cannot skip the order.
    for _ in range(4):
        task._post_action(None)
    assert not task._check_success()
    state["valid"] = False
    task._post_action(None)
    state["valid"] = True
    task._post_action(None)
    state["valid"] = False
    task._post_action(None)
    assert task._operation_hold == 0
    state["valid"] = True
    for _ in range(3):
        task._post_action(None)
    assert task._check_success()
    assert capsys.readouterr().out.count("stage 2/2") == 1
    task._post_action(None)
    assert not capsys.readouterr().out
    state["valid"] = False
    task._post_action(None)
    assert not task._check_success()
    task._reset_task_progress()
    assert task._approach_stage == task._operation_hold == 0
    assert not task._completion_reported
    assert not task._check_success()


@pytest.mark.parametrize("task_class", TASKS[:3])
def test_lift_requires_30cm_without_grasp(task_class):
    task = object.__new__(task_class)
    task._lift_initial_z = 0.5
    state = {"z": 0.8}
    task._target_root_z = lambda: state["z"]
    task.g1_hand_grasping_object = lambda name: pytest.fail("must not check grasp")
    assert task._raw_operation_success()
    state["z"] = 0.799
    assert not task._raw_operation_success()


@pytest.mark.parametrize("task_class", TASKS[3:])
def test_basket_uses_atomic_contact_distance_and_release_checks(task_class):
    task = object.__new__(task_class)
    task.success_xy_tolerance = 0.12
    task.release_distance = 0.25
    state = {"contact": True, "eef": np.array([0.5, 0, 0.1])}
    positions = np.array([[0.10, 0, 0.04], [0, 0, 0]])
    task.objects = {
        task.REMOTE_NAME: SimpleNamespace(root_body=task.REMOTE_NAME),
        task.BASKET_NAME: SimpleNamespace(root_body=task.BASKET_NAME),
    }
    task.obj_body_id = {task.REMOTE_NAME: 0, task.BASKET_NAME: 1}
    task.sim = SimpleNamespace(data=SimpleNamespace(body_xpos=positions))
    task.object_pos = lambda name: positions[task.obj_body_id[name]]
    task.eef_positions = lambda: [np.array([-0.5, 0, 0.1]), state["eef"]]

    def check_contact(source, basket):
        assert source is task.objects[task.REMOTE_NAME]
        assert basket is task.objects[task.BASKET_NAME]
        return state["contact"]

    task.check_contact = check_contact
    task.object_bbox_points = lambda name: pytest.fail("no composite-only bbox gate")
    atomic_task = task_class.__bases__[1]
    assert atomic_task._raw_success(task)
    assert task._raw_operation_success()

    state["contact"] = False
    assert not task._raw_operation_success()
    state["contact"] = True
    positions[0, 0] = 0.121
    assert not task._raw_operation_success()
    positions[0, 0] = 0.10
    state["eef"] = positions[0].copy()
    assert not task._raw_operation_success()
    state["eef"] = np.array([0.5, 0, 0.1])
    assert task._raw_operation_success()
