"""Order, geometry, and scene coverage for the book/book/mug composite."""

from itertools import product
from types import SimpleNamespace

import numpy as np
import pytest

import robocasa.environments.ithor_living_room as room
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.tasks.composite.stack_books_and_place_mug import (
    IThorLivingRoomStackBooksAndPlaceMug as Task,
)
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


def test_all_scenes_have_left_right_books_and_supported_layouts():
    assert REGISTERED_ENVS[Task.__name__] is Task
    assert room.REGISTERED_ITHOR_LIVING_ROOM_ENVS[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (202, 207, 209, 210, 212, 226)
    assert set(Task.SCENE_SPECS) == {f"FloorPlan{i}" for i in Task.SUPPORTED_SCENE_IDS}
    for scene, spec in Task.SCENE_SPECS.items():
        stack_spec = get_ithor_livingroom_task_spec(scene, "stack_book_on_book")
        assert spec["moving_book_pitch"] == stack_spec["moving_book_pitch"]
        assert spec["moving_book_pitch"] == pytest.approx(-np.pi / 2, abs=1e-5)
        robot = np.asarray(spec["robot_pose"]["pos"][:2])
        yaw = spec["robot_pose"]["yaw"]
        left = np.array([-np.sin(yaw), np.cos(yaw)])
        forward = np.array([np.cos(yaw), np.sin(yaw)])
        base = np.asarray(spec["base_book_pos"][:2])
        moving = np.asarray(spec["moving_book_pos"][:2])
        assert (base - robot) @ left > 0
        assert (moving - robot) @ left < 0
        assert (base - robot) @ forward > 0
        assert (moving - robot) @ forward > 0
        assert np.linalg.norm(base - moving) > 0.3
        support = spec["work_surface_support"]
        center = np.asarray(support["center"][:2])
        half = np.asarray(support["half_xy"])
        for name in ("base_book", "moving_book", "mug"):
            # Project the book's calibrated footprint into world XY.
            angle = spec[f"{name}_yaw"]
            c, s = abs(np.cos(angle)), abs(np.sin(angle))
            size = np.array(spec["source_bbox_size"]) * spec["source_scale"]
            if name == "moving_book":
                pitch = spec["moving_book_pitch"]
                size[1] = abs(np.cos(pitch)) * size[1] + abs(np.sin(pitch)) * size[2]
            footprint = np.array([c * size[0] + s * size[1], s * size[0] + c * size[1]])
            if name == "mug":
                footprint = np.array([0.06, 0.06])
            assert np.all(
                np.abs(np.array(spec[f"{name}_pos"][:2]) - center) + footprint < half
            )
        assert spec["hold_seconds"] == 1.0


def test_strict_order_continuous_one_second_hold_and_reset(monkeypatch, capsys):
    monkeypatch.setattr(IThorLivingRoomEnv, "_post_action", lambda self, action: None)
    task = object.__new__(Task)
    task._stack_hold_steps = 5
    task.HOLD_STEPS = 20
    task._reset_task_progress()
    state = dict(books=True, mug=True, released=True)
    task.release_distance = 0.25
    task.object_released = lambda *args, **kwargs: state["released"]
    task._books_stacked = lambda: state["books"]
    task._mug_touches_top_book = lambda: state["mug"]
    task._mug_on_top_book = lambda: state["mug"]

    def advance(steps):
        for _ in range(steps):
            task._post_action(None)

    # Starting with the mug already on the book cannot bypass the first stage.
    advance(25)
    assert task._stack_stage == 0
    assert not task._check_success()
    state["mug"] = False
    advance(4)
    assert task._stack_stage == 0
    advance(1)
    assert task._stack_stage == 1
    assert "stage 1/2" in capsys.readouterr().out
    # A hand bringing the mug near the top book must not reset the first stage.
    state["released"] = False
    advance(5)
    assert task._stack_stage == 1
    state["mug"] = True
    advance(5)
    assert task._stack_stage == 1 and not task._check_success()
    state["released"] = True
    advance(19)
    assert not task._check_success()
    state["mug"] = False
    advance(1)
    assert task._success_hold_steps == 0
    state["mug"] = True
    advance(20)
    assert task._check_success()
    assert "stage 2/2" in capsys.readouterr().out
    state["books"] = False
    advance(1)
    assert task._stack_stage == 0
    assert not task._check_success()
    task._stack_stage = 1
    task._success_hold_steps = 20
    task._reset_task_progress()
    assert not task._check_success()


def _bbox(center, half):
    return np.asarray(center) + np.asarray(list(product((-1, 1), repeat=3))) * half


@pytest.mark.parametrize(
    "xy,bottom,contact,released,upright,expected",
    [
        ((0, 0), 0.802, True, True, True, True),
        ((0, 0), 0.802, False, True, True, False),
        ((0, 0), 0.84, True, True, True, False),
        ((0.12, 0), 0.775, True, True, True, False),
        ((0.06, 0), 0.802, True, True, True, False),
        ((0, 0), 0.802, True, False, True, False),
        ((0, 0), 0.802, True, True, False, False),
    ],
)
def test_mug_requires_top_support_full_footprint_and_release(
    xy, bottom, contact, released, upright, expected
):
    task = object.__new__(Task)
    task.release_distance = 0.25
    task.objects = {
        name: SimpleNamespace(root_body=name) for name in ("moving_book", "mug")
    }
    task._mug_touches_top_book = lambda: contact
    task.object_released = lambda *args, **kwargs: released
    boxes = {
        "moving_book": _bbox([0, 0, 0.789], [0.078, 0.104, 0.013]),
        "mug": _bbox([*xy, bottom + 0.0375], [0.045, 0.045, 0.0375]),
    }
    task.object_bbox_points = boxes.__getitem__
    task._collision_z_bounds = lambda name: (
        boxes[name][:, 2].min(),
        boxes[name][:, 2].max(),
    )
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            get_body_xmat=lambda name: (
                np.diag([1, -1, -1]) if name == "mug" and not upright else np.eye(3)
            )
        )
    )
    assert task._mug_on_top_book() is expected


def test_book_stack_geometry_is_independent_of_hand_release():
    task = object.__new__(Task)
    task.XY_TOLERANCE = .08
    task.ORIENTATION_COSINE = .94
    task.work_surface_region = object()
    task.objects = {name: object() for name in ("base_book", "moving_book")}
    boxes = {
        "base_book": _bbox([0, 0, .763], [.078, .104, .013]),
        "moving_book": _bbox([0, 0, .789], [.078, .104, .013]),
    }
    task.object_bbox_points = boxes.__getitem__
    task.check_contact = lambda *args: True
    task._book_face_alignment = lambda: 1.
    task.object_on_region = lambda *args, **kwargs: True
    task.object_released = lambda *args, **kwargs: pytest.fail("hand proximity is not stack geometry")
    assert task._books_stacked()
    boxes["base_book"], boxes["moving_book"] = boxes["moving_book"], boxes["base_book"]
    assert not task._books_stacked()  # Reversing the initial left/right books fails.
