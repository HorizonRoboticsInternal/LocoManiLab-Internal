"""Regression tests for strict iTHOR Living Room task calibrations."""

from types import SimpleNamespace

import pytest

from robocasa.environments.ithor_living_room.tasks.atomic.box_flap import (
    IThorLivingRoomCloseBoxFlap,
    IThorLivingRoomOpenBoxFlap,
)
from robocasa.environments.ithor_living_room.tasks.atomic.close_laptop import (
    IThorLivingRoomCloseLaptop,
)
from robocasa.environments.ithor_living_room.tasks.atomic.open_laptop import (
    IThorLivingRoomOpenLaptop,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_cup_among_objects import (
    IThorLivingRoomLiftCupAmongObjects,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_dumbbell import (
    IThorLivingRoomLiftDumbbell,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_statue import (
    IThorLivingRoomLiftStatue,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_watering_can import (
    IThorLivingRoomLiftWateringCan,
)
from robocasa.environments.ithor_living_room.tasks.atomic.push_basketball import (
    IThorLivingRoomPushBasketball,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.thor_semantics import SemanticJoint
from robocasa.models.scenes.ithor_livingroom_task_specs import _load_specs


def test_drawer_tasks_use_symmetric_twenty_eighty_thresholds():
    specs = _load_specs()["scenes"]

    for scene_name in (
        "FloorPlan209",
        "FloorPlan210",
        "FloorPlan226",
    ):
        open_spec = specs[scene_name]["open_drawer"]
        assert open_spec["init_open_fraction"] == pytest.approx(0.20)
        assert open_spec["success_open_fraction"] == pytest.approx(0.80)

    for scene_name in ("FloorPlan209", "FloorPlan210", "FloorPlan226"):
        close_spec = specs[scene_name]["close_drawer"]
        assert close_spec["init_open_fraction"] == pytest.approx(0.80)
        assert close_spec["success_close_fraction"] == pytest.approx(0.20)


def test_laptop_tasks_use_twenty_to_ninety_percent_travel():
    specs = _load_specs()["scenes"]

    for scene_name in (
        "FloorPlan202",
        "FloorPlan207",
        "FloorPlan209",
        "FloorPlan210",
        "FloorPlan212",
        "FloorPlan226",
    ):
        open_spec = specs[scene_name]["open_laptop"]
        close_spec = specs[scene_name]["close_laptop"]

        assert open_spec["init_open_fraction"] == pytest.approx(0.2)
        assert open_spec["success_open_fraction"] == pytest.approx(0.9)
        assert close_spec["init_open_fraction"] == pytest.approx(0.9)
        assert close_spec["success_close_fraction"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    ("task_class", "scene_210_lang", "other_scene_lang"),
    (
        (
            IThorLivingRoomLiftStatue,
            "lift the statue from the edge of the support surface",
            "lift the statue from the edge of the table",
        ),
        (
            IThorLivingRoomLiftDumbbell,
            "lift the dumbbell from the edge of the support surface",
            "lift the dumbbell from the edge of the table",
        ),
        (
            IThorLivingRoomLiftWateringCan,
            "lift the watering can from the edge of the support surface",
            "lift the watering can from the edge of the table",
        ),
        (
            IThorLivingRoomLiftCupAmongObjects,
            "lift the cup from the support surface with five objects",
            "lift the cup from the table with five objects",
        ),
        (
            IThorLivingRoomPushBasketball,
            "push the basketball across the support surface",
            "push the basketball across the table",
        ),
    ),
)
def test_scene_210_support_surface_language_is_scene_local(
    task_class, scene_210_lang, other_scene_lang, monkeypatch
):
    monkeypatch.setattr(IThorLivingRoomEnv, "get_ep_meta", lambda _self: {})
    task = object.__new__(task_class)
    task.scene_name = "FloorPlan210"
    if task_class is IThorLivingRoomPushBasketball:
        task.PUSH_DISTANCE = 0.3

    assert task.get_ep_meta()["lang"] == scene_210_lang

    task.scene_name = "FloorPlan202"
    assert task.get_ep_meta()["lang"] == other_scene_lang


def test_scene_210_cup_layout_stays_inside_support_without_jitter():
    spec = _load_specs()["scenes"]["FloorPlan210"]["lift_cup_among_objects"]
    support = spec["work_surface_support"]
    center = support["center"]
    half_xy = support["half_xy"]
    for pose in spec["object_poses"].values():
        delta = [pose["pos"][axis] - center[axis] for axis in (0, 1)]
        assert all(
            abs(value) <= half_xy[axis] - 0.05
            for axis, value in enumerate(delta)
        )
    assert spec["object_jitter_radius"] == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("task_class", "initial_fraction"),
    ((IThorLivingRoomOpenLaptop, 0.2), (IThorLivingRoomCloseLaptop, 0.9)),
)
def test_laptop_placement_spec_does_not_require_an_exact_target_joint(
    task_class, initial_fraction
):
    task = object.__new__(task_class)
    task.scene_name = "FloorPlan202"
    task.sim = None
    task._g1_target_z = lambda: 0.8
    task._joint_world_z = lambda _joint: 0.8
    lid = SemanticJoint(
        stable_id="spawned_laptop:joint:lid",
        name="spawned_laptop_Laptop_13_laptop_lid_13_joint_0",
        role="lid",
        range=(0.0, 1.5708),
        closed_qpos=1.5708,
        open_qpos=0.0,
    )
    task.CLOSED_QPOS = lid.closed_qpos
    fixture = SimpleNamespace(joints={"lid": lid})

    # Placement calibration has no scene-native fixture/joint IDs. The spawned
    # laptop must still select its lid and initialize it from that calibration.
    task._get_scene_robot_pose_calibration()
    assert task._select_target_joint(fixture) is lid
    assert task._initial_joint_qpos(lid) == pytest.approx(
        lid.closed_qpos * (1.0 - initial_fraction)
    )


@pytest.mark.parametrize(
    ("task_class", "initial_fraction", "success_open_fraction"),
    (
        (IThorLivingRoomOpenBoxFlap, 0.2, 0.8),
        (IThorLivingRoomCloseBoxFlap, 0.8, 0.2),
    ),
)
def test_box_flap_tasks_use_twenty_eighty_percent_travel(
    task_class,
    initial_fraction,
    success_open_fraction,
):
    task = object.__new__(task_class)
    task.target_closed_qpos = -2.5
    task.target_open_qpos = 0.0

    assert task.INITIAL_OPEN_FRACTION == pytest.approx(initial_fraction)
    assert task._initial_target_qpos() == pytest.approx(-2.5 + 2.5 * initial_fraction)

    open_fraction = {"value": success_open_fraction}
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            get_joint_qpos=lambda _joint: -2.5 + 2.5 * open_fraction["value"]
        )
    )
    task.target_joint = "target_flap"
    assert task._target_fraction() == pytest.approx(0.8)
    open_fraction["value"] += -0.01 if task.BEHAVIOR == "open" else 0.01
    assert task._target_fraction() == pytest.approx(0.79)

    specs = _load_specs()["scenes"]
    for scene_name in (
        "FloorPlan202",
        "FloorPlan207",
        "FloorPlan209",
        "FloorPlan210",
        "FloorPlan212",
        "FloorPlan226",
    ):
        spec = specs[scene_name][task_class.SPEC_TASK]
        assert spec["success_fraction"] == pytest.approx(
            success_open_fraction
            if task_class.BEHAVIOR == "open"
            else 1.0 - success_open_fraction
        )
