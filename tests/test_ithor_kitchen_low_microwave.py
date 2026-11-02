"""Grounded microwave support, attached handle, and ordered door/apple stages."""

from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_kitchen import (
    IThorKitchenCloseThenOpenLowMicrowave as CloseOpen,
    IThorKitchenOpenLowMicrowaveAndLiftApple as OpenLift,
    IThorKitchenOpenLowMicrowaveAndPutAppleInside as OpenPut,
    IThorKitchenPutAppleInLowMicrowave as Put,
    IThorKitchenOpenCloseLowMicrowave as Door,
    IThorKitchenTakeAppleFromOpenLowMicrowaveToTop as TakeToTop,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks import composite


TASKS = (Door, Put)
NEW_TASKS = (OpenLift, TakeToTop, CloseOpen, OpenPut)
NEW_APPLE_TASKS = (OpenLift, TakeToTop, OpenPut)


def make_task(cls):
    task = object.__new__(cls)
    task.scene_name = "FloorPlan13"
    task.control_freq = 20
    task._atomic_tasks = task._get_atomic_tasks()
    task._reset_task_progress()
    task._configure_articulation_task()
    task.SUCCESS_CLOSE_FRACTION = task._get_task_spec()["success_close_fraction"]
    return task


@pytest.mark.parametrize("cls", TASKS)
@pytest.mark.parametrize("scene_id", (6, 7, 13, 24, 25))
def test_registration_and_shared_fixed_microwave_layout(cls, scene_id):
    assert REGISTERED_ENVS[cls.__name__] is cls
    assert REGISTERED_ITHOR_KITCHEN_ENVS[cls.__name__] is cls
    assert cls.__name__ in composite.__all__
    assert "_LowMicrowaveMixin" not in REGISTERED_ENVS
    assert cls.SUPPORTED_SCENE_IDS == (6, 7, 13, 24, 25)
    assert set(cls.SCENE_SPECS) == {f"FloorPlan{s}" for s in cls.SUPPORTED_SCENE_IDS}
    scene = f"FloorPlan{scene_id}"
    spec = cls.SCENE_SPECS[scene]
    assert "lang" not in spec
    assert spec["init_open_fraction"] == 0.0
    assert spec["success_open_fraction"] == 0.4
    assert spec["success_close_fraction"] == 0.1
    assert spec["table_center"][2] == 0.45
    other = Put if cls is Door else Door
    assert {k: v for k, v in spec.items() if k != "apple_pos"} == {
        k: v for k, v in other.SCENE_SPECS[scene].items() if k != "apple_pos"
    }


@pytest.mark.parametrize("cls", NEW_TASKS)
@pytest.mark.parametrize("scene_id", (6, 7, 13, 24, 25))
def test_new_tasks_are_public_with_explicit_scene_specs(cls, scene_id):
    assert REGISTERED_ENVS[cls.__name__] is cls
    assert REGISTERED_ITHOR_KITCHEN_ENVS[cls.__name__] is cls
    assert cls.__name__ in composite.__all__
    assert cls.SUPPORTED_SCENE_IDS == (6, 7, 13, 24, 25)
    assert set(cls.SCENE_SPECS) == {f"FloorPlan{s}" for s in cls.SUPPORTED_SCENE_IDS}
    spec = cls.SCENE_SPECS[f"FloorPlan{scene_id}"]
    assert "lang" not in spec
    assert spec["success_open_fraction"] == 0.4
    assert spec["success_close_fraction"] == 0.1
    assert spec["table_center"][2] == 0.45
    base_spec = Door.SCENE_SPECS[f"FloorPlan{scene_id}"]
    for field in (
        "target_fixture",
        "target_joint",
        "microwave_body",
        "microwave_pos",
        "microwave_quat",
        "microwave_receptacle_site",
    ):
        assert spec[field] == base_spec[field]


@pytest.mark.parametrize("scene_id", (6, 7, 13, 24, 25))
def test_new_task_initial_states_and_shared_assets_are_explicit(scene_id):
    scene = f"FloorPlan{scene_id}"
    assert OpenLift.SCENE_SPECS[scene]["init_open_fraction"] == 0.2
    assert OpenPut.SCENE_SPECS[scene]["init_open_fraction"] == 0.0
    assert TakeToTop.SCENE_SPECS[scene]["init_open_fraction"] == 1.0
    assert CloseOpen.SCENE_SPECS[scene]["init_open_fraction"] == 0.8
    assert OpenLift.APPLE_INITIAL_LOCATION == "inside"
    assert TakeToTop.APPLE_INITIAL_LOCATION == "inside"
    assert OpenPut.APPLE_INITIAL_LOCATION == "top"
    assert all(cls.APPLE_XML == Put.APPLE_XML for cls in NEW_APPLE_TASKS)
    assert OpenLift.MIN_LIFT_DELTA == 0.30
    assert OpenLift.PICK_LIFT_DELTA == 0.05
    assert "apple_top_pos" not in OpenLift.SCENE_SPECS[scene]
    assert "apple_top_pos" in TakeToTop.SCENE_SPECS[scene]
    assert "apple_top_pos" in OpenPut.SCENE_SPECS[scene]
    for task_class in (OpenLift, TakeToTop):
        apple_xy = np.asarray(task_class.SCENE_SPECS[scene]["apple_inside_xy"])
        assert apple_xy.shape == (2,)
        assert not np.allclose(
            apple_xy, task_class.SCENE_SPECS[scene]["microwave_pos"][:2]
        )
    assert "apple_inside_xy" not in Put.SCENE_SPECS[scene]
    assert "_LowMicrowaveAppleMixin" not in REGISTERED_ENVS


@pytest.mark.parametrize("scene_id", (6, 7, 24, 25))
def test_close_then_open_uses_open_lift_robot_pose_for_corrected_scenes(scene_id):
    scene = f"FloorPlan{scene_id}"
    assert (
        CloseOpen.SCENE_SPECS[scene]["robot_pose"]
        == OpenLift.SCENE_SPECS[scene]["robot_pose"]
    )


def test_close_then_open_floorplan13_robot_pose_is_unchanged():
    assert CloseOpen.SCENE_SPECS["FloorPlan13"]["robot_pose"] == {
        "pos": [-1.75, 3.20, 0.0],
        "yaw": 3.141592653589793,
    }


def test_inside_apple_calibrations_account_for_native_site_floor():
    for task_class in (OpenLift, TakeToTop):
        assert task_class.SCENE_SPECS["FloorPlan6"][
            "initial_inside_tolerance"
        ] == pytest.approx(0.01)
        assert task_class.SCENE_SPECS["FloorPlan24"][
            "initial_inside_tolerance"
        ] == pytest.approx(0.02)
        for scene_id in (7, 13, 25):
            assert (
                "initial_inside_tolerance"
                not in task_class.SCENE_SPECS[f"FloorPlan{scene_id}"]
            )


def test_requested_inside_apple_tasks_use_calibrated_spawns():
    assert OpenLift.APPLE_INSIDE_XY_KEY == "apple_inside_xy"
    assert TakeToTop.APPLE_INSIDE_XY_KEY == "apple_inside_xy"
    assert Put.APPLE_INSIDE_XY_KEY is None
    assert OpenLift.APPLE_INSIDE_SUPPORT_HALF_XY == (0.10, 0.10)
    assert TakeToTop.APPLE_INSIDE_SUPPORT_HALF_XY == (0.10, 0.10)
    assert Put.APPLE_INSIDE_SUPPORT_HALF_XY == (0.05, 0.05)
    assert OpenLift.SCENE_SPECS["FloorPlan13"]["robot_pose"]["pos"] == [
        -2.05554,
        3.56620,
        0.0,
    ]
    assert TakeToTop.SCENE_SPECS["FloorPlan13"]["robot_pose"]["pos"] == [
        -2.15904,
        3.19529,
        0.0,
    ]


@pytest.mark.parametrize("scene_id", (6, 7, 13, 24, 25))
def test_take_to_top_spawn_is_slightly_right_of_the_apple_approach_line(scene_id):
    spec = TakeToTop.SCENE_SPECS[f"FloorPlan{scene_id}"]
    robot_xy = np.asarray(spec["robot_pose"]["pos"][:2])
    apple_xy = np.asarray(spec["apple_inside_xy"])
    yaw = spec["robot_pose"]["yaw"]
    forward = np.array([np.cos(yaw), np.sin(yaw)])
    apple_offset = apple_xy - robot_xy

    assert np.dot(apple_offset, forward) > 0.0
    lateral_offset = forward[0] * apple_offset[1] - forward[1] * apple_offset[0]
    assert lateral_offset == pytest.approx(0.06, abs=1e-4)


def test_inside_apple_xy_calibration_is_opt_in_for_the_requested_tasks():
    center = np.array([1.0, 2.0, 0.6])
    for task_class in (OpenLift, TakeToTop):
        task = make_task(task_class)
        task._microwave_receptacle_pose = lambda: (center, None, None)
        task._get_task_spec = lambda: {"apple_inside_xy": [1.1, 1.9]}
        np.testing.assert_allclose(task._apple_inside_xy(), [1.1, 1.9])
    task = make_task(Put)
    task._microwave_receptacle_pose = lambda: (center, None, None)
    np.testing.assert_allclose(task._apple_inside_xy(), center[:2])


def test_floorplan24_take_to_top_uses_a_task_only_top_collision_guard():
    task = make_task(TakeToTop)
    task.scene_name = "FloorPlan24"
    world = ET.Element("worldbody")
    microwave = ET.SubElement(
        world,
        "body",
        name=task._get_task_spec()["microwave_body"],
    )
    task.mujoco_arena = SimpleNamespace(worldbody=world)

    task._add_apple_top_collision_guard()

    guard = microwave.find("./geom[@name='task_apple_top_collision_guard']")
    assert guard is not None
    assert guard.get("type") == "box"
    assert guard.get("contype") == "1"
    assert guard.get("conaffinity") == "15"
    np.testing.assert_allclose(
        np.fromstring(guard.get("size"), sep=" "), [0.115, 0.105, 0.002]
    )
    assert "apple_top_collision_guard" not in OpenPut.SCENE_SPECS["FloorPlan24"]

    task.scene_name = "FloorPlan13"
    task.mujoco_arena = SimpleNamespace(worldbody=ET.Element("worldbody"))
    task._add_apple_top_collision_guard()
    assert not list(task.mujoco_arena.worldbody)


def test_open_lift_task_language_describes_taking_out_the_apple():
    assert OpenLift.TASK_LANG == (
        "crouch down, open the low microwave, and take out the apple"
    )


def test_model_edit_fixes_housing_and_preserves_door_and_native_receptacle():
    task = make_task(Door)
    spec = task._get_task_spec()
    world = ET.Element("worldbody")
    housing = ET.SubElement(world, "body", name=spec["microwave_body"])
    ET.SubElement(housing, "joint", name="housing_free", type="free")
    door = ET.SubElement(housing, "body", name="door")
    hinge = ET.SubElement(door, "joint", name=spec["target_joint"], type="hinge")
    receptacle = ET.SubElement(housing, "site", name=spec["microwave_receptacle_site"])
    task.mujoco_arena = SimpleNamespace(worldbody=world, asset=ET.Element("asset"))
    task._setup_baked_scene_fixtures()
    assert housing.find("./joint") is None
    assert door.find("./joint") is hinge
    assert housing.find("./site") is receptacle
    assert np.allclose(
        np.fromstring(housing.get("pos"), sep=" "), spec["microwave_pos"]
    )
    table = world.find("./body[@name='task_low_microwave_table']")
    assert len(table.findall("geom")) == 6
    for side in ("left_support", "right_support"):
        support = table.find(f"./geom[@name='task_low_microwave_table_{side}_visual']")
        pos = np.fromstring(support.get("pos"), sep=" ")
        size = np.fromstring(support.get("size"), sep=" ")
        assert pos[2] - size[2] == pytest.approx(-0.002)
        assert pos[2] + size[2] == pytest.approx(spec["table_center"][2] - 0.05)
    handle = door.find("./body[@name='task_microwave_handle']")
    assert handle is not None
    assert len(handle.findall("./geom[@type='capsule']")) == 3
    mesh = task.mujoco_arena.asset.find("./mesh[@name='task_microwave_handle_mesh']")
    assert np.allclose(
        np.fromstring(mesh.get("scale"), sep=" "),
        np.array([0.065, 0.16143, 0.18]) * 1.5,
    )
    grip = handle.find("./geom[@name='task_microwave_handle_grip']")
    assert float(grip.get("size")) == pytest.approx(0.018)
    endpoints = np.fromstring(grip.get("fromto"), sep=" ").reshape(2, 3)
    assert np.linalg.norm(endpoints[1] - endpoints[0]) == pytest.approx(0.195)
    mount = handle.find("./geom[@name='task_microwave_handle_lower_mount']")
    mount_start = np.fromstring(mount.get("fromto"), sep=" ")[:3]
    handle_pos = np.fromstring(handle.get("pos"), sep=" ")
    assert (handle_pos + mount_start)[2] == pytest.approx(0.03)
    assert Put.HANDLE_SCALE == 1.0
    assert not handle.findall(".//joint")
    assert task._get_obj_cfgs() == []
    assert "apple_pos" not in spec
    assert "IThorKitchenTakeAppleFromLowMicrowave" not in REGISTERED_ENVS
    assert not table.findall(".//joint")


@pytest.mark.parametrize("cls", TASKS)
@pytest.mark.parametrize("control_freq", (20, 30))
def test_ordered_stages_require_apple_placement_only_for_put(
    cls, control_freq, monkeypatch, capsys
):
    monkeypatch.setattr(
        RoomOpenArticulation, "_post_action", lambda self, action: (0.0, False, {})
    )
    task = make_task(cls)
    task.control_freq = control_freq
    state = {"fraction": 0.0, "inside": False}
    task.sim = object()
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    if cls is Put:
        task._apple_in_microwave = lambda: state["inside"]

    def step(count):
        for _ in range(count):
            task._post_action(None)

    step(control_freq)
    assert task.stage_index == 0 and not task._check_success()
    state["fraction"] = np.nextafter(0.4, 0.0)
    step(4)
    assert task.stage_index == 0
    state["fraction"] = 0.399
    step(1)
    assert task._stage_hold_steps == 0
    state["fraction"] = np.nextafter(0.4, 0.0)
    step(5)
    assert task.stage_index == 1
    if cls is Put:
        assert [t.name for t in task._atomic_tasks] == [
            "open_microwave",
            "put_apple_in_microwave",
            "close_microwave",
        ]
        # Closing the door without putting the apple inside cannot skip placement.
        state["fraction"] = 0.1
        step(control_freq)
        assert task.stage_index == 1 and not task._check_success()
        state["inside"] = True
        step(5)
        assert task.stage_index == 1  # Placement requires the door to remain open.
        state["fraction"] = 0.4
        step(5)
        assert task.stage_index == 2
        state["inside"] = False
        state["fraction"] = 0.1
        step(control_freq)
        assert not task._check_success()  # Apple must remain inside through closure.
        state["inside"] = True
    else:
        assert [t.name for t in task._atomic_tasks] == [
            "open_microwave",
            "close_microwave",
        ]
    state["fraction"] = np.nextafter(0.1, 1.0)
    step(control_freq - 1)
    assert not task._check_success()
    state["fraction"] = 0.101
    step(1)
    assert task._stage_hold_steps == 0
    state["fraction"] = np.nextafter(0.1, 1.0)
    step(control_freq)
    assert task._check_success()
    output = capsys.readouterr().out
    if cls is Put:
        assert "stage 1/3, next stage: put_apple_in_microwave" in output
        assert "stage 2/3, next stage: close_microwave" in output
        assert "stage 3/3" in output
    else:
        assert "stage 1/2, next stage: close_microwave" in output
        assert "stage 2/2" in output
    task._reset_task_progress()
    assert task.stage_index == 0 and not task._check_success()


def test_apple_range_uses_native_site_rotation_and_full_three_dimensions():
    task = make_task(Put)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    site_center = np.array([1.0, 2.0, 0.4])
    half_size = np.array([0.2, 0.12, 0.1])
    task._microwave_receptacle_site_id = 0
    task.sim = SimpleNamespace(
        model=SimpleNamespace(site_size=np.array([half_size])),
        data=SimpleNamespace(
            site_xpos=np.array([site_center]), site_xmat=np.array([rotation.ravel()])
        ),
    )
    state = {"center": site_center + rotation @ np.array([0.15, 0.0, 0.0])}
    task.object_bbox_points = lambda name: np.array([state["center"]])
    assert task._apple_in_microwave()
    for offset in (
        [0.201, 0.0, 0.0],
        [0.0, 0.121, 0.0],
        [0.0, 0.0, 0.101],
        [0.0, 0.0, -0.101],
    ):
        state["center"] = site_center + rotation @ np.array(offset)
        assert not task._apple_in_microwave()


@pytest.mark.parametrize(
    ("cls", "stage_names"),
    (
        (OpenLift, ("open_microwave", "lift_apple")),
        (TakeToTop, ("pick_up_apple", "place_apple_on_microwave")),
        (CloseOpen, ("close_microwave", "open_microwave")),
        (
            OpenPut,
            ("open_microwave", "pick_up_apple", "put_apple_in_microwave"),
        ),
    ),
)
def test_new_tasks_enforce_declared_stage_order(cls, stage_names, monkeypatch):
    monkeypatch.setattr(
        RoomOpenArticulation, "_post_action", lambda self, action: (0.0, False, {})
    )
    task = make_task(cls)
    state = {stage.name: False for stage in task._atomic_tasks}
    for stage in task._atomic_tasks:
        setattr(task, stage.success, lambda name=stage.name: state[name])

    assert tuple(stage.name for stage in task._atomic_tasks) == stage_names
    for index, stage in enumerate(task._atomic_tasks):
        state[stage.name] = False
        for future in task._atomic_tasks[index + 1 :]:
            state[future.name] = True
        task._post_action(None)
        assert task.stage_index == index

        state[stage.name] = True
        for _ in range(task._required_stage_hold_steps(stage)):
            task._post_action(None)
        assert task.stage_index == index + 1
    assert task._check_success()


def test_lift_requires_source_height_grasp_open_door_and_30cm_delta():
    task = make_task(OpenLift)
    task._apple_initial_root_z = 0.4
    task._grasped_at_source_height = False
    state = {"height": 0.51, "grasped": True, "door_open": True}
    task._apple_root_z = lambda: state["height"]
    task.g1_hand_grasping_object = lambda name: state["grasped"]
    task._raw_success = lambda: state["door_open"]

    assert not task._apple_lifted_30cm_with_door_open()
    assert not task._grasped_at_source_height
    state["height"] = 0.4
    assert not task._apple_lifted_30cm_with_door_open()
    assert task._grasped_at_source_height
    state["height"] = np.nextafter(0.7, 0.0)
    assert not task._apple_lifted_30cm_with_door_open()
    state["height"] = 0.7
    assert task._apple_lifted_30cm_with_door_open()
    state["door_open"] = False
    assert not task._apple_lifted_30cm_with_door_open()
    state["door_open"] = True
    state["grasped"] = False
    assert not task._apple_lifted_30cm_with_door_open()


def test_pick_up_stage_requires_a_source_height_grasp_and_5cm_delta():
    task = make_task(TakeToTop)
    task._apple_initial_root_z = 0.4
    task._grasped_at_source_height = False
    state = {"height": 0.4, "grasped": True}
    task._apple_root_z = lambda: state["height"]
    task.g1_hand_grasping_object = lambda name: state["grasped"]
    task._raw_success = lambda: True

    assert not task._apple_picked_up_with_door_open()
    state["height"] = np.nextafter(0.45, 0.0)
    assert not task._apple_picked_up_with_door_open()
    state["height"] = 0.45
    assert task._apple_picked_up_with_door_open()


def test_open_put_stages_do_not_require_grasp_or_release_state():
    task = make_task(OpenPut)
    task._apple_initial_root_z = 0.4
    task._apple_root_z = lambda: 0.45
    task.g1_hand_grasping_object = lambda name: pytest.fail("grasp must not be queried")
    task._raw_success = lambda: True
    assert task._apple_picked_up_with_door_open()

    site_center = np.array([1.0, 2.0, 0.4])
    task._microwave_receptacle_pose = lambda: (
        site_center,
        np.array([0.2, 0.12, 0.1]),
        np.eye(3),
    )
    task.object_bbox_points = lambda name: np.array([site_center])
    task._apple_contacts_target_fixture = lambda: True
    task.object_released = lambda *args, **kwargs: pytest.fail(
        "release must not be queried"
    )
    assert task._apple_inside_with_door_open()


def test_new_placement_predicates_require_center_containment_and_contact():
    task = make_task(OpenPut)
    site_center = np.array([1.0, 2.0, 0.4])
    half_size = np.array([0.2, 0.12, 0.1])
    rotation = np.eye(3)
    task._microwave_receptacle_pose = lambda: (site_center, half_size, rotation)
    state = {
        "points": np.array([site_center]),
        "contact": True,
        "door_open": True,
    }
    task.object_bbox_points = lambda name: state["points"]
    task._apple_contacts_target_fixture = lambda: state["contact"]
    task._raw_success = lambda: state["door_open"]

    assert task._apple_inside_with_door_open()
    state["points"] = np.array([site_center, site_center + np.array([0.21, 0.0, 0.0])])
    assert task._apple_in_microwave()
    assert task._apple_inside_with_door_open()
    state["points"] = np.array([site_center])
    state["contact"] = False
    assert not task._apple_inside_with_door_open()
    state["contact"] = True
    state["points"] = np.array([site_center + np.array([0.21, 0.0, 0.0])])
    assert not task._apple_inside_with_door_open()
    state["points"] = np.array([site_center])
    state["door_open"] = False
    assert not task._apple_inside_with_door_open()


def test_top_placement_requires_support_contact_release_and_open_door():
    task = make_task(TakeToTop)
    state = {"on_top": True, "released": True, "door_open": True}
    task._apple_top_region = lambda: object()
    task.object_on_region = lambda name, region: state["on_top"]
    task.object_released = lambda name, th: state["released"]
    task._raw_success = lambda: state["door_open"]

    assert task._apple_on_top_with_door_open()
    state["on_top"] = False
    assert not task._apple_on_top_with_door_open()
    state["on_top"] = True
    state["released"] = False
    assert not task._apple_on_top_with_door_open()
    state["released"] = True
    state["door_open"] = False
    assert not task._apple_on_top_with_door_open()
