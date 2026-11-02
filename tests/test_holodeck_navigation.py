import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

from robocasa.environments.holodeck.tasks.approach_window import (
    HolodeckApproachWindow,
)
from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckApproachDeskThenOpenLaptop,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_SPECS_PATH = REPO_ROOT / "robocasa/models/assets/scenes/holodeck_task_specs.yaml"
SCENES_DIR = REPO_ROOT / "robocasa/models/assets/scenes/holodeck-objaverse-train"


def _body_xy(scene_id, body_name):
    scene_path = SCENES_DIR / f"train_{scene_id}.xml"
    root = ET.parse(scene_path).getroot()
    body = root.find(f".//body[@name='{body_name}']")
    assert body is not None
    return tuple(float(value) for value in body.attrib["pos"].split()[:2])


@pytest.mark.parametrize(
    ("scene_id", "trash_can_body"),
    (
        (117, "holodeck117_trash_can"),
        (1018, "holodeck1018_trash_can"),
        (1197, "holodeck1197_trash_can"),
        (1664, "holodeck1664_trash_can"),
        (2025, "holodeck2025_trash_can"),
    ),
)
def test_approach_trash_can_spawn_faces_target(scene_id, trash_can_body):
    with TASK_SPECS_PATH.open() as file:
        spec = yaml.safe_load(file)["scenes"][f"train_{scene_id}"]["approach_trash_can"]

    robot_xy = tuple(float(value) for value in spec["robot_pose"]["pos"][:2])
    target_xy = _body_xy(scene_id, trash_can_body)
    expected_yaw = math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0])
    yaw_error = math.atan2(
        math.sin(float(spec["robot_pose"]["yaw"]) - expected_yaw),
        math.cos(float(spec["robot_pose"]["yaw"]) - expected_yaw),
    )

    assert abs(yaw_error) < 1e-6
    assert math.dist(robot_xy, target_xy) > float(spec["success_radius"]) + 0.15


@pytest.mark.parametrize("scene_id", (117, 1018))
def test_desk_laptop_trash_composite_faces_desk(scene_id):
    with TASK_SPECS_PATH.open() as file:
        scenes = yaml.safe_load(file)["scenes"]
    spec = scenes[f"train_{scene_id}"]["approach_trash_can"]

    composite_pose = spec["desk_composite_robot_pose"]
    robot_xy = tuple(float(value) for value in composite_pose["pos"][:2])
    trash_pose = spec["composite_robot_pose"]
    trash_robot_xy = tuple(float(value) for value in trash_pose["pos"][:2])
    trash_xy = _body_xy(scene_id, spec["trash_can_body"])
    assert math.dist(trash_robot_xy, trash_xy) > float(spec["success_radius"]) + 0.15
    target_xy = _body_xy(
        scene_id,
        scenes[f"train_{scene_id}"]["approach_desk"]["desk_body"],
    )

    expected_yaw = math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0])
    yaw_error = math.atan2(
        math.sin(float(composite_pose["yaw"]) - expected_yaw),
        math.cos(float(composite_pose["yaw"]) - expected_yaw),
    )
    assert abs(yaw_error) < 1e-6


def test_1197_approach_desk_open_laptop_uses_its_remote_composite_spawn():
    with TASK_SPECS_PATH.open() as file:
        scenes = yaml.safe_load(file)["scenes"]

    task = object.__new__(HolodeckApproachDeskThenOpenLaptop)
    task.navigation_spec = scenes["train_1197"]["approach_desk"]
    assert (
        task._approach_desk_spawn_pose()
        == task.navigation_spec["desk_composite_robot_pose"]
    )

    # This fallback preserves this composite's configured starts in all other
    # supported scenes, which have no laptop-specific remote desk spawn.
    for scene_id in (117, 1018, 1664, 2025):
        task.navigation_spec = scenes[f"train_{scene_id}"]["approach_desk"]
        assert task._approach_desk_spawn_pose() == task.navigation_spec["robot_pose"]


def test_1018_desk_laptop_window_composite_uses_remote_spawn():
    with TASK_SPECS_PATH.open() as file:
        spec = yaml.safe_load(file)["scenes"]["train_1018"]["approach_window"]

    composite_pose = spec["composite_robot_pose"]
    robot_xy = tuple(float(value) for value in composite_pose["pos"][:2])
    target_xy = _body_xy(1018, spec["window_body"])

    assert math.dist(robot_xy, target_xy) > float(spec["success_radius"]) + 0.15


def test_1197_trash_desk_chair_composite_uses_remote_desk_spawn():
    with TASK_SPECS_PATH.open() as file:
        scene_specs = yaml.safe_load(file)["scenes"]["train_1197"]

    trash_spec = scene_specs["approach_trash_can"]
    composite_pose = trash_spec.get("composite_robot_pose")
    assert composite_pose is not None

    robot_xy = tuple(float(value) for value in composite_pose["pos"][:2])
    desk_xy = _body_xy(1197, scene_specs["approach_desk_then_pull_chair"]["desk_body"])
    assert (
        math.dist(robot_xy, desk_xy)
        > float(scene_specs["approach_desk_then_pull_chair"]["success_radius"]) + 0.15
    )

    trash_xy = _body_xy(1197, trash_spec["trash_can_body"])
    expected_yaw = math.atan2(trash_xy[1] - robot_xy[1], trash_xy[0] - robot_xy[0])
    yaw_error = math.atan2(
        math.sin(float(composite_pose["yaw"]) - expected_yaw),
        math.cos(float(composite_pose["yaw"]) - expected_yaw),
    )
    assert abs(yaw_error) < 1e-6


@pytest.mark.parametrize("scene_id", (117, 1018, 1197, 1664, 2025))
def test_desk_laptop_window_composite_has_explicit_remote_spawn(scene_id):
    with TASK_SPECS_PATH.open() as file:
        scene_specs = yaml.safe_load(file)["scenes"][f"train_{scene_id}"]
    spec = scene_specs["approach_window"]

    composite_pose = spec.get("composite_robot_pose")
    assert composite_pose is not None
    assert len(composite_pose["pos"]) == 3
    assert math.isfinite(float(composite_pose["yaw"]))
    robot_xy = tuple(float(value) for value in composite_pose["pos"][:2])
    desk_spec = scene_specs["approach_desk"]
    desk_xy = _body_xy(scene_id, desk_spec["desk_body"])
    assert math.dist(robot_xy, desk_xy) > 1.15
    assert math.dist(robot_xy, _body_xy(scene_id, spec["window_body"])) > 1.15
    expected_yaw = math.atan2(desk_xy[1] - robot_xy[1], desk_xy[0] - robot_xy[0])
    yaw_error = math.atan2(
        math.sin(float(composite_pose["yaw"]) - expected_yaw),
        math.cos(float(composite_pose["yaw"]) - expected_yaw),
    )
    assert abs(yaw_error) < 1e-6


@pytest.mark.parametrize("scene_id", (117, 1018))
def test_approach_window_atomic_uses_rightmost_calibration(scene_id):
    assert HolodeckApproachWindow.SPEC_TASK == "approach_window"

    with TASK_SPECS_PATH.open() as file:
        scene_specs = yaml.safe_load(file)["scenes"][f"train_{scene_id}"]

    atomic_spec = scene_specs["approach_window"]
    composite_spec = scene_specs["approach_window"]
    robot_xy = tuple(float(value) for value in atomic_spec["robot_pose"]["pos"][:2])
    target_xy = _body_xy(scene_id, atomic_spec["window_body"])
    expected_yaw = math.atan2(target_xy[1] - robot_xy[1], target_xy[0] - robot_xy[0])
    yaw_error = math.atan2(
        math.sin(float(atomic_spec["robot_pose"]["yaw"]) - expected_yaw),
        math.cos(float(atomic_spec["robot_pose"]["yaw"]) - expected_yaw),
    )

    assert atomic_spec["window_body"] == composite_spec["window_body"]
    assert atomic_spec["lang"] == "walk to the rightmost window and stand nearby"
    assert composite_spec["lang"] == "walk to the rightmost window and stand nearby"
    assert abs(yaw_error) < 1e-6


@pytest.mark.parametrize("scene_id", (117, 1018, 1197, 1664, 2025))
def test_approach_trash_can_applies_calibrated_pose(scene_id):
    import robosuite
    import robocasa  # noqa: F401 - registers Holodeck environments

    env = robosuite.make(
        env_name="HolodeckApproachTrashCan",
        robots="G1FloatingBody",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        scene_ids=scene_id,
    )
    try:
        env.reset()
        root_body = env.robots[0].robot_model.root_body
        body_id = env.sim.model.body_name2id(root_body)
        actual_xy = np.asarray(env.sim.data.xpos[body_id], dtype=float)[:2]
        spec = yaml.safe_load(TASK_SPECS_PATH.read_text())["scenes"][
            f"train_{scene_id}"
        ]["approach_trash_can"]
        expected_xy = np.asarray(spec["robot_pose"]["pos"][:2], dtype=float)
        assert np.allclose(actual_xy, expected_xy, atol=1e-6)

        quat = np.asarray(env.sim.data.xquat[body_id], dtype=float)
        w, x, y, z = quat
        forward = np.array([1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y + w * z)])
        target_xy = np.asarray(
            env.sim.data.get_body_xpos(spec["trash_can_body"]), dtype=float
        )[:2]
        target_direction = target_xy - actual_xy
        target_direction /= np.linalg.norm(target_direction)
        assert np.dot(forward, target_direction) > 1.0 - 1e-6
    finally:
        env.close()


@pytest.mark.parametrize("scene_id", (1664, 2025))
def test_approach_desk_applies_calibrated_laptop_pose(scene_id):
    import numpy as np
    import robosuite
    import robocasa  # noqa: F401 - registers Holodeck environments

    env = robosuite.make(
        env_name="HolodeckApproachDesk",
        robots="G1FloatingBody",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        scene_ids=scene_id,
    )
    try:
        env.reset()
        scene_name = f"train_{scene_id}"
        with TASK_SPECS_PATH.open() as file:
            spec = yaml.safe_load(file)["scenes"][scene_name]["approach_desk"]
        body_id = env.sim.model.body_name2id(
            spec.get("target_body", "spawned_laptop_Laptop_26")
        )
        actual_pos = np.asarray(env.sim.data.xpos[body_id], dtype=float)
        expected_pos = np.asarray(spec["laptop_pose"]["pos"], dtype=float)
        assert np.allclose(actual_pos, expected_pos, atol=1e-6)
    finally:
        env.close()


@pytest.mark.parametrize("scene_id", (1664, 2025))
def test_approach_trash_can_reuses_open_laptop_desktop_layout(scene_id):
    import robosuite
    import robocasa  # noqa: F401 - registers Holodeck environments

    env = robosuite.make(
        env_name="HolodeckApproachTrashCan",
        robots="G1FloatingBody",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        scene_ids=scene_id,
    )
    try:
        env.reset()
        scene_name = f"train_{scene_id}"
        specs = yaml.safe_load(TASK_SPECS_PATH.read_text())["scenes"][scene_name]
        laptop_spec = specs["open_laptop"]

        assert laptop_spec["source_body"] not in env.sim.model.body_names
        body_id = env.sim.model.body_name2id("spawned_laptop_Laptop_26")
        assert np.allclose(
            env.sim.data.xpos[body_id], laptop_spec["laptop_pose"]["pos"], atol=1e-6
        )
    finally:
        env.close()


def test_holodeck_default_asset_layout_is_configured_for_unrelated_tasks():
    from robocasa.models.scenes.holodeck_task_specs import (
        get_holodeck_default_asset_layout,
    )

    assert get_holodeck_default_asset_layout("train_1664") == "open_laptop"
    assert get_holodeck_default_asset_layout("train_2025") == "open_laptop"
    assert get_holodeck_default_asset_layout("train_1197") is None


@pytest.mark.parametrize("scene_id", (1664, 2025))
def test_unrelated_holodeck_task_receives_default_asset_layout(scene_id):
    import robosuite
    import robocasa  # noqa: F401 - registers Holodeck environments

    env = robosuite.make(
        env_name="HolodeckApproachWindow",
        robots="G1FloatingBody",
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        scene_ids=scene_id,
    )
    try:
        env.reset()
        assert "spawned_laptop_Laptop_26" in env.sim.model.body_names
    finally:
        env.close()


@pytest.mark.parametrize("scene_id", (117, 1018, 1197, 1664, 2025))
def test_approach_trash_can_scene_has_named_collision_body(scene_id):
    root = ET.parse(SCENES_DIR / f"train_{scene_id}.xml").getroot()
    body_name = f"holodeck{scene_id}_trash_can"
    body = root.find(f".//body[@name='{body_name}']")
    assert body is not None
    collision = body.find(f"./geom[@name='{body_name}_collision']")
    assert collision is not None
    assert collision.get("contype") != "0"
    assert collision.get("conaffinity") != "0"
