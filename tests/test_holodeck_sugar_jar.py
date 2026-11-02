from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from robocasa.environments.holodeck.tasks.mug_lid import (
    HolodeckCloseMugLid,
    HolodeckOpenMugLid,
)
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckMultiHopComposite,
)
from robocasa.environments.holodeck.tasks.multi_hop.object_workflows import (
    HolodeckOpenJarAddSugarThenCloseJar,
    HolodeckSugarCubeThenCloseJar,
)
from robocasa.environments.holodeck.tasks.sugar_cube_in_jar import (
    HolodeckSugarCubeInJar,
)


@pytest.mark.parametrize(
    "task_class",
    (HolodeckSugarCubeThenCloseJar, HolodeckOpenJarAddSugarThenCloseJar),
)
def test_jar_close_stage_uses_only_open_then_closed_target_state(task_class):
    task = object.__new__(task_class)
    task._lid_open_when_closing = True
    task._lid_is_closed = lambda: False

    assert not task._lid_closed_after_sugar()

    task._lid_is_closed = lambda: True
    assert task._lid_closed_after_sugar()


def test_lid_atomic_predicates_do_not_require_grasp_history():
    open_task = object.__new__(HolodeckOpenMugLid)
    open_task._objects_in_contact = lambda *_names: False
    assert open_task._lid_is_open()

    close_task = object.__new__(HolodeckCloseMugLid)
    close_task._closed_lid_offset = np.zeros(3)
    close_task._lid_offset = lambda: np.array([0.01, 0.0, 0.0])
    close_task.task_spec = {"close_distance": 0.04}
    assert close_task._lid_is_closed()


def test_sugar_jar_composite_open_predicate_rejects_closed_pose_without_contact():
    """A contact-free closed pose must not complete the open-jar stage."""
    task = object.__new__(HolodeckOpenJarAddSugarThenCloseJar)
    task._lid_is_closed = lambda: True
    task._objects_in_contact = lambda *_names: False
    assert not task._lid_is_open()

    task._lid_is_closed = lambda: False
    assert task._lid_is_open()


def test_composite_close_target_follows_sugar_jar_pose():
    task = object.__new__(HolodeckSugarCubeThenCloseJar)
    task.object_pos = lambda _name: np.array([5.15, 1.45, 0.70740])
    task.close_lid_spec = {
        "mug_pose": {"pos": [5.02380371, 1.45, 0.70740]},
        "closed_lid_pose": {
            "pos": [5.02380371, 1.45, 0.80349],
            "quat": [1.0, 0.0, 0.0, 0.0],
        },
        "close_distance": 0.04,
    }

    target = task._closed_lid_pose_for_jar()
    assert target["pos"] == pytest.approx([5.15, 1.45, 0.80349])


def test_sugar_then_close_jar_uses_composite_only_layout():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        scenes = yaml.safe_load(file)["scenes"]

    atomic = scenes["train_117"]["sugar_cube_in_jar"]
    composite = scenes["train_117"]["sugar_cube_then_close_jar"]
    assert HolodeckSugarCubeInJar.SPEC_TASK == "sugar_cube_in_jar"
    assert HolodeckSugarCubeThenCloseJar.SPEC_TASK == "sugar_cube_then_close_jar"
    assert atomic["jar_pose"]["pos"] == [5.15, 1.45, 0.70740]
    assert np.allclose(
        [
            composite[name]["pos"][:2]
            for name in ("sugar_pose", "jar_pose", "lid_pose")
        ],
        [[5.12380371, 1.25], [5.12380371, 1.45], [5.12380371, 1.65]],
    )
    composite_1018 = scenes["train_1018"]["sugar_cube_then_close_jar"]
    assert np.allclose(
        [
            composite_1018[name]["pos"][:2]
            for name in ("sugar_pose", "jar_pose", "lid_pose")
        ],
        [[3.36892754, 1.05], [3.36892754, 1.25], [3.36892754, 1.45]],
    )


def test_new_sugar_jar_scenes_include_open_desk_support_calibration():
    with open("robocasa/models/assets/scenes/holodeck_task_specs.yaml") as file:
        scenes = yaml.safe_load(file)["scenes"]

    for scene_name, support_z, support_top_z in (
        ("train_1664", 0.7473, 0.7486),
        ("train_2025", 0.7678, 0.7688),
    ):
        spec = scenes[scene_name]["sugar_cube_in_jar"]
        assert spec["support_z"] == pytest.approx(support_z)
        assert spec["support_top_z"] == pytest.approx(support_top_z)


def test_sugar_cube_requires_all_bbox_points_inside_outer_octagon():
    task = object.__new__(HolodeckSugarCubeInJar)
    jar_points = np.array(
        [
            [-1.0, -1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, 1.0, 1.0],
            [1.0, -1.0, -1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, -1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    task._sugar_and_jar_local_bbox_points = lambda: (
        np.array(
            [[0.2, 0.2, -0.1], [0.2, 0.2, 0.1]]
        ),
        jar_points,
    )
    assert task._sugar_in_jar()

    task._sugar_and_jar_local_bbox_points = lambda: (
        np.array(
            [[0.9, 0.9, -0.1], [0.9, 0.9, 0.1]]
        ),
        jar_points,
    )
    assert not task._sugar_in_jar()


@pytest.mark.parametrize(
    "task_class",
    (HolodeckSugarCubeThenCloseJar, HolodeckOpenJarAddSugarThenCloseJar),
)
def test_jar_composite_stages_inherit_one_second_atomic_hold(task_class):
    composite = object.__new__(HolodeckMultiHopComposite)
    composite.control_freq = 20
    jar_stages = [
        stage for stage in task_class.ATOMIC_TASKS if stage.task_cls is not None
    ]
    assert all(stage.hold_steps is None for stage in jar_stages)
    assert all(
        HolodeckMultiHopComposite._required_stage_hold_steps(composite, stage) == 20
        for stage in jar_stages
    )


def test_sugar_jar_success_diagnostics_log_every_five_seconds(capsys):
    task = object.__new__(HolodeckSugarCubeThenCloseJar)
    task.scene_name = "train_117"
    task.sim = SimpleNamespace(data=SimpleNamespace(time=4.99))
    task._success_debug_next_log_time = 5.0
    task._sugar_in_jar = lambda: True
    task._sugar_and_jar_local_bbox_points = lambda: (
        np.zeros((8, 3)),
        np.zeros((8, 3)),
    )
    task.JAR_BBOX_TOLERANCE = 0.001
    task._lid_open_when_closing = True
    task._lid_is_closed = lambda: True
    task._atomic_tasks = (
        SimpleNamespace(name="add_sugar"),
        SimpleNamespace(name="close_jar"),
    )
    task._stage_index = 1
    task._stage_hold_steps = 12
    task._required_stage_hold_steps = lambda _stage: 20

    task._log_success_conditions()
    assert capsys.readouterr().out == ""

    task.sim.data.time = 5.0
    task._log_success_conditions()
    assert "t=5.00s" in capsys.readouterr().out
    assert task._success_debug_next_log_time == 10.0
