"""Focused contracts, ordering, and one reset for the 1477 cube workflow."""

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import robosuite

from robocasa.environments import holodeck
from robocasa.environments.holodeck import tasks
from robocasa.environments.holodeck.holodeck import (
    HolodeckEnv,
    REGISTERED_HOLODECK_ENVS,
)
from robocasa.environments.holodeck.tasks.multi_hop.cube_laptop_workflow import (
    HolodeckApproachWindowThenCubeInPlateThenOpenLaptop as Workflow,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robosuite.environments.base import REGISTERED_ENVS
from validate_ithor_tasks import _load_reset_controller


SCENES = Path(__file__).resolve().parents[1] / "robocasa/models/assets/scenes"


def test_registration_and_exact_scene_117_laptop_copy():
    for catalog in (holodeck, tasks):
        assert getattr(catalog, Workflow.__name__) is Workflow
        assert catalog.__all__.count(Workflow.__name__) == 1
    assert REGISTERED_HOLODECK_ENVS[Workflow.__name__] is Workflow
    assert REGISTERED_ENVS[Workflow.__name__] is Workflow
    assert Workflow.SUPPORTED_SCENE_IDS == (1477,)
    assert Workflow.SPEC_TASK == "cube_in_plate"
    assert Workflow.USE_DEFAULT_ASSET_LAYOUT is False

    task = object.__new__(Workflow)
    task.workflow_spec = get_holodeck_task_spec(
        "train_1477", Workflow.WORKFLOW_SPEC_TASK
    )
    task.laptop_spec = get_holodeck_task_spec("train_117", "open_laptop")
    captured = {}
    task._append_xml_fixture_model = lambda root, *args, **kwargs: captured.update(
        root=root,
        kwargs=kwargs,
    )
    task._add_scene_117_laptop()
    source = ET.parse(SCENES / "holodeck-objaverse-train/train_117.robocasa.xml")
    source_body = source.find(
        f"./worldbody/body[@name='{task.laptop_spec['target_body']}']"
    )
    assert ET.tostring(captured["root"].find("./worldbody/body")) == ET.tostring(
        source_body
    )
    assert captured["kwargs"]["ensure_inertials"] is False
    assert all(
        "Laptop" in asset.get("name", "")
        for asset in captured["root"].findall("./asset/*")
    )
    room = ET.parse(SCENES / "holodeck-objaverse-train/train_1477.robocasa.xml")
    windows = [
        b
        for b in room.findall("./worldbody/body")
        if b.get("name", "").startswith("window_")
    ]
    rightmost = max(windows, key=lambda b: float(b.get("pos").split()[0]))
    assert rightmost.get("name") == task.workflow_spec["window_body"]


def _fake_workflow(monkeypatch, control_freq):
    monkeypatch.setattr(
        HolodeckEnv, "_post_action", lambda self, action: (0, False, {})
    )
    task = object.__new__(Workflow)
    task.control_freq = control_freq
    task._atomic_tasks = task.ATOMIC_TASKS
    task._reset_composite_progress()
    task._cube_stage_armed = False
    task._laptop_stage_armed = False
    state = {"window": False, "cube": False, "qpos": 1.5708}
    task._window_reached = lambda: state["window"]
    task._cube_placed = lambda: state["cube"]
    task.laptop_spec = {"closed_qpos": 1.5708, "open_qpos": 0.0}
    task.laptop_joint = "lid"
    task.sim = SimpleNamespace(
        data=SimpleNamespace(get_joint_qpos=lambda name: state["qpos"])
    )
    return task, state


def _advance(task, count):
    for _ in range(count):
        task._post_action(None)


@pytest.mark.parametrize("control_freq", (20, 50))
def test_ordered_success_and_interrupted_final_hold(monkeypatch, control_freq):
    task, state = _fake_workflow(monkeypatch, control_freq)
    state["window"] = True
    _advance(task, 10)
    assert task.stage_index == 1
    state["cube"] = True
    _advance(task, control_freq)
    assert task.stage_index == 2
    state["qpos"] = 0.0
    _advance(task, control_freq - 1)
    assert not task._check_success()
    state["cube"] = False
    task._post_action(None)
    assert task._stage_hold_steps == 0
    state["cube"] = True
    _advance(task, control_freq)
    assert task._check_success()


def test_early_placement_and_opening_must_be_undone(monkeypatch):
    task, state = _fake_workflow(monkeypatch, 20)
    state.update(cube=True, qpos=0.0)
    _advance(task, 30)
    assert task.stage_index == 0
    state["window"] = True
    _advance(task, 10)
    _advance(task, 30)
    assert task.stage_index == 1
    state["cube"] = False
    task._post_action(None)
    state["cube"] = True
    _advance(task, 20)
    assert task.stage_index == 2
    _advance(task, 30)
    assert not task._check_success()
    state["qpos"] = 1.5708
    task._post_action(None)
    state["qpos"] = 0.0
    _advance(task, 20)
    assert task._check_success()


def test_one_headless_reset_and_tabletop_stability():
    env = robosuite.make(
        env_name=Workflow.__name__,
        robots="G1FloatingBody",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        initialization_noise=None,
        control_freq=50,
        scene_ids=[1477],
        seed=0,
    )
    try:
        env.reset()
        assert env.scene_name == "train_1477"
        assert env.stage_index == 0 and not env._check_success()
        assert env.task_spec == get_holodeck_task_spec("train_1477", "cube_in_plate")
        assert env.count_body_collisions() == 0
        assert env.room_of_point([*env.robot_base_pos[:2], 0.0]) is not None
        assert not env._cube_placed()
        assert env._laptop_open_fraction() == pytest.approx(0.0)
        assert env.workflow_spec["success_radius"] < env.__class__.ATOMIC_TASKS[
            0
        ].task_cls.success_distance(env)
        for key in ("target", "plate"):
            np.testing.assert_allclose(
                env.sim.data.get_body_xpos(env.task_spec[f"{key}_body"]),
                env.task_spec[f"{key}_pose"]["pos"],
            )
        initial_laptop = env.sim.data.get_body_xpos(env.laptop_body).copy()
        # The free imported laptop must keep its support during navigation.
        for _ in range(100):
            env.sim.step()
        np.testing.assert_allclose(
            env.sim.data.get_body_xpos(env.laptop_body),
            initial_laptop,
            atol=0.01,
        )
        assert env._laptop_open_fraction() < 0.05
        assert not env._cube_placed()
        assert env._cube_contacts_body("table_body")
        assert not env._cube_contacts_body("plate_body")
    finally:
        env.close()
