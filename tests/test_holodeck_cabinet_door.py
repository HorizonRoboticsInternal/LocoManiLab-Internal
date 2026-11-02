"""Contract and focused runtime checks for the scene 117 cabinet door."""

from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import pytest
import robosuite
import yaml

from robocasa.environments import holodeck
from robocasa.environments.holodeck import tasks
from robocasa.environments.holodeck.holodeck import (
    HolodeckEnv,
    REGISTERED_HOLODECK_ENVS,
)
from robocasa.environments.holodeck.tasks.cabinet_door import HolodeckOpenCabinetDoor
from robosuite.environments.base import REGISTERED_ENVS
from validate_ithor_tasks import _load_reset_controller


SCENES_DIR = Path(__file__).resolve().parents[1] / "robocasa/models/assets/scenes"


def _spec():
    return yaml.safe_load((SCENES_DIR / "holodeck_task_specs.yaml").read_text())[
        "scenes"
    ]["train_117"]["open_cabinet_door"]


def test_cabinet_door_registration_and_scene_contract():
    cls = HolodeckOpenCabinetDoor
    assert holodeck.HolodeckOpenCabinetDoor is cls
    assert tasks.HolodeckOpenCabinetDoor is cls
    assert REGISTERED_HOLODECK_ENVS[cls.__name__] is cls
    assert REGISTERED_ENVS[cls.__name__] is cls
    assert cls.SUPPORTED_SCENE_IDS == (117,)
    assert cls.USE_DEFAULT_ASSET_LAYOUT is False
    assert cls.TARGET_JOINT_TYPES == ("hinge",)

    spec = _spec()
    assert spec["init_open_fraction"] == pytest.approx(0.20)
    assert spec["success_open_fraction"] == pytest.approx(0.80)
    assert spec["hold_seconds"] == pytest.approx(1.0)
    root = ET.parse(SCENES_DIR / "holodeck-objaverse-train/train_117.xml").getroot()
    fixture = root.find(f".//body[@name='{spec['target_fixture']}']")
    assert fixture is not None
    joint = fixture.find(f".//joint[@name='{spec['target_joint']}']")
    assert joint is not None
    assert joint.get("type", "hinge") == "hinge"
    assert tuple(map(float, joint.get("range").split())) == pytest.approx(
        (spec["open_qpos"], spec["closed_qpos"])
    )


@pytest.mark.parametrize("control_freq", (20, 50))
def test_cabinet_door_requires_continuous_one_second(monkeypatch, control_freq):
    monkeypatch.setattr(HolodeckEnv, "_post_action", lambda self, action: None)
    task = object.__new__(HolodeckOpenCabinetDoor)
    task.control_freq = control_freq
    task._get_articulation_task_spec = _spec
    task._configure_articulation_task()
    task._success_hold = 0
    state = {"fraction": 0.20}
    task.target_semantic_joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task.sim = None
    assert task.HOLD_STEPS == control_freq

    task._post_action(None)
    assert not task._check_success()
    state["fraction"] = 0.80
    for _ in range(control_freq - 1):
        task._post_action(None)
        assert not task._check_success()
    state["fraction"] = 0.799
    task._post_action(None)
    assert task._success_hold == 0
    state["fraction"] = 0.80
    for _ in range(control_freq):
        task._post_action(None)
    assert task._check_success()
    state["fraction"] = 0.799
    task._post_action(None)
    assert not task._check_success()


def test_cabinet_door_117_headless_reset():
    env = robosuite.make(
        env_name="HolodeckOpenCabinetDoor",
        robots="G1",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        initialization_noise=None,
        control_freq=50,
        scene_ids=[117],
        seed=0,
    )
    try:
        env.reset()
        spec = _spec()
        joint = env.target_semantic_joint
        joint_id = env.sim.model.joint_name2id(env.target_joint)
        assert env.sim.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE
        assert joint.fraction(env.sim) == pytest.approx(0.20)
        assert not env._raw_success()
        assert not env._check_success()
        assert env.sim.data.get_joint_qvel(env.target_joint) == pytest.approx(0.0)
        assert env.count_body_collisions() == 0
        # Holodeck floors are represented by room polygons, not iTHOR fixtures.
        base_xy = env._robot_base_xy()
        offsets = [np.zeros(2)] + [
            0.25 * np.array([np.cos(angle), np.sin(angle)])
            for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ]
        assert all(
            env.room_of_point([*(base_xy + offset), 0.0]) is not None
            for offset in offsets
        )
        assert env.get_ep_meta()["target_joint"] == spec["target_joint"]
        assert env.HOLD_STEPS == 50

        env.sim.data.set_joint_qpos(
            env.target_joint,
            spec["closed_qpos"] + 0.80 * (spec["open_qpos"] - spec["closed_qpos"]),
        )
        env.sim.forward()
        assert env._raw_success()
        assert env.count_body_collisions() == 0
    finally:
        env.close()
