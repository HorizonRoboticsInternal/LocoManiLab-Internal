"""Contracts and focused reset checks for the Holodeck 1477 atomic tasks."""

from pathlib import Path
from types import SimpleNamespace
import json
import xml.etree.ElementTree as ET

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
from robocasa.environments.holodeck.tasks.cube_in_plate import HolodeckCubeInPlate
from robocasa.environments.holodeck.tasks.lift_plant_from_low_cabinet import (
    HolodeckLiftPlantFromLowCabinet,
)
from robosuite.environments.base import REGISTERED_ENVS
from validate_ithor_tasks import _load_reset_controller


SCENES_DIR = Path(__file__).resolve().parents[1] / "robocasa/models/assets/scenes"


def _spec(task_name):
    return yaml.safe_load((SCENES_DIR / "holodeck_task_specs.yaml").read_text())[
        "scenes"
    ]["train_1477"][task_name]


def test_1477_task_registration_and_scene_contract():
    plant_cls = HolodeckLiftPlantFromLowCabinet
    cube_cls = HolodeckCubeInPlate
    for cls in (plant_cls, cube_cls):
        assert getattr(holodeck, cls.__name__) is cls
        assert getattr(tasks, cls.__name__) is cls
        assert REGISTERED_HOLODECK_ENVS[cls.__name__] is cls
        assert REGISTERED_ENVS[cls.__name__] is cls
        assert cls.SUPPORTED_SCENE_IDS == (1477,)
        assert cls.USE_DEFAULT_ASSET_LAYOUT is False

    root = ET.parse(
        SCENES_DIR / "holodeck-objaverse-train/train_1477.robocasa.xml"
    ).getroot()
    metadata = json.loads(
        (SCENES_DIR / "holodeck-objaverse-train/train_1477_metadata.json").read_text()
    )["objects"]
    plant = _spec(plant_cls.SPEC_TASK)
    cube = _spec(cube_cls.SPEC_TASK)
    assert plant["source_body"] == ("stand_007a3ac7acf8cfc943375cf9dded27c1_1_0_0")
    assert plant["target_body"] == ("houseplant_70451f91981395edf773114de3a28b67_1_0_0")
    for spec_key, spec in (("plant", plant), ("cube", cube)):
        target = root.find(f".//body[@name='{spec['target_body']}']")
        assert target is not None, spec_key
        assert target.find("./joint[@type='free']") is not None, spec_key
    source = root.find(f".//body[@name='{plant['source_body']}']")
    assert source is not None
    assert metadata[plant["source_body"]]["asset_id"] == "Side_Table_309_1"
    assert metadata[plant["source_body"]]["category"] == "SideTable"
    assert metadata[plant["target_body"]]["category"] == "HousePlant"
    assert root.find(f".//body[@name='{cube['plate_body']}']") is not None
    assert root.find(f".//body[@name='{cube['table_body']}']") is not None


def test_plant_lift_requires_height_and_continuous_hold(monkeypatch):
    monkeypatch.setattr(HolodeckEnv, "_post_action", lambda self, action: None)
    task = object.__new__(HolodeckLiftPlantFromLowCabinet)
    task.control_freq = 20
    task._plant_initial_z = 1.0
    task._success_hold = 0
    state = {"z": 1.0}
    task._plant_root_z = lambda: state["z"]

    state["z"] = 1.29
    task._post_action(None)
    assert task._success_hold == 0

    state["z"] = 1.30
    for _ in range(task.hold_steps):
        task._post_action(None)
    assert task._check_success()

    state["z"] = 1.29
    task._post_action(None)
    assert task._success_hold == 0
    assert not task._check_success()


@pytest.mark.parametrize(
    "contact_pairs, expected",
    (
        ((), False),
        (((0, 1),), True),
        (((1, 0),), True),
        (((0, 2),), False),
        (((0, 1), (0, 2)), False),
        (((1, 0), (2, 0)), False),
        (((0, 1), (1, 2)), True),
        (((0, 1), (0, 3)), True),
    ),
)
def test_cube_placement_uses_plate_and_table_contacts(contact_pairs, expected):
    task = object.__new__(HolodeckCubeInPlate)
    # Contacts belong to child bodies: cube=2, plate=4, table=6, hand=7.
    task._body_id = lambda key: {
        "target_body": 1,
        "plate_body": 3,
        "table_body": 5,
    }[key]
    task._body_tree_ids = lambda roots: {roots[0], roots[0] + 1}
    task.sim = SimpleNamespace(
        model=SimpleNamespace(geom_bodyid=np.array([2, 4, 6, 7])),
        data=SimpleNamespace(
            ncon=len(contact_pairs),
            contact=[
                SimpleNamespace(geom1=first, geom2=second)
                for first, second in contact_pairs
            ],
        ),
    )

    assert task._cube_placed() is expected


@pytest.mark.parametrize("control_freq", (20, 50))
@pytest.mark.parametrize("interruption", ("plate_lost", "table_contact"))
def test_cube_placement_requires_one_second_of_continuous_contact(
    monkeypatch,
    control_freq,
    interruption,
):
    monkeypatch.setattr(HolodeckEnv, "_post_action", lambda self, action: None)
    task = object.__new__(HolodeckCubeInPlate)
    task.control_freq = control_freq
    task._success_hold = 0
    contacts = {"plate_body": True, "table_body": False}
    task._cube_contacts_body = lambda key: contacts[key]

    assert task.hold_steps == control_freq
    for _ in range(task.hold_steps - 1):
        task._post_action(None)
    assert not task._check_success()

    interrupted_body = "plate_body" if interruption == "plate_lost" else "table_body"
    contacts[interrupted_body] = not contacts[interrupted_body]
    task._post_action(None)
    assert task._success_hold == 0
    assert not task._check_success()

    contacts.update(plate_body=True, table_body=False)
    for _ in range(task.hold_steps - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    assert task.reward() == 1.0

    contacts[interrupted_body] = not contacts[interrupted_body]
    task._post_action(None)
    assert task._success_hold == 0
    assert not task._check_success()
    assert task.reward() == 0.0


@pytest.mark.parametrize(
    "env_name, task_name, robot_name",
    (
        ("HolodeckLiftPlantFromLowCabinet", "lift_plant_from_low_cabinet", "G1"),
        ("HolodeckCubeInPlate", "cube_in_plate", "G1FloatingBody"),
    ),
)
def test_1477_atomic_task_headless_reset(env_name, task_name, robot_name):
    env = robosuite.make(
        env_name=env_name,
        robots=robot_name,
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
        assert env.task_spec == _spec(task_name)
        assert not env._check_success()
        if task_name == "cube_in_plate":
            assert env.count_body_collisions() == 0
            assert all(
                body not in env.sim.model.body_names
                for body in env.task_spec["remove_scene_bodies"]
            )
            for key in ("target", "plate"):
                body_id = env._body_id(f"{key}_body")
                np.testing.assert_allclose(
                    env.sim.data.body_xpos[body_id],
                    env.task_spec[f"{key}_pose"]["pos"],
                )
    finally:
        env.close()
