"""Structural checks for the asset-free default teleoperation scene."""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from robosuite.models.bases import NoActuationBase
from robosuite.models.robots.manipulators.g1_robot import G1FloatingBody
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS
from robosuite.environments.base import REGISTERED_ENVS

import robocasa  # noqa: F401 - register public environments
from robocasa.environments.empty.empty_env import EmptyEnv
from robocasa.models.arenas import NativeFloorArena


def test_empty_env_is_registered_and_uses_native_floor():
    assert REGISTERED_ENVS["EmptyEnv"] is EmptyEnv
    assert EmptyEnv.manual_episode_save is True
    assert isinstance(EmptyEnv.__new__(EmptyEnv)._make_arena(), NativeFloorArena)


def test_benchmark_g1_overrides_optional_robosuite_models_registration():
    assert REGISTERED_ROBOTS["G1FloatingBody"] is G1FloatingBody


def test_native_floor_has_one_geom_and_no_file_backed_assets():
    arena_xml = (
        Path(robocasa.__file__).parent
        / "models"
        / "assets"
        / "arenas"
        / "native_floor_arena.xml"
    )
    root = ET.parse(arena_xml).getroot()
    geoms = list(root.find("worldbody").iter("geom"))

    assert [(geom.get("name"), geom.get("type")) for geom in geoms] == [
        ("floor", "plane")
    ]
    assert geoms[0].get("size") == "0 0 0.05"
    assert geoms[0].get("friction") == "0.75 .1 .1"
    assert geoms[0].get("material") == "groundplane"
    assert geoms[0].get("group") == "1"
    assert root.find("visual/headlight") is not None
    assert root.find("visual/global").get("azimuth") == "-140"
    assert all("file" not in element.attrib for element in root.iter())


def test_floating_g1_pelvis_inertia_moves_with_the_physical_robot():
    robot = G1FloatingBody(idn=0)
    robot.add_base(NoActuationBase(idn=0))

    root = next(body for body in robot.worldbody if body.get("name") == robot.root_body)
    mount = root.find(".//body[@name='manipulator_mount']")

    assert mount is not None
    assert root.find("freejoint") is not None
    assert float(root.find("inertial").get("mass")) == pytest.approx(1e-6)

    pelvis_inertial = mount.find("inertial")
    assert pelvis_inertial is not None
    assert float(pelvis_inertial.get("mass")) == pytest.approx(3.813)
    assert float(pelvis_inertial.get("pos").split()[2]) == pytest.approx(-0.07605)
