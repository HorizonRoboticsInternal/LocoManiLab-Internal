"""G1-only runtime boundaries, using bundled MJCF resources and no policies.

These checks construct XML model wrappers, not simulation environments. They
work against a source checkout or an installed robosuite wheel and need no
scene packs, controller weights, camera/robot SDKs, or network access.
"""

from pathlib import Path

import numpy as np
import pytest
import mujoco

import robosuite
from robosuite.models.bases import BASE_MAPPING, robot_base_factory
from robosuite.models.grippers import GRIPPER_MAPPING, NullGripper, gripper_factory
from robosuite.models.robots.robot_model import REGISTERED_ROBOTS, create_robot
from robosuite.robots import ROBOT_CLASS_MAPPING, LeggedRobot


G1_NAMES = {"G1", "G1FixedLowerBody", "G1FloatingBody"}


def test_default_robot_registries_expose_only_concrete_g1_variants():
    assert set(REGISTERED_ROBOTS) == G1_NAMES
    assert set(ROBOT_CLASS_MAPPING) == G1_NAMES
    assert all(ROBOT_CLASS_MAPPING[name] is LeggedRobot for name in G1_NAMES)


@pytest.mark.parametrize("name", ["PandaOmron", "Panda", "GR1", "UR5e"])
def test_removed_robot_is_rejected_before_reading_assets(name, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Unsupported robot lookup attempted to load an asset")

    monkeypatch.setattr("robosuite.models.base.ET.parse", forbidden)
    with pytest.raises(Exception) as error:
        create_robot(name)
    message = str(error.value)
    assert name in message
    assert "G1" in message


@pytest.mark.parametrize("name, joint_count, floating", [
    ("G1", 43, True),
    ("G1FixedLowerBody", 31, False),
    ("G1FloatingBody", 43, True),
])
def test_g1_variants_keep_joint_layout_and_complete_asset_dependencies(
    name, joint_count, floating,
):
    robot = create_robot(name, idn=0)
    assert len(robot.joints) == joint_count
    assert len(robot.actuators) == joint_count
    assert np.asarray(robot.init_qpos).shape == (joint_count,)
    assert (robot.worldbody.find(".//freejoint") is not None) is floating
    assert robot.default_gripper == {"right": None, "left": None}
    assert robot.default_base in BASE_MAPPING

    dependencies = [Path(node.get("file")) for node in robot.asset if node.get("file")]
    assert dependencies, "G1 model unexpectedly has no mesh dependencies"
    for path in dependencies:
        assert path.is_file(), f"G1 model dependency was omitted: {path}"
    model = mujoco.MjModel.from_xml_string(robot.get_xml())
    assert model.nu == joint_count
    assert model.nq == joint_count + (7 if floating else 0)
    assert model.nv == joint_count + (6 if floating else 0)


def test_g1_null_gripper_factory_keeps_a_complete_no_attachment_model():
    assert set(GRIPPER_MAPPING) == {None}
    gripper = gripper_factory(None, idn="right")
    assert isinstance(gripper, NullGripper)
    assert gripper.dof == 0
    assert gripper.worldbody is not None


@pytest.mark.parametrize("name", ["FloatingLeggedBase", "NoActuationBase"])
def test_g1_required_base_factories_keep_bundled_resources(name):
    base = robot_base_factory(name, idn=0)
    assert base.worldbody.find("body") is not None
    assert np.asarray(base.top_offset).shape == (3,)
    for node in base.asset:
        if node.get("file"):
            assert Path(node.get("file")).is_file()


def test_distribution_has_no_retired_robot_asset_directories():
    robot_assets = Path(robosuite.__file__).resolve().parent / "models/assets/robots"
    assert {path.name for path in robot_assets.iterdir() if path.is_dir()} == {"g1"}
