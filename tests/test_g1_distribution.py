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


G1_NAMES = {"G1", "G1FixedLowerBody", "G1FloatingBody", "G1Revo2"}


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
    ("G1Revo2", 51, True),
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
    assert {path.name for path in robot_assets.iterdir() if path.is_dir()} == {"g1", "g1_revo2"}


def test_revo2_tactile_extensions_survive_xml_merge():
    from robosuite.models.base import MujocoXML
    from robosuite.utils.mjcf_utils import xml_path_completion
    robot = create_robot("G1Revo2", idn=0)
    arena = MujocoXML(xml_path_completion("arenas/empty_arena.xml"))
    arena.merge(robot)
    model = arena.get_model()
    assert (model.nq, model.nv, model.nu, model.neq, model.nsensor) == (58, 57, 51, 10, 122)
    assert model.nplugin > 0
    for side in ("left", "right"):
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"robot0_{side}_wrist_cam_mount")
        assert model.body_mass[index] == 0


def test_mounted_revo2_reports_physical_pelvis_instead_of_floor_freejoint():
    from types import SimpleNamespace
    from robocasa.g1_host.base import G1CompositeBase

    robot = create_robot("G1Revo2", idn=0)
    robot.add_base(robot_base_factory(robot.default_base, idn=0))
    model = robot.get_model()
    data = mujoco.MjData(model)
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot.root_body)
    addr = model.jnt_qposadr[model.body_jntadr[root_id]]
    data.qpos[addr:addr + 3] = [0.1, 0.2, 0.0]
    mujoco.mj_forward(model, data)
    host = SimpleNamespace(
        robot_model=robot,
        sim=SimpleNamespace(
            model=SimpleNamespace(body_name2id=lambda name: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, name)),
            data=SimpleNamespace(body_xpos=data.xpos),
        ),
        _free_qpos_addr=addr,
    )
    np.testing.assert_allclose(G1CompositeBase._read_root_pos(host), [0.1, 0.2, 0.793], atol=1e-6)
    assert data.qpos[addr + 2] == 0.0
