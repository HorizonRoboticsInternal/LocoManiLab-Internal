"""Scene 315 bottom contact and uninterrupted, grasp-independent success."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from robosuite.utils.binding_utils import MjSim

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_bedroom.tasks.object_in_receptacle import (
    IThorBedroomCanIntoBasket,
)
from robocasa.environments.ithor_bedroom.tasks.trash_can import (
    IThorBedroomApproachTrashCan,
    IThorBedroomCanIntoTrashCan,
)
from robocasa.environments.ithor_task_catalog import (
    BEDROOM_PACKAGE,
    EXPECTED_TASK_MATRIX,
    discover_task_catalog,
    task_counts,
    validate_task_catalog,
)


BOTTOM_GEOM = "ashcan_27ba7c49bbccf220f68beea1b3ff0f8e_1_1_0_collision_7"


def make_env(cls, control_freq=20):
    env = object.__new__(cls)
    env.control_freq = control_freq
    env._configure_success_hold_duration()
    env._reset_task_progress()
    return env


def test_catalog_scene_scope_and_canonical_can():
    entries = discover_task_catalog([BEDROOM_PACKAGE])
    issues = validate_task_catalog(entries)
    # Existing composite expectations predate the routine task scene matrix.
    assert not [issue for issue in issues if ": task matrix changed" not in issue]
    assert task_counts(entries)[BEDROOM_PACKAGE][315] == {"atomic": 2, "composite": 2}
    assert EXPECTED_TASK_MATRIX[BEDROOM_PACKAGE][315] == {"atomic": 2, "composite": 2}
    tasks = (IThorBedroomApproachTrashCan, IThorBedroomCanIntoTrashCan)
    assert {
        entry.cls
        for entry in entries
        if 315 in entry.scene_ids and entry.family != "composite"
    } == set(tasks)
    for cls in tasks:
        assert cls.SUPPORTED_SCENE_IDS == (315,)
        with pytest.raises(ValueError, match="uncalibrated scene ids"):
            cls(scene_ids=[329])
    place = IThorBedroomCanIntoTrashCan
    assert place.SOURCE_MJCF_PATH == IThorBedroomCanIntoBasket.SOURCE_MJCF_PATH
    assert place.SOURCE_SCALE == IThorBedroomCanIntoBasket.SOURCE_SCALE
    nav_spec, place_spec = (cls.SCENE_SPECS["FloorPlan315"] for cls in tasks)
    for key in ("fixture", "bin_region"):
        assert nav_spec[key] == place_spec[key]


@pytest.mark.parametrize(
    "pairs, expected",
    [
        ([], False),
        ([(0, 2)], False),  # Side or rim contact is insufficient.
        ([(3, 1)], False),  # Another object touches the bottom.
        ([(0, 4)], False),  # Hand contact alone is insufficient.
        ([(0, 1)], True),
        ([(1, 0)], True),
        ([(0, 1), (0, 4)], True),  # Holding the can is allowed.
    ],
)
def test_only_can_bottom_contact_satisfies_success(pairs, expected):
    env = make_env(IThorBedroomCanIntoTrashCan)
    env.objects = {"item": SimpleNamespace(contact_geoms=["can_collision"])}
    env.bin_bottom_geoms = (BOTTOM_GEOM,)
    names = ["can_collision", BOTTOM_GEOM, "bin_wall", "other", "hand"]
    env.sim = SimpleNamespace(
        model=SimpleNamespace(geom_id2name=names.__getitem__),
        data=SimpleNamespace(
            ncon=len(pairs),
            contact=[SimpleNamespace(geom1=a, geom2=b) for a, b in pairs],
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("success must not inspect bounds, orientation or grasp")

    env.object_bbox_points = env.object_released = env._check_grasp = forbidden
    assert env._can_contacts_trash_can_bottom() is expected


@pytest.mark.parametrize(
    "cls", [IThorBedroomApproachTrashCan, IThorBedroomCanIntoTrashCan]
)
@pytest.mark.parametrize("control_freq", [10, 20, 30])
def test_one_second_hold_interruption_and_reset_without_grasp(cls, control_freq):
    env = make_env(cls, control_freq)
    state = {"valid": True}
    if cls is IThorBedroomApproachTrashCan:
        env._navigation_success_distance = 0.35
        env.target_fixture = object()
        env.robot_fixture_collision_distance = lambda *args, **kwargs: (
            0.35 if state["valid"] else 0.3501
        )
    else:
        env._can_contacts_trash_can_bottom = lambda: state["valid"]
    # Any accidental grasp/release requirement must fail this regression.
    def forbidden(*args, **kwargs):
        raise AssertionError("success must not inspect grasp or release")

    env.object_released = env._check_grasp = forbidden
    with patch.object(IThorBedroomEnv, "_post_action", return_value=None):
        for _ in range(control_freq - 1):
            env._post_action(None)
        assert not env._check_success()
        assert not env._check_success()  # queries cannot accumulate hold time
        state["valid"] = False
        env._post_action(None)
        assert not env._check_success()
        state["valid"] = True
        for _ in range(control_freq - 1):
            env._post_action(None)
        assert not env._check_success()
        env._post_action(None)
        assert env._check_success()
        env._reset_task_progress()
        assert not env._check_success()
        for _ in range(control_freq):
            env._post_action(None)
        assert env._check_success()
        state["valid"] = False
        env._post_action(None)
        assert not env._check_success()


def test_315_physical_bottom_contact_in_any_orientation():
    """Use native collision geometry without unrelated robot initialization."""
    task = IThorBedroomCanIntoTrashCan
    assert task.SCENE_SPECS["FloorPlan315"]["bin_bottom_geoms"] == [BOTTOM_GEOM]
    assets_dir = Path(task.SOURCE_MJCF_PATH).parents[4]
    scene = ET.parse(assets_dir / "scenes/ithor/FloorPlan315_physics.xml")
    bin_body = deepcopy(scene.find(f".//geom[@name='{BOTTOM_GEOM}']/../.."))
    for body in bin_body.iter("body"):
        for element in list(body):
            if element.tag != "body" and (
                element.tag != "geom" or "_collision_" not in element.get("name", "")
            ):
                body.remove(element)
            elif element.tag == "geom":
                element.attrib.pop("class", None)
    model = ET.Element("mujoco")
    asset = ET.SubElement(model, "asset")
    world = ET.SubElement(model, "worldbody")
    world.append(bin_body)
    can = ET.SubElement(world, "body", name="can", pos="0 0 1")
    ET.SubElement(can, "freejoint", name="can_joint")
    can_xml = ET.parse(task.SOURCE_MJCF_PATH)
    can_geoms = []
    for index, mesh in enumerate(can_xml.findall("./asset/mesh")):
        if not mesh.get("name").startswith("can_collision_mesh_"):
            continue
        mesh = deepcopy(mesh)
        mesh.set("file", str(Path(task.SOURCE_MJCF_PATH).parent / mesh.get("file")))
        scale = np.fromstring(mesh.get("scale"), sep=" ") * task.SOURCE_SCALE
        mesh.set("scale", " ".join(map(str, scale)))
        asset.append(mesh)
        name = f"can_collision_{index}"
        can_geoms.append(name)
        ET.SubElement(can, "geom", name=name, type="mesh", mesh=mesh.get("name"))
    env = make_env(task)
    env.sim = MjSim.from_xml_string(ET.tostring(model, encoding="unicode"))
    env.objects = {"item": SimpleNamespace(contact_geoms=can_geoms)}
    env.bin_bottom_geoms = (BOTTOM_GEOM,)
    env.sim.forward()
    bottom_id = env.sim.model.geom_name2id(BOTTOM_GEOM)
    rotation = env.sim.data.geom_xmat[bottom_id].reshape(3, 3)
    center = env.sim.data.geom_xpos[bottom_id].copy()
    center[2] += (np.abs(rotation) @ env.sim.model.geom_size[bottom_id])[2]
    for angle in (0.0, np.pi / 2, np.pi / 4):
        qpos = np.array(
            [*center[:2], 1.0, np.cos(angle / 2), np.sin(angle / 2), 0.0, 0.0]
        )
        env.sim.data.set_joint_qpos("can_joint", qpos)
        env.sim.forward()
        lowest_z = np.inf
        for name in can_geoms:
            geom = env.sim.model.geom_name2id(name)
            mesh = env.sim.model.geom_dataid[geom]
            start = env.sim.model.mesh_vertadr[mesh]
            count = env.sim.model.mesh_vertnum[mesh]
            vertices = env.sim.model.mesh_vert[start : start + count]
            points = vertices @ env.sim.data.geom_xmat[geom].reshape(3, 3).T
            points += env.sim.data.geom_xpos[geom]
            lowest_z = min(lowest_z, points[:, 2].min())
        qpos[2] += center[2] - lowest_z - 0.0005
        env.sim.data.set_joint_qpos("can_joint", qpos)
        env.sim.forward()
        assert env._can_contacts_trash_can_bottom(), f"angle={angle}"
        env._reset_task_progress()
        with patch.object(IThorBedroomEnv, "_post_action", return_value=None):
            for _ in range(env.HOLD_STEPS - 1):
                env._post_action(None)
            assert not env._check_success()
            env._post_action(None)
            assert env._check_success()
            qpos[2] += 0.10
            env.sim.data.set_joint_qpos("can_joint", qpos)
            env.sim.forward()
            assert not env._can_contacts_trash_can_bottom()
            env._post_action(None)
            assert not env._check_success()
