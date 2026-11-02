"""Scene 301 state-only predicates and uninterrupted one-second holds."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET
from unittest.mock import patch

import numpy as np
import pytest

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_bedroom.tasks.bedside_objects import (
    IThorBedroomLiftStatueFromSofa,
    IThorBedroomStandPillowAtHeadboard,
)
from robocasa.environments.ithor_task_catalog import (
    BEDROOM_PACKAGE,
    discover_task_catalog,
    task_counts,
    validate_task_catalog,
)


TASKS = (IThorBedroomStandPillowAtHeadboard, IThorBedroomLiftStatueFromSofa)


def test_registration_and_exact_scene_scope():
    entries = discover_task_catalog([BEDROOM_PACKAGE])
    assert task_counts(entries)[BEDROOM_PACKAGE][301] == {"atomic": 2, "composite": 2}
    # The pre-existing composite matrix is stale; registration/duplicates and
    # the new scene's exact matrix are checked independently of that old issue.
    issues = validate_task_catalog(entries)
    assert not [issue for issue in issues if ": task matrix changed" not in issue]
    for cls in TASKS:
        assert sum(entry.cls is cls for entry in entries) == 1
        assert cls.SUPPORTED_SCENE_IDS == (301,)
        with pytest.raises(ValueError, match="uncalibrated scene ids"):
            cls(scene_ids=[329])


def make_pillow():
    env = object.__new__(IThorBedroomStandPillowAtHeadboard)
    env.target_fixture = SimpleNamespace(body_name="pillow")
    env._pillow_bodies = {1}
    env._headboard_geom = 1
    env._mattress_geom = 2
    # Pillow face normal and headboard thickness direction both point along Y.
    pillow_rotation = np.eye(3)
    headboard_rotation = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    contacts = [
        SimpleNamespace(geom1=0, geom2=1, dist=0.0),
        SimpleNamespace(geom1=2, geom2=0, dist=0.0),
    ]
    env.sim = SimpleNamespace(
        model=SimpleNamespace(
            body_name2id=lambda name: 0,
            geom_bodyid=np.array([1, 2, 2]),
        ),
        data=SimpleNamespace(
            body_xmat=np.array([pillow_rotation.ravel()]),
            geom_xmat=np.array(
                [np.eye(3).ravel(), headboard_rotation.ravel(), np.eye(3).ravel()]
            ),
            contact=contacts,
            ncon=len(contacts),
        ),
    )
    return env


def test_pillow_requires_upright_mattress_and_headboard_contact():
    env = make_pillow()
    assert env._target_state_reached()
    env.sim.data.ncon = 1  # Against headboard but lifted above mattress.
    assert not env._target_state_reached()
    env.sim.data.contact = env.sim.data.contact[1:]
    assert not env._target_state_reached()  # Upright in the middle of the bed.
    env = make_pillow()
    env.sim.data.body_xmat[0] = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]).ravel()
    assert not env._target_state_reached()  # Flat, even if touching both.
    env = make_pillow()
    env.sim.data.contact[0].dist = -0.01
    assert not env._target_state_reached()  # Penetrating the headboard.


def test_statue_height_is_relative_to_reset_without_grasp():
    env = object.__new__(IThorBedroomLiftStatueFromSofa)
    center = np.array([1.0, 0.44, 0.58])
    env.target_fixture = SimpleNamespace(world_center=lambda sim: center)
    env.sim = None
    env._initial_target_z = float(center[2])
    assert not env._target_state_reached()
    center[2] = env._initial_target_z + 0.299
    assert not env._target_state_reached()
    center[2] = env._initial_target_z + 0.30
    assert env._target_state_reached()


@pytest.mark.parametrize("cls", TASKS)
@pytest.mark.parametrize("control_freq", [10, 20, 30])
def test_uninterrupted_second_counts_actions_not_success_queries(cls, control_freq):
    env = object.__new__(cls)
    env.control_freq = control_freq
    env._configure_success_hold_duration()
    env._state_hold_steps = 0
    reached = True
    env._target_state_reached = lambda: reached
    # There are no robot/grasp attributes: success depends only on the object.
    with patch.object(IThorBedroomEnv, "_post_action", return_value=None):
        for _ in range(control_freq - 1):
            env._post_action(None)
        assert not env._check_success()
        assert not env._check_success()
        reached = False
        env._post_action(None)
        assert env._state_hold_steps == 0
        reached = True
        for _ in range(control_freq):
            env._post_action(None)
        assert env._check_success()
        reached = False
        env._post_action(None)
        assert not env._check_success()


def test_default_sofa_replacement_and_semantics(tmp_path):
    from robocasa.models.arenas.molmospaces_arena import preprocess_scene_xml
    from robocasa.models.scenes.ithor_scene_defaults import (
        BEDROOM_301_OLD_SEAT_BODY,
        BEDROOM_301_SOFA_BODY,
        BEDROOM_301_SOFA_ID,
        apply_default_scene_metadata,
    )
    from robocasa.models.scenes.molmospaces_scene_registry import get_scene_xml

    source = Path(get_scene_xml(301))
    before = ET.parse(source).getroot()
    cleaned = ET.parse(preprocess_scene_xml(source, tmp_path / "scene.xml")).getroot()
    assert (
        before.find(f"worldbody/body[@name='{BEDROOM_301_OLD_SEAT_BODY}']") is not None
    )
    assert cleaned.find(f"worldbody/body[@name='{BEDROOM_301_OLD_SEAT_BODY}']") is None
    sofa = cleaned.find(f"worldbody/body[@name='{BEDROOM_301_SOFA_BODY}']")
    assert sofa is not None and sofa.find(".//joint") is None
    assert len(list(sofa.iter("geom"))) == 11  # Four visuals and seven colliders.
    before_names = {b.get("name") for b in before.find("worldbody").findall("body")}
    after_names = {b.get("name") for b in cleaned.find("worldbody").findall("body")}
    assert after_names == before_names - {BEDROOM_301_OLD_SEAT_BODY} | {
        BEDROOM_301_SOFA_BODY
    }
    for asset in cleaned.find("asset"):
        if asset.get("name", "").startswith(BEDROOM_301_SOFA_BODY) and asset.get(
            "file"
        ):
            assert Path(asset.get("file")).is_file()

    with source.with_name("FloorPlan301_physics_metadata.json").open() as handle:
        metadata = json.load(handle)
    original = copy.deepcopy(metadata)
    apply_default_scene_metadata(metadata, "FloorPlan301")
    assert BEDROOM_301_OLD_SEAT_BODY not in metadata["objects"]
    fixture = metadata["objects"][BEDROOM_301_SOFA_BODY]
    assert fixture["category"] == "Sofa" and fixture["is_static"]
    assert f"FloorPlan301:{fixture['object_id']}" == BEDROOM_301_SOFA_ID
    assert IThorBedroomLiftStatueFromSofa.SOFA_STABLE_ID == BEDROOM_301_SOFA_ID
    for name, value in original["objects"].items():
        if name != BEDROOM_301_OLD_SEAT_BODY:
            assert metadata["objects"][name] == value
    for name in fixture["name_map"]["bodies"]:
        assert cleaned.find(f".//body[@name='{name}']") is not None
    for name in fixture["name_map"]["sites"]:
        assert cleaned.find(f".//site[@name='{name}']") is not None

    # Fixed furniture does not generate fixed/fixed contacts. Query distances
    # directly so a replacement embedded in a wall cannot pass this check.
    import mujoco

    for asset in cleaned.find("asset"):
        if asset.get("file"):
            asset.set("file", str((source.parent / asset.get("file")).resolve()))
    model = mujoco.MjModel.from_xml_string(ET.tostring(cleaned, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    sofa_bodies = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        for name in fixture["name_map"]["bodies"]
    }
    colliders = [g for g in range(model.ngeom) if model.geom_contype[g]]
    sofa_geoms = [g for g in colliders if int(model.geom_bodyid[g]) in sofa_bodies]
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    assert min(
        mujoco.mj_geomDistance(model, data, g, floor, 0.01, None) for g in sofa_geoms
    ) == pytest.approx(0.0, abs=0.001)
    for sofa_geom in sofa_geoms:
        for other_geom in colliders:
            if int(model.geom_bodyid[other_geom]) in sofa_bodies:
                continue
            assert (
                mujoco.mj_geomDistance(model, data, sofa_geom, other_geom, 0.01, None)
                >= -0.003
            )


def test_default_sofa_replacement_leaves_other_scenes_unchanged():
    from robocasa.models.scenes.ithor_scene_defaults import (
        apply_default_scene_furniture,
        apply_default_scene_metadata,
    )

    root = ET.fromstring('<mujoco><worldbody><body name="seat"/></worldbody></mujoco>')
    before = ET.tostring(root)
    metadata = {"objects": {"seat": {"category": "DogBed"}}}
    original = copy.deepcopy(metadata)
    apply_default_scene_furniture(root, "/scenes/FloorPlan302_physics.xml")
    apply_default_scene_metadata(metadata, "FloorPlan302")
    assert ET.tostring(root) == before
    assert metadata == original


def test_statue_replacement_is_shared_and_task_local():
    from robocasa.environments.ithor_bedroom.tasks.bedside_objects import (
        _BedsideObjectStateMixin,
        _SofaStatueMixin,
    )
    from robocasa.environments.ithor_bedroom.tasks.bedside_composites import (
        IThorBedroomOpenBoxThenStatueFromSofaIntoBox,
    )

    from robocasa.environments.ithor_living_room.tasks.atomic.lift_statue import (
        IThorLivingRoomLiftStatue,
    )
    from robocasa.environments.ithor_bedroom.tasks.assets import asset_xml_path

    source = Path(asset_xml_path(_SofaStatueMixin.STATUE_MODEL))
    source_xml = source.read_bytes()
    assert IThorLivingRoomLiftStatue.TARGET_XML.endswith(_SofaStatueMixin.STATUE_MODEL)
    assert _SofaStatueMixin.STATUE_SCALE == IThorLivingRoomLiftStatue.TARGET_SCALE
    scene = Path(__file__).resolve().parents[1] / (
        "robocasa/models/assets/scenes/ithor/FloorPlan301_physics.xml"
    )
    original = ET.parse(scene).getroot()
    original_xml = ET.tostring(original)
    basketball_body = "basketball_13f63d02d4de4b9acc00b38afd72266b_1_0_0"
    for cls in (
        IThorBedroomLiftStatueFromSofa,
        IThorBedroomOpenBoxThenStatueFromSofaIntoBox,
    ):
        env = object.__new__(cls)
        root = copy.deepcopy(original)
        env.mujoco_arena = SimpleNamespace(root=root, worldbody=root.find("worldbody"))
        basketball = SimpleNamespace(body_name=basketball_body)
        env.get_fixture = lambda *, id: (
            basketball if id == "FloorPlan301:Basketball_1_7" else None
        )
        expected = copy.deepcopy(original)
        expected_worldbody = expected.find("worldbody")
        expected_worldbody.remove(
            expected_worldbody.find(f"body[@name='{basketball_body}']")
        )

        def create_fixture(**kwargs):
            assert kwargs["name"] == "task_sofa_statue"
            assert kwargs["asset_id"] == "Stone_Statue_4"
            assert kwargs["ensure_inertials"] is False
            statue_root = ET.parse(kwargs["xml_path"]).getroot()
            statue_body = statue_root.find("./worldbody/body")
            assert statue_body.get("pos") == "0 0 0"
            assert statue_body.get("quat") == "1 0 0 0"
            colliders = [g for g in statue_root.iter("geom") if g.get("contype") == "1"]
            assert len(colliders) == 7
            assert all(g.get("type") == "box" for g in colliders)
            assert not any(
                g.get("name", "").endswith("reg_bbox") for g in statue_root.iter("geom")
            )
            root.find("worldbody").append(statue_root.find("./worldbody/body"))
            return SimpleNamespace()

        env.create_scene_fixture = create_fixture
        with patch.object(_BedsideObjectStateMixin, "_setup_task_fixtures"):
            _SofaStatueMixin._setup_task_fixtures(env)
        # Only the ball is removed; the new statue is a separate task-local body.
        spawned = root.find("worldbody")[-1]
        assert spawned.get("name") == "statue_main"
        root.find("worldbody").remove(spawned)
        assert ET.tostring(root) == ET.tostring(expected)
        assert cls.TARGET_STABLE_ID == "FloorPlan301:spawned:task_sofa_statue"
        assert not hasattr(cls, "BASKETBALL_MASS")
    assert ET.tostring(original) == original_xml
    assert source.read_bytes() == source_xml
    assert not issubclass(IThorBedroomStandPillowAtHeadboard, _SofaStatueMixin)
