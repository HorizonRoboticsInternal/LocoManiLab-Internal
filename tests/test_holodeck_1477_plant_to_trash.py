"""Scene-local layout, approach/lift/place order, and plant containment."""

import json
from pathlib import Path
from types import SimpleNamespace
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
from robocasa.environments.holodeck.tasks.multi_hop import (
    HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan as Workflow,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robosuite.environments.base import REGISTERED_ENVS
from validate_ithor_tasks import _load_reset_controller


ROOT = Path(__file__).resolve().parents[1]


def test_registration_and_scene_contract():
    for catalog in (holodeck, tasks):
        assert getattr(catalog, Workflow.__name__) is Workflow
        assert catalog.__all__.count(Workflow.__name__) == 1
    assert REGISTERED_HOLODECK_ENVS[Workflow.__name__] is Workflow
    assert REGISTERED_ENVS[Workflow.__name__] is Workflow
    old_name = "HolodeckPickPlantThenApproachTrashCanThenThrowPlant"
    for registry in (REGISTERED_HOLODECK_ENVS, REGISTERED_ENVS):
        assert old_name not in registry
    for catalog in (holodeck, tasks):
        assert not hasattr(catalog, old_name)
    assert [stage.name for stage in Workflow.ATOMIC_TASKS] == [
        "approach_trash_can",
        "lift_plant",
        "place_plant_in_trash_can",
    ]
    assert Workflow.SUPPORTED_SCENE_IDS == (1477,)
    assert Workflow.USE_DEFAULT_ASSET_LAYOUT is False
    assert Workflow.SPEC_TASK == "lift_plant_from_low_cabinet"
    path = "robocasa/models/assets/scenes/holodeck_task_specs.yaml"
    specs = yaml.safe_load((ROOT / path).read_text())["scenes"]
    assert [
        scene for scene, tasks in specs.items() if Workflow.WORKFLOW_SPEC_TASK in tasks
    ] == ["train_1477"]
    source = specs["train_1477"][Workflow.SPEC_TASK]
    assert source["target_body"] == "houseplant_70451f91981395edf773114de3a28b67_1_0_0"
    assert source["target_pose"]["pos"] == [0.75, 5.4, 0.66533]
    assert source["robot_pose"]["pos"] == [0.75, 5.05, 0.0]
    workflow = specs["train_1477"][Workflow.WORKFLOW_SPEC_TASK]
    assert workflow["bin_pos"] == [1.1, 4.9, 0.0]
    assert workflow["robot_pose"]["pos"] != source["robot_pose"]["pos"]
    task = object.__new__(Workflow)
    task.workflow_spec = workflow
    assert task._get_robot_base_pose_config() == workflow["robot_pose"]


def test_bin_reuses_scene_117_meshes_and_materials():
    task = object.__new__(Workflow)
    task.workflow_spec = get_holodeck_task_spec("train_1477", task.WORKFLOW_SPEC_TASK)
    copied = {}
    task._append_xml_fixture_model = lambda model, *args, **kwargs: copied.update(
        model=model
    )
    task._add_scene_117_trash_can()
    source = ET.parse(
        ROOT
        / "robocasa/models/assets/scenes/holodeck-objaverse-train/train_117.robocasa.xml"
    ).getroot()
    body = copied["model"].find("./worldbody/body")
    original = source.find(f"./worldbody/body[@name='{task.TRASH_CAN_SOURCE_BODY}']")
    assert body.get("quat") == original.get("quat")
    assert body.get("name") == "trash_can"
    for asset in copied["model"].findall("./asset/*"):
        original_asset = source.find(
            f"./asset/{asset.tag}[@name='{asset.get('name')}']"
        )
        assert ET.tostring(asset) == ET.tostring(original_asset)
    visual = body.findall("./geom[@group='1']")
    assert len(visual) == 3
    assert all(geom.get("type") == "mesh" for geom in visual)
    for geom, original_geom in zip(body.findall("geom"), original.findall("geom")):
        attrs = dict(original_geom.attrib)
        attrs["name"] = attrs["name"].replace(task.TRASH_CAN_SOURCE_BODY, "trash_can")
        assert geom.attrib == attrs


def _reject_hand_check(*args, **kwargs):
    pytest.fail("Task predicates and debug logging must not read hand state/contact")


def _fake_task(monkeypatch, frequency):
    monkeypatch.setattr(
        HolodeckEnv, "_post_action", lambda self, action: (0, False, {})
    )
    task = object.__new__(Workflow)
    task.control_freq = frequency
    task.SUCCESS_DEBUG_ENABLED = False
    task._atomic_tasks = task.ATOMIC_TASKS
    task._reset_task_progress()
    task._plant_initial_z = 0.6
    task.workflow_spec = get_holodeck_task_spec("train_1477", task.WORKFLOW_SPEC_TASK)
    task.sim = SimpleNamespace()
    state = dict(z=0.6, xy=np.zeros(2), distance=2.0, placed=False)
    task._plant_grasped = task.g1_hand_closed = task.check_contact = _reject_hand_check
    task._plant_root_z = lambda: state["z"]
    task._robot_base_xy = lambda: state["xy"]
    task.success_distance = lambda: state["distance"]
    task._plant_in_bin = lambda: task._release_stage_ready and state["placed"]
    return task, state


def _advance(task, steps):
    for _ in range(steps):
        task._post_action(None)


@pytest.mark.parametrize("frequency", (20, 50))
def test_order_approach_lift_and_interrupted_placement(monkeypatch, frequency):
    task, state = _fake_task(monkeypatch, frequency)
    state.update(z=0.91, placed=True)
    _advance(task, frequency)
    assert task.stage_index == 0  # Early lifting/placement cannot skip navigation.
    state.update(distance=0.2)
    _advance(task, 4)
    assert task.stage_index == 0
    state["distance"] = 0.5
    _advance(task, 1)
    assert task._stage_hold_steps == 0
    state["distance"] = 0.2
    _advance(task, 5)
    assert task.stage_index == 1
    _advance(task, frequency)
    assert task.stage_index == 1  # A plant lifted before arrival must be lowered.
    state.update(z=0.89, placed=False)
    _advance(task, 1)
    state["z"] = 0.91
    _advance(task, frequency - 1)
    assert task.stage_index == 1
    state["z"] = 0.89
    _advance(task, 1)
    assert task._stage_hold_steps == 0
    state["z"] = 0.91
    _advance(task, frequency)
    assert task.stage_index == 2
    _advance(task, frequency)
    assert not task._check_success()
    state["placed"] = True
    _advance(task, frequency - 1)
    assert not task._check_success()
    state["placed"] = False
    task._post_action(None)
    assert task._stage_hold_steps == 0
    state["placed"] = True
    _advance(task, frequency)
    assert task._check_success()
    state["placed"] = False
    assert not task._check_success()
    task._reset_task_progress()
    assert task.stage_index == 0 and not task._release_stage_ready
    assert not task._lift_stage_armed


def test_containment_rejects_wall_rim_under_floor_without_hand_checks():
    task = object.__new__(Workflow)
    task.workflow_spec = get_holodeck_task_spec("train_1477", task.WORKFLOW_SPEC_TASK)
    task._release_stage_ready = True
    task._plant_bodies = {1}
    task.g1_hand_closed = task.check_contact = _reject_hand_check
    state = {"contact": True}
    task._plant_contacts_floor = lambda: state["contact"]
    center = np.array(task.workflow_spec["bin_pos"])
    rotation = np.array([[1.0, 0, 0], [0, 0, -1.0], [0, 1.0, 0]])
    task.trash_can_body = "bin"
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            get_body_xpos=lambda _: center,
            get_body_xmat=lambda _: rotation,
        )
    )
    points = np.array([[-0.04, 0.08, -0.04], [0.04, 0.15, 0.04]])
    task._plant_collision_points = lambda: points @ rotation.T + center
    assert task._plant_in_bin()
    for shift in ([0.25, 0, 0], [0, 0.60, 0], [0, -0.04, 0]):
        task._plant_collision_points = lambda: (points + shift) @ rotation.T + center
        assert not task._plant_in_bin()
    task._plant_collision_points = lambda: points @ rotation.T + center
    state["contact"] = False
    assert not task._plant_in_bin()
    state["contact"] = True
    task._release_stage_ready = False
    assert not task._plant_in_bin()


def test_one_headless_reset_support_containment_and_bin_clearance():
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
        assert env.stage_index == 0 and not env._check_success()
        assert env.count_body_collisions() == 0
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            xy = env._robot_base_xy() + 0.25 * np.array([np.cos(angle), np.sin(angle)])
            assert env.room_of_point([*xy, 0]) is not None
        np.testing.assert_allclose(
            env.sim.data.get_body_xpos(env.trash_can_body), [1.1, 4.9, 0]
        )
        initial = env.sim.data.body_xpos[env._target_body_id()].copy()
        source_bodies = env._body_tree_ids(
            [env.sim.model.body_name2id(env.task_spec["source_body"])]
        )

        def source_contact():
            for i in range(env.sim.data.ncon):
                c = env.sim.data.contact[i]
                bodies = {
                    int(env.sim.model.geom_bodyid[c.geom1]),
                    int(env.sim.model.geom_bodyid[c.geom2]),
                }
                if bodies & source_bodies and bodies & env._plant_bodies:
                    return True
            return False

        assert source_contact(), "reset plant must already contact its side table"
        _settle_plant_only(env, 100)
        np.testing.assert_allclose(
            env.sim.data.body_xpos[env._target_body_id()], initial, atol=0.01
        )
        assert source_contact()
        print("plant_start", initial, "spawn_to_bin", env.success_distance())
        assert env.success_distance() > env.workflow_spec["success_radius"] + 0.15
        assert np.linalg.norm(
            initial[:2] - env.workflow_spec["bin_pos"][:2]
        ) == pytest.approx(np.hypot(0.35, 0.50))
        _assert_native_bin_geometry(env)
        # Include all scene objects and their visible meshes. A large furniture
        # asset beside the bin has no category in the scene metadata.
        import mujoco

        metadata = json.loads(
            (
                ROOT
                / "robocasa/models/assets/scenes/holodeck-objaverse-train"
                / "train_1477_metadata.json"
            ).read_text()
        )
        protected = [
            name
            for name in metadata["objects"]
            if name in env.sim.model.body_names and name != env.task_spec["target_body"]
        ]
        assert "obja_fa7e87085e0e4b24afa961dfa1553d20_1_0_0" in protected
        bin_id = env.sim.model.body_name2id(env.trash_can_body)
        bin_bodies = env._body_tree_ids([bin_id])
        bin_geoms = env._collidable_geoms(bin_bodies)
        bin_min, bin_max = _visual_mesh_bounds(env, bin_bodies)
        for table in protected:
            table_bodies = env._body_tree_ids([env.sim.model.body_name2id(table)])
            table_geoms = env._collidable_geoms(table_bodies)
            clearance = min(
                mujoco.mj_geomDistance(
                    env.sim.model._model, env.sim.data._data, a, b, np.inf, None
                )
                for a in table_geoms
                for b in bin_geoms
            )
            table_min, table_max = _visual_mesh_bounds(env, table_bodies)
            gap = np.maximum(np.maximum(table_min - bin_max, bin_min - table_max), 0)
            visual_clearance = np.linalg.norm(gap)
            if min(clearance, visual_clearance) < 0.5:
                print(
                    "bin_object_clearance",
                    table,
                    clearance,
                    "visual_aabb",
                    visual_clearance,
                )
            assert clearance > 0.05, table
            assert visual_clearance > 0.05, table
        # In the same instance, place the real native pot in the real hollow bin.
        qpos = env.sim.data.get_joint_qpos(env._target_joint_name()).copy()
        qpos[:2] = env.workflow_spec["bin_pos"][:2]
        env.sim.data.set_joint_qpos(env._target_joint_name(), qpos)
        env.sim.forward()
        qpos[2] += 0.08 - env._plant_collision_points()[:, 2].min()
        env.sim.data.set_joint_qpos(env._target_joint_name(), qpos)
        env.sim.data.set_joint_qvel(env._target_joint_name(), np.zeros(6))
        env.sim.forward()
        env._release_stage_ready = True
        _settle_plant_only(env, 30)
        assert env._plant_contacts_floor()
        assert env._plant_in_bin()
        qpos = env.sim.data.get_joint_qpos(env._target_joint_name()).copy()
        qpos[0] += 0.35
        env.sim.data.set_joint_qpos(env._target_joint_name(), qpos)
        env.sim.forward()
        assert not env._plant_in_bin()
    finally:
        env.close()


def _assert_native_bin_geometry(env):
    import mujoco

    model = env.sim.model._model
    bin_id = model.body(env.trash_can_body).id
    bin_geoms = np.flatnonzero(model.geom_bodyid == bin_id)
    visible = bin_geoms[model.geom_group[bin_geoms] == 1]
    collision = bin_geoms[model.geom_group[bin_geoms] == 0]
    assert len(visible) == 3 and len(collision) == 5
    assert np.all(model.geom_type[visible] == int(mujoco.mjtGeom.mjGEOM_MESH))
    assert np.all(model.geom_contype[visible] == 0)
    assert np.all(model.geom_conaffinity[visible] == 0)
    assert np.all(model.geom_contype[collision] != 0)


def test_debug_reports_blockers_and_throttles_without_advancing(monkeypatch, capsys):
    task, state = _fake_task(monkeypatch, 50)
    task.SUCCESS_DEBUG_ENABLED = True
    task.trash_can_body = "bin"
    task._plant_bodies = {1}
    task._plant_contacts_floor = lambda: False
    task._plant_collision_points = lambda: np.array(
        [[0.2, 0.08, 0], [0.25, 0.15, 0.04]]
    )
    task.sim.data = SimpleNamespace(
        time=0.0,
        get_body_xpos=lambda _: np.zeros(3),
        get_body_xmat=lambda _: np.eye(3),
    )
    task._log_success_conditions()
    output = capsys.readouterr().out
    assert "approach_trash_can" in output and "离桶过远" in output
    assert "双手=" not in output and "闭手=" not in output
    assert "保持=0/5" in output
    task.sim.data.time = 0.5
    task._log_success_conditions()
    assert capsys.readouterr().out == ""
    task._stage_index = 1
    task._log_success_conditions()
    output = capsys.readouterr().out
    assert "lift_plant" in output and "抬升不足" in output
    assert "需先放低提前举起的盆栽" in output and "保持=0/50" in output
    task._stage_index = 2
    task._release_stage_ready = True
    task._log_success_conditions()
    output = capsys.readouterr().out
    assert "盆体超出桶内容纳范围" in output and "盆栽未接触桶底" in output
    assert "上界越界=" in output and "投放阶段就绪=1" in output
    task.sim.data.time = 1.5
    task._log_success_conditions()
    assert "place_plant_in_trash_can" in capsys.readouterr().out
    assert task.stage_index == 2 and task._stage_hold_steps == 0
    state["placed"] = True
    task._stage_index = 3
    task._log_success_conditions()
    assert "任务成功=1" in capsys.readouterr().out
    task._reset_task_progress()
    assert task._plant_debug_last_state is None
    assert task._success_debug_next_log_time == 0.0


def _visual_mesh_bounds(env, bodies):
    """World bounds of actual visible vertices, independent of collision proxies."""
    import mujoco

    model, data = env.sim.model._model, env.sim.data._data
    points = []
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] not in bodies or model.geom_group[geom] != 1:
            continue
        assert model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        mesh = model.geom_dataid[geom]
        start = model.mesh_vertadr[mesh]
        vertices = model.mesh_vert[start : start + model.mesh_vertnum[mesh]]
        points.append(
            vertices @ data.geom_xmat[geom].reshape(3, 3).T + data.geom_xpos[geom]
        )
    assert points
    points = np.concatenate(points)
    return points.min(axis=0), points.max(axis=0)


def _settle_plant_only(env, steps):
    """Settle the pot while keeping the uncontrolled robot and furniture still."""
    model, data = env.sim.model, env.sim.data
    joint = model.joint_name2id(env._target_joint_name())
    qadr, dadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
    qmask, vmask = np.ones(model.nq, bool), np.ones(model.nv, bool)
    qmask[qadr : qadr + 7] = False
    vmask[dadr : dadr + 6] = False
    fixed = data.qpos.copy()
    for _ in range(steps):
        env.sim.step()
        data.qpos[qmask] = fixed[qmask]
        data.qvel[vmask] = 0
        env.sim.forward()
