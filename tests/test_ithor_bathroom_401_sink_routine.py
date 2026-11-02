"""Ordered sink routine, isolated specs, and one Scene 401 initialization."""

from copy import deepcopy
from dataclasses import replace

import mujoco
import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_bathroom.tasks.composite.sink_routine import (
    IThorBathroomApproachSinkAndPutSpongeInSinkThenLiftSoapBottle as Task,
)
from robocasa.environments.ithor_bathroom.tasks.lift import _BathroomNativeLiftTask
from robocasa.environments.ithor_bathroom.tasks.sink_tasks import (
    IThorBathroomPutSpongeInSink,
)
from robocasa.environments.ithor_task_catalog import (
    BATHROOM_PACKAGE,
    discover_task_catalog,
    validate_task_catalog,
)
from validate_ithor_tasks import _capture_robot_joint_state, _load_reset_controller


def test_catalog_and_private_spec_copy():
    entries = discover_task_catalog((BATHROOM_PACKAGE,))
    issues = validate_task_catalog(entries)
    # Audit this task and detect duplicate physical signatures across the catalog.
    assert not [issue for issue in issues if Task.__name__ in issue]
    assert not [issue for issue in issues if "duplicate" in issue]
    assert [e.scene_ids for e in entries if e.cls is Task] == [(401,)]
    original = IThorBathroomPutSpongeInSink.SCENE_SPECS["FloorPlan401"]
    spec = Task.SCENE_SPECS["FloorPlan401"]
    assert spec is not original
    assert all(
        spec[key] == original[key]
        for key in original
        if key not in ("lang", "source_region")
    )
    assert spec["source_region"] is not original["source_region"]
    assert spec["source_region"]["half_size"] == original["source_region"]["half_size"]
    saved = deepcopy(original)
    try:
        spec["source_region"]["center"][0] += 1.0
        assert original == saved
    finally:
        spec["source_region"]["center"][0] -= 1.0


def _unexpected_grasp_check(fixture):
    raise AssertionError("The sink routine must not check bottle grasping")


def _machine(frequency):
    task = object.__new__(Task)
    state = dict(distance=3.0, sponge=False, height=0.0)
    task.control_freq = frequency
    task._configure_success_hold_duration()
    task.PRINT_STAGE_MESSAGES = False
    task.sink_fixture = object()
    task.source_fixture = object()
    task.robot_fixture_collision_distance = lambda *args, **kw: state["distance"]
    task._raw_success = lambda: state["sponge"]
    task._source_root_z = lambda: state["height"]
    task._g1_hand_grasping_fixture = _unexpected_grasp_check
    task._reset_task_progress()
    return task, state


def _steps(task, count):
    for _ in range(count):
        task._advance_sink_routine()


def _approach(task, state):
    state["distance"] = 0.30
    _steps(task, task.APPROACH_HOLD_STEPS)
    assert task._routine_stage == 1


def _place(task, state):
    state["sponge"] = True
    _steps(task, task.HOLD_STEPS)
    assert task._routine_stage == 2


@pytest.mark.parametrize("frequency", [10, 20])
def test_order_hold_interruptions_and_fixed_reset(frequency):
    task, state = _machine(frequency)
    state.update(sponge=True, height=0.11)
    _steps(task, frequency * 2)
    assert task._routine_stage == 0
    assert not task._check_success()
    _approach(task, state)
    _steps(task, frequency * 2)
    assert task._routine_stage == 1  # Preplaced sponge cannot bypass ordering.
    state["sponge"] = False
    _steps(task, 1)
    _place(task, state)
    _steps(task, frequency * 2)
    assert not task._check_success()  # Prelifted bottle must be lowered first.
    state["sponge"] = False
    _steps(task, 1)
    state["sponge"] = True
    _steps(task, frequency)
    assert not task._check_success()  # Contact interruption cannot bypass order.
    state["height"] = 0.0
    _steps(task, 1)
    state["height"] = 0.11
    _steps(task, frequency - 1)
    assert not task._check_success()
    state["height"] = 0.09
    _steps(task, 1)
    assert task._routine_hold == 0
    state["height"] = 0.11
    _steps(task, frequency)
    assert task._check_success()
    state["sponge"] = False
    _steps(task, 1)
    assert not task._check_success()
    state.update(height=0.0, distance=3.0)
    task._reset_task_progress()
    assert task._routine_stage == 0
    assert task._routine_hold == 0
    assert not task._check_success()


def test_lift_without_grasp_requires_height_and_full_contact_hold():
    assert Task.SUPPORTED_SCENE_IDS == (401,)
    assert Task.MIN_LIFT_DELTA == 0.10
    assert _BathroomNativeLiftTask.MIN_LIFT_DELTA == 0.30
    task, state = _machine(20)
    _approach(task, state)
    state["sponge"] = True
    _steps(task, 19)
    assert task._routine_stage == 1
    state["sponge"] = False
    _steps(task, 1)
    _place(task, state)
    state["height"] = 0.09
    _steps(task, 40)
    assert not task._check_success()
    state["height"] = 0.10
    _steps(task, task.HOLD_STEPS - 1)
    assert not task._check_success()
    state["sponge"] = False
    _steps(task, 1)
    assert task._routine_hold == 0
    state["sponge"] = True
    _steps(task, task.HOLD_STEPS - 1)
    assert not task._check_success()
    _steps(task, 1)
    assert task._check_success()


def _assert_clear_of_visible_sink(env):
    """Ray-test object visual vertices against the actual sink surface."""
    model, data = env.sim.model, env.sim.data
    sink_bodies = env._fixture_body_ids(env.sink_fixture, include_siblings=True)
    subjects = {
        "sponge": env._body_tree_ids([env.obj_body_id["sponge"]]),
        "bottle": env._fixture_body_ids(env.source_fixture, include_siblings=True),
    }
    visual_meshes = [
        geom
        for geom in range(model.ngeom)
        if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_MESH
        and model.geom_contype[geom] == 0
        and model.geom_conaffinity[geom] == 0
    ]
    sink_meshes = [
        geom for geom in visual_meshes if model.geom_bodyid[geom] in sink_bodies
    ]
    direction = np.array([0.0, 0.0, -1.0])
    for name, bodies in subjects.items():
        overlaps = []
        for geom in visual_meshes:
            if model.geom_bodyid[geom] not in bodies:
                continue
            mesh = model.geom_dataid[geom]
            start = model.mesh_vertadr[mesh]
            count = model.mesh_vertnum[mesh]
            vertices = (
                model.mesh_vert[start : start + count]
                @ data.geom_xmat[geom].reshape(3, 3).T
                + data.geom_xpos[geom]
            )
            for vertex in vertices:
                origin = vertex + [0.0, 0.0, 0.25]
                for sink_mesh in sink_meshes:
                    distance = mujoco.mj_rayMesh(
                        model._model, data._data, sink_mesh, origin, direction
                    )
                    if distance >= 0.0:
                        overlaps.append(0.25 - distance)
        assert overlaps, f"{name}: no visible sink surface sampled"
        depth = max(overlaps)
        print(f"{name} maximum visible sink overlap: {depth:.6f} m")
        assert depth <= 0.0005, f"{name}: visible sink overlap {depth:.6f} m"


def test_401_headless_layout_and_physical_goals(monkeypatch):
    env = robosuite.make(
        env_name=Task.__name__,
        robots="G1",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        control_freq=20,
        renderer="mujoco",
        scene_ids=[401],
        seed=0,
    )
    try:
        env.reset()
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert env._initial_sink_distance >= 2.30
        assert not env._check_success()
        assert not env._raw_success()
        assert env._fixture_contacts_fixture(env.source_fixture, env.sink_fixture)
        assert env._fixture_is_upright(env.source_fixture, (1, 1.0))
        assert env.check_contact(
            env.objects["sponge"],
            env._geom_names_for_bodies(
                env._fixture_body_ids(env.sink_fixture, include_siblings=True)
            ),
        )
        _assert_clear_of_visible_sink(env)
        # New bottle staging needs a short stability check with the robot held.
        robot = _capture_robot_joint_state(env)
        initial_bottle = env.source_fixture.world_center(env.sim).copy()
        initial_sponge = env.object_pos("sponge").copy()
        for _ in range(int(0.3 / env.sim.model.opt.timestep)):
            env.sim.step()
            for qs, q, vs, v in robot:
                env.sim.data.qpos[qs] = q
                env.sim.data.qvel[vs] = v
            env.sim.forward()
        assert (
            np.linalg.norm(env.source_fixture.world_center(env.sim) - initial_bottle)
            < 0.01
        )
        assert np.linalg.norm(env.object_pos("sponge") - initial_sponge) < 0.01
        assert env._fixture_contacts_fixture(env.source_fixture, env.sink_fixture)
        assert env._fixture_is_upright(env.source_fixture, (1, 1.0))
        assert not env._fixture_has_illegal_penetration(
            env.source_fixture, env.sink_fixture.body_names
        )
        _assert_clear_of_visible_sink(env)
        env._set_exact_robot_pose(dict(xy=[-2.40, 1.531], yaw=np.pi), "sink operation")
        assert env._fixture_bbox_points(env.source_fixture)[:, 2].min() > 1.0
        print(
            "sink distance:",
            env._initial_sink_distance,
            "bottle centre:",
            env.source_fixture.world_center(env.sim),
            "bottle span:",
            np.ptp(env._fixture_bbox_points(env.source_fixture), axis=0),
        )
        _steps(env, env.APPROACH_HOLD_STEPS)
        assert env._routine_stage == 1
        geom_id = env.sim.model.geom_name2id(env.sink_bottom_geoms[0])
        center = env.sim.data.geom_xpos[geom_id].copy()
        rotation = env.sim.data.geom_xmat[geom_id].reshape(3, 3)
        size = np.abs(rotation) @ env.sim.model.geom_size[geom_id]
        center[2] += size[2]
        bottom = replace(
            env.source_region,
            _world_center=center,
            half_size=np.array([size[0], size[1], 0.01]),
        )
        assert env.place_object_on_support_contact(
            "sponge",
            bottom,
            fixture=env.sink_fixture,
            xy=center[:2] + [-0.08, 0.0],
            inward_fractions=(0.0,),
            probe_above=0.01,
            probe_below=0.02,
            probe_step=0.001,
        )
        assert env._raw_success()
        _steps(env, env.HOLD_STEPS)
        assert env._routine_stage == 2
        # Exercise actual lift geometry without any grasp query.
        monkeypatch.setattr(env, "_g1_hand_grasping_fixture", _unexpected_grasp_check)
        _steps(env, 1)
        joint = env._fixture_free_joint_name(env.source_fixture)
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        qpos[2] += 0.11
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        _steps(env, env.HOLD_STEPS - 1)
        assert not env._check_success()
        _steps(env, 1)
        assert env._check_success()
        env._set_task_object_bottom_at("sponge", center[:2], center[2] + 0.10)
        _steps(env, 1)
        assert not env._check_success()
        meta = env.get_ep_meta()
        assert meta["stage_order"] == list(Task.COMPOSITE_ORDERING)
        assert meta["final_grasp_required"] is False
    finally:
        env.close()
