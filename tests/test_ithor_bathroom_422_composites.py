"""Scene-422 registration, ordered goals, and one reset per new layout."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.ithor_bathroom import tasks
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.environments.ithor_bathroom.tasks.composite.scene422_routines import (
    IThorBathroomOpenDrawerThenLiftCandleFromStoolAndPutInDrawer as Candle,
    IThorBathroomOpenThenCloseUpperCabinetDoorAndApproachSink as Cabinet,
)
from robocasa.environments.ithor_bathroom.tasks.floor_lift import (
    IThorBathroomLiftCandleFromFloor as Lift,
)
from robocasa.environments.ithor_bathroom.tasks.storage_articulation import (
    IThorBathroomOpenUpperCabinetDoor as Open,
)
from robocasa.environments.ithor_task_catalog import (
    BATHROOM_PACKAGE,
    discover_task_catalog,
    validate_task_catalog,
)


def test_registration_scene_limits_and_independent_specs():
    from robosuite.environments.base import REGISTERED_ENVS

    for cls, atom, scene in (
        (Cabinet, Open, "FloorPlan422"),
        (Candle, Lift, "FloorPlan401"),
    ):
        assert cls.SUPPORTED_SCENE_IDS == (422,)
        assert set(cls.SCENE_SPECS) == {"FloorPlan422"}
        assert cls in tasks.COMPOSITE_TASK_CLASSES
        assert cls not in tasks.ATOMIC_TASK_CLASSES
        assert REGISTERED_ENVS[cls.__name__] is cls
        assert getattr(tasks, cls.__name__) is cls
        original = deepcopy(atom.SCENE_SPECS)
        xy = cls.SCENE_SPECS["FloorPlan422"]["robot_pose"]["xy"]
        saved = xy[0]
        try:
            xy[0] = 123
            assert atom.SCENE_SPECS == original
        finally:
            xy[0] = saved
        with pytest.raises(ValueError, match="uncalibrated scene ids"):
            cls(scene_ids=[418])
        assert scene in atom.SCENE_SPECS
    assert Candle.TARGET_XML == Lift.TARGET_XML
    assert Candle.TARGET_SCALE == Lift.TARGET_SCALE
    assert Candle.STOOL_SCALE == Lift.STOOL_SCALE
    assert validate_task_catalog(discover_task_catalog((BATHROOM_PACKAGE,))) == []


def test_candle_facing_target_is_local_to_drawer_routine():
    source = np.array([-1.12, 0.76])
    atomic = object.__new__(Lift)
    assert atomic._robot_facing_target_xy(source) is source
    composite = object.__new__(Candle)
    composite.drawer_fixture = SimpleNamespace(center=np.array([-0.81, 0.38, 0.96]))
    composite.scene_task_spec = lambda: Candle.SCENE_SPECS["FloorPlan422"]
    np.testing.assert_array_equal(
        composite._robot_facing_target_xy(source), [-0.81, 0.81885]
    )


def _stub(cls, frequency=20):
    task = object.__new__(cls)
    task.PRINT_STAGE_MESSAGES = False
    task.control_freq = frequency
    task._configure_success_hold_duration()
    task.sim = None
    state = dict(
        fraction=0.2,
        distance=1.0,
        xy=np.zeros(2),
        height=0.5,
        grasp=False,
        stool=True,
        placed=False,
    )
    joint = SimpleNamespace(fraction=lambda sim: state["fraction"])
    task.target_semantic_joint = task.drawer_joint = joint
    task.SUCCESS_OPEN_FRACTION = 0.8
    task.SUCCESS_CLOSE_FRACTION = 0.2
    task.sink_fixture = object()
    task._robot_base_xy = lambda: state["xy"]
    task.robot_fixture_collision_distance = lambda *a, **kw: state["distance"]
    task._source_root_z = lambda: state["height"]
    task.g1_hand_grasping_object = lambda *a: state["grasp"]
    task._candle_is_on_stool = lambda: state["stool"]
    task._candle_inside_drawer = lambda: state["placed"]
    task._candle_contacts_drawer_bottom = lambda: state["placed"]
    task._candle_released = lambda: not state["grasp"]
    task.illegal_contact_pairs_for_bodies = lambda *a, **kw: []
    task._fixture_body_ids = lambda *a: {1}
    task.drawer_fixture = object()
    task.obj_body_id = {"candle": 2}
    task._reset_task_progress()
    return task, state


def _tick(task, count):
    for _ in range(count):
        task._post_action(None)


@pytest.mark.parametrize("frequency", (10, 20))
def test_cabinet_order_requires_movement_after_closing(monkeypatch, frequency):
    monkeypatch.setattr(IThorBathroomEnv, "_post_action", lambda *a: None)
    task, state = _stub(Cabinet, frequency)
    state.update(distance=0.1, xy=np.array([1.0, 0.0]))
    _tick(task, frequency)
    assert task._routine_stage == 0
    state["fraction"] = 0.8
    _tick(task, 5)
    assert task._routine_stage == 1
    state["fraction"] = 0.2
    _tick(task, 5)
    assert task._routine_stage == 2
    _tick(task, frequency)
    assert not task._check_success()  # Early navigation earns no displacement.
    state["xy"] = np.array([1.31, 0.0])
    _tick(task, frequency - 1)
    assert not task._check_success()
    _tick(task, 1)
    assert task._check_success()
    state["fraction"] = 0.21
    assert not task._check_success()
    _tick(task, 1)
    assert task._routine_hold == 0
    task._reset_task_progress()
    assert task._routine_stage == task._routine_hold == 0


@pytest.mark.parametrize("frequency", (10, 20))
def test_candle_order_lift_and_place_without_grasp(monkeypatch, frequency):
    monkeypatch.setattr(IThorBathroomEnv, "_post_action", lambda *a: None)
    task, state = _stub(Candle, frequency)
    task.g1_hand_grasping_object = lambda *a: pytest.fail(
        "The candle routine must not query current or historical grasping"
    )
    state.update(placed=True, stool=False)
    _tick(task, frequency)
    assert task._routine_stage == 0
    state.update(fraction=0.8, placed=False, stool=True)
    _tick(task, 5)
    assert task._routine_stage == 1
    state.update(height=0.79, grasp=False, stool=False)
    _tick(task, 5)
    assert task._routine_stage == 1  # Less than the 0.30 m lift threshold.
    state.update(height=0.80, fraction=0.79)
    _tick(task, 5)
    assert task._routine_stage == 1  # The drawer must stay open.
    state["fraction"] = 0.8
    _tick(task, 4)
    assert task._routine_stage == 1
    state["height"] = 0.79
    _tick(task, 1)
    assert task._routine_hold == 0
    state["height"] = 0.80
    _tick(task, 5)
    assert task._routine_stage == 2  # Height alone suffices, with no grasp history.
    task._candle_released = lambda: pytest.fail(
        "The final stage must not query release status"
    )
    task.g1_hand_object_contact = lambda *a: pytest.fail(
        "The final stage must not query hand contact"
    )
    state.update(placed=True, grasp=True)
    _tick(task, frequency - 1)
    assert not task._check_success()
    _tick(task, 1)
    assert task._check_success()  # Holding the candle does not block success.
    state["grasp"] = False
    assert task._check_success()
    state["placed"] = False
    assert not task._check_success()
    _tick(task, 1)
    assert task._routine_hold == 0
    task._reset_task_progress()
    assert task._initial_source_z == state["height"]
    assert task._routine_stage == task._routine_hold == 0


@pytest.mark.parametrize("frequency", (10, 20))
def test_candle_already_placed_when_final_stage_starts(monkeypatch, frequency):
    monkeypatch.setattr(IThorBathroomEnv, "_post_action", lambda *a: None)
    task, state = _stub(Candle, frequency)
    state["fraction"] = 0.8
    _tick(task, 5)
    assert task._routine_stage == 1
    state.update(height=0.80, stool=False)
    _tick(task, 4)
    assert task._routine_stage == 1
    # Placement can finish on the same control step as the lift hold.
    state["placed"] = True
    _tick(task, 1)
    assert task._routine_stage == 2
    assert task._raw_success()
    assert task._routine_hold == 0
    _tick(task, frequency - 1)
    assert not task._check_success()
    _tick(task, 1)
    assert task._check_success()


@pytest.mark.parametrize("frequency", (10, 20))
def test_candle_success_diagnostic_every_five_seconds(monkeypatch, capsys, frequency):
    from robocasa.environments.ithor_bathroom.tasks.composite import scene422_routines

    clock = [0.0]
    monkeypatch.setattr(scene422_routines, "monotonic", lambda: clock[0])
    monkeypatch.setattr(IThorBathroomEnv, "_post_action", lambda *a: None)
    task, state = _stub(Candle, frequency)
    task._routine_stage = 2
    state.update(fraction=0.79, stool=False, placed=True, height=0.80)
    task.drawer_fixture = SimpleNamespace(stable_id="FloorPlan422:Drawer_452cc903_0")
    task.drawer_region = SimpleNamespace(
        world_pose=lambda sim: (np.zeros(3), 0.0),
        half_size=np.full(3, 0.2),
    )
    state["bbox_upper_x"] = 0.3
    task.object_bbox_points = lambda name: np.array(
        [
            [-0.1, -0.1, -0.1],
            [state["bbox_upper_x"], 0.1, 0.1],
        ]
    )
    task.object_pos = lambda name: np.array([0.1, 0.0, 0.0])
    task._candle_collision_points = lambda: task.object_bbox_points(task.TARGET_NAME)
    task._candle_inside_drawer = Candle._candle_inside_drawer.__get__(task)
    task._candle_contacts_drawer_bottom = Candle._candle_contacts_drawer_bottom.__get__(
        task
    )
    task._body_tree_ids = lambda roots: {2}
    task._drawer_bottom_geom_id = 0
    contacts = [
        SimpleNamespace(geom1=1, geom2=2, frame=[0, 0, 0.25], dist=-0.001),
        SimpleNamespace(geom1=2, geom2=1, frame=[0, 0, 1.0], dist=0.0005),
    ]
    task.sim = SimpleNamespace(
        model=SimpleNamespace(
            geom_id2name=lambda i: f"geom_{i}", geom_bodyid=np.array([1, 2, 3])
        ),
        data=SimpleNamespace(ncon=len(contacts), contact=contacts),
    )
    pairs = {(2, 7)}
    task.illegal_contact_pairs_for_bodies = lambda *a, **kw: pairs
    clock[0] = 4.99
    _tick(task, 1)
    assert capsys.readouterr().out == ""
    clock[0] = 5.0
    _tick(task, 2)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    line = lines[0]
    for detail in (
        "stage=3/3",
        "armed=True",
        "stage_goal=False",
        f"hold=0/{frequency}steps",
        "drawer_open=False",
        "drawer_fraction=0.7900/0.8000",
        "inside_drawer=False",
        "bottom_contact=False",
        "no_illegal_penetration=False",
        "lift_delta=0.3000/0.3000m",
        "placement_ok=False",
        "success=False",
        "illegal_pairs=[('geom_2', 'geom_7')]",
        "target_drawer=FloorPlan422:Drawer_452cc903_0",
        "candle_xyz=[0.1, 0.0, 0.0]",
        "drawer_xyz=[0.0, 0.0, 0.0]",
        "bbox_local_min=[-0.1, -0.1, -0.1]",
        "bounds_source=collision_mesh",
        "bbox_local_max=[0.3, 0.1, 0.1]",
        "allowed_half=[0.205, 0.205, 0.205]",
        "outside_xyz_m=[0.095, 0.0, 0.0]",
        "target_bottom=geom_0",
        "candle_contacts={'geom_2': {'dist_min_m': -0.001, 'dist_max_m': 0.0005, 'up_normal_z': 1.0}}",
    ):
        assert detail in line
    clock[0] = 9.99
    _tick(task, 1)
    assert capsys.readouterr().out == ""
    clock[0] = 10.0
    state["fraction"] = 0.8
    state["bbox_upper_x"] = 0.1
    contacts[:] = [SimpleNamespace(geom1=0, geom2=1, frame=[0, 0, 1.0], dist=0.0)]
    task.sim.data.ncon = len(contacts)
    pairs.clear()
    _tick(task, frequency)
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "placement_ok=True" in lines[0]
    assert "illegal_pairs=[]" in lines[0]
    assert "outside_xyz_m=[0.0, 0.0, 0.0]" in lines[0]
    assert "inside_drawer=True" in lines[0]
    assert "bottom_contact=True" in lines[0]
    assert "candle_contacts={'geom_0':" in lines[0]
    assert task._check_success()
    clock[0] = 15.0
    _tick(task, 1)
    assert "success=True" in capsys.readouterr().out
    task._reset_task_progress()
    clock[0] = 19.99
    _tick(task, 1)
    assert capsys.readouterr().out == ""
    clock[0] = 20.0
    _tick(task, 1)
    assert len(capsys.readouterr().out.splitlines()) == 1


@pytest.mark.parametrize("drawer_yaw", (0.0, np.pi / 2))
def test_candle_containment_uses_mesh_not_empty_rotated_box_corners(drawer_yaw):
    task = object.__new__(Candle)
    center = np.array([1.2, -0.6, 0.9])
    c, s = np.cos(drawer_yaw), np.sin(drawer_yaw)
    drawer_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    c, s = np.cos(np.pi / 4), np.sin(np.pi / 4)
    candle_rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    vertices = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [0, 0, 0.5],
            [0, 0, -0.5],
        ]
    )
    task._candle_collision_meshes = [(0, vertices)]
    task.sim = SimpleNamespace(
        data=SimpleNamespace(
            geom_xmat=np.array([(drawer_rotation @ candle_rotation).ravel()]),
            geom_xpos=np.array([center.copy()]),
        )
    )
    task.drawer_region = SimpleNamespace(
        world_pose=lambda sim: (center, drawer_yaw),
        half_size=np.ones(3),
    )
    corners = np.array(
        [[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-0.5, 0.5)]
    )
    # The old rotated box exceeds the drawer despite the whole mesh fitting.
    assert np.max(corners @ candle_rotation.T) > 1 + task.BOUNDS_TOLERANCE
    assert task._candle_inside_drawer()
    task.sim.data.geom_xpos[0] += drawer_rotation @ np.array([0.4, 0, 0])
    assert not task._candle_inside_drawer()  # Actual material outside still fails.


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "failure", (None, "side", "other_geom", "penetration", "hover")
)
def test_bottom_contact_requires_exact_geom_upward_normal_and_shallow_contact(
    reverse, failure
):
    task = object.__new__(Candle)
    task.obj_body_id = {"candle": 2}
    task._body_tree_ids = lambda *a: {2}
    task._drawer_bottom_geom_id = 0
    drawer_geom = 3 if failure == "other_geom" else 0
    geoms = (drawer_geom, 1) if not reverse else (1, drawer_geom)
    normal = 0.0 if failure == "side" else (-1.0 if reverse else 1.0)
    contact = SimpleNamespace(
        geom1=geoms[0],
        geom2=geoms[1],
        frame=[0, 0, normal],
        dist=-0.01
        if failure == "penetration"
        else (0.01 if failure == "hover" else 0.0),
    )
    task.sim = SimpleNamespace(
        model=SimpleNamespace(geom_bodyid=np.array([1, 2, 3, 1])),
        data=SimpleNamespace(ncon=1, contact=[contact]),
    )
    assert task._candle_contacts_drawer_bottom() is (failure is None)


def _sonic_stance(env):
    # Match the loaded SONIC stance, without requiring a locomotion policy.
    angles = dict(
        hip_pitch=-0.312,
        hip_roll=0.0,
        hip_yaw=0.0,
        knee=0.669,
        ankle_pitch=-0.363,
        ankle_roll=0.0,
        shoulder_pitch=0.2,
        shoulder_yaw=0.0,
        elbow=0.6,
        wrist_roll=0.0,
        wrist_pitch=0.0,
        wrist_yaw=0.0,
    )
    targets = {f"waist_{axis}": 0.0 for axis in ("yaw", "roll", "pitch")}
    for side, roll in (("left", 0.2), ("right", -0.2)):
        targets.update({f"{side}_{name}": value for name, value in angles.items()})
        targets[f"{side}_shoulder_roll"] = roll
    for name, value in targets.items():
        joint = f"robot0_{name}_joint"
        env.sim.data.set_joint_qpos(joint, value)
        env.sim.data.set_joint_qvel(joint, 0.0)
    env.sim.forward()
    return len(targets)


@pytest.mark.parametrize("cls", (Cabinet, Candle))
def test_headless_422_reset_and_physical_goals(monkeypatch, cls):
    import robosuite
    from robocasa.utils import env_utils
    from validate_ithor_tasks import (
        _assert_body_contact_state,
        _load_reset_controller,
        _settle_scene_with_frozen_robot,
        _task_body_contact_contracts,
    )

    monkeypatch.setattr(env_utils, "_apply_controller_spawn_pose", _sonic_stance)
    env = robosuite.make(
        env_name=cls.__name__,
        robots="G1FloatingBody",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        control_freq=20,
        renderer="mujoco",
        scene_ids=[422],
        seed=0,
    )
    try:
        env.reset()
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert not env._check_success()
        assert not env._raw_success()
        assert env.HOLD_STEPS == 20
        if cls is Cabinet:
            assert env.target_semantic_joint.fraction(env.sim) == pytest.approx(0.2)
            assert (
                env.robot_fixture_collision_distance(env.sink_fixture, max_distance=5.0)
                > 0.5
            )
            assert env._robot_clears_joint_sweep()
            # Pose probe verifies a collision-free, floor-supported sink endpoint.
            env._set_exact_robot_pose = (
                tasks.IThorBathroomLiftCandleFromFloor._set_exact_robot_pose.__get__(
                    env
                )
            )
            env._set_exact_robot_pose(
                {"xy": [-1.25, 0.29], "yaw": np.pi}, "422 sink endpoint"
            )
            assert env._raw_success()
            print(
                "422 cabinet/sink: collision-free initial stance, door sweep and sink endpoint"
            )
        else:
            assert env.drawer_fixture.stable_id == "FloorPlan422:Drawer_452cc903_0"
            assert (
                env.drawer_joint.name
                == "drawer_e69d502ac6388ef5b2b2cb0a94a4962b_1_1_0_joint_0"
            )
            assert env.drawer_joint.open_qpos == pytest.approx(0.22)
            assert env.drawer_joint.fraction(env.sim) == pytest.approx(0.2)
            assert env.scene_task_spec()["source_xy"] == [-0.87953, 1.00641]
            assert env.scene_task_spec()["stool_xy"] == [-0.87953, 1.00641]
            np.testing.assert_allclose(env._robot_base_xy(), [-1.19715, 0.81885])
            assert np.linalg.norm(
                env.object_pos(env.TARGET_NAME)[:2] - env._robot_base_xy()
            ) == pytest.approx(float(np.hypot(0.31762, 0.18756)), abs=1e-3)
            offset = env.object_pos(env.TARGET_NAME)[:2] - env._robot_base_xy()
            yaw = env._robot_base_yaw()
            forward = np.array([np.cos(yaw), np.sin(yaw)])
            left = np.array([-np.sin(yaw), np.cos(yaw)])
            assert 0.18 < offset @ left < 0.20
            assert 0.31 < offset @ forward < 0.33
            np.testing.assert_allclose(forward, [1.0, 0.0], atol=1e-6)
            assert env.scene_task_spec()["robot_pose"]["yaw"] == 0.0
            assert env._candle_is_on_stool()
            initial = env.object_pos(env.TARGET_NAME).copy()
            for name, subjects, supports in _task_body_contact_contracts(env):
                _assert_body_contact_state(env, name, subjects, supports, "reset")
            _settle_scene_with_frozen_robot(env, int(0.5 / env.sim.model.opt.timestep))
            assert np.linalg.norm(env.object_pos(env.TARGET_NAME) - initial) < 0.01
            assert env._candle_is_on_stool()
            env.sim.data.set_joint_qpos(
                env.drawer_joint.name, env.drawer_joint.open_qpos
            )
            env.sim.forward()
            # Full-open drawer clearance is omitted at the user's request.
            center = env.drawer_region.world_pose(env.sim)[0]
            # The semantic lower face matches this drawer's physical bottom.
            support = replace(
                env.drawer_region,
                kind="support",
                _world_offset=env.drawer_region._world_offset
                + np.array([0.0, 0.0, -env.drawer_region.half_size[2]]),
            )
            assert env.place_object_on_support_contact(
                env.TARGET_NAME,
                support,
                env.drawer_fixture,
                xy=center[:2],
                inward_fractions=(0.0,),
                probe_above=0.01,
                probe_below=0.01,
                probe_step=0.0005,
            )
            assert env._raw_success()
            # A tilted candle can touch the bottom near a wall while empty
            # corners of its enclosing box protrude through both surfaces.
            joint = env.objects[env.TARGET_NAME].joints[0]
            qpos = env.sim.data.get_joint_qpos(joint).copy()
            tilt, yaw = np.deg2rad(65), np.deg2rad(35)
            ct, st = np.cos(tilt / 2), np.sin(tilt / 2)
            cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
            qpos[3:] = [cy * ct, -sy * st, cy * st, sy * ct]
            env.sim.data.set_joint_qpos(joint, qpos)
            env.sim.forward()
            center, drawer_yaw = env.drawer_region.world_pose(env.sim)
            c, s = np.cos(drawer_yaw), np.sin(drawer_yaw)
            rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            local = (env._candle_collision_points() - center) @ rotation
            bottom = env._drawer_bottom_geom_id
            bottom_rotation = env.sim.data.geom_xmat[bottom].reshape(3, 3)
            bottom_z = env.sim.data.geom_xpos[bottom, 2] + (
                np.abs(bottom_rotation[2]) @ env.sim.model.geom_size[bottom]
            )
            shift = np.array(
                [
                    -env.drawer_region.half_size[0] + 0.001 - local[:, 0].min(),
                    -(local[:, 1].min() + local[:, 1].max()) / 2,
                    bottom_z - 0.0001 - center[2] - local[:, 2].min(),
                ]
            )
            qpos[:3] += rotation @ shift
            env.sim.data.set_joint_qpos(joint, qpos)
            env.sim.forward()
            box = (env.object_bbox_points(env.TARGET_NAME) - center) @ rotation
            half = env.drawer_region.half_size + env.BOUNDS_TOLERANCE
            assert np.any(box < -half) or np.any(box > half)
            assert env._candle_inside_drawer()
            assert env._candle_contacts_drawer_bottom()
            assert env._raw_success()
            print(
                "422 tilted candle: mesh fits and touches bottom while old box exceeds bounds"
            )
            env._set_task_object_bottom_at(
                env.TARGET_NAME, center[:2], env.object_bottom_z(env.TARGET_NAME) + 0.03
            )
            assert not env._raw_success()  # Hovering inside the semantic volume.
            print(
                "422 candle: stable stool contact, drawer bottom placement positive and hover negative"
            )
    finally:
        env.close()
