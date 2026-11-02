"""Scene 410 drawer registration, bottom contact, lift height and hold contracts."""

from types import SimpleNamespace

import numpy as np
import pytest

from robocasa.environments.ithor_bathroom import tasks
from robocasa.environments.ithor_bathroom.tasks.drawer_pnp import (
    IThorBathroomTakeSoapBottleFromDrawer as Take,
    IThorBathroomToiletPaperRollIntoDrawer as Put,
)
from robocasa.environments.ithor_bathroom.tasks.object_manipulation import (
    _BathroomWorkspaceMixin,
)
from robocasa.environments.ithor_task_catalog import (
    BATHROOM_PACKAGE,
    discover_task_catalog,
    validate_task_catalog,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion


def test_410_tasks_are_registered_only_for_410():
    for cls in (Put, Take):
        assert cls.SUPPORTED_SCENE_IDS == (410,)
        assert cls in tasks.ATOMIC_TASK_CLASSES
        assert cls.__name__ in tasks.__all__
        spec = cls.SCENE_SPECS["FloorPlan410"]
        assert spec["drawer_fixture"] == "FloorPlan410:Drawer_c81edb51_0"
        assert (
            spec["drawer_joint"]
            == "drawer_3593bc74e14afaf576668610fec0f56f_1_1_0_joint_0"
        )
    assert validate_task_catalog(discover_task_catalog((BATHROOM_PACKAGE,))) == []
    assert 410 not in tasks.IThorBathroomLiftSoapBottle.SUPPORTED_SCENE_IDS
    assert 410 not in tasks.IThorBathroomLiftToiletPaperRoll.SUPPORTED_SCENE_IDS


def test_drawer_bounds_follow_articulated_region_and_require_full_containment():
    task = object.__new__(Put)
    task.sim = None
    task.source_fixture = object()
    center = np.array([-1.3, 2.27, 0.608])
    task.drawer_region = SimpleNamespace(
        world_pose=lambda sim: (center, np.pi / 2),
        half_size=np.array([0.37874, 0.206928, 0.0907076]),
    )
    points = np.array([[-0.05, -0.06, -0.08], [0.05, 0.06, 0.05]])
    task._fixture_bbox_points = lambda fixture: points + center
    assert task._source_inside_drawer()
    assert not task._source_clear_of_drawer()
    # Partial overlap, including an object whose center is outside, cannot pass.
    points[:, 2] += 0.12
    assert not task._source_inside_drawer()
    assert not task._source_clear_of_drawer()
    points[:, 2] += 0.10
    assert task._source_clear_of_drawer()
    # The volume follows the drawer, rather than using its closed-world pose.
    center[1] += 0.15
    assert task._source_clear_of_drawer()


@pytest.mark.parametrize("cls", (Put, Take))
@pytest.mark.parametrize("control_freq", (10, 20))
def test_success_predicates_and_one_second_hold(monkeypatch, cls, control_freq):
    task = object.__new__(cls)
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    task._success_hold = 0
    task.source_fixture = object()
    task.drawer_fixture = SimpleNamespace(body_names=("drawer",))
    task.robots = [SimpleNamespace(robot_model=SimpleNamespace(root_body="robot"))]
    state = dict(
        inside=False,
        clear=False,
        contact=False,
        released=True,
        grasp=False,
        penetration=False,
        height=0.5,
    )
    task._initial_source_z = state["height"]
    task._source_root_z = lambda: state["height"]
    task._source_inside_drawer = lambda: state["inside"]
    task._source_clear_of_drawer = lambda: state["clear"]
    task._fixture_contacts_fixture = lambda *args: state["contact"]
    task._source_contacts_drawer_bottom = lambda: state["contact"]
    task._fixture_released = lambda *args: state["released"]
    task._g1_hand_grasping_fixture = lambda *args: state["grasp"]
    task._fixture_has_illegal_penetration = lambda *args: state["penetration"]
    monkeypatch.setattr(
        _BathroomWorkspaceMixin,
        "_post_action",
        lambda *args: None,
        raising=False,
    )
    assert not task._raw_success()
    if cls is Put:
        # Bottom contact is sufficient even if held or protruding above the rim.
        state.update(inside=False, contact=True, released=False, grasp=True)
        assert task._raw_success()
        state["contact"] = False
        assert not task._raw_success()
        state["contact"] = True
    else:
        # Moving clear or grasping below the height threshold is insufficient.
        state.update(clear=True, grasp=True, height=0.799)
        assert not task._raw_success()
        # Exact 0.30 m threshold, even without grasp and while touching the drawer.
        state.update(height=0.8, grasp=False, clear=False, contact=True)
        assert task._raw_success()
        state["grasp"] = True
        assert task._raw_success()
        state["grasp"] = False
    for _ in range(control_freq - 1):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()
    if cls is Put:
        state["contact"] = False
    else:
        state["height"] = 0.799
    task._post_action(None)
    assert not task._check_success()
    assert task._success_hold == 0
    if cls is Take:
        state["height"] = 0.8
        for _ in range(control_freq - 1):
            task._post_action(None)
        assert not task._check_success()
        task._post_action(None)
        assert task._check_success()


def test_soap_lift_reset_records_post_placement_height(monkeypatch):
    from robocasa.environments.ithor_bathroom.tasks.drawer_pnp import (
        _BathroomDrawerTask,
    )

    task = object.__new__(Take)
    task.source_fixture = SimpleNamespace(body_name="soap")
    positions = np.array([[0.0, 0.0, -1.0]])
    task.sim = SimpleNamespace(
        model=SimpleNamespace(body_name2id=lambda name: 0),
        data=SimpleNamespace(body_xpos=positions),
    )

    def reset_parent(self):
        # Placement must finish before the initial height is recorded.
        positions[0, 2] = 0.6
        self._success_hold = 0

    monkeypatch.setattr(_BathroomDrawerTask, "_reset_internal", reset_parent)
    for _ in range(2):
        positions[0, 2] = 2.0
        task._success_hold = 20
        task._reset_internal()
        assert task._initial_source_z == pytest.approx(0.6)
        assert task._success_hold == 0
        assert not task._raw_success()
        positions[0, 2] = 0.91
        assert task._raw_success()


@pytest.mark.parametrize(
    "geom_pair, normal, point, distance, expected",
    (
        ((1, 0), (0, 0, 1), (0, 0, 0.5), -0.001, True),
        ((1, 0), (0, 0, 1), (0, 0, 0.512), 0.0004, True),  # bbox offset
        ((0, 1), (0, 0, -1), (0, 0, 0.5), -0.001, True),
        ((1, 0), (1, 0, 0), (0, 0, 0.5), -0.001, False),  # side wall
        ((1, 0), (0, 0, -1), (0, 0, 0.5), -0.001, False),  # underside
        ((1, 0), (0, 0, 1), (0, 0, 0.68), -0.001, False),  # rim
        ((1, 0), (0, 0, 1), (0.5, 0, 0.5), -0.001, False),  # outside bottom
        ((2, 0), (0, 0, 1), (0, 0, 0.5), -0.001, False),  # another drawer
        ((1, 0), (0, 0, 1), (0, 0, 0.5), 0.01, False),  # hovering
        ((1, 0), (0, 0, 1), (0, 0, 0.5), -0.01, False),  # penetrating
    ),
)
def test_roll_bottom_contact_uses_target_surface(
    geom_pair,
    normal,
    point,
    distance,
    expected,
):
    task = object.__new__(Put)
    task.source_fixture = object()
    task.drawer_fixture = object()
    task._fixture_body_ids = lambda fixture, **kwargs: (
        {10} if fixture is task.source_fixture else {20}
    )
    contact = SimpleNamespace(
        geom1=geom_pair[0],
        geom2=geom_pair[1],
        frame=np.asarray((*normal, 0, 0, 0, 0, 0, 0)),
        pos=np.asarray(point),
        dist=distance,
    )
    task.sim = SimpleNamespace(
        model=SimpleNamespace(geom_bodyid=np.array([10, 20, 30])),
        data=SimpleNamespace(ncon=1, contact=[contact]),
    )
    task.drawer_support_region = SemanticRegion(
        stable_id="bottom",
        fixture_id="drawer",
        kind="support",
        parent_body=None,
        local_center=np.zeros(3),
        half_size=np.array([0.38, 0.21, 0.02]),
        _world_center=np.array([0, 0, 0.5]),
    )
    assert task._raw_success() is expected
    task.sim.data.ncon = 0
    assert not task._raw_success()


@pytest.mark.parametrize("cls", (Put, Take))
def test_viewer_stance_reset_and_drawer_placement(cls):
    import robosuite
    from robocasa.g1_host.bridge import load_sim2real_controller_config
    from validate_ithor_tasks import (
        _assert_body_contact_state,
        _settle_scene_with_frozen_robot,
        _task_body_contact_contracts,
    )

    env = robosuite.make(
        env_name=cls.__name__,
        robots="G1FloatingBody",
        controller_configs=load_sim2real_controller_config("sonic", "mock"),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=True,
        control_freq=20,
        renderer="mujoco",
        scene_ids=[410],
        seed=0,
    )
    try:
        env.reset()
        assert env.count_body_collisions() == 0
        assert env.robot_footprint_is_on_floor(env._robot_base_xy(), radius=0.25)
        assert not env._raw_success()
        assert not env._check_success()
        assert env.HOLD_STEPS == 20
        assert env.sim.data.get_joint_qpos(env.drawer_joint.name) == pytest.approx(
            0.386
        )
        assert env._source_horizontal_distance < 0.70
        initial = env.source_fixture.world_center(env.sim).copy()
        if cls is Take:
            source_body_id = env.sim.model.body_name2id(env.source_fixture.body_name)
            # The bottle's native +Y axis runs from its base toward its pump.
            assert env.sim.data.body_xmat[source_body_id].reshape(3, 3)[2, 1] > 0.999
        _settle_scene_with_frozen_robot(env, int(0.5 / env.sim.model.opt.timestep))
        drift = np.linalg.norm(env.source_fixture.world_center(env.sim) - initial)
        assert drift < 0.01
        assert env.count_body_collisions() == 0
        assert not env._raw_success()
        for name, subjects, supports in _task_body_contact_contracts(env):
            _assert_body_contact_state(env, name, subjects, supports, "after settle")
        if cls is Take:
            assert env.sim.data.body_xmat[source_body_id].reshape(3, 3)[2, 1] > 0.99
            # An upright bottle extends above the rim; its base stays in the drawer.
            assert env._fixture_bbox_inside_region_xy(
                env.source_fixture,
                env.drawer_support_region,
            )
            assert env._fixture_contacts_fixture(env.source_fixture, env.drawer_fixture)
            assert env._initial_source_z == pytest.approx(
                env._source_root_z(), abs=0.01
            )
            assert not env._g1_hand_grasping_fixture(env.source_fixture)
            joint = env._fixture_free_joint_name(env.source_fixture)
            qpos = env.sim.data.get_joint_qpos(joint).copy()
            for lift, expected in ((0.29, False), (0.31, True)):
                qpos[2] = env._initial_source_z + lift
                env.sim.data.set_joint_qpos(joint, qpos)
                env.sim.forward()
                assert env._raw_success() is expected
            assert not env._g1_hand_grasping_fixture(env.source_fixture)
            return
        target_xy = env.drawer_region.world_pose(env.sim)[0][:2]
        poses = (
            ("upright", env.scene_task_spec()["source_quat"]),
            ("sideways", [1, 0, 0, 0]),
            # Tilt 45 degrees about a diagonal horizontal axis from upright.
            ("tilted", [-0.19134172, 0.19134172, 0.46193977, 0.84462320]),
        )
        protruding_pose_passed = False
        for name, quaternion in poses:
            env._set_fixture_quaternion(env.source_fixture, quaternion)
            assert env._place_fixture_on_support(
                env.source_fixture,
                env.drawer_support_region,
                env.drawer_fixture,
                target_xy,
                contract_name="roll_inside_drawer",
            )
            assert env._raw_success(), name
            protruding_pose_passed |= not env._source_inside_drawer()
        assert protruding_pose_passed
        joint = env._fixture_free_joint_name(env.source_fixture)
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        # Hovering inside the volume has no physical support and must fail.
        qpos[2] += 0.01
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        assert not env._raw_success()
        # The neighbouring upper drawer is not a valid destination.
        qpos[0] -= 0.86233
        qpos[2] -= 0.01
        env.sim.data.set_joint_qpos(joint, qpos)
        env.sim.forward()
        assert not env._raw_success()
        print(
            {
                "source_distance": env._source_horizontal_distance,
                "settle_drift": float(drift),
                "hold_steps": env.HOLD_STEPS,
            }
        )
    finally:
        env.close()
