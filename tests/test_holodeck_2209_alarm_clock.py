"""Focused contracts, geometry and reset checks for scene 2209's clock tasks."""

from types import SimpleNamespace

import numpy as np
import pytest
import robosuite

from robocasa.environments import holodeck
from robocasa.environments.holodeck import tasks
from robocasa.environments.holodeck.holodeck import (
    HolodeckEnv,
    REGISTERED_HOLODECK_ENVS,
)
from robocasa.environments.holodeck.tasks.alarm_clock import (
    HolodeckPickAlarmClockFromSofa,
    HolodeckPlaceAlarmClockInDrawer,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robosuite.environments.base import REGISTERED_ENVS
from validate_ithor_tasks import _load_reset_controller


CLASSES = (HolodeckPickAlarmClockFromSofa, HolodeckPlaceAlarmClockInDrawer)


def test_registration_scene_scope_and_permanent_cleanup():
    for cls in CLASSES:
        assert getattr(holodeck, cls.__name__) is cls
        assert getattr(tasks, cls.__name__) is cls
        assert REGISTERED_HOLODECK_ENVS[cls.__name__] is cls
        assert REGISTERED_ENVS[cls.__name__] is cls
        assert cls.SUPPORTED_SCENE_IDS == (2209,)
        assert not cls.USE_DEFAULT_ASSET_LAYOUT
        spec = get_holodeck_task_spec("train_2209", cls.SPEC_TASK)
        assert "lang" not in spec
        with pytest.raises(KeyError):
            get_holodeck_task_spec("train_117", cls.SPEC_TASK)
    assert "_HolodeckAlarmClockTask" not in REGISTERED_ENVS
    assert "_HolodeckAlarmClockTask" not in REGISTERED_HOLODECK_ENVS
    assert HolodeckEnv.HIDDEN_SCENE_BODIES["train_2209"] == (
        "statue_59b4e93be5e649fe613e418b442aec0b_2_0_0",
        "obja_e55514b2057c4ad7853722d0cdcf52f7_1_0_0",
    )
    assert "train_1477" not in HolodeckEnv.HIDDEN_SCENE_BODIES
    notebook = "obja_e573304fce364cf299027494fc1afede_1_0_0"
    assert HolodeckPickAlarmClockFromSofa.HIDDEN_SCENE_BODIES == {
        **HolodeckEnv.HIDDEN_SCENE_BODIES,
        "train_2209": (*HolodeckEnv.HIDDEN_SCENE_BODIES["train_2209"], notebook),
    }
    assert (
        HolodeckPlaceAlarmClockInDrawer.HIDDEN_SCENE_BODIES
        is HolodeckEnv.HIDDEN_SCENE_BODIES
    )


def _fake_task(cls):
    task = object.__new__(cls)
    task.control_freq = 20
    task._clock_initial_z = 1.0
    task._clock_body_id = 0
    task._success_hold = 0
    task.sim = SimpleNamespace(
        data=SimpleNamespace(body_xpos=np.array([[0.0, 0.0, 1.0]]))
    )
    return task


@pytest.mark.parametrize("control_freq", (20, 50))
def test_lift_requires_only_height_and_continuous_hold(monkeypatch, control_freq):
    monkeypatch.setattr(HolodeckEnv, "_post_action", lambda self, action: None)
    task = _fake_task(HolodeckPickAlarmClockFromSofa)
    task.control_freq = control_freq
    # Neither current grasp nor prior grasp is required.
    task._clock_grasped = lambda: pytest.fail("success must not query grasp state")
    task.sim.data.body_xpos[0, 2] = 1.31
    for _ in range(task.hold_steps - 1):
        task._post_action(None)
    assert not task._check_success()
    task.sim.data.body_xpos[0, 2] = 1.299
    task._post_action(None)
    assert task._success_hold == 0
    assert not task._check_success()
    task.sim.data.body_xpos[0, 2] = 1.30
    for _ in range(task.hold_steps):
        task._post_action(None)
    assert task._check_success()
    assert task.reward() == 1.0
    task.sim.data.body_xpos[0, 2] = 1.299
    task._post_action(None)
    assert not task._check_success()
    assert task.reward() == 0.0


@pytest.mark.parametrize("control_freq", (20, 50))
def test_placement_requires_only_continuous_support_contact(monkeypatch, control_freq):
    monkeypatch.setattr(HolodeckEnv, "_post_action", lambda self, action: None)
    task = _fake_task(HolodeckPlaceAlarmClockInDrawer)
    task.control_freq = control_freq
    task._drawer_bottom_id = 7
    task._clock_geoms = ["clock_collision"]
    task.sim.model = SimpleNamespace(
        geom_id2name=lambda geom: "drawer_floor" if geom == 7 else "drawer_wall"
    )
    task._clock_grasped = lambda: pytest.fail("success must not query grasp state")
    state = {"contact": False}

    def contact(geoms, clock_geoms):
        assert geoms == ["drawer_floor"]
        assert clock_geoms == task._clock_geoms
        return state["contact"]

    task.check_contact = contact
    task._post_action(None)
    assert not task._check_success()
    state["contact"] = True
    for _ in range(task.hold_steps - 1):
        task._post_action(None)
    assert not task._check_success()
    # Any interruption resets the one-second timer.
    state["contact"] = False
    task._post_action(None)
    assert task._success_hold == 0
    state["contact"] = True
    for _ in range(task.hold_steps):
        task._post_action(None)
    assert task._check_success()
    assert task.reward() == 1.0
    state["contact"] = False
    task._post_action(None)
    assert not task._check_success()
    assert task.reward() == 0.0


def _settle_clock_only(env, seconds=0.4):
    """Check target/support stability while holding unrelated scene state fixed."""
    model, data = env.sim.model, env.sim.data
    joint_id = model.joint_name2id(env._clock_joint)
    qadr, dadr = int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
    fixed_qpos = data.qpos.copy()
    qmask, vmask = np.ones(model.nq, dtype=bool), np.ones(model.nv, dtype=bool)
    qmask[qadr : qadr + 7] = False
    vmask[dadr : dadr + 6] = False
    for _ in range(int(np.ceil(seconds / model.opt.timestep))):
        env.sim.step()
        data.qpos[qmask] = fixed_qpos[qmask]
        data.qvel[vmask] = 0
        env.sim.forward()


def _set_clock(env, pos):
    pose = env.sim.data.get_joint_qpos(env._clock_joint).copy()
    pose[:3] = pos
    env.sim.data.set_joint_qpos(env._clock_joint, pose)
    env.sim.data.set_joint_qvel(env._clock_joint, np.zeros(6))
    env.sim.forward()


@pytest.mark.parametrize("cls", CLASSES, ids=lambda cls: cls.__name__)
def test_headless_reset_source_stability_and_drawer_geometry(cls):
    env = robosuite.make(
        env_name=cls.__name__,
        robots="G1FloatingBody",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        initialization_noise=None,
        control_freq=50,
        scene_ids=[2209],
        seed=0,
    )
    try:
        env.reset()
        assert env.scene_name == "train_2209"
        assert not env._check_success()
        assert not env._clock_grasped()
        assert env.count_body_collisions() == 0
        assert not getattr(env, "_spawn_cleared_geom_state", {})
        for body in cls.HIDDEN_SCENE_BODIES["train_2209"]:
            assert body not in env.sim.model.body_names
        # Other native clocks, books and the furniture are preserved.
        for body in (
            "alarmclock_7e4d181f2517b07b61d9406b0a04ede2_1_0_0",
            "alarmclock_7e4d181f2517b07b61d9406b0a04ede2_2_0_0",
            env.task_spec["source_body"],
            "book_6807c2e0132b1d71cafcc3a5ea9619e9_1_0_0",
            "book_1dc4b51c02d3770f5204b6d1c7c714d5_1_0_0",
        ):
            assert body in env.sim.model.body_names
        robot_ids = env._body_tree_ids(
            [env.sim.model.body_name2id(env.robots[0].robot_model.root_body)]
        )
        for contact in env.sim.data.contact[: env.sim.data.ncon]:
            b1, b2 = env.sim.model.geom_bodyid[[contact.geom1, contact.geom2]]
            if (b1 in robot_ids) != (b2 in robot_ids):
                assert contact.dist >= -0.0002
        source_ids = env._body_tree_ids(
            [env.sim.model.body_name2id(env.task_spec["source_body"])]
        )
        source_geoms = env._geom_names_for_bodies(source_ids)
        assert env.check_contact(env._clock_geoms, source_geoms)
        initial = env.sim.data.body_xpos[env._clock_body_id].copy()
        _settle_clock_only(env)
        assert (
            np.linalg.norm(env.sim.data.body_xpos[env._clock_body_id] - initial) < 0.01
        )
        assert env.check_contact(env._clock_geoms, source_geoms)
        assert not env._check_success()
        if cls is HolodeckPlaceAlarmClockInDrawer:
            assert env.sim.data.get_joint_qpos(
                env.task_spec["drawer_joint"]
            ) == pytest.approx(0.294)
            center = env.sim.data.site_xpos[env._drawer_site_id].copy()
            bottom = env.sim.data.geom_xpos[env._drawer_bottom_id, 2]
            bottom += env.sim.model.geom_size[env._drawer_bottom_id, 1]
            root_to_bottom = (
                env.sim.data.body_xpos[env._clock_body_id, 2]
                - env._clock_points()[:, 2].min()
            )
            inside = np.array(
                [center[0], center[1] - 0.08, bottom + root_to_bottom + 0.0002]
            )
            _set_clock(env, inside)
            _settle_clock_only(env)
            # Direct placement meets the contact condition without grasping.
            assert env._goal_satisfied()
            # Velocity is irrelevant while the support contact is present.
            env.sim.data.set_joint_qvel(env._clock_joint, [0.2, 0, 0, 0, 0, 0])
            assert env._goal_satisfied()
            # Floating or exterior placements lack the named floor contact.
            for offset in ([0, 0, 0.03], [0, -0.25, 0], [0.5, 0, 0]):
                _set_clock(env, inside + offset)
                assert not env._goal_satisfied()
    finally:
        env.close()
