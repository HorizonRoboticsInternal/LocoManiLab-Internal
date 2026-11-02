"""Scene-local registration, ordered stages, and representative initialization."""

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
from robocasa.environments.holodeck.tasks.multi_hop import (
    alarm_clock_workflows as workflows,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robosuite.environments.base import REGISTERED_ENVS
from validate_ithor_tasks import _load_reset_controller


BOX = workflows.HolodeckPlaceAlarmClockThenCloseDrawerThenPickBox
TRASH = workflows.HolodeckOpenDrawerThenPlaceAlarmClockThenApproachTrashCan


def test_registration_scope_and_unchanged_atomic_contracts():
    for cls in (BOX, TRASH):
        for catalog in (holodeck, tasks, workflows):
            assert getattr(catalog, cls.__name__) is cls
            assert catalog.__all__.count(cls.__name__) == 1
        assert REGISTERED_ENVS[cls.__name__] is cls
        assert REGISTERED_HOLODECK_ENVS[cls.__name__] is cls
        assert cls.SUPPORTED_SCENE_IDS == (2209,)
        assert not cls.USE_DEFAULT_ASSET_LAYOUT
        with pytest.raises(KeyError):
            get_holodeck_task_spec("train_117", cls.WORKFLOW_SPEC_TASK)
    for registry in (REGISTERED_ENVS, REGISTERED_HOLODECK_ENVS):
        assert "_HolodeckClockDrawerWorkflow" not in registry
    for cls in (
        workflows.HolodeckOpenDrawer,
        workflows.HolodeckCloseDrawer,
        workflows.HolodeckPickBoxFromFloor,
        workflows.HolodeckApproachTrashCan,
    ):
        assert cls.SUPPORTED_SCENE_IDS == (117, 1018, 1197, 1664, 2025)
    atomic = get_holodeck_task_spec("train_2209", "place_alarm_clock_in_drawer")
    assert atomic["drawer_open_qpos"] == 0.294
    assert atomic["robot_pose"]["pos"] == [2.92, 5.04, 0.0]
    assert "box_pose" not in atomic and "bin_pos" not in atomic


def _fake(monkeypatch, cls, frequency):
    monkeypatch.setattr(
        HolodeckEnv, "_post_action", lambda self, action: (0, False, {})
    )
    task = object.__new__(cls)
    task.control_freq = frequency
    task._atomic_tasks = task.ATOMIC_TASKS
    task._reset_composite_progress()
    task._success_hold = 0
    task._clock_stage_armed = False
    task._close_stage_armed = False
    state = dict(
        clock=False,
        drawer=1.0 if cls is BOX else 0.3,
        box=False,
        distance=2.0,
        xy=np.zeros(2),
    )
    task._goal_satisfied = lambda: state["clock"]
    task._clock_placed = lambda: state["clock"]
    task._drawer_fraction = lambda: state["drawer"]
    task._box_lifted = lambda: state["box"]
    task._trash_distance = lambda: state["distance"]
    task._robot_base_xy = lambda: state["xy"]
    task.workflow_spec = {"success_radius": 0.35}
    task._enter_stage(task.current_atomic_task)
    return task, state


def _advance(task, steps=None):
    if steps is None:
        steps = task.control_freq
    for _ in range(steps):
        task._post_action(None)


@pytest.mark.parametrize("frequency", (20, 50))
def test_place_close_pick_order_and_continuous_hold(monkeypatch, frequency):
    task, state = _fake(monkeypatch, BOX, frequency)
    state.update(drawer=0.2, box=True)
    _advance(task)
    assert task.stage_index == 0
    state.update(clock=True, drawer=1.0)
    _advance(task)
    assert task.stage_index == 1
    # Lifting early does not count as lifting after closing.
    state["drawer"] = 0.2
    _advance(task)
    assert task.stage_index == 2
    _advance(task)
    assert not task._check_success()
    state["box"] = False
    _advance(task, 1)
    state["box"] = True
    _advance(task, frequency - 1)
    assert not task._check_success()
    state["clock"] = False
    _advance(task, 1)
    assert task._stage_hold_steps == 0
    state["clock"] = True
    _advance(task)
    assert task._check_success()
    task._reset_composite_progress()
    assert not task._check_success()


@pytest.mark.parametrize("frequency", (20, 50))
def test_open_place_walk_rejects_early_goals(monkeypatch, frequency):
    task, state = _fake(monkeypatch, TRASH, frequency)
    state.update(clock=True, distance=0.2)
    _advance(task)
    assert task.stage_index == 0
    state["drawer"] = 0.8
    _advance(task)
    assert task.stage_index == 1
    _advance(task)
    assert task.stage_index == 1
    state["clock"] = False
    _advance(task, 1)
    state["clock"] = True
    _advance(task)
    assert task.stage_index == 2
    state["xy"] = np.array([0.5, 0.0])
    _advance(task)
    assert not task._check_success()
    state["distance"] = 0.5
    _advance(task, 1)
    state["distance"] = 0.35
    _advance(task, frequency - 1)
    assert not task._check_success()
    state["distance"] = 0.351
    _advance(task, 1)
    assert task._stage_hold_steps == 0
    state["distance"] = 0.35
    _advance(task)
    assert task._check_success()


@pytest.mark.parametrize("cls", (BOX, TRASH), ids=lambda cls: cls.__name__)
def test_one_headless_reset_and_support_stability(cls):
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
        assert not env._check_success() and env.stage_index == 0
        assert not env._clock_placed()
        assert env.count_body_collisions() == 0
        assert env._drawer_fraction() == pytest.approx(
            env.workflow_spec["drawer_initial_fraction"]
        )
        base = env.sim.data.get_body_xpos(env.robots[0].robot_model.root_body)[:2]
        for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            point = base + 0.25 * np.array([np.cos(angle), np.sin(angle)])
            assert env.room_of_point([*point, 0.0]) is not None
        source_geoms = env._geom_names_for_bodies(
            env._body_tree_ids(
                [env.sim.model.body_name2id(env.task_spec["source_body"])],
            )
        )
        assert env.check_contact(env._clock_geoms, source_geoms)
        if cls is BOX:
            assert env.objects["box"]._scale == pytest.approx([env.BOX_SCALE] * 3)
            assert env._box_bottom_z() == pytest.approx(0, abs=0.001)
            assert not env._box_lifted()
        else:
            assert "box" not in env.objects
            assert env._trash_distance() > env.workflow_spec["success_radius"] + 0.15
            floor_id = env.sim.model.geom_name2id(f"{env.trash_can_body}_inner_floor")
            floor_z = env.sim.data.geom_xpos[floor_id, 2]
            assert abs(floor_z - 0.04) < 0.001
        # Allow only the newly reused free objects to move; hold robot/furniture fixed.
        model, data = env.sim.model, env.sim.data
        qmask, vmask = np.ones(model.nq, bool), np.ones(model.nv, bool)
        joints = [env._clock_joint]
        if cls is BOX:
            joints.append(env.objects["box"].joints[0])
        for name in joints:
            joint = model.joint_name2id(name)
            qadr, dadr = model.jnt_qposadr[joint], model.jnt_dofadr[joint]
            qmask[qadr : qadr + 7] = False
            vmask[dadr : dadr + 6] = False
        fixed = data.qpos.copy()
        initial_clock = data.body_xpos[env._clock_body_id].copy()
        initial_box = env._box_root_z() if cls is BOX else None
        for _ in range(int(np.ceil(0.4 / model.opt.timestep))):
            env.sim.step()
            data.qpos[qmask] = fixed[qmask]
            data.qvel[vmask] = 0
            env.sim.forward()
        assert np.linalg.norm(data.body_xpos[env._clock_body_id] - initial_clock) < 0.01
        assert env.check_contact(env._clock_geoms, source_geoms)
        if cls is BOX:
            assert abs(env._box_root_z() - initial_box) < 0.01
        assert not env._check_success()
    finally:
        env.close()
