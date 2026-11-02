"""Focused timing, scope and physical checks for scene 2025's candle task."""

import math
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import robosuite

from robocasa.environments import holodeck
from robocasa.environments.holodeck import tasks
from robocasa.environments.holodeck.holodeck import (
    HolodeckEnv,
    REGISTERED_HOLODECK_ENVS,
    get_holodeck_scene_xml,
)
from robocasa.environments.holodeck.tasks.stand_candle_upright import (
    HolodeckStandCandleUpright,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robosuite.environments.base import REGISTERED_ENVS
from validate_ithor_tasks import _load_reset_controller


TASK = HolodeckStandCandleUpright


def test_registration_and_scene_scope():
    name = TASK.__name__
    assert getattr(holodeck, name) is getattr(tasks, name) is TASK
    assert name in holodeck.__all__ and name in tasks.__all__
    assert REGISTERED_HOLODECK_ENVS[name] is REGISTERED_ENVS[name] is TASK
    assert TASK.SUPPORTED_SCENE_IDS == (2025,)
    assert TASK.ATOMIC_SKILL == "PlaceObject"
    spec = get_holodeck_task_spec("train_2025", TASK.SPEC_TASK)
    assert "lang" not in spec
    for scene in (117, 1477, 2209):
        with pytest.raises(KeyError):
            get_holodeck_task_spec(f"train_{scene}", TASK.SPEC_TASK)
    for ids in ([], [117], [2025, 2209], "train_2209"):
        with pytest.raises(ValueError, match="only supports scene 2025"):
            TASK(scene_ids=ids)
    # No other registered task declares this target/skill pair.
    assert [
        cls
        for cls in REGISTERED_HOLODECK_ENVS.values()
        if getattr(cls, "TARGET_BODY", None) == TASK.TARGET_BODY
        and getattr(cls, "ATOMIC_SKILL", None) == TASK.ATOMIC_SKILL
    ] == [TASK]


@pytest.mark.parametrize("control_freq", (20, 50))
def test_upright_contact_and_uninterrupted_one_second(monkeypatch, control_freq):
    monkeypatch.setattr(HolodeckEnv, "_post_action", lambda self, action: None)
    task = object.__new__(TASK)
    task.control_freq = control_freq
    task._success_hold = 0
    task._candle_body_id = 0
    task._candle_geoms = ["target"]
    task.task_spec = {"tabletop_geom": "designated_tabletop"}
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    task.sim = SimpleNamespace(data=SimpleNamespace(body_xmat=rotation.reshape(1, 9)))
    state = {"contact": True}

    def contact(first, second):
        assert first == ["designated_tabletop"] and second == ["target"]
        return state["contact"]

    task.check_contact = contact
    assert task.hold_steps == control_freq
    assert task._goal_satisfied()
    for _ in range(task.hold_steps - 1):
        task._post_action(None)
        assert not task._check_success()
    # Reward and success queries must not advance the timer.
    for _ in range(3):
        assert task.reward() == 0.0 and not task._check_success()
    state["contact"] = False
    task._post_action(None)
    assert task._success_hold == 0
    state["contact"] = True
    for _ in range(task.hold_steps):
        task._post_action(None)
    assert task._check_success() and task.reward() == 1.0
    # A horizontal candle touching the correct table interrupts success.
    task.sim.data.body_xmat[:] = np.eye(3).reshape(1, 9)
    task._post_action(None)
    assert task._success_hold == 0 and not task._check_success()
    # An upside-down candle also fails.
    rotation[2, 1] = -1.0
    task.sim.data.body_xmat[:] = rotation.reshape(1, 9)
    assert not task._goal_satisfied()
    # Orientation tolerance has a meaningful boundary.
    for angle, expected in ((9.0, True), (11.0, False)):
        theta = math.radians(90.0 - angle)
        task.sim.data.body_xmat[:] = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, math.cos(theta), -math.sin(theta)],
                [0.0, math.sin(theta), math.cos(theta)],
            ]
        ).reshape(1, 9)
        assert task._goal_satisfied() == expected


def _set_candle(env, pos, quat):
    env.sim.data.set_joint_qpos(env._candle_joint, np.array([*pos, *quat]))
    env.sim.data.set_joint_qvel(env._candle_joint, np.zeros(6))
    env.sim.forward()


def _settle_candle_only(env, seconds=0.4):
    """Hold the robot and unrelated objects still during a short target check."""
    model, data = env.sim.model, env.sim.data
    joint = model.joint_name2id(env._candle_joint)
    qadr, dadr = int(model.jnt_qposadr[joint]), int(model.jnt_dofadr[joint])
    fixed_qpos = data.qpos.copy()
    qmask, vmask = np.ones(model.nq, dtype=bool), np.ones(model.nv, dtype=bool)
    qmask[qadr : qadr + 7] = False
    vmask[dadr : dadr + 6] = False
    for _ in range(int(np.ceil(seconds / model.opt.timestep))):
        env.sim.step()
        data.qpos[qmask] = fixed_qpos[qmask]
        data.qvel[vmask] = 0
        env.sim.forward()


def test_headless_reset_stability_and_physical_success(monkeypatch):
    env = robosuite.make(
        env_name=TASK.__name__,
        robots="G1FloatingBody",
        controller_configs=_load_reset_controller(),
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        ignore_done=True,
        hard_reset=False,
        initialization_noise=None,
        control_freq=50,
        scene_ids=[2025],
        seed=0,
    )
    try:
        env.reset()
        assert env.scene_name == "train_2025"
        assert not env._check_success() and not env._goal_satisfied()
        assert env.count_body_collisions() == 0
        assert not getattr(env, "_spawn_cleared_geom_state", {})
        model, data = env.sim.model, env.sim.data
        source = ET.parse(get_holodeck_scene_xml(2025)).getroot()
        # Removal is task-local: the source asset still contains the laptop,
        # while its bodies, collision/visual geoms and joints are absent here.
        laptop_name = "laptop_63fbcd680e0db93dfb0454951f2114f3_1_0_0"
        laptop = source.find(f".//body[@name='{laptop_name}']")
        assert laptop is not None
        for tag, compiled_names in (
            ("body", model.body_names),
            ("geom", model.geom_names),
            ("joint", model.joint_names),
        ):
            for element in laptop.iter(tag):
                assert element.get("name") not in compiled_names
        assert "train_2025" not in HolodeckEnv.HIDDEN_SCENE_BODIES
        # The other candles retain their native compiled free-joint poses.
        for name in (
            "candle_e556573454d609117d1a666932ebfcb2_1_0_0",
            "candle_c6e5d947796037dc5ccc452b1320f772_2_0_0",
        ):
            body = source.find(f".//body[@name='{name}']")
            joint = body.find("joint").get("name")
            qadr = model.jnt_qposadr[model.joint_name2id(joint)]
            native = np.fromstring(body.get("pos") + " " + body.get("quat"), sep=" ")
            native[3:] /= np.linalg.norm(native[3:])
            np.testing.assert_allclose(model.qpos0[qadr : qadr + 7], native, atol=1e-8)
        # Verify the named support is the farther table, rather than its neighbor.
        desk = data.body_xpos[
            model.body_name2id("obja_42f1d3c373784c9996d35cbf08f2dbbb_1_0_0")
        ]
        far = data.body_xpos[model.body_name2id(env.task_spec["table_body"])]
        near = data.body_xpos[
            model.body_name2id("stand_dba5fb19566f0e1caaf78947145fe7fc_2_0_0")
        ]
        assert np.linalg.norm(far[:2] - desk[:2]) > np.linalg.norm(near[:2] - desk[:2])
        tabletop = [env.task_spec["tabletop_geom"]]
        assert env.check_contact(tabletop, env._candle_geoms)
        candle_geom_ids = {model.geom_name2id(name) for name in env._candle_geoms}
        for contact in data.contact:
            if contact.geom1 in candle_geom_ids or contact.geom2 in candle_geom_ids:
                assert env._tabletop_geom_id in (contact.geom1, contact.geom2)
                assert contact.dist > -1e-4
        assert np.linalg.norm(data.get_joint_qvel(env._candle_joint)) == 0
        initial = data.body_xpos[env._candle_body_id].copy()
        _settle_candle_only(env)
        assert np.linalg.norm(data.body_xpos[env._candle_body_id] - initial) < 0.01
        assert env.check_contact(tabletop, env._candle_geoms)
        assert not env._goal_satisfied()
        upright = [0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)]
        pos = [2.26769, 0.531398, 0.84290]
        _set_candle(env, pos, upright)
        _settle_candle_only(env)
        assert env._goal_satisfied()
        # Count through the real predicate, bypassing unrelated controller work.
        monkeypatch.setattr(HolodeckEnv, "_post_action", lambda self, action: None)
        for _ in range(env.hold_steps - 1):
            env._post_action(None)
        assert not env._check_success()
        env._post_action(None)
        assert env._check_success() and env.reward() == 1.0
        for bad_pos, bad_quat in (
            ([2.23, 0.531398, 0.95], upright),  # Suspended above correct table.
            ([3.30, 0.531398, 0.84290], upright),  # Supported by other table.
            (pos, [1.0, 0.0, 0.0, 0.0]),  # Horizontal.
            (pos, [np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0]),  # Inverted.
        ):
            _set_candle(env, bad_pos, bad_quat)
            assert not env._goal_satisfied()
            env._post_action(None)
            assert not env._check_success() and env._success_hold == 0
        meta = env.get_ep_meta()
        assert meta["target_body"] == TASK.TARGET_BODY and meta["hold_seconds"] == 1.0
        assert "farther from the desk" in meta["lang"]
    finally:
        env.close()
