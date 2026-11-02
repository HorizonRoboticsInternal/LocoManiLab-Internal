"""Scene 401 sponge placement depends only on sustained sink-bottom contact."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import robosuite

from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.environments.ithor_bathroom.tasks.sink_tasks import (
    IThorBathroomPutSpongeInSink as Task,
)
from validate_ithor_tasks import _load_reset_controller


BOTTOM_GEOM = "sink_d516d145d37ab002f8a700bafd85476c_1_1_0_collision_32"


def _contact_task():
    task = object.__new__(Task)
    task.objects = {"sponge": SimpleNamespace(contact_geoms=["sponge_collision"])}
    task.sink_bottom_geoms = (BOTTOM_GEOM,)
    names = ["sponge_collision", BOTTOM_GEOM, "sink_rim", "sink_slope", "hand", "other"]
    task.sim = SimpleNamespace(
        model=SimpleNamespace(geom_id2name=names.__getitem__),
        data=SimpleNamespace(ncon=0, contact=[]),
    )

    def unexpected_constraint(*args, **kwargs):
        raise AssertionError("Success must depend only on sponge/bottom contact")

    task.object_bbox_points = unexpected_constraint
    task._object_fixture_contact_is_shallow = unexpected_constraint
    task.object_released = unexpected_constraint
    task.g1_hand_object_contact = unexpected_constraint
    return task


def _set_contacts(task, pairs):
    task.sim.data.contact = [
        SimpleNamespace(geom1=geom1, geom2=geom2) for geom1, geom2 in pairs
    ]
    task.sim.data.ncon = len(pairs)


@pytest.mark.parametrize(
    "pairs, expected",
    [
        ([], False),
        ([(0, 2)], False),  # Rim contact.
        ([(0, 3)], False),  # Sloping wall contact.
        ([(5, 1)], False),  # Another object contacts the bottom.
        ([(0, 4)], False),  # Hand contact alone.
        ([(0, 1)], True),
        ([(1, 0)], True),  # Contact pair ordering is irrelevant.
        ([(0, 1), (0, 4)], True),  # Holding the sponge does not prevent success.
    ],
)
def test_only_sponge_bottom_contact_satisfies_raw_success(pairs, expected):
    task = _contact_task()
    _set_contacts(task, pairs)
    assert task._raw_success() is expected


@pytest.mark.parametrize("control_freq", (10, 20))
def test_one_second_contact_hold_resets_on_interruption(monkeypatch, control_freq):
    task = _contact_task()
    task.control_freq = control_freq
    task._configure_success_hold_duration()
    task._success_hold_steps = 0
    monkeypatch.setattr(
        IThorBathroomEnv,
        "_post_action",
        lambda self, action: (self.reward(action), False, {}),
    )
    assert task.HOLD_STEPS == control_freq
    _set_contacts(task, [(0, 1)])
    for _ in range(control_freq - 1):
        assert task._post_action(None)[0] == 0.0
    assert not task._check_success()

    _set_contacts(task, [])
    assert task._post_action(None)[0] == 0.0
    assert task._success_hold_steps == 0

    _set_contacts(task, [(0, 1), (0, 4)])
    for _ in range(control_freq - 1):
        assert task._post_action(None)[0] == 0.0
    assert task._post_action(None)[0] == 1.0
    assert task._check_success()

    _set_contacts(task, [(0, 2)])
    assert task._post_action(None)[0] == 0.0
    assert task._success_hold_steps == 0
    assert not task._check_success()


def test_401_physical_bottom_contact_and_rim_counterexample():
    assert Task.SUPPORTED_SCENE_IDS == (401,)
    assert Task.SCENE_SPECS["FloorPlan401"]["sink_bottom_geoms"] == [BOTTOM_GEOM]
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
        assert env.check_contact(
            env.objects["sponge"],
            "sink_d516d145d37ab002f8a700bafd85476c_1_1_0_collision_8",
        )
        assert not env._raw_success()
        assert not env._check_success()
        np.testing.assert_allclose(env._robot_base_xy(), [-2.40, 1.531], atol=0.005)

        geom_id = env.sim.model.geom_name2id(BOTTOM_GEOM)
        assert int(env.sim.model.geom_bodyid[geom_id]) in env._fixture_body_ids(
            env.sink_fixture, include_siblings=True
        )
        center = np.asarray(env.sim.data.geom_xpos[geom_id]).copy()
        rotation = np.asarray(env.sim.data.geom_xmat[geom_id]).reshape(3, 3)
        half_size = np.abs(rotation) @ env.sim.model.geom_size[geom_id]
        center[2] += half_size[2]
        bottom_region = replace(
            env.source_region,
            stable_id=f"{env.sink_fixture.stable_id}:region:support:sink_bottom",
            _world_center=center,
            half_size=np.array([half_size[0], half_size[1], 0.01]),
        )
        # Keep the positive example clear of the raised central drain.
        assert env.place_object_on_support_contact(
            "sponge",
            bottom_region,
            fixture=env.sink_fixture,
            xy=center[:2] + [-0.08, 0.0],
            inward_fractions=(0.0,),
            probe_above=0.01,
            probe_below=0.02,
            probe_step=0.001,
        )
        assert env._raw_success()
        for _ in range(env.HOLD_STEPS - 1):
            assert env._post_action(None)[0] == 0.0
        assert not env._check_success()
        assert env._post_action(None)[0] == 1.0
        assert env._check_success()

        env._set_task_object_bottom_at("sponge", center[:2], center[2] + 0.10)
        assert not env._raw_success()
        env._post_action(None)
        assert not env._check_success()
        assert env._success_hold_steps == 0
        meta = env.get_ep_meta()
        assert meta["target_contact_geoms"] == [BOTTOM_GEOM]
        assert meta["success_requires_release"] is False
    finally:
        env.close()
