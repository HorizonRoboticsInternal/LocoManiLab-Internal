"""Scene 301 ordering, geometric counterexamples, and catalog isolation."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_bedroom.tasks.bedside_composites import (
    IThorBedroomApproachBedStandPillowThenDesk as PillowRoutine,
    IThorBedroomOpenBoxThenStatueFromSofaIntoBox as StatueRoutine,
)
from robocasa.environments.ithor_bedroom.tasks.bedside_objects import (
    IThorBedroomStandPillowAtHeadboard as PillowTask,
    IThorBedroomLiftStatueFromSofa as StatueTask,
)
from robocasa.environments.ithor_task_catalog import (
    BEDROOM_PACKAGE,
    discover_task_catalog,
    task_counts,
    validate_task_catalog,
)


def test_catalog_and_native_specs_are_isolated():
    assert "IThorBedroomLiftBasketballFromSofa" not in REGISTERED_ENVS
    assert "IThorBedroomOpenBoxThenBasketballFromSofaIntoBox" not in REGISTERED_ENVS
    entries = discover_task_catalog([BEDROOM_PACKAGE])
    assert task_counts(entries)[BEDROOM_PACKAGE][301] == {"atomic": 2, "composite": 2}
    assert not [
        issue
        for issue in validate_task_catalog(entries)
        if ": task matrix changed" not in issue
    ]
    assert PillowRoutine.PILLOW_SCALE == PillowTask.PILLOW_SCALE
    assert (
        PillowRoutine._setup_baked_scene_fixtures
        is PillowTask._setup_baked_scene_fixtures
    )
    for routine, atomic in ((PillowRoutine, PillowTask), (StatueRoutine, StatueTask)):
        assert routine.SUPPORTED_SCENE_IDS == (301,)
        assert REGISTERED_ENVS[routine.__name__] is routine
        assert sum(entry.cls is routine for entry in entries) == 1
        assert not issubclass(routine, atomic)
        original = deepcopy(atomic.SCENE_SPECS)
        spec = routine.SCENE_SPECS["FloorPlan301"]
        assert spec["source_xy"] == original["FloorPlan301"]["source_xy"]
        assert spec["source_quat"] == original["FloorPlan301"]["source_quat"]
        assert spec["source_xy"] is not atomic.SCENE_SPECS["FloorPlan301"]["source_xy"]
        assert routine.TARGET_STABLE_ID == atomic.TARGET_STABLE_ID
        with pytest.raises(ValueError, match="uncalibrated scene ids"):
            routine(scene_ids=[329])


def sequence(cls, freq=20):
    env = object.__new__(cls)
    env.PRINT_STAGE_MESSAGES = False
    env.control_freq = freq
    env._configure_success_hold_duration()
    env._reset_sequence()
    return env


def tick(env, count=1):
    with patch.object(IThorBedroomEnv, "_post_action", return_value=(0, False, {})):
        for _ in range(count):
            env._post_action(None)


def pillow_sequence():
    env = sequence(PillowRoutine)
    state = dict(
        near_bed=False, upright=False, near_desk=False, xy=np.array([1.3, 0.9])
    )
    env.source_fixture = object()
    env.desk_fixture = object()
    env._bed_arrival_xy = None
    env._target_state_reached = lambda: state["upright"]
    env._robot_base_xy = lambda: state["xy"]
    env._near_fixture = lambda fixture: state[
        "near_bed" if fixture is env.source_fixture else "near_desk"
    ]
    return env, state


def test_pillow_requires_bed_then_new_reorientation_then_desk():
    env, state = pillow_sequence()
    state.update(upright=True, near_desk=True)
    tick(env, 30)
    assert env._sequence_stage == 0 and not env._check_success()
    state.update(
        near_bed=True, upright=False, near_desk=False, xy=np.array([0.3, 1.48])
    )
    tick(env, 5)
    assert env._sequence_stage == 1
    state.update(near_bed=False, upright=True, near_desk=True)
    tick(env, 10)
    assert env._sequence_stage == 1  # Reorientation must occur at the bed.
    state.update(near_bed=True, near_desk=False)
    tick(env, 5)
    assert env._sequence_stage == 2
    state.update(near_bed=False, near_desk=True, xy=np.array([2.0, -0.3]))
    tick(env, 19)
    assert not env._check_success()
    state["upright"] = False
    tick(env)
    assert env._sequence_hold == 0
    state["upright"] = True
    tick(env, 20)
    assert env._check_success()
    # Queries are read-only; reset discards all completed stages.
    for _ in range(30):
        assert env._check_success()
    env._reset_sequence()
    assert not env._check_success() and env._sequence_stage == 0


@pytest.mark.parametrize("freq", [10, 20, 30])
def test_box_requires_open_then_lift_then_released_placement(freq):
    env = sequence(StatueRoutine, freq)
    state = dict(open=False, on_sofa=True, lifted=False, inside=False)
    env._box_open = lambda: state["open"]
    env._statue_on_sofa = lambda: state["on_sofa"]
    env._target_state_reached = lambda: state["lifted"]
    env._statue_inside_box = lambda: state["inside"]
    state.update(lifted=True, on_sofa=False, inside=True)
    tick(env, 30)
    state["open"] = True
    tick(env, 5)
    assert env._sequence_stage == 0  # Taking statue before opening earns no credit.
    state.update(on_sofa=True, lifted=False, inside=False)
    tick(env, 5)
    assert env._sequence_stage == 1
    state.update(on_sofa=False, inside=True)
    tick(env, freq)
    assert env._sequence_stage == 1  # Direct placement cannot skip the lift.
    state.update(lifted=True, inside=False)
    tick(env, 5)
    assert env._sequence_stage == 2
    state.update(lifted=False, inside=True)
    tick(env, freq - 1)
    assert not env._check_success()
    state["open"] = False
    tick(env)
    assert env._sequence_hold == 0
    state["open"] = True
    tick(env, freq)
    assert env._check_success()


def test_future_goal_already_complete_must_be_undone():
    env = sequence(StatueRoutine)
    goals = [True, True, True]
    env._sequence_goal = lambda stage: goals[stage]
    tick(env, 50)
    assert env._sequence_stage == 1 and not env._check_success()
    goals[1] = False
    tick(env)
    goals[1] = True
    tick(env, 5)
    assert env._sequence_stage == 2
    tick(env, 30)
    assert not env._check_success()


def box_geometry():
    env = object.__new__(StatueRoutine)
    env.target_fixture = object()
    env.box_fixture = SimpleNamespace(body_names=["box"])
    env._box_site_id = 0
    env._box_floor_id = 1
    # Mock a box with Z as its vertical site axis, testing frame independence.
    env.sim = SimpleNamespace(
        model=SimpleNamespace(
            site_size=np.array([[0.20, 0.20, 0.25]]),
            geom_size=np.array([[0.12, 0.12, 0.12], [0.21, 0.21, 0.01]]),
            geom_bodyid=np.array([1, 2]),
        ),
        data=SimpleNamespace(
            site_xpos=np.array([[0.0, 0.0, 0.25]]),
            site_xmat=np.array([np.eye(3).ravel()]),
            geom_xpos=np.array([[0.0, 0.0, 0.13], [0.0, 0.0, 0.0]]),
            geom_xmat=np.array([np.eye(3).ravel(), np.eye(3).ravel()]),
            contact=[SimpleNamespace(geom1=0, geom2=1, dist=0.0)],
            ncon=1,
        ),
    )
    points = np.array(
        [[x, y, z] for x in (-0.12, 0.12) for y in (-0.12, 0.12) for z in (0.01, 0.25)]
    )
    env._statue_collision_bounds = lambda: points
    env._hands_touch_fixture = lambda fixture: False
    env._fixture_has_illegal_penetration = lambda *args: False
    env._fixture_body_ids = lambda fixture: {1}
    return env, points


def test_box_containment_requires_full_bounds_floor_contact_and_release():
    env, points = box_geometry()
    assert env._statue_inside_box()
    points[:, 0] += 0.12
    assert not env._statue_inside_box()  # Center inside, statue straddles side wall.
    env, points = box_geometry()
    points[:, 2] += 0.30
    assert not env._statue_inside_box()  # Hovering above the rim.
    env, points = box_geometry()
    points[:, 2] -= 0.03
    assert not env._statue_inside_box()  # Through the bottom.
    env, _ = box_geometry()
    env.sim.data.ncon = 0
    assert not env._statue_inside_box()  # Floating inside.
    env.sim.data.ncon = 1
    env._hands_touch_fixture = lambda fixture: True
    assert not env._statue_inside_box()  # Still held.
    env._hands_touch_fixture = lambda fixture: False
    env.sim.data.contact[0].dist = -0.01
    assert not env._statue_inside_box()  # Penetrating floor contact.
    env.sim.data.contact[0].dist = 0.0
    env._fixture_has_illegal_penetration = lambda *args: True
    assert not env._statue_inside_box()


def test_statue_box_bounds_follow_collider_orientation():
    env = object.__new__(StatueRoutine)
    env._statue_geom_ids = [0]
    env._box_site_id = 0
    rotation = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    env.sim = SimpleNamespace(
        model=SimpleNamespace(geom_size=np.array([[0.02, 0.10, 0.03]])),
        data=SimpleNamespace(
            geom_xpos=np.array([[1.0, 2.0, 3.0]]),
            geom_xmat=np.array([rotation.ravel()]),
            site_xmat=np.array([np.eye(3).ravel()]),
        ),
    )
    points = env._statue_collision_bounds()
    assert np.allclose(np.ptp(points, axis=0), [0.06, 0.20, 0.04])
    assert np.allclose((points.min(axis=0) + points.max(axis=0)) / 2, [1.0, 2.0, 3.0])
