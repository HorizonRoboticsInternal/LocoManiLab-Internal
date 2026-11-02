"""Scene 315 registration, ordered stages, hold timing and goal counterexamples."""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_bedroom.tasks.scene_315_composites import (
    IThorBedroomApproachDeskAndPutMugThenCanInBasket as Pack,
    IThorBedroomApproachTrashCanAndPutCanInTrashThenLiftBasket as Trash,
)
from robocasa.environments.ithor_bedroom.tasks.trash_can import (
    IThorBedroomCanIntoTrashCan as AtomicTrash,
)
from robocasa.environments.ithor_task_catalog import (
    BEDROOM_PACKAGE,
    discover_task_catalog,
    task_counts,
    validate_task_catalog,
)


def test_catalog_and_isolated_scene_specs():
    entries = discover_task_catalog([BEDROOM_PACKAGE])
    assert validate_task_catalog(entries) == []
    assert task_counts(entries)[BEDROOM_PACKAGE][315] == {"atomic": 2, "composite": 2}
    for cls in (Trash, Pack):
        assert REGISTERED_ENVS[cls.__name__] is cls
        assert cls.SUPPORTED_SCENE_IDS == (315,)
        assert not issubclass(cls, AtomicTrash)
        with pytest.raises(ValueError, match="uncalibrated scene ids"):
            cls(scene_ids=[329])
    baseline = deepcopy(AtomicTrash.SCENE_SPECS)
    spec = Trash.SCENE_SPECS["FloorPlan315"]
    assert spec["bin_bottom_geoms"] == baseline["FloorPlan315"]["bin_bottom_geoms"]
    assert (
        spec["bin_bottom_geoms"]
        is not AtomicTrash.SCENE_SPECS["FloorPlan315"]["bin_bottom_geoms"]
    )
    assert Trash.OBJECT_ASSETS["can"][1] == AtomicTrash.SOURCE_SCALE
    assert AtomicTrash.SCENE_SPECS == baseline


def sequence(cls, freq=20):
    env = object.__new__(cls)
    env.control_freq = freq
    env.PRINT_STAGE_MESSAGES = False
    env._configure_success_hold_duration()
    env._reset_task_progress()
    state = dict(
        near=False,
        unchanged=True,
        can_waits=True,
        bottom=False,
        mug=False,
        can=False,
        grasp=False,
        height=1.0,
    )
    env.bin_fixture = object()
    env.work_surface_fixture = object()
    env._near = lambda fixture: state["near"]
    env._unchanged = lambda names: state["unchanged"] and (
        state["can_waits"] if "can" in names else True
    )
    env._can_contacts_trash_can_bottom = lambda: state["bottom"]
    env._placed = lambda name: state[name]
    env.object_pos = lambda name: np.array([0.0, 0.0, state["height"]])
    env.g1_hand_grasping_object = lambda name: state["grasp"]
    env._lift_initial_z = 1.0
    return env, state


def tick(env, count=1):
    with patch.object(IThorBedroomEnv, "_post_action", return_value=(0, False, {})):
        for _ in range(count):
            env._post_action(None)


@pytest.mark.parametrize("freq", [10, 20, 30])
def test_pack_requires_navigation_mug_then_can_and_continuous_final_hold(freq):
    env, state = sequence(Pack, freq)
    state.update(mug=True, can=True, unchanged=False)
    tick(env, 30)
    assert env._sequence_stage == 0
    state.update(near=True, unchanged=True, mug=False, can=False)
    tick(env, 5)
    assert env._sequence_stage == 1
    state.update(can=True, can_waits=False)
    tick(env, 10)
    assert env._sequence_stage == 1
    state.update(mug=True)
    tick(env, 10)
    assert env._sequence_stage == 1  # Can placed early must return to the desk.
    state.update(can=False, can_waits=True)
    tick(env, 5)
    assert env._sequence_stage == 2
    state.update(can=True)
    tick(env, freq - 1)
    assert not env._check_success()
    state.update(mug=False)
    tick(env)
    assert env._sequence_hold == 0
    state.update(mug=True)
    tick(env, freq)
    assert env._check_success()
    hold = env._sequence_hold
    for _ in range(20):
        assert env._check_success()
    assert env._sequence_hold == hold
    env._reset_task_progress()
    assert env._sequence_stage == 0 and not env._check_success()


@pytest.mark.parametrize("freq", [10, 20, 30])
def test_trash_requires_30cm_lift_with_can_retained_without_grasp(freq, capsys):
    env, state = sequence(Trash, freq)
    state.update(bottom=True, grasp=True, height=1.4, unchanged=False)
    tick(env, 30)
    assert env._sequence_stage == 0
    state.update(near=True, bottom=False, height=1.0, grasp=False, unchanged=True)
    tick(env, 5)
    assert env._sequence_stage == 1
    state.update(bottom=True, unchanged=False)
    tick(env, 10)
    assert env._sequence_stage == 1  # Basket must wait while disposing the can.
    state.update(unchanged=True)
    tick(env, 5)
    assert env._sequence_stage == 2
    # No low-height grasp was recorded, and grasp detection must not be called.
    env.g1_hand_grasping_object = lambda name: pytest.fail("unexpected grasp check")
    state.update(grasp=False)
    state.update(height=1.299)
    tick(env, freq)
    assert not env._check_success()
    state.update(height=1.3)
    tick(env, freq - 1)
    assert not env._check_success()
    state.update(bottom=False)
    tick(env)
    assert env._sequence_hold == 0
    state.update(bottom=True)
    env.PRINT_STAGE_MESSAGES = True
    tick(env, freq)
    assert env._check_success()
    assert "stage 3/3" in capsys.readouterr().out
    tick(env)
    assert env._check_success()  # No grasp is required to maintain success.
    state.update(height=1.299)
    tick(env)
    assert not env._check_success()
    env._reset_task_progress()
    assert env._sequence_stage == 0 and not env._check_success()


@pytest.mark.parametrize("missing", ["support", "inside", "released", None])
def test_placement_requires_supported_basket_interior_contact_and_release(missing):
    env = object.__new__(Pack)
    env.work_surface_region = object()
    env._receptacle_upright_released_on_region = lambda *args: missing != "support"
    env._object_in_named_receptacle = lambda *args, **kwargs: missing != "inside"
    env.object_released = lambda *args, **kwargs: missing != "released"
    assert env._placed("mug") is (missing is None)
    assert env._placed("can") is (missing is None)
