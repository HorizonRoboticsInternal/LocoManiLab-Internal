"""Focused contracts for ordered multi-step iTHOR kitchen tasks."""

from types import SimpleNamespace

import pytest
from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_kitchen import __all__ as KITCHEN_EXPORTS
from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.composite.multi_step_manipulation import (
    IThorKitchenApproachCabinetAndStoreStatue,
    IThorKitchenBowlToSinkThenAppleOnPlate,
    IThorKitchenCloseThenOpenCabinet,
    IThorKitchenOpenThenCloseCabinet,
    IThorKitchenOpenThenCloseDrawer,
    IThorKitchenStoreAppleInFridge,
)
from robocasa.environments.ithor_kitchen.tasks.composite.take_statue_from_cabinet import (
    IThorKitchenTakeStatueFromCabinet,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


WORKFLOWS = (
    IThorKitchenStoreAppleInFridge,
    IThorKitchenOpenThenCloseDrawer,
    IThorKitchenBowlToSinkThenAppleOnPlate,
    IThorKitchenApproachCabinetAndStoreStatue,
)


def test_multi_step_tasks_are_public_and_registered():
    for task_class in WORKFLOWS:
        assert task_class.__name__ in KITCHEN_EXPORTS
        assert REGISTERED_ENVS[task_class.__name__] is task_class
        assert REGISTERED_ITHOR_KITCHEN_ENVS[task_class.__name__] is task_class
        assert task_class.TASK_FAMILY == "composite"


@pytest.mark.parametrize("task_class", WORKFLOWS)
def test_pickup_is_not_an_atomic_success_stage(task_class):
    task = object.__new__(task_class)
    stages = task_class._normalize_atomic_tasks(task._get_atomic_tasks())

    assert all(
        not stage.name.startswith(("pick_", "take_", "lift_")) for stage in stages
    )
    assert all("grasp" not in str(stage.success) for stage in stages)


@pytest.mark.parametrize(
    ("task_class", "expected_scenes"),
    (
        (IThorKitchenStoreAppleInFridge, (13, 24, 25)),
        (IThorKitchenOpenThenCloseDrawer, (13, 24, 25)),
        (IThorKitchenBowlToSinkThenAppleOnPlate, (13, 24)),
        (IThorKitchenApproachCabinetAndStoreStatue, (7, 13)),
    ),
)
def test_multi_step_task_specs_are_explicit_and_complete(task_class, expected_scenes):
    assert task_class.SUPPORTED_SCENE_IDS == expected_scenes
    assert tuple(task_class.SCENE_SPECS) == tuple(
        f"FloorPlan{scene_id}" for scene_id in expected_scenes
    )
    for scene_id in expected_scenes:
        spec = get_ithor_kitchen_task_spec(f"FloorPlan{scene_id}", task_class.SPEC_TASK)
        assert not set(task_class.REQUIRED_SPEC_FIELDS) - set(spec)
        assert spec["hold_seconds"] == pytest.approx(1.0)
        assert len(spec["robot_pose"]["pos"]) == 3
        assert "lang" not in spec
        assert task_class.TASK_LANG


@pytest.mark.parametrize(
    ("task_class", "spec_task", "initial_fraction"),
    (
        (IThorKitchenOpenThenCloseCabinet, "cabinet_open_close_cycle", 0.2),
        (IThorKitchenCloseThenOpenCabinet, "cabinet_close_open_cycle", 0.8),
    ),
)
def test_cabinet_cycles_have_explicit_calibrations_for_every_kitchen_scene(
    task_class, spec_task, initial_fraction
):
    expected_scenes = (6, 7, 13, 24, 25)
    assert task_class.SUPPORTED_SCENE_IDS == expected_scenes
    assert tuple(task_class.SCENE_SPECS) == tuple(
        f"FloorPlan{scene_id}" for scene_id in expected_scenes
    )
    for scene_id in expected_scenes:
        scene = f"FloorPlan{scene_id}"
        workflow = get_ithor_kitchen_task_spec(scene, spec_task)
        opened = get_ithor_kitchen_task_spec(scene, "open_cabinet")
        closed = get_ithor_kitchen_task_spec(scene, "close_cabinet")
        assert (
            workflow["target_fixture"]
            == opened["target_fixture"]
            == closed["target_fixture"]
        )
        assert (
            workflow["target_joint"]
            == opened["target_joint"]
            == closed["target_joint"]
        )
        assert workflow["init_open_fraction"] == pytest.approx(initial_fraction)
        assert workflow["success_open_fraction"] == pytest.approx(0.8)
        assert workflow["success_close_fraction"] == pytest.approx(0.2)


def test_fridge_and_drawer_specs_bind_the_same_joint_for_open_and_close():
    for scene_id in IThorKitchenStoreAppleInFridge.SUPPORTED_SCENE_IDS:
        scene = f"FloorPlan{scene_id}"
        workflow = get_ithor_kitchen_task_spec(scene, "store_apple_in_fridge")
        opened = get_ithor_kitchen_task_spec(scene, "open_fridge")
        closed = get_ithor_kitchen_task_spec(scene, "close_fridge")
        assert workflow["target_fixture"] == opened["target_fixture"]
        assert workflow["target_fixture"] == closed["target_fixture"]
        assert workflow["target_joint"] == opened["target_joint"]
        assert workflow["target_joint"] == closed["target_joint"]
        assert workflow["fridge_receptacle_region"].startswith(
            f"{workflow['target_fixture']}:region:receptacle:"
        )

    for scene_id in IThorKitchenOpenThenCloseDrawer.SUPPORTED_SCENE_IDS:
        scene = f"FloorPlan{scene_id}"
        workflow = get_ithor_kitchen_task_spec(scene, "drawer_open_close_cycle")
        opened = get_ithor_kitchen_task_spec(scene, "open_drawer")
        closed = get_ithor_kitchen_task_spec(scene, "close_drawer")
        assert workflow["target_fixture"] == opened["target_fixture"]
        assert workflow["target_fixture"] == closed["target_fixture"]
        assert workflow["target_joint"] == opened["target_joint"]
        assert workflow["target_joint"] == closed["target_joint"]


def test_later_stage_condition_must_be_recompleted_after_stage_entry():
    task = object.__new__(IThorKitchenBowlToSinkThenAppleOnPlate)
    task._stage_condition_armed = False

    # A plate state produced before this stage cannot be inherited as success.
    assert not task._active_stage_condition(True)
    assert not task._stage_condition_armed
    assert not task._active_stage_condition(False)
    assert task._stage_condition_armed
    assert task._active_stage_condition(True)


def test_object_workflow_success_predicates_never_query_grasp_state():
    fail_on_grasp = lambda *_args, **_kwargs: pytest.fail(
        "picking/grasping must not be a success condition"
    )

    fridge = object.__new__(IThorKitchenStoreAppleInFridge)
    fridge.object_in_region = lambda *_args, **_kwargs: True
    fridge._fixture_contact = lambda *_args, **_kwargs: True
    fridge.object_released = lambda *_args, **_kwargs: True
    fridge.g1_hand_grasping_object = fail_on_grasp
    fridge.fridge_receptacle_region = object()
    fridge.target_fixture = object()
    assert fridge._apple_in_fridge()

    bowl_plate = object.__new__(IThorKitchenBowlToSinkThenAppleOnPlate)
    bowl_plate.object_in_region = lambda *_args, **_kwargs: True
    bowl_plate._fixture_contact = lambda *_args, **_kwargs: True
    bowl_plate.object_released = lambda *_args, **_kwargs: True
    bowl_plate.g1_hand_grasping_object = fail_on_grasp
    bowl_plate.sink_region = object()
    bowl_plate.sink_fixture = object()
    bowl_plate.objects = {"apple": object(), "plate": object()}
    bowl_plate.sim = SimpleNamespace()
    assert bowl_plate._bowl_in_sink()

    statue = object.__new__(IThorKitchenApproachCabinetAndStoreStatue)
    statue.object_in_region = lambda *_args, **_kwargs: True
    statue._fixture_contact = lambda *_args, **_kwargs: True
    statue.object_released = lambda *_args, **_kwargs: True
    statue.g1_hand_grasping_object = fail_on_grasp
    statue.cabinet_receptacle_region = object()
    statue.cabinet_receptacle_fixture = object()
    assert statue._statue_in_cabinet()


def test_final_close_predicates_retain_placed_object(monkeypatch):
    fridge = object.__new__(IThorKitchenStoreAppleInFridge)
    fridge._stage_condition_armed = True
    fridge._articulation_is_closed = lambda: True
    fridge._apple_in_fridge = lambda: True
    assert fridge._fridge_closed_with_apple()
    fridge._apple_in_fridge = lambda: False
    assert not fridge._fridge_closed_with_apple()

    statue = object.__new__(IThorKitchenApproachCabinetAndStoreStatue)
    statue._stage_condition_armed = True
    statue._articulation_is_closed = lambda: True
    statue._statue_in_cabinet = lambda: True
    assert statue._cabinet_closed_with_statue()
    statue._statue_in_cabinet = lambda: False
    assert not statue._cabinet_closed_with_statue()


def test_final_stage_hold_tracks_control_frequency():
    task = object.__new__(IThorKitchenOpenThenCloseDrawer)
    task.scene_name = "FloorPlan13"
    task.control_freq = 20
    task._atomic_tasks = task_class_stages = task._normalize_atomic_tasks(
        task._get_atomic_tasks()
    )

    assert task._required_stage_hold_steps(task_class_stages[0]) == 5
    assert task._required_stage_hold_steps(task_class_stages[-1]) == 20


def test_take_statue_uses_calibrated_regions_and_twenty_percent_close_limit():
    assert IThorKitchenTakeStatueFromCabinet.SUPPORTED_SCENE_IDS == (
        6,
        7,
        13,
        24,
        25,
    )
    for scene_id in IThorKitchenTakeStatueFromCabinet.SUPPORTED_SCENE_IDS:
        spec = get_ithor_kitchen_task_spec(
            f"FloorPlan{scene_id}", "take_statue_from_cabinet"
        )
        assert len(spec["statue_pos"]) == 3
        assert len(spec["success_region"]["center"]) == 3
        assert len(spec["success_region"]["half_size"]) == 3
        assert spec["success_close_fraction"] == pytest.approx(0.20)

    task = object.__new__(IThorKitchenTakeStatueFromCabinet)
    task._get_task_spec = lambda: {"success_close_fraction": 0.20}
    task.sim = object()
    task.target_semantic_joint = SimpleNamespace(fraction=lambda _sim: 0.20)
    assert task._articulation_is_closed()

    task.target_semantic_joint = SimpleNamespace(fraction=lambda _sim: 0.201)
    assert not task._articulation_is_closed()


def test_take_statue_success_region_requires_containment_contact_and_release():
    task = object.__new__(IThorKitchenTakeStatueFromCabinet)
    task.success_region = object()
    task.table_fixture = object()
    state = {"inside": True, "contact": True, "released": True}
    task.object_in_region = lambda *args, **kwargs: state["inside"]
    task._fixture_contact = lambda *args, **kwargs: state["contact"]
    task.object_released = lambda *args, **kwargs: state["released"]

    assert task._statue_in_success_region()
    for failed_condition in state:
        state[failed_condition] = False
        assert not task._statue_in_success_region()
        state[failed_condition] = True

    task._close_active_stage_condition = lambda: True
    assert task._cabinet_closed_with_statue_on_table()
    task._close_active_stage_condition = lambda: False
    assert not task._cabinet_closed_with_statue_on_table()
