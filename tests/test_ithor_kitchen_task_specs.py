"""Regression tests for strict iTHOR Kitchen task calibrations."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from robocasa.environments.ithor_kitchen import (
    REGISTERED_ITHOR_KITCHEN_ENVS,
    IThorKitchenAppleBananaPlateSwap,
    IThorKitchenBananaBasketToTable,
    IThorKitchenBowlToSink,
    IThorKitchenCloseOven,
    IThorKitchenLiftSaltShaker,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    _load_specs,
    get_ithor_kitchen_task_spec,
)


@pytest.mark.parametrize(
    ("scene_name", "target_fixture"),
    (
        ("FloorPlan7", "FloorPlan7:Fridge_7_10"),
        ("FloorPlan13", "FloorPlan13:Fridge_13_9"),
        ("FloorPlan24", "FloorPlan24:Fridge_24_12"),
        ("FloorPlan25", "FloorPlan25:Fridge_25_16"),
    ),
)
def test_approach_fridge_calibrations(scene_name, target_fixture):
    spec = get_ithor_kitchen_task_spec(
        scene_name,
        "approach_fridge",
        required=(
            "target_fixture",
            "robot_pose",
            "target_xy",
            "target_tolerance",
        ),
    )

    assert spec["target_fixture"] == target_fixture
    assert len(spec["robot_pose"]["pos"]) == 3
    assert isinstance(spec["robot_pose"]["yaw"], float)
    assert len(spec["target_xy"]) == 2
    assert len(spec["target_tolerance"]) == 2
    assert all(value >= 0.0 for value in spec["target_tolerance"])
    assert isinstance(spec.get("show_target_region", False), bool)
    assert "initial_pose_jitter" not in spec
    assert "robot_jitter_radius" not in spec
    assert "robot_scale" not in spec


def test_articulation_specs_only_override_relevant_task_defaults():
    specs = _load_specs()["scenes"]
    for tasks in specs.values():
        for task_name, spec in tasks.items():
            if not task_name.startswith(("open_", "close_")):
                continue
            assert "initial_pose_jitter" not in spec
            assert "robot_jitter_radius" not in spec
            assert "hold_steps" not in spec
            assert "robot_scale" not in spec
            assert "closed_qpos" not in spec
            assert "open_qpos" not in spec
            assert "open_init_fraction" not in spec
            assert 0.0 <= spec["init_open_fraction"] <= 1.0
            if task_name.startswith("open_"):
                assert 0.0 <= spec["success_open_fraction"] <= 1.0
                assert "success_close_fraction" not in spec
            else:
                assert 0.0 <= spec["success_close_fraction"] <= 1.0
                assert "success_open_fraction" not in spec


def test_drawer_tasks_use_symmetric_twenty_eighty_thresholds():
    specs = _load_specs()["scenes"]
    for scene_name in (
        "FloorPlan6",
        "FloorPlan7",
        "FloorPlan13",
        "FloorPlan24",
        "FloorPlan25",
    ):
        open_spec = specs[scene_name]["open_drawer"]
        assert open_spec["init_open_fraction"] == pytest.approx(0.20)
        assert open_spec["success_open_fraction"] == pytest.approx(0.80)

        close_spec = specs[scene_name]["close_drawer"]
        assert close_spec["init_open_fraction"] == pytest.approx(0.80)
        assert close_spec["success_close_fraction"] == pytest.approx(0.20)

        drawer_pnp = specs[scene_name]["drawer_pnp"]
        assert drawer_pnp["success_close_fraction"] == pytest.approx(0.20)


def test_close_oven_uses_twenty_to_ten_percent_travel():
    scene_names = (
        "FloorPlan6",
        "FloorPlan7",
        "FloorPlan13",
        "FloorPlan24",
        "FloorPlan25",
    )
    assert IThorKitchenCloseOven.SUPPORTED_SCENE_IDS == (6, 7, 13, 24, 25)
    assert IThorKitchenCloseOven.OPEN_INIT_FRACTION == pytest.approx(0.20)
    assert IThorKitchenCloseOven.SUCCESS_CLOSE_FRACTION == pytest.approx(0.10)

    specs = _load_specs()["scenes"]
    for scene_name in scene_names:
        spec = specs[scene_name]["close_oven"]
        assert spec["init_open_fraction"] == pytest.approx(0.20)
        assert spec["success_close_fraction"] == pytest.approx(0.10)

    task = object.__new__(IThorKitchenCloseOven)
    fraction = {"value": 0.10}
    task.sim = object()
    task._success_joints = lambda: [
        SimpleNamespace(fraction=lambda _sim: fraction["value"])
    ]
    assert task._raw_success()
    fraction["value"] = 0.11
    assert not task._raw_success()


def test_task_specs_have_no_cross_scene_fallback():
    with pytest.raises(KeyError, match="FloorPlan14/approach_fridge"):
        get_ithor_kitchen_task_spec("FloorPlan14", "approach_fridge")


def test_lift_tasks_require_thirty_centimeters_for_one_second():
    specs = _load_specs()["scenes"]
    for scene_name in (
        "FloorPlan6",
        "FloorPlan7",
        "FloorPlan13",
        "FloorPlan24",
        "FloorPlan25",
    ):
        for task_name in ("lift_eggplant", "lift_plate", "lift_salt_shaker"):
            spec = specs[scene_name][task_name]
            assert spec["lift_height"] == pytest.approx(0.30)
            assert spec["hold_seconds"] == pytest.approx(1.0)
        assert "lift_can" not in specs[scene_name]


def test_lift_salt_shaker_replaces_lift_can_registration():
    assert (
        REGISTERED_ITHOR_KITCHEN_ENVS["IThorKitchenLiftSaltShaker"]
        is IThorKitchenLiftSaltShaker
    )
    assert "IThorKitchenLiftCan" not in REGISTERED_ITHOR_KITCHEN_ENVS

    task = object.__new__(IThorKitchenLiftSaltShaker)
    object_cfg = task._get_obj_cfgs()[0]
    model_path = Path(object_cfg["obj_groups"])
    assert model_path.name == "model.xml"
    assert model_path.parent.name == "SaltShaker002"
    assert model_path.is_file()


def test_lift_success_still_requires_one_second_above_threshold():
    task = object.__new__(IThorKitchenLiftSaltShaker)
    task._lift_initial_z = 1.0
    task.min_lift_delta = 0.30
    task.lift_hold_seconds = 1.0
    task.control_freq = 20
    task._lift_hold_steps = 19
    task._target_root_z = lambda: 1.30

    assert task._raw_success()
    assert not task._check_success()
    task._lift_hold_steps = 20
    assert task._check_success()


@pytest.mark.parametrize("banana_first", (False, True))
def test_apple_banana_plate_swap_requires_simultaneous_final_state(
    monkeypatch, banana_first
):
    monkeypatch.setattr(IThorKitchenEnv, "_post_action", lambda self, action: None)
    task = object.__new__(IThorKitchenAppleBananaPlateSwap)
    task.scene_name = "FloorPlan6"
    task.success_hold_seconds = 1.0
    task.control_freq = 20
    task._success_hold_steps = 0
    task.support_region = object()
    task.apple_region_z_tolerance = 0.08
    task.objects = {name: name for name in ("apple", "banana", "plate")}
    task.obj_body_id = {"apple": 0, "banana": 1, "plate": 2}
    task.sim = SimpleNamespace(data=SimpleNamespace(body_xpos=[[0.0, 0.0, 1.0]] * 3))
    contacts = {("apple", "plate")}
    state = {"apple_on_table": False}
    task.check_contact = lambda obj, receptacle: (obj, receptacle) in contacts
    task.object_on_region = lambda *args, **kwargs: state["apple_on_table"]

    def advance():
        for _ in range(20):
            task._post_action(None)

    advance()
    assert not task._check_success()

    if banana_first:
        contacts.add(("banana", "plate"))
    else:
        contacts.remove(("apple", "plate"))
        state["apple_on_table"] = True
    advance()
    assert not task._check_success()

    contacts.discard(("apple", "plate"))
    contacts.add(("banana", "plate"))
    state["apple_on_table"] = True
    for _ in range(19):
        task._post_action(None)
    assert not task._check_success()
    task._post_action(None)
    assert task._check_success()

    # Returning the apple to the plate invalidates even a table-region match.
    contacts.add(("apple", "plate"))
    task._post_action(None)
    assert not task._raw_success()
    assert not task._check_success()
    assert task._success_hold_steps == 0


def test_cup_stack_specs_have_no_table_success_parameters():
    cup_stack_specs = [
        tasks["cup_stack"]
        for tasks in _load_specs()["scenes"].values()
        if "cup_stack" in tasks
    ]
    assert cup_stack_specs
    for spec in cup_stack_specs:
        assert "support_region" not in spec
        assert set(spec) == {
            "hold_seconds",
            "robot_pose",
            "cup_1_pos",
            "cup_2_pos",
        }


@pytest.mark.parametrize(
    "scene_name",
    ("FloorPlan6", "FloorPlan7", "FloorPlan13", "FloorPlan24", "FloorPlan25"),
)
def test_direct_object_calibrations_use_bottom_xyz(scene_name):
    tasks = _load_specs()["scenes"][scene_name]

    bowl = tasks["bowl_to_sink"]
    assert len(bowl["object_pos"]) == 3
    assert "object_xy" not in bowl
    assert "source_bottom_z" not in bowl

    loaded_basket = tasks["banana_basket_to_table"]
    assert len(loaded_basket["basket_pos"]) == 3
    assert len(loaded_basket["banana_pos"]) == 3


def test_direct_object_tasks_share_bottom_xyz_placement_primitive():
    calls = []
    bowl = object.__new__(IThorKitchenBowlToSink)
    bowl.scene_name = "FloorPlan6"
    bowl._get_task_spec = lambda: {"object_pos": [1.0, 2.0, 0.9]}
    bowl._set_task_object_bottom_at = lambda name, xy, bottom_z: calls.append(
        (name, list(xy), float(bottom_z))
    )
    bowl._place_source_object()

    loaded_basket = object.__new__(IThorKitchenBananaBasketToTable)
    loaded_basket.scene_name = "FloorPlan6"
    loaded_basket._get_task_spec = lambda: {
        "basket_pos": [3.0, 4.0, 0.8],
        "banana_pos": [3.0, 4.0, 0.81],
    }
    loaded_basket._set_task_object_bottom_at = lambda name, xy, bottom_z: calls.append(
        (name, list(xy), float(bottom_z))
    )
    loaded_basket._place_task_objects()

    assert calls == [
        ("bowl", [1.0, 2.0], 0.9),
        ("basket", [3.0, 4.0], 0.8),
        ("banana", [3.0, 4.0], 0.81),
    ]


@pytest.mark.parametrize(
    "scene_name", ("FloorPlan7", "FloorPlan13", "FloorPlan24", "FloorPlan25")
)
def test_composite_pick_place_specs_are_complete(scene_name):
    drawer_pnp = get_ithor_kitchen_task_spec(
        scene_name,
        "drawer_pnp",
        required=(
            "target_fixture",
            "target_joint",
            "robot_pose",
            "drawer_receptacle_region",
            "apple_pos",
        ),
    )
    assert len(drawer_pnp["apple_pos"]) == 3
    assert "source_region" not in drawer_pnp

    take_apple = get_ithor_kitchen_task_spec(
        scene_name,
        "open_drawer_take_apple",
        required=(
            "target_fixture",
            "target_joint",
            "robot_pose",
            "drawer_receptacle_region",
            "apple_pos",
            "target_region",
            "show_target_region",
        ),
    )
    assert len(take_apple["target_region"]["center"]) == 3
    assert len(take_apple["target_region"]["half_xy"]) == 2
    assert take_apple["target_region"]["z_tolerance"] >= 0.0
    assert isinstance(take_apple["show_target_region"], bool)

    cabinet_pnp = get_ithor_kitchen_task_spec(
        scene_name,
        "apple_cabinet_pnp",
        required=(
            "target_fixture",
            "target_joint",
            "robot_pose",
            "cabinet_receptacle_region",
            "apple_pos",
        ),
    )
    assert cabinet_pnp["cabinet_receptacle_region"].startswith(scene_name)
    assert ":region:receptacle:" in cabinet_pnp["cabinet_receptacle_region"]
    assert cabinet_pnp["success_close_fraction"] == pytest.approx(0.08)
    assert "success_open_fraction" not in cabinet_pnp
    if scene_name == "FloorPlan7":
        assert cabinet_pnp["target_fixture"] == "FloorPlan7:Cabinet_b8962d2f_14"
        assert cabinet_pnp["cabinet_receptacle_region"] == (
            "FloorPlan7:StandardCounterHeightWidth_d610e478_57:region:receptacle:6"
        )

    swap = get_ithor_kitchen_task_spec(
        scene_name,
        "apple_banana_plate_swap",
        required=(
            "robot_pose",
            "support_region",
            "plate_pos",
            "banana_pos",
        ),
    )
    assert len(swap["support_region"]["center"]) == 3


def test_multistage_navigation_specs_have_complete_language():
    route_names = {
        "oven_then_fridge",
        "fridge_then_oven",
        "fridge_then_dishwasher",
        "dishwasher_then_fridge",
        "oven_then_dishwasher",
        "dishwasher_then_oven",
    }
    found = set()

    for tasks in _load_specs()["scenes"].values():
        for task_name in route_names & tasks.keys():
            first, second = task_name.split("_then_")
            spec = tasks[task_name]

            assert set(spec) == {"stages", "robot_pose", "lang"}
            assert spec["lang"] == (f"walk to the {first}, then walk to the {second}")
            found.add(task_name)

    assert found == route_names


def test_floorplan24_routes_use_white_oven_and_face_first_fixture():
    oven_first = get_ithor_kitchen_task_spec(
        "FloorPlan24",
        "oven_then_fridge",
    )
    fridge_first = get_ithor_kitchen_task_spec(
        "FloorPlan24",
        "fridge_then_oven",
    )

    assert oven_first["stages"] == [
        "FloorPlan24:oven_FP24_0",
        "FloorPlan24:Fridge_24_12",
    ]
    assert oven_first["robot_pose"]["yaw"] == pytest.approx(-0.5544493430401705)
    assert fridge_first["stages"] == [
        "FloorPlan24:Fridge_24_12",
        "FloorPlan24:oven_FP24_0",
    ]
    assert fridge_first["robot_pose"]["pos"] == [-1.0, 3.1, 0.0]
    assert fridge_first["robot_pose"]["yaw"] == pytest.approx(-2.571045336646057)


def test_kitchen_scene_specs_are_in_numeric_id_order():
    assert list(_load_specs()["scenes"]) == [
        "FloorPlan6",
        "FloorPlan7",
        "FloorPlan13",
        "FloorPlan24",
        "FloorPlan25",
    ]


def test_task_spec_returns_a_deep_copy():
    first = get_ithor_kitchen_task_spec("FloorPlan13", "approach_fridge")
    first["robot_pose"]["pos"][0] = 123.0

    second = get_ithor_kitchen_task_spec("FloorPlan13", "approach_fridge")
    assert second["robot_pose"]["pos"][0] == pytest.approx(-2.4525105172645785)
