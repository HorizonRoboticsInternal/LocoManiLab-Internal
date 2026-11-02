"""The desk-laptop composites must not inherit bedside-table placement."""

import pytest

from robocasa.environments.ithor_bedroom.tasks.articulation_composites import (
    IThorBedroomApproachDeskAndCloseLaptop,
    IThorBedroomApproachDeskAndOpenLaptop,
)
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
)


@pytest.mark.parametrize(
    "task_class",
    (IThorBedroomApproachDeskAndOpenLaptop, IThorBedroomApproachDeskAndCloseLaptop),
)
@pytest.mark.parametrize("scene_id", (302, 323))
def test_approach_laptop_uses_main_desk_only_in_affected_scenes(task_class, scene_id):
    scene_name = f"FloorPlan{scene_id}"
    task = object.__new__(task_class)
    task.scene_name = scene_name
    desk = get_ithor_bedroom_task_spec(scene_name, "grasp_workspace")
    workspace = task._workspace_spec()

    assert (workspace["fixture"], workspace["region"]) == (
        desk["fixture"],
        desk["region"],
    )
    assert workspace["laptop_seating"] == desk["seating"]
    assert workspace["remove_fixtures"] == [
        item
        for item in desk["remove_fixtures"]
        if item != get_ithor_bedroom_task_spec(scene_name, "laptop")["fixture"]
    ]
    assert (
        workspace["fixture"]
        != get_ithor_bedroom_task_spec(scene_name, "workspace")["fixture"]
    )


@pytest.mark.parametrize(
    "task_class",
    (IThorBedroomApproachDeskAndOpenLaptop, IThorBedroomApproachDeskAndCloseLaptop),
)
def test_approach_laptop_other_scenes_use_calibrated_work_surface(task_class):
    for scene_id in (306, 310, 329):
        scene_name = f"FloorPlan{scene_id}"
        task = object.__new__(task_class)
        task.scene_name = scene_name
        expected = get_ithor_bedroom_task_spec(scene_name, "workspace")
        laptop = task_class.SCENE_SPECS[scene_name]
        if "surface_fixture" in laptop:
            expected["fixture"] = laptop["surface_fixture"]
            expected["region"] = laptop["surface_region"]
        if "replace_fixture" in laptop:
            expected["remove_fixtures"] = [
                item
                for item in expected.get("remove_fixtures", ())
                if item != laptop["replace_fixture"]
            ]
        assert task._workspace_spec() == expected


@pytest.mark.parametrize(
    "task_class",
    (IThorBedroomApproachDeskAndOpenLaptop, IThorBedroomApproachDeskAndCloseLaptop),
)
@pytest.mark.parametrize("scene_id", (302, 306, 310, 323))
def test_approach_replaces_only_native_laptop_with_329_asset(task_class, scene_id):
    scene_name = f"FloorPlan{scene_id}"
    spec = task_class.SCENE_SPECS[scene_name]
    original = get_ithor_bedroom_task_spec(scene_name, "laptop")
    reference = get_ithor_bedroom_task_spec("FloorPlan329", "laptop")
    task = object.__new__(task_class)
    task.scene_name = scene_name
    workspace = task._workspace_spec()

    assert spec["fixture_mode"] == "spawn_asset"
    assert spec["replace_fixture"] == original["fixture"]
    assert spec["asset"] == reference["asset"]
    assert spec["asset_id"] == reference["asset_id"]
    assert spec["spawn_quat"] == reference["spawn_quat"]
    assert spec["spawn_scale"] == reference["spawn_scale"]
    assert spec["fixture"] == f"{scene_name}:spawned:{spec['spawn_name']}"
    assert spec["joint"] == f"{spec['spawn_name']}_Laptop_13_laptop_lid_13_joint_0"
    assert (spec["surface_fixture"], spec["surface_region"]) == (
        workspace["fixture"],
        workspace["region"],
    )
    assert spec["replace_fixture"] not in workspace.get("remove_fixtures", ())


@pytest.mark.parametrize(
    "task_class",
    (IThorBedroomApproachDeskAndOpenLaptop, IThorBedroomApproachDeskAndCloseLaptop),
)
def test_approach_329_laptop_spec_is_unchanged(task_class):
    spec = task_class.SCENE_SPECS["FloorPlan329"]
    reference = get_ithor_bedroom_task_spec("FloorPlan329", "laptop")
    assert spec["fixture_mode"] == reference["fixture_mode"]
    assert "replace_fixture" not in spec
    for key in (
        "asset",
        "asset_id",
        "spawn_name",
        "spawn_root_pos",
        "spawn_quat",
        "joint",
    ):
        assert spec[key] == reference[key]
