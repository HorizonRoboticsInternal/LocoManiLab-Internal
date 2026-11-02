"""Contracts for ordered iTHOR living-room navigation tasks."""

from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_living_room.ithor_living_room import (
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robocasa.environments.ithor_living_room.tasks.composite.base import (
    IThorLivingRoomMultiStageComposite,
)
from robocasa.environments.ithor_living_room.tasks.composite.multi_stage_navigation import (
    IThorLivingRoomBinThenSofa,
    IThorLivingRoomBinThenTable,
    IThorLivingRoomSofaThenTable,
    IThorLivingRoomSofaThenBin,
    IThorLivingRoomTableThenSofa,
    IThorLivingRoomTableThenBin,
)


TASKS = (
    (IThorLivingRoomSofaThenTable, ("sofa", "table")),
    (IThorLivingRoomTableThenSofa, ("table", "sofa")),
    (IThorLivingRoomBinThenTable, ("bin", "table")),
    (IThorLivingRoomBinThenSofa, ("bin", "sofa")),
    (IThorLivingRoomTableThenBin, ("table", "bin")),
    (IThorLivingRoomSofaThenBin, ("sofa", "bin")),
)


def test_living_room_composite_base_is_not_registered():
    name = IThorLivingRoomMultiStageComposite.__name__

    assert name not in REGISTERED_ENVS
    assert name not in REGISTERED_ITHOR_LIVING_ROOM_ENVS


def test_living_room_navigation_tasks_have_complete_ordered_specs():
    expected_scenes = {
        "FloorPlan202",
        "FloorPlan207",
        "FloorPlan209",
        "FloorPlan212",
        "FloorPlan226",
    }

    for task_class, stage_names in TASKS:
        assert task_class.STAGE_NAMES == stage_names
        assert set(task_class.SCENE_SPECS) == expected_scenes
        for spec in task_class.SCENE_SPECS.values():
            assert len(spec["stages"]) == 2
            assert all(
                stage_name in fixture_id.lower()
                for stage_name, fixture_id in zip(stage_names, spec["stages"])
            )
            assert spec["robot_pose"]["pos"][2] == 0.0
            assert "then" in spec["lang"]


def test_living_room_navigation_uses_only_new_public_names():
    for task_class, _stage_names in TASKS:
        assert task_class.__name__ in REGISTERED_ENVS
        assert task_class.__name__ in REGISTERED_ITHOR_LIVING_ROOM_ENVS

    old_name = "IThorLivingRoomWalkToTableThenSofa"
    assert old_name not in REGISTERED_ENVS
    assert old_name not in REGISTERED_ITHOR_LIVING_ROOM_ENVS
    assert "TwoStageNavigation" not in REGISTERED_ENVS
