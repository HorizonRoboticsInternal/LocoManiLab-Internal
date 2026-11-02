"""Ordered sofa, table, and bin navigation for iTHOR living rooms."""

from robocasa.environments.ithor_composite.navigation import (
    MultiStageNavigationMixin,
)
from robocasa.environments.ithor_living_room.tasks.composite.base import (
    IThorLivingRoomMultiStageComposite,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_specs,
)


_SUPPORTED_SCENE_IDS = (202, 207, 209, 212, 226)
_REQUIRED_SPEC_FIELDS = ("stages", "robot_pose")


class IThorLivingRoomSofaThenTable(
    MultiStageNavigationMixin,
    IThorLivingRoomMultiStageComposite,
):
    SPEC_TASK = "sofa_then_table"
    TASK_LANG = 'walk to the sofa, then walk to the table'
    STAGE_NAMES = ("sofa", "table")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS
    SCENE_SPECS = get_ithor_livingroom_task_specs(
        SPEC_TASK, required=_REQUIRED_SPEC_FIELDS
    )


class IThorLivingRoomTableThenSofa(
    MultiStageNavigationMixin,
    IThorLivingRoomMultiStageComposite,
):
    SPEC_TASK = "table_then_sofa"
    TASK_LANG = 'walk to the table, then walk to the sofa'
    STAGE_NAMES = ("table", "sofa")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS
    SCENE_SPECS = get_ithor_livingroom_task_specs(
        SPEC_TASK, required=_REQUIRED_SPEC_FIELDS
    )


class IThorLivingRoomBinThenTable(
    MultiStageNavigationMixin,
    IThorLivingRoomMultiStageComposite,
):
    SPEC_TASK = "bin_then_table"
    TASK_LANG = 'walk to the trash bin, then walk to the table'
    STAGE_NAMES = ("bin", "table")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS
    SCENE_SPECS = get_ithor_livingroom_task_specs(
        SPEC_TASK, required=_REQUIRED_SPEC_FIELDS
    )


class IThorLivingRoomBinThenSofa(
    MultiStageNavigationMixin,
    IThorLivingRoomMultiStageComposite,
):
    SPEC_TASK = "bin_then_sofa"
    TASK_LANG = 'walk to the trash bin, then walk to the sofa'
    STAGE_NAMES = ("bin", "sofa")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS
    SCENE_SPECS = get_ithor_livingroom_task_specs(
        SPEC_TASK, required=_REQUIRED_SPEC_FIELDS
    )


class IThorLivingRoomTableThenBin(
    MultiStageNavigationMixin,
    IThorLivingRoomMultiStageComposite,
):
    SPEC_TASK = "table_then_bin"
    TASK_LANG = 'walk to the table, then walk to the trash bin'
    STAGE_NAMES = ("table", "bin")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS
    SCENE_SPECS = get_ithor_livingroom_task_specs(
        SPEC_TASK, required=_REQUIRED_SPEC_FIELDS
    )


class IThorLivingRoomSofaThenBin(
    MultiStageNavigationMixin,
    IThorLivingRoomMultiStageComposite,
):
    SPEC_TASK = "sofa_then_bin"
    TASK_LANG = 'walk to the sofa, then walk to the trash bin'
    STAGE_NAMES = ("sofa", "bin")
    SUPPORTED_SCENE_IDS = _SUPPORTED_SCENE_IDS
    SCENE_SPECS = get_ithor_livingroom_task_specs(
        SPEC_TASK, required=_REQUIRED_SPEC_FIELDS
    )


__all__ = [
    "IThorLivingRoomBinThenSofa",
    "IThorLivingRoomBinThenTable",
    "IThorLivingRoomSofaThenTable",
    "IThorLivingRoomSofaThenBin",
    "IThorLivingRoomTableThenSofa",
    "IThorLivingRoomTableThenBin",
]
