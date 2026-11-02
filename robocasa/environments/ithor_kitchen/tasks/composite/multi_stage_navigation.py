"""Ordered multi-stage appliance navigation for iTHOR kitchens."""

from robocasa.environments.ithor_composite.navigation import (
    MultiStageNavigationMixin,
)
from robocasa.environments.ithor_kitchen.tasks.composite.base import (
    IThorKitchenMultiStageComposite,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_specs,
)


class IThorKitchenOvenThenFridge(
    MultiStageNavigationMixin,
    IThorKitchenMultiStageComposite,
):
    SPEC_TASK = "oven_then_fridge"
    TASK_LANG = 'walk to the oven, then walk to the fridge'
    STAGE_NAMES = ("oven", "fridge")
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK, required=("stages", "robot_pose")
    )


class IThorKitchenFridgeThenOven(
    MultiStageNavigationMixin,
    IThorKitchenMultiStageComposite,
):
    SPEC_TASK = "fridge_then_oven"
    TASK_LANG = 'walk to the fridge, then walk to the oven'
    STAGE_NAMES = ("fridge", "oven")
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK, required=("stages", "robot_pose")
    )


class IThorKitchenFridgeThenDishwasher(
    MultiStageNavigationMixin,
    IThorKitchenMultiStageComposite,
):
    SPEC_TASK = "fridge_then_dishwasher"
    TASK_LANG = 'walk to the fridge, then walk to the dishwasher'
    STAGE_NAMES = ("fridge", "dishwasher")
    SUPPORTED_SCENE_IDS = (6, 7, 13, 25)
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK, required=("stages", "robot_pose")
    )


class IThorKitchenDishwasherThenFridge(
    MultiStageNavigationMixin,
    IThorKitchenMultiStageComposite,
):
    SPEC_TASK = "dishwasher_then_fridge"
    TASK_LANG = 'walk to the dishwasher, then walk to the fridge'
    STAGE_NAMES = ("dishwasher", "fridge")
    SUPPORTED_SCENE_IDS = (6, 7, 13, 25)
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK, required=("stages", "robot_pose")
    )


class IThorKitchenOvenThenDishwasher(
    MultiStageNavigationMixin,
    IThorKitchenMultiStageComposite,
):
    SPEC_TASK = "oven_then_dishwasher"
    TASK_LANG = 'walk to the oven, then walk to the dishwasher'
    STAGE_NAMES = ("oven", "dishwasher")
    SUPPORTED_SCENE_IDS = (6, 7, 13, 25)
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK, required=("stages", "robot_pose")
    )


class IThorKitchenDishwasherThenOven(
    MultiStageNavigationMixin,
    IThorKitchenMultiStageComposite,
):
    SPEC_TASK = "dishwasher_then_oven"
    TASK_LANG = 'walk to the dishwasher, then walk to the oven'
    STAGE_NAMES = ("dishwasher", "oven")
    SUPPORTED_SCENE_IDS = (6, 7, 13, 25)
    SCENE_SPECS = get_ithor_kitchen_task_specs(
        SPEC_TASK, required=("stages", "robot_pose")
    )


__all__ = [
    "IThorKitchenDishwasherThenFridge",
    "IThorKitchenDishwasherThenOven",
    "IThorKitchenFridgeThenDishwasher",
    "IThorKitchenFridgeThenOven",
    "IThorKitchenOvenThenDishwasher",
    "IThorKitchenOvenThenFridge",
]
