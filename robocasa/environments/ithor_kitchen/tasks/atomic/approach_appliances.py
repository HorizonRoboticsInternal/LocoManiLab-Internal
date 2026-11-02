"""iTHOR kitchen navigation tasks for the dishwasher and oven."""

from robocasa.environments.ithor_kitchen.tasks.atomic.approach_fridge import (
    IThorKitchenApproachFridge,
)


class IThorKitchenApproachDishwasher(IThorKitchenApproachFridge):
    """Walk from the composite-route spawn to the named dishwasher."""

    SUPPORTED_SCENE_IDS = (6, 7, 13, 25)
    SPEC_TASK = "approach_dishwasher"
    TARGET_CATEGORY = "Dishwasher"


class IThorKitchenApproachOven(IThorKitchenApproachFridge):
    """Walk from the composite-route spawn to the named oven."""

    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SPEC_TASK = "approach_oven"
    TARGET_CATEGORY = "Oven"


__all__ = [
    "IThorKitchenApproachDishwasher",
    "IThorKitchenApproachOven",
]
