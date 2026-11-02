"""iTHOR kitchen variants that reuse the AppleOnPlate scene calibration."""

from robocasa.environments.ithor_kitchen.tasks.atomic.apple_on_plate import (
    IThorKitchenAppleOnPlate,
)


class IThorKitchenBreadOnPlate(IThorKitchenAppleOnPlate):
    ATOMIC_SKILL = "BreadOnPlate"
    SOURCE_NAME = "bread"
    SOURCE_GROUP = "bread"
    SOURCE_LANG = "bread"


class IThorKitchenCheeseOnPlate(IThorKitchenAppleOnPlate):
    ATOMIC_SKILL = "CheeseOnPlate"
    SOURCE_NAME = "cheese"
    SOURCE_GROUP = "cheese"
    SOURCE_LANG = "cheese"


class IThorKitchenMugOnPlate(IThorKitchenAppleOnPlate):
    ATOMIC_SKILL = "MugOnPlate"
    SOURCE_NAME = "mug"
    SOURCE_GROUP = "mug"
    SOURCE_LANG = "mug"


__all__ = [
    "IThorKitchenBreadOnPlate",
    "IThorKitchenCheeseOnPlate",
    "IThorKitchenMugOnPlate",
]
