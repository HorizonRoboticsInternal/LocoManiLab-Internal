"""iTHOR kitchen: put a cup into the sink."""

from robocasa.environments.ithor_kitchen.tasks.atomic.bowl_to_sink import (
    IThorKitchenBowlToSink,
)


class IThorKitchenCupToSink(IThorKitchenBowlToSink):
    """Cup variant that reuses the BowlToSink scene calibration."""

    ATOMIC_SKILL = "CupToSink"
    SOURCE_NAME = "cup"
    SOURCE_LANG = "cup"
    SOURCE_XML = "models/assets/objects/objaverse/cup/cup_2/model.xml"
    SOURCE_SCALE = 1.0


__all__ = ["IThorKitchenCupToSink"]
