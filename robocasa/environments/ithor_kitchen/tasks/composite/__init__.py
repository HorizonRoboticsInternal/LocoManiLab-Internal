"""Composite iTHOR kitchen task modules."""

from robocasa.environments.ithor_kitchen.tasks.composite.base import (
    IThorKitchenAtomicTask,
    IThorKitchenMultiStageComposite,
)
from robocasa.environments.ithor_kitchen.tasks.composite.multi_stage_navigation import (
    IThorKitchenDishwasherThenFridge,
    IThorKitchenDishwasherThenOven,
    IThorKitchenFridgeThenDishwasher,
    IThorKitchenFridgeThenOven,
    IThorKitchenOvenThenDishwasher,
    IThorKitchenOvenThenFridge,
)
from robocasa.environments.ithor_kitchen.tasks.composite.bend_and_put_apple_on_plate import (
    IThorKitchenBendAndPutAppleOnPlate,
)
from robocasa.environments.ithor_kitchen.tasks.composite.approach_table_manipulation import (
    IThorKitchenApproachTableAndLiftApple,
    IThorKitchenApproachTableAndLiftBowl,
    IThorKitchenApproachTableAndLiftSaltShaker,
    IThorKitchenApproachTableAndPutAppleOnPlate,
    IThorKitchenApproachTableAndPutBreadOnPlate,
    IThorKitchenApproachTableAndPutCheeseOnPlate,
    IThorKitchenApproachTableAndPutMugOnPlate,
)
from robocasa.environments.ithor_kitchen.tasks.composite.multi_step_manipulation import (
    IThorKitchenApproachCabinetAndStoreStatue,
    IThorKitchenBowlToSinkThenAppleOnPlate,
    IThorKitchenOpenThenCloseDrawer,
    IThorKitchenStoreAppleInFridge,
    IThorKitchenOpenThenCloseCabinet,
    IThorKitchenCloseThenOpenCabinet,
)
from robocasa.environments.ithor_kitchen.tasks.composite.bread_off_plate_then_apple_on_plate import (
    IThorKitchenBreadOffPlateThenAppleOnPlate,
)
from robocasa.environments.ithor_kitchen.tasks.composite.cup_from_stool_to_sink import (
    IThorKitchenCupFromStoolToSink,
)
from robocasa.environments.ithor_kitchen.tasks.composite.pan_to_burner_then_egg_into_pan import (
    IThorKitchenPanToRightBurnerThenEggIntoPan,
)
from robocasa.environments.ithor_kitchen.tasks.composite.canned_drink_to_fridge_then_close import (
    IThorKitchenCannedDrinkIntoFridgeThenCloseDoor,
)
from robocasa.environments.ithor_kitchen.tasks.composite.canned_drink_into_drawer import (
    IThorKitchenOpenDrawerThenPutCannedDrinkInside,
)
from robocasa.environments.ithor_kitchen.tasks.composite.canned_drink_from_fridge_to_trash import (
    IThorKitchenCannedDrinkFromFridgeToTrash,
)

from robocasa.environments.ithor_kitchen.tasks.composite.approach_fixture_manipulation import (
    IThorKitchenApproachTableAndStandCannedDrinkUpright,
    IThorKitchenApproachStoveAndPutPan,
    IThorKitchenApproachSinkAndPutBowl,
    IThorKitchenApproachSinkAndPutCup,
    IThorKitchenApproachDrawerAndOpen,
    IThorKitchenApproachDrawerAndClose,
    IThorKitchenApproachCabinetAndOpen,
    IThorKitchenApproachCabinetAndClose,
)

from robocasa.environments.ithor_kitchen.tasks.composite.approach_fixture_manipulation import (
    IThorKitchenApproachTableAndStandCannedDrinkUpright,
    IThorKitchenApproachStoveAndPutPan,
    IThorKitchenApproachSinkAndPutBowl,
    IThorKitchenApproachSinkAndPutCup,
    IThorKitchenApproachDrawerAndOpen,
    IThorKitchenApproachDrawerAndClose,
    IThorKitchenApproachCabinetAndOpen,
    IThorKitchenApproachCabinetAndClose,
)

from robocasa.environments.ithor_kitchen.tasks.composite.take_statue_from_cabinet import (
    IThorKitchenTakeStatueFromCabinet,
)

from robocasa.environments.ithor_kitchen.tasks.composite.low_microwave_apple import (
    IThorKitchenCloseThenOpenLowMicrowave,
    IThorKitchenOpenCloseLowMicrowave,
    IThorKitchenOpenLowMicrowaveAndLiftApple,
    IThorKitchenOpenLowMicrowaveAndPutAppleInside,
    IThorKitchenPutAppleInLowMicrowave,
    IThorKitchenTakeAppleFromOpenLowMicrowaveToTop,
)

__all__ = [
    "IThorKitchenCannedDrinkFromFridgeToTrash",
    "IThorKitchenOpenDrawerThenPutCannedDrinkInside",
    "IThorKitchenCannedDrinkIntoFridgeThenCloseDoor",
    "IThorKitchenPanToRightBurnerThenEggIntoPan",
    "IThorKitchenCupFromStoolToSink",
    "IThorKitchenBreadOffPlateThenAppleOnPlate",
    "IThorKitchenCloseThenOpenLowMicrowave",
    "IThorKitchenOpenCloseLowMicrowave",
    "IThorKitchenOpenLowMicrowaveAndLiftApple",
    "IThorKitchenOpenLowMicrowaveAndPutAppleInside",
    "IThorKitchenPutAppleInLowMicrowave",
    "IThorKitchenTakeAppleFromOpenLowMicrowaveToTop",
    "IThorKitchenTakeStatueFromCabinet",
    "IThorKitchenApproachTableAndStandCannedDrinkUpright",
    "IThorKitchenApproachStoveAndPutPan",
    "IThorKitchenApproachSinkAndPutBowl",
    "IThorKitchenApproachSinkAndPutCup",
    "IThorKitchenApproachDrawerAndOpen",
    "IThorKitchenApproachDrawerAndClose",
    "IThorKitchenApproachCabinetAndOpen",
    "IThorKitchenApproachCabinetAndClose",
    "IThorKitchenAtomicTask",
    "IThorKitchenMultiStageComposite",
    "IThorKitchenDishwasherThenFridge",
    "IThorKitchenDishwasherThenOven",
    "IThorKitchenFridgeThenDishwasher",
    "IThorKitchenFridgeThenOven",
    "IThorKitchenOvenThenDishwasher",
    "IThorKitchenOvenThenFridge",
    "IThorKitchenBendAndPutAppleOnPlate",
    "IThorKitchenApproachTableAndLiftApple",
    "IThorKitchenApproachTableAndLiftBowl",
    "IThorKitchenApproachTableAndLiftSaltShaker",
    "IThorKitchenApproachTableAndPutAppleOnPlate",
    "IThorKitchenApproachTableAndPutBreadOnPlate",
    "IThorKitchenApproachTableAndPutCheeseOnPlate",
    "IThorKitchenApproachTableAndPutMugOnPlate",
    "IThorKitchenStoreAppleInFridge",
    "IThorKitchenOpenThenCloseCabinet",
    "IThorKitchenCloseThenOpenCabinet",
    "IThorKitchenOpenThenCloseDrawer",
    "IThorKitchenBowlToSinkThenAppleOnPlate",
    "IThorKitchenApproachCabinetAndStoreStatue",
]
