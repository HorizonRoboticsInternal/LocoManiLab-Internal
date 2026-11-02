from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    IThorKitchenEnv,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.approach_fridge import (
    IThorKitchenApproachFridge,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.approach_appliances import (
    IThorKitchenApproachDishwasher,
    IThorKitchenApproachOven,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.articulation import (
    IThorKitchenOpenCabinet,
    IThorKitchenOpenDrawer,
    IThorKitchenCloseFridge,
    IThorKitchenCloseCabinet,
    IThorKitchenCloseDrawer,
    IThorKitchenOpenFridge,
    IThorKitchenOpenOven,
    IThorKitchenCloseOven,
)
from robocasa.environments.ithor_kitchen.tasks.composite.approach_open_oven import (
    IThorKitchenApproachOpenOven,
)
from robocasa.environments.ithor_kitchen.tasks.composite.approach_open_fridge import (
    IThorKitchenApproachOpenFridge,
)
from robocasa.environments.ithor_kitchen.tasks.composite.drawer_pnp import (
    IThorKitchenDrawerPnP,
)
from robocasa.environments.ithor_kitchen.tasks.composite.apple_cabinet_pnp import (
    IThorKitchenAppleCabinetPnP,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.apple_on_plate import (
    IThorKitchenAppleOnPlate,
)
from robocasa.environments.ithor_kitchen.tasks.composite.apple_banana_plate_swap import (
    IThorKitchenAppleBananaPlateSwap,
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
from robocasa.environments.ithor_kitchen.tasks.atomic.pack_produce_into_box import (
    IThorKitchenPackProduceIntoBox,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.objects_into_box import (
    IThorKitchenAppleIntoBox,
    IThorKitchenCupIntoBox,
    IThorKitchenStatueIntoBox,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.banana_basket_to_table import (
    IThorKitchenBananaBasketToTable,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.coffee_drink_to_trash import (
    IThorKitchenCoffeeDrinkToTrash,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.cup_stack import (
    IThorKitchenCupStack,
)
from robocasa.environments.ithor_kitchen.tasks.composite.open_drawer_take_apple import (
    IThorKitchenOpenDrawerTakeApple,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.bowl_to_sink import (
    IThorKitchenBowlToSink,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.cup_to_sink import (
    IThorKitchenCupToSink,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.stand_canned_drink_upright import (
    IThorKitchenStandCannedDrinkUpright,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.pan_table_to_stove import (
    IThorKitchenPanTableToStove,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.objects_on_plate import (
    IThorKitchenBreadOnPlate,
    IThorKitchenCheeseOnPlate,
    IThorKitchenMugOnPlate,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.lift_objects import (
    IThorKitchenLiftEggplant,
    IThorKitchenLiftPlate,
    IThorKitchenLiftSaltShaker,
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

from robocasa.environments.ithor_kitchen.tasks.composite.approach_open_close import (
    IThorKitchenApproachDrawerAndOpenClose,
    IThorKitchenApproachCabinetAndOpenClose,
)
from robocasa.environments.ithor_kitchen.tasks.composite.apple_to_trash import (
    IThorKitchenPickAppleAndTurnToTrash,
    IThorKitchenOpenDrawerAndTurnAppleToTrash,
)

from robocasa.environments.ithor_kitchen.tasks.composite.low_microwave_apple import (
    IThorKitchenCloseThenOpenLowMicrowave,
    IThorKitchenOpenCloseLowMicrowave,
    IThorKitchenOpenLowMicrowaveAndLiftApple,
    IThorKitchenOpenLowMicrowaveAndPutAppleInside,
    IThorKitchenPutAppleInLowMicrowave,
    IThorKitchenTakeAppleFromOpenLowMicrowaveToTop,
)

from robocasa.environments.ithor_kitchen.tasks.atomic.egg_into_jar import (
    IThorKitchenEggIntoJar,
)

from robocasa.environments.ithor_kitchen.tasks.atomic.wine_bottle_onto_stool import (
    IThorKitchenWineBottleOntoStool,
)

from robocasa.environments.ithor_kitchen.tasks.atomic.sponge_into_open_drawer import (
    IThorKitchenSpongeIntoOpenDrawer,
)

from robocasa.environments.ithor_kitchen.tasks.atomic.pan_left_to_right_burner import (
    IThorKitchenPanLeftToRightBurner,
)


from robocasa.environments.ithor_kitchen.tasks.atomic.canned_drink_into_pan import (
    IThorKitchenCannedDrinkIntoPan,
)

from robocasa.environments.ithor_kitchen.tasks.atomic.salt_shaker_stool_to_sink_counter import (
    IThorKitchenSaltShakerStoolToSinkCounter,
)

__all__ = [
    "IThorKitchenCannedDrinkFromFridgeToTrash",
    "IThorKitchenOpenDrawerThenPutCannedDrinkInside",
    "IThorKitchenCannedDrinkIntoFridgeThenCloseDoor",
    "IThorKitchenPanToRightBurnerThenEggIntoPan",
    "IThorKitchenCupFromStoolToSink",
    "IThorKitchenBreadOffPlateThenAppleOnPlate",
    "IThorKitchenSaltShakerStoolToSinkCounter",
    "IThorKitchenCannedDrinkIntoPan",
    "IThorKitchenPanLeftToRightBurner",
    "IThorKitchenSpongeIntoOpenDrawer",
    "IThorKitchenWineBottleOntoStool",
    "IThorKitchenEggIntoJar",
    "IThorKitchenCloseThenOpenLowMicrowave",
    "IThorKitchenOpenCloseLowMicrowave",
    "IThorKitchenOpenLowMicrowaveAndLiftApple",
    "IThorKitchenOpenLowMicrowaveAndPutAppleInside",
    "IThorKitchenPutAppleInLowMicrowave",
    "IThorKitchenTakeAppleFromOpenLowMicrowaveToTop",
    "IThorKitchenApproachDrawerAndOpenClose",
    "IThorKitchenApproachCabinetAndOpenClose",
    "IThorKitchenPickAppleAndTurnToTrash",
    "IThorKitchenOpenDrawerAndTurnAppleToTrash",
    "IThorKitchenTakeStatueFromCabinet",
    "IThorKitchenApproachTableAndStandCannedDrinkUpright",
    "IThorKitchenApproachStoveAndPutPan",
    "IThorKitchenApproachSinkAndPutBowl",
    "IThorKitchenApproachSinkAndPutCup",
    "IThorKitchenApproachDrawerAndOpen",
    "IThorKitchenApproachDrawerAndClose",
    "IThorKitchenApproachCabinetAndOpen",
    "IThorKitchenApproachCabinetAndClose",
    "IThorKitchenEnv",
    "REGISTERED_ITHOR_KITCHEN_ENVS",
    "IThorKitchenApproachDishwasher",
    "IThorKitchenApproachFridge",
    "IThorKitchenApproachOven",
    "IThorKitchenOpenCabinet",
    "IThorKitchenOpenDrawer",
    "IThorKitchenOpenFridge",
    "IThorKitchenCloseFridge",
    "IThorKitchenCloseCabinet",
    "IThorKitchenCloseDrawer",
    "IThorKitchenOpenOven",
    "IThorKitchenCloseOven",
    "IThorKitchenApproachOpenOven",
    "IThorKitchenApproachOpenFridge",
    "IThorKitchenDrawerPnP",
    "IThorKitchenAppleCabinetPnP",
    "IThorKitchenAppleOnPlate",
    "IThorKitchenAppleBananaPlateSwap",
    "IThorKitchenStoreAppleInFridge",
    "IThorKitchenOpenThenCloseCabinet",
    "IThorKitchenCloseThenOpenCabinet",
    "IThorKitchenOpenThenCloseDrawer",
    "IThorKitchenBowlToSinkThenAppleOnPlate",
    "IThorKitchenApproachCabinetAndStoreStatue",
    "IThorKitchenPackProduceIntoBox",
    "IThorKitchenAppleIntoBox",
    "IThorKitchenCupIntoBox",
    "IThorKitchenStatueIntoBox",
    "IThorKitchenBananaBasketToTable",
    "IThorKitchenCoffeeDrinkToTrash",
    "IThorKitchenCupStack",
    "IThorKitchenOpenDrawerTakeApple",
    "IThorKitchenBowlToSink",
    "IThorKitchenCupToSink",
    "IThorKitchenStandCannedDrinkUpright",
    "IThorKitchenPanTableToStove",
    "IThorKitchenBreadOnPlate",
    "IThorKitchenCheeseOnPlate",
    "IThorKitchenMugOnPlate",
    "IThorKitchenLiftEggplant",
    "IThorKitchenLiftPlate",
    "IThorKitchenLiftSaltShaker",
    "IThorKitchenOvenThenFridge",
    "IThorKitchenFridgeThenOven",
    "IThorKitchenFridgeThenDishwasher",
    "IThorKitchenDishwasherThenFridge",
    "IThorKitchenOvenThenDishwasher",
    "IThorKitchenDishwasherThenOven",
    "IThorKitchenBendAndPutAppleOnPlate",
    "IThorKitchenApproachTableAndLiftApple",
    "IThorKitchenApproachTableAndLiftBowl",
    "IThorKitchenApproachTableAndLiftSaltShaker",
    "IThorKitchenApproachTableAndPutAppleOnPlate",
    "IThorKitchenApproachTableAndPutBreadOnPlate",
    "IThorKitchenApproachTableAndPutCheeseOnPlate",
    "IThorKitchenApproachTableAndPutMugOnPlate",
]
