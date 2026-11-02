from robocasa.environments.ithor_living_room.tasks.atomic.approach_sofa import (
    IThorLivingRoomApproachSofa,
)
from robocasa.environments.ithor_living_room.tasks.atomic.box_from_chair_to_floor import (
    IThorLivingRoomBoxFromChairToFloor,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_bowl_from_coffee_table import (
    IThorLivingRoomLiftBowlFromCoffeeTable,
)
from robocasa.environments.ithor_living_room.tasks.atomic.box_from_sofa_to_floor import (
    IThorLivingRoomBoxFromSofaToFloor,
)
from robocasa.environments.ithor_living_room.tasks.atomic.statue_chair_to_table import (
    IThorLivingRoomStatueChairToTable,
)
from robocasa.environments.ithor_living_room.tasks.atomic.statue_into_box import (
    IThorLivingRoomStatueIntoBox,
)
from robocasa.environments.ithor_living_room.tasks.atomic.remote_into_box import (
    IThorLivingRoomRemoteIntoBox,
)
from robocasa.environments.ithor_living_room.tasks.atomic.sit_on_sofa import (
    IThorLivingRoomSitOnSofa,
)
from robocasa.environments.ithor_living_room.tasks.atomic.approach_table import (
    IThorLivingRoomApproachTable,
)
from robocasa.environments.ithor_living_room.tasks.atomic.articulation import (
    IThorLivingRoomCloseDrawer,
    IThorLivingRoomOpenDrawer,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_statue import (
    IThorLivingRoomLiftStatue,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_watering_can import (
    IThorLivingRoomLiftWateringCan,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_dumbbell import (
    IThorLivingRoomLiftDumbbell,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_cup_among_objects import (
    IThorLivingRoomLiftCupAmongObjects,
)
from robocasa.environments.ithor_living_room.tasks.atomic.pour_block_from_basket import (
    IThorLivingRoomPourBlockFromBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.pour_ice_from_cup import (
    IThorLivingRoomPourIceFromCup,
)
from robocasa.environments.ithor_living_room.tasks.atomic.coffee_drink_to_bin import (
    IThorLivingRoomCoffeeDrinkToBin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.remote_to_bin import (
    IThorLivingRoomRemoteToBin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.push_basketball import (
    IThorLivingRoomPushBasketball,
)
from robocasa.environments.ithor_living_room.tasks.atomic.basket_to_sofa import (
    IThorLivingRoomBasketToSofa,
)
from robocasa.environments.ithor_living_room.tasks.atomic.dumbbell_to_sofa import (
    IThorLivingRoomDumbbellToSofa,
)
from robocasa.environments.ithor_living_room.tasks.atomic.statue_to_sofa import (
    IThorLivingRoomStatueToSofa,
)
from robocasa.environments.ithor_living_room.tasks.atomic.remote_to_basket import (
    IThorLivingRoomRemoteToBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.candle_to_basket import (
    IThorLivingRoomCandleToBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.tissue_box_to_basket import (
    IThorLivingRoomTissueBoxToBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.open_laptop import (
    IThorLivingRoomOpenLaptop,
)
from robocasa.environments.ithor_living_room.tasks.atomic.book_into_basket import (
    IThorLivingRoomBookIntoBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.stack_book_on_book import (
    IThorLivingRoomStackBookOnBook,
)
from robocasa.environments.ithor_living_room.tasks.atomic.push_block_to_target_region import (
    IThorLivingRoomPushBlockToTargetRegion,
)
from robocasa.environments.ithor_living_room.tasks.atomic.box_flap import (
    IThorLivingRoomCloseBoxFlap,
    IThorLivingRoomOpenBoxFlap,
)
from robocasa.environments.ithor_living_room.tasks.composite.pack_candle_statue import (
    IThorLivingRoomPackCandleStatueIntoBasket,
)
from robocasa.environments.ithor_living_room.tasks.composite.walk_and_close_drawer import (
    IThorLivingRoomWalkAndCloseDrawer,
)
from robocasa.environments.ithor_living_room.tasks.composite.walk_and_close_low_drawer import (
    IThorLivingRoomWalkAndCloseLowDrawer,
)
from robocasa.environments.ithor_living_room.tasks.composite.low_drawer_pnp import (
    IThorLivingRoomLowDrawerPnP,
)
from robocasa.environments.ithor_living_room.tasks.composite.walk_and_open_low_drawer import (
    IThorLivingRoomWalkAndOpenLowDrawer,
)
from robocasa.environments.ithor_living_room.tasks.composite.walk_and_open_drawer import (
    IThorLivingRoomWalkAndOpenDrawer,
)
from robocasa.environments.ithor_living_room.tasks.composite.walk_and_open_close_drawer import (
    IThorLivingRoomWalkAndOpenCloseDrawer,
)
from robocasa.environments.ithor_living_room.tasks.composite.walk_to_table_and_put_can_in_basket import (
    IThorLivingRoomWalkToTableAndPutCanInBasket,
)
from robocasa.environments.ithor_living_room.tasks.composite.mug_from_coffee_table_to_box_on_sofa import (
    IThorLivingRoomMugFromCoffeeTableToBoxOnSofa,
)
from robocasa.environments.ithor_living_room.tasks.composite.mug_from_sofa_to_coffee_table import (
    IThorLivingRoomMugFromSofaToCoffeeTable,
)
from robocasa.environments.ithor_living_room.tasks.composite.close_top_right_then_top_left_drawer import (
    IThorLivingRoomCloseTopRightThenTopLeftDrawer,
)
from robocasa.environments.ithor_living_room.tasks.composite.remote_from_side_table_into_box_on_sofa import (
    IThorLivingRoomRemoteFromSideTableIntoBoxOnSofa,
)
from robocasa.environments.ithor_living_room.tasks.composite.statue_from_table_to_trash_bin import (
    IThorLivingRoomStatueFromTableToTrashBin,
)
from robocasa.environments.ithor_living_room.tasks.composite.canned_drink_from_table_into_box_on_armchair import (
    IThorLivingRoomCannedDrinkFromTableIntoBoxOnArmchair,
)
from robocasa.environments.ithor_living_room.tasks.composite import (
    IThorLivingRoomBinThenSofa,
    IThorLivingRoomBinThenTable,
    IThorLivingRoomSofaThenTable,
    IThorLivingRoomSofaThenBin,
    IThorLivingRoomTableThenSofa,
    IThorLivingRoomTableThenBin,
)

from robocasa.environments.ithor_living_room.tasks.composite.stack_books_and_place_mug import (
    IThorLivingRoomStackBooksAndPlaceMug,
)
from robocasa.environments.ithor_living_room.tasks.composite.approach_table_manipulation import (
    IThorLivingRoomApproachTableAndLiftDumbbell,
    IThorLivingRoomApproachTableAndLiftStatue,
    IThorLivingRoomApproachTableAndLiftWateringCan,
    IThorLivingRoomApproachTableAndPutRemoteInBasket,
    IThorLivingRoomApproachTableAndPutCandleInBasket,
    IThorLivingRoomApproachTableAndPutTissueBoxInBasket,
    IThorLivingRoomApproachTableAndPutBookInBasket,
)
