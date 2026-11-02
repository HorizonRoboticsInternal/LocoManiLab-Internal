"""Composite iTHOR living-room task modules."""

from robocasa.environments.ithor_living_room.tasks.composite.base import (
    IThorLivingRoomAtomicTask,
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

__all__ = [
    "IThorLivingRoomStackBooksAndPlaceMug",
    "IThorLivingRoomApproachTableAndLiftDumbbell",
    "IThorLivingRoomApproachTableAndLiftStatue",
    "IThorLivingRoomApproachTableAndLiftWateringCan",
    "IThorLivingRoomApproachTableAndPutRemoteInBasket",
    "IThorLivingRoomApproachTableAndPutCandleInBasket",
    "IThorLivingRoomApproachTableAndPutTissueBoxInBasket",
    "IThorLivingRoomApproachTableAndPutBookInBasket",
    "IThorLivingRoomAtomicTask",
    "IThorLivingRoomMultiStageComposite",
    "IThorLivingRoomBinThenSofa",
    "IThorLivingRoomBinThenTable",
    "IThorLivingRoomSofaThenTable",
    "IThorLivingRoomSofaThenBin",
    "IThorLivingRoomTableThenSofa",
    "IThorLivingRoomTableThenBin",
    "IThorLivingRoomPackCandleStatueIntoBasket",
    "IThorLivingRoomWalkAndCloseDrawer",
    "IThorLivingRoomWalkAndOpenDrawer",
    "IThorLivingRoomWalkAndOpenLowDrawer",
    "IThorLivingRoomWalkAndCloseLowDrawer",
    "IThorLivingRoomLowDrawerPnP",
    "IThorLivingRoomWalkAndOpenCloseDrawer",
    "IThorLivingRoomWalkToTableAndPutCanInBasket",
    "IThorLivingRoomMugFromCoffeeTableToBoxOnSofa",
    "IThorLivingRoomMugFromSofaToCoffeeTable",
    "IThorLivingRoomCloseTopRightThenTopLeftDrawer",
    "IThorLivingRoomRemoteFromSideTableIntoBoxOnSofa",
    "IThorLivingRoomStatueFromTableToTrashBin",
    "IThorLivingRoomCannedDrinkFromTableIntoBoxOnArmchair",
]
