"""Canonical Bedroom task catalog."""

from .articulation import (
    IThorBedroomCloseTopDrawer,
    IThorBedroomOpenTopDrawer,
)
from .articulation_composites import (
    IThorBedroomApproachDeskAndCloseLaptop,
    IThorBedroomApproachDeskAndOpenLaptop,
    IThorBedroomApproachStorageAndCloseBoxFlap,
    IThorBedroomApproachStorageAndCloseDrawer,
    IThorBedroomApproachStorageAndCloseHamperLid,
    IThorBedroomApproachStorageAndCloseSafeDoor,
    IThorBedroomApproachStorageAndOpenBoxFlap,
    IThorBedroomApproachStorageAndOpenDrawer,
    IThorBedroomApproachStorageAndOpenHamperLid,
    IThorBedroomApproachStorageAndOpenSafeDoor,
    IThorBedroomOpenThenCloseBoxFlap,
    IThorBedroomOpenThenCloseDrawer,
    IThorBedroomOpenThenCloseHamperLid,
    IThorBedroomOpenThenCloseLaptop,
    IThorBedroomOpenThenCloseSafeDoor,
    IThorBedroomOpenThenCloseTopDrawer,
)
from .bedside_composites import (
    IThorBedroomApproachBedStandPillowThenDesk,
    IThorBedroomOpenBoxThenStatueFromSofaIntoBox,
)
from .bedside_objects import (
    IThorBedroomStandPillowAtHeadboard,
    IThorBedroomLiftStatueFromSofa,
)
from .bedside_window_hamper import (
    IThorBedroomApproachBedsideWindowThenOpenHamperLid,
)
from .box_flap import IThorBedroomCloseBoxFlap, IThorBedroomOpenBoxFlap
from .drawer import IThorBedroomCloseDrawer, IThorBedroomOpenDrawer
from .floor_basket_composite import (
    IThorBedroomLiftBasketFromFloorThenPlaceOnDeskAndPutBoxedFoodInBasket,
)
from .hamper_lid import (
    IThorBedroomCloseHamperLid,
    IThorBedroomOpenHamperLid,
)
from .laptop import IThorBedroomCloseLaptop, IThorBedroomOpenLaptop
from .lift_baseball_bat import IThorBedroomLiftBaseballBat
from .lift_basket import (
    IThorBedroomLiftBasket,
    IThorBedroomLiftBasketFromFloor,
)
from .navigation import (
    IThorBedroomApproachBed,
    IThorBedroomApproachBedsideWindow,
    IThorBedroomApproachDesk,
    IThorBedroomNavigateToStorage,
)
from .navigation_composites import (
    IThorBedroomBedThenDesk,
    IThorBedroomBedThenStorage,
    IThorBedroomDeskThenBed,
    IThorBedroomDeskThenStorage,
    IThorBedroomStorageThenBed,
    IThorBedroomStorageThenDesk,
)
from .object_in_receptacle import (
    IThorBedroomBoxedFoodIntoBasket,
    IThorBedroomCanIntoBasket,
    IThorBedroomMugIntoBasket,
)
from .safe_door import IThorBedroomCloseSafeDoor, IThorBedroomOpenSafeDoor
from .routine_composites import *
from .routine_composites import TASK_CLASSES as _ROUTINE_TASK_CLASSES
from .scene_315_composites import (
    IThorBedroomApproachTrashCanAndPutCanInTrashThenLiftBasket,
    IThorBedroomApproachDeskAndPutMugThenCanInBasket,
)
from .trash_can import IThorBedroomApproachTrashCan, IThorBedroomCanIntoTrashCan
from .workspace_composites import (
    IThorBedroomApproachDeskAndLiftBasket,
    IThorBedroomApproachDeskAndPutBoxedFoodInBasket,
    IThorBedroomApproachDeskAndPutCanInBasket,
    IThorBedroomApproachDeskAndPutMugInBasket,
)


TASK_CLASSES = (
    IThorBedroomApproachBed,
    IThorBedroomApproachBedsideWindow,
    IThorBedroomApproachDesk,
    IThorBedroomNavigateToStorage,
    IThorBedroomApproachTrashCan,
    IThorBedroomCanIntoTrashCan,
    IThorBedroomOpenSafeDoor,
    IThorBedroomCloseSafeDoor,
    IThorBedroomOpenHamperLid,
    IThorBedroomCloseHamperLid,
    IThorBedroomOpenBoxFlap,
    IThorBedroomCloseBoxFlap,
    IThorBedroomOpenDrawer,
    IThorBedroomCloseDrawer,
    IThorBedroomOpenLaptop,
    IThorBedroomCloseLaptop,
    IThorBedroomCanIntoBasket,
    IThorBedroomMugIntoBasket,
    IThorBedroomBoxedFoodIntoBasket,
    IThorBedroomOpenTopDrawer,
    IThorBedroomCloseTopDrawer,
    IThorBedroomLiftBasket,
    IThorBedroomLiftBasketFromFloor,
    IThorBedroomLiftBaseballBat,
    IThorBedroomStandPillowAtHeadboard,
    IThorBedroomLiftStatueFromSofa,
    IThorBedroomApproachBedStandPillowThenDesk,
    IThorBedroomOpenBoxThenStatueFromSofaIntoBox,
    IThorBedroomBedThenDesk,
    IThorBedroomDeskThenBed,
    IThorBedroomBedThenStorage,
    IThorBedroomStorageThenBed,
    IThorBedroomDeskThenStorage,
    IThorBedroomStorageThenDesk,
    IThorBedroomApproachBedsideWindowThenOpenHamperLid,
    IThorBedroomApproachDeskAndOpenLaptop,
    IThorBedroomApproachDeskAndCloseLaptop,
    IThorBedroomApproachDeskAndLiftBasket,
    IThorBedroomApproachDeskAndPutCanInBasket,
    IThorBedroomApproachDeskAndPutMugInBasket,
    IThorBedroomApproachDeskAndPutBoxedFoodInBasket,
    IThorBedroomLiftBasketFromFloorThenPlaceOnDeskAndPutBoxedFoodInBasket,
    IThorBedroomApproachStorageAndOpenSafeDoor,
    IThorBedroomApproachStorageAndCloseSafeDoor,
    IThorBedroomApproachStorageAndOpenHamperLid,
    IThorBedroomApproachStorageAndCloseHamperLid,
    IThorBedroomApproachStorageAndOpenDrawer,
    IThorBedroomApproachStorageAndCloseDrawer,
    IThorBedroomApproachStorageAndOpenBoxFlap,
    IThorBedroomApproachStorageAndCloseBoxFlap,
    IThorBedroomOpenThenCloseDrawer,
    IThorBedroomOpenThenCloseTopDrawer,
    IThorBedroomOpenThenCloseSafeDoor,
    IThorBedroomOpenThenCloseHamperLid,
    IThorBedroomOpenThenCloseBoxFlap,
    IThorBedroomOpenThenCloseLaptop,
    IThorBedroomApproachTrashCanAndPutCanInTrashThenLiftBasket,
    IThorBedroomApproachDeskAndPutMugThenCanInBasket,
    *_ROUTINE_TASK_CLASSES,
)

__all__ = [task_class.__name__ for task_class in TASK_CLASSES]
