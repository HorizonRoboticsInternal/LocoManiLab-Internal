"""Public Bathroom task catalog; this tuple is the only launch-task list."""

from robocasa.environments.ithor_bathroom.tasks.articulation import (
    IThorBathroomCloseToiletLid,
    IThorBathroomOpenToiletLid,
)
from robocasa.environments.ithor_bathroom.tasks.bathtub_tasks import (
    IThorBathroomPutSpongeInBathtub,
)
from robocasa.environments.ithor_bathroom.tasks.cabinet_lift import (
    IThorBathroomLiftSoapBottleFromCabinet,
)
from robocasa.environments.ithor_bathroom.tasks.composite import *  # noqa: F403
from robocasa.environments.ithor_bathroom.tasks.composite import (
    TASK_CLASSES as COMPOSITE_TASK_CLASSES,
)
from robocasa.environments.ithor_bathroom.tasks.drawer_pnp import (
    IThorBathroomTakeSoapBottleFromDrawer,
    IThorBathroomToiletPaperRollIntoDrawer,
)
from robocasa.environments.ithor_bathroom.tasks.floor_lift import (
    IThorBathroomLiftCandleFromFloor,
)
from robocasa.environments.ithor_bathroom.tasks.lift import (
    IThorBathroomLiftScrubBrush,
    IThorBathroomLiftSoapBottle,
    IThorBathroomLiftSprayBottle,
    IThorBathroomLiftToiletPaperRoll,
)
from robocasa.environments.ithor_bathroom.tasks.navigation import (
    IThorBathroomApproachGarbageCan,
    IThorBathroomApproachSink,
    IThorBathroomApproachToilet,
    IThorBathroomNavigateToBathingArea,
)
from robocasa.environments.ithor_bathroom.tasks.pick_place import (
    IThorBathroomToiletPaperRollIntoBin,
)
from robocasa.environments.ithor_bathroom.tasks.reorientation import (
    IThorBathroomStandSoapBottleUpright,
)
from robocasa.environments.ithor_bathroom.tasks.shelf_pnp import (
    IThorBathroomSoapToLowerShelf,
)
from robocasa.environments.ithor_bathroom.tasks.shower_door import (
    IThorBathroomCloseShowerDoor,
    IThorBathroomOpenShowerDoor,
)
from robocasa.environments.ithor_bathroom.tasks.sink_tasks import (
    IThorBathroomPutSpongeInSink,
)
from robocasa.environments.ithor_bathroom.tasks.storage_articulation import (
    IThorBathroomCloseCabinetDoor,
    IThorBathroomCloseLowerDrawer,
    IThorBathroomCloseUpperCabinetDoor,
    IThorBathroomOpenCabinetDoor,
    IThorBathroomOpenLowerDrawer,
    IThorBathroomOpenUpperCabinetDoor,
)


ATOMIC_TASK_CLASSES = (
    IThorBathroomApproachGarbageCan,
    IThorBathroomApproachSink,
    IThorBathroomApproachToilet,
    IThorBathroomCloseCabinetDoor,
    IThorBathroomCloseLowerDrawer,
    IThorBathroomCloseUpperCabinetDoor,
    IThorBathroomCloseShowerDoor,
    IThorBathroomCloseToiletLid,
    IThorBathroomLiftCandleFromFloor,
    IThorBathroomLiftScrubBrush,
    IThorBathroomLiftSoapBottle,
    IThorBathroomLiftSoapBottleFromCabinet,
    IThorBathroomLiftSprayBottle,
    IThorBathroomLiftToiletPaperRoll,
    IThorBathroomNavigateToBathingArea,
    IThorBathroomOpenCabinetDoor,
    IThorBathroomOpenLowerDrawer,
    IThorBathroomOpenUpperCabinetDoor,
    IThorBathroomOpenShowerDoor,
    IThorBathroomOpenToiletLid,
    IThorBathroomPutSpongeInBathtub,
    IThorBathroomPutSpongeInSink,
    IThorBathroomSoapToLowerShelf,
    IThorBathroomStandSoapBottleUpright,
    IThorBathroomToiletPaperRollIntoBin,
    IThorBathroomToiletPaperRollIntoDrawer,
    IThorBathroomTakeSoapBottleFromDrawer,
)

TASK_CLASSES = (*ATOMIC_TASK_CLASSES, *COMPOSITE_TASK_CLASSES)

__all__ = [task_class.__name__ for task_class in TASK_CLASSES]
