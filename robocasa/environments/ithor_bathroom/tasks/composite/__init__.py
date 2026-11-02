"""Public composite iTHOR Bathroom task classes."""

from .approach_manipulation import (
    IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin,
    IThorBathroomApproachSinkAndCloseCabinetDoor,
    IThorBathroomApproachSinkAndLiftScrubBrush,
    IThorBathroomApproachSinkAndLiftSoapBottle,
    IThorBathroomApproachSinkAndLiftSprayBottle,
    IThorBathroomApproachSinkAndLiftToiletPaperRoll,
    IThorBathroomApproachSinkAndOpenCabinetDoor,
    IThorBathroomApproachToiletAndCloseToiletLid,
    IThorBathroomApproachToiletAndOpenToiletLid,
    IThorBathroomNavigateToBathingAreaAndCloseShowerDoor,
    IThorBathroomNavigateToBathingAreaAndLiftScrubBrush,
    IThorBathroomNavigateToBathingAreaAndLiftSoapBottle,
    IThorBathroomNavigateToBathingAreaAndOpenShowerDoor,
    IThorBathroomNavigateToBathingAreaAndPutSpongeInBathtub,
    IThorBathroomNavigateToBathingAreaAndStandSoapBottleUpright,
)
from .articulation_cycles import (
    IThorBathroomCloseThenOpenCabinetDoor,
    IThorBathroomCloseThenOpenLowerDrawer,
    IThorBathroomCloseThenOpenShowerDoor,
    IThorBathroomCloseThenOpenToiletLid,
    IThorBathroomOpenThenCloseCabinetDoor,
    IThorBathroomOpenThenCloseLowerDrawer,
    IThorBathroomOpenThenCloseShowerDoor,
    IThorBathroomOpenThenCloseToiletLid,
)
from .navigation_routes import (
    IThorBathroomNavigateToBathingAreaThenGarbageCan,
    IThorBathroomNavigateToBathingAreaThenSink,
    IThorBathroomNavigateToBathingAreaThenToilet,
    IThorBathroomNavigateToGarbageCanThenBathingArea,
    IThorBathroomNavigateToGarbageCanThenSink,
    IThorBathroomNavigateToGarbageCanThenToilet,
    IThorBathroomNavigateToSinkThenBathingArea,
    IThorBathroomNavigateToSinkThenGarbageCan,
    IThorBathroomNavigateToSinkThenToilet,
    IThorBathroomNavigateToToiletThenBathingArea,
    IThorBathroomNavigateToToiletThenGarbageCan,
    IThorBathroomNavigateToToiletThenSink,
)
from .sink_routine import IThorBathroomApproachSinkAndPutSpongeInSinkThenLiftSoapBottle
from .cabinet_soap_routine import *  # noqa: F403
from .cabinet_soap_routine import TASK_CLASSES as CABINET_SOAP_TASK_CLASSES
from .drawer_routines import *  # noqa: F403
from .drawer_routines import TASK_CLASSES as DRAWER_ROUTINE_TASK_CLASSES
from .scene422_routines import *  # noqa: F403
from .scene422_routines import TASK_CLASSES as SCENE422_TASK_CLASSES
from .routines import *  # noqa: F403
from .routines import TASK_CLASSES as ROUTINE_TASK_CLASSES


TASK_CLASSES = (
    IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin,
    IThorBathroomApproachSinkAndCloseCabinetDoor,
    IThorBathroomApproachSinkAndLiftScrubBrush,
    IThorBathroomApproachSinkAndLiftSoapBottle,
    IThorBathroomApproachSinkAndLiftSprayBottle,
    IThorBathroomApproachSinkAndLiftToiletPaperRoll,
    IThorBathroomApproachSinkAndOpenCabinetDoor,
    IThorBathroomApproachToiletAndCloseToiletLid,
    IThorBathroomApproachToiletAndOpenToiletLid,
    IThorBathroomCloseThenOpenCabinetDoor,
    IThorBathroomCloseThenOpenLowerDrawer,
    IThorBathroomCloseThenOpenShowerDoor,
    IThorBathroomCloseThenOpenToiletLid,
    IThorBathroomNavigateToBathingAreaAndCloseShowerDoor,
    IThorBathroomNavigateToBathingAreaAndLiftScrubBrush,
    IThorBathroomNavigateToBathingAreaAndLiftSoapBottle,
    IThorBathroomNavigateToBathingAreaAndOpenShowerDoor,
    IThorBathroomNavigateToBathingAreaAndPutSpongeInBathtub,
    IThorBathroomNavigateToBathingAreaAndStandSoapBottleUpright,
    IThorBathroomNavigateToBathingAreaThenGarbageCan,
    IThorBathroomNavigateToBathingAreaThenSink,
    IThorBathroomNavigateToBathingAreaThenToilet,
    IThorBathroomNavigateToGarbageCanThenBathingArea,
    IThorBathroomNavigateToGarbageCanThenSink,
    IThorBathroomNavigateToGarbageCanThenToilet,
    IThorBathroomNavigateToSinkThenBathingArea,
    IThorBathroomNavigateToSinkThenGarbageCan,
    IThorBathroomNavigateToSinkThenToilet,
    IThorBathroomNavigateToToiletThenBathingArea,
    IThorBathroomNavigateToToiletThenGarbageCan,
    IThorBathroomNavigateToToiletThenSink,
    IThorBathroomOpenThenCloseCabinetDoor,
    IThorBathroomOpenThenCloseLowerDrawer,
    IThorBathroomOpenThenCloseShowerDoor,
    IThorBathroomOpenThenCloseToiletLid,
    IThorBathroomApproachSinkAndPutSpongeInSinkThenLiftSoapBottle,
    *ROUTINE_TASK_CLASSES,
    *DRAWER_ROUTINE_TASK_CLASSES,
    *CABINET_SOAP_TASK_CLASSES,
    *SCENE422_TASK_CLASSES,
)

__all__ = [task_class.__name__ for task_class in TASK_CLASSES]
