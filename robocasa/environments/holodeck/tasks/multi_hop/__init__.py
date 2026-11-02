"""Organized multi-hop task families for the Holodeck benchmark.

The modules are grouped by the fixture or workflow they calibrate.  Concrete
task classes remain available from ``robocasa.environments.holodeck.tasks``.
"""

from .base import HolodeckAtomicTask, HolodeckMultiHopComposite
from .cube_laptop_workflow import HolodeckApproachWindowThenCubeInPlateThenOpenLaptop
from .desk_workflows import (
    HolodeckApproachDeskThenCloseDrawer,
    HolodeckApproachDeskThenMoveCupToCenter,
    HolodeckApproachDeskThenOpenJar,
    HolodeckApproachDeskThenPickCup,
)
from .object_workflows import (
    HolodeckApproachDeskThenCloseLaptopMoveCup,
    HolodeckApproachDeskThenOpenLaptop,
    HolodeckApproachDeskPullChairThenOpenLaptop,
    HolodeckOpenDrawerTakeTissueBoxToDesk,
    HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
    HolodeckOpenDrawerThenTakeTissueBox,
    HolodeckOpenJarAddSugarThenCloseJar,
    HolodeckPickBookFromSourceToDesk,
    HolodeckTakeTissueBoxThenPlaceCup,
    HolodeckPickBookFromSourceToDeskThenApproachTrashCan,
    HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan,
    HolodeckApproachDeskThenIceCubeInGlass,
    HolodeckCloseJarThenApproachDeskThenApproachTrashCan,
    HolodeckApproachTrashCanThenApproachDeskThenOpenJar,
    HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop,
    HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop,
    HolodeckCloseLaptopCloseDrawerThenApproachWindow,
    HolodeckApproachTrashCanThenPickBoxFromFloor,
    HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk,
    HolodeckCloseLaptopThenApproachTrashCanThenPickBox,
    HolodeckApproachWindowThenPickBox,
    HolodeckPullChairThenApproachTrashCanThenPickBox,
    HolodeckRemoveFolderFromShelfToDesk,
    HolodeckSugarCubeThenCloseJar,
    HolodeckMoveCupToCenterThenOpenLaptop,
    HolodeckIceCubeThenApproachWindowThenPickBox,
    HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer,
    HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat,
    HolodeckPlaceBoxOnFloorThenPlaceBoxOnDesk,
    HolodeckPlaceBoxOnFloorThenPickBook,
)
from .navigation import (
    HolodeckApproachDeskThenPullChair,
    HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan,
    HolodeckApproachTrashCanThenApproachDeskThenPullChair,
    HolodeckApproachDeskThenOpenLaptopThenApproachWindow,
    HolodeckApproachTrashCanThenThrowBottle,
    HolodeckApproachTrashCanThenThrowBottleThenApproachDesk,
    HolodeckApproachWindowThenDesk,
    HolodeckApproachWindowThenDeskThenPickCup,
    HolodeckIceCubeThenPlaceCup,
    HolodeckApproachDeskIceCubePlaceCupThenPickCup,
)

from .plant_to_trash import (
    HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan,
)

from .alarm_clock_workflows import (
    HolodeckPlaceAlarmClockThenCloseDrawerThenPickBox,
    HolodeckOpenDrawerThenPlaceAlarmClockThenApproachTrashCan,
)

__all__ = [
    "HolodeckPlaceAlarmClockThenCloseDrawerThenPickBox",
    "HolodeckOpenDrawerThenPlaceAlarmClockThenApproachTrashCan",
    "HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan",
    "HolodeckApproachWindowThenCubeInPlateThenOpenLaptop",
    "HolodeckAtomicTask",
    "HolodeckMultiHopComposite",
    "HolodeckApproachDeskThenPickCup",
    "HolodeckApproachDeskThenMoveCupToCenter",
    "HolodeckApproachDeskThenCloseDrawer",
    "HolodeckApproachDeskThenOpenJar",
    "HolodeckApproachDeskThenCloseLaptopMoveCup",
    "HolodeckApproachDeskThenOpenLaptop",
    "HolodeckOpenDrawerTakeTissueBoxToDesk",
    "HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer",
    "HolodeckOpenDrawerThenTakeTissueBox",
    "HolodeckOpenJarAddSugarThenCloseJar",
    "HolodeckSugarCubeThenCloseJar",
    "HolodeckApproachDeskThenPullChair",
    "HolodeckApproachWindowThenDeskThenPickCup",
    "HolodeckIceCubeThenPlaceCup",
    "HolodeckRemoveFolderFromShelfToDesk",
    "HolodeckApproachDeskPullChairThenOpenLaptop",
    "HolodeckMoveCupToCenterThenOpenLaptop",
    "HolodeckPickBookFromSourceToDesk",
    "HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop",
    "HolodeckApproachDeskIceCubePlaceCupThenPickCup",
    "HolodeckCloseLaptopCloseDrawerThenApproachWindow",
    "HolodeckApproachTrashCanThenPickBoxFromFloor",
    "HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk",
    "HolodeckCloseLaptopThenApproachTrashCanThenPickBox",
    "HolodeckApproachWindowThenPickBox",
    "HolodeckPullChairThenApproachTrashCanThenPickBox",
    "HolodeckTakeTissueBoxThenPlaceCup",
    "HolodeckApproachTrashCanThenThrowBottle",
    "HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop",
    "HolodeckApproachWindowThenDesk",
    "HolodeckApproachTrashCanThenThrowBottleThenApproachDesk",
    "HolodeckApproachDeskThenOpenLaptopThenApproachWindow",
    "HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan",
    "HolodeckApproachTrashCanThenApproachDeskThenPullChair",
    "HolodeckPickBookFromSourceToDeskThenApproachTrashCan",
    "HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan",
    "HolodeckApproachDeskThenIceCubeInGlass",
    "HolodeckCloseJarThenApproachDeskThenApproachTrashCan",
    "HolodeckApproachTrashCanThenApproachDeskThenOpenJar",
    "HolodeckIceCubeThenApproachWindowThenPickBox",
    "HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer",
    "HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat",
    "HolodeckPlaceBoxOnFloorThenPlaceBoxOnDesk",
]
