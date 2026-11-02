from .holodeck import HolodeckEnv, REGISTERED_HOLODECK_ENVS
from .tasks.approach_desk import HolodeckApproachDesk
from .tasks.approach_trash_can import HolodeckApproachTrashCan
from .tasks.approach_window import HolodeckApproachWindow
from .tasks.laptop import HolodeckCloseLaptop, HolodeckOpenLaptop
from .tasks.articulation import HolodeckCloseDrawer, HolodeckOpenDrawer
from .tasks.cabinet_door import HolodeckOpenCabinetDoor
from .tasks.alarm_clock import (
    HolodeckPickAlarmClockFromSofa,
    HolodeckPlaceAlarmClockInDrawer,
)
from .tasks.cube_in_plate import HolodeckCubeInPlate
from .tasks.stand_candle_upright import HolodeckStandCandleUpright
from .tasks.ice_cube_in_glass import HolodeckIceCubeInGlass
from .tasks.mug_lid import HolodeckCloseMugLid, HolodeckOpenMugLid
from .tasks.sugar_cube_in_jar import HolodeckSugarCubeInJar
from .tasks.move_cup import (
    HolodeckMoveCupToCenter,
    HolodeckMoveCupToCorner,
)
from .tasks.pick_book_from_dresser import HolodeckPickBookFromDresser
from .tasks.lift_plant_from_low_cabinet import HolodeckLiftPlantFromLowCabinet
from .tasks.pick_cup_from_desk import HolodeckPickCupFromDesk
from .tasks.pick_box_from_floor import HolodeckPickBoxFromFloor
from .tasks.place_box_on_floor import (
    HolodeckPlaceBoxFromDeskToFloor,
    HolodeckPlaceBoxOnFloor,
)
from .tasks.place_box_on_desk import HolodeckPlaceBoxOnDesk
from .tasks.place_cup_on_placemat import HolodeckPlaceCupOnPlacemat
from .tasks.pull_chair_from_desk import HolodeckPullChairFromDesk
from .tasks.push_chair_to_desk import HolodeckPushChairToDesk
from .tasks.throw_bottle_into_trash_can import HolodeckThrowBottleIntoTrashCan
from .tasks.shelf_folder import (
    HolodeckInsertFolderIntoShelf,
    HolodeckRemoveFolderFromShelf,
)
from .tasks.take_tissue_box_from_open_drawer import HolodeckTakeTissueBoxFromOpenDrawer
from .tasks.multi_hop import (
    HolodeckPlaceAlarmClockThenCloseDrawerThenPickBox,
    HolodeckOpenDrawerThenPlaceAlarmClockThenApproachTrashCan,
    HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan,
    HolodeckApproachWindowThenCubeInPlateThenOpenLaptop,
    HolodeckApproachDeskThenCloseLaptopMoveCup,
    HolodeckApproachDeskIceCubePlaceCupThenPickCup,
    HolodeckApproachDeskThenOpenLaptop,
    HolodeckApproachDeskThenPullChair,
    HolodeckApproachDeskPullChairThenOpenLaptop,
    HolodeckApproachWindowThenDeskThenPickCup,
    HolodeckAtomicTask,
    HolodeckIceCubeThenPlaceCup,
    HolodeckMoveCupToCenterThenOpenLaptop,
    HolodeckMultiHopComposite,
    HolodeckOpenDrawerTakeTissueBoxToDesk,
    HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer,
    HolodeckOpenDrawerThenTakeTissueBox,
    HolodeckOpenJarAddSugarThenCloseJar,
    HolodeckPickBookFromSourceToDesk,
    HolodeckRemoveFolderFromShelfToDesk,
    HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop,
    HolodeckCloseLaptopCloseDrawerThenApproachWindow,
    HolodeckApproachTrashCanThenPickBoxFromFloor,
    HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk,
    HolodeckCloseLaptopThenApproachTrashCanThenPickBox,
    HolodeckApproachWindowThenPickBox,
    HolodeckPullChairThenApproachTrashCanThenPickBox,
    HolodeckApproachTrashCanThenThrowBottle,
    HolodeckApproachTrashCanThenThrowBottleThenApproachDesk,
    HolodeckApproachDeskThenOpenLaptopThenApproachWindow,
    HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan,
    HolodeckApproachWindowThenDesk,
    HolodeckApproachTrashCanThenApproachDeskThenPullChair,
    HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop,
    HolodeckTakeTissueBoxThenPlaceCup,
    HolodeckPickBookFromSourceToDeskThenApproachTrashCan,
    HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan,
    HolodeckApproachDeskThenIceCubeInGlass,
    HolodeckCloseJarThenApproachDeskThenApproachTrashCan,
    HolodeckApproachTrashCanThenApproachDeskThenOpenJar,
    HolodeckSugarCubeThenCloseJar,
    HolodeckIceCubeThenApproachWindowThenPickBox,
    HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer,
    HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat,
    HolodeckPlaceBoxOnFloorThenPlaceBoxOnDesk,
    HolodeckPlaceBoxOnFloorThenPickBook,
)

__all__ = [
    "HolodeckPlaceAlarmClockThenCloseDrawerThenPickBox",
    "HolodeckOpenDrawerThenPlaceAlarmClockThenApproachTrashCan",
    "HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan",
    "HolodeckApproachWindowThenCubeInPlateThenOpenLaptop",
    "HolodeckEnv",
    "REGISTERED_HOLODECK_ENVS",
    "HolodeckApproachDesk",
    "HolodeckApproachTrashCan",
    "HolodeckApproachWindow",
    "HolodeckOpenLaptop",
    "HolodeckCloseLaptop",
    "HolodeckOpenDrawer",
    "HolodeckOpenCabinetDoor",
    "HolodeckPickAlarmClockFromSofa",
    "HolodeckPlaceAlarmClockInDrawer",
    "HolodeckCubeInPlate",
    "HolodeckStandCandleUpright",
    "HolodeckCloseDrawer",
    "HolodeckIceCubeInGlass",
    "HolodeckOpenMugLid",
    "HolodeckCloseMugLid",
    "HolodeckSugarCubeInJar",
    "HolodeckMoveCupToCenter",
    "HolodeckMoveCupToCorner",
    "HolodeckPickBookFromDresser",
    "HolodeckLiftPlantFromLowCabinet",
    "HolodeckPickCupFromDesk",
    "HolodeckPickBoxFromFloor",
    "HolodeckPlaceBoxOnFloor",
    "HolodeckPlaceBoxFromDeskToFloor",
    "HolodeckPlaceBoxOnDesk",
    "HolodeckPlaceCupOnPlacemat",
    "HolodeckPullChairFromDesk",
    "HolodeckPushChairToDesk",
    "HolodeckThrowBottleIntoTrashCan",
    "HolodeckInsertFolderIntoShelf",
    "HolodeckRemoveFolderFromShelf",
    "HolodeckTakeTissueBoxFromOpenDrawer",
    "HolodeckAtomicTask",
    "HolodeckMultiHopComposite",
    "HolodeckApproachDeskThenOpenLaptop",
    "HolodeckOpenDrawerThenTakeTissueBox",
    "HolodeckOpenDrawerTakeTissueBoxToDesk",
    "HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer",
    "HolodeckApproachDeskThenCloseLaptopMoveCup",
    "HolodeckSugarCubeThenCloseJar",
    "HolodeckOpenJarAddSugarThenCloseJar",
    "HolodeckRemoveFolderFromShelfToDesk",
    "HolodeckIceCubeThenPlaceCup",
    "HolodeckApproachDeskThenPullChair",
    "HolodeckApproachWindowThenDeskThenPickCup",
    "HolodeckPickBookFromSourceToDesk",
    "HolodeckApproachDeskPullChairThenOpenLaptop",
    "HolodeckMoveCupToCenterThenOpenLaptop",
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
    "HolodeckApproachTrashCanThenThrowBottleThenApproachDesk",
    "HolodeckApproachDeskThenOpenLaptopThenApproachWindow",
    "HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan",
    "HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop",
    "HolodeckApproachWindowThenDesk",
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
