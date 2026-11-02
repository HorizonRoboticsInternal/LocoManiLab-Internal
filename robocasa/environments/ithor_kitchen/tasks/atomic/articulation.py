"""iTHOR kitchen articulation atomic tasks."""

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class IThorKitchenOpenFridge(RoomOpenArticulation, IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "open_fridge"
    ATOMIC_SKILL = "OpenFridge"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Fridge",)
    TARGET_LANG = "fridge door"
    TARGET_JOINT_ROLES = ("door",)
    OPEN_STAND_DISTANCE = 0.05


class IThorKitchenOpenCabinet(RoomOpenArticulation, IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "open_cabinet"
    ATOMIC_SKILL = "OpenCabinet"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Cabinet",)
    TARGET_LANG = "cabinet door"
    TARGET_JOINT_ROLES = ("door",)
    SUCCESS_JOINT_SELECTION = "matching"
    OPEN_STAND_DISTANCE = 0.35


class IThorKitchenOpenDrawer(RoomOpenArticulation, IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "open_drawer"
    ATOMIC_SKILL = "OpenDrawer"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Drawer",)
    TARGET_LANG = "drawer"
    TARGET_JOINT_ROLES = ("drawer",)
    TARGET_JOINT_TYPES = ("slide",)
    OPEN_STAND_DISTANCE = 0.45


class IThorKitchenCloseFridge(RoomOpenArticulation, IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "close_fridge"
    ATOMIC_SKILL = "CloseFridge"
    BEHAVIOR = "close"
    TARGET_CATEGORIES = ("Fridge",)
    TARGET_LANG = "fridge door"
    TARGET_JOINT_ROLES = ("door",)
    CLOSE_STAND_DISTANCE = 0.05


class IThorKitchenCloseCabinet(RoomOpenArticulation, IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "close_cabinet"
    ATOMIC_SKILL = "CloseCabinet"
    BEHAVIOR = "close"
    TARGET_CATEGORIES = ("Cabinet",)
    TARGET_LANG = "cabinet door"
    TARGET_JOINT_ROLES = ("door",)
    SUCCESS_JOINT_SELECTION = "matching"
    CLOSE_STAND_DISTANCE = 0.25


class IThorKitchenCloseDrawer(RoomOpenArticulation, IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "close_drawer"
    ATOMIC_SKILL = "CloseDrawer"
    BEHAVIOR = "close"
    TARGET_CATEGORIES = ("Drawer",)
    TARGET_LANG = "drawer"
    TARGET_JOINT_ROLES = ("drawer",)
    TARGET_JOINT_TYPES = ("slide",)
    CLOSE_STAND_DISTANCE = 0.20


class IThorKitchenOpenOven(RoomOpenArticulation, IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "open_oven"
    ATOMIC_SKILL = "OpenOven"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Oven",)
    TARGET_LANG = "oven door"
    TARGET_JOINT_ROLES = ("door",)
    OPEN_STAND_DISTANCE = 0.55


class IThorKitchenCloseOven(RoomOpenArticulation, IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_kitchen_task_spec)
    SPEC_TASK = "close_oven"
    ATOMIC_SKILL = "CloseOven"
    BEHAVIOR = "close"
    TARGET_CATEGORIES = ("Oven",)
    TARGET_LANG = "oven door"
    TARGET_JOINT_ROLES = ("door",)
    CLOSE_STAND_DISTANCE = 0.35
    OPEN_INIT_FRACTION = 0.20
    SUCCESS_CLOSE_FRACTION = 0.10
