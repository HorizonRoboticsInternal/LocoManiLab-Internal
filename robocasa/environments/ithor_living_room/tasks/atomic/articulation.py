"""iTHOR living-room articulation tasks."""

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomOpenDrawer(RoomOpenArticulation, IThorLivingRoomEnv):
    SUPPORTED_SCENE_IDS = (209, 210, 226)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_livingroom_task_spec)
    SPEC_TASK = "open_drawer"
    HOLD_SECONDS = 1.0
    ATOMIC_SKILL = "OpenDrawer"
    BEHAVIOR = "open"
    TARGET_CATEGORIES = ("Drawer", "Dresser")
    TARGET_LANG = "drawer"
    TARGET_JOINT_ROLES = ("drawer",)
    TARGET_JOINT_TYPES = ("slide",)
    OPEN_STAND_DISTANCE = 0.45


class IThorLivingRoomCloseDrawer(RoomOpenArticulation, IThorLivingRoomEnv):
    SUPPORTED_SCENE_IDS = (209, 210, 226)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_ithor_livingroom_task_spec)
    SPEC_TASK = "close_drawer"
    HOLD_SECONDS = 1.0
    ATOMIC_SKILL = "CloseDrawer"
    BEHAVIOR = "close"
    TARGET_CATEGORIES = ("Drawer", "Dresser")
    TARGET_LANG = "drawer"
    TARGET_JOINT_ROLES = ("drawer",)
    TARGET_JOINT_TYPES = ("slide",)
    CLOSE_STAND_DISTANCE = 0.45


__all__ = ["IThorLivingRoomOpenDrawer", "IThorLivingRoomCloseDrawer"]
