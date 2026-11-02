"""Navigate to a living-room drawer, then open it."""

from robocasa.environments.ithor_living_room.tasks.atomic.articulation import (
    IThorLivingRoomOpenDrawer,
)
from .drawer_navigation import DrawerNavigationGateMixin


class IThorLivingRoomWalkAndOpenDrawer(
    DrawerNavigationGateMixin, IThorLivingRoomOpenDrawer
):
    SPEC_TASK = "walk_and_open_drawer"
    COMPOSITE_KIND = "navigate_then_open_drawer"
    ATOMIC_SKILL = "ApproachAndOpenDrawer"


__all__ = ["IThorLivingRoomWalkAndOpenDrawer"]
