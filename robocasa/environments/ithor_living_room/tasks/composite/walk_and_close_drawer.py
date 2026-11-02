"""Navigate to a living-room drawer, then close it."""

from robocasa.environments.ithor_living_room.tasks.atomic.articulation import (
    IThorLivingRoomCloseDrawer,
)
from .drawer_navigation import DrawerNavigationGateMixin


class IThorLivingRoomWalkAndCloseDrawer(
    DrawerNavigationGateMixin, IThorLivingRoomCloseDrawer
):
    SPEC_TASK = "walk_and_close_drawer"
    COMPOSITE_KIND = "navigate_then_close_drawer"
    ATOMIC_SKILL = "ApproachAndCloseDrawer"


__all__ = ["IThorLivingRoomWalkAndCloseDrawer"]
