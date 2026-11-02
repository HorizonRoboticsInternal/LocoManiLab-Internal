"""Ordered multi-stage base for iTHOR living-room composite tasks."""

from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)


IThorLivingRoomAtomicTask = IThorAtomicTask


class IThorLivingRoomMultiStageComposite(
    IThorMultiStageCompositeMixin,
    IThorLivingRoomEnv,
):
    """Room base for ordered iTHOR living-room composite tasks."""

    SUCCESS_HOLD_SECONDS = 1.0
    NAVIGATION_SUCCESS_DISTANCES = {"table": 0.45}


REGISTERED_ITHOR_LIVING_ROOM_ENVS.pop(IThorLivingRoomMultiStageComposite.__name__, None)
REGISTERED_ENVS.pop(IThorLivingRoomMultiStageComposite.__name__, None)


__all__ = [
    "IThorLivingRoomAtomicTask",
    "IThorLivingRoomMultiStageComposite",
]
