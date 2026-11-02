"""ProcTHOR semantic open-door task."""

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.procthor.procthor import ProcTHOREnv


class ProcTHOROpenDoor(RoomOpenArticulation, ProcTHOREnv):
    ATOMIC_SKILL = "OpenDoor"
    BEHAVIOR = "open"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "articulation"
    TARGET_CATEGORIES = ("Doorway", "Door")
    TARGET_LANG = "door"
    TARGET_JOINT_ROLES = ("door",)
    OPEN_STAND_DISTANCE = 0.8


__all__ = ["ProcTHOROpenDoor"]
