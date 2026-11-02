"""ProcTHOR semantic open-drawer task."""

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.procthor.procthor import ProcTHOREnv


class ProcTHOROpenDrawer(RoomOpenArticulation, ProcTHOREnv):
    ATOMIC_SKILL = "OpenDrawer"
    BEHAVIOR = "open"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "articulation"
    TARGET_CATEGORIES = ("Dresser", "Drawer")
    TARGET_LANG = "dresser drawer"
    TARGET_JOINT_ROLES = ("drawer",)
    TARGET_JOINT_TYPES = ("slide",)
    OPEN_STAND_DISTANCE = 0.7
    DESCRIBE_TARGET_JOINT_LEVEL = True
    TARGET_JOINT_LEVELS = ("top", "middle", "bottom")
    RANDOMIZE_TARGET_JOINT = True
    ARTICULATION_STAND_POSE_JITTER = True


__all__ = ["ProcTHOROpenDrawer"]
