"""The G1 hands are part of its MJCF; only a null attachment is needed."""
from .gripper_model import GripperModel
from .gripper_factory import gripper_factory
from .null_gripper import NullGripper

GRIPPER_MAPPING = {None: NullGripper}
ALL_GRIPPERS = GRIPPER_MAPPING.keys()


def register_gripper(target_class):
    GRIPPER_MAPPING[target_class.__name__] = target_class
    return target_class
