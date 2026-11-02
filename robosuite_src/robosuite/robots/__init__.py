from .fixed_base_robot import FixedBaseRobot
from .mobile_robot import MobileRobot
from .wheeled_robot import WheeledRobot
from .legged_robot import LeggedRobot

from robosuite.models.robots.robot_model import REGISTERED_ROBOTS

from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

ALL_ROBOTS = REGISTERED_ROBOTS.keys()

# Runtime wrappers remain generic; the bundled model catalog is G1-only.
ROBOT_CLASS_MAPPING = {
    "G1": LeggedRobot,
    "G1FixedLowerBody": LeggedRobot,
    "G1FloatingBody": LeggedRobot,
}

target_type_mapping = {
    "FixedBaseRobot": FixedBaseRobot,
    "MobileRobot": MobileRobot,
    "WheeledRobot": WheeledRobot,
    "LeggedRobot": LeggedRobot,
}


def register_robot_class(target_type, **kwargs):
    def decorator(target_class):
        # Store the class in the registry with additional arguments
        ROBOT_CLASS_MAPPING.update(
            {target_class.__name__: target_type_mapping[target_type]}
        )

        return target_class  # Return the class itself

    return decorator
