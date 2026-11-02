"""Robot model API and the G1 variants maintained by this benchmark."""
from .robot_model import RobotModel, create_robot
from .manipulators import G1, G1FixedLowerBody, G1FloatingBody, ManipulatorModel


def is_robosuite_robot(robot: str) -> bool:
    from .robot_model import REGISTERED_ROBOTS

    return robot in REGISTERED_ROBOTS
