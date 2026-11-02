from robosuite.environments.base import make

# Benchmark environments are registered by robocasa, not upstream robot demos.

from robosuite.environments import ALL_ENVIRONMENTS
from robosuite.controllers import (
    ALL_PART_CONTROLLERS,
    load_part_controller_config,
    ALL_COMPOSITE_CONTROLLERS,
    load_composite_controller_config,
)
from robosuite.robots import ALL_ROBOTS
from robosuite.models.grippers import ALL_GRIPPERS
from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

# This maintained fork intentionally registers only its bundled G1 variants.
# Do not import external model packages or robot-specific example controllers:
# import-time registration can overwrite G1 with an incompatible implementation.

__version__ = "1.5.2"
__logo__ = """
      ;     /        ,--.
     ["]   ["]  ,<  |__**|
    /[_]\  [~]\/    |//  |
     ] [   OOO      /o|__|
"""
