"""Shared base abstractions and the lightweight bases used by G1."""
from .mount_model import MountModel
from .robot_base_model import RobotBaseModel
from .robot_base_factory import robot_base_factory
from .mobile_base_model import MobileBaseModel
from .leg_base_model import LegBaseModel
from .null_base_model import NullBaseModel
from .null_mount import NullMount
from .null_mobile_base import NullMobileBase
from .no_actuation_base import NoActuationBase
from .floating_legged_base import FloatingLeggedBase
from .null_base import NullBase

BASE_MAPPING = {
    "NullMount": NullMount,
    "NullMobileBase": NullMobileBase,
    "NoActuationBase": NoActuationBase,
    "FloatingLeggedBase": FloatingLeggedBase,
    "NullBase": NullBase,
}
ALL_BASES = BASE_MAPPING.keys()


def register_base(target_class):
    BASE_MAPPING[target_class.__name__] = target_class
    return target_class
