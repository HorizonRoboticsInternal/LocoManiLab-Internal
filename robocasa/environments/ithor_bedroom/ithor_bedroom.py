"""iTHOR bedroom environment category (FloorPlan 301-330).

Thin re-export shim — the base class is built by
:func:`robocasa.environments.ithor_base.make_room_env_category`.
"""

from robocasa.environments.ithor_base import (  # noqa: F401
    IThorBedroomEnv,
    REGISTERED_ITHOR_BEDROOM_ENVS,
)
