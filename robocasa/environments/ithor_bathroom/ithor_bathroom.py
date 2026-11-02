"""iTHOR bathroom environment category (FloorPlan 401-430).

Thin re-export shim — the base class is built by
:func:`robocasa.environments.ithor_base.make_room_env_category`.
"""

from robocasa.environments.ithor_base import (  # noqa: F401
    IThorBathroomEnv,
    REGISTERED_ITHOR_BATHROOM_ENVS,
)
