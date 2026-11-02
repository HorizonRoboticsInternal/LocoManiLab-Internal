"""iTHOR living-room environment category (FloorPlan 201-230).

Peer of kitchen / living_room: own runnable base env + registry, backed by a
MolmoSpaces iTHOR living-room as background. Tasks under ``tasks/``.

Thin re-export shim — the base class is built by
:func:`robocasa.environments.ithor_base.make_room_env_category`.
"""

from robocasa.environments.ithor_base import (  # noqa: F401
    IThorLivingRoomEnv,
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
