"""Ordered multi-stage base for iTHOR kitchen composite tasks.

Concrete tasks declare ``ATOMIC_TASKS`` directly or through a task-specific
mixin. Every completed stage prints one terminal message. When combining this
base with another task mixin, keep the task-specific mixin first so its setup
and metadata hooks run before this room base.
"""

from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import (
    IThorKitchenEnv,
    REGISTERED_ITHOR_KITCHEN_ENVS,
)


IThorKitchenAtomicTask = IThorAtomicTask


class IThorKitchenMultiStageComposite(
    IThorMultiStageCompositeMixin,
    IThorKitchenEnv,
):
    """Runnable-room base for ordered iTHOR kitchen composite tasks."""

    SUCCESS_HOLD_SECONDS = 1.0

    def _get_task_spec(self):
        """Use the same parameter interface as the other Kitchen tasks."""
        return self.scene_task_spec()


# The room metaclass registers every environment subclass. This class is an
# implementation base rather than a runnable benchmark task, so keep it out of
# both registries while allowing its concrete subclasses to register normally.
REGISTERED_ITHOR_KITCHEN_ENVS.pop(IThorKitchenMultiStageComposite.__name__, None)
REGISTERED_ENVS.pop(IThorKitchenMultiStageComposite.__name__, None)


__all__ = ["IThorKitchenAtomicTask", "IThorKitchenMultiStageComposite"]
