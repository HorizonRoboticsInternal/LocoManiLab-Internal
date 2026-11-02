from robocasa.environments.ithor_bedroom.ithor_bedroom import (
    IThorBedroomEnv,
    REGISTERED_ITHOR_BEDROOM_ENVS,
)

from . import tasks as _tasks
from .tasks import *  # noqa: F403

TASK_CLASSES = _tasks.TASK_CLASSES


__all__ = [
    "IThorBedroomEnv",
    "REGISTERED_ITHOR_BEDROOM_ENVS",
    *_tasks.__all__,
]
