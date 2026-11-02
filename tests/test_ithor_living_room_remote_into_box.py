"""Focused registration and task identity checks for L06."""

from robocasa.environments.ithor_living_room import (
    IThorLivingRoomRemoteIntoBox as Task,
    REGISTERED_ITHOR_LIVING_ROOM_ENVS,
)
from robosuite.environments.base import REGISTERED_ENVS
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


def test_registered_scene226_only():
    for registry in (REGISTERED_ENVS, REGISTERED_ITHOR_LIVING_ROOM_ENVS):
        assert registry[Task.__name__] is Task
    assert Task.SUPPORTED_SCENE_IDS == (226,)


def test_uses_scene226_native_remote_and_box():
    assert Task.OBJECT_BODY.startswith("remotecontrol_bbeb7464")
    assert Task.BOX_BODY.startswith("box_b1c47d670")
    assert len(Task.FLAP_JOINTS) == 4
    spec = get_ithor_livingroom_task_spec("FloorPlan226", "remote_into_box")
    assert spec["object_quat"] == [1.0, 0.0, 0.0, 0.0]
