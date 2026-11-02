"""iTHOR living-room: lift a dumbbell from a table."""

from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_lift import (
    FixedObjectLiftMixin,
)


class IThorLivingRoomLiftDumbbell(
    FixedObjectLiftMixin,
    RoomLiftMixin,
    IThorLivingRoomEnv,
):
    """Pick up a dumbbell from the table edge and lift it."""

    TARGET_NAME = "dumbbell"
    TARGET_XML = (
        "models/assets/objects/thor/Miscellaneous Objects/Dumbbell/Prefabs/"
        "Dumbbell_1_1/Dumbbell_1_1.xml"
    )
    SPEC_TASK = "lift_dumbbell"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        support = "support surface" if self.scene_name == "FloorPlan210" else "table"
        ep_meta["lang"] = f"lift the dumbbell from the edge of the {support}"
        return ep_meta


__all__ = ["IThorLivingRoomLiftDumbbell"]
