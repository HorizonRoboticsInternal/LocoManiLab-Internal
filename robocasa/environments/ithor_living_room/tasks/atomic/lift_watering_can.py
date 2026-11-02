"""iTHOR living-room: lift a watering can from a table."""

from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_lift import (
    FixedObjectLiftMixin,
)


class IThorLivingRoomLiftWateringCan(
    FixedObjectLiftMixin,
    RoomLiftMixin,
    IThorLivingRoomEnv,
):
    """Pick up a watering can from the table edge and lift it."""

    TARGET_NAME = "watering_can"
    TARGET_XML = (
        "models/assets/objects/thor/Miscellaneous Objects/WateringCan/Prefabs/"
        "Watering_Can_1/Watering_Can_1.xml"
    )
    SPEC_TASK = "lift_watering_can"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        support = "support surface" if self.scene_name == "FloorPlan210" else "table"
        ep_meta["lang"] = f"lift the watering can from the edge of the {support}"
        return ep_meta


__all__ = ["IThorLivingRoomLiftWateringCan"]
