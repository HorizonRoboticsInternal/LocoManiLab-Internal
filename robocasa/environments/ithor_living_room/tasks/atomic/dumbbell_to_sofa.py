"""iTHOR living-room: move a dumbbell onto the sofa."""

from robocasa.environments.ithor_living_room.tasks.atomic.basket_to_sofa import (
    IThorLivingRoomBasketToSofa,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_to_sofa import (
    FixedObjectToSofaMixin,
    RaisedFixedObjectToSofaMixin,
)


class IThorLivingRoomDumbbellToSofa(
    RaisedFixedObjectToSofaMixin,
    FixedObjectToSofaMixin,
    IThorLivingRoomBasketToSofa,
):
    SOURCE_NAME = "dumbbell"
    SOURCE_XML = (
        "models/assets/objects/thor/Miscellaneous Objects/Dumbbell/Prefabs/"
        "Dumbbell_1_1/Dumbbell_1_1.xml"
    )
    SPEC_TASK = "dumbbell_to_sofa"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the dumbbell and place it on the sofa"
        return ep_meta


__all__ = ["IThorLivingRoomDumbbellToSofa"]
