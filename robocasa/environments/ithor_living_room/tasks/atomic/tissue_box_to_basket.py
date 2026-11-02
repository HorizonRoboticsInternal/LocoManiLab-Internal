"""iTHOR living-room: place a tissue box into a basket."""

from robocasa.environments.ithor_living_room.tasks.atomic.remote_to_basket import (
    IThorLivingRoomRemoteToBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_to_basket import (
    FixedObjectToBasketMixin,
)


class IThorLivingRoomTissueBoxToBasket(
    FixedObjectToBasketMixin, IThorLivingRoomRemoteToBasket
):
    """Pick up a tissue box and place it into a basket."""

    REMOTE_NAME = "tissue_box"
    REMOTE_XML = (
        "models/assets/objects/thor/Miscellaneous Objects/TissueBox/Prefabs/"
        "Tissue_Box_1/Tissue_Box_1.xml"
    )
    SPEC_TASK = "tissue_box_to_basket"

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the tissue box and place it in the basket"
        return ep_meta


__all__ = ["IThorLivingRoomTissueBoxToBasket"]
