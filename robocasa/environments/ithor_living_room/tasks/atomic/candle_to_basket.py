"""iTHOR living-room: place a candle into a basket."""

from robocasa.environments.ithor_living_room.tasks.atomic.remote_to_basket import (
    IThorLivingRoomRemoteToBasket,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_to_basket import (
    FixedObjectToBasketMixin,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.raised_work_surface import (
    TableSizedRaisedSurfaceMixin,
)


class IThorLivingRoomCandleToBasket(
    TableSizedRaisedSurfaceMixin,
    FixedObjectToBasketMixin,
    IThorLivingRoomRemoteToBasket,
):
    """Pick up a candle and place it into a basket."""

    REMOTE_NAME = "candle"
    REMOTE_XML = (
        "models/assets/objects/thor/Miscellaneous Objects/Candle/Prefabs/"
        "Candle_1/Candle_1.xml"
    )
    SPEC_TASK = "candle_to_basket"

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self._setup_table_sized_work_surface([spec["remote_pos"]])

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and getattr(self, "work_surface_region", None) is not None
        )

    def _reset_internal(self):
        super()._reset_internal()
        spec = self._get_task_spec()
        for name, pos_key, yaw_key in (
            (self.BASKET_NAME, "basket_pos", "basket_yaw"),
            (self.REMOTE_NAME, "remote_pos", "remote_yaw"),
        ):
            set_free_joint_bottom_at(
                self,
                self.objects[name],
                self._work_surface_bottom_position(spec[pos_key]),
                yaw=spec[yaw_key],
            )
        self.sim.forward()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the candle and place it in the basket"
        return ep_meta


__all__ = ["IThorLivingRoomCandleToBasket"]
