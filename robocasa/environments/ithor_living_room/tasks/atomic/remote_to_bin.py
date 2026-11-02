"""iTHOR living-room task for placing a remote control into the bin."""

import os

from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.coffee_drink_to_bin import (
    IThorLivingRoomCoffeeDrinkToBin,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_to_basket import (
    FixedObjectToBasketMixin,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomRemoteToBin(
    FixedObjectToBasketMixin, IThorLivingRoomCoffeeDrinkToBin
):
    """Pick up a remote from the table and put it into the bin beside G1."""

    SOURCE_NAME = "remote"
    SOURCE_GROUP = None
    SPEC_TASK = "remote_to_bin"
    REMOTE_NAME = SOURCE_NAME
    REMOTE_XML = (
        "models/assets/objects/thor/Living Room Objects/Remotes/Prefabs/"
        "Remote_1/Remote_1.xml"
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _asset_path(self, relative_path):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", relative_path)
        )

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "bin_fixture",
                    "object_pos",
                    "object_yaw",
                    "bin_target_xy",
                    "bin_xy_tolerance",
                    "bin_z_tolerance",
                    "robot_pose",
                    "hold_seconds",
                    "source_category",
                    "source_wrapper_pos",
                    "source_wrapper_quat",
                    "source_bbox_pos",
                    "source_bbox_size",
                    "source_scale",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_obj_cfgs(self):
        spec = self._get_task_spec()
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups=self._asset_path(self.REMOTE_XML),
                object_scale=float(spec["source_scale"]),
            )
        ]

    def _create_objects(self):
        self._remote_temp_xmls = []
        try:
            super()._create_objects()
        finally:
            for path in self._remote_temp_xmls:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            self._remote_temp_xmls = []

    def _place_source_object(self):
        spec = self._get_task_spec()
        set_free_joint_bottom_at(
            self,
            self.objects[self.SOURCE_NAME],
            spec["object_pos"],
            yaw=spec["object_yaw"],
        )
        self.sim.forward()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the remote and put it into the bin beside you"
        return ep_meta


__all__ = ["IThorLivingRoomRemoteToBin"]
