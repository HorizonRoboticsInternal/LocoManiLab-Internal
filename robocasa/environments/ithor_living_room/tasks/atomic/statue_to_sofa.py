"""iTHOR living-room: move a statue onto the sofa."""

import numpy as np

from robocasa.environments.ithor_kitchen.tasks.atomic.objects_into_box import (
    LivingRoomStatueObject,
)
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_oriented_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.basket_to_sofa import (
    IThorLivingRoomBasketToSofa,
)
from robocasa.environments.ithor_living_room.tasks.atomic.lift_statue import (
    IThorLivingRoomLiftStatue,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_to_sofa import (
    RaisedFixedObjectToSofaMixin,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomStatueToSofa(
    RaisedFixedObjectToSofaMixin, IThorLivingRoomBasketToSofa
):
    SOURCE_NAME = "statue"
    SOURCE_XML = IThorLivingRoomLiftStatue.TARGET_XML
    SOURCE_SCALE = IThorLivingRoomLiftStatue.TARGET_SCALE
    SPEC_TASK = "statue_to_sofa"

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=("sofa_fixture", "robot_pose", "object_pos", "object_yaw"),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _create_objects(self):
        self.object_cfgs = self._get_obj_cfgs()
        statue = LivingRoomStatueObject(
            name=self.SOURCE_NAME,
            source_xml=self._source_xml_path(),
            scale=self.SOURCE_SCALE,
        )
        self.objects = {statue.name: statue}
        self.model.merge_objects([statue])

    def _reset_internal(self):
        IThorLivingRoomEnv._reset_internal(self)
        spec = self._get_task_spec()
        set_free_joint_oriented_bottom_at(
            self,
            self.objects[self.SOURCE_NAME],
            self._work_surface_bottom_position(spec["object_pos"]),
            yaw=spec["object_yaw"],
            # Match LiftStatue: rotate the asset's local Y axis upright.
            pitch=np.pi / 2.0,
        )
        self.sim.forward()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the statue and place it on the sofa"
        return ep_meta


__all__ = ["IThorLivingRoomStatueToSofa"]
