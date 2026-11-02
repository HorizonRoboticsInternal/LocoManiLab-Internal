"""iTHOR living-room: pick up a statue and lift it above a target height."""

import os

import numpy as np

from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_kitchen.tasks.atomic.objects_into_box import (
    LivingRoomStatueObject,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_oriented_bottom_at,
)
from robocasa.environments.ithor_living_room.tasks.atomic.object_lift import (
    FixedObjectLiftMixin,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomLiftStatue(
    FixedObjectLiftMixin, RoomLiftMixin, IThorLivingRoomEnv
):
    """Pick up a vertically staged statue from a cleared table edge and lift it."""

    TARGET_NAME = "statue"
    TARGET_SCALE = 0.5
    TARGET_XML = (
        "models/assets/objects/thor/Miscellaneous Objects/Statue/Prefabs/"
        "Stone_Statue_4/Stone_Statue_4.xml"
    )
    SPEC_TASK = "lift_statue"

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "statue_pos",
                    "statue_yaw",
                    "lift_height",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.TARGET_NAME,
                obj_groups=self.TARGET_XML,
                object_scale=self.TARGET_SCALE,
            )
        ]

    def _create_objects(self):
        self.objects = {}
        self.object_cfgs = self._get_obj_cfgs()
        source_xml = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.TARGET_XML)
        )
        statue = LivingRoomStatueObject(
            name=self.TARGET_NAME,
            source_xml=source_xml,
            scale=self.TARGET_SCALE,
        )
        self.objects[statue.name] = statue
        self.model.merge_objects([statue])

    def _set_target_pose_from_spec(self):
        spec = self._get_task_spec()
        set_free_joint_oriented_bottom_at(
            self,
            self.objects[self.TARGET_NAME],
            spec["statue_pos"],
            yaw=spec["statue_yaw"],
            # Stone_Statue_4's long axis is local Y; +90 deg about X makes it vertical.
            pitch=np.pi / 2.0,
        )
        self.sim.forward()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        support = "support surface" if self.scene_name == "FloorPlan210" else "table"
        ep_meta["lang"] = f"lift the statue from the edge of the {support}"
        return ep_meta


__all__ = ["IThorLivingRoomLiftStatue"]
