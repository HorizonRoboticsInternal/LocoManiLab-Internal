"""iTHOR living-room task for pushing a basketball across a table."""

import numpy as np

from robocasa.environments.ithor_atomic.lift import RoomLiftMixin
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.tasks.atomic.object_lift import (
    FixedObjectLiftMixin,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomPushBasketball(
    FixedObjectLiftMixin,
    RoomLiftMixin,
    IThorLivingRoomEnv,
):
    """Push a basketball a calibrated planar distance without dropping it."""

    ATOMIC_SKILL = "PushObject"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "push"
    TARGET_NAME = "basketball"
    TARGET_XML = (
        "models/assets/objects/thor/Bedroom Objects/Basketball/Prefab/"
        "Basketball_1/Basketball_1.xml"
    )
    SPEC_TASK = "push_basketball"

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "object_pos",
                    "object_yaw",
                    "push_distance",
                    "max_height_change",
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

    def _setup_task_fixtures(self):
        IThorLivingRoomEnv._setup_task_fixtures(self)
        spec = self._get_task_spec()
        self.PUSH_DISTANCE = float(spec["push_distance"])
        self.MAX_HEIGHT_CHANGE = float(spec["max_height_change"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])
        if min(self.PUSH_DISTANCE, self.MAX_HEIGHT_CHANGE, self.HOLD_SECONDS) <= 0:
            raise ValueError(f"{self.scene_name}/{self.SPEC_TASK}: invalid threshold")

    def _reset_internal(self):
        IThorLivingRoomEnv._reset_internal(self)
        self._set_target_pose_from_spec()
        self._push_initial_pos = self._basketball_pos().copy()
        self._basketball_hand_contacted = False
        self._push_hold_steps = 0

    def _basketball_pos(self):
        root = self.objects[self.TARGET_NAME].root_body
        return np.asarray(self.sim.data.get_body_xpos(root), dtype=float)

    def _raw_push_success(self):
        delta = self._basketball_pos() - self._push_initial_pos
        return bool(
            self._basketball_hand_contacted
            and np.linalg.norm(delta[:2]) >= self.PUSH_DISTANCE
            and abs(float(delta[2])) <= self.MAX_HEIGHT_CHANGE
        )

    def _post_action(self, action):
        ret = IThorLivingRoomEnv._post_action(self, action)
        if any(
            self.g1_hand_object_contact(side, self.TARGET_NAME)
            for side in ("left", "right")
        ):
            self._basketball_hand_contacted = True
        self._push_hold_steps = (
            self._push_hold_steps + 1 if self._raw_push_success() else 0
        )
        return ret

    def _check_success(self):
        return self._push_hold_steps >= int(
            round(self.HOLD_SECONDS * self.control_freq)
        )

    def get_ep_meta(self):
        ep_meta = IThorLivingRoomEnv.get_ep_meta(self)
        support = "support surface" if self.scene_name == "FloorPlan210" else "table"
        ep_meta.update(
            lang=f"push the basketball across the {support}",
            push_distance=self.PUSH_DISTANCE,
        )
        return ep_meta


__all__ = ["IThorLivingRoomPushBasketball"]
