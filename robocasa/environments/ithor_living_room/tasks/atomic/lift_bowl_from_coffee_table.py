"""Scene 201: lift the native bowl from the coffee table."""

import math

import numpy as np

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomLiftBowlFromCoffeeTable(IThorLivingRoomEnv):
    SUPPORTED_SCENE_IDS = (201,)
    SPEC_TASK = "lift_bowl_from_coffee_table"
    ATOMIC_SKILL = "LiftObject"
    TASK_FAMILY = "lift"
    DIFFICULTY_LEVEL = "L1"
    INITIAL_POSE_JITTER = False

    BOWL_BODY = "bowl_6befd62f08fd322391939c2b44d3f839_1_0_0"
    BOWL_JOINT = BOWL_BODY + "_jntfree_0"

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        return get_ithor_livingroom_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("robot_pose", "lift_height", "hold_seconds"),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _setup_references(self):
        super()._setup_references()
        self._bowl_body_id = self.sim.model.body_name2id(self.BOWL_BODY)

    def _bowl_root_z(self):
        return float(self.sim.data.body_xpos[self._bowl_body_id, 2])

    def _raw_success(self):
        return bool(
            self._bowl_root_z()
            >= self._bowl_initial_z + float(self._get_task_spec()["lift_height"])
        )

    def _reset_internal(self):
        super()._reset_internal()
        self.sim.forward()
        self._bowl_initial_z = self._bowl_root_z()
        self._success_hold_steps = 0

    def _post_action(self, action):
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return super()._post_action(action)

    def _check_success(self):
        required = math.ceil(
            float(self._get_task_spec()["hold_seconds"]) * self.control_freq
        )
        return self._success_hold_steps >= required

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="lift the bowl from the coffee table",
            task_id=type(self).__name__,
            source_entity="FloorPlan201:Bowl_1_8",
            source_support="FloorPlan201:Coffee_Table_201_1_0",
            success_definition="bowl root at least 30 cm above its initial height continuously for 1 second",
        )
        return meta


__all__ = ["IThorLivingRoomLiftBowlFromCoffeeTable"]
