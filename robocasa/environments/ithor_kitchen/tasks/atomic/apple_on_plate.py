"""iTHOR kitchen: place an apple on a plate."""

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robocasa.utils import object_utils as OU


class IThorKitchenAppleOnPlate(IThorKitchenEnv):
    ATOMIC_SKILL = "AppleOnPlate"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "pick_place"
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SPEC_TASK = "apple_on_plate"
    SOURCE_NAME = "apple"
    SOURCE_GROUP = "apple"
    SOURCE_LANG = "apple"

    TABLE_MARGIN = 0.05
    EDGE_MARGIN = 0.025
    EDGE_FRACTION = 0.98
    APPLE_ON_PLATE_XY_TOL = 0.13
    USE_BASE_SUCCESS_HOLD = True

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        if getattr(self, "_apple_on_plate_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "plate_pos",
                    "apple_pos",
                    "hold_seconds",
                ),
            )
            self._apple_on_plate_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _get_obj_cfgs(self):
        return [
            dict(
                name="plate",
                obj_groups="plate",
            ),
            dict(
                name=self.SOURCE_NAME,
                obj_groups=self.SOURCE_GROUP,
            ),
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualAppleOnPlateSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _reset_internal(self):
        super()._reset_internal()
        self.success_hold_seconds = float(self._get_task_spec()["hold_seconds"])
        self._place_task_objects()
        self._success_hold_steps = 0

    def _place_task_objects(self):
        plate_pos, apple_pos = self._configured_object_positions()

        self._set_free_joint_bottom_at(
            self.objects["plate"], plate_pos[:2], plate_pos[2]
        )
        self._set_free_joint_bottom_at(
            self.objects[self.SOURCE_NAME], apple_pos[:2], apple_pos[2]
        )
        self.sim.forward()

    def _configured_object_positions(self):
        spec = self._get_task_spec()
        plate_pos = np.asarray(spec["plate_pos"], dtype=float)
        apple_pos = np.asarray(spec["apple_pos"], dtype=float)
        if plate_pos.shape != (3,) or apple_pos.shape != (3,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: "
                "plate_pos and apple_pos must contain XYZ"
            )
        return plate_pos, apple_pos

    def _edge_layout_xy(self):
        plate_pos, apple_pos = self._configured_object_positions()
        return plate_pos[:2], apple_pos[:2]

    def _set_free_joint_bottom_at(self, obj, xy, bottom_z):
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        qpos[:2] = np.asarray(xy, dtype=float)[:2]
        qpos[2] = float(bottom_z - obj.bottom_offset[2])
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        try:
            qvel_addr = self.sim.model.get_joint_qvel_addr(obj.joints[0])
        except Exception:
            qvel_addr = None
        if qvel_addr is not None:
            if isinstance(qvel_addr, tuple):
                self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
            else:
                self.sim.data.qvel[qvel_addr] = 0.0

    def _raw_success(self):
        return bool(
            OU.check_obj_in_receptacle(
                self, self.SOURCE_NAME, "plate", th=self.APPLE_ON_PLATE_XY_TOL
            )
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self.USE_BASE_SUCCESS_HOLD:
            self._success_hold_steps = (
                self._success_hold_steps + 1 if self._raw_success() else 0
            )
        return ret

    def _check_success(self):
        required_steps = max(
            1, int(round(self.success_hold_seconds * self.control_freq))
        )
        return self._success_hold_steps >= required_steps

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = f"put the {self.SOURCE_LANG} on the plate on the table"
        return ep_meta


__all__ = ["IThorKitchenAppleOnPlate"]
