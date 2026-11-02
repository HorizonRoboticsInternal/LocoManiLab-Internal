"""iTHOR kitchen: stack two cups on a table."""

import numpy as np

from robocasa.environments.ithor_kitchen.tasks.atomic.apple_on_plate import (
    IThorKitchenAppleOnPlate,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class IThorKitchenCupStack(IThorKitchenAppleOnPlate):
    USE_BASE_SUCCESS_HOLD = False
    ATOMIC_SKILL = "CupStack"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "pick_place"
    SPEC_TASK = "cup_stack"

    CUP_NAMES = ("cup_1", "cup_2")
    CUP_XML = "objaverse/cup/cup_4/model.xml"
    CUP_SCALE = 1.2
    STACK_XY_TOL = 0.05
    MIN_STACK_Z_GAP = 0.005
    MAX_STACK_Z_GAP = 0.12
    AXIS_UP_DOT = np.cos(np.deg2rad(45))
    BOTTOM_AXIS_UP_DOT = np.cos(np.deg2rad(10))

    def _get_task_spec(self):
        if getattr(self, "_cup_stack_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    "cup_1_pos",
                    "cup_2_pos",
                    "hold_seconds",
                ),
            )
            self._cup_stack_spec_scene = self.scene_name
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
                name=name,
                info=dict(mjcf_path=self.CUP_XML),
                object_scale=self.CUP_SCALE,
            )
            for name in self.CUP_NAMES
        ]

    def _place_task_objects(self):
        spec = self._get_task_spec()
        for name in self.CUP_NAMES:
            pos = np.asarray(spec[f"{name}_pos"], dtype=float)
            if pos.shape != (3,):
                raise ValueError(
                    f"{self.scene_name}/{self.SPEC_TASK}: {name}_pos needs XYZ"
                )
            self._set_free_joint_bottom_at(self.objects[name], pos[:2], pos[2])
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        self.stack_hold_seconds = float(self._get_task_spec()["hold_seconds"])
        self._stack_hold_steps = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._cups_are_stacked():
            self._stack_hold_steps += 1
        else:
            self._stack_hold_steps = 0
        return ret

    def _check_success(self):
        return self._stack_hold_steps >= self.stack_hold_steps

    @property
    def stack_hold_steps(self):
        return max(1, int(round(self.stack_hold_seconds * self.control_freq)))

    def _cups_are_stacked(self):
        bounds = {name: self._cup_bounds(name) for name in self.CUP_NAMES}
        first, second = self.CUP_NAMES
        lower, upper = (
            (first, second)
            if bounds[first][0][2] <= bounds[second][0][2]
            else (second, first)
        )
        stacked = self._pair_is_stacked(lower, upper, bounds)
        upright = all(
            self._axis_up(name) >= self.AXIS_UP_DOT for name in self.CUP_NAMES
        )
        bottom_upright = self._axis_up(lower) >= self.BOTTOM_AXIS_UP_DOT
        cups_touch = self.check_contact(self.objects[first], self.objects[second])
        return bool(stacked and upright and bottom_upright and cups_touch)

    def _cup_bounds(self, name):
        points = self.object_bbox_points(name)
        return np.mean(points, axis=0), np.ptp(points, axis=0)

    def _pair_is_stacked(self, lower, upper, bounds):
        lower_center, lower_size = bounds[lower]
        upper_center, upper_size = bounds[upper]
        xy_dis = np.linalg.norm(upper_center[:2] - lower_center[:2])
        z_dis = upper_center[2] - lower_center[2]
        cup_height = min(float(lower_size[2]), float(upper_size[2]))
        xy_tol = self.STACK_XY_TOL * float(self.CUP_SCALE)
        min_z_gap = max(self.MIN_STACK_Z_GAP, 0.02 * cup_height)
        max_z_gap = max(
            self.MAX_STACK_Z_GAP * float(self.CUP_SCALE),
            0.75 * cup_height,
        )
        return bool(xy_dis <= xy_tol and min_z_gap <= z_dis <= max_z_gap)

    def _axis_up(self, name):
        body_id = self.obj_body_id[name]
        mat = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
        return float(np.dot(mat[:, 2], np.array([0.0, 0.0, 1.0])))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "stack the two cups on the table"
        return ep_meta


__all__ = ["IThorKitchenCupStack"]
