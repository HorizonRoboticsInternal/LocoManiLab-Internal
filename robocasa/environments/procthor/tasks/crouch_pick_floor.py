"""ProcTHOR task: crouch down in place."""

import numpy as np

from robocasa.environments.procthor.tasks.place_to_place import (
    ProcTHORPlaceToPlaceTask,
    room_label,
)


class ProcTHORCrouchPickCylinder(ProcTHORPlaceToPlaceTask):
    ATOMIC_SKILL = "Crouch"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "crouch"

    ROBOT_SAFE_HALF = 0.10
    CROUCH_HEAD_DELTA = 0.50
    HOLD_STEPS = 5

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        rooms, _ = self.get_room_layout()
        if not rooms:
            self._scene_invalid_reason = f"{self.scene_name}: no rooms"
            return
        self.source_room = max(rooms, key=self._room_area)

    def _is_scene_valid(self):
        return super()._is_scene_valid() and getattr(self, "source_room", None)

    def _get_obj_cfgs(self):
        return []

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold = 0
        pose = self._sample_robot_pose()
        if pose is None:
            raise RuntimeError(f"{self.scene_name}: no crouch floor pose")
        robot_xy, yaw = pose
        if not (
            self._set_robot_free_pose(robot_xy, yaw)
            and self._calibrate_robot_free_joint_height()
        ):
            raise RuntimeError(f"{self.scene_name}: cannot place G1 for crouch")
        self.robot_base_pos = np.array([robot_xy[0], robot_xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self.source_robot_xy = np.array(robot_xy, dtype=float)
        self.source_robot_yaw = float(yaw)
        self._initial_head_z = self._head_z()

    def _reset_internal(self):
        super()._reset_internal()
        self._success_hold = 0
        if hasattr(self, "source_robot_xy"):
            self._set_robot_free_pose(self.source_robot_xy, self.source_robot_yaw)
            self._calibrate_robot_free_joint_height()
            self.robot_base_pos = np.array(
                [self.source_robot_xy[0], self.source_robot_xy[1], 0.0]
            )
            self.robot_base_ori = np.array([0.0, 0.0, self.source_robot_yaw])
        self._initial_head_z = self._head_z()

    def _sample_robot_pose(self):
        ignore = ()
        center = self._room_center(self.source_room)
        for robot_xy in sorted(
            self._room_grid(self.source_room, margin=0.75),
            key=lambda p: np.linalg.norm(p - center),
        ):
            if not (
                self._point_is_clear(robot_xy, ignore)
                and self._safe_box_in_room(
                    robot_xy, self.source_room, self.ROBOT_SAFE_HALF
                )
            ):
                continue
            forward = center - robot_xy
            norm = np.linalg.norm(forward)
            if norm < 1e-6:
                forward = np.array([1.0, 0.0])
            else:
                forward = forward / norm
            yaw = float(np.arctan2(forward[1], forward[0]))
            return robot_xy, yaw
        return None

    def _head_z(self):
        model, data = self.sim.model, self.sim.data
        if "robot0_head_camera" in model.camera_names:
            return float(data.cam_xpos[model.camera_name2id("robot0_head_camera")][2])
        for body_name in ("robot0_head_link", "robot0_torso_link", "robot0_pelvis"):
            if body_name in model.body_names:
                return float(data.body_xpos[model.body_name2id(body_name)][2])
        return float(self._robot_base_xy()[1])

    def _head_is_crouched(self):
        return self._head_z() <= self._initial_head_z - self.CROUCH_HEAD_DELTA

    def _raw_success(self):
        return self._head_is_crouched()

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = self._success_hold + 1 if self._raw_success() else 0
        return ret

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        rooms, _ = self.get_room_layout()
        ep_meta.update(
            source_room=self.source_room,
            crouch_head_threshold=float(self._initial_head_z - self.CROUCH_HEAD_DELTA),
            lang=(f"crouch down in the {room_label(rooms[self.source_room]['type'])}"),
        )
        return ep_meta


__all__ = ["ProcTHORCrouchPickCylinder"]
