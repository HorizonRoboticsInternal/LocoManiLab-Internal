"""Shared implementation for iTHOR lift tasks."""

import numpy as np


class RoomLiftMixin:
    """Lift an object from a support surface by a required height."""

    ATOMIC_SKILL = "LiftObject"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "lift"

    TARGET_NAME = None
    MIN_LIFT_DELTA = 0.3
    PLACEMENT_MARGIN = 0.03
    STAND_DISTANCE = 0.4
    HOLD_SECONDS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.05
    INITIAL_OBJECT_JITTER_RADIUS = 0.05
    SEAT_TOKENS = ("stool", "chair")
    SEAT_CLEAR_RADIUS = 1.15

    def _setup_scene(self):
        super()._setup_scene()
        self._place_robot_at_target()

    def _reset_internal(self):
        super()._reset_internal()
        self._lift_initial_z = self._target_root_z()
        self._lift_hold_steps = 0

    def _place_robot_at_target(self):
        target_xy, front = self._lift_target_pose()
        valid_count = 0
        for distance in np.arange(self.STAND_DISTANCE, 1.0 + 0.005, 0.01):
            xy = target_xy + front * float(distance)
            yaw = float(np.arctan2(target_xy[1] - xy[1], target_xy[0] - xy[0]))
            if (
                self._set_robot_free_pose(xy, yaw)
                and self._calibrate_robot_free_joint_height()
            ):
                valid_count += 1
                if valid_count < 3:
                    continue
                self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                self.robot_base_ori = np.array([0.0, 0.0, yaw])
                return
        raise RuntimeError(
            f"{self.scene_name}: no collision-free G1 pose at {self.TARGET_NAME}"
        )

    def _target_root_z(self):
        root = self.objects[self.TARGET_NAME].root_body
        return float(self.sim.data.get_body_xpos(root)[2])

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._target_root_z() >= self._lift_initial_z + self.MIN_LIFT_DELTA:
            self._lift_hold_steps += 1
        else:
            self._lift_hold_steps = 0
        return ret

    def _check_success(self):
        return self._lift_hold_steps >= int(
            round(self.HOLD_SECONDS * self.control_freq)
        )

    def _constrain_initial_jitter_candidate(self):
        super()._constrain_initial_jitter_candidate()
        obj = self.objects[self.TARGET_NAME]
        qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
        target_xy, front = self._lift_target_pose()
        right = np.array([-front[1], front[0]], dtype=float)
        current_xy = qpos[:2] + self._object_bbox_center_xy(self.TARGET_NAME)
        lateral = float(np.dot(current_xy - target_xy, right))
        lateral = np.clip(
            lateral,
            -self.INITIAL_OBJECT_JITTER_RADIUS,
            self.INITIAL_OBJECT_JITTER_RADIUS,
        )
        target_xy = target_xy + right * lateral
        qpos[:2] = target_xy - self._object_bbox_center_xy(self.TARGET_NAME)
        self.sim.data.set_joint_qpos(obj.joints[0], qpos)
        self.sim.forward()

    def _setup_lift_support(self, requirements, ideal_height, blocked_tokens=()):
        candidates = []
        robot_xy = np.asarray(self.robot_base_pos[:2], dtype=float)
        for region in self.get_region(kind="support", return_all=True):
            fixture = self.get_fixture(id=region.fixture_id)
            if fixture is None or not self._support_matches_requirements(
                region, fixture, requirements
            ):
                continue
            text = self.semantic_fixture_text(fixture)
            if any(token in text for token in blocked_tokens):
                continue
            center, _ = region.world_pose()
            score = float(np.linalg.norm(center[:2] - robot_xy))
            score += 0.6 * abs(float(center[2]) - ideal_height)
            candidates.append((score, region.stable_id, region, fixture))
        if not candidates:
            self.lift_region = None
            self.lift_fixture = None
            return
        _, _, self.lift_region, self.lift_fixture = min(candidates)
        self.fixture_refs["lift_surface"] = self.lift_fixture
        self.region_refs["lift_surface"] = self.lift_region
        self.clear_support_region_clutter(self.lift_region, self.lift_fixture)
        target_xy, front = self._lift_target_pose()
        for xy in (target_xy, target_xy + front * self.STAND_DISTANCE):
            self.remove_semantic_fixtures_near(
                self.SEAT_TOKENS,
                xy,
                self.SEAT_CLEAR_RADIUS,
            )

    def _lift_target_pose(self, toward_xy=None):
        return self.support_region_edge_pose(
            self.lift_region,
            self._object_half_xy(self.TARGET_NAME),
            self.robot_base_pos[:2] if toward_xy is None else toward_xy,
            margin=self.PLACEMENT_MARGIN,
            sim=self.sim if getattr(self, "sim", None) is not None else None,
        )

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid() and getattr(self, "lift_region", None) is not None
        )


__all__ = ["RoomLiftMixin"]
