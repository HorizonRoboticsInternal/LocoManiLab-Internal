"""Pull the movable office chair out from under a desk in Holodeck."""

import xml.etree.ElementTree as ET

import numpy as np

from .push_chair_to_desk import HolodeckPushChairToDesk


class HolodeckPullChairFromDesk(HolodeckPushChairToDesk):
    """Pull a calibrated office chair out from under its desk."""

    BLUE_CHAIR_SCENES = frozenset(("train_1018", "train_1664", "train_2025"))
    MIN_PULL_DISPLACEMENT = 0.50
    ATOMIC_SKILL = "PullObject"
    TASK_FAMILY = "pull"
    SPEC_TASK = "pull_chair_from_desk"
    # Used when this initializer is reached through a composite MRO.
    PUSH_SPEC_TASK = SPEC_TASK

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self._pull_origin_xy = np.asarray(
            self.task_spec["chair_start_pose"]["pos"][:2], dtype=float
        )
        self._disable_desk_drawer_collisions()

        # Pulling requires the robot to stand on the side toward which the
        # chair will move.  The parent Push task places it on the opposite
        # side, so replace that placement after the shared fixture setup.
        start_xy = np.asarray(
            self.task_spec["chair_start_pose"]["pos"][:2], dtype=float
        )
        pull_delta = self.target_xy - start_xy
        pull_norm = float(np.linalg.norm(pull_delta))
        if pull_norm <= 1e-6:
            return
        pull_axis = pull_delta / pull_norm
        robot_pose = self.task_spec["robot_pose"]
        robot_pos = np.asarray(robot_pose["pos"], dtype=float).copy()
        robot_pos[:2] = start_xy + pull_axis * self.PUSH_STANDOFF
        # Face the chair while standing on the side toward which it will move.
        robot_yaw = float(np.arctan2(-pull_axis[1], -pull_axis[0]))
        self.set_robot_base_pose(robot_pos, robot_yaw)

    def _setup_scene(self):
        # Composite tasks may replace ``task_spec`` after fixture setup; refresh
        # the origin immediately before the shared scene reset uses it.
        self._pull_origin_xy = np.asarray(
            self.task_spec["chair_start_pose"]["pos"][:2], dtype=float
        )
        super()._setup_scene()
        if self.chair_body not in self.sim.model.body_names:
            raise RuntimeError(
                f"{self.scene_name}: calibrated chair body is missing: "
                f"{self.chair_body}"
            )
        if self._success_position_condition():
            raise RuntimeError(
                f"{self.scene_name}: pull-chair reset already exceeds the "
                "required displacement"
            )

    def _chair_has_min_pull_displacement(self, chair_body=None, origin_xy=None):
        """Return whether a chair has cleared its reset origin by the threshold."""
        body_name = self.chair_body if chair_body is None else chair_body
        origin = self._pull_origin_xy if origin_xy is None else origin_xy
        body_id = self.sim.model.body_name2id(body_name)
        chair_xy = np.asarray(self.sim.data.body_xpos[body_id][:2], dtype=float)
        displacement = np.linalg.norm(chair_xy - np.asarray(origin, dtype=float))
        return bool(displacement >= HolodeckPullChairFromDesk.MIN_PULL_DISPLACEMENT)

    def _success_position_condition(self):
        """Require the chair to leave its reset region by the pull threshold."""
        return self._chair_has_min_pull_displacement()

    def _chair_in_target_region(self, task_spec=None):
        """Keep the calibrated target predicate available to composite tasks."""
        return super()._chair_in_target_region(task_spec)

    def _disable_desk_drawer_collisions(self):
        """Keep the chair from opening desk drawers while it is under the desk."""
        desk_body = self.find_scene_body(self.task_spec["desk_body"])
        if desk_body is None:
            return
        contact = self.mujoco_arena.root.find("contact")
        if contact is None:
            contact = ET.SubElement(self.mujoco_arena.root, "contact")
        if self.task_spec.get("disable_chair_desk_collisions", False):
            # Both fixtures use an outer body with collision geoms attached to
            # nested bodies. MuJoCo excludes body pairs by their exact body
            # ids; excluding only the outer body does not cover descendants.
            chair_body = self.find_scene_body(self.chair_body)
            chair_parts = list(chair_body.iter("body")) if chair_body is not None else []
            desk_parts = list(desk_body.iter("body"))
            for chair_part in chair_parts:
                chair_name = chair_part.get("name")
                if not chair_name:
                    continue
                for desk_part in desk_parts:
                    desk_name = desk_part.get("name")
                    if desk_name:
                        ET.SubElement(
                            contact,
                            "exclude",
                            body1=chair_name,
                            body2=desk_name,
                        )
        if not self.task_spec.get("disable_desk_drawer_collisions", True):
            return

        # Holodeck desks encode their drawers as direct children numbered
        # 2..7; leave the surrounding desk frame collidable for the pull task.
        for child in desk_body.iter("body"):
            name = child.get("name", "")
            if not any(name.endswith(f"_1_{i}_0") for i in range(2, 10)):
                continue
            ET.SubElement(
                contact,
                "exclude",
                body1=self.chair_body,
                body2=name,
            )
            for geom in child.iter("geom"):
                if "_visual_" not in (geom.get("name") or ""):
                    geom.set("contype", "0")
                    geom.set("conaffinity", "0")

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        chair_color = (
            "blue" if self.scene_name in self.BLUE_CHAIR_SCENES else "black"
        )
        meta["lang"] = f"pull the {chair_color} office chair out from under the desk"
        meta["success_definition"] = (
            "move the chair center at least "
            f"{self.MIN_PULL_DISPLACEMENT:.2f} m from its reset position and "
            "keep it there continuously for 1 second"
        )
        return meta


__all__ = ["HolodeckPullChairFromDesk"]
