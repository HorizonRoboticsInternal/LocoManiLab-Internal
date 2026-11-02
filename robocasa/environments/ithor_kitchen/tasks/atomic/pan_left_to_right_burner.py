"""Scene 8: move the native pan between the two front stove burners."""

import math

import numpy as np
from scipy.spatial import ConvexHull

from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_spec


class IThorKitchenPanLeftToRightBurner(IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (8,)
    SPEC_TASK = "pan_left_to_right_burner"
    ATOMIC_SKILL = "MovePanBetweenBurners"
    TASK_FAMILY = "pick_place"
    DIFFICULTY_LEVEL = "L2"
    INITIAL_POSE_JITTER = False
    PAN_BODY = "cookingpan_4e32276b6d5211f77b4dbcba87f5c921_1_0_0"
    PAN_JOINT = PAN_BODY + "_jntfree_0"
    PAN_SITE = (
        "cookingpan_4e32276b6d5211f77b4dbcba87f5c921_1_1_0_pan8ReceptacleCollider0_0"
    )
    # Native coil support sits above the oven's lower decorative grate boxes.
    TARGET_BURNER_GEOMS = ("mesh_e00097fe8685339a7b23451cba3bb274_1_1_0_collision_3",)

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        return get_ithor_kitchen_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("robot_pose", "hold_seconds", "min_overlap_ratio"),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _setup_references(self):
        super()._setup_references()
        bodies = self._body_tree_ids([self.sim.model.body_name2id(self.PAN_BODY)])
        self._pan_geom_ids = self._collidable_geoms(bodies)
        self._pan_site_id = self.sim.model.site_name2id(self.PAN_SITE)
        self._target_geom_ids = [
            self.sim.model.geom_name2id(n) for n in self.TARGET_BURNER_GEOMS
        ]

    def _before_object_placement(self):
        super()._before_object_placement()
        # Pan_8_32 already rests on the left-front burner in the native scene.
        start, end = self.sim.model.get_joint_qvel_addr(self.PAN_JOINT)
        self.sim.data.qvel[start:end] = 0
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _reset_task_progress(self):
        self._success_hold_steps = 0

    def _pan_points(self):
        model, data = self.sim.model._model, self.sim.data._data
        signs = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        points = []
        for geom in self._pan_geom_ids:
            if int(model.geom_type[geom]) == 7:
                mesh = int(model.geom_dataid[geom])
                start, count = int(model.mesh_vertadr[mesh]), int(
                    model.mesh_vertnum[mesh]
                )
                local = model.mesh_vert[start : start + count]
            else:
                box = model.geom_aabb[geom]
                local = box[:3] + signs * box[3:]
            rotation = data.geom_xmat[geom].reshape(3, 3)
            points.append(local @ rotation.T + data.geom_xpos[geom])
        return np.concatenate(points)

    def _target_bounds_xy(self):
        points = []
        signs = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        for geom in self._target_geom_ids:
            box = np.asarray(self.sim.model.geom_aabb[geom])
            rotation = np.asarray(self.sim.data.geom_xmat[geom]).reshape(3, 3)
            points.extend(
                (box[:3] + signs * box[3:]) @ rotation.T + self.sim.data.geom_xpos[geom]
            )
        points = np.asarray(points)[:, :2]
        return points.min(0), points.max(0)

    @staticmethod
    def _overlap_ratio(points_xy, lower, upper):
        """Intersection / pan footprint area; clip its convex outline to the burner."""
        points = np.unique(np.asarray(points_xy, dtype=float), axis=0)
        if len(points) < 3 or np.linalg.matrix_rank(points - points[0]) < 2:
            return 0.0
        polygon = points[ConvexHull(points).vertices]

        def area(vertices):
            if len(vertices) < 3:
                return 0.0
            vertices = np.asarray(vertices)
            return (
                abs(
                    np.dot(vertices[:, 0], np.roll(vertices[:, 1], -1))
                    - np.dot(vertices[:, 1], np.roll(vertices[:, 0], -1))
                )
                / 2
            )

        total = area(polygon)
        for axis, boundary, direction in (
            (0, lower[0], 1),
            (0, upper[0], -1),
            (1, lower[1], 1),
            (1, upper[1], -1),
        ):
            clipped = []
            if len(polygon) == 0:
                return 0.0
            previous = polygon[-1]
            prev_inside = direction * (previous[axis] - boundary) >= 0
            for current in polygon:
                inside = direction * (current[axis] - boundary) >= 0
                if inside != prev_inside:
                    fraction = (boundary - previous[axis]) / (
                        current[axis] - previous[axis]
                    )
                    clipped.append(previous + fraction * (current - previous))
                if inside:
                    clipped.append(current)
                previous, prev_inside = current, inside
            polygon = clipped
        return float(np.clip(area(polygon) / total, 0.0, 1.0))

    def _raw_success(self):
        lower, upper = self._target_bounds_xy()
        ratio = self._overlap_ratio(self._pan_points()[:, :2], lower, upper)
        return ratio + 1e-9 >= float(self._get_task_spec()["min_overlap_ratio"])

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
            lang="move the pan from the front-left burner to the front-right burner",
            task_id=type(self).__name__,
            target_entity="FloorPlan8:Pan_8_32",
            source_burner="front_left_facing_stove",
            target_burner="front_right_facing_stove",
            target_burner_geoms=list(self.TARGET_BURNER_GEOMS),
            success_definition="at least 70% of the pan horizontal footprint overlaps the front-right burner support area continuously for 1 second",
        )
        return meta


__all__ = ["IThorKitchenPanLeftToRightBurner"]
