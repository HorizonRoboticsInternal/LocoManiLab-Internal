"""Scene 8: move the pan to the front-right burner, then put an egg in it."""

import math

import numpy as np

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.environments.ithor_kitchen.tasks.atomic.pan_left_to_right_burner import (
    IThorKitchenPanLeftToRightBurner,
)
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class IThorKitchenPanToRightBurnerThenEggIntoPan(
    IThorMultiStageCompositeMixin,
    IThorKitchenEnv,
):
    """Move the pan between front burners, then place the nearby egg in it."""

    SUPPORTED_SCENE_IDS = (8,)
    SPEC_TASK = "pan_to_right_burner_then_egg_into_pan"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False

    PAN_BODY = "cookingpan_4e32276b6d5211f77b4dbcba87f5c921_1_0_0"
    EGG_BODY = "egg_15cf0be37a6e2484728ad7e6fc53f181_1_0_0"
    PAN_JOINT = PAN_BODY + "_jntfree_0"
    EGG_JOINT = EGG_BODY + "_jntfree_0"
    PAN_SITES = (
        "cookingpan_4e32276b6d5211f77b4dbcba87f5c921_1_1_0_pan8ReceptacleCollider0_0",
        "cookingpan_4e32276b6d5211f77b4dbcba87f5c921_1_1_0_pan8ReceptacleCollider1_1",
    )
    TARGET_BURNER_GEOMS = ("mesh_e00097fe8685339a7b23451cba3bb274_1_1_0_collision_3",)

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        return get_ithor_kitchen_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "robot_pose",
                "hold_seconds",
                "min_overlap_ratio",
                "egg_xy",
                "egg_support_z",
                "egg_in_pan_tolerance",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="move_pan_to_front_right_burner",
                success="_pan_on_target_burner",
                task_cls="IThorKitchenPanLeftToRightBurner",
                hold_steps=5,
            ),
            IThorAtomicTask(
                name="place_egg_in_pan",
                success="_egg_in_pan_with_pan_retained",
                task_cls="IThorKitchenEggIntoPan",
            ),
        )

    def _required_stage_hold_steps(self, task):
        if task is self._atomic_tasks[-1]:
            return math.ceil(
                float(self._get_task_spec()["hold_seconds"]) * self.control_freq
            )
        return super()._required_stage_hold_steps(task)

    def _setup_references(self):
        super()._setup_references()
        pan_bodies = self._body_tree_ids([self.sim.model.body_name2id(self.PAN_BODY)])
        egg_bodies = self._body_tree_ids([self.sim.model.body_name2id(self.EGG_BODY)])
        self._pan_geom_ids = self._collidable_geoms(pan_bodies)
        self._egg_geom_ids = self._collidable_geoms(egg_bodies)
        self._pan_site_ids = [
            self.sim.model.site_name2id(name) for name in self.PAN_SITES
        ]
        self._target_geom_ids = [
            self.sim.model.geom_name2id(name) for name in self.TARGET_BURNER_GEOMS
        ]

    @staticmethod
    def _box_signs():
        return np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )

    def _geom_points(self, geom_ids):
        model, data = self.sim.model._model, self.sim.data._data
        points = []
        for geom_id in geom_ids:
            if int(model.geom_type[geom_id]) == 7:
                mesh_id = int(model.geom_dataid[geom_id])
                start = int(model.mesh_vertadr[mesh_id])
                count = int(model.mesh_vertnum[mesh_id])
                local = model.mesh_vert[start : start + count]
            else:
                bounds = model.geom_aabb[geom_id]
                local = bounds[:3] + self._box_signs() * bounds[3:]
            rotation = data.geom_xmat[geom_id].reshape(3, 3)
            points.append(local @ rotation.T + data.geom_xpos[geom_id])
        return np.concatenate(points)

    def _target_bounds_xy(self):
        points = []
        for geom_id in self._target_geom_ids:
            bounds = self.sim.model.geom_aabb[geom_id]
            local = bounds[:3] + self._box_signs() * bounds[3:]
            rotation = np.asarray(self.sim.data.geom_xmat[geom_id]).reshape(3, 3)
            points.extend(local @ rotation.T + self.sim.data.geom_xpos[geom_id])
        points = np.asarray(points)[:, :2]
        return points.min(0), points.max(0)

    def _place_egg(self):
        spec = self._get_task_spec()
        qpos = self.sim.data.get_joint_qpos(self.EGG_JOINT).copy()
        qpos[:2] = np.asarray(spec["egg_xy"], dtype=float)
        self.sim.data.set_joint_qpos(self.EGG_JOINT, qpos)
        self.sim.forward()
        qpos = self.sim.data.get_joint_qpos(self.EGG_JOINT).copy()
        qpos[2] += float(spec["egg_support_z"]) - float(
            self._geom_points(self._egg_geom_ids)[:, 2].min()
        )
        self.sim.data.set_joint_qpos(self.EGG_JOINT, qpos)
        start, end = self.sim.model.get_joint_qvel_addr(self.EGG_JOINT)
        self.sim.data.qvel[start:end] = 0
        self.sim.forward()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._place_egg()
        for joint in (self.PAN_JOINT, self.EGG_JOINT):
            start, end = self.sim.model.get_joint_qvel_addr(joint)
            self.sim.data.qvel[start:end] = 0
        self.sim.forward()

    def _reset_task_progress(self):
        super()._reset_task_progress()
        self._egg_stage_armed = False

    def _pan_on_target_burner(self):
        lower, upper = self._target_bounds_xy()
        ratio = IThorKitchenPanLeftToRightBurner._overlap_ratio(
            self._geom_points(self._pan_geom_ids)[:, :2],
            lower,
            upper,
        )
        return ratio + 1e-9 >= float(self._get_task_spec()["min_overlap_ratio"])

    def _egg_in_pan(self):
        points = self._geom_points(self._egg_geom_ids)
        egg_center = (points.min(0) + points.max(0)) / 2
        tolerance = float(self._get_task_spec()["egg_in_pan_tolerance"])
        for site_id in self._pan_site_ids:
            site_center = np.asarray(self.sim.data.site_xpos[site_id])
            rotation = np.asarray(self.sim.data.site_xmat[site_id]).reshape(3, 3)
            half_size = np.asarray(self.sim.model.site_size[site_id]) + tolerance
            local = (egg_center - site_center) @ rotation
            if np.all(np.abs(local) <= half_size):
                return True
        return False

    def _egg_in_pan_with_pan_retained(self):
        egg_in_pan = self._egg_in_pan()
        if not self._egg_stage_armed:
            if not egg_in_pan:
                self._egg_stage_armed = True
            return False
        return egg_in_pan and self._pan_on_target_burner()

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "move the pan from the front-left burner to the front-right "
                "burner, then put the nearby egg in the pan"
            ),
            target_entity="FloorPlan8:Egg_8_4",
            target_receptacle="FloorPlan8:Pan_8_32",
            task_id=type(self).__name__,
            success_definition=(
                "first overlap at least 70 percent of the pan footprint with "
                "the front-right burner; then keep the egg center inside the "
                "pan while the pan remains on that burner continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorKitchenPanToRightBurnerThenEggIntoPan"]
