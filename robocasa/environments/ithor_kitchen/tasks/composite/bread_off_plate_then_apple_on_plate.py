"""Scene 3: move bread off its plate, then put an apple on the plate."""

import math
import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class IThorKitchenBreadOffPlateThenAppleOnPlate(
    SupportContactPlacementMixin,
    IThorMultiStageCompositeMixin,
    IThorKitchenEnv,
):
    """Move the bread to the right counter, then put the apple on the plate."""

    SUPPORTED_SCENE_IDS = (3,)
    SPEC_TASK = "bread_off_plate_then_apple_on_plate"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False

    BREAD_BODY = "bread_02590a72dc962788d308c804909b928a_1_0_0"
    PLATE_BODY = "plate_078f7b7772accf1f6c4a960a7e5bdde0_1_0_0"
    BREAD_JOINT = BREAD_BODY + "_jntfree_0"
    PLATE_JOINT = PLATE_BODY + "_jntfree_0"
    APPLE_NAME = "apple"
    APPLE_XML = "environments/ithor_kitchen/tasks/assets/Apple_25/model.xml"

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
                "plate_xy",
                "apple_xy",
                "counter_top_z",
                "bread_support_z_tolerance",
                "bread_target",
                "apple_plate_xy_tolerance",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _get_obj_cfgs(self):
        path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.APPLE_XML)
        )
        return [dict(name=self.APPLE_NAME, obj_groups=path)]

    def _create_objects(self):
        """Load the verified Apple_25 mesh directly without registry sampling."""

        self.objects = {}
        self.object_cfgs = self._get_obj_cfgs()
        apple = MJCFObject(
            name=self.APPLE_NAME,
            mjcf_path=self.object_cfgs[0]["obj_groups"],
            scale=1.0,
        )
        self.objects[apple.name] = apple
        self.model.merge_objects([apple])

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualBreadApplePlateSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="place_bread_right_of_plate",
                success="_bread_on_right_counter",
                task_cls="IThorKitchenBreadOnCounter",
                hold_steps=5,
            ),
            IThorAtomicTask(
                name="place_apple_on_plate",
                success="_apple_on_plate_with_bread_retained",
                task_cls="IThorKitchenAppleOnPlate",
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
        bread_bodies = self._body_tree_ids(
            [self.sim.model.body_name2id(self.BREAD_BODY)]
        )
        plate_bodies = self._body_tree_ids(
            [self.sim.model.body_name2id(self.PLATE_BODY)]
        )
        apple_bodies = self._body_tree_ids(
            [self.sim.model.body_name2id(self.objects[self.APPLE_NAME].root_body)]
        )
        self._bread_geom_ids = self._collidable_geoms(bread_bodies)
        self._plate_geom_ids = self._collidable_geoms(plate_bodies)
        self._apple_geom_ids = self._collidable_geoms(apple_bodies)

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

    def _place_native_body(self, joint_name, geom_ids, xy, bottom_z):
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:2] = np.asarray(xy, dtype=float)
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.forward()
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[2] += float(bottom_z) - float(self._geom_points(geom_ids)[:, 2].min())
        self.sim.data.set_joint_qpos(joint_name, qpos)
        start, end = self.sim.model.get_joint_qvel_addr(joint_name)
        self.sim.data.qvel[start:end] = 0
        self.sim.forward()

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()
        plate_xy = spec["plate_xy"]
        counter_top_z = float(spec["counter_top_z"]) + 0.001
        self._place_native_body(
            self.PLATE_JOINT,
            self._plate_geom_ids,
            plate_xy,
            counter_top_z,
        )
        plate_top = float(self._geom_points(self._plate_geom_ids)[:, 2].max())
        self._place_native_body(
            self.BREAD_JOINT,
            self._bread_geom_ids,
            plate_xy,
            plate_top + 0.002,
        )
        self._set_task_object_bottom_at(
            self.APPLE_NAME,
            spec["apple_xy"],
            counter_top_z,
        )

    @staticmethod
    def _points_in_volume(points, volume):
        center = np.asarray(volume["center"], dtype=float)
        half_size = np.asarray(volume["half_size"], dtype=float)
        tolerance = np.asarray(volume.get("tolerance", [0.0, 0.0, 0.0]))
        return bool(
            len(points)
            and np.all(points >= center - half_size - tolerance)
            and np.all(points <= center + half_size + tolerance)
        )

    def _bread_on_right_counter(self):
        points = self._geom_points(self._bread_geom_ids)
        spec = self._get_task_spec()
        return bool(
            self._points_in_volume(points, spec["bread_target"])
            and abs(float(points[:, 2].min()) - float(spec["counter_top_z"]))
            <= float(spec["bread_support_z_tolerance"])
        )

    def _apple_contacts_plate(self):
        for contact in self.sim.data.contact[: self.sim.data.ncon]:
            pair = {int(contact.geom1), int(contact.geom2)}
            if pair.intersection(self._apple_geom_ids) and pair.intersection(
                self._plate_geom_ids
            ):
                return float(contact.dist) >= -0.003
        return False

    def _apple_on_plate(self):
        apple_points = self._geom_points(self._apple_geom_ids)
        plate_points = self._geom_points(self._plate_geom_ids)
        apple_center = (apple_points.min(axis=0) + apple_points.max(axis=0)) / 2
        plate_center = (plate_points.min(axis=0) + plate_points.max(axis=0)) / 2
        return bool(
            np.linalg.norm(apple_center[:2] - plate_center[:2])
            <= float(self._get_task_spec()["apple_plate_xy_tolerance"])
            and self._apple_contacts_plate()
            and self.object_released(self.APPLE_NAME)
        )

    def _apple_on_plate_with_bread_retained(self):
        return self._apple_on_plate() and self._bread_on_right_counter()

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "take the bread off the plate and place it on the counter to the "
                "right, then put the apple from the left side onto the plate"
            ),
            target_entity="FloorPlan3:Bread_3_24 and task apple",
            target_receptacle="FloorPlan3:Plate_3_18",
            task_id=type(self).__name__,
            success_definition=(
                "first place the whole bread anywhere on the counter to the right "
                "of the sink; then release the apple on the plate while the bread "
                "remains on that counter continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorKitchenBreadOffPlateThenAppleOnPlate"]
