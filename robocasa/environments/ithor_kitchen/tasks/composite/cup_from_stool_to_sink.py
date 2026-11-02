"""Scene 3: take a cup from a stool, then carry it to the sink."""

import math

import numpy as np

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class IThorKitchenCupFromStoolToSink(
    IThorMultiStageCompositeMixin,
    IThorKitchenEnv,
):
    """Take the stool's cup, carry it to the sink, and put it inside."""

    SUPPORTED_SCENE_IDS = (3,)
    SPEC_TASK = "cup_from_stool_to_sink"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False

    CUP_BODY = "cup_dae0e5a693f3ef8e56f0000c68e4dafe_1_0_0"
    STOOL_BODY = "stool_80b0f89d3b5199daba395c1cfedbc901_1_0_0"
    CUP_JOINT = CUP_BODY + "_jntfree_0"
    STOOL_JOINT = STOOL_BODY + "_jntfree_0"
    SEAT_GEOM = "stool_80b0f89d3b5199daba395c1cfedbc901_1_1_0_collision_1"

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
                "stool_xy",
                "sink_volume",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="take_cup_from_stool",
                success="_cup_removed_from_stool",
                hold_steps=5,
            ),
            IThorAtomicTask(
                name="put_cup_in_sink",
                success="_cup_in_sink",
                task_cls="IThorKitchenCupToSink",
            ),
        )

    def _required_stage_hold_steps(self, task):
        if task is self._atomic_tasks[-1]:
            return math.ceil(
                float(self._get_task_spec()["hold_seconds"]) * self.control_freq
            )
        return super()._required_stage_hold_steps(task)

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        return roots - {self.sim.model.body_name2id(self.STOOL_BODY)}

    def _setup_references(self):
        super()._setup_references()
        cup_bodies = self._body_tree_ids([self.sim.model.body_name2id(self.CUP_BODY)])
        self._cup_geom_ids = self._collidable_geoms(cup_bodies)
        self._seat_geom_id = self.sim.model.geom_name2id(self.SEAT_GEOM)

    @staticmethod
    def _box_signs():
        return np.asarray(
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

    def _single_geom_points(self, geom_id):
        model, data = self.sim.model._model, self.sim.data._data
        bounds = model.geom_aabb[geom_id]
        local = bounds[:3] + self._box_signs() * bounds[3:]
        rotation = data.geom_xmat[geom_id].reshape(3, 3)
        return local @ rotation.T + data.geom_xpos[geom_id]

    def _place_body_xy(self, joint_name, xy):
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:2] = np.asarray(xy, dtype=float)
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.data.set_joint_qvel(joint_name, np.zeros(6))
        self.sim.forward()

    def _place_body_bottom(self, joint_name, geom_ids, xy, bottom_z):
        self._place_body_xy(joint_name, xy)
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[2] += float(bottom_z) - float(self._geom_points(geom_ids)[:, 2].min())
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.data.set_joint_qvel(joint_name, np.zeros(6))
        self.sim.forward()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._place_body_xy(self.STOOL_JOINT, self._get_task_spec()["stool_xy"])
        seat_points = self._single_geom_points(self._seat_geom_id)
        seat_center_xy = (seat_points[:, :2].min(0) + seat_points[:, :2].max(0)) / 2
        self._place_body_bottom(
            self.CUP_JOINT,
            self._cup_geom_ids,
            seat_center_xy,
            float(seat_points[:, 2].max()) + 0.002,
        )

    def _cup_on_stool(self):
        cup_points = self._geom_points(self._cup_geom_ids)
        seat_points = self._single_geom_points(self._seat_geom_id)
        cup_center = (cup_points.min(axis=0) + cup_points.max(axis=0)) / 2
        seat_low = seat_points.min(axis=0)
        seat_high = seat_points.max(axis=0)
        return bool(
            np.all(cup_center[:2] >= seat_low[:2])
            and np.all(cup_center[:2] <= seat_high[:2])
            and abs(float(cup_points[:, 2].min()) - float(seat_high[2])) <= 0.05
        )

    def _cup_removed_from_stool(self):
        return not self._cup_on_stool()

    @staticmethod
    def _point_in_volume(point, volume):
        point = np.asarray(point, dtype=float)
        center = np.asarray(volume["center"], dtype=float)
        half_size = np.asarray(volume["half_size"], dtype=float)
        tolerance = np.asarray(volume.get("tolerance", [0.0, 0.0, 0.0]))
        return bool(
            np.all(point >= center - half_size - tolerance)
            and np.all(point <= center + half_size + tolerance)
        )

    def _cup_in_sink(self):
        points = self._geom_points(self._cup_geom_ids)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        return self._point_in_volume(center, self._get_task_spec()["sink_volume"])

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "take the cup from the stool, turn around and walk to the sink, "
                "then put the cup into the sink"
            ),
            target_entity="FloorPlan3:Cup_3_20",
            source_support="FloorPlan3:Stool_2_3_1",
            target_receptacle="FloorPlan3:Sink_8f39e796_17",
            task_id=type(self).__name__,
            success_definition=(
                "after the cup leaves the stool, keep its center inside the sink "
                "continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorKitchenCupFromStoolToSink"]
