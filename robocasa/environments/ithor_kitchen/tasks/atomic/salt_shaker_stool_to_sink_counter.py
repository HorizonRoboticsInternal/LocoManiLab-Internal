"""Scene 25: move the native salt shaker from a nearby stool to the counter."""

import math

import numpy as np

from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_spec


class IThorKitchenSaltShakerStoolToSinkCounter(IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (25,)
    SPEC_TASK = "salt_shaker_stool_to_sink_counter"
    ATOMIC_SKILL = "PickPlaceSaltShaker"
    TASK_FAMILY = "pick_place"
    DIFFICULTY_LEVEL = "L2"
    INITIAL_POSE_JITTER = False

    STOOL_BODY = "stool_ddf34c98948ee78cfc554a575e4ac59e_1_0_0"
    STOOL_JOINT = STOOL_BODY + "_jntfree_0"
    STOOL_SEAT_GEOM = "stool_ddf34c98948ee78cfc554a575e4ac59e_1_1_0_collision_1"
    SALT_BODY = "saltshaker_4a6645cf2cdf6c8afa19b4998a0f4061_1_0_0"
    SALT_JOINT = SALT_BODY + "_jntfree_0"
    TARGET_COUNTER_GEOM = "sphere_c8fc031a3410345444c5efceccbb63f3_1_1_0_collision_7"

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
                "stool_xy",
                "salt_xy",
                "hold_seconds",
                "height_tolerance",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        protected = {
            self.sim.model.body_name2id(self.STOOL_BODY),
            self.sim.model.body_name2id(self.SALT_BODY),
        }
        return roots - protected

    def _setup_references(self):
        super()._setup_references()
        salt_bodies = self._body_tree_ids([self.sim.model.body_name2id(self.SALT_BODY)])
        stool_bodies = self._body_tree_ids(
            [self.sim.model.body_name2id(self.STOOL_BODY)]
        )
        self._salt_geom_ids = self._collidable_geoms(salt_bodies)
        self._stool_geom_ids = self._collidable_geoms(stool_bodies)
        self._seat_geom_id = self.sim.model.geom_name2id(self.STOOL_SEAT_GEOM)
        self._target_geom_id = self.sim.model.geom_name2id(self.TARGET_COUNTER_GEOM)

    def _geom_tree_points(self, geom_ids):
        model, data = self.sim.model._model, self.sim.data._data
        signs = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        points = []
        for geom in geom_ids:
            if int(model.geom_type[geom]) == 7:
                mesh = int(model.geom_dataid[geom])
                start = int(model.mesh_vertadr[mesh])
                count = int(model.mesh_vertnum[mesh])
                local = model.mesh_vert[start : start + count]
            else:
                box = model.geom_aabb[geom]
                local = box[:3] + signs * box[3:]
            rotation = data.geom_xmat[geom].reshape(3, 3)
            points.append(local @ rotation.T + data.geom_xpos[geom])
        return np.concatenate(points)

    def _salt_points(self):
        return self._geom_tree_points(self._salt_geom_ids)

    def _before_object_placement(self):
        super()._before_object_placement()
        spec = self._get_task_spec()

        stool_qpos = self.sim.data.get_joint_qpos(self.STOOL_JOINT).copy()
        seat_xy = np.asarray(self.sim.data.geom_xpos[self._seat_geom_id])[:2]
        stool_qpos[:2] += np.asarray(spec["stool_xy"], dtype=float) - seat_xy
        self.sim.data.set_joint_qpos(self.STOOL_JOINT, stool_qpos)
        self.sim.forward()

        seat_box = np.asarray(self.sim.model.geom_aabb[self._seat_geom_id])
        seat_rotation = np.asarray(self.sim.data.geom_xmat[self._seat_geom_id]).reshape(
            3, 3
        )
        seat_top = (
            self.sim.data.geom_xpos[self._seat_geom_id][2]
            + (seat_rotation @ seat_box[:3])[2]
            + np.abs(seat_rotation[2]) @ seat_box[3:]
        )

        salt_qpos = self.sim.data.get_joint_qpos(self.SALT_JOINT).copy()
        points = self._salt_points()
        salt_qpos[:2] += (
            np.asarray(spec["salt_xy"], dtype=float)
            - (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
        )
        salt_qpos[2] += seat_top - points[:, 2].min() - 0.0001
        self.sim.data.set_joint_qpos(self.SALT_JOINT, salt_qpos)

        for joint in (self.STOOL_JOINT, self.SALT_JOINT):
            start, end = self.sim.model.get_joint_qvel_addr(joint)
            self.sim.data.qvel[start:end] = 0
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        self._success_hold_steps = 0

    def _target_bounds_and_top(self):
        geom = self._target_geom_id
        box = np.asarray(self.sim.model.geom_aabb[geom])
        rotation = np.asarray(self.sim.data.geom_xmat[geom]).reshape(3, 3)
        signs = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        points = (box[:3] + signs * box[3:]) @ rotation.T + self.sim.data.geom_xpos[
            geom
        ]
        return (
            points[:, :2].min(axis=0),
            points[:, :2].max(axis=0),
            points[:, 2].max(),
        )

    @staticmethod
    def _on_counter(points, lower, upper, top, height_tolerance):
        points = np.asarray(points)
        return bool(
            np.all(points[:, :2] >= np.asarray(lower) - 0.002)
            and np.all(points[:, :2] <= np.asarray(upper) + 0.002)
            and abs(float(points[:, 2].min()) - float(top)) <= height_tolerance
        )

    def _raw_success(self):
        lower, upper, top = self._target_bounds_and_top()
        return self._on_counter(
            self._salt_points(),
            lower,
            upper,
            top,
            float(self._get_task_spec()["height_tolerance"]),
        )

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
            lang="move the salt shaker from the stool beside you to the counter left of the sink",
            task_id=type(self).__name__,
            target_entity="FloorPlan25:Salt_Shaker_2_10",
            source_support="FloorPlan25:Stool_3_1_26",
            target_counter="counter_left_of_sink_facing_sink",
            target_counter_geom=self.TARGET_COUNTER_GEOM,
            success_definition="whole salt shaker footprint on the counter left of the sink at countertop height continuously for 1 second",
        )
        return meta


__all__ = ["IThorKitchenSaltShakerStoolToSinkCounter"]
