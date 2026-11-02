"""Place Scene 8's native sponge in the initially open drawer beside it."""

import math

import numpy as np

from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_spec


class IThorKitchenSpongeIntoOpenDrawer(IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (8,)
    SPEC_TASK = "sponge_into_open_drawer"
    ATOMIC_SKILL = "PlaceIntoDrawer"
    TASK_FAMILY = "pick_place"
    DIFFICULTY_LEVEL = "L2"
    INITIAL_POSE_JITTER = False
    SPONGE_BODY = "sponge_41cc9aa65073b4cd1fc4d9871335148d_1_0_0"
    SPONGE_JOINT = SPONGE_BODY + "_jntfree_0"
    DRAWER_JOINT = "drawer_1e0a6cf3f02ee8aa272eb3ad3f084372_1_1_0_joint_0"
    DRAWER_SITE = (
        "drawer_1e0a6cf3f02ee8aa272eb3ad3f084372_1_1_0_Mesh01bf79b35bReceptacle0_0"
    )

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
                "init_open_fraction",
                "hold_seconds",
                "sponge_xy",
                "sponge_quat",
                "sponge_support_geom",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _setup_references(self):
        super()._setup_references()
        bodies = self._body_tree_ids([self.sim.model.body_name2id(self.SPONGE_BODY)])
        self._sponge_geom_ids = self._collidable_geoms(bodies)
        self._drawer_site_id = self.sim.model.site_name2id(self.DRAWER_SITE)
        self._drawer_joint_id = self.sim.model.joint_name2id(self.DRAWER_JOINT)
        if any(int(self.sim.model.geom_type[g]) != 6 for g in self._sponge_geom_ids):
            raise ValueError(
                "Scene 8 native sponge must use its original box collision"
            )

    def _before_object_placement(self):
        super()._before_object_placement()
        low, high = self.sim.model.jnt_range[self._drawer_joint_id]
        fraction = float(self._get_task_spec()["init_open_fraction"])
        self.sim.data.set_joint_qpos(self.DRAWER_JOINT, low + fraction * (high - low))
        self.sim.data.qvel[self.sim.model.get_joint_qvel_addr(self.DRAWER_JOINT)] = 0
        self.sim.forward()
        spec = self._get_task_spec()
        qpos = self.sim.data.get_joint_qpos(self.SPONGE_JOINT).copy()
        qpos[:2] = np.asarray(spec["sponge_xy"], dtype=float)
        qpos[3:7] = np.asarray(spec["sponge_quat"], dtype=float)
        self.sim.data.set_joint_qpos(self.SPONGE_JOINT, qpos)
        self.sim.forward()
        points = self._sponge_points()
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        qpos[:2] += np.asarray(spec["sponge_xy"]) - center[:2]
        support = self.sim.model.geom_name2id(spec["sponge_support_geom"])
        box = np.asarray(self.sim.model.geom_aabb[support])
        rotation = np.asarray(self.sim.data.geom_xmat[support]).reshape(3, 3)
        top = (
            self.sim.data.geom_xpos[support][2]
            + (rotation @ box[:3])[2]
            + np.abs(rotation[2]) @ box[3:]
        )
        qpos[2] += top - points[:, 2].min() - 0.0001
        self.sim.data.set_joint_qpos(self.SPONGE_JOINT, qpos)
        start, end = self.sim.model.get_joint_qvel_addr(self.SPONGE_JOINT)
        self.sim.data.qvel[start:end] = 0
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _reset_task_progress(self):
        self._success_hold_steps = 0

    def _sponge_points(self):
        signs = np.array(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        model, data = self.sim.model, self.sim.data
        points = []
        for geom in self._sponge_geom_ids:
            box = np.asarray(model.geom_aabb[geom])
            local = box[:3] + signs * box[3:]
            rotation = np.asarray(data.geom_xmat[geom]).reshape(3, 3)
            points.append(local @ rotation.T + data.geom_xpos[geom])
        return np.concatenate(points)

    @staticmethod
    def _inside_drawer(points, center, rotation, half_size):
        # The native receptacle's local Y points down; only permit a 2 mm
        # tolerance at its floor, not above the rim or outside its walls.
        local = (points - center) @ rotation
        return bool(
            np.all(local >= -half_size)
            and np.all(local <= half_size + np.array([0.0, 0.002, 0.0]))
        )

    def _raw_success(self):
        site = self._drawer_site_id
        return self._inside_drawer(
            self._sponge_points(),
            np.asarray(self.sim.data.site_xpos[site]),
            np.asarray(self.sim.data.site_xmat[site]).reshape(3, 3),
            np.asarray(self.sim.model.site_size[site]),
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
            lang="put the sponge into the open top drawer beside it",
            task_id=type(self).__name__,
            target_entity="FloorPlan8:Dish_Sponge_1_25",
            target_receptacle="FloorPlan8:Drawer_8c0ca37e_5",
            target_joint=self.DRAWER_JOINT,
            success_definition="whole sponge inside the specified drawer's moving receptacle continuously for 1 second; drawer starts 100 percent open",
        )
        return meta


__all__ = ["IThorKitchenSpongeIntoOpenDrawer"]
