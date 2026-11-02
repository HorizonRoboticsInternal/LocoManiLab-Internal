"""Scene 3: place the native egg in the open pot beside it."""

import math

import mujoco
import numpy as np

from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import get_ithor_kitchen_task_spec


class IThorKitchenEggIntoJar(IThorKitchenEnv):
    SUPPORTED_SCENE_IDS = (3,)
    SPEC_TASK = "egg_into_jar"
    ATOMIC_SKILL = "PlaceIntoReceptacle"
    TASK_FAMILY = "pick_place"
    DIFFICULTY_LEVEL = "L2"
    INITIAL_POSE_JITTER = False
    EGG_BODY = "egg_45a3d68915c9f19164541ddf4da76856_1_0_0"
    JAR_BODY = "pot_b832ca1a0e9cb0a662847a7c02bd0466_1_0_0"
    EGG_JOINT = EGG_BODY + "_jntfree_0"
    JAR_JOINT = JAR_BODY + "_jntfree_0"
    EGG_SCALE = 0.075 / 0.06848599
    RECEPTACLE_SITES = (
        "pot_b832ca1a0e9cb0a662847a7c02bd0466_1_1_0_pot3ReceptacleCollider0_0",
        "pot_b832ca1a0e9cb0a662847a7c02bd0466_1_1_0_pot3ReceptacleCollider1_1",
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        return get_ithor_kitchen_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("robot_pose", "hold_seconds"),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        # Scale only this task's in-memory egg. The scene file stays unchanged.
        egg = self.mujoco_arena.worldbody.find(f".//body[@name='{self.EGG_BODY}']")
        if egg is None:
            raise ValueError("Scene 3 native egg is missing")
        meshes = {g.get("mesh") for g in egg.iter("geom") if g.get("mesh")}
        for mesh in self.mujoco_arena.asset.findall("mesh"):
            if mesh.get("name") in meshes:
                scale = np.fromstring(mesh.get("scale", "1 1 1"), sep=" ")
                mesh.set("scale", " ".join(map(str, scale * self.EGG_SCALE)))
        for body in egg.findall(".//body"):
            pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
            body.set("pos", " ".join(map(str, pos * self.EGG_SCALE)))
        quat = np.fromstring(egg.get("quat"), sep=" ")
        rotated = np.empty(4)
        mujoco.mju_mulQuat(rotated, np.array([2**-0.5, 0, 0, 2**-0.5]), quat)
        egg.set("quat", " ".join(map(str, rotated)))

    def _setup_references(self):
        super()._setup_references()
        self._egg_body_ids = self._body_tree_ids(
            [self.sim.model.body_name2id(self.EGG_BODY)]
        )
        self._egg_geom_ids = self._collidable_geoms(self._egg_body_ids)
        self._site_ids = [self.sim.model.site_name2id(n) for n in self.RECEPTACLE_SITES]

    def _egg_vertices(self):
        model, data = self.sim.model._model, self.sim.data._data
        points = []
        for geom in self._egg_geom_ids:
            mesh = int(model.geom_dataid[geom])
            start, count = int(model.mesh_vertadr[mesh]), int(model.mesh_vertnum[mesh])
            vertices = model.mesh_vert[start : start + count]
            rotation = data.geom_xmat[geom].reshape(3, 3)
            points.append(vertices @ rotation.T + data.geom_xpos[geom])
        return np.concatenate(points)

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()
        # Place the scaled, rotated native egg in shallow contact with its
        # original support, rather than dropping it after reset.
        points = self._egg_vertices()
        model, data = self.sim.model._model, self.sim.data._data
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        geomid = np.array([-1], dtype=np.int32)
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([center[0], center[1], center[2] + 0.15]),
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            int(model.geom_bodyid[self._egg_geom_ids[0]]),
            geomid,
        )
        if distance < 0:
            raise ValueError("No physical support under Scene 3 egg")
        support_z = center[2] + 0.15 - distance
        qpos = self.sim.data.get_joint_qpos(self.EGG_JOINT).copy()
        qpos[2] += support_z - float(points[:, 2].min()) - 0.0001
        self.sim.data.set_joint_qpos(self.EGG_JOINT, qpos)
        self._egg_support_geom = int(geomid[0])
        for joint in (self.EGG_JOINT, self.JAR_JOINT):
            start, end = self.sim.model.get_joint_qvel_addr(joint)
            self.sim.data.qvel[start:end] = 0
        self.sim.forward()

    def _reset_task_progress(self):
        self._success_hold_steps = 0

    @staticmethod
    def _containment_mask(points, center, rotation, half_size):
        local = (points - center) @ rotation
        # The native site's Y axis is vertical; allow shallow floor contact.
        bottom_tolerance = np.array([0.0, 0.002, 0.0])
        return np.all(local >= -half_size, axis=1) & np.all(
            local <= half_size + bottom_tolerance, axis=1
        )

    @classmethod
    def _contained(cls, points, center, rotation, half_size):
        return bool(np.all(cls._containment_mask(points, center, rotation, half_size)))

    @classmethod
    def _inside_receptacle(cls, points, volumes):
        # The two native sites jointly describe one cavity. Requiring the
        # intersection wrongly rejects an egg resting off centre in the pot.
        inside = np.zeros(len(points), dtype=bool)
        for center, rotation, half_size in volumes:
            inside |= cls._containment_mask(points, center, rotation, half_size)
        return bool(len(points) and np.all(inside))

    def _raw_success(self):
        model, data = self.sim.model, self.sim.data
        volumes = [
            (
                np.asarray(data.site_xpos[site]),
                np.asarray(data.site_xmat[site]).reshape(3, 3),
                np.asarray(model.site_size[site]),
            )
            for site in self._site_ids
        ]
        return self._inside_receptacle(self._egg_vertices(), volumes)

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
            lang="put the egg into the open pot beside it",
            target_entity="FloorPlan3:Egg_3_21",
            target_receptacle="FloorPlan3:Pot_3_22",
            task_id=type(self).__name__,
            success_definition="whole egg inside the union of the native pot receptacle volumes continuously for 1 second",
        )
        return meta


__all__ = ["IThorKitchenEggIntoJar"]
