"""Scene 205: move the native cardboard box from the sofa to the floor."""

import math

import numpy as np

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomBoxFromSofaToFloor(IThorLivingRoomEnv):
    SUPPORTED_SCENE_IDS = (205,)
    SPEC_TASK = "box_from_sofa_to_floor"
    ATOMIC_SKILL = "PickPlaceBox"
    TASK_FAMILY = "pick_place"
    DIFFICULTY_LEVEL = "L2"
    INITIAL_POSE_JITTER = False

    SOFA_BODY = "sofa_2e5f2b1b88237f983400420d29e5f613_1_0_0"
    SOFA_SITE = (
        "sofa_2e5f2b1b88237f983400420d29e5f613_1_1_0_" "sofa205ReceptacleCollider0_0"
    )
    BOX_BODY = "box_376bd8a46eebebb53a1cbbec2a1cada0_1_0_0"
    BOX_JOINT = BOX_BODY + "_jntfree_0"

    def __init__(self, *args, scene_ids=None, **kwargs):
        scene_ids = self.SUPPORTED_SCENE_IDS if scene_ids is None else scene_ids
        kwargs.setdefault("initialization_noise", None)
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        return get_ithor_livingroom_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "robot_pose",
                "box_xy",
                "box_quat",
                "sofa_surface_offset",
                "release_distance",
                "hold_seconds",
                "max_linear_speed",
                "max_angular_speed",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        protected = {
            self.sim.model.body_name2id(self.SOFA_BODY),
            self.sim.model.body_name2id(self.BOX_BODY),
        }
        return roots - protected

    def _setup_references(self):
        super()._setup_references()
        box_root = self.sim.model.body_name2id(self.BOX_BODY)
        self._box_body_ids = self._body_tree_ids([box_root])
        self._box_geom_ids = self._collidable_geoms(self._box_body_ids)
        self._sofa_body_ids = self._body_tree_ids(
            [self.sim.model.body_name2id(self.SOFA_BODY)]
        )
        self._sofa_site_id = self.sim.model.site_name2id(self.SOFA_SITE)

    @staticmethod
    def _box_signs():
        return np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )

    def _geom_points(self, geom_ids):
        model, data = self.sim.model._model, self.sim.data._data
        points = []
        for geom in geom_ids:
            bounds = np.asarray(model.geom_aabb[geom], dtype=float)
            local = bounds[:3] + self._box_signs() * bounds[3:]
            rotation = np.asarray(data.geom_xmat[geom], dtype=float).reshape(3, 3)
            points.append(local @ rotation.T + data.geom_xpos[geom])
        return np.concatenate(points)

    def _box_points(self):
        return self._geom_points(self._box_geom_ids)

    def _sofa_site_top(self):
        model, data = self.sim.model._model, self.sim.data._data
        size = np.asarray(model.site_size[self._sofa_site_id], dtype=float)
        rotation = np.asarray(data.site_xmat[self._sofa_site_id], dtype=float).reshape(
            3, 3
        )
        points = (
            self._box_signs() * size @ rotation.T + data.site_xpos[self._sofa_site_id]
        )
        return float(points[:, 2].max())

    def _set_task_layout(self):
        spec = self._get_task_spec()
        quat = np.asarray(spec["box_quat"], dtype=float)
        if quat.shape != (4,) or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: box_quat needs normalized WXYZ"
            )

        qpos = self.sim.data.get_joint_qpos(self.BOX_JOINT).copy()
        qpos[:2] = np.asarray(spec["box_xy"], dtype=float)
        qpos[3:7] = quat
        self.sim.data.set_joint_qpos(self.BOX_JOINT, qpos)
        self.sim.forward()

        qpos[2] += (
            self._sofa_site_top()
            + float(spec["sofa_surface_offset"])
            - float(self._box_points()[:, 2].min())
        )
        self.sim.data.set_joint_qpos(self.BOX_JOINT, qpos)
        self.sim.data.set_joint_qvel(self.BOX_JOINT, np.zeros(6))
        self.sim.forward()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._set_task_layout()

    def _constrain_initial_jitter_candidate(self):
        self._set_task_layout()

    def _reset_internal(self):
        super()._reset_internal()
        self._set_task_layout()
        self._success_hold_steps = 0

    def _box_contacts(self, target_body_ids):
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in self._box_body_ids and body_2 in target_body_ids) or (
                body_2 in self._box_body_ids and body_1 in target_body_ids
            ):
                return True
        return False

    def _box_contacts_floor(self):
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if body_1 in self._box_body_ids and self._is_floor_geom(contact.geom2):
                return True
            if body_2 in self._box_body_ids and self._is_floor_geom(contact.geom1):
                return True
        return False

    def _box_is_released(self):
        center = np.asarray(
            self.sim.data.body_xpos[self.sim.model.body_name2id(self.BOX_BODY)]
        )
        threshold = float(self._get_task_spec()["release_distance"])
        return all(
            np.linalg.norm(np.asarray(eef) - center) > threshold
            for eef in self.eef_positions()
        )

    def _box_is_stable(self):
        start, end = self.sim.model.get_joint_qvel_addr(self.BOX_JOINT)
        velocity = np.asarray(self.sim.data.qvel[start:end])
        spec = self._get_task_spec()
        return bool(
            np.linalg.norm(velocity[:3]) <= float(spec["max_linear_speed"])
            and np.linalg.norm(velocity[3:]) <= float(spec["max_angular_speed"])
        )

    def _raw_success(self):
        return bool(
            self._box_contacts_floor()
            and self._box_is_released()
            and self._box_is_stable()
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

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="take the cardboard box off the sofa and place it on the floor",
            task_id=type(self).__name__,
            target_entity="FloorPlan205:Box_5_13",
            source_support="FloorPlan205:Sofa_205_1_5",
            target_support="FloorPlan205:Floor",
            success_definition=(
                "the released cardboard box contacts the floor and remains stable "
                "continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomBoxFromSofaToFloor"]
