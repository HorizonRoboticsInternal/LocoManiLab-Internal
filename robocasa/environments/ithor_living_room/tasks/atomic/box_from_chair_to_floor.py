"""Scene 201: move a task-local cardboard box from a chair to the floor."""

import math
import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomBoxFromChairToFloor(IThorLivingRoomEnv):
    SUPPORTED_SCENE_IDS = (201,)
    SPEC_TASK = "box_from_chair_to_floor"
    ATOMIC_SKILL = "PickPlaceBox"
    TASK_FAMILY = "pick_place"
    DIFFICULTY_LEVEL = "L2"
    INITIAL_POSE_JITTER = False

    BOX_XML = "models/assets/objects/manual/open_cardboard_box/model.xml"
    CHAIR_BODY = "chair_ae19c4fab4a513a60b5ce3c1ea49f205_3_0_0"
    CHAIR_SEAT_GEOM = "chair_ae19c4fab4a513a60b5ce3c1ea49f205_3_1_0_collision_2"

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
                "chair_xy",
                "chair_quat",
                "box_yaw",
                "seat_center_offset",
                "release_distance",
                "hold_seconds",
                "max_linear_speed",
                "max_angular_speed",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _get_obj_cfgs(self):
        return [dict(name="cardboard_box", obj_groups=self._box_xml_path())]

    def _box_xml_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.BOX_XML)
        )

    def sample_object(self, groups, **kwargs):
        if groups == self._box_xml_path():
            scale = kwargs.get("object_scale") or 1.0
            return (
                dict(mjcf_path=groups, scale=scale),
                dict(
                    mjcf_path=groups,
                    cat="cardboard_box",
                    groups=["cardboard_box"],
                    groups_containing_sampled_obj=["cardboard_box"],
                    split=None,
                ),
            )
        return super().sample_object(groups, **kwargs)

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualBoxFromChairSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _spawn_clearable_fixture_roots(self):
        roots = super()._spawn_clearable_fixture_roots()
        return roots - {self.sim.model.body_name2id(self.CHAIR_BODY)}

    def _setup_references(self):
        super()._setup_references()
        self._chair_body_ids = self._body_tree_ids(
            [self.sim.model.body_name2id(self.CHAIR_BODY)]
        )
        self._chair_seat_geom_id = self.sim.model.geom_name2id(self.CHAIR_SEAT_GEOM)
        box_root = self.sim.model.body_name2id(self.objects["cardboard_box"].root_body)
        self._box_body_ids = self._body_tree_ids([box_root])

    def _seat_pose(self):
        geom = self._chair_seat_geom_id
        model, data = self.sim.model._model, self.sim.data._data
        bounds = np.asarray(model.geom_aabb[geom], dtype=float)
        rotation = np.asarray(data.geom_xmat[geom], dtype=float).reshape(3, 3)
        signs = np.asarray(
            [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)]
        )
        points = (bounds[:3] + signs * bounds[3:]) @ rotation.T + data.geom_xpos[geom]
        center_xy = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
        offset = np.asarray(self._get_task_spec()["seat_center_offset"], dtype=float)
        if offset.shape != (2,):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: seat_center_offset needs XY"
            )
        return center_xy + offset, float(points[:, 2].max())

    def _place_box_on_chair(self):
        spec = self._get_task_spec()
        obj = self.objects["cardboard_box"]
        joint = obj.joints[0]
        xy, seat_top = self._seat_pose()
        yaw = float(spec["box_yaw"])
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[:2] = xy
        qpos[3:7] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.forward()
        qpos[2] += seat_top - self.object_bottom_z("cardboard_box") - 0.001
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()

    def _set_task_layout(self):
        chair_joint = self.CHAIR_BODY + "_jntfree_0"
        chair_qpos = self.sim.data.get_joint_qpos(chair_joint).copy()
        chair_qpos[:2] = np.asarray(self._get_task_spec()["chair_xy"], dtype=float)
        chair_quat = np.asarray(self._get_task_spec()["chair_quat"], dtype=float)
        if chair_quat.shape != (4,) or not np.isclose(
            np.linalg.norm(chair_quat), 1.0, atol=1e-5
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: chair_quat needs normalized WXYZ"
            )
        chair_qpos[3:7] = chair_quat
        self.sim.data.set_joint_qpos(chair_joint, chair_qpos)
        self.sim.data.set_joint_qvel(chair_joint, np.zeros(6))
        self.sim.forward()
        self._place_box_on_chair()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._set_task_layout()

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

    def _box_is_stable(self):
        joint = self.objects["cardboard_box"].joints[0]
        start, end = self.sim.model.get_joint_qvel_addr(joint)
        velocity = np.asarray(self.sim.data.qvel[start:end])
        spec = self._get_task_spec()
        return bool(
            np.linalg.norm(velocity[:3]) <= float(spec["max_linear_speed"])
            and np.linalg.norm(velocity[3:]) <= float(spec["max_angular_speed"])
        )

    def _raw_success(self):
        return bool(
            self._box_contacts_floor()
            and self.object_released(
                "cardboard_box",
                th=float(self._get_task_spec()["release_distance"]),
            )
            and self._box_is_stable()
        )

    def _reset_internal(self):
        super()._reset_internal()
        self._set_task_layout()
        self._success_hold_steps = 0

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
            lang="take the cardboard box off the chair and place it on the floor",
            task_id=type(self).__name__,
            target_entity="cardboard_box",
            source_fixture="FloorPlan201:Chair_201_1_28",
            target_fixture="FloorPlan201:Floor",
            success_definition=(
                "the released cardboard box contacts the floor and remains stable "
                "continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomBoxFromChairToFloor"]
