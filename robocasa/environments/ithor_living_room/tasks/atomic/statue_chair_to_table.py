"""Scene 205: move the native statue from a dining chair to the table."""

import math
import os

import numpy as np

from robocasa.environments.ithor_kitchen.tasks.atomic.objects_into_box import (
    LivingRoomStatueObject,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomStatueChairToTable(IThorLivingRoomEnv):
    SUPPORTED_SCENE_IDS = (205,)
    SPEC_TASK = "statue_chair_to_table"
    ATOMIC_SKILL = "PickPlaceStatue"
    TASK_FAMILY = "pick_place"
    DIFFICULTY_LEVEL = "L2"
    INITIAL_POSE_JITTER = False

    STATUE_NAME = "statue"
    STATUE_XML = (
        "models/assets/objects/thor/Miscellaneous Objects/Statue/Prefabs/"
        "Stone_Statue_4/Stone_Statue_4.xml"
    )
    STATUE_SCALE = 0.5
    CHAIR_BODY = "chair_4e34d3fa7376ee5f6db663a0032cd3c5_1_0_0"
    CHAIR_JOINT = CHAIR_BODY + "_jntfree_0"
    CHAIR_SEAT_GEOM = "chair_4e34d3fa7376ee5f6db663a0032cd3c5_1_1_0_collision_3"
    NATIVE_STATUE_BODY = "statue_67e198ecd7bd687ce6b619c6d9b6331b_1_0_0"
    TABLE_GEOM = "diningtable_7bd9af02966b7e2fb8bfb44fd45f9dc7_1_1_0_collision_2"

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
                "chair_pose",
                "statue_xy",
                "statue_quat",
                "hold_seconds",
                "height_tolerance",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _statue_xml_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.STATUE_XML)
        )

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.STATUE_NAME,
                obj_groups=self._statue_xml_path(),
                object_scale=self.STATUE_SCALE,
            )
        ]

    def _create_objects(self):
        self.objects = {}
        self.object_cfgs = self._get_obj_cfgs()
        statue = LivingRoomStatueObject(
            name=self.STATUE_NAME,
            source_xml=self._statue_xml_path(),
            scale=self.STATUE_SCALE,
        )
        self.objects[statue.name] = statue
        self.model.merge_objects([statue])

    def _setup_baked_scene_fixtures(self):
        super()._setup_baked_scene_fixtures()
        removed = self.remove_scene_bodies((self.NATIVE_STATUE_BODY,))
        if self.NATIVE_STATUE_BODY not in removed:
            raise RuntimeError(
                f"{self.scene_name}: could not replace the native statue"
            )

    def _setup_references(self):
        super()._setup_references()
        statue = self.objects[self.STATUE_NAME]
        self._statue_joint = statue.joints[0]
        statue_root = self.sim.model.body_name2id(statue.root_body)
        self._statue_geom_ids = self._collidable_geoms(
            self._body_tree_ids([statue_root])
        )
        self._seat_geom_id = self.sim.model.geom_name2id(self.CHAIR_SEAT_GEOM)
        self._table_geom_id = self.sim.model.geom_name2id(self.TABLE_GEOM)

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

    def _statue_points(self):
        return self._geom_tree_points(self._statue_geom_ids)

    def _set_task_layout(self):
        spec = self._get_task_spec()
        chair_pose = spec["chair_pose"]
        chair_qpos = np.concatenate(
            (
                np.asarray(chair_pose["pos"], dtype=float),
                np.asarray(chair_pose["quat"], dtype=float),
            )
        )
        self.sim.data.set_joint_qpos(self.CHAIR_JOINT, chair_qpos)
        self.sim.forward()

        seat_points = self._geom_tree_points([self._seat_geom_id])
        seat_top = float(seat_points[:, 2].max())
        statue_qpos = self.sim.data.get_joint_qpos(self._statue_joint).copy()
        statue_qpos[3:] = np.asarray(spec["statue_quat"], dtype=float)
        self.sim.data.set_joint_qpos(self._statue_joint, statue_qpos)
        self.sim.forward()
        points = self._statue_points()
        center_xy = (points[:, :2].min(axis=0) + points[:, :2].max(axis=0)) / 2
        statue_qpos[:2] += np.asarray(spec["statue_xy"]) - center_xy
        statue_qpos[2] += seat_top - points[:, 2].min() - 0.001
        self.sim.data.set_joint_qpos(self._statue_joint, statue_qpos)

        for joint in (self.CHAIR_JOINT, self._statue_joint):
            start, end = self.sim.model.get_joint_qvel_addr(joint)
            self.sim.data.qvel[start:end] = 0.0
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

    def _table_bounds_and_top(self):
        points = self._geom_tree_points([self._table_geom_id])
        return (
            points[:, :2].min(axis=0),
            points[:, :2].max(axis=0),
            float(points[:, 2].max()),
        )

    @staticmethod
    def _on_table(points, lower, upper, top, height_tolerance):
        points = np.asarray(points)
        return bool(
            np.all(points[:, :2] >= np.asarray(lower) - 0.002)
            and np.all(points[:, :2] <= np.asarray(upper) + 0.002)
            and abs(float(points[:, 2].min()) - float(top)) <= height_tolerance
        )

    def _raw_success(self):
        lower, upper, top = self._table_bounds_and_top()
        return self._on_table(
            self._statue_points(),
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
            lang="move the statue from the chair to the dining table",
            task_id=type(self).__name__,
            target_entity="Stone_Statue_4",
            source_support="FloorPlan205:Chair_002_1_1",
            target_table="FloorPlan205:Dining_Table_205_1_12",
            replaced_scene_entity="FloorPlan205:Waving_Statue_3_23",
            success_definition="whole statue footprint on the dining table at tabletop height continuously for 1 second",
        )
        return meta


__all__ = ["IThorLivingRoomStatueChairToTable"]
