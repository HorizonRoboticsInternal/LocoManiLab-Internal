"""Scene 209: place the native statue into the open native cardboard box."""

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


class IThorLivingRoomStatueIntoBox(IThorLivingRoomEnv):
    SUPPORTED_SCENE_IDS = (209,)
    SPEC_TASK = "statue_into_box"
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
    BOX_BODY = "box_e7e029a9fdad1078011b9d2693eb5ef1_1_0_0"
    BOX_JOINT = BOX_BODY + "_jntfree_0"
    BOX_BASE_BODY = "box_e7e029a9fdad1078011b9d2693eb5ef1_1_5_0"
    BOX_BOTTOM_GEOM = BOX_BASE_BODY.replace("_1_5_0", "_1_5_0_collision_9")
    BOX_WALL_Y_POS = BOX_BASE_BODY.replace("_1_5_0", "_1_5_0_collision_10")
    BOX_WALL_Y_NEG = BOX_BASE_BODY.replace("_1_5_0", "_1_5_0_collision_11")
    BOX_WALL_X_NEG = BOX_BASE_BODY.replace("_1_5_0", "_1_5_0_collision_12")
    BOX_WALL_X_POS = BOX_BASE_BODY.replace("_1_5_0", "_1_5_0_collision_13")
    FLAP_JOINTS = (
        "box_e7e029a9fdad1078011b9d2693eb5ef1_1_1_0_box9flapouter2joint0_1",
        "box_e7e029a9fdad1078011b9d2693eb5ef1_1_2_0_box9flapinner1joint0_2",
        "box_e7e029a9fdad1078011b9d2693eb5ef1_1_3_0_box9flapinner2joint0_3",
        "box_e7e029a9fdad1078011b9d2693eb5ef1_1_4_0_box9flapouter1joint0_4",
    )
    NATIVE_STATUE_BODY = "statue_ad73194bc098e0c7a3246019c7443373_1_0_0"

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
                "table_top_z",
                "box_xy",
                "box_quat",
                "object_xy",
                "object_quat",
                "flap_qpos",
                "horizontal_tolerance",
                "height_tolerance",
                "hold_seconds",
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
        box_root = self.sim.model.body_name2id(self.BOX_BODY)
        statue = self.objects[self.STATUE_NAME]
        self._object_joint = statue.joints[0]
        object_root = self.sim.model.body_name2id(statue.root_body)
        self._box_geom_ids = self._collidable_geoms(self._body_tree_ids([box_root]))
        self._object_geom_ids = self._collidable_geoms(
            self._body_tree_ids([object_root])
        )
        self._box_bottom_geom_id = self.sim.model.geom_name2id(self.BOX_BOTTOM_GEOM)
        self._box_wall_geom_ids = {
            "y_pos": self.sim.model.geom_name2id(self.BOX_WALL_Y_POS),
            "y_neg": self.sim.model.geom_name2id(self.BOX_WALL_Y_NEG),
            "x_neg": self.sim.model.geom_name2id(self.BOX_WALL_X_NEG),
            "x_pos": self.sim.model.geom_name2id(self.BOX_WALL_X_POS),
        }

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

    def _set_scalar_joint(self, joint_name, value):
        self.sim.data.set_joint_qpos(joint_name, float(value))
        joint_id = self.sim.model.joint_name2id(joint_name)
        self.sim.data.qvel[int(self.sim.model.jnt_dofadr[joint_id])] = 0.0

    def _place_free_body(self, joint_name, geom_ids, xy, quat, bottom_z):
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:2] = np.asarray(xy, dtype=float)
        qpos[3:] = np.asarray(quat, dtype=float)
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.forward()
        points = self._geom_tree_points(geom_ids)
        qpos[2] += float(bottom_z) - float(points[:, 2].min()) - 0.0001
        self.sim.data.set_joint_qpos(joint_name, qpos)
        start, end = self.sim.model.get_joint_qvel_addr(joint_name)
        self.sim.data.qvel[start:end] = 0.0
        self.sim.forward()

    def _set_task_layout(self):
        spec = self._get_task_spec()
        flap_qpos = spec["flap_qpos"]
        for joint in self.FLAP_JOINTS:
            self._set_scalar_joint(joint, flap_qpos[joint])
        self._place_free_body(
            self.BOX_JOINT,
            self._box_geom_ids,
            spec["box_xy"],
            spec["box_quat"],
            spec["table_top_z"],
        )
        self._place_free_body(
            self._object_joint,
            self._object_geom_ids,
            spec["object_xy"],
            spec["object_quat"],
            spec["table_top_z"],
        )

    def _before_object_placement(self):
        super()._before_object_placement()
        self._set_task_layout()

    def _constrain_initial_jitter_candidate(self):
        self._set_task_layout()

    def _reset_internal(self):
        super()._reset_internal()
        self._set_task_layout()
        self._success_hold_steps = 0

    def _box_frame(self):
        walls = {
            key: self._geom_tree_points([geom_id])
            for key, geom_id in self._box_wall_geom_ids.items()
        }
        centers = {key: points.mean(axis=0) for key, points in walls.items()}
        x_axis = centers["x_pos"] - centers["x_neg"]
        y_axis = centers["y_pos"] - centers["y_neg"]
        x_axis /= np.linalg.norm(x_axis)
        y_axis -= np.dot(y_axis, x_axis) * x_axis
        y_axis /= np.linalg.norm(y_axis)
        z_axis = np.cross(x_axis, y_axis)
        if z_axis[2] < 0.0:
            z_axis *= -1.0
        rotation = np.column_stack((x_axis, y_axis, z_axis))
        origin = np.mean(list(centers.values()), axis=0)
        return origin, rotation, walls

    def _points_in_box_frame(self, points, frame=None):
        origin, rotation, _ = self._box_frame() if frame is None else frame
        return (np.asarray(points) - origin) @ rotation

    def _box_interior(self):
        frame = self._box_frame()
        walls = {
            key: self._points_in_box_frame(points, frame)
            for key, points in frame[2].items()
        }
        bottom = self._points_in_box_frame(
            self._geom_tree_points([self._box_bottom_geom_id]), frame
        )
        lower = np.array([walls["x_neg"][:, 0].max(), walls["y_neg"][:, 1].max()])
        upper = np.array([walls["x_pos"][:, 0].min(), walls["y_pos"][:, 1].min()])
        rim_top = min(float(points[:, 2].max()) for points in walls.values())
        return lower, upper, float(bottom[:, 2].max()), rim_top

    @staticmethod
    def _inside_open_box(
        points,
        lower,
        upper,
        base_top,
        rim_top,
        horizontal_tolerance,
        height_tolerance,
    ):
        points = np.asarray(points)
        bottom = float(points[:, 2].min())
        return bool(
            np.all(points[:, :2] >= np.asarray(lower) - horizontal_tolerance)
            and np.all(points[:, :2] <= np.asarray(upper) + horizontal_tolerance)
            and bottom >= float(base_top) - height_tolerance
            and bottom <= float(rim_top) + height_tolerance
        )

    def _raw_success(self):
        spec = self._get_task_spec()
        lower, upper, base_top, rim_top = self._box_interior()
        object_points = self._points_in_box_frame(
            self._geom_tree_points(self._object_geom_ids)
        )
        return self._inside_open_box(
            object_points,
            lower,
            upper,
            base_top,
            rim_top,
            float(spec["horizontal_tolerance"]),
            float(spec["height_tolerance"]),
        )

    def _post_action(self, action):
        result = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return result

    def _check_success(self):
        required = math.ceil(
            float(self._get_task_spec()["hold_seconds"]) * self.control_freq
        )
        return self._success_hold_steps >= required

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang="place the statue into the open cardboard box on the coffee table",
            task_id=type(self).__name__,
            target_entity="Stone_Statue_4",
            target_receptacle="FloorPlan209:Box_9_20",
            support="FloorPlan209:Coffee_Table_209_1_1",
            replaced_scene_entity="FloorPlan209:Towel_Statue_1_22",
            success_definition=(
                "the statue footprint and bottom remain inside the cardboard box "
                "continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomStatueIntoBox"]
