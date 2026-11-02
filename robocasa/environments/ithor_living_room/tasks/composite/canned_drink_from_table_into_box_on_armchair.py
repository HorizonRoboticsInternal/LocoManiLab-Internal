"""Scene 226: carry a canned drink into an open box on the armchair."""

import math
import os

import numpy as np
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_kitchen.tasks.atomic.stand_canned_drink_upright import (
    IThorKitchenStandCannedDrinkUpright,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomCannedDrinkFromTableIntoBoxOnArmchair(
    IThorMultiStageCompositeMixin,
    IThorLivingRoomEnv,
):
    """Take the table can to the open box on the nearby armchair."""

    SUPPORTED_SCENE_IDS = (226,)
    SPEC_TASK = "canned_drink_from_table_into_box_on_armchair"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False

    CAN_NAME = "canned_drink"
    CAN_XML = IThorKitchenStandCannedDrinkUpright.CAN_XML
    TABLE_BODY = "coffeetable_9ffe419303e0e264ed57bd3b87ddecb2_1_0_0"
    ARMCHAIR_BODY = "armchair_2bbe61654860561b5e0240380a9b5943_1_0_0"
    ARMCHAIR_SEAT_GEOM = "armchair_2bbe61654860561b5e0240380a9b5943_1_1_0_collision_3"
    BOX_BODY = "box_b1c47d67056cd714cb582cc351224939_1_0_0"
    BOX_JOINT = BOX_BODY + "_jntfree_0"
    BOX_BASE_BODY = "box_b1c47d67056cd714cb582cc351224939_1_4_0"
    BOX_BOTTOM_GEOM = BOX_BASE_BODY + "_collision_7"
    BOX_WALL_X_NEG = BOX_BASE_BODY + "_collision_8"
    BOX_WALL_X_POS = BOX_BASE_BODY + "_collision_9"
    BOX_WALL_Y_NEG = BOX_BASE_BODY + "_collision_10"
    BOX_WALL_Y_POS = BOX_BASE_BODY + "_collision_11"
    FLAP_JOINTS = (
        "box_b1c47d67056cd714cb582cc351224939_1_1_0_" "box26flapouter1joint0_1",
        "box_b1c47d67056cd714cb582cc351224939_1_2_0_" "box26flapouter2joint0_2",
        "box_b1c47d67056cd714cb582cc351224939_1_3_0_" "box26flapinner2joint0_3",
        "box_b1c47d67056cd714cb582cc351224939_1_5_0_" "box26flapinner1joint0_4",
    )

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
                "can_xy",
                "can_yaw",
                "box_xy",
                "box_quat",
                "flap_qpos",
                "horizontal_tolerance",
                "height_tolerance",
                "hold_seconds",
            ),
        )

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _can_xml_path(self):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.CAN_XML)
        )

    def _get_obj_cfgs(self):
        return [dict(name=self.CAN_NAME, obj_groups=self._can_xml_path())]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualCanToArmchairBoxSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="take_canned_drink_from_table",
                success="_can_removed_from_table",
                hold_steps=5,
            ),
            IThorAtomicTask(
                name="put_canned_drink_into_box_on_armchair",
                success="_can_in_box_on_armchair",
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
        protected = {
            self.sim.model.body_name2id(self.TABLE_BODY),
            self.sim.model.body_name2id(self.ARMCHAIR_BODY),
            self.sim.model.body_name2id(self.BOX_BODY),
        }
        return roots - protected

    def _setup_references(self):
        super()._setup_references()
        can = self.objects[self.CAN_NAME]
        can_root = self.sim.model.body_name2id(can.root_body)
        table_root = self.sim.model.body_name2id(self.TABLE_BODY)
        armchair_root = self.sim.model.body_name2id(self.ARMCHAIR_BODY)
        box_root = self.sim.model.body_name2id(self.BOX_BODY)
        self._can_geom_ids = self._collidable_geoms(self._body_tree_ids([can_root]))
        self._table_geom_ids = self._collidable_geoms(self._body_tree_ids([table_root]))
        self._armchair_body_ids = self._body_tree_ids([armchair_root])
        self._box_body_ids = self._body_tree_ids([box_root])
        self._armchair_seat_geom_id = self.sim.model.geom_name2id(
            self.ARMCHAIR_SEAT_GEOM
        )
        self._box_bottom_geom_id = self.sim.model.geom_name2id(self.BOX_BOTTOM_GEOM)
        self._box_wall_geom_ids = {
            "y_pos": self.sim.model.geom_name2id(self.BOX_WALL_Y_POS),
            "y_neg": self.sim.model.geom_name2id(self.BOX_WALL_Y_NEG),
            "x_neg": self.sim.model.geom_name2id(self.BOX_WALL_X_NEG),
            "x_pos": self.sim.model.geom_name2id(self.BOX_WALL_X_POS),
        }

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

    def _set_scalar_joint(self, joint_name, value):
        self.sim.data.set_joint_qpos(joint_name, float(value))
        joint_id = self.sim.model.joint_name2id(joint_name)
        self.sim.data.qvel[int(self.sim.model.jnt_dofadr[joint_id])] = 0.0

    def _place_can_on_table(self):
        spec = self._get_task_spec()
        can = self.objects[self.CAN_NAME]
        joint = can.joints[0]
        yaw = float(spec["can_yaw"])
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[:2] = np.asarray(spec["can_xy"], dtype=float)
        qpos[3:7] = np.asarray([math.cos(yaw / 2.0), 0.0, 0.0, math.sin(yaw / 2.0)])
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.forward()
        table_top = float(self._geom_points(self._table_geom_ids)[:, 2].max())
        can_bottom = float(self._geom_points(self._can_geom_ids)[:, 2].min())
        qpos[2] += table_top - can_bottom - 0.001
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))
        self.sim.forward()

    def _place_open_box_on_armchair(self):
        spec = self._get_task_spec()
        for joint in self.FLAP_JOINTS:
            self._set_scalar_joint(joint, spec["flap_qpos"][joint])
        qpos = self.sim.data.get_joint_qpos(self.BOX_JOINT).copy()
        qpos[:2] = np.asarray(spec["box_xy"], dtype=float)
        qpos[3:7] = np.asarray(spec["box_quat"], dtype=float)
        self.sim.data.set_joint_qpos(self.BOX_JOINT, qpos)
        self.sim.forward()
        seat_top = float(self._geom_points([self._armchair_seat_geom_id])[:, 2].max())
        box_bottom = float(self._geom_points([self._box_bottom_geom_id])[:, 2].min())
        qpos[2] += seat_top - box_bottom - 0.001
        self.sim.data.set_joint_qpos(self.BOX_JOINT, qpos)
        self.sim.data.set_joint_qvel(self.BOX_JOINT, np.zeros(6))
        self.sim.forward()

    def _set_task_layout(self):
        self._place_open_box_on_armchair()
        self._place_can_on_table()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._set_task_layout()

    def _constrain_initial_jitter_candidate(self):
        self._set_task_layout()

    def _can_on_table(self):
        can = self._geom_points(self._can_geom_ids)
        table = self._geom_points(self._table_geom_ids)
        center = (can.min(axis=0) + can.max(axis=0)) / 2
        return bool(
            np.all(center[:2] >= table.min(axis=0)[:2])
            and np.all(center[:2] <= table.max(axis=0)[:2])
            and abs(float(can[:, 2].min()) - float(table[:, 2].max())) <= 0.05
        )

    def _can_removed_from_table(self):
        return not self._can_on_table()

    def _box_contacts_armchair(self):
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in self._box_body_ids and body_2 in self._armchair_body_ids) or (
                body_2 in self._box_body_ids and body_1 in self._armchair_body_ids
            ):
                return True
        return False

    def _box_frame(self):
        walls = {
            key: self._geom_points([geom_id])
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

    @staticmethod
    def _points_in_box_frame(points, frame):
        origin, rotation, _ = frame
        return (np.asarray(points) - origin) @ rotation

    def _can_inside_box(self):
        frame = self._box_frame()
        walls = {
            key: self._points_in_box_frame(points, frame)
            for key, points in frame[2].items()
        }
        bottom = self._points_in_box_frame(
            self._geom_points([self._box_bottom_geom_id]), frame
        )
        lower = np.array([walls["x_neg"][:, 0].max(), walls["y_neg"][:, 1].max()])
        upper = np.array([walls["x_pos"][:, 0].min(), walls["y_pos"][:, 1].min()])
        base_top = float(bottom[:, 2].max())
        rim_top = min(float(points[:, 2].max()) for points in walls.values())
        can = self._points_in_box_frame(self._geom_points(self._can_geom_ids), frame)
        spec = self._get_task_spec()
        horizontal_tolerance = float(spec["horizontal_tolerance"])
        height_tolerance = float(spec["height_tolerance"])
        can_bottom = float(can[:, 2].min())
        return bool(
            np.all(can[:, :2] >= lower - horizontal_tolerance)
            and np.all(can[:, :2] <= upper + horizontal_tolerance)
            and can_bottom >= base_top - height_tolerance
            and can_bottom <= rim_top + height_tolerance
        )

    def _can_in_box_on_armchair(self):
        return bool(
            self._box_contacts_armchair()
            and self._can_inside_box()
            and self.object_released(self.CAN_NAME)
        )

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "take the canned drink from the coffee table, walk to the nearby "
                "armchair, and put the canned drink into the open cardboard box"
            ),
            task_id=type(self).__name__,
            target_entity="canned_drink",
            source_support="FloorPlan226:Coffee_Table_203_1_13",
            target_receptacle="FloorPlan226:Box_26_0",
            target_support="FloorPlan226:Armchair_226_1_15",
            success_definition=(
                "after the canned drink leaves the coffee table, release it inside "
                "the open cardboard box while the box remains on the armchair "
                "continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomCannedDrinkFromTableIntoBoxOnArmchair"]
