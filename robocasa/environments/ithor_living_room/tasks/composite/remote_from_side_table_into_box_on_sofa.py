"""Scene 205: carry the upright remote into an open box on the sofa."""

import math

import numpy as np

from robocasa.environments.ithor_composite.base import (
    IThorAtomicTask,
    IThorMultiStageCompositeMixin,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomRemoteFromSideTableIntoBoxOnSofa(
    IThorMultiStageCompositeMixin,
    IThorLivingRoomEnv,
):
    """Take the upright remote from the side table and put it in the sofa box."""

    SUPPORTED_SCENE_IDS = (205,)
    SPEC_TASK = "remote_from_side_table_into_box_on_sofa"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    INITIAL_POSE_JITTER = False

    REMOTE_BODY = "remotecontrol_9e0a946ebd9862fe7ffbf14025fdf9f9_1_0_0"
    REMOTE_JOINT = REMOTE_BODY + "_jntfree_0"
    SIDE_TABLE_BODY = "stand_9985276433646fd9350dc0cc79ad70c5_1_0_0"
    SOFA_BODY = "sofa_2e5f2b1b88237f983400420d29e5f613_1_0_0"
    BOX_BODY = "box_376bd8a46eebebb53a1cbbec2a1cada0_1_0_0"
    BOX_JOINT = BOX_BODY + "_jntfree_0"
    BOX_BASE_BODY = "box_376bd8a46eebebb53a1cbbec2a1cada0_1_5_0"
    BOX_BOTTOM_GEOM = BOX_BASE_BODY + "_collision_9"
    BOX_WALL_Y_POS = BOX_BASE_BODY + "_collision_10"
    BOX_WALL_Y_NEG = BOX_BASE_BODY + "_collision_11"
    BOX_WALL_X_NEG = BOX_BASE_BODY + "_collision_12"
    BOX_WALL_X_POS = BOX_BASE_BODY + "_collision_13"
    FLAP_JOINTS = (
        "box_376bd8a46eebebb53a1cbbec2a1cada0_1_1_0_box5flapinner1joint0_1",
        "box_376bd8a46eebebb53a1cbbec2a1cada0_1_2_0_box5flapouter1joint0_2",
        "box_376bd8a46eebebb53a1cbbec2a1cada0_1_3_0_box5flapouter2joint0_3",
        "box_376bd8a46eebebb53a1cbbec2a1cada0_1_4_0_box5flapinner2joint0_4",
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
                "remote_xy",
                "remote_quat",
                "box_pos",
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

    def _get_atomic_tasks(self):
        return (
            IThorAtomicTask(
                name="take_remote_from_side_table",
                success="_remote_removed_from_side_table",
                hold_steps=5,
            ),
            IThorAtomicTask(
                name="put_remote_into_box_on_sofa",
                success="_remote_in_box_on_sofa",
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
            self.sim.model.body_name2id(self.REMOTE_BODY),
            self.sim.model.body_name2id(self.SIDE_TABLE_BODY),
            self.sim.model.body_name2id(self.SOFA_BODY),
            self.sim.model.body_name2id(self.BOX_BODY),
        }
        return roots - protected

    def _setup_references(self):
        super()._setup_references()
        remote_root = self.sim.model.body_name2id(self.REMOTE_BODY)
        table_root = self.sim.model.body_name2id(self.SIDE_TABLE_BODY)
        sofa_root = self.sim.model.body_name2id(self.SOFA_BODY)
        box_root = self.sim.model.body_name2id(self.BOX_BODY)
        self._remote_geom_ids = self._collidable_geoms(
            self._body_tree_ids([remote_root])
        )
        self._side_table_geom_ids = self._collidable_geoms(
            self._body_tree_ids([table_root])
        )
        self._sofa_body_ids = self._body_tree_ids([sofa_root])
        self._box_body_ids = self._body_tree_ids([box_root])
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

    def _place_remote_on_side_table(self):
        spec = self._get_task_spec()
        quat = np.asarray(spec["remote_quat"], dtype=float)
        qpos = self.sim.data.get_joint_qpos(self.REMOTE_JOINT).copy()
        qpos[:2] = np.asarray(spec["remote_xy"], dtype=float)
        qpos[3:7] = quat
        self.sim.data.set_joint_qpos(self.REMOTE_JOINT, qpos)
        self.sim.forward()
        table_top = float(self._geom_points(self._side_table_geom_ids)[:, 2].max())
        remote_bottom = float(self._geom_points(self._remote_geom_ids)[:, 2].min())
        qpos[2] += table_top - remote_bottom - 0.001
        self.sim.data.set_joint_qpos(self.REMOTE_JOINT, qpos)
        self.sim.data.set_joint_qvel(self.REMOTE_JOINT, np.zeros(6))
        self.sim.forward()

    def _place_open_box_on_sofa(self):
        spec = self._get_task_spec()
        for joint in self.FLAP_JOINTS:
            self._set_scalar_joint(joint, spec["flap_qpos"][joint])
        qpos = self.sim.data.get_joint_qpos(self.BOX_JOINT).copy()
        qpos[:3] = np.asarray(spec["box_pos"], dtype=float)
        qpos[3:7] = np.asarray(spec["box_quat"], dtype=float)
        self.sim.data.set_joint_qpos(self.BOX_JOINT, qpos)
        self.sim.data.set_joint_qvel(self.BOX_JOINT, np.zeros(6))
        self.sim.forward()

    def _set_task_layout(self):
        self._place_open_box_on_sofa()
        self._place_remote_on_side_table()

    def _before_object_placement(self):
        super()._before_object_placement()
        self._set_task_layout()

    def _constrain_initial_jitter_candidate(self):
        self._set_task_layout()

    def _remote_on_side_table(self):
        remote_points = self._geom_points(self._remote_geom_ids)
        table_points = self._geom_points(self._side_table_geom_ids)
        remote_center = (remote_points.min(axis=0) + remote_points.max(axis=0)) / 2
        table_low = table_points.min(axis=0)
        table_high = table_points.max(axis=0)
        return bool(
            np.all(remote_center[:2] >= table_low[:2])
            and np.all(remote_center[:2] <= table_high[:2])
            and abs(float(remote_points[:, 2].min()) - float(table_high[2])) <= 0.05
        )

    def _remote_removed_from_side_table(self):
        return not self._remote_on_side_table()

    def _box_contacts_sofa(self):
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in self._box_body_ids and body_2 in self._sofa_body_ids) or (
                body_2 in self._box_body_ids and body_1 in self._sofa_body_ids
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

    @staticmethod
    def _remote_within_expanded_box(
        remote,
        lower,
        upper,
        base_top,
        rim_top,
        horizontal_tolerance,
        height_tolerance,
    ):
        """Accept an upright or tilted remote whose center and lower end enter the box."""

        remote = np.asarray(remote, dtype=float)
        remote_center = (remote.min(axis=0) + remote.max(axis=0)) / 2
        remote_bottom = float(remote[:, 2].min())
        return bool(
            np.all(remote_center[:2] >= np.asarray(lower) - horizontal_tolerance)
            and np.all(remote_center[:2] <= np.asarray(upper) + horizontal_tolerance)
            and remote_bottom >= float(base_top) - height_tolerance
            and remote_bottom <= float(rim_top) + height_tolerance
        )

    def _remote_inside_box(self):
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
        remote = self._points_in_box_frame(
            self._geom_points(self._remote_geom_ids), frame
        )
        spec = self._get_task_spec()
        tolerance = float(spec["horizontal_tolerance"])
        height_tolerance = float(spec["height_tolerance"])
        return self._remote_within_expanded_box(
            remote,
            lower,
            upper,
            base_top,
            rim_top,
            tolerance,
            height_tolerance,
        )

    def _remote_in_box_on_sofa(self):
        return bool(self._box_contacts_sofa() and self._remote_inside_box())

    def _success_region_bounds(self):
        return []

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "take the upright remote control from the small table beside the "
                "television, walk to the long sofa, and put the remote into the "
                "open cardboard box on the sofa"
            ),
            task_id=type(self).__name__,
            target_entity="FloorPlan205:Remote_4_15",
            source_support="FloorPlan205:Side_Table_205_2_1",
            target_receptacle="FloorPlan205:Box_5_13",
            target_support="FloorPlan205:Sofa_205_1_5",
            initial_orientation="standing upright with the long axis vertical",
            success_definition=(
                "after the remote leaves the side table, keep its center and lower "
                "end within the expanded open-box volume while the box remains on "
                "the long sofa continuously for 1 second; the upright remote may "
                "extend above the rim"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomRemoteFromSideTableIntoBoxOnSofa"]
