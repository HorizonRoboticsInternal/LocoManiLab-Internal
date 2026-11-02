"""iTHOR living-room close-laptop task."""

import numpy as np
from robocasa.utils.mjcf_xml_utils import mjcf_xml_bounds, quat_multiply

from robocasa.environments.ithor_atomic.articulation import (
    RoomCloseLaptop,
    RoomOpenArticulation,
)
from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomCloseLaptop(RoomCloseLaptop, IThorLivingRoomEnv):
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    SPEC_TASK = "close_laptop"
    HOLD_SECONDS = 1.0

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_laptop_task_spec(self):
        """Load placement calibration separately from exact-fixture joint specs."""
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=("laptop_pose", "robot_pose", "init_open_fraction"),
            )
            self._task_spec_scene = self.scene_name
        self.HOLD_SECONDS = float(self.task_spec.get("hold_seconds", 1.0))
        self.OPEN_INIT_FRACTION = float(self.task_spec["init_open_fraction"])
        if self.BEHAVIOR == "close":
            self.SUCCESS_CLOSE_FRACTION = float(
                self.task_spec["success_close_fraction"]
            )
        else:
            self.SUCCESS_OPEN_FRACTION = float(self.task_spec["success_open_fraction"])
        return self.task_spec

    def _get_scene_surface_calibration(self):
        return None

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_laptop_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _fixed_laptop_pose(self):
        pose = self._get_laptop_task_spec()["laptop_pose"]
        pos = np.asarray(pose.get("pos"), dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: laptop_pose needs XYZ + yaw"
            )
        yaw = float(pose["yaw"])
        yaw_quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
        upright_quat = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])
        quat = quat_multiply(yaw_quat, upright_quat)
        front = np.array([np.sin(yaw), -np.cos(yaw)], dtype=float)
        return pos, quat, front

    def _setup_task_fixtures(self):
        self._laptop_xml_path = self._laptop_asset_xml_path()
        (
            self._laptop_local_center,
            self._laptop_half_size,
            self._laptop_local_bottom,
        ) = mjcf_xml_bounds(self._laptop_xml_path)
        self.set_robot_base_pose(*self._get_scene_robot_pose_calibration())
        pos, quat, front = self._fixed_laptop_pose()
        self.target_fixture = self.create_scene_fixture(
            xml_path=self._laptop_xml_path,
            pos=pos,
            quat=quat,
            name="spawned_laptop",
            category="Laptop",
            source_category="Laptop",
            asset_id="Laptop_13",
            front=front,
            half_size=self._laptop_half_size,
        )
        self.fixture_refs["articulation_target"] = self.target_fixture

    def _is_scene_valid(self):
        return RoomOpenArticulation._is_scene_valid(self)

    def _initial_jitter_extra_joints(self):
        return ()

    def _initial_joint_qpos(self, joint):
        """Use the YAML open fraction for both open and close laptop tasks."""
        self._get_laptop_task_spec()
        return float(
            self.CLOSED_QPOS
            + self.OPEN_INIT_FRACTION * (joint.open_qpos - self.CLOSED_QPOS)
        )

    def _finalize_articulation_robot_pose(self, distance):
        self.sim.forward()
        return self.count_body_collisions() == 0

    def _move_laptop_to_fixed_pose(self):
        pos, quat, _ = self._fixed_laptop_pose()
        root_joint = self._laptop_free_joint()
        qpos = self.sim.data.get_joint_qpos(root_joint).copy()
        qpos[:3] = pos
        qpos[3:7] = quat
        self.sim.data.set_joint_qpos(root_joint, qpos)
        self.sim.forward()

    def _setup_scene(self):
        RoomOpenArticulation._setup_scene(self)
        self._move_laptop_to_fixed_pose()
        self._set_laptop_lid_limits()
        self.target_fixture_xy = self.target_fixture.world_center(self.sim)[:2].copy()
        self.target_joint_pos = self._joint_target_pos(self.target_semantic_joint)
        self.sim.data.set_joint_qpos(
            self.target_joint, self._initial_joint_qpos(self.target_semantic_joint)
        )
        self.sim.forward()
        self.HOLD_STEPS = int(round(self.HOLD_SECONDS * self.control_freq))
