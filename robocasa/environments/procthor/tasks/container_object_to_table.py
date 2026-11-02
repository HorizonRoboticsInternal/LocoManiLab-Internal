"""ProcTHOR task: take an object from a drawer/cabinet to a generated table."""

import mujoco
import numpy as np
from robosuite.utils import transform_utils as T

from robocasa.environments.procthor.procthor import ProcTHOREnv
from robocasa.environments.procthor.tasks.place_to_place import (
    ProcTHORPlaceToPlaceTask,
    room_label,
)


class ProcTHORContainerObjectToTable(ProcTHORPlaceToPlaceTask):
    ATOMIC_SKILL = "PickPlaceOnSurface"
    DIFFICULTY_LEVEL = "L4"
    TASK_FAMILY = "pick_place"

    SOURCE_NAME = "item"
    SOURCE_LANG = "bottled drink"
    SOURCE_TABLE_NAME = "__unused_source_table__"
    TARGET_TABLE_NAME = "procthor_task_container_target_table"

    CONTAINER_CATEGORIES = {"cabinet", "drawer", "dresser"}
    CONTAINER_JOINT_TYPES = {"slide", "hinge"}
    CONTAINER_STAND_DISTANCE = 0.70
    OBJECT_IN_CONTAINER_Z_OFFSET = 0.08
    HORIZONTAL_OBJECT_PITCH = np.pi / 2
    REQUIRE_RELEASE = False

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("obj_registries", ("aigen", "objaverse", "lightwheel"))
        super().__init__(*args, **kwargs)

    def _setup_task_fixtures(self):
        ProcTHOREnv._setup_task_fixtures(self)
        self.container_fixture = self._select_container_fixture()
        if self.container_fixture is None:
            self._scene_invalid_reason = f"{self.scene_name}: no drawer/cabinet"
            return
        self.source_room = self.container_fixture.room_id
        self.target_room = self._pick_target_room()
        self._add_table(self.TARGET_TABLE_NAME, "target", self.target_room)

    def _is_scene_valid(self):
        return (
            ProcTHOREnv._is_scene_valid(self)
            and getattr(self, "container_fixture", None) is not None
        )

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups="bottled_drink",
                object_scale=1.5,
                placement=dict(margin=0.0),
            )
        ]

    def _setup_scene(self):
        ProcTHOREnv._setup_scene(self)
        self._success_hold = 0
        self.container_joint = self._select_container_joint()
        if self.container_joint is None:
            raise RuntimeError(
                f"{self.scene_name}: no slide/hinge drawer/cabinet joint"
            )
        self.container_joint_name = self.container_joint.name
        self.closed_qpos = float(self.container_joint.closed_qpos)
        self.open_qpos = float(self.container_joint.open_qpos)
        self.container_joint_range = abs(self.open_qpos - self.closed_qpos)
        self.container_joint_pos = self._joint_target_pos(self.container_joint_name)
        self.container_type = self._joint_type_name(self.container_joint_name)
        self.sim.data.set_joint_qpos(self.container_joint_name, self.closed_qpos)
        self.sim.forward()

        target_xy = self._free_point(
            self.target_room, margin=0.65, require_table_space=True
        )
        if target_xy is None:
            raise RuntimeError(f"{self.scene_name}: no free target-table floor point")
        self._move_body(self.TARGET_TABLE_NAME, target_xy)
        self._update_table_semantics(
            self.target_fixture, self.target_region, target_xy, self.target_room
        )
        target_center, _ = self.target_region.world_pose(self.sim)
        self.target_surface = (target_center, self.target_region.half_size[:2])

        if not self._place_robot_for_container():
            raise RuntimeError(f"{self.scene_name}: cannot place G1 at container")

    def _reset_internal(self):
        ProcTHOREnv._reset_internal(self)
        self.sim.data.set_joint_qpos(self.container_joint_name, self.closed_qpos)
        self.sim.forward()
        self._place_object_in_container()
        self._place_robot_for_container()

    def _apply_object_placements(self, placements):
        ProcTHOREnv._apply_object_placements(self, placements)
        self._place_object_in_container()

    # ---- container selection ----------------------------------------------------

    def _select_container_fixture(self):
        candidates = []
        for fixture in self.get_fixture(capabilities=("openable",), return_all=True):
            if fixture.category not in self.CONTAINER_CATEGORIES:
                continue
            if not fixture.joints:
                continue
            candidates.append(fixture)
        if not candidates:
            return None
        center = (
            self._room_center(candidates[0].room_id)
            if candidates[0].room_id
            else np.zeros(2)
        )
        return min(
            candidates,
            key=lambda fixture: (
                float(np.linalg.norm(np.asarray(fixture.center[:2]) - center)),
                fixture.name,
            ),
        )

    def _pick_target_room(self):
        rooms, edges = self.get_room_layout()
        source = self.source_room
        neighbors = [b for a, b in edges if a == source] + [
            a for a, b in edges if b == source
        ]
        choices = [room for room in neighbors if room in rooms and room != source]
        if not choices:
            choices = [room for room in rooms if room != source]
        if not choices:
            return source
        return max(choices, key=self._room_area)

    def _select_container_joint(self):
        joints = []
        for joint in self.container_fixture.joints.values():
            joint_type = self._joint_type_name(joint.name)
            if joint_type not in self.CONTAINER_JOINT_TYPES:
                continue
            text = f"{joint.name} {joint.role}".lower()
            if "handle" in text or "free" in text:
                continue
            joints.append(joint)
        if not joints:
            return None
        return min(
            joints, key=lambda joint: (self._joint_world_z(joint.name), joint.name)
        )

    def _joint_type_name(self, joint_name):
        try:
            joint_id = self.sim.model.joint_name2id(joint_name)
            joint_type = int(self.sim.model.jnt_type[joint_id])
        except Exception:
            return None
        if joint_type == int(mujoco.mjtJoint.mjJNT_SLIDE):
            return "slide"
        if joint_type == int(mujoco.mjtJoint.mjJNT_HINGE):
            return "hinge"
        return str(joint_type)

    def _joint_body_ids(self, joint_name):
        try:
            joint_id = self.sim.model.joint_name2id(joint_name)
            root_id = int(self.sim.model.jnt_bodyid[joint_id])
        except Exception:
            return []
        ids = {root_id}
        changed = True
        while changed:
            changed = False
            for body_id in range(self.sim.model.nbody):
                if body_id in ids:
                    continue
                if int(self.sim.model.body_parentid[body_id]) in ids:
                    ids.add(body_id)
                    changed = True
        return sorted(ids)

    def _joint_target_pos(self, joint_name):
        body_ids = self._joint_body_ids(joint_name)
        if not body_ids:
            return self.container_fixture.world_center(self.sim).copy()
        points = [
            np.asarray(self.sim.data.body_xpos[bid], dtype=float) for bid in body_ids
        ]
        for geom_id in range(self.sim.model.ngeom):
            if int(self.sim.model.geom_bodyid[geom_id]) in body_ids:
                points.append(np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float))
        return np.mean(np.asarray(points), axis=0)

    def _joint_world_z(self, joint_name):
        return float(self._joint_target_pos(joint_name)[2])

    # ---- object and robot placement --------------------------------------------

    def _place_object_in_container(self):
        qpos = self.sim.data.get_joint_qpos(
            self.objects[self.SOURCE_NAME].joints[0]
        ).copy()
        pos = self._object_spawn_pos()
        qpos[:3] = pos
        quat_xyzw = T.mat2quat(T.euler2mat([0.0, self.HORIZONTAL_OBJECT_PITCH, 0.0]))
        qpos[3:] = T.convert_quat(quat_xyzw, to="wxyz")
        self.sim.data.set_joint_qpos(self.objects[self.SOURCE_NAME].joints[0], qpos)
        self.sim.forward()
        qpos = self.sim.data.get_joint_qpos(
            self.objects[self.SOURCE_NAME].joints[0]
        ).copy()
        qpos[2] += pos[2] - self.object_bottom_z(self.SOURCE_NAME)
        self.sim.data.set_joint_qpos(self.objects[self.SOURCE_NAME].joints[0], qpos)
        self.sim.forward()

    def _object_spawn_pos(self):
        region = self.container_fixture.get_region("interior", required=False)
        if region is not None:
            center, _ = region.world_pose(self.sim)
            jitter = np.minimum(region.half_size[:2] * 0.35, 0.08)
            return center + np.array(
                [
                    self.rng.uniform(-jitter[0], jitter[0]),
                    self.rng.uniform(-jitter[1], jitter[1]),
                    self.OBJECT_IN_CONTAINER_Z_OFFSET,
                ]
            )
        front = np.asarray(self.container_fixture.front, dtype=float)
        front = front / (np.linalg.norm(front) + 1e-9)
        pos = self.container_joint_pos.copy()
        pos[:2] -= front * 0.10
        pos[2] += self.OBJECT_IN_CONTAINER_Z_OFFSET
        return pos

    def _place_robot_for_container(self):
        target = np.asarray(self.container_joint_pos[:2], dtype=float)
        front = np.asarray(self.container_fixture.front, dtype=float)
        front = front / (np.linalg.norm(front) + 1e-9)
        directions = [front]
        base_angle = float(np.arctan2(front[1], front[0]))
        directions.extend(
            np.array(
                [np.cos(base_angle + np.deg2rad(d)), np.sin(base_angle + np.deg2rad(d))]
            )
            for d in (35, -35, 70, -70, 110, -110, 180)
        )
        for direction in directions:
            right = np.array([-direction[1], direction[0]])
            for distance in (self.CONTAINER_STAND_DISTANCE, 0.85, 1.00, 1.20):
                for lateral in (0.0, 0.20, -0.20):
                    xy = target + direction * distance + right * lateral
                    yaw = float(np.arctan2(target[1] - xy[1], target[0] - xy[0]))
                    if (
                        self._set_robot_free_pose(xy, yaw)
                        and self._calibrate_robot_free_joint_height()
                    ):
                        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                        self.robot_base_ori = np.array([0.0, 0.0, yaw])
                        return True
        return False

    # ---- success / metadata -----------------------------------------------------

    def _raw_success(self):
        bbox = self.object_bbox_points(self.SOURCE_NAME)
        center = np.mean(bbox, axis=0)
        bottom = float(np.min(bbox[:, 2]))
        target_center, _ = self.target_region.world_pose(self.sim)
        half = self.target_region.half_size
        return bool(
            abs(float(center[0] - target_center[0])) <= float(half[0])
            and abs(float(center[1] - target_center[1])) <= float(half[1])
            and self.TABLE_HEIGHT - 0.08 <= bottom <= self.TABLE_HEIGHT + 0.15
        )

    def get_ep_meta(self):
        ep_meta = ProcTHOREnv.get_ep_meta(self)
        rooms, _ = self.get_room_layout()
        source_label = room_label(rooms[self.source_room]["type"])
        target_label = room_label(rooms[self.target_room]["type"])
        container_lang = "drawer" if self.container_type == "slide" else "cabinet"
        ep_meta.update(
            source_object=self.SOURCE_NAME,
            source_room=self.source_room,
            target_room=self.target_room,
            container_fixture=self.container_fixture.name,
            container_joint=self.container_joint_name,
            container_type=container_lang,
            source_pos=[float(v) for v in self.object_pos(self.SOURCE_NAME)],
            target_surface=[float(v) for v in self.target_surface[0]],
            lang=(
                f"open the {container_lang} in the {source_label}, take out the "
                f"{self.SOURCE_LANG}, navigate to the table in the {target_label}, "
                "and place it on top"
            ),
        )
        return ep_meta


__all__ = ["ProcTHORContainerObjectToTable"]
