"""Place a cardboard box from a calibrated desk onto the Holodeck floor."""

import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.holodeck.tasks.pick_box_from_floor import (
    HolodeckPickBoxFromFloor,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckPlaceBoxOnFloor(HolodeckPickBoxFromFloor):
    """Move the calibrated desk box down until it contacts the room floor."""

    ATOMIC_SKILL = "PickPlace"
    TASK_FAMILY = "pick_place"
    SPEC_TASK = "place_box_on_floor"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    HOLD_SECONDS = 1.0
    CONTACT_MARGIN = 0.003
    USE_DEFAULT_ASSET_LAYOUT = False

    def _setup_baked_scene_fixtures(self):
        spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
        remove_scene_bodies = tuple(spec.get("remove_scene_bodies", ()))
        if remove_scene_bodies:
            removed = self.remove_scene_bodies(remove_scene_bodies)
            missing = set(remove_scene_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured box scene bodies are missing: "
                    f"{sorted(missing)}"
                )
        self._add_desk_support(spec)
        super()._setup_baked_scene_fixtures()

    def _add_desk_support(self, spec):
        """Add a level task-local support under the large box in narrow scenes."""
        if self.scene_name not in ("train_1664", "train_2025"):
            return
        body = self.find_scene_body(spec["table_body"])
        if body is None:
            raise RuntimeError(
                f"{self.scene_name}: missing desk for box support: "
                f"{spec['table_body']}"
            )
        support_top_z = float(spec["support_top_z"])
        body_pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        body_quat = np.fromstring(body.get("quat", "1 0 0 0"), sep=" ")
        from robosuite.utils import transform_utils as T

        body_rot = T.quat2mat(T.convert_quat(body_quat, to="xyzw"))
        local_quat = T.convert_quat(T.mat2quat(body_rot.T), to="wxyz")
        box_pos = np.asarray(spec["box_pose"]["pos"], dtype=float)
        support_center = np.asarray(
            [box_pos[0], box_pos[1], support_top_z - 0.005], dtype=float
        )
        local_pos = body_rot.T @ (support_center - body_pos)
        geom_name = f"{self.scene_name}_place_box_support"
        if body.find(f"./geom[@name='{geom_name}']") is not None:
            return
        ET.SubElement(
            body,
            "geom",
            name=geom_name,
            type="box",
            pos=" ".join(str(float(value)) for value in local_pos),
            quat=" ".join(str(float(value)) for value in local_quat),
            size="0.22 0.14 0.005",
            contype="8",
            conaffinity="15",
            group="0",
            friction="0.9 0.9 0.001",
            solimp="0.95 0.98 0.001",
            solref="0.004 1",
        )

    def _setup_task_fixtures(self):
        HolodeckEnv._setup_task_fixtures(self)
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=(
                "room_id",
                "table_body",
                "robot_pose",
                "box_pose",
                "box_scale",
            ),
        )
        table_body = self.task_spec["table_body"]
        if self.find_scene_body(table_body) is None:
            raise RuntimeError(f"{self.scene_name}: desk body is missing: {table_body}")
        self.desk_body = table_body
        pose = self.task_spec["robot_pose"]
        self.set_robot_base_pose(np.asarray(pose["pos"], dtype=float), float(pose["yaw"]))

    def _is_scene_valid(self):
        return self.find_scene_body(self.task_spec["table_body"]) is not None

    def _reset_internal(self):
        HolodeckEnv._reset_internal(self)
        box = self.objects["box"]
        qpos = self.sim.data.get_joint_qpos(box.joints[0]).copy()
        pose = self.task_spec["box_pose"]
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(box.joints[0], qpos)
        self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
        self.sim.forward()
        support_top_z = self.task_spec.get("support_top_z")
        if support_top_z is not None:
            points, _ = self._body_geom_points([box.root_body])
            if not len(points):
                raise RuntimeError(f"{self.scene_name}: box has no compiled geometry")
            # Use a shallow 1 mm overlap so MuJoCo emits a stable contact
            # instead of treating an exact zero-gap placement as separated.
            qpos[2] += float(support_top_z) - float(np.min(points[:, 2])) - 0.001
            self.sim.data.set_joint_qpos(box.joints[0], qpos)
            self.sim.data.set_joint_qvel(box.joints[0], np.zeros(6))
            self.sim.forward()
        self._box_initial_z = self._box_root_z()
        self._box_floor_geom_id = self.sim.model.geom_name2id("floor")
        desk_id = self.sim.model.body_name2id(self.desk_body)
        self._desk_body_ids = set(self._body_tree_ids([desk_id]))
        if self.count_body_collisions() != 0:
            raise RuntimeError(
                f"{self.scene_name}: box-on-floor reset has robot collision"
            )
        if self._box_contacts_floor():
            raise RuntimeError(f"{self.scene_name}: box starts on the floor")
        if not self._box_contacts_desk() and not self._box_at_support_height():
            raise RuntimeError(f"{self.scene_name}: box is not supported by the desk")
        self._success_hold = 0

    def _box_contacts_floor(self):
        box_id = self.sim.model.body_name2id(self.objects["box"].root_body)
        box_bodies = set(self._body_tree_ids([box_id]))
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            if self._box_floor_geom_id not in (contact.geom1, contact.geom2):
                continue
            other_geom = (
                contact.geom2
                if contact.geom1 == self._box_floor_geom_id
                else contact.geom1
            )
            if (
                int(self.sim.model.geom_bodyid[other_geom]) in box_bodies
                and float(contact.dist) <= self.CONTACT_MARGIN
            ):
                return True
        return False

    def _box_contacts_desk(self):
        box_id = self.sim.model.body_name2id(self.objects["box"].root_body)
        box_bodies = set(self._body_tree_ids([box_id]))
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body_1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body_2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body_1 in box_bodies and body_2 in self._desk_body_ids) or (
                body_2 in box_bodies and body_1 in self._desk_body_ids
            ):
                if float(contact.dist) <= self.CONTACT_MARGIN:
                    return True
        return False

    def _box_at_support_height(self):
        """Accept the calibrated shallow overlap when MuJoCo has no ncon yet."""
        support_top_z = self.task_spec.get("support_top_z")
        if support_top_z is None:
            return False
        points, _ = self._body_geom_points([self.objects["box"].root_body])
        return bool(
            len(points)
            and abs(float(np.min(points[:, 2])) - float(support_top_z))
            <= self.CONTACT_MARGIN + 0.001
        )

    def _post_action(self, action):
        _, done, info = HolodeckEnv._post_action(self, action)
        contacting = self._box_contacts_floor()
        self._success_hold = self._success_hold + 1 if contacting else 0
        success = bool(self._check_success())
        info["task_success"] = success
        info["success"] = success
        info["success_hold"] = self._success_hold
        return float(success), done, info

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _check_success(self):
        return self._success_hold >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = HolodeckEnv.get_ep_meta(self)
        meta["lang"] = "pick up the cardboard box from the desk and place it on the floor"
        meta["source_object"] = "box"
        meta["source_fixture"] = self.desk_body
        meta["target_object"] = "floor"
        meta["success_definition"] = (
            "the box must contact the room floor continuously "
            f"for {self.HOLD_SECONDS:.1f} s"
        )
        return meta


# Keep the source-to-target wording available to callers that prefer it.
HolodeckPlaceBoxFromDeskToFloor = HolodeckPlaceBoxOnFloor

__all__ = ["HolodeckPlaceBoxOnFloor", "HolodeckPlaceBoxFromDeskToFloor"]
