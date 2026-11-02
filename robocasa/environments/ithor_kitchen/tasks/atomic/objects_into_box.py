"""iTHOR kitchen atomic tasks for placing one object into a box."""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
from robosuite.utils.mjcf_utils import string_to_array

from robocasa.environments.ithor_kitchen.tasks.atomic.pack_produce_into_box import (
    IThorKitchenPackProduceIntoBox,
)
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)


class LivingRoomStatueObject(MJCFObject):
    """Runtime adapter for the existing Living Room Dog Statue asset."""

    def __init__(self, name, source_xml, scale):
        tree = ET.parse(source_xml)
        root = tree.getroot()
        source_dir = os.path.dirname(source_xml)
        for asset in root.findall("./asset/*"):
            asset_path = asset.get("file")
            if asset_path and not os.path.isabs(asset_path):
                asset.set("file", os.path.abspath(os.path.join(source_dir, asset_path)))
        default = root.find("default")
        if default is not None:
            root.remove(default)
        for elem in root.iter():
            class_name = elem.attrib.pop("class", None)
            if elem.tag == "geom" and class_name == "__VISUAL_MJT__":
                elem.set("contype", "0")
                elem.set("conaffinity", "0")
                elem.set("group", "1")
            elif elem.tag == "geom" and class_name == "__DYNAMIC_MJT__":
                elem.set("contype", "1")
                elem.set("conaffinity", "1")
                elem.set("group", "0")
                elem.set("friction", "0.95 0.3 0.1")
        worldbody = root.find("worldbody")
        object_body = worldbody.find("body")
        for joint in list(object_body.findall("joint")):
            object_body.remove(joint)
        statue_body = object_body.find("body")
        statue_body.set("name", "object")
        statue_body.append(
            ET.Element(
                "geom",
                name="reg_bbox",
                type="box",
                pos="0.0451288 0 0.0029972",
                size="0.1346252 0.1049926 0.0618679",
                group="1",
                contype="0",
                conaffinity="0",
                rgba="0 1 0 0",
            )
        )
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", delete=False
        ) as file:
            temp_path = file.name
            tree.write(file, encoding="utf-8", xml_declaration=False)
        try:
            super().__init__(name=name, mjcf_path=temp_path, scale=scale)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


class SingleObjectIntoBoxMixin:
    """Single-object variant with an independent scene calibration."""

    ATOMIC_SKILL = "PlaceIntoReceptacle"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "pick_place"

    TARGET_NAME = None
    TARGET_GROUP = None
    TARGET_LANG = None
    TARGET_XML = None
    TARGET_SCALE = 1.0
    ITEMS = ()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        if self.TARGET_XML is not None:
            self._reinforce_box_floor()

    def _reinforce_box_floor(self):
        floor = next(
            geom
            for geom in self.usd_box.get_obj().iter("geom")
            if (geom.get("name") or "").endswith("box_floor_col")
        )
        pos = string_to_array(floor.get("pos"))
        size = string_to_array(floor.get("size"))
        floor_top = float(pos[2] + size[2])
        size[2] = max(float(size[2]), 0.01)
        pos[2] = floor_top - size[2]
        floor.set("pos", " ".join(str(value) for value in pos))
        floor.set("size", " ".join(str(value) for value in size))

    def _get_task_spec(self):
        if getattr(self, "_single_box_spec_scene", None) != self.scene_name:
            required = [
                "hold_seconds",
                "robot_pose",
                "support_region",
                "box_pos",
                "box_yaw",
                "object_pos",
            ]
            if self.TARGET_XML is not None:
                required.append("object_quat")
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=tuple(required),
            )
            self._single_box_spec_scene = self.scene_name
        return self.task_spec

    def _get_obj_cfgs(self):
        cfg = dict(name=self.TARGET_NAME, object_scale=self.TARGET_SCALE)
        if self.TARGET_XML is None:
            cfg["obj_groups"] = self.TARGET_GROUP
        else:
            cfg["obj_groups"] = self.TARGET_XML
        return [cfg]

    def _create_objects(self):
        if self.TARGET_XML is None:
            return super()._create_objects()
        self.objects = {}
        self.object_cfgs = self._get_obj_cfgs()
        model = LivingRoomStatueObject(
            name=self.TARGET_NAME,
            source_xml=self.TARGET_XML,
            scale=self.TARGET_SCALE,
        )
        self.objects[model.name] = model
        self.model.merge_objects([model])

    def _place_task_objects(self):
        spec = self._get_task_spec()
        self._set_free_joint_bottom_at(self.usd_box, spec["box_pos"])
        if self.TARGET_XML is None:
            self._set_free_joint_bottom_at(
                self.objects[self.TARGET_NAME], spec["object_pos"]
            )
        else:
            self._set_oriented_object_bottom_at(
                self.objects[self.TARGET_NAME],
                spec["object_pos"],
                spec["object_quat"],
            )
        self.sim.forward()

    def _set_oriented_object_bottom_at(self, obj, pos, quat):
        pos = np.asarray(pos, dtype=float)
        quat = np.asarray(quat, dtype=float)
        if (
            pos.shape != (3,)
            or quat.shape != (4,)
            or not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-5)
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: object_pos needs XYZ and "
                "object_quat must be normalized WXYZ"
            )
        joint_name = obj.joints[0]
        qpos = self.sim.data.get_joint_qpos(joint_name).copy()
        qpos[:3] = pos
        qpos[3:7] = quat
        self.sim.data.set_joint_qpos(joint_name, qpos)
        self.sim.forward()
        bbox_points = np.asarray(self.object_bbox_points(self.TARGET_NAME), dtype=float)
        bottom_z = float(np.min(bbox_points[:, 2]))
        qpos[2] += pos[2] - bottom_z
        self.sim.data.set_joint_qpos(joint_name, qpos)
        try:
            qvel_addr = self.sim.model.get_joint_qvel_addr(joint_name)
        except Exception:
            qvel_addr = None
        if qvel_addr is not None:
            if isinstance(qvel_addr, tuple):
                self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
            else:
                self.sim.data.qvel[qvel_addr] = 0.0

    def _raw_success(self):
        if self.TARGET_XML is not None:
            return bool(
                self._object_inside_box_volume()
                and self.check_contact(self.objects[self.TARGET_NAME], self.usd_box)
                and not self.g1_hand_grasping_object(self.TARGET_NAME)
            )
        return self.object_inside_box_collision(self.TARGET_NAME)

    def _object_inside_box_volume(self):
        root_id = self.sim.model.body_name2id(self.usd_box.root_body)
        root_pos = np.asarray(self.sim.data.body_xpos[root_id], dtype=float)
        root_rot = np.asarray(self.sim.data.body_xmat[root_id], dtype=float).reshape(
            3, 3
        )
        centers = {}
        sizes = {}
        for suffix in (
            "box_floor_col",
            "box_wall_x_pos",
            "box_wall_x_neg",
            "box_wall_y_pos",
            "box_wall_y_neg",
        ):
            geom_id = self.sim.model.geom_name2id(
                f"{self.usd_box.naming_prefix}{suffix}"
            )
            centers[suffix] = root_rot.T @ (self.sim.data.geom_xpos[geom_id] - root_pos)
            sizes[suffix] = np.asarray(self.sim.model.geom_size[geom_id], dtype=float)
        low = np.array(
            [
                centers["box_wall_x_neg"][0] + sizes["box_wall_x_neg"][0],
                centers["box_wall_y_neg"][1] + sizes["box_wall_y_neg"][1],
            ]
        )
        high = np.array(
            [
                centers["box_wall_x_pos"][0] - sizes["box_wall_x_pos"][0],
                centers["box_wall_y_pos"][1] - sizes["box_wall_y_pos"][1],
            ]
        )
        bbox_points = np.asarray(self.object_bbox_points(self.TARGET_NAME), dtype=float)
        local_points = (bbox_points - root_pos) @ root_rot
        local_center = 0.5 * (
            np.min(local_points, axis=0) + np.max(local_points, axis=0)
        )
        tolerance = 0.005
        return bool(
            np.all(local_center[:2] >= low - tolerance)
            and np.all(local_center[:2] <= high + tolerance)
            and local_center[2]
            >= (centers["box_floor_col"][2] + sizes["box_floor_col"][2] - tolerance)
            and local_center[2]
            <= (
                min(
                    centers["box_wall_x_pos"][2] + sizes["box_wall_x_pos"][2],
                    centers["box_wall_x_neg"][2] + sizes["box_wall_x_neg"][2],
                    centers["box_wall_y_pos"][2] + sizes["box_wall_y_pos"][2],
                    centers["box_wall_y_neg"][2] + sizes["box_wall_y_neg"][2],
                )
                - tolerance
            )
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang=f"put the {self.TARGET_LANG} into the box",
            target_entity=self.TARGET_NAME,
            target_receptacle="usd_box",
            success_definition=(
                (
                    f"the {self.TARGET_LANG} bounding-box center must remain "
                    "inside the box volume and contact the box without "
                    "being grasped, for the configured hold duration"
                    if self.TARGET_XML is not None
                    else f"the {self.TARGET_LANG} center must remain inside the "
                    "box for the configured hold duration"
                )
            ),
        )
        return ep_meta


class IThorKitchenAppleIntoBox(
    SingleObjectIntoBoxMixin, IThorKitchenPackProduceIntoBox
):
    SPEC_TASK = "apple_into_box"
    TARGET_NAME = "apple"
    TARGET_GROUP = "apple"
    TARGET_LANG = "apple"
    ITEMS = ((TARGET_NAME, TARGET_GROUP),)


class IThorKitchenCupIntoBox(SingleObjectIntoBoxMixin, IThorKitchenPackProduceIntoBox):
    SPEC_TASK = "cup_into_box"
    TARGET_NAME = "cup"
    TARGET_GROUP = "cup"
    TARGET_LANG = "cup"
    ITEMS = ((TARGET_NAME, TARGET_GROUP),)


class IThorKitchenStatueIntoBox(
    SingleObjectIntoBoxMixin, IThorKitchenPackProduceIntoBox
):
    SPEC_TASK = "statue_into_box"
    TARGET_NAME = "statue"
    TARGET_LANG = "statue"
    TARGET_XML = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../../../../models/assets/objects/thor/Miscellaneous Objects/Statue/"
            "Prefabs/Stone_Statue_4/Stone_Statue_4.xml",
        )
    )
    TARGET_SCALE = 0.30
    ITEMS = ((TARGET_NAME, TARGET_XML),)


__all__ = [
    "IThorKitchenAppleIntoBox",
    "IThorKitchenCupIntoBox",
    "IThorKitchenStatueIntoBox",
]
