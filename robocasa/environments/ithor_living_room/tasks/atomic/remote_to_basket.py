"""iTHOR living-room: place a remote control into a basket."""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np
import robocasa.utils.object_utils as OU
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_living_room.ithor_living_room import IThorLivingRoomEnv
from robocasa.environments.ithor_living_room.task_spec_utils import (
    set_free_joint_bottom_at,
)
from robocasa.models.scenes.ithor_livingroom_task_specs import (
    get_ithor_livingroom_task_spec,
)


class IThorLivingRoomRemoteToBasket(IThorLivingRoomEnv):
    """Pick up a remote control and place it into a basket."""

    ATOMIC_SKILL = "PlaceIntoReceptacle"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "pick_place"

    REMOTE_NAME = "remote"
    BASKET_NAME = "basket"
    REMOTE_XML = (
        "models/assets/objects/thor/Living Room Objects/Remotes/Prefabs/"
        "Remote_1/Remote_1.xml"
    )
    BASKET_XML = "models/assets/objects/lightwheel/basket/Basket039/model.xml"
    SUPPORTED_SCENE_IDS = (202, 207, 209, 210, 212, 226)
    SPEC_TASK = "remote_to_basket"

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_livingroom_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "basket_pos",
                    "basket_yaw",
                    "remote_pos",
                    "remote_yaw",
                    "robot_pose",
                    "success_xy_tolerance",
                    "release_distance",
                    "hold_seconds",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_surface_calibration(self):
        return None

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.success_xy_tolerance = float(spec["success_xy_tolerance"])
        self.release_distance = float(spec["release_distance"])
        self.HOLD_SECONDS = float(spec["hold_seconds"])
        if (
            min(self.success_xy_tolerance, self.release_distance, self.HOLD_SECONDS)
            <= 0.0
        ):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: success parameters "
                "must be positive"
            )
        self.set_robot_base_pose(*self._get_scene_robot_pose_calibration())

    def _asset_path(self, relative_path):
        return os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", relative_path)
        )

    def _get_obj_cfgs(self):
        return [
            dict(name=self.BASKET_NAME, obj_groups=self._asset_path(self.BASKET_XML)),
            dict(name=self.REMOTE_NAME, obj_groups=self._asset_path(self.REMOTE_XML)),
        ]

    def _materialize_remote_xml(self, source_path):
        source_dir = os.path.dirname(source_path)
        tree = ET.parse(source_path)
        root = tree.getroot()
        default = root.find("default")
        if default is not None:
            root.remove(default)
        for element in root.iter():
            class_name = element.attrib.pop("class", None)
            if element.tag != "geom":
                continue
            if class_name == "__VISUAL_MJT__":
                element.set("contype", "0")
                element.set("conaffinity", "0")
                element.set("group", "1")
            elif class_name in ("__DYNAMIC_MJT__", "__STRUCTURAL_MJT__"):
                element.set("contype", "1")
                element.set("conaffinity", "15")
                element.set("group", "0")
                element.set("friction", "0.9 0.9 0.001")
                element.set("solref", "0.025 1")
                element.set("solimp", "0.998 0.998 0.001")
        for asset in root.findall("./asset/*"):
            file_path = asset.get("file")
            if file_path:
                asset.set("file", os.path.abspath(os.path.join(source_dir, file_path)))

        worldbody = root.find("worldbody")
        remote_body = root.find("./worldbody/body")
        if remote_body is None:
            raise ValueError(f"{source_path}: missing remote root body")
        worldbody.remove(remote_body)
        outer_body = ET.SubElement(worldbody, "body")
        outer_body.append(remote_body)
        remote_body.set("name", "object")
        for joint in list(remote_body.findall("joint")):
            remote_body.remove(joint)

        model_bodies = list(remote_body.findall("body"))
        wrapper = ET.Element(
            "body",
            pos="0 0 0.0341",
            quat="0.7071068 0 0.7071068 0",
        )
        for body in model_bodies:
            remote_body.remove(body)
            wrapper.append(body)
        remote_body.append(wrapper)
        remote_body.append(
            ET.Element(
                "geom",
                name="reg_bbox",
                type="box",
                pos="0 0 0.0341",
                size="0.108 0.010 0.0341",
                group="1",
                contype="0",
                conaffinity="0",
                rgba="0 1 0 0",
            )
        )

        handle = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", prefix="robocasa_remote_1_", delete=False
        )
        with handle:
            tree.write(handle, encoding="utf-8", xml_declaration=True)
        self._remote_temp_xmls.append(handle.name)
        return handle.name

    def _create_objects(self):
        self._remote_temp_xmls = []
        try:
            super()._create_objects()
        finally:
            for path in self._remote_temp_xmls:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
            self._remote_temp_xmls = []

    def sample_object(self, groups, **kwargs):
        if groups == self._asset_path(self.REMOTE_XML):
            scale = kwargs.get("object_scale") or 1.0
            temp_xml = self._materialize_remote_xml(groups)
            return (
                dict(mjcf_path=temp_xml, scale=scale),
                dict(
                    mjcf_path=groups,
                    cat="remote_control",
                    groups=["remote_control"],
                    groups_containing_sampled_obj=["remote_control"],
                    split=None,
                ),
            )
        return super().sample_object(groups, **kwargs)

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name="ManualLivingRoomRemoteToBasketSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _reset_internal(self):
        super()._reset_internal()
        spec = self._get_task_spec()
        set_free_joint_bottom_at(
            self,
            self.objects[self.BASKET_NAME],
            spec["basket_pos"],
            yaw=spec["basket_yaw"],
        )
        set_free_joint_bottom_at(
            self,
            self.objects[self.REMOTE_NAME],
            spec["remote_pos"],
            yaw=spec["remote_yaw"],
        )
        self.sim.forward()
        self._success_hold_steps = 0

    def _setup_scene(self):
        super()._setup_scene()
        self._success_hold_steps = 0
        self.HOLD_STEPS = int(round(self.HOLD_SECONDS * self.control_freq))

    def _raw_success(self):
        return bool(
            OU.check_obj_in_receptacle(
                self,
                self.REMOTE_NAME,
                self.BASKET_NAME,
                th=self.success_xy_tolerance,
            )
            and self.object_released(self.REMOTE_NAME, th=self.release_distance)
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "pick up the remote control and place it in the basket"
        return ep_meta


__all__ = ["IThorLivingRoomRemoteToBasket"]
