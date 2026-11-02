"""Actuated conveyor fixture for TeleLab environments."""

import os
from copy import deepcopy

import numpy as np
from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import (
    array_to_string,
    get_elements,
    scale_mjcf_model,
    string_to_array,
)

from robocasa.models import assets_root


class ConveyorBeltFixture(MujocoXML):
    """Complete conveyor MJCF model, including joints and actuators.

    This intentionally derives from :class:`MujocoXML`, rather than the usual
    object-style ``Fixture`` base. ``ManipulationTask.merge_objects`` only
    appends object bodies and assets; using ``MujocoXML.merge`` also preserves
    the conveyor's actuator section.

    Args:
        pos: Translation applied to the floor-aligned pose authored in the XML.
        quat: Optional root-body quaternion in MuJoCo ``w x y z`` order.
    """

    BODY_NAME = "conveyor_belt_single"
    ACTUATOR_PREFIX = "belt_drive_"
    JOINT_PREFIX = "belt_slide_"

    def __init__(self, pos=(0.0, 0.0, 0.0), quat=None, scale=(1.0, 1.0, 1.0)):
        xml_path = os.path.join(
            assets_root,
            "objects",
            "manual",
            "conveyor_belt",
            "conveyor_belt_single.xml",
        )
        super().__init__(xml_path)

        scale = np.asarray(scale, dtype=float)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError("scale must contain three positive finite values")
        if not np.allclose(scale, 1.0):
            scale_mjcf_model(
                obj=self.worldbody,
                asset_root=self.asset,
                scale=scale,
                get_elements_func=get_elements,
            )
            self._fit_end_rollers_inside_frame(scale)

        body = self.worldbody.find(f"./body[@name='{self.BODY_NAME}']")
        if body is None:
            raise ValueError(f"{xml_path}: missing root body {self.BODY_NAME!r}")

        self._add_robosuite_visual_geoms(body)

        authored_pos = string_to_array(body.get("pos", "0 0 0"))
        body.set("pos", array_to_string(authored_pos + np.asarray(pos, dtype=float)))
        if quat is not None:
            quat = np.asarray(quat, dtype=float)
            if quat.shape != (4,):
                raise ValueError("quat must contain four values in w x y z order")
            body.set("quat", array_to_string(quat))

    def _fit_end_rollers_inside_frame(self, scale):
        """Keep scaled end rollers inside the scaled frame footprint."""
        frame = self.worldbody.find(
            f"./body[@name='{self.BODY_NAME}']/geom[@name='conveyor_frame']"
        )
        if frame is None:
            return
        frame_half_length = string_to_array(frame.get("size"))[0]
        for name, sign in (("belt_end_left", -1.0), ("belt_end_right", 1.0)):
            roller = self.worldbody.find(
                f"./body[@name='{self.BODY_NAME}']/geom[@name='{name}']"
            )
            if roller is None:
                continue
            roller_pos = string_to_array(roller.get("pos"))
            roller_size = string_to_array(roller.get("size"))
            # The cylinders are rotated 90 degrees around X, so their local
            # axial half-length lies along model Y. The generic scaler treats
            # the second cylinder size as model Z and can make the rollers
            # wider than the frame under anisotropic scaling.
            roller_size[1] *= scale[1] / scale[2]
            roller.set("size", array_to_string(roller_size))
            roller_radius = roller_size[0]
            roller_pos[0] = sign * (frame_half_length - roller_radius)
            roller.set("pos", array_to_string(roller_pos))

    @staticmethod
    def _add_robosuite_visual_geoms(root_body):
        """Split fixture geometry into robosuite collision and visual groups."""
        for parent in root_body.iter("body"):
            visual_geoms = []
            for geom in parent.findall("./geom"):
                noncolliding = (
                    geom.get("contype") == "0" and geom.get("conaffinity") == "0"
                )
                if noncolliding:
                    geom.set("group", "1")
                    continue

                geom.set("group", "0")
                visual = deepcopy(geom)
                visual.set("name", f"{geom.get('name')}_visual")
                visual.set("group", "1")
                visual.set("contype", "0")
                visual.set("conaffinity", "0")
                visual.set("density", "0")
                visual.attrib.pop("mass", None)
                visual_geoms.append(visual)
            parent.extend(visual_geoms)

    @property
    def actuator_names(self):
        return tuple(
            actuator.get("name")
            for actuator in self.actuator
            if actuator.get("name", "").startswith(self.ACTUATOR_PREFIX)
        )

    @property
    def joint_names(self):
        return tuple(
            joint.get("name")
            for joint in self.worldbody.iter("joint")
            if joint.get("name", "").startswith(self.JOINT_PREFIX)
        )
