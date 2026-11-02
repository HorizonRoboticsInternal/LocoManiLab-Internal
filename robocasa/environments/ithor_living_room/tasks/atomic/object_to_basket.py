"""Shared fixed-object MJCF adapter for living-room basket tasks."""

import os
import tempfile
import xml.etree.ElementTree as ET


def _vec(values):
    return " ".join(str(float(value)) for value in values)


class FixedObjectToBasketMixin:
    """Load one THOR object while reusing RemoteToBasket scene poses."""

    def _get_object_adapter(self):
        spec = self._get_task_spec()
        required = (
            "source_category",
            "source_wrapper_pos",
            "source_wrapper_quat",
            "source_bbox_pos",
            "source_bbox_size",
            "source_scale",
        )
        missing = [key for key in required if key not in spec]
        if missing:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing object adapter "
                f"parameters {missing}"
            )
        return spec

    def _get_obj_cfgs(self):
        adapter = self._get_object_adapter()
        return [
            dict(name=self.BASKET_NAME, obj_groups=self._asset_path(self.BASKET_XML)),
            dict(
                name=self.REMOTE_NAME,
                obj_groups=self._asset_path(self.REMOTE_XML),
                object_scale=float(adapter["source_scale"]),
            ),
        ]

    def _materialize_remote_xml(self, source_path):
        adapter = self._get_object_adapter()
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
            elif class_name == "region":
                element.set("contype", "0")
                element.set("conaffinity", "0")
                element.set("group", "1")
                element.set("rgba", "0 1 0 0")
                element.set("mass", "0")
                if element.get("name") == "reg_bbox":
                    element.set("name", "source_bbox_marker")
        for asset in root.findall("./asset/*"):
            file_path = asset.get("file")
            if file_path:
                asset.set("file", os.path.abspath(os.path.join(source_dir, file_path)))

        worldbody = root.find("worldbody")
        object_body = root.find("./worldbody/body")
        if object_body is None:
            raise ValueError(f"{source_path}: missing object root body")
        worldbody.remove(object_body)
        outer_body = ET.SubElement(worldbody, "body")
        outer_body.append(object_body)
        object_body.set("name", "object")
        # Objects routed through this adapter are intentionally rigid. THOR assets
        # such as Book_3 also contain nested cover hinges; leaving those joints in
        # place makes the hinge appear before the generated free joint in
        # ``obj.joints`` and breaks reset-time object jitter.
        for body in object_body.iter("body"):
            for joint in list(body.findall("joint")):
                body.remove(joint)

        model_bodies = list(object_body.findall("body"))
        wrapper = ET.Element(
            "body",
            pos=_vec(adapter["source_wrapper_pos"]),
            quat=_vec(adapter["source_wrapper_quat"]),
        )
        for body in model_bodies:
            object_body.remove(body)
            wrapper.append(body)
        object_body.append(wrapper)
        object_body.append(
            ET.Element(
                "geom",
                name="reg_bbox",
                type="box",
                pos=_vec(adapter["source_bbox_pos"]),
                size=_vec(adapter["source_bbox_size"]),
                group="1",
                contype="0",
                conaffinity="0",
                rgba="0 1 0 0",
            )
        )

        handle = tempfile.NamedTemporaryFile(
            mode="wb", suffix=".xml", prefix="robocasa_basket_object_", delete=False
        )
        with handle:
            tree.write(handle, encoding="utf-8", xml_declaration=True)
        self._remote_temp_xmls.append(handle.name)
        return handle.name

    def sample_object(self, groups, **kwargs):
        if groups == self._asset_path(self.REMOTE_XML):
            adapter = self._get_object_adapter()
            scale = kwargs.get("object_scale") or 1.0
            temp_xml = self._materialize_remote_xml(groups)
            category = str(adapter["source_category"])
            return (
                dict(mjcf_path=temp_xml, scale=scale),
                dict(
                    mjcf_path=groups,
                    cat=category,
                    groups=[category],
                    groups_containing_sampled_obj=[category],
                    split=None,
                ),
            )
        return super().sample_object(groups, **kwargs)


__all__ = ["FixedObjectToBasketMixin"]
