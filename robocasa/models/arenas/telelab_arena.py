"""Arena adapter for hand-authored TeleLab real-to-sim MJCF scenes."""

import os
import tempfile
import xml.etree.ElementTree as ET
from copy import deepcopy

from robosuite.models.arenas import Arena

from robocasa.models.arenas.molmospaces_arena import inline_defaults


class TeleLabArena(Arena):
    """Load an ordinary TeleLab MJCF while preserving its visual materials.

    Unlike MolmoSpaces exports, these scenes need no default-class expansion or
    asset path rewriting. robosuite only requires a direct worldbody geom named
    ``floor`` and its visual-group convention.
    """

    def __init__(self, scene_xml):
        self.scene_xml = os.path.abspath(os.fspath(scene_xml))
        if not os.path.isfile(self.scene_xml):
            raise FileNotFoundError(f"TeleLab scene XML not found: {self.scene_xml}")

        cleaned = self._prepare_scene_xml(self.scene_xml)
        try:
            super().__init__(cleaned)
        finally:
            os.unlink(cleaned)

    @staticmethod
    def _prepare_scene_xml(scene_xml):
        tree = ET.parse(scene_xml)
        root = tree.getroot()
        if root.findall("./asset/model") or root.findall(".//attach"):
            root = TeleLabArena._flatten_attached_models(scene_xml, root)
            # Attached fixtures bring nested named defaults, which robosuite's
            # single-level defaults parser cannot consume.
            inline_defaults(root)
            tree = ET.ElementTree(root)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"{scene_xml}: missing <worldbody>")

        floor = worldbody.find("./geom[@name='floor']")
        if floor is None:
            floor = next(
                (
                    geom
                    for geom in worldbody.findall("./geom")
                    if geom.get("type") == "plane"
                ),
                None,
            )
        if floor is None:
            raise ValueError(
                f"{scene_xml}: expected a direct worldbody plane geom for the floor"
            )
        floor.set("name", "floor")
        floor.set("group", "0")

        # ``floor`` must remain group 0 for robosuite's floor reference and
        # collision handling, but collision meshes are hidden by default. Keep
        # a material-bearing visual copy so the authored floor texture renders.
        floor_visual = deepcopy(floor)
        floor_visual.set("name", "floor_visual")
        floor_visual.set("group", "1")
        floor_visual.set("contype", "0")
        floor_visual.set("conaffinity", "0")
        floor_visual.set("density", "0")
        floor_visual.attrib.pop("mass", None)
        floor_index = list(worldbody).index(floor)
        worldbody.insert(floor_index + 1, floor_visual)

        if worldbody.find("./camera[@name='frontview']") is None:
            ET.SubElement(
                worldbody,
                "camera",
                name="frontview",
                mode="fixed",
                pos="0 -2.7 1.6",
                xyaxes="1 0 0 0 0.45 0.89",
            )

        # robosuite only computes inertia from group 0 and recolors that group.
        # Split ordinary authored geoms into a collision original plus a
        # material-preserving, non-colliding visual copy. Already-authored group
        # values are respected, allowing future XMLs to provide their own split.
        existing_names = {
            geom.get("name")
            for geom in worldbody.iter("geom")
            if geom.get("name") is not None
        }
        unnamed_visual_index = 1
        for parent in worldbody.iter():
            additions = []
            for geom in list(parent):
                if geom.tag != "geom" or geom is floor or geom.get("group") is not None:
                    continue
                # MjSpec may omit an attached model's explicit group="0"
                # because zero is MuJoCo's default, while retaining the
                # fixture collision class's diagnostic color (legacy dark red
                # or transparent). Treat those geoms as authored collision
                # geometry; copying them to group 1 would expose collision
                # boxes in the normal visual pass.
                rgba = geom.get("rgba", "").split()
                geom_name = geom.get("name", "")
                is_fixture_collision = rgba in (
                    ["0.5", "0", "0", "0.5"],
                    ["0", "0", "0", "0"],
                ) or geom_name.endswith("_collision")
                if is_fixture_collision:
                    geom.set("group", "0")
                    continue
                noncolliding = (
                    geom.get("contype") == "0" and geom.get("conaffinity") == "0"
                )
                if noncolliding:
                    geom.set("group", "1")
                    continue

                geom.set("group", "0")
                visual = deepcopy(geom)
                if geom.get("name") is None:
                    base_name = f"telelab_geom_visual_{unnamed_visual_index}"
                    unnamed_visual_index += 1
                else:
                    base_name = f"{geom.get('name')}_visual"
                visual_name = base_name
                index = 2
                while visual_name in existing_names:
                    visual_name = f"{base_name}_{index}"
                    index += 1
                existing_names.add(visual_name)
                visual.set("name", visual_name)
                visual.set("group", "1")
                visual.set("contype", "0")
                visual.set("conaffinity", "0")
                visual.set("density", "0")
                visual.attrib.pop("mass", None)
                additions.append(visual)
            parent.extend(additions)

        # Keeping the temporary file beside the source preserves relative asset
        # and include paths while MujocoXML loads and absolutizes them.
        handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=".telelab_",
            suffix=".xml",
            dir=os.path.dirname(scene_xml),
            delete=False,
        )
        try:
            tree.write(handle, encoding="utf-8", xml_declaration=True)
        finally:
            handle.close()
        return handle.name

    @staticmethod
    def _flatten_attached_models(scene_xml, source_root):
        """Resolve MuJoCo sub-model attachments before robosuite merges G1.

        MjSpec expands ``asset/model + attach`` but writes child asset paths
        relative to the child model. Once flattened into the parent, make those
        paths absolute so robosuite's compiler ``meshdir`` cannot reinterpret
        them.
        """
        import mujoco

        scene_dir = os.path.dirname(scene_xml)
        model_dirs = {}
        for model in source_root.findall("./asset/model"):
            name = model.get("name")
            path = model.get("file")
            if name and path:
                model_dirs[name] = os.path.dirname(
                    os.path.abspath(os.path.join(scene_dir, path))
                )
        prefix_dirs = []
        for attach in source_root.findall(".//attach"):
            model_dir = model_dirs.get(attach.get("model"))
            prefix = attach.get("prefix")
            if model_dir is not None and prefix:
                prefix_dirs.append((prefix, model_dir))

        spec = mujoco.MjSpec.from_file(scene_xml)
        for collection_name in ("meshes", "textures"):
            for asset in getattr(spec, collection_name):
                path = asset.file
                if not path or os.path.isabs(path):
                    continue
                asset_dir = next(
                    (
                        model_dir
                        for prefix, model_dir in prefix_dirs
                        if asset.name.startswith(prefix)
                    ),
                    scene_dir,
                )
                asset.file = os.path.abspath(os.path.join(asset_dir, path))
        return ET.fromstring(spec.to_xml())
