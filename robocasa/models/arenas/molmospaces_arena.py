"""MolmoSpacesArena -- use a MolmoSpaces (iTHOR/ProcTHOR) scene MJCF as the
background arena for a RoboCasa environment.

MolmoSpaces scenes are full, self-contained MuJoCo MJCF rooms (walls, floor,
built-in furniture, articulated appliances) that reference meshes/textures via
relative paths. They use a NESTED ``<default>`` class hierarchy
(``/`` -> ``__VISUAL_MJT__`` / ``__STRUCTURAL_MJT__`` / ...). robosuite's
``MujocoXML`` default-class parser is single-level and crashes on this nesting
(``KeyError: '__VISUAL_MJT__'``), so we pre-process the scene XML first:

  1. Inline every default class onto its elements (proper MuJoCo inheritance,
     no ``childclass`` is used by these scenes), then drop ``<default>`` and all
     ``class=`` attributes -> robosuite sees a defaults-free model.
  2. Drop the legacy ``<size njmax/nconmax>`` cap so MuJoCo auto-sizes once the
     robot + task objects are merged in.
  3. Guarantee a worldbody geom named ``floor`` (robosuite's ``Arena`` looks it
     up); MolmoSpaces scenes already provide one, but we assert it.

The cleaned XML is written next to the source (so the relative mesh paths still
resolve) and handed to ``Arena.__init__``, which absolutizes the asset paths.
"""

import os
import tempfile
import xml.etree.ElementTree as ET

import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import new_element


def _build_class_tables(default_root):
    """Return (attrs_by_class, parent_of) for a nested ``<default>`` tree.

    attrs_by_class[class_name][tag] -> {attr: value} declared *directly* on that
    class; parent_of[class_name] -> enclosing class name (or None).
    """
    attrs_by_class = {}
    parent_of = {}

    def walk(el, parent_name):
        for d in el.findall("default"):
            name = d.get("class")
            attrs_by_class[name] = {
                child.tag: dict(child.attrib) for child in d if child.tag != "default"
            }
            parent_of[name] = parent_name
            walk(d, name)

    walk(default_root, None)
    return attrs_by_class, parent_of


def _effective_attrs(cls, tag, attrs_by_class, parent_of):
    """Resolve inherited default attrs for (class, element-tag): root-most
    ancestor first so the leaf class overrides its parents."""
    chain = []
    n = cls
    while n is not None:
        chain.append(n)
        n = parent_of.get(n)
    resolved = {}
    for n in reversed(chain):
        resolved.update(attrs_by_class.get(n, {}).get(tag, {}))
    return resolved


def inline_defaults(root):
    """Bake the nested ``<default>`` classes onto elements and strip them.

    Mutates ``root`` in place. Safe for scenes that use direct ``class="..."``
    references and no ``childclass`` (true for iTHOR/ProcTHOR exports)."""
    default_root = root.find("default")
    if default_root is None:
        return
    attrs_by_class, parent_of = _build_class_tables(default_root)

    def apply(el):
        cls = el.attrib.pop("class", None)
        if cls is not None:
            for k, v in _effective_attrs(
                cls, el.tag, attrs_by_class, parent_of
            ).items():
                el.attrib.setdefault(k, v)  # explicit attrs win over defaults
        for child in list(el):
            apply(child)

    for child in list(root):
        if child.tag != "default":
            apply(child)
    root.remove(default_root)


def remap_geom_groups(root):
    """Translate MolmoSpaces geom-group convention to robosuite's.

    MolmoSpaces uses group 0 = VISUAL, groups 3/4 = collision/structural.
    Some visual geoms inherit no explicit group and rely on ``contype=0`` /
    ``conaffinity=0`` instead. robosuite uses group 0 as collision
    (``recolor_collision_geoms`` strips ``material``), group 1 as visual. We map
    visual geoms -> 1 and collision/structural geoms -> 0.
    """
    for geom in root.iter("geom"):
        g = geom.get("group")
        is_noncolliding_visual = (
            geom.get("contype") == "0" and geom.get("conaffinity") == "0"
        )
        if g == "0" or (g is None and is_noncolliding_visual):
            geom.set("group", "1")
        elif g in ("3", "4"):
            geom.set("group", "0")


def remap_objaverse_release_paths(root, src_xml, release="20260131"):
    """Point ProcTHOR Objaverse assets at the dated MolmoSpaces release dir.

    ProcTHOR scene exports reference ``../../objects/objaverse/<uid>/...`` while
    the downloaded resources live under
    ``../../objects/objaverse/<release>/<uid>/...``. Rewrite only paths where
    the original file is absent and the dated file exists.
    """
    prefix = "../../objects/objaverse/"
    release_prefix = prefix + release + "/"
    scene_dir = os.path.dirname(src_xml)

    for el in root.iter():
        asset_file = el.get("file")
        if asset_file is None:
            continue
        if not asset_file.startswith(prefix) or asset_file.startswith(release_prefix):
            continue

        original_abs = os.path.normpath(os.path.join(scene_dir, asset_file))
        if os.path.exists(original_abs):
            continue

        dated_file = release_prefix + asset_file[len(prefix) :]
        dated_abs = os.path.normpath(os.path.join(scene_dir, dated_file))
        if os.path.exists(dated_abs):
            el.set("file", dated_file)


def preprocess_scene_xml(src_xml, dst_xml=None):
    """Produce a robosuite-ingestible copy of a MolmoSpaces scene MJCF.

    Args:
        src_xml: path to the MolmoSpaces scene .xml.
        dst_xml: output path. Defaults to ``<src>.robocasa.xml`` in the same
            directory (must stay in the same dir so relative mesh paths resolve).
    Returns:
        The output path.
    """
    src_xml = os.path.abspath(src_xml)
    if dst_xml is None:
        base = os.path.basename(src_xml).rsplit(".", 1)[0]
        dst_xml = os.path.join(os.path.dirname(src_xml), base + ".robocasa.xml")

    tree = ET.parse(src_xml)
    root = tree.getroot()

    inline_defaults(root)
    remap_geom_groups(root)
    remap_objaverse_release_paths(root, src_xml)

    from robocasa.models.scenes.ithor_scene_defaults import (
        apply_default_scene_furniture,
    )

    apply_default_scene_furniture(root, src_xml)

    # Drop legacy size cap -> let MuJoCo auto-size after robot/objects merge in.
    size = root.find("size")
    if size is not None:
        for k in ("njmax", "nconmax", "nstack"):
            size.attrib.pop(k, None)
        if not size.attrib:
            root.remove(size)

    # robosuite's Arena requires a worldbody geom named "floor".
    wb = root.find("worldbody")
    floor = wb.find(".//geom[@name='floor']") if wb is not None else None
    if floor is None:
        raise ValueError(f"{src_xml}: no geom named 'floor' found for Arena")

    tree.write(dst_xml)
    return dst_xml


class MolmoSpacesArena(Arena):
    """A RoboCasa arena backed by a MolmoSpaces scene MJCF.

    Args:
        scene_xml (str): path to a MolmoSpaces scene .xml (load it from the
            symlink resources dir so its relative ``../../objects`` mesh paths
            resolve).
    """

    def __init__(self, scene_xml):
        self.scene_xml = os.path.abspath(scene_xml)
        cleaned = preprocess_scene_xml(self.scene_xml)
        super().__init__(cleaned)
        self._add_default_cameras()

    def _add_default_cameras(self):
        """Add robosuite's standard fixed cameras (frontview / birdview /
        agentview / sideview) looking at the room centre, since MolmoSpaces
        scenes don't define them and robosuite/render expects them.

        Cameras are positioned relative to the scene's xy centre and overall
        extent (estimated from worldbody body positions), so they frame rooms of
        different sizes reasonably."""
        existing = {c.get("name") for c in self.worldbody.findall("./camera")}

        # room centre/size from WORLD geom positions (compile the model once;
        # body @pos attrs are local to their parents, so can't be used directly)
        import mujoco

        model = mujoco.MjModel.from_xml_string(
            ET.tostring(self.root, encoding="unicode")
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        xyz = data.geom_xpos
        lo, hi = xyz.min(0), xyz.max(0)
        cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
        ex, ey = (hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2
        ceil_z = float(hi[2])
        target = np.array([cx, cy, 1.0])

        def look_at(eye):
            """xyaxes for a camera at eye looking at target (avoids quats)."""
            fwd = target - eye
            fwd = fwd / (np.linalg.norm(fwd) + 1e-9)
            # use +y as reference up when looking near-vertically (e.g. birdview),
            # else world +z, so right = fwd x up is never degenerate
            world_up = (
                np.array([0, 1.0, 0]) if abs(fwd[2]) > 0.95 else np.array([0, 0, 1.0])
            )
            right = np.cross(fwd, world_up)
            right = right / (np.linalg.norm(right) + 1e-9)
            up = np.cross(right, fwd)
            return " ".join(f"{v:.5f}" for v in [*right, *up])

        # interior cameras (the room is enclosed by walls, so a camera outside
        # the bbox would only see the exterior shell): tuck them just inside the
        # walls at eye height, looking across at the room centre.
        cams = {
            "frontview": np.array([cx + 0.85 * ex, cy, 1.6]),
            "sideview": np.array([cx, cy + 0.85 * ey, 1.6]),
            "agentview": np.array([cx + 0.6 * ex, cy + 0.6 * ey, 1.5]),
            "birdview": np.array([cx, cy, max(1.0, 0.9 * ceil_z)]),
        }
        for name, eye in cams.items():
            if name in existing:
                continue
            self.worldbody.append(
                new_element(
                    tag="camera",
                    name=name,
                    mode="fixed",
                    pos=" ".join(f"{v:.5f}" for v in eye),
                    xyaxes=look_at(eye),
                )
            )

        # MolmoSpaces rooms ship a single ceiling light, and robosuite's world
        # base (merged over the arena) drops the scene's bright headlight -> the
        # env renders dark in the corners. Add a grid of overhead lights so the
        # whole room is evenly lit for rendering / camera observations.
        lz = max(1.0, ceil_z - 0.1)
        for sx in (-0.5, 0.5):
            for sy in (-0.5, 0.5):
                self.worldbody.append(
                    new_element(
                        tag="light",
                        name=None,
                        pos=f"{cx + sx * ex:.4f} {cy + sy * ey:.4f} {lz:.4f}",
                        dir="0 0 -1",
                        directional="false",
                        diffuse="0.8 0.8 0.8",
                        specular="0.15 0.15 0.15",
                        attenuation="0.5 0 0",
                        castshadow="false",
                    )
                )
