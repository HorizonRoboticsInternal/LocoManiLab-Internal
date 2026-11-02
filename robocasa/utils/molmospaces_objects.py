"""Wrap a MolmoSpaces objaverse object as a RoboCasa ``MJCFObject``.

MolmoSpaces objaverse objects ship as a self-contained ``<uid>.xml`` (visual
mesh + texture + convex collider meshes), but in MolmoSpaces' own MJCF
convention (nested ``__VISUAL_MJT__`` default classes, group 0 = visual). RoboCasa's
``MJCFObject`` expects a specific layout instead: a ``<body><body name="object">``
nest, a ``class="visual"`` geom, ``class="collision"`` geoms, and a ``reg_bbox``
region box describing the bounding box.

``make_objaverse_object`` reads the objaverse meshes, computes the object's
bounding box (by compiling the raw object once), and emits a fresh
RoboCasa-format ``model.xml`` that ``MJCFObject`` can ingest directly.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np
import mujoco

from robocasa.models.objects.objects import MJCFObject


def get_objaverse_dir():
    import robocasa

    return os.path.join(robocasa.models.assets_root, "objects", "objaverse", "20260131")


def list_objaverse_uids():
    d = get_objaverse_dir()
    if not os.path.isdir(d):
        return []
    return sorted(
        n for n in os.listdir(d) if os.path.isfile(os.path.join(d, n, f"{n}.xml"))
    )


def _bbox_from_visual(obj_dir, uid):
    """Compile the visual mesh alone to get its (center, half_size) AABB."""
    vis = os.path.join(obj_dir, f"{uid}_visual.obj")
    xml = f"""
    <mujoco>
      <asset><mesh name="m" file="{vis}" scale="-1 1 1"/></asset>
      <worldbody><body name="object">
        <geom type="mesh" mesh="m"/>
      </body></worldbody>
    </mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    # mesh vertex bounds in object frame
    v0 = m.mesh_vertadr[0]
    nv = m.mesh_vertnum[0]
    verts = m.mesh_vert[v0 : v0 + nv]
    lo, hi = verts.min(0), verts.max(0)
    center = (lo + hi) / 2
    half = (hi - lo) / 2
    return center, half


def make_objaverse_object(uid, name, scale=1.0, fit_size=None):
    """Return an ``MJCFObject`` for the MolmoSpaces objaverse object ``uid``.

    Args:
        scale (float): explicit uniform scale.
        fit_size (float or None): if set, auto-scale the object so its largest
            dimension equals ``fit_size`` metres (e.g. 0.15 for a tabletop prop).
            objaverse objects span furniture-scale to tiny, so a task placing
            them on a counter should pass ``fit_size`` to keep them graspable.
    """
    obj_dir = os.path.join(get_objaverse_dir(), uid)
    src_xml = os.path.join(obj_dir, f"{uid}.xml")
    if not os.path.isfile(src_xml):
        raise FileNotFoundError(f"objaverse object not found: {src_xml}")

    center, half = _bbox_from_visual(obj_dir, uid)

    if fit_size is not None:
        max_dim = float(2 * half.max())
        if max_dim > 1e-6:
            scale = fit_size / max_dim

    # absolute mesh/texture paths so the emitted xml is location-independent
    vis_obj = os.path.join(obj_dir, f"{uid}_visual.obj")
    tex_png = os.path.join(obj_dir, f"{uid}_visual_0.png")
    colliders = sorted(
        f
        for f in os.listdir(obj_dir)
        if f.startswith(f"{uid}_collider") and f.endswith(".obj")
    )

    mj = ET.Element("mujoco", model=f"objaverse_{uid}")
    ET.SubElement(mj, "compiler", angle="radian")

    asset = ET.SubElement(mj, "asset")
    ET.SubElement(asset, "mesh", name=f"{uid}_vis", file=vis_obj, scale="-1 1 1")
    if os.path.isfile(tex_png):
        ET.SubElement(asset, "texture", name=f"{uid}_tex", type="2d", file=tex_png)
        ET.SubElement(asset, "material", name=f"{uid}_mat", texture=f"{uid}_tex")
    for i, c in enumerate(colliders):
        ET.SubElement(
            asset,
            "mesh",
            name=f"{uid}_col{i}",
            file=os.path.join(obj_dir, c),
            scale="-1 1 1",
        )

    default = ET.SubElement(mj, "default")
    dv = ET.SubElement(default, "default", **{"class": "visual"})
    ET.SubElement(dv, "geom", conaffinity="0", contype="0", group="1", type="mesh")
    dc = ET.SubElement(default, "default", **{"class": "collision"})
    ET.SubElement(dc, "geom", group="0", type="mesh")
    dr = ET.SubElement(default, "default", **{"class": "region"})
    ET.SubElement(
        dr,
        "geom",
        group="1",
        conaffinity="0",
        contype="0",
        rgba="0 1 0 0",
        mass="0",
        type="box",
    )

    wb = ET.SubElement(mj, "worldbody")
    outer = ET.SubElement(wb, "body")
    obj_body = ET.SubElement(outer, "body", name="object")
    vis_kwargs = dict(
        **{"class": "visual"},
        mesh=f"{uid}_vis",
        type="mesh",
        density="100",
        friction="0.95 0.3 0.1",
    )
    if os.path.isfile(tex_png):
        vis_kwargs["material"] = f"{uid}_mat"
    ET.SubElement(obj_body, "geom", **vis_kwargs)
    # bounding-box region geom (RoboCasa placement uses this)
    ET.SubElement(
        obj_body,
        "geom",
        **{"class": "region"},
        name="reg_bbox",
        type="box",
        pos=" ".join(f"{v:.6f}" for v in center),
        size=" ".join(f"{max(v, 1e-4):.6f}" for v in half),
    )
    for i in range(len(colliders)):
        ET.SubElement(
            obj_body,
            "geom",
            **{"class": "collision"},
            mesh=f"{uid}_col{i}",
            type="mesh",
        )

    out_xml = os.path.join(obj_dir, f"{uid}.robocasa.xml")
    ET.ElementTree(mj).write(out_xml)

    return MJCFObject(name=name, mjcf_path=out_xml, scale=scale)
