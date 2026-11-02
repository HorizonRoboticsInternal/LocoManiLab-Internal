"""Convert an Illusion-style USD articulated asset into a RoboCasa MJCF fixture.

The source asset is a binary USDC stage (Y-up, centimetres) describing a small
number of rigid groups connected by PhysX revolute joints, plus a sidecar
``joint_info.yaml`` that names the parent/child group of each joint.

This script:
  * exports every USD Mesh as an OBJ (positions + face-varying UVs + normals),
  * resolves each mesh's bound material to a single MuJoCo diffuse texture or
    solid colour,
  * converts geometry from USD (Y-up, cm) to MuJoCo (Z-up, m),
  * builds a body hierarchy (base -> hinged child bodies) with hinge joints,
    convex-mesh collision geoms, and reg_main/reg_int region boxes.

Usage:
    python usd_to_mjcf_fixture.py <asset_dir> <output_dir> [--name NAME]
"""

import argparse
import math
import os
import shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom

import numpy as np
import yaml
from pxr import Usd, UsdGeom, UsdShade, Gf


# ---------------------------------------------------------------------------
# coordinate conversion: USD (Y-up, cm) -> MuJoCo (Z-up, m)
# ---------------------------------------------------------------------------
# Optional extra rotation applied AFTER the Y-up->Z-up conversion, so a fixture
# can be born in its desired upright/facing orientation (baked into the mesh,
# pivot and axis) instead of being rotated per-scene with set_euler. Set via
# --extra-rot-x/y/z (degrees) on the CLI; identity by default.
_EXTRA_ROT = np.eye(3)
_USD_TO_MJ = np.eye(3)


def _rot_x(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def _rot_y(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def _rot_z(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def set_extra_rotation(rx_deg, ry_deg, rz_deg):
    """Build the post-conversion rotation matrix (Rz @ Ry @ Rx)."""
    global _EXTRA_ROT
    _EXTRA_ROT = _rot_z(rz_deg) @ _rot_y(ry_deg) @ _rot_x(rx_deg)


def set_stage_up_axis(up_axis):
    """Configure the USD-to-MuJoCo axis conversion for this USD stage.

    MuJoCo is Z-up.  USD assets can be Y-up or Z-up; treating every USD as
    Y-up lays a Z-up prop on its side.  Unit conversion remains separate.
    """
    global _USD_TO_MJ
    if str(up_axis).upper() == "Z":
        _USD_TO_MJ = np.eye(3)
    elif str(up_axis).upper() == "Y":
        _USD_TO_MJ = _rot_x(90.0)
    else:
        raise ValueError(f"Unsupported USD up axis: {up_axis!r}")


def usd_point_to_mj(p, meters_per_unit):
    """Vec3 in USD world (units = 1/meters_per_unit) -> MuJoCo metres, Z-up."""
    x, y, z = p[0], p[1], p[2]
    s = meters_per_unit
    v = _USD_TO_MJ @ np.array([x * s, y * s, z * s], dtype=np.float64)
    return _EXTRA_ROT @ v


def usd_dir_to_mj(d):
    """Direction vector (no scaling) USD -> MuJoCo."""
    x, y, z = d[0], d[1], d[2]
    v = _USD_TO_MJ @ np.array([x, y, z], dtype=np.float64)
    v = _EXTRA_ROT @ v
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------------------
# material resolution
# ---------------------------------------------------------------------------
def resolve_material(mat_prim):
    """Return dict(name, texture_basename or None, rgba).

    Handles three material conventions found in Illusion / Omniverse USDs:

    1. Illusion MDL (``BaseColor_Color``, ``IsBaseColorTex``, ``BaseColor_Tex``)
    2. Omniverse MDL (``diffuse_color_constant``)
    3. UsdPreviewSurface (``UsdUVTexture`` child shader + ``opacity``)
    """
    name = mat_prim.GetName()

    # Collect all shader children (a material may have several — the first
    # non-utility shader wins for colour, others provide texture info).
    shaders = []
    for c in mat_prim.GetChildren():
        if c.GetTypeName() == "Shader":
            shaders.append(UsdShade.Shader(c))

    if not shaders:
        return {"name": name, "texture": None, "rgba": [0.8, 0.8, 0.8, 1.0]}

    def _gi(shader, attr):
        inp = shader.GetInput(attr)
        return inp.Get() if inp else None

    rgba = [0.8, 0.8, 0.8, 1.0]
    texture = None

    for sh in shaders:
        sid = sh.GetShaderId() or ""

        # -- solid colour -------------------------------------------------
        # Illusion MDL
        bc = _gi(sh, "BaseColor_Color")
        if bc is not None:
            rgba = [float(bc[0]), float(bc[1]), float(bc[2]), 1.0]
        # Omniverse MDL
        dc = _gi(sh, "diffuse_color_constant")
        if dc is not None:
            rgba = [float(dc[0]), float(dc[1]), float(dc[2]), 1.0]

        # -- opacity ------------------------------------------------------
        op = _gi(sh, "Opacity") or _gi(sh, "opacity")
        if op is not None:
            rgba[3] = float(op)

        # -- texture ------------------------------------------------------
        # Illusion-style: BaseColor_Tex attribute on the colour shader.
        btx = _gi(sh, "BaseColor_Tex")
        if btx and (_gi(sh, "IsBaseColorTex") or True):
            tex_val = float(_gi(sh, "IsBaseColorTex") or 1.0)
            if tex_val >= 0.5:
                texture = os.path.basename(str(btx).strip("@"))

        # UsdPreviewSurface / Omniverse: look for a UsdUVTexture child shader.
        if not texture and "UsdUVTexture" in str(sid):
            f = _gi(sh, "file")
            if f:
                # file is an Sdf.AssetPath; resolve to a local basename
                texture = os.path.basename(str(f).rstrip("@"))

        # Omniverse: texture reader shader (stReader / diffuseTexture)
        if not texture:
            f = _gi(sh, "file")
            if f and any(
                k in sid.lower() for k in ("streader", "diffusetexture", "uvtexture")
            ):
                texture = os.path.basename(str(f).rstrip("@"))

        # Also check other shaders in the material for texture file.
        if not texture:
            f = _gi(sh, "file")
            if f:
                texture = os.path.basename(str(f).rstrip("@"))

    if texture:
        rgba = [1.0, 1.0, 1.0, rgba[3]]

    return {"name": name, "texture": texture, "rgba": rgba}


# ---------------------------------------------------------------------------
# mesh export
# ---------------------------------------------------------------------------
def export_mesh_obj(mesh_prim, xform_cache, meters_per_unit, out_path):
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = mesh.GetPointsAttr().Get()
    fvc = mesh.GetFaceVertexCountsAttr().Get()
    fvi = mesh.GetFaceVertexIndicesAttr().Get()
    if not pts or not fvc or not fvi:
        return False

    m = xform_cache.GetLocalToWorldTransform(mesh_prim)
    rot = m.ExtractRotationMatrix()

    # transform + convert positions
    verts = []
    for p in pts:
        wp = m.Transform(Gf.Vec3d(p[0], p[1], p[2]))
        verts.append(usd_point_to_mj(wp, meters_per_unit))

    # face-varying UVs / normals
    pv = UsdGeom.PrimvarsAPI(mesh_prim)
    st_pv = pv.GetPrimvar("st")
    st_vals, st_idx, st_interp = None, None, None
    if st_pv and st_pv.IsDefined():
        st_vals = st_pv.Get()
        st_idx = st_pv.GetIndices()
        st_interp = st_pv.GetInterpolation()

    normals = mesh.GetNormalsAttr().Get()

    def st_at(corner, vidx):
        if st_vals is None:
            return None
        if st_interp == "faceVarying":
            i = st_idx[corner] if st_idx else corner
        elif st_interp == "vertex" or st_interp == "varying":
            i = st_idx[vidx] if st_idx else vidx
        else:  # constant
            i = 0
        uv = st_vals[i]
        return (uv[0], uv[1])

    def n_at(corner, vidx):
        if not normals:
            return None
        if len(normals) == len(fvi):  # faceVarying
            nn = normals[corner]
        elif len(normals) == len(pts):  # per-vertex
            nn = normals[vidx]
        else:
            return None
        wn = rot * Gf.Vec3d(nn[0], nn[1], nn[2])
        return usd_dir_to_mj(wn)

    # build OBJ
    lines = ["# converted from USD"]
    for v in verts:
        lines.append(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}")

    vt_list, vn_list, faces = [], [], []
    corner = 0
    for count in fvc:
        face_corners = []
        for k in range(count):
            vidx = fvi[corner]
            uv = st_at(corner, vidx)
            nrm = n_at(corner, vidx)
            vt_i = vn_i = None
            if uv is not None:
                vt_list.append(uv)
                vt_i = len(vt_list)
            if nrm is not None:
                vn_list.append(nrm)
                vn_i = len(vn_list)
            face_corners.append((vidx + 1, vt_i, vn_i))
            corner += 1
        # fan-triangulate (counts are already 3 here, but stay safe)
        for t in range(1, count - 1):
            faces.append([face_corners[0], face_corners[t], face_corners[t + 1]])

    for uv in vt_list:
        lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
    for n in vn_list:
        lines.append(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}")
    for f in faces:
        parts = []
        for (vi, ti, ni) in f:
            if ti and ni:
                parts.append(f"{vi}/{ti}/{ni}")
            elif ti:
                parts.append(f"{vi}/{ti}")
            elif ni:
                parts.append(f"{vi}//{ni}")
            else:
                parts.append(f"{vi}")
        lines.append("f " + " ".join(parts))

    with open(out_path, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    allv = np.array(verts)
    return allv  # return verts for bbox computation


# ---------------------------------------------------------------------------
# main conversion
# ---------------------------------------------------------------------------
def convert(asset_dir, out_dir, name, static=False, open_top_receptacle=False):
    usd_path = os.path.join(asset_dir, "instance.usd")
    info_path = os.path.join(asset_dir, "joint_info.yaml")
    tex_dir = os.path.join(asset_dir, "Materials", "Textures")

    if os.path.exists(info_path):
        with open(info_path) as fh:
            jinfo = yaml.safe_load(fh) or {}
    elif static:
        # Many scanned prop USDs have no articulation sidecar.  They are still
        # useful as fixed fixtures / receptacles, so keep every mesh on one
        # static root body rather than requiring a fabricated joint map.
        jinfo = {"joints": []}
    else:
        raise FileNotFoundError(
            f"{info_path} is required for articulated conversion; use --static "
            "for an asset with no joint_info.yaml"
        )

    stage = Usd.Stage.Open(usd_path)
    if stage is None:
        raise RuntimeError(f"Could not open USD stage: {usd_path}")
    set_stage_up_axis(UsdGeom.GetStageUpAxis(stage))
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    xc = UsdGeom.XformCache()

    vis_dir = os.path.join(out_dir, "visuals")
    os.makedirs(vis_dir, exist_ok=True)

    # ---- map each group prim path -> list of mesh prims under it ----
    groups = {}
    mesh_to_group = {}
    for p in stage.Traverse():
        if p.GetTypeName() == "Mesh":
            grp = p.GetParent().GetPath().pathString
            groups.setdefault(grp, []).append(p)
            mesh_to_group[p.GetPath().pathString] = grp

    # ---- joints: build child-group -> joint info ----
    joints = []
    for j in jinfo.get("joints", []):
        joints.append(
            {
                "name": j["name"].split("/")[-1],
                "type": j["type"],
                "body0": j["body0"],  # parent group path
                "body1": j["body1"],  # child group path
                "axis": j["axis"],
                "range": j["motion_range"],
                "prim": j["name"],
            }
        )
    child_groups = {j["body1"] for j in joints}

    # base = group that is not a child of any joint
    base_groups = [g for g in groups if g not in child_groups]

    # ---- export meshes, gather materials, bbox per group ----
    materials = {}  # mat name -> resolved dict
    mesh_records = []  # list of dict(group, mesh_name, obj_file, mat_name, verts)

    for grp, mesh_prims in groups.items():
        for mp in mesh_prims:
            mesh_name = mp.GetName()
            obj_file = f"{mesh_name}.obj"
            verts = export_mesh_obj(mp, xc, mpu, os.path.join(vis_dir, obj_file))
            if verts is False:
                continue
            # material
            bound = UsdShade.MaterialBindingAPI(mp).ComputeBoundMaterial()[0]
            mat_name = None
            if bound:
                mres = resolve_material(bound.GetPrim())
                mat_name = mres["name"]
                materials[mat_name] = mres
            # Detect degenerate (planar) meshes: MuJoCo's qhull cannot build a
            # convex hull for them and the model fails to compile. We still
            # keep the visual geom; just skip emitting a mesh-collision geom.
            # (Collision is rewritten as primitive boxes in Step 2 anyway.)
            extents = verts.max(axis=0) - verts.min(axis=0)
            degenerate = bool((extents < 1e-5).any())
            mesh_records.append(
                {
                    "group": grp,
                    "mesh_name": mesh_name,
                    "obj": obj_file,
                    "mat": mat_name,
                    "verts": verts,
                    "degenerate": degenerate,
                }
            )

    # ---- copy textures actually referenced ----
    used_tex = {m["texture"] for m in materials.values() if m["texture"]}
    for t in used_tex:
        src = os.path.join(tex_dir, t)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(vis_dir, t))

    # ---- compute joint pivot/axis in MuJoCo frame ----
    for j in joints:
        jp = stage.GetPrimAtPath(j["prim"])
        from pxr import UsdPhysics

        rj = UsdPhysics.RevoluteJoint(jp)
        body0_prim = stage.GetPrimAtPath(j["body0"])
        m0 = xc.GetLocalToWorldTransform(body0_prim)
        lp0 = rj.GetLocalPos0Attr().Get()
        lr0 = rj.GetLocalRot0Attr().Get()  # Gf.Quatf
        # pivot in body0 local -> world
        pivot_world = m0.Transform(Gf.Vec3d(lp0[0], lp0[1], lp0[2]))
        j["pivot_mj"] = usd_point_to_mj(pivot_world, mpu)
        # axis: axis attr letter -> vec, rotate by localRot0 then body0 rotation
        axis_letter = rj.GetAxisAttr().Get()
        axis_vec = {
            "X": Gf.Vec3d(1, 0, 0),
            "Y": Gf.Vec3d(0, 1, 0),
            "Z": Gf.Vec3d(0, 0, 1),
        }[axis_letter]
        q = Gf.Quatd(lr0.GetReal(), Gf.Vec3d(*lr0.GetImaginary()))
        axis_jointframe = q.Transform(axis_vec)
        axis_world = m0.ExtractRotationMatrix() * axis_jointframe
        j["axis_mj"] = usd_dir_to_mj(axis_world)
        lo = math.radians(float(j["range"]["min"]))
        hi = math.radians(float(j["range"]["max"]))
        j["range_rad"] = (lo, hi)

    return build_mjcf(
        out_dir,
        name,
        groups,
        base_groups,
        joints,
        mesh_records,
        materials,
        static=static,
        open_top_receptacle=open_top_receptacle,
    )


def bbox_of(verts_list):
    allv = np.concatenate([v for v in verts_list], axis=0)
    return allv.min(axis=0), allv.max(axis=0)


def build_mjcf(
    out_dir,
    name,
    groups,
    base_groups,
    joints,
    mesh_records,
    materials,
    static=False,
    open_top_receptacle=False,
):
    mujoco = ET.Element("mujoco", model="model")
    ET.SubElement(mujoco, "option", timestep="0.002", gravity="0.0 0.0 -9.81")
    ET.SubElement(mujoco, "compiler", angle="radian")

    asset = ET.SubElement(mujoco, "asset")
    # meshes. Degenerate (planar) meshes need inertia="shell" so MuJoCo does
    # not try to compute volume-based inertia and crash with "mesh volume is
    # too small". They are also excluded from collision (see add_group_geoms).
    for rec in mesh_records:
        attrib = {"file": f"visuals/{rec['obj']}", "name": f"{rec['mesh_name']}_vis"}
        if rec.get("degenerate", False):
            attrib["inertia"] = "shell"
        ET.SubElement(asset, "mesh", attrib=attrib)
    # materials + textures
    for mname, m in materials.items():
        if m["texture"]:
            ET.SubElement(
                asset,
                "texture",
                file=f"visuals/{m['texture']}",
                name=f"{mname}_tex",
                type="2d",
            )
            ET.SubElement(
                asset,
                "material",
                name=f"{mname}_mat",
                texture=f"{mname}_tex",
                rgba=" ".join(f"{c:.4f}" for c in m["rgba"]),
                specular="0.3",
                shininess="0.3",
            )
        else:
            ET.SubElement(
                asset,
                "material",
                name=f"{mname}_mat",
                rgba=" ".join(f"{c:.4f}" for c in m["rgba"]),
                specular="0.3",
                shininess="0.3",
            )

    # defaults
    default = ET.SubElement(mujoco, "default")
    d_vis = ET.SubElement(default, "default", attrib={"class": "visual"})
    ET.SubElement(d_vis, "geom", conaffinity="0", contype="0", group="1", type="mesh")
    d_col = ET.SubElement(default, "default", attrib={"class": "collision"})
    ET.SubElement(d_col, "geom", group="0", rgba="0.5 0 0 0.5")
    d_reg = ET.SubElement(default, "default", attrib={"class": "region"})
    ET.SubElement(
        d_reg, "geom", group="1", conaffinity="0", contype="0", rgba="0 1 0 0"
    )

    worldbody = ET.SubElement(mujoco, "worldbody")
    outer = ET.SubElement(worldbody, "body")
    obj_body = ET.SubElement(outer, "body", name="object")

    mat_lookup = {rec["mesh_name"]: rec["mat"] for rec in mesh_records}

    def add_group_geoms(parent_elem, grp):
        for rec in mesh_records:
            if rec["group"] != grp:
                continue
            attrib = {
                "mesh": f"{rec['mesh_name']}_vis",
                "type": "mesh",
                "class": "visual",
            }
            if rec["mat"]:
                attrib["material"] = f"{rec['mat']}_mat"
            ET.SubElement(parent_elem, "geom", attrib=attrib)
            # Convex hulls fill an open container's cavity.  For an authored
            # open-top receptacle use its explicit floor / walls below instead.
            if not open_top_receptacle and not rec.get("degenerate", False):
                ET.SubElement(
                    parent_elem,
                    "geom",
                    attrib={
                        "mesh": f"{rec['mesh_name']}_vis",
                        "type": "mesh",
                        "class": "collision",
                        "solimp": "0.998 0.998 0.001",
                        "solref": "0.001 1",
                        "friction": "0.95 0.3 0.1",
                        "name": f"{rec['mesh_name']}_col",
                    },
                )

    # A static conversion preserves all visual mesh groups but deliberately
    # drops their USD articulation: the fixture is a fixed prop.
    if static:
        for grp in groups:
            add_group_geoms(obj_body, grp)
    else:
        for bg in base_groups:
            add_group_geoms(obj_body, bg)

    # hinged child bodies
    actuators = []
    for j in joints:
        child = ET.SubElement(obj_body, "body", name=_body_name(j["body1"]))
        px, py, pz = j["pivot_mj"]
        ax, ay, az = j["axis_mj"]
        lo, hi = j["range_rad"]
        jname = _body_name(j["body1"]) + "_joint"
        ET.SubElement(
            child,
            "joint",
            axis=f"{ax:.6f} {ay:.6f} {az:.6f}",
            pos=f"{px:.6f} {py:.6f} {pz:.6f}",
            limited="true",
            name=jname,
            range=f"{lo:.6f} {hi:.6f}",
            type="hinge",
            damping="1",
            frictionloss="1",
            armature=".01",
            stiffness="0.05",
        )
        add_group_geoms(child, j["body1"])
        actuators.append((jname, jname + "_actuator"))

    # ---- regions: reg_main (full bbox) + reg_int (shrunk interior) ----
    all_verts = [rec["verts"] for rec in mesh_records]
    bmin, bmax = bbox_of(all_verts)
    center = (bmin + bmax) / 2.0
    half = (bmax - bmin) / 2.0
    ET.SubElement(
        obj_body,
        "geom",
        attrib={
            "class": "region",
            "name": "reg_main",
            "type": "box",
            "pos": " ".join(f"{c:.6f}" for c in center),
            "size": " ".join(f"{c:.6f}" for c in half),
        },
    )
    int_half = half * 0.55
    ET.SubElement(
        obj_body,
        "geom",
        attrib={
            "class": "region",
            "name": "reg_int",
            "type": "box",
            "pos": " ".join(f"{c:.6f}" for c in center),
            "size": " ".join(f"{c:.6f}" for c in int_half),
        },
    )

    if open_top_receptacle:
        # A conservative open box: its visuals remain the original USD mesh,
        # while collision is represented by a floor and four thin walls.  The
        # interior stays collision-free and is exported as reg_receptacle for
        # placement / success checks.
        # The USD AABB includes small visual protrusions. Shrink the collision
        # footprint uniformly about its centre to fit the visible box shell.
        collision_footprint = 0.80
        wall = max(float(min(half[0], half[1])) * 0.02, 0.003)
        inner_xy = np.maximum(half[:2] * collision_footprint - wall, 0.01)
        receptacle_inner_z = max(float(half[2]) * 0.94, 0.02)
        wall_inner_z = max(float(half[2]) * 0.98, receptacle_inner_z)
        base_z = float(bmin[2])
        receptacle_center = np.array(
            [center[0], center[1], base_z + wall + receptacle_inner_z]
        )
        shell = ET.SubElement(obj_body, "body", name="receptacle_shell")
        collision = {
            "class": "collision",
            "type": "box",
            "friction": "0.95 0.3 0.1",
            "solimp": "0.998 0.998 0.001",
            "solref": "0.001 1",
        }

        def shell_geom(name, pos, size):
            attrib = dict(collision, name=name)
            attrib["pos"] = " ".join(f"{v:.6f}" for v in pos)
            attrib["size"] = " ".join(f"{v:.6f}" for v in size)
            ET.SubElement(shell, "geom", attrib=attrib)

        shell_geom(
            "box_floor_col",
            [center[0], center[1], base_z + wall / 2],
            [inner_xy[0] + wall, inner_xy[1] + wall, wall / 2],
        )
        wall_z = base_z + wall + wall_inner_z
        shell_geom(
            "box_wall_x_pos",
            [center[0] + inner_xy[0] + wall / 2, center[1], wall_z],
            [wall / 2, inner_xy[1] + wall, wall_inner_z],
        )
        shell_geom(
            "box_wall_x_neg",
            [center[0] - inner_xy[0] - wall / 2, center[1], wall_z],
            [wall / 2, inner_xy[1] + wall, wall_inner_z],
        )
        shell_geom(
            "box_wall_y_pos",
            [center[0], center[1] + inner_xy[1] + wall / 2, wall_z],
            [inner_xy[0], wall / 2, wall_inner_z],
        )
        shell_geom(
            "box_wall_y_neg",
            [center[0], center[1] - inner_xy[1] - wall / 2, wall_z],
            [inner_xy[0], wall / 2, wall_inner_z],
        )
        ET.SubElement(
            obj_body,
            "geom",
            attrib={
                "class": "region",
                "name": "reg_receptacle",
                "type": "box",
                "pos": " ".join(f"{c:.6f}" for c in receptacle_center),
                "size": " ".join(
                    f"{c:.6f}" for c in [inner_xy[0], inner_xy[1], receptacle_inner_z]
                ),
            },
        )

    act_elem = ET.SubElement(mujoco, "actuator")
    for jname, aname in actuators:
        ET.SubElement(act_elem, "motor", name=aname, joint=jname, gear="50")

    # pretty write
    rough = ET.tostring(mujoco, "utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    out_xml = os.path.join(out_dir, "model.xml")
    with open(out_xml, "w") as fh:
        fh.write(pretty)
    # Fail at conversion time instead of leaving a broken generated fixture for
    # a task reset to discover later.  Keep the converter usable in minimal
    # asset-only environments where mujoco is not installed.
    try:
        import mujoco

        mujoco.MjModel.from_xml_path(out_xml)
    except ImportError:
        pass
    return out_xml, [a[0] for a in actuators], (bmin, bmax)


def _body_name(group_path):
    return group_path.split("/")[-1]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("asset_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--name", default="articulated")
    ap.add_argument(
        "--static",
        action="store_true",
        help="convert a USD prop with no articulation sidecar",
    )
    ap.add_argument(
        "--open-top-receptacle",
        action="store_true",
        help="author a collision-free open receptacle volume",
    )
    ap.add_argument(
        "--extra-rot-x",
        type=float,
        default=0.0,
        help="extra rotation about X (deg) applied after Y-up->Z-up, "
        "to bake the upright/facing orientation into the asset",
    )
    ap.add_argument(
        "--extra-rot-y", type=float, default=0.0, help="extra rotation about Y (deg)"
    )
    ap.add_argument(
        "--extra-rot-z",
        type=float,
        default=0.0,
        help="extra rotation about Z (deg, yaw)",
    )
    args = ap.parse_args()
    set_extra_rotation(args.extra_rot_x, args.extra_rot_y, args.extra_rot_z)
    xml, jnames, bbox = convert(
        args.asset_dir,
        args.out_dir,
        args.name,
        static=args.static,
        open_top_receptacle=args.open_top_receptacle,
    )
    print("wrote:", xml)
    print("joints:", jnames)
    print("bbox(min,max) m:", bbox)
    if (args.extra_rot_x, args.extra_rot_y, args.extra_rot_z) != (0, 0, 0):
        print(
            f"extra rotation applied (deg): "
            f"x={args.extra_rot_x} y={args.extra_rot_y} z={args.extra_rot_z}"
        )
