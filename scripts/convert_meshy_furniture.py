"""Convert a Meshy-AI (or any single-mesh OBJ) export into a robocasa furniture asset.

This automates the manual steps used to turn a Meshy AI ``.obj/.mtl/.png`` export
into a robocasa object-format asset (see ``coffee_tables/CoffeeTable001`` for the
reference output):

  1. Copy the mesh/material/texture into ``<asset>/visual/`` with ASCII names
     (``model.obj`` / ``model.mtl`` / ``texture.png``) and fix the ``mtllib`` /
     ``map_Kd`` references (MuJoCo chokes on the original non-ASCII filenames).
  2. Compute the mesh axis-aligned bounding box to derive scale and ``reg_bbox``.
  3. Convert the source up-axis to MuJoCo's Z-up (Meshy/Blender export Y-up).
  4. Recenter the mesh at the body origin.
  5. Emit ``model.xml`` (visual mesh + texture/material + ``reg_bbox`` region +
     an approximate collision box).
  6. Optionally render a verification screenshot.

Output goes to ``robocasa/models/assets/furniture/<category>/<name>/`` (this tree
is gitignored — assets are not committed).

Examples
--------
    # by target real-world size (longest final dimension = 1.1 m):
    python scripts/convert_meshy_furniture.py \
        --src ./meshy_export \
        --category coffee_tables --name CoffeeTable001 \
        --target-size 1.1 --render

    # by explicit scale, source already Z-up, no collision box:
    python scripts/convert_meshy_furniture.py \
        --src ./some_export --category sofas --name Sofa001 \
        --scale 0.6 --up z --collision none

Run in the ``tele_benchmark`` conda env. ``--render`` needs a working GL
(headless: prefix with ``MUJOCO_GL=egl``).
"""

import argparse
import os
import shutil
import sys

import numpy as np

import robocasa


# --------------------------------------------------------------------------- #
# Input discovery
# --------------------------------------------------------------------------- #
def find_inputs(src, obj, mtl, texture):
    """Resolve the obj / mtl / texture paths from a source dir or explicit flags."""
    def _first(globpat):
        import glob

        hits = sorted(glob.glob(os.path.join(src, globpat))) if src else []
        return hits[0] if hits else None

    obj = obj or _first("*.obj")
    mtl = mtl or _first("*.mtl")
    texture = texture or _first("*.png") or _first("*.jpg") or _first("*.jpeg")

    if not obj or not os.path.isfile(obj):
        sys.exit(f"[error] no .obj found (looked in --src={src!r}, --obj={obj!r})")
    if not texture or not os.path.isfile(texture):
        sys.exit(f"[error] no texture (.png/.jpg) found (--src={src!r}, --texture={texture!r})")
    return obj, mtl, texture


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #
def compute_aabb(obj_path):
    """Return (min_xyz, max_xyz) over the OBJ 'v' (vertex) lines."""
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                _, x, y, z = line.split()[:4]
                v = np.array([float(x), float(y), float(z)])
                lo = np.minimum(lo, v)
                hi = np.maximum(hi, v)
    if not np.all(np.isfinite(lo)):
        sys.exit(f"[error] no vertices parsed from {obj_path}")
    return lo, hi


# --------------------------------------------------------------------------- #
# Asset file copy + reference fixup
# --------------------------------------------------------------------------- #
def copy_assets(obj, mtl, texture, visual_dir):
    """Copy mesh/material/texture into visual_dir with ASCII names; fix references."""
    os.makedirs(visual_dir, exist_ok=True)
    tex_ext = os.path.splitext(texture)[1].lower()
    tex_name = "texture" + tex_ext

    # obj: rewrite the mtllib line to point at our renamed mtl.
    with open(obj, "r", errors="ignore") as fin, open(
        os.path.join(visual_dir, "model.obj"), "w"
    ) as fout:
        for line in fin:
            if line.startswith("mtllib "):
                fout.write("mtllib model.mtl\n")
            else:
                fout.write(line)

    # mtl: rewrite map_Kd to the renamed texture (create a minimal one if absent).
    mtl_out = os.path.join(visual_dir, "model.mtl")
    if mtl and os.path.isfile(mtl):
        with open(mtl, "r", errors="ignore") as fin, open(mtl_out, "w") as fout:
            for line in fin:
                if line.strip().lower().startswith("map_kd"):
                    fout.write(f"map_Kd {tex_name}\n")
                else:
                    fout.write(line)
    else:
        with open(mtl_out, "w") as fout:
            fout.write(f"newmtl Material\nmap_Kd {tex_name}\n")

    shutil.copyfile(texture, os.path.join(visual_dir, tex_name))
    os.chmod(os.path.join(visual_dir, tex_name), 0o644)
    return tex_name


# --------------------------------------------------------------------------- #
# model.xml emission
# --------------------------------------------------------------------------- #
def compute_transform(lo, hi, scale, up):
    """Return (half_extents, geom_pos, euler_attr) in the final Z-up frame.

    The visual geom is rotated (if needed) into Z-up and translated so the mesh
    bounding-box center lands on the body origin.
    """
    center = (lo + hi) / 2.0
    half_src = (hi - lo) / 2.0  # source half-extents (x, y, z)
    cx, cy, cz = center
    if up == "y":
        # Y-up -> Z-up: rotate +90 deg about X. Maps (x,y,z)->(x,-z,y); spans swap Y<->Z.
        euler_attr = ' euler="1.5708 0 0"'
        half = np.array([half_src[0], half_src[2], half_src[1]]) * scale
        # Recenter: pos = -R*(scale*center) = scale*(-cx, cz, -cy).
        geom_pos = scale * np.array([-cx, cz, -cy])
    elif up == "z":
        euler_attr = ""
        half = half_src * scale
        geom_pos = scale * np.array([-cx, -cy, -cz])
    else:
        sys.exit(f"[error] --up must be 'y' or 'z', got {up!r}")
    return half, geom_pos, euler_attr


def write_model_xml(asset_dir, name, tex_name, half, geom_pos, euler_attr, scale, up, collision):
    """Write model.xml. Returns (final_half_extents, suggested_floor_z)."""
    hx, hy, hz = half
    px, py, pz = geom_pos

    if collision == "box":
        collision_geom = (
            '        <!-- Approximate collision: full bounding box. Refine per asset\n'
            '             (e.g. split into tabletop + legs, or carve out shelves). -->\n'
            f'        <geom class="collision" type="box" pos="0 0 0" '
            f'size="{hx:.4f} {hy:.4f} {hz:.4f}"/>\n'
        )
    elif collision == "none":
        collision_geom = "        <!-- No collision geom (visual-only). -->\n"
    else:
        sys.exit(f"[error] --collision must be 'box' or 'none', got {collision!r}")

    xml = f"""<?xml version='1.0' encoding='UTF-8'?>
<mujoco model="{name}">
  <option timestep="0.002" gravity="0.0 0.0 -9.81"/>
  <compiler angle="radian"/>
  <asset>
    <!-- Generated by scripts/convert_meshy_furniture.py.
         Source up-axis: {up}.  scale={scale:.4f}.
         Final extents (m): {2*hx:.3f} x {2*hy:.3f} x {2*hz:.3f}.
         Rest feet on the floor by placing the body at z={hz:.3f}. -->
    <mesh file="visual/model.obj" name="{name}_vis" scale="{scale:.6f} {scale:.6f} {scale:.6f}"/>
    <texture file="visual/{tex_name}" name="{name}_tex" type="2d"/>
    <material name="{name}_mat" texture="{name}_tex" specular="0.2" shininess="0.1"/>
  </asset>
  <default>
    <default class="visual">
      <geom conaffinity="0" contype="0" group="1" type="mesh"/>
    </default>
    <default class="collision">
      <geom group="0" rgba="0.5 0 0 0.5"/>
    </default>
    <default class="region">
      <geom group="1" conaffinity="0" contype="0" rgba="0 1 0 0" mass="0"/>
    </default>
  </default>
  <worldbody>
    <body>
      <body name="object">
        <!-- Visible mesh (non-colliding), recentered + rotated into Z-up. -->
        <geom mesh="{name}_vis" class="visual" material="{name}_mat"{euler_attr} pos="{px:.4f} {py:.4f} {pz:.4f}"/>
        <!-- Bounding-box region for the placement sampler. -->
        <geom class="region" name="reg_bbox" type="box" pos="0 0 0" size="{hx:.4f} {hy:.4f} {hz:.4f}"/>
{collision_geom}      </body>
    </body>
  </worldbody>
</mujoco>
"""
    out = os.path.join(asset_dir, "model.xml")
    with open(out, "w") as f:
        f.write(xml)
    return half, hz


# --------------------------------------------------------------------------- #
# Optional verification render
# --------------------------------------------------------------------------- #
def render_verify(asset_dir, name, tex_name, scale, half, geom_pos, euler_attr, hz, out_png):
    """Render the asset on a floor with a 0.5 m reference cube, to a PNG."""
    try:
        import mujoco
        import imageio.v2 as imageio
    except Exception as e:  # noqa: BLE001
        print(f"[render] skipped (import failed: {e})")
        return
    hx, hy, _ = half
    px, py, pz = geom_pos
    xml = f"""
<mujoco>
  <compiler angle="radian" assetdir="{asset_dir}"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <asset>
    <mesh file="visual/model.obj" name="m" scale="{scale:.6f} {scale:.6f} {scale:.6f}"/>
    <texture file="visual/{tex_name}" name="t" type="2d"/>
    <material name="mat" texture="t" specular="0.2" shininess="0.1"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.3 0.3 0.3" rgb2="0.4 0.4 0.4" width="300" height="300"/>
    <material name="floor" texture="grid" texrepeat="10 10"/>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.5 0.7 0.9" rgb2="0.1 0.1 0.15" width="32" height="32"/>
  </asset>
  <worldbody>
    <light pos="0 0 4" dir="0 0 -1"/><light pos="2 -2 3" dir="-1 1 -1"/>
    <geom name="floor" type="plane" size="3 3 0.1" material="floor"/>
    <geom name="ref" type="box" pos="1.2 0 0.25" size="0.25 0.25 0.25" rgba="0.2 0.6 1 0.5"/>
    <body name="object" pos="0 0 {hz:.4f}">
      <geom mesh="m" material="mat" type="mesh" contype="0" conaffinity="0" group="1"{euler_attr} pos="{px:.4f} {py:.4f} {pz:.4f}"/>
    </body>
    <camera name="v" pos="2.2 -2.2 1.7" xyaxes="0.7 0.7 0 -0.35 0.35 0.87"/>
  </worldbody>
</mujoco>
"""
    try:
        m = mujoco.MjModel.from_xml_string(xml)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        r = mujoco.Renderer(m, height=720, width=960)
        r.update_scene(d, camera="v")
        imageio.imwrite(out_png, r.render())
        print(f"[render] wrote {out_png}")
    except Exception as e:  # noqa: BLE001
        print(f"[render] failed: {e}")


def main():
    p = argparse.ArgumentParser(
        description="Convert a Meshy-AI OBJ export into a robocasa furniture asset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--src", help="source dir containing the .obj/.mtl/.png export")
    p.add_argument("--obj", help="explicit .obj path (overrides --src lookup)")
    p.add_argument("--mtl", help="explicit .mtl path (optional)")
    p.add_argument("--texture", help="explicit texture path (overrides --src lookup)")
    p.add_argument("--category", required=True, help="furniture category dir, e.g. coffee_tables")
    p.add_argument("--name", required=True, help="instance name, e.g. CoffeeTable001")
    p.add_argument("--scale", type=float, help="explicit uniform scale factor")
    p.add_argument(
        "--target-size",
        type=float,
        help="set the longest FINAL dimension (m); computes scale automatically. "
        "Use this for Meshy exports (normalized to [-1,1]).",
    )
    p.add_argument("--up", default="y", choices=["y", "z"],
                   help="source up-axis (Meshy/Blender = y)")
    p.add_argument("--collision", default="box", choices=["box", "none"],
                   help="collision geom: full bounding box, or none (visual-only)")
    p.add_argument("--out-root", default=None,
                   help="asset root (default: robocasa/models/assets/furniture)")
    p.add_argument("--render", action="store_true", help="write a verification PNG")
    p.add_argument("--force", action="store_true", help="overwrite an existing asset dir")
    args = p.parse_args()

    obj, mtl, texture = find_inputs(args.src, args.obj, args.mtl, args.texture)

    out_root = args.out_root or os.path.join(
        robocasa.models.assets_root, "furniture"
    )
    asset_dir = os.path.join(out_root, args.category, args.name)
    if os.path.exists(asset_dir) and not args.force:
        sys.exit(f"[error] {asset_dir} already exists (use --force to overwrite)")

    lo, hi = compute_aabb(obj)
    spans = hi - lo
    print(f"[aabb] source spans (x,y,z) = ({spans[0]:.3f}, {spans[1]:.3f}, {spans[2]:.3f})")

    # Resolve scale.
    if args.scale is not None:
        scale = args.scale
    elif args.target_size is not None:
        # Longest FINAL dimension: with y-up, Y<->Z swap doesn't change the max span.
        scale = args.target_size / float(np.max(spans))
        print(f"[scale] target-size {args.target_size} m -> scale {scale:.4f}")
    else:
        scale = 1.0
        print("[warn] no --scale/--target-size given; using scale=1.0. "
              "Meshy exports are normalized to [-1,1] — you likely want --target-size.")

    half, geom_pos, euler_attr = compute_transform(lo, hi, scale, args.up)
    visual_dir = os.path.join(asset_dir, "visual")
    tex_name = copy_assets(obj, mtl, texture, visual_dir)
    half, hz = write_model_xml(
        asset_dir, args.name, tex_name, half, geom_pos, euler_attr,
        scale, args.up, args.collision,
    )

    print(f"[done] wrote {asset_dir}/model.xml")
    print(f"[done] final extents (m): {2*half[0]:.3f} x {2*half[1]:.3f} x {2*half[2]:.3f}")
    print(f"[done] place the body at z={hz:.3f} to rest it on the floor.")
    print(f"[use ] FurnitureObject(name='{args.name.lower()}', "
          f"mjcf_path=furniture_mjcf_path('{args.category}', '{args.name}'), "
          f"pos=(x, y, {hz:.3f}))")

    if args.render:
        render_verify(
            asset_dir, args.name, tex_name, scale, half, geom_pos, euler_attr, hz,
            os.path.join(asset_dir, "preview.png"),
        )


if __name__ == "__main__":
    main()
