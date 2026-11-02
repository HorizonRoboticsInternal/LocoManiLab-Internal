"""Decimate the high-poly furniture visual meshes for faster rendering.

The Meshy-AI furniture exports (sofas / coffee_tables / tv_stands / side_tables)
are ~120K-270K vertices each. With several in a scene the on-screen MuJoCo
viewer becomes choppy (the viewer renders them every frame and contends with
SONIC on the GPU). A visual mesh only needs a few thousand faces, and collision
is a separate box (see convert_meshy_furniture.py), so decimating is a pure
render speedup with no physics change.

UV/texture coords are preserved (MuJoCo reads the texture from model.xml's
``<texture>`` element + the obj's ``vt`` coords, not from the obj's mtl), so the
appearance is unchanged. Originals are backed up to ``model.orig.obj`` and every
run re-decimates from that backup, so it's idempotent and reversible.

Usage:
    python scripts/decimate_furniture.py                       # all furniture, 20k faces
    python scripts/decimate_furniture.py --target-faces 15000
    python scripts/decimate_furniture.py --dirs sofas/Sofa001 tv_stands/TVStand001
    python scripts/decimate_furniture.py --restore             # put the originals back
"""

import argparse
import pathlib
import shutil

import pymeshlab as ml

import robocasa

FIXTURES = pathlib.Path(robocasa.__file__).parent / "models/assets/fixtures"
DEFAULT_CATEGORIES = ["sofas", "coffee_tables", "tv_stands", "side_tables"]


def find_furniture_objs(dirs):
    objs = []
    if dirs:
        for d in dirs:
            p = FIXTURES / d / "visual" / "model.obj"
            if p.exists():
                objs.append(p)
            else:
                print(f"  skip (no visual/model.obj): {d}")
    else:
        for cat in DEFAULT_CATEGORIES:
            catdir = FIXTURES / cat
            if not catdir.is_dir():
                continue
            for inst in sorted(catdir.iterdir()):
                p = inst / "visual" / "model.obj"
                if p.exists():
                    objs.append(p)
    return objs


def _label(obj_path):
    # .../fixtures/<cat>/<inst>/visual/model.obj -> "<cat>/<inst>"
    return f"{obj_path.parents[2].name}/{obj_path.parents[1].name}"


def decimate(obj_path, target_faces):
    backup = obj_path.with_name("model.orig.obj")
    if not backup.exists():
        shutil.copy(obj_path, backup)  # first run: snapshot the original

    ms = ml.MeshSet()
    ms.load_new_mesh(str(backup))  # always decimate from the original (idempotent)
    f0 = ms.current_mesh().face_number()
    if f0 <= target_faces:
        print(f"  {_label(obj_path)}: {f0} faces <= target, skip")
        return

    ms.meshing_decimation_quadric_edge_collapse_with_texture(
        targetfacenum=target_faces,
        preserveboundary=True,
        preservenormal=True,
        optimalplacement=True,
    )
    f1 = ms.current_mesh().face_number()
    ms.save_current_mesh(str(obj_path), save_wedge_texcoord=True)
    sz0 = backup.stat().st_size / 1e6
    sz1 = obj_path.stat().st_size / 1e6
    print(f"  {_label(obj_path)}: {f0} -> {f1} faces | obj {sz0:.0f} MB -> {sz1:.1f} MB")


def restore(obj_path):
    backup = obj_path.with_name("model.orig.obj")
    if backup.exists():
        shutil.copy(backup, obj_path)
        print(f"  {_label(obj_path)}: restored from model.orig.obj")
    else:
        print(f"  {_label(obj_path)}: no backup to restore")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-faces", type=int, default=20000)
    ap.add_argument("--dirs", nargs="*", default=None,
                    help="specific <category>/<instance> dirs; default = all furniture")
    ap.add_argument("--restore", action="store_true",
                    help="restore the original (pre-decimation) meshes from backup")
    args = ap.parse_args()

    objs = find_furniture_objs(args.dirs)
    if args.restore:
        print(f"restoring {len(objs)} furniture mesh(es) from backups...")
        for o in objs:
            restore(o)
        return

    print(f"decimating {len(objs)} furniture mesh(es) to ~{args.target_faces} faces...")
    for o in objs:
        decimate(o, args.target_faces)
    print("done. originals backed up to model.orig.obj (re-run is idempotent; "
          "--restore to undo).")


if __name__ == "__main__":
    main()
