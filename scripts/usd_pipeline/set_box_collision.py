"""Replace an articulated fixture's mesh collision with non-interlocking boxes.

Step 2 of the USD->RoboCasa pipeline (see ``docs/usd_articulated_asset_pipeline.md``
§2). This is the general tool that replaces the old per-asset
``set_<asset>_box_collision.py`` copies: the hand-tuned box geometry now lives
as data in ``box_configs/<AssetName>.json`` and this script applies it.

Why boxes (not mesh / hull / CoACD): for a lidded appliance the door mesh and
body mesh share volume at the seam; any mesh-derived collision leaves the two
interlocked at closed and geometrically locks the hinge. Native RoboCasa
appliances use a few hand-placed boxes for the same reason.

What it does to ``<asset_dir>/model.xml``:
  * drop every collision ``<mesh ... name="..._col"/>`` from ``<asset>``,
  * drop every ``<geom class="collision">`` currently on the base + door bodies,
  * add the box geoms from the JSON config.
Visual meshes and joints are untouched.

Box config JSON format (coordinates in the fixture's local Z-up metre frame):

    {
      "asset": "robocasa/models/assets/fixtures/articulated/ElectricCooker001",
      "body": [
        {"name": "body_back", "pos": [0.0, 0.0805, 0.0], "size": [0.123, 0.0235, 0.151]},
        ...
      ],
      "door": [
        {"name": "door_col", "pos": [0.0, -0.082, 0.0105], "size": [0.111, 0.022, 0.1305]}
      ]
    }

``body`` boxes go on the base ``<body name="object">``; ``door`` boxes go on the
first body that has a ``<joint>`` child (the converter's single-hinge output).

Usage:
    python scripts/usd_pipeline/set_box_collision.py box_configs/ElectricCooker001.json
    python scripts/usd_pipeline/set_box_collision.py /path/to/myasset.json
    # override the asset dir from the JSON (e.g. while iterating):
    python scripts/usd_pipeline/set_box_collision.py cfg.json --asset path/to/AssetDir
"""

import argparse
import json
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom


COMMON = {
    "type": "box",
    "class": "collision",
    "solimp": "0.998 0.998 0.001",
    "solref": "0.001 1",
    "friction": "0.95 0.3 0.1",
}


def _fmt(t):
    return " ".join(f"{v:.6f}" for v in t)


def apply_box_collision(asset_dir, body_boxes, door_boxes):
    """Rewrite ``<asset_dir>/model.xml`` to use the given box collision geoms."""
    xml_path = os.path.join(asset_dir, "model.xml")
    tree = ET.parse(xml_path)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")

    # 1) remove every collision mesh from <asset> (names end with _col)
    for mesh in list(asset.findall("mesh")):
        if mesh.get("name", "").endswith("_col"):
            asset.remove(mesh)

    # 2) find base ("object") and door (first body with a <joint> child)
    base_body = worldbody.find(".//body[@name='object']")
    if base_body is None:
        raise RuntimeError("could not find <body name='object'> in model.xml")
    door_body = None
    for body in worldbody.iter("body"):
        if body.find("joint") is not None:
            door_body = body
            break

    # 3) remove existing collision geoms from base (+ door if present)
    targets = [base_body] + ([door_body] if door_body is not None else [])
    for body in targets:
        for g in list(body.findall("geom")):
            if g.get("class") == "collision":
                body.remove(g)

    # 4) add the new box geoms
    for box in body_boxes:
        ET.SubElement(base_body, "geom", attrib={
            **COMMON, "name": box["name"],
            "pos": _fmt(box["pos"]), "size": _fmt(box["size"]),
        })
    if door_boxes:
        if door_body is None:
            raise RuntimeError(
                "config has 'door' boxes but model.xml has no body with a <joint>"
            )
        for box in door_boxes:
            ET.SubElement(door_body, "geom", attrib={
                **COMMON, "name": box["name"],
                "pos": _fmt(box["pos"]), "size": _fmt(box["size"]),
            })

    rough = ET.tostring(root, "utf-8")
    pretty = "\n".join(
        l for l in minidom.parseString(rough).toprettyxml(indent="  ").splitlines()
        if l.strip()
    )
    with open(xml_path, "w") as fh:
        fh.write(pretty + "\n")
    print(f"rewrote {xml_path}")
    print(f"  base body: {base_body.get('name')}  ({len(body_boxes)} box geoms)")
    if door_boxes:
        print(f"  door body: {door_body.get('name')}  ({len(door_boxes)} box geoms)")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", help="path to the box-config JSON (see module docstring)")
    ap.add_argument("--asset", default=None,
                    help="override the asset dir from the JSON config")
    args = ap.parse_args()

    cfg_path = args.config
    if not os.path.isabs(cfg_path) and not os.path.exists(cfg_path):
        # allow naming a config in this script's box_configs/ dir directly
        here = os.path.join(os.path.dirname(__file__), "box_configs", cfg_path)
        if os.path.exists(here):
            cfg_path = here
    with open(cfg_path) as fh:
        cfg = json.load(fh)

    asset_dir = args.asset or cfg["asset"]
    if not os.path.isabs(asset_dir):
        # resolve relative to the repo root (two levels up from scripts/usd_pipeline/)
        repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
        asset_dir = os.path.normpath(os.path.join(repo_root, asset_dir))

    apply_box_collision(asset_dir, cfg.get("body", []), cfg.get("door", []))


if __name__ == "__main__":
    main()
