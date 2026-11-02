"""Convert a USDZ prop into a RoboCasa MJCF XML asset.

This is a thin preparation wrapper around ``usd_to_mjcf_fixture.py``:

  1. extract the USDZ archive to a temporary asset directory,
  2. expose the archive's first USD/USDC file as ``instance.usd``,
  3. mirror common texture folders into ``Materials/Textures``,
  4. run the existing USD -> MJCF converter.

Example:
    python scripts/usd_pipeline/usdz_to_mjcf.py \
        robocasa/models/assets/objects/usd/charger/object.usdz \
        robocasa/models/assets/objects/usd/charger/converted \
        --name charger --static
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile
from xml.dom import minidom

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from usd_to_mjcf_fixture import convert, set_extra_rotation  # noqa: E402


def _site_name(*parts):
    text = "_".join(str(part) for part in parts if str(part))
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in text).strip("_")


def _metadata_support_site_name(key, asset_name=None):
    parts = [part for part in str(key).split("/") if part]
    if asset_name and parts and parts[0] == asset_name:
        parts = parts[1:]
    return _site_name(*parts, "site")


def _add_site(body, name, frame, rgba):
    if not frame or len(frame) < 3:
        return
    attrib = {
        "name": name,
        "type": "sphere",
        "size": "0.004",
        "pos": " ".join(f"{float(v):.6f}" for v in frame[:3]),
        "rgba": rgba,
        "group": "5",
    }
    if len(frame) >= 7:
        attrib["quat"] = " ".join(f"{float(v):.8f}" for v in frame[3:7])
    ET.SubElement(body, "site", attrib=attrib)


def _add_bbox_region(xml_path, bbox):
    if not bbox:
        return False
    mins, maxs = bbox
    center = [(float(lo) + float(hi)) * 0.5 for lo, hi in zip(mins, maxs)]
    half = [max((float(hi) - float(lo)) * 0.5, 0.001) for lo, hi in zip(mins, maxs)]

    tree = ET.parse(xml_path)
    root = tree.getroot()
    body = root.find(".//body[@name='object']")
    if body is None:
        return False
    for geom in body.findall("geom"):
        if geom.get("name") == "reg_bbox":
            return False
    ET.SubElement(
        body,
        "geom",
        attrib={
            "name": "reg_bbox",
            "class": "region",
            "type": "box",
            "pos": " ".join(f"{v:.6f}" for v in center),
            "size": " ".join(f"{v:.6f}" for v in half),
        },
    )
    rough = ET.tostring(root, "utf-8")
    pretty = minidom.parseString(rough).toprettyxml(indent="  ")
    with open(xml_path, "w") as handle:
        handle.write(pretty)
    return True


def _add_metadata_sites(xml_path, metadata_path, asset_name=None):
    if not metadata_path or not os.path.exists(metadata_path):
        return []
    with open(metadata_path) as handle:
        metadata = json.load(handle)

    tree = ET.parse(xml_path)
    root = tree.getroot()
    body = root.find(".//body[@name='object']")
    if body is None:
        return []

    added = []
    insert_frames = (
        metadata.get("active", {})
        .get("functional", {})
        .get("insert", {})
        .get("frame", [])
    )
    for index, frame in enumerate(insert_frames):
        name = "insert_site" if index == 0 else f"insert_site_{index}"
        _add_site(body, name, frame, "1 0.15 0.05 1")
        added.append(name)

    supports = metadata.get("passive", {}).get("support", {})
    for key, spec in sorted(supports.items()):
        centers = spec.get("center", [])
        if not centers:
            continue
        name = _metadata_support_site_name(key, asset_name=asset_name)
        _add_site(body, name, centers[0], "0.1 0.4 1 1")
        added.append(name)

    if added:
        rough = ET.tostring(root, "utf-8")
        pretty = minidom.parseString(rough).toprettyxml(indent="  ")
        with open(xml_path, "w") as handle:
            handle.write(pretty)
    return added


def _safe_extract_usdz(usdz_path, out_dir):
    with zipfile.ZipFile(usdz_path) as archive:
        for info in archive.infolist():
            target = os.path.abspath(os.path.join(out_dir, info.filename))
            if not target.startswith(os.path.abspath(out_dir) + os.sep):
                raise RuntimeError(f"unsafe path in USDZ archive: {info.filename}")
        archive.extractall(out_dir)


def _find_stage_file(asset_dir):
    candidates = []
    for root, _, files in os.walk(asset_dir):
        for filename in files:
            lower = filename.lower()
            if lower.endswith((".usd", ".usdc", ".usda")):
                path = os.path.join(root, filename)
                rel = os.path.relpath(path, asset_dir)
                priority = 0 if filename.lower() in {"main.usdc", "main.usd"} else 1
                candidates.append((priority, rel.count(os.sep), rel, path))
    if not candidates:
        raise RuntimeError(f"{asset_dir}: USDZ archive contains no USD stage")
    return sorted(candidates)[0][-1]


def _prepare_asset_dir(usdz_path, work_dir):
    _safe_extract_usdz(usdz_path, work_dir)
    stage = _find_stage_file(work_dir)
    instance = os.path.join(work_dir, "instance.usd")
    if os.path.abspath(stage) != os.path.abspath(instance):
        shutil.copyfile(stage, instance)

    textures = os.path.join(work_dir, "Materials", "Textures")
    os.makedirs(textures, exist_ok=True)
    for root, _, files in os.walk(work_dir):
        if os.path.abspath(root) == os.path.abspath(textures):
            continue
        if "texture" not in root.lower():
            continue
        for filename in files:
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                shutil.copyfile(
                    os.path.join(root, filename),
                    os.path.join(textures, filename),
                )
    return instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("usdz_path")
    parser.add_argument("out_dir")
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--static",
        action="store_true",
        help="convert as a non-articulated prop",
    )
    parser.add_argument("--open-top-receptacle", action="store_true")
    parser.add_argument("--extra-rot-x", type=float, default=0.0)
    parser.add_argument("--extra-rot-y", type=float, default=0.0)
    parser.add_argument("--extra-rot-z", type=float, default=0.0)
    parser.add_argument(
        "--metadata",
        default=None,
        help="optional metadata.json used to author functional MJCF sites",
    )
    parser.add_argument(
        "--keep-workdir",
        action="store_true",
        help="keep the extracted USDZ directory next to the output for debugging",
    )
    args = parser.parse_args()

    usdz_path = os.path.abspath(args.usdz_path)
    out_dir = os.path.abspath(args.out_dir)
    name = args.name or os.path.splitext(os.path.basename(usdz_path))[0]
    os.makedirs(out_dir, exist_ok=True)
    metadata_path = args.metadata
    if metadata_path is None:
        sibling = os.path.join(os.path.dirname(usdz_path), "metadata.json")
        metadata_path = sibling if os.path.exists(sibling) else None
    elif metadata_path:
        metadata_path = os.path.abspath(metadata_path)

    if args.keep_workdir:
        work_dir = os.path.join(out_dir, "_usdz_extract")
        if os.path.exists(work_dir):
            shutil.rmtree(work_dir)
        os.makedirs(work_dir)
        _prepare_asset_dir(usdz_path, work_dir)
        asset_dir = work_dir
        set_extra_rotation(args.extra_rot_x, args.extra_rot_y, args.extra_rot_z)
        xml, joints, bbox = convert(
            asset_dir,
            out_dir,
            name,
            static=args.static,
            open_top_receptacle=args.open_top_receptacle,
        )
    else:
        with tempfile.TemporaryDirectory(prefix="usdz_to_mjcf_") as work_dir:
            _prepare_asset_dir(usdz_path, work_dir)
            set_extra_rotation(args.extra_rot_x, args.extra_rot_y, args.extra_rot_z)
            xml, joints, bbox = convert(
                work_dir,
                out_dir,
                name,
                static=args.static,
                open_top_receptacle=args.open_top_receptacle,
            )

    has_bbox = _add_bbox_region(xml, bbox)
    sites = _add_metadata_sites(xml, metadata_path, asset_name=name)
    print("wrote:", xml)
    print("joints:", joints)
    print("bbox(min,max) m:", bbox)
    print("bbox_region:", has_bbox)
    print("sites:", sites)


if __name__ == "__main__":
    main()
