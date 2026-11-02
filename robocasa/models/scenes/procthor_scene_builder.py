"""ProcTHOR scene builder helpers.

This module is the ProcTHOR counterpart to the kitchen scene builder, but the
input starts from exported MolmoSpaces assets rather than hand-authored
RoboCasa layout YAML.  ProcTHOR already ships a structured ``train_N.json``
with rooms, walls, doors, and objects; the MJCF XML supplies the baked body /
geom / joint names.  The first slice here converts those sources into
Kitchen-like fixture configs without re-instantiating the fixtures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import json
import os
import re
import xml.etree.ElementTree as ET

import numpy as np


_TRAIN_XML_RE = re.compile(r"(train_\d+)(?:\.robocasa)?\.xml$")


def _load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def _scene_sidecar_paths(scene_xml):
    """Return the ProcTHOR JSON and metadata paths for a scene XML path."""
    scene_xml = os.path.abspath(scene_xml)
    scene_dir = os.path.dirname(scene_xml)
    m = _TRAIN_XML_RE.fullmatch(os.path.basename(scene_xml))
    if m is None:
        raise ValueError(f"not a ProcTHOR train XML path: {scene_xml}")
    scene_name = m.group(1)
    return (
        os.path.join(scene_dir, f"{scene_name}.json"),
        os.path.join(scene_dir, f"{scene_name}_metadata.json"),
    )


def _xyz_to_mjcf(pos):
    """Convert ProcTHOR JSON coordinates (x, y-up, z) to MJCF (x, y, z-up)."""
    return [float(pos["x"]), float(pos["z"]), float(pos["y"])]


def _rotation_to_yaw(rotation):
    """Return the horizontal yaw in radians from ProcTHOR's y-axis rotation."""
    return float(np.deg2rad(rotation.get("y", 0.0)))


def _sanitize_name(name):
    out = re.sub(r"[^0-9a-zA-Z_]+", "_", str(name)).strip("_").lower()
    return out or "fixture"


def _body_table(scene_xml):
    root = ET.parse(scene_xml).getroot()
    bodies = {}
    for body in root.findall(".//body"):
        name = body.get("name")
        if name:
            bodies[name] = dict(body.attrib)
    return bodies


def _metadata_by_object_id(metadata):
    out = {}
    for body_name, info in metadata.get("objects", {}).items():
        object_id = info.get("object_id")
        if object_id:
            out[object_id] = (body_name, info)
    return out


def _metadata_by_body(metadata):
    return metadata.get("objects", {})


def _object_fixture_type(object_type):
    mapping = {
        "CounterTop": "counter",
        "DiningTable": "table",
        "Dresser": "dresser",
        "Fridge": "fridge",
        "GarbageBag": "trash",
        "Sofa": "sofa",
        "TVStand": "tv_stand",
        "ShelvingUnit": "shelf",
        "Bed": "bed",
        "HousePlant": "plant",
        "Painting": "wall_accessory",
    }
    return mapping.get(object_type, _sanitize_name(object_type))


@dataclass
class ProcTHORFixtureRecord:
    """Lightweight fixture handle for a baked ProcTHOR body.

    Kitchen fixtures are instantiated MJCF objects. ProcTHOR fixtures already
    exist inside the loaded arena, so this record points to their baked bodies
    while exposing the metadata tasks need.
    """

    name: str
    fixture_type: str
    body_name: str | None = None
    asset_id: str | None = None
    object_id: str | None = None
    room_id: str | None = None
    pos: np.ndarray | None = None
    yaw: float | None = None
    body_names: list[str] = field(default_factory=list)
    joint_names: list[str] = field(default_factory=list)
    site_names: list[str] = field(default_factory=list)
    source: dict = field(default_factory=dict)

    @property
    def nat_lang(self):
        return self.fixture_type.replace("_", " ")

    def _source_aliases(self, kind, name):
        values = self.source.get(f"{kind}_names", {})
        if isinstance(values, dict):
            alias = values.get(name)
            if alias is None:
                return []
            return alias if isinstance(alias, (list, tuple)) else [alias]
        return []

    def _find_named_part(
        self,
        kind,
        token=None,
        fallback=None,
        required=True,
        exclude_tokens=(),
    ):
        names = list(getattr(self, f"{kind}_names"))
        tokens = [] if token is None else ([token] if isinstance(token, str) else token)
        exclude_tokens = (
            [exclude_tokens] if isinstance(exclude_tokens, str) else exclude_tokens
        )
        if token is not None:
            for name in names:
                haystack = " ".join(
                    str(value).lower()
                    for value in [name, *self._source_aliases(kind, name)]
                )
                if all(str(t).lower() in haystack for t in tokens) and not any(
                    str(t).lower() in haystack for t in exclude_tokens
                ):
                    return name
        if fallback is not None and fallback in names:
            return fallback
        if token is None and names:
            return names[0]
        if not required:
            return None
        token_msg = "" if token is None else f" matching {token!r}"
        raise RuntimeError(
            f"Could not find {kind}{token_msg} on fixture {self.name}; "
            f"available {kind}s: {names}"
        )

    def find_body(self, token=None, fallback=None, required=True, exclude_tokens=()):
        return self._find_named_part("body", token, fallback, required, exclude_tokens)

    def find_joint(self, token=None, fallback=None, required=True, exclude_tokens=()):
        return self._find_named_part("joint", token, fallback, required, exclude_tokens)

    def find_site(self, token=None, fallback=None, required=True, exclude_tokens=()):
        return self._find_named_part("site", token, fallback, required, exclude_tokens)

    def body_prefixes(self):
        return tuple(self.body_names or ([self.body_name] if self.body_name else []))


def extract_procthor_layout_and_style(scene_xml):
    """Extract Kitchen-style layout/style dictionaries from a ProcTHOR scene.

    The layout contains semantic room / wall / door / object records. The style
    maps fixture types and asset ids back to their baked MJCF body roots.
    """
    scene_xml = os.path.abspath(scene_xml)
    scene_json_path, metadata_path = _scene_sidecar_paths(scene_xml)
    scene = _load_json(scene_json_path)
    metadata = _load_json(metadata_path)
    body_attrs = _body_table(scene_xml)
    meta_by_object_id = _metadata_by_object_id(metadata)

    layout = {
        "scene_xml": scene_xml,
        "scene_json": scene_json_path,
        "metadata_json": metadata_path,
        "rooms": [],
        "walls": [],
        "doors": [],
        "fixtures": [],
    }
    style = {
        "assets": {},
        "body_roots": {},
    }

    for room in scene.get("rooms", []):
        room_id = room["id"].split("|")[-1]
        layout["rooms"].append(
            {
                "name": f"room_{room_id}",
                "type": "floor",
                "room_id": room_id,
                "room_type": room.get("roomType"),
                "floor_polygon": [_xyz_to_mjcf(p) for p in room["floorPolygon"]],
                "material": room.get("floorMaterial", {}).get("name"),
                "body_name": f"room_{room_id}"
                if f"room_{room_id}" in body_attrs
                else None,
            }
        )

    for i, wall in enumerate(scene.get("walls", [])):
        room_id = wall["roomId"].split("|")[-1]
        name = f"wall_{room_id}_{i}"
        layout["walls"].append(
            {
                "name": name,
                "type": "wall",
                "room_id": room_id,
                "wall_id": wall.get("id"),
                "polygon": [_xyz_to_mjcf(p) for p in wall["polygon"]],
                "material": wall.get("material", {}).get("name"),
                "body_name": name if name in body_attrs else None,
            }
        )

    for door in scene.get("doors", []):
        body_name, info = meta_by_object_id.get(door.get("id"), (None, {}))
        layout["doors"].append(
            {
                "name": _sanitize_name(door.get("id")),
                "type": "door",
                "door_id": door.get("id"),
                "asset_id": door.get("assetId"),
                "rooms": [door.get("room0"), door.get("room1")],
                "walls": [door.get("wall0"), door.get("wall1")],
                "position": _xyz_to_mjcf(door["assetPosition"]),
                "body_name": body_name,
                "body_names": info.get("name_map", {}).get("bodies", []),
                "joint_names": info.get("name_map", {}).get("joints", []),
                "site_names": info.get("name_map", {}).get("sites", []),
            }
        )

    for obj in scene.get("objects", []):
        object_id = obj.get("id")
        body_name, info = meta_by_object_id.get(object_id, (None, {}))
        object_type = obj.get("objectType")
        fixture_type = _object_fixture_type(object_type)
        asset_id = obj.get("assetId")
        fixture = {
            "name": _sanitize_name(object_id),
            "type": fixture_type,
            "object_type": object_type,
            "object_id": object_id,
            "asset_id": asset_id,
            "room_id": str(info.get("room_id", "") or "").strip() or None,
            "position": _xyz_to_mjcf(obj["position"]),
            "yaw": _rotation_to_yaw(obj.get("rotation", {})),
            "body_name": body_name,
            "body_names": info.get("name_map", {}).get("bodies", []),
            "joint_names": info.get("name_map", {}).get("joints", []),
            "site_names": info.get("name_map", {}).get("sites", []),
            "children": [child.get("id") for child in obj.get("children", [])],
        }
        layout["fixtures"].append(fixture)
        if asset_id is not None:
            style["assets"].setdefault(fixture_type, {})[asset_id] = {
                "body_name": body_name,
                "object_type": object_type,
            }
        if body_name is not None:
            style["body_roots"][body_name] = {
                "fixture_type": fixture_type,
                "asset_id": asset_id,
                "object_id": object_id,
            }

    return layout, style


def create_fixtures(layout_config, style_config=None, rng=None):
    """Create lightweight fixture records from extracted ProcTHOR layout data.

    Mirrors ``scene_builder.create_fixtures`` at the return-shape level
    (``name -> model``), but the models are references to baked scene bodies.
    """
    del style_config, rng
    fixtures = {}
    for section in ("rooms", "walls", "doors", "fixtures"):
        for cfg in layout_config.get(section, []):
            name = cfg["name"]
            fixtures[name] = ProcTHORFixtureRecord(
                name=name,
                fixture_type=cfg["type"],
                body_name=cfg.get("body_name"),
                asset_id=cfg.get("asset_id"),
                object_id=cfg.get("object_id") or cfg.get("door_id"),
                room_id=cfg.get("room_id"),
                pos=(
                    np.array(cfg["position"], dtype=float)
                    if "position" in cfg
                    else None
                ),
                yaw=cfg.get("yaw"),
                body_names=list(cfg.get("body_names", [])),
                joint_names=list(cfg.get("joint_names", [])),
                site_names=list(cfg.get("site_names", [])),
                source=cfg,
            )
    return fixtures


def get_fixture_cfgs(fixtures):
    """Return KitchenArena-style fixture cfgs for fixture records."""
    return [
        {
            "name": name,
            "type": "fixture",
            "model": fixture,
            "fixture_type": fixture.fixture_type,
            "body_name": fixture.body_name,
        }
        for name, fixture in fixtures.items()
    ]


def summarize_fixtures(fixtures):
    counts = {}
    for fixture in fixtures.values():
        counts[fixture.fixture_type] = counts.get(fixture.fixture_type, 0) + 1
    return dict(sorted(counts.items()))


def main():
    parser = argparse.ArgumentParser(description="Inspect ProcTHOR fixture layout")
    parser.add_argument("scene_xml", help="Path to train_N.xml or train_N.robocasa.xml")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    layout, style = extract_procthor_layout_and_style(args.scene_xml)
    fixtures = create_fixtures(layout, style)
    print("scene_xml:", layout["scene_xml"])
    print(
        "rooms/walls/doors/fixtures:",
        *(len(layout[k]) for k in ("rooms", "walls", "doors", "fixtures")),
    )
    print("fixture type counts:", summarize_fixtures(fixtures))
    for i, fixture in enumerate(fixtures.values()):
        if i >= args.limit:
            print(f"... ({len(fixtures) - args.limit} more)")
            break
        pos = None if fixture.pos is None else [round(float(v), 4) for v in fixture.pos]
        print(
            f"{fixture.name:32s} type={fixture.fixture_type:14s} "
            f"body={fixture.body_name} pos={pos} joints={len(fixture.joint_names)}"
        )


if __name__ == "__main__":
    main()
