"""Audit semantic fixtures and regions extracted from downloaded THOR scenes."""

import argparse
import os

from robocasa.models.arenas.molmospaces_arena import MolmoSpacesArena
from robocasa.models.scenes.molmospaces_scene_registry import (
    RoomType,
    get_ithor_scenes_dir,
    get_procthor_scenes_dir,
    list_procthor_scenes,
    list_scenes,
)
from robocasa.models.scenes.procthor_scene_builder import (
    extract_procthor_layout_and_style,
)
from robocasa.models.scenes.thor_semantics import build_semantic_registry
from robocasa.utils.molmospaces_placement import analyze_scene


def _procthor_rooms(layout):
    rooms = {}
    for room in layout.get("rooms", []):
        room_id = str(room.get("room_id"))
        rooms[room_id] = {
            "type": room.get("room_type"),
            "poly": [(point[0], point[1]) for point in room["floor_polygon"]],
        }
    edges = []
    for door in layout.get("doors", []):
        room_ids = [str(room).split("|")[-1] for room in door.get("rooms", [])]
        if len(room_ids) == 2 and room_ids[0] != room_ids[1]:
            edges.append(tuple(room_ids))
    return rooms, edges


def audit_ithor(limit=None):
    scenes = {}
    for room_type in (
        RoomType.KITCHEN,
        RoomType.LIVING_ROOM,
        RoomType.BEDROOM,
        RoomType.BATHROOM,
    ):
        scenes.update(list_scenes(room_type))
    items = sorted(scenes.items())[:limit]
    root = get_ithor_scenes_dir()
    return _audit_items(
        (
            scene_name,
            scene_xml,
            os.path.join(root, f"{scene_name}_physics_metadata.json"),
            None,
            None,
        )
        for scene_name, scene_xml in items
    )


def audit_procthor(limit=None):
    items = sorted(list_procthor_scenes().items())[:limit]
    specs = []
    for scene_name, scene_xml in items:
        layout, _ = extract_procthor_layout_and_style(scene_xml)
        rooms, edges = _procthor_rooms(layout)
        specs.append((scene_name, scene_xml, layout["metadata_json"], rooms, edges))
    return _audit_items(specs)


def _audit_items(items):
    results = []
    for scene_name, scene_xml, metadata_path, rooms, edges in items:
        arena = MolmoSpacesArena(scene_xml)
        info = analyze_scene(arena)
        registry = build_semantic_registry(
            arena,
            scene_name,
            metadata_path,
            surfaces=info.get("surfaces", []),
            rooms=rooms,
            room_edges=edges,
            scene_center=info.get("room_center"),
            override_path=os.path.join(
                os.path.dirname(
                    get_procthor_scenes_dir()
                    if scene_name.startswith("train_")
                    else get_ithor_scenes_dir()
                ),
                "thor_semantic_overrides.yaml",
            ),
        )
        issues = registry.audit()
        results.append((scene_name, len(registry.fixtures), issues))
        print(
            f"{scene_name}: fixtures={len(registry.fixtures)} " f"issues={len(issues)}"
        )
        for issue in issues:
            print(f"  {issue}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=("all", "ithor", "procthor"), default="all")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    results = []
    if args.family in ("all", "ithor"):
        results.extend(audit_ithor(args.limit))
    if args.family in ("all", "procthor"):
        results.extend(audit_procthor(args.limit))
    issue_count = sum(len(issues) for _, _, issues in results)
    print(f"audited={len(results)} issues={issue_count}")
    raise SystemExit(1 if issue_count else 0)


if __name__ == "__main__":
    main()
