"""ProcTHOR-Objaverse multi-room environment category.

Backed by MolmoSpaces ProcTHOR train scenes under
``assets/scenes/procthor-objaverse-train``. Unlike the iTHOR room categories,
these scenes are full multi-room homes and do not encode a single room type in
their id, so they use automatic surface and robot-pose detection by default.
"""

import os

from robocasa.environments.ithor_base import IThorRoomEnv, make_ithor_registry
from robocasa.models.scenes.procthor_scene_builder import (
    create_fixtures,
    extract_procthor_layout_and_style,
    get_fixture_cfgs,
)
from robocasa.models.scenes.molmospaces_scene_registry import (
    RoomType,
    get_procthor_scenes_dir,
    get_procthor_scene_xml,
    list_procthor_scenes,
    normalize_procthor_scene_name,
)
from robocasa.models.scenes.thor_semantics import build_semantic_registry

REGISTERED_PROCTHOR_ENVS, ProcTHOREnvMeta = make_ithor_registry()


def _room_num(room_id):
    return str(room_id).split("|")[-1]


def _point_in_poly_xy(point, poly):
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            x_intersect = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < x_intersect:
                inside = not inside
        j = i
    return inside


class ProcTHOREnv(IThorRoomEnv, metaclass=ProcTHOREnvMeta):
    """Runnable ProcTHOR multi-room scene (free-roam; tasks subclass this)."""

    ROOM_TYPE = RoomType.MULTI_ROOM
    SCENE_FAMILY = "ProcTHOR-Objaverse train"
    REQUIRE_SCENE_CALIBRATION = False

    def _list_downloaded_scenes(self):
        return list_procthor_scenes()

    def _scene_ids_to_names(self, scene_ids, downloaded):
        ids = [scene_ids] if isinstance(scene_ids, (int, str)) else list(scene_ids)
        names = []
        for scene_id in ids:
            name = normalize_procthor_scene_name(scene_id)
            if name not in downloaded:
                raise ValueError(
                    f"{self.SCENE_ID_ARG} {scene_id!r} resolves to {name!r}, "
                    f"but it is not downloaded. Downloaded ProcTHOR scenes: "
                    f"{sorted(downloaded.keys())}."
                )
            names.append(name)
        return names

    def _get_scene_xml(self, scene_name):
        return get_procthor_scene_xml(scene_name)

    def _setup_baked_scene_fixtures(self):
        layout_config, style_config = extract_procthor_layout_and_style(
            self.mujoco_arena.scene_xml
        )
        self.procthor_layout_config = layout_config
        self.procthor_style_config = style_config
        self.fixtures = create_fixtures(layout_config, style_config, rng=self.rng)
        self.fixture_cfgs = get_fixture_cfgs(self.fixtures)

    def _build_semantic_scene(self, scene_info):
        rooms, edges = self.get_room_layout()
        metadata_path = self.procthor_layout_config["metadata_json"]
        override_path = os.path.join(
            os.path.dirname(get_procthor_scenes_dir()),
            "thor_semantic_overrides.yaml",
        )
        return build_semantic_registry(
            self.mujoco_arena,
            self.scene_name,
            metadata_path,
            surfaces=scene_info.get("surfaces", []),
            rooms=rooms,
            room_edges=edges,
            scene_center=scene_info.get("room_center"),
            override_path=override_path,
        )

    def get_room_layout(self):
        rooms_src = self.procthor_layout_config.get("rooms", [])
        doors_src = self.procthor_layout_config.get("doors", [])

        rooms = {}
        for room in rooms_src:
            rid = _room_num(room.get("room_id", room.get("name", "")))
            poly = [(p[0], p[1]) for p in room["floor_polygon"]]
            rooms[rid] = {
                "type": room.get("room_type", f"room {rid}"),
                "poly": poly,
            }

        edges = set()
        for door in doors_src:
            door_rooms = door.get("rooms") or []
            if len(door_rooms) < 2:
                continue
            r0, r1 = _room_num(door_rooms[0]), _room_num(door_rooms[1])
            if r0 == r1 or r0 not in rooms or r1 not in rooms:
                continue
            if float(door.get("openness", 1.0)) <= 0:
                continue
            edges.add(tuple(sorted((r0, r1))))
        return rooms, sorted(edges)

    def room_of_point(self, point, rooms=None):
        if rooms is None:
            rooms, _ = self.get_room_layout()
        for rid, info in rooms.items():
            if _point_in_poly_xy(point[:2], info["poly"]):
                return rid
        return None

    def spawn_point_matches_rooms(self, spawn, rooms=None):
        if rooms is None:
            rooms, _ = self.get_room_layout()
        source_room = str(spawn["source_room"])
        target_room = str(spawn["target_room"])
        robot_room = self.room_of_point(spawn["robot_pos"], rooms)
        source_surface_room = self.room_of_point(spawn["source_surface_center"], rooms)
        target_surface_room = self.room_of_point(spawn["target_surface_center"], rooms)
        return (
            robot_room == source_room
            and source_surface_room == source_room
            and target_surface_room == target_room
            and source_room != target_room
        )
