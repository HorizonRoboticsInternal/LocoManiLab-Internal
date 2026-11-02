"""Semantic fixture and region registry for baked iTHOR / ProcTHOR scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import yaml


SUPPORT_CATEGORIES = {
    "bed",
    "coffeetable",
    "countertop",
    "desk",
    "diningtable",
    "dresser",
    "shelf",
    "shelvingunit",
    "sidetable",
    "sofa",
    "table",
    "tvstand",
}
SUPPORT_CATEGORY_PREFIXES = (
    "countermesh",
    "countertop",
    "standardcounterheightwidth",
)
INTERIOR_CATEGORIES = {
    "cabinet",
    "dishwasher",
    "drawer",
    "dresser",
    "fridge",
    "microwave",
    "oven",
}
RECEPTACLE_TOKENS = ("basket", "bin", "bowl", "can", "receptacle", "sink")
SHELF_SITE_SUPPORT_CATEGORIES = {
    "shelf",
    "shelvingunit",
    "sidetable",
    "wallshelf",
}


def normalize_category(value):
    value = "" if value is None else str(value)
    return "".join(ch for ch in value.lower() if ch.isalnum() or ch == "_")


def _is_support_category(category):
    return category in SUPPORT_CATEGORIES or category.startswith(
        SUPPORT_CATEGORY_PREFIXES
    )


def _is_shelf_site_support(category, source_category, site_name, site_alias):
    haystack = " ".join(
        normalize_category(value)
        for value in (category, source_category, site_name, site_alias)
    )
    if "receptacle" not in haystack:
        return False
    if "drawer" in haystack:
        return False
    return category in SHELF_SITE_SUPPORT_CATEGORIES or "shelf" in haystack


def _as_names(value):
    if isinstance(value, dict):
        return list(value.keys())
    return list(value or [])


def _fixture_region_bounds(fixture, name):
    region = fixture._regions[name]
    center = np.array(
        [
            np.mean((region["p0"][0], region["px"][0])),
            np.mean((region["p0"][1], region["py"][1])),
            np.mean((region["p0"][2], region["pz"][2])),
        ],
        dtype=float,
    )
    half_size = (
        np.array(
            [
                region["px"][0] - region["p0"][0],
                region["py"][1] - region["p0"][1],
                region["pz"][2] - region["p0"][2],
            ],
            dtype=float,
        )
        / 2
    )
    return center, half_size


def semantic_fixture_from_robocasa_fixture(
    fixture,
    stable_id,
    category,
    source_category=None,
    asset_id=None,
    region_kinds=None,
):
    main_center, main_half = _fixture_region_bounds(fixture, "main")
    yaw = float(fixture.euler[2])
    semantic_regions = {}
    for region_name, kind in (region_kinds or {}).items():
        center, half_size = _fixture_region_bounds(fixture, region_name)
        region = SemanticRegion(
            stable_id=f"{stable_id}:region:{kind}",
            fixture_id=stable_id,
            kind=kind,
            parent_body=fixture.root_body,
            local_center=center,
            half_size=half_size,
        )
        semantic_regions[region.stable_id] = region

    body_names = [
        body.get("name") for body in fixture.worldbody.iter("body") if body.get("name")
    ]
    return SemanticFixture(
        stable_id=stable_id,
        name=fixture.name,
        category=str(category).lower(),
        source_category=source_category or category,
        room_id=None,
        body_name=fixture.root_body,
        body_names=body_names,
        site_names=[],
        asset_id=asset_id,
        object_id=fixture.name,
        center=np.asarray(fixture.pos, dtype=float) + main_center,
        half_size=main_half,
        yaw=yaw,
        front=np.array([np.cos(yaw), np.sin(yaw)]),
        capabilities={region.kind for region in semantic_regions.values()},
        regions=semantic_regions,
        local_center=main_center,
        local_half_size=main_half,
    )


def _yaw_from_mat(mat):
    mat = np.asarray(mat, dtype=float).reshape(3, 3)
    return float(np.arctan2(mat[1, 0], mat[0, 0]))


def _angle_delta(a, b):
    return float((a - b + np.pi) % (2 * np.pi) - np.pi)


def _local_pose(data, body_id, world_center, world_yaw):
    body_pos = np.asarray(data.xpos[body_id], dtype=float)
    body_mat = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    local_center = body_mat.T @ (np.asarray(world_center) - body_pos)
    local_yaw = _angle_delta(world_yaw, _yaw_from_mat(body_mat))
    return local_center, local_yaw


def _merged_surface_patches(patches):
    def record(center, half):
        center = np.asarray(center, dtype=float)
        half = np.asarray(half, dtype=float)
        return {
            "lo": center[:2] - half[:2],
            "hi": center[:2] + half[:2],
            "z_values": [float(center[2])],
        }

    def area(patch):
        size = np.maximum(patch["hi"] - patch["lo"], 0.0)
        return float(size[0] * size[1])

    def can_merge(a, b):
        z_gap = abs(float(np.median(a["z_values"])) - float(np.median(b["z_values"])))
        if z_gap > 0.05:
            return False
        gap = np.maximum(
            np.maximum(a["lo"], b["lo"]) - np.minimum(a["hi"], b["hi"]),
            0.0,
        )
        if np.any(gap > 0.10):
            return False
        lo = np.minimum(a["lo"], b["lo"])
        hi = np.maximum(a["hi"], b["hi"])
        merged_area = float(np.prod(np.maximum(hi - lo, 0.0)))
        return merged_area <= 1.60 * (area(a) + area(b))

    merged = [record(center, half) for center, half in patches]
    changed = True
    while changed:
        changed = False
        next_patches = []
        used = [False] * len(merged)
        for i, patch in enumerate(merged):
            if used[i]:
                continue
            current = patch
            used[i] = True
            for j in range(i + 1, len(merged)):
                if used[j] or not can_merge(current, merged[j]):
                    continue
                current = {
                    "lo": np.minimum(current["lo"], merged[j]["lo"]),
                    "hi": np.maximum(current["hi"], merged[j]["hi"]),
                    "z_values": [*current["z_values"], *merged[j]["z_values"]],
                }
                used[j] = True
                changed = True
            next_patches.append(current)
        merged = next_patches
    result = []
    for patch in merged:
        center_xy = 0.5 * (patch["lo"] + patch["hi"])
        half_xy = np.maximum(0.5 * (patch["hi"] - patch["lo"]), 0.025)
        center = np.array(
            [center_xy[0], center_xy[1], float(np.median(patch["z_values"]))]
        )
        result.append((center, half_xy))
    return result


def _support_region_from_patch(
    fixture_id,
    index,
    parent_body,
    parent_id,
    data,
    patch_center,
    patch_half,
):
    local_center, local_yaw = _local_pose(data, parent_id, patch_center, 0.0)
    return SemanticRegion(
        stable_id=f"{fixture_id}:region:support:{index}",
        fixture_id=fixture_id,
        kind="support",
        parent_body=parent_body,
        local_center=local_center,
        half_size=np.array([patch_half[0], patch_half[1], 0.02]),
        local_yaw=local_yaw,
        _world_center=patch_center,
        _world_yaw=0.0,
    )


@dataclass
class SemanticJoint:
    stable_id: str
    name: str
    role: str
    range: tuple[float, float]
    closed_qpos: float
    open_qpos: float

    def fraction(self, sim):
        qpos = float(sim.data.get_joint_qpos(self.name))
        denom = self.open_qpos - self.closed_qpos
        if abs(denom) < 1e-8:
            return 0.0
        return float(np.clip((qpos - self.closed_qpos) / denom, 0.0, 1.0))


@dataclass
class SemanticRegion:
    stable_id: str
    fixture_id: str
    kind: str
    parent_body: str | None
    local_center: np.ndarray
    half_size: np.ndarray
    local_yaw: float = 0.0
    _world_center: np.ndarray = field(default_factory=lambda: np.zeros(3))
    _world_yaw: float = 0.0
    _world_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def world_pose(self, sim=None):
        world_offset = np.asarray(self._world_offset, dtype=float)
        if sim is None or self.parent_body is None:
            return (
                np.asarray(self._world_center, dtype=float).copy() + world_offset,
                float(self._world_yaw),
            )
        try:
            body_id = sim.model.body_name2id(self.parent_body)
        except Exception:
            return (
                np.asarray(self._world_center, dtype=float).copy() + world_offset,
                float(self._world_yaw),
            )
        body_pos = np.asarray(sim.data.body_xpos[body_id], dtype=float)
        body_mat = np.asarray(sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
        center = body_pos + body_mat @ self.local_center + world_offset
        return center, _yaw_from_mat(body_mat) + self.local_yaw

    def contains_xy(self, point, sim=None, margin=0.0):
        center, yaw = self.world_pose(sim)
        delta = np.asarray(point, dtype=float)[:2] - center[:2]
        c, s = np.cos(-yaw), np.sin(-yaw)
        local = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
        return bool(np.all(np.abs(local) <= np.maximum(self.half_size[:2] - margin, 0)))


@dataclass
class SemanticFixture:
    stable_id: str
    name: str
    category: str
    source_category: str
    room_id: str | None
    body_name: str | None
    body_names: list[str]
    site_names: list[str]
    asset_id: str | None
    object_id: str | None
    center: np.ndarray
    half_size: np.ndarray
    yaw: float
    front: np.ndarray
    capabilities: set[str] = field(default_factory=set)
    joints: dict[str, SemanticJoint] = field(default_factory=dict)
    regions: dict[str, SemanticRegion] = field(default_factory=dict)
    source: dict = field(default_factory=dict)
    local_center: np.ndarray | None = None
    local_half_size: np.ndarray | None = None

    @property
    def pos(self):
        return self.center

    @property
    def fixture_type(self):
        return self.category

    @property
    def nat_lang(self):
        return self.source_category.replace("_", " ").lower()

    @property
    def joint_names(self):
        return [joint.name for joint in self.joints.values()]

    def body_prefixes(self):
        return tuple(self.body_names or ([self.body_name] if self.body_name else []))

    def world_center(self, sim=None):
        if sim is None or self.body_name is None or self.local_center is None:
            return self.center.copy()
        try:
            body_id = sim.model.body_name2id(self.body_name)
            body_pos = np.asarray(sim.data.body_xpos[body_id], dtype=float)
            body_mat = np.asarray(sim.data.body_xmat[body_id], dtype=float).reshape(
                3, 3
            )
            return body_pos + body_mat @ self.local_center
        except Exception:
            return self.center.copy()

    def radial_extent(self, direction):
        """Conservative distance from fixture center to its XY boundary."""
        direction = np.asarray(direction, dtype=float)[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return float(np.max(self.half_size[:2]))
        direction = direction / norm
        return float(np.dot(np.abs(direction), self.half_size[:2]))

    def long_edge_direction(self, sim=None):
        """Unit XY direction of the fixture's longest local horizontal edge."""
        local_half = (
            self.half_size if self.local_half_size is None else self.local_half_size
        )
        long_axis = int(np.argmax(local_half[:2]))
        if sim is not None and self.body_name is not None:
            try:
                body_id = sim.model.body_name2id(self.body_name)
                body_mat = np.asarray(sim.data.body_xmat[body_id], dtype=float).reshape(
                    3, 3
                )
                direction = body_mat[:2, long_axis]
            except Exception:
                direction = np.array(
                    [
                        np.cos(self.yaw + long_axis * np.pi / 2),
                        np.sin(self.yaw + long_axis * np.pi / 2),
                    ]
                )
        else:
            direction = np.array(
                [
                    np.cos(self.yaw + long_axis * np.pi / 2),
                    np.sin(self.yaw + long_axis * np.pi / 2),
                ]
            )
        norm = np.linalg.norm(direction)
        return direction / norm if norm > 1e-8 else np.array([1.0, 0.0])

    def long_edge_normal(self, sim=None):
        long_edge = self.long_edge_direction(sim)
        return np.array([-long_edge[1], long_edge[0]])

    def get_region(self, kind, required=True):
        matches = [r for r in self.regions.values() if r.kind == kind]
        if matches:
            return max(matches, key=lambda r: float(np.prod(r.half_size[:2])))
        if required:
            raise LookupError(f"{self.name} has no {kind!r} region")
        return None

    def find_joint(self, token=None, fallback=None, required=True, exclude_tokens=()):
        del fallback
        tokens = [] if token is None else ([token] if isinstance(token, str) else token)
        excludes = (
            [exclude_tokens] if isinstance(exclude_tokens, str) else exclude_tokens
        )
        for joint in self.joints.values():
            haystack = f"{joint.name} {joint.role}".lower()
            if all(str(t).lower() in haystack for t in tokens) and not any(
                str(t).lower() in haystack for t in excludes
            ):
                return joint.name
        if token is None and self.joints:
            return next(iter(self.joints.values())).name
        if required:
            raise RuntimeError(
                f"Could not find joint matching {token!r} on {self.name}; "
                f"available: {self.joint_names}"
            )
        return None

    def semantic_joint(self, role=None, required=True):
        if role is not None:
            for joint in self.joints.values():
                if joint.role == role:
                    return joint
        if self.joints:
            return next(iter(self.joints.values()))
        if required:
            raise LookupError(f"{self.name} has no semantic joint")
        return None


class SemanticSceneRegistry:
    """Queryable semantic graph backed by immutable baked scene bodies."""

    def __init__(self, scene_name, fixtures, rooms=None, room_edges=None):
        self.scene_name = scene_name
        self.fixtures = {fixture.name: fixture for fixture in fixtures}
        self._by_id = {fixture.stable_id: fixture for fixture in fixtures}
        self.rooms = rooms or {}
        self.room_edges = set(tuple(sorted(edge)) for edge in (room_edges or []))
        self.sim = None

    def bind(self, sim):
        self.sim = sim
        return self

    def add_fixture(self, fixture):
        self.fixtures[fixture.name] = fixture
        self._by_id[fixture.stable_id] = fixture
        return fixture

    def get_fixture(
        self,
        id=None,
        category=None,
        capabilities=(),
        room_id=None,
        ref=None,
        relation=None,
        return_all=False,
    ):
        if isinstance(id, SemanticFixture):
            return id
        if id is not None:
            if id in self.fixtures:
                return self.fixtures[id]
            if id in self._by_id:
                return self._by_id[id]
        id_token = None
        if id is not None:
            id_token = normalize_category(id)
        categories = set()
        if category is not None:
            values = (
                category if isinstance(category, (list, tuple, set)) else [category]
            )
            categories = {normalize_category(v) for v in values}
        required_caps = set(capabilities or ())
        ref_fixture = self.get_fixture(id=ref) if ref is not None else None
        matches = []
        for fixture in self.fixtures.values():
            if id_token and not any(
                id_token in normalize_category(value)
                for value in (
                    fixture.name,
                    fixture.stable_id,
                    fixture.category,
                    fixture.source_category,
                    fixture.object_id or "",
                )
            ):
                continue
            if categories and fixture.category not in categories:
                continue
            if required_caps and not required_caps.issubset(fixture.capabilities):
                continue
            if room_id is not None and str(fixture.room_id) != str(room_id):
                continue
            if ref_fixture is fixture:
                continue
            if ref_fixture is not None and relation in {"near", "above", "hosts"}:
                delta = fixture.center - ref_fixture.center
                if relation == "near" and np.linalg.norm(delta[:2]) > 1.25:
                    continue
                if relation == "above" and not (
                    delta[2] > 0.05 and np.linalg.norm(delta[:2]) < 0.8
                ):
                    continue
                if relation == "hosts" and not (
                    np.all(np.abs(delta[:2]) <= fixture.half_size[:2] + 0.15)
                ):
                    continue
            matches.append(fixture)
        matches.sort(key=lambda f: (f.room_id or "", f.name))
        if return_all:
            return matches
        return matches[0] if matches else None

    def get_region(
        self,
        fixture=None,
        kind=None,
        min_size=None,
        ref=None,
        relation=None,
        return_all=False,
    ):
        candidates = []
        fixtures = (
            [self.get_fixture(id=fixture)]
            if fixture is not None
            else list(self.fixtures.values())
        )
        ref_fixture = self.get_fixture(id=ref) if ref is not None else None
        for item in fixtures:
            if item is None:
                continue
            for region in item.regions.values():
                if kind is not None and region.kind != kind:
                    continue
                if min_size is not None and np.any(
                    2 * region.half_size[:2] < np.asarray(min_size)[:2]
                ):
                    continue
                if ref_fixture is not None and relation in {"near", "above"}:
                    center, _ = region.world_pose(self.sim)
                    delta = center - ref_fixture.world_center(self.sim)
                    footprint_near = region.contains_xy(
                        ref_fixture.world_center(self.sim), self.sim, margin=-0.20
                    )
                    if relation == "near" and not (
                        footprint_near or np.linalg.norm(delta[:2]) <= 1.25
                    ):
                        continue
                    if relation == "above" and not (
                        delta[2] > 0.05
                        and (footprint_near or np.linalg.norm(delta[:2]) < 0.8)
                    ):
                        continue
                candidates.append(region)
        candidates.sort(key=lambda r: r.stable_id)
        if return_all:
            return candidates
        return candidates[0] if candidates else None

    def adjacent_rooms(self, room_id):
        room_id = str(room_id)
        return [
            b if a == room_id else a for a, b in self.room_edges if room_id in (a, b)
        ]

    @staticmethod
    def point_in_room(point, polygon):
        return _point_in_poly(point, polygon)

    def audit(self):
        issues = []
        if len(self._by_id) != len(self.fixtures):
            issues.append("duplicate fixture stable_id")
        for fixture in self.fixtures.values():
            if not fixture.body_names:
                issues.append(f"{fixture.stable_id}: no valid body")
            for joint in fixture.joints.values():
                if abs(joint.range[1] - joint.range[0]) <= 1e-8:
                    issues.append(f"{joint.stable_id}: zero joint range")
            for region in fixture.regions.values():
                if np.any(region.half_size <= 0):
                    issues.append(f"{region.stable_id}: invalid region size")
        return issues


def _geom_bounds(model, data, body_ids):
    corners = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) not in body_ids:
            continue
        aabb = np.asarray(model.geom_aabb[geom_id], dtype=float)
        center_local, half = aabb[:3], np.maximum(aabb[3:], 1e-4)
        rot = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        center = np.asarray(data.geom_xpos[geom_id], dtype=float) + rot @ center_local
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    corners.append(center + rot @ (half * [sx, sy, sz]))
    if not corners:
        positions = [np.asarray(data.xpos[bid], dtype=float) for bid in body_ids]
        if not positions:
            return np.zeros(3), np.full(3, 0.05)
        corners = positions
    corners = np.asarray(corners)
    lo, hi = corners.min(axis=0), corners.max(axis=0)
    return (lo + hi) / 2, np.maximum((hi - lo) / 2, 0.02)


def _geom_half_size_in_body_frame(model, data, body_ids, root_id):
    root_pos = np.asarray(data.xpos[root_id], dtype=float)
    root_mat = np.asarray(data.xmat[root_id], dtype=float).reshape(3, 3)
    corners = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) not in body_ids:
            continue
        aabb = np.asarray(model.geom_aabb[geom_id], dtype=float)
        geom_center, half = aabb[:3], np.maximum(aabb[3:], 1e-4)
        geom_mat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        world_center = np.asarray(data.geom_xpos[geom_id], dtype=float)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    world = world_center + geom_mat @ (
                        geom_center + half * [sx, sy, sz]
                    )
                    corners.append(root_mat.T @ (world - root_pos))
    if not corners:
        return np.full(3, 0.05)
    corners = np.asarray(corners)
    return np.maximum((corners.max(axis=0) - corners.min(axis=0)) / 2, 0.02)


def _joint_role(category, name, alias="", body_name="", body_alias=""):
    text = f"{category} {name} {alias} {body_name} {body_alias}".lower()
    if "handle" in text:
        return "handle"
    if "drawer" in text or category == "drawer":
        return "drawer"
    if "door" in text or category in {"fridge", "cabinet", "oven", "dishwasher"}:
        return "door"
    return "articulation"


def _load_overrides(path, scene_name):
    if path is None or not os.path.isfile(path):
        return {}
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return data.get("scenes", {}).get(scene_name, {})


def _point_in_poly(point, poly):
    x, y = float(point[0]), float(point[1])
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            cross = (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
            if x < cross:
                inside = not inside
        j = i
    return inside


def build_semantic_registry(
    arena,
    scene_name,
    metadata_path,
    surfaces=(),
    rooms=None,
    room_edges=None,
    scene_center=None,
    override_path=None,
):
    """Build semantic handles from THOR metadata and compiled MJCF geometry."""
    with open(metadata_path) as f:
        metadata = json.load(f)
    from robocasa.models.scenes.ithor_scene_defaults import apply_default_scene_metadata

    apply_default_scene_metadata(metadata, scene_name)
    model = mujoco.MjModel.from_xml_string(ET.tostring(arena.root, encoding="unicode"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    overrides = _load_overrides(override_path, scene_name)
    fixture_overrides = overrides.get("fixtures", {})
    dynamic_body_names = {
        body_name
        for obj in metadata.get("objects", {}).values()
        if not obj.get("is_static", False)
        for body_name in _as_names(obj.get("name_map", {}).get("bodies"))
    }
    ray_geom_id = np.zeros(1, dtype=np.int32)

    def supporting_body_id(point):
        distance = mujoco.mj_ray(
            model,
            data,
            np.array([point[0], point[1], 1.5]),
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            -1,
            ray_geom_id,
        )
        if distance < 0 or ray_geom_id[0] < 0:
            return None
        return int(model.geom_bodyid[ray_geom_id[0]])

    room_centers = {}
    for room_id, room in (rooms or {}).items():
        poly = np.asarray(room.get("poly", []), dtype=float)
        if len(poly):
            room_centers[str(room_id)] = poly.mean(axis=0)
    scene_center = np.asarray(
        np.zeros(2) if scene_center is None else scene_center, dtype=float
    )[:2]

    fixtures = []
    semantic_body_ids = set()
    for hash_name, obj in metadata.get("objects", {}).items():
        object_id = obj.get("object_id") or hash_name
        override = fixture_overrides.get(
            object_id, fixture_overrides.get(hash_name, {})
        )
        if override.get("disabled", False):
            continue
        source_category = override.get("category", obj.get("category", "object"))
        category = normalize_category(source_category)
        name_map = obj.get("name_map", {})
        body_names = _as_names(name_map.get("bodies"))
        body_ids = []
        valid_body_names = []
        for body_name in body_names:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                body_ids.append(int(body_id))
                valid_body_names.append(body_name)
        if not body_ids:
            continue
        semantic_body_ids.update(body_ids)
        root_body = valid_body_names[0]
        root_id = body_ids[0]
        center, half_size = _geom_bounds(model, data, set(body_ids))
        body_yaw = _yaw_from_mat(data.xmat[root_id])
        room_id = str(obj.get("room_id")) if obj.get("room_id") is not None else None
        room_center = room_centers.get(room_id)
        if room_center is None:
            room_center = scene_center
        front = np.asarray(override.get("front", room_center - center[:2]), dtype=float)
        if np.linalg.norm(front) < 1e-6:
            front = np.array([np.cos(body_yaw), np.sin(body_yaw)])
        front = front[:2] / np.linalg.norm(front[:2])
        stable_id = f"{scene_name}:{object_id}"

        aliases = name_map.get("joints", {})
        body_aliases = name_map.get("bodies", {})
        joints = {}
        for joint_name in _as_names(aliases):
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0 or int(model.jnt_type[joint_id]) == int(
                mujoco.mjtJoint.mjJNT_FREE
            ):
                continue
            lo, hi = (float(v) for v in model.jnt_range[joint_id])
            if abs(hi - lo) <= 1e-8:
                continue
            alias = aliases.get(joint_name, "") if isinstance(aliases, dict) else ""
            body_id = int(model.jnt_bodyid[joint_id])
            body_name = (
                mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            )
            body_alias = (
                body_aliases.get(body_name, "")
                if isinstance(body_aliases, dict)
                else ""
            )
            role = (
                override.get("joints", {})
                .get(joint_name, {})
                .get(
                    "role",
                    _joint_role(category, joint_name, alias, body_name, body_alias),
                )
            )
            if role == "handle":
                continue
            closed = float(
                override.get("joints", {}).get(joint_name, {}).get("closed_qpos", 0.0)
            )
            open_qpos = hi if abs(hi - closed) >= abs(lo - closed) else lo
            joints[role if role not in joints else joint_name] = SemanticJoint(
                stable_id=f"{stable_id}:joint:{role}",
                name=joint_name,
                role=role,
                range=(lo, hi),
                closed_qpos=closed,
                open_qpos=float(open_qpos),
            )

        capabilities = set()
        if joints:
            capabilities.update(("articulated", "openable"))
        if _is_support_category(category):
            capabilities.add("support")
        if category in INTERIOR_CATEGORIES:
            capabilities.add("interior")
        if any(token in category for token in RECEPTACLE_TOKENS):
            capabilities.add("receptacle")

        site_aliases = name_map.get("sites") if isinstance(name_map, dict) else {}
        if not isinstance(site_aliases, dict):
            site_aliases = {}

        fixture = SemanticFixture(
            stable_id=stable_id,
            name=hash_name,
            category=category,
            source_category=source_category,
            room_id=room_id,
            body_name=root_body,
            body_names=valid_body_names,
            site_names=_as_names(name_map.get("sites")),
            asset_id=obj.get("asset_id"),
            object_id=object_id,
            center=center,
            half_size=half_size,
            yaw=body_yaw,
            front=front,
            capabilities=capabilities,
            joints=joints,
            source=obj,
            local_center=_local_pose(data, root_id, center, body_yaw)[0],
            local_half_size=_geom_half_size_in_body_frame(
                model, data, set(body_ids), root_id
            ),
        )

        regions = {}
        for site_name in fixture.site_names:
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id < 0:
                continue
            site_center = np.asarray(data.site_xpos[site_id], dtype=float)
            site_mat = np.asarray(data.site_xmat[site_id], dtype=float).reshape(3, 3)
            site_size = np.maximum(
                np.asarray(model.site_size[site_id], dtype=float), 0.01
            )
            projected = []
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        projected.append(site_mat @ (site_size * [sx, sy, sz]))
            projected = np.asarray(projected)
            site_half = np.maximum(
                (projected.max(axis=0) - projected.min(axis=0)) / 2, 0.025
            )
            parent_id = int(model.site_bodyid[site_id])
            parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id)
            local_center, local_yaw = _local_pose(data, parent_id, site_center, 0.0)
            site_lower = site_name.lower()
            site_alias = str(site_aliases.get(site_name, "")).lower()
            kinds = []
            if "receptacle" in site_lower:
                kinds.append("receptacle")
            if _is_shelf_site_support(
                category,
                source_category,
                site_lower,
                site_alias,
            ):
                kinds.append("support")
            if not kinds and category in INTERIOR_CATEGORIES:
                kinds.append("interior")
            if not kinds:
                continue
            for kind in kinds:
                region_center = site_center
                region_half = site_half
                region_local_center = local_center
                region_local_yaw = local_yaw
                if kind == "support":
                    region_center = site_center.copy()
                    region_center[2] -= site_half[2]
                    region_half = np.array([site_half[0], site_half[1], 0.02])
                    region_local_center, region_local_yaw = _local_pose(
                        data, parent_id, region_center, 0.0
                    )
                region = SemanticRegion(
                    stable_id=f"{stable_id}:region:{kind}:{len(regions)}",
                    fixture_id=stable_id,
                    kind=kind,
                    parent_body=parent_name,
                    local_center=region_local_center,
                    half_size=region_half,
                    local_yaw=region_local_yaw,
                    _world_center=region_center,
                    _world_yaw=0.0,
                )
                regions[region.stable_id] = region
                fixture.capabilities.add(kind)

        if _is_support_category(category):
            support_patches = []
            for raw_center, raw_half in surfaces or ():
                patch_center = np.asarray(raw_center, dtype=float)
                patch_half = np.asarray(raw_half, dtype=float)
                support_body_id = supporting_body_id(patch_center)
                if support_body_id not in body_ids:
                    continue
                if not np.all(
                    np.abs(patch_center[:2] - center[:2]) <= half_size[:2] + 0.15
                ):
                    continue
                if not (
                    center[2] <= patch_center[2] <= center[2] + half_size[2] + 0.25
                ):
                    continue
                support_patches.append((patch_center, patch_half))

            for patch_center, patch_half in _merged_surface_patches(support_patches):
                region = _support_region_from_patch(
                    stable_id,
                    len(regions),
                    root_body,
                    root_id,
                    data,
                    patch_center,
                    patch_half,
                )
                regions[region.stable_id] = region
                fixture.capabilities.add("support")

        interaction_center = center.copy()
        interaction_center[:2] += front * (fixture.radial_extent(front) + 0.75)
        interaction_center[2] = 0.0
        local_center, local_yaw = _local_pose(
            data, root_id, interaction_center, np.arctan2(-front[1], -front[0])
        )
        interaction = SemanticRegion(
            stable_id=f"{stable_id}:region:interaction",
            fixture_id=stable_id,
            kind="interaction",
            parent_body=root_body,
            local_center=local_center,
            half_size=np.array([0.35, 0.25, 0.05]),
            local_yaw=local_yaw,
            _world_center=interaction_center,
            _world_yaw=np.arctan2(-front[1], -front[0]),
        )
        regions[interaction.stable_id] = interaction

        # Geometry fallback for articulated interiors whose exporter omitted sites.
        if "interior" in fixture.capabilities and not any(
            r.kind in {"interior", "receptacle"} for r in regions.values()
        ):
            interior_center = center.copy()
            local_center, local_yaw = _local_pose(
                data, root_id, interior_center, body_yaw
            )
            interior = SemanticRegion(
                stable_id=f"{stable_id}:region:interior:fallback",
                fixture_id=stable_id,
                kind="interior",
                parent_body=root_body,
                local_center=local_center,
                half_size=np.maximum(half_size * np.array([0.55, 0.55, 0.35]), 0.04),
                local_yaw=local_yaw,
                _world_center=interior_center,
                _world_yaw=body_yaw,
            )
            regions[interior.stable_id] = interior

        fixture.regions = regions
        fixtures.append(fixture)

    # Raycast surfaces often belong to anonymous counter meshes rather than a
    # metadata object categorized as CounterTop. Group patches by the actual
    # supporting body and expose each body as a synthetic semantic fixture.
    surface_groups = {}
    for patch_center, patch_half in surfaces or ():
        patch_center = np.asarray(patch_center, dtype=float)
        body_id = supporting_body_id(patch_center)
        if body_id is None:
            continue
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        if (
            not body_name
            or body_name in dynamic_body_names
            or body_id in semantic_body_ids
        ):
            continue
        group = surface_groups.setdefault(
            body_name, {"body_id": body_id, "patches": []}
        )
        group["patches"].append(
            (patch_center.copy(), np.asarray(patch_half, dtype=float).copy())
        )

    existing_ids = {fixture.stable_id for fixture in fixtures}
    for index, (body_name, group) in enumerate(sorted(surface_groups.items())):
        merged_patches = _merged_surface_patches(group["patches"])
        patch_los = [center[:2] - half[:2] for center, half in merged_patches]
        patch_his = [center[:2] + half[:2] for center, half in merged_patches]
        lo = np.min(patch_los, axis=0)
        hi = np.max(patch_his, axis=0)
        center_xy = (lo + hi) / 2
        half_xy = np.maximum((hi - lo) / 2, 0.025)
        center = np.array(
            [
                center_xy[0],
                center_xy[1],
                np.median([patch[0][2] for patch in merged_patches]),
            ]
        )
        body_id = group["body_id"]
        stable_id = f"{scene_name}:surface:{body_name}"
        if stable_id in existing_ids:
            stable_id = f"{stable_id}:{index}"
        nearest = min(
            fixtures,
            key=lambda fixture: np.linalg.norm(fixture.center[:2] - center[:2]),
            default=None,
        )
        room_id = None
        for candidate_room_id, room in (rooms or {}).items():
            if _point_in_poly(center[:2], room.get("poly", [])):
                room_id = str(candidate_room_id)
                break
        if room_id is None:
            room_id = nearest.room_id if nearest is not None else None
        room_center = room_centers.get(room_id, scene_center)
        front = room_center - center[:2]
        if np.linalg.norm(front) < 1e-6:
            front = -center[:2]
        if np.linalg.norm(front) < 1e-6:
            front = np.array([1.0, 0.0])
        front = front / np.linalg.norm(front)
        regions = {}
        for patch_index, (patch_center, patch_half) in enumerate(merged_patches):
            region = _support_region_from_patch(
                stable_id,
                patch_index,
                body_name,
                body_id,
                data,
                patch_center,
                patch_half,
            )
            regions[region.stable_id] = region
        interaction_center = center.copy()
        boundary_distance = float(np.dot(np.abs(front), half_xy))
        interaction_center[:2] += front * (boundary_distance + 0.75)
        interaction_center[2] = 0.0
        i_local, i_yaw = _local_pose(
            data,
            body_id,
            interaction_center,
            np.arctan2(-front[1], -front[0]),
        )
        interaction = SemanticRegion(
            stable_id=f"{stable_id}:region:interaction",
            fixture_id=stable_id,
            kind="interaction",
            parent_body=body_name,
            local_center=i_local,
            half_size=np.array([0.35, 0.25, 0.05]),
            local_yaw=i_yaw,
            _world_center=interaction_center,
            _world_yaw=np.arctan2(-front[1], -front[0]),
        )
        regions[interaction.stable_id] = interaction
        fixtures.append(
            SemanticFixture(
                stable_id=stable_id,
                name=f"semantic_surface_{index}",
                category="surface",
                source_category="Surface",
                room_id=room_id,
                body_name=body_name,
                body_names=[body_name],
                site_names=[],
                asset_id=None,
                object_id=body_name,
                center=center,
                half_size=np.array([half_xy[0], half_xy[1], 0.05]),
                yaw=0.0,
                front=front,
                capabilities={"support"},
                regions=regions,
                local_center=_local_pose(data, body_id, center, 0.0)[0],
                local_half_size=_geom_half_size_in_body_frame(
                    model, data, {body_id}, body_id
                ),
            )
        )

    return SemanticSceneRegistry(
        scene_name, fixtures, rooms=rooms, room_edges=room_edges
    )


__all__ = [
    "SemanticFixture",
    "SemanticJoint",
    "SemanticRegion",
    "SemanticSceneRegistry",
    "build_semantic_registry",
    "normalize_category",
    "semantic_fixture_from_robocasa_fixture",
]
