"""Find a collision-free floor spot to spawn the robot in a MolmoSpaces room.

MolmoSpaces scenes are dense furnished rooms, so a fixed spawn (e.g. the origin)
often lands inside furniture or a wall. Rather than trust the occupancy map's
(undocumented) pixel->world mapping, we work directly on the compiled MuJoCo
model with raycasts:

  * vertical ray down at a candidate (x, y): the first geom hit must be the
    ``floor`` -> the spot has clear headroom (no furniture/counter above it);
  * horizontal rays at torso height: the nearest geom must be > a clearance
    radius away -> the robot won't spawn jammed against a wall/furniture.

The most open candidate (largest min-clearance), biased toward the room centre,
is returned as an ``(x, y, 0.0)`` base position.
"""

import numpy as np
import xml.etree.ElementTree as ET
import mujoco


def _compile(arena):
    xml_str = ET.tostring(arena.root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml_str)


def _flat_patches(pts, patch_radius=0.22, flatness_tol=0.03, min_points=6):
    """Cluster elevated ray hits into flat patches, sorted best-first.

    Returns a list of ``(center_xyz, half_xy)`` for each flat patch
    (deduplicated by centre), ordered by size/flatness.
    """
    cand = []
    for (px, py, pz) in pts:
        d = np.linalg.norm(pts[:, :2] - np.array([px, py]), axis=1)
        near = pts[d <= patch_radius]
        if len(near) < min_points or near[:, 2].std() > flatness_tol:
            continue
        score = len(near) - 10 * near[:, 2].std()
        cx, cy = float(near[:, 0].mean()), float(near[:, 1].mean())
        top_z = float(np.median(near[:, 2]))
        half = np.clip(
            (near[:, :2].max(0) - near[:, :2].min(0)) / 2 - 0.05,
            0.05,
            0.60,
        )
        cand.append((score, np.array([cx, cy, top_z]), half))
    cand.sort(key=lambda c: -c[0])
    # dedup patches whose centres are within patch_radius of a kept one
    kept = []
    for score, center, half in cand:
        if all(np.linalg.norm(center[:2] - k[0][:2]) > patch_radius for k in kept):
            kept.append((center, half))
    return kept


def find_surfaces(model, data, **kwargs):
    """All flat candidate surfaces (best-first) as ``[(center_xyz, half_xy), ...]``."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    xy = data.geom_xpos[:, :2]
    lo, hi = xy.min(0), xy.max(0)
    cast_z = kwargs.get("cast_z", 1.5)
    height_range = kwargs.get("height_range", (0.35, 1.25))
    grid_step = kwargs.get("grid_step", 0.05)
    geomid = np.zeros(1, dtype=np.int32)
    pts = []
    for x in np.arange(lo[0], hi[0], grid_step):
        for y in np.arange(lo[1], hi[1], grid_step):
            dist = mujoco.mj_ray(
                model,
                data,
                np.array([x, y, cast_z]),
                np.array([0.0, 0.0, -1.0]),
                None,
                1,
                -1,
                geomid,
            )
            if dist < 0 or geomid[0] == floor_id:
                continue
            z = cast_z - dist
            if height_range[0] <= z <= height_range[1]:
                pts.append((x, y, z))
    if not pts:
        return []
    return _flat_patches(
        np.array(pts),
        kwargs.get("patch_radius", 0.35),
        kwargs.get("flatness_tol", 0.05),
        kwargs.get("min_points", 4),
    )


def analyze_scene(arena, rng=None):
    """Compile the arena once and return placement info for an env reset.

    Returns dict with:
      * ``base_pos``: free-floor (x, y, 0) spawn for the robot.
      * ``surface``: (center_xyz, half_xy) of a tabletop to place objects on,
        or None if no suitable surface was found.
    """
    rng = rng or np.random.default_rng()
    model = _compile(arena)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    # Couple robot pose + work surface so the task is REACHABLE: stand the robot
    # in clear open floor, then use the nearest surface *big enough* for the
    # objects within arm's reach of that spot, facing it. If none is reachable
    # (dense rooms whose counters are all against the walls behind an island),
    # fall back to standing in front of the best surface, else the open spot +
    # best surface (objects may then be out of reach -- pin an exact pose via the
    # per-scene registry / ``robot_base_pos`` kwarg). ``_spot_ok``'s headroom
    # check keeps the spawn inside the room (the floor is an infinite plane).
    min_half = 0.12  # surface must be wide enough to hold the task objects
    surfaces = find_surfaces(model, data)
    big = [s for s in surfaces if min(s[1]) >= min_half] or surfaces
    base_xy = _free_base_pos(model, data, rng)[:2]

    surface, base_yaw = None, 0.0
    in_reach = [s for s in big if np.linalg.norm(s[0][:2] - base_xy) <= 1.1]
    if in_reach:
        surface = min(in_reach, key=lambda s: np.linalg.norm(s[0][:2] - base_xy))
        t = surface[0][:2]
        base_yaw = float(np.arctan2(t[1] - base_xy[1], t[0] - base_xy[0]))
    else:
        surface = big[0] if big else None
        if surface is not None:
            facing = _base_pose_facing(model, data, surface[0][:2], rng)
            if facing is not None:
                base_xy, base_yaw = facing[0][:2], facing[1]

    return {
        "base_pos": np.array([base_xy[0], base_xy[1], 0.05]),
        "base_yaw": base_yaw,
        "surface": surface,
        "surfaces": surfaces,
        "room_center": _room_center_xy(model, data),
        "room_half_extent": _room_half_extent_xy(model, data),
        "open_pos": _free_base_pos(model, data, rng),
    }


# 8 horizontal directions for clearance probing
_H_DIRS = np.array(
    [[np.cos(a), np.sin(a), 0] for a in np.linspace(0, 2 * np.pi, 8, endpoint=False)]
)


def _room_center_xy(model, data):
    """Room centre from visual (group-1) geom positions."""
    vis = data.geom_xpos[np.asarray(model.geom_group) == 1][:, :2]
    if len(vis) == 0:
        vis = data.geom_xpos[:, :2]
    return (vis.min(0) + vis.max(0)) / 2


def _room_half_extent_xy(model, data):
    """Room half-extent from visual (group-1) geom positions."""
    vis = data.geom_xpos[np.asarray(model.geom_group) == 1][:, :2]
    if len(vis) == 0:
        vis = data.geom_xpos[:, :2]
    return (vis.max(0) - vis.min(0)) / 2


def _spot_ok(model, data, x, y, floor_id, clearance=0.4, headroom=1.45):
    """Is (x, y) a valid standing spot: on the floor, under the roof with clear
    headroom (NOT outside the building and NOT under a counter), and with enough
    radial clearance from walls/furniture?

    The floor geom is an *infinite plane*, so "ray-down hits floor" alone also
    accepts points outside the room — the headroom ray up is what confines the
    spot to inside the walls (a point outside the building has no ceiling above
    it; a point jammed under a counter hits it within ``headroom``).
    """
    gid = np.zeros(1, dtype=np.int32)
    # 1) on the floor
    down = mujoco.mj_ray(
        model, data, np.array([x, y, 1.5]), np.array([0.0, 0.0, -1.0]), None, 1, -1, gid
    )
    if down < 0 or gid[0] != floor_id:
        return False
    # 2) under the roof with clear headroom (excludes outside + under-counter)
    up = mujoco.mj_ray(
        model, data, np.array([x, y, 0.3]), np.array([0.0, 0.0, 1.0]), None, 1, -1, gid
    )
    if up < 0 or up < headroom:
        return False
    # 3) radial clearance
    for d in _H_DIRS:
        dd = mujoco.mj_ray(model, data, np.array([x, y, 0.4]), d, None, 1, -1, gid)
        if 0 <= dd < clearance:
            return False
    return True


def _base_pose_facing(
    model,
    data,
    target_xy,
    rng=None,
    reach=(0.75, 0.85, 0.65, 0.95, 1.05),
    clearance=0.3,
):
    """Deterministic robot pose standing in front of the work surface, facing it.

    Places the robot on the *room-interior* side of the surface (toward the room
    centre) at a reachable distance — so it stands in the open floor facing the
    counter/table, not jammed into the wall corner behind it. Falls back to an
    interior-biased angular scan if the straight-in spot is blocked. Returns
    ((x, y, 0.0), yaw) with yaw pointing the robot's +x at the surface, or None.
    """
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    target_xy = np.asarray(target_xy, dtype=float)
    interior = _room_center_xy(model, data) - target_xy
    n = np.linalg.norm(interior)
    interior = interior / n if n > 1e-6 else np.array([1.0, 0.0])

    def pose_at(x, y):
        yaw = float(np.arctan2(target_xy[1] - y, target_xy[0] - x))
        return np.array([x, y, 0.0]), yaw

    # Scan a dense set of (distance, angle) candidates around the surface and
    # collect every valid standing spot. Dense furnished rooms (kitchen islands,
    # narrow walkways) often have NO spot straight in front, so we search all
    # directions and pick the closest reachable one, breaking ties toward the
    # room interior (open side) -- this avoids both the wall corner and a
    # too-far fallback.
    base_ang = float(np.arctan2(interior[1], interior[0]))
    dists = np.arange(reach[0] if reach else 0.6, 1.45, 0.1)
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    best, best_cost = None, np.inf
    for dist in dists:
        for a in angles:
            x, y = target_xy + dist * np.array([np.cos(a), np.sin(a)])
            if not _spot_ok(model, data, x, y, floor_id, clearance):
                continue
            # cost: closer is better (reachable); prefer the interior side
            interior_bonus = -0.4 * np.cos(a - base_ang)  # small when toward room
            cost = dist + interior_bonus
            if cost < best_cost:
                best_cost, best = cost, (x, y)
    return pose_at(*best) if best is not None else None


def find_free_base_pos(
    arena,
    rng=None,
    clearance=0.45,
    wall_margin=0.7,
    n_candidates=400,
    torso_heights=(0.35, 1.0),
):
    """Return an (x, y, 0.0) base position on free floor of the arena.

    Args:
        arena (MolmoSpacesArena): the built arena (its xml is compiled here).
        rng: numpy Generator for candidate sampling / tie-breaking.
        clearance (float): required radial free space around the spawn (m).
        wall_margin (float): keep candidates this far inside the room bbox (m).
        n_candidates (int): random candidate spawns to evaluate.
        torso_heights (tuple): heights at which to cast horizontal clearance rays.
    """
    rng = rng or np.random.default_rng()
    model = _compile(arena)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return _free_base_pos(
        model, data, rng, clearance, wall_margin, n_candidates, torso_heights
    )


def _free_base_pos(
    model,
    data,
    rng,
    clearance=0.45,
    wall_margin=0.7,
    n_candidates=400,
    torso_heights=(0.35, 1.0),
):
    """Raycast-based free-floor search on an already-compiled model."""
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    # room xy bounds from visual geom positions (group 1 after our remap)
    xy = data.geom_xpos[:, :2]
    lo = xy.min(0) + wall_margin
    hi = xy.max(0) - wall_margin
    if np.any(hi <= lo):  # tiny room; relax margin
        lo = xy.min(0)
        hi = xy.max(0)
    center = (lo + hi) / 2

    dirs = np.array(
        [
            [1, 0, 0],
            [-1, 0, 0],
            [0, 1, 0],
            [0, -1, 0],
            [1, 1, 0],
            [1, -1, 0],
            [-1, 1, 0],
            [-1, -1, 0],
        ],
        dtype=float,
    )
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    cand = rng.uniform(lo, hi, size=(n_candidates, 2))
    # evaluate centre first so an open central spot wins ties
    cand = np.vstack([center, cand])

    geomid = np.zeros(1, dtype=np.int32)
    best_pos, best_score = None, -np.inf
    for (x, y) in cand:
        # must be a valid standing spot: on the floor, under the roof with clear
        # headroom (the floor is an infinite plane, so this is what keeps the
        # spawn INSIDE the room, not outside the walls), and clear of furniture
        if not _spot_ok(model, data, x, y, floor_id, clearance):
            continue

        # radial clearance (for scoring open-ness) at torso heights
        min_clear = np.inf
        for h in torso_heights:
            origin = np.array([x, y, h])
            for d in dirs:
                dd = mujoco.mj_ray(model, data, origin, d, None, 1, -1, geomid)
                if dd >= 0:
                    min_clear = min(min_clear, dd)

        # score: prefer open spots near the room centre
        score = min_clear - 0.15 * np.linalg.norm([x - center[0], y - center[1]])
        if score > best_score:
            best_score, best_pos = score, np.array([x, y, 0.0])

    if best_pos is None:
        # fallback: room centre on the floor
        best_pos = np.array([center[0], center[1], 0.0])
    return best_pos


def get_scene_furniture_pos(sim, scene_name, category):
    """Return the (x, y, z) world position of a furniture piece by semantic category.

    Reads ``FloorPlanN_physics_metadata.json``, finds the first static object
    whose ``category`` matches (case-insensitive), takes the first entry in
    ``name_map.bodies`` as the MuJoCo body name, and returns its world xpos.

    Args:
        sim: MuJoCo MjSim instance (with compiled model + forward kinematics).
        scene_name: ``"FloorPlan301"``, etc.
        category: semantic category string, e.g. ``"Bed"``, ``"Toilet"``,
            ``"Sofa"``, ``"Fridge"``.

    Returns:
        (x, y, z) world position of the matching body.

    Raises:
        FileNotFoundError: the metadata json is missing.
        LookupError: no object with the requested category was found, or the
            body name is not in the MuJoCo model.
    """
    import json
    import os

    from robocasa.models.scenes.molmospaces_scene_registry import get_ithor_scenes_dir

    json_path = os.path.join(
        get_ithor_scenes_dir(), f"{scene_name}_physics_metadata.json"
    )
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"metadata not found: {json_path}")

    with open(json_path) as f:
        data = json.load(f)

    target_body = None
    for _hash, obj in data.get("objects", {}).items():
        if obj.get("category", "").lower() == category.lower():
            bodies = obj.get("name_map", {}).get("bodies", {})
            if bodies:
                # name_map.bodies maps hash_name -> friendly_name.
                # The MuJoCo body name is the hash_name (first key).
                target_body = next(iter(bodies.keys()))
                break

    if target_body is None:
        raise LookupError(f"no furniture with category '{category}' in {scene_name}")

    try:
        body_id = sim.model.body_name2id(target_body)
    except Exception:
        raise LookupError(
            f"body '{target_body}' (category '{category}') not found "
            f"in MuJoCo model for {scene_name}"
        )

    return sim.data.body_xpos[body_id].copy()
