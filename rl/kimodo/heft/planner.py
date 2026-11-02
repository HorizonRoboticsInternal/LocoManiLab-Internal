"""A path for the body, because Kimodo cannot see the room.

Kimodo receives a prompt, an end-effector goal and a root goal. It receives NO
scene geometry, so it does not avoid furniture -- it walks through it. Measured:
in ``IThorBedroomLiftTeddyBear`` a hip stayed against a wall for 205 ticks and the
physics filter rejected the episode at 47 degrees of tilt. That failure says
nothing about the generator's quality; it is a question the generator was never
asked.

⚠️ **This planner cannot hand Kimodo a trajectory, and that shapes the whole
design.** The only spatial instruction Kimodo accepts is where the ROOT should be
at the last frame of a chunk. So a plan is not executed by the planner -- it is
CHOPPED into hops short enough for one chunk, and each hop becomes one chunk's
root goal. That makes this a generalisation of the two-stage approach already in
``episode.py``, not a new mechanism beside it.

Three things this deliberately does not do:

  * **It does not plan for the arms.** The occupancy grid is a floor plan; a hand
    passing over a table is not a collision it can see. Arm collisions belong to
    the physics filter, which already catches them.
  * **It does not use the robot's true footprint.** It inflates obstacles by a
    radius instead, which is the standard trade and is conservative in the
    direction that matters -- it refuses tight gaps rather than clipping them.
  * **It does not run per tick.** One plan per waypoint, recomputed only when the
    robot ends up somewhere the plan did not predict.
"""

from __future__ import annotations

import heapq

import numpy as np

#: Metres per grid cell. 0.10 m resolves a chair leg; finer mostly buys memory.
CELL = 0.10

#: How far to inflate obstacles, metres. The G1's standing footprint is roughly
#: 0.25 m across at the feet but the swinging arms and the torso are wider, and a
#: walking gait needs lateral room -- 0.35 is measured off the widest body geom
#: rather than the feet, because it is the shoulders that hit door frames.
ROBOT_RADIUS = 0.35

#: Longest hop handed to one chunk. A 3 s chunk was measured to cover about
#: 1.5 m, so 1.2 leaves room to arrive rather than to still be walking.
HOP = 1.2

#: Longest leg a chunk is allowed to be handed, metres. A 3 s chunk was measured
#: to cover about 1.5 m, and delivery is 66% of the ask above 1.2 m against
#: 33-40% below it -- so a leg slightly over `HOP` is better walked in one piece
#: than split into two weak ones. `HOP` remains the target spacing; this is the
#: bar a leg has to exceed before it is cut at all.
MAX_HOP = 1.5

#: Geoms whose names start with these are floor, not obstacle.
GROUND_PREFIXES = ("floor", "ground", "wall_floor")


def _obstacle_geoms(model, data, z_lo: float, z_hi: float):
    """Geom indices that block the body between two heights.

    ⚠️ Filtered by GEOM name, never by body name. robocasa attaches floor and
    wall geoms to the worldbody, so every one of them reports a body called
    ``world`` -- a body-name filter passes the floor through as an obstacle and
    the whole room becomes impassable. That exact confusion already cost this
    project a misdiagnosis once, in ``_scene_contacts``.
    """
    import mujoco

    out = []
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if name.startswith(GROUND_PREFIXES):
            continue
        if name.startswith("robot0_") or name.startswith("gripper0_"):
            continue
        # ⚠️ Only geoms that actually COLLIDE. robocasa carries visual meshes and
        # placement-region markers with contype=conaffinity=0, and they are often
        # LARGER than the collision body: the sofa's visual geom reaches 0.90 m
        # from its centre while the colliding one reaches 0.57. Treating markers
        # as obstacles walled off space nothing occupies, and the plan stopped
        # 1.25 m from a sofa the robot demonstrably stands 0.63 m from.
        if model.geom_contype[g] == 0 and model.geom_conaffinity[g] == 0:
            continue
        # A geom the body can walk over or duck under is not an obstacle.
        R = np.abs(np.asarray(data.geom_xmat[g]).reshape(3, 3))
        half_z = float((R @ np.asarray(model.geom_size[g], dtype=np.float64))[2])
        lo = float(data.geom_xpos[g][2]) - half_z
        hi = float(data.geom_xpos[g][2]) + half_z
        if hi < z_lo or lo > z_hi:
            continue
        out.append(g)
    return out


class Occupancy:
    """A 2-D floor plan of what the body cannot walk through."""

    def __init__(self, model, data, *, cell: float = CELL,
                 radius: float = ROBOT_RADIUS, z_lo: float = 0.15,
                 z_hi: float = 1.60, margin: float = 1.0):
        self.cell = float(cell)
        gs = _obstacle_geoms(model, data, z_lo, z_hi)
        if not gs:
            raise SystemExit("[planner] no obstacle geoms found -- the name "
                             "filter is wrong for this scene, and an empty grid "
                             "would plan straight lines through walls")
        # ⚠️ Each obstacle's footprint is its ROTATED extent projected onto the
        # floor, not a disc of its largest dimension. Discs were the first
        # attempt and they are catastrophic for walls: a wall is a box of size
        # (3.0, 0.05, 0.75), so max(size) makes it a disc of radius 3 m, four of
        # those fill the room, and the grid came out 53% occupied with no path
        # anywhere. The failure looked like "this scene is impassable".
        boxes = [self._footprint(model, data, g) for g in gs]
        lo = np.min([b[0] for b in boxes], axis=0) - margin
        hi = np.max([b[1] for b in boxes], axis=0) + margin
        self.origin = lo
        self.shape = tuple(np.maximum(
            2, np.ceil((hi - lo) / self.cell).astype(int) + 1))
        self.grid = np.zeros(self.shape, dtype=bool)

        # Rasterise each obstacle as a disc of its own extent, then inflate by
        # the robot radius. Discs rather than the true geom shape: a box rotated
        # 45 degrees would otherwise leak through its own corners, and being
        # conservative here refuses a gap rather than clipping a wall.
        rr = np.arange(self.shape[0]) * self.cell + self.origin[0]
        cc = np.arange(self.shape[1]) * self.cell + self.origin[1]
        X, Y = np.meshgrid(rr, cc, indexing="ij")
        for blo, bhi in boxes:
            self.grid |= ((X >= blo[0] - radius) & (X <= bhi[0] + radius) &
                          (Y >= blo[1] - radius) & (Y <= bhi[1] + radius))

    @staticmethod
    def _footprint(model, data, g):
        """(lo_xy, hi_xy) of a geom's floor footprint, rotation included.

        The half-extent along each world axis of a rotated box is
        ``|R| @ size`` -- the absolute-valued rotation matrix times the half
        sizes. That is exact for boxes and a safe over-estimate for everything
        else, which is the right direction: it refuses a gap rather than
        clipping a wall.
        """
        R = np.abs(np.asarray(data.geom_xmat[g]).reshape(3, 3))
        half = R @ np.asarray(model.geom_size[g], dtype=np.float64)
        p = np.asarray(data.geom_xpos[g], dtype=np.float64)
        return p[:2] - half[:2], p[:2] + half[:2]

    def to_cell(self, xy) -> tuple[int, int]:
        c = np.round((np.asarray(xy, dtype=np.float64)[:2] - self.origin) / self.cell)
        return (int(np.clip(c[0], 0, self.shape[0] - 1)),
                int(np.clip(c[1], 0, self.shape[1] - 1)))

    def to_world(self, cell) -> np.ndarray:
        return self.origin + np.asarray(cell, dtype=np.float64) * self.cell

    def free(self, cell) -> bool:
        return not self.grid[cell[0], cell[1]]

    def occupancy_fraction(self) -> float:
        return float(self.grid.mean())


def clearance_at(model, data, xy, *, z_lo: float = 0.15, z_hi: float = 1.60):
    """``(distance, geom_name)`` from a floor point to the nearest obstacle.

    Distance to the obstacle's FOOTPRINT, so it is negative when the point is
    inside one. No grid is built -- this answers "how much room is there here",
    which is a different question from "can a path be found".

    ⚠️ **It does NOT discriminate a scene that works from one that does not**,
    and that is worth writing down because it looked like it would. Spawn
    clearance, measured over three seeds each:

        SquatPickPlaceTVStand    0.100   (known birth clipping)
        StackCanOnPlate          0.250
        IThorBedroomLiftTeddyBear 0.272-0.696
        ApproachTable            0.302   <- succeeds 5/5
        TabletopPickPlace        0.325
        IThorLivingRoomLiftStatue 0.333  <- succeeds 0/5
        ApproachSofa             1.150

    ``IThorLivingRoomLiftStatue`` has MORE room at its spawn than the task that
    succeeds every time, so "the robot starts inside the inflated obstacle" does
    not explain why it ends up on the coffee table. Every scene here spawns the
    robot within the 0.35 m inflation radius; that is normal, not a fault.
    """
    import mujoco

    p = np.asarray(xy, dtype=np.float64)[:2]
    best, name = float("inf"), ""
    for g in _obstacle_geoms(model, data, z_lo, z_hi):
        lo, hi = Occupancy._footprint(model, data, g)
        d = np.maximum(lo - p, p - hi)
        # Standard point-to-box distance: outside, the positive components give
        # the offset; inside, every component is negative and the largest is the
        # signed distance to the nearest face.
        dist = (float(np.linalg.norm(np.maximum(d, 0.0))) if (d > 0).any()
                else float(d.max()))
        if dist < best:
            best = dist
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    return best, name


def astar(occ: Occupancy, start_xy, goal_xy, *, allow_blocked_goal: bool = True,
          arrive_radius: float | None = None):
    """Grid path from start to goal, or None. World coordinates in and out.

    ``allow_blocked_goal`` matters more than it sounds: the goal is usually a
    piece of FURNITURE, so its own cell is occupied by construction, and a
    planner that refuses blocked goals would refuse every approach task. The
    search instead settles for the reachable cell nearest the goal.
    """
    s, g = occ.to_cell(start_xy), occ.to_cell(goal_xy)
    # ⚠️ The goal is FURNITURE, so its own cell is occupied by construction and
    # so is everything within the inflation radius of it. Planning to the centre
    # therefore stops as far short as the inflation is wide -- measured 1.25 m on
    # ApproachSofa, against a success radius of 1.0, i.e. a plan that is correct
    # and still fails the task. The target is a REGION: the free cell nearest the
    # goal, which is where a body can actually stand.
    if not occ.free(g):
        g = _nearest_free(occ, g)
        if g is None:
            return None
        # ⚠️ Nearest-free is not good enough on its own. The free region starts
        # one inflation radius out, so on ApproachSofa the plan ended 1.25 m from
        # the sofa while the task scores at 1.0 m -- geometrically correct and a
        # task failure. When the caller knows the radius that scores, prefer a
        # free cell INSIDE it, and only fall back to nearest-free when the whole
        # scoring region is blocked.
        if arrive_radius is not None:
            inside = _free_within(occ, occ.to_cell(goal_xy), arrive_radius)
            if inside is not None:
                g = inside
    # ⚠️ And the start can be blocked too -- the robot spawns within an arm's
    # length of a table in most scenes, and the inflated table swallows it. The
    # search cannot expand out of a blocked cell, which returned a one-point
    # "path" that every check in h15 passed trivially. Step it out first.
    if not occ.free(s):
        s2 = _nearest_free(occ, s)
        if s2 is not None:
            s = s2
    if s == g:
        return [np.asarray(start_xy, dtype=np.float64)[:2], occ.to_world(g)]

    H, W = occ.shape
    def h(c):
        return np.hypot(c[0] - g[0], c[1] - g[1])

    open_ = [(h(s), 0.0, s)]
    came, cost = {s: None}, {s: 0.0}
    best, best_h = s, h(s)
    steps = 0
    while open_:
        steps += 1
        if steps > 400_000:
            break
        _, cg, cur = heapq.heappop(open_)
        if cur == g:
            best = cur
            break
        if h(cur) < best_h and occ.free(cur):
            best, best_h = cur, h(cur)
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nxt = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nxt[0] < H and 0 <= nxt[1] < W):
                    continue
                if not occ.free(nxt) and nxt != g:
                    continue
                nc = cg + np.hypot(dx, dy)
                if nc < cost.get(nxt, np.inf):
                    cost[nxt] = nc
                    came[nxt] = cur
                    heapq.heappush(open_, (nc + h(nxt), nc, nxt))
    if best_h > h(s) or (best == s and best != g):
        if not allow_blocked_goal:
            return None
    node, path = best, []
    while node is not None:
        path.append(occ.to_world(node))
        node = came.get(node)
    return path[::-1]


def _free_within(occ: Occupancy, centre, radius: float):
    """A free cell inside ``radius`` of ``centre``, nearest the centre, or None.

    ⚠️ Nearest to the CENTRE, not to the caller -- the task measures the robot's
    distance to the fixture, so the cell that scores best is the innermost free
    one, and picking the nearest-to-the-robot instead would stop at the far edge
    of the scoring ring and pass by a margin of centimetres.
    """
    r_cells = int(np.ceil(radius / occ.cell))
    H, W = occ.shape
    best, best_d = None, None
    for dx in range(-r_cells, r_cells + 1):
        for dy in range(-r_cells, r_cells + 1):
            d2 = dx * dx + dy * dy
            if d2 > r_cells * r_cells:
                continue
            c = (centre[0] + dx, centre[1] + dy)
            if not (0 <= c[0] < H and 0 <= c[1] < W):
                continue
            if occ.grid[c[0], c[1]]:
                continue
            if best_d is None or d2 < best_d:
                best, best_d = c, d2
    return best


def _nearest_free(occ: Occupancy, cell, max_r: int = 60):
    """The free cell nearest ``cell``, or None. Spiral search on the grid."""
    if occ.free(cell):
        return cell
    H, W = occ.shape
    for r in range(1, max_r + 1):
        best, best_d = None, None
        for dx in range(-r, r + 1):
            for dy in (-r, r) if abs(dx) != r else range(-r, r + 1):
                c = (cell[0] + dx, cell[1] + dy)
                if not (0 <= c[0] < H and 0 <= c[1] < W):
                    continue
                if not occ.grid[c[0], c[1]]:
                    d = dx * dx + dy * dy
                    if best_d is None or d < best_d:
                        best, best_d = c, d
        if best is not None:
            return best
    return None


def simplify(path, occ: Occupancy):
    """Drop waypoints a straight line already covers.

    A grid path has a corner every cell. Handing those to the generator as root
    goals would ask for dozens of 0.1 m chunks, which is both wasteful and worse
    -- Kimodo produces a natural WALK when given a metre to cover and something
    closer to a shuffle when given a decimetre.
    """
    if not path or len(path) < 3:
        return list(path or [])
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not _clear(out[-1], path[j], occ):
            j -= 1
        out.append(path[j])
        i = j
    return out


def _clear(a, b, occ: Occupancy) -> bool:
    n = max(2, int(np.linalg.norm(np.asarray(b) - np.asarray(a)) / (occ.cell * 0.5)))
    for t in np.linspace(0.0, 1.0, n):
        if not occ.free(occ.to_cell(np.asarray(a) + t * (np.asarray(b) - np.asarray(a)))):
            return False
    return True


#: Hops closer together than this are merged, metres. A hop is an instruction to
#: walk somewhere, and an instruction to walk 0.2 m is one the robot does not
#: carry out: over 1838 navigation chunks, delivery is 33-40% of the ask below
#: 1.2 m, so a 0.2 m hop moves the robot about 7 cm and costs a whole chunk.
#:
#: This is the same number as `MIN_USEFUL_ASK` in h7 and for the same reason,
#: but it is deliberately NOT imported from there: the planner does not depend on
#: the driver, and a shared constant across that boundary is how the scene side's
#: PALM_OFFSET ended up scoring the training model's reward (I-13).
MIN_HOP_SPACING = 0.45


def hops(path, hop: float = HOP, min_spacing: float = MIN_HOP_SPACING, occ=None):
    """Re-sample a path into steps a single chunk can actually walk.

    This is the step that makes the plan executable at all: Kimodo takes one root
    position per chunk, so a 6 m path is not an instruction it can follow, but
    five 1.2 m hops are five instructions it can.

    ⚠️ EVERY CORNER OF THE SIMPLIFIED PATH USED TO BECOME A HOP, however close it
    was to the one before. `simplify` drops points a straight line covers, but
    the corners it keeps are not evenly spaced, and near the goal they bunch:
    measured on `IThorBathroomApproachToilet`, identically at two seeds --

        hop0  1.69 m from the toilet   1.23 m from the previous point
        hop1  1.09 m                   1.20 m
        hop2  1.09 m                   0.20 m   <- same place
        hop3  1.05 m                   0.14 m   <- same place

    -- three waypoints piled on the arrival ring, each costing a chunk that moves
    the robot a few centimetres. That is the "shuffles along the edge before
    finally stepping in" this removes.

    ⚠️ The LAST point of a cluster is kept, not the first. Keeping the first
    would cut the corner OUTWARD, away from the path the planner proved to be
    clear; keeping the last only shortens the approach to a point already on it.
    ⚠️ The final point is always kept -- it is where the path ends, and dropping
    it would quietly move the destination.
    """
    if not path:
        return []
    pts = [np.asarray(p, dtype=np.float64)[:2] for p in path]
    # ⚠️ ``out`` must include the START, or the first hop is measured from a
    # point that is not in the list and the longest-hop check silently misses it.
    # That is how a 1.39 m hop survived a 1.2 m bar.
    out, carry = [pts[0]], pts[0]
    for nxt in pts[1:]:
        d = float(np.linalg.norm(nxt - carry))
        # ⚠️ EVENLY, not "full hops plus whatever is left". Stepping a fixed 1.2
        # and keeping the remainder turns a 1.40 m leg into 1.20 + 0.20, and that
        # 0.20 m stub is a whole chunk spent moving the robot about 7 cm --
        # delivery is 33-40% of the ask below 1.2 m. Measured on
        # `IThorBathroomApproachToilet`, identically at two seeds: the path runs
        # 2.73 -> 1.69 -> 1.09 -> 1.09 -> 0.97 m from the toilet, and the 1.69 ->
        # 1.09 leg is 1.40 m, so the chopper put a point 0.20 m short of a real
        # corner. The robot then took three moves along the arrival ring's edge
        # before stepping in, which is what this looked like from the outside.
        #
        # Dividing into `ceil(d / hop)` equal parts keeps every point ON the
        # segment (so nothing is cut), keeps every gap under `hop`, and cannot
        # emit a stub: the shortest piece a leg produces is now d / ceil(d/hop),
        # which for 1.40 m is 0.70 rather than 0.20.
        # ⚠️ Split against MAX_HOP, not `hop`. Dividing a 1.40 m leg by the 1.2 m
        # nominal gives two 0.70 m pieces, and 0.70 is in the flat part of the
        # delivery curve (33-40%) while 1.40 is in the good one (66%) -- so
        # "tidier" spacing bought two weak chunks in place of one strong one, and
        # measured worse: 1 fall and 1 physics rejection in 10 against 0 and 0.
        # `hop` stays the target; MAX_HOP is how far a chunk was actually
        # measured to cover, so a leg is only cut when it exceeds THAT.
        n = max(1, int(np.ceil(d / MAX_HOP - 1e-9)))
        for k in range(1, n + 1):
            out.append(carry + (nxt - carry) * (k / n))
        carry = nxt
    if min_spacing <= 0.0 or len(out) < 3:
        return out
    # Thin the clusters, SKIPPING points rather than replacing the one before
    # them.
    #
    # ⚠️ Replacing was the obvious way and it is wrong: on a path with points at
    # 0 / 1.2 / 1.4 / 1.5 / 2.6 it collapses to 0 / 1.5 / 2.6, and that first gap
    # is 1.5 m -- LONGER than the hop this function exists to bound. A hop over
    # the bar is the failure mode the `out` list is built to make visible, so
    # producing one while tidying the list would be a bad trade.
    #
    # So a point is dropped only when the gap it leaves behind still fits in a
    # chunk. Everything else is kept, however tightly packed.
    kept, i, last = [out[0]], 1, len(out) - 1
    while i <= last:
        p = out[i]
        gap0 = float(np.linalg.norm(p - kept[-1]))
        if gap0 < 1e-6:
            # ⚠️ Exact duplicate. The chopping loop steps to `hop` and then also
            # appends the segment end, so a segment that is an exact multiple of
            # `hop` emits the same point twice. A zero-length hop can never be
            # what keeps a gap under the bar, so it goes unconditionally -- and a
            # waypoint the robot is already standing on is precisely the "walks
            # in place for a chunk" this whole change is about.
            i += 1
            continue
        if gap0 >= min_spacing:
            kept.append(p)
        elif (i < last and float(np.linalg.norm(out[i + 1] - kept[-1])) <= hop
              and (occ is None or _clear(kept[-1], out[i + 1], occ))):
            # ⚠️ AND THE SHORTCUT HAS TO BE CLEAR. Dropping a point means walking
            # the straight line past it, and some of the points that bunch up are
            # real corners -- the path goes round something. Without this test the
            # thinning tripled the scene contacts (0.16 -> 0.44 and 0.13 -> 0.32
            # per chunk, both seeds) while physics still passed, which is the
            # order those two move in: contact duration is what degrades root
            # delivery long before anything trips the filter (bottleneck §1).
            #
            # `_clear` is the same line-of-sight test `simplify` uses, against the
            # same inflated occupancy, so a shortcut this accepts is one the
            # planner would have taken itself.
            pass                                  # drop: the next one covers it
        elif (i == last and len(kept) > 1
              and float(np.linalg.norm(p - kept[-2])) <= hop):
            # The end point replaces the one before it, but ONLY when the gap it
            # inherits still fits a chunk -- [0, 1.2, 1.3] collapsing to
            # [0, 1.3] would have produced a 1.3 m hop against a 1.2 m bar.
            # ⚠️ And never when `kept` is just the start: replacing there loses
            # the start, which the longest-hop check is measured from.
            kept[-1] = p
        else:
            kept.append(p)                        # cannot drop without a long hop
        i += 1
    return kept


def plan_hops(model, data, start_xy, goal_xy, *, arrive_radius=None, **kw):
    """Occupancy -> A* -> simplify -> hops. Returns ([], occ) when no path exists.

    ``arrive_radius`` should be the TASK's own success radius when there is one,
    so the plan aims at a spot that scores rather than at the edge of the
    inflation. Passing it is the difference between a correct plan and a
    successful one.

    ⚠️ **Tries a SEQUENCE of inflation radii, widest first.** The single 0.35 m
    radius was refusing half the episodes outright: measured on
    ``IThorLivingRoomLiftStatue``, 11 of 20 printed "NO PLAN", and the seed with 8
    of them is exactly the seed whose hand never got closer than 1.97 m while the
    seed with 3 reached 0.91 m. Flood-filling the free space says why -- on
    FloorPlan201 at 0.35 the goal is in a DIFFERENT CONNECTED COMPONENT from the
    robot, so no search could have found a path; at 0.25 they are connected and
    A* returns 37-56 points. FloorPlan202 is connected at both and unaffected.

    So the choice is not "0.35 or 0.25". A narrower body is a real risk -- 0.35
    was measured off the widest body geom because it is the shoulders that hit
    door frames -- but refusing to plan is not the safe option, it is the option
    where the robot drives straight at the furniture instead. Widest first,
    narrow only when the room leaves nothing else, and say which was used.
    """
    radii = kw.get("radii") or (kw.get("radius") or ROBOT_RADIUS, 0.28, 0.22)
    okw = {k: v for k, v in kw.items() if k in ("cell", "z_lo", "z_hi")}
    occ = None
    best = None                      # (end distance, hops, occ) across radii
    for radius in radii:
        occ = Occupancy(model, data, radius=radius, **okw)
        path = astar(occ, start_xy, goal_xy, arrive_radius=arrive_radius)
        # ⚠️ A DEGENERATE PATH IS A FAILURE, and it has to be reported as one.
        # When the goal is unreachable A* returns its best-so-far, which can be
        # the start cell itself -- a one-point "path" that gets simplified to one
        # point, chopped into one hop, and inserted as zero waypoints. Every stage
        # succeeds and nothing happens. Measured: the caller printed "planned 2
        # hops", the episode ran with its waypoint list unchanged, and the robot
        # walked into furniture exactly as it would have with no planner.
        #
        # The test is whether the path ENDS CLOSER than it started.
        if not path:
            continue
        start = np.asarray(start_xy, dtype=np.float64)[:2]
        goal = np.asarray(goal_xy, dtype=np.float64)[:2]
        if len(path) < 2 or (np.linalg.norm(np.asarray(path[-1])[:2] - goal)
                             >= np.linalg.norm(start - goal) - occ.cell):
            continue
        occ.radius_used = float(radius)
        end = float(np.linalg.norm(np.asarray(path[-1])[:2] - goal))
        made = hops(simplify(path, occ), kw.get("hop", HOP), occ=occ)
        # ⚠️ A path that is merely BETTER THAN NOTHING is not good enough to stop
        # searching on. `IThorBedroomLiftTeddyBear` plans to 1.25 m short of a goal
        # the task scores at 1.0 m -- geometrically correct, and a task failure --
        # and the wide radius accepts it because it does end closer than it
        # started. When the caller says what distance scores, keep narrowing until
        # the plan reaches it, and only then settle for the best of the lot.
        if arrive_radius is None or end <= arrive_radius:
            return made, occ
        if best is None or end < best[0]:
            best = (end, made, occ)
    if best is not None:
        return best[1], best[2]
    if occ is not None:
        occ.radius_used = None
    return [], occ
