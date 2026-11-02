"""How a generated chunk TRAVELS: forwards, backwards, or sideways.

Kimodo will happily walk a chunk straight backwards. h16 measured it: a goal
1.5 m behind the robot, prompt ``"a person walks forward"``, closes from 1.50 m
to 0.53 m with a heading change of **+1 degree** -- 1.9 m of travel, none of it
turning. The generator is doing what it was asked; nothing in the ask says the
body has to face where it is going.

That is a defect for three reasons, in increasing order of how much they cost:

  * **The scene punishes it.** The robot cannot see behind itself and neither can
    the planner's hops, which are aimed at the goal, not at what is between. The
    measured walk delivery rate is 82-120% in an empty arena and 8-16% once
    furniture is touched.
  * **The tracker is worse at it.** HEFT's training distribution is forward
    locomotion; a backwards reference is out-of-distribution for the one part of
    the pipeline that is frozen.
  * **The data is not what anyone wants.** These rollouts exist to become
    demonstrations. A humanoid moonwalking to a drawer is a correct solution to
    the reward and an unusable demonstration.

This module is the measurement, the gate and the picker -- all three, because
the gate is only trustworthy if the number behind it is the same one reported in
the logs.

    stats = travel_stats(q)            # what the chunk does
    ok, why = travel_ok(stats)         # does it obey
    i = best_chunk(qs, goal_local)     # which of K candidates to keep

⚠️ Everything here reads the GENERATED reference, not the executed state. That
is deliberate: a constraint the generator violates cannot be repaired by the
tracker, and checking the reference costs no simulation.
"""

from __future__ import annotations

import numpy as np

#: Chunk frames are 30 Hz.
FPS = 30.0

#: Below this the chunk is standing still and the direction of "travel" is noise.
#: 5 cm over 3 s -- an order of magnitude under the 0.6-1.2 m a working approach
#: chunk covers.
STILL = 0.05

#: How much of the path may be walked backwards, in metres. Not zero: the gait
#: itself has a small retrograde component at each footfall, and a hard zero
#: would reject every chunk including the ones that work.
MAX_BACK = 0.15

#: How much of the path may be strafed, as a fraction of the total. Sideways
#: travel is legitimate in small amounts -- the final step onto a standoff point
#: is often a shuffle -- but a chunk that is mostly strafe has not turned.
MAX_SIDE_FRAC = 0.45


def heading(q: np.ndarray) -> np.ndarray:
    """[T] root yaw in radians, from the reference's own wxyz quaternion."""
    a = np.asarray(q, dtype=np.float64)
    w, x, y, z = a[:, 3], a[:, 4], a[:, 5], a[:, 6]
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def travel_stats(q: np.ndarray) -> dict:
    """Decompose a chunk's root path into forward / backward / sideways.

    ⚠️ Against the **per-frame** heading, not the heading at the start of the
    chunk. A robot that turns 90 degrees and then walks has travelled forwards,
    and scoring it against where it was first pointing would call that motion
    sideways -- which would ban the one behaviour this whole module is trying to
    encourage.
    """
    a = np.asarray(q, dtype=np.float64)
    xy = a[:, 0:2]
    if len(xy) < 2:
        return {"travel": 0.0, "path": 0.0, "fwd": 0.0, "back": 0.0,
                "side": 0.0, "side_frac": 0.0, "fwd_frac": 1.0, "dyaw": 0.0}
    d = np.diff(xy, axis=0)
    yaw = heading(a)[:-1]
    f = np.stack([np.cos(yaw), np.sin(yaw)], axis=1)
    l = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
    along = np.einsum("ij,ij->i", d, f)
    lat = np.einsum("ij,ij->i", d, l)
    path = float(np.abs(np.linalg.norm(d, axis=1)).sum())
    fwd = float(np.clip(along, 0.0, None).sum())
    back = float(np.clip(-along, 0.0, None).sum())
    side = float(np.abs(lat).sum())
    dyaw = float(np.degrees(np.unwrap(heading(a))[-1] - heading(a)[0]))
    return {
        "travel": round(float(np.linalg.norm(xy[-1] - xy[0])), 4),
        "path": round(path, 4),
        "fwd": round(fwd, 4),
        "back": round(back, 4),
        "side": round(side, 4),
        "side_frac": round(side / path, 4) if path > 1e-9 else 0.0,
        "fwd_frac": round(fwd / path, 4) if path > 1e-9 else 1.0,
        "dyaw": round(dyaw, 1),
    }


def travel_ok(stats: dict, *, max_back: float = MAX_BACK,
              max_side_frac: float = MAX_SIDE_FRAC) -> tuple[bool, str]:
    """Does this chunk move the way a person walking somewhere moves?

    A chunk that barely moves passes: standing still is not travelling
    backwards, and the reach chunks -- which are most of them -- must not be
    rejected by a locomotion rule.
    """
    if float(stats.get("path", 0.0)) < STILL:
        return True, "still"
    if float(stats["back"]) > max_back:
        return False, f"backward {stats['back']:.2f}m > {max_back:.2f}"
    if float(stats["side_frac"]) > max_side_frac:
        return False, f"sideways {stats['side_frac']:.0%} > {max_side_frac:.0%}"
    return True, "ok"


def closure(q: np.ndarray, goal_xy) -> float:
    """How much closer to ``goal_xy`` the chunk ends than it started, in metres.

    Positive is progress. In the chunk's own frame, which is where both the
    generated root and the root goal handed to the generator live.
    """
    a = np.asarray(q, dtype=np.float64)[:, 0:2]
    g = np.asarray(goal_xy, dtype=np.float64).reshape(2)
    return float(np.linalg.norm(a[0] - g) - np.linalg.norm(a[-1] - g))


def score(stats: dict, gain: float) -> float:
    """Rank a candidate: progress first, then how it got there.

    The penalty is in metres so it is commensurate with the progress it is
    traded against -- a chunk that closes 5 cm more by walking 30 cm backwards
    loses to one that closes less and stays facing forward.
    """
    return (float(gain)
            - 2.0 * float(stats["back"])
            - 0.5 * max(0.0, float(stats["side_frac"]) - MAX_SIDE_FRAC)
            * float(stats["path"]))


def best_chunk(qs: list, goal_xy=None, **gate) -> tuple[int, list]:
    """Pick from K candidate chunks. Returns (index, per-candidate report).

    Legal candidates -- those passing ``travel_ok`` -- always outrank illegal
    ones, however much better the illegal one's progress is. Within a class the
    ranking is by ``score``. If every candidate is illegal the least bad is
    returned rather than nothing: a rejected chunk is a stalled episode, and the
    reason travels with it so the caller can log what it had to accept.
    """
    reports = []
    for q in qs:
        st = travel_stats(q)
        ok, why = travel_ok(st, **gate)
        gain = 0.0 if goal_xy is None else closure(q, goal_xy)
        reports.append({**st, "ok": bool(ok), "why": why,
                        "gain": round(gain, 4), "score": round(score(st, gain), 4)})
    order = sorted(range(len(reports)),
                   key=lambda i: (reports[i]["ok"], reports[i]["score"]))
    return order[-1], reports
