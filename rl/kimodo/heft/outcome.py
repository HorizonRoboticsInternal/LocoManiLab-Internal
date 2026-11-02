"""Did Kimodo do the job it was given? One label per ask, not a pile of errors.

The pipeline gives Kimodo exactly two kinds of instruction:

    a target ROOT position   -> walk there
    a target EEF position    -> put the hand there

Everything downstream -- navigation planning, grasp point selection, gripper
timing -- is built on those two working. So the first thing an evaluation has to
say is which of them failed and HOW, and the current logs cannot: they report
``root_err_final 0.42`` for a robot that walked most of the way and stopped
short, for one that never moved at all, and for one that walked briskly in the
wrong direction. Those are three different bugs with the same number.

The labels, and what each one indicts:

    delivered   inside tolerance at some point            -- nothing
    overshoot   got there, then left                      -- the horizon/stop rule
    short       moved towards it, ran out                 -- capability or budget
    stuck       barely moved while being asked to         -- the tracker or a wall
    deviated    moved, but not towards the target         -- the generator's aim
    infeasible  the ask was outside the measured envelope -- the PLANNER, not Kimodo

``infeasible`` is the one worth having on principle. A pipeline that grades only
the executor will keep blaming it for asks nobody could satisfy, and the fix for
"the plan put the goal 1.2 m from the shoulder" is not in the generator.

⚠️ Progress is measured against the ask's own starting distance, so a chunk that
begins already inside tolerance is `delivered` and not something to celebrate --
`d_start` travels with the label for exactly that reason.
"""

from __future__ import annotations

import numpy as np

#: Travel below which "it moved" is not a claim worth making, in metres. A 3 s
#: chunk at a walking pace covers 0.6-1.2 m, so 5 cm is comfortably noise.
STILL = 0.05

#: Fraction of the initial gap that has to close before the attempt counts as
#: aimed at the target rather than merely energetic.
MIN_PROGRESS_FRAC = 0.15

#: Measured reach of the arm from the pelvis with the feet planted: h18 puts the
#: standing envelope at 0.80 m and `eef_goals` measures the arm alone at p95
#: 0.639 m. An eef ask beyond this is a planning error, not an execution one.
STANDING_REACH = 0.80


def classify(d_start: float, err_min: float, err_final: float, travel: float,
             tol: float, *, feasible: bool = True,
             still: float = STILL,
             min_progress_frac: float = MIN_PROGRESS_FRAC) -> dict:
    """One ask -> one label. All distances in metres.

    ``d_start``   how far the target was when the chunk began
    ``err_min``   closest the thing ever got
    ``err_final`` where it ended up
    ``travel``    path length actually covered (NOT displacement -- a robot that
                  walks a circle has travelled, and calling that "stuck" would
                  hide a real and different failure)
    ``tol``       what counts as arrived for this ask
    """
    d_start = float(d_start)
    progress = d_start - float(err_min)
    frac = progress / d_start if d_start > 1e-6 else 1.0
    out = {"d_start": round(d_start, 4), "err_min": round(float(err_min), 4),
           "err_final": round(float(err_final), 4), "travel": round(float(travel), 4),
           "progress": round(progress, 4), "progress_frac": round(frac, 3),
           "tol": round(float(tol), 4)}

    if not feasible:
        return {**out, "label": "infeasible"}
    if err_min <= tol:
        # Arrived. The only question left is whether it stayed.
        return {**out, "label": "delivered" if err_final <= tol else "overshoot"}
    if travel < still:
        return {**out, "label": "stuck"}
    if frac < min_progress_frac:
        return {**out, "label": "deviated"}
    return {**out, "label": "short"}


def eef_feasible(d_root_to_goal: float, reach: float = STANDING_REACH) -> bool:
    """Could a planted robot have put its hand there at all?

    Horizontal distance from the ROOT, because that is the axis h18 measured and
    the one the legs would have to fix. Deliberately generous: this exists to
    catch asks that are plainly impossible, not to adjudicate the last 5 cm.
    """
    return float(d_root_to_goal) <= float(reach)


def summarise(rows: list, key: str = "label") -> dict:
    """Label histogram, ordered worst-first so the top line is the thing to fix."""
    order = ["stuck", "deviated", "short", "overshoot", "infeasible", "delivered"]
    hist = {k: 0 for k in order}
    for r in rows:
        lab = r.get(key)
        if lab is not None:
            hist[lab] = hist.get(lab, 0) + 1
    n = sum(hist.values()) or 1
    return {**hist, "n": sum(hist.values()),
            "delivered_rate": round(hist.get("delivered", 0) / n, 3)}


def table(groups: dict, title: str = "") -> str:
    """``{name: [row, ...]}`` -> a printable label table."""
    cols = ["stuck", "deviated", "short", "overshoot", "infeasible", "delivered"]
    out = [f"{title}" if title else ""]
    out.append(f"{'':<34}{'n':>5}" + "".join(f"{c[:9]:>10}" for c in cols))
    for name, rows in groups.items():
        h = summarise(rows)
        out.append(f"{name:<34}{h['n']:>5}"
                   + "".join(f"{h.get(c, 0):>10}" for c in cols))
    return "\n".join(x for x in out if x)
