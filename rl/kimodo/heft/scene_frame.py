"""Scene geometry for Kimodo goals: where to aim the wrist, and where to stand.

Kimodo is conditioned on a WRIST position -- the tracker controls the chain up to
``wrist_yaw_link`` and nothing beyond it. Objects, however, sit where the HAND
must close. Handing the object's centre straight to Kimodo puts the wrist on the
object and the palm through it: a collision, not a grasp.

Measured on robosuite's G1 (``EmptyEnv``, right hand, in the wrist frame):

    thumb_2 tip            x = 0.0748
    index_0 / middle_0     x = 0.1415      <- finger roots
    index_1 / middle_1     x = 0.1943
    all hand collision geoms   x in [0.0684, 0.2252]

so the point an object is enclosed at sits roughly midway between the thumb tip
and the finger mid-joints. ``PALM_OFFSET`` is that midpoint. ``measure_palm_offset``
recomputes it from whatever model is loaded, so a different hand does not silently
inherit this number.
"""

from __future__ import annotations

import numpy as np

#: Grasp centre ahead of ``wrist_yaw_link``, along the wrist's +x, in metres.
#: Midway between the thumb tip (0.075) and the finger mid-joints (0.194).
# ⚠️ The training side has its OWN number and they must not be unified: the
# mjlab model g1_5.xml has no fingers at all, and its ``hand_mimic`` frame sits
# at 0.1160 m. This 0.13 is correct HERE because robosuite's G1 really does have
# the hand geometry it was measured from. See kimodo_rl/rollout.py::PALM_OFFSET.
PALM_OFFSET = 0.13

#: How far from the target the robot should place its root before reaching.
#: Reused from the mjlab training side so the geometry Kimodo saw in training is
#: the geometry it sees here.
DEFAULT_STANDOFF = 0.45


def measure_palm_offset(model, data, prefix: str = "robot0_",
                        side: str = "right") -> dict:
    """Re-derive the hand geometry from a live model. Diagnostic, not a constant.

    Returns the span and the landmarks so a scene run can print them; a hand that
    differs from the measurement above should be noticed, not absorbed.
    """
    import mujoco

    wid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            f"{prefix}{side}_wrist_yaw_link")
    if wid < 0:
        raise SystemExit(f"[scene_frame] no {prefix}{side}_wrist_yaw_link")
    R, o = data.xmat[wid].reshape(3, 3), data.xpos[wid]
    xs = []
    for g in range(model.ngeom):
        bn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                               model.geom_bodyid[g]) or ""
        if f"{side}_hand" in bn and (model.geom_contype[g] | model.geom_conaffinity[g]):
            xs.append(float((R.T @ (data.geom_xpos[g] - o))[0]))
    if not xs:
        return {"n_geoms": 0, "offset_in_use": PALM_OFFSET}
    return {"n_geoms": len(xs), "x_min": min(xs), "x_max": max(xs),
            "x_mean": float(np.mean(xs)), "offset_in_use": PALM_OFFSET}


def resolve_floor_z(model, data, min_extent: float = 0.5) -> float:
    """World z of the walking surface, which is NOT 0 by default.

    robocasa does not build its floor from a ``<plane>``: in a living-room scene
    it is ``floor_room_g0``, a BOX whose centre is at z = -0.02 with a 0.02 m
    half-height, so the surface is its TOP FACE. Reading ``geom_xpos`` would put
    the datum 2 cm underground, and the reference's root height is measured from
    this surface -- DEBUG_LOG #5/#19 are both this mistake.

    Selected by size rather than by name: "floor" also appears in
    ``toaster_main_group_sideL_slotL_floor``. Anything large enough to stand on
    and near ground level qualifies; a sole-clearance check at the call site is
    what actually confirms the answer.

    Lives here rather than on the driver because the driver and the scene-side
    tracker must agree on it to the millimetre -- two copies of this rule would
    put the two paths on floors a couple of centimetres apart and every root
    height would differ by that much without anything failing.
    """
    import mujoco

    best = None
    for g in range(model.ngeom):
        if not (model.geom_contype[g] | model.geom_conaffinity[g]):
            continue
        gtype = model.geom_type[g]
        size = model.geom_size[g]
        pos = data.geom_xpos[g]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            top = float(pos[2])
        elif gtype == mujoco.mjtGeom.mjGEOM_BOX:
            if size[0] < min_extent or size[1] < min_extent:
                continue
            top = float(pos[2] + size[2])
        else:
            continue
        if not (-1.0 <= top <= 0.5):        # a table top is not the floor
            continue
        best = top if best is None else max(best, top)
    if best is None:
        raise SystemExit(
            f"[scene_frame] found no ground-level surface at least {min_extent} m "
            "across to use as the floor datum")
    return best


#: How far PAST the grasp point to aim, metres. Not a fudge factor: the reach
#: undershoots along the approach direction by a consistent amount, measured on
#: p35d over three episodes as -0.086 / -0.085 / -0.117 m, so a target placed
#: exactly on the object is a target the arm systematically stops short of.
#:
#: ⚠️ Aiming at the object's SURFACE -- the intuitive fix, and the one this
#: project briefly adopted -- moves the target the WRONG WAY. It shortens the
#: ask by the object's radius (0.056 m for the mug) on top of an undershoot that
#: is already 0.09, which is why the sign here is positive.
#:
#: ⚠️ AND IT DID NOT WORK EITHER. Swept over 16 waypoints per condition:
#:
#:     bias 0.00   median 0.116   62% under 0.12   best 0.094
#:     bias 0.08   median 0.101   62% under 0.12   best 0.082
#:
#: a +0.015 m improvement whose bootstrap 95% interval is [-0.010, +0.033] --
#: it straddles zero, so by criterion 1 there is no effect to report. Variance
#: also rose (0.173, 0.177 appear, worse than anything in the baseline): biasing
#: pulls the reachable ones closer and pushes the unreachable ones further.
#:
#: Kept as a knob, default 0.0 = the original behaviour bit for bit. The two
#: results together say the target's POSITION is not the lever -- the hand cannot
#: get to the target it already has, and moving that target is compensating
#: geometrically for what is an execution-accuracy problem.
REACH_BIAS = 0.0


def wrist_goal(grasp_point: np.ndarray, approach_dir: np.ndarray,
               offset: float = PALM_OFFSET, bias: float = None) -> np.ndarray:
    """Where the WRIST must be for the hand to enclose ``grasp_point``.

    ``approach_dir`` is the direction the hand travels in as it closes on the
    object, i.e. the wrist's +x at contact. The wrist therefore sits ``offset``
    BEHIND the grasp point along it, less ``bias`` to compensate for the
    measured undershoot.
    """
    d = np.asarray(approach_dir, dtype=np.float64)
    n = np.linalg.norm(d)
    if n < 1e-9:
        raise ValueError("approach_dir is degenerate")
    b = REACH_BIAS if bias is None else float(bias)
    return np.asarray(grasp_point, dtype=np.float64) - (d / n) * (float(offset) - b)


def palm_axis(model, data, prefix: str = "robot0_", side: str = "right") -> np.ndarray:
    """The direction the hand is actually pointing: the wrist frame's +x, world.

    ``approach_from_root`` guesses this direction from the geometry, and the
    guess is wrong by 30-34 degrees in practice -- because Kimodo's end-effector
    constraint pins the hand's POSITION and inherits its ORIENTATION from the
    conditioning prefix, so a reach arrives at the right place with roughly the
    orientation the robot was already standing in.

    Measured on ``TabletopPickPlace`` with a wrist error of 0.070 m, the object
    still sat at (x +0.10, y -0.06, z +0.12) in the wrist frame: inside the
    fingers' x span but 12 cm off their axis, so closing grabbed air. Aiming the
    next chunk with this axis instead of the guess is what closes that gap.
    """
    import mujoco

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                            f"{prefix}{side}_wrist_yaw_link")
    if bid < 0:
        raise SystemExit(f"[scene_frame] no {prefix}{side}_wrist_yaw_link")
    return data.xmat[bid].reshape(3, 3)[:, 0].copy()


def approach_from_root(grasp_point: np.ndarray, root_pos: np.ndarray,
                       horizontal: bool = True) -> np.ndarray:
    """Default approach: straight out from the robot towards the object.

    Kept horizontal by default. A reach that dives at the object also pitches the
    wrist, and the tracker's wrist authority is the weakest on the robot -- effort
    limit 5 Nm against 139 at the knee.
    """
    d = np.asarray(grasp_point, dtype=np.float64) - np.asarray(root_pos, dtype=np.float64)
    if horizontal:
        d = d.copy()
        d[2] = 0.0
    n = np.linalg.norm(d)
    return np.array([1.0, 0.0, 0.0]) if n < 1e-6 else d / n


def root_standoff(goal: np.ndarray, root_pos: np.ndarray,
                  standoff: float = DEFAULT_STANDOFF) -> np.ndarray:
    """A root XY that leaves the goal within arm's reach.

    Mirrors ``eef_goals.root_target`` from the training side: walk towards the
    goal but stop ``standoff`` short of it, so the arm does the last stretch.
    """
    g = np.asarray(goal, dtype=np.float64)[:2]
    r = np.asarray(root_pos, dtype=np.float64)[:2]
    d = g - r
    n = np.linalg.norm(d)
    if n <= standoff:
        return r.copy()          # already close enough; do not walk backwards
    return r + d * (1.0 - standoff / n)


def free_standoff(env, goal: np.ndarray, root_pos: np.ndarray,
                  standoff: float = DEFAULT_STANDOFF, step: float = 0.05,
                  max_extra: float = 0.35) -> tuple[np.ndarray, float]:
    """A standoff point that is not inside the furniture.

    The nominal standoff is a geometric guess and a scene is full of things to
    stand inside. Walks the point outward until the robot's bodies are clear.
    Returns ``(xy, used_standoff)`` so a caller can report when it had to back off.
    """
    import mujoco

    model = env.sim.model._model
    data = env.sim.data._data
    from rl.kimodo.heft.model_patch import free_joint_id

    qadr = int(model.jnt_qposadr[free_joint_id(model)])
    saved = data.qpos[qadr:qadr + 3].copy()

    n_steps = int(max_extra / step) + 1
    for k in range(n_steps):
        s = standoff + k * step
        xy = root_standoff(goal, root_pos, s)
        data.qpos[qadr:qadr + 2] = xy
        mujoco.mj_forward(model, data)
        hits = 0
        for c in range(data.ncon):
            g1, g2 = data.contact[c].geom1, data.contact[c].geom2
            b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   model.geom_bodyid[g1]) or ""
            b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                   model.geom_bodyid[g2]) or ""
            r1, r2 = b1.startswith("robot0_"), b2.startswith("robot0_")
            if r1 != r2 and data.contact[c].dist < -1e-3:
                hits += 1
        if hits == 0:
            data.qpos[qadr:qadr + 3] = saved
            mujoco.mj_forward(model, data)
            return xy, s
    data.qpos[qadr:qadr + 3] = saved
    mujoco.mj_forward(model, data)
    return root_standoff(goal, root_pos, standoff), standoff
