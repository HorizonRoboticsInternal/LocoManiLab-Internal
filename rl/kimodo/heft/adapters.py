"""Turn a task scene into the waypoints Kimodo is conditioned on.

Kimodo takes a 3-D wrist goal and an optional root sub-goal. A scene has objects,
fixtures and a ``_check_success()``. This module is the translation, and it is
deliberately **outside** the task files: adapters are keyed by env class name, so
adding Kimodo support to a task requires no edit to the task, and a task that has
no adapter fails loudly instead of being driven at a guessed goal.

The geometry that actually matters is in ``scene_frame``: an object's position is
a GRASP point, and the wrist has to sit ``PALM_OFFSET`` behind it or the palm goes
through the object.

Adapters read only attributes the tasks already expose -- ``_obj_pos``,
``_target_point``, ``objects[...]``, ``sim.data`` -- so they stay correct as long
as the task's own success predicate does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from rl.kimodo.heft.scene_frame import (
    DEFAULT_STANDOFF,
    approach_from_root,
    root_standoff,
    wrist_goal,
)

#: Where the arm alone stops closing the gap, measured by h18: inside this the
#: robot can reach without moving its feet, past it the error is just the
#: shortfall (0.26 m at 1.0, 0.46 m at 1.2).
STANDING_REACH = 0.80
#: What the generator can actually DELIVER a hand to without stepping, metres.
#: Distinct from `STANDING_REACH`, which is what the ARM can span (h18: feet
#: planted, lean allowed, about 0.80 m). The kinematics and the model disagree,
#: and the model is the binding constraint:
#:
#:     ask distance   n    ref     end     held 10 ticks inside 0.10 m
#:     0.35-0.50      9   0.224   0.068          7/9
#:     0.50-0.64      8   0.240   0.172          2/8
#:     0.64-0.80     12   0.469   0.275          1/12
#:     0.80+          6   0.677   0.335          0/6
#:
#: (h27, empty floor, p41c, targets drawn 0.20-0.85.) The cliff sits at 0.64,
#: which is the measured p95 arm reach -- inside it the hand arrives and stays,
#: past it it does not, and the reference itself doubles its error.
#:
#: This matters because the adapters gated WALKING on 0.80 and the tasks ask for
#: more than 0.64 most of the time: on `IThorKitchenAppleOnPlate` the horizontal
#: ask runs 0.489-0.805 with a median of 0.664. Every one of those was marked
#: "already in reach, do not move your feet" and then missed.
DELIVERABLE_REACH = 0.60

# Grasp phases. The scripted gripper acts on these; Kimodo never sees them.
OPEN, CLOSE, HOLD, RELEASE = "open", "close", "hold", "release"


@dataclass
class Waypoint:
    """One Kimodo chunk's worth of intent."""

    grasp_point: np.ndarray          # where the HAND should end up, world
    phase: str = OPEN                # OPEN / CLOSE / HOLD / RELEASE
    eef: str = "RightHand"           # Kimodo joint name for the constraint
    label: str = ""
    standoff: float = DEFAULT_STANDOFF
    needs_root: bool = True          # walk towards it, or reach from where we are
    #: Name of the scene object this waypoint acts on, when there is one. The
    #: scripted grasp needs it to know whose geoms count as "touching" -- without
    #: it the hand can only close on a schedule, which is what it used to do.
    grasp_object: str | None = None
    #: A LOCOMOTION waypoint: the task is to get the BODY somewhere, and there is
    #: no end-effector goal at all. Distinct from a reach whose hand target
    #: happens to be far away -- handing an approach task an eef goal asks the
    #: chunk to walk and arrive reaching, which is the ask a 3 s chunk cannot
    #: meet, and the hand has nothing to do with the success criterion anyway.
    approach_only: bool = False
    #: How close the ROOT has to get for this waypoint to count as done, when
    #: there is nothing better to ask. ⚠️ A RADIUS ON THE ROOT IS NOT THE
    #: CRITERION any iThor navigation task uses -- see ``arrive_test``. Kept for
    #: the planner's own hops, which are waypoints the task has no opinion about,
    #: and as the number the report prints beside `root_err`.
    arrive_radius: float | None = None
    #: The task's OWN test for "we are there", when it exposes one. Preferred
    #: over ``arrive_radius`` because a radius on the root cannot express what
    #: these tasks measure, and h31 measured how badly:
    #:
    #:   * the judge is `mj_geomDistance` SURFACE to surface, held 10 steps;
    #:     the radius is root centre to fixture CENTRE. Different quantities.
    #:   * the conversion between them is per fixture AND per azimuth. On
    #:     `IThorBedroomApproachBed`, one root-to-centre distance of 1.2 m gives
    #:     surface distances from 0.027 to 0.34 m depending on which side of the
    #:     bed the robot stands. No single radius represents that.
    #:   * the value in use was 1.0 m, at which the toilet task passes on 0 of
    #:     12 azimuths (surface 0.369 m against its 0.1 m bar).
    #:
    #: A callable rather than a number because the task already computes it, and
    #: recomputing it here is how the driver and the judge drift apart.
    arrive_test: object | None = None
    #: Which way the hand travels as it arrives, world frame. ``None`` means
    #: "straight out from the robot, horizontally", which is right for picking
    #: something off a surface and WRONG for pressing something down: measured on
    #: a laptop lid, a horizontal arrival drove the hinge to -0.05 rad, i.e. it
    #: pushed the lid further OPEN. A push task knows the direction its target has
    #: to move, and that is the direction the palm should come from.
    approach_dir: np.ndarray | None = None

    def wrist_target(self, root_pos: np.ndarray) -> np.ndarray | None:
        """The wrist goal handed to Kimodo, palm offset applied. None if this is
        a pure locomotion waypoint."""
        if self.approach_only:
            return None
        d = (self.approach_dir if self.approach_dir is not None
             else approach_from_root(self.grasp_point, root_pos))
        return wrist_goal(self.grasp_point, d)

    def root_target(self, root_pos: np.ndarray) -> np.ndarray | None:
        if not self.needs_root:
            return None
        # An approach waypoint aims the BODY at the target itself, not at a
        # standoff: the success criterion is the root's own distance, so backing
        # off by an arm's length would deliberately stop short of passing.
        if self.approach_only:
            return np.asarray(self.grasp_point, dtype=np.float64).copy()
        return root_standoff(self.grasp_point, root_pos, self.standoff)


@dataclass
class SceneGoals:
    waypoints: list[Waypoint]
    lang: str = ""
    notes: dict = field(default_factory=dict)


_REGISTRY: dict[str, Callable] = {}


def register(*env_names):
    def deco(fn):
        for n in env_names:
            _REGISTRY[n] = fn
        return fn
    return deco


def adapter_for(env) -> Callable:
    name = type(getattr(env, "unwrapped", env)).__name__
    if name not in _REGISTRY:
        raise SystemExit(
            f"[adapters] no SceneGoalAdapter for {name!r}. Registered: "
            f"{sorted(_REGISTRY)}. Driving an unknown scene at a guessed goal is "
            f"how a run produces plausible numbers about nothing.")
    return _REGISTRY[name]


def goals_for(env) -> SceneGoals:
    base = getattr(env, "unwrapped", env)
    g = adapter_for(env)(base)
    if not g.lang:
        try:
            g.lang = base.get_ep_meta().get("lang", "")
        except Exception:  # noqa: BLE001
            g.lang = ""
    return g


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
@register("SquatPickPlaceTVStand")
def _squat_pick_place(env) -> SceneGoals:
    """Squat to the apple on a low table, then place it on the TV stand behind.

    The task's own ``_obj_pos`` / ``_target_point`` are the source of truth --
    the same two quantities its ``_check_success`` is computed from, so the goal
    the robot is driven at and the criterion it is judged by cannot drift apart.
    """
    obj = np.asarray(env._obj_pos(), dtype=np.float64)
    tgt = np.asarray(env._target_point(), dtype=np.float64)
    return SceneGoals(
        waypoints=[
            Waypoint(obj, phase=CLOSE, label="pick apple", grasp_object="apple"),
            # Place slightly above the stand top: the success test is a 0.30 m
            # sphere around the top centre, so releasing at the surface risks
            # dropping through it before the check runs.
            Waypoint(tgt + np.array([0.0, 0.0, 0.10]), phase=RELEASE,
                     label="place on tv stand"),
        ],
        notes={"target_radius": float(getattr(env, "TARGET_RADIUS", 0.30))},
    )


@register("TabletopPickPlace", "StackCanOnPlate")
def _tabletop_pick_place(env) -> SceneGoals:
    """Pick the first object and put it on the target, both from the task."""
    names = list(getattr(env, "objects", {}).keys())
    if not names:
        raise SystemExit(f"[adapters] {type(env).__name__} exposes no objects")
    src = _object_pos(env, names[0])
    tgt = (np.asarray(env._target_point(), dtype=np.float64)
           if hasattr(env, "_target_point") else _object_pos(env, names[-1]))
    return SceneGoals(waypoints=[
        Waypoint(src, phase=CLOSE, label=f"pick {names[0]}", grasp_object=names[0]),
        Waypoint(tgt + np.array([0.0, 0.0, 0.08]), phase=RELEASE, label="place",
                 grasp_object=names[0]),
    ])


@register("CarryObjectToTable")
def _carry(env) -> SceneGoals:
    """Pick, walk, place. The walk is expressed only as a root sub-goal."""
    names = list(getattr(env, "objects", {}).keys())
    src = _object_pos(env, names[0])
    tgt = np.asarray(env._target_point(), dtype=np.float64)
    return SceneGoals(waypoints=[
        Waypoint(src, phase=CLOSE, label=f"pick {names[0]}", grasp_object=names[0]),
        # Hold while walking: the wrist goal stays with the object, which is now
        # in the hand, so the reach constraint is satisfied and the root sub-goal
        # does the work.
        Waypoint(tgt, phase=HOLD, label="carry", standoff=0.9, grasp_object=names[0]),
        Waypoint(tgt + np.array([0.0, 0.0, 0.08]), phase=RELEASE, label="place",
                 grasp_object=names[0]),
    ])


# ⚠️ `IThorBedroomLiftTeddyBear` is NOT here. It is unregistered in
# `robocasa/__init__.py` on this branch and deleted on `main`, whose commit
# 4e2602f says why: "Change task: IThorBedroomLiftTeddyBear ->
# IThorBedroomLiftBasket". It was SUPERSEDED, not lost. Keeping it registered
# here bought one `Environment not found` per batch and nothing else.
@register("IThorLivingRoomLiftStatue", "IThorBedroomLiftBasket")
def _room_lift(env) -> SceneGoals:
    """Reach the object and lift it. One waypoint, and the grasp is the task.

    The iTHOR lift tasks are the right place to test a grasp, because their
    success condition is ``_object_is_lifted() and _hand_supports_object()``
    held for HOLD_STEPS -- so nothing succeeds by walking past. Note that
    ``_hand_supports_object`` is itself a PROXIMITY test (0.16-0.24 m depending
    on the task), which is why the pipeline keeps its own contact-and-carry
    check in ``grasp.py`` rather than trusting the task's verdict as evidence
    that the robot held anything.

    ⚠️ **Only some of these place the robot in front of the object.**
    ``lift_teddy_bear`` calls ``place_robot_at_fixture`` and starts within reach;
    ``lift_statue`` does not, and measured spawn-to-object distance there is
    2.1-3.0 m with the object typically BEHIND the robot. So the approach
    waypoint asks for the root and the lift waypoint does not -- by then the
    robot is where it needs to be, and a second root sub-goal would only walk it
    back out of the pose it just achieved.
    """
    # The task names its own target (`RoomLiftMixin.TARGET_NAME`). The old
    # two-way guess worked only while there were exactly two lift tasks, and
    # `IThorBedroomLiftBasket` -- a third one, on the same mixin -- would have
    # been driven at a teddy bear that does not exist in its scene.
    name = getattr(env, "TARGET_NAME", None)
    if name not in getattr(env, "objects", {}):
        name = (
            "statue"
            if "statue" in getattr(env, "objects", {})
            else "basket"
        )
    obj = _object_pos(env, name)
    lift = np.array([0.0, 0.0, float(getattr(env, "MIN_LIFT_DELTA", 0.12)) + 0.10])
    return SceneGoals(waypoints=[
        Waypoint(obj, phase=CLOSE, label=f"grasp {name}", grasp_object=name),
        # Same object, raised: once the hand has closed, the thing to ask for is
        # that hand a little higher. A fixed world point would fight whatever
        # pose the grasp actually ended in.
        Waypoint(obj + lift, phase=HOLD, label=f"lift {name}", needs_root=False,
                 grasp_object=name),
    ])


#: How much of a swing one waypoint may cover, radians. 20 degrees keeps the
#: chord within ~1.5%% of the arc it approximates, which is under the hand's own
#: 0.064 m palm-to-fingertip distance for any door this pipeline has met.
ARC_STEP = np.deg2rad(20.0)
#: Candidates within this fraction of the best travel count as tied and are
#: decided on reach instead. 0.8 keeps a real lever advantage decisive while
#: letting a 60-geom door be chosen sensibly.
TIE_FRACTION = 0.8
#: Height above the pelvis where a standing reach is most comfortable, metres.
#: The G1's shoulder sits about here, and h25's sweep is best at `dz +0.15`.
COMFORT_DZ = 0.15
#: A candidate must still carry this fraction of the best travel, or "pushing"
#: it moves the joint too little to reach its bar.
MIN_TRAVEL_FRACTION = 0.25
#: How far above or below the comfort height the arc may wander, metres.
MAX_DZ = 0.45
#: Cap, because each waypoint costs up to `--max-chunks` chunks of 3 s.
MAX_ARC_STEPS = 5


#: Every task built on ``RoomOpenArticulation``: a joint, a threshold, and a
#: robot the task itself parks in front of the fixture. They are the middle tier
#: -- scored on an ANGLE (or a slide), so the hand only has to touch and keep
#: going, and palm orientation never enters the criterion.
#:
#: ⚠️ Half of them are not winnable with an open hand and that is a property of
#: the task, not of the model: OPENING a drawer or a door means pulling it
#: towards you, which needs the fingers to hold a handle. CLOSING means pushing,
#: which a flat palm does. `_articulation` reports which kind each episode is
#: rather than failing quietly on the pulls.
_ARTICULATION_TASKS = (
    "IThorLivingRoomCloseLaptop", "IThorBedroomCloseLaptop",
    "IThorKitchenCloseDrawer", "IThorKitchenCloseCabinet",
    "IThorKitchenCloseFridge", "IThorKitchenCloseOven",
    "IThorKitchenOpenDrawer", "IThorKitchenOpenCabinet",
    "IThorKitchenOpenFridge", "IThorKitchenOpenOven",
    # The other three rooms. Every one of these is a `RoomOpenArticulation`
    # subclass, so the adapter below already handles them -- they were simply
    # never listed, and an unlisted task fails with "no SceneGoalAdapter" in the
    # middle of a batch rather than at its start.
    "IThorBathroomCloseToiletLid", "IThorBathroomOpenToiletLid",
    "IThorBedroomCloseTopDrawer", "IThorBedroomOpenTopDrawer",
    "IThorLivingRoomOpenDrawer",
    # ⚠️ The `Walk*` / `Approach*` variants are the SAME fixture with the robot
    # spawned across the room: h19 puts the joint 1.46-3.36 m away at reset
    # against 0.35-0.70 for the others. They are listed because the adapter's
    # own standoff logic already walks the root in, but expect the planner to
    # carry the episode and the articulation to be the last 3 seconds of it.
    "IThorBathroomApproachToiletAndOpenToiletLid",
    "IThorBathroomApproachToiletAndCloseToiletLid",
    "IThorBedroomApproachDeskAndCloseLaptop",
    "IThorKitchenApproachOpenFridge", "IThorKitchenApproachOpenOven",
)


@register(*_ARTICULATION_TASKS)
def _articulation(env, behavior: str | None = None) -> SceneGoals:
    """Push an articulated fixture to where the task wants it. Contact scale, no grasp.

    Everything this pipeline has succeeded at is scored on the ROOT at 1.0 m, and
    everything it has failed at needs the fingers to close on a 0.02-0.04 m
    envelope. These sit between: success is a joint reaching a threshold and
    HOLDING it, so the hand only has to touch and keep going, and palm
    orientation -- where every grasp attempt has died -- never enters the
    criterion. h19 puts every one of them inside the reach envelope, 0.64-0.98 m
    with a clear stance at 0.45-0.76 m.

    ⚠️ **Half of them are not winnable with an open hand**, and that is the task's
    shape rather than the model's limit. OPENING a drawer or a door pulls it
    towards you, which needs fingers on a handle; CLOSING pushes, which a flat
    palm does. The verdict is computed and reported per episode rather than
    discovered from a run of zeros -- see ``notes["kind"]``.

    Three things this has to get right, two of them found by getting them wrong
    on the laptop:

      * **Aim at the point that MOVES, not at the body's centre.** The task's own
        ``target_joint_pos`` is the mean of every point on the moving body; on a
        laptop lid that sits half a lid-length below the edge, has the shortest
        lever on the hinge, and travels only 0.08 m over the whole sweep -- so
        aiming there asks the hand to stop where the lid already is. This picks
        the point with the LARGEST travel between now and the target pose, which
        is the far edge of a door and any point of a drawer.
      * **Arrive from the direction the part must travel.** The first version
        left the approach at its default (straight out from the robot,
        horizontal) and drove a laptop hinge to -0.05 rad, i.e. it pushed the lid
        further OPEN.
      * **Aim past the bar.** Reaches undershoot along the approach by a measured
        0.085-0.117 m, and a push has no penalty for overshooting -- the joint
        limit stops the part, and a hand that keeps going rests on a shut drawer.
    """
    import mujoco

    sim = env.sim
    m, d = sim.model._model, sim.data._data
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, env.target_joint)
    if jid < 0:
        raise SystemExit(f"[adapters] {env.target_joint!r} is not in the model")
    bid, adr = int(m.jnt_bodyid[jid]), int(m.jnt_qposadr[jid])

    # ⚠️ Written against the task's OWN success predicate, which changed when the
    # vetted `RoomOpenArticulation` was adopted from the wjb branch. It is now a
    # FRACTION of each joint's own travel -- `close` wants every success joint at
    # or below `SUCCESS_CLOSE_FRACTION` (0.05), `open` at or above
    # `SUCCESS_OPEN_FRACTION` (0.90) -- and there is no `MIN_SUCCESS_QPOS_DELTA`
    # any more. The previous version read that attribute and would now raise;
    # worse, the old bars were 0.40/0.40, so every articulation number measured
    # before this was against a much laxer criterion than the task now applies.
    # `behavior` overrides the task's own for the COMPOSITE tasks, where the
    # same fixture is opened and then closed inside one episode and the env
    # carries no single answer. None keeps reading the task, as before.
    opening = (behavior or getattr(env, "BEHAVIOR", "close")) == "open"
    j = env.target_semantic_joint
    closed, opened = float(j.closed_qpos), float(j.open_qpos)
    now = float(d.qpos[adr])
    want_frac = (float(env.SUCCESS_OPEN_FRACTION) if opening
                 else float(env.SUCCESS_CLOSE_FRACTION))
    # The joint value that satisfies the predicate, and the extreme beyond it.
    want = closed + want_frac * (opened - closed)
    far = opened if opening else closed
    rng = abs(opened - closed)
    thr = abs(want - (closed if not opening else opened))

    def moving_points(qpos: float) -> np.ndarray:
        """Every geom centre on the moving body, at a given joint value."""
        d.qpos[adr] = float(qpos)
        mujoco.mj_forward(m, d)
        pts = [np.asarray(d.geom_xpos[g], dtype=np.float64)
               for g in range(m.ngeom) if int(m.geom_bodyid[g]) == bid]
        return np.asarray(pts) if pts else np.asarray(d.xpos[bid])[None]

    try:
        here, there = moving_points(now), moving_points(far)
        at_bar = moving_points(want)
    finally:
        d.qpos[adr] = now
        mujoco.mj_forward(m, d)

    if here.shape != there.shape or len(here) == 0:
        raise SystemExit(f"[adapters] {env.target_joint!r}: moving body has no geoms")
    # ⚠️ "THE POINT THAT TRAVELS FURTHEST" IS BARELY A CHOICE ON A DOOR.
    # `IThorKitchenCloseFridge`'s door carries 60 geoms and the top eight by
    # travel are within 0.02 m of each other (0.51-0.53) -- a near-tie that
    # `argmax` breaks arbitrarily. Their heights above the pelvis span +0.10 to
    # +1.23 m, so the same task picks a waist-high panel in one scene and a point
    # above the robot's head in the next, which is why its results swing between
    # floor plans for no reason the model controls.
    #
    # Keep the near-ties on travel, then choose among them by whether a standing
    # robot could actually put a hand there. Nothing here is annotation: the
    # joint comes from the task and the rest is forward kinematics.
    travel = np.linalg.norm(there - here, axis=1)
    best = float(travel.max())
    if best <= 1e-6:
        raise SystemExit(f"[adapters] {env.target_joint!r}: nothing on this body moves")
    root_now = np.asarray(env.sim.data.body_xpos[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_support")],
        dtype=np.float64)

    # ⚠️ NOT the point with the most travel. That is the longest lever and also
    # the point that runs away fastest: on `IThorKitchenCloseFridge` the furthest
    # travelling geom goes from 0.32 m to 1.29 m from a robot that is not
    # supposed to take a step. The task is done standing still, so the point to
    # push is the one whose WHOLE ARC stays inside the arm -- nearer the hinge,
    # shorter lever, but reachable from the first frame to the last.
    #
    # Travel still has to be non-trivial or the "push" is a point that barely
    # moves and the joint never reaches its bar.
    n_probe = 7
    qs = np.linspace(now, far, n_probe)
    sweep = np.stack([moving_points(q) for q in qs])       # [probe, geom, 3]
    d.qpos[adr] = now
    mujoco.mj_forward(m, d)
    radial = np.linalg.norm(sweep[:, :, :2] - root_now[None, None, :2], axis=2)
    worst = radial.max(axis=0)                             # furthest it ever gets
    dz_all = np.abs(sweep[:, :, 2] - (root_now[2] + COMFORT_DZ)).max(axis=0)

    ok = np.flatnonzero((travel >= MIN_TRAVEL_FRACTION * best)
                        & (worst <= STANDING_REACH) & (dz_all <= MAX_DZ))
    if len(ok) == 0:
        # Nothing stays in reach for the whole swing. Fall back to "as reachable
        # as possible", and say so in the notes rather than pretending.
        ok = np.flatnonzero(travel >= MIN_TRAVEL_FRACTION * best)
        reachable_arc = False
    else:
        reachable_arc = True
    k = int(min(ok, key=lambda i: (worst[i] + dz_all[i])))
    contact, goal_pt = here[k], there[k]
    push = goal_pt - contact
    n = float(np.linalg.norm(push))
    push = push / n if n > 1e-6 else np.array([0.0, 0.0, -1.0])

    # PUSH or PULL, from the robot's point of view. A part that has to travel
    # towards the robot cannot be moved by an open palm -- reported so a run of
    # zeros is legible as "this task needs a grasp" instead of as a model failure.
    root = np.asarray(env.sim.data.body_xpos[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_support")],
        dtype=np.float64) if mujoco.mj_name2id(
            m, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_support") >= 0 else contact
    away = contact - root
    away = away / max(float(np.linalg.norm(away)), 1e-6)
    kind = "push" if float(np.dot(push, away)) > -0.15 else "pull (needs a grasp)"

    # ⚠️ Do not ask for the root when the part is already in reach. These tasks
    # park the robot in front of the fixture on purpose, so a root sub-goal is an
    # instruction to walk when there is nowhere to walk to. Measured on
    # `IThorKitchenCloseDrawer`: the two failing episodes had d_xy go 0.76 -> 1.63
    # and 0.59 -> 1.27, i.e. the robot walked AWAY from a drawer it could already
    # touch, while the two that succeeded never moved (travel 0.02).
    #
    # ⚠️ The threshold is `DELIVERABLE_REACH`, NOT `STANDING_REACH`. It was the
    # latter, and 0.80 is what the ARM can span rather than what the generator
    # delivers: past 0.64 m the hand holds its target in 1 of 12 draws instead of
    # 7 of 9, and the tasks ask past it most of the time.
    reach = float(np.linalg.norm(contact - root))
    needs_root = bool(reach > DELIVERABLE_REACH)

    # WHICH HAND. A part on the robot's left pushed with the right hand asks the
    # arm to cross the body -- the longest path, and the one the generator has
    # seen least of. Taken from the sign of the lateral offset in the robot's own
    # frame: computed, not annotated.
    quat = np.asarray(env.sim.data.body_xquat[
        mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "mobilebase0_support")],
        dtype=np.float64)
    w_, x_, y_, z_ = quat
    yaw = np.arctan2(2 * (w_ * z_ + x_ * y_), 1 - 2 * (y_ * y_ + z_ * z_))
    lat = float(np.array([-np.sin(yaw), np.cos(yaw)]) @ (contact - root)[:2])
    side = "left" if lat > 0.05 else "right"
    eef = "LeftHand" if side == "left" else "RightHand"

    label = getattr(env, "TARGET_LANG", "part")
    wp_standoff = DEFAULT_STANDOFF

    # ⚠️ A HINGE SWEEPS AN ARC AND A CHORD IS NOT IT. Two waypoints -- "touch it
    # here, press it to there" -- describe the straight line between the ends,
    # which is exactly right for a drawer (prismatic: the chord IS the path) and
    # wrong for a door. Measured on `IThorKitchenCloseFridge`, a 59 degree swing:
    #
    #     chord 0.964 m, arc 1.007 m, effective radius 0.99 m
    #     chord direction vs the tangent at the start:   29 degrees
    #     the true path leaves the chord by:             0.126 m
    #
    # The palm sits 0.064 m from the fingertips, so a hand tracking the chord has
    # left the door face by the halfway point -- and it is pushing 29 degrees off
    # the direction the door is actually going, which slides along the face
    # instead of driving it. That is a planning error, not a tracking one, and it
    # is why the drawer tasks work and the fridge does not.
    #
    # So a swing is followed round: waypoints along the arc, each aimed with the
    # tangent AT THAT POINT.
    swing = float(abs(now - far)) if int(m.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_HINGE else 0.0
    n_steps = int(np.clip(np.ceil(swing / ARC_STEP), 1, MAX_ARC_STEPS))
    stops = []
    if n_steps > 1:
        qs = np.linspace(now, far, n_steps + 1)
        arc = [moving_points(q)[k] for q in qs]
        d.qpos[adr] = now
        mujoco.mj_forward(m, d)
        for i in range(1, len(arc)):
            step = np.asarray(arc[i]) - np.asarray(arc[i - 1])
            nn = float(np.linalg.norm(step))
            stops.append((np.asarray(arc[i]),
                          step / nn if nn > 1e-6 else push))
    else:
        stops = [(goal_pt, push)]

    waypoints = [
        # Touch it where it is before driving at where it has to go. Aiming
        # straight at the end pose from a metre out has the hand arrive already
        # past the part -- sweeping the space it must move INTO, not the space it
        # is in.
        Waypoint(contact, phase=OPEN, label=f"touch {label}", approach_dir=push,
                 needs_root=needs_root, eef=eef),
    ]
    for i, (pt, tangent) in enumerate(stops):
        # ⚠️ FOLLOW THE PART. A swinging door does not stay put while it is
        # pushed: on `IThorKitchenCloseFridge` the contact point runs from 0.32 m
        # to 1.29 m from the robot as the door closes, and the old plan marked
        # every press waypoint `needs_root=False` -- asking a stationary robot to
        # push something that is leaving. That is impossible before the model is
        # involved at all.
        #
        # So each press asks for the root whenever its own target has moved out
        # of range, at the operating standoff rather than the nominal one.
        # ⚠️ NO ROOT GOAL. These tasks park the robot in front of the fixture and
        # the whole job is done standing: an earlier version asked the root to
        # follow the swinging door and that is not what the task wants. The
        # contact point is now chosen so the arc stays in reach, which is what
        # makes standing still viable -- the two changes only work together.
        waypoints.append(
            Waypoint(pt, phase=OPEN, approach_dir=tangent, eef=eef,
                     needs_root=False,
                     label=(f"press {label}" if len(stops) == 1
                            else f"press {label} {i + 1}/{len(stops)}")))
    return SceneGoals(waypoints=waypoints,
                      lang=getattr(env, "get_ep_meta", lambda: {})().get("lang", ""),
        notes={"joint": env.target_joint, "kind": kind,
               "reach_m": round(reach, 3), "needs_root": needs_root,
               "travel_m": round(float(np.linalg.norm(goal_pt - contact)), 4),
               # Positive = this much further to move. Follows the task's own
               # predicate: a CLOSE task wants |q-closed| under the bar, an OPEN
               # task wants it over, so one formula reports an untouched drawer
               # as already done.
               # Positive = this much further to move, in the joint's own units.
               "to_go_rad": round(float(abs(now - want)), 4),
               "frac_now": round(float(abs(now - closed) / max(rng, 1e-9)), 4),
               "frac_needed": round(want_frac, 3),
               "arc_steps": len(stops),
               "contact_dz": round(float((contact - root_now)[2]), 3),
               "arc_in_reach": bool(reachable_arc),
               "arc_max_reach": round(float(worst[k]), 3),
               "candidates": int(len(ok)),
               "side": side,
               "swing_deg": round(float(np.degrees(swing)), 1),
               "push_dir": [round(float(v), 3) for v in push],
               "bar_point": [round(float(v), 4) for v in at_bar[k]]})


@register("TwoStageNavigation")
def _two_stage(env) -> SceneGoals:
    """Sofa first, THEN the tv stand. Two waypoints, because it is two goals.

    ⚠️ This was originally handled by the single-goal approach adapter, and the
    failure was invisible in the success rate alone: the robot walked to the sofa
    and stopped, scoring 0/5 while ApproachSofa scored 4/5. What gave it away was
    that the two tasks reported root_err_min_median 2.3349 and arrive_rate 0.4444
    IDENTICALLY, to four decimals -- the same trajectory, because the adapter was
    driving at the same single point. Two scenes cannot agree to four decimals by
    chance; that pair of numbers was the bug's fingerprint.
    """
    radius = float(getattr(env, "SUCCESS_RADIUS", 1.0))
    out = []
    for attr, label in (("sofa", "approach sofa"), ("tv_stand", "approach tv stand")):
        fx = getattr(env, attr, None)
        if fx is None or not hasattr(fx, "pos"):
            raise SystemExit(f"[adapters] TwoStageNavigation has no {attr!r}")
        p = np.asarray(fx.pos, dtype=np.float64)
        out.append(Waypoint(np.array([p[0], p[1], 0.9]), phase=OPEN, label=label,
                            approach_only=True, arrive_radius=radius))
    return SceneGoals(waypoints=out,
                      lang=getattr(env, "get_ep_meta", lambda: {})().get("lang", ""))


@register("ApproachSofa", "ApproachTable",
          "IThorLivingRoomApproachSofa", "IThorBedroomApproachBed",
          "IThorKitchenApproachFridge", "IThorBathroomApproachToilet",
          # Taken from the wjb branch once the bathroom floor plans were on disk.
          # Navigation is the one category this pipeline does well -- 93% on
          # ApproachToilet -- so every task that lands in it is close to free.
          "IThorBathroomApproachSink", "IThorBathroomApproachGarbageCan",
          "IThorBedroomApproachDesk", "IThorLivingRoomApproachTable")
def _approach(env) -> SceneGoals:
    """Walk to a piece of furniture and stay there. No hand involved at all.

    These are the cheapest tasks this pipeline can attempt and it is worth being
    precise about why: success is the robot's own geometry coming within
    ``SUCCESS_DISTANCE`` of the fixture's geometry -- `mj_geomDistance`, SURFACE
    to surface, 0.1 m for most of them -- held for ``HOLD_STEPS`` = 10 control
    steps. It does not depend on palm orientation, which is where every grasp
    attempt so far has failed.

    ⚠️ **THIS DOCSTRING USED TO SAY SOMETHING ELSE, AND THE CODE BELIEVED IT.**
    It read "the ROOT's planar distance to a fixture centre under
    ``SUCCESS_RADIUS`` -- 1.0 m", and the code did
    ``getattr(env, "SUCCESS_RADIUS", 1.0)``. **No iThor navigation task defines
    ``SUCCESS_RADIUS``** -- it exists only on the living-room tasks and
    `ithor_atomic/carry.py` -- so every task this adapter is registered for fell
    through to the default. h31 measured what that bar means on the toilet:
    at a root 1.0 m from the centre the surfaces are **0.369 m** apart against a
    0.1 m criterion, i.e. **0 of 12 azimuths would have passed**. The number was
    not merely loose, it was a different measurement.

    So the waypoint now carries the task's own test, and the radius is left as
    the number the report prints. ``tasks.md``'s "判据（根到水槽 <1.0 m）" came
    from this docstring and is wrong for the same reason.

    ⚠️ What they DO depend on is the thing the generator has no information
    about: Kimodo receives no scene geometry, so it walks through furniture. In
    IThorBedroomLiftTeddyBear that showed up as a hip against a wall for 205
    ticks. So these tasks do not dodge the hard part, they swap which hard part
    is being tested -- which is exactly why they are worth running.

    The target is read from the task's own fixture where it exposes one, so the
    point being driven at and the point being judged cannot drift.
    """
    target = _approach_target(env)
    return SceneGoals(
        waypoints=[Waypoint(target, phase=OPEN, label="approach",
                            approach_only=True,
                            arrive_radius=_arrive_radius(env),
                            arrive_test=_arrive_test(env))],
        lang=getattr(env, "get_ep_meta", lambda: {})().get("lang", ""))


def _arrive_test(env):
    """When to STOP WALKING, or None if the task exposes no test of its own.

    ⚠️ The INSTANTANEOUS predicate (``_robot_is_near_target_fixture``), not
    ``_check_success``. This is a release for the walk, and the difference is
    ``HOLD_STEPS`` = 10 control steps of walking that happen after the robot is
    already in range: the final chunk's ask is clamped to the fixture CENTRE, so
    those ten ticks are ten ticks of reference driving the robot deeper.
    Measured on `IThorBathroomApproachToilet` against a 0.1 m criterion, the
    surface distance at the moment ``_check_success`` fires is **0.033-0.036 m**
    -- two thirds of the margin spent, close enough that the settle alone then
    puts a limb on the porcelain.

    ⚠️ **THIS DOCSTRING USED TO ARGUE THE OPPOSITE**, that only the held
    criterion should release the waypoint, because "a waypoint released on the
    instant of arrival would let go of a robot that is still moving through the
    target". That risk is real and is now answered where it belongs: the robot
    stops walking here and then STANDS, tracking a frozen pose, while the task's
    own ``HOLD_STEPS`` accumulate (``h7_scene_rollout.hold_still``). The verdict
    is still ``_check_success`` -- what changed is that the hold is served
    standing rather than walking.
    """
    if getattr(env, "target_fixture", None) is None:
        return None
    near = getattr(env, "_robot_is_near_target_fixture", None)
    if callable(near):
        return lambda e=env: bool(e._robot_is_near_target_fixture())
    fn = getattr(env, "_check_success", None)
    if not callable(fn):
        return None
    return lambda e=env: bool(e._check_success())


#: Release bar for an approach waypoint, metres, root centre to target centre.
#: This is the "did the BODY get there" criterion -- horizontal only, so a robot
#: that crouches does not read as further away.
#:
#: 1.0 m, and the number comes from two measurements pulling in opposite
#: directions:
#:
#:   * h31, per fixture, the largest radius at which every non-penetrating
#:     azimuth ALSO satisfies the task's own surface criterion:
#:         GarbageCan 0.40  Sink 0.50  Toilet 0.60  Desk 0.60
#:         Fridge     0.80  Table 1.00  Bed none    Sofa none
#:   * where the robot ACTUALLY stops on the toilet task, measured over the
#:     runs so far: 0.58, 0.605, 0.692, 0.72 m.
#:
#: A 0.6 m bar sits in the middle of that second list and rejects half of the
#: arrivals a human would call "it is standing at the toilet". Since this bar
#: exists to answer "is the body there", not to predict the task's own verdict
#: (which `success_rate` reports separately and which no radius can reproduce --
#: on ApproachBed a single radius spans surface distances 0.027 to 0.34 m), it
#: is set to enclose the arrivals rather than to chase the surface test.
#:
#: ⚠️ Per run, use `--success-radius`; this is only the default.
DEFAULT_ARRIVE_RADIUS = 1.0


def _arrive_radius(env) -> float:
    """What the report compares `root_err` against. Not the criterion.

    ⚠️ ``SUCCESS_RADIUS`` is read only because the living-room tasks really do
    define it and really do judge on a root radius. For everything else this is
    a printed number, and pretending otherwise is what this whole change is
    undoing.
    """
    r = getattr(env, "SUCCESS_RADIUS", None)
    return float(r) if r is not None else DEFAULT_ARRIVE_RADIUS


def _approach_target(env) -> np.ndarray:
    """The fixture centre the task measures the robot's distance to.

    Resolved by asking the task, in the order of how authoritative the answer is:
    the attribute its own ``success_distance`` reads, then a generic target
    point. Guessing a fixture by name would let the driver aim at one thing while
    the judge scores another, which is the failure ``goals_for`` raises to avoid.
    """
    for attr in ("sofa", "origin_table", "bed", "fridge", "toilet", "sink",
                 "garbage_can", "desk", "table", "target_fixture"):
        fx = getattr(env, attr, None)
        if fx is not None and hasattr(fx, "pos"):
            p = np.asarray(fx.pos, dtype=np.float64)
            return np.array([p[0], p[1], 0.9])       # chest height over the centre
    for attr in ("_target_point", "target_point"):
        fn = getattr(env, attr, None)
        if callable(fn):
            return np.asarray(fn(), dtype=np.float64)
    raise SystemExit(
        f"[adapters] {type(env).__name__} exposes no fixture this adapter knows "
        "how to aim at. Add its attribute to _approach_target rather than "
        "guessing -- driving at one point while the task scores another is the "
        "failure mode this raises to avoid.")


#: Pick-and-place tasks, as (source object, where it has to end up). The second
#: entry is resolved at runtime: a string names another OBJECT (put the apple on
#: the plate), and ``"@target"`` means the task's own ``target_region`` (carry it
#: to the other table).
#:
#: ⚠️ Only the SINGLE-move members of the family are here. `AppleBananaPlateSwap`
#: (apple off the plate, then banana onto it) and `BowlStack` (three bowls in
#: order) need four to six waypoints each, and every one of them has to end in a
#: successful grasp -- at a measured grasp rate of 0/20 on the easiest object in
#: the set, expressing them would be writing a test for a capability that does
#: not exist yet. They go in when a grasp lands.
_PICK_PLACE = {
    "IThorKitchenAppleOnPlate": ("apple", "plate"),
    "IThorKitchenBananaPlateToTable": ("plate", "@target"),
}


@register(*_PICK_PLACE)
def _pick_place(env) -> SceneGoals:
    """Grasp one object, carry it onto another object or surface.

    Two waypoints, and the second one is what makes this different from a lift:
    the hand has to arrive somewhere SPECIFIC while still holding, so the task
    is scored on the object's final position rather than on the hand's.

    ⚠️ The place waypoint does not ask for the root. By then the robot is
    standing where the grasp happened, and a second root sub-goal would walk it
    back out of the pose it just achieved -- the same reason `_room_lift`'s lift
    waypoint does not ask for one.

    ⚠️ These are honest 0s until a grasp lands. h14 puts the closing envelope at
    0.02 m for an apple and 0.12 m for a box, against a measured scene palm
    error of 0.09-0.16 m, so `AppleOnPlate` is being asked for five times the
    precision the pipeline has. Registered anyway, because "never measured" and
    "measured at zero" are different states and only one of them can be tracked.
    """
    src, dst = _PICK_PLACE[type(getattr(env, "unwrapped", env)).__name__]
    obj = _object_pos(env, src)
    if dst == "@target":
        region = getattr(env, "target_region", None)
        if region is None:
            raise SystemExit(
                f"[adapters] {type(env).__name__} exposes no target_region; the "
                "place waypoint would be aimed at a guess.")
        centre, _ = region.world_pose(env.sim)
        goal = np.asarray(centre, dtype=np.float64).copy()
        # Clear of the surface by the object's own half-height, so the ask is
        # "hold it above the table", not "push it through the table".
        goal[2] += 0.12
    else:
        goal = _object_pos(env, dst)
        goal[2] += 0.08

    return SceneGoals(waypoints=[
        Waypoint(obj, phase=CLOSE, label=f"grasp {src}", grasp_object=src),
        Waypoint(goal, phase=HOLD, label=f"place {src} -> {dst.lstrip('@')}",
                 needs_root=False, grasp_object=src),
    ], lang=str(env.get_ep_meta().get("lang", "")))


def _object_pos(env, name: str) -> np.ndarray:
    """World position of a placed object, via its free joint."""
    obj = env.objects[name]
    return np.asarray(
        env.sim.data.get_joint_qpos(obj.joints[0])[:3], dtype=np.float64)


def registered() -> list[str]:
    return sorted(_REGISTRY)
