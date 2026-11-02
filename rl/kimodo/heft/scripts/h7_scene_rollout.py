"""h7: a whole task scene, driven by Kimodo through the ported HEFT tracker.

h6 established that the fine-tuned weights still help once the tracker is the
ported one, in a bare arena with a sampled wrist target. This adds the three
things a task has that an arena does not:

  * **the goal comes from the scene**, via ``adapters.goals_for(env)`` -- the
    same ``_obj_pos`` / ``_target_point`` the task's own ``_check_success``
    is computed from, so the robot is driven at the criterion it is judged by
    rather than at a guess;
  * **it takes more than one chunk**, so the chunk boundary is exercised;
  * **the hand has to close**, which is a channel the tracker does not touch.

Success is polled every control step and the RISING EDGE is recorded, matching
``EpisodeRecorder``: a task that succeeds and then drops the object still
succeeded, and a run that only checked at the end would miss it.

    conda activate kimodo_rl
    python -m rl.kimodo.heft.scripts.h7_scene_rollout --env SquatPickPlaceTVStand \\
        --ckpt rl/kimodo/_out/heft/ckpt/old_lr1e-5_nohands_kimodo_best.pt
    python -m rl.kimodo.heft.scripts.h7_scene_rollout --list
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np
from dataclasses import replace

#: Where the `kimodo_rl` package lives (the generator, the actor, the text cache,
#: the goal sampler). It is a separate repo on purpose -- the weights were
#: trained against that code and importing it is what makes a rollout here
#: comparable with a training number -- and it is put on `sys.path` at the
#: bottom of `main` rather than installed.
#:
#: Select the external checkout explicitly; no machine-specific fallback.
HEFT_WORK = os.environ.get("HEFT_WORK", "")

#: Whether the pose ask includes a body turn. Palm-pointing and facing are
#: separable and were measured separately: palm alone takes the hand from 34 to
#: 13-19 degrees off target, facing alone turns the body but makes the hand worse
#: (nothing corrects for the rotation it applies to the arm), and the two together
#: land at 17-23. Kept as a switch because which one a scene needs depends on
#: whether the object is in front of the robot already.
FACE_ASK = True

#: Grip trigger per phase. The tracker owns 29 body DoF; these 14 finger DoF are
#: an independent channel, so a scripted grasp cannot perturb what it is doing.
GRIP = {"open": 0.0, "close": 1.0, "hold": 1.0, "release": 0.0}
RELEASE = "release"


_BRIDGE = None


def _wrist_world_from_ref(r, wi: int) -> np.ndarray:
    """[T, 3] the reference's wrist in the CHUNK's own frame, rotation included.

    ⚠️ `local_body_pos` IS ROOT-FRAME. The bridge says so in its own header --
    "R_root^T (x_world - root), i.e. the FULL root frame" -- so recovering a
    position takes `R_root @ local + root`, and this code added `root` alone.

    The omission is invisible in an empty room and only there. `EmptyEnv` spawns
    the robot facing +x with an identity root quaternion, so R_root is the
    identity and the two agree to 0.035 m. A scene spawns it facing whatever the
    room dictates -- FloorPlan13 is +85 degrees -- and then the error is the
    wrist's own offset from the pelvis, rotated by the room's heading:

        empty room   ref wrist - root [ 0.088, -0.165]   robot [ 0.055, -0.175]
        FloorPlan13  ref wrist - root [ 0.034, -0.159]   robot [ 0.185, +0.072]

    Rotating the scene's reference offset by +85 gives [0.161, 0.020] against the
    robot's [0.185, 0.072] -- agreement to 0.057, the same order as the empty
    room's residual. The bridge's header even warns that a yaw-only frame is off
    by 0.63 m; the measured tick-0 disagreements were 0.28 to 0.70.

    ⚠️ EVERY SCENE `ref_err` AND `track_err` PRINTED BEFORE THIS COMMIT IS WRONG,
    by a constant of roughly this size. That is what four failed explanations were
    chasing: the heading, the prefix, the ask distance and the pose constraints
    were all being asked to account for a flat ~0.54 that was this.
    """
    import scipy.spatial.transform as _st

    R = _st.Rotation.from_quat(np.asarray(r["root_rot"], dtype=np.float64))
    local = np.asarray(r["local_body_pos"], dtype=np.float64)[:, wi]
    return R.apply(local) + np.asarray(r["root_pos"], dtype=np.float64)


def _reference_wrist_err(qpos_30hz: np.ndarray, wrist_goal_local: np.ndarray,
                         root_l: np.ndarray, side: str = "right") -> float:
    """Closest the GENERATED reference's own wrist gets to the goal. FK, no physics.

    ``root_l`` is the robot's root at the time the chunk was generated, and it is
    not optional. A Kimodo chunk comes back in ITS OWN frame: ``_encode_continue``
    re-anchors the prefix so its first frame sits at the origin, and subtracts the
    same offset from the goal, which is why the drive is translation-invariant and
    correct. The chunk's coordinates are therefore relative to wherever the robot
    was standing, not to the scene.

    Comparing the chunk against a scene-frame goal without undoing that shift
    inflates the error by exactly the robot's accumulated travel -- which showed
    up as ref_err GROWING chunk after chunk (1.45 -> 1.96 -> 2.68) while the robot
    was visibly closing on the target, i.e. reading as "the generator degrades as
    the episode goes on" when nothing of the sort was happening.

    Only XY is undone. Kimodo's re-anchoring zeroes its up axis (``off[1] = 0``),
    so heights stay absolute in the floor-referenced frame.
    """
    global _BRIDGE
    from rl.kimodo.heft_bridge import HeftReference

    # ⚠️ `side` IS NOT OPTIONAL WHEN THE ASK NAMES A HAND. This measured the
    # right wrist unconditionally, so every number reported for a LEFT-hand ask
    # was the RIGHT wrist's distance to a goal the left hand was told to reach --
    # which read as "the left hand is bad even in the base model, ref 0.473
    # against 0.090". It is an artifact of this line, not a property of Kimodo.
    if _BRIDGE is None:
        _BRIDGE = HeftReference(["pelvis", "left_wrist_yaw_link",
                                 "right_wrist_yaw_link"])
    r = _BRIDGE.from_qpos36(np.asarray(qpos_30hz, dtype=np.float32),
                            src_fps=30.0, resample_to=50.0)
    wi = list(r["body_names"]).index(f"{side}_wrist_yaw_link")
    w = _wrist_world_from_ref(r, wi)
    goal = np.asarray(wrist_goal_local, dtype=np.float64).copy()
    goal[0] -= float(root_l[0])
    goal[1] -= float(root_l[1])
    return float(np.linalg.norm(w - goal[None], axis=1).min())


def _reference_wrist_path(qpos_30hz: np.ndarray, root_l: np.ndarray,
                          side: str = "right") -> np.ndarray:
    """[T, 3] where the REFERENCE's own wrist is, tick by tick, in LOCAL frame.

    The partner of ``_reference_wrist_err``, and the reason both exist: comparing
    the robot's wrist against this, at the same tick, is the only measurement
    that separates the two failures this project keeps confusing.

        reference far from the goal, robot ON the reference   -> GENERATION
        reference on the goal, robot far from the reference   -> EXECUTION

    Everything reported until now conflated them, because ``ref_err`` is measured
    against the goal and ``wrist_err`` is measured against the goal, and two
    distances to a third point say nothing about the distance between the two.

    Same anchoring undo as ``_reference_wrist_err``: a chunk comes back in its
    own frame with the prefix's root at the origin, so XY is shifted back by the
    root at generation time. Heights are already floor-referenced.
    """
    global _BRIDGE
    from rl.kimodo.heft_bridge import HeftReference

    # ⚠️ `side` IS NOT OPTIONAL WHEN THE ASK NAMES A HAND. This measured the
    # right wrist unconditionally, so every number reported for a LEFT-hand ask
    # was the RIGHT wrist's distance to a goal the left hand was told to reach --
    # which read as "the left hand is bad even in the base model, ref 0.473
    # against 0.090". It is an artifact of this line, not a property of Kimodo.
    if _BRIDGE is None:
        _BRIDGE = HeftReference(["pelvis", "left_wrist_yaw_link",
                                 "right_wrist_yaw_link"])
    r = _BRIDGE.from_qpos36(np.asarray(qpos_30hz, dtype=np.float32),
                            src_fps=30.0, resample_to=50.0)
    wi = list(r["body_names"]).index(f"{side}_wrist_yaw_link")
    w = _wrist_world_from_ref(r, wi)
    w = np.asarray(w, dtype=np.float64).copy()
    w[:, 0] += float(root_l[0])
    w[:, 1] += float(root_l[1])
    return w


def _tracking_error(ref_local: np.ndarray, wrist_world: np.ndarray,
                    tr) -> tuple[float, float]:
    """(median, max) distance between the robot's wrist and the reference's.

    Tick-aligned over the overlap, so a chunk cut short by a fall is compared
    only where both exist rather than being padded with its own last frame --
    padding would score the moment of falling as perfect tracking.
    """
    if ref_local is None or wrist_world is None or len(ref_local) == 0:
        return float("nan"), float("nan")
    ref_w = np.asarray([tr.to_world(p) for p in ref_local], dtype=np.float64)
    n = min(len(ref_w), len(wrist_world))
    if n == 0:
        return float("nan"), float("nan")
    d = np.linalg.norm(np.asarray(wrist_world)[:n] - ref_w[:n], axis=1)
    return float(np.median(d)), float(d.max())


def _azimuth(delta_xyz, quat_wxyz) -> float:
    """Bearing to a point in the robot's OWN frame, degrees. +-180 is behind it.

    Worth reporting per waypoint because the prompt vocabulary has no word for
    "turn around": ``prompt_for`` picks the walking prompt on distance alone, so
    a goal behind the robot is asked for with "walks forward".
    """
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    az = np.degrees(np.arctan2(delta_xyz[1], delta_xyz[0]) - yaw)
    return float((az + 180.0) % 360.0 - 180.0)


def _scene_contacts(tr) -> int:
    """Robot-vs-scene contacts, excluding ground and walls.

    Filtered on the GEOM name, not the body name. robocasa attaches the floor and
    wall geoms to the worldbody, so every one of them reports a body called
    ``world`` -- a body-name filter passes them all through and reports a robot
    standing quietly on the floor as eight collisions with the furniture. That
    misreading is what made a pure over-travel look like a collision problem.
    """
    import mujoco

    model, data = tr.model, tr.data
    n = 0
    for c in range(data.ncon):
        g1, g2 = data.contact[c].geom1, data.contact[c].geom2
        b1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g1]) or ""
        b2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, model.geom_bodyid[g2]) or ""
        r1, r2 = b1.startswith("robot0_"), b2.startswith("robot0_")
        if r1 == r2:
            continue
        gn = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g2 if r1 else g1) or ""
        if gn.startswith("floor") or gn.startswith("wall"):
            continue
        n += 1
    return n


#: How far a single 3 s chunk actually carries the robot. Measured, not assumed:
#: on goals at d_xy 2.6-4.7 m the GENERATED reference's own wrist ends
#: ref_err ~= d_xy - 1.5, i.e. Kimodo covers about 1.5 m and stops -- the
#: shortfall is in the reference, before physics. So a waypoint further than this
#: has to be approached over several chunks; handing the whole distance to one
#: chunk asks for a motion that does not exist.
CHUNK_REACH = 1.5


#: How far the retry is allowed to shift the aim in total. Half the measured
#: palm offset: past that the correction is no longer describing where the object
#: is relative to the hand, it is describing a reach that never got close.
GRASP_RETRY_CAP = 0.15

#: How near the palm has to get before "retry until you hold it" is the right
#: rule. Beyond this the problem is not the grasp, it is that the robot has not
#: arrived, and re-attempting burns the chunks that would have got it there.
GRASP_RETRY_GATE = 0.30


#: Robot bodies that count as "the hand did it". Everything else touching the
#: moving part is the body bumping into furniture.
HAND_BODIES = ("wrist", "hand", "thumb", "index", "middle", "ring", "pinky", "palm")


def _articulation_touchers(tr, joint_name: str) -> dict:
    """Which robot bodies are touching the part the task scores, right now.

    A task predicate cannot tell a manipulation from a body check, and neither can
    the physics filter -- knees touching furniture is deliberately allowed there,
    because a robot reaching over a coffee table legitimately grazes it. So which
    part of the robot moved the thing has to be measured on its own.

    ⚠️ It was written to confirm a suspicion and refuted it. `IThorKitchenCloseDrawer`
    succeeded at tick 41 and 56, before the first chunk had finished, and the
    physics culprit list showed `right_knee_link` against a drawer for 47 ticks --
    so the successes looked like the robot walking into the cabinet. Measured
    against THE scored joint's own body, both are

        right_wrist_yaw_link 52 ticks + left_hand_index 5     body 0
        right_wrist_yaw_link 21 ticks + right_hand_index 2    body 0

    The knee was against a DIFFERENT drawer. A kitchen has many, and the culprit
    list is not filtered to the one being scored.

    ⚠️ It does catch the pattern elsewhere: on `IThorKitchenCloseFridge` the door
    collects 994 body ticks against 583 hand ticks, and the physics filter rejects
    5 of 6 episodes.
    """
    import mujoco

    m, d = tr.model, tr.data
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return {}
    bid = int(m.jnt_bodyid[jid])
    out = {}
    for c in range(d.ncon):
        g1, g2 = d.contact[c].geom1, d.contact[c].geom2
        b1, b2 = int(m.geom_bodyid[g1]), int(m.geom_bodyid[g2])
        if bid not in (b1, b2):
            continue
        other = b2 if b1 == bid else b1
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, other) or ""
        if name.startswith("robot0_") or name.startswith("gripper0_"):
            out[name] = out.get(name, 0) + 1
    return out


def _articulation_progress(tr) -> dict:
    """How far the task's target joint has travelled, and how far it still has to.

    Empty for a task with no articulation, so the column simply does not appear
    rather than reporting a zero that reads like "it did not move".
    """
    env = getattr(tr, "env", None)
    name = getattr(env, "target_joint", None)
    if not name:
        return {}
    try:
        import mujoco

        jid = mujoco.mj_name2id(tr.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            return {}
        q = float(tr.data.qpos[int(tr.model.jnt_qposadr[jid])])
        # The task's own predicate, which is a FRACTION of the joint's travel
        # since the vetted articulation base was adopted -- `close` wants <= 0.05
        # and `open` wants >= 0.90, and `MIN_SUCCESS_QPOS_DELTA` no longer exists.
        j = env.target_semantic_joint
        closed, opened = float(j.closed_qpos), float(j.open_qpos)
        frac = abs(q - closed) / max(abs(opened - closed), 1e-9)
        opening = getattr(env, "BEHAVIOR", "close") == "open"
        need = (float(env.SUCCESS_OPEN_FRACTION) if opening
                else float(env.SUCCESS_CLOSE_FRACTION))
        # Positive = this much more of the travel still to cover.
        to_go = (need - frac) if opening else (frac - need)
        return {"joint_qpos": round(q, 4), "joint_frac": round(frac, 4),
                "joint_to_go": round(to_go, 4)}
    except Exception:  # noqa: BLE001  -- a diagnostic must never fail a rollout
        return {}


def drive_chunk(tr, actor, wp, prefix, n_frames: int, n_ticks: int, *, reboot: bool,
                grasp=None, use_palm_axis: bool = False, pose_ask: bool = False,
                standoff_frac: float | None = None,
                physics=None, watchers=(), on_goal=None, aim_correction=None,
                tries: int = 1, root_gain: float = 1.0, hand: str = "auto",
                canon_mode: str = "auto", success_hold: int = 0):
    """Aim one Kimodo chunk at one waypoint from wherever the robot is now.

    ``aim_correction`` is a world-frame vector added to the grasp point: the
    second-attempt half of "reach, miss, reach again". It is measured, not
    guessed -- the residual between where the palm actually closed and where the
    object actually was, on the previous attempt.
    """
    import torch

    from kimodo_rl.eef_goals import EefGoalSampler
    from kimodo_rl.prompts import (LEAN, LEAN_L, RAISE, RAISE_L, REACH, REACH_L,
                                   TURN_BEARING, TURN_WALK, WALK, WALK_L,
                                   WALK_ONLY, prompt_for)

    # Right-handed prompt -> its left-handed twin. Built from the module's own
    # constants so a change there cannot leave this silently mapping to nothing.
    LEFT_PROMPT = {REACH: REACH_L, LEAN: LEAN_L, WALK: WALK_L, RAISE: RAISE_L}

    from rl.kimodo.heft.canon import Canonical, yaw_of as canon_yaw_of
    from rl.kimodo.heft.outcome import classify, eef_feasible
    from rl.kimodo.heft.travel import best_chunk

    from rl.kimodo.heft.scene_frame import (approach_from_root, free_standoff,
                                            palm_axis, wrist_goal)

    root_l = tr.root_pos_local()
    grasp_l = tr.to_local(wp.grasp_point)
    if aim_correction is not None:
        grasp_l = grasp_l + np.asarray(aim_correction, dtype=np.float64)
    # The wrist goal, not the object's centre: the tracker controls the chain up
    # to wrist_yaw_link and nothing beyond it, so aiming the wrist AT the object
    # puts the palm through it. PALM_OFFSET is measured, not guessed.
    #
    # WHICH direction to offset along is the harder half. The first chunk has to
    # guess it from the geometry, because the hand has not moved yet. From the
    # second chunk on, the hand's REALIZED axis is available and is the better
    # answer: Kimodo constrains end-effector position and inherits orientation
    # from the prefix, so the guess is reliably ~30 degrees out and leaves the
    # object beside the fingers rather than between them. Re-aiming along the
    # realized axis makes the rolling horizon a fixed-point iteration on the one
    # quantity the generator will not take as an instruction.
    # A LOCOMOTION waypoint has no hand goal. Giving it one would ask a 3 s chunk
    # to walk AND arrive reaching -- the ask that does not fit -- and the hand is
    # not in the success criterion anyway, which is the root's own distance.
    approach_only = bool(getattr(wp, "approach_only", False))
    # ⚠️ WHICH HAND, from the waypoint. This was hardcoded to "RightHand", so an
    # adapter that had worked out the part was on the robot's LEFT still asked
    # the right arm to cross the body for it. The adapters compute the side; the
    # driver has to honour it or the computation is decoration.
    side = "left" if "left" in str(getattr(wp, "eef", "")).lower() else "right"
    eef_name = "LeftHand" if side == "left" else "RightHand"
    # A waypoint that KNOWS which way its target has to move outranks both the
    # guess and the realized axis: re-aiming along the palm's own direction is a
    # fixed-point iteration towards wherever the hand happens to point, which for
    # a press is exactly the quantity being corrected.
    ask_dir = getattr(wp, "approach_dir", None)
    axis = (np.asarray(ask_dir, dtype=np.float64) if ask_dir is not None
            else palm_axis(tr.model, tr.data, tr.prefix, side)
            if use_palm_axis else approach_from_root(grasp_l, root_l))
    wrist_l = None if approach_only else wrist_goal(grasp_l, axis)

    # ⚠️ WHICH HAND IS A PER-CHUNK DECISION, like every other geometric quantity
    # here. The adapter picks one when the episode is built and the robot turns
    # while it works: measured on `IThorKitchenCloseFridge`, the adapter chose
    # the right hand at reset and by the time the chunk ran the target sat behind
    # the robot's LEFT shoulder, with the right arm asked to cross the body.
    #
    # ⚠️ Judged on the PART, not on the wrist goal. Reaching for the wrist goal
    # sounds like the right question -- that is where the hand ends up -- but it
    # gives the wrong answer on a door: the goal sits a palm's length back along
    # the push, which drags it towards the midline, and a fridge door lying at
    # +150 degrees was answered with a wrist goal at +19 and therefore the RIGHT
    # arm. The hand has to get AROUND the part to its far face, so the arm that
    # can do it is the one on the PART's side.
    #
    # The dead band keeps a target near the midline on whichever hand the plan
    # committed to instead of flapping chunk to chunk.
    az_hand = None
    if hand in ("left", "right"):
        # Forced from the command line: the geometry is overruled on purpose, so
        # a plan can be tested against a human's reading of the scene.
        side = hand
        eef_name = "LeftHand" if side == "left" else "RightHand"
    elif wrist_l is not None:
        az_hand = _azimuth(grasp_l - root_l, tr.root_quat())
        if az_hand > HAND_SWITCH_DEG:
            side = "left"
        elif az_hand < -HAND_SWITCH_DEG:
            side = "right"
        eef_name = "LeftHand" if side == "left" else "RightHand"

    # P2-7. Say the other half out loud: which way to face, and which way to hold
    # the hand. Both are channels EndEffectorConstraintSet writes anyway; leaving
    # them at the prefix's values is an instruction to keep the current heading
    # and the current palm direction, not an absence of one.
    hand_fix, body_yaw = None, None
    if pose_ask and wp.grasp_object is not None and not approach_only:
        from rl.kimodo.heft.grasp_pose import grasp_pose_for

        ask = grasp_pose_for(tr, wp.grasp_point, face=FACE_ASK)
        hand_fix, body_yaw = ask["hand_fix"], ask["body_yaw"]
        # Re-derive the wrist goal along the direction we are now ASKING the palm
        # to point, so position and orientation agree by construction.
        wrist_l = wrist_goal(grasp_l, ask["hand_dir"])

    # free_standoff, not root_standoff: the nominal 0.45 m standoff was measured
    # against the arm's reach in an empty arena, and in a living room the point
    # it lands on can be inside the coffee table. It probes for collisions by
    # writing the free joint, so it works in WORLD coordinates while the
    # generator wants the answer in local -- the two frames coincide only while
    # the spawn is at the origin, which is not something to rely on.
    # ⚠️ EVERY reach gets a root sub-goal, including the ones the waypoint marks
    # `needs_root=False`. The code used to send none, reasoning that "by then the
    # robot is where it needs to be and a second root sub-goal would only walk it
    # back out of the pose it just achieved". Both halves of that are wrong, and
    # both were measured:
    #
    #   * it does NOT stay put. With the walk leg on, `IThorKitchenAppleOnPlate`
    #     went leg 0.07 -> reach 0.50 -> 0.82 -> 1.00 -> 1.54 m from the target,
    #     median root travel 0.142 m per reach chunk, max 0.732.
    #   * Kimodo closes eef distance by WALKING. Without a root goal it barely
    #     extends the arm: h25's sweep, same asks with and without one --
    #
    #         d          0.50    0.60    0.70    0.80
    #         no root   0.206   0.268   0.326   0.403
    #         root      0.170   0.166   0.149   0.191
    #
    #     Under 0.45 m the two are IDENTICAL, because a root goal at the robot's
    #     own position is silently dropped -- "stand still" is not expressible.
    #
    # So `needs_root=False` now means "aim the root at the standoff point rather
    # than at the target", not "say nothing about the root".
    root_goal, used_standoff = (None, None)
    if True:
        # ⚠️ AN APPROACH USED TO ASK FOR standoff 0, i.e. it aimed the root at the
        # fixture's CENTRE -- a point inside the furniture. `free_standoff` walks
        # it out only until the bodies clear (at most 0.35 m), so the robot was
        # told to walk into the toilet, could never get there, and kept pushing
        # at the edge for the rest of its chunk budget. That is the "lingering at
        # the boundary" this fixes, and it also explains the bearing flipping
        # sign between chunks (-47.6 deg then -8.9 deg on one waypoint): a goal
        # underfoot changes azimuth wildly for a few centimetres of shuffle,
        # which then trips the turn prompt.
        #
        # Aim at a point INSIDE the arrival ring but OUTSIDE the fixture:
        # `APPROACH_STANDOFF_FRAC` of the bar, leaving the rest as margin for the
        # tracker's undershoot. `free_standoff` still pushes it further out if
        # even that lands in geometry.
        stand = wp.standoff
        if approach_only:
            bar = float(getattr(wp, "arrive_radius", None) or 1.0)
            stand = (APPROACH_STANDOFF_FRAC if standoff_frac is None
                     else float(standoff_frac)) * bar
        # A waypoint already walked to keeps its standoff -- the goal then lands
        # on or near where the robot is standing, which costs nothing, and pulls
        # it back in when a previous chunk has drifted.

        xy_w, used_standoff = free_standoff(
            tr.env, np.asarray(wp.grasp_point, dtype=np.float64),
            tr.data.xpos[tr.bid_root].copy(), stand)
        root_goal = np.asarray(xy_w, dtype=np.float32) - tr.origin_xy.astype(np.float32)
        if root_gain and root_gain != 1.0:
            # ⚠️ ASK FOR MORE THAN YOU WANT, because the tracker delivers about
            # half. Measured over 207 navigation chunks: the generated reference
            # walks its root to 0.034 m of the goal and the ROBOT stops 0.425 m
            # out; reference travel 1.17 m against robot travel 0.59 m -- 50%
            # delivery with NO furniture contact at all, 38-40% with it.
            #
            # The frozen tracker observes neither root position nor root linear
            # velocity (they are privileged), so root translation is open loop
            # and lags. Compensating is exactly what the RL fine-tuning exists to
            # learn; this is the same compensation applied by hand, at the ask.
            here_l = np.asarray(root_l[:2], dtype=np.float32)
            root_goal = here_l + (root_goal - here_l) * float(root_gain)

    local = EefGoalSampler.to_pelvis_frame(
        (grasp_l if wrist_l is None else wrist_l)[None], root_l[None])
    d_xy = float(np.linalg.norm((grasp_l - root_l)[:2]))
    # ⚠️ WHICH WAY THE GOAL IS decides the prompt. Hardcoding "walks forward"
    # for every approach is how the robot ends up MOONWALKING: h20 measured a
    # goal 1.5 m at 180 degrees answered with 1.55 m of backward travel and no
    # turn at all. `TURN_WALK` at the same bearing costs 0.03 m of closure and
    # takes the backward travel to 0.09 m -- and it wins from 45 degrees up, so
    # the threshold is set by the measurement, not by the word "around".
    if approach_only and root_goal is not None:
        # ⚠️ ASK FAR ENOUGH TO MOVE, then let the undershoot land it. Delivery is
        # a step function of the ask (1838 chunks): 33-40% below 1.2 m, 66%
        # above. Moving the aim point off the fixture centre and onto a standoff
        # shortened the final leg from a median 1.154 m to 0.673 m -- straight
        # into the flat part -- and the symptom was the robot hovering at the
        # ring's edge for a chunk or two before stepping in.
        #
        # So the AIM is unchanged (still the reachable standoff) and only the
        # REQUESTED distance is stretched, along the same direction, to
        # MIN_USEFUL_ASK. The robot delivers a fraction of it and lands near the
        # true standoff; `arrived` still stops it, so the stretched ask cannot
        # turn back into the endless push at an unreachable point that aiming at
        # the centre used to cause.
        here = np.asarray(root_l[:2], dtype=np.float32)
        v = np.asarray(root_goal, dtype=np.float32) - here
        n = float(np.linalg.norm(v))
        if 1e-6 < n < MIN_APPROACH_ASK:
            u = v / n
            # ⚠️ CLAMPED AT THE TARGET CENTRE, never past it. Stretching the ask
            # walks the aim point from the standoff towards the fixture, and the
            # centre is exactly where it used to aim -- a configuration whose
            # physics pass rate was measured at 1.00 over six episodes. Beyond it
            # the aim would be on the far side of the furniture, which is a
            # direction nothing has ever been tested walking in.
            cap = float(np.dot(np.asarray(grasp_l[:2], dtype=np.float32) - here, u))
            root_goal = here + u * min(MIN_APPROACH_ASK, max(n, cap))
    to_goal = (np.array([root_goal[0] - root_l[0], root_goal[1] - root_l[1], 0.0])
               if root_goal is not None else grasp_l - root_l)
    bearing = _azimuth(to_goal, tr.root_quat())
    turning = approach_only and abs(np.radians(bearing)) > TURN_BEARING
    # ⚠️ THE PROMPT CARRIES THE HAND, and `prompt_for` only knows right-handed
    # ones -- "a person reaches out with the RIGHT hand". So a left-hand waypoint
    # was constraining the LEFT wrist while telling the model in words to use the
    # right, and the words won: the right arm moved. `prompts._BY_SIDE` has had
    # the left variants all along; nothing was reading them.
    if approach_only:
        prompt = TURN_WALK if turning else WALK_ONLY
    else:
        prompt = prompt_for(local)[0]
        if side == "left":
            prompt = LEFT_PROMPT.get(prompt, prompt)
    # One batched sample of K, not K samples: the diffusion pass is batched, so
    # candidates cost the GPU almost nothing and the pick is free. Default 1 --
    # h20 measured K=4 adding nothing once the prompt is right (legal rate 0.80
    # either way), so it is available rather than on.
    k = max(1, int(tries))
    # ⚠️ HEADING CANONICALISATION -- MEASURED, AND IT DOES NOT TRANSFER. Off by
    # default. See the block below for what it does buy and where it stops. Generate in the frame the model was trained in -- rotate the
    # prefix and every goal about the robot's own root until it faces +x -- then
    # rotate the chunk back. Measured in the empty room, p41c, n=20 per cell,
    # with the ask held bit-for-bit identical in the ROBOT's frame:
    #
    #     spawn yaw      raw end / dwell      canon end / dwell
    #        90            0.153   6/20         0.093  15/20   <- fully recovered
    #       135            0.224   1/20         0.183   1/20   <- marginal
    #       180            0.876   1/20         2.150   1/20   <- 16 of 20 FELL
    #
    # The yaw-0 baseline is 0.089 / 12-20, so at 90 degrees this recovers the
    # whole loss. At 180 it is far WORSE than doing nothing and the reason is not
    # understood -- so `auto` applies it inside CANON_MAX_DEG and leaves the rest
    # alone rather than pretending the transform is general.
    #
    # That gate is not academic. The four collection rooms spawn at fixed
    # headings, identical across seeds:
    #
    #     kitchen    FloorPlan13    +85     21 tasks
    #     bathroom   FloorPlan401   +85     11 tasks
    #     livingroom FloorPlan209     0      7 tasks   already the trained frame
    #     bedroom    FloorPlan329  -180      9 tasks   worst heading there is
    #
    # ⚠️⚠️ AND YET IT DOES NOT HELP THERE. Run on IThorKitchenAppleOnPlate, which
    # spawns at +85, n=6 episodes, 27 manipulation chunks each way:
    #
    #                   ref_err   palm    reach_end   root_err
    #     --canon off    0.544    0.359     0.575      0.054
    #     --canon auto   0.566    0.377     0.550      0.046
    #
    # Nothing, or slightly worse, on the metric the empty room said would halve.
    # Two differences between the two settings, neither tested: `auto` fired on
    # only 11 of 27 chunks because the heading DRIFTS past the gate mid-episode
    # (the values it did fire on ran 82 to 120), and the scene's prefix is five
    # frames of real motion while the empty-room prefix was a settled stand with
    # one quaternion overwritten -- `Canonical` reads the yaw of the LAST prefix
    # frame and applies it to all five, which is exact only if the robot was not
    # turning during them.
    # How much the heading MOVES across the prefix. `Canonical` reads the yaw of
    # the last frame and applies it to all five, which is exact only if this is
    # near zero -- one of the two live explanations for why canonicalisation
    # halves the error in the empty room (static prefix) and does nothing in a
    # scene (five frames of real motion). Recorded always, so the question can be
    # answered from any run rather than needing a special one.
    _py = np.degrees([canon_yaw_of(np.asarray(f, dtype=np.float64)[3:7])
                      for f in np.asarray(prefix)])
    _py = np.unwrap(np.radians(_py))
    prefix_yaw_spread = float(np.degrees(_py.max() - _py.min()))
    canon = None
    if canon_mode != "off":
        yaw_now = abs(np.degrees(canon_yaw_of(tr.root_quat())))
        if canon_mode == "on" or yaw_now <= CANON_MAX_DEG:
            canon = Canonical(prefix, pivot=root_l)
    gen_prefix = canon.prefix(prefix) if canon else prefix
    gen_wrist = canon.point(wrist_l) if (canon and wrist_l is not None) else wrist_l
    gen_root = canon.xy(root_goal) if (canon and root_goal is not None) else root_goal
    with torch.no_grad():
        s = actor.sample_actions_batch(
            [prompt] * k, n_frames, [gen_prefix] * k,
            [None] * k if gen_wrist is None
            else [(eef_name, np.asarray(gen_wrist, dtype=np.float32))] * k,
            root_goals=None if gen_root is None else [gen_root] * k,
            hand_fixes=[hand_fix] * k, body_yaws=[body_yaw] * k)
    qs = [np.asarray(x["qpos_30hz"], dtype=np.float32) for x in s]
    if canon is not None:
        qs = [canon.unrotate_chunk(q) if np.isfinite(q).all() else q for q in qs]
    pick, reports = best_chunk(qs, root_goal if root_goal is not None else None)
    q, travel = qs[pick], reports[pick]
    base = {"label": wp.label, "phase": wp.phase, "d_xy": round(d_xy, 4),
            "canon": None if canon is None else round(float(np.degrees(canon.yaw)), 1),
            "prefix_yaw_spread": round(prefix_yaw_spread, 2),
            # Which hand was asked for, and in which words. Both, because they
            # are set in different places and disagreed silently: the constraint
            # named the left wrist while the prompt said "with the right hand".
            "eef": eef_name, "side": side,
            "az_hand": None if az_hand is None else round(float(az_hand), 1),
            "body_yaw_deg": None if body_yaw is None else round(float(np.degrees(body_yaw)), 1),
            "azimuth_deg": round(bearing, 1),
            "prompt": "turn" if turning else ("walk" if approach_only else "reach"),
            # What the GENERATED motion does, before any of it is executed. A
            # chunk that walks backwards is a generation defect and no amount of
            # tracking fixes it, so it is recorded here rather than inferred
            # later from the executed root path.
            "travel_back": travel["back"], "travel_side_frac": travel["side_frac"],
            "travel_dyaw": travel["dyaw"], "travel_ok": travel["ok"],
            "travel_why": travel["why"],
            "travel_tries": k,
            "standoff_used": None if used_standoff is None else round(float(used_standoff), 3)}
    if not np.isfinite(q).all() or np.abs(q[:, 0:3]).max() > 50.0:
        return {**base, "unusable": True}, 0, {"success_at": None, "fell": False}

    # The hand is driven per tick by the grasp state machine when the waypoint
    # names an object, and only falls back to the phase schedule when it does
    # not. The fallback is what every waypoint used to get, and it curls the
    # fingers while the hand is still half a metre out.
    min_palm_dist = float("inf")
    best_residual = None
    art_joint = getattr(getattr(tr, "env", None), "target_joint", None)
    touchers: dict = {}
    if grasp is not None and wp.phase == RELEASE:
        grasp.release()
    elif grasp is None:
        g = GRIP.get(wp.phase, 0.0)
        tr.set_grip(g, g)

    # ⚠️ Contacts counted PER TICK across the chunk, not sampled at its end.
    # `_scene_contacts` alone is a snapshot, and a snapshot cannot tell a robot
    # that ploughed through a bed for two seconds from one that brushed a chair
    # on the last frame -- which is exactly the distinction needed to attribute
    # the root shortfall to furniture or to the tracker.
    contact_ticks = [0]

    def _count_contacts(i=None, t=None):
        if _scene_contacts(tr) > 0:
            contact_ticks[0] += 1

    # ⚠️ Registered HERE, before `on_tick` is defined. `on_tick` binds the
    # watcher tuple as a default argument, so anything appended afterwards is
    # captured by nothing and silently never runs.
    watchers = tuple(watchers) + (_count_contacts,)

    # One hook, several consumers. The grasp machine and the physics filter both
    # need per-tick state and neither can be hoisted to chunk granularity: "has
    # the hand arrived" and "did anything clip" are both about instants between
    # steps. `watchers` are passive observers (viewer, state recorder) -- they
    # run LAST and must never influence what the others see.
    def on_tick(i, _tr, _g=grasp, _p=physics, _w=tuple(watchers)):
        nonlocal min_palm_dist, best_residual
        if _g is not None:
            d_now = _g.step()["dist"]
            if d_now < min_palm_dist:
                min_palm_dist = d_now
                # The vector from where the hand closed to where the object was.
                # A distance says "it missed"; this says WHICH WAY, which is the
                # only part a second attempt can act on.
                best_residual = _g.object_pos() - _g.palm_pos()
        if _p is not None:
            _p.step()
        if art_joint:
            for b, n in _articulation_touchers(_tr, art_joint).items():
                touchers[b] = touchers.get(b, 0) + n
        for w in _w:
            w(i, _tr)
    # Publish this chunk's targets before it runs, so a recorder or viewer can
    # draw them. The goal is the WRIST goal (a palm's length short of the
    # object), which is what the chunk was actually asked for -- drawing the
    # object alone would make every correct reach look 0.13 m short.
    if on_goal is not None:
        # ⚠️ AN APPROACH WAYPOINT HAS NO HAND GOAL, AND THE PICTURE MUST NOT
        # INVENT ONE. This used to pass `wp.grasp_point` into the HAND slot when
        # there was no wrist goal, so a navigation task drew a white "right hand
        # target" ball on the toilet -- next to the red root-goal ball at
        # essentially the same point, since an approach waypoint has standoff 0.
        # Kimodo is genuinely given no eef constraint here (`gen_wrist` is None
        # above); the ball was the drawing, not the ask, and it read as the
        # opposite. The original bug this replaced was a CRASH -- `to_world(None)`
        # raises `IndexError` -- and the fix aimed at the wrong slot.
        # Three things: the hand goal, the object, and where the BODY was asked
        # to go. Everything that works is scored on the root.
        on_goal(None if wrist_l is None else tr.to_world(wrist_l),
                None if wrist_l is None else wp.grasp_point,
                None if root_goal is None else tr.to_world(
                    np.array([root_goal[0], root_goal[1], 0.0])),
                # Which hand, so the two end-effector goals get their own colours.
                "left" if "left" in str(getattr(wp, "eef", "")).lower() else "right")
    tr.load_chunk(q, reboot=reboot)
    # An approach stops the moment it has arrived; everything else runs its full
    # chunk. Only locomotion, because a reach or a grasp is not finished when the
    # hand first gets somewhere -- the grasp state machine needs the rest of the
    # chunk to close, and cutting it short would be cutting the task off.
    stop_when = None
    if approach_only:
        if wp.arrive_test is not None:
            reached = wp.arrive_test
        elif wp.arrive_radius:
            bar, tgt = float(wp.arrive_radius), np.asarray(grasp_l[:2])
            reached = (lambda: float(np.linalg.norm(
                np.asarray(tr.root_pos_local()[:2]) - tgt)) <= bar)
        else:
            reached = None
        # ⚠️ RELEASED ON THE FIRST TICK THE PREDICATE FIRES, and the hold is
        # served STANDING STILL afterwards (`_hold_still`) rather than by
        # counting ticks here. Counting them here was measured, and what it
        # counted was the robot walking further in: the ask on the final chunk is
        # clamped to the fixture CENTRE, so every tick of hold is a tick of
        # reference that pushes deeper. Over three episodes of
        # `IThorBathroomApproachToilet` the criterion fired at tick 707 of 732
        # with the surface 0.035 m away, and the 25 held ticks took it to
        # **0.000** -- i.e. the hold was exactly the window in which the robot
        # closed the last 3.5 cm and leaned on the porcelain. 22-26 ticks per
        # episode sat under 0.02 m and all of them were inside that window.
        stop_when = reached
    res = tr.rollout(n_ticks, track=side, on_tick=on_tick, stop_on_success=False,
                     success_hold=success_hold, stop_when=stop_when)

    # Three errors, and the triple is the diagnosis. Measuring only the last of
    # them says "it did not work" without saying which stage failed.
    #   ref_err    the GENERATED reference's own wrist vs the wrist goal -- did
    #              Kimodo even aim there? Large means the ask was outside what
    #              the prompt and constraints can express, e.g. a target 3 m away
    #              handed to a chunk that covers ~1.5 m.
    #   wrist_err  the ROBOT's wrist vs the same goal. This is h6's exec_err, so
    #              scene numbers are comparable to arena ones.
    #   hand_err   the robot's wrist vs the OBJECT. Necessarily >= PALM_OFFSET
    #              when the reach is perfect, because the wrist is meant to stop
    #              a palm's length short. Never compare it against exec_err.
    # An approach waypoint is judged on the ROOT, so the wrist columns would be
    # noise dressed as measurement -- reported as None rather than as a number
    # nobody should read.
    #
    # ⚠️ But the ref/exec split still has to be reported, on the root instead of
    # the wrist. Without it a stalled approach reads as one undifferentiated
    # "it did not move", and the two causes need opposite fixes: a reference that
    # never asks for the travel is a GENERATION problem, and a reference that
    # asks for it while the robot stays put is an EXECUTION one. `IThorLivingRoom
    # LiftStatue` sat in exactly that ambiguity -- hops behind the robot, 0.02-0.30 m
    # of travel per 3 s chunk against 0.6-1.2 m on a task that works -- and there
    # was no column that could tell the two apart.
    if approach_only:
        d_root = np.linalg.norm(
            res["root_world"][:, :2] - np.asarray(wp.grasp_point)[None, :2], axis=1)
        # The chunk comes back in its own frame, anchored so the prefix's root is
        # the origin (`_encode_continue` subtracts `smooth[0]`), which is the same
        # re-anchoring `_reference_wrist_err` undoes. The prefix IS the robot's
        # current state, so chunk-frame + root_l is the local frame.
        ref_xy = np.asarray(q, dtype=np.float64)[:, :2] + np.asarray(root_l)[None, :2]
        d_ref = np.linalg.norm(ref_xy - np.asarray(grasp_l)[None, :2], axis=1)
        row = {**base, "unusable": False, "approach": True,
               "root_err_min": round(float(d_root.min()), 4),
               "root_err_final": round(float(d_root[-1]), 4),
               "arrive_radius": wp.arrive_radius,
               # ⚠️ THE TASK'S OWN IN-RANGE TEST WHEN IT HAS ONE -- "the body got
               # there", which is what a WALK can be judged on. The task's
               # verdict is `success`, and it needs the hold that `hold_still`
               # serves after this row is written. A radius on the root is
               # a different measurement from the surface distance these tasks
               # judge, and h31 measured the gap: at the 1.0 m bar that used to
               # be here, `IThorBathroomApproachToilet` passes on 0 of 12
               # azimuths. Falling back to the radius only for the planner's own
               # hops, which the task has no opinion about.
               "arrived": bool(wp.arrive_test() if wp.arrive_test is not None
                               else d_root.min() <= (wp.arrive_radius or 1.0)),
               "arrive_by": "task" if wp.arrive_test is not None else "radius",
               "ref_err": None, "wrist_err_min": None, "hand_err_min": None,
               "ref_root_err_min": round(float(d_ref.min()), 4),
               "ref_root_err_final": round(float(d_ref[-1]), 4),
               "ref_travel": round(float(np.linalg.norm(
                   ref_xy[-1] - ref_xy[0])), 4),
               "root_travel": round(float(np.linalg.norm(
                   res["root_world"][-1][:2] - res["root_world"][0][:2])), 4),
               "scene_contacts": _scene_contacts(tr),
           "contact_ticks": contact_ticks[0],
               "fell": bool(res["fell"]), "success_at": res["success_at"]}
        # ⚠️ PATH length, not displacement. A robot that walks a loop and comes
        # back has not been "stuck", and labelling it so would hide a real and
        # differently-caused failure.
        path = float(np.linalg.norm(np.diff(res["root_world"][:, :2], axis=0),
                                    axis=1).sum())
        row["root_outcome"] = classify(
            d_start=float(d_root[0]), err_min=float(d_root.min()),
            err_final=float(d_root[-1]), travel=path,
            tol=float(wp.arrive_radius or 1.0))
        row["root_path"] = round(path, 4)
        return row, res["ticks"], res

    goal_w = tr.to_world(wrist_l)
    d_wrist = np.linalg.norm(res["wrist_world"] - goal_w[None], axis=1)
    # An articulated task is scored on an ANGLE, and none of the distance columns
    # can see it: the hand can be 0.03 m from where it was asked to be and have
    # moved the hinge not at all, or sweep past the target and close it anyway.
    # Reported for any task that names a target joint, so a push task's progress
    # is legible in the same log as a reach's.
    art = _articulation_progress(tr)
    if art and touchers:
        hand = {b: n for b, n in touchers.items()
                if any(k in b for k in HAND_BODIES)}
        art["joint_touched_by"] = dict(sorted(touchers.items(), key=lambda kv: -kv[1])[:4])
        # The question a success on this task has to answer.
        art["joint_hand_ticks"] = int(sum(hand.values()))
        art["joint_body_ticks"] = int(sum(touchers.values()) - sum(hand.values()))
    d_hand = np.linalg.norm(res["wrist_world"] - np.asarray(wp.grasp_point)[None], axis=1)
    # Where the wrist was when the ask was made -- without it "progress" has no
    # denominator and a chunk that starts already on target reads the same as one
    # that closed a metre.
    d_wrist_start = float(np.linalg.norm(res["wrist_world"][0] - goal_w))
    row = {**base, "unusable": False, "approach": False,
           "ref_err": round(_reference_wrist_err(q, wrist_l, root_l, side=side), 4),
           "wrist_err_min": round(float(d_wrist.min()), 4),
           "wrist_err_final": round(float(d_wrist[-1]), 4),
           "hand_err_min": round(float(d_hand.min()), 4),
           "root_travel": round(float(np.linalg.norm(
               res["root_world"][-1][:2] - res["root_world"][0][:2])), 4),
           "scene_contacts": _scene_contacts(tr),
           "contact_ticks": contact_ticks[0],
           "fell": bool(res["fell"]),
           "success_at": res["success_at"], **art}
    wrist_path = float(np.linalg.norm(np.diff(res["wrist_world"], axis=0),
                                      axis=1).sum())
    # THE ATTRIBUTION COLUMN. `ref_err` says how far the generated reference is
    # from the goal; `track_err` says how far the robot is from that reference.
    # Together they say which component to fix, which no pair of goal-distances
    # can.
    t_med, t_max = _tracking_error(_reference_wrist_path(q, root_l, side=side),
                                   res["wrist_world"], tr)
    # ⚠️ THE DISAGREEMENT AT TICK 0, which has to be near zero and is the only
    # way to tell a tracking failure from a frame bug. `load_chunk(reboot=True)`
    # puts the robot ON the chunk's first frame, so at tick 0 the robot's wrist
    # and the reference's frame-0 wrist are the same point by construction --
    # measured at 0.045 m in the empty room, all of it the offset between the FK
    # body and the sim body. If the scene reads far larger, `ref_err` and
    # `track_err` are being computed against a frame the robot is not in, and the
    # constant ~0.54 `ref_err` that survives every explanation is a measurement
    # artifact rather than a property of the generator.
    _ref_path = _reference_wrist_path(q, root_l, side=side)
    if len(_ref_path) and len(res["wrist_world"]):
        _d0 = (np.asarray(res["wrist_world"][0], dtype=np.float64)
               - np.asarray(tr.to_world(_ref_path[0]), dtype=np.float64))
        row["track_err_t0"] = round(float(np.linalg.norm(_d0)), 4)
        # ⚠️ Split, because the two components accuse different code. A HORIZONTAL
        # offset is the anchoring (`root_l`, `origin_xy`); a VERTICAL one is the
        # floor -- the scene's floor is a box whose top face is the walking
        # surface and is not at z = 0, while `_reference_wrist_path` deliberately
        # leaves heights alone ("already floor-referenced").
        row["track_t0_xy"] = round(float(np.linalg.norm(_d0[:2])), 4)
        row["track_t0_z"] = round(float(_d0[2]), 4)
    row["track_err_med"] = None if np.isnan(t_med) else round(t_med, 4)
    row["track_err_max"] = None if np.isnan(t_max) else round(t_max, 4)
    # ⚠️ Feasibility is judged on the ASK, before the execution is graded. A goal
    # further from the root than the arm can reach is a PLANNER defect, and
    # grading the generator on it is how a pipeline spends months improving the
    # wrong component.
    row["eef_outcome"] = classify(
        d_start=d_wrist_start, err_min=float(d_wrist.min()),
        err_final=float(d_wrist[-1]), travel=wrist_path,
        tol=GRASP_RETRY_GATE,
        feasible=eef_feasible(float(np.linalg.norm((goal_w - tr.data.xpos[tr.bid_root])[:2]))))
    row["wrist_path"] = round(wrist_path, 4)
    # ⚠️ HOW FAR THE GOAL IS FROM THE PELVIS, at the start of the chunk and at
    # the end. `eef_feasible` above judges the ASK, i.e. the start -- but the
    # body moves during the chunk, and if it ends up further away than the arm
    # can span then the second half of the chunk was reaching for something
    # outside the workspace no matter how good the hand is. Measured envelope:
    # p95 0.639 m seated, about 0.80 m standing with a lean (h18).
    _pel = np.asarray(tr.data.xpos[tr.bid_root], dtype=np.float64)
    row["reach_end"] = round(float(np.linalg.norm(goal_w - _pel)), 4)
    row["reach_start"] = round(float(np.linalg.norm(
        goal_w - np.asarray(res["root_world"][0]))), 4)
    if grasp is not None:
        g = grasp.info()
        # Reported separately from the task's verdict on purpose: the task asks
        # "is a hand within 0.16-0.24 m", this asks "did the object travel with
        # the wrist". They disagree exactly on the episodes worth throwing away.
        row.update({"grasp_state": g["state"], "grasp_fingers": g["fingers"],
                    "grasp_trigger": g["approach"], "grasp_rearms": g["rearms"],
                    "grasp_min_dist": round(float(min_palm_dist), 4),
                    "grasp_residual": (None if best_residual is None else
                                       [round(float(v), 4) for v in best_residual]),
                    "grasp_travel": g["max_travel"], "grasp_slip": g["max_slip"],
                    "holding": bool(grasp.holding)})
    return row, res["ticks"], res


def _place_in_front(tr, dist: float, height: float) -> None:
    """Move every graspable object to ``dist`` in front of the robot.

    A DIFFICULTY KNOB, not a fix. Everything measured so far conflates two
    questions -- can the robot get to the object, and can it grasp once there --
    because every episode starts with a walk. Removing the walk answers the
    second one on its own, and if the answer is still no, then every metre of
    locomotion improvement would have been wasted effort.

    Done by writing the object's free joint at rollout time rather than by
    editing the task: the task file is shared with other branches (see plan.md
    D5), and a difficulty setting that only exists during an experiment should
    not become a permanent property of the scene.
    """
    import mujoco

    root = tr.data.xpos[tr.bid_root]
    q = tr.data.qpos
    yaw = 2 * np.arctan2(tr.root_quat()[3], tr.root_quat()[0])
    fwd = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    for name in list(getattr(tr.env, "objects", {})):
        jid = mujoco.mj_name2id(tr.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint0")
        if jid < 0:
            continue
        adr = tr.model.jnt_qposadr[jid]
        q[adr:adr + 3] = root + fwd * float(dist) + np.array([0, 0, 0.0])
        q[adr + 2] = tr.floor_z + float(height)
        q[adr + 3:adr + 7] = [1, 0, 0, 0]
        dof = tr.model.jnt_dofadr[jid]
        tr.data.qvel[dof:dof + 6] = 0
        # Only the first object: moving the plate too would make "place" trivial
        # in a way that hides whether the pick worked.
        break
    mujoco.mj_forward(tr.model, tr.data)
    print(f"[h7] EASY MODE: object moved to {dist:.2f} m ahead, "
          f"{height:.2f} m up -- locomotion removed, this is not a normal run")


#: Where the arm alone stops closing the gap, measured by h18: past this, palm
#: error is just the shortfall (0.26 m at d_xy 1.0, 0.46 m at 1.2) while the same
#: goal with room to walk stays at 0.03-0.15 m all the way out.
STANDING_REACH = 0.80


#: How far out the GENERATOR still puts the wrist where it was asked, metres.
#: h25's sweep: 0.021 at 0.20 m, 0.057 at 0.30, 0.131 at 0.40 -- against a grasp
#: radius of 0.045-0.07. This is a much tighter number than the ARM's 0.80 m
#: reach, and the difference is why splitting a reach used to be gated on the
#: wrong one.
GEN_RADIUS = 0.35

#: Past this azimuth the target belongs to the hand on that side, degrees. A dead
#: band, so a target near the midline stays on whichever hand the plan chose
#: instead of alternating between chunks.
HAND_SWITCH_DEG = 20.0

#: Beyond this |heading|, canonicalisation is REFUSED under `--canon auto`.
#: Measured (h27 --spawn-yaw, p41c, n=20): at 90 it recovers the whole loss
#: (end 0.153 -> 0.093, dwell 6/20 -> 15/20 against a yaw-0 baseline of 0.089 /
#: 12-20), at 135 it is marginal, and at 180 the robot fell in 16 of 20 draws and
#: the hand finished 2.15 m out -- far worse than leaving the heading alone. The
#: 180 failure is NOT understood, so the gate is drawn where the evidence stops.
CANON_MAX_DEG = 120.0


def _expand_with_reach_legs(tr, goals, reach: float = GEN_RADIUS,
                            verbose: bool = True):
    """Give a reach that starts out of range its own chunk to walk in first.

    A waypoint with a hand goal asks ONE chunk to cover ground and arrive
    reaching -- the ask this pipeline has repeatedly measured as not fitting in
    3 s. The planner already splits that for locomotion, but only past its hop
    length of 1.2 m, so the 0.8-1.2 m band falls between the two and gets the
    combined ask. That band is where the furniture tasks live: measured on
    `IThorBedroomCloseLaptop`, the lid is 0.76-1.31 m away and the robot closes
    1.03 -> 0.86 m over three chunks before the stall detector gives up, 0.3 m
    short, with `standoff_used` 0.45 in every one of them. It is asked to come
    closer and never gets there.

    Split, the walk is judged on the ROOT -- which is what it is actually doing --
    and the reach starts from inside the envelope.
    """
    from rl.kimodo.heft.adapters import OPEN, SceneGoals, Waypoint
    from rl.kimodo.heft.scene_frame import free_standoff

    out = []
    here = tr.data.xpos[tr.bid_root].copy()
    for wp in goals.waypoints:
        target = np.asarray(wp.grasp_point, dtype=np.float64)
        approach_only = bool(getattr(wp, "approach_only", False))
        d = float(np.linalg.norm(target - here))
        if approach_only or not wp.needs_root or d <= reach:
            out.append(wp)
            here = target if approach_only else here
            continue
        # ⚠️ Stand inside the OPERATING RANGE, not at the waypoint's nominal
        # standoff. The whole point of the split is that the reach chunk starts
        # with its target already somewhere the generator can hit: leaving the
        # leg to stop at 0.45 while the range is 0.35 hands the reach an ask it
        # was just measured to miss (h25: 0.057 at 0.30 m, 0.131 at 0.40).
        want = min(float(wp.standoff), float(reach))
        xy, used = free_standoff(tr.env, target, here, want)
        stand = np.array([xy[0], xy[1], target[2]])
        out.append(Waypoint(stand, phase=OPEN, label=f"step to {wp.label}",
                            approach_only=True,
                            # The bar is the reach envelope, not an arm's length:
                            # arriving anywhere inside it is enough, and a tighter
                            # one would spend the whole chunk budget on the walk.
                            # ⚠️ Tight on purpose. The old bar was
                            # `max(0.35, reach - standoff)`, which let the walk
                            # leg finish 0.35 m short -- and the whole point of
                            # the leg is to get the hand goal inside a radius of
                            # that same size. A leg that ends 0.35 m out has
                            # bought nothing.
                            arrive_radius=0.25))
        # By the time the reach runs the robot is where it needs to be; a second
        # root sub-goal would only walk it back out of the pose it just achieved.
        out.append(replace(wp, needs_root=False))
        here = np.array([xy[0], xy[1], here[2]])
        if verbose:
            print(f"[h7]      +1 step leg before {wp.label!r} "
                  f"(target {d:.2f} m out, range {reach:.2f}, "
                  f"stand at {used:.2f}{' -- pushed out by furniture' if used > want + 1e-3 else ''})")
    return SceneGoals(waypoints=out, lang=goals.lang, notes=goals.notes)


def _expand_with_plan(tr, goals, hop: float, verbose: bool = True,
                      hop_radius: float = 0.2):
    """Replace each root-seeking waypoint with the hops of a planned path.

    Done by REWRITING the waypoint list rather than by teaching drive_chunk about
    paths: a hop is already exactly what an approach waypoint is -- a root goal
    with no hand target -- so the planner's output slots into machinery that
    already exists and is already tested. Nothing downstream needs to know a
    planner ran.

    The final waypoint keeps its original identity (phase, grasp object, hand
    goal); only the travel in front of it becomes hops. A grasp waypoint whose
    object is already within reach gets no hops at all.
    """
    from rl.kimodo.heft.adapters import OPEN, SceneGoals, Waypoint
    from rl.kimodo.heft.planner import plan_hops

    out = []
    here = tr.data.xpos[tr.bid_root][:2].copy()
    for wp in goals.waypoints:
        goal_xy = np.asarray(wp.grasp_point, dtype=np.float64)[:2]
        if not wp.needs_root or float(np.linalg.norm(goal_xy - here)) < hop:
            out.append(wp)
            here = goal_xy
            continue
        try:
            hp, occ = plan_hops(tr.model, tr.data, here, goal_xy,
                                arrive_radius=wp.arrive_radius, hop=hop)
        except SystemExit as e:
            if verbose:
                print(f"[h7]      planner unavailable ({e}); driving direct")
            out.append(wp)
            continue
        # The last hop is the destination the original waypoint already asks for,
        # and the first is USUALLY where the robot already stands -- keeping
        # either would spend a chunk going nowhere.
        #
        # ⚠️ "Usually". When the robot's own cell is BLOCKED -- and it often is,
        # every scene here spawns inside the 0.35 m inflation -- `astar` steps the
        # start out to the nearest free cell and the path begins THERE. `hp[0]` is
        # then not where the robot stands; it is the one waypoint that gets it
        # out. Dropping it hands the robot a leg from inside an obstacle straight
        # to `hp[1]`, and on `IThorLivingRoomLiftStatue` that leg is 25 of 28 sampled
        # points inside inflated obstacles (h17), against 1 of 17 on the task that
        # succeeds. The robot enters the coffee table's footprint at tick 57 and
        # spends 68% of the episode inside it.
        #
        # ⚠️ It does not always bite, which is why it survived: `ApproachTable`
        # also starts in a blocked cell, but its step-out happens to point the way
        # it was going anyway.
        keep_first = not occ.free(occ.to_cell(here))
        mid = hp[(0 if keep_first else 1):-1]
        for i, p in enumerate(mid):
            out.append(Waypoint(np.array([p[0], p[1], 0.9]), phase=OPEN,
                                label=f"hop{i}", approach_only=True,
                                arrive_radius=hop_radius))
        out.append(wp)
        if verbose:
            if mid:
                r = getattr(occ, "radius_used", None)
                print(f"[h7]      +{len(mid)} hop waypoints before {wp.label!r} "
                      f"(grid {occ.occupancy_fraction():.0%} occupied"
                      + (f", body radius {r:.2f}" if r else "") + ")")
            else:
                print(f"[h7]      NO PLAN for {wp.label!r} "
                      f"(grid {occ.occupancy_fraction():.0%} occupied) -- driving "
                      f"direct, which is what walks into furniture")
        here = goal_xy
    return SceneGoals(waypoints=out, lang=goals.lang)


def _fan_out(fns):
    """One callback that calls several. None entries are dropped."""
    fns = [f for f in fns if f is not None]
    if not fns:
        return None
    return lambda *a, **k: [f(*a, **k) for f in fns]


def _render_subprocess(states_path: str, video_path: str,
                       model_path: str | None = None) -> str | None:
    """Render a saved episode to mp4 in a SEPARATE process.

    Not an optimisation and not tidiness: this process holds a CUDA context for
    Kimodo, and an EGL offscreen context in the same process deadlocks on shared
    multi-GPU nodes (DEBUG_LOG #11). The child imports no torch, so it can render
    safely. Failure here is reported, never fatal -- the episode's data is
    already on disk and a missing video must not lose it.
    """
    import subprocess

    cmd = [sys.executable, "-m", "rl.kimodo.heft.scripts.view_rollout",
           "--states", states_path, "--video", video_path]
    if model_path:
        cmd += ["--model", model_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"[h7] render failed ({r.returncode}); states kept at {states_path}")
        print((r.stderr or "").strip()[-500:])
        return None
    print(f"[h7] wrote {video_path}")
    return video_path


def _ego_subprocess(states_path: str, video_path: str,
                    model_path: str) -> str | None:
    """Render the HEAD camera to mp4, in a separate process. See _render_subprocess.

    Separate from the third-person render because the two answer different
    questions and only one of them is what a VLA is trained on. A third-person
    shot says whether the robot walked through the table; the ego view says
    whether the object was ever in frame -- and the camera bug that filmed 751
    frames of bathroom wall was invisible in the log and obvious here.
    """
    import subprocess

    cmd = [sys.executable, "-m", "rl.kimodo.heft.scripts.view_rollout",
           "--states", states_path, "--video", video_path,
           "--model", model_path, "--ego"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        print(f"[h7] ego render failed ({r.returncode}); states kept at {states_path}")
        print((r.stderr or "").strip()[-500:])
        return None
    print(f"[h7] wrote {video_path}")
    return video_path


#: Where an approach aims the ROOT, as a fraction of the waypoint's arrival bar.
#: 0.6 of a 1.0 m ring puts the goal at 0.6 m from the target -- outside the
#: fixture for every approach task h31 measured, and 0.4 m inside the bar, which
#: is the margin the tracker's undershoot needs (it delivers 33-40% of a short
#: ask, so it stops short of whatever it is aimed at).
#:
#: ⚠️ NOT 1.0. Aiming AT the ring means stopping outside it.
#: ⚠️ NOT 0.0, which is what it used to be -- see the comment at the use site.
APPROACH_STANDOFF_FRAC = 0.6

#: Below this ask, a locomotion chunk does not move the robot enough to be worth
#: spending. Measured over **1838 navigation chunks** -- robot travel against the
#: reference's own travel, by how far the chunk was asked to go:
#:
#:     ask 0.0-0.3   n=155   ref 0.099   robot 0.039   40%
#:     ask 0.3-0.5   n=218   ref 0.405   robot 0.135   33%
#:     ask 0.5-0.8   n=312   ref 0.696   robot 0.238   34%
#:     ask 0.8-1.2   n=496   ref 1.175   robot 0.457   39%
#:     ask 1.2+      n=657   ref 1.430   robot 0.940   66%
#:
#: A 0.3 m residual therefore takes two to three chunks out of a budget of four,
#: and it does not converge while it spends them -- measured on one `hop3`:
#: root error 0.307 -> 0.265 -> 0.275 over three chunks, robot travel 0.043,
#: 0.103, 0.011. That is the stall. The next hop is 1.2 m away, i.e. in the 66%
#: band, so moving on is strictly more productive than closing the residual.
MIN_USEFUL_ASK = 0.45

#: Shortest distance an APPROACH will ask for, metres. Read straight off the same
#: table: delivery is flat at 33-40% everywhere below 1.2 m and jumps to 66%
#: above it, so 1.2 is where the ask stops being wasted rather than a round
#: number. Only the requested distance is stretched to it -- the aim point is
#: unchanged and the stretch is capped at the target's centre.
#:
#: ⚠️ Different from MIN_USEFUL_ASK, which is a RELEASE bar for hops. One decides
#: when to give up on a gap, the other how far to ask when closing one.
MIN_APPROACH_ASK = 1.2


def _hop_spent(wp, k: int, waypoints, err: float) -> bool:
    """Give up on a planner hop whose residual is too small to be walkable.

    ⚠️ ONLY for the planner's own hops, and only when another waypoint follows.
    A hop is a stepping stone -- nothing is scored on reaching it exactly -- while
    the task's own waypoint is the thing being judged, and releasing that one
    early would be inventing an arrival.

    ⚠️ NOT the same as widening `--hop-radius`. That bar was tightened 0.6 -> 0.2
    on measurement (root error 0.57 -> 0.39, physics 0.94 -> 1.00, bottleneck
    §5b) because following the path closely is what keeps the legs out of the
    furniture. This keeps the bar where it is and only stops the robot from
    spending its chunk budget on a gap it cannot walk.
    """
    if not str(getattr(wp, "label", "")).startswith("hop"):
        return False
    if k >= len(waypoints) - 1:
        return False
    return bool(err < MIN_USEFUL_ASK)


def _gate_reason(row: dict, require_natural: bool = True) -> str:
    """Why this episode is not data. One string, in the order the gate checks.

    ``require_natural`` has to be threaded through: with ``--no-natural`` the
    smoothness screen is not part of the gate, and reporting "naturalness" as
    the reason an episode was dropped when the gate never consulted it would
    send the reader after the wrong number.
    """
    if not row.get("success"):
        return "no success"
    if row.get("fell"):
        return "fell"
    if row.get("physics_ok") is None:
        return "physics screen did not run"
    if not row.get("physics_ok"):
        return f"physics: {','.join(sorted(row.get('physics_reasons') or {})) or '?'}"
    if require_natural and not row.get("natural_ok"):
        stats = row.get("natural_stats") or {}
        near = ", ".join(f"{k} {stats[k]:.2f}" for k in ("joint_dlj", "wrist_dlj")
                         if isinstance(stats.get(k), (int, float)))
        return f"naturalness ({near})" if near else "naturalness"
    return "?"


def _median_of(rows, key: str):
    """Median of a per-waypoint field over the waypoints that HAVE it."""
    v = [w[key] for r in rows for w in r["waypoints"]
         if not w.get("unusable") and w.get(key) is not None]
    return round(float(np.median(v)), 4) if v else None


def _rate_of(rows, key: str):
    v = [bool(w[key]) for r in rows for w in r["waypoints"] if key in w]
    return round(float(np.mean(v)), 4) if v else None


def _reason_hist(rows) -> dict:
    """Which filter is the bottleneck. Accepts counted too, or the yield is invisible."""
    hist = {"accepted": 0, "rejected": 0}
    for r in rows:
        if r.get("physics_ok") is None:
            continue
        hist["accepted" if r["physics_ok"] else "rejected"] += 1
        for name in r.get("physics_reasons", {}):
            hist[name] = hist.get(name, 0) + 1
    return hist


def _print_row(r: dict) -> None:
    hand = f" [{r.get('side', '?')}]" if not r.get("approach") else ""

    if r.get("unusable"):
        print(f"[h7] wp{r['waypoint']}.{r['chunk']} {r['label']!r}{hand}  UNUSABLE reference")
        return
    if r.get("approach"):
        print(f"[h7] wp{r['waypoint']}.{r['chunk']} {r['label']!r} APPROACH  "
              f"d_xy {r['d_xy']:.2f} az {r['azimuth_deg']:+.0f}deg  "
              f"root {r['root_err_min']:.2f}/{r['root_err_final']:.2f} "
              f"(bar {r['arrive_radius']})  "
              f"ref_root {r.get('ref_root_err_final', float('nan')):.2f}  "
              f"travel ref {r.get('ref_travel', float('nan')):.2f} / "
              f"exec {r['root_travel']:.2f}  "
              f"contacts {r['scene_contacts']}"
              + ('  ARRIVED' if r['arrived'] else '')
              + ('  FELL' if r['fell'] else '')
              + ('  SUCCESS' if r['success_at'] is not None else ''))
        return
    print(f"[h7] wp{r['waypoint']}.{r['chunk']} {r['label']!r} {r['phase']}  "
          f"d_xy {r['d_xy']:.2f} az {r['azimuth_deg']:+.0f}deg  "
          f"ref {r['ref_err']:.3f}  wrist {r['wrist_err_min']:.3f}/"
          f"{r['wrist_err_final']:.3f}  hand {r['hand_err_min']:.3f}  "
          f"travel {r['root_travel']:.2f}  contacts {r['scene_contacts']}"
          + (f"  joint {r['joint_qpos']:+.3f} (to go {r['joint_to_go']:+.3f})"
             if r.get("joint_qpos") is not None else "")
          + (f"  touched by hand {r['joint_hand_ticks']} / body {r['joint_body_ticks']} ticks"
             if r.get("joint_hand_ticks") is not None else "")
          + (f"  yaw {r['body_yaw_deg']:+.0f}deg" if r.get("body_yaw_deg") is not None else "")
          + (f"  grasp={r['grasp_state']}/{r['grasp_fingers']}f "
             f"palm {r['grasp_min_dist']:.3f}" if "grasp_state" in r else "")
          + ('  FELL' if r['fell'] else '')
          + ('  SUCCESS' if r['success_at'] is not None else ''))


def hold_still(tr, ticks: int, *, watchers=(), physics=None, check=None,
               success_hold: int = 0) -> dict:
    """Stand where you are for ``ticks``, tracking a frozen pose.

    ⚠️ **THE HOLD AFTER ARRIVAL MUST NOT BE SPENT WALKING.** A waypoint's
    release predicate stops the CHUNK, but the criterion these tasks use is a
    held one -- `RoomNavMixin` wants `SUCCESS_DISTANCE` for `HOLD_STEPS` control
    steps -- and any hold served by letting the walking reference run is a hold
    served by walking further into the furniture. Measured on
    `IThorBathroomApproachToilet`: the surface distance was 0.035 m when the
    criterion fired and 0.000 m twenty-five ticks later, because the final
    chunk's ask is clamped to the fixture centre and nothing in the reference
    knows the job is done.

    A frozen reference is the whole mechanism: the tracker is handed the robot's
    own realized pose, tiled, so "keep doing what you are doing" means "stand".
    The ticks still belong to the episode -- the physics filter steps, the
    watchers see them, a fall during the hold is a fall -- because a hold the
    recorders cannot see would silently drop the last half second of every
    navigation episode from the exported data.
    """
    ticks = int(ticks)
    if ticks <= 0:
        return {"ticks": 0, "fell": False, "held": None, "success_at": None}
    # The chunk is resampled 30 -> 50 Hz, so `ticks` frames is already 1.67x more
    # reference than the hold consumes. Running out mid-hold would leave the
    # tracker with no reference, which is the one state it does not survive.
    q = np.tile(tr.realized_qpos36(), (ticks + 4, 1)).astype(np.float32)
    tr.load_chunk(q, reboot=False)
    inside = []

    def on_tick(i, t, _p=physics, _w=tuple(watchers), _c=check):
        if _p is not None:
            _p.step()
        if _c is not None:
            inside.append(bool(_c()))
        for w in _w:
            w(i, t)

    res = tr.rollout(ticks, track=None, on_tick=on_tick, stop_on_success=False,
                     success_hold=success_hold)
    return {"ticks": int(res["ticks"]), "fell": bool(res["fell"]),
            # Did it STAY arrived, or did standing let it drift back out? The
            # answer belongs on the row: an arrival that does not survive half a
            # second of standing is not the arrival the criterion means.
            "held": (None if not inside else
                     round(float(sum(inside) / len(inside)), 3)),
            "success_at": res["success_at"]}


def episode(tr, actor, goals, *, seconds: float, prefix_frames: int,
            settle: int, max_chunks: int = 4, arrive_tol: float = 0.25,
            palm_axis_aim: bool = False, pose_ask: bool = True,
            use_physics: bool = True, physics_limits=None, grasp_retry: float = 0.0,
            travel_tries: int = 1, root_gain: float = 1.0, hand: str = "auto",
            canon_mode: str = "auto", success_hold: int = 0,
            standoff_frac: float | None = None, arrive_hold: int = 50,
            watchers=(), on_goal=None, verbose: bool = True) -> dict:
    """One episode over the waypoint list, rolling-horizon within each waypoint.

    A waypoint is not one chunk. Each chunk re-reads the robot's realized pose
    and re-aims at the same waypoint, so a distant target is approached over
    several chunks and the geometry is recomputed from where the robot actually
    ended up rather than from where the last chunk hoped it would.
    """
    from rl.kimodo.heft.filters import PhysicsFilter, PhysicsLimits
    from rl.kimodo.heft.grasp import ScriptedGrasp

    n_frames = int(round(seconds * 30))
    n_ticks = int(round(seconds * 50))

    stand = np.tile(tr.realized_qpos36(), (prefix_frames, 1)).astype(np.float32)
    log, success_tick, total_ticks, n_chunks = [], None, 0, 0
    fell = False

    # One grasp controller per object, kept across waypoints: a "lift" waypoint
    # that follows a "grasp" one must not restart the state machine, or the
    # carry test loses the anchor it was going to prove the hold against.
    grasps = {}
    # Accumulated across chunks: both filters judge the EPISODE. A per-chunk
    # smoothness score would miss a discontinuity at the boundary, which is the
    # one place this pipeline is most likely to produce one.
    all_targets, all_wrists = [], []
    # P3-2 runs for the WHOLE episode, across chunk boundaries. A per-chunk
    # filter would reset the fall detector at every boundary and pass an episode
    # that toppled slowly over three chunks.
    physics = PhysicsFilter(tr, physics_limits or PhysicsLimits()) if use_physics else None

    def grasp_for(wp):
        name = getattr(wp, "grasp_object", None)
        if name is None:
            return None
        if name not in grasps:
            grasps[name] = ScriptedGrasp(tr, name)
        return grasps[name]

    for k, wp in enumerate(goals.waypoints):
        prev_err = None
        # "Reach, miss, reach again." The correction is the MEASURED residual
        # between where the palm closed and where the object was, not a constant:
        # a fixed global bias was tried (`--reach-bias`) and its bootstrap
        # interval crossed zero, because the miss is not the same size or
        # direction every time. Accumulated across attempts within one waypoint
        # and dropped at the next, since it describes this reach and no other.
        aim_fix = np.zeros(3)
        for c in range(max_chunks):
            pre = stand if n_chunks == 0 else tr.prefix_frames(prefix_frames)
            row, ticks, res = drive_chunk(tr, actor, wp, pre, n_frames, n_ticks,
                                          standoff_frac=standoff_frac,
                                          canon_mode=canon_mode,
                                          success_hold=success_hold,
                                          reboot=(n_chunks == 0),
                                          grasp=grasp_for(wp),
                                          use_palm_axis=(c > 0 and palm_axis_aim),
                                          pose_ask=pose_ask, physics=physics,
                                          tries=travel_tries, root_gain=root_gain,
                                          hand=hand,
                                          watchers=watchers, on_goal=on_goal,
                                          aim_correction=(aim_fix if grasp_retry
                                                          and aim_fix.any() else None))
            n_chunks += 1
            total_ticks += ticks
            if res.get("target_q_29") is not None:
                all_targets.append(res["target_q_29"])
            if res.get("wrist_world") is not None:
                all_wrists.append(res["wrist_world"])
            row.update({"waypoint": k, "chunk": c})
            log.append(row)
            if verbose:
                _print_row(row)
            if row.get("unusable"):
                break
            if res["success_at"] is not None and success_tick is None:
                success_tick = total_ticks - ticks + res["success_at"]
            if res["fell"]:
                fell = True
                break

            # For a waypoint with an object, "arrived" means the PALM reached the
            # object -- the wrist reaching its own goal is not the same claim, and
            # on TabletopPickPlace the two differ by 0.07 vs 0.136 m.
            if row.get("approach"):
                # Judged on the ROOT, against the TASK's own radius rather than
                # arrive_tol -- that one is an arm's-length tolerance and would
                # keep walking long after the task had already scored.
                err = row["root_err_final"]
                if row["arrived"] or _hop_spent(wp, k, goals.waypoints, err):
                    # ⚠️ Only at the DESTINATION, and only on a real arrival. A
                    # hop is a stepping stone: standing still on each of them
                    # would add half a second of nothing per hop, and there is
                    # no criterion there to verify. `_hop_spent` is a give-up,
                    # not an arrival, and holding at a point the robot never
                    # reached would report a verified stop that never happened.
                    if (row["arrived"] and arrive_hold
                            and k == len(goals.waypoints) - 1):
                        h = hold_still(tr, arrive_hold, watchers=watchers,
                                       physics=physics,
                                       check=(wp.arrive_test if wp.arrive_test
                                              is not None else None))
                        total_ticks += h["ticks"]
                        row["hold_ticks"], row["hold_held"] = h["ticks"], h["held"]
                        if h["success_at"] is not None and success_tick is None:
                            success_tick = total_ticks - h["ticks"] + h["success_at"]
                        if h["fell"]:
                            fell = True
                    break
            else:
                err = (row["grasp_min_dist"] if "grasp_min_dist" in row
                       else row["wrist_err_final"])
                # ⚠️ `arrive_tol` is 0.25 m and the hand closes at 0.02-0.04 m,
                # so on an object waypoint "arrived" fires 5-10x outside the
                # envelope and the rolling horizon stops improving long before a
                # grasp is possible. Measured: the retry below never fired once,
                # because the loop had already declared success at palm 0.20 m.
                # With retries on, only actually HOLDING it counts as done.
                # ⚠️ And only once the hand is actually NEAR. "Retry until you
                # hold it" applied at any distance spends the whole chunk budget
                # re-attempting a grasp from a metre and a half away instead of
                # walking closer: measured on `IThorLivingRoomLiftStatue`, the palm's
                # closest approach went to 1.86 m median with 0/92 chunks inside
                # the box's 0.12 m envelope, against 0.74 m median without it.
                near = err <= GRASP_RETRY_GATE
                done = (row.get("holding") if (grasp_retry and wp.grasp_object and near)
                        else (row.get("holding") or err <= arrive_tol))
                if done:
                    break                  # arrived; move to the next waypoint
                # Missed. Aim the next attempt at where the object actually was
                # relative to where the hand actually closed. Damped, because the
                # next chunk starts from a different pose and a full correction
                # would chase its own overshoot; clipped, because a residual of
                # half a metre means the hand never got near and the fix for that
                # is not a nudge.
                res = row.get("grasp_residual")
                if grasp_retry and res is not None:
                    v = np.asarray(res, dtype=np.float64)
                    n = float(np.linalg.norm(v))
                    if n <= 0.30:
                        aim_fix = aim_fix + grasp_retry * v
                        # ⚠️ CLIPPED, because this is integral control over a
                        # plant that barely responds. Measured on the first
                        # version: the correction ran 0.16 -> 0.26 -> 0.35 -> 0.41
                        # while the palm distance did NOT improve monotonically
                        # (0.203 -> 0.125 -> 0.223). The hand's landing point is
                        # not very sensitive to the aim point -- the same
                        # insensitivity behind the systematic 0.085-0.117 m
                        # undershoot -- so an uncapped integrator just winds up
                        # and starts aiming at empty space.
                        mag = float(np.linalg.norm(aim_fix))
                        if mag > GRASP_RETRY_CAP:
                            aim_fix = aim_fix * (GRASP_RETRY_CAP / mag)
                        if verbose:
                            print(f"[h7]      retry: palm closed {n:.3f} m off; "
                                  f"aim now {np.linalg.norm(aim_fix):.3f} m "
                                  f"({'capped' if mag > GRASP_RETRY_CAP else 'free'})")
            if prev_err is not None and err > prev_err - 0.05:
                # No meaningful progress from a whole extra chunk. Spending the
                # rest of the budget re-asking for the same thing produces more
                # data about the same failure, not a better outcome.
                if verbose:
                    print(f"[h7]      stalled at {err:.3f} m (was {prev_err:.3f}); "
                          "moving on")
                break
            prev_err = err
        if fell or log[-1].get("unusable"):
            break

    # Let the scene settle: a place that succeeds only after the object stops
    # rolling is still a success, and the reference has run out by now so the
    # tracker holds the last frame.
    if settle and not any(r.get("fell") for r in log):
        res = tr.rollout(settle, track="right", stop_on_success=True,
                         success_hold=success_hold,
                         on_tick=(lambda i, t: [w(i, t) for w in watchers]
                                  if watchers else None))
        total_ticks += res["ticks"]
        if res["success_at"] is not None and success_tick is None:
            success_tick = total_ticks

    held = {n: g.holding for n, g in grasps.items()}
    # P3-3, offline over the whole episode: jerk is a third derivative, so it is
    # only defined over a window and a per-tick verdict would need a lag anyway.
    from rl.kimodo.heft.naturalness import judge

    # Per chunk, never concatenated: the boundary discontinuity is a separate,
    # named defect and would otherwise swamp the smoothness score.
    nv = judge(targets_q29=all_targets or None, wrist_world=all_wrists or None)
    pv = physics.verdict() if physics is not None else None
    if verbose:
        print(f"[h7]      natural: {nv.summary()}  "
              f"{ {k: round(v, 2) for k, v in nv.stats.items()} }")
    if verbose and pv is not None:
        print(f"[h7]      physics: {pv.summary()}  {pv.stats}")
        if pv.culprits:
            print(f"[h7]      contacts: {pv.culprits}")
    return {"waypoints": log, "success_tick": success_tick,
            # P3-2 is reported beside success, never folded into it. An episode
            # that "succeeds" with a knee inside the sofa is a physics failure
            # with a success label, and the pipeline has to be able to find it.
            "physics_ok": None if pv is None else pv.accepted,
            "physics_reasons": {} if pv is None else pv.reasons,
            "physics_stats": {} if pv is None else pv.stats,
            "physics_culprits": {} if pv is None else pv.culprits,
            "natural_ok": nv.accepted, "natural_stats": nv.stats,
            "success": success_tick is not None,
            # Reported next to `success` and never merged into it. The task's
            # verdict and "the robot actually held the object" are different
            # claims, and the episodes where they disagree are exactly the ones
            # a data pipeline must be able to find.
            "held": held, "any_held": any(held.values()) if held else None,
            "fell": any(r.get("fell") for r in log),
            "ticks": total_ticks}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # iThor by default. The other scene families' collision setup has not been
    # checked to the same standard, and a rollout that fails there cannot be
    # attributed to the model without first ruling out the room.
    ap.add_argument("--env", default="IThorLivingRoomApproachTable")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--text-cache", help="existing cache (default: KIMODO_TEXT_CACHE or HEFT_WORK/assets/text_cache.pt)")
    ap.add_argument("--base", action="store_true", help="run the PRETRAINED Kimodo, loading no checkpoint at all -- the model before any of this project's RL touched it. Mutually exclusive with --ckpt. ⚠️ Expect it to LOOK more accurate and DELIVER worse: its reference lands 0.053 m from the ask against p41c's 0.085, and its hand holds the target in 5 of 30 draws against 19 of 30. Fine-tuning spends reference accuracy to buy overshoot the tracker can land on")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--seconds", type=float, default=3.0)
    ap.add_argument("--steps", type=int, default=6, help="denoising steps")
    ap.add_argument("--prefix", type=int, default=5)
    ap.add_argument("--settle", type=int, default=50, help="extra ticks after the last chunk")
    ap.add_argument("--success-hold", type=int, default=25, metavar="TICKS",
                    help="how many CONSECUTIVE ticks `_check_success()` must stay "
                         "true before the episode is credited, at 50 Hz. ⚠️ 0 "
                         "credits the first tick it fires, which counts a hand "
                         "sweeping through the target, a lid bouncing past its "
                         "threshold, and an object that is briefly inside a "
                         "region on its way out of it. `success_at` is still "
                         "stamped at the tick the run STARTED, so the reported "
                         "time keeps meaning \"when it got there\". 0 reproduces "
                         "every number measured before 2026-08-09")
    ap.add_argument("--max-chunks", type=int, default=4,
                    help="chunks allowed per waypoint; a 3 s chunk covers ~1.5 m")
    ap.add_argument("--arrive-tol", type=float, default=0.25,
                    help="palm-to-object (or wrist-to-goal) distance counted as arrived")
    ap.add_argument("--no-physics", action="store_true",
                    help="do not run the P3-2 plausibility filter")
    ap.add_argument("--no-natural", action="store_true",
                    help="drop the SMOOTHNESS screen from the export gate. The "
                         "DLJ numbers are still measured and still written to "
                         "`natural_stats` / `natural_ok_rate` -- only the gate "
                         "stops consulting them, and the summary records that "
                         "with `natural_gated: false`. Use when the criterion "
                         "and the physics screen are what you are collecting "
                         "on and the motion's smoothness is not; ⚠️ the "
                         "resulting `usable_rate` is NOT comparable with a "
                         "normal run's")
    ap.add_argument("--no-face", action="store_true",
                    help="ask for the palm direction but not for a body turn")
    ap.add_argument("--no-pose-ask", action="store_true",
                    help="do NOT ask for a facing direction or a palm direction "
                         "(P2-7 off) -- the behaviour every number before it was "
                         "measured under")
    ap.add_argument("--palm-axis", action="store_true",
                    help="re-aim later chunks along the REALIZED wrist axis. Off by "
                         "default: it was tried and it makes things worse (palm "
                         "distance 0.134 -> 0.161 m), because each chunk regenerates "
                         "a whole motion and the final orientation depends on the "
                         "trajectory, so the previous chunk's axis does not predict "
                         "the next one's")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--list", action="store_true", help="print the registered adapters and exit")
    ap.add_argument("--no-plan", action="store_true",
                    help="do NOT route around furniture. The planner is ON by "
                         "default and decides per waypoint: a target closer than "
                         "--hop is driven at directly, anything further is routed. "
                         "Kimodo receives no scene geometry, so turning this off "
                         "means walking through walls -- it exists to reproduce "
                         "old numbers, not for normal use")
    ap.add_argument("--hop", type=float, default=1.2,
                    help="metres per planned hop; a 3 s chunk covers about 1.5")
    ap.add_argument("--easy-dist", type=float, default=None,
                    help="DIFFICULTY KNOB: teleport the graspable object to this "
                         "horizontal distance directly in front of the robot, at "
                         "a comfortable height. Removes locomotion from the task "
                         "so the reach can be measured on its own. 0.45 is a "
                         "planted arm's reach. Does NOT edit the task file")
    ap.add_argument("--easy-height", type=float, default=0.95,
                    help="height for --easy-dist, metres above the floor")
    ap.add_argument("--reach-bias", type=float, default=None,
                    help="aim this far PAST the grasp point, metres, to cancel the "
                         "measured undershoot (p35d: 0.085-0.117 m short along the "
                         "approach). Default 0 = the original behaviour")
    ap.add_argument("--no-ghost", action="store_true",
                    help="do NOT draw the reference pose in --view. On by default: "
                         "without it the window cannot separate 'the robot lags "
                         "its reference' (execution) from 'the reference is in the "
                         "wrong place' (generation), and the log's palm number is "
                         "consistent with both")
    ap.add_argument("--view", action="store_true",
                    help="open a live viewer window and follow the rollout. Needs "
                         "a DISPLAY, and paces the episode to real time -- do not "
                         "quote wall-clock numbers from a viewed run")
    ap.add_argument("--save-states", action="store_true",
                    help="record every tick's sim state to <out>/states/<tag>_ep<N>.npz, "
                         "so the episode can be watched or rendered afterwards")
    ap.add_argument("--video", action="store_true",
                    help="implies --save-states, then renders each episode to mp4 "
                         "IN A SUBPROCESS. Rendering never happens in this process: "
                         "an EGL context alongside CUDA deadlocks (DEBUG_LOG #11)")
    ap.add_argument("--canon", choices=("auto", "on", "off"), default="off",
                    metavar="MODE",
                    help="generate in the frame Kimodo was TRAINED in: rotate the "
                         "prefix and the goals about the robot's own root until "
                         "it faces +x, generate, rotate the chunk back. ⚠️ OFF BY "
                         "DEFAULT because it DOES NOT HELP IN A SCENE. In the "
                         "empty room at 90 degrees it recovers the entire heading "
                         "loss (h27: end 0.153 -> 0.093, dwell 6/20 -> 15/20); on "
                         "IThorKitchenAppleOnPlate, which spawns at +85, the same "
                         "transform moved ref_err 0.544 -> 0.566 and the palm "
                         "0.359 -> 0.377, i.e. nothing, or slightly worse. `auto` "
                         "gates on |heading| <= 120 (at 180 the robot fell 16 of "
                         "20 times). Kept so the difference can be chased, not "
                         "because it is a fix")
    ap.add_argument("--task-set", choices=("legacy", "48"), default="legacy",
                    help="which registry resolves the task. `48` is the "
                         "collection list (33 atomic + 15 composite) and pins "
                         "the room it is collected in; `legacy` is the older set "
                         "this repo measured before it. ⚠️ Kept apart on purpose: "
                         "a number that could have come from either set cannot be "
                         "attributed to one")
    ap.add_argument("--scene-id", type=int, default=None, metavar="N",
                    help="pin the iThor floor plan. ⚠️ Without this every reset "
                         "RESAMPLES the room, and that is the largest noise "
                         "source in every measurement this pipeline makes: the "
                         "same ask measured 0.107 m in one room and 0.628 in the "
                         "next, purely because they spawn the robot at different "
                         "headings. The collection line pins one room per space "
                         "-- kitchen 13, livingroom 209, bedroom 329, "
                         "bathroom 401 -- and so should any run meant to be "
                         "compared with another")
    ap.add_argument("--scene-seed", type=int, default=None, metavar="N",
                    help="fix the object placement draw inside that room")
    ap.add_argument("--loop", action="store_true",
                    help="keep starting new episodes until the window is closed. "
                         "Each one resets the scene, and iThor resamples the "
                         "floor plan on reset, so this walks through rooms as "
                         "well as attempts. Needs --view")
    ap.add_argument("--hand", choices=("auto", "left", "right"), default="auto",
                    help="force the arm. `auto` re-decides every chunk from the "
                         "WRIST GOAL's bearing in the robot's current frame "
                         "(dead band 20 deg, adapter's choice inside it). Force "
                         "one to test a plan against your own reading of the "
                         "scene -- the log prints `az_hand` so the two can be "
                         "compared")
    # ⚠️ STILL 0.2, deliberately. The residual left at a hop (0.26-0.31 m
    # measured) sits just outside this bar, which looks like an argument for
    # widening it -- but the bar was tightened 0.6 -> 0.2 on measurement
    # (root error 0.57 -> 0.39, physics pass 0.94 -> 1.00, bottleneck §5b),
    # because following the path closely is what keeps the legs out of the
    # furniture. `_hop_spent` fixes the stall without giving that back: the bar
    # stays where it is, and the robot simply stops spending chunks on a gap it
    # cannot walk.
    ap.add_argument("--arrive-hold", type=int, default=50, metavar="TICKS",
                    help="ticks the robot STANDS STILL at the destination once "
                         "it has arrived, at 50 Hz (default 50 = 1 s), tracking "
                         "its own frozen pose. 0 ends the episode the instant "
                         "the criterion fires. ⚠️ This is a hold, not extra "
                         "walking: serving it by letting the approach chunk run "
                         "on was measured to close the last 3.5 cm and lean the "
                         "robot on the fixture, because the final ask is clamped "
                         "to the fixture centre")
    ap.add_argument("--approach-standoff-frac", type=float, default=None,
                    metavar="F",
                    help="where an approach aims the ROOT, as a fraction of the "
                         "arrival bar (default %.2f). Governs a measured "
                         "trade-off: 0.0 (what it used to be) aims at the "
                         "fixture centre and lands closest but burns chunks "
                         "pushing at an unreachable point; higher lands further "
                         "out and walks straighter" % APPROACH_STANDOFF_FRAC)
    ap.add_argument("--hop-radius", type=float, default=0.2, metavar="M",
                    help="how close the root must get before a planned hop counts"
                         "as done. WAS `max(0.4, hop*0.5)` = 0.6 m against a 1.2 m "
                         "hop -- ⚠️ HALF A HOP, so most waypoints got exactly ONE "
                         "chunk, the rolling horizon never re-aimed, and the "
                         "tracker's 50%% per-chunk under-travel was never "
                         "corrected. Measured on IThorBedroomApproachBed, root "
                         "error median: 0.6 -> 0.5701, 0.35 -> 0.5265, "
                         "0.2 -> 0.3901 (n=16), with physics 0.94 -> 1.00 and 19%% "
                         "more wall clock. ⚠️ Only bites where the planner "
                         "inserts hops: on a close manipulation waypoint the "
                         "output is bit-identical")
    ap.add_argument("--root-gain", type=float, default=1.0, metavar="G",
                    help="scale every root sub-goal's distance by G. The tracker "
                         "delivers about HALF the reference's own root travel "
                         "(measured 50%% with no contact, 38-40%% with), because "
                         "it observes neither root position nor root velocity. "
                         "G=2 asks for double so the shortfall lands on target. "
                         "⚠️ Overshooting past the goal is its own failure -- "
                         "read arrive_rate and root_err_final together")
    ap.add_argument("--standoff", type=float, default=None, metavar="M",
                    help="override every manipulation waypoint's standoff -- how "
                         "close the ROOT is asked to end up to the target. The "
                         "default (0.45, inherited from the training sampler) was "
                         "chosen for the arm's reach; h25 measured the GENERATOR "
                         "accurate only within 0.25-0.30 m, which is a different "
                         "and much tighter constraint. This flag exists to test "
                         "whether standing closer moves the ask into the zone "
                         "where the reference actually goes")
    ap.add_argument("--travel-tries", type=int, default=1, metavar="K",
                    help="sample K candidate chunks in ONE batch and keep the "
                         "one that travels best (forwards, facing where it is "
                         "going). Default 1: h20 measured K=4 adding nothing "
                         "once the prompt turns with the bearing -- legal rate "
                         "0.80 either way. The gate itself is always on and "
                         "always reported (`travel_ok` / `travel_back`)")
    ap.add_argument("--grasp-retry", type=float, default=0.0, metavar="GAIN",
                    help="after a missed grasp, aim the next chunk at the MEASURED "
                         "residual between where the palm closed and where the "
                         "object was, times this gain (0.8 is a reasonable start, "
                         "0 = off). A fixed global bias was tried instead and its "
                         "interval crossed zero -- the miss is not the same size "
                         "or direction every time")
    ap.add_argument("--no-reach-leg", action="store_true",
                    help="do NOT split a far reach into walk-then-reach. The split "
                         "is the protocol this pipeline is built on -- a far "
                         "target gets a root goal and nothing else, and the hand "
                         "goal is only issued once the target is inside the "
                         "operating range -- so it is ON by default")
    ap.add_argument("--reach-leg", nargs="?", type=float, const=GEN_RADIUS,
                    default=GEN_RADIUS, metavar="M",
                    help="the range inside which a hand goal is issued at all. "
                         "Beyond it the waypoint is split into a walk chunk (root "
                         "goal only) and a reach chunk. "
                         f"Default {GEN_RADIUS} m, the radius inside which h25 "
                         "measured the GENERATOR still lands where it was asked. "
                         "⚠️ An earlier version gated this on the ARM's 0.80 m "
                         "reach and measured worse; the arm being able to get "
                         "there is not the same claim as the generator asking it "
                         "to")
    ap.add_argument("--keep-model", action="store_true",
                    help="keep the compiled .mjb next to the states so the "
                         "episode can be re-rendered later. ~170 MB each")
    ap.add_argument("--save-data", metavar="DIR", default=None,
                    help="export the USABLE episodes as a heft-format LeRobot "
                         "dataset at DIR (must not exist). Implies --save-states "
                         "and --ego-view: the dataset's images ARE the ego view. "
                         "⚠️ Usable, not successful -- see --gate")
    ap.add_argument("--gate", choices=("usable", "success"), default="usable",
                    help="what --save-data exports. 'usable' (default) needs "
                         "success AND no fall AND the physics and naturalness "
                         "screens; 'success' is the criterion alone, which on "
                         "IThorBathroomApproachSink admits 100%% of episodes "
                         "while 20%% are data -- the rest have a shoulder "
                         "inside the sink")
    ap.add_argument("--show-success", action="store_true",
                    help="in --view, draw a RING on the floor at the waypoint's "
                         "arrive radius, plus a dot where the root actually is. "
                         "Amber outside, green inside")
    ap.add_argument("--success-radius", type=float, default=None, metavar="M",
                    help="judge arrival on the ROOT being this far (horizontal) "
                         "from the target, instead of on the task's own "
                         "surface-distance test, and draw that radius. ⚠️ Changes "
                         "what `arrived` means, NOT what `success` means -- "
                         "success_rate and the export gate stay with the task's "
                         "own verdict")
    ap.add_argument("--ego-view", action="store_true",
                    help="render the robot's HEAD camera to <tag>_ep<N>.ego.mp4 "
                         "-- what the VLA would see, as opposed to --video's "
                         "third-person shot. Offline, in a subprocess, for the "
                         "same reason --video is")
    ap.add_argument("--out", default="rl/kimodo/_out/heft/h7")
    args = ap.parse_args()

    # ⚠️ Resolved here, and passing both is an ERROR rather than a precedence
    # rule. "--base --ckpt something" has two readings and silently picking one
    # produces a number labelled with the model that did not generate it.
    if args.base and args.ckpt:
        raise SystemExit("[h7] --base and --ckpt are mutually "
                         "exclusive: --base loads nothing")
    if not args.base and args.ckpt is None:
        args.ckpt = None
    print(f"[h7] model: "
          + ("PRETRAINED Kimodo, no checkpoint" if not args.ckpt else args.ckpt))

    heft_work = os.path.abspath(os.path.expanduser(HEFT_WORK)) if HEFT_WORK else ""
    if not heft_work or not os.path.isdir(os.path.join(heft_work, "kimodo_rl")):
        raise SystemExit(
            f"[h7] no kimodo_rl package under {heft_work or '(HEFT_WORK is unset)'}. That repo holds the\n"
            f"     generator, the actor and the text cache this rollout drives;\n"
            f"     the weights were trained against it, so it is imported rather\n"
            f"     than re-implemented here.\n"
            f"     Set $HEFT_WORK to the checkout, e.g.\n"
            f"         HEFT_WORK=/path/to/heft_work python -m {__spec__.name if __spec__ else 'rl.kimodo.heft.scripts.h7_scene_rollout'} ...")
    if heft_work not in sys.path:
        sys.path.insert(0, heft_work)
    global FACE_ASK
    FACE_ASK = not args.no_face
    if args.reach_bias is not None:
        from rl.kimodo.heft import scene_frame

        scene_frame.REACH_BIAS = float(args.reach_bias)
        print(f"[h7] aiming {args.reach_bias:+.3f} m past the grasp point")

    # ⚠️ The two task sets resolve through DIFFERENT registries and neither falls
    # back to the other: a task that is not in the selected set fails loudly, so
    # a number can always be attributed to the set it came from.
    if args.task_set == "48":
        from rl.kimodo.heft.adapters48 import goals_for, registered
    else:
        from rl.kimodo.heft.adapters import goals_for, registered

    if args.list:
        # One per line, because the only thing anyone does with this is loop
        # over it. A Python list repr hands `for T in $(... --list)` tokens like
        # "['ApproachSofa'," and the loop fails on the first task.
        for name in registered():
            print(name)
        return 0

    os.makedirs(args.out, exist_ok=True)
    tag = args.tag or f"{args.env}_{'lora' if args.ckpt else 'base'}"

    from rl.kimodo.heft.scene_track import KimodoHeftTracker
    from rl.kimodo.heft.scripts.h6_kimodo_empty import build_actor

    actor = build_actor(args.ckpt, args.steps, heft_work=heft_work, text_cache=args.text_cache)
    # ⚠️ The per-episode retry below does not cover THIS. The tracker is built
    # once, before the loop, and its constructor resets too -- so a seed whose
    # first scene is one of the unbuildable kitchen floor plans dies here with
    # nothing run at all. Measured: seed 2 of `IThorKitchenCloseDrawer` failed
    # identically for two different checkpoints and cost 20 episodes of a
    # comparison. The scene is drawn per reset, so retrying draws a different one.
    # ⚠️ And the retry has to PERTURB THE SEED, which the first version of it did
    # not. The constructor's scene is deterministic in the seed it is given, so
    # retrying with the same one drew FloorPlan3 six times and failed six times.
    # A different seed is a different scene, which is legitimate here -- the floor
    # plan is random anyway -- but it makes the run no longer the one the command
    # named, so it is printed and recorded.
    tr, build_seed = None, args.seed
    # Pinning the room, when asked. ⚠️ Without it every reset resamples the floor
    # plan, which is the largest noise source in every measurement here.
    if args.task_set == "48":
        from rl.kimodo.heft import adapters48

        # The room travels with the task: their numbers and ours are comparable
        # only in the same one, and leaving it unpinned reintroduces the largest
        # noise source there is.
        pinned = adapters48.scene_id_for(args.env)
        if pinned is not None and args.scene_id is None:
            args.scene_id = pinned
            print(f"[h7] task-set 48: pinning {args.env} to FloorPlan{pinned}")
        broken = adapters48.CANNOT_BUILD.get(args.env)
        if broken:
            raise SystemExit(
                f"[h7] {args.env} does not build:\n    {broken}\n"
                f"Checked on three seeds in its pinned room. Nothing to measure.")
        trouble = adapters48.KNOWN_TROUBLE.get(args.env)
        if trouble:
            print(f"[h7] ⚠️ the collection line already found: {trouble}")
    scene_kw = {}
    if args.scene_id is not None:
        scene_kw["scene_ids"] = [int(args.scene_id)]
    if args.scene_seed is not None:
        scene_kw["fixed_scene_seed"] = int(args.scene_seed)
    for attempt in range(6):
        build_seed = args.seed + 1000 * attempt
        try:
            tr = KimodoHeftTracker(env_name=args.env, seed=build_seed, **scene_kw)
            break
        except BaseException as e:  # noqa: BLE001
            print(f"[h7] first scene will not build at seed {build_seed} "
                  f"({type(e).__name__}: {str(e)[:80]}); retrying with a different seed")
    if tr is None:
        raise SystemExit(f"[h7] {args.env}: six unbuildable scenes in a row -- this "
                         "is not the occasional bad floor plan, something is wrong")
    if build_seed != args.seed:
        print(f"[h7] ⚠️ built at seed {build_seed}, not {args.seed}")

    # ⚠️ A dataset needs BOTH recorders and the compiled scene: the columns come
    # from the tick watcher, the ego frames are rendered offline from the states
    # plus the .mjb, and a missing .mjb cannot be reconstructed for these rooms
    # (the reset resamples objects). So --save-data turns all three on rather
    # than failing later with two thirds of an episode on disk.
    save_states = args.save_states or args.video or args.save_data or args.ego_view
    keep_model = args.keep_model or bool(args.save_data)
    if args.save_data and os.path.exists(args.save_data):
        raise SystemExit(f"[h7] --save-data {args.save_data} already exists -- "
                         "LeRobot refuses to write into an existing root, and "
                         "appending to someone else's dataset is not what this "
                         "flag means. Pick a new directory.")
    state_dir = os.path.join(args.out, "states")
    if save_states:
        os.makedirs(state_dir, exist_ok=True)
    if args.view:
        from rl.kimodo.heft.view import LiveViewer
    if save_states:
        from rl.kimodo.heft.view import RolloutRecorder

    from rl.kimodo.heft.export import episode_usable

    def _episode_usable(row: dict) -> bool:
        """The export gate. `--gate success` deliberately keeps the loose one.

        Both branches are named the same thing on the row so the summary's
        `usable_rate` always describes whatever was actually exported -- a rate
        computed against a different population than the filter is this line of
        work's most-repeated measurement bug (P3-2, P3-3).
        """
        if args.gate == "success":
            return bool(row.get("success"))
        return episode_usable(row, require_natural=not args.no_natural)

    bundles: list[str] = []
    temp_models: list[str] = []
    # ⚠️ Registered rather than left to the end of main, and SIGTERM is turned
    # into a SystemExit so that `atexit` runs for it too. Without both halves an
    # interrupted collection keeps every compiled scene it had staged for the
    # export -- measured at 1.1 GB after nine episodes. `_drop_temp_models`
    # empties the list, so running twice is free.
    import atexit
    import signal

    atexit.register(_drop_temp_models, temp_models, args)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    rows, skipped, ep_notes, t0 = [], [], [], time.time()
    if args.standoff is not None:
        print(f"[h7] standoff override: every manipulation waypoint -> "
              f"{args.standoff:.2f} m (default 0.45)")
    # Imported unconditionally: the handler below references it whether or not
    # a window was opened, and a NameError inside an except clause would surface
    # as a crash at exactly the moment the user was trying to stop cleanly.
    from rl.kimodo.heft.view import ResetRequested, ViewerClosed

    # ⚠️ `--loop` needs a window to stop it. Without one the only exit is the
    # episode count, and an infinite headless run is a way to lose a GPU.
    if args.loop and not args.view:
        raise SystemExit("[h7] --loop needs --view: nothing else can end it")
    for ep in (itertools.count() if args.loop else range(args.episodes)):
        import torch

        torch.manual_seed(args.seed + ep)
        np.random.seed(args.seed + ep)
        # This seed IS the scene's identity under hard_reset, because the reset
        # below resamples the objects from numpy's global stream. Handed to the
        # tracker so a recording can carry it and be rebuilt exactly -- see
        # view._rebuild, where counting resets was not enough.
        tr.episode_seed = args.seed + ep
        # ⚠️ One unbuildable scene must cost one episode, not the run. Some
        # kitchen floor plans -- FloorPlan3 is the one that surfaced this -- make
        # `resolve_root_body` raise SystemExit ("the two models are different
        # robots", best 361.79 mm against a 50 mm bar). A ten-episode run died on
        # its sixth and wrote no results at all, losing the five that had already
        # succeeded. Resampling is legitimate here because the floor plan is drawn
        # per reset anyway; what is NOT legitimate is doing it quietly, so every
        # skip is printed and counted.
        for attempt in range(6):
            try:
                tr.reset()
                break
            except BaseException as e:  # noqa: BLE001
                skipped.append({"episode": ep, "attempt": attempt,
                                "error": f"{type(e).__name__}: {str(e)[:120]}"})
                print(f"[h7] ep {ep}: scene will not build ({type(e).__name__}); "
                      f"resampling ({attempt + 1}/6)")
        else:
            print(f"[h7] ep {ep}: giving up after 6 unbuildable scenes")
            continue
        if args.easy_dist is not None:
            _place_in_front(tr, args.easy_dist, args.easy_height)
        goals = goals_for(tr.env)
        # The adapter's own verdict on the episode. For an articulated task it
        # says push or pull, and a pull cannot be done with an open hand -- so a
        # run of zeros on those is a property of the task, not of the model.
        # Computed and then thrown away in the first version, which is how a
        # deliberate diagnostic becomes invisible.
        if goals.notes:
            print(f"[h7]      adapter: {goals.notes}")
            ep_notes.append({"episode": ep, **{k: v for k, v in goals.notes.items()
                                               if not isinstance(v, (list, dict))}})
        # Planning is a property of the DISTANCE, not of the command line. The
        # expansion already skips any waypoint closer than one hop, so leaving it
        # on costs nothing on near targets and is the difference between 0/5 and
        # 5/5 on far ones (ApproachTable, measured). Having it behind a flag meant
        # every run that forgot the flag quietly walked into the furniture.
        if args.standoff is not None:
            # Before planning: the hops are laid out towards the standing point,
            # so moving the standing point after the fact would leave the last
            # hop aimed at the old one.
            goals = replace(goals, waypoints=[
                w if getattr(w, "approach_only", False)
                else replace(w, standoff=float(args.standoff))
                for w in goals.waypoints])
        # ⚠️ CAPTURED BEFORE THE PLANNER RUNS. `_expand_with_plan` prepends its
        # hops, and a hop is `approach_only` too -- so picking "the first
        # approach waypoint" out of the EXPANDED list finds hop0 and draws the
        # ring around the first stepping stone instead of around the target.
        # That is what it did. Taking the last of the task's OWN waypoints is
        # unambiguous and does not depend on matching a label.
        ring_wp = next((w for w in reversed(goals.waypoints)
                        if getattr(w, "approach_only", False)), None)
        if args.success_radius is not None:
            # ⚠️ Applied BEFORE the planner, and only to the task's own approach
            # waypoints -- the planner's hops get their own `hop_radius`, and
            # overriding those too would silently retune the rolling horizon
            # while the user was only asking to change the final bar.
            goals = replace(goals, waypoints=[
                replace(w, arrive_radius=float(args.success_radius),
                        arrive_test=None)
                if getattr(w, "approach_only", False) else w
                for w in goals.waypoints])
        if not args.no_plan:
            goals = _expand_with_plan(tr, goals, args.hop,
                                      hop_radius=args.hop_radius)
        if args.reach_leg and not args.no_reach_leg:
            goals = _expand_with_reach_legs(tr, goals, reach=float(args.reach_leg))
        print(f"\n[h7] ep {ep}: {len(goals.waypoints)} waypoints -- {goals.lang!r}")
        # A viewer is opened PER EPISODE on purpose. Under hard_reset,
        # env.reset() rebuilds the MjModel, so a window held across episodes is
        # pointing at a model that no longer exists.
        watchers, recorder, view, heft_rec = [], None, None, None
        if args.view:
            probe = None
            if args.show_success:
                from rl.kimodo.heft.success_view import root_ring_probe

                # The ring is drawn at the radius the run is JUDGING by, so the
                # picture cannot show one bar while the report prints another.
                # `ring_wp` comes from above, before the planner added its hops.
                ring_r = (args.success_radius if args.success_radius is not None
                          else (getattr(ring_wp, "arrive_radius", None) or 0.6))
                if ring_wp is None:
                    print("[h7] --show-success: no approach waypoint in this "
                          "task, nothing to draw")
                else:
                    probe = root_ring_probe(tr, ring_wp.grasp_point, ring_r)
                    # ⚠️ Say which criterion is actually judging. Without
                    # --success-radius the waypoint still releases on the TASK's
                    # own surface test, so the ring is a reference line and not
                    # the bar -- and a ring that silently means something other
                    # than the number in the report is the exact failure this
                    # whole line of work keeps having to undo.
                    judging = ("this ring" if args.success_radius is not None
                               else "the TASK's own surface test, NOT this ring "
                                    "(pass --success-radius to judge on it)")
                    print(f"[h7] --show-success: ring at {ring_r:.2f} m "
                          f"(root to target, horizontal); `arrived` is judged by "
                          f"{judging}")
            view = LiveViewer(tr, success=probe,
                              ghost=(None if args.no_ghost
                                     else tr.reference_points)).__enter__()
            watchers.append(view.on_tick)
        if save_states:
            recorder = RolloutRecorder(tr)
            watchers.append(recorder.on_tick)
        if args.save_data:
            from rl.kimodo.heft.export import HeftFrameRecorder

            heft_rec = HeftFrameRecorder(tr)
            watchers.append(heft_rec.on_tick)

        try:
            r = episode(tr, actor, goals, seconds=args.seconds,
                        prefix_frames=args.prefix, settle=args.settle,
                        canon_mode=args.canon,
                        success_hold=args.success_hold,
                        max_chunks=args.max_chunks, arrive_tol=args.arrive_tol,
                        standoff_frac=args.approach_standoff_frac,
                        arrive_hold=args.arrive_hold,
                        palm_axis_aim=args.palm_axis,
                        pose_ask=not args.no_pose_ask,
                        use_physics=not args.no_physics,
                        grasp_retry=args.grasp_retry,
                        travel_tries=args.travel_tries,
                        root_gain=args.root_gain, hand=args.hand,
                        watchers=tuple(watchers),
                        on_goal=_fan_out(
                            [recorder.set_marks if recorder is not None else None,
                             view.set_marks if view is not None else None]))
        except ResetRequested:
            # R in the window: this episode is abandoned on purpose, so it is not
            # scored -- counting a deliberately aborted run would quietly poison
            # whatever rate is printed at the end.
            print("[h7] reset requested -- next episode")
            if view is not None:
                view.__exit__(None, None, None)
            continue
        except ViewerClosed:
            # Closing the window is a request to stop, not an error. Everything
            # already gathered stays; the episode simply ends here.
            print("[h7] viewer closed -- stopping")
            if view is not None:
                view.__exit__(None, None, None)
            break
        r["episode"] = ep
        # Decided HERE, where the screens that feed it are, and stored on the
        # row -- so the rate printed in the summary and the gate the exporter
        # applies are one function of one set of facts, not two readings of it.
        r["usable"] = _episode_usable(r)
        rows.append(r)
        if recorder is not None and len(recorder):
            sp = os.path.join(state_dir, f"{tag}_ep{ep}.npz")
            recorder.save(sp)
            r["states"] = sp
            print(f"[h7] ep {ep}: {len(recorder)} states -> {sp}")
            # The compiled scene travels with the states, because rebuilding it
            # from env+seed draws a DIFFERENT room whenever the reset resamples
            # objects. ~170 MB for a living room, so it is written for this
            # render and removed again -- keep it (--keep-model) only when the
            # episode is worth re-rendering later. ⚠️ `--keep-model` alone (no
            # video) is the useful combination for screening: states without the
            # model cannot be re-rendered in these rooms at all.
            if args.video or keep_model or args.ego_view:
                mp = sp[:-4] + ".mjb"
                recorder.save_model(mp)
                if args.video:
                    r["video"] = _render_subprocess(sp, sp[:-4] + ".mp4", mp)
                if args.ego_view:
                    r["ego_video"] = _ego_subprocess(sp, sp[:-4] + ".ego.mp4", mp)
                if keep_model:
                    r["model"] = mp
                    if not args.keep_model:
                        # Kept ONLY so the export can render the ego view at the
                        # end of the run, and deleted again once it has. ⚠️ A
                        # compiled scene is 120-210 MB: a 45-task sweep at 10
                        # episodes would leave ~72 GB behind, which on this
                        # machine is half the free space. Wanting the model
                        # afterwards is what --keep-model is for.
                        temp_models.append(mp)
                else:
                    os.remove(mp)
        # The columns go beside the states whether or not this episode passes:
        # a rejected episode is the only evidence of WHY the gate rejects, and
        # `plan.md`'s rule for the first batch is that accepts and rejects are
        # both kept with a reason. Only the passing ones are handed to LeRobot.
        if heft_rec is not None and heft_rec.rows:
            bp = os.path.join(state_dir, f"{tag}_ep{ep}.heft.npz")
            heft_rec.save(bp, lang=goals.lang, env_name=args.env,
                          seed=args.seed + ep, usable=r["usable"])
            r["heft_bundle"] = bp
            if r["usable"]:
                bundles.append(bp)
            else:
                print(f"[h7] ep {ep}: NOT exported "
                      f"({_gate_reason(r, not args.no_natural)})")
        if view is not None:
            view.__exit__(None, None, None)
        print(f"[h7] ep {ep}: success={r['success']} fell={r['fell']} ticks={r['ticks']}")

    tr.close()
    summary = {
        "tag": tag, "env": args.env, "ckpt": args.ckpt, "episodes": len(rows),
        "success_rate": round(float(np.mean([r["success"] for r in rows])), 4),
        # ⚠️ READ THIS ONE, not success_rate, when the question is "how much data
        # did we get". They are not close: IThorBathroomApproachSink succeeds
        # 100% and is usable 20%. `gate` records which definition produced it.
        "usable_rate": (round(float(np.mean([bool(r.get("usable")) for r in rows])), 4)
                        if rows else None),
        "gate": args.gate,
        # ⚠️ Which screens the gate above actually consulted. Without this the
        # `usable_rate` of a --no-natural run and of a normal one are the same
        # field name meaning two different measurements, and nothing in the file
        # would say so.
        "natural_gated": (args.gate != "success") and not args.no_natural,
        "physics_gated": (args.gate != "success") and not args.no_physics,
        "held_rate": (round(float(np.mean([bool(r["any_held"]) for r in rows])), 4)
                      if any(r.get("any_held") is not None for r in rows) else None),
        "physics_ok_rate": (round(float(np.mean([bool(r["physics_ok"]) for r in rows])), 4)
                            if any(r.get("physics_ok") is not None for r in rows) else None),
        "natural_ok_rate": round(float(np.mean([bool(r["natural_ok"]) for r in rows])), 4),
        "reject_reasons": _reason_hist(rows),
        "fall_rate": round(float(np.mean([r["fell"] for r in rows])), 4),
        # Per waypoint KIND. A reach waypoint has no root criterion and an
        # approach waypoint has no hand one, so pooling them would average a
        # number against a None -- which is how this first crashed -- and
        # averaging them successfully would be worse, because the result would
        # mean nothing.
        "hand_err_min_median": _median_of(rows, "hand_err_min"),
        "reach_end_median": _median_of(rows, "reach_end"),
        "root_err_min_median": _median_of(rows, "root_err_min"),
        "arrive_rate": _rate_of(rows, "arrived"),
        "seconds": args.seconds, "seed": args.seed,
        # Reported, never silent: a run that quietly resampled its way
        # past four broken scenes is not the run the command asked for.
        "scenes_skipped": len(skipped), "skipped": skipped,
        "build_seed": build_seed,
        "adapter_notes": ep_notes,
        "secs": round(time.time() - t0, 1),
    }
    print("\n=== h7 summary ===")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")
    path = os.path.join(args.out, f"{tag}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "episodes": rows}, f, indent=2)
    print(f"[h7] saved {path}")

    # The dataset is written LAST, in a subprocess, after the rollout's CUDA
    # context is gone: writing it needs the ego camera, and rendering needs a
    # process without CUDA (DEBUG_LOG #11). Failure here loses nothing -- every
    # bundle, state file and compiled scene is already on disk and
    # `python -m rl.kimodo.heft.export --bundle ... --out ...` re-runs it.
    if args.save_data:
        if not bundles:
            print(f"[h7] ⚠️ NOTHING TO EXPORT: 0 of {len(rows)} episodes passed "
                  f"the '{args.gate}' gate, so {args.save_data} was not created. "
                  f"The bundles are in {state_dir} and say why.")
        else:
            import subprocess

            print(f"[h7] exporting {len(bundles)}/{len(rows)} episodes -> "
                  f"{args.save_data}")
            cmd = [sys.executable, "-m", "rl.kimodo.heft.export",
                   "--out", args.save_data, "--bundle", *bundles]
            rc = subprocess.run(cmd)
            if rc.returncode:
                print(f"[h7] export failed ({rc.returncode}). Nothing is lost -- "
                      f"re-run:\n  python -m rl.kimodo.heft.export --out "
                      f"{args.save_data} --bundle {state_dir}/{tag}_ep*.heft.npz")
                # The models stay: without them the re-run above cannot render,
                # and 2 GB of recoverable disk is worth less than an episode
                # that can never be exported again.
                temp_models = []
    _drop_temp_models(temp_models, args)
    return 0


def _drop_temp_models(paths, args) -> None:
    """Delete the compiled scenes that were only held for the export.

    ⚠️ Called from a ``finally``, not just at the end of a clean run. Measured:
    interrupting a sweep 9 episodes in left 1.1 GB of ``.mjb`` behind, on a disk
    at 92%. An interrupted run is the NORMAL case for a long collection -- it is
    the one that has to clean up, not the one that gets excused from it.
    """
    freed, n = 0, 0
    for mp in list(paths):
        try:
            freed += os.path.getsize(mp)
            os.remove(mp)
            n += 1
        except OSError:
            pass
    paths.clear()
    if freed:
        print(f"[h7] removed {n} compiled scenes ({freed / 2**30:.1f} GB); "
              f"pass --keep-model to keep them")


if __name__ == "__main__":
    raise SystemExit(main())
