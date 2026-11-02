"""What to ask Kimodo for besides a wrist position: where to face, where to point.

``EndEffectorConstraintSet`` writes the hand's rotation and the root's heading as
well as the hand's position, and the pipeline was handing it the conditioning
prefix's values for both. That is not a neutral default -- it is an instruction:
*end this chunk facing the way you started, with your hand held the way it is
now, but with the hand over there.* Both halves of that instruction were wrong
for manipulation, and both showed up in measurements before the cause was found:

  * a goal to the side is served by leaning and side-stepping instead of turning
    to face it, because the heading channel forbids the turn;
  * the hand arrives 30-34 degrees off the approach direction, leaving the object
    0.12 m off the fingers' axis with the wrist itself only 0.070 m from its goal.

This module computes the two corrections in MuJoCo world terms, where the scene
lives. The actor conjugates them into Kimodo's y-up basis at the boundary.

Both are expressed as DELTAS from the robot's current pose, never as absolute
orientations, and that is deliberate: Kimodo's ``right_wrist_yaw_skel`` frame and
MuJoCo's ``right_wrist_yaw_link`` frame differ by a rotation that was measured
and found NOT to be constant (2.5-9.7 degrees across arm poses, because
``qpos_to_motion_dict`` retargets rather than re-bases). A delta applied on the
left cancels that unknown exactly; an absolute orientation would inherit it.
"""

from __future__ import annotations

import numpy as np


def min_rotation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``a`` to unit vector ``b``."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = a / max(np.linalg.norm(a), 1e-9)
    b = b / max(np.linalg.norm(b), 1e-9)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))
    if s < 1e-9:
        if c > 0:
            return np.eye(3)
        # Antiparallel: any perpendicular axis, half turn.
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis /= np.linalg.norm(axis)
        K = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        return np.eye(3) + 2 * K @ K
    K = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + K + K @ K * ((1 - c) / (s * s))


def yaw_matrix(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def palm_fix(tr, hand_dir: np.ndarray, side: str = "right",
             body_yaw: float | None = None) -> np.ndarray:
    """World rotation that points the palm along ``hand_dir``.

    The CURRENT palm axis is read from MuJoCo rather than from the Kimodo pose,
    which is what makes the delta exact: the conditioning prefix IS the robot's
    realized state, so the two describe the same instant, and reading it on the
    MuJoCo side avoids the retargeting residual entirely.

    ``body_yaw`` must be passed whenever a turn is also being asked for. The
    actor applies the turn to the whole final pose FIRST, so by the time this
    correction lands the palm is already at ``Ry @ current`` -- composing a
    correction computed from the unturned axis stacks the two rotations. Measured
    with them composed wrongly: asking for the palm alone brought the hand from
    34 to 13-19 degrees off target, and asking for both put it back to 20-29.
    """
    import mujoco

    bid = mujoco.mj_name2id(tr.model, mujoco.mjtObj.mjOBJ_BODY,
                            f"{tr.prefix}{side}_wrist_yaw_link")
    if bid < 0:
        raise SystemExit(f"[grasp_pose] no {tr.prefix}{side}_wrist_yaw_link")
    current = tr.data.xmat[bid].reshape(3, 3)[:, 0]
    if body_yaw:
        current = yaw_matrix(float(body_yaw)) @ current
    return min_rotation(current, hand_dir)


def yaw_of(quat_wxyz) -> float:
    w, x, y, z = np.asarray(quat_wxyz, dtype=np.float64)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def face_yaw(tr, target_world, *, max_turn: float = np.pi) -> float:
    """How far the body should turn to face ``target_world``, radians about +z.

    Clipped to ``max_turn`` because a single chunk has a limited turn rate the
    same way it has a limited reach (~1.5 m), and an unreachable constraint is
    not free -- it is one the denoiser cannot satisfy and has to trade against
    everything else in the chunk.
    """
    root = tr.data.xpos[tr.bid_root]
    d = np.asarray(target_world, dtype=np.float64)[:2] - root[:2]
    if np.linalg.norm(d) < 1e-6:
        return 0.0
    want = float(np.arctan2(d[1], d[0]))
    delta = want - yaw_of(tr.root_quat())
    delta = (delta + np.pi) % (2 * np.pi) - np.pi
    return float(np.clip(delta, -max_turn, max_turn))


def grasp_pose_for(tr, grasp_point, *, side: str = "right",
                   face: bool = True, point_palm: bool = True,
                   max_turn: float = np.deg2rad(75.0)) -> dict:
    """The full ask for one grasp waypoint: where to stand facing, how to hold.

    ``hand_dir`` is the horizontal direction from the root to the object, the
    same one the position offset already uses -- the point is precisely that the
    position goal and the orientation goal should tell the same story, and until
    now only the position half was being said out loud.
    """
    from rl.kimodo.heft.scene_frame import approach_from_root

    root = tr.data.xpos[tr.bid_root]
    hand_dir = approach_from_root(np.asarray(grasp_point, dtype=np.float64), root)
    out = {"hand_dir": hand_dir, "hand_fix": None, "body_yaw": None}
    if face:
        out["body_yaw"] = face_yaw(tr, grasp_point, max_turn=max_turn)
    if point_palm:
        # Computed AFTER the turn, and told about it: the actor rotates the whole
        # final pose before applying this, so the two must compose in that order.
        out["hand_fix"] = palm_fix(tr, hand_dir, side, body_yaw=out["body_yaw"])
    return out
