"""P4-2: record states during a rollout, render only the episodes worth keeping.

Two reasons this exists, and the second one is the one that makes the batch path
possible at all:

  * **Cost follows yield, not attempts.** Rendering every rollout pays the camera
    cost for episodes the filters are about to throw away. Recording
    ``sim.get_state()`` instead and rendering afterwards means the render bill
    scales with how many episodes pass, which at an unknown yield is the
    difference between a batch that finishes and one that does not.
  * **A worker that renders cannot also generate.** An EGL offscreen context in a
    process that also holds a CUDA context deadlocks on shared multi-GPU nodes
    (DEBUG_LOG #11). Kimodo lives on the GPU, so the rollout process must not
    render, full stop. This is not an optimisation, it is the constraint the
    whole batch design is built around.

⚠️ **Re-rendering is only faithful into the same model.** ``env.reset()`` under
``hard_reset`` rebuilds the MjModel and resamples the objects, so replaying
recorded qpos into it draws a different scene. h10 measured this as joint drift
3.1e-2 versus exactly 0.0 with a soft reset; here it would show up as a
re-render that looks plausible and is of another episode. ``StateRecorder``
therefore stores the model's identity alongside the states and refuses to render
into a model that does not match.
"""

from __future__ import annotations

import numpy as np


class StateRecorder:
    """Collects per-tick sim states, plus enough identity to detect a mismatch."""

    def __init__(self, tr):
        self.tr = tr
        self.reset()

    def reset(self) -> None:
        self.qpos: list[np.ndarray] = []
        self.qvel: list[np.ndarray] = []
        # nq/nbody/ngeom together are a cheap fingerprint of "same model". A
        # resampled object set changes at least one of them, and comparing the
        # full model would be both slow and impossible to store per episode.
        m = self.tr.model
        self.fingerprint = (int(m.nq), int(m.nv), int(m.nbody), int(m.ngeom))

    def step(self, i=None, tr=None) -> None:
        d = self.tr.data
        self.qpos.append(d.qpos.copy())
        self.qvel.append(d.qvel.copy())

    def to_dict(self) -> dict:
        return {"qpos": np.asarray(self.qpos), "qvel": np.asarray(self.qvel),
                "fingerprint": np.asarray(self.fingerprint)}


#: Azimuths tried, in degrees, when the requested one is blocked. Ordered so the
#: search gives up the requested view as slowly as possible -- a camera that
#: jumps to the far side of the room every time a chair edge clips the ray is
#: harder to read than one that stays put.
AZIMUTH_SWEEP = (0.0, 30.0, -30.0, 60.0, -60.0, 90.0, -90.0,
                 120.0, -120.0, 150.0, -150.0, 180.0)


#: Where the clearance ray starts, in metres out from the lookat point. Inside
#: this radius are the robot's own geoms, which are the subject, not occluders.
CAM_NEAR = 0.35
#: Stop backing off this far short of whatever the ray hit.
CAM_MARGIN = 0.20
#: A view with at least this much room is good enough to keep -- the camera only
#: goes looking for a better angle below it. Pure hysteresis: without it the
#: camera hunts every time the ray grazes a chair leg.
CAM_GOOD = 1.6


def visual_only_option(cutaway: bool = False):
    """The render options robosuite itself uses for a camera image.

    ⚠️ MuJoCo's default shows geom groups 0, 1 and 2, and in a robocasa scene
    group 0 is the COLLISION geometry -- 465 opaque boxes and capsules that sit
    on top of the visual meshes. Rendering with the default produces flat olive
    slabs where the walls are and green blobs where the plant is; the same frame
    with group 0 hidden is a wood-grain table, a plant with leaves and a lamp
    with a shade. That is not a preference, it is the difference between the
    dataset's images and the images the teleop collector produced.

    robosuite's own setting is the authority and this reproduces it exactly:
    ``environments/base.py`` writes ``vopt.geomgroup[0] = render_collision_mesh``
    and ``[1] = render_visual_mesh``, whose defaults are False and True. Groups
    2 and up are left at MuJoCo's default, which is why the end-effector target
    markers still appear -- they appear in the collector's videos too.
    """
    import mujoco

    opt = mujoco.MjvOption()
    opt.geomgroup[0] = 0            # collision: off, as robosuite does
    opt.geomgroup[1] = 1            # visual: on
    if cutaway:
        opt.flags[mujoco.mjtVisFlag.mjVIS_STATIC] = 0
    return opt


def camera_clearance(m, d, lookat, azimuth, elevation, want,
                     *, near: float = CAM_NEAR, margin: float = CAM_MARGIN) -> float:
    """How far back a free camera can sit on this view axis before it hits something.

    Returns ``want`` when nothing is in the way, otherwise the reduced distance.

    ⚠️ The ray is cast **outward from the robot**, not inward from the camera,
    and the direction matters more than it looks. Casting from the eye is the
    obvious way round and it fails in exactly the case that matters: a camera
    already embedded in a wall starts INSIDE that geom, ``mj_ray`` returns no
    hit, and the frame is scored as a clear view. That is not hypothetical --
    `IThorBathroomApproachToilet` rendered 751 frames of flat olive wall with the
    old test reporting every one of them unobstructed. From the robot outward
    the start point is always in free space, so the hit is real, and its distance
    is also the answer to "how close does the camera have to come".

    ⚠️ ``flg_static=1``: walls are static geoms on the worldbody and are the
    entire point.
    """
    import mujoco

    az, el = np.radians(float(azimuth)), np.radians(float(elevation))
    # MuJoCo's free camera sits BEHIND its lookat, along -forward.
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    back = -fwd
    pnt = np.asarray(lookat, dtype=np.float64) + float(near) * back
    gid = np.zeros(1, dtype=np.int32)
    hit = float(mujoco.mj_ray(m, d, pnt, back, None, 1, -1, gid))
    if hit < 0.0:
        return float(want)
    return float(min(float(want), max(0.0, near + hit - margin)))


def camera_blocked(m, d, lookat, azimuth, elevation, distance,
                   *, margin: float = CAM_MARGIN) -> bool:
    """Would a camera at ``distance`` on this axis be looking through something?"""
    return camera_clearance(m, d, lookat, azimuth, elevation, distance,
                            margin=margin) < float(distance) - 1e-6


def _eye(lookat, azimuth, elevation, distance) -> np.ndarray:
    az, el = np.radians(float(azimuth)), np.radians(float(elevation))
    fwd = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    return np.asarray(lookat, dtype=np.float64) - float(distance) * fwd


def visible_count(m, d, eye, points, *, margin: float = 0.06) -> int:
    """How many of ``points`` this eye can actually SEE.

    The clearance test asks whether the line to the robot's centre is clear. That
    is not the same question as whether the thing being looked at is visible, and
    a kitchen makes the difference concrete: on `IThorKitchenAppleOnPlate` the
    line to the pelvis was clear while the pelvis itself sat behind a counter,
    so the frame was a perfectly good picture of a room with the subject hidden
    in it -- 0 of 2 markers on screen.
    """
    import mujoco

    eye = np.asarray(eye, dtype=np.float64)
    seen = 0
    gid = np.zeros(1, dtype=np.int32)
    for p in points:
        p = np.asarray(p, dtype=np.float64).reshape(3)
        vec = p - eye
        n = float(np.linalg.norm(vec))
        if n < 1e-9:
            seen += 1
            continue
        hit = float(mujoco.mj_ray(m, d, eye, vec / n, None, 1, -1, gid))
        if hit < 0.0 or hit >= n - margin:
            seen += 1
    return seen


def render_states(tr, states: dict, *, size=(480, 640), azimuth: float = 90.0,
                  elevation: float = -8.0, distance: float = 2.6,
                  lookat=None, stride: int = 1, marks=None,
                  cutaway: bool = False) -> list:
    """Draw recorded states through the CURRENT sim. Returns a list of frames.

    Must run in a process with no CUDA context. The caller is responsible for
    that; asserting it here is not possible without importing torch, which is
    exactly what must not happen.

    With ``lookat=None`` the camera FOLLOWS the robot and steps around whatever
    is in the way. Both halves were forced by watching the output: the fixed
    camera was aimed once, at the reset pose, so an episode that walks 2 m leaves
    it filming an empty patch of floor; and on `IThorLivingRoomLiftStatue` all 2000
    frames were a flat yellow rectangle, the camera being inside a wall
    (known_issues I-2). Pass an explicit ``lookat`` to pin the camera exactly --
    which is what a pixel-for-pixel comparison such as h12 needs.
    """
    import mujoco

    m, d = tr.model, tr.data
    fp = (int(m.nq), int(m.nv), int(m.nbody), int(m.ngeom))
    got = tuple(int(x) for x in states["fingerprint"])
    if fp != got:
        raise SystemExit(
            f"[render] model fingerprint {fp} does not match the recording's "
            f"{got}. The scene was rebuilt between recording and rendering -- "
            "almost certainly a hard reset -- so these states describe a "
            "different episode and rendering them would produce a convincing "
            "video of something that never happened.")

    h, w = size
    renderer = mujoco.Renderer(m, height=h, width=w)
    # ⚠️ CUTAWAY hides every STATIC geom -- walls, floor, counters, fixtures --
    # leaving the robot, the free objects and the markers. It is the answer to
    # the room the camera cannot get out of: `IThorKitchenAppleOnPlate` puts the
    # robot in an aisle with 0.2-0.6 m of clearance in every direction, so the
    # honest full-context frame is a close-up of a torso.
    #
    # It is OFF by default and always will be: a picture with the furniture
    # removed cannot answer "is the robot walking through the table", which is
    # the question most of these videos exist to answer.
    # ⚠️ ALWAYS set, not only for cutaway. Leaving it unset is what drew the
    # collision geometry over every third-person video this pipeline has
    # produced -- see `visual_only_option`.
    renderer._scene_option = visual_only_option(cutaway)
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(m, cam)
    cam.elevation = elevation
    cam.azimuth = azimuth
    cam.distance = distance
    follow = lookat is None
    cam.lookat[:] = (np.asarray(lookat, dtype=np.float64) if not follow
                     else d.xpos[tr.bid_root] + np.array([0.6, 0.0, 0.2]))

    frames = []
    saved_q, saved_v = d.qpos.copy(), d.qvel.copy()
    try:
        for k in range(0, len(states["qpos"]), stride):
            d.qpos[:] = states["qpos"][k]
            d.qvel[:] = states["qvel"][k]
            mujoco.mj_forward(m, d)
            if follow:
                cam.lookat[:] = d.xpos[tr.bid_root] + np.array([0.0, 0.0, 0.2])
                want = [np.asarray(mk[0], dtype=np.float64)
                        for mk in (marks(k) if callable(marks) else (marks or []))
                        if mk[0] is not None and np.isfinite(mk[0]).all()]
                # Two knobs, in this order: turn, then come closer. Turning alone
                # cannot help in a room smaller than `distance` -- a 2 m bathroom
                # has no azimuth at which a 2.6 m camera is outside the walls, so
                # the sweep would pick a direction and still film tiling.
                #
                # ⚠️ Scored on what is VISIBLE, not only on what is unblocked.
                # Clearance asks "is the line to the robot's centre free"; a
                # counter can leave that line free while hiding everything worth
                # looking at behind it.
                def score(az):
                    c = camera_clearance(m, d, cam.lookat, az, elevation, distance)
                    if not want:
                        return (0, c)
                    eye = _eye(cam.lookat, az, elevation, max(0.6, c))
                    return (visible_count(m, d, eye, want), c)

                best = score(cam.azimuth)
                if best[0] < len(want) or best[1] < min(distance, CAM_GOOD):
                    for off in AZIMUTH_SWEEP:
                        cand = score(azimuth + off)
                        if cand > best:
                            best, cam.azimuth = cand, azimuth + off
                        if best[0] >= len(want) and best[1] >= min(distance, CAM_GOOD):
                            break
                # Never further than the room allows, never so close that the
                # camera is inside the robot.
                cam.distance = max(0.6, best[1])
            renderer.update_scene(d, camera=cam)
            if marks is not None:
                from rl.kimodo.heft.view import _draw_markers

                _draw_markers(renderer.scene, marks(k) if callable(marks) else marks)
            frames.append(renderer.render().copy())
    finally:
        d.qpos[:] = saved_q
        d.qvel[:] = saved_v
        mujoco.mj_forward(m, d)
        renderer.close()
    return frames


def write_mp4(frames, path: str, fps: int = 50) -> str:
    import imageio.v2 as imageio

    with imageio.get_writer(path, fps=fps, macro_block_size=1) as w:
        for f in frames:
            w.append_data(f)
    return path
