"""Draw the arrival criterion as a ring on the floor, and judge the root by it.

    the ring     radius R around the target, drawn flat on the floor
    amber        the root is outside it
    green        the root is inside it
    the dot      where the root actually is, at floor height, so "how far short"
                 is readable without counting pixels

This is a ROOT-TO-TARGET criterion: horizontal distance from the robot's root to
the target point, against one radius. It is deliberately simpler than what the
task's own ``_check_success`` computes (``mj_geomDistance`` surface to surface,
held 10 steps), and it is the one to draw when the question being asked is "did
the body get there".

⚠️ **THE TWO CAN DISAGREE AND THE PICTURE ONLY SHOWS THIS ONE.** h31 measured
the relation on the toilet: a root 1.0 m from the fixture centre is 0.369 m from
its surface, so a 1.0 m ring would enclose poses the task refuses; and the
conversion is per fixture and per azimuth -- on `IThorBedroomApproachBed` a
single radius of 1.2 m spans surface distances from 0.027 to 0.34 m depending
which side of the bed the robot stands. Pick the radius knowing that, and read
`success_rate` (which is still the task's own verdict) separately from the ring.
"""

from __future__ import annotations

import numpy as np

#: Ring colours: outside, inside.
FAR_RGBA = (1.0, 0.65, 0.0, 0.85)
NEAR_RGBA = (0.1, 0.95, 0.25, 0.9)
#: Segments the ring is drawn from. MjvScene takes primitives only, so a circle
#: is a fan of capsules; 48 is smooth at the distances these rings are viewed at
#: and leaves plenty of room under `maxgeom`.
SEGMENTS = 48
#: Capsule radius of the ring itself, metres.
WIDTH = 0.012
#: How far above the floor the ring sits, metres. Just enough not to z-fight
#: with the floor geom, which otherwise makes it flicker.
LIFT = 0.01


def root_ring_probe(tr, target, radius: float):
    """A callable returning ``(dist, root_xyz, inside)`` for the current pose.

    ``target`` is the point the ring is centred on, in WORLD. ``dist`` is
    horizontal: the root's height is not part of "did the body get there", and
    including it would make a robot that crouches read as further away.
    """
    c = np.asarray(target, dtype=np.float64)[:2]
    r = float(radius)

    def probe():
        root = np.asarray(tr.data.xpos[tr.bid_root], dtype=np.float64)
        d = float(np.linalg.norm(root[:2] - c))
        return d, root, bool(d <= r)

    probe.radius = r
    probe.centre = c
    return probe


def draw(scene, state, probe) -> None:
    """Add the ring and the root dot to an MjvScene."""
    import mujoco

    from rl.kimodo.heft.view import add_marker

    if state is None:
        return
    dist, root, inside = state
    rgba = np.asarray(NEAR_RGBA if inside else FAR_RGBA, dtype=np.float32)
    c, r = probe.centre, probe.radius
    # Floor height taken from the ROOT's own z minus its standing height would
    # need a standing height; the scene's floor is at z=0 in these rooms and the
    # ring is lifted off it, which is enough to stop the z-fight.
    z = LIFT
    prev = None
    for k in range(SEGMENTS + 1):
        a = 2.0 * np.pi * k / SEGMENTS
        p = np.array([c[0] + r * np.cos(a), c[1] + r * np.sin(a), z])
        if prev is not None:
            if scene.ngeom >= scene.maxgeom:
                break
            g = scene.geoms[scene.ngeom]
            mujoco.mjv_initGeom(
                g, type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=np.zeros(3),
                pos=np.zeros(3), mat=np.eye(3).reshape(9), rgba=rgba)
            mujoco.mjv_connector(g, int(mujoco.mjtGeom.mjGEOM_CAPSULE), WIDTH,
                                 prev, p)
            scene.ngeom += 1
        prev = p
    # Where the root actually is, projected to the ring's own plane. Without it
    # the ring says whether the robot is in, but not by how much it missed --
    # and "0.05 m short" and "0.6 m short" are different bugs.
    add_marker(scene, np.array([root[0], root[1], z]), rgba, 0.05)
