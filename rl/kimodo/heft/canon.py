"""Generate in the frame Kimodo was trained in, not the one the room happens to use.

⚠️ THE SINGLE LARGEST DEFECT FOUND IN THIS PIPELINE. Training's prefix is always
the standing pose with an identity quaternion -- ``EefReachEnv.prefix`` is
``np.tile(self.stand, (5, 1))`` and ``stand`` faces +x. ``_encode_continue``
re-anchors the chunk's TRANSLATION so the prefix starts at the origin, and does
nothing about its HEADING. A scene spawns the robot facing whatever the room
dictates, so every rollout in a room that is not aligned with world +x asks the
generator for something it has never seen.

Measured on `IThorKitchenAppleOnPlate`, same ask relative to the robot, same
checkpoint, reference wrist distance to the goal:

    scene          yaw    as-is    yaw canonicalised
    FloorPlan22      0    0.107    0.107
    FloorPlan14     90    0.628    0.104
    FloorPlan18     90    0.625    0.101

Six times worse at 90 degrees, and fully recovered by rotating. It explains a
day of confusion: `IThorKitchenCloseDrawer` spawns near yaw 0 and measured 0.037,
`AppleOnPlate` spawns at 90 and measured 0.6, and the "3.4x room-to-room spread"
was rooms differing in spawn heading.

So: rotate the prefix and every goal about the robot's own root until it faces
+x, generate, and rotate the chunk back.

⚠️⚠️ **NOT WIRED IN, AND NOT CORRECT YET.** The finding above is solid -- it
reproduces across scenes and the canonical-frame generation really does land at
0.10 where the raw one lands at 0.63. The TRANSFORM here is not: `prefix()`
rotates about the robot's root while `unrotate_chunk()` rotates about the chunk's
own anchored origin, and those two are not inverses (round-trip position error
0.407 m on a synthetic case). Wiring it into the driver measured ref_err
0.648 -> 0.857, i.e. worse than leaving the heading alone.

What is still needed: work out exactly which frame `_encode_continue` returns the
chunk in -- it subtracts the prefix's first frame, so the answer is probably
"anchored at the prefix start, in the prefix's heading" -- and make the two
methods exact inverses under that convention. Until then this module is a
recorded diagnosis, not a fix.
"""

from __future__ import annotations

import numpy as np


def yaw_of(quat_wxyz) -> float:
    """Heading in radians from a wxyz quaternion."""
    w, x, y, z = (float(v) for v in quat_wxyz)
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _qmul(a, b) -> np.ndarray:
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], dtype=np.float64)


def _qz(angle: float) -> np.ndarray:
    return np.array([np.cos(angle / 2.0), 0.0, 0.0, np.sin(angle / 2.0)])


def _rot(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _sign_continuous(frames) -> np.ndarray:
    """Make the quaternion track continuous in SIGN, frame to frame.

    ⚠️ Not cosmetic, and the failure it prevents is spectacular rather than
    subtle. q and -q are the same rotation, so nothing downstream that reads one
    frame notices -- but the reference conversion gets ROOT ANGULAR VELOCITY by
    differencing consecutive quaternions, and one sign flip inside the sequence
    differences as a full turn in a 20 ms step.

    It bites exactly at a half turn. Rotating a near-upright pose by 180 degrees
    lands it next to the antipode, where w passes through 0 and the sign of the
    whole quaternion is decided by numerical noise. Measured, on the empty-room
    probe at --spawn-yaw 180: without this the robot fell in 16 of 20 draws and
    the hand finished 2.15 m out; the same transform at 90 degrees, nowhere near
    the antipode, worked without it (dwell 6/20 -> 15/20).
    """
    out = np.asarray(frames, dtype=np.float32).copy()
    for i in range(1, len(out)):
        if float(np.dot(out[i][3:7], out[i - 1][3:7])) < 0.0:
            out[i][3:7] = -out[i][3:7]
    return out


class Canonical:
    """Rotate about ``pivot`` so ``yaw`` becomes 0; rotate results back.

    ``pivot`` is the robot's root in the same frame the prefix and goals use --
    rotating about the root rather than the origin keeps the robot where it is
    and only changes which way it is looking.
    """

    def __init__(self, prefix, pivot):
        self.pivot = np.asarray(pivot, dtype=np.float64).copy()
        self.yaw = yaw_of(np.asarray(prefix, dtype=np.float64)[-1][3:7])
        self.R_to = _rot(-self.yaw)
        self.R_back = _rot(self.yaw)

    # ------------------------------------------------------------------ in
    def prefix(self, prefix) -> np.ndarray:
        """[F, 36] prefix rotated to face +x."""
        out = np.asarray(prefix, dtype=np.float32).copy()
        q = _qz(-self.yaw)
        for f in range(len(out)):
            out[f][:3] = (self.pivot + self.R_to
                          @ (np.asarray(out[f][:3], dtype=np.float64) - self.pivot))
            out[f][3:7] = _qmul(q, np.asarray(out[f][3:7], dtype=np.float64))
        return _sign_continuous(out)

    def point(self, p):
        """A world/local 3-D goal, rotated into the canonical frame."""
        if p is None:
            return None
        v = np.asarray(p, dtype=np.float64)
        return (self.pivot + self.R_to @ (v - self.pivot)).astype(np.float32)

    def xy(self, p):
        """A 2-D root goal. Same rotation, kept separate because the caller's
        arrays are 2-D and silently broadcasting a 3-D pivot would not raise."""
        if p is None:
            return None
        v = np.asarray(p, dtype=np.float64).reshape(2)
        r = self.pivot[:2] + self.R_to[:2, :2] @ (v - self.pivot[:2])
        return r.astype(np.float32)

    # ----------------------------------------------------------------- out
    def unrotate_chunk(self, q36) -> np.ndarray:
        """[T, 36] generated chunk, rotated back into the scene's frame.

        ⚠️ About the ORIGIN, not about the pivot. A chunk comes back in its own
        anchored frame -- `_encode_continue` translates the prefix's first frame
        to zero -- so the position that has to be rotated is already relative,
        and rotating it about the robot's root as well adds the pivot offset
        twice. The first version did that and made the rollout WORSE than no fix
        at all (ref_err 0.648 -> 0.857) while the isolated test said 0.628 ->
        0.104; the difference was entirely this line.
        """
        out = np.asarray(q36, dtype=np.float32).copy()
        q = _qz(self.yaw)
        for t in range(len(out)):
            out[t][:3] = self.R_back @ np.asarray(out[t][:3], dtype=np.float64)
            out[t][3:7] = _qmul(q, np.asarray(out[t][3:7], dtype=np.float64))
        return _sign_continuous(out)
