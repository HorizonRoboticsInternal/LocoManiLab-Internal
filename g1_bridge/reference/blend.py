"""Smooth re-entry for a motion-tracking reference: hold a pose, then ramp back
onto the live stream instead of snapping to it.

The problem this solves. ``RealtimeMotionBuffer`` already stitches a reference
discontinuity, but only in the two dimensions it can reason about: its
``motion_first_frame`` handler re-attaches an incoming segment with a *yaw-only
rotation plus xy translation*. Height, roll, pitch and all 29 joint angles pass
straight through. So when an operator stops tracking, moves, and starts again,
the robot's reference teleports from the pose it was holding to whatever shape
the operator is now in -- inside a single control tick, with a finite-difference
velocity to match.

The reference transition anchors on the held pose, aligns the resuming stream
to it, then outputs a weighted blend of the
two whose weight ramps 0 -> 1 over ``blend_s``. After that the live frame passes
through untouched.

Two choices here are load-bearing and easy to get wrong:

* **Blend in qpos space, not in body-pose space.** ``PoseBlender`` returns a
  flat qpos; the caller FKs it (``DefaultReferenceFrame.payload_from_qpos``) to
  get the body arrays the buffer wants. Lerping the 33 body positions directly
  is not kinematically valid -- a 90-degree elbow difference foreshortens the
  forearm by ~29% at alpha=0.5 -- and it is worst exactly when the two poses
  differ most, which is the case the blend exists for. It also guarantees the
  blend's endpoint equals the live payload's own arrays bit-for-bit, since
  those are themselves FK of its qpos.

* **alpha advances on the motion stream's clock, not per push.** The buffer
  finite-differences reference velocity over the timestamps we hand it, so
  making alpha a function of those same timestamps means it derives the true
  derivative of the trajectory we are commanding. Per-push alpha would instead
  make the induced reference velocity scale with the sim/wall-clock ratio -- a
  headless sim running 10x real time would inject 10x the reference velocity
  (the hazard already documented at
  ``composite._push_synthetic_default_once``). A stalled stream is handled for
  free: the timestamp stops advancing, so alpha does too, and the frames pushed
  meanwhile are content-identical.
"""

from __future__ import annotations

import time

import numpy as np

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from sim2real.rl_policy.utils.motion import _quat_slerp_batch
from sim2real.utils.math import quat_conjugate, quat_mul, yaw_quat

# The publisher's flat-qpos convention: free root then the RobotCfg joints.
# RealtimeMotionBuffer assumes the same thing when it auto-slices a 36-long
# "qpos" payload key as [7:7+njoints].
QPOS_ROOT_POS = slice(0, 3)
QPOS_ROOT_QUAT = slice(3, 7)
QPOS_JOINTS = slice(7, None)

#: Wall-clock multiple of ``blend_s`` after which a blend is force-completed.
#: Without it, a motion stream whose clock froze mid-blend would strand the
#: reference partway between the anchor and the live pose forever.
_STALL_FACTOR = 3.0


def qpos_from_payload(payload: dict, root_body_idx: int = 0) -> np.ndarray:
    """The payload's pose as a flat qpos.

    Prefers the payload's own ``qpos`` key (the publisher writes it for every
    frame, live or paused). Falls back to assembling it from the root body's
    world pose plus ``joint_pos``, which is the same information -- the buffer
    itself reads root pose out of ``body_pos_w``/``body_quat_w`` at
    ``root_body_idx``.
    """
    qpos = payload.get("qpos")
    if qpos is not None:
        return np.asarray(qpos, dtype=np.float64).reshape(-1)
    body_pos = np.asarray(payload["body_pos_w"], dtype=np.float64)
    body_quat = np.asarray(payload["body_quat_w"], dtype=np.float64)
    joints = np.asarray(payload["joint_pos"], dtype=np.float64).reshape(-1)
    return np.concatenate([body_pos[root_body_idx], body_quat[root_body_idx], joints])


def align_qpos_to(anchor: np.ndarray, live: np.ndarray) -> np.ndarray:
    """Re-express ``anchor`` at ``live``'s root xy and heading.

    Yaw only, and xy only: the anchor keeps its own height, roll, pitch and
    joint angles, and just gets moved to stand where the resuming stream says
    the root is. That is what lets the blend run in the raw stream's own frame
    while ``RealtimeMotionBuffer``'s segment transform stays the single
    authority on where the reference sits in the world.

    Why it comes out exactly continuous: the buffer's transform is a yaw-only
    rotation plus an xy translation, ``yaw_quat`` commutes with such a rotation,
    and forward kinematics is equivariant under a rigid root transform. So when
    the caller stamps ``motion_first_frame`` on the first blended frame (which
    is this aligned anchor, at alpha=0), the transform the buffer computes maps
    it onto precisely the frame it was already holding -- the stamp is a
    provable no-op rather than a small jump.
    """
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    live = np.asarray(live, dtype=np.float64).reshape(-1)
    anchor_yaw = yaw_quat(anchor[QPOS_ROOT_QUAT].astype(np.float32).reshape(1, 4))
    live_yaw = yaw_quat(live[QPOS_ROOT_QUAT].astype(np.float32).reshape(1, 4))
    yaw_off = quat_mul(live_yaw, quat_conjugate(anchor_yaw))

    out = anchor.copy()
    out[QPOS_ROOT_POS] = [
        live[0],
        live[1],
        anchor[2],  # keep the anchor's own height
    ]
    out[QPOS_ROOT_QUAT] = np.asarray(
        quat_mul(yaw_off, anchor[QPOS_ROOT_QUAT].astype(np.float32).reshape(1, 4)),
        dtype=np.float64,
    ).reshape(4)
    return out


def stand_in_place_qpos(anchor: np.ndarray, standing_joints, height=None) -> np.ndarray:
    """A standing pose placed exactly where ``anchor`` already stands.

    Keeps the anchor's root xy and *yaw*, drops its roll/pitch, and swaps in the
    standing joint angles. The point is the frame: ``anchor`` is the last frame
    pushed, so it lives in whatever raw frame the buffer's current motion
    segment is expressed in, and this target lands in that same frame. Building
    the target from the robot's own world pose instead -- which is the obvious
    thing to write, and was written first -- silently mixes two coordinate
    systems: the buffer goes on applying its segment transform (operator frame
    -> robot frame) to a frame that is *already* in the destination frame, so
    "stand still where you are" is delivered rotated and metres away, and the
    error compounds on every disengage.
    """
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    out = anchor.copy()
    out[QPOS_ROOT_QUAT] = np.asarray(
        yaw_quat(anchor[QPOS_ROOT_QUAT].astype(np.float32).reshape(1, 4)),
        dtype=np.float64,
    ).reshape(4)
    if height is not None:
        out[2] = float(height)
    out[QPOS_JOINTS] = np.asarray(standing_joints, dtype=np.float64).reshape(-1)
    return out


def blend_qpos(anchor: np.ndarray, live: np.ndarray, alpha: float) -> np.ndarray:
    """``anchor`` at alpha=0, ``live`` at alpha=1: lerp position and joints,
    shortest-path slerp the root orientation."""
    anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
    live = np.asarray(live, dtype=np.float64).reshape(-1)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    out = (1.0 - alpha) * anchor + alpha * live
    out[QPOS_ROOT_QUAT] = np.asarray(
        _quat_slerp_batch(
            anchor[QPOS_ROOT_QUAT].astype(np.float32).reshape(1, 4),
            live[QPOS_ROOT_QUAT].astype(np.float32).reshape(1, 4),
            np.asarray([alpha], dtype=np.float32),
        ),
        dtype=np.float64,
    ).reshape(4)
    return out


class PoseBlender:
    """One ramp from a held pose onto a moving (or fixed) target.

    Two shapes of use, both on the same object:

    * ``begin(anchor, live, t_ns)`` -- resuming tracking. The anchor is aligned
      to ``live``'s root (see :func:`align_qpos_to`) and every later
      :meth:`step` blends it against that tick's live frame, so the target
      keeps moving with the operator while the weight ramps.
    * ``begin(anchor, target, t_ns, align=False)`` -- ramping to a fixed pose
      (the standing default, on disengage). No alignment: the target already
      sits where it should, and ``step`` is called with that same target every
      tick.
    """

    def __init__(self, blend_s: float):
        self.blend_s = float(blend_s)
        self._anchor = None
        self._t0_ns = None
        self._wall_t0 = None
        self._alpha = 1.0

    @property
    def active(self) -> bool:
        return self._anchor is not None

    @property
    def alpha(self) -> float:
        """Weight of the live/target pose in the last :meth:`step` -- 1.0 when
        no blend is running (i.e. straight pass-through)."""
        return self._alpha

    def cancel(self) -> None:
        self._anchor = None
        self._t0_ns = None
        self._wall_t0 = None
        self._alpha = 1.0

    def begin(self, anchor, live, t_ns: int, *, align: bool = True) -> None:
        """Arm a ramp from ``anchor`` toward ``live`` starting at stream time
        ``t_ns``. A non-positive ``blend_s`` disables blending entirely, which
        is the documented way to get the old snap-to-live behaviour back."""
        if self.blend_s <= 0.0 or anchor is None:
            self.cancel()
            return
        anchor = np.asarray(anchor, dtype=np.float64).reshape(-1)
        self._anchor = align_qpos_to(anchor, live) if align else anchor
        self._t0_ns = int(t_ns)
        self._wall_t0 = time.monotonic()
        self._alpha = 0.0

    def restart_alignment(self, live) -> None:
        """Re-align the in-flight anchor to a stream that just re-anchored
        itself (the publisher stamps ``motion_first_frame`` after an XR
        reconnect). Keeps the elapsed weight; only the frame changes."""
        if self._anchor is not None:
            self._anchor = align_qpos_to(self._anchor, live)

    def step(self, live, t_ns: int) -> tuple[np.ndarray, float]:
        """This tick's reference qpos and the weight used for it.

        Returns ``live`` unchanged at alpha 1.0 when no blend is armed, so the
        caller has a single code path for blending and pass-through.
        """
        live = np.asarray(live, dtype=np.float64).reshape(-1)
        if self._anchor is None:
            self._alpha = 1.0
            return live, 1.0

        alpha = (int(t_ns) - self._t0_ns) / (self.blend_s * 1e9)
        # The stream's clock is the honest one (it is what the buffer
        # differences over), but it can stop; wall-clock is the backstop.
        if time.monotonic() - self._wall_t0 > _STALL_FACTOR * self.blend_s:
            alpha = 1.0
        alpha = float(np.clip(alpha, 0.0, 1.0))

        out = blend_qpos(self._anchor, live, alpha)
        self._alpha = alpha
        if alpha >= 1.0:
            self.cancel()
            self._alpha = 1.0
        return out, alpha
