"""Assemble the tracker's 1729-dim observation. Pure numpy, no simulator.

Ported from ``heft_work/kimodo_rl/rollout.py``'s ``observe`` / ``_push_history``,
with one deliberate structural change: **the state reads are not in here.** This
class is handed a state snapshot and a reference and returns a vector. mjlab
reads ``robot.data.*``; robosuite reads ``sim.data``; both produce the same four
quantities, and keeping the assembly separate means the arithmetic -- which is
where the subtle errors live -- can be verified against the torch original on
synthetic states, with no simulator of either kind involved.

The concatenation order IS the contract. It is copied line for line from the
runtime; there is no way to check it from the outside, because any permutation
of 1729 floats is still 1729 floats and the policy will happily consume it.

Frame conventions, each of which has a wrong answer that runs fine:

  * ``root_pos`` is in the env-local frame, i.e. world minus the scene's floor
    datum, because the reference lives there. In robosuite it must also come
    from the ANATOMICAL pelvis: the body named ``pelvis`` is the floating base
    at floor level, 0.793 m below (see model_patch.resolve_root_body).
  * ``root_quat`` is wxyz -- MuJoCo's ``xquat`` order. robosuite's
    ``mat2quat`` returns xyzw, so ``obs["robot0_base_quat"]`` is the wrong order
    AND the wrong body.
  * ``ang_vel`` is in the BODY frame. mjlab exposes it in the world frame and
    ``_push_history`` rotates it; a MuJoCo free joint's ``qvel[3:6]`` is already
    body-local, so rotating it a second time corrupts 27 dims. This class takes
    body frame and says so, rather than guessing which one it was handed.
  * ``joint_pos`` / ``joint_vel`` are in CANONICAL order. The reference arrives
    in dataset order and is permuted here, once, at set_reference.

Note ``projected_gravity`` in the history IS normalised while the target gravity
in the observation is NOT. That asymmetry is in the runtime; it is reproduced,
not tidied.
"""

from __future__ import annotations

import numpy as np

from rl.kimodo.heft.tracker_const import (
    BOOT_INDICATOR_MAX,
    DS_TO_CANON,
    FUTURE_STEPS,
    HISTORY_STEPS,
    N_JOINTS,
    OBS_DIM,
    PREV_ACTION_STEPS,
    obs_slices,
    quat_apply_inv,
    quat_to_rot6d_rel,
)

_GRAVITY_W = np.array([0.0, 0.0, -1.0])


class TrackerObs:
    """Reference + rolling state history -> [N, 1729].

    Usage per 50 Hz control tick, in this order (the order matters: the history
    must contain the current state before the observation is built, and the
    action must be pushed after the policy has produced it):

        obs.push_history(root_quat, ang_vel_b, qpos_canon, qvel_canon)
        v = obs.observe(root_pos_local, root_quat, qpos_canon)
        delta, raw = policy(v)
        obs.push_action(raw)
        obs.advance()
    """

    def __init__(self, num_envs: int = 1):
        self.n = int(num_envs)
        self.fut = np.asarray(FUTURE_STEPS, dtype=np.int64)
        self.hist = np.asarray(HISTORY_STEPS, dtype=np.int64)
        self.hist_len = int(self.hist.max()) + 1
        self.slices = obs_slices()
        self.ref = None
        self.reset_buffers()

    # ------------------------------------------------------------------ buffers
    def reset_buffers(self) -> None:
        n = self.n
        self.h_angvel = np.zeros((n, self.hist_len, 3))
        self.h_grav = np.zeros((n, self.hist_len, 3))
        self.h_qpos = np.zeros((n, self.hist_len, N_JOINTS))
        self.h_qvel = np.zeros((n, self.hist_len, N_JOINTS))
        self.prev_act = np.zeros((n, PREV_ACTION_STEPS, N_JOINTS))
        self.last_action = np.zeros((n, N_JOINTS))
        self.idx = np.zeros(n, dtype=np.int64)
        self.boot = np.full(n, float(BOOT_INDICATOR_MAX))

    def set_reference(self, ref_qpos36: np.ndarray, reboot: bool = True) -> None:
        """``ref_qpos36`` [N, T, 36] at 50 Hz, in DATASET joint order.

        Does not touch the history buffers. At a chunk boundary that is the whole
        point: the continuation was generated from the robot's realized state, so
        clearing history would erase the drift the project exists to measure.
        """
        ref = np.asarray(ref_qpos36, dtype=np.float64)
        if ref.ndim == 2:
            ref = ref[None]
        if ref.shape[0] != self.n or ref.shape[2] != 36:
            raise ValueError(f"expected [{self.n}, T, 36], got {ref.shape}")
        if not np.isfinite(ref).all():
            # A non-finite reference does not raise in MuJoCo -- it poisons the
            # state with NaN, every later reward is NaN, and the step rate
            # collapses by 10x. Catch it at the door.
            bad = np.argwhere(~np.isfinite(ref))[:3]
            raise ValueError(f"reference has non-finite entries, e.g. at {bad.tolist()}")

        self.ref = ref
        self.T = ref.shape[1]
        # Permute the reference's joints into canonical order ONCE. self.ref
        # stays in dataset order so callers can compare against it without
        # juggling a second convention.
        self.ref_canon = np.concatenate(
            [ref[..., 0:7], ref[..., 7:36][..., DS_TO_CANON]], axis=-1)
        self.idx = np.zeros(self.n, dtype=np.int64)
        if reboot:
            self.boot = np.full(self.n, float(BOOT_INDICATOR_MAX))

    # ------------------------------------------------------------------ updates
    def push_history(self, root_quat_wxyz: np.ndarray, ang_vel_body: np.ndarray,
                     joint_pos_canon: np.ndarray, joint_vel_canon: np.ndarray) -> None:
        """Shift the rolling buffers and write the current state at index 0.

        ``ang_vel_body`` is ALREADY in the body frame -- see the module docstring.
        """
        q = np.atleast_2d(np.asarray(root_quat_wxyz, dtype=np.float64))
        g = quat_apply_inv(q, np.broadcast_to(_GRAVITY_W, (q.shape[0], 3)))
        g = g / np.maximum(np.linalg.norm(g, axis=-1, keepdims=True), 1e-8)

        for buf, val in (
            (self.h_angvel, np.atleast_2d(ang_vel_body)),
            (self.h_grav, g),
            (self.h_qpos, np.atleast_2d(joint_pos_canon)),
            (self.h_qvel, np.atleast_2d(joint_vel_canon)),
        ):
            buf[:, 1:] = buf[:, :-1]
            buf[:, 0] = val

    def push_action(self, raw_canon: np.ndarray) -> None:
        """Store the policy's RAW output -- clipped, NOT scaled, canonical order.

        The runtime fills PrevActions with ``last_action``, which is the
        unscaled value. Storing the scaled delta instead corrupts 232 of the
        1729 dims and raises nothing (heft_work silent bug #5).
        """
        raw = np.atleast_2d(np.asarray(raw_canon, dtype=np.float64))
        self.prev_act[:, 1:] = self.prev_act[:, :-1]
        self.prev_act[:, 0] = raw
        self.last_action = raw.copy()

    def advance(self) -> None:
        """One control tick of the boot counter and the reference cursor.

        Must run exactly once per 50 Hz tick. robosuite calls its controller once
        per 0.002 s physics substep, so this belongs inside the policy-rate gate;
        outside it, the cursor runs 10x fast and every history buffer is garbage.
        """
        self.boot = np.maximum(self.boot - 1.0, 0.0)
        self.idx = np.minimum(self.idx + 1, self.T - 1)

    # ------------------------------------------------------------------ observe
    def _ref_at(self, offsets: np.ndarray) -> np.ndarray:
        """[N, len(offsets), 36] canonical, clamped at both ends like the runtime."""
        i = np.clip(self.idx[:, None] + offsets[None, :], 0, self.T - 1)
        return np.take_along_axis(self.ref_canon, i[..., None], axis=1)

    def observe(self, root_pos_local: np.ndarray, root_quat_wxyz: np.ndarray,
                joint_pos_canon: np.ndarray) -> np.ndarray:
        if self.ref is None:
            raise RuntimeError("set_reference() before observe()")
        n = self.n
        root_quat = np.atleast_2d(np.asarray(root_quat_wxyz, dtype=np.float64))
        qj = np.atleast_2d(np.asarray(joint_pos_canon, dtype=np.float64))
        _ = np.atleast_2d(np.asarray(root_pos_local, dtype=np.float64))

        ref = self._ref_at(self.fut)                       # [N, F, 36] canonical
        r_pos, r_quat, r_qj = ref[..., 0:3], ref[..., 3:7], ref[..., 7:36]
        n_f = len(self.fut)

        # TrackingCommandObsRaw: inter-step displacement expressed in the
        # REFERENCE's own root frame (not the robot's), then rot6d of the
        # reference against the robot's CURRENT orientation.
        pos_diff_w = r_pos[:, 1:] - r_pos[:, 0:1]          # [N, F-1, 3]
        q0 = np.broadcast_to(r_quat[:, 0:1], (n, n_f - 1, 4))
        pos_diff_b = quat_apply_inv(q0, pos_diff_w)
        cur_q = np.broadcast_to(root_quat[:, None, :], (n, n_f, 4))
        rot6d = quat_to_rot6d_rel(cur_q, r_quat)

        tgt_minus_cur = r_qj - qj[:, None, :]
        root_z = r_pos[..., 2]
        # NOT normalised here, unlike the history's projected gravity. That
        # asymmetry is in the runtime.
        tgt_grav = quat_apply_inv(r_quat, np.broadcast_to(_GRAVITY_W, (n, n_f, 3)))

        out = np.concatenate([
            (self.boot / BOOT_INDICATOR_MAX)[:, None],
            pos_diff_b.reshape(n, -1), rot6d.reshape(n, -1),
            r_qj.reshape(n, -1), tgt_minus_cur.reshape(n, -1),
            root_z,
            tgt_grav.reshape(n, -1),
            self.h_angvel[:, self.hist].reshape(n, -1),
            self.h_grav[:, self.hist].reshape(n, -1),
            self.h_qpos[:, self.hist].reshape(n, -1),
            self.h_qvel[:, self.hist].reshape(n, -1),
            self.prev_act.reshape(n, -1),
        ], axis=1)

        if out.shape[1] != OBS_DIM:
            raise RuntimeError(f"assembled {out.shape[1]} dims, expected {OBS_DIM}")
        return out

    # ------------------------------------------------------------------ probes
    def split(self, obs: np.ndarray) -> dict[str, np.ndarray]:
        """Slice a 1729-vector into its 10 named blocks.

        The per-block view is what makes a mismatch diagnosable: a joint-order
        error moves 1276 dims together, a quaternion-order error moves only the
        gravity blocks, and a scaled-PrevActions error moves exactly 232.
        """
        obs = np.atleast_2d(obs)
        return {name: obs[:, sl] for name, sl in self.slices.items()}
