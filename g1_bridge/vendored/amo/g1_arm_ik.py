"""G1 dual-arm inverse kinematics (pinocchio, damped least squares).

Adapted from Psi0's ``G1_29_ArmIK`` (real/teleop/robot_control/robot_arm_ik.py),
but solved with a numerical DLS Jacobian iteration instead of casadi/ipopt — the
pip ``pin`` wheel doesn't ship the ``pinocchio.casadi`` bridge, and DLS is fast
enough (~0.15 ms) for 50 Hz teleop.

Locks the legs / waist / hands and solves the 14 arm joints (shoulder x3 +
elbow + wrist x3 per side) so the two wrist end-effector frames (``L_ee`` /
``R_ee``, 0.05 m past wrist_yaw) reach target SE3 poses (in the robot pelvis
frame). Model is loaded kinematics-only (no meshes).

Why this is a *guarded* Gauss-Newton and not a plain DLS loop
-------------------------------------------------------------
The first version of this port was a plain DLS iteration: full Newton step, no
line search, ``np.clip`` to the joint limits inside the loop, and a convergence
test that could never fire. Measured, it did not converge at all — it ran a
chaotic orbit at ~3 rad per iteration for all 30 iterations and returned
iterate #30, which is a *discontinuous* function of the target. At the operator
level that is violent twitching, and it started at an ordinary reach distance
(~0.25 m past the rest pose; the arm only has ~9 cm of radial headroom).

Four mechanisms keep it well behaved now. All four come from what Psi0's IPOPT
formulation gets for free, so this stays a faithful port rather than a
different algorithm:

* **Step clamp** (:data:`_STEP_MAX`) — keeps each step inside the region where
  the ``log6`` linearisation is valid. This is the dominant term: on the
  measured 0.30 m trajectory it alone takes the peak commanded joint velocity
  from 115 rad/s to 9.2 rad/s.
* **Backtracking line search** on the actual objective — makes the iteration a
  guaranteed-descent method, so a 30-iteration blow-up is structurally
  impossible rather than merely unlikely.
* **Temporal smoothness** (:data:`_SMOOTH_W`) — Psi0's ``0.1*sumsqr(q -
  q_last)``, which the original port dropped.
* **Limit handling by step scaling** instead of mid-iteration clipping, which
  destroyed the descent property and caused chatter on the bounds.

Convergence is tested on the *gradient*, not the residual: at the edge of the
workspace the residual stays large while the solution is perfectly optimal, and
the old residual+posture test could not fire at all. Typical cost is now ~3.5
iterations / 0.15 ms, i.e. the fixed solver is ~3.7x faster than the broken one
that always burned all 30.
"""

import os
from collections import deque

import numpy as np
import pinocchio as pin

_URDF = os.path.join(os.path.dirname(__file__), "g1_body29_hand14.urdf")

# joints locked away so only the 14 arm DOF remain (== Psi0 mixed_jointsToLockIDs)
_LOCK_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
    "right_hand_thumb_0_joint", "right_hand_thumb_1_joint", "right_hand_thumb_2_joint",
    "right_hand_index_0_joint", "right_hand_index_1_joint",
    "right_hand_middle_0_joint", "right_hand_middle_1_joint",
]


class WeightedMovingFilter:
    def __init__(self, weights, data_size=14):
        self._weights = np.array(weights, dtype=float)
        self._window = len(weights)
        self._buf = deque(maxlen=self._window)
        self._filtered = np.zeros(data_size)

    def add_data(self, x):
        x = np.asarray(x, dtype=float)
        self._buf.append(x)
        if len(self._buf) < self._window:
            self._filtered = x.copy()
        else:
            self._filtered = sum(
                w * d for w, d in zip(self._weights, reversed(self._buf))
            )

    def prime(self, x):
        """Fill the window with ``x`` so the next filtered output IS ``x``.

        Call this whenever the input stream restarts (arm teleop engage /
        disengage, episode reset). Two reasons it is ``prime`` and not a plain
        ``reset``:

        * A *stale* window is a real bug — the buffer survives disengage, so
          the first solve after re-engaging returns 0.4*new + 0.6*(the solution
          from before the operator let go), i.e. a lurch back toward the old
          pose on every re-engage.
        * An *empty* window is not the fix: ``add_data`` passes raw through
          while ``len(buf) < window``, so clearing would swap the lurch for a
          3-sample discontinuity in the filter's own dynamics right at engage.

        Priming with the measured pose makes the first tick continuous by
        construction.
        """
        x = np.asarray(x, dtype=float)
        self._buf.clear()
        for _ in range(self._window):
            self._buf.append(x.copy())
        self._filtered = x.copy()

    @property
    def filtered_data(self):
        return self._filtered


# Reduced-model joint order (14): L[sh pitch, sh roll, sh yaw, elbow,
# wr roll, wr pitch, wr yaw], then R[same].
#
# The "weird elbow" is the elbow *swivel* — the 1-DOF redundancy of a 7-DOF arm
# under a 6-DOF wrist task (the elbow rotating about the shoulder->wrist line).
# That swivel lives mainly in shoulder-roll/yaw, NOT in the elbow joint: the
# elbow *flexion* angle is fixed by how far the wrist reaches, so pinning it
# only adds tracking error. We therefore regularize shoulder roll+yaw, and
# leave the elbow flexion free.
_IK_SH_ROLL_IDX = (1, 8)
_IK_SH_YAW_IDX = (2, 9)

# Natural rest arm pose used as the null-space posture reference (== the AMO
# default arm config, in the reduced-model order). The posture term pulls the
# *redundant* DOF toward this so the elbow swivel settles naturally.
_Q_REF = np.array([0.5, 0.0, 0.2, 0.3, 0.0, 0.0, 0.0,
                   0.5, 0.0, -0.2, 0.3, 0.0, 0.0, 0.0], dtype=float)

# Per-joint posture (null-space) weight. Baseline is small (like Psi0's 0.02
# casadi regularization) so it only acts where the wrist task leaves slack; the
# swivel joints (shoulder roll+yaw) get a stronger pull so the elbow stops
# drifting. TUNE _POSTURE_W_SH_YAW up if the elbow still looks off, down if the
# arm feels stiff or the wrist mistracks.
_POSTURE_W_BASE = 0.02
_POSTURE_W_SH_ROLL = 0.1
_POSTURE_W_SH_YAW = 0.3

# Temporal-smoothness weight -- Psi0's ``0.1 * sumsqr(q - q_last)``
# (robot_body_ik.py:199,217), which this port originally dropped. Distinct from
# the posture term above and doing a different job: _POSTURE_W_* is a *pose*
# prior (pull the redundant DOF toward _Q_REF), _SMOOTH_W is a *temporal* prior
# (pull toward where the arm already is). Psi0 has both.
_SMOOTH_W = 0.1

# Inf-norm cap on a single Newton step, in rad. The single most load-bearing
# number in this file: measured, it alone takes the peak commanded joint
# velocity on a 0.30 m reach from 115 rad/s to 9.2 rad/s.
#
# 0.15 rad is ~2.5x the largest step legitimate tracking ever asks for (a hand
# moving at the 3 rad/s joint rate the caller rate-limits to covers 0.06 rad per
# 50 Hz tick), so it never binds during normal teleop -- it only bounds the
# nonsense. Do NOT drop below 0.10 (the engage transient takes visibly longer to
# converge) or raise above 0.25 (the chaotic regime returns).
_STEP_MAX = 0.15

# Keep-out margin from the joint limits, in rad. Steps are scaled to land inside
# [lower+margin, upper-margin]; the hard clip only ever runs as a post-loop
# safety net (see solve_ik).
_LIMIT_MARGIN = 0.01

# Max halvings in the backtracking line search before giving up on this
# iteration (2^-6 = 1/64 of the clamped step -- below that the step is not
# worth taking and the solver is at a local minimum).
_LS_MAX_HALVINGS = 6


class G1ArmIK:
    """Dual-arm IK. ``solve_ik(left_wrist, right_wrist, q, dq)`` -> (q14, tauff14)."""

    def __init__(
        self,
        urdf_path=_URDF,
        damping=0.1,
        max_iters=20,
        tol=1e-4,
        pos_weight=50.0,
        rot_weight=1.0,
        posture_weight=None,
        q_ref=None,
        smooth_weight=_SMOOTH_W,
    ):
        full = pin.buildModelFromUrdf(urdf_path)
        lock_ids = [full.getJointId(n) for n in _LOCK_JOINTS if full.existJointName(n)]
        self.model = pin.buildReducedModel(full, lock_ids, np.zeros(full.nq))
        for name, jname in (("L_ee", "left_wrist_yaw_joint"),
                            ("R_ee", "right_wrist_yaw_joint")):
            self.model.addFrame(
                pin.Frame(
                    name,
                    self.model.getJointId(jname),
                    pin.SE3(np.eye(3), np.array([0.05, 0.0, 0.0])),
                    pin.FrameType.OP_FRAME,
                )
            )
        self.data = self.model.createData()
        self.nq = self.model.nq  # 14
        self.L_id = self.model.getFrameId("L_ee")
        self.R_id = self.model.getFrameId("R_ee")
        self.lower = np.array(self.model.lowerPositionLimit)
        self.upper = np.array(self.model.upperPositionLimit)
        # Soft bounds the line search steers inside. Clamped to the joint's own
        # midpoint so a joint whose full range is under 2*margin can't invert.
        mid = 0.5 * (self.lower + self.upper)
        self._soft_lower = np.minimum(self.lower + _LIMIT_MARGIN, mid)
        self._soft_upper = np.maximum(self.upper - _LIMIT_MARGIN, mid)
        # Levenberg term. NOTE it is NOT what makes this iteration stable: at
        # damping=0.1 it contributes 0.01 against posture+smoothness weights
        # summing to >=0.12, and against a task weight of 50 the effective
        # damping ratio is 2e-4. The step clamp and the line search are what
        # keep the solve well behaved; this is kept only so the knob's meaning
        # doesn't silently change for anyone who was tuning it.
        self._damping = damping
        self._max_iters = max_iters
        self._tol = tol
        # per-row task weight (translation prioritized over rotation, like Psi0's
        # 50:1 casadi cost) for the L/R 6D errors: [transL(3), rotL(3), transR, rotR].
        self._W = np.array(
            [pos_weight] * 3 + [rot_weight] * 3 + [pos_weight] * 3 + [rot_weight] * 3
        )
        # Null-space posture task: bias the redundant arm DOF toward _Q_REF so
        # the elbow settles naturally (the wrist-only task leaves the elbow
        # swivel free). Per-joint weight, with the elbow + shoulder-yaw emphasized.
        self._q_ref = np.asarray(q_ref if q_ref is not None else _Q_REF, dtype=float)
        if posture_weight is None:
            pw = np.full(self.nq, _POSTURE_W_BASE)
            for i in _IK_SH_ROLL_IDX:
                pw[i] = _POSTURE_W_SH_ROLL
            for i in _IK_SH_YAW_IDX:
                pw[i] = _POSTURE_W_SH_YAW
        else:
            pw = np.broadcast_to(np.asarray(posture_weight, dtype=float), (self.nq,)).copy()
        self._posture_w = pw
        self._smooth_w = float(smooth_weight)
        self.init_data = np.zeros(self.nq)
        self.smooth_filter = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), self.nq)
        # Last solve's diagnostics, for the caller's debug log / the smoothness
        # test. Attributes rather than extra return values so solve_ik keeps its
        # (q, tauff) signature.
        self.last_iters = 0
        self.last_grad_inf = 0.0
        self.last_err_norm = 0.0

    def prime_filter(self, q14):
        """Re-seed the output filter at ``q14`` -- see WeightedMovingFilter.prime."""
        self.smooth_filter.prime(np.asarray(q14, dtype=float).reshape(self.nq))

    def _cost_and_err(self, q, tgtL, tgtR, q_anchor):
        """Objective value and the 12-D task error at ``q``.

        The objective is exactly the one the normal equations in ``solve_ik``
        Gauss-Newton on, which is what makes the line search valid:
            C(q) = 1/2*sum(W*err^2) + 1/2*sum(Wp*(q-q_ref)^2)
                   + 1/2*Ws*sum((q-q_anchor)^2)
        """
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        eL = pin.log6(self.data.oMf[self.L_id].actInv(tgtL)).vector
        eR = pin.log6(self.data.oMf[self.R_id].actInv(tgtR)).vector
        err = np.concatenate([eL, eR])
        dq_ref = q - self._q_ref
        dq_anchor = q - q_anchor
        cost = 0.5 * (
            float(self._W @ (err * err))
            + float(self._posture_w @ (dq_ref * dq_ref))
            + self._smooth_w * float(dq_anchor @ dq_anchor)
        )
        return cost, err

    def _max_feasible_scale(self, q, dq):
        """Largest t in [0, 1] with ``q + t*dq`` inside the soft joint bounds."""
        bounds = np.where(dq > 0, self._soft_upper, self._soft_lower)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(dq != 0.0, (bounds - q) / dq, np.inf)
        return float(np.clip(np.min(t), 0.0, 1.0))

    def solve_ik(
        self, left_wrist, right_wrist, current_q=None, current_dq=None, q_anchor=None
    ):
        """left_wrist / right_wrist: 4x4 SE3 target poses in the pelvis frame.

        ``current_q`` seeds the iteration (pass the measured arm state).
        ``q_anchor`` is what the temporal-smoothness term pulls toward and
        defaults to the seed, matching Psi0, which anchors its ``smooth_cost``
        to the measured joints (robot_body_ik.py:306-321). Anchoring on the
        measurement rather than on the last command is deliberate: it cannot
        wind up, so if the arm is blocked or saturated the next solve replans
        from where the arm actually is instead of snapping when it clears.

        Returns (q14, tauff14). Diagnostics land on ``last_iters`` /
        ``last_grad_inf`` / ``last_err_norm``.
        """
        q = np.array(
            current_q if current_q is not None else self.init_data, dtype=float
        )
        # The seed comes from the sim's measured qpos, which can sit a hair
        # outside the URDF limits; starting outside makes the first feasible
        # step scale zero.
        q = np.clip(q, self.lower, self.upper)
        anchor = q.copy() if q_anchor is None else np.asarray(q_anchor, dtype=float)
        tgtL = pin.SE3(np.asarray(left_wrist)[:3, :3], np.asarray(left_wrist)[:3, 3])
        tgtR = pin.SE3(np.asarray(right_wrist)[:3, :3], np.asarray(right_wrist)[:3, 3])
        lam2 = self._damping**2
        I14 = np.eye(self.nq)
        W = self._W
        Wp = self._posture_w
        Ws = self._smooth_w
        q_ref = self._q_ref

        cost, err = self._cost_and_err(q, tgtL, tgtR, anchor)
        iters = 0
        grad_inf = 0.0
        for _ in range(self._max_iters):
            JL = pin.computeFrameJacobian(
                self.model, self.data, q, self.L_id, pin.LOCAL
            )
            JR = pin.computeFrameJacobian(
                self.model, self.data, q, self.R_id, pin.LOCAL
            )
            J = np.vstack([JL, JR])  # 12 x 14
            # Gauss-Newton on the objective in _cost_and_err (b == -grad C):
            #   dq = (J^T W J + (lam^2 + Ws) I + diag(Wp))^-1
            #        (J^T W err + Wp (q_ref - q) + Ws (q_anchor - q))
            # The task term drives the wrists; the posture term acts only where
            # the task leaves slack (the elbow swivel null space); the
            # smoothness term keeps consecutive solves near each other.
            JtW = J.T * W  # 14x12
            A = JtW @ J + (lam2 + Ws) * I14 + np.diag(Wp)
            b = JtW @ err + Wp * (q_ref - q) + Ws * (anchor - q)

            # Converge on the GRADIENT, not the residual. At the edge of the
            # workspace the residual stays large while the solution is already
            # optimal -- the old test (residual AND posture error both tiny)
            # could never fire, so the loop always ran to max_iters and
            # returned whatever the last chaotic iterate happened to be.
            grad_inf = float(np.max(np.abs(b)))
            if grad_inf < self._tol:
                break

            dq = np.linalg.solve(A, b)
            step = float(np.max(np.abs(dq)))
            if step > _STEP_MAX:
                dq *= _STEP_MAX / step
            # Freeze components that would push a joint further past a bound it
            # is already on. Scaling handles the rest -- clipping mid-iteration
            # (what this used to do) breaks the descent property and chatters.
            dq = np.where((q <= self._soft_lower) & (dq < 0.0), 0.0, dq)
            dq = np.where((q >= self._soft_upper) & (dq > 0.0), 0.0, dq)
            if not np.any(dq):
                break

            t = self._max_feasible_scale(q, dq)
            accepted = False
            for _ls in range(_LS_MAX_HALVINGS):
                if t <= 0.0:
                    break
                q_try = pin.integrate(self.model, q, t * dq)
                cost_try, err_try = self._cost_and_err(q_try, tgtL, tgtR, anchor)
                if cost_try < cost:
                    q, cost, err = q_try, cost_try, err_try
                    accepted = True
                    break
                t *= 0.5
            if not accepted:
                break  # no descent direction left -- this is the local minimum
            iters += 1

        q = np.clip(q, self.lower, self.upper)  # safety net; the loop stays inside
        self.smooth_filter.add_data(q)
        q = self.smooth_filter.filtered_data
        self.init_data = q
        self.last_iters = iters
        self.last_grad_inf = grad_inf
        self.last_err_norm = float(np.linalg.norm(err))
        tauff = pin.rnea(self.model, self.data, q, np.zeros(self.nq), np.zeros(self.nq))
        return q, tauff
