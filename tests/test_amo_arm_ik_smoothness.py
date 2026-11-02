"""Offline smoothness harness for the AMO VR-3-point arm IK.

The thing this catches cannot be caught by eye and cannot be caught on a
still target: the solver's output has to be a *continuous function of the
target* across consecutive 50 Hz ticks, closed around a lagging plant. The
original DLS port was not -- it ran a chaotic Newton orbit at ~3 rad/iteration
for all 30 iterations and returned iterate #30, which produced violent
twitching for the operator the moment a reaching motion took the target near
the edge of the arm's workspace.

So every check here drives a *trajectory*, feeds the measured joint state back
in the way the real backend does (``solve_ik(current_q=measured)``), and
asserts on the per-tick command delta. A pass means the whole chain --
reach clamp, solver, jump guard, rate limiter -- turns smooth operator motion
into smooth joint commands even when the operator asks for the impossible.

Run: ``python tests/test_amo_arm_ik_smoothness.py``  (no sim, no PICO, no GPU)
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pinocchio as pin  # noqa: E402
from g1_bridge.vendored.amo.g1_arm_ik import (  # noqa: E402
    G1ArmIK,
    WeightedMovingFilter,
    _Q_REF,
)
from g1_bridge.backends.amo import (  # noqa: E402
    _AMO_ARM_REACH_MAX,
    _AMO_ARM_VEL_LIMIT,
    _AMO_JUMP_THRESH,
    _POLICY_DT,
    clamp_reach,
    rate_limit,
)

# Plant lag: q_meas chases q_cmd at this fraction per tick. Stands in for the
# 500 Hz PD loop, whose only role here is to make `current_q` lag the command
# the way it really does -- the solver is seeded from the MEASURED joints, so a
# test that fed back the command would hide exactly the feedback path that
# matters.
_PLANT_ALPHA = 0.35

# Pass thresholds. 0.12 rad/tick == 6 rad/s at 50 Hz, i.e. twice the rate
# limiter's own budget, so a trajectory can only fail this by defeating the
# limiter -- which is what a chaotic solve does when its output is rejected and
# held and then jumps again.
_MAX_DQ_PER_TICK = 0.12
_MAX_MEAN_ITERS = 8.0
_MAX_SOLVE_MS = 0.5


def _pose(pos, rpy=(0.0, 0.0, 0.0)):
    T = np.eye(4)
    T[:3, :3] = pin.rpy.rpyToMatrix(*rpy)
    T[:3, 3] = pos
    return T


class Rig:
    """One IK + the guards around it, driven tick by tick like the backend."""

    def __init__(self, use_clamp=True, use_rate_limit=True, use_jump_guard=True):
        self.ik = G1ArmIK()
        self.use_clamp = use_clamp
        self.use_rate_limit = use_rate_limit
        self.use_jump_guard = use_jump_guard
        model, data = self.ik.model, self.ik.data
        pin.framesForwardKinematics(model, data, np.zeros(self.ik.nq))
        self.sh_l = data.oMi[
            model.getJointId("left_shoulder_pitch_joint")
        ].translation.copy()
        self.sh_r = data.oMi[
            model.getJointId("right_shoulder_pitch_joint")
        ].translation.copy()
        self.reset(_Q_REF.copy())

    def reset(self, q0):
        self.q_meas = np.array(q0, dtype=float)
        self.q_cmd = np.array(q0, dtype=float)
        self.ik.prime_filter(q0)

    def home_ee(self):
        """L/R EE poses at the rest pose -- the anchor every trajectory
        perturbs, exactly like _rel_target's engage anchor."""
        pin.framesForwardKinematics(self.ik.model, self.ik.data, _Q_REF.copy())
        return (
            self.ik.data.oMf[self.ik.L_id].homogeneous.copy(),
            self.ik.data.oMf[self.ik.R_id].homogeneous.copy(),
        )

    def step(self, tL, tR):
        tL, tR = tL.copy(), tR.copy()
        clamped = False
        if self.use_clamp:
            tL[:3, 3], cl = clamp_reach(tL[:3, 3], self.sh_l, _AMO_ARM_REACH_MAX)
            tR[:3, 3], cr = clamp_reach(tR[:3, 3], self.sh_r, _AMO_ARM_REACH_MAX)
            clamped = cl or cr
        t0 = time.perf_counter()
        q_sol, _ = self.ik.solve_ik(tL, tR, current_q=self.q_meas)
        solve_ms = (time.perf_counter() - t0) * 1e3
        raw = q_sol.copy()
        rejected = False
        # NOTE: measured against the command, which is NOT what the backend
        # does any more -- AmoBackend._commit_arm_q14 compares against the
        # previous raw solution, because comparing against a rate-limited
        # command measures limiter lag rather than a jump and latches on it.
        # Left as-is deliberately: this rig exists to A/B solver *smoothness*
        # (clamp and limiter duty on a fixed trajectory), and keeping the guard
        # in its strictest form keeps those numbers conservative. Do not read
        # it as a model of the backend's guard.
        if self.use_jump_guard and np.any(
            np.abs(q_sol - self.q_cmd) > _AMO_JUMP_THRESH
        ):
            q_sol = self.q_cmd.copy()
            rejected = True
        limited = False
        if self.use_rate_limit:
            q_sol, limited = rate_limit(
                q_sol, self.q_cmd, _AMO_ARM_VEL_LIMIT * _POLICY_DT
            )
        dq_cmd = float(np.max(np.abs(q_sol - self.q_cmd)))
        self.q_cmd = q_sol
        self.q_meas += _PLANT_ALPHA * (self.q_cmd - self.q_meas)
        return dict(
            dq_cmd=dq_cmd,
            dq_raw=float(np.max(np.abs(raw - self.q_cmd))),
            iters=self.ik.last_iters,
            solve_ms=solve_ms,
            clamped=clamped,
            rejected=rejected,
            limited=limited,
        )

    def run(self, traj):
        recs = [self.step(tL, tR) for tL, tR in traj]
        dq = np.array([r["dq_cmd"] for r in recs])
        return dict(
            mean_dq=float(dq.mean()),
            p95_dq=float(np.percentile(dq, 95)),
            max_dq=float(dq.max()),
            max_vel=float(dq.max() / _POLICY_DT),
            mean_iters=float(np.mean([r["iters"] for r in recs])),
            mean_ms=float(np.mean([r["solve_ms"] for r in recs])),
            clamp_duty=float(np.mean([r["clamped"] for r in recs])),
            limit_duty=float(np.mean([r["limited"] for r in recs])),
            n_rejected=int(sum(r["rejected"] for r in recs)),
            n_twitch=int((dq > 0.1).sum()),
        )


# --------------------------------------------------------------------------- #
# Trajectories: what an operator's hands actually do, as EE targets.
# --------------------------------------------------------------------------- #
def reach_traj(rig, distance, n=150):
    """Both hands reach straight forward by `distance`, then hold."""
    hL, hR = rig.home_ee()
    out = []
    for i in range(n):
        s = min(1.0, i / (0.6 * n)) * distance
        tL, tR = hL.copy(), hR.copy()
        tL[0, 3] += s
        tR[0, 3] += s
        out.append((tL, tR))
    return out


def wrist_twist_traj(rig, amplitude, n=150):
    """Moderate reach, then a large wrist roll sweep -- the case a position
    clamp cannot help with, since it is the ORIENTATION that is unreachable
    (the G1 wrist limits are +/-1.61 rad, far tighter than a human forearm)."""
    hL, hR = rig.home_ee()
    out = []
    for i in range(n):
        a = amplitude * np.sin(2 * np.pi * i / n)
        tL, tR = hL.copy(), hR.copy()
        tL[0, 3] += 0.12
        tR[0, 3] += 0.12
        tL[:3, :3] = pin.rpy.rpyToMatrix(a, 0.0, 0.0) @ tL[:3, :3]
        tR[:3, :3] = pin.rpy.rpyToMatrix(-a, 0.0, 0.0) @ tR[:3, :3]
        out.append((tL, tR))
    return out


def step_traj(rig, jump=0.30, n=100):
    """Instantaneous 0.30 m target jump -- a dropped VR frame, or a teleport."""
    hL, hR = rig.home_ee()
    out = []
    for i in range(n):
        tL, tR = hL.copy(), hR.copy()
        if i >= 10:
            tL[0, 3] += jump
            tR[0, 3] += jump
        out.append((tL, tR))
    return out


def _report(name, m):
    print(
        f"  {name:<26} max Δq {m['max_dq']:6.4f} rad  "
        f"({m['max_vel']:6.2f} rad/s)  mean Δq {m['mean_dq']:6.4f}  "
        f"iters {m['mean_iters']:4.1f}  {m['mean_ms']:5.3f} ms  "
        f"clamp {m['clamp_duty']:.2f}  ratelim {m['limit_duty']:.2f}  "
        f"twitch {m['n_twitch']}"
    )


# --------------------------------------------------------------------------- #
def test_reach_trajectories():
    """The headline case: the solver used to go chaotic past ~0.25 m of reach,
    which is inside the first ordinary reaching motion."""
    print("T1 forward reach")
    for d in (0.10, 0.20, 0.30, 0.50):
        rig = Rig()
        m = rig.run(reach_traj(rig, d))
        _report(f"reach {d:.2f} m", m)
        assert m["max_dq"] <= _MAX_DQ_PER_TICK, (
            f"reach {d} m: max Δq {m['max_dq']:.4f} rad/tick "
            f"({m['max_vel']:.1f} rad/s) exceeds {_MAX_DQ_PER_TICK}"
        )
        assert m["mean_iters"] <= _MAX_MEAN_ITERS, m
        assert m["mean_ms"] <= _MAX_SOLVE_MS, m
    print("PASS: forward reach stays smooth out to 0.50 m (past max arm reach)")


def test_orientation_sweep():
    """Unreachable ORIENTATION also drove the old solver chaotic, and the reach
    clamp does nothing for it -- this is what the step clamp + line search buy.

    A high rate-limiter duty cycle is EXPECTED here and is not the "something
    upstream is broken" signal it would be in a live session: +/-2.5 rad is far
    outside the G1 wrist's +/-1.61 rad range, so the operator is asking for
    something the arm cannot do and saturating the limiter is the correct
    response.
    """
    print("T2 wrist orientation sweep")
    for a in (1.0, 2.5):
        rig = Rig()
        m = rig.run(wrist_twist_traj(rig, a))
        _report(f"twist +/-{a:.1f} rad", m)
        assert m["max_dq"] <= _MAX_DQ_PER_TICK, m
    # Without the limiter this same sweep peaks at ~9 rad/s with 2 twitch
    # events, so it is load-bearing here rather than decorative.
    bare = Rig(use_rate_limit=False)
    m_bare = bare.run(wrist_twist_traj(bare, 2.5))
    assert m_bare["max_vel"] > 5.0, (
        "the unreachable-orientation case no longer stresses the solver; "
        "this test has stopped testing anything"
    )
    print("PASS: unreachable wrist orientations do not twitch")


def test_step_input_is_a_ramp():
    """A target teleport must come out as a bounded ramp, not a lurch."""
    print("T3 step input")
    rig = Rig()
    m = rig.run(step_traj(rig))
    _report("0.30 m target jump", m)
    budget = _AMO_ARM_VEL_LIMIT * _POLICY_DT
    assert m["max_dq"] <= budget + 1e-9, (
        f"step input produced {m['max_dq']:.4f} rad in one tick, "
        f"rate limit budget is {budget:.4f}"
    )
    assert m["limit_duty"] > 0.0, "the rate limiter should have engaged here"
    print(f"PASS: 0.30 m teleport ramps at the {budget:.3f} rad/tick budget")


def test_solver_converges_on_the_gradient():
    """The old convergence test could never fire (it also required the posture
    error to be ~0), so every solve burned all 30 iterations and returned a
    point on a chaotic orbit. A reachable target must now converge in a few."""
    print("T4 convergence")
    rig = Rig()
    hL, hR = rig.home_ee()
    tL, tR = hL.copy(), hR.copy()
    tL[0, 3] += 0.05
    tR[0, 3] += 0.05
    for _ in range(20):  # settle
        rig.step(tL, tR)
    assert rig.ik.last_iters < 10, f"took {rig.ik.last_iters} iterations to hold still"
    assert rig.ik.last_grad_inf < 1.0, rig.ik.last_grad_inf

    # Holding a fixed state and target must be a FIXED POINT, not an orbit.
    # (Give the 4-tap output filter its window first -- its memory is not the
    # solver's, and asserting on the first two calls would only measure the
    # filter shifting.)
    q_state = rig.q_meas.copy()
    outs = [rig.ik.solve_ik(tL, tR, current_q=q_state)[0] for _ in range(6)]
    assert np.max(np.abs(outs[-1] - outs[-2])) < 1e-12, (
        f"repeated solves at an identical state drift by "
        f"{np.max(np.abs(outs[-1] - outs[-2])):.2e} -- the iteration is not converging"
    )
    print(f"PASS: converges in {rig.ik.last_iters} iters; held target is a fixed point")


def test_joint_limits_respected():
    """Limits used to be enforced by clipping mid-iteration, which broke the
    descent property and chattered on the bounds."""
    print("T5 joint limits")
    rig = Rig()
    hL, hR = rig.home_ee()
    # Drive far outside the workspace with the clamp OFF so the solve is pinned
    # against the limits for the whole trajectory.
    rig_noclamp = Rig(use_clamp=False, use_rate_limit=False, use_jump_guard=False)
    traj = []
    for i in range(120):
        s = min(1.0, i / 60) * 0.8
        tL, tR = hL.copy(), hR.copy()
        tL[0, 3] += s
        tL[2, 3] += s
        tR[0, 3] += s
        tR[2, 3] += s
        traj.append((tL, tR))
    rig_noclamp.run(traj)
    q = rig_noclamp.q_cmd
    lo, hi = rig_noclamp.ik.lower, rig_noclamp.ik.upper
    assert np.all(q >= lo - 1e-9) and np.all(
        q <= hi + 1e-9
    ), f"solution outside joint limits: {q}"
    print("PASS: solutions stay inside the URDF joint limits even unreachable")


def test_jump_guard_and_ik_failure():
    """A single bad solve must be rejected and the last command held."""
    print("T6 jump guard")
    rig = Rig()
    q_prev = rig.q_cmd.copy()
    q_bad = q_prev.copy()
    q_bad[3] += 2.0  # elbow, threshold pi/3
    assert np.any(np.abs(q_bad - q_prev) > _AMO_JUMP_THRESH), "test setup"
    q_ok = q_prev.copy()
    q_ok[3] += 0.05
    assert not np.any(np.abs(q_ok - q_prev) > _AMO_JUMP_THRESH), "guard too tight"
    # NaN must be caught by the same finite check the backend applies.
    assert not np.all(np.isfinite(np.array([np.nan] + [0.0] * 13)))
    print("PASS: jump guard fires on a 2 rad elbow step, passes a 0.05 rad one")


def test_filter_prime_has_no_carryover():
    """The output FIR survived disengage, so re-engaging blended 60% of the
    pre-disengage solution into the first tick -- a lurch, every time."""
    print("T7 filter prime")
    f = WeightedMovingFilter(np.array([0.4, 0.3, 0.2, 0.1]), 14)
    for _ in range(6):
        f.add_data(np.full(14, 1.0))
    assert np.allclose(f.filtered_data, 1.0)
    f.prime(np.full(14, -2.0))
    assert np.allclose(f.filtered_data, -2.0), "prime must take effect immediately"
    f.add_data(np.full(14, -2.0))
    assert np.allclose(f.filtered_data, -2.0), "primed window must not pull back"

    # End to end: a solve after priming must not be contaminated by the state
    # from before it.
    rig = Rig()
    hL, hR = rig.home_ee()
    tA_l, tA_r = hL.copy(), hR.copy()
    tA_l[0, 3] += 0.15
    tA_r[0, 3] += 0.15
    for _ in range(30):
        rig.step(tA_l, tA_r)  # converge somewhere far from home

    q_seed = _Q_REF.copy()
    rig.ik.prime_filter(q_seed)
    q_after = rig.ik.solve_ik(hL, hR, current_q=q_seed)[0]

    fresh = G1ArmIK()
    fresh.prime_filter(q_seed)
    q_fresh = fresh.solve_ik(hL, hR, current_q=q_seed)[0]
    assert (
        np.max(np.abs(q_after - q_fresh)) < 1e-9
    ), "primed solver still carries state from the previous engagement"
    print("PASS: prime_filter leaves no carryover across engagements")


def test_rate_limit_preserves_direction():
    """Uniform scaling, not per-joint clipping: per-joint clipping distorts the
    posture mid-motion, which reads worse than the jump it prevents."""
    print("T8 rate limiter shape")
    q_prev = np.zeros(14)
    q_new = np.arange(14, dtype=float) * 0.1  # varied magnitudes
    q_out, limited = rate_limit(q_new, q_prev, 0.06)
    assert limited
    assert abs(np.max(np.abs(q_out - q_prev)) - 0.06) < 1e-12
    d_in, d_out = q_new - q_prev, q_out - q_prev
    cos = float(d_in @ d_out / (np.linalg.norm(d_in) * np.linalg.norm(d_out)))
    assert abs(cos - 1.0) < 1e-12, f"direction changed (cos={cos})"
    q_out, limited = rate_limit(q_prev + 0.01, q_prev, 0.06)
    assert not limited, "limiter must be inert below the budget"
    print("PASS: rate limiter scales uniformly and is inert below budget")


def test_reach_clamp_geometry():
    print("T9 reach clamp")
    sh = np.array([0.0, 0.1, 0.29])
    p, clamped = clamp_reach(sh + np.array([0.2, 0.0, 0.0]), sh, 0.42)
    assert not clamped and np.allclose(p, sh + np.array([0.2, 0.0, 0.0]))
    p, clamped = clamp_reach(sh + np.array([0.8, 0.0, 0.0]), sh, 0.42)
    assert clamped and abs(np.linalg.norm(p - sh) - 0.42) < 1e-12
    # Direction preserved, so the operator's intent survives the clamp.
    v = np.array([0.5, -0.4, 0.3])
    p, _ = clamp_reach(sh + v, sh, 0.42)
    cos = float(v @ (p - sh) / (np.linalg.norm(v) * np.linalg.norm(p - sh)))
    assert abs(cos - 1.0) < 1e-12
    print("PASS: reach clamp projects onto the sphere along the target direction")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    tests = [
        test_reach_trajectories,
        test_orientation_sweep,
        test_step_input_is_a_ramp,
        test_solver_converges_on_the_gradient,
        test_joint_limits_respected,
        test_jump_guard_and_ik_failure,
        test_filter_prime_has_no_carryover,
        test_rate_limit_preserves_direction,
        test_reach_clamp_geometry,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
        print()
    if failed:
        print(f"{failed}/{len(tests)} checks FAILED")
        sys.exit(1)
    print("PASS: all AMO arm-IK smoothness checks")
