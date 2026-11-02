"""P3-3: is the motion natural, or merely successful.

A tracker driven at a goal can reach it by juddering. That still exports, still
passes the physics filter, and is exactly the sort of demonstration a VLA should
not be taught -- so "did it reach" and "did it move like something worth
imitating" have to be measured separately.

The measure is **dimensionless log jerk** (Balasubramanian et al.), which is the
standard smoothness metric in the human-movement literature and the reason to use
it here is that it is scale-free:

    DLJ = -ln( (T^3 / v_peak^2) * integral |d3x/dt3|^2 dt )

Higher is smoother. The normalisation by duration and peak speed is what makes a
1 s reach and a 3 s walk comparable -- a raw jerk integral is dominated by how
fast and how long the motion was, so thresholding it would reject long motions
and pass short jerky ones, which is backwards.

Two channels, because they fail differently:

    joint   the 29 commanded joint targets. Catches a policy whose ACTION is
            noisy even when the body filters it into something that looks fine.
    wrist   the tracked end-effector in world. Catches the opposite: a smooth
            action stream whose realised motion chatters against contact.

Thresholds are the measured population's p05 (low DLJ = jerky), not a constant.
``h11_naturalness`` reports the distribution and what it would set; the defaults
here are those numbers. The rule is the same one P3-3 was specified with: the
bar is the population's own quantile, because "smooth" has no absolute meaning.

⚠️ **This can only see jerk that survives execution.** Roughening the generated
REFERENCE by sigma=0.02 rad moved the realised action stream by 0.43 nats -- the
tracker low-passes it. Injecting the same noise into the action stream directly
moves DLJ by 0.83, and sigma=0.05 by 2.24, monotonically, which is what says the
metric works. So a generator that emits juddering references is largely invisible
here; that failure belongs upstream, in the generation-time KL against the base
model, which is the other half of P3-3.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: The tracker's control rate. Jerk is a third derivative, so getting this wrong
#: scales the metric by dt^-3 -- a factor of 1000 for a 10x error.
CONTROL_DT = 1.0 / 50.0


def dimensionless_log_jerk(x: np.ndarray, dt: float = CONTROL_DT) -> float:
    """[T] or [T, D] trajectory -> DLJ. Higher is smoother; NaN if too short.

    Multi-dimensional input is treated as one trajectory in D-space, i.e. the
    jerk magnitude is the norm across dimensions, not a per-dimension average.
    Averaging would let one juddering joint hide behind 28 quiet ones.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    T = x.shape[0]
    if T < 5:
        return float("nan")

    v = np.gradient(x, dt, axis=0)
    a = np.gradient(v, dt, axis=0)
    j = np.gradient(a, dt, axis=0)

    duration = (T - 1) * dt
    v_peak = float(np.linalg.norm(v, axis=1).max())
    if v_peak < 1e-9:
        # A motionless trajectory is perfectly smooth, and dividing by its peak
        # speed is how that becomes an infinity instead.
        return float("inf")
    integral = float(np.sum(np.linalg.norm(j, axis=1) ** 2) * dt)
    if integral <= 0:
        return float("inf")
    return float(-np.log((duration ** 3 / v_peak ** 2) * integral))


@dataclass
class NaturalnessLimits:
    """Lower bounds on DLJ. Below the bar is too jerky. See h11_naturalness."""

    #: On the 29 commanded joint targets (action.wbc). Measured population p05
    #: minus a nat: live runs sit at -15.3 median / -15.8 p05, and injecting
    #: sigma=0.05 rad of noise into the action stream drops it to -17.3.
    joint_dlj: float = -16.8
    #: On the tracked wrist in world. Live -12.4 median / -13.2 p05.
    wrist_dlj: float = -14.2
    #: Generation-time KL against the BASE model, nats. None disables the check.
    #: The floor is not zero and must not be treated as zero: the non-degenerate
    #: null is the KL between two base samples under the same conditioning, which
    #: h11 measures. A threshold below that floor rejects the base model itself.
    max_kl: float | None = None

    def as_dict(self) -> dict:
        return {"joint_dlj": self.joint_dlj, "wrist_dlj": self.wrist_dlj,
                "max_kl": self.max_kl}


@dataclass
class NaturalnessVerdict:
    accepted: bool
    reasons: dict
    stats: dict

    def summary(self) -> str:
        if self.accepted:
            return "accept"
        return "reject [" + ", ".join(sorted(self.reasons)) + "]"


def boundary_jump(chunks, dt: float = CONTROL_DT) -> float:
    """Largest velocity discontinuity across chunk joins, in units/s.

    Reported on its own rather than folded into DLJ, because it is a KNOWN and
    specific defect with a known cause: ``_realized_prefix`` tiles one realized
    frame K times, so the prefix velocity is identically zero and the generator
    is told "you are stationary here" at every boundary. Lumping it into a global
    smoothness score both hides which defect fired and swamps the score -- a
    two-chunk episode measured end-to-end reads 2 nats worse than either of its
    halves, so a bar set on single chunks rejects every multi-chunk episode.
    """
    worst = 0.0
    for a, b in zip(chunks, chunks[1:]):
        a = np.asarray(a, dtype=np.float64)
        b = np.asarray(b, dtype=np.float64)
        if len(a) < 2 or len(b) < 2:
            continue
        v_end = (a[-1] - a[-2]) / dt
        v_start = (b[1] - b[0]) / dt
        worst = max(worst, float(np.linalg.norm(v_end - v_start)))
    return worst


def _worst_dlj(chunks, dt: float) -> float:
    """DLJ of the roughest chunk. Per chunk, never on the concatenation."""
    vals = [dimensionless_log_jerk(c, dt) for c in chunks if len(c) >= 5]
    vals = [v for v in vals if np.isfinite(v)]
    return min(vals) if vals else float("nan")


def judge(targets_q29=None, wrist_world=None, kl: float | None = None,
          limits: NaturalnessLimits = NaturalnessLimits(),
          dt: float = CONTROL_DT) -> NaturalnessVerdict:
    """Offline: score a finished episode's action and wrist streams.

    Both arguments are a LIST OF CHUNKS, not one concatenated array. Smoothness
    is scored per chunk and the roughest one decides, while the joins are
    measured separately by ``boundary_jump``. A single array is accepted and
    treated as one chunk.

    Offline on purpose. Jerk is a third derivative, so it is only defined over a
    window; computing it per tick would either need a lag or a growing buffer,
    and the answer is not actionable mid-episode anyway.
    """
    stats, reasons = {}, {}

    def as_chunks(x):
        if x is None:
            return []
        return [np.asarray(x)] if isinstance(x, np.ndarray) and x.ndim == 2 else list(x)

    tc, wc = as_chunks(targets_q29), as_chunks(wrist_world)
    if tc:
        d = _worst_dlj(tc, dt)
        stats["joint_dlj"] = d
        stats["joint_boundary_jump"] = boundary_jump(tc, dt)
        if np.isfinite(d) and d < limits.joint_dlj:
            reasons["jerky_action"] = d
    if wc:
        d = _worst_dlj(wc, dt)
        stats["wrist_dlj"] = d
        stats["wrist_boundary_jump"] = boundary_jump(wc, dt)
        if np.isfinite(d) and d < limits.wrist_dlj:
            reasons["jerky_motion"] = d
    if kl is not None:
        stats["kl"] = float(kl)
        if limits.max_kl is not None and kl > limits.max_kl:
            reasons["far_from_base"] = float(kl)
    return NaturalnessVerdict(accepted=not reasons, reasons=reasons, stats=stats)
