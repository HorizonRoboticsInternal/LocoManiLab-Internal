"""P3-2: is this episode physically plausible, and if not, which way did it fail.

An episode that reaches the goal by falling through a table, or by ending with a
knee 6 cm inside the sofa, is not training data -- it is a rendering of a physics
failure with a success label attached. This runs every control step and answers
two questions that are deliberately separate:

    accepted        did anything fire
    reasons         WHICH checks fired, and at which tick each first fired

The second one is the point. plan.md's rule for the first batch is that accepts
and rejects are both recorded with a reason, because the DISTRIBUTION of reasons
is what tells you which filter is the bottleneck -- without it you find out on
the third batch.

Every threshold here is measured, not chosen. ``h9_physics_calibrate`` runs the
checks over rollouts that are known to be fine and reports the p95 and the max of
each quantity; the defaults below are those numbers with headroom, and the script
prints what it would have set so the two cannot drift apart silently. A guessed
threshold is worse than none: it rejects real data at a rate nobody can quote.

One check is not a threshold at all. A non-finite reference does not raise in
MuJoCo -- it poisons the state with NaN, every later number is NaN, and the step
rate collapses about tenfold while the run continues. ``isfinite`` is checked
first and rejects immediately.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

#: Bodies whose contact with scene geometry is disqualifying on its own: the
#: TRUNK. A torso or a shoulder inside the furniture means the robot is through a
#: solid, and no task outcome redeems that.
#:
#: Hips and knees are deliberately NOT here, and that took a measurement to get
#: right. Including them rejected 4/4 of StackCanOnPlate on knee-and-hip contact
#: with the coffee table at penetrations of 8-10 mm -- i.e. the robot leaning its
#: legs against a low table while reaching over it, which is what reaching over a
#: low table looks like. Rejecting an entire scene for that is a filter deciding
#: the task is invalid. Legs are covered by the penetration check instead: touch
#: is allowed, 2 cm inside the table is not.
NEVER_CONTACT = (
    "torso_link", "waist_yaw_link", "waist_roll_link",
    "left_shoulder_pitch_link", "right_shoulder_pitch_link",
    "left_shoulder_roll_link", "right_shoulder_roll_link",
    "left_shoulder_yaw_link", "right_shoulder_yaw_link",
)

#: Reported but never disqualifying: informative when an episode is rejected for
#: something else, because "which limb was in the furniture" is usually the story.
WATCH_CONTACT = (
    "left_hip_pitch_link", "right_hip_pitch_link",
    "left_hip_roll_link", "right_hip_roll_link",
    "left_hip_yaw_link", "right_hip_yaw_link",
    "left_knee_link", "right_knee_link",
)


@dataclass
class PhysicsLimits:
    """Defaults are calibrated numbers with headroom. See h9_physics_calibrate."""

    #: Root drop below its starting height, metres. Against the MINIMUM over the
    #: episode, never the last frame -- a stumble that recovers is still a
    #: stumble, and judging on the final frame passed an episode that went
    #: 0.798 -> 0.341 -> back up.
    fall_drop: float = 0.30
    #: Tilt of the root's up-axis from vertical, degrees.
    tilt_deg: float = 45.0
    #: Absolute root height band above the floor datum, metres. Catches both
    #: free-fall through a finite floor (root z reached -67 m once) and a robot
    #: that has burrowed.
    root_z_range: tuple[float, float] = (0.35, 1.20)
    #: Penetration depth into scene geometry that counts as clipping, metres.
    #: MuJoCo's solver always allows a little; this is about a limb inside a
    #: solid, not about contact.
    penetration: float = 0.02
    #: How far a joint may exceed its own range before it counts, radians.
    joint_margin: float = 0.05
    #: Ticks a never-contact body may touch the scene before it counts. One tick
    #: is a brush; a sustained one is a limb inside the furniture.
    contact_ticks: int = 5
    #: Ticks the tilt may exceed `tilt_deg` before it counts. Reaching into a low
    #: drawer takes the root past 45 degrees for a moment; going over keeps it
    #: there. Measured over 66 episodes of two task families, on the seven that
    #: exceeded the bar at all:
    #:
    #:     leans, did not fall   0, 7, 15, 16, 22 ticks   tilt 45.2-48.0   drop 0.08-0.10
    #:     tipping falls            40, 51 ticks          tilt 79.4-105.1  drop 0.66-0.72
    #:
    #: Any bar in (22, 40) separates them; 30 is the middle. ⚠️ It rests on TWO
    #: positive examples, so it is a starting point, not a settled number -- but
    #: the two groups differ by a factor of two in tilt and eight in root drop, so
    #: they are not near cases. A fall that never tips (one in this sample, root
    #: drop 0.304 at 17 degrees) is caught by `fell` and does not need this.
    tipped_ticks: int = 30

    def as_dict(self) -> dict:
        return {
            "fall_drop": self.fall_drop, "tilt_deg": self.tilt_deg,
            "tipped_ticks": self.tipped_ticks,
            "root_z_range": list(self.root_z_range),
            "penetration": self.penetration, "joint_margin": self.joint_margin,
            "contact_ticks": self.contact_ticks,
        }


@dataclass
class PhysicsVerdict:
    accepted: bool
    reasons: dict = field(default_factory=dict)     # name -> first tick it fired
    stats: dict = field(default_factory=dict)       # name -> the worst value seen
    culprits: dict = field(default_factory=dict)    # "body <-> geom" -> ticks
    ticks: int = 0

    def summary(self) -> str:
        if self.accepted:
            return f"accept ({self.ticks} ticks)"
        worst = ", ".join(f"{k}@{v}" for k, v in sorted(self.reasons.items(),
                                                        key=lambda kv: kv[1]))
        return f"reject [{worst}]"


class PhysicsFilter:
    """Per-tick physical plausibility. Cheap enough to run at 50 Hz."""

    def __init__(self, tr, limits: PhysicsLimits = PhysicsLimits(),
                 record_stats: bool = True):
        import mujoco

        self.mj = mujoco
        self.tr = tr
        self.lim = limits
        self.record_stats = bool(record_stats)
        self._never = self._bodies(NEVER_CONTACT)
        self._watch = self._bodies(WATCH_CONTACT)
        # ⚠️ "Is this body part of the ROBOT" is a kinematic-tree question, not a
        # name-prefix question. Six of the robot's own bodies do not carry the
        # `robot0_` prefix -- `mobilebase0_support`, `manipulator_mount` and the
        # four gripper stubs -- so a prefix test reports them as SCENE, and a hip
        # resting against the robot's own mount was being logged as a forbidden
        # contact with the room. Measured: 12 of the culprit hits across the
        # articulation re-baseline were `hip_roll_link <-> robot0_g0_col`, which
        # is a geom on `manipulator_mount`.
        self._robot_root = int(self.tr.model.body_rootid[self.tr.bid_root])
        self._jnt_lo, self._jnt_hi = self._joint_ranges()
        self.reset()

    # ------------------------------------------------------------------ setup
    def _bodies(self, names) -> set[int]:
        out = set()
        for name in names:
            b = self.mj.mj_name2id(self.tr.model, self.mj.mjtObj.mjOBJ_BODY,
                                   f"{self.tr.prefix}{name}")
            if b >= 0:
                out.add(int(b))
        return out

    def _joint_ranges(self):
        m = self.tr.model
        lo, hi = [], []
        for a in self.tr._q_ds:
            # q address -> joint id, so limits come from the model rather than
            # from a transcribed table that can drift out of date.
            jid = int(np.where(m.jnt_qposadr == a)[0][0])
            if m.jnt_limited[jid]:
                lo.append(float(m.jnt_range[jid, 0]))
                hi.append(float(m.jnt_range[jid, 1]))
            else:
                lo.append(-np.inf)
                hi.append(np.inf)
        return np.asarray(lo), np.asarray(hi)

    # ------------------------------------------------------------------ cycle
    def reset(self) -> None:
        self.tick = 0
        self.reasons: dict[str, int] = {}
        self.stats: dict[str, float] = {}
        self._contact_run = 0
        self.culprits: dict[str, int] = {}
        self._start_z = float(self.tr.root_pos_local()[2])

    def _fire(self, name: str) -> None:
        self.reasons.setdefault(name, self.tick)

    def _note(self, name: str, value: float, worse=max) -> None:
        if not self.record_stats:
            return
        cur = self.stats.get(name)
        self.stats[name] = value if cur is None else worse(cur, value)

    # ------------------------------------------------------------------ step
    def step(self, i: int | None = None) -> None:
        self.tick = self.tick if i is None else int(i)
        d, m = self.tr.data, self.tr.model

        # 1. Non-finite state. Checked first and on its own, because everything
        # below it becomes meaningless once NaN is in the state, and MuJoCo will
        # not have told anyone.
        qp = d.qpos[self.tr._q_ds]
        qv = d.qvel[self.tr._v_ds]
        if not (np.isfinite(qp).all() and np.isfinite(qv).all()
                and np.isfinite(d.xpos[self.tr.bid_root]).all()):
            self._fire("nonfinite")
            return

        # 2. Root height, absolute and relative.
        z = float(self.tr.root_pos_local()[2])
        self._note("root_drop", self._start_z - z)
        self._note("root_z_min", z, min)
        self._note("root_z_max", z)
        if self._start_z - z > self.lim.fall_drop:
            self._fire("fell")
        if not (self.lim.root_z_range[0] <= z <= self.lim.root_z_range[1]):
            self._fire("root_height")

        # 3. Tilt. The root's own up-axis against world up; a robot at 60 degrees
        # is on its way down even if it has not lost height yet.
        R = d.xmat[self.tr.bid_root].reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))
        self._note("tilt_deg", tilt)
        if tilt > self.lim.tilt_deg:
            # ⚠️ Count HOW LONG, not just that it happened. A max on its own
            # cannot tell a robot leaning into a low drawer for three ticks from
            # one going over, and that distinction is the whole of I-9: on
            # `IThorKitchenCloseDrawer` 11 of 18 rejections were `tipped`, the
            # threshold was calibrated on single-chunk arena episodes that never
            # reach down, and deciding whether to move it needs the duration.
            # Answering it by replaying a recording needed the compiled model,
            # which is deleted by default -- the filter already had the number
            # and was discarding everything but the maximum.
            self._tipped_ticks = getattr(self, "_tipped_ticks", 0) + 1
            self._note("tipped_ticks", float(self._tipped_ticks))
            if self._tipped_ticks > self.lim.tipped_ticks:
                self._fire("tipped")

        # 4. Joint limits, read off the model rather than a transcribed table.
        over = np.maximum(self._jnt_lo - qp, qp - self._jnt_hi)
        worst = float(np.max(over)) if over.size else 0.0
        self._note("joint_over", worst)
        if worst > self.lim.joint_margin:
            self._fire("joint_limit")

        # 5. Penetration and forbidden contact, in one pass over the contacts.
        # The offending pair is recorded, not just the count: "forbidden_contact"
        # on its own cannot be told apart from a threshold that is too tight, and
        # deciding that needs to know whether it was a knee in a sofa or a
        # shoulder brushing a wall.
        deepest, forbidden = 0.0, None
        for c in range(d.ncon):
            con = d.contact[c]
            g1, g2 = con.geom1, con.geom2
            b1 = int(m.geom_bodyid[g1])
            b2 = int(m.geom_bodyid[g2])
            n1 = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_BODY, b1) or ""
            n2 = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_BODY, b2) or ""
            r1 = int(m.body_rootid[b1]) == self._robot_root
            r2 = int(m.body_rootid[b2]) == self._robot_root
            if r1 == r2:
                continue                       # self-contact or scene-scene
            # Ground and walls are filtered by GEOM name: robocasa hangs them off
            # the worldbody, so every one reports a body called "world" and a
            # body-name filter passes them all through.
            gn = self.mj.mj_id2name(m, self.mj.mjtObj.mjOBJ_GEOM,
                                    g2 if r1 else g1) or ""
            if gn.startswith("floor") or gn.startswith("wall"):
                continue
            deepest = min(deepest, float(con.dist))
            rb = b1 if r1 else b2
            if rb in self._never or rb in self._watch:
                key = f"{(n1 if r1 else n2).replace(self.tr.prefix, '')} <-> {gn}"
                self.culprits[key] = self.culprits.get(key, 0) + 1
                if rb in self._never:
                    forbidden = key
        self._note("penetration", -deepest)
        if -deepest > self.lim.penetration:
            self._fire("clipping")
        self._contact_run = self._contact_run + 1 if forbidden else 0
        self._note("forbidden_contact_ticks", float(self._contact_run))
        if self._contact_run >= self.lim.contact_ticks:
            self._fire("forbidden_contact")

        self.tick += 1

    # ------------------------------------------------------------------ out
    @property
    def accepted(self) -> bool:
        return not self.reasons

    def verdict(self) -> PhysicsVerdict:
        return PhysicsVerdict(accepted=self.accepted,
                              reasons=dict(self.reasons),
                              stats={k: round(float(v), 4)
                                     for k, v in self.stats.items()},
                              culprits=dict(sorted(self.culprits.items(),
                                                   key=lambda kv: -kv[1])[:5]),
                              ticks=self.tick)


def reason_histogram(verdicts) -> dict:
    """Rejection reasons across episodes, accepts included as ``accepted``.

    This is the diagnostic the first batch exists to produce: which filter is
    the bottleneck. Counting only rejects would hide the yield.
    """
    hist: dict[str, int] = {"accepted": 0, "rejected": 0}
    for v in verdicts:
        if v.accepted:
            hist["accepted"] += 1
            continue
        hist["rejected"] += 1
        for name in v.reasons:
            hist[name] = hist.get(name, 0) + 1
    return hist
