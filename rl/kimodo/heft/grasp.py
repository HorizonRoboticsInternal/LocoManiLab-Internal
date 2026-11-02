"""Scripted grasp that closes on ARRIVAL and confirms the object is actually held.

The version this replaces closed the hand at the start of the chunk whose phase
said "close". That is a schedule, not a grasp: the fingers curl while the hand is
still half a metre away, and by the time it arrives it is a fist. It also never
checked the outcome, so an episode where the hand swept past the object and
closed on air was indistinguishable from one that picked it up.

Neither does the task's own criterion distinguish them. ``RoomLiftMixin`` asks
``_hand_supports_object()``, and the iTHOR lift tasks implement that as "an
end-effector is within HAND_DISTANCE (0.24 m) of the object" -- a proximity test.
Combined with ``_object_is_lifted()`` that is a reasonable success condition for
a benchmark, but it is not evidence that the robot HELD anything, and a data
pipeline that exports on it will export episodes of a hand hovering next to an
object that some other contact happened to lift.

So there are three states here, and they are deliberately separate:

    reached   the palm got close enough to close on
    contact   >= MIN_FINGERS distinct finger links touch the object's geoms
    carried   the object stays put IN THE WRIST FRAME while the wrist travels

``carried`` is the one that means "held". Contact alone is satisfied by pressing
an object against the table it is sitting on, which is the single most likely way
for this pipeline to manufacture false positives -- the hand comes down on the
object, the fingers touch it, and nothing about that is a grasp. Requiring the
object to move WITH the hand costs one stored vector and rules it out.

Distances are measured from the PALM POINT, not the wrist. The tracker controls
the chain up to ``wrist_yaw_link`` and the hand sticks out ``PALM_OFFSET`` beyond
it, so a wrist-based trigger fires 13 cm early or late depending on the approach.
"""

from __future__ import annotations

import numpy as np

from rl.kimodo.heft.scene_frame import PALM_OFFSET

OPEN, CLOSING, CONTACT, CARRIED, MISSED = "open", "closing", "contact", "carried", "missed"


class ScriptedGrasp:
    """One hand, one object, driven once per 50 Hz control tick."""

    def __init__(self, tr, obj_name: str, side: str = "right", *,
                 approach: float | None = None, stable_ticks: int = 3,
                 ramp_ticks: int = 10, min_fingers: int = 2,
                 contact_ticks: int = 5, carry_travel: float = 0.05,
                 carry_slip: float = 0.06, max_rearm: int = 3,
                 verbose: bool = False):
        """
        ``approach``      palm-to-object distance that counts as arrived. None
                          (the default) SIZES IT TO THE OBJECT -- see below.
        ``max_rearm``     how many times a missed grasp may be retried.
        ``stable_ticks``  consecutive ticks inside ``approach`` before closing --
                          a single tick can be a fly-through at 50 Hz.
        ``ramp_ticks``    ticks to go from open to closed. The finger PD is a
                          position target; slamming it shut punches light objects
                          away rather than holding them.
        ``min_fingers``   distinct finger LINKS that must touch. One fingertip
                          brushing the object is contact, not a grasp.
        ``carry_travel``  how far the wrist must move before the carry test means
                          anything -- with a stationary hand every object trivially
                          "stays put in the wrist frame".
        ``carry_slip``    how far the object may drift in the wrist frame over
                          that travel and still count as carried.
        """
        import mujoco

        self.mj = mujoco
        self.tr = tr
        self.side = side
        self.obj_name = obj_name
        # ⚠️ SIZED TO THE OBJECT, not fixed. The old constant 0.12 m closed the
        # hand 0.076 m from an apple whose half-extent is 0.03 -- the fingertips
        # were 4.6 cm from anything, so the grasp could not have succeeded no
        # matter how good the reach was, and because a miss used to be terminal
        # that one early close ended the episode's chance. Measured on
        # `IThorKitchenAppleOnPlate` ep0: closed at 0.076, contact on 0 fingers.
        #
        # The rule is the object's own half-extent plus how far the fingers close
        # in from the palm point, clipped to the range h14 measured envelopes
        # over (0.02 for an apple, 0.12 for a box).
        self.approach = (float(approach) if approach is not None
                         else self._size_trigger(tr, obj_name))
        self.max_rearm = int(max_rearm)
        self.stable_ticks = int(stable_ticks)
        self.ramp_ticks = int(ramp_ticks)
        self.min_fingers = int(min_fingers)
        self.contact_ticks = int(contact_ticks)
        self.carry_travel = float(carry_travel)
        self.carry_slip = float(carry_slip)
        self.verbose = bool(verbose)

        self.bid_wrist = mujoco.mj_name2id(
            tr.model, mujoco.mjtObj.mjOBJ_BODY, f"{tr.prefix}{side}_wrist_yaw_link")
        if self.bid_wrist < 0:
            raise SystemExit(f"[grasp] no {tr.prefix}{side}_wrist_yaw_link")

        if obj_name not in getattr(tr.env, "objects", {}):
            raise SystemExit(
                f"[grasp] {type(tr.env).__name__} has no object {obj_name!r}; it has "
                f"{sorted(getattr(tr.env, 'objects', {}))}. An adapter naming an "
                "object the scene does not have would otherwise fall back to the "
                "phase schedule and look like a grasp that never confirms.")
        obj = tr.env.objects[obj_name]
        self.obj_geoms = {
            mujoco.mj_name2id(tr.model, mujoco.mjtObj.mjOBJ_GEOM, g)
            for g in obj.contact_geoms
        } - {-1}
        if not self.obj_geoms:
            raise SystemExit(f"[grasp] object {obj_name!r} has no resolvable geoms")
        self.obj_joint = obj.joints[0]
        # Finger links of THIS hand only. A left-hand contact is not evidence that
        # the right hand is holding anything.
        self.finger_bodies = {
            b for b in range(tr.model.nbody)
            if (mujoco.mj_id2name(tr.model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
            .startswith(f"{tr.prefix}{side}_hand_")
        }
        self.reset()

    @staticmethod
    def _size_trigger(tr, obj_name: str, *, finger: float = 0.05,
                      lo: float = 0.04, hi: float = 0.14) -> float:
        """Palm-to-object distance at which closing can actually reach it."""
        import mujoco

        m = tr.model
        best = 0.0
        for g in range(m.ngeom):
            gn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if gn.startswith(obj_name):
                best = max(best, float(np.max(m.geom_size[g])))
        if best <= 0.0:
            return 0.12                      # unknown object: the old constant
        return float(np.clip(best + finger, lo, hi))

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        self.state = OPEN
        self.trigger = 0.0
        self._near = 0
        self._contact = 0
        self._ramp = 0
        self._anchor_wrist = None      # wrist pos when contact was first confirmed
        self._anchor_local = None      # object pos in the wrist frame at that moment
        self.max_travel = 0.0
        self.max_slip = 0.0
        self.rearms = 0
        self._missed_at = None

    # ------------------------------------------------------------------ state
    def palm_pos(self) -> np.ndarray:
        """Where the hand actually closes, in world coordinates."""
        d = self.tr.data
        R = d.xmat[self.bid_wrist].reshape(3, 3)
        return d.xpos[self.bid_wrist] + R[:, 0] * PALM_OFFSET

    def object_pos(self) -> np.ndarray:
        return np.asarray(
            self.tr.env.sim.data.get_joint_qpos(self.obj_joint)[:3], dtype=np.float64)

    def distance(self) -> float:
        return float(np.linalg.norm(self.palm_pos() - self.object_pos()))

    def touching_fingers(self) -> int:
        """Distinct finger links of this hand in contact with the object."""
        d, m = self.tr.data, self.tr.model
        hit = set()
        for c in range(d.ncon):
            g1, g2 = d.contact[c].geom1, d.contact[c].geom2
            if g1 in self.obj_geoms:
                other = g2
            elif g2 in self.obj_geoms:
                other = g1
            else:
                continue
            b = int(m.geom_bodyid[other])
            if b in self.finger_bodies:
                hit.add(b)
        return len(hit)

    def _object_in_wrist_frame(self) -> np.ndarray:
        d = self.tr.data
        R = d.xmat[self.bid_wrist].reshape(3, 3)
        return R.T @ (self.object_pos() - d.xpos[self.bid_wrist])

    # ------------------------------------------------------------------ tick
    def step(self) -> dict:
        """Advance the state machine one control tick and drive the trigger."""
        dist = self.distance()

        if self.state == OPEN:
            self._near = self._near + 1 if dist <= self.approach else 0
            if self._near >= self.stable_ticks:
                self.state = CLOSING
                self._ramp = 0
                if self.verbose:
                    print(f"[grasp] closing at {dist:.3f} m")

        elif self.state == CLOSING:
            self._ramp += 1
            self.trigger = min(1.0, self._ramp / max(self.ramp_ticks, 1))
            if self._ramp >= self.ramp_ticks:
                n = self.touching_fingers()
                self._contact = self._contact + 1 if n >= self.min_fingers else 0
                if self._contact >= self.contact_ticks:
                    self.state = CONTACT
                    self._anchor_wrist = self.tr.data.xpos[self.bid_wrist].copy()
                    self._anchor_local = self._object_in_wrist_frame()
                    if self.verbose:
                        print(f"[grasp] contact on {n} fingers")
                elif self._ramp > self.ramp_ticks + 100:
                    # Two seconds shut with nothing between the fingers.
                    self.state = MISSED

        elif self.state in (CONTACT, CARRIED):
            travel = float(np.linalg.norm(
                self.tr.data.xpos[self.bid_wrist] - self._anchor_wrist))
            slip = float(np.linalg.norm(self._object_in_wrist_frame() - self._anchor_local))
            self.max_travel = max(self.max_travel, travel)
            self.max_slip = max(self.max_slip, slip)
            if travel >= self.carry_travel:
                # The object came along -> held. The object stayed behind while
                # the hand left -> it was being pressed, not carried.
                self.state = CARRIED if slip <= self.carry_slip else MISSED
                if self.verbose and self.state == MISSED:
                    print(f"[grasp] lost it: hand moved {travel:.3f} m, "
                          f"object slipped {slip:.3f} m in the wrist frame")
            if self.state != MISSED and self.touching_fingers() == 0:
                self.state = MISSED

        elif self.state == MISSED:
            # ⚠️ A MISS USED TO BE TERMINAL, and that threw away recoveries the
            # robot had already earned: on `IThorKitchenAppleOnPlate` ep0 the
            # hand closed on air, and a later chunk then brought the palm to
            # 0.076 m -- closer than the attempt that failed -- with the machine
            # permanently written off. Re-arm when the palm comes back inside the
            # trigger, bounded, because a hand oscillating around the threshold
            # would otherwise open and shut forever.
            if dist <= self.approach and self.rearms < self.max_rearm:
                self.rearms += 1
                self.state = OPEN
                self._near = self._contact = self._ramp = 0
                if self.verbose:
                    print(f"[grasp] re-arm {self.rearms}/{self.max_rearm} "
                          f"at {dist:.3f} m")

        # Drive the channel every tick. Holding requires continuous torque; the
        # fingers are position-controlled and would relax if the trigger lapsed.
        if self.state in (CONTACT, CARRIED):
            self.trigger = 1.0
        elif self.state == MISSED:
            self.trigger = 0.0
        self._write()
        return self.info()

    def release(self) -> None:
        self.state = OPEN
        self.trigger = 0.0
        self._near = self._contact = self._ramp = 0
        self._write()

    def _write(self) -> None:
        if self.side == "right":
            self.tr.set_grip(0.0, self.trigger)
        else:
            self.tr.set_grip(self.trigger, 0.0)

    def info(self) -> dict:
        return {"state": self.state, "trigger": round(self.trigger, 3),
                "dist": round(self.distance(), 4),
                "fingers": self.touching_fingers(),
                # Reported because the trigger is no longer a constant: a log
                # that says "closed at 0.09" means nothing without the distance
                # at which THIS object's fingers can reach.
                "approach": round(self.approach, 4),
                "rearms": self.rearms,
                "max_travel": round(self.max_travel, 4),
                "max_slip": round(self.max_slip, 4)}

    @property
    def holding(self) -> bool:
        """The only claim this module is willing to make about holding."""
        return self.state == CARRIED
