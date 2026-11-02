"""The heft/scalebfm OFF/IDLE/POSE machine and its re-entry pose blend.

No sim, no ONNX, no GPU, no PICO: the backend takes a Sim2RealContext
*Protocol*, so a fake with a handful of attributes is a valid context, and
RealtimeMotionBuffer can be built standalone with motion_zmq_connect=None.

Three groups:

  T1  ModeManager chord profiles -- including a golden sequence for the
      default "sonic" profile, which is what proves this change did not touch
      sonic's own state machine.
  T2  GenericTrackingBackend against a stubbed motion buffer: which ticks push
      at all, and what the blended frames contain.
  T3  The same push sequence through the REAL RealtimeMotionBuffer on a
      controlled clock, asserting on what the policy would actually observe.
      This is the one that would have caught the original bug.

Run: python tests/test_tracking_state_machine.py
"""

import sys
import types

import numpy as np

from g1_bridge.pico.mode_manager import ManagerMode, ModeManager
from sim2real.config.robots.g1 import G1_CFG

from g1_bridge.backends.base import BackendResources, TickInput
from g1_bridge.backends.generic import GenericTrackingBackend, RefPhase
from g1_bridge.reference.default import default_reference_frame
from g1_bridge.reference.blend import (
    QPOS_JOINTS,
    QPOS_ROOT_POS,
    QPOS_ROOT_QUAT,
    stand_in_place_qpos,
)


def _yaw_of(qpos):
    """Heading of a flat qpos, in radians."""
    w, x, y, z = np.asarray(qpos, dtype=np.float64)[QPOS_ROOT_QUAT]
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


DT_NS = 20_000_000  # 50 Hz, one policy tick
BLEND_S = 0.2  # 10 ticks -- short enough to keep the tests quick
BLEND_TICKS = int(round(BLEND_S / 0.02))
FAILURES = []


def check(cond, msg):
    if cond:
        print(f"  ok   {msg}")
    else:
        print(f"  FAIL {msg}")
        FAILURES.append(msg)


# --------------------------------------------------------------------------- #
# T1 -- ModeManager chord profiles
# --------------------------------------------------------------------------- #
def buttons(*pressed):
    return {k: (k in pressed) for k in ("a", "b", "x", "y", "left_axis_click")}


def drive(mgr, chords):
    """Press and release each chord in turn; return the mode after each press."""
    out = []
    for chord in chords:
        mgr.tick(buttons(*chord))
        out.append(mgr.current)
        mgr.tick(buttons())
    return out


def t1_mode_profiles():
    print("T1 ModeManager profiles")
    seq = [
        ("a", "b", "x", "y"),  # engage
        ("a", "x"),  # -> POSE
        ("b", "y"),  # -> FROZEN_UPPER (sonic only)
        ("b", "y"),  # -> POSE
        ("a", "x"),  # -> PLANNER
        ("left_axis_click",),  # -> VR_3PT (sonic only)
        ("left_axis_click",),  # -> back to parent
        ("a", "b", "x", "y"),  # stop
    ]
    golden_sonic = [
        ManagerMode.PLANNER,
        ManagerMode.POSE,
        ManagerMode.PLANNER_FROZEN_UPPER,
        ManagerMode.POSE,
        ManagerMode.PLANNER,
        ManagerMode.VR_3PT,
        ManagerMode.PLANNER,
        ManagerMode.OFF,
    ]
    got = drive(ModeManager(verbose=False), seq)
    check(got == golden_sonic, f"default profile unchanged: {[m.name for m in got]}")

    got = drive(ModeManager(verbose=False, profile="tracking"), seq)
    golden_tracking = [
        ManagerMode.PLANNER,  # engage
        ManagerMode.POSE,  # A+X
        ManagerMode.POSE,  # B+Y muted
        ManagerMode.POSE,  # B+Y muted
        ManagerMode.PLANNER,  # A+X back to idle
        ManagerMode.PLANNER,  # L-stick muted
        ManagerMode.PLANNER,  # L-stick muted
        ManagerMode.OFF,  # stop
    ]
    check(
        got == golden_tracking,
        f"tracking profile is OFF/PLANNER/POSE only: {[m.name for m in got]}",
    )
    check(
        set(got) <= {ManagerMode.OFF, ManagerMode.PLANNER, ManagerMode.POSE},
        "tracking profile never reaches FROZEN_UPPER or VR_3PT",
    )

    try:
        ModeManager(profile="nope")
        check(False, "unknown profile raises")
    except ValueError:
        check(True, "unknown profile raises")


# --------------------------------------------------------------------------- #
# Fakes shared by T2/T3
# --------------------------------------------------------------------------- #
class FakeBuffer:
    def __init__(self):
        self.pushed = []

    def push_payload(self, payload):
        self.pushed.append(payload)


class Clock:
    """The GMR publisher's wall clock, independent of a sim host's clock."""

    def __init__(self, ns=1_700_000_000_000_000_000):
        self.ns = ns

    def advance(self, ns=DT_NS):
        self.ns += ns


class FakeRelay:
    """A GMR publisher whose operator is walking and slowly raising both arms."""

    def __init__(self, fk, clock, moving=True):
        self._fk = fk
        self._clock = clock
        self._k = 0
        # A stationary operator still sits 2.5 m away and 40 deg round from the
        # robot, so the segment transform is just as non-trivial -- but any
        # heading change in the delivered reference is then unambiguously the
        # bridge's doing, not the operator's.
        self._moving = moving
        self.paused = False
        self.payload = None
        self.advance()

    def advance(self):
        k = self._k if self._moving else 0
        qpos = np.asarray(G1_CFG.default_qpos, dtype=np.float64).copy()
        qpos[0] = 3.0 + 0.01 * k  # walking in +x, far from the robot
        qpos[1] = -2.0
        yaw = 0.7 + 0.005 * k
        qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        qpos[QPOS_JOINTS] += 0.4 + 0.002 * k  # a pose well off the default
        # Body arrays are FK of the qpos -- exactly how the publisher builds them
        # (pico_retarget_pub._pose_arrays_from_qpos is their only producer).
        payload = self._fk.payload_from_qpos(qpos)
        payload["qpos"] = qpos.tolist()
        payload["pico_recv_time_ns"] = self._clock.ns
        payload["paused"] = self.paused
        self.payload = payload
        self._k += 1
        return payload

    def latest(self):
        return self.payload

    def default(self):
        return self.payload


class FakeCtx:
    def __init__(self, robot_cfg, buffer):
        self.tracking = types.SimpleNamespace(robot_cfg=robot_cfg, motion_buffer=buffer)
        self.io = None
        self.pico_source = None
        self.mode_manager = ModeManager(verbose=False, profile="tracking")
        self.sim = types.SimpleNamespace(data=types.SimpleNamespace(time=0.0))
        self.left_trigger = 0.0
        self.right_trigger = 0.0
        self.synthetic_pushes = 0
        self.settle_qpos = None
        self._standing_joints = np.asarray(G1_CFG.default_qpos, dtype=np.float64)[7:]
        self.pelvis_quat = np.array([1.0, 0.0, 0.0, 0.0])

    def standing_joints(self):
        return self._standing_joints

    def clock_s(self):
        # The host's clock: sim time here, wall time on the real-robot host. The
        # session answers it from clock_ms(); backends use it for rate limiting
        # instead of ctx.sim.data.time, which a hardware host does not have.
        return float(self.sim.data.time)

    def _host_clock_ns(self):
        return int(self.clock_s() * 1_000_000_000)

    def read_root_pos(self):
        return np.array([0.5, 0.25, 0.79])

    def read_pelvis_state(self):
        return self.pelvis_quat, np.zeros(3)

    def read_joint_state(self):
        return np.zeros(29), np.zeros(29)

    def read_body_pose_w(self, name):
        return np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0])

    def step_policy(self, qpos):
        return np.zeros(29, np.float32), np.ones(29), np.ones(29)

    def hold_pose(self):
        return np.zeros(29, np.float32), np.ones(29), np.ones(29)

    def push_synthetic_default_once(self, qpos):
        self.synthetic_pushes += 1


def make_session(buffer=None, blend_s=BLEND_S, clock=None, moving=True):
    fk = default_reference_frame(G1_CFG)
    buf = buffer if buffer is not None else FakeBuffer()
    ctx = FakeCtx(G1_CFG, buf)
    # TeleopSession.configure installs this exact clock in production. The GMR
    # relay below deliberately keeps its epoch wall timestamps, reproducing the
    # two time domains present in a real PICO + MuJoCo session.
    if hasattr(buf, "set_clock"):
        buf.set_clock(ctx._host_clock_ns)
    relay = FakeRelay(fk, clock if clock is not None else Clock(), moving=moving)
    backend = GenericTrackingBackend()
    backend.on_load(
        ctx,
        {"sim2real_pose_blend_s": blend_s},
        BackendResources(tracking=ctx.tracking, io=None, extra={"gmr_relay": relay}),
    )
    backend.reset(ctx)
    return backend, ctx, relay, buf


class Driver:
    """Runs ticks the way the composite does: mode edge first, then step."""

    def __init__(self, backend, ctx, relay, clock, use_on_mode_change=True):
        self.backend, self.ctx, self.relay, self.clock = backend, ctx, relay, clock
        self.mode = ManagerMode.OFF
        self.sim_t = 0.0
        self._hook = use_on_mode_change

    def set_mode(self, mode):
        if mode != self.mode and self._hook:
            self.backend.on_mode_change(self.ctx, self.mode, mode)
        self.mode = mode

    def tick(self, n=1):
        for _ in range(n):
            self.relay.advance()  # a payload stamped "now"...
            self.sim_t += 0.02
            self.ctx.sim.data.time = self.sim_t
            self.backend.step(
                self.ctx,
                TickInput(
                    qpos=np.zeros(29),
                    qvel=np.zeros(29),
                    sim_time=self.sim_t,
                    mode=self.mode,
                    buttons=None,
                    sample=None,
                ),
            )
            self.clock.advance()  # ...consumed before the clock moves on


# --------------------------------------------------------------------------- #
# T2 -- backend behaviour against a stubbed buffer
# --------------------------------------------------------------------------- #
def t2_backend(use_on_mode_change=True):
    label = "on_mode_change" if use_on_mode_change else "force_mode fallback"
    print(f"T2 backend state machine + blend ({label})")
    clock = Clock()
    backend, ctx, relay, buf = make_session(clock=clock)
    d = Driver(backend, ctx, relay, clock, use_on_mode_change)

    # seed_reference: anchors on the standing pose settling left behind, and
    # must not push (that frame is already in the buffer).
    backend.seed_reference(
        ctx,
        TickInput(np.zeros(29), np.zeros(29), 0.0, ManagerMode.OFF, None, None),
    )
    anchor = np.array(backend._anchor_qpos, copy=True)
    check(len(buf.pushed) == 0, "seed_reference pushes nothing")

    # --- OFF and IDLE are silent -------------------------------------------
    d.tick(5)
    check(len(buf.pushed) == 0, "OFF at episode start pushes nothing")
    d.set_mode(ManagerMode.PLANNER)
    d.tick(5)
    check(len(buf.pushed) == 0, "IDLE pushes nothing (the freeze)")

    # --- first POSE entry: ramp from the standing anchor --------------------
    d.set_mode(ManagerMode.POSE)
    d.tick(1)
    check(len(buf.pushed) == 1, "POSE pushes every tick")
    first = buf.pushed[0]
    live0 = np.asarray(relay.payload["qpos"], dtype=np.float64)
    q0 = np.asarray(first["qpos"], dtype=np.float64)
    check(first.get("motion_first_frame") is True, "entry frame stamps a new segment")
    check(
        first["pico_recv_time_ns"] == ctx._host_clock_ns()
        and first["pico_recv_time_ns"] != relay.payload["pico_recv_time_ns"],
        "buffer frame uses sim-host time, not the GMR publisher's wall time",
    )
    check(
        np.allclose(q0[QPOS_JOINTS], anchor[QPOS_JOINTS], atol=1e-6),
        "entry frame's joints are the held pose, not the operator's",
    )
    check(
        np.allclose(q0[QPOS_ROOT_POS][:2], live0[QPOS_ROOT_POS][:2], atol=1e-6),
        "entry frame's root xy is re-expressed onto the live stream",
    )
    check(
        abs(q0[2] - anchor[2]) < 1e-9,
        "entry frame keeps the anchor's height (yaw/xy alignment only)",
    )

    d.tick(BLEND_TICKS)
    check(backend._blender.active is False, f"blend completes in {BLEND_TICKS} ticks")
    last = buf.pushed[-1]
    check(
        np.allclose(last["joint_pos"], relay.payload["joint_pos"], atol=1e-5),
        "after the blend the live frame passes through untouched",
    )
    check(
        last.get("motion_first_frame") is None,
        "steady-state frames are not stamped as new segments",
    )

    # Continuity: no push may jump more than the ramp itself can move.
    joints = np.array([np.asarray(p["joint_pos"]) for p in buf.pushed])
    step_max = float(np.abs(np.diff(joints, axis=0)).max())
    bound = float(np.abs(live0[QPOS_JOINTS] - anchor[QPOS_JOINTS]).max()) / BLEND_TICKS
    check(
        step_max <= bound * 1.5 + 1e-3,
        f"per-tick joint step {step_max:.4f} stays within the ramp bound {bound:.4f}",
    )

    # --- exit mid-nothing, re-enter: anchor is where we froze ---------------
    frozen = np.asarray(buf.pushed[-1]["qpos"], dtype=np.float64)
    n_before = len(buf.pushed)
    d.set_mode(ManagerMode.PLANNER)
    d.tick(5)
    check(
        len(buf.pushed) == n_before, "IDLE after POSE pushes nothing (holds the pose)"
    )

    d.set_mode(ManagerMode.POSE)
    d.tick(1)
    re_entry = np.asarray(buf.pushed[-1]["qpos"], dtype=np.float64)
    check(
        np.allclose(re_entry[QPOS_JOINTS], frozen[QPOS_JOINTS], atol=1e-6),
        "re-entry blends from the pose it froze in",
    )

    # --- interrupt a blend: the partial pose becomes the new anchor ---------
    d.tick(2)  # part-way up the ramp
    partial = np.asarray(buf.pushed[-1]["qpos"], dtype=np.float64)
    d.set_mode(ManagerMode.PLANNER)
    d.tick(3)
    d.set_mode(ManagerMode.POSE)
    d.tick(1)
    resumed = np.asarray(buf.pushed[-1]["qpos"], dtype=np.float64)
    check(
        np.allclose(resumed[QPOS_JOINTS], partial[QPOS_JOINTS], atol=1e-6),
        "a blend interrupted mid-ramp resumes from where it stopped",
    )

    # --- stop chord: ramp back to standing, then fall silent ----------------
    # Compared over ALL 36 qpos entries, not just the joints. Slicing the root
    # off is what let the frame-mixing bug through: the joints were right in
    # both the correct and the broken version, and only the root pos/quat --
    # which is what the buffer's segment transform acts on -- differed.
    held = np.asarray(buf.pushed[-1]["qpos"], dtype=np.float64)
    d.set_mode(ManagerMode.OFF)
    d.tick(1)
    off_first = np.asarray(buf.pushed[-1]["qpos"], dtype=np.float64)
    check(
        np.allclose(off_first, held, atol=1e-6),
        "the stop ramp starts at exactly the frame that was being held",
    )
    d.tick(BLEND_TICKS)
    fk = default_reference_frame(G1_CFG)
    standing = stand_in_place_qpos(
        held, fk.standing_joints(ctx.standing_joints()), height=fk.default_root_height
    )
    landed = np.asarray(buf.pushed[-1]["qpos"], dtype=np.float64)
    check(
        np.allclose(landed, standing, atol=1e-6),
        "the stop ramp ends on the standing pose, in the held frame",
    )
    check(
        np.allclose(landed[QPOS_ROOT_POS][:2], held[QPOS_ROOT_POS][:2], atol=1e-9),
        "standing up does not move the reference in xy "
        "(a robot-world target here would teleport it)",
    )
    check(
        abs(_yaw_of(landed) - _yaw_of(held)) < 1e-6,
        "standing up keeps the held heading "
        "(this is the ratchet that walked the robot's facing)",
    )
    n_before = len(buf.pushed)
    d.tick(5)
    check(len(buf.pushed) == n_before, "OFF falls silent once it has stood up")

    # --- publisher pause vetoes tracking ------------------------------------
    d.set_mode(ManagerMode.POSE)
    d.tick(2)
    n_before = len(buf.pushed)
    relay.paused = True
    d.tick(3)
    check(
        len(buf.pushed) == n_before,
        "a paused publisher holds instead of tracking its standing frames",
    )
    relay.paused = False
    return backend, ctx, buf


def t2_recording_gate():
    print("T2 recording gate")
    clock = Clock()
    backend, ctx, relay, buf = make_session(clock=clock)
    d = Driver(backend, ctx, relay, clock)
    backend.seed_reference(
        ctx, TickInput(np.zeros(29), np.zeros(29), 0.0, ManagerMode.OFF, None, None)
    )
    d.set_mode(ManagerMode.PLANNER)
    d.tick(3)
    check(
        backend.teleop_sample(ctx).get("ref_joint_pos") is None,
        "IDLE reports not-ready, so collection skips those frames",
    )
    d.set_mode(ManagerMode.POSE)
    d.tick(3)
    check(
        backend.teleop_sample(ctx).get("ref_joint_pos") is not None,
        "POSE reports ready, including during the blend",
    )
    d.set_mode(ManagerMode.OFF)
    d.tick(2)
    check(
        backend.teleop_sample(ctx).get("ref_joint_pos") is None,
        "the stop ramp is not recorded either",
    )
    check(
        _phase_of(backend) is RefPhase.OFF,
        "phase tracking follows the mode",
    )


def _phase_of(backend):
    return backend._phase


# --------------------------------------------------------------------------- #
# T3 -- the same pushes through the real RealtimeMotionBuffer
# --------------------------------------------------------------------------- #
def t5_engage_ratchet():
    """BUG B: repeated A+B+X+Y walked the robot's standing heading around.

    Runs the real buffer so the segment transform is genuinely in force, with
    the operator far away and turned 40 deg from the robot -- the offset the
    transform absorbs, and the offset a robot-world stop target would re-apply
    on top of it. Four engage/disengage cycles; the delivered standing heading
    must not creep.
    """
    print("T5 A+B+X+Y engage/disengage ratchet (real buffer)")
    from sim2real.rl_policy.utils import motion_buffer as mb_mod

    from sim2real.rl_policy.utils.motion_buffer import RealtimeMotionBuffer

    clock = Clock()
    fake_time = types.SimpleNamespace(time_ns=lambda: clock.ns)
    real_time = mb_mod.time
    mb_mod.time = fake_time
    try:
        buf = RealtimeMotionBuffer(
            robot_cfg=G1_CFG, future_steps=[0], motion_zmq_connect=None, dt_s=0.02
        )
        backend, ctx, relay, _ = make_session(buffer=buf, clock=clock, moving=False)
        d = Driver(backend, ctx, relay, clock)
        fk = default_reference_frame(G1_CFG)
        pelvis = fk.body_names.index("pelvis")
        for _ in range(4):
            buf.push_payload(
                fk.payload(
                    ctx.read_root_pos(),
                    ctx.read_pelvis_state()[0],
                    ctx.standing_joints(),
                )
            )
            clock.advance()
        backend.seed_reference(
            ctx, TickInput(np.zeros(29), np.zeros(29), 0.0, ManagerMode.OFF, None, None)
        )

        headings = []
        for _ in range(4):
            d.set_mode(ManagerMode.PLANNER)
            d.tick(3)
            d.set_mode(ManagerMode.POSE)
            d.tick(BLEND_TICKS + 4)
            d.set_mode(ManagerMode.OFF)
            d.tick(BLEND_TICKS + 6)
            q = buf.get_obs().body_quat_w[0, 0, pelvis]
            headings.append(
                float(
                    np.arctan2(
                        2 * (q[0] * q[3] + q[1] * q[2]),
                        1 - 2 * (q[2] * q[2] + q[3] * q[3]),
                    )
                )
            )
        drift = np.abs(np.diff(np.unwrap(headings)))
        check(
            float(drift.max()) < np.deg2rad(2.0),
            f"standing heading does not ratchet across 4 engage cycles "
            f"(max step {np.rad2deg(float(drift.max())):.2f} deg)",
        )
    finally:
        mb_mod.time = real_time


def t6_second_episode():
    """BUG A: the second demo of a session spun up and fell over.

    One buffer, two episodes. Episode 1 runs a POSE session so a non-identity
    segment transform exists, then the composite's per-episode teardown runs
    (reset_stream) and episode 2 settles from a DIFFERENT spawn heading.

    The assertion is the mechanism, not the symptom: with a stale transform the
    delivered reference is `T . yaw(robot)`, so the heading error the policy
    sees stays pinned at T's yaw no matter where the robot turns -- the
    reference chases the robot instead of opposing it. So sweep the robot's yaw
    and require the error to TRACK it.
    """
    print("T6 second episode after a POSE session (real buffer)")
    from sim2real.rl_policy.utils import motion_buffer as mb_mod

    from sim2real.rl_policy.utils.motion_buffer import RealtimeMotionBuffer

    clock = Clock()
    fake_time = types.SimpleNamespace(time_ns=lambda: clock.ns)
    real_time = mb_mod.time
    mb_mod.time = fake_time
    try:
        buf = RealtimeMotionBuffer(
            robot_cfg=G1_CFG, future_steps=[0], motion_zmq_connect=None, dt_s=0.02
        )
        fk = default_reference_frame(G1_CFG)
        pelvis = fk.body_names.index("pelvis")

        # ---- episode 1: settle, engage, track, stop ----
        backend, ctx, relay, _ = make_session(buffer=buf, clock=clock)
        d = Driver(backend, ctx, relay, clock)
        for _ in range(4):
            buf.push_payload(
                fk.payload(
                    ctx.read_root_pos(),
                    ctx.read_pelvis_state()[0],
                    ctx.standing_joints(),
                )
            )
            clock.advance()
        backend.seed_reference(
            ctx, TickInput(np.zeros(29), np.zeros(29), 0.0, ManagerMode.OFF, None, None)
        )
        d.set_mode(ManagerMode.PLANNER)
        d.tick(3)
        d.set_mode(ManagerMode.POSE)
        d.tick(BLEND_TICKS + 10)
        seg = np.asarray(buf._segment_yaw_offset_quat, dtype=np.float64)
        seg_yaw = abs(
            float(
                np.arctan2(
                    2 * (seg[0] * seg[3] + seg[1] * seg[2]),
                    1 - 2 * (seg[2] * seg[2] + seg[3] * seg[3]),
                )
            )
        )
        check(
            seg_yaw > np.deg2rad(5.0),
            f"episode 1 leaves a real segment transform to be cleared "
            f"({np.rad2deg(seg_yaw):.1f} deg) -- otherwise this test proves nothing",
        )

        # ---- the composite's per-episode teardown ----
        buf.reset_stream()
        backend2, ctx2, relay2, _ = make_session(buffer=buf, clock=clock)

        # ---- episode 2: new spawn heading, settling holds one frozen frame ----
        spawn_yaw = -0.9
        ctx2.pelvis_quat = np.array(
            [np.cos(spawn_yaw / 2), 0.0, 0.0, np.sin(spawn_yaw / 2)]
        )
        settle_frame = fk.payload(
            ctx2.read_root_pos(), ctx2.read_pelvis_state()[0], ctx2.standing_joints()
        )
        errs = []
        for k in range(12):
            # sweep the robot's heading, exactly as a spinning robot would
            robot_yaw = spawn_yaw + 0.05 * k
            ctx2.pelvis_quat = np.array(
                [np.cos(robot_yaw / 2), 0.0, 0.0, np.sin(robot_yaw / 2)]
            )
            buf.push_payload(dict(settle_frame))
            clock.advance()
            q = buf.get_obs().body_quat_w[0, 0, pelvis]
            ref_yaw = float(
                np.arctan2(
                    2 * (q[0] * q[3] + q[1] * q[2]),
                    1 - 2 * (q[2] * q[2] + q[3] * q[3]),
                )
            )
            errs.append(ref_yaw - robot_yaw)

        errs = np.asarray(errs)
        swept = abs(errs[-1] - errs[0])
        check(
            swept > np.deg2rad(20.0),
            f"the heading error tracks the robot instead of staying pinned "
            f"(swept {np.rad2deg(swept):.1f} deg over a 31 deg turn) -- a "
            f"constant error here is the self-chasing reference that spun it up",
        )
        check(
            abs(errs[0]) < np.deg2rad(2.0),
            f"episode 2 opens with the reference on the robot's own heading "
            f"({np.rad2deg(abs(errs[0])):.1f} deg off)",
        )
    finally:
        mb_mod.time = real_time


def t3_real_buffer(blend_s=BLEND_S, negative_control=False):
    if negative_control:
        print("T3b negative control: same run with the blend disabled")
    else:
        print("T3 real RealtimeMotionBuffer, controlled clock")
    from sim2real.rl_policy.utils import motion_buffer as mb_mod

    from sim2real.rl_policy.utils.motion_buffer import RealtimeMotionBuffer

    clock = Clock()
    fake_time = types.SimpleNamespace(time_ns=lambda: clock.ns)
    # One patch covers both paths now: the buffer samples with its own time_ns,
    # and push_payload -- which stamps un-timestamped frames (the synthetic
    # standing default) -- lives in this same module. Leave either on the real
    # wall clock and those frames outlive everything, so the reference silently
    # falls back to them: a bug in the harness, not in the buffer.
    real_time = mb_mod.time
    mb_mod.time = fake_time
    try:
        buf = RealtimeMotionBuffer(
            robot_cfg=G1_CFG,
            future_steps=[0, 1, 2],
            motion_zmq_connect=None,
            dt_s=0.02,
        )
        backend, ctx, relay, _ = make_session(buffer=buf, clock=clock, blend_s=blend_s)
        d = Driver(backend, ctx, relay, clock)
        fk = default_reference_frame(G1_CFG)
        pelvis = fk.body_names.index("pelvis")
        world = fk.body_names.index("world")

        # Reproduce the composite's settling window: it pushes the standing
        # default frame every tick until the robot is quiet. Without it the
        # buffer would still be holding its own built-in world-origin default,
        # and the first stamped frame would legitimately re-anchor the segment
        # -- which is a different situation from the one under test.
        for _ in range(4):
            buf.push_payload(
                fk.payload(ctx.read_root_pos(), ctx.read_pelvis_state()[0])
            )
            clock.advance()
        backend.seed_reference(
            ctx, TickInput(np.zeros(29), np.zeros(29), 0.0, ManagerMode.OFF, None, None)
        )

        obs_joints, obs_root, obs_bodies, vel_max = [], [], [], []

        def run(n):
            for _ in range(n):
                d.tick(1)
                o = buf.get_obs()
                obs_joints.append(o.joint_pos[0, 0].copy())
                obs_root.append(o.body_pos_w[0, 0, pelvis].copy())
                # Skip "world": it sits at the origin in every raw frame, so
                # the buffer reports it as the image of the origin under the
                # segment transform. That is not a robot body and no
                # observation tracks it -- it moves whenever the transform is
                # re-anchored, by construction.
                obs_bodies.append(np.delete(o.body_pos_w[0, 0], world, axis=0))
                vel_max.append(
                    max(
                        float(np.abs(o.joint_vel).max()),
                        float(np.abs(o.body_lin_vel_w).max()),
                        float(np.abs(o.body_ang_vel_w).max()),
                    )
                )

        d.set_mode(ManagerMode.PLANNER)
        run(6)
        idle_end = len(obs_joints)
        d.set_mode(ManagerMode.POSE)
        run(BLEND_TICKS + 6)
        pose_end = len(obs_joints)
        d.set_mode(ManagerMode.PLANNER)
        run(10)

        obs_joints = np.array(obs_joints)
        obs_root = np.array(obs_root)
        obs_bodies = np.array(obs_bodies)
        steps = np.abs(np.diff(obs_joints, axis=0)).max(axis=1)
        # The reference the policy reads lags the push by delay_ns (4 ticks
        # here), so the handover shows up a few observations later than the
        # mode edge -- widen the window rather than pretending otherwise.
        edge = slice(idle_end - 2, idle_end + 6)

        if negative_control:
            # Same code path, blend_s=0 -- i.e. the behaviour this change
            # replaced. If this does NOT show a jump, the continuity checks
            # above are measuring nothing.
            check(
                float(steps[edge].max()) > 0.3,
                f"with the blend off the handover really does snap "
                f"({float(steps[edge].max()):.4f} rad in one tick)",
            )
            return

        check(
            float(steps[edge].max()) < 0.05,
            f"no reference jump at the IDLE->POSE handover "
            f"(max step {float(steps[edge].max()):.4f} rad)",
        )
        root_jump = float(np.abs(np.diff(obs_root[edge], axis=0)).max())
        check(
            root_jump < 0.02,
            f"root position stays continuous across the handover ({root_jump:.5f} m)",
        )
        # Every body, not just the pelvis: the entry frame's motion_first_frame
        # stamp does change the segment transform (it has to absorb the anchor's
        # re-expression onto the live root), and this is the assertion that the
        # transformed output nonetheless does not move.
        # Bound the handover by the ramp itself rather than by a hand-picked
        # constant: spreading the anchor-to-live difference over BLEND_TICKS is
        # exactly what the blend promises, so no single tick may move a body
        # more than that -- which is what "no teleport" means here.
        span = float(np.abs(obs_bodies[pose_end - 1] - obs_bodies[idle_end - 1]).max())
        body_bound = span / BLEND_TICKS * 1.5 + 1e-3
        body_jump = float(np.abs(np.diff(obs_bodies[edge], axis=0)).max())
        check(
            body_jump <= body_bound,
            f"no robot body teleports across the handover, despite the segment "
            f"transform changing under it ({body_jump:.5f} m, ramp bound "
            f"{body_bound:.5f} m over a {span:.3f} m pose change)",
        )
        tail = slice(pose_end + 6, None)
        check(
            float(np.abs(np.diff(obs_joints[tail], axis=0)).max()) == 0.0,
            "the IDLE hold is exactly motionless",
        )
        check(
            max(vel_max[pose_end + 6 :]) == 0.0,
            "the IDLE hold reports exactly zero reference velocity",
        )
        check(
            float(steps.max()) < 0.1,
            f"no single-tick reference jump anywhere (max {float(steps.max()):.4f} rad)",
        )
    finally:
        mb_mod.time = real_time


if __name__ == "__main__":
    t1_mode_profiles()
    t2_backend(use_on_mode_change=True)
    t2_backend(use_on_mode_change=False)
    t2_recording_gate()
    t3_real_buffer()
    t3_real_buffer(blend_s=0.0, negative_control=True)
    t5_engage_ratchet()
    t6_second_episode()
    print()
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS: all heft/scalebfm state-machine and blend checks")
