"""The GMR motion-tracking backends: heft, scalebfm.

These differ only by checkpoint. Each runs GMR (the retargeting publisher) in a
subprocess inside sim2real's own pico venv, subscribes to its ZMQ payload
stream, and decides per tick what reference the policy tracks. The policy always
runs; only the reference changes.

Three states, and unlike sonic they are genuinely distinct here (see RefPhase):

    OFF  --A+B+X+Y-->  IDLE  --A+X-->  POSE  --A+X-->  IDLE  --A+X-->  POSE
     ^                                                                  |
     +---------------------- A+B+X+Y --------------------------- (any) -+

  POSE  tracks the live GMR stream -- the operator's body.
  IDLE  holds whatever pose POSE left off in, motionless, by simply not
        pushing: the buffer's own last-frame fallback repeats its last
        transformed frame with zero velocity for as long as we stay quiet.
  OFF   ramps the reference back to the standing default and then holds
        there, so the stop chord actually stands the robot up.

Entering POSE never snaps. The held pose is the anchor, the resuming stream is
aligned to it, and the reference blends from one to the other over
``sim2real_pose_blend_s`` -- see reference_blend.py for the mechanism and why it
runs in qpos space on the motion stream's own clock. This applies to the first
engage of an episode too, where the anchor is the standing pose the settling
window left behind.

These backends have no kPlanner and no VR-3pt controller, so PLANNER is reused
as IDLE and B+Y / L-stick are muted at the ModeManager (``mode_profile``).
"""

from __future__ import annotations

import time
from enum import IntEnum

import numpy as np
from termcolor import colored

from g1_bridge.pico.mode_manager import ManagerMode
from sim2real.config.robots.base import MOTION_FIRST_FRAME_KEY
from sim2real.rl_policy.families import get_family

# The SAME four RealtimeMotionBuffer uses for its segment transform (see its own
# import line). teleop_sample records the inverse of that transform, so sharing
# the implementation is what makes "inverse" a fact rather than an intention.
from sim2real.utils.math import (
    quat_canonical,
    quat_conjugate,
    quat_mul,
    quat_rotate_numpy,
    yaw_quat,
)

from g1_bridge.backends.base import (
    BackendResources,
    Sim2RealBackend,
    Sim2RealContext,
    TickInput,
    read_hand_state,
)
from g1_bridge.reference.default import default_reference_frame
from g1_bridge.reference.blend import (
    PoseBlender,
    qpos_from_payload,
    stand_in_place_qpos,
)

#: Seconds to ramp from the held pose onto the live stream (and, on the stop
#: chord, back to the standing default). Matches the reference implementation's
#: ``transition_steps: 100`` at 50 Hz; its own notes advise against going below
#: ~1 s on hardware. 0 disables blending -- the old snap-to-live behaviour.
_DEFAULT_BLEND_S = 2.0


class RefPhase(IntEnum):
    """What the reference is doing, independent of which ManagerMode said so.

    Deliberately total (see :func:`_phase_for`): the ``"tracking"`` chord
    profile should make PLANNER_FROZEN_UPPER and VR_3PT unreachable, but a
    ``force_mode`` call or a future profile change must not be able to produce
    an undefined reference.
    """

    OFF = 0
    IDLE = 1
    POSE = 2


def _phase_for(mode) -> RefPhase:
    if mode == ManagerMode.OFF:
        return RefPhase.OFF
    if mode == ManagerMode.POSE:
        return RefPhase.POSE
    return RefPhase.IDLE


def _root_body_idx(payload: dict) -> int:
    """Index of the pelvis in the payload's body arrays -- the buffer's own
    ``root_body_name`` default. Only needed for the qpos fallback."""
    names = payload.get("body_names")
    if names and "pelvis" in names:
        return int(names.index("pelvis"))
    return 0


class GenericTrackingBackend(Sim2RealBackend):
    """Reference-tracking policy fed by a GMR retargeting subprocess."""

    name = "heft"
    controller_json = "g1_generic.json"
    uses_gmr = True
    mode_profile = "tracking"
    #: The 50 Hz reference-motion contract -- what teleop_sample records here
    #: (ref_root_pos/quat + ref_joint_pos) and what every tracking family in
    #: sim2real consumes. A policy trained on this drives heft, scalebfm,
    #: mimic-lite, holomotion, twist2 and bfm-zero without changes.
    vla_action_space = "ref_qpos"

    #: Default policy.yaml per backend name -- these share all logic and differ
    #: only in checkpoint, so one class serves both (see backends/__init__).
    #: The paths live in sim2real's family table (the single source; a
    #: consistency test in tests/test_backend_registry_consistency.py holds
    #: the two packages together). Alternates: heft ships a sibling
    #: policy-semantic.onnx and scalebfm a humanoid_transformer_xl -- select
    #: either with an explicit --policy-config.
    POLICY_CONFIGS = {
        name: get_family(name).policy_config for name in ("heft", "scalebfm")
    }

    def make_vla_adapter(self, ctx):
        """Hand the adapter the SAME forward kinematics the live GMR stream is
        FK'd through (``pico_retarget_pub._pose_arrays_from_qpos``).

        The motion buffer consumes body poses, not joint angles, so a policy's
        predicted qpos has to be FK'd somewhere. Doing it with a different
        model than the recording used would put the reference in a subtly
        different frame from the one the policy was trained against -- the
        kind of mismatch that reads as a slightly-wrong pose, not an error.
        """
        from sim2real.vla.adapters import get_adapter

        fk = getattr(self, "_fk", None) or default_reference_frame(ctx.tracking.robot_cfg)
        return get_adapter("ref_qpos", frame_builder=fk.payload_from_qpos)

    def __init__(self):
        self._relay = None
        self._last_ref_qpos = None
        #: ``(anchor_xy(1,3), inverse_anchor_yaw_quat(1,4))`` defining this
        #: episode's reference frame, or None before its first recorded frame.
        #: Cleared by on_episode_start (every episode, on both hosts) AND by
        #: reset (env.reset in sim); see on_episode_start for why one of those
        #: is not enough.
        self._ref_anchor = None
        self._debug_log_t = -1e9
        # FK helper (resolved in on_load, cached process-wide by robot).
        self._fk = None
        self._blend_s = _DEFAULT_BLEND_S
        self._record_idle = False
        # The last pushed qpos with its root re-expressed through the motion
        # buffer's segment transform -- i.e. the reference as the policy
        # tracks it (schema v8). What _last_ref_qpos is set from.
        self._last_pushed_qpos_tracked = None
        self._blender = PoseBlender(_DEFAULT_BLEND_S)
        # The pose currently being held == the last frame we pushed. Doubles as
        # the blend anchor and as what teleop_sample reports, so the two can
        # never disagree about what the policy is actually tracking.
        self._anchor_qpos = None
        self._off_target_qpos = None
        # Phase this backend last acted on, and the two "a blend is owed"
        # flags set on the mode edge (see on_mode_change).
        self._phase = None
        self._blend_pending = False
        self._off_blend_pending = False
        self._pushed_any = False
        # Operator-facing diagnostics (see _warn / _check_stream_health).
        self._warned = {}
        self._last_stream_t_ns = None
        self._stream_stall_ticks = 0
        self._starved_ticks = 0

    # ------------------------------------------------------------------ #
    # Construction (cache-key-guarded: about once per process)
    # ------------------------------------------------------------------ #
    def build_tracking(self, cfg, policy_config_path, inference_backend):
        from g1_bridge.bridge_runtime import make_bridge_tracking
        from g1_bridge.gmr_utils.publisher_proc import start_gmr_if_available
        from g1_bridge.gmr_utils.relay import GMRRelay

        gmr_proc = start_gmr_if_available(cfg)  # None on a host without the pico venv (eval)
        connect = cfg.get("gmr_motion_zmq_connect") or "tcp://127.0.0.1:28701"
        gmr_relay = GMRRelay(connect=connect)
        # push, not zmq: we own the subscription (gmr_relay) so we can choose
        # default-vs-live per tick -- see step().
        tracking, io = make_bridge_tracking(
            policy_config_path,
            motion_mode="push",
            inference_backend=inference_backend,
            motion_backend="zmq",
        )
        print(
            f"[g1_bridge] G1_BRIDGE armed, backend={self.name} "
            f"(policy_config={policy_config_path}, GMR subprocess {'live' if gmr_proc else 'not started'}). "
            "A+B+X+Y engages (IDLE, holds pose); A+X starts/stops tracking "
            "your body; A+B+X+Y again stands back up. Entering tracking "
            "blends from the held pose over sim2real_pose_blend_s."
        )
        return BackendResources(
            tracking=tracking,
            io=io,
            extra={"gmr_proc": gmr_proc, "gmr_relay": gmr_relay},
        )

    def make_pico_source(self, cfg):
        """Buttons/trigger over GMR's own ZMQ publisher when hardware is live.

        GMR already owns an xrobotoolkit_sdk connection for full-body tracking.
        Opening a SECOND real connection here (just for buttons/trigger) starves
        whichever connection isn't "primary" of live body data -- verified live:
        the non-primary connection's body pose freezes and its body timestamp
        sticks at 0, which is exactly the "hand works, body doesn't follow" (and
        its mirror image) bug reported in practice. See gmr_button_relay.py for
        the full trade-off (VR_3PT is unreachable through it).
        """
        if cfg.get("pico_source", "mock") != "real":
            return None
        from g1_bridge.gmr_utils.button_relay import GMRButtonRelay

        # pico_source_kwargs (service_script/init_timeout_s/verbose/...) are
        # written for the SDK-connection sources; GMRButtonRelay takes no such
        # options, it only has ZMQ endpoints to connect to.
        source = GMRButtonRelay()
        source.start()
        return source

    # ------------------------------------------------------------------ #
    # Per env.reset()
    # ------------------------------------------------------------------ #
    def on_load(self, ctx: Sim2RealContext, cfg: dict, res: BackendResources) -> None:
        self._relay = res.extra["gmr_relay"]
        self._blend_s = float(cfg.get("sim2real_pose_blend_s", _DEFAULT_BLEND_S))
        self._record_idle = bool(cfg.get("sim2real_record_idle", False))
        self._blender = PoseBlender(self._blend_s)
        # Same MJCF the publisher retargets into, so blended frames carry the
        # identical joint_names/body_names as the live stream -- a layout change
        # would make RealtimeMotionBuffer drop its last transformed frame and
        # reset the segment transform, exactly at the hand-off that needs them.
        self._fk = default_reference_frame(ctx.tracking.robot_cfg)

    def reset(self, ctx: Sim2RealContext) -> None:
        self._last_ref_qpos = None
        self._ref_anchor = None
        self._debug_log_t = -1e9
        self._anchor_qpos = None
        self._off_target_qpos = None
        self._phase = None
        self._blend_pending = False
        self._off_blend_pending = False
        self._pushed_any = False
        # Operator-facing diagnostics (see _warn / _check_stream_health).
        self._warned = {}
        self._last_stream_t_ns = None
        self._stream_stall_ticks = 0
        self._starved_ticks = 0
        self._blender.cancel()

    # ------------------------------------------------------------------ #
    # Per tick
    # ------------------------------------------------------------------ #
    def on_mode_change(self, ctx: Sim2RealContext, prev, new) -> None:
        """Arm (or disarm) a blend on the ModeManager edge.

        Runs at substep rate, so it only sets flags -- the work happens on the
        next throttled ``step``. Mirrors sonic's ``_pose_align_pending``.
        """
        self._arm_for_phase(_phase_for(prev), _phase_for(new))

    def _arm_for_phase(self, prev_phase, new_phase) -> None:
        if new_phase == prev_phase:
            return
        self._blend_pending = new_phase == RefPhase.POSE
        if new_phase != RefPhase.POSE:
            # Whatever ramp was running, stop where it got to: that partial
            # pose is now the held pose, and re-entering POSE must blend from
            # there rather than from where the previous ramp started.
            self._blender.cancel()
        # Only ramp back to standing when we are actually leaving an engaged
        # state -- not on the episode's opening None -> OFF, where the settling
        # window has already put the standing frame in the buffer.
        self._off_blend_pending = new_phase == RefPhase.OFF and prev_phase in (
            RefPhase.IDLE,
            RefPhase.POSE,
        )

    def seed_reference(self, ctx: Sim2RealContext, tick: TickInput) -> None:
        """Anchor the episode on the pose settling left the robot in.

        No push of its own: the settling window's last standing frame is
        already in the buffer, already anchored to where the robot came to
        rest, so ``_end_settling``'s contract is met. What this adds is the
        anchor for the episode's *first* engage, which therefore ramps from
        standing into the operator's pose exactly like every later re-entry
        instead of snapping.

        Takes the settling window's own captured frame rather than rebuilding
        one, so the anchor is exactly the pose already sitting in the buffer.
        """
        self._anchor_qpos = getattr(ctx, "settle_qpos", None)
        if self._anchor_qpos is None:
            self._anchor_qpos = self._fk.default_qpos_at(
                ctx.read_root_pos(),
                ctx.read_pelvis_state()[0],
                ctx.standing_joints(),
            )
        self._push_reference(ctx, tick)

    def step(self, ctx: Sim2RealContext, tick: TickInput):
        """Update the reference for this tick's phase, then run the policy."""
        self._push_reference(ctx, tick)
        return ctx.step_policy(tick.qpos)

    def _push_reference(self, ctx: Sim2RealContext, tick: TickInput) -> None:
        """Feed the motion buffer this tick's reference, or deliberately don't.

        Falls back to the composite's synthetic default payload -- still driven
        through the real trained policy -- when no GMR payload has arrived at
        all, e.g. the subprocess can't reach a real PICO service in a
        mock/no-hardware environment. That case used to fall through to a raw,
        un-anchored PD hold and reliably made the robot collapse.

        Not pushing is a real outcome here, not a failure path: it is how IDLE
        holds a pose. ``RealtimeMotionBuffer`` ages its frames out after
        ``history_ns`` and then repeats its last *transformed* frame with all
        velocities exactly zero (``_fill_from_last_frame_locked``), and clamps
        to the same frame while it drains -- so silence is a ~0.16 s roll-out
        into a clean, motionless hold, and re-pushing a frozen frame at 50 Hz
        would only add churn and a second timestamp convention.

        ``_last_ref_qpos`` (what teleop_sample reports, and the readiness gate
        collection records against) is the frame actually pushed, and is left
        None whenever we are not tracking -- so IDLE and the stop ramp do not
        land in a dataset as if the operator had commanded them.
        """
        payload = self._relay.latest()
        if payload is None:
            self._last_ref_qpos = None
            ctx.push_synthetic_default_once(tick.qpos)
            return

        phase = self._resolve_phase(tick.mode, payload)
        if phase != self._phase:
            # Belt and braces for on_mode_change: force_mode() (used by
            # --force-engage and the E/T keys) does not go through it, and two
            # edges can land inside one 20 ms policy tick.
            self._arm_for_phase(self._phase, phase)
            self._phase = phase

        stream_t_ns = int(
            payload.get("pico_recv_time_ns")
            or payload.get("publish_t_ns")
            or ctx._host_clock_ns()
        )
        # The publisher timestamp belongs to the GMR process's wall clock. It
        # is useful for detecting a stalled source, but it is NOT necessarily
        # the motion buffer's clock: a sim host installs MuJoCo sim time on the
        # buffer so sampling is deterministic. Stamp every pushed frame on the
        # host clock instead; on hardware the host clock is wall time, so this
        # remains the same time base as the publisher there.
        host_t_ns = ctx._host_clock_ns()
        if phase == RefPhase.POSE:
            self._check_stream_health(ctx, stream_t_ns)
            alpha = self._push_tracking(ctx, payload, host_t_ns)
        else:
            alpha = self._push_stopping(ctx, host_t_ns)
        self._debug_log_payload(ctx, payload, phase, alpha)

    def _resolve_phase(self, mode, payload: dict) -> RefPhase:
        """Phase for this tick, with the publisher's pause as a veto.

        ``gmr_publisher_launcher`` disables the publisher's own A+X pause
        toggle precisely so A+X stays ours, so ``paused`` should never be True
        here. But ``GMRSubprocess`` reuses an already-bound publisher on port
        28701 if it finds one, and a publisher started by hand
        (``-m sim2real.teleop.pico_retarget_pub``) still has the toggle live.
        Its paused frames are a standing default, not the operator -- tracking
        those would silently look like "the robot ignores me", so hold instead.
        """
        phase = _phase_for(mode)
        if phase == RefPhase.POSE and bool(payload.get("paused", False)):
            self._warn(
                "paused",
                "the GMR publisher says it is PAUSED, so tracking is on hold "
                "even though A+X engaged POSE. A+X is bound to the publisher's "
                "own pause toggle too, and gmr_publisher_launcher only disables "
                "that for a publisher IT started -- GMRSubprocess reuses one "
                "already bound to port 28701. Kill the stray publisher and let "
                "this session start its own, or press A+X again to re-sync the "
                "two.",
            )
            return RefPhase.IDLE
        return phase

    def _warn(self, key: str, message: str, period: float = 5.0) -> None:
        """Rate-limited operator-facing warning, on wall time.

        These are the paths where the robot silently stops following: without a
        printed reason the operator sees only "it ignores me", which is
        indistinguishable from a policy or hardware fault and sends the next
        hour of debugging in the wrong direction.
        """
        now = time.monotonic()
        if now - self._warned.get(key, -1e9) < period:
            return
        self._warned[key] = now
        print(colored(f"[g1_bridge] {message}", "yellow"), flush=True)

    def _check_stream_health(self, ctx, t_ns: int) -> None:
        """Warn when the reference the policy tracks has stopped advancing.

        Two independent ways this bites, both of which look identical from the
        operator's chair (the robot just stops following):

        * the GMR stream stalls -- the relay keeps handing back its last
          payload, so the timestamp stops moving;
        * the buffer starves because too few frames have reached it. A sim host
          stamps and samples on sim time; hardware stamps and samples on wall
          time, so clock drift is not a legitimate reason for starvation.
        """
        if t_ns == self._last_stream_t_ns:
            self._stream_stall_ticks += 1
            # 25 policy ticks ~ 0.5 s of the same frame
            if self._stream_stall_ticks == 25:
                self._warn(
                    "stall",
                    "the GMR reference has not advanced for ~0.5 s -- the robot "
                    "is holding, not tracking. Check the GMR subprocess and the "
                    "headset's body tracking.",
                )
        else:
            self._stream_stall_ticks = 0
            self._last_stream_t_ns = t_ns

        buf = getattr(ctx.tracking, "motion_buffer", None)
        frames = getattr(buf, "_timestamps_ns", None)
        if frames is None or len(frames) > 1:
            self._starved_ticks = 0
            return
        # Sustained, not momentary: the buffer is legitimately empty for a tick
        # or two right after reset_stream() and at the first push of a POSE
        # session, and warning about that would train the operator to ignore it.
        self._starved_ticks += 1
        if self._starved_ticks == 15:
            self._warn(
                "starved",
                f"the motion buffer has been down to {len(frames)} frame(s) for "
                "~0.3 s: the policy is tracking a frozen reference with zero "
                "velocity. Check the GMR publisher and relay delivery.",
            )

    def _push_tracking(self, ctx: Sim2RealContext, payload: dict, t_ns: int) -> float:
        """POSE: blend from the held pose onto the live stream, then pass through."""
        live = qpos_from_payload(payload, _root_body_idx(payload))
        entering = self._blend_pending or not self._pushed_any
        if entering:
            self._blender.begin(self._anchor_qpos, live, t_ns)
            self._blend_pending = False
        elif payload.get(MOTION_FIRST_FRAME_KEY) and self._blender.active:
            # The publisher re-anchored itself mid-ramp (it flags that after an
            # XR reconnect). Re-align the anchor to the new frame; the elapsed
            # weight is unaffected.
            self._blender.restart_alignment(live)

        out, alpha = self._blender.step(live, t_ns)
        # Stamp a new motion segment when the reference discontinuously changes
        # where it sits in the world, so RealtimeMotionBuffer re-attaches it to
        # the frame it is already holding (yaw-only rotation + xy translation).
        # At alpha 0 the first blended frame IS the aligned anchor, which makes
        # that re-attachment a provable no-op -- see reference_blend.
        stamp = entering or bool(payload.get(MOTION_FIRST_FRAME_KEY))
        if alpha >= 1.0:
            # Steady state: forward the publisher's own arrays untouched rather
            # than round-tripping them through our FK. They are identical (both
            # are FK of this qpos on the same MJCF), but "untouched" needs no
            # such argument to stay true.
            frame = self._frame_from_payload(payload, t_ns)
        else:
            frame = self._fk.payload_from_qpos(out)
        self._push(ctx, frame, out, t_ns, stamp)
        # Recorded as the policy tracks it (segment frame), not the raw stream.
        self._last_ref_qpos = self._last_pushed_qpos_tracked
        return alpha

    def _push_stopping(self, ctx: Sim2RealContext, t_ns: int) -> float:
        """IDLE and OFF: hold the pose, or ramp it back to standing first.

        IDLE pushes nothing at all. OFF ramps to the standing default once and
        then also falls silent, so both end in the same motionless hold -- the
        difference is only which pose is held.
        """
        if not (self._off_blend_pending or self._blender.active):
            self._blender.cancel()
            self._last_ref_qpos = (
                self._tracked_qpos(ctx, self._anchor_qpos)
                if (self._record_idle and self._anchor_qpos is not None)
                else None
            )
            return 1.0

        if self._off_blend_pending:
            # Stand up WHERE THE ANCHOR ALREADY STANDS, not where the robot is
            # in world coordinates. The anchor is the last frame we pushed, so
            # it is expressed in whatever raw frame this motion segment uses;
            # the buffer's segment transform then carries the target to the
            # right place in the world, exactly as it does every other frame.
            # Building the target from ctx.read_root_pos()/read_pelvis_state()
            # instead -- the obvious thing, and what this did first -- feeds a
            # robot-world frame into a buffer that is still applying an
            # operator->robot transform, so "stand still" arrives rotated and
            # metres away, and ratchets a bit further on every disengage.
            self._off_target_qpos = stand_in_place_qpos(
                self._anchor_qpos,
                self._fk.standing_joints(ctx.standing_joints()),
                height=self._fk.default_root_height,
            )
            # align=False: the target is already in the anchor's frame, and the
            # ramp starts exactly at the anchor, so the segment stays continuous
            # and needs no motion_first_frame stamp.
            self._blender.begin(
                self._anchor_qpos, self._off_target_qpos, t_ns, align=False
            )
            self._off_blend_pending = False

        out, alpha = self._blender.step(self._off_target_qpos, t_ns)
        self._push(ctx, self._fk.payload_from_qpos(out), out, t_ns, stamp=False)
        self._last_ref_qpos = (
            self._last_pushed_qpos_tracked if self._record_idle else None
        )
        return alpha

    def _frame_from_payload(self, payload: dict, t_ns: int) -> dict:
        """The publisher's own frame, rebuilt from scratch.

        From scratch and not ``{**payload}``: that would carry
        ``publish_t_ns``/``smplx_t_ns``/``paused`` and the ``xrobot_*`` viewer
        keys along, and would make ``__append_payload`` resolve a timestamp we
        did not choose. One clock across ramp and pass-through is what keeps
        every frame we push strictly newer than the last -- otherwise the
        buffer bisect-inserts it into the past and its last-frame cache rejects
        it outright.
        """
        frame = {
            "joint_names": payload["joint_names"],
            "body_names": payload["body_names"],
            "joint_pos": payload["joint_pos"],
            "body_pos_w": payload["body_pos_w"],
            "body_quat_w": payload["body_quat_w"],
            "pico_recv_time_ns": t_ns,
        }
        if payload.get("qpos") is not None:
            frame["qpos"] = payload["qpos"]
        return frame

    def _push(self, ctx, frame: dict, qpos, t_ns: int, stamp: bool) -> None:
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        frame["pico_recv_time_ns"] = t_ns
        # Carried for parity with the publisher's own frames, and so a pushed
        # frame is self-describing. The buffer prefers "joint_pos" and only
        # falls back to slicing "qpos", so the two cannot disagree.
        frame.setdefault("qpos", qpos.tolist())
        if stamp:
            frame[MOTION_FIRST_FRAME_KEY] = True
        ctx.tracking.motion_buffer.push_payload(frame)
        # The pose we are now holding, for whenever we next stop pushing. RAW
        # (segment) frame on purpose: the blender and the buffer both work in
        # it, and stand_in_place_qpos builds the OFF target from it.
        self._anchor_qpos = qpos
        self._pushed_any = True
        # ...and the same pose as the POLICY sees it: root passed through the
        # segment transform push_payload just (re)installed. This is what gets
        # recorded -- see _tracked_qpos.
        self._last_pushed_qpos_tracked = self._tracked_qpos(ctx, qpos)

    @staticmethod
    def _tracked_qpos(ctx, qpos):
        """``qpos`` (36: root pos/quat + joints) with its root re-expressed in
        the motion buffer's current segment frame -- the frame the tracking
        policy consumes (schema_version >= 8).

        The raw stream's root is in the operator's GMR world; the buffer
        re-attaches every new segment (POSE re-entry, XR reconnect) onto the
        pose it is already holding with a yaw-only + xy-only transform
        (RealtimeMotionBuffer._update_segment_transform_locked), so the policy
        never sees the operator's displacement during a pause. Recording the
        raw root instead put that displacement into ``action.ref_qpos`` as a
        single-frame teleport with no counterpart in the image or the state --
        and no counterpart at deploy either, where RefQposAdapter stamps one
        segment per episode. z, roll, pitch and the 29 joints are untouched.
        """
        q = np.asarray(qpos, dtype=np.float32).reshape(-1).copy()
        buf = getattr(getattr(ctx, "tracking", None), "motion_buffer", None)
        transform = getattr(buf, "transform_root_pose", None)
        if transform is None:  # a host without a buffer: raw frame, as before
            return q
        pos, quat = transform(q[0:3], q[3:7])
        q[0:3] = pos
        q[3:7] = quat
        return q

    def _debug_log_payload(self, ctx, payload, phase, alpha) -> None:
        """~1 Hz fingerprint of what the policy is actually tracking.

        Reports the phase and blend weight next to the payload, so a live
        session can tell an intentional IDLE hold from a stalled relay, and a
        ramp still in progress from a finished one -- independently of whether
        ModeManager engagement itself works.
        """
        now = ctx.clock_s()
        if now - self._debug_log_t < 1.0:
            return
        self._debug_log_t = now
        try:
            joint_pos = payload.get("joint_pos")
            sample = (
                np.asarray(joint_pos, dtype=np.float32).reshape(-1)[:3].tolist()
                if joint_pos is not None
                else None
            )
        except Exception:
            sample = None
        print(
            f"[g1_bridge][debug] gmr payload: phase={phase.name} "
            f"alpha={alpha:.2f} paused={payload.get('paused')} "
            f"seq={payload.get('seq')} publish_t_ns={payload.get('publish_t_ns')} "
            f"smplx_t_ns={payload.get('smplx_t_ns')} joint_pos[:3]={sample}"
        )

    # ------------------------------------------------------------------ #
    # Recording
    # ------------------------------------------------------------------ #
    def on_episode_start(self) -> None:
        """Forget this episode's reference origin; the next sample re-takes it."""
        self._ref_anchor = None

    def _episode_relative_root(self, root_pos, root_quat):
        """``(pos, quat)`` of a reference root, in THIS episode's own frame.

        The input root is the SEGMENT-FRAME root (schema v8: ``_tracked_qpos``,
        the root the motion buffer serves to the policy), so what comes out is
        continuous across POSE re-entries -- an operator walking around during
        a pause moves nothing here, exactly as it moves nothing for the policy.

        The episode's first recorded reference defines the frame, so that frame's
        own root comes out at ``[0, 0, z]`` with zero yaw. Only yaw and xy are
        removed -- exactly, and only, what the consumer normalises away:

        * ``RealtimeMotionBuffer`` re-anchors each motion segment with a yaw-only
          rotation plus an xy translation (``_transform_body_pose_by_xy_yaw``,
          driven by ``yaw_quat``/``_pos_no_z``). Height, roll and pitch pass
          through it untouched -- they are gravity-observable and are real signal,
          so removing them would destroy information rather than noise.
        * At deploy the same cancellation happens to the POLICY's output:
          ``RefQposAdapter`` stamps ``motion_first_frame`` on the episode's first
          push, so the buffer measures the whole episode from the first predicted
          frame. Recording anything else means training the model to predict a
          constant that the runtime is guaranteed to subtract.

        Recording the raw stream instead (what this did before 2026-08-11) put
        the operator's absolute GMR-world xy/yaw in the action. That origin is
        per-GMR-process, not per-episode -- the subprocess is built in
        ``build_tracking``, which is cache-key-guarded and runs about once per
        process -- so episode N continued from wherever episode N-1 stopped.

        Taking the origin from the reference stream itself (rather than from the
        robot's own pose) is what keeps sim and hardware datasets comparable:
        ``read_root_pos()`` is real world coordinates in sim and a documented
        constant zero on a G1, which has no odometry.
        """
        anchor_xy, anchor_yaw_inv = self._ref_anchor
        pos = np.asarray(root_pos, dtype=np.float32).reshape(1, 3) - anchor_xy
        quat = np.asarray(root_quat, dtype=np.float32).reshape(1, 4)
        return (
            quat_rotate_numpy(anchor_yaw_inv, pos).reshape(3).astype(np.float32),
            # w >= 0, because this quaternion is RECORDED, not just consumed. Any
            # turn past pi walks the continuous stream into the negative half of
            # the double cover and it stays there for the rest of the episode --
            # on this task, 74.6% of all frames -- so the same heading reaches
            # training as both q and -q and the regression learns their mean:
            # a root that never turns. See quat_canonical.
            quat_canonical(quat_mul(anchor_yaw_inv, quat)).reshape(4).astype(np.float32),
        )

    def teleop_sample(self, ctx: Sim2RealContext) -> dict:
        """State = measured joint state grouped by limb; action = the GMR
        reference actually pushed into the policy this tick.

        The reference (root pos/quat + 29 joint angles) is what the policy is
        tracking -- not the low-level PD joint target, which is what the
        sonic-style target_q_29/action.wbc schema records. Joint velocity is
        intentionally not recorded. Hands are reported as 7-per-side target
        joint angles rather than a raw trigger scalar, matching AMO.

        The reference ROOT is reported in this episode's own frame -- see
        _episode_relative_root. ``ref_joint_pos`` needs no such treatment: joint
        angles do not live in a world frame at all.
        """
        from sim2real.utils.dex3 import grip_to_joint_targets

        if self._last_ref_qpos is None:
            return {"ref_joint_pos": None}
        qpos, _qvel = ctx.read_joint_state()
        ref = self._last_ref_qpos  # 36: [root_pos(3), root_quat(4), joint_pos(29)]
        if self._ref_anchor is None:
            # First recorded frame of the episode: it defines the frame, and so
            # comes out at xy=0, yaw=0 by construction. `_pos_no_z` and
            # `yaw_quat` here are the buffer's own conventions, imported from
            # the same module it uses -- a second implementation of "the yaw
            # part of a quaternion" is exactly how the two would drift apart.
            xy = np.asarray(ref[0:3], dtype=np.float32).reshape(1, 3).copy()
            xy[0, 2] = 0.0
            anchor_yaw = yaw_quat(np.asarray(ref[3:7], dtype=np.float32).reshape(1, 4))
            self._ref_anchor = (xy, quat_conjugate(anchor_yaw))
        ref_root_pos, ref_root_quat = self._episode_relative_root(ref[0:3], ref[3:7])
        hands = read_hand_state(ctx)
        zeros7 = np.zeros(7, dtype=np.float32)
        return {
            "left_leg": np.asarray(qpos[0:6], dtype=np.float32),
            "right_leg": np.asarray(qpos[6:12], dtype=np.float32),
            "waist": np.asarray(qpos[12:15], dtype=np.float32),
            "left_arm": np.asarray(qpos[15:22], dtype=np.float32),
            "right_arm": np.asarray(qpos[22:29], dtype=np.float32),
            "left_hand_state": np.asarray(hands.get("left", zeros7), dtype=np.float32),
            "right_hand_state": np.asarray(hands.get("right", zeros7), dtype=np.float32),
            "ref_root_pos": ref_root_pos,
            "ref_root_quat": ref_root_quat,
            "ref_joint_pos": np.asarray(ref[7:36], dtype=np.float32),
            "left_hand_action": grip_to_joint_targets(
                float(np.clip(ctx.left_trigger, 0.0, 1.0)), "left"
            ).astype(np.float32),
            "right_hand_action": grip_to_joint_targets(
                float(np.clip(ctx.right_trigger, 0.0, 1.0)), "right"
            ).astype(np.float32),
        }
