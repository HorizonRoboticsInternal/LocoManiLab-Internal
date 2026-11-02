"""G1BridgeComposite -- the robosuite adapter for :mod:`g1_bridge.session`.

This file is deliberately thin. Everything that is *teleoperation* -- the
sim2real policy runtime, the PICO source, the ModeManager, the backends, the
standing-reference bookkeeping -- lives in ``TeleopSession``, which knows
nothing about robosuite. What is left here is what only a robosuite host can
do: satisfy the composite-controller API, read/write MuJoCo, and run the
per-episode settling window that robocasa's airborne spawn requires.

The split exists so the controller that records a dataset and the controller
that deploys a policy on hardware can be the same code: a second host (real
robot, or sim2real's own MuJoCo loop) implements the same handful of methods
this class does -- ``read_joint_state``/``read_pelvis_state``/``read_root_pos``/
``read_body_pose_w``/``clock_ms`` plus the ``sim``/``qpos_addrs``/``qvel_addrs``/
``hand_mapping`` attributes -- and gets the identical control path.

Registers itself as ``"G1_BRIDGE"`` via robosuite's
``@register_composite_controller`` decorator purely as an import side effect
-- selecting it only requires ``import robocasa.g1_host.bridge`` before
``robosuite.make()`` runs (see ``robocasa.scripts.collect_demos_g1``, the
unified collection entrypoint for every backend); nothing in robosuite's own
``composite/__init__.py`` needs to change.

Every episode opens with a **settling window** (``_run_settling``), because
robocasa spawns the G1 airborne and it free-falls onto the floor over the first
handful of control steps. Throughout the window the real policy runs -- it is
the only thing that can catch the landing -- against sim2real's own default
standing frame (``default_reference.py``), and it is at the *start* of the
window, with the robot's real post-teleport pose live, that the observations'
one-shot robot-vs-motion yaw alignment gets calibrated (``_begin_settling``).
Only once the robot is quiet does the episode's real reference take over
(``_end_settling``). This is not the "interpolate-then-handoff" scheme that was
tried and rejected early on: nothing hands the policy a half-interpolated,
out-of-distribution pose -- the policy is in control the whole time. The window
is host-specific by nature, which is why it stays here rather than in the
session.

After settling it mirrors ``wbt_g1_composite.WholeBodyG1Composite``'s own
control pattern as closely as possible: the inner policy runs *continuously*,
in every mode including OFF, and what changes with ModeManager state is only
the reference it tracks --

* disengaged (OFF): a **default standing reference**, mirroring WBT's own
  ``_update_default_pose_reference`` -- the policy is still doing active,
  trained balance control, just against a static target instead of the live
  operator. This is NOT a naive fixed-pose PD hold (tried first; the robot
  doesn't stay standing).
* engaged: the live PICO-driven reference.

Changing reference is a teleport -- the two sources' root xy/yaw are unrelated
-- so a swap is flagged with ``motion_first_frame`` and
``RealtimeMotionBuffer`` re-attaches the new segment to its last transformed
output. sim2real's motion-stream contract puts that responsibility on whoever
owns the reference source; here that is this composite, because unlike the real
deployment it subscribes to GMR itself instead of letting ``pico_retarget_pub``
keep its own pause/unpause continuous. That re-attachment is yaw and xy only,
though, so the GMR tracking backends additionally ramp the *pose* across the
swap rather than stepping it -- see ``g1_bridge/reference/blend.py``,
``GenericTrackingBackend._push_reference`` and sim2real's
``docs/policy_yaw_alignment.md``.

Two backend families:

* ``sim2real_backend="sonic"``: two checkpoint flavors, distinguished by
  ``isinstance(self._tracking, SplitOnnxBridgeTracking)`` (see
  ``bridge_runtime.py``):

  * smpl (the historical default, ``smpl/policy.yaml``): only POSE mode is
    driven by sim2real, GMR-free -- tele's own PICO SMPL sample is pushed
    straight in, see ``smpl_payload.py``. OFF / PLANNER / PLANNER_FROZEN_UPPER
    hold the SONIC standing pose (no kPlanner reuse yet). VR_3PT's fallback
    to tele's own vendored WBTControllerSonicTeleop is not yet implemented
    either -- both are deferred, since both need kPlanner-driven lower-body
    locomotion which is out of scope here.
  * g1 split-onnx (encoder+decoder ONNX pair, e.g.
    GR00T-WholeBodyControl/gear_sonic_deploy's low_latency checkpoint, see
    ``g1_bridge/sonic_utils/encoder_obs.py``): no live PICO/GMR reference
    is wired up for this flavor yet, so it runs continuously in every mode
    (including OFF) against ``RealtimeMotionBuffer``'s own built-in default
    standing-pose frame -- enough to validate the encoder/decoder pair
    end-to-end, not a live teleop reference.

  The full 5-mode ModeManager state machine still runs throughout, so
  ``teleop.stream_mode`` semantics in recorded data stay identical to
  WBT-SONIC even while most modes are stubbed.

* ``sim2real_backend="amo"`` (AmoBridgeTracking): AMO (Li/Cheng/Huang/Wang,
  RSS 2025) only ever controls 15 of its 23 DOF via the neural policy --
  legs + waist. The other 8 (shoulder/elbow) plus the 6 wrist joints the
  policy doesn't model at all are driven by IK against the operator's
  PICO-calibrated 3-point wrist pose -- a completely separate channel from
  the neural policy, ported verbatim from the pre-existing standalone
  ``AMOG1Composite`` (``amo_g1_composite.py``, since removed:
  ``G1ArmIK`` + ``_tick_arms``/``_read_3pt_pose``/``_rel_target``). Locomotion
  itself is command-velocity-driven (PICO joysticks -> vx/vy/vyaw), not
  motion-tracking-driven, so there is no GMR subprocess and no motion
  payload pushed at all for this backend -- ``_step_amo`` ticks PICO
  joysticks -> ``self._tracking.amo_commands``/``amo_vyaw`` (read by
  ``sim2real.rl_policy.observations.amo``) instead, then overwrites the
  8+6 IK-driven slots in ``step_policy``'s ``q_target`` before returning.

* ``sim2real_backend`` = any other sim2real full-body tracking policy family
  (``heft``, ``scalebfm``, ``bfm_zero``, ...): driven by real GMR-retargeted
  PICO motion, published by a ``gmr_subprocess.GMRSubprocess``. Unlike the
  sonic path, this composite owns the ZMQ subscription itself
  (``gmr_relay.GMRRelay``) and *pushes* whichever payload is appropriate
  (default or latest-live) into a push-mode BridgeTracking each tick --
  sim2real's own auto-connecting ``RealtimeMotionBuffer`` (motion_mode=
  "zmq") can't be pointed at a frozen reference on demand, which is exactly
  what reproducing WBT's OFF behavior needs. These policies don't get
  tele's 5-mode PICO UX -- ModeManager's OFF/non-OFF only selects which
  reference feeds the policy and gates DATA RECORDING; hand trigger gating
  matches WBT's own convention (open in OFF) too, exactly as below.
"""

from __future__ import annotations

import numpy as np

from robosuite.controllers import load_composite_controller_config
from robosuite.controllers.composite.composite_controller import (
    register_composite_controller,
)
from robocasa.g1_host.base import G1CompositeBase
from robocasa.g1_host.mujoco_io import build_foot_geom_ids, feet_in_contact

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import

from g1_bridge.backends import (
    TickInput,
    backend_names,  # noqa: F401 (re-exported)
    get_backend_class,  # noqa: F401 (re-exported)
)
from g1_bridge.backends.base import BRIDGE_KEY_HELP
from g1_bridge.config import (
    INFERENCE_BACKENDS,  # noqa: F401 (re-exported)
    SPEC_KEY,
    build_bridge_spec,
    config_path,
    resolve_policy_config_path,  # noqa: F401 (re-exported)
)
from robocasa.g1_host.dex3_grip import build_hand_mapping, write_gripper_torques
from g1_bridge.session import TeleopSession
from g1_bridge.modality import (
    G1_BACKEND_TELEOP_KEYS,  # noqa: F401 (re-exported)
)


# ------------------------------------------------------------------------- #
# Controller-config loading -- moved here from the now-retired
# collect_demos_sim2real.py so the unified robocasa/scripts/collect_demos_g1.py
# can build a G1BridgeComposite config for any --backend without needing
# its own copy of this logic.
#
# The backend/checkpoint/PICO resolution itself is NOT robosuite's business and
# lives in g1_bridge.config.build_bridge_spec, which a real-robot host uses too.
# What is left here is the robosuite wrapper: the same JSON, loaded through
# robosuite's own loader so the result also carries type/body_parts.
# ------------------------------------------------------------------------- #

def load_sim2real_controller_config(
    backend: str,
    pico_source: str,
    policy_config: str | None = None,
    gmr_viewer: bool = False,
    inference_backend: str | None = None,
    amo_arm_source: str | None = None,
) -> dict:
    """Build a G1_BRIDGE controller config for any --backend. Used by the
    unified collect_demos_g1.py/collect_fixed_task.py in place of the old
    tele-native load_g1_controller_config (robocasa.utils.g1_data_collection).

    ``inference_backend`` overrides the controller JSON's
    ``sim2real_inference_backend`` (default ``onnx-cpu``). Note the policy is
    only one of several things competing for the tick budget here -- MuJoCo,
    rendering and the GMR subprocess share the machine -- so moving it to the
    GPU helps the policy's own latency, not necessarily the loop rate.
    """
    cfg = load_composite_controller_config(controller=str(config_path(backend)))
    cfg[SPEC_KEY] = build_bridge_spec(
        backend,
        pico_source,
        policy_config=policy_config,
        gmr_viewer=gmr_viewer,
        inference_backend=inference_backend,
        amo_arm_source=amo_arm_source,
    )
    return cfg


# Settling-window defaults, overridable per run via
# composite_controller_specific_configs. See _run_settling.
_DEFAULT_SETTLE_MAX_TICKS = 250  # 5 s at 50 Hz: hard cap, never hang an episode
_DEFAULT_SETTLE_QVEL = 2.0  # rad/s, max joint speed that still counts as settled
_DEFAULT_SETTLE_HOLD_TICKS = 5  # consecutive quiet ticks required (the landing bounces)

# G1_BACKEND_TELEOP_KEYS (sample keys/layout get_teleop_data_sample()
# actually returns, keyed by --backend) now lives in g1_bridge/
# modality.py, imported above -- schema is kept separate from this file's
# runtime logic. See that module for the per-backend field documentation.


@register_composite_controller
class G1BridgeComposite(G1CompositeBase):
    name = "G1_BRIDGE"

    # Printed once by demo_empty_teleop.run_sonic_viewer_loop's
    # setup_after_reset() whenever handle_keypress is defined (see below) --
    # required attribute, not optional, once that hasattr check passes.
    # The text itself lives with the keys, in backends/base.py, because sonic's
    # '/' handler prints it too and backends must not import this module.
    KEY_HELP = BRIDGE_KEY_HELP

    def __init__(self, sim, robot_model, grippers):
        # Address maps (_qpos_addrs/_qvel_addrs/_actuator_ids/_free_*) are
        # declared and rebuilt by G1CompositeBase.
        super().__init__(sim, robot_model, grippers)
        # One session per composite instance, i.e. per episode -- robosuite
        # rebuilds this object on every env.reset(). Expensive handles come
        # from session._shared, so "fresh per episode" costs nothing.
        self._session = TeleopSession(self)
        self._hand_mapping = None
        # Per-episode settling window -- see _run_settling.
        self._settling = True
        self._settle_ticks = 0
        self._settle_quiet_ticks = 0
        # None => operator-driven. Set by set_vla_action to run under a policy.
        self._vla_action = None
        self._foot_geom_ids = frozenset()
        self._settle_max_ticks = _DEFAULT_SETTLE_MAX_TICKS
        self._settle_qvel = _DEFAULT_SETTLE_QVEL
        self._settle_hold_ticks = _DEFAULT_SETTLE_HOLD_TICKS

    # ------------------------------------------------------------------ #
    # TeleopHost: what the session reads back out of the sim. The four
    # read_* methods come from G1CompositeBase; these are the rest.
    # ------------------------------------------------------------------ #
    @property
    def session(self) -> TeleopSession:
        return self._session

    @property
    def hand_mapping(self):
        return self._hand_mapping

    def read_hand_state(self):
        """Measured finger joints, straight off the sim's own qpos.

        The backends used to read this themselves through ``ctx.hand_mapping`` +
        ``ctx.sim``; going through the host instead is what lets the same
        ``teleop_sample`` run on hardware, where the numbers come from the Dex3
        driver's DDS feedback rather than from a simulator.
        """
        if self._hand_mapping is None:
            return None
        out = {}
        for side in ("left", "right"):
            addrs = self._hand_mapping.get(side)
            if addrs is None:
                return None
            out[side] = np.asarray(
                [float(self.sim.data.qpos[a]) if a is not None else 0.0
                 for a in addrs["qpos"]],
                dtype=np.float32,
            )
        return out

    def spawn_joint_targets(self):
        """``(qpos_addr, qvel_addr, angle)`` for the stance this episode's
        policy actually holds, for robocasa to spawn the robot in.

        Optional protocol, called by ``env_utils.set_robot_base`` while it
        places the robot; a composite without this method keeps the MJCF
        ``init_qpos`` stance and the behaviour it had before.

        Why it exists: the MJCF stance and the policy's equilibrium stance are
        different poses, so the robot used to spend the first ~0.2 s of every
        episode bending from one into the other under load -- measured 4.5 cm of
        pelvis descent while the feet moved 2.5 mm, i.e. leg flexion, not a
        fall. That transient is what slides the robot off its intended spawn
        point. Starting from the pose the legs are going to end up in removes
        the transient rather than damping it.

        Returns an empty tuple when the checkpoint declares no default pose, so
        the caller falls back to the MJCF stance.
        """
        angles = self._session.standing_joints()
        addrs, vaddrs = self.qpos_addrs, self.qvel_addrs
        if angles is None or addrs is None or vaddrs is None:
            return ()
        angles = np.asarray(angles, dtype=np.float64).reshape(-1)
        if len(addrs) != angles.shape[0]:
            return ()
        return tuple(
            (qa, qv, float(a))
            for qa, qv, a in zip(addrs, vaddrs, angles)
            if qa is not None and qv is not None
        )

    def clock_ms(self) -> int:
        """Monotonic-enough tick stamp for sim2real's RobotState. Sim time, not
        wall time: the policy's own notion of "now" must follow the sim clock,
        which in a headless run advances far faster than real time."""
        return int(self.sim.data.time * 1000)

    # ------------------------------------------------------------------ #
    # Init flow: skip the base class's part-controller machinery, exactly
    # like WholeBodyG1Composite -- we write ctrl directly.
    # ------------------------------------------------------------------ #
    def load_controller_config(
        self, part_controller_config, composite_controller_specific_config=None
    ):
        cfg = composite_controller_specific_config or {}
        self.composite_controller_specific_config = cfg
        self._init_robosuite_state(part_controller_config)

        # robosuite calls load_controller_config() again on every env.reset()
        # (Robot.reset() -> _load_controller(), unconditionally -- confirmed
        # via a live episode-reset test). sim/sim.model are genuinely new
        # objects each reset (robocasa scenes re-randomize), so the joint/
        # actuator address maps below MUST be rebuilt every call. What must
        # NOT be rebuilt -- the PICO SDK connection, sim2real Tracking/ONNX
        # session, GMR subprocess/relay -- is guarded inside the session's
        # process-wide cache. See TeleopSession.configure.
        self._build_mappings()
        self._hand_mapping = build_hand_mapping(self.sim, self.robot_model)
        # Geom ids are model-relative and robocasa hands out a new model every
        # reset, so this belongs with the address maps, not with the cache.
        self._foot_geom_ids = build_foot_geom_ids(self.sim, self.robot_model)
        self._settle_max_ticks = int(
            cfg.get("sim2real_settle_max_ticks", _DEFAULT_SETTLE_MAX_TICKS)
        )
        self._settle_qvel = float(cfg.get("sim2real_settle_qvel", _DEFAULT_SETTLE_QVEL))
        self._settle_hold_ticks = int(
            cfg.get("sim2real_settle_hold_ticks", _DEFAULT_SETTLE_HOLD_TICKS)
        )
        self._session.configure(cfg)

    # The no-env-action stubs (setup_action_split_idx / set_goal /
    # update_state / action_limits / create_action_vector /
    # create_action_dict_from_action_vector) come from G1CompositeBase.

    def reset(self):
        # Re-arm the settling window, then let the session clear the policy
        # runtime's own state. Ordering within the session is load-bearing
        # (buffer stream, then per-episode fields, then the backend) -- see
        # TeleopSession.reset.
        self._settling = True
        self._settle_ticks = 0
        self._settle_quiet_ticks = 0
        self._vla_action = None
        self._session.reset()
        self._session.vla_reset()

    # ------------------------------------------------------------------ #
    # Per-episode settling window
    # ------------------------------------------------------------------ #
    def _run_settling(self, tick: TickInput):
        """Ride out robocasa's airborne spawn, then hand over the real reference.

        Robocasa does not place the G1 on the floor: it spawns airborne and
        free-falls ~13 cm at ~9.1 m/s^2 over the first ~5 control steps of every
        episode (zero foot contacts throughout, then a bounce). Two things went
        wrong when the episode's reference and heading alignment were both
        committed during that fall:

        * the frozen reference captured a mid-air leg shape the landed robot
          could not hold, and
        * the alignment was calibrated against a stale, pre-teleport heading.

        So this window runs the **real policy** -- it is the only thing that can
        actually catch the landing; a plain PD hold at the default pose does not
        keep a humanoid upright, and neither does sim2real's own ``init`` ramp,
        which assumes a robot that is already standing still -- against the
        default standing frame captured ONCE at the spawn pose
        (``_begin_settling``) and re-pushed unchanged every tick. Once the robot
        is quiet, ``_end_settling`` swaps in the reference the episode will
        actually track.

        Captured once, not re-anchored to the live robot each tick (which is
        what this used to do), for two reasons. A reference rebuilt from the
        robot's own heading every tick makes the *heading error the policy sees*
        independent of how far the robot has actually turned -- the reference
        chases the robot instead of correcting it, so a yaw disturbance is never
        opposed and can wind up without bound. And re-pushing identical content
        makes the buffer's wall-clock finite difference exactly zero, instead of
        dividing a moving frame by a headless sim's tiny real-time dt -- the
        hazard ``TeleopSession.push_synthetic_default_once`` documents and this
        window used to violate on every one of its ticks.

        Nothing here is recorded: get_teleop_data_sample() reports not-ready for
        the whole window.
        """
        if self._settle_ticks == 0:
            self._begin_settling(tick)
        self._settle_ticks += 1
        self._session.push_standing_frame()
        out = self._session.step_policy(tick.qpos)
        if self._settled(tick.qvel):
            self._end_settling(tick)
        return out

    def _begin_settling(self, tick: TickInput) -> None:
        """Calibrate the observations' one-shot robot-vs-motion yaw alignment.

        This is the only place it happens, and the timing is the whole point.
        The alignment is computed inside ``Tracking.reset()``, from the robot
        pose in ``state_processor`` and the reference in the motion buffer --
        so both have to be live and mutually consistent at this instant. Here
        they are: the robot's real post-teleport spawn pose, and a reference
        anchored to that same pose.

        sim2real's motion-stream contract makes the offset permanent for the
        episode and explicitly forbids recomputing it later
        (``docs/policy_yaw_alignment.md``), which is why ``_end_settling`` swaps
        the reference *without* re-running this -- it flags the swap as a new
        motion segment instead and lets the buffer re-attach it.

        What this replaces: the alignment used to be calibrated in ``reset()``,
        which robosuite runs BEFORE the scene teleports the robot, pairing a
        stale heading (-23.9 deg measured) with ``RealtimeMotionBuffer``'s
        built-in default frame (world origin, yaw 0) while the robot actually
        spawned at +33.7 deg. Every ``target_body_rot_future_to_robot_base``
        entry then carried a constant ~40-60 deg heading error for the whole
        episode -- including the pelvis's own target, which is identity by
        construction. There *was* a repair path
        (``push_synthetic_default_once``), but it only ran when no GMR payload
        existed, and the GMR subprocess publishes even under
        ``--pico-source mock``, so whether it fired at all was a startup race.
        """
        # Freeze the heading the whole window will aim at, then push before
        # calibrating: the alignment is computed from exactly this frame paired
        # with exactly this robot pose.
        self._session.set_anchor_heading(self._read_pelvis_state()[0])
        self._session.push_standing_frame()
        self._session.calibrate_alignment()

    def _settled(self, qvel) -> bool:
        """Feet down and the robot quiet for _settle_hold_ticks in a row.

        Consecutive ticks, not one: the landing bounces (measured foot contacts
        appearing at step 4, gone again by step 10), so a single quiet tick
        proves nothing. The overall cap is not a formality either -- a
        fixed-lower-body G1 variant never makes foot contact at all, and would
        otherwise sit in this window forever.
        """
        if self._settle_ticks >= self._settle_max_ticks:
            return True
        if not feet_in_contact(self.sim, self._foot_geom_ids):
            self._settle_quiet_ticks = 0
            return False
        if float(np.abs(np.asarray(qvel)).max()) > self._settle_qvel:
            self._settle_quiet_ticks = 0
            return False
        self._settle_quiet_ticks += 1
        return self._settle_quiet_ticks >= self._settle_hold_ticks

    def _end_settling(self, tick: TickInput) -> None:
        # Under a VLA there is no operator sample to seed from, and none is
        # needed: the adapter flags its first push as a new motion segment, so
        # RealtimeMotionBuffer re-attaches the reference to the pose settling
        # left the robot holding -- exactly what seed_reference arranges for
        # the operator path.
        if getattr(self, "_vla_action", None) is None:
            self._session.seed_reference(tick)
        self._settling = False

    # ------------------------------------------------------------------ #
    # Control loop
    # ------------------------------------------------------------------ #
    #: Does the settling window apply when a POLICY, not an operator, drives?
    #:
    #: No, and this is not a tuning knob -- it is the difference between a
    #: working sonic eval and a broken one. The window is right for the operator
    #: path: nobody is commanding anything during the spawn, so the whole-body
    #: policy tracks its own standing frame until the robot is quiet.
    #:
    #: Under a VLA that same window switches the decoder's input stream out from
    #: under it mid-episode. Throughout the window the decoder is driven by
    #: ENCODER-produced tokens (encoded from the standing reference), and its
    #: ten-frame action history fills up with the answers to those; at handover
    #: the tokens start coming from the policy instead, into a history that
    #: belongs to a different token stream. Measured on gr00t
    #: ``bench_sonic_KitchenOpenFridge_v2`` (100 episodes, 2026-08-09):
    #: settle-then-hand-over 60/100, policy-from-tick-0 82/100 -- the latter
    #: matching ``WholeBodyG1Composite``, which has no window at all and scores
    #: 83/100. Letting the policy catch its own landing costs nothing measurable.
    #:
    #: What the window also owns -- the one-shot yaw calibration in
    #: ``_begin_settling`` -- still happens, at tick 0, against the live spawn
    #: pose. That was never about settling; see ``run_controller``.
    SETTLE_UNDER_VLA = False

    @property
    def accepts_vla_action(self) -> bool:
        """Whether the driver should be querying its policy yet.

        Only ever False when :attr:`SETTLE_UNDER_VLA` puts the policy behind the
        settling window: actions handed over during the window are DISCARDED, so
        a driver that kept querying anyway would burn the head of its first
        action chunk on frames the robot was still falling through and then
        apply whatever index it had reached. Wait on this instead.
        """
        return not self._settling or not self.SETTLE_UNDER_VLA

    def set_vla_action(self, action: dict | None) -> None:
        """Hand this composite one timestep of policy output.

        Setting an action switches the tick from operator-driven to
        policy-driven; passing None switches back. Call it once per new
        timestep the driver wants applied -- the value is held until replaced,
        so a control tick that arrives between policy chunks re-applies the
        last one rather than dropping to a null command.

        The action's keys must match the backend's ``vla_action_space``; that
        is checked once, on the first tick, by the adapter.
        """
        self._vla_action = action

    def run_controller(self, enabled_parts):
        qpos, qvel = self._read_joint_state()
        sim_time = float(self.sim.data.time)

        vla_action = getattr(self, "_vla_action", None)
        if vla_action is None:
            sample = self._session.sample_pico()
            mode, buttons = self._session.begin_tick(sample, self._settling)
        else:
            # A policy is driving: there is no operator to sample. robosuite
            # calls this once per PHYSICS substep -- ten times per control step
            # -- and polling the headset (or, in sim eval, its mock stand-in)
            # that often costs 1.7 ms of every 3.5 ms this controller spends,
            # for a sample nothing on this path reads: vla_step_backend takes
            # the command from the policy, the hands from
            # vla_hand_targets(), and consults neither tick.mode nor
            # tick.buttons. Skipping it is most of the gap to
            # WholeBodyG1Composite (measured 51 -> 56 steps/s on
            # IThorKitchenOpenFridge, versus wbt's 58).
            #
            # The ModeManager is deliberately left where it is rather than
            # advanced from a synthetic sample: with no buttons arriving it has
            # nothing to advance on, and its state still describes the episode
            # for get_teleop_data_sample().
            sample, buttons = None, None
            mode = self._session.mode_manager.current

        if self._session.policy_due(sim_time):
            tick = TickInput(
                qpos=qpos,
                qvel=qvel,
                sim_time=sim_time,
                mode=mode,
                buttons=buttons,
                sample=sample,
            )
            # A policy is driving: calibrate the yaw alignment on the live spawn
            # pose and hand over in the same tick, instead of running the window
            # on encoder-produced tokens and switching the decoder's input
            # stream under it later. See SETTLE_UNDER_VLA.
            if self._settling and vla_action is not None and not self.SETTLE_UNDER_VLA:
                self._begin_settling(tick)
                self._end_settling(tick)
            if self._settling:
                target_q, kps, kds = self._run_settling(tick)
            elif vla_action is not None:
                target_q, kps, kds = self._session.vla_step_backend(tick, vla_action)
            else:
                target_q, kps, kds = self._session.step_backend(tick)
            self._session.cache_command(target_q, kps, kds, sim_time)

        target_q, kps, kds = self._session.cached_command
        self._write_body_torques(target_q, kps, kds, qpos, qvel)

        # Hands: predicted joint angles when the policy provides them, else the
        # operator's trigger. A VLA that predicts hands but is driven through
        # the trigger path would have its grasp silently replaced by a scalar.
        hands = None if vla_action is None else self._session.vla_hand_targets(vla_action)
        left, right = hands or (self._session.left_trigger, self._session.right_trigger)
        write_gripper_torques(self.sim, self._hand_mapping, left, right)
        return {}

    # ------------------------------------------------------------------ #
    # Operator / collection surface -- delegated to the session.
    # ------------------------------------------------------------------ #
    def handle_keypress(self, keycode):
        self._session.handle_keypress(keycode)

    @property
    def hud_cheatsheet(self):
        return self._session.hud_cheatsheet

    def get_teleop_data_sample(self) -> dict:
        """This tick's recording sample, from the active backend.

        Empty during the settling window: the policy is holding the robot's own
        pose while it lands, not tracking anything an operator asked for.
        build_frame()'s readiness gate reads that as "not ready" and skips the
        frame, which is what keeps the landing out of the dataset.

        The amo backend uses its OWN joint-grouped schema (hand/arm/leg + torso
        rpy/height + locomotion commands); every other backend matches
        WholeBodyG1Composite's target_q_43/vr_3pt/smpl field contract
        (build_frame requires target_q_43 -- the 43-wide action.wbc, schema v6;
        target_q_29 before that -- and, whenever that's non-None,
        stream_mode, locomotion_mode, facing_direction too; everything else is
        optional). See g1_bridge/modality.py for the per-backend fields.
        """
        if self._settling:
            return {}
        return self._session.teleop_sample()

    # -- Compatibility accessors used by collection and viewer tooling. The
    # underlying state lives on the session.
    @property
    def _ctrl_type(self):
        return self._session.ctrl_type

    @property
    def _mode_manager(self):
        return self._session.mode_manager

    @property
    def _pico_source(self):
        return self._session.pico_source

    @property
    def _tracking(self):
        return self._session.tracking
