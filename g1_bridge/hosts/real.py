"""The real-robot host: the second implementation of the teleop host seam.

``TeleopSession`` reaches its host through five methods (``read_joint_state`` /
``read_pelvis_state`` / ``read_root_pos`` / ``read_body_pose_w`` / ``clock_ms``)
plus a handful of optional attributes. In the benchmark those are answered out of
MuJoCo; here they are answered out of the robot's own state stream, and the
control path in between -- policy runtime, PICO source, ModeManager, backends,
reference blending, and the ``teleop_sample()`` that *becomes the dataset* -- is
byte-for-byte the same code. That identity is the whole point: a dataset recorded
on hardware and one recorded in sim are then the same thing to a trainer.

Shape of one tick::

    state = robot_io.read_state()          # G1RobotIO (inline DDS) or ZMQRobotIO
    host._state = state                    # feeds read_joint_state/read_pelvis_state
    target_q, kps, kds = session.step_backend(tick)
                                           # -> push_state -> tracking.step()
    robot_io.write_command(target_q, ...)  # what the runtime just computed

Note the two RobotIOs. ``session`` drives sim2real in ``push`` mode (an in-process
``PushRobotIO`` it owns, see ``bridge_runtime``); this host owns the *hardware* one
and pumps between them. Do not "simplify" that by pointing the session at the
hardware io directly: ``BridgeTracking`` refuses any robot_io but ``push``, and
push mode is what lets the bridge decide per tick which reference the policy sees.

Startup is not the sim's. ``BridgeTracking.__init__`` arms the policy immediately,
which is safe in a simulator that spawns the robot airborne and rides out the
landing, and is not safe on hardware. So this host starts in ``zero`` (hold the
measured pose), ramps to the default stance on ``init``, and only arms the policy
when the operator explicitly says so -- see :meth:`RealRobotRunner.arm`.
"""

from __future__ import annotations

import select
import sys
import termios
import time
import tty
from typing import Any

import json

import numpy as np
import zmq
from loguru import logger
from termcolor import colored

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import

from g1_bridge.backends.base import TickInput
from g1_bridge.session import TeleopSession

#: Consecutive ticks with no fresh robot state before the host drops to zero
#: mode. The policy would otherwise keep running on a frozen observation --
#: ``PushRobotIO.read_state`` deliberately returns the last state rather than
#: blocking, so a dead bridge looks exactly like a stationary robot.
_STALE_TICK_LIMIT = 5


def _axis_or_none(buttons, key: str):
    """One PICO stick as ``[x, y]``, or None when the source publishes no axes.

    JSON-ready (a list, not a tuple) because this only ever feeds the status
    payload, and tolerant of a malformed value for the same reason: a teleop
    pane that raises is worse than one that says "unknown".
    """
    value = (buttons or {}).get(key)
    if value is None:
        return None
    try:
        x, y = value
        return [float(x), float(y)]
    except (TypeError, ValueError):
        return None


class RealRobotHost:
    """Host surface over a hardware ``RobotIO``.

    Two parts of the sim host's surface have no hardware source: absolute root
    position (no odometry -- ``read_root_pos`` returns zeros) and world body
    poses (``read_body_pose_w`` raises). What a backend needs of those two is
    therefore the whole question of whether it runs here:

    * heft/scalebfm -- neither. Supported.
    * almi -- neither either, which is not obvious and was checked term by term:
      its observations read only body-frame angular velocity, the root
      quaternion, joint angles/velocities, the previous action and its own
      command (``sim2real/rl_policy/observations/almi.py``) -- ``base_height`` is
      a *command*, not a measurement; its neutral reference forward-kinematics
      the policy's own default pose rather than asking the host; and its EE
      targets come out of the GMR payload, which the publisher already FK'd.
      It also never feeds the motion buffer, so the synthetic standing frame
      built from those zero roots is not something it reads. Supported.
    * sonic -- finger joint angles, which the Dex3 driver's DDS feedback supplies
      (``read_hand_state``). Supported.
    * amo -- a real torso height (``read_root_pos()[2]``), which is genuinely
      unobservable; the real recording layout drops that one state group.
    """

    #: No simulator. The session exposes these to backends via getattr, and a
    #: backend that actually needs one will fail loudly rather than silently
    #: reading zeros.
    sim = None
    robot_model = None
    hand_mapping = None
    qpos_addrs = None
    qvel_addrs = None

    def __init__(self, robot_io, *, joint_count: int = 29, expected_rate_hz: float | None = None):
        self._io = robot_io
        self._joint_count = joint_count
        self._state = None
        self._stale_ticks = 0
        self._skipped = 0
        #: Set by the runner once the hand driver exists; read_hand_state uses it.
        self._dex3_driver = None
        #: Expected interval between state packets, for gap accounting only.
        self._expected_dt_ms = 1000.0 / expected_rate_hz if expected_rate_hz else None
        self.session = TeleopSession(self)

    # ------------------------------------------------------------------ #
    # State pump
    # ------------------------------------------------------------------ #
    @property
    def io(self):
        return self._io

    @property
    def state(self):
        return self._state

    def poll(self) -> bool:
        """Latch this tick's robot state. False if nothing fresh arrived.

        ``ZMQRobotIO`` returns the last decoded message (or None before the first
        one); ``G1RobotIO`` reads the SDK synchronously and always has one.
        Staleness is judged on the state's own tick, which both back-ends stamp.
        """
        state = self._io.read_state()
        if state is None:
            self._stale_ticks += 1
            return False
        if self._state is not None and int(state.tick) == int(self._state.tick):
            self._stale_ticks += 1
            return False
        self._skipped += self._gap(state)
        self._state = state
        self._stale_ticks = 0
        return True

    def poll_next(self, timeout_s: float = 1.0, poll_dt: float = 0.0005) -> bool:
        """Block until a state NEWER than the last one arrives. False on timeout.

        This is the loop's clock, deliberately -- one policy step per state packet,
        the way LocoManiLab's original HeFT integration drives its controller
        (``src/deploy.py``: ``process_state(wait_next=True)`` then ``send_cmd()``).
        A free-running wall-clock loop reading "whatever is latest" instead beats
        against the bridge's own publish clock: some ticks re-use a state already
        acted on, others skip one, and the phase error walks. On hardware that is
        felt as joint shake, and it is the first thing the repo's own docs tell you
        to suspect (``docs/robot_io.md``). Locking to the packet removes the beat
        entirely -- observation and command then refer to the same instant.
        """
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.poll():
                return True
            time.sleep(poll_dt)
        return False

    def _gap(self, state) -> int:
        """How many state packets went missing before this one (0 when in step)."""
        if self._state is None or self._expected_dt_ms is None:
            return 0
        delta = int(state.tick) - int(self._state.tick)
        return max(0, int(round(delta / self._expected_dt_ms)) - 1)

    @property
    def skipped(self) -> int:
        return self._skipped

    @property
    def stale(self) -> bool:
        return self._stale_ticks >= _STALE_TICK_LIMIT

    @property
    def ready(self) -> bool:
        return self._state is not None

    # ------------------------------------------------------------------ #
    # The host surface TeleopSession consumes
    # ------------------------------------------------------------------ #
    def read_joint_state(self):
        """(qpos, qvel) for the 29 body joints, in G1_JOINT_NAMES order.

        Layout comes from the io itself: ``qpos = [root_pos(3), quat(4), joints]``
        and ``qvel = [root_lin(3), omega(3), joints]`` -- see
        ``G1RobotIO.read_state`` / ``ZMQRobotIO._message_to_state``, which fill
        only the IMU and motor slots.
        """
        s = self._require_state()
        return (
            np.asarray(s.qpos[7:], dtype=np.float32),
            np.asarray(s.qvel[6:], dtype=np.float32),
        )

    def read_pelvis_state(self):
        """(root quaternion wxyz, body-frame angular velocity) from the IMU."""
        s = self._require_state()
        quat = np.asarray(s.qpos[3:7], dtype=np.float32)
        n = float(np.linalg.norm(quat))
        quat = quat / n if n > 1e-8 else np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        return quat, np.asarray(s.qvel[3:6], dtype=np.float32)

    def read_root_pos(self):
        """Zeros: a G1 has no odometry, so world root position is unobservable.

        This is safe rather than approximate, and only because of where the value
        is consumed. ``DefaultReferenceFrame.default_qpos_at`` uses ``root_pos[:2]``
        and nothing else, i.e. it only decides *where in the world* the standing
        reference is drawn; ``TeleopSession.step_policy`` already pushes zeros as
        the root for BOTH hosts, so the policy never observes this number; and
        ``RealtimeMotionBuffer`` re-anchors each new segment to the robot's own
        pose regardless. What would break is a consumer of absolute position --
        amo's torso height (``read_root_pos()[2]``) is exactly that, which is one
        of the reasons amo is not supported on this host yet.
        """
        return np.zeros(3, dtype=np.float32)

    def read_body_pose_w(self, body_name: str):
        raise NotImplementedError(
            "read_body_pose_w needs forward kinematics on hardware (the robot "
            "reports joints and IMU, not body poses). Implement it via "
            "g1_bridge.reference.default.DefaultReferenceFrame's MJCF + "
            "mujoco.mj_forward when adding a backend that needs it; none of "
            "heft, scalebfm and almi do."
        )

    def read_hand_state(self):
        """Measured Dex3 finger joints, or None when no hand is attached.

        Same shape the sim host returns, from the driver's own DDS feedback
        (``rt/dex3/{left,right}/state``). The driver is handed over by the runner
        because it owns the hand's lifetime; before it exists, or with
        ``--dex3 none``, this is None and the recorder simply has no hand columns
        to fill rather than filling them with zeros.
        """
        driver = self._dex3_driver
        if driver is None:
            return None
        try:
            feedback = driver.read_feedback()
        except Exception as exc:  # noqa: BLE001 -- a hand fault must not stop control
            logger.debug(f"dex3 feedback unavailable: {exc}")
            return None
        if not feedback:
            return None
        return {
            side: np.asarray(feedback.get(side, np.zeros(7)), dtype=np.float32)
            for side in ("left", "right")
        }

    def clock_ms(self) -> int:
        """The state's own timestamp -- wall clock, in milliseconds.

        Taken from the io rather than read fresh here so the observation, the
        policy clock and the recorded frame all refer to the same instant.
        """
        s = self._require_state()
        return int(s.tick)

    def _require_state(self):
        if self._state is None:
            raise RuntimeError(
                "no robot state yet -- call poll() until it returns True before "
                "stepping (with --robot-io zmq this means the bridge process "
                "isn't publishing on tcp://*:5590 yet)"
            )
        return self._state

    def close(self):
        try:
            self._io.close()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"robot io close failed: {exc}")


class RealRobotRunner:
    """The 50 Hz control loop, the arming state machine, and the operator keys.

    Owns nothing about datasets: a caller passes ``on_tick`` and gets one call per
    tick after the command has been written, which is where the collector builds
    and records its frame.
    """

    def __init__(
        self,
        host: RealRobotHost,
        *,
        spec: dict,
        rate_hz: float = 50.0,
        dex3_driver: Any = None,
        controller: Any = None,
        on_tick=None,
        on_episode_command=None,
    ):
        self.host = host
        self.session = host.session
        self.rate_hz = float(rate_hz)
        self.dex3_driver = dex3_driver
        host._dex3_driver = dex3_driver
        self.controller = controller
        #: True when `controller` reads THIS terminal (sim2real's KeyboardController)
        #: rather than a separate pane publishing over ZMQ.
        self._keys_are_local = getattr(controller, "name", "") != "zmq-keyboard"
        self.on_tick = on_tick
        self.on_episode_command = on_episode_command
        self._armed = False
        self._overruns = 0
        self._ticks = 0
        # Operator flow, ported from LocoManiLab's original HeFT integration (src/deploy.py): hold zero
        # torque until the operator confirms once, then ramp to the default pose
        # by itself, then wait for the key that hands the robot to the policy. One
        # confirmation and one arming key -- not a sequence to memorise.
        self._last_skip_warn = 0.0
        self._last_prompt = 0.0
        self._last_refuse = 0.0
        self._stdin = None
        self._stdin_settings = None
        self._armed_at = None
        self._tick_costs: list[float] = []
        self._over_budget = 0
        self._last_cost_report = 0.0
        self._stage = "confirm"          # confirm -> ramping -> ready -> armed
        self._ramp_started_tick = None
        self._prompted = False
        # Builds the backend, the policy runtime, the PICO source and (for the
        # GMR families) the retarget subprocess. Everything expensive happens
        # here, so it must run before the loop's first deadline.
        self.session.configure(spec)
        source = self.session.pico_source
        logger.info(
            "PICO source: {} — on the GMR backends the buttons come from the "
            "retarget publisher's ZMQ sockets (5592/5593), not from a second SDK "
            "connection in this process. `python -m g1_bridge.pico_probe` shows "
            "every hop of that chain.",
            type(source).__name__ if source is not None else "none (mock/no operator)",
        )
        self._buttons_seen = False
        self._no_button_warned = False
        self._last_status_pub = 0.0
        try:
            from g1_bridge.teleop_monitor import DEFAULT_STATUS_PORT

            self._status_socket = zmq.Context.instance().socket(zmq.PUB)
            self._status_socket.setsockopt(zmq.LINGER, 0)
            self._status_socket.bind(f"tcp://*:{DEFAULT_STATUS_PORT}")
        except Exception as exc:  # noqa: BLE001 -- a taken port must not stop collection
            logger.debug(f"teleop status port unavailable: {exc}")
            self._status_socket = None

    # ------------------------------------------------------------------ #
    # Control modes. sim2real's own three, driven explicitly instead of by
    # BridgeTracking's constructor (which arms the policy immediately).
    # ------------------------------------------------------------------ #
    @property
    def armed(self) -> bool:
        return self._armed

    def zero(self, source: str = "real_host"):
        """Hold the measured pose. Startup state, and the e-stop."""
        self._armed = False
        if self._stage == "armed":
            # An e-stop drops the operator back to "confirm again before moving",
            # rather than leaving a single keypress between here and a live policy.
            self._stage = "confirm"
            self._prompted = False
        self.session.tracking.set_zero_mode(source=source)
        logger.info("control mode: ZERO (holding measured pose)")

    def init(self, source: str = "real_host"):
        """Ramp to the policy's default stance over sim2real's own init window."""
        self._armed = False
        self._stage = "ramping"
        self._ramp_started_tick = self._ticks
        self._prompted = False
        self.session.tracking.set_init_mode(source=source)
        logger.info("control mode: INIT (ramping to the default stance)")

    #: Joint speed under which the robot counts as standing still, rad/s. Same
    #: value the sim host's settling window uses (``_DEFAULT_SETTLE_QVEL``).
    _ARM_QVEL_LIMIT = 2.0

    def arm(self):
        """Hand the robot to the policy, calibrating the one-shot yaw alignment.

        The order is load-bearing and mirrors what the sim host does per episode
        (``TeleopSession.reset`` then ``G1BridgeComposite._begin_settling``):

        1. ``session.reset()`` -- resets the motion buffer's STREAM, primes the
           state processor with the robot's real pose, then resets Tracking. The
           stream reset is the part that matters here: the buffer's segment
           transform is what maps the operator's frame onto the robot's, and a
           transform left over from a previous arming silently rotates this
           session's reference. That is felt as "the robot starts off yawed", and
           it is exactly the failure ``TeleopSession.reset``'s own docstring
           describes for a second episode in sim.
        2. freeze the heading the reference will be built at,
        3. push a standing frame at that heading,
        4. calibrate against exactly that pair (``set_policy_mode`` -> ``reset``),
        5. seed the episode's reference.

        sim2real makes the alignment offset permanent for the episode and forbids
        recomputing it later (``docs/policy_yaw_alignment.md``), so this happens
        once, here -- which is also why it must happen with the robot standing
        still: see ``_request_arm``.
        """
        self.session.reset()
        quat, _ = self.host.read_pelvis_state()
        self.session.set_anchor_heading(quat)
        self.session.push_standing_frame()
        self.session.calibrate_alignment()
        qpos, qvel = self.host.read_joint_state()
        self.session.seed_reference(
            TickInput(
                qpos=qpos,
                qvel=qvel,
                sim_time=time.monotonic(),
                mode=self.session.mode_manager.current,
                buttons=None,
                sample=None,
            )
        )
        self._armed = True
        self._armed_at = time.monotonic()
        self._stage = "armed"
        logger.info("control mode: POLICY (armed) — the robot is now tracking")

    # ------------------------------------------------------------------ #
    # Loop
    # ------------------------------------------------------------------ #
    def wait_for_state(self, timeout_s: float = 10.0) -> bool:
        return self.host.poll_next(timeout_s=timeout_s)

    def tick(self, fresh: bool = True) -> None:
        """One control tick: state -> policy -> command -> (recording hook).

        ``fresh`` says whether this tick has a state packet of its own; the loop
        below only ever calls it with one.
        """
        if not self.host.ready:
            return
        if self.host.stale and self._armed:
            logger.error(
                "robot state stale for {} ticks — dropping to ZERO", _STALE_TICK_LIMIT
            )
            self.zero()

        self._drain_operator_input()
        self._advance_stage()

        sample = self.session.sample_pico()
        mode, buttons = self.session.begin_tick(sample, settling=False)
        self._watch_buttons(buttons)
        self._publish_teleop_status(mode, buttons, sample)
        self._arm_on_engage_chord(mode)

        qpos, qvel = self.host.read_joint_state()
        tick = TickInput(
            qpos=qpos,
            qvel=qvel,
            # Wall clock, not sim time: on hardware they are the same thing, and
            # every rate limiter downstream (blend ramps, stream-health checks)
            # is written against seconds that actually elapse.
            sim_time=time.monotonic(),
            mode=mode,
            buttons=buttons,
            sample=sample,
        )
        target_q, kps, kds = self.session.step_backend(tick)
        self.session.cache_command(target_q, kps, kds, tick.sim_time)
        self._write(target_q, kps, kds)

        if self.dex3_driver is not None:
            self.dex3_driver.apply_grips(
                self.session.left_trigger, self.session.right_trigger
            )
        self._ticks += 1
        if self.on_tick is not None:
            self.on_tick(self, fresh)

    def _write(self, target_q, kps, kds):
        """Forward the runtime's own command to the robot.

        ``q_target``/``kp``/``kd`` come from the backend's return value, not from
        ``last_command``: amo overwrites the arm and wrist slots *after* the
        policy step, and those edits would be dropped otherwise. ``dq_target`` and
        ``tau_ff`` are whatever the runtime's CommandSender produced this tick.
        """
        last = self.session.io.last_command
        n = len(target_q)
        dq = last[1] if last is not None else np.zeros(n, dtype=np.float32)
        tau = last[2] if last is not None else np.zeros(n, dtype=np.float32)
        self.host.io.write_command(
            q_target=np.asarray(target_q, dtype=np.float32),
            dq_target=np.asarray(dq, dtype=np.float32),
            tau_ff=np.asarray(tau, dtype=np.float32),
            kp=np.asarray(kps, dtype=np.float32),
            kd=np.asarray(kds, dtype=np.float32),
        )

    #: Ticks the runtime's init ramp takes (BasePolicy interpolates over 500 of
    #: them). Used only to know when to tell the operator the robot has arrived.
    _RAMP_TICKS = 500

    #: How often an unanswered prompt is re-printed, seconds. The controller pane
    #: keeps logging underneath it, so a question asked once scrolls away and the
    #: operator is left looking at a bare cursor -- heft's own gate re-prints for
    #: exactly this reason (``deploy.zero_torque_state``).
    _PROMPT_REPEAT_S = 3.0

    def _prompt(self):
        """Say what the robot is waiting for, and keep saying it until answered."""
        now = time.monotonic()
        if self._prompted and now - self._last_prompt < self._PROMPT_REPEAT_S:
            return
        self._prompted = True
        self._last_prompt = now
        if self._stage == "confirm":
            print(
                colored(
                    "  Confirm start? Robot will move from zero torque to the "
                    "default pose.  [y/Enter = yes, q = quit]  (type here)",
                    "yellow",
                ),
                flush=True,
            )
        elif self._stage == "ready":
            print(
                colored(
                    "  Robot is standing in the default pose. Press the PICO chord "
                    "A+B+X+Y (or 'a' here) to enter the tracking policy. "
                    "'o' = zero torque / e-stop.",
                    "green",
                ),
                flush=True,
            )
        else:
            self._prompted = False   # nothing to ask in this stage

    def _drain_operator_input(self):
        """The operator keys, in the heft fork's vocabulary.

        ==========  ====================================================
        y / Enter   confirm the startup gate: ramp to the default pose
        a           enter the tracking policy (arm) -- only once standing
        c / Enter   start / stop+save an episode (after arming)
        x           discard the episode in progress
        o           zero torque (e-stop); re-arms the [Y/n] gate
        q           quit (drops to zero torque first)
        e / p       ModeManager chord fallbacks (engage / POSE)
        i / ]       sim2real's own init / policy keys, kept as aliases
        ==========  ====================================================

        Enter means "confirm" before the gate is passed and "toggle recording"
        after, because that is when each of them exists -- the gate happens once,
        at startup, and recording only after arming.

        The full PICO engage chord may also arm once the robot is ready; see
        ``_arm_on_engage_chord``. A bare ``A`` or ``A+B`` does not arm, because
        those are intermediate button states while pressing ``A+B+X+Y``.
        Recording is keyboard-only on this host: PICO ``Y`` and ``Y+A`` do not
        start or discard an episode.
        """
        for key in self._read_own_stdin():
            # This terminal is the controller's own: every key works here, which
            # is also where the prompts that ask for them are printed.
            self._handle_key(key)
        if self.controller is None:
            return
        if self._keys_are_local:
            # A local keyboard controller IS this terminal (sim2real's own stdin
            # listener), so it gets the same full set.
            mode = self.controller.get_control_mode()
            if mode == "zero":
                self.zero()
            elif mode == "init":
                self.init()
            elif mode == "policy":
                self._request_arm()
        for key in self.controller.get_extra_keys():
            self._handle_key(key, record_only=not self._keys_are_local)

    def _read_own_stdin(self) -> list[str]:
        """Keys typed into THIS process's terminal (the controller pane).

        cbreak, non-blocking: the control loop must never wait on a keypress. Set
        up once, lazily, and only when stdin is a terminal -- under tmux
        send-keys, a pipe, or a service manager it simply yields nothing.
        """
        if self._stdin is False:
            return []
        if self._stdin is None:
            try:
                if not sys.stdin.isatty():
                    self._stdin = False
                    return []
                self._stdin_settings = termios.tcgetattr(sys.stdin.fileno())
                tty.setcbreak(sys.stdin.fileno())
                self._stdin = sys.stdin
            except Exception:  # noqa: BLE001 -- no terminal, no local keys
                self._stdin = False
                return []
        keys = []
        while select.select([self._stdin], [], [], 0)[0]:
            ch = self._stdin.read(1)
            if not ch:
                break
            keys.append("enter" if ch in ("\n", "\r") else ch.lower())
        return keys

    def _restore_stdin(self) -> None:
        if self._stdin and self._stdin_settings is not None:
            try:
                termios.tcsetattr(
                    sys.stdin.fileno(), termios.TCSADRAIN, self._stdin_settings
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"could not restore terminal settings: {exc}")

    def _handle_key(self, key: str, *, record_only: bool = False) -> None:
        """One operator key.

        ``record_only`` is what the keyboard PANE gets: it publishes over a socket
        any process on this machine could bind to, so it is allowed to start,
        stop and discard episodes and nothing else. Everything that moves the
        robot -- the startup gate, arming, zero torque, quitting -- is accepted
        only from this process's own terminal, which is where the prompts asking
        for those keys are printed.
        """
        from sim2real.rl_policy.control_mode import resolve_keyboard_episode_command

        key = key.lower()
        if record_only:
            command = resolve_keyboard_episode_command(key)
            if command is None and key == "enter":
                command = "toggle"
            if command is not None and self.on_episode_command is not None:
                self.on_episode_command(command)
            elif command is None:
                logger.debug(f"ignoring {key!r} from the keyboard pane (record keys only)")
            return
        if key == "q":
            raise KeyboardInterrupt
        if key in ("y", "s") or (key == "enter" and self._stage == "confirm"):
            if self._stage == "confirm":
                print(colored("  Confirmed -- moving to the default pose.", "green"),
                      flush=True)
                self.init()
            return
        if key == "a":
            self._request_arm()
            return
        if key == "o":
            self.zero()
            return
        if key == "i":
            self.init()
            return
        if key in ("e", "p"):
            # The session's own keyboard fallbacks for the two PICO chords
            # (OFF<->engage, POSE toggle). It expects the uppercase keycode,
            # because in sim these arrive from the MuJoCo viewer.
            self.session.handle_keypress(ord(key.upper()))
            return
        command = resolve_keyboard_episode_command(key)
        if command is None and key == "enter":
            command = "toggle"          # heft's keyboard_control binds ENTER to 'c'
        if command is not None and self.on_episode_command is not None:
            self.on_episode_command(command)

    def _request_arm(self):
        """Arm, but only from a robot that is actually standing where it should be.

        Refusing early is the point: arming calibrates the episode's one-shot yaw
        alignment against the pose the robot is in, so arming mid-ramp bakes a
        half-way pose into the whole session.
        """
        if self._stage == "armed":
            return
        if self._stage != "ready":
            self._refuse_arm("arming")
            return
        _qpos, qvel = self.host.read_joint_state()
        speed = float(np.abs(np.asarray(qvel)).max())
        if speed > self._ARM_QVEL_LIMIT:
            # The yaw alignment is computed from the pose the robot is in at this
            # instant and then frozen for the session, so arming a robot that is
            # still moving freezes a pose it is about to leave. The sim host waits
            # for the same quiet condition before handing over (``_settled``).
            print(
                colored(
                    f"  not yet -- the robot is still moving ({speed:.2f} rad/s > "
                    f"{self._ARM_QVEL_LIMIT} rad/s). Let it settle, then press 'a' again.",
                    "yellow",
                ),
                flush=True,
            )
            return
        self.arm()

    def _refuse_arm(self, what: str) -> None:
        now = time.monotonic()
        if now - self._last_refuse < 2.0:
            return
        self._last_refuse = now
        print(
            colored(
                f"  {what} ignored -- "
                + (
                    "confirm with 'y' first (the robot is still at zero torque)."
                    if self._stage == "confirm"
                    else "still ramping to the default pose; wait for the prompt."
                ),
                "yellow",
            ),
            flush=True,
        )

    def _advance_stage(self):
        """Ramp -> ready, once the runtime's init interpolation has finished."""
        if self._stage == "ramping" and self._ramp_started_tick is not None:
            if self._ticks - self._ramp_started_tick >= self._RAMP_TICKS:
                self._stage = "ready"
                self._prompted = False
        self._prompt()

    def _arm_on_engage_chord(self, mode) -> None:
        """The PICO chord enters the tracking policy, as it does in the heft fork.

        There, ``A+B+X+Y`` (or the remote's A) is what takes the robot from
        "standing in the default pose" into the policy; here the same chord is
        ModeManager's engage. Treat the first engage as both: arm, then let the
        engage stand. Only from ``ready`` -- a chord pressed while the robot is
        still at zero torque or mid-ramp is refused with the same message the
        keyboard path gives, because the yaw alignment is frozen from the pose at
        this instant.

        A bare ``A`` or ``A+B`` is still NOT wired to anything: those are states
        the operator passes through on the way into the four-button chord, so
        binding them would arm the robot by accident. The chord is deliberate.
        """
        from g1_bridge.pico.mode_manager import ManagerMode

        if self._armed or mode == ManagerMode.OFF:
            return
        if self._stage != "ready":
            self._refuse_arm("the PICO chord")
            return
        print(colored("  PICO chord — entering the tracking policy", "green"), flush=True)
        self._request_arm()

    def _publish_teleop_status(self, mode, buttons, sample) -> None:
        """Publish what the operator's input looks like from in here.

        sonic and amo open their PICO connection inside this process (a
        multiprocessing child), so nothing outside can observe it -- which leaves
        "the chord does nothing" with no middle to inspect. Publishing the view
        the ModeManager actually gets gives the teleop pane the same job on every
        backend. Status only: nothing subscribes to this to control anything.

        The stick axes are here for the same reason the buttons are, one failure
        later: on the GMR backends both come through ``GMRButtonRelay``, and that
        channel once carried the buttons but not the axes -- chords worked, arms
        followed, and the sticks silently did nothing (see
        ``PicoControllerStateMessage`` and tests/test_pico_controller_axes.py).
        That is invisible on heft/scalebfm, which do not steer, and total on
        almi, whose entire locomotion command is the two sticks. In sim the HUD
        shows them; on hardware this pane is the only place they can be seen.
        """
        if self._status_socket is None:
            return
        now = time.monotonic()
        if now - self._last_status_pub < 0.1:
            return
        self._last_status_pub = now
        stamp = (sample or {}).get("timestamp")
        payload = {
            "source": type(self.session.pico_source).__name__
            if self.session.pico_source is not None
            else "none",
            "buttons": {k: bool((buttons or {}).get(k)) for k in ("a", "b", "x", "y")},
            # None (not zeros) when the source never published an axis at all --
            # "the relay is not sending sticks" and "the operator's thumb is at
            # rest" are exactly the two cases this pane exists to tell apart.
            "left_axis": _axis_or_none(buttons, "left_axis"),
            "right_axis": _axis_or_none(buttons, "right_axis"),
            "left_trigger": float(self.session.left_trigger),
            "right_trigger": float(self.session.right_trigger),
            "mode": getattr(mode, "name", str(mode)),
            "armed": bool(self._armed),
            "sample_age_s": (now - stamp) if isinstance(stamp, (int, float)) else None,
        }
        try:
            self._status_socket.send_string(json.dumps(payload), zmq.NOBLOCK)
        except Exception as exc:  # noqa: BLE001 -- status must never break control
            logger.debug(f"teleop status publish failed: {exc}")

    def _watch_buttons(self, buttons) -> None:
        """Say something when the operator's buttons never arrive.

        Silence here is indistinguishable from "the operator has not pressed
        anything yet", and the chord is the only way into teleop -- so an armed
        robot that has never seen a single button edge is worth naming, once,
        with the one command that shows which hop is dead.
        """
        if self._buttons_seen or self._no_button_warned:
            return
        if buttons and any(bool(buttons.get(k)) for k in ("a", "b", "x", "y")):
            self._buttons_seen = True
            logger.info("PICO buttons are reaching the collector")
            return
        if not self._armed or self._armed_at is None:
            return
        if time.monotonic() - self._armed_at < 20.0:
            return
        self._no_button_warned = True
        logger.warning(
            "armed for 20 s and no PICO button has been seen yet. The chord "
            "(A+B+X+Y) travels headset -> XRoboToolkit service -> GMR publisher "
            "-> ZMQ 5592 -> this process; run `python -m g1_bridge.pico_probe` "
            "in the spare pane to see which hop is dead. Keyboard 'e' engages "
            "and 'p' toggles POSE meanwhile."
        )

    # Recording is KEYBOARD-ONLY on this host: 'c' toggles, 'x' discards, both
    # through _drain_operator_input below.
    #
    # There used to be a PICO path here as well ('Y' tap = toggle, 'Y+A' =
    # discard, mirroring sim2real's own --controller pico). It is gone because a
    # bare rising edge on one button is the wrong trigger for the one action in
    # this loop that silently changes what lands on disk. The operator's hands
    # are on the sticks for the whole session and Y sits next to the A+B+X+Y
    # engage chord; a brush against it starts a recording with nothing but a
    # green line in a scrolling log to say so, and a second brush saves it. That
    # is not hypothetical -- it happened on an amo run (2026-08-12). Y was chosen
    # originally *because* it was the only free bit on the message, which is the
    # same thing as saying nothing else protects it.
    #
    # Keyboard 'c'/'x' are deliberate by construction: they need a hand off the
    # sticks and eyes on the terminal that prints the result. sim2real's
    # standalone PicoController (rl_policy/controllers/pico.py) still has its own
    # Y binding -- that is a different entry point, reached only via
    # base_policy.py's --controller pico, never by this collector.
    def run(self, max_ticks: int | None = None, *, zero_on_exit: bool = True,
            state_timeout_s: float = 1.0):
        """One policy step per state packet -- the loop LocoManiLab's original HeFT integration runs.

        Not a fixed-rate wall-clock loop: the control rate IS the rate the robot
        publishes state at (its bridge decimates 1 kHz down to the 50 Hz the
        policies were trained at), so stepping on the packet keeps observation and
        command on the same clock. A free-running loop instead beats against that
        clock -- re-using a state here, skipping one there -- which on hardware is
        felt as joint shake and is the first suspect the repo's own robot_io doc
        names. Run the bridge at the control rate (the launcher does) and this
        paces itself.

        ``max_ticks`` counts from this call, so a caller can run the INIT ramp for
        a fixed number of ticks and then continue with the real budget.
        ``zero_on_exit=False`` keeps the current control mode on return -- only for
        such staged runs; the final one must leave the robot in zero.
        """
        started_at = self._ticks
        skipped_before = self.host.skipped
        period = 1.0 / self.rate_hz
        last_step = 0.0
        try:
            while max_ticks is None or (self._ticks - started_at) < max_ticks:
                if not self.host.poll_next(timeout_s=state_timeout_s):
                    logger.warning(
                        "no robot state for {}s — is the bridge publishing?",
                        state_timeout_s,
                    )
                    if self._armed:
                        self.zero()
                    continue
                # Rate guard. With the bridge at the control rate (what the
                # launcher starts) every packet is a step and this never fires.
                # It exists for the hosts that publish faster -- sim2sim's
                # base_sim runs at 200 Hz -- where stepping per packet would run
                # the policy at 4x the rate it was trained at. Keeping the guard
                # here rather than sleeping to a deadline preserves the property
                # that matters: the step always acts on the freshest state.
                now = time.monotonic()
                if now - last_step < period * 0.9:
                    continue
                last_step = now
                t0 = time.perf_counter()
                self.tick(fresh=True)
                self._record_tick_cost(time.perf_counter() - t0)
                if self.host.skipped != skipped_before:
                    # heft's controller prints the same thing, and for the same
                    # reason: a dropped packet is a tick the policy never saw, and
                    # it shows up as a twitch rather than as an error.
                    if now - self._last_skip_warn > 5.0:
                        logger.warning(
                            "skipped {} state packet(s) since the last report — the "
                            "bridge or this loop is not keeping up",
                            self.host.skipped - skipped_before,
                        )
                        self._last_skip_warn = now
                        skipped_before = self.host.skipped
        except KeyboardInterrupt:
            zero_on_exit = True
        finally:
            if zero_on_exit:
                self.zero()

    def _record_tick_cost(self, seconds: float) -> None:
        """Track how long a tick takes, and say so when it stops fitting.

        The budget is the control period: policy inference, the reference push and
        whatever the recording hook does all have to finish inside it, or the next
        state packet is already waiting when this one returns and the command
        stream stutters. On hardware a stuttering command stream is felt as joint
        shake, which is why this is measured rather than assumed -- LocoManiLab's heft
        fork sidesteps the question entirely by recording in a separate process.
        """
        self._tick_costs.append(seconds)
        budget = 1.0 / self.rate_hz
        if seconds > budget:
            self._over_budget += 1
        now = time.monotonic()
        if now - self._last_cost_report < 10.0 or len(self._tick_costs) < 50:
            return
        self._last_cost_report = now
        costs = sorted(self._tick_costs)
        message = (
            f"tick cost ms: p50={costs[len(costs)//2]*1000:.1f} "
            f"p95={costs[int(len(costs)*0.95)]*1000:.1f} max={costs[-1]*1000:.1f} "
            f"(budget {budget*1000:.0f}) over-budget {self._over_budget}/{len(costs)}"
        )
        (logger.warning if self._over_budget else logger.info)(message)
        self._tick_costs.clear()
        self._over_budget = 0

    @property
    def stats(self) -> dict:
        return {
            "ticks": self._ticks,
            "overruns": self._overruns,
            "skipped_state_packets": self.host.skipped,
        }

    def close(self):
        self._restore_stdin()
        if self._status_socket is not None:
            self._status_socket.close(0)
        try:
            self.zero()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"could not drop to zero mode on close: {exc}")
        if self.controller is not None:
            self.controller.close()
        if self.dex3_driver is not None and hasattr(self.dex3_driver, "close"):
            self.dex3_driver.close()
        pico = self.session.pico_source
        if pico is not None and hasattr(pico, "stop"):
            pico.stop()
        self.host.close()
