"""Real-robot teleop data collection -- the hardware twin of the sim collector.

    uv run python -m g1_bridge.collect_real --backend heft \\
        --pico-source real --robot-io zmq --camera-connect tcp://127.0.0.1:5594 \\
        --task-prompt "pick up the cup" --out /abs/path/to/dataset

Same backends, same reference/blend/mode-manager path, same LeRobot schema as
``robocasa.scripts.collect_demos_g1`` -- the collection arguments are literally
the same function (``g1_bridge.recording.cli``), and the per-tick sample is the
same ``session.teleop_sample()``. What differs is only what a host can observe:
the ego frame comes from a camera server instead of a renderer, the root
orientation from the IMU instead of ``robot0_base_quat``, and the sim-only
reward/done columns (``next.reward``/``next.done``) are not recorded.
For SONIC, ``observation.eef_state`` is computed by the same pelvis-frame
forward kinematics on both hosts (see ``recording/g1_fk.py``).

Processes this expects to be running alongside it (heft/scalebfm/almi):

* the GMR retarget publisher -- started automatically by the backend, from
  ``<repo>/venv/pico/.venv``; publishes motion on 28701, PICO buttons **and
  stick axes** on 5592 and hand grips on 5593. It owns the only xrobotoolkit
  connection, so on these backends every scrap of operator input comes through
  it -- including almi's locomotion sticks.
* ``python -m sim2real.camera.server --cameras.ego_view.source realsense --fps 50``
* with ``--robot-io zmq`` (the default): ``python scripts/g1/real_bridge.py``.
  With ``--robot-io inline`` this process talks DDS itself -- note the repo's own
  warning that combining inline with Dex3 control is not validated for
  DDS-participant conflicts.

Operator flow (see also the banner this prints at startup):

1. The robot starts at zero torque. Confirm with ``y``/Enter in the collector
   terminal to ramp to the policy's default stance.
2. Once standing, ``a`` or the PICO chord A+B+X+Y arms the tracking policy
   and captures its one-shot yaw alignment. Early or moving-state arming is
   refused; a bare PICO A or A+B does not arm this host.
3. A+B+X+Y toggles engagement; A+X toggles POSE. Keyboard fallbacks in the
   collector terminal are ``e`` and ``p``.
4. ``c``/Enter starts or stops and saves an episode; ``x`` discards it.
   Recording is keyboard-only. The separate keyboard pane accepts only these
   recording keys; startup confirmation and arming use the collector terminal.
5. ``o`` returns to zero torque and resets the confirmation gate; ``q`` quits.

"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import zmq
from loguru import logger
from termcolor import colored

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import

from g1_bridge.backends import backend_names
from g1_bridge.config import build_bridge_spec
from g1_bridge.modality import layout_for
from g1_bridge.recording.cli import add_shared_collection_args
from g1_bridge.recording.dataset import make_dataset, timestamped_out_dir
from g1_bridge.recording.episode import OperatorEpisodeRecorder
from g1_bridge.recording.worker import RecordingWorker
from g1_bridge.recording.features import (
    EGO_H,
    EGO_W,
    LEGACY_MODALITY_ASSET,
    REAL_EGO_H,
    REAL_EGO_W,
)
from g1_bridge.recording.schema import ModalityLayout
from g1_bridge.recording.frames import build_blob_frame, build_legacy_frame, sonic_state_43

#: Every backend the bridge knows. What differs on hardware is only what the
#: robot can observe:
#:
#: * heft/scalebfm -- nothing missing.
#: * sonic -- nothing missing either: its finger joints come from the Dex3
#:   driver's DDS feedback instead of the sim's qpos (read_hand_state).
#: * amo -- nothing missing since schema v7: its rpy/height STATE is the
#:   previous tick's torso command (Psi0's definition), not a measurement, so
#:   the real layout is the sim layout (32-wide state + root_orientation /
#:   projected_gravity from the IMU). Its reset-time joint teleport degrades to
#:   the operator's own INIT ramp.
#: * almi -- nothing missing, which is worth stating because it reads a root
#:   quaternion and a base height and neither is a hardware gap: the quaternion
#:   is the IMU's, and the height is a *command* the operator dials in, never a
#:   measurement. Its EE targets come out of the GMR payload (already FK'd by the
#:   publisher), so it needs no world body poses either. See RealRobotHost.
#:   One real-only behaviour worth knowing: ``session.reset()`` happens once, at
#:   arm time, not per episode. Everything a dataset defines per episode is
#:   therefore re-taken in ``on_episode_start`` (wired below to the recorder's
#:   episode toggle): amo's heading anchor and torso integrators, almi's
#:   recording gate and base height, heft's reference origin.
#:
#: Derived from the backend registry so a new backend is not silently
#: rejected here after being registered there.
SUPPORTED_BACKENDS = backend_names()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g1_bridge.collect_real",
        description="Teleoperate a real G1 and record a LeRobot dataset.",
    )
    # No episode cap by default: a hardware session ends when the operator
    # says so, not after an arbitrary count.
    add_shared_collection_args(parser, max_episodes=0)
    parser.add_argument(
        "--robot", type=str, default="g1", help="sim2real robot cfg name."
    )
    parser.add_argument(
        "--robot-io",
        type=str,
        default="zmq",
        choices=["zmq", "inline"],
        help="'zmq' (default) talks to scripts/g1/real_bridge.py over "
        "tcp://*:5590/5591, keeping DDS out of this process -- also the mode "
        "sim2real documents as known-safe with Dex3 hand control, and the one "
        "that lets you smoke-test against sim_env/base_sim.py. 'inline' opens "
        "the robot's DDS here (lower jitter, upstream's default for deploy).",
    )
    parser.add_argument(
        "--robot-interface",
        type=str,
        default="eth0",
        help="network interface for --robot-io inline.",
    )
    parser.add_argument(
        "--camera-connect",
        type=str,
        default="tcp://127.0.0.1:5594",
        help="sim2real.camera.server address. Images are required: a dataset "
        "with no ego view trains nothing.",
    )
    parser.add_argument(
        "--ego-size",
        type=str,
        default=f"{REAL_EGO_W}x{REAL_EGO_H}",
        help=f"WxH to store the ego frames at (default {REAL_EGO_W}x{REAL_EGO_H}, "
        "the G1 camera's own resolution), or 'native' to follow whatever the "
        f"camera sends. The simulator renders {EGO_W}x{EGO_H}; both are 4:3, so a "
        "model that resizes sees the same picture -- but a config that CROPS at a "
        "fixed pixel size (LocoManiLab's dp.yaml takes a 216x288 crop) frames a much "
        "smaller part of a 640x480 image, so check that before mixing the two.",
    )
    parser.add_argument(
        "--camera-timeout-s",
        type=float,
        default=30.0,
        help="how long to wait for the first camera frame before giving up.",
    )
    parser.add_argument(
        "--dex3",
        type=str,
        default="double",
        choices=["none", "left", "right", "double"],
        help="in-process Dex3 hand control, driven by the same PICO triggers "
        "the dataset records as action.hands.",
    )
    parser.add_argument(
        "--dex3-dry-run",
        action="store_true",
        help="map grips to joint targets without touching unitree_sdk2py/DDS "
        "(dev machines, sim2sim).",
    )
    parser.add_argument("--dex3-domain-id", type=int, default=None)
    parser.add_argument("--dex3-interface", type=str, default=None)
    parser.add_argument(
        "--keyboard",
        default="local",
        choices=["local", "zmq", "none"],
        help="where the operator keys come from. 'local' reads this process's own "
        "stdin; 'zmq' subscribes to g1_bridge.keyboard_control running in its own "
        "pane (what the tmux launcher uses, so the keys have a window of their own "
        "and the focus never has to move); 'none' is PICO-only.",
    )
    parser.add_argument(
        "--keyboard-connect",
        default=None,
        help="--keyboard zmq: address of the key publisher "
        "(default tcp://127.0.0.1:5580).",
    )
    parser.add_argument(
        "--no-keyboard",
        action="store_true",
        help="alias for --keyboard none.",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override one bridge-spec key (repeatable), e.g. "
        "--set sim2real_pose_blend_s=1.0 or --set sim2real_record_idle=true. "
        "These are the same keys the controller JSONs carry; values are parsed "
        "as JSON when possible, else kept as strings.",
    )
    parser.add_argument(
        "--arm-on-start",
        action="store_true",
        help="arm the policy as soon as state arrives. Debug/sim2sim only -- on "
        "hardware, confirm with 'y'/Enter and arm with 'a' or the PICO engage "
        "chord once the robot is standing.",
    )
    parser.add_argument(
        "--init-seconds",
        type=float,
        default=0.0,
        help="with --arm-on-start: hold INIT (ramp to the default stance) for "
        "this long before arming, instead of arming straight away. sim2real's "
        "ramp is 500 ticks / 10 s, so that is the useful value; 0 skips it.",
    )
    return parser


def _parse_ego_size(value: str, native_hw):
    if value.strip().lower() == "native":
        return native_hw
    try:
        w, h = (int(v) for v in value.lower().split("x"))
    except ValueError:
        raise SystemExit(f"--ego-size must be WxH or 'native', got {value!r}")
    return (h, w)


def _parse_overrides(items):
    """``["a=1", "b=true"]`` -> ``{"a": 1, "b": True}`` (JSON values, else strings)."""
    import json

    out = {}
    for item in items or []:
        if "=" not in item:
            raise SystemExit(f"--set expects KEY=VALUE, got {item!r}")
        key, _, value = item.partition("=")
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def _resize(image, target_hw):
    if image.shape[:2] == tuple(target_hw):
        return image
    import cv2

    h, w = target_hw
    return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)


def _wait_for_first_frame(camera, timeout_s: float):
    """Block until the camera server sends something. Returns (name, frame)."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        message = camera.read()
        images = (message or {}).get("images") or {}
        if images:
            name = "ego_view" if "ego_view" in images else sorted(images)[0]
            return name, images[name]
        time.sleep(0.05)
    raise SystemExit(
        "no camera frame arrived. Start the server first:\n"
        "  uv run python -m sim2real.camera.server "
        "--cameras.ego_view.source realsense --fps 50"
    )


def _network_interfaces() -> list[str]:
    """The machine's NIC names, for error messages that name the alternatives."""
    import socket

    try:
        return [name for _, name in socket.if_nameindex() if name != "lo"]
    except Exception:  # noqa: BLE001
        return []


def _build_dex3(args):
    if args.dex3 == "none":
        return None
    from sim2real.utils.dex3 import DryRunDex3HandDriver, Dex3HandDriver

    if args.dex3_dry_run:
        return DryRunDex3HandDriver(args.dex3)
    # hand_dir is keyword-ONLY on the real driver (and positional on the dry-run
    # one, which is why passing it positionally to both looked fine until a real
    # hand was configured).
    kwargs = {"hand_dir": args.dex3}
    if args.dex3_domain_id is not None:
        kwargs["domain_id"] = args.dex3_domain_id
    # The hand is on the robot's own DDS network -- the same wire the arms and legs
    # are on -- so it follows --robot-interface unless told otherwise. The driver's
    # own default is "eth0", which is right onboard the G1 and wrong on every
    # workstation, where it surfaces as an opaque "channel factory init error".
    interface = args.dex3_interface or args.robot_interface
    kwargs["interface"] = interface
    if args.robot_io == "inline":
        logger.warning(
            "--dex3 with --robot-io inline is NOT validated upstream for "
            "DDS-participant conflicts with the debug-mode handshake; the "
            "known-safe combination is --robot-io zmq."
        )
    try:
        return Dex3HandDriver(**kwargs)
    except Exception as exc:  # noqa: BLE001 -- the SDK raises a bare Exception
        available = ", ".join(_network_interfaces()) or "none found"
        raise SystemExit(
            f"could not open the Dex3 hand on DDS interface {interface!r}: {exc}\n"
            f"  interfaces on this machine: {available}\n"
            f"  --dex3-interface <nic> sets it (it follows --robot-interface by "
            f"default), --dex3-dry-run maps grips without touching DDS, and "
            f"--dex3 none runs without hands."
        ) from exc


def _banner(args, out_dir, keyboard_where='local'):
    print(
        colored(
            "\n".join(
                [
                    "",
                    "=" * 72,
                    f"  real-robot collection — backend={args.backend} "
                    f"robot_io={args.robot_io} dex3={args.dex3} "
                    f"max_episodes={args.max_episodes or 'unlimited'}",
                    f"  dataset: {out_dir}",
                    "=" * 72,
                    "  robot keys — THIS terminal only:",
                    "    y / Enter  confirm the startup gate -> ramp to default pose",
                    "    a          enter the tracking policy (arm)",
                    "    o          zero torque (e-stop; re-arms the [y/n] gate)",
                    "    q          quit          e / p  engage / POSE fallbacks",
                    f"  recording keys — {keyboard_where}:",
                    "    c / Enter  start -> stop+save an episode",
                    "    x          discard the episode in progress",
                    "  PICO (same as in sim):",
                    "    A+B+X+Y  engage <-> OFF        A+X  POSE (follow me) toggle",
                    "    triggers hand grip (recorded as action.hands)",
                    "  NOTE arming AND recording are keyboard-only: a bare PICO",
                    "       A/A+B would fire on the way into the A+B+X+Y chord,",
                    "       and a brushed Y used to start a recording silently.",
                    "=" * 72,
                    "",
                ]
            ),
            "cyan",
        ),
        flush=True,
    )


def _exit_on_signals() -> None:
    """Turn SIGTERM/SIGHUP into KeyboardInterrupt so cleanup actually runs.

    Python's default action for both is to die immediately -- no ``finally``, no
    ``atexit``. That is how a killed tmux session used to leave the GMR publisher
    holding port 28701: the collector never got to call its stop(). Raising
    instead runs the same path Ctrl-C does.
    """
    import signal

    def _raise(signum, _frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        try:
            signal.signal(sig, _raise)
        except (ValueError, OSError):
            pass


def main(argv=None) -> int:
    _exit_on_signals()
    args = build_parser().parse_args(argv)
    if args.backend not in SUPPORTED_BACKENDS:
        raise SystemExit(
            f"unknown --backend {args.backend} "
            f"(supported: {', '.join(SUPPORTED_BACKENDS)})"
        )
    layout = layout_for(args.backend, real=True)
    if layout is not layout_for(args.backend, real=False):
        logger.info(
            "%s on hardware records a different layout from the sim (see "
            "g1_bridge/modality.py G1_BACKEND_TELEOP_KEYS_REAL).", args.backend
        )

    # Resolve everything that can fail cheaply BEFORE touching hardware or the
    # XRoboToolkit SDK.
    spec = build_bridge_spec(
        args.backend,
        args.pico_source,
        policy_config=args.policy_config,
        gmr_viewer=args.gmr_viewer,
        inference_backend=args.inference_backend,
        amo_arm_source=args.amo_arm_source,
    )
    spec.update(_parse_overrides(args.overrides))
    out_dir = pathlib.Path(
        args.out
        # outputs_real/, not outputs/: the sim collector (tele_humanoid_benchmark's
        # g1_data_collection) writes the identical schema under outputs_sim/, and
        # nothing downstream can tell a hardware episode from a simulated one once
        # the two are in the same tree.
        or timestamped_out_dir(pathlib.Path.cwd() / "outputs_real", "real", args.backend)
    )
    if not out_dir.is_absolute():
        raise SystemExit(f"--out must be an absolute path, got {out_dir}")

    from sim2real.camera.client import CameraClient

    camera = CameraClient(args.camera_connect)
    camera_name, first_frame = _wait_for_first_frame(camera, args.camera_timeout_s)
    ego_hw = _parse_ego_size(args.ego_size, first_frame.shape[:2])
    logger.info(
        "camera '{}' {}x{} -> recording {}x{}",
        camera_name,
        first_frame.shape[1],
        first_frame.shape[0],
        ego_hw[1],
        ego_hw[0],
    )

    # Before anything starts the XRoboToolkit SDK: it leaves C++ streaming
    # threads alive, and importing pyarrow/video native modules afterwards has
    # been observed to segfault (same ordering constraint as the sim collector).
    dataset = make_dataset(
        out_dir,
        layout,
        fps=args.control_freq,
        collection_info={
            "backend": args.backend,
            "pico_source": args.pico_source,
            "policy_config": spec.get("sim2real_policy_config"),
            "robot": "g1-real",
            "robot_io": args.robot_io,
            "dex3": args.dex3,
            "camera": args.camera_connect,
            "ego_size": f"{ego_hw[1]}x{ego_hw[0]}",
        },
        extra_features=None,  # no eef_state / reward / done on hardware
        ego_hw=ego_hw,
        robot_type="G1Real",
        # sonic's legacy per-signal schema ships its modality.json as an asset;
        # the blob-mode backends generate theirs from the layout and ignore this.
        modality_asset=LEGACY_MODALITY_ASSET,
    )
    # Recording status and events go to the recording pane over a socket, not to
    # this terminal: the controller pane is what an operator watches for the
    # robot, and recording noise there scrolls away the line they needed. (The
    # heft fork got this separation for free by running its exporter as its own
    # process; the recorder has to be in-process here, so the OUTPUT is what
    # moves instead.)
    from g1_bridge.recording_monitor import DEFAULT_RECORDING_PORT

    status_pub = zmq.Context.instance().socket(zmq.PUB)
    status_pub.setsockopt(zmq.LINGER, 0)
    try:
        status_pub.bind(f"tcp://*:{DEFAULT_RECORDING_PORT}")
    except Exception as exc:  # noqa: BLE001 -- a taken port must not stop collection
        logger.warning(f"recording status port unavailable ({exc}); logging here instead")
        status_pub = None

    def publish(payload: dict) -> None:
        if status_pub is None:
            if payload.get("type") == "event":
                print(payload["message"], flush=True)
            return
        try:
            status_pub.send_string(json.dumps(payload), zmq.NOBLOCK)
        except Exception:  # noqa: BLE001
            pass

    recorder = OperatorEpisodeRecorder(
        dataset,
        args.task_prompt or "",
        layout,
        max_episodes=args.max_episodes or 10**9,   # 0 = no limit
        on_event=lambda kind, message: publish(
            {"type": "event", "kind": kind, "message": message}
        ),
        # One hardware run is many episodes inside a single backend reset, so
        # this is the ONLY signal a backend gets that a new episode began.
        # Late-bound on purpose: `runner` is built further down, and every
        # reset_episode() comes from the run loop, i.e. strictly after that.
        on_episode_start=lambda: runner.session.on_episode_start(),
    )
    # Everything the writer does -- the first episode's image-writer spin-up, the
    # per-frame PNG staging, and above all save_episode's mp4 encode -- happens on
    # this thread instead of the control loop's. That is what stops the robot
    # lurching when the operator presses 'c'.
    writer = RecordingWorker(
        recorder,
        on_event=lambda kind, message: publish(
            {"type": "event", "kind": kind, "message": message}
        ),
    )

    from sim2real.config.robots import get_robot_cfg
    from sim2real.rl_policy.robot_io.factory import create_robot_io

    from g1_bridge.hosts.real import RealRobotHost, RealRobotRunner

    robot_io = create_robot_io(
        mode=args.robot_io,
        robot_name=args.robot,
        robot_cfg=get_robot_cfg(args.robot),
        interface=args.robot_interface,
    )
    host = RealRobotHost(robot_io, expected_rate_hz=args.control_freq)

    keyboard = "none" if args.no_keyboard else args.keyboard
    if keyboard == "none" and not (args.arm_on_start or args.debug_record_steps):
        raise SystemExit(
            "--keyboard none leaves no way to arm the policy (arming is keyboard-only, "
            "see the module docstring). Use --keyboard local/zmq, or pass "
            "--arm-on-start for an unattended/sim2sim run."
        )
    controller = None
    if keyboard == "local":
        from sim2real.rl_policy.controllers.keyboard import KeyboardController

        controller = KeyboardController()
    elif keyboard == "zmq":
        from g1_bridge.keyboard_control import DEFAULT_KEYBOARD_PORT, ZmqKeyController

        controller = ZmqKeyController(
            args.keyboard_connect or f"tcp://127.0.0.1:{DEFAULT_KEYBOARD_PORT}"
        )

    require_ready = not args.debug_skip_readiness
    state = {"dropped_stale_camera": 0, "last_stamp": None, "repeats": 0,
             "last_status": 0.0}

    def publish_status(runner):
        st = recorder.status
        publish(
            {
                "type": "status",
                "recording": st["recording"],
                "episodes": st["episodes"],
                "frames": st["frames"],
                "armed": bool(runner.armed),
                "queued": writer.pending,
                "saving": writer.saving,
                "dropped": writer.dropped,
                "dataset": str(out_dir),
            }
        )

    def _tick_status(runner, now=None):
        now = now if now is not None else time.monotonic()
        if now - state["last_status"] > 0.5:
            state["last_status"] = now
            publish_status(runner)

    def record(runner, _fresh_state):
        # Status first, and unconditionally: the interesting moments (no camera
        # frame, mode OFF, the writer draining after 'c') are exactly the ones
        # where the frame path below returns early.
        _tick_status(runner)
        message = camera.read() or {}
        images = message.get("images") or {}
        image = images.get(camera_name)
        if image is None:
            state["dropped_stale_camera"] += 1
            return
        stamp = (message.get("timestamps") or {}).get(camera_name)
        if stamp is not None and stamp == state["last_stamp"]:
            state["repeats"] += 1
            if state["repeats"] == 3:
                logger.warning(
                    "the same camera frame has been recorded 3 ticks in a row -- "
                    "start the server with --fps {} to match the control rate.",
                    args.control_freq,
                )
        else:
            state["repeats"] = 0
        state["last_stamp"] = stamp

        from g1_bridge.pico.mode_manager import ManagerMode

        if runner.session.mode_manager.current == ManagerMode.OFF and require_ready:
            # Same gate as the sim: OFF means nothing the operator commanded.
            return
        sample = runner.session.teleop_sample()
        img = _resize(image, ego_hw)
        root_quat = host.read_pelvis_state()[0]
        if isinstance(layout, ModalityLayout):
            frame = build_blob_frame(
                layout, sample, image=img, root_quat=root_quat,
                require_ready=require_ready,
            )
        else:
            # sonic's legacy per-signal schema. Its observation.state is the
            # 43-DoF vector (29 body joints with the 14 finger joints
            # interleaved between the arms) -- assembled by the same shared
            # helper the sim collector uses, off the same two host reads, so the
            # two hosts cannot drift apart on column order again.
            frame = build_legacy_frame(
                layout, sample, image=img, state=sonic_state_43(host),
                root_quat=root_quat, require_ready=require_ready,
            )
        if recorder.recording:
            writer.add_frame(frame)
        _tick_status(runner, now=time.monotonic())

    def on_episode_command(command):
        # The toggle itself is cheap and must be immediate (it decides whether the
        # NEXT frame is recorded); only the writing is deferred.
        if command == "toggle":
            if recorder.recording:
                recorder.recording = False
                if recorder.ep_frames or writer.pending:
                    writer.save("operator stop")
                else:
                    print(colored("[collect] recording stopped — nothing recorded", "yellow"))
            else:
                recorder.recording = True
                recorder.reset_episode()
                print(colored("[collect] recording STARTED", "green"), flush=True)
        elif command == "discard":
            recorder.recording = False
            writer.discard("operator discard")

    runner = RealRobotRunner(
        host,
        spec=spec,
        rate_hz=args.control_freq,
        dex3_driver=_build_dex3(args),
        controller=controller,
        on_tick=record,
        on_episode_command=on_episode_command,
    )

    try:
        if not runner.wait_for_state():
            raise SystemExit(
                "no robot state. With --robot-io zmq, start the bridge:\n"
                "  uv run python scripts/g1/real_bridge.py "
                f"--interface {args.robot_interface} --rate {args.control_freq}\n"
                "  (--rate must be the control rate: this loop steps once per "
                "state packet)"
            )
        runner.zero()
        if args.force_engage:
            from g1_bridge.pico.mode_manager import ManagerMode

            runner.session.mode_manager.force_mode(ManagerMode.POSE)
        if args.arm_on_start or args.debug_record_steps:
            # Unattended path: skip the [y/n] gate the operator would answer.
            runner._stage = "ready"
            if args.init_seconds > 0:
                # The ramp has to be *ticked*: sim2real advances it one step per
                # policy tick, so simply sleeping here would leave the robot in
                # whatever pose it started in and then arm on top of that.
                runner.init()
                runner.run(
                    max_ticks=int(args.init_seconds * args.control_freq),
                    zero_on_exit=False,
                )
            runner.arm()
        _banner(args, out_dir, keyboard_where=(
            'the keyboard-control pane' if keyboard == 'zmq' else keyboard))

        if args.debug_record_steps:
            recorder.toggle()  # open an episode; nothing is operator-driven here
            runner.run(max_ticks=args.debug_record_steps)
            recorder.toggle()  # stop + save
            print(
                colored(
                    f"[collect] debug: plumbing OK, {recorder.n_saved} episode(s), "
                    f"{runner.stats}",
                    "green",
                )
            )
        else:
            runner.run()
    finally:
        writer.close()               # drains, encodes the last episode's video
        recorder.close()
        runner.close()
        if status_pub is not None:
            status_pub.close(0)
        if state["dropped_stale_camera"]:
            logger.info(
                "{} tick(s) had no camera frame and were not recorded",
                state["dropped_stale_camera"],
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
