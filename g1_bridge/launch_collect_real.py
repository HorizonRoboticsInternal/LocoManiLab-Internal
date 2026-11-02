"""tmux launcher for real-robot collection -- one command, every component visible.

    uv run python -m g1_bridge.launch_collect_real --interface eth0 -- \\
        --backend heft --pico-source real --task-prompt "pick up the cup" \\
        --out /abs/path/to/dataset

Layout is the launcher LocoManiLab's original HeFT integration used, pane for pane, so the muscle memory
carries over -- including which pane holds the focus (the keyboard one) and what the
camera window is for (watching the link, not serving it: the camera server lives on
the robot).

    session: g1_data_collection
    window '<backend>' (4 panes; 5 on the GMR backends -- see below)
      ┌────────────────────────────┬────────────────────────────┐
      │ 0: 控制器 controller       │ 2: 数据导出器 recording    │
      │    collect_real (robot)    │    status bar + events     │
      │                            ├────────────────────────────┤
      │                            │ 3: 遥操作状态 teleop       │
      │                            │    what the collector got  │
      ├────────────────────────────┼────────────────────────────┤
      │ 1: XR 遥操作服务 teleop    │ 4: 键盘控制 keyboard       │
      │    GMR retarget publisher  │    <- FOCUS, keys go here  │
      └────────────────────────────┴────────────────────────────┘
    window 'bridge'   scripts/g1/real_bridge.py   (or sim_env/base_sim.py with --sim)
    window 'camera'   camera-link monitor (fps / frame size), --camera-server to serve

The extra pane exists only when pane 1 is the GMR publisher (heft/scalebfm/almi).
On sonic/amo pane 1 is already the collector's view of the operator, so there is
nothing to add and the layout is the fork's original four. Pane indices therefore
differ between the two layouts, which is why ``_create_session`` returns a role map
instead of anyone writing a number down.

One difference is forced by the architecture rather than chosen: the fork's pane 0
(controller) and pane 2 (data exporter) were two processes, and here recording is
inside the control loop -- it reads ``session.teleop_sample()`` in-process, at the
same 50 Hz. So pane 0 is the controller *and* the exporter, and pane 2 watches what
the exporter has actually written. The keyboard did go back to its own pane
(``g1_bridge.keyboard_control``, publishing over the same ZMQ port the fork used).

Every argument after ``--`` goes to ``g1_bridge.collect_real`` untouched (and is
validated against its parser *before* any tmux window is created, so a typo fails
in a second instead of in a pane you have to go read).

Starting the GMR publisher in a visible pane is deliberate: the backend would
otherwise spawn it as a child with its logs interleaved into the collector's, and
``GMRSubprocess`` reuses a publisher already bound to 28701 -- so a pane here is
the same process, just where you can see it.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import re
import shlex
import shutil
import socket
import signal
import subprocess
import sys
import time

#: Session and window names mirror the heft fork's launcher
#: (heft_data_collection / windows: <backend> | bridge | camera).
SESSION_DEFAULT = "g1_data_collection"
GMR_PORT = 28701


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="g1_bridge.launch_collect_real",
        description="Start real-robot collection in a tmux session (see module docstring).",
        epilog="Everything after `--` is passed to g1_bridge.collect_real.",
    )
    parser.add_argument(
        "--interface", default="eth0", help="robot network interface for the bridge pane."
    )
    parser.add_argument(
        "--sim",
        action="store_true",
        help="run sim_env/base_sim.py in the bridge window instead of the robot "
        "bridge -- the sim2sim smoke path, no hardware needed.",
    )
    parser.add_argument(
        "--camera-source",
        default="realsense",
        help="sim2real.camera.server source (realsense | cv2 | ...).",
    )
    parser.add_argument(
        "--camera-device", type=int, default=None, help="device index for --camera-source cv2."
    )
    parser.add_argument(
        "--camera-fps",
        type=float,
        default=50.0,
        help="camera server fps. Match the control rate or frames repeat in the dataset.",
    )
    parser.add_argument(
        "--camera-connect",
        default=None,
        help="camera address for the monitor window (default: whatever the "
        "collector was given after `--`, else localhost).",
    )
    parser.add_argument(
        "--camera-display",
        default="auto",
        choices=["auto", "tk", "http", "none"],
        help="camera window mode: auto = a native window if a display is "
        "reachable, else an MJPEG stream in the browser; none = fps only.",
    )
    parser.add_argument("--camera-http-port", type=int, default=8081)
    parser.add_argument(
        "--camera-server",
        action="store_true",
        help="start a camera SERVER in the camera window instead of a monitor. Only "
        "for a camera physically attached to this machine -- on a G1 the server runs "
        "onboard and this window just watches the link.",
    )
    parser.add_argument(
        "--no-camera", action="store_true", help="don't open the camera window at all."
    )
    parser.add_argument(
        "--no-bridge",
        action="store_true",
        help="don't start the bridge window (one is already running elsewhere, or "
        "--robot-io inline is in use).",
    )
    parser.add_argument(
        "--no-teleop",
        action="store_true",
        help="don't start the XR/GMR teleop pane (let the backend spawn its own child).",
    )
    parser.add_argument(
        "--no-keyboard-pane",
        action="store_true",
        help="don't start the keyboard-control pane; the collector then reads its own "
        "stdin in pane 0 (--keyboard local).",
    )
    parser.add_argument("--session", default=SESSION_DEFAULT, help="tmux session name.")
    parser.add_argument(
        "--no-attach", action="store_true", help="leave the session detached and return."
    )
    parser.add_argument(
        "--kill", action="store_true", help="kill an existing session of this name and exit."
    )
    parser.add_argument(
        "--gmr-timeout-s",
        type=float,
        default=60.0,
        help="how long to wait for the GMR publisher to bind port 28701 before "
        "starting the collector. It loads GMR/mujoco/the XR SDK first, so this is "
        "tens of seconds on a cold start.",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="kill leftover processes from a previous session (a stale GMR "
        "publisher on 28701 is otherwise silently reused) instead of refusing.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="run the prerequisite checks and print the plan, then exit WITHOUT "
        "creating anything. Use this to verify a setup: a normal run starts the "
        "bridge, and starting the bridge releases the robot's motion mode -- on a "
        "standing robot that means it goes limp.",
    )
    return parser


#: Ports this session's own processes bind. A leftover on any of them is what
#: makes the next launch "reuse" a publisher nobody is driving, or fail to bind.
_SESSION_PORTS = {
    28701: "GMR motion publisher",
    5580: "keyboard control",
    5581: "teleop status",
    5582: "recording status",
}
#: Only ever kill processes whose command line is one of ours.
_OUR_COMMANDS = (
    "publisher_launcher",
    "pico_retarget_pub",
    "g1_bridge.collect_real",
    "g1_bridge.keyboard_control",
    "g1_bridge.recording_monitor",
    "g1_bridge.teleop_monitor",
    "g1_bridge.camera_monitor",
)


def _port_holders() -> list[tuple[int, int, str, bool]]:
    """(pid, port, cmdline, is_ours) for every process holding a session port.

    Reports EVERYTHING, not just our own modules. A foreign process on one of
    these ports is the more dangerous case, not the safer one: port 28701 held by
    anything at all makes the GMR publisher die at bind (measured -- a leftover
    test publisher took a whole session down that way), and a name-filtered scan
    reports "all clear" right before it happens. The filter belongs on *killing*,
    which is where it stays: --clean kills ours and names the rest for a human.
    """
    found: list[tuple[int, int, str, bool]] = []
    seen: set[int] = set()
    for port in _SESSION_PORTS:
        try:
            out = subprocess.run(
                ["bash", "-lc", f"ss -ltnpH 'sport = :{port}' 2>/dev/null"],
                capture_output=True, text=True, timeout=10,
            ).stdout
        except Exception:  # noqa: BLE001
            continue
        for match in re.finditer(r"pid=(\d+)", out):
            pid = int(match.group(1))
            if pid in seen:
                continue
            try:
                cmd = pathlib.Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            except Exception:  # noqa: BLE001
                cmd = "(unreadable)"
            seen.add(pid)
            found.append((pid, port, cmd.strip(), any(n in cmd for n in _OUR_COMMANDS)))
    return found


def _leftover_processes() -> list[tuple[int, int, str]]:
    """Only the holders that are ours to kill."""
    return [(pid, port, cmd) for pid, port, cmd, ours in _port_holders() if ours]


def _wait_for_port(port: int, *, timeout_s: float, label: str) -> bool:
    """Block until something is listening on ``port``. False on timeout.

    Printed rather than silent: on a cold start this is the longest pause in the
    whole launch, and an operator staring at a blank pane needs to know it is
    loading rather than hung.
    """
    deadline = time.time() + timeout_s
    waited = 0.0
    while time.time() < deadline:
        if _port_bound(port):
            if waited:
                print(f"  {label}: up after {waited:.0f}s")
            return True
        time.sleep(1.0)
        waited += 1.0
        if waited in (5.0, 15.0, 30.0):
            print(f"  waiting for the {label} to bind :{port} ({waited:.0f}s)...", flush=True)
    print(
        f"  {label} did not bind :{port} within {timeout_s:.0f}s — continuing; the "
        "backend will start its own (check that pane for the reason)"
    )
    return False


def _port_bound(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.3)
        return sock.connect_ex((host, port)) == 0


def _kill_leftovers(leftovers) -> None:
    """SIGTERM each leftover's process group, then SIGKILL what is left."""
    for pid, _port, _cmd in leftovers:
        for sig in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(os.getpgid(pid), sig)
            except Exception:  # noqa: BLE001 -- already gone, or not ours to signal
                try:
                    os.kill(pid, sig)
                except Exception:  # noqa: BLE001
                    break
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except OSError:
                break          # gone; no need to escalate


def _q(value) -> str:
    return shlex.quote(str(value))


def _tmux(*args: str, check: bool = False):
    return subprocess.run(["tmux", *args], capture_output=True, check=check)


def _session_exists(session: str) -> bool:
    return _tmux("has-session", "-t", session).returncode == 0


def _sim2real_root() -> pathlib.Path:
    import sim2real

    return pathlib.Path(sim2real.__file__).resolve().parent.parent


def _missing_modules(python: str, modules: list[str]) -> list[str]:
    """Which of ``modules`` the interpreter that will run the panes cannot import."""
    probe = (
        "import importlib.util as u, sys;"
        "print(' '.join(m for m in sys.argv[1:] if u.find_spec(m) is None))"
    )
    try:
        out = subprocess.run(
            [python, "-c", probe, *modules], capture_output=True, text=True, timeout=120
        )
    except Exception:  # noqa: BLE001 -- a broken interpreter is the caller's problem
        return []
    return out.stdout.split()


def _install_hint(root: pathlib.Path, extra: str, packages: str) -> str:
    """How to install an optional dependency, phrased for the env actually in use."""
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    if conda_env:
        return f"conda activate {conda_env} && pip install {packages}"
    return f"cd {root} && uv sync --extra {extra}"


def _check_prerequisites(args, rest, root: pathlib.Path):
    """Fail here, once, instead of in three panes the operator has to go read.

    Every window below runs a different subset of the optional dependencies
    (``robot-g1`` for the bridge, ``camera`` for the camera server), and a missing
    one surfaces as a ModuleNotFoundError inside its own pane -- three separate
    red tracebacks that look like "the environment is broken" when it is one
    `pip install` away.
    """
    errors = []
    if not shutil.which("tmux"):
        errors.append("tmux is not installed. Install with: sudo apt install tmux")

    python = sys.executable
    missing = _missing_modules(
        python, ["sim2real", "g1_bridge", "lerobot", "av", "cv2", "msgpack", "zmq"]
    )
    if missing:
        errors.append(
            f"the collector's own dependencies are missing from {python}: "
            f"{', '.join(missing)} -- "
            + _install_hint(root, "lerobot --extra camera", "lerobot==0.3.3 av opencv-python-headless msgpack msgpack-numpy")
        )
    if not args.no_bridge and not args.sim and _missing_modules(python, ["unitree_sdk2py"]):
        errors.append(
            "the robot bridge needs unitree_sdk2py (the 'robot-g1' extra) -- "
            + _install_hint(
                root, "robot-g1",
                "'unitree-sdk2py @ git+https://github.com/EGalahad/unitree_sdk2_python.git'",
            )
            + "\n    (or pass --no-bridge and run scripts/g1/real_bridge.py wherever it "
            "is installed, e.g. onboard the robot)"
        )
    if not args.no_camera and args.camera_server:
        # Only a camera SERVER needs a capture backend. The default camera window
        # is a monitor, which needs nothing the collector does not already have.
        needed = {"realsense": "pyrealsense2", "cv2": "cv2"}.get(args.camera_source)
        if needed and _missing_modules(python, [needed]):
            errors.append(
                f"--camera-server needs {needed} for --camera-source "
                f"{args.camera_source} -- "
                + _install_hint(root, "camera", "pyrealsense2 opencv-python-headless msgpack msgpack-numpy")
                + "\n    (drop --camera-server if the camera runs on the robot: the "
                "camera window then just watches the link)"
            )
    # Validate the collector's own arguments before building any windows.
    from g1_bridge import collect_real

    try:
        collect_real.build_parser().parse_args(rest)
    except SystemExit:
        errors.append(
            "the arguments after `--` are not valid for g1_bridge.collect_real (see above)"
        )
    out = _arg_of(rest, "--out")
    if out and pathlib.Path(out).exists():
        errors.append(
            f"--out already exists: {out}\n    LeRobot refuses to write into an "
            "existing dataset dir (it would mix two runs' episode indices). Pick a "
            "new one, or move the old one aside."
        )
    # Who needs the isolated GMR venv: heft/scalebfm for the retarget publisher,
    # and amo for its own arm-retargeting worker (--amo-arm-source gmr, the
    # default with real hardware). sonic needs neither.
    needs_pico_venv = (not args.no_teleop and _uses_gmr_motion(rest)) or (
        _backend_of(rest) == "amo" and _arg_of(rest, "--amo-arm-source") != "ik"
    )
    if needs_pico_venv:
        pico_python = root / "venv" / "pico" / ".venv" / "bin" / "python"
        if not pico_python.exists():
            errors.append(
                f"GMR venv missing: {pico_python} -- run `uv sync --project "
                f"venv/pico` in {root}"
            )
    if not args.no_bridge and not args.sim:
        bridge = root / "scripts" / "g1" / "real_bridge.py"
        if not bridge.is_file():
            errors.append(f"robot bridge not found: {bridge}")
    # A stale publisher is worse than a missing one: GMRSubprocess deliberately
    # REUSES whatever is bound to 28701 (two operators, one headset), so a leftover
    # from a killed session makes the next run look perfectly connected while
    # nothing moves. Catch it here, where it is one line to fix, instead of in the
    # teleop pane twenty minutes later.
    holders = _port_holders()
    if holders:
        print("  something is already holding this session's ports:")
        for pid, port, cmd, ours in holders:
            tag = "ours" if ours else "NOT ours"
            print(f"    pid {pid:>7}  port {port}  ({_SESSION_PORTS.get(port, '?')})  [{tag}]")
            print(f"             {cmd[:88]}")
        ours = [(pid, port, cmd) for pid, port, cmd, is_ours in holders if is_ours]
        foreign = [h for h in holders if not h[3]]
        if args.clean and ours:
            _kill_leftovers(ours)
            print("  --clean: sent SIGTERM to ours; continuing")
        if foreign:
            errors.append(
                "the ports above marked NOT ours belong to something this launcher "
                "will not kill for you.\n"
                "    Port 28701 in particular is fatal rather than shared: the GMR "
                "publisher dies at bind, and\n"
                "    the pane where it happens is not the one you are watching. Kill "
                "the pid yourself, then re-run."
            )
        elif ours and not args.clean:
            errors.append(
                "re-run with --clean to kill them, or "
                f"`python -m g1_bridge.launch_collect_real --kill --session {args.session}`.\n"
                "    (a leftover GMR publisher is silently REUSED, so the session "
                "would look connected and not move)"
            )

    if errors:
        print("ERROR: prerequisites not met:\n")
        for e in errors:
            print(f"  - {e}")
        print()
        sys.exit(1)


def _pane(session: str, index: int) -> str:
    return f"{session}:0.{index}"


def _set_pane_title(session: str, index: int, title: str):
    _tmux("select-pane", "-t", _pane(session, index), "-T", title)


def _send(session: str, index: int, cmd: str, wait: float = 1.0):
    """Run ``cmd`` in a pane, verbatim."""
    _tmux("send-keys", "-t", _pane(session, index), cmd, "C-m")
    time.sleep(wait)


def _new_window(session: str, name: str, cmd: str, wait: float = 1.0):
    _tmux("new-window", "-t", session, "-n", name)
    _tmux("send-keys", "-t", f"{session}:{name}", cmd, "C-m")
    time.sleep(wait)


#: Environment variables a pane needs to be *this* environment rather than
#: whatever the tmux server happened to be started with. A new pane inherits the
#: tmux SERVER's environment -- and the server is usually a days-old process from
#: an unrelated session -- so without this the panes silently run under someone
#: else's conda env, ld path or HF cache. Deliberately an allowlist: copying the
#: whole environment would drag TMUX/TERM/SSH_* along and confuse the new panes.
_SESSION_ENV_KEYS = (
    "PATH",
    "CONDA_PREFIX",
    "CONDA_DEFAULT_ENV",
    "CONDA_EXE",
    "VIRTUAL_ENV",
    "LD_LIBRARY_PATH",
    "PYTHONPATH",
    "MUJOCO_GL",
    "SIM2REAL_ROOT",
    "HF_HOME",
    "HF_ENDPOINT",
    "HUGGINGFACE_HUB_CACHE",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_HOME",
    # Without these the camera window cannot open a window at all from inside a
    # pane -- the tmux server predates this login's X session.
    "DISPLAY",
    "XAUTHORITY",
    "WAYLAND_DISPLAY",
    "XDG_RUNTIME_DIR",
)


def _session_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _SESSION_ENV_KEYS if os.environ.get(k)}


def _env_prefix() -> str:
    """A shell prefix that re-activates this environment inside a pane.

    The panes run commands through an interactive shell, whose rc re-runs conda's
    own shell hook -- which resets CONDA_PREFIX to empty in a shell that was never
    `conda activate`d, even though the inherited PATH still points at the env. The
    absolute interpreter path below means nothing *we* start depends on that, but
    an operator typing `python` or `pip` in a pane would get the wrong one, so
    activate properly instead of relying on inheritance.
    """
    conda_env = os.environ.get("CONDA_DEFAULT_ENV")
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_env and conda_env != "base" and conda_exe:
        conda_sh = pathlib.Path(conda_exe).resolve().parent.parent / "etc/profile.d/conda.sh"
        if conda_sh.is_file():
            return f"source {_q(conda_sh)} && conda activate {_q(conda_env)} && "
    venv = os.environ.get("VIRTUAL_ENV")
    if venv and (pathlib.Path(venv) / "bin/activate").is_file():
        return f"source {_q(pathlib.Path(venv) / 'bin/activate')} && "
    return ""


def _create_session(session: str, window: str, *, teleop_status_pane: bool = False):
    """Build the window and return {role: pane index}.

    The map is returned rather than written down at each call site because the
    indices are not stable across layouts: tmux renumbers by position, so the
    extra pane below shifts everything after it. Roles do not shift.
    """
    env = _session_env()
    new_session = ["tmux", "new-session", "-d", "-s", session, "-n", window]
    for key, value in env.items():
        new_session += ["-e", f"{key}={value}"]
    subprocess.run(new_session, check=True)
    # Also on the session itself, so windows/panes created below inherit it even
    # though they are not part of the new-session command.
    for key, value in env.items():
        _tmux("set-environment", "-t", session, key, value)
    _tmux("set-option", "-t", session, "-g", "mouse", "on")
    # A pane whose program exits stays, showing its last output. Otherwise a
    # component that dies at startup (a port already bound, a missing dependency)
    # takes its pane with it and the session just looks empty.
    _tmux("set-option", "-t", session, "-g", "remain-on-exit", "on")
    _tmux("bind-key", "-T", "root", "C-\\", "kill-session")
    # Titled, coloured pane borders: which component is which, at a glance.
    _tmux("set-option", "-t", session, "-g", "pane-border-status", "top")
    _tmux("set-option", "-t", session, "-g", "pane-border-format",
          " #[bold]#{pane_index}: #{pane_title}#[default] ")
    _tmux("set-option", "-t", session, "-g", "pane-border-style", "fg=colour240")
    _tmux("set-option", "-t", session, "-g", "pane-active-border-style", "fg=colour46,bold")
    _tmux("set-option", "-t", session, "-g", "status-style", "bg=colour24,fg=white")
    _tmux("set-option", "-t", session, "-g", "status-left",
          "#[bg=colour46,fg=black,bold] g1 collect #[default] ")
    _tmux("set-option", "-t", session, "-g", "status-left-length", "24")
    _tmux("set-window-option", "-t", session, "-g", "window-status-current-style",
          "bg=colour46,fg=black,bold")
    #   0 | 2              0 | 2
    #  ---|---     or     ---| 3    <- the extra status pane, on the GMR
    #   1 | 3              1 | 4       backends (whose pane 1 is the publisher,
    #                                  not the collector's view of the operator)
    _tmux("split-window", "-t", f"{session}:0", "-h")
    _tmux("split-window", "-t", _pane(session, 0), "-v")
    _tmux("split-window", "-t", _pane(session, 2), "-v")
    if not teleop_status_pane:
        return {"collector": 0, "teleop": 1, "recording": 2, "keyboard": 3}
    # Splitting the RIGHT column keeps the two panes an operator's hands and eyes
    # know where to find -- the collector top-left, the keys bottom-right -- and
    # puts the new one next to the other read-only monitor.
    _tmux("split-window", "-t", _pane(session, 2), "-v")
    return {"collector": 0, "teleop": 1, "recording": 2,
            "teleop_status": 3, "keyboard": 4}


def _arg_of(rest: list[str], flag: str):
    """The value the collector was given for ``flag``, if any."""
    for i, item in enumerate(rest):
        if item == flag and i + 1 < len(rest):
            return rest[i + 1]
        if item.startswith(f"{flag}="):
            return item.split("=", 1)[1]
    return None


def _uses_gmr_motion(rest: list[str]) -> bool:
    """Whether this backend consumes the GMR retarget publisher's motion stream.

    Read off the backend class rather than hardcoded: heft/scalebfm track a
    GMR-retargeted body and almi reads its waist/arm reference and both stick
    axes out of the same stream, while sonic and amo open their own direct PICO
    connection. Starting the publisher next to those two would put a SECOND
    xrobotoolkit_sdk connection on the same headset, and whichever is not
    "primary" gets a frozen body pose forever -- the failure gmr_button_relay.py
    documents. (amo still uses GMR for its ARMS, but through its own worker
    process, which the backend starts itself.)
    """
    try:
        from g1_bridge.backends import get_backend_class

        return bool(get_backend_class(_backend_of(rest)).uses_gmr)
    except Exception:  # noqa: BLE001 -- an unknown backend fails later, with a better message
        return True


def _backend_of(rest: list[str]) -> str:
    """Name the main window after the controller, as the fork did ('heft')."""
    return _arg_of(rest, "--backend") or "collect"


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, rest = argv[:split], argv[split + 1:]
    args = build_parser().parse_args(argv)

    if args.kill:
        _tmux("kill-session", "-t", args.session)
        print(f"killed tmux session {args.session!r} (if it existed)")
        time.sleep(0.5)
        leftovers = _leftover_processes()
        if leftovers:
            print("  leftovers still holding this session's ports:")
            for pid, port, cmd in leftovers:
                print(f"    pid {pid:>7}  port {port}  {cmd[:70]}")
            _kill_leftovers(leftovers)
            print("  sent SIGTERM to each; re-run --kill to check")
        else:
            print("  no leftover publisher/collector processes")
        return 0

    root = _sim2real_root()
    _check_prerequisites(args, rest, root)
    if _session_exists(args.session):
        print(
            f"tmux session {args.session!r} already exists. Attach with "
            f"`tmux attach -t {args.session}`, or drop it with "
            f"`python -m g1_bridge.launch_collect_real --kill --session {args.session}`."
        )
        return 1

    python = _q(sys.executable)
    cd_root = f"{_env_prefix()}cd {_q(root)}"

    def run(program: str) -> str:
        """Environment, then cd, then the program -- with the pane's shell kept.

        Deliberately NOT `exec`. Replacing the shell does make tmux's SIGHUP land
        directly on the program, which is tidy in theory; in practice it also
        means that a program which exits takes its pane (and, for a window with
        one pane, the whole window) with it. Measured here: a bridge that failed
        to bind its port left no window and no traceback -- the operator sees a
        session that "has nothing in it" instead of the error that explains why.

        Keeping the shell keeps the error on screen, and `remain-on-exit` keeps
        the pane itself. Cleanup of processes that outlive their pane is handled
        where it actually belongs: the collector turns SIGTERM/SIGHUP into a
        normal shutdown, GMRSubprocess kills its whole process group, and the
        launcher refuses to start (or, with --clean, clears) leftovers still
        holding a port.
        """
        return f"{cd_root} && {program}"
    window = _backend_of(rest)
    keyboard_pane = not args.no_keyboard_pane
    collector_args = list(rest)
    if keyboard_pane:
        # The keys get their own pane, as they did in the fork; the collector
        # subscribes instead of owning this terminal.
        collector_args += ["--keyboard", "zmq"]
    collector = run(f"{python} -m g1_bridge.collect_real {' '.join(_q(a) for a in collector_args)}")
    camera_connect = args.camera_connect or _arg_of(rest, "--camera-connect") or \
        f"tcp://127.0.0.1:{5594}"

    print("=" * 68)
    print("  G1 real-robot collection launcher")
    print("=" * 68)
    print(f"  session:    {args.session}   (window '{window}')")
    print(f"  repo:       {root}")
    print(f"  python:     {sys.executable}")
    print(f"  bridge:     {'sim2sim (base_sim.py)' if args.sim else ('skipped' if args.no_bridge else f'real_bridge.py --interface {args.interface}')}")
    print(f"  camera:     {'no window' if args.no_camera else (f'SERVER {args.camera_source} @ {args.camera_fps} fps' if args.camera_server else f'monitor {camera_connect}')}")
    teleop_pane = not args.no_teleop and _uses_gmr_motion(rest)
    print(
        "  teleop:     "
        + (
            "GMR retarget publisher pane + teleop monitor pane"
            if teleop_pane
            else (
                "pane disabled"
                if args.no_teleop
                else "teleop monitor pane (backend owns its own PICO link)"
            )
        )
    )
    print(f"  keys:       {'keyboard-control pane (ZMQ)' if keyboard_pane else 'collector pane 0 (stdin)'}")
    print(f"  collector:  {' '.join(rest) or '(no arguments!)'}")
    print("=" * 68)

    if args.check_only:
        print("\n  --check-only: prerequisites pass; nothing was started.")
        return 0

    panes = _create_session(args.session, window, teleop_status_pane=teleop_pane)

    # Bridge first, in its own window: the collector blocks on the robot's state
    # stream, so it needs something publishing before it can arm anything.
    #
    # NOTE this is the moment the robot is touched: RealBridge.__init__ force-
    # releases the active Unitree motion mode, so a robot that was standing under
    # its built-in controller goes limp. Have it on the hoist. Nothing above this
    # line talks to hardware -- that is what --check-only stops before.
    if not args.no_bridge:
        if args.sim:
            bridge_cmd = run(f"{python} sim2real/sim_env/base_sim.py --robot g1")
        else:
            # --rate = the control rate, not real_bridge's own 100 Hz default: the
            # collector steps once per state packet (as LocoManiLab's original HeFT integration does), so
            # the publish rate IS the control rate. heft's C++ bridge does the same
            # thing by decimating its 1 kHz loop by 20.
            bridge_cmd = run(
                f"{python} scripts/g1/real_bridge.py "
                f"--interface {_q(args.interface)} "
                f"--rate {_q(_arg_of(rest, '--control-freq') or 50)}"
            )
        _new_window(args.session, "bridge", bridge_cmd, wait=2.0)

    # Camera window: a MONITOR by default, because the camera server runs on the
    # robot in every deployment this launcher is used for -- the question at
    # launch is "is the link up and fast enough", which is what it answers.
    if not args.no_camera:
        if args.camera_server:
            camera_cmd = run(
                f"{python} -m sim2real.camera.server "
                f"--cameras.ego_view.source {_q(args.camera_source)} "
                f"--fps {_q(args.camera_fps)}"
            )
            if args.camera_device is not None:
                camera_cmd += f" --cameras.ego_view.device {_q(args.camera_device)}"
        else:
            camera_cmd = run(
                f"{python} -m g1_bridge.camera_monitor "
                f"--connect {_q(camera_connect)} "
                f"--display {_q(args.camera_display)} "
                f"--http-port {_q(args.camera_http_port)} "
                f"--control-freq {_q(_arg_of(rest, '--control-freq') or 50)}"
            )
        _new_window(args.session, "camera", camera_cmd, wait=1.0)

    _tmux("select-window", "-t", f"{args.session}:0")

    # The 'teleop' pane (bottom-left): XR / GMR teleop service -- the fork's pane
    # 1. Every backend gets something here, so the pane means the same thing
    # everywhere: for the GMR backends the retarget publisher itself, for
    # sonic/amo (whose PICO link lives inside the collector, in a multiprocessing
    # child with no socket to subscribe to) a view of what the collector is
    # actually receiving. Leaving it empty for half the backends would make the
    # pane differ exactly where an operator looks when the chord "does nothing".
    if args.no_teleop:
        _set_pane_title(args.session, panes["teleop"], "XR teleop (pane disabled)")
    elif not teleop_pane:
        _set_pane_title(args.session, panes["teleop"],
                        "XR 遥操作 / teleop input (collector's view)")
        _send(args.session, panes["teleop"],
              run(f"{python} -m g1_bridge.teleop_monitor"), wait=0.5)
    else:
        pico_python = root / "venv" / "pico" / ".venv" / "bin" / "python"
        launcher = root / "g1_bridge" / "gmr_utils" / "publisher_launcher.py"
        _set_pane_title(args.session, panes["teleop"],
                        "XR 遥操作服务 / GMR retarget publisher")
        _send(args.session, panes["teleop"],
              run(f"{_q(pico_python)} {_q(launcher)} --no-viewer"), wait=0.5)
        # Wait for it to actually BIND before the collector starts. This is a race
        # with teeth: the backend starts its own publisher whenever 28701 is free,
        # and the pane's one takes 10-20 s to get there (it imports GMR, mujoco and
        # the XR SDK first). Start the collector too early and both bind, the loser
        # dies with "Address already in use", and which one lost is a coin flip.
        _wait_for_port(GMR_PORT, timeout_s=args.gmr_timeout_s, label="GMR publisher")

        # ... and, on these backends only, a pane for the collector's own view of
        # the operator. The publisher's log above says what it SENT; this says
        # what arrived. The two differ on the one channel that matters most here:
        # the sticks reach the collector through GMRButtonRelay, which once
        # carried the buttons and not the axes. heft/scalebfm never noticed (they
        # do not steer); almi's entire locomotion command is those two sticks, and
        # on hardware there is no MuJoCo HUD to see them in.
        _set_pane_title(args.session, panes["teleop_status"],
                        "遥操作状态 / teleop input (collector's view)")
        _send(args.session, panes["teleop_status"],
              run(f"{python} -m g1_bridge.teleop_monitor"), wait=0.5)

    # The 'recording' pane (top-right): the fork's data-exporter pane. The
    # exporter itself is a thread inside the collector's process, so what lives
    # here is its OUTPUT -- the status bar and every recording event -- carried
    # over a socket. Same pane, same information, and pane 0 stays about the robot.
    _set_pane_title(args.session, panes["recording"], "数据导出器 / recording status")
    _send(args.session, panes["recording"],
          run(f"{python} -m g1_bridge.recording_monitor"), wait=0.5)

    # Pane 0 (top-left): the controller (and, unlike the fork, the exporter too).
    _set_pane_title(args.session, panes["collector"], "控制器 / controller + recording")
    _send(args.session, panes["collector"], collector, wait=0.5)

    # The 'keyboard' pane (bottom-right): keyboard control, and the focus --
    # exactly as in the fork, so the pane the operator types into is the one that
    # only takes keys.
    if keyboard_pane:
        _set_pane_title(args.session, panes["keyboard"],
                        "键盘控制 / keyboard control  <- FOCUS")
        _send(args.session, panes["keyboard"],
              run(f"{python} -m g1_bridge.keyboard_control"), wait=0.5)
        _tmux("select-pane", "-t", _pane(args.session, panes["keyboard"]))
    else:
        _set_pane_title(args.session, panes["keyboard"], "shell")
        prefix = _env_prefix()
        if prefix:
            _send(args.session, panes["keyboard"], prefix.rstrip(" &"), wait=0.2)
        _tmux("select-pane", "-t", _pane(args.session, panes["collector"]))

    print()
    print(f"  window '{window}':")
    print("    pane 0 (top-left)      控制器 controller + recording (collect_real)")
    print(
        "    pane 1 (bottom-left)   "
        + (
            "XR 遥操作服务 GMR publisher"
            if teleop_pane
            else ("(disabled)" if args.no_teleop else "XR 遥操作 teleop monitor")
        )
    )
    print("    pane 2 (top-right)     数据导出器 recording monitor")
    print("    pane 3 (bottom-right)  " + ("键盘控制 keyboard control  <- you are here" if keyboard_pane else "shell"))
    if not args.no_bridge:
        print(f"  window 'bridge':  {'MuJoCo sim2sim' if args.sim else 'G1 robot bridge'}")
    if not args.no_camera:
        print(f"  window 'camera':  {'camera server' if args.camera_server else 'camera-link monitor'}")
    print()
    print("  operating order:")
    print("    1. CONTROLLER pane (0): 'y' (or Enter) to confirm — robot ramps to the default pose")
    print("    2. PICO: A+B+X+Y — enters the tracking policy and engages ('a' in pane 0 also arms)")
    print("    3. PICO: A+X to start following you")
    print("    4. KEYBOARD pane (3): 'c' start / stop+save an episode, 'x' discard")
    print("    e-stop: 'o' in the controller pane. The keyboard pane only records.")
    print()
    print("  tmux:  Ctrl-b <arrows> panes | Ctrl-b n/p windows | Ctrl-b d detach | Ctrl-\\ kill")
    print("=" * 68)

    if args.no_attach:
        print(f"detached. Attach with: tmux attach -t {args.session}")
        return 0
    try:
        subprocess.run(["tmux", "attach", "-t", args.session])
    except KeyboardInterrupt:
        pass
    if _session_exists(args.session):
        print(f"\nsession {args.session!r} is still running.")
        print(f"  reattach: tmux attach -t {args.session}")
        print(f"  kill:     tmux kill-session -t {args.session}")
    return 0


def _signal_handler(sig, frame):
    print("\nshutdown requested; leaving the tmux session running.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, _signal_handler)
    sys.exit(main())
