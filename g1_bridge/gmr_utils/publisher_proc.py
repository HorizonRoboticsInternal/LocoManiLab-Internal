"""Launches sim2real's own PICO GMR retargeting publisher as a subprocess.

``general_motion_retargeting`` (GMR) -- the package sim2real's
``sim2real.teleop.pico_retarget_pub.LiveRetargetPublisher`` needs for full-
body PICO retargeting -- is only installed in sim2real's isolated
``venv/pico/.venv`` (Python 3.10); it does not import under the Python 3.11
``tele_benchmark`` env this bridge otherwise runs in (confirmed: not even
listed as a dependency of sim2real's main package). So unlike the SONIC/smpl
path (Flavor A, in-process push), full-body GMR retargeting for Flavor B runs
as a genuinely separate OS process, using ``pico_retarget_pub.py`` completely
unmodified, publishing over its normal ZMQ interface (default
``tcp://*:28701``) exactly as it's designed to.

Verified this and tele's own ``RealPicoVRSource`` can hold independent
concurrent connections to the same running PICO PC Service
(``RoboticsServiceProcess``, port 60061) without conflict -- two separate
``xrobotoolkit_sdk.init()`` calls (one from each Python env) ran side by side
successfully.

IMPORTANT -- ``LiveRetargetPublisher`` starts with ``self.paused = True``
(pico_retarget_pub.py's own default, no CLI flag to change it) and only
flips to live tracking on a **rising edge of the left controller's plain X
button**, read by *this specific process's own* independent PICO SDK
connection (not part of any A+B+X+Y-style chord -- that's tele's own
ModeManager, a completely separate PICO connection/state machine). In
practice this proved painful to operate: this process's stdout (where the
toggle confirmation prints) is rarely the terminal the operator is
watching, so there's no feedback on whether a press registered, and two
independent SDK connections both polling the same physical button made
debugging "did my press even go anywhere" slow. So when *this module*
launches a fresh publisher, it runs it through ``publisher_launcher.py``
instead of the module directly, which subclasses ``LiveRetargetPublisher``
to force ``paused = False`` from the start (X still works as a normal
pause/resume toggle afterwards). A publisher this module *reuses* instead
(because something was already bound to the port -- see ``start()``) was
not necessarily started this way and may still be paused; confirmed by
directly subscribing to its published feed: a paused publisher's
``body_pos_w``/``joint_pos`` are frozen at a default pose (and the JSON
payload has ``"paused": true``) no matter how much the operator moves --
looks exactly like "nothing is happening" but is not a code bug.
"""

from __future__ import annotations

import atexit
import os
import pathlib
import signal
import socket
import subprocess
import sys
import time

import sim2real

_SIM2REAL_ROOT = pathlib.Path(sim2real.__file__).resolve().parent.parent
_PICO_VENV_PYTHON = _SIM2REAL_ROOT / "venv" / "pico" / ".venv" / "bin" / "python"
_DEFAULT_BIND_PORT = 28701  # matches PublisherArgs.bind default "tcp://*:28701"


def _port_open(port: int, host: str = "127.0.0.1", timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


class GMRSubprocess:
    """Manages one ``sim2real.teleop.pico_retarget_pub`` child process.

    This is a shared PICO PC Service (``RoboticsServiceProcess``) and shared
    machine -- another operator may already have a compatible publisher
    running (confirmed happens in practice: someone's own long-running
    ``pico_retarget_pub.py`` session was found bound to the default ports).
    Since ZMQ PUB sockets support unlimited SUB clients with zero
    interference, ``start()`` checks the target port first and, if something
    is already publishing there, reuses it instead of trying (and failing)
    to bind a duplicate -- and correspondingly never tries to kill a process
    it didn't launch.
    """

    def __init__(self, extra_args: list[str] | None = None, bind_port: int = _DEFAULT_BIND_PORT):
        if not _PICO_VENV_PYTHON.exists():
            raise FileNotFoundError(
                f"sim2real's isolated pico venv not found at {_PICO_VENV_PYTHON} "
                "-- GMR retargeting needs this (see module docstring)."
            )
        self._extra_args = extra_args or []
        self._bind_port = bind_port
        self._proc: subprocess.Popen | None = None
        self._external = False

    def start(self, *, warmup_s: float = 2.0) -> None:
        if self.is_alive() or self._external:
            return
        if _port_open(self._bind_port):
            print(
                f"[g1_bridge] a GMR publisher is already bound to port "
                f"{self._bind_port} (likely another operator's session) -- "
                "reusing it instead of launching a duplicate.\n"
                "[g1_bridge] it may be paused (starts paused by "
                "default, and there's no way to tell from here) -- if the "
                "robot doesn't respond to your body movement, press the "
                "LEFT CONTROLLER'S PLAIN X BUTTON ONCE (not part of any "
                "chord) to toggle it into live tracking."
            )
            self._external = True
            return
        cmd = [
            str(_PICO_VENV_PYTHON),
            str(pathlib.Path(__file__).parent / "publisher_launcher.py"),
            *self._extra_args,
        ]
        print(f"[g1_bridge] launching GMR subprocess (auto-live): {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd,
            cwd=str(_SIM2REAL_ROOT),
            stdout=sys.stdout,
            stderr=sys.stderr,
            # Its own process group, so stop() can take down the publisher AND
            # anything it spawned with one signal. Without this, a child that
            # outlives its parent keeps port 28701 bound and the next session
            # silently "reuses" a publisher nobody is driving.
            start_new_session=True,
        )
        atexit.register(self.stop)
        if warmup_s > 0:
            time.sleep(warmup_s)
        if not self.is_alive():
            hint = ""
            if _port_open(self._bind_port):
                hint = (
                    f"\n  Port {self._bind_port} is bound by something else: another "
                    "publisher started at almost the same moment (the launcher now "
                    "waits for one to bind before starting the collector), or a "
                    "leftover from a previous session. "
                    "`python -m g1_bridge.launch_collect_real --kill` clears ours."
                )
            raise RuntimeError(
                f"GMR subprocess exited immediately (returncode="
                f"{self._proc.returncode}) -- check its stdout/stderr above.{hint}"
            )

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        """Take the publisher down, process group and all.

        The group is what matters: ``terminate()`` alone signals the launcher
        process, and anything it started (or anything that re-parented) keeps
        running -- holding port 28701, which the next session then "reuses"
        instead of starting a publisher of its own. That reuse is a real feature
        (two operators, one headset), which is exactly why a stale one is so
        confusing: everything looks connected and nothing moves.
        """
        if self._external or self._proc is None:
            return
        if self._proc.poll() is None:
            self._signal_group(signal.SIGTERM)
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._signal_group(signal.SIGKILL)
                try:
                    self._proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    print(
                        f"[g1_bridge] GMR subprocess {self._proc.pid} did not exit; "
                        f"port {self._bind_port} may stay bound"
                    )
        self._proc = None

    def _signal_group(self, sig) -> None:
        """Signal the publisher's whole process group, falling back to the pid."""
        try:
            os.killpg(os.getpgid(self._proc.pid), sig)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                self._proc.send_signal(sig)
            except ProcessLookupError:
                pass


def start_gmr_if_available(cfg: dict):
    """``GMRSubprocess`` started, or None when this host cannot run one and does
    not need to.

    The GMR publisher needs sim2real's isolated pico venv. A box without it
    (a training cluster running ``lab eval --sim``, a dev box with only the
    policy venvs) has no operator stream anyway: with ``pico_source`` mock or
    keyboard the backend is either driven by a VLA (eval -- the reference
    comes from the adapter, not from GMR) or recording nothing (mock PICO has
    no body tracking). Failing hard there turned every such eval into
    ``FileNotFoundError: ... venv/pico`` before the first step. With
    ``pico_source: real`` the venv IS required and the error stands.
    """
    try:
        proc = GMRSubprocess(extra_args=cfg.get("gmr_extra_args", []))
    except FileNotFoundError:
        if cfg.get("pico_source", "mock") == "real":
            raise
        print(
            "[g1_bridge] GMR subprocess not started: no pico venv on this host and "
            f"pico_source={cfg.get('pico_source', 'mock')!r} (no operator stream to "
            "retarget). Fine for VLA-driven eval; a mock-PICO collection records nothing."
        )
        return None
    proc.start(warmup_s=float(cfg.get("gmr_warmup_s", 2.0)))
    return proc

