"""Main-process side of the GMR arm retargeter: owns the worker and the wire.

Pairs with ``arm_worker.py`` (read its docstring first -- it explains why AMO
cannot use the ``GMRSubprocess``/``GMRButtonRelay`` path at all). The split of
responsibility is the whole point: **this process keeps the PICO connection**,
so joysticks, ModeManager and wrist calibration are untouched; the worker is a
pure function of the body frame we hand it.

Both directions are PUB/SUB with ``CONFLATE=1`` and never block the caller.
That is deliberate for a 50 Hz control loop: a worker that stalls must cost the
arms *staleness*, which ``latest()`` reports and the backend can act on, rather
than stalling the loop that is also balancing the robot. Same reasoning, and
the same shape, as ``relay.py``'s ``GMRRelay``.
"""

from __future__ import annotations

import atexit
import pathlib
import struct
import subprocess
import sys
import threading
import time

import numpy as np
import zmq

import sim2real

from g1_bridge.gmr_utils.arm_worker import (
    DEFAULT_IN_PORT,
    DEFAULT_OUT_PORT,
    _REQ_STRUCT,
    _RSP_BYTES,
)

#: Resolved through the package, never by directory name -- the checkout can
#: live anywhere. Same contract as publisher_proc.py.
_SIM2REAL_ROOT = pathlib.Path(sim2real.__file__).resolve().parent.parent
_PICO_VENV_PYTHON = _SIM2REAL_ROOT / "venv" / "pico" / ".venv" / "bin" / "python"
_WORKER_PATH = pathlib.Path(__file__).parent / "arm_worker.py"


class GMRArmRelay:
    """One GMR arm worker plus the sockets to it.

    ``submit`` is fire-and-forget, ``latest`` returns the newest answer and how
    old it is. Nothing here raises during a control tick.
    """

    def __init__(
        self,
        *,
        actual_human_height: float = 1.60,
        robot: str = "unitree_g1",
        in_port: int = DEFAULT_IN_PORT,
        out_port: int = DEFAULT_OUT_PORT,
        viewer: bool = False,
        viewer_port: int | None = None,
    ):
        if not _PICO_VENV_PYTHON.exists():
            raise FileNotFoundError(
                f"sim2real's isolated pico venv not found at {_PICO_VENV_PYTHON} "
                "-- GMR arm retargeting needs it (GMR does not install under "
                "this env's Python). Run scripts/setup.sh tele-benchmark."
            )
        self._height = float(actual_human_height)
        self._robot = robot
        self._in_port = in_port
        self._out_port = out_port
        self._viewer = bool(viewer)
        self._viewer_port = viewer_port
        self._proc: subprocess.Popen | None = None

        ctx = zmq.Context.instance()
        self._pub = ctx.socket(zmq.PUB)
        self._pub.bind(f"tcp://127.0.0.1:{in_port}")
        self._sub = ctx.socket(zmq.SUB)
        self._sub.setsockopt(zmq.SUBSCRIBE, b"")
        self._sub.setsockopt(zmq.CONFLATE, 1)
        self._sub.setsockopt(zmq.RCVTIMEO, 200)
        self._sub.connect(f"tcp://127.0.0.1:{out_port}")

        self._lock = threading.Lock()
        self._q14: np.ndarray | None = None
        self._stamp: float = 0.0
        self._seq_out = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="GMRArmRelay"
        )

    # -- lifecycle -------------------------------------------------------- #
    def start(self, *, warmup_s: float = 3.0) -> None:
        cmd = [
            str(_PICO_VENV_PYTHON),
            str(_WORKER_PATH),
            "--in-port", str(self._in_port),
            "--out-port", str(self._out_port),
            "--robot", self._robot,
            "--actual-human-height", f"{self._height:.3f}",
        ]
        if self._viewer:
            cmd.append("--viewer")
            if self._viewer_port is not None:
                cmd += ["--viewer-port", str(self._viewer_port)]
        print(f"[g1_bridge] launching GMR arm worker: {' '.join(cmd)}")
        self._proc = subprocess.Popen(
            cmd, cwd=str(_SIM2REAL_ROOT), stdout=sys.stdout, stderr=sys.stderr
        )
        atexit.register(self.stop)
        self._thread.start()
        # GMR builds a MuJoCo model and a mink IK configuration at import; the
        # first solve is also the slowest. Give it that before the first tick
        # asks for an answer.
        if warmup_s > 0:
            time.sleep(warmup_s)
        if not self.is_alive():
            raise RuntimeError(
                f"GMR arm worker exited immediately (returncode="
                f"{self._proc.returncode}) -- see its output above."
            )

    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self._proc is not None and self._proc.poll() is None:
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._proc.kill()
                self._proc.wait(timeout=timeout)
        self._proc = None
        self._pub.close(linger=0)
        self._sub.close(linger=0)

    # -- the wire --------------------------------------------------------- #
    def submit(self, raw_smpl) -> None:
        """Hand one (24, 7) body frame to the worker. Never blocks, never raises."""
        if raw_smpl is None:
            return
        arr = np.asarray(raw_smpl, dtype=np.float32)
        if arr.shape != (24, 7) or not np.isfinite(arr).all():
            return
        self._seq_out += 1
        try:
            self._pub.send(
                _REQ_STRUCT.pack(self._seq_out) + arr.tobytes(), flags=zmq.NOBLOCK
            )
        except zmq.ZMQError:
            pass

    def latest(self):
        """``(q14, age_seconds)`` for the newest solution, or ``None``.

        ``age`` is what a caller should gate on: the worker going quiet is the
        realistic failure (body tracking dropped, the process died), and it
        looks identical to "the operator is holding still" in the values alone.
        """
        with self._lock:
            if self._q14 is None:
                return None
            return self._q14.copy(), time.monotonic() - self._stamp

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                msg = self._sub.recv()
            except zmq.Again:
                continue
            except (zmq.ContextTerminated, zmq.ZMQError):
                return
            if len(msg) != _RSP_BYTES:
                continue
            q14 = np.frombuffer(
                msg, dtype=np.float32, offset=_REQ_STRUCT.size
            ).astype(np.float64)
            if not np.isfinite(q14).all():
                continue
            with self._lock:
                self._q14 = q14
                self._stamp = time.monotonic()
