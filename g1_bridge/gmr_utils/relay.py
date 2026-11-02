"""Owns the ZMQ subscription to a GMR publisher (gmr_subprocess.py) directly,
instead of letting sim2real's own RealtimeMotionBuffer auto-connect to it.

Why: WBT's own composite (wbt_g1_composite.WholeBodyG1Composite) never lets
its inner SONIC controller run on a raw, unmanaged live reference -- in OFF
it explicitly feeds a frozen default-pose reference into the SAME policy
that handles engaged tracking (_update_default_pose_reference), so the
policy is always doing active, trained balance control against a sane
target, never naive fixed-pose PD and never a half-finished custom
interpolation hack (both tried here first; both worse than this pattern in
practice -- naive-PD/interpolation-then-handoff caused instability).
Reproducing that for the GMR-driven backends means this composite needs to
choose, per tick, whether the policy sees the live GMR feed or a frozen
default -- which isn't possible if sim2real's own RealtimeMotionBuffer
manages the ZMQ subscription internally (motion_mode="zmq" in
bridge_runtime.py). So instead: this class holds the one real subscription
(CONFLATE=1, always the latest message), and composite.py pushes either
``latest()`` or ``default()`` into a push-mode BridgeTracking
(motion_mode="push", same push path Flavor A/SONIC already uses) depending
on ModeManager state.
"""

from __future__ import annotations

import json
import threading

import zmq


class GMRRelay:
    def __init__(self, connect: str = "tcp://127.0.0.1:28701"):
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.SUB)
        self._sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._sock.setsockopt(zmq.CONFLATE, 1)
        self._sock.setsockopt(zmq.RCVTIMEO, 200)
        self._sock.connect(connect)
        self._lock = threading.Lock()
        self._latest: dict | None = None
        self._default: dict | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="GMRRelay")
        self._thread.start()

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                raw = self._sock.recv()
            except zmq.Again:
                continue
            except zmq.ContextTerminated:
                return
            try:
                payload = json.loads(raw)
            except (ValueError, TypeError):
                continue
            with self._lock:
                self._latest = payload
                if self._default is None:
                    self._default = payload

    def latest(self) -> dict | None:
        """Most recent payload, or None if nothing received yet."""
        with self._lock:
            return self._latest

    def default(self) -> dict | None:
        """The first payload ever received, held fixed -- a frozen reference
        for the policy to hold station against while disengaged (OFF),
        mirroring WBT's own _update_default_pose_reference pattern. None
        until the first message arrives."""
        with self._lock:
            return self._default

    def stop(self, timeout: float = 1.0) -> None:
        self._stop.set()
        self._thread.join(timeout=timeout)
        self._sock.close(linger=0)
