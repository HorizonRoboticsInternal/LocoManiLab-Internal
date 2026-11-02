"""Publish the G1's first-person frame on a ZMQ PUB socket.

Why this exists: during teleoperation the operator wears a PICO headset, but the
sim's ego view only ever appeared in a small Tk window on the workstation screen
(``EgoCameraPreview`` in ``robocasa/demos/demo_empty_teleop.py``) -- so the
operator had to look away from the headset to see what the robot sees. On the
real robot that view comes from the Orin's video sender; there was no sim-side
equivalent. This is one half of it: the frames leave the process here, and
LocoManiLab's ``locomanilab.runtime.headset_video`` encodes them to the headset.

The wire format is deliberately NOT a new one. It is
``sim2real.camera.messages``, the same format the robot's own camera server
publishes on port 5555:

    {"timestamps": {name: float}, "images": {name: <JPEG bytes>}}

so the existing viewer works on this endpoint unchanged::

    python -m g1_bridge.camera_monitor --connect tcp://127.0.0.1:5557

Off by default, and entirely env-var driven: no CLI flag has to be threaded
through collect_demos_g1 / collect_fixed_task / g1_data_collection to reach the
control loop.

    LOCOMANILAB_EGO_STREAM          bind endpoint, e.g. tcp://127.0.0.1:5557 (or just
                                   "5557"). Unset/empty -> this module does nothing.
    LOCOMANILAB_EGO_STREAM_KEY      image name in the payload  [default: ego_view]
    LOCOMANILAB_EGO_STREAM_STRIDE   publish every Nth frame    [default: 1]
    LOCOMANILAB_EGO_STREAM_QUALITY  JPEG quality 1-100         [default: 85]

The corresponding legacy ``SOMA_EGO_STREAM*`` variables remain fallbacks only
when the new variable is unset. An explicitly empty new endpoint disables
publishing even if the legacy endpoint is set.

Every failure mode degrades to "print one line, publish nothing" -- this must
never be able to take a collection session down with it, the same contract
``EgoCameraPreview`` holds.
"""

from __future__ import annotations

import os
import time

import numpy as np
from termcolor import colored

ENV_ENDPOINT = "LOCOMANILAB_EGO_STREAM"
ENV_KEY = "LOCOMANILAB_EGO_STREAM_KEY"
ENV_STRIDE = "LOCOMANILAB_EGO_STREAM_STRIDE"
ENV_QUALITY = "LOCOMANILAB_EGO_STREAM_QUALITY"

_LEGACY_ENV_NAMES = {
    ENV_ENDPOINT: "SOMA_EGO_STREAM",
    ENV_KEY: "SOMA_EGO_STREAM_KEY",
    ENV_STRIDE: "SOMA_EGO_STREAM_STRIDE",
    ENV_QUALITY: "SOMA_EGO_STREAM_QUALITY",
}


def _environment_value(name: str, default: str) -> str:
    """Prefer the current variable, including an explicitly empty value."""
    return os.environ.get(name, os.environ.get(_LEGACY_ENV_NAMES[name], default))


DEFAULT_KEY = "ego_view"
DEFAULT_QUALITY = 85


def _normalise_endpoint(value: str) -> str:
    """Accept ``5557``, ``127.0.0.1:5557`` or a full ``tcp://...`` endpoint."""
    value = value.strip()
    if "://" in value:
        return value
    if ":" in value:
        return f"tcp://{value}"
    return f"tcp://127.0.0.1:{value}"


class EgoFramePublisher:
    """Non-blocking PUB socket for one named camera stream."""

    def __init__(self, endpoint: str, key: str = DEFAULT_KEY, stride: int = 1,
                 quality: int = DEFAULT_QUALITY):
        import zmq  # local: only this optional feature needs pyzmq

        from sim2real.camera.messages import encode_camera_frame

        self._encode = encode_camera_frame
        self._key = key
        self._stride = max(1, int(stride))
        self._quality = int(quality)
        self._i = 0
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.PUB)
        # Bounded queue + DONTWAIT + no linger: a slow or absent subscriber must
        # cost the control loop nothing. Same policy as the robot sender's own
        # publisher (SNDHWM 5, LINGER 0, ZMQ_DONTWAIT).
        self._sock.setsockopt(zmq.SNDHWM, 2)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(endpoint)
        self._noblock = zmq.NOBLOCK
        print(colored(
            f"[ego-stream] publishing {key!r} on {endpoint} "
            f"(every {self._stride} frame(s), JPEG q{self._quality})", "cyan"))

    @classmethod
    def maybe(cls) -> "EgoFramePublisher | None":
        """Build one from the environment, or return None (the normal case)."""
        raw = _environment_value(ENV_ENDPOINT, "").strip()
        if not raw:
            return None
        try:
            return cls(
                _normalise_endpoint(raw),
                key=_environment_value(ENV_KEY, DEFAULT_KEY),
                stride=int(_environment_value(ENV_STRIDE, "1")),
                quality=int(_environment_value(ENV_QUALITY, str(DEFAULT_QUALITY))),
            )
        except Exception as exc:
            print(colored(
                f"[ego-stream] disabled ({type(exc).__name__}: {exc})", "yellow"))
            return None

    def publish(self, rgb: np.ndarray) -> None:
        """Send one upright RGB frame. Never raises."""
        if rgb is None:
            return
        self._i += 1
        if self._i % self._stride != 0:
            return
        try:
            payload = self._encode(
                {self._key: np.ascontiguousarray(rgb)},
                {self._key: time.time()},
                jpeg_quality=self._quality,
            )
            self._sock.send(payload, self._noblock)
        except Exception as exc:
            # zmq.Again (queue full) is routine back-pressure, not a fault; only
            # a real error is worth disabling ourselves over.
            if type(exc).__name__ == "Again":
                return
            print(colored(
                f"[ego-stream] error ({type(exc).__name__}: {exc}); disabling",
                "yellow"))
            self.close()

    def close(self) -> None:
        sock, self._sock = getattr(self, "_sock", None), None
        if sock is not None:
            try:
                sock.close(0)
            except Exception:
                pass
        # A closed publisher must stay harmless if publish() is called again.
        self.publish = lambda *_a, **_k: None  # type: ignore[assignment]
