"""Client for a GR00T-compatible policy server.

Every VLA served to this stack -- gr00t, fastwam, lingbot-va, starVLA, pi0.5,
PRTS -- speaks the same wire protocol, so one client covers all of them:

    transport   ZMQ REQ  ->  tcp://<host>:<port>  (server binds REP)
    encoding    msgpack-numpy
    request     {"endpoint": str, "data": {...}}
    endpoints   ping | kill | reset | get_action | get_modality_config
    get_action  data={"observation": {...}, "options": None}
                -> [action_dict, info_dict]

Action keys come back either bare (``motion_token``) or prefixed
(``action.motion_token``) depending on the server; :func:`action_field` and
:func:`strip_action_prefix` accept both, because which one you get is a
property of the server implementation, not of the model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

#: Where every server in this stack listens unless told otherwise.
DEFAULT_PORT = 5550


class PolicyClient:
    """Thin request/reply client. One in-flight call at a time (ZMQ REQ).

    Not thread-safe by itself: a REQ socket must see strict send/recv
    alternation. Drive it from one thread -- see ``sim2real.vla.driver``,
    which keeps inference on a single worker for exactly this reason.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = DEFAULT_PORT,
        timeout_ms: int = 20000,
    ) -> None:
        import msgpack_numpy as mnp
        import zmq

        self._mnp = mnp
        self._zmq = zmq
        self.host, self.port, self.timeout_ms = host, port, timeout_ms
        self.ctx = zmq.Context.instance()
        self.sock = None
        self._connect()

    def _connect(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, self.timeout_ms)
        self.sock.setsockopt(self._zmq.SNDTIMEO, self.timeout_ms)
        self.sock.connect(f"tcp://{self.host}:{self.port}")

    def _call(self, request: dict) -> Any:
        self.sock.send(self._mnp.packb(request))
        msg = self.sock.recv()
        if msg == b"ERROR":
            raise RuntimeError(
                f"policy server tcp://{self.host}:{self.port} returned ERROR "
                f"for endpoint {request.get('endpoint')!r}"
            )
        return self._mnp.unpackb(msg, raw=False)

    def ping(self) -> bool:
        """True if the server answered. On failure the socket is rebuilt.

        A REQ socket that timed out is stuck mid-handshake and will refuse the
        next send, so a failed ping must reset it or every later call fails
        too -- which reads as "the server never came up" long after it did.
        """
        try:
            self._call({"endpoint": "ping"})
            return True
        except Exception:
            self._connect()
            return False

    def reset(self, options: dict | None = None) -> Any:
        return self._call({"endpoint": "reset", "data": options or {}})

    def get_modality_config(self) -> Any:
        return self._call({"endpoint": "get_modality_config"})

    def get_action(self, observation: dict, options: dict | None = None) -> dict:
        resp = self._call(
            {
                "endpoint": "get_action",
                "data": {"observation": observation, "options": options},
            }
        )
        if isinstance(resp, (list, tuple)):
            return resp[0]
        return resp

    def kill(self) -> None:
        try:
            self._call({"endpoint": "kill"})
        except Exception:
            pass

    def close(self) -> None:
        if self.sock is not None:
            self.sock.close(linger=0)
            self.sock = None


def strip_action_prefix(action: dict) -> dict:
    """Drop the ``action.`` prefix from every key that has one."""
    return {k.replace("action.", "", 1): v for k, v in action.items()}


def action_field(action: dict, key: str) -> np.ndarray:
    """Fetch ``key`` from an action dict, with or without the prefix."""
    for k in (key, f"action.{key}"):
        value = action.get(k)
        if value is not None:
            return np.asarray(value)
    raise KeyError(
        f"action field {key!r} (or 'action.{key}') not in {sorted(action)}"
    )


def as_chunk(value, dim: int | None = None) -> np.ndarray:
    """Normalise an action field to ``(horizon, dim)``.

    Servers return ``(B, T, D)``, ``(T, D)`` or ``(D,)`` depending on model and
    horizon; downstream code only ever wants to index by time.

    The batch axis is dropped by INDEXING, never by ``np.squeeze``: a 1-D action
    group (amo's ``height``/``torso_vx``/``target_yaw``, almi's ``ang_vel_z``/
    ``base_height``) comes back as ``(1, T, 1)``, and squeezing that gives
    ``(T,)`` -- which then reads as ONE step of T dims. ``VlaDriver`` takes the
    chunk length as the minimum over fields, so that one field pinned every
    chunk to length 1 and the robot re-applied step 0 forever: a closed-loop
    amo/almi policy that looked like "holds still, never starts" (0/20 on a
    checkpoint that reproduces the training actions open-loop to 1e-4).
    """
    arr = np.asarray(value, dtype=np.float32)
    while arr.ndim > 2 and arr.shape[0] == 1:   # (1, T, D) / (1, 1, T, D) -> (T, D)
        arr = arr[0]
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        # (D,) one step -- unless the caller says the field is dim-1/dim-d and
        # this is a horizon of it.
        arr = arr.reshape(1, -1) if dim is None or arr.shape[0] == dim else arr.reshape(-1, dim)
    elif arr.ndim > 2:  # (B>1, T, D): no batch semantics downstream, flatten
        arr = arr.reshape(-1, arr.shape[-1])
    return arr.reshape(-1, arr.shape[-1] if dim is None else dim)
