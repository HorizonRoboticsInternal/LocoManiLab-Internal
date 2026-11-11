"""The closed loop between a VLA policy server and a whole-body controller.

Control runs at 50 Hz; a VLA answers at 2.5-10 Hz. Everything hard about
driving one from the other lives in that gap:

**Chunked replanning.** Each ``get_action`` returns a horizon of actions, not
one. The loop walks that chunk until the next one lands.

**Latency compensation.** By the time a chunk arrives, the world has moved on
by however long inference took. Starting at index 0 replays stale actions; the
fix is to enter the chunk at ``round(delay * control_freq)``. Capped by
``max_latency_skip``, because a pathological delay would otherwise skip the
robot far into a plan made for a state it is no longer in.

**Chunk-boundary smoothing.** Consecutive chunks disagree at the seam. An EMA
on the first action of each new chunk (``ema_alpha < 1``) trades a little
responsiveness for not stepping the controller's input discontinuously.

**Async inference** (``async_inference=True``). Real deployment cannot stall a
50 Hz control loop on a 400 ms model call, so inference runs on a worker and
the loop keeps publishing from the current chunk. A non-realtime simulator has
the opposite preference -- determinism -- so it re-plans synchronously and this
defaults off.

The driver never touches the robot. It produces the action to apply and hands
it to an :mod:`~sim2real.vla.adapters` adapter, which knows what the
controller consumes. That split is why one loop serves every combination.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from loguru import logger

from sim2real.vla.adapters import ActionAdapter, check_action_keys
from sim2real.vla.client import as_chunk, strip_action_prefix


@dataclass
class VlaDriverConfig:
    #: Controller tick rate (Hz). Actions are consumed at this rate.
    control_freq: float = 50.0
    #: How often to request a fresh chunk (Hz). 2.5 for PyTorch serving,
    #: 10.0 with TRT. Raising it past what the server can actually sustain
    #: makes the policy re-plan faster than it can act on its own plan.
    inference_rate: float = 2.5
    #: Chunk length the model emits. Only bounds the latency skip; the chunk
    #: is indexed as returned.
    action_horizon: int = 64
    #: EMA weight on the first action of a new chunk. 1.0 = snap.
    ema_alpha: float = 1.0
    #: Cap on the latency-compensated entry index, in control steps.
    max_latency_skip: int = 3
    #: Run inference on a worker thread instead of blocking the loop.
    async_inference: bool = False
    #: Fields to interpolate at chunk boundaries. Defaults to every field.
    ema_fields: tuple[str, ...] | None = None


class VlaDriver:
    """Owns chunk state; you own the loop.

    Call :meth:`step` once per control tick with a freshly built observation.
    It returns the action dict for this tick (already indexed and smoothed),
    or None before the first chunk has arrived.
    """

    def __init__(
        self,
        client,
        adapter: ActionAdapter | None = None,
        config: VlaDriverConfig | None = None,
        observation_fn: Callable[[], dict] | None = None,
        command_space: str | None = None,
    ) -> None:
        """``adapter`` is only needed when the caller wants :meth:`step` to
        apply actions itself (``apply_to=``). A host that applies actions
        through its own path (the g1_bridge backends build their own adapter,
        with host-supplied constructor args this driver cannot know) should
        pass ``command_space`` instead -- the driver still validates the
        policy's action keys against it, without carrying a second adapter
        whose ``apply()`` never runs and whose construction may require
        arguments (e.g. ref_qpos's ``frame_builder``) the caller cannot supply.
        """
        if adapter is None and command_space is None:
            raise ValueError("give VlaDriver an adapter or a command_space")
        self.client = client
        self.adapter = adapter
        self.command_space = command_space if command_space is not None else adapter.command_space
        self.cfg = config or VlaDriverConfig()
        self._reference_windows = self.command_space == "holomotion_reference_v1"
        if self._reference_windows and (
            self.cfg.control_freq != 50 or self.cfg.inference_rate != 3.125
            or self.cfg.ema_alpha != 1 or self.cfg.async_inference
        ):
            raise ValueError("HoloMotion requires 50 Hz / 16-step synchronous execution without action EMA")
        self._reference_step = 0
        self.observation_fn = observation_fn

        self.replan_steps = max(
            1, int(round(self.cfg.control_freq / max(self.cfg.inference_rate, 1e-6)))
        )
        self.inference_interval = 1.0 / max(self.cfg.inference_rate, 1e-6)

        self._chunk: dict[str, np.ndarray] | None = None
        self._chunk_len = 0
        self._idx = 0
        self._new_chunk = False
        self._prev: dict[str, np.ndarray] | None = None
        self._keys_checked = False
        self.n_inferences = 0

        self._worker: threading.Thread | None = None
        self._req: queue.Queue = queue.Queue(maxsize=1)
        self._res: queue.Queue = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._busy = threading.Event()
        self._last_request_t = -1e9
        if self.cfg.async_inference:
            self._start_worker()

    # -- lifecycle --------------------------------------------------------
    def reset(self) -> None:
        """Drop chunk state at an episode boundary."""
        self._reference_step = 0
        self._chunk = None
        self._chunk_len = 0
        self._idx = 0
        self._new_chunk = False
        self._prev = None
        self._last_request_t = -1e9
        self.n_inferences = 0
        if self.adapter is not None:
            self.adapter.reset()
        while not self._res.empty():
            try:
                self._res.get_nowait()
            except queue.Empty:
                break

    def close(self) -> None:
        self._stop.set()
        if self._worker is not None:
            try:
                self._req.put_nowait(None)
            except queue.Full:
                pass
            self._worker.join(timeout=2.0)
            self._worker = None

    # -- inference --------------------------------------------------------
    def _start_worker(self) -> None:
        def loop():
            while not self._stop.is_set():
                try:
                    item = self._req.get(timeout=0.1)
                except queue.Empty:
                    continue
                if item is None:
                    break
                observation, started = item
                self._busy.set()
                try:
                    action = self.client.get_action(observation)
                    self._res.put((action, started))
                except Exception as exc:  # keep the control loop alive
                    logger.error("VLA inference failed: {}", exc)
                finally:
                    self._busy.clear()

        self._worker = threading.Thread(target=loop, daemon=True)
        self._worker.start()

    def _ingest(self, action: dict, started: float) -> None:
        if not self._keys_checked:
            check_action_keys(self.command_space, action)
            self._keys_checked = True
        chunk = {k: as_chunk(v) for k, v in strip_action_prefix(action).items()
                 if v is not None and np.asarray(v).dtype != object}
        if not chunk:
            raise ValueError(f"policy returned no usable action arrays: {sorted(action)}")
        if self._reference_windows:
            reference = chunk.get("reference")
            if reference is None or reference.shape != (50, 87) or not np.isfinite(reference).all():
                raise ValueError("HoloMotion server must return finite reference[50,87]")
            if np.any(np.abs(np.linalg.norm(reference[:, 61:65], axis=1) - 1) > .1):
                raise ValueError("HoloMotion reference has invalid root quaternions")
        self._chunk = chunk
        self._chunk_len = min(len(v) for v in chunk.values())
        # Enter the chunk where the world already is, not at its start -- but
        # ONLY when the world actually moved while the model was thinking.
        # Under synchronous inference the caller is blocked, so nothing
        # advanced: skipping there would silently discard the first actions of
        # every chunk (at 2.5 Hz, 3 of every 20 steps, right after each
        # re-plan) and measurably costs task success.
        if self.cfg.async_inference:
            delay = max(0.0, time.monotonic() - started)
            skip = int(np.clip(round(delay * self.cfg.control_freq), 0, self._chunk_len - 1))
            self._idx = min(skip, self.cfg.max_latency_skip)
        else:
            self._idx = 0
        self._new_chunk = True
        self.n_inferences += 1

    # -- per control tick -------------------------------------------------
    def step(self, observation: dict | None = None, *, apply_to=None) -> dict | None:
        """Advance one control tick.

        Args:
            observation: this tick's observation. Optional if the driver was
                given an ``observation_fn``; it is only ever built when a
                request is actually about to be sent.
            apply_to: when given, the adapter applies this tick's action to it.

        Returns the action for this tick, or None while waiting for the first
        chunk (async mode only -- a synchronous driver always has one).
        """
        now = time.monotonic()
        needs_chunk = self._chunk is None or self._idx >= self.replan_steps

        if self.cfg.async_inference:
            while True:
                try:
                    action, started = self._res.get_nowait()
                except queue.Empty:
                    break
                self._ingest(action, started)
            due = now - self._last_request_t >= self.inference_interval
            if (self._chunk is None or due) and not self._busy.is_set():
                obs = observation if observation is not None else self._observe()
                try:
                    self._req.put_nowait((obs, now))
                    self._last_request_t = now
                except queue.Full:
                    pass
        elif needs_chunk:
            obs = observation if observation is not None else self._observe()
            started = time.monotonic()
            self._ingest(self.client.get_action(obs), started)

        if self._chunk is None:
            return None

        i = min(self._idx, self._chunk_len - 1)
        if self._reference_windows:
            current = {
                "reference_window": self._chunk["reference"][i:i + 11].copy(),
                "reference_step": self._reference_step,
                "reference_new_chunk": self._new_chunk,
            }
            self._new_chunk = False
            self._idx += 1
            self._reference_step += 1
            if apply_to is not None:
                raise ValueError("Deliver HoloMotion reference windows through composite.set_vla_action")
            return current
        current = {k: v[i].copy() for k, v in self._chunk.items()}
        if self._new_chunk and self._prev is not None and self.cfg.ema_alpha < 1.0:
            a = self.cfg.ema_alpha
            fields = self.cfg.ema_fields or tuple(current)
            for k in fields:
                if k in current and k in self._prev and current[k].shape == self._prev[k].shape:
                    current[k] = a * current[k] + (1.0 - a) * self._prev[k]
        self._new_chunk = False
        self._prev = {k: v.copy() for k, v in current.items()}
        self._idx += 1

        if apply_to is not None:
            if self.adapter is None:
                raise ValueError(
                    "VlaDriver.step(apply_to=...) needs the adapter this "
                    "driver was constructed without -- pass one, or apply "
                    "the returned action yourself."
                )
            self.adapter.apply(apply_to, current)
        return current

    def _observe(self) -> dict:
        if self.observation_fn is None:
            raise ValueError(
                "VlaDriver.step needs an observation, or an observation_fn "
                "given at construction."
            )
        return self.observation_fn()


class ReplayPolicy:
    """Stands in for a policy server, replaying recorded actions.

    The single most useful smoke test in this stack: it removes the model from
    the picture entirely, so if the robot does the recorded thing, everything
    downstream of the policy -- adapter, command space, controller -- is
    correct, and if it does not, the model was never the problem.
    """

    def __init__(self, frames: list[dict[str, np.ndarray]]) -> None:
        self.frames = frames
        self.i = 0

    def get_action(self, observation: dict, options: dict | None = None) -> dict:
        frame = self.frames[min(self.i, len(self.frames) - 1)]
        self.i += 1
        return {k: np.asarray(v)[None] for k, v in frame.items()}

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None


class ZeroPolicy:
    """Zeros of the right shape: does the controller hold itself up at all?

    Separates "the controller stack is broken" from "the policy is bad" -- a
    robot that falls over under zeros has a problem no checkpoint will fix.
    """

    def __init__(self, shapes: dict[str, int]) -> None:
        self.shapes = shapes

    def get_action(self, observation: dict, options: dict | None = None) -> dict:
        return {k: np.zeros((1, d), np.float32) for k, d in self.shapes.items()}

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        return None
