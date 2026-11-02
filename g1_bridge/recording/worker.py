"""Recording on its own thread, so writing data never stalls the robot.

LocoManiLab's original HeFT integration put the data exporter in a separate *process* for this reason.
Merging it into the control loop bought one ``teleop_sample()`` call instead of a
ZMQ hop -- and brought the writer's worst case with it. Measured on the collection
box: a normal recorded tick costs ~4 ms against a 20 ms budget, but the first tick
of an episode costs ~38 ms (LeRobot spins up its image-writer threads) and
``save_episode`` blocks for *seconds* while it encodes the mp4. Every one of those
milliseconds is a control tick that does not happen, and on hardware that is felt
as the robot lurching the moment you press 'c'.

So the control loop hands frames to this queue and moves on. The worker owns the
dataset object entirely -- add_frame, save_episode, discard, and the staged-PNG
sweep -- and the loop touches none of it.

Backpressure: the queue is bounded. If the writer ever falls behind for longer
than the queue holds, frames are dropped and counted rather than blocking the
control loop, because a dropped frame costs one row of a dataset and a blocked
tick costs a stumble. In practice it does not fill: recording stops before
``save_episode`` runs, so the drain has the whole save to catch up in.
"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass

from termcolor import colored


@dataclass
class _Save:
    reason: str


@dataclass
class _Discard:
    reason: str


class RecordingWorker:
    """Feeds an ``EpisodeRecorder`` from a background thread."""

    def __init__(self, recorder, max_queued: int = 1500, on_event=None):
        self._recorder = recorder
        #: Same rationale as EpisodeRecorder.on_event: these belong in the
        #: recording pane, not interleaved with the controller's logs.
        self._on_event = on_event or (lambda kind, message: print(message, flush=True))
        self._queue: queue.Queue = queue.Queue(maxsize=max_queued)
        self._stop = threading.Event()
        self._dropped = 0
        self._last_drop_warn = 0.0
        #: True while the worker is inside save_episode (the mp4 encode). The
        #: recording pane turns this into a progress bar -- it is the wait after
        #: pressing 'c', and it used to be invisible.
        self._saving = False
        self._thread = threading.Thread(
            target=self._loop, name="RecordingWorker", daemon=True
        )
        self._thread.start()

    # -- called from the control loop; must never block ------------------- #
    def add_frame(self, frame) -> None:
        if frame is None:
            return
        try:
            self._queue.put_nowait(frame)
        except queue.Full:
            self._dropped += 1
            now = time.monotonic()
            if now - self._last_drop_warn > 5.0:
                self._last_drop_warn = now
                self._on_event(
                    "dropped",
                    colored(
                        f"[collect] recording is behind — dropped {self._dropped} "
                        "frame(s) rather than stalling the control loop",
                        "yellow",
                    ),
                )

    def save(self, reason: str = "operator stop") -> None:
        self._queue.put(_Save(reason))

    def discard(self, reason: str = "operator discard") -> None:
        self._queue.put(_Discard(reason))

    # -- worker ----------------------------------------------------------- #
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if isinstance(item, _Save):
                    self._saving = True
                    try:
                        self._recorder.save(reason=item.reason)
                    finally:
                        self._saving = False
                elif isinstance(item, _Discard):
                    self._recorder.discard(reason=item.reason)
                else:
                    # append, not add_frame: the producer already decided
                    # this frame belongs to the episode (see append_frame).
                    self._recorder.append_frame(item)
            except Exception as exc:  # noqa: BLE001 -- never kill the writer thread
                self._on_event("error", colored(f"[collect] recording error: {exc}", "red"))
            finally:
                self._queue.task_done()

    @property
    def saving(self) -> bool:
        return self._saving

    @property
    def pending(self) -> int:
        return self._queue.qsize()

    @property
    def dropped(self) -> int:
        return self._dropped

    def close(self, timeout_s: float = 120.0) -> None:
        """Drain, then stop. The last episode's video encode happens here."""
        pending = self._queue.qsize()
        if pending:
            self._on_event(
                "flush", colored(f"[collect] flushing {pending} queued frame(s)...", "cyan")
            )
        deadline = time.monotonic() + timeout_s
        while not self._queue.empty() and time.monotonic() < deadline:
            time.sleep(0.05)
        self._stop.set()
        self._thread.join(timeout=5.0)
