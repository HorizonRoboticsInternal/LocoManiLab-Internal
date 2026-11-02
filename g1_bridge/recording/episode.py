"""Episode bookkeeping: append / save / discard, and the ``max_episodes`` stop.

Host-agnostic on purpose, because *when* an episode ends is exactly what differs
between hosts and nothing else does:

* sim: the task's own ``_check_success()`` rising edge saves the episode; ``Q``
  discards it (``robocasa.utils.g1_data_collection.EpisodeRecorder``).
* hardware: there is no success signal at all, so the operator says so, from the
  keyboard only -- ``c`` starts/stops, ``x`` discards. There is no PICO gesture
  for this on purpose; see ``g1_bridge/hosts/real.py``.

Both drive this class; the frames, the staged-PNG cleanup and the
interrupted-episode handling are shared.
"""

from __future__ import annotations

from termcolor import colored

from g1_bridge.recording.dataset import purge_image_cache


class EpisodeRecorder:
    """Append frames to a LeRobot dataset and manage episode boundaries."""

    def __init__(self, dataset, lang, teleop_keys, max_episodes=10**9, on_event=None,
                 on_episode_start=None):
        self.dataset = dataset
        self.lang = lang
        self.teleop_keys = teleop_keys
        self.max_episodes = max_episodes
        self.n_saved = 0
        self.ep_frames = 0
        self.episode_saved = False
        #: Called at every episode boundary (see reset_episode). Collectors wire
        #: it to ``TeleopSession.on_episode_start``, which is how a backend that
        #: records episode-relative quantities learns where "the start" is --
        #: ``Sim2RealBackend.reset`` cannot tell it, because on hardware one
        #: reset spans every episode of the run.
        self._on_episode_start = on_episode_start
        #: Where recording events go. Default: this process's stdout, which is
        #: right for the sim collector (one process, one terminal). The real
        #: collector passes a callback that publishes them instead, so they land
        #: in the recording pane rather than interleaved with the controller's
        #: 50 Hz logs -- the separation the heft fork got for free by running its
        #: exporter as its own process.
        self._on_event = on_event or (lambda kind, message: print(message, flush=True))

    def _event(self, kind: str, message: str) -> None:
        self._on_event(kind, message)

    def reset_episode(self):
        self.ep_frames = 0
        self.episode_saved = False
        if self._on_episode_start is not None:
            self._on_episode_start()

    def append_frame(self, frame, task=None):
        """Append one already-built frame, unconditionally.

        The gate on "is the operator recording" belongs to whoever produces the
        frame, not here: with a RecordingWorker the producer decided one queue-
        length ago, and re-checking on the writer thread would silently drop the
        tail of every episode -- exactly the frames where the demo ends.
        """
        if frame is None:
            return False
        self.dataset.add_frame(frame, task=self.lang if task is None else task)
        self.ep_frames += 1
        return True

    def add_frame(self, frame, task=None):
        """Append one frame if this recorder is currently accepting them."""
        return self.append_frame(frame, task=task)

    def save(self, reason="SUCCESS"):
        """Close the current episode. Raises KeyboardInterrupt at ``max_episodes``."""
        if self.episode_saved or self.ep_frames == 0:
            return False
        self.dataset.save_episode()
        self.n_saved += 1
        self.episode_saved = True
        # Sweep the staged PNGs now, not at close: LeRobot removes the episode's
        # own dir when it encodes the video, but an interrupted or partially
        # written one leaves frames behind, and a finished dataset should carry
        # videos/ only. Doing it per episode also keeps the tree small enough to
        # copy off the robot while the session is still running.
        purge_image_cache(self.dataset)
        self._event(
            "saved",
            colored(
                f"[collect] {reason} — saved episode {self.n_saved} "
                f"({self.ep_frames} frames).",
                "green",
            ),
        )
        if self.n_saved >= self.max_episodes:
            raise KeyboardInterrupt
        return True

    def discard(self, reason="Q"):
        if self.ep_frames > 0:
            # clear_episode_buffer also deletes this episode's staged PNGs.
            try:
                self.dataset.clear_episode_buffer()
            except OSError:
                pass  # ignore rmtree ENOTEMPTY race / stale NFS handles
            self._event(
                "discarded",
                colored(
                    f"[collect] {reason} — discarded {self.ep_frames} frames", "yellow"
                ),
            )
        self.reset_episode()

    def close(self):
        # An episode still in the buffer here was interrupted (Ctrl-C) rather than
        # saved or discarded: drop it, so its staged PNGs go with it.
        if self.ep_frames > 0 and not self.episode_saved:
            self.discard(reason="interrupted")
        if hasattr(self.dataset, "stop_image_writer"):
            self.dataset.stop_image_writer()
        purge_image_cache(self.dataset)
        self._event(
            "done",
            colored(
                f"[collect] done — {self.n_saved} episode(s) saved at "
                f"{self.dataset.root}",
                "green",
            ),
        )


class OperatorEpisodeRecorder(EpisodeRecorder):
    """Operator-driven episodes, for hosts with no success signal (hardware).

    ``toggle`` starts an episode and, on the next press, saves it; ``discard``
    drops whatever has been recorded since the last start. Frames are only
    accepted while an episode is open, so nothing gets recorded before the
    operator says "now".
    """

    def __init__(self, dataset, lang, teleop_keys, max_episodes=10**9, on_event=None,
                 on_episode_start=None):
        super().__init__(dataset, lang, teleop_keys, max_episodes, on_event=on_event,
                         on_episode_start=on_episode_start)
        self.recording = False

    def add_frame(self, frame, task=None):
        if not self.recording:
            return False
        return self.append_frame(frame, task=task)

    @property
    def status(self) -> dict:
        """What the status line shows: recording?, episodes saved, frames so far."""
        return {
            "recording": self.recording,
            "episodes": self.n_saved,
            "frames": self.ep_frames,
        }

    def toggle(self):
        """Start (or stop+save) an episode. Returns True when one was saved."""
        if not self.recording:
            self.recording = True
            self.reset_episode()
            self._event("started", colored("[collect] recording STARTED", "green"))
            return False
        self.recording = False
        if self.ep_frames == 0:
            self._event(
                "empty",
                colored("[collect] recording stopped — nothing recorded", "yellow"),
            )
            return False
        return self.save(reason="operator stop")

    def discard(self, reason="operator discard"):
        self.recording = False
        super().discard(reason=reason)
