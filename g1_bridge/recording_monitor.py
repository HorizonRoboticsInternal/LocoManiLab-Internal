"""The recording pane: status bar + event log, fed by the collector.

    uv run python -m g1_bridge.recording_monitor

LocoManiLab's original HeFT integration ran its data exporter as its own process, so everything about
recording -- "started", "saved episode 3 (412 frames)", "discarded" -- appeared in
the exporter's pane and nowhere else. Here the recorder is a thread inside the
collector (it has to be: it reads ``session.teleop_sample()`` in-process), so its
prints would land in the controller's pane, interleaved with 50 Hz policy logs.
That is the pane an operator watches for the robot, and recording noise in it
means the one line you needed scrolls away.

So the collector publishes recording state and events on a socket instead, and
this renders them: a colored status bar that says at a glance whether you are
recording, and the event log underneath. Read-only -- nothing here controls
anything; the keys are in the keyboard pane.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import zmq
from termcolor import colored

#: Port the collector publishes recording status/events on.
DEFAULT_RECORDING_PORT = 5582


def _term_width(default: int = 80) -> int:
    import shutil

    return max(40, shutil.get_terminal_size((default, 24)).columns)


def _bar(fraction: float, width: int, colour: str) -> str:
    """A filled bar: ``fraction`` of ``width`` painted, the rest dim."""
    filled = max(0, min(width, int(round(fraction * width))))
    return colored(" " * filled, "white", f"on_{colour}") + colored(
        " " * (width - filled), "white", "on_grey"
    )


def _status_bar(status: dict, age: float) -> list[str]:
    """The pane's bottom block: one filled colour bar plus a progress bar.

    A full-width bar rather than a line of text because this is read from across
    the room, mid-teleop, with both hands busy: colour and length carry at a
    glance, a sentence does not. Red while recording, grey while idle, and the
    progress bar shows the one thing that used to be invisible -- how much of the
    episode is still queued for writing after you press 'c'.
    """
    width = _term_width()
    recording = bool(status.get("recording"))
    episodes = int(status.get("episodes", 0))
    frames = int(status.get("frames", 0))
    queued = int(status.get("queued", 0))
    dropped = int(status.get("dropped", 0))
    saving = bool(status.get("saving")) or queued > 0

    label = (
        f" ● REC   episode {episodes + 1}   {frames:5d} frames   "
        f"{frames / 50.0:5.1f}s "
        if recording
        else f" ○ IDLE   {episodes} episode(s) saved   {frames:5d} frames "
    )
    if dropped:
        label += f"  DROPPED {dropped} "
    label += " " * max(0, width - len(label))
    colour = "red" if recording else ("blue" if saving else "grey")
    bar = colored(label[:width], "white", f"on_{colour}", attrs=["bold"])

    # Progress: while recording it fills with the episode's own length (a full
    # bar means "this is already a long demo"); while saving it drains with the
    # write queue, which is exactly the wait after pressing 'c'.
    if saving and not recording:
        peak = max(int(status.get("queue_peak", queued)), 1)
        done = 1.0 - (queued / peak)
        detail = f"  saving  {queued:4d} frame(s) left "
        progress = _bar(done, max(10, width - len(detail) - 2), "blue")
    elif recording:
        target = float(status.get("progress_target_s", 30.0)) * 50.0
        detail = f"  buffer  {queued:4d} queued      "
        progress = _bar(min(1.0, frames / target), max(10, width - len(detail) - 2), "green")
    else:
        detail = "  idle                     "
        progress = _bar(0.0, max(10, width - len(detail) - 2), "grey")

    lines = [bar, detail + progress]
    if age > 3.0:
        lines.append(colored(f"  collector silent for {age:.0f}s", "red"))
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="g1_bridge.recording_monitor")
    parser.add_argument("--connect", default=f"tcp://127.0.0.1:{DEFAULT_RECORDING_PORT}")
    args = parser.parse_args(argv)

    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.RCVTIMEO, 0)
    socket.connect(args.connect)

    print(colored("  recording monitor", "cyan", attrs=["bold"]), flush=True)
    print(f"  {args.connect}\n", flush=True)

    status: dict = {}
    last_seen = time.monotonic()
    dataset_shown = False
    queue_peak = 0
    lines_drawn = 0
    try:
        while True:
            while True:
                try:
                    message = json.loads(socket.recv_string(zmq.NOBLOCK))
                except zmq.Again:
                    break
                last_seen = time.monotonic()
                if message.get("type") == "event":
                    # Events scroll above the bar; the bar is redrawn under them.
                    sys.stdout.write("\r\033[K" + message.get("message", "") + "\n")
                    lines_drawn = 0
                else:
                    status = message
                    queue_peak = (
                        max(queue_peak, int(status.get("queued", 0)))
                        if status.get("queued")
                        else 0
                    )
                    status["queue_peak"] = queue_peak
                    if not dataset_shown and status.get("dataset"):
                        dataset_shown = True
                        sys.stdout.write(f"  dataset: {status['dataset']}\n")

            age = time.monotonic() - last_seen
            block = (
                _status_bar(status, age)
                if status
                else [colored("  waiting for the collector...", "yellow")]
            )
            # Redraw in place: move up over the previous block, then rewrite it.
            if lines_drawn:
                sys.stdout.write(f"\033[{lines_drawn}A")
            for line in block:
                sys.stdout.write("\r\033[K" + line + "\n")
            lines_drawn = len(block)
            sys.stdout.flush()
            time.sleep(0.2)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
