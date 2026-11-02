"""What the collector is actually receiving from the operator, live.

    uv run python -m g1_bridge.teleop_monitor

The tmux session gives teleop its own pane for every backend, but what belongs in
that pane differs by how the backend gets its PICO input:

* heft / scalebfm / almi read a GMR-retargeted body over ZMQ, so the retarget
  publisher gets a pane of its own -- but its log only says what the *publisher*
  sent, so this still runs alongside it, saying what the collector received.
* sonic / amo open their own PICO connection **inside the collector**
  (``ProcessPicoVRSource``, a multiprocessing child), so there is no separate
  process to watch and no socket to subscribe to. The collector publishes what it
  sees instead, and this renders it.

Same pane, same question either way: is the operator's input arriving, and what
does the robot think it says. Without this, "the chord does nothing" on sonic/amo
has no observable middle -- you can see the headset is connected and you can see
the robot not moving, with nothing in between.

The sticks are shown for the GMR backends' sake in particular. They come through
``GMRButtonRelay``, a channel that once carried the buttons but not the axes;
heft/scalebfm never noticed (they do not steer), while on almi the two sticks
*are* the entire locomotion command. ``sticks --`` here means the relay is
publishing no axes at all, which is a different fault from a stick reading 0.00.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import zmq
from termcolor import colored

#: Port the collector publishes its teleop status on. Deliberately not one of the
#: PICO/GMR data ports: this is a status feed, and nothing controls anything from it.
DEFAULT_STATUS_PORT = 5581

#: What a stick reads as when the collector published no axis for it. Same width
#: as a rendered pair, so the line does not jitter, and visibly not a number --
#: "no axes on the wire" must not look like "stick centred".
_NO_AXIS = "    --,--    "  # 13 chars, as wide as "(+0.42,-0.10)"


def _stick(axis) -> str:
    """``(+0.42,-0.10)`` for a live stick, ``--,--`` when there is no axis.

    Also the fallback for a collector predating the axes in the status payload,
    which is why a missing key is not an error here.
    """
    try:
        x, y = axis
        return f"({float(x):+.2f},{float(y):+.2f})"
    except (TypeError, ValueError):
        return _NO_AXIS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="g1_bridge.teleop_monitor")
    parser.add_argument("--connect", default=f"tcp://127.0.0.1:{DEFAULT_STATUS_PORT}")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args(argv)

    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.RCVTIMEO, 0)
    socket.connect(args.connect)
    print(f"teleop status from {args.connect} (Ctrl-C to stop)\n", flush=True)

    latest = None
    last_seen = 0.0
    try:
        while True:
            try:
                latest = json.loads(socket.recv_string(zmq.NOBLOCK))
                last_seen = time.monotonic()
            except zmq.Again:
                pass
            time.sleep(0.02)

            if latest is None:
                if time.monotonic() - last_seen > 3.0:
                    print(
                        colored(
                            "\r  waiting for the collector to publish teleop status..."
                            "   ",
                            "yellow",
                        ),
                        end="",
                        flush=True,
                    )
                    last_seen = time.monotonic()
                continue

            age = time.monotonic() - last_seen
            if age > 2.0:
                print(
                    colored(f"\r  collector silent for {age:.0f}s   ", "red"),
                    end="",
                    flush=True,
                )
                time.sleep(args.interval)
                continue

            buttons = latest.get("buttons") or {}
            pressed = "".join(k.upper() for k in "abxy" if buttons.get(k)) or "-"
            sample_age = latest.get("sample_age_s")
            stale = sample_age is None or sample_age > 0.5
            print(
                "\r  "
                + colored(f" {latest.get('source', '?'):20s} ", "white", "on_blue")
                + f"  buttons [{pressed:4s}]"
                + f"  trig L={latest.get('left_trigger', 0.0):.2f} "
                f"R={latest.get('right_trigger', 0.0):.2f}"
                + f"  sticks {_stick(latest.get('left_axis'))}"
                f"/{_stick(latest.get('right_axis'))}"
                + f"  mode {latest.get('mode', '?'):8s}"
                + ("  " + colored("ARMED", "green") if latest.get("armed") else "  safe ")
                + (
                    colored(f"  sample stale ({sample_age})", "red")
                    if stale
                    else f"  sample {sample_age * 1000:.0f} ms"
                )
                + "   ",
                end="",
                flush=True,
            )
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
