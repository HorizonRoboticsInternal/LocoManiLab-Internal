"""Keyboard controller as its own process, published over ZMQ.

    uv run python -m g1_bridge.keyboard_control            # collection banner

Ported from LocoManiLab's original HeFT integration (``vla/keyboard_control.py``), including the port and
the key vocabulary, so an operator moving between the two stacks keeps the same
window: the keys live in their own tmux pane, that pane holds the focus, and the
collector reads them over ZMQ instead of owning the terminal.

Why a separate process at all, when the collector could read its own stdin (and
does, with ``--keyboard local``): the tmux session puts five things on screen, and
whichever pane owns the keyboard is the one the operator must keep focused. Making
that a pane whose *only* job is keys means focus never has to move back to a
scrolling log to press 'x'.

Keys -- recording only, exactly the fork's ``--mode record`` set:

===========  ==========================================================
c / Enter    start an episode / stop it and save
x            discard the episode in progress
q / Ctrl-C   quit this controller (the collector keeps running)
===========  ==========================================================

Deliberately nothing else. The keys that move the robot -- the startup gate,
arming, zero torque -- live on the collector's own terminal, where the prompts
they answer are printed. Splitting them was not cosmetic: this pane publishes
over a socket any process on the machine can bind to, and "the robot armed
because something else typed into a tmux pane" is not a failure mode worth
keeping. It also matches the fork, where the recording pane only ever spoke
c / x / q.
"""

from __future__ import annotations

import argparse
import select
import sys
import termios
import time
import tty

import zmq
from termcolor import colored

#: Same default as the heft fork's keyboard bus, so the two are interchangeable.
DEFAULT_KEYBOARD_PORT = 5580

#: The only keys this pane publishes. See the module docstring.
_RECORD_KEYS = frozenset({"c", "x", "enter"})

BANNER = """\
Keyboard control — recording
----------------------------
  c / ENTER   Toggle recording  (start episode  |  stop & save)
  x           Discard the current episode
  q / Ctrl-C  Quit this controller (does not stop the collector)

  Robot control (start gate, arming, e-stop) is in the CONTROLLER pane,
  where its prompts are printed. The PICO chord A+B+X+Y arms and engages.
"""


class ZmqKeyController:
    """The collector-side reader: sim2real's ControllerBase surface over ZMQ.

    Same three methods ``BasePolicy``/``RealRobotRunner`` expect from a local
    keyboard controller, so the collector cannot tell the difference.
    """

    name = "zmq-keyboard"

    def __init__(self, connect: str = f"tcp://127.0.0.1:{DEFAULT_KEYBOARD_PORT}"):
        from sim2real.rl_policy.control_mode import resolve_keyboard_control_mode

        self._resolve_mode = resolve_keyboard_control_mode
        self._pending_mode = None
        self._pending_keys: list[str] = []
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.SUB)
        self._socket.setsockopt(zmq.SUBSCRIBE, b"")
        self._socket.setsockopt(zmq.RCVTIMEO, 0)
        self._socket.connect(connect)
        self._connect = connect

    def _drain(self):
        while True:
            try:
                key = self._socket.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                return
            mode = self._resolve_mode(key)
            if mode is not None:
                self._pending_mode = mode
            else:
                self._pending_keys.append(key)

    def get_control_mode(self):
        self._drain()
        mode, self._pending_mode = self._pending_mode, None
        return mode

    def get_extra_keys(self) -> list[str]:
        self._drain()
        keys, self._pending_keys = self._pending_keys, []
        return keys

    def close(self):
        self._socket.close(0)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="g1_bridge.keyboard_control")
    parser.add_argument("--port", type=int, default=DEFAULT_KEYBOARD_PORT)
    parser.add_argument(
        "--bind-host", default="*", help="'*' = any interface, 'localhost' to restrict."
    )
    args = parser.parse_args(argv)

    socket = zmq.Context.instance().socket(zmq.PUB)
    try:
        socket.bind(f"tcp://{args.bind_host}:{args.port}")
    except zmq.ZMQError as exc:
        # Almost always a leftover from a previous session still holding the
        # port. A raw ZMQError here reads as "the tool is broken"; it is one
        # command to fix, so say which one.
        print(
            colored(
                f"cannot publish keys on port {args.port}: {exc}\n"
                "  Another keyboard controller (probably from a previous session) "
                "still holds it.\n"
                "  Clear it with: python -m g1_bridge.launch_collect_real --kill\n"
                "  or run the collector with --keyboard local and type in its own pane.",
                "red",
            ),
            flush=True,
        )
        return 1
    print(colored(BANNER, "cyan"), flush=True)
    print(f"publishing keys on tcp://{args.bind_host}:{args.port}\n", flush=True)
    # Give SUBscribers a moment to finish connecting; ZMQ PUB drops messages sent
    # before a subscription lands, and the first key is usually pressed within
    # seconds of launch.
    time.sleep(0.5)

    is_tty = sys.stdin.isatty()
    if not is_tty:
        print("stdin is not a terminal; nothing to read.", flush=True)
        return 1
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        while True:
            ready, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not ready:
                continue
            ch = sys.stdin.read(1)
            if ch in ("\x03",):                      # Ctrl-C
                break
            key = "enter" if ch in ("\n", "\r") else ch.lower()
            if key == "q":
                # Quits THIS controller only -- the collector runs on. Nothing is
                # published: 'q' reaching the collector would end the session, and
                # this pane is not where that decision is made.
                break
            if key not in _RECORD_KEYS:
                print(
                    colored(
                        f"  {key!r} ignored — this pane only does c (record) and "
                        "x (discard). Robot control is in the controller pane.",
                        "yellow",
                    ),
                    flush=True,
                )
                continue
            socket.send_string(key)
            print(f"  -> {key}", flush=True)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
        socket.close(0)
    print("\nkeyboard controller stopped (the collector is still running).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
