"""Where did the PICO buttons stop? Watch every hop of the chain at once.

    uv run python -m g1_bridge.pico_probe

On the GMR backends (heft, scalebfm) the operator's buttons do NOT reach the
collector directly. The chain is::

    headset -> XRoboToolkit PC service -> GMR publisher (pico_retarget_pub,
      the pane running publisher_launcher.py) -> ZMQ 5592 buttons / 5593
      triggers / 28701 motion -> GMRButtonRelay -> ModeManager -> the chord

Only the GMR publisher may hold the ``xrobotoolkit_sdk`` connection -- a second
one in the collector would starve whichever end is not "primary" (that is the
whole reason ``GMRButtonRelay`` exists). So "the headset says connected" and
"the collector sees buttons" are two very different claims, with three
processes between them, and this prints all three so the dead hop is obvious
rather than inferred:

* motion (28701): is the publisher retargeting a live body at all?
* buttons (5592): press A / B / X / Y and watch them light up
* triggers (5593): squeeze; these are also what the hands follow

If motion moves but buttons never change, the publisher is alive and the button
socket is the problem. If nothing arrives at all, the publisher is not running
(or is bound on another host/port) -- which is also what the collector sees.
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import zmq
from termcolor import colored

from sim2real.utils.common import PORTS, HandGripMessage, PicoControllerStateMessage


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="g1_bridge.pico_probe")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--controller-port", type=int, default=PORTS["pico_controller"])
    parser.add_argument("--hand-grip-port", type=int, default=PORTS["hand_grip"])
    parser.add_argument("--motion-port", type=int, default=28701)
    parser.add_argument("--interval", type=float, default=1.0)
    args = parser.parse_args(argv)

    ctx = zmq.Context.instance()

    def sub(port: int):
        s = ctx.socket(zmq.SUB)
        s.setsockopt(zmq.SUBSCRIBE, b"")
        s.setsockopt(zmq.CONFLATE, 1)
        s.setsockopt(zmq.RCVTIMEO, 0)
        s.connect(f"tcp://{args.host}:{port}")
        return s

    buttons_sock = sub(args.controller_port)
    grip_sock = sub(args.hand_grip_port)
    motion_sock = sub(args.motion_port)

    print(
        f"watching {args.host}: buttons :{args.controller_port}  "
        f"triggers :{args.hand_grip_port}  motion :{args.motion_port}\n"
        "press A / B / X / Y on the PICO and squeeze the triggers "
        "(Ctrl-C to stop)\n",
        flush=True,
    )

    counts = {"buttons": 0, "triggers": 0, "motion": 0}
    state = {"buttons": None, "grips": (0.0, 0.0), "motion_seq": None, "joint0": None}
    seen_pressed: set[str] = set()
    last_report = time.monotonic()
    try:
        while True:
            try:
                msg = PicoControllerStateMessage.from_bytes(buttons_sock.recv(zmq.NOBLOCK))
                counts["buttons"] += 1
                state["buttons"] = {"A": msg.A, "B": msg.B, "X": msg.X, "Y": msg.Y}
                seen_pressed.update(k for k, v in state["buttons"].items() if v)
            except zmq.Again:
                pass
            try:
                grip = HandGripMessage.from_bytes(grip_sock.recv(zmq.NOBLOCK))
                counts["triggers"] += 1
                state["grips"] = (grip.left_grip, grip.right_grip)
            except zmq.Again:
                pass
            try:
                payload = json.loads(motion_sock.recv_string(zmq.NOBLOCK))
                counts["motion"] += 1
                state["motion_seq"] = payload.get("seq")
                joints = payload.get("joint_pos") or []
                state["joint0"] = round(float(joints[0]), 4) if joints else None
            except zmq.Again:
                pass
            except Exception:
                counts["motion"] += 1

            time.sleep(0.002)
            now = time.monotonic()
            if now - last_report < args.interval:
                continue
            elapsed = now - last_report
            last_report = now

            def rate(key):
                r = counts[key] / elapsed
                counts[key] = 0
                return r

            b_rate, g_rate, m_rate = rate("buttons"), rate("triggers"), rate("motion")
            pressed = (
                "".join(k for k, v in (state["buttons"] or {}).items() if v) or "-"
            )
            line = (
                f"  buttons {b_rate:5.1f} Hz  [{pressed:4s}]   "
                f"triggers {g_rate:5.1f} Hz  L={state['grips'][0]:.2f} "
                f"R={state['grips'][1]:.2f}   "
                f"motion {m_rate:5.1f} Hz  seq={state['motion_seq']} "
                f"j0={state['joint0']}"
            )
            if b_rate == 0 and m_rate == 0:
                line += colored("   <- nothing from the GMR publisher at all", "red")
            elif b_rate == 0:
                line += colored(
                    "   <- motion flows but NO buttons: the publisher is alive, "
                    "its button socket is not reaching here",
                    "yellow",
                )
            print(line, flush=True)
    except KeyboardInterrupt:
        pass
    if seen_pressed:
        print(f"\nbuttons seen pressed during this run: {sorted(seen_pressed)}")
        print("=> the chord reaches this machine; if the robot still does not "
              "engage, the problem is downstream (ModeManager / collector).")
    else:
        print("\nno button press was ever observed on this socket.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
