#!/usr/bin/env python3
"""Receive normalized PICO grip commands over ZMQ and control the Dex3 hand."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import zmq

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sim2real.utils.common import PORTS, Dex3HandStateMessage
from sim2real.utils.dex3 import (
    DEFAULT_DOMAIN_ID,
    DEFAULT_INTERFACE,
    DEFAULT_KD,
    DEFAULT_KP,
    DEFAULT_MAX_DELTA_Q,
    DEFAULT_THUMB_KD,
    DEFAULT_THUMB_KP,
    Dex3HandDriver,
    DryRunDex3HandDriver,
    HandGripConsumer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--connect",
        default=f"tcp://127.0.0.1:{PORTS['hand_grip']}",
        help="HandGripMessage ZMQ publisher endpoint.",
    )
    parser.add_argument("--hand-dir", choices=["left", "right", "double"], default="double")
    parser.add_argument("--domain-id", type=int, default=DEFAULT_DOMAIN_ID)
    parser.add_argument("--interface", default=DEFAULT_INTERFACE)
    parser.add_argument("--kp", type=float, default=DEFAULT_KP)
    parser.add_argument("--kd", type=float, default=DEFAULT_KD)
    parser.add_argument("--thumb-kp", type=float, default=DEFAULT_THUMB_KP)
    parser.add_argument("--thumb-kd", type=float, default=DEFAULT_THUMB_KD)
    parser.add_argument("--max-delta-q", type=float, default=DEFAULT_MAX_DELTA_Q)
    parser.add_argument("--hwm", type=int, default=1)
    parser.add_argument("--stale-timeout-s", type=float, default=1.0)
    parser.add_argument("--status-interval-s", type=float, default=2.0)
    parser.add_argument(
        "--publish-state-addr",
        default=None,
        help=(
            "Optional ZMQ PUB bind address (e.g. tcp://*:5595) for a "
            "Dex3HandStateMessage stream carrying target+measured joint "
            "angles, consumed by the dex3 LeRobot RecordingExtra."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Decode and map commands without importing unitree_sdk2py or moving hardware.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stale_timeout_s <= 0 or args.status_interval_s <= 0:
        raise ValueError("stale-timeout-s and status-interval-s must be > 0")

    driver = (
        DryRunDex3HandDriver(hand_dir=args.hand_dir)
        if args.dry_run
        else Dex3HandDriver(
            hand_dir=args.hand_dir,
            domain_id=args.domain_id,
            interface=args.interface,
            kp=args.kp,
            kd=args.kd,
            thumb_kp=args.thumb_kp,
            thumb_kd=args.thumb_kd,
            max_delta_q=args.max_delta_q,
        )
    )
    consumer = HandGripConsumer(driver)

    socket = zmq.Context.instance().socket(zmq.SUB)
    socket.setsockopt(zmq.LINGER, 0)
    socket.setsockopt(zmq.RCVHWM, int(args.hwm))
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    socket.connect(args.connect)

    state_socket = None
    if args.publish_state_addr:
        state_socket = zmq.Context.instance().socket(zmq.PUB)
        state_socket.setsockopt(zmq.LINGER, 0)
        state_socket.bind(args.publish_state_addr)

    print(
        f"[Dex3] connect={args.connect} hand_dir={args.hand_dir} "
        f"dry_run={args.dry_run} publish_state_addr={args.publish_state_addr}",
        flush=True,
    )
    last_status = 0.0
    stale = False
    try:
        while True:
            try:
                raw = socket.recv(flags=zmq.DONTWAIT)
            except zmq.Again:
                raw = None

            now = time.monotonic()
            if raw is not None:
                try:
                    message = consumer.consume(raw, received_monotonic=now)
                except Exception as exc:
                    print(f"[Dex3] ignored invalid command: {exc}", flush=True)
                else:
                    if stale:
                        print("[Dex3] grip stream recovered", flush=True)
                    stale = False
                    if args.dry_run and now - last_status >= args.status_interval_s:
                        print(
                            f"[Dex3 dry-run] left={message.left_grip:.3f} "
                            f"right={message.right_grip:.3f}",
                            flush=True,
                        )
                        last_status = now

            if consumer.is_stale(now_monotonic=now, timeout_s=args.stale_timeout_s):
                stale = True
                if now - last_status >= args.status_interval_s:
                    print(
                        "[Dex3] grip stream stale; holding the last commanded pose",
                        flush=True,
                    )
                    last_status = now

            if state_socket is not None:
                targets = driver.read_targets()
                measured = driver.read_feedback()
                state_message = Dex3HandStateMessage(
                    timestamp_ns=time.time_ns(),
                    left_target=targets.get("left"),
                    left_measured=measured.get("left"),
                    right_target=targets.get("right"),
                    right_measured=measured.get("right"),
                )
                state_socket.send(state_message.to_bytes())

            time.sleep(0.005)
    except KeyboardInterrupt:
        print("[Dex3] KeyboardInterrupt, disconnecting without opening the hand", flush=True)
    finally:
        socket.close(0)
        if state_socket is not None:
            state_socket.close(0)
        driver.close()


if __name__ == "__main__":
    main()
