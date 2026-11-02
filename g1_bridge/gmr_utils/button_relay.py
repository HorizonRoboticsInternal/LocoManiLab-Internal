"""ZMQ-based PicoVRSource that reads buttons/hand-trigger from GMR's own
publisher, instead of opening a second independent xrobotoolkit_sdk
connection in this process.

Why this exists: ``sim2real.teleop.pico_retarget_pub.LiveRetargetPublisher``
(the GMR full-body retargeting subprocess launched by ``gmr_subprocess.py``)
opens its own ``xrobotoolkit_sdk.init()`` connection to read PICO body
tracking. This composite used to ALSO open a second, independent connection
in the main process (via robosuite's ``RealPicoVRSource``/
``ProcessPicoVRSource``), purely to read buttons/hand-trigger for the
ModeManager chord and gripper control.

Verified live (2026-07-29) that two concurrent ``xrobotoolkit_sdk.init()``
connections do NOT actually run side by side cleanly, contrary to
``gmr_subprocess.py``'s docstring: directly subscribing to the GMR
publisher's motion feed while the main process's own PICO source was
reading fresh buttons showed ``joint_pos`` bit-identical across many
seconds and ``smplx_t_ns`` permanently 0 -- i.e. whichever connection is
NOT "primary" gets a stale, frozen body pose forever. This is exactly what
caused "hand trigger works, body doesn't follow" (and the mirror-image
"body follows, hand trigger doesn't" from an earlier session where the
GMR subprocess happened to end up primary instead).

The fix: never open a second PICO connection at all when a GMR subprocess
already owns the link. ``LiveRetargetPublisher`` already publishes A/B/X/Y
button state (port ``PORTS["pico_controller"]``, ``PicoControllerStateMessage``)
and left/right index-trigger values (port ``PORTS["hand_grip"]``,
``HandGripMessage`` -- misleadingly named "grip" but built from
``index_trig``, i.e. the same front trigger our gripper control needs) over
ZMQ specifically so a second process doesn't need its own SDK connection.
This class subscribes to both and adapts them into the same sample-dict
shape ``RealPicoVRSource``/``ProcessPicoVRSource`` return, so it's a drop-in
``PicoVRSource`` for GMR-backed backends (heft, scalebfm).

Trade-off: GMR's publisher does not expose axis-click, menu buttons, or the raw
calibrated 3-point wrist/head pose over ZMQ. ModeManager's ``VR_3PT`` mode
(left-stick-click transition) is therefore unreachable through this source --
OFF/PLANNER/POSE/PLANNER_FROZEN_UPPER (the modes these backends actually use)
are unaffected.

The joystick *axes* used to be on that list too, which broke any GMR-backed
backend that steers with a stick: ALMI's chords and hand triggers arrived, its
body tracked, and its sticks read centred forever, so the robot simply would
not walk. The publisher now carries them in ``PicoControllerStateMessage`` and
they are surfaced here under the same ``left_axis``/``right_axis`` keys the
direct sources use.
"""

from __future__ import annotations

import threading
import time
from typing import Optional

import numpy as np
import zmq
from g1_bridge.pico.vr_source import (
    PicoVRSource,
    empty_buttons,
    empty_device_poses,
)
from sim2real.utils.common import PORTS, HandGripMessage, PicoControllerStateMessage


class GMRButtonRelay(PicoVRSource):
    def __init__(
        self,
        controller_connect: str = f"tcp://127.0.0.1:{PORTS['pico_controller']}",
        hand_grip_connect: str = f"tcp://127.0.0.1:{PORTS['hand_grip']}",
    ):
        ctx = zmq.Context.instance()
        self._controller_sock = ctx.socket(zmq.SUB)
        self._controller_sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._controller_sock.setsockopt(zmq.CONFLATE, 1)
        self._controller_sock.connect(controller_connect)

        self._hand_grip_sock = ctx.socket(zmq.SUB)
        self._hand_grip_sock.setsockopt(zmq.SUBSCRIBE, b"")
        self._hand_grip_sock.setsockopt(zmq.CONFLATE, 1)
        self._hand_grip_sock.connect(hand_grip_connect)

        self._lock = threading.Lock()
        self._buttons = empty_buttons()
        self._stop_evt = threading.Event()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="GMRButtonRelay"
        )

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()
            print(
                "[GMRButtonRelay] started -- reading buttons/trigger from GMR's "
                f"own PICO connection (ports {PORTS['pico_controller']}/"
                f"{PORTS['hand_grip']}) instead of opening a second one. "
                "VR_3PT mode is unreachable through this source (see module "
                "docstring); OFF/PLANNER/POSE/PLANNER_FROZEN_UPPER are fine."
            )

    def stop(self) -> None:
        self._stop_evt.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._controller_sock.close(linger=0)
        self._hand_grip_sock.close(linger=0)

    def calibrate(self) -> bool:
        # No calibration hook over this channel -- GMR's own publisher
        # handles its own heading/pose calibration internally.
        return False

    def get_latest(self) -> Optional[dict]:
        with self._lock:
            buttons = dict(self._buttons)
        return {
            # This channel carries buttons only -- GMR owns the PICO
            # connection, so there is no headset/controller pose here. Flagged
            # invalid rather than zero-filled silently: a consumer that needs
            # metric poses (amo's absolute arm teleop) refuses to engage and
            # says why, instead of driving the arms from identity.
            **empty_device_poses(),
            "vr_3point_local_target": np.zeros(9, dtype=np.float32),
            "vr_3point_local_orn_target": np.zeros(12, dtype=np.float32),
            "calibrated": True,
            "timestamp": time.monotonic(),
            "raw_smpl": None,
            "buttons": buttons,
            "pose_smpl_joints_local": None,
            "pose_smpl_root_quat": None,
            "pose_anchor_6d": None,
            "pose_wrist_joints": None,
        }

    def _apply_controller_message(self, msg: PicoControllerStateMessage) -> None:
        """Fold one decoded controller message into the button snapshot.

        Split out of ``_loop`` so the mapping can be tested without a socket --
        the axes were missing here for a while and nothing caught it.
        """
        with self._lock:
            self._buttons["a"] = msg.A
            self._buttons["b"] = msg.B
            self._buttons["x"] = msg.X
            self._buttons["y"] = msg.Y
            # Same (x, y) convention the direct sources use, so a backend's
            # stick mapping does not care which source it was handed.
            self._buttons["left_axis"] = msg.left_axis
            self._buttons["right_axis"] = msg.right_axis

    def _loop(self) -> None:
        while not self._stop_evt.is_set():
            got_any = False
            try:
                raw = self._controller_sock.recv(flags=zmq.NOBLOCK)
                self._apply_controller_message(
                    PicoControllerStateMessage.from_bytes(raw)
                )
                got_any = True
            except zmq.Again:
                pass
            except Exception as e:
                print(f"[GMRButtonRelay] controller decode error: {e}")

            try:
                raw = self._hand_grip_sock.recv(flags=zmq.NOBLOCK)
                msg = HandGripMessage.from_bytes(raw)
                with self._lock:
                    self._buttons["left_trigger"] = msg.left_grip
                    self._buttons["right_trigger"] = msg.right_grip
                got_any = True
            except zmq.Again:
                pass
            except Exception as e:
                print(f"[GMRButtonRelay] hand-grip decode error: {e}")

            if not got_any:
                time.sleep(0.005)
