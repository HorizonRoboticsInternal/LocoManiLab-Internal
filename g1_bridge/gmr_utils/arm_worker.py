#!/usr/bin/env python3
"""GMR arm retargeting as a pure compute worker: body skeleton in, 14 joints out.

This exists instead of reusing ``GMRSubprocess`` because that path is unusable
for AMO, for two independent reasons:

1. ``pico_retarget_pub.py`` opens **its own** ``xrobotoolkit_sdk`` connection.
   Two concurrent SDK connections do not coexist -- the non-primary one's body
   pose freezes forever (measured 2026-07-29; see ``button_relay.py``'s module
   docstring). So a GMR-backed backend must give up its own PICO connection.
2. Which means reading buttons through ``GMRButtonRelay`` -- and that channel
   carries A/B/X/Y and the two triggers, and nothing else. **No joystick axes,
   no axis-click.** AMO's entire command surface is joysticks: locomotion,
   torso height and yaw all come off the two sticks (``_tick_amo_commands``),
   and arm engage is the right-stick click. Routed through that relay, AMO
   would stand still with its arms parked, permanently.

So this worker owns no hardware at all. tele keeps its single PICO connection
and simply hands over the 24-joint body frame it is already receiving
(``sample["raw_smpl"]``); this process retargets it and hands back joint
angles. Nothing about joysticks, ModeManager, or wrist calibration changes.

Runs under sim2real's isolated ``venv/pico/.venv`` (Python 3.10) because GMR
does not install under the 3.11 ``tele_benchmark`` env -- so it must import
nothing from ``g1_bridge`` or ``robocasa``, neither of which exists there.
``arm_relay.py`` launches it by absolute path, the same way
``publisher_proc.py`` launches ``publisher_launcher.py``.

Wire format, both directions, little-endian and fixed-size (a short read is a
protocol error, not a partial frame):

    request   uint64 seq | 24*7 float32   raw xrobotoolkit body pose, Unity axes
    response  uint64 seq | 14   float32   qpos[22:36], AMO's IK-14 order
"""

from __future__ import annotations

import argparse
import struct
import sys
import time

import numpy as np
import zmq

DEFAULT_IN_PORT = 28711
DEFAULT_OUT_PORT = 28712

_REQ_STRUCT = struct.Struct("<Q")
_N_BODY = 24
_REQ_BYTES = _REQ_STRUCT.size + _N_BODY * 7 * 4
_RSP_BYTES = _REQ_STRUCT.size + 14 * 4

#: qpos[22:36] of GMR's g1_mocap_29dof.xml is L[sh p,r,y, elbow, wr r,p,y] then
#: R[same] -- byte-for-byte the order AMO's IK-14 vector already uses, so the
#: hand-off needs no permutation on either side.
ARM_QPOS_SLICE = slice(22, 36)

#: PICO body-tracking joint order (xrobotoolkit_sdk.get_body_joints_pose()).
#: Kept in sync by hand with scripts/gmr_arm_ab.py's copy -- that one is a
#: standalone offline tool that must not import this package.
BODY_JOINT_NAMES = (
    "Pelvis", "Left_Hip", "Right_Hip", "Spine1", "Left_Knee", "Right_Knee",
    "Spine2", "Left_Ankle", "Right_Ankle", "Spine3", "Left_Foot", "Right_Foot",
    "Neck", "Left_Collar", "Right_Collar", "Head", "Left_Shoulder",
    "Right_Shoulder", "Left_Elbow", "Right_Elbow", "Left_Wrist", "Right_Wrist",
    "Left_Hand", "Right_Hand",
)

#: Unity (Y-up, left-handed) -> the right-handed frame GMR retargets in.
#: Verbatim from general_motion_retargeting.xrobot_utils.
_UNITY_TO_RH = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64)


def frame_from_raw(raw: np.ndarray) -> dict:
    """(24, 7) ``[x,y,z,qx,qy,qz,qw]`` in Unity axes -> GMR's human-pose dict."""
    from scipy.spatial.transform import Rotation as R

    pos = np.asarray(raw[:, 0:3], dtype=np.float64) @ _UNITY_TO_RH.T
    rot = R.from_matrix(_UNITY_TO_RH) * R.from_quat(
        np.asarray(raw[:, 3:7], dtype=np.float64)
    )
    quat_wxyz = rot.as_quat(scalar_first=True)
    return {name: [pos[i], quat_wxyz[i]] for i, name in enumerate(BODY_JOINT_NAMES)}


def _open_viewer(robot: str, port):
    """A viser web view of the retargeted robot, served on localhost.

    Shows GMR's **whole-body** solution -- which is more than this process
    hands over. Only ``qpos[22:36]``, the 14 arm joints, leaves here; the legs
    and waist you will see moving are solved (the retargeter always solves the
    full body) and then discarded, because AMO's own policy owns them. So the
    viewer answers "is GMR reading my body correctly", not "is this what the
    robot will do".

    Same viewer sim2real's own pico_retarget_pub uses, so it behaves the same
    and needs nothing that is not already in the pico venv.
    """
    import mujoco
    from general_motion_retargeting import ROBOT_XML_DICT
    from sim2real.utils.mjviser_viewer import MjviserMujocoViewer

    # Loaded as-is, deliberately. pico_retarget_pub wraps its model in mjhub's
    # temp_mjcf_with_floor first, but GMR's g1_mocap_29dof.xml already ships a
    # "floor" plane geom and its own light -- adding another floor throws
    # `XML Error: repeated name 'groundplane' in texture`.
    model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT[robot]))
    data = mujoco.MjData(model)
    viewer = MjviserMujocoViewer(model, data, label="gmr-arm", port=port)
    print(
        "[gmr-arm] viewer shows the FULL-BODY retarget; only the 14 arm joints "
        "are sent to the robot (legs/waist come from the AMO policy).",
        flush=True,
    )
    return mujoco, model, data, viewer


def run(args) -> int:
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    retarget = GMR(
        src_human="xrobot",
        tgt_robot=args.robot,
        actual_human_height=args.actual_human_height,
        verbose=False,
    )

    mj = model = mj_data = viewer = None
    if args.viewer:
        try:
            mj, model, mj_data, viewer = _open_viewer(args.robot, args.viewer_port)
        except Exception as e:  # a viewer must never cost us the teleop link
            print(f"[gmr-arm] viewer unavailable ({type(e).__name__}: {e}); "
                  "continuing headless.", flush=True)
            viewer = None
    # The retarget loop runs at 50 Hz; redrawing that often buys nothing and
    # competes with the solve for the same core.
    view_period = 1.0 / 30.0
    t_view = 0.0

    ctx = zmq.Context.instance()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    # Always the newest frame: this is a 50 Hz control loop, and a retarget that
    # ran long must catch up by SKIPPING, never by working through a backlog.
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.setsockopt(zmq.RCVTIMEO, 500)
    sub.connect(f"tcp://127.0.0.1:{args.in_port}")

    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{args.out_port}")

    print(
        f"[gmr-arm] ready: sub :{args.in_port} -> pub :{args.out_port}, "
        f"robot={args.robot}, actual_human_height={args.actual_human_height:.2f}",
        flush=True,
    )

    n = 0
    t_report = time.monotonic()
    worst_ms = 0.0
    while True:
        try:
            msg = sub.recv()
        except zmq.Again:
            continue
        except (zmq.ContextTerminated, KeyboardInterrupt):
            break
        if len(msg) != _REQ_BYTES:
            print(f"[gmr-arm] bad request size {len(msg)} (want {_REQ_BYTES}); dropped",
                  flush=True)
            continue
        (seq,) = _REQ_STRUCT.unpack_from(msg, 0)
        raw = np.frombuffer(msg, dtype=np.float32, offset=_REQ_STRUCT.size).reshape(
            _N_BODY, 7
        )
        if not np.isfinite(raw).all():
            continue  # a dropout frame; the relay holds its last value

        t0 = time.perf_counter()
        try:
            # Stateful on purpose: the configuration warm-starts from the
            # previous frame, which is what makes a per-frame solve cheap.
            qpos = retarget.retarget(frame_from_raw(raw), offset_to_ground=False)
        except Exception as e:  # never let one bad frame kill the worker
            print(f"[gmr-arm] retarget failed ({type(e).__name__}: {e}); skipped",
                  flush=True)
            continue
        dt_ms = (time.perf_counter() - t0) * 1e3
        worst_ms = max(worst_ms, dt_ms)

        q14 = np.asarray(qpos[ARM_QPOS_SLICE], dtype=np.float32)
        if q14.shape != (14,) or not np.isfinite(q14).all():
            continue
        pub.send(_REQ_STRUCT.pack(seq) + q14.tobytes())

        n += 1
        now = time.monotonic()
        if viewer is not None and now - t_view >= view_period:
            t_view = now
            try:
                mj_data.qpos[:] = qpos
                mj.mj_forward(model, mj_data)  # sync() reads xpos/xmat, not qpos
                viewer.sync()
            except Exception as e:
                print(f"[gmr-arm] viewer stopped ({type(e).__name__}: {e})", flush=True)
                viewer = None
        if now - t_report >= 5.0:
            print(f"[gmr-arm] {n / (now - t_report):5.1f} Hz, worst solve "
                  f"{worst_ms:5.2f} ms", flush=True)
            n, t_report, worst_ms = 0, now, 0.0
    if viewer is not None:
        viewer.close()
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--in-port", type=int, default=DEFAULT_IN_PORT)
    p.add_argument("--out-port", type=int, default=DEFAULT_OUT_PORT)
    p.add_argument("--robot", default="unitree_g1")
    p.add_argument(
        "--actual-human-height",
        type=float,
        default=1.60,
        help="Set BELOW the operator's real height: PICO's skeleton estimate "
        "runs long (TWIST2's teleop.sh says so explicitly). Offline sweeps on "
        "recorded sessions showed the arm angles are barely sensitive to it, "
        "so this is not a knob worth agonising over.",
    )
    p.add_argument(
        "--viewer",
        action="store_true",
        help="serve a viser web view of the retargeted robot on localhost",
    )
    p.add_argument(
        "--viewer-port",
        type=int,
        default=None,
        help="fixed port for the viewer (default: viser picks one and prints it)",
    )
    return run(p.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
