#!/usr/bin/env python3
"""Offline A/B: AMO's 3-point arm IK vs. GMR retargeting, on the same session.

The question this answers is "would GMR place the arms more like the operator's
than the 3-point IK does", and it answers it without touching the live control
path. ``_ArmIkLog`` (g1_bridge/backends/amo.py, enabled with
``SIM2REAL_AMO_IK_DEBUG=1``) already records, per engaged tick, the IK's raw
solution, the command it actually issued, and which guards fired. It now also
records the operator themselves -- the raw 24-joint PICO body skeleton. That is
exactly GMR's input, so the recorded session can simply be replayed through GMR
and the two answers scored against the same human.

Why the two solvers differ at all, since both are IK: the 3-point solver sees
only the headset and the two controllers. With the hand pose fixed, the elbow
is still free to swing around the shoulder-wrist axis -- the arm's redundant
DoF is *undetermined by the input*, and the solver settles it with a posture
prior (``_Q_REF``) that knows nothing about where the operator's elbow is. GMR
sees all 24 body joints and matches shoulder/elbow/wrist link *orientations*
(``xrobot_to_g1.json``: pos_weight 0, rot_weight 10), so the elbow is pinned by
the operator's actual elbow. That is the hypothesis metric A tests.

Run it under sim2real's isolated pico venv -- GMR needs Python 3.10 and does not
install into the 3.11 tele_benchmark env, which is also why this script imports
nothing from ``g1_bridge`` or ``robocasa``: none of it is importable there. It
reads an npz and nothing else.

    <g1_backend>/venv/pico/.venv/bin/python scripts/gmr_arm_ab.py \
        --input amo_ik_log_000.npz --actual-human-height 1.55,1.60,1.70

That writes ``<input>_ab.npz`` next to the input. Plotting is a separate pass,
because matplotlib is not in the pico venv -- run it from tele_benchmark:

    python scripts/gmr_arm_ab.py --plot-only amo_ik_log_000_ab.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Conventions
# --------------------------------------------------------------------------- #
#: PICO body-tracking joint order, as xrobotoolkit_sdk.get_body_joints_pose()
#: returns it. Same list as sim2real/teleop/benchmark_smplx_retarget.py and
#: GMR's own XRobotStreamer.body_joint_names -- duplicated rather than imported
#: because importing the latter constructs a streamer, which calls xrt.init().
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

#: The 14 arm joints, in the order BOTH sides already use: GMR's qpos[22:36]
#: (MJCF declaration order in g1_mocap_29dof.xml) and AMO's IK-14 order
#: (amo.py's _AMO_IK14_JOINT_NAMES) are the same sequence, so the comparison is
#: a slice on one side and a straight read on the other -- no permutation.
ARM_QPOS_SLICE = slice(22, 36)
ARM_JOINT_NAMES = (
    "left_shoulder_pitch", "left_shoulder_roll", "left_shoulder_yaw",
    "left_elbow", "left_wrist_roll", "left_wrist_pitch", "left_wrist_yaw",
    "right_shoulder_pitch", "right_shoulder_roll", "right_shoulder_yaw",
    "right_elbow", "right_wrist_roll", "right_wrist_pitch", "right_wrist_yaw",
)

#: Robot link <-> human body pairs the geometric metrics are computed over.
#: The robot links are the ones xrobot_to_g1.json itself drives.
_LIMB_CHAIN = {
    "left": {
        "shoulder": ("left_shoulder_yaw_link", "Left_Shoulder"),
        "elbow": ("left_elbow_link", "Left_Elbow"),
        "wrist": ("left_wrist_yaw_link", "Left_Wrist"),
    },
    "right": {
        "shoulder": ("right_shoulder_yaw_link", "Right_Shoulder"),
        "elbow": ("right_elbow_link", "Right_Elbow"),
        "wrist": ("right_wrist_yaw_link", "Right_Wrist"),
    },
}

#: A per-tick |dq| above this counts as a twitch. Same threshold the live 1 Hz
#: summary and tests/test_amo_arm_ik_smoothness.py use, so the numbers here are
#: readable against the ones printed during a session.
_TWITCH_DQ = 0.1


# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
def frame_from_raw(raw: np.ndarray) -> dict:
    """One (24, 7) PICO body frame -> the dict GMR's ``retarget`` consumes.

    Input rows are ``[x, y, z, qx, qy, qz, qw]`` in Unity axes (scalar-LAST);
    GMR wants ``{name: [pos, quat_wxyz]}`` in right-handed axes. Same two steps
    as xrobot_utils.get_processed_body_data + coordinate_transform_unity_data.
    """
    from scipy.spatial.transform import Rotation as R

    pos = np.asarray(raw[:, 0:3], dtype=np.float64) @ _UNITY_TO_RH.T
    # scalar-last -> Rotation, then left-multiply by the frame change.
    rot = R.from_matrix(_UNITY_TO_RH) * R.from_quat(np.asarray(raw[:, 3:7], dtype=np.float64))
    quat_wxyz = rot.as_quat(scalar_first=True)
    return {
        name: [pos[i], quat_wxyz[i]] for i, name in enumerate(BODY_JOINT_NAMES)
    }


def load_log(path: Path) -> dict:
    """Read an _ArmIkLog dump and fail loudly on the ones that cannot be used."""
    with np.load(path, allow_pickle=False) as d:
        missing = [k for k in ("raw_smpl", "q_cmd", "q_raw", "scalars", "columns") if k not in d]
        if missing:
            raise SystemExit(
                f"{path}: missing {missing}. This npz predates the raw_smpl "
                "capture -- re-record with SIM2REAL_AMO_IK_DEBUG=1 on a build "
                "that includes it."
            )
        out = {k: d[k] for k in d.files}
    out["columns"] = [str(c) for c in out["columns"]]

    smpl = out["raw_smpl"]
    valid = np.isfinite(smpl).all(axis=(1, 2))
    if not valid.any():
        raise SystemExit(
            f"{path}: every one of the {len(smpl)} frames has no body-tracking "
            "data (all NaN). That is what a --pico-source mock session looks "
            "like; GMR has nothing to retarget. Re-record with "
            "--pico-source real and the headset's body tracking active."
        )
    out["valid"] = valid
    return out


# --------------------------------------------------------------------------- #
# Forward kinematics
# --------------------------------------------------------------------------- #
class RobotFK:
    """FK on GMR's own G1 model, used for BOTH trajectories.

    One model for both sides is the point: it removes any MJCF-vs-URDF
    discrepancy from the comparison, so a difference in the numbers is a
    difference in the joint angles and nothing else.
    """

    def __init__(self):
        import mujoco
        from general_motion_retargeting import ROBOT_XML_DICT

        self._mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(ROBOT_XML_DICT["unitree_g1"]))
        self.data = mujoco.MjData(self.model)

    def body_positions(self, qpos: np.ndarray, body_names) -> dict:
        self.data.qpos[:] = qpos
        self._mj.mj_kinematics(self.model, self.data)
        return {
            name: self.data.body(name).xpos.copy() for name in body_names
        }


def _unit(v: np.ndarray):
    n = np.linalg.norm(v)
    return None if n < 1e-9 else v / n


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    ua, ub = _unit(a), _unit(b)
    if ua is None or ub is None:
        return np.nan
    return float(np.degrees(np.arccos(np.clip(float(ua @ ub), -1.0, 1.0))))


# --------------------------------------------------------------------------- #
# The replay
# --------------------------------------------------------------------------- #
def replay(log: dict, human_height: float, fk: RobotFK) -> dict:
    """Retarget every valid frame and score both solvers against the operator.

    Geometry is compared in GMR's world, where the scaled human and the
    retargeted robot are co-located by construction. To put the IK's arms in
    that same world, its 14 joint angles are written into GMR's own solution
    for that frame -- so the two configurations share an identical root, legs
    and waist and differ ONLY in the arms. Anything the metrics see is then
    attributable to the arm solve.
    """
    from general_motion_retargeting import GeneralMotionRetargeting as GMR

    retarget = GMR(
        src_human="xrobot",
        tgt_robot="unitree_g1",
        actual_human_height=human_height,
        verbose=False,
    )

    raw = log["raw_smpl"]
    q_cmd = log["q_cmd"]
    valid = log["valid"]
    n = len(raw)

    body_names = [
        link for side in _LIMB_CHAIN.values() for link, _ in side.values()
    ]

    q_gmr = np.full_like(q_cmd, np.nan)
    elbow_err = np.full((n, 2, 2), np.nan, dtype=np.float32)  # [t, side, (gmr, ik)]
    wrist_err = np.full((n, 2, 2), np.nan, dtype=np.float32)
    solve_ms = np.full(n, np.nan, dtype=np.float32)

    for t in range(n):
        if not valid[t]:
            continue
        frame = frame_from_raw(raw[t])
        t0 = time.perf_counter()
        # Stateful by design: the configuration warm-starts from the previous
        # frame, exactly as it would live. Frames must be fed in order and not
        # subsampled, or the trajectory this produces is not the one a live
        # session would have produced.
        qpos = retarget.retarget(frame, offset_to_ground=False)
        solve_ms[t] = (time.perf_counter() - t0) * 1e3

        q_gmr[t] = qpos[ARM_QPOS_SLICE]
        qpos_ik = qpos.copy()
        qpos_ik[ARM_QPOS_SLICE] = q_cmd[t]

        p_gmr = fk.body_positions(qpos, body_names)
        p_ik = fk.body_positions(qpos_ik, body_names)
        human = retarget.scaled_human_data

        for s, (side, chain) in enumerate(_LIMB_CHAIN.items()):
            sh_link, sh_body = chain["shoulder"]
            el_link, el_body = chain["elbow"]
            wr_link, wr_body = chain["wrist"]
            if sh_body not in human or el_body not in human:
                continue
            h_sh = np.asarray(human[sh_body][0], dtype=np.float64)
            h_el = np.asarray(human[el_body][0], dtype=np.float64)
            h_upper = h_el - h_sh
            # A: does the upper arm point where the operator's does?
            elbow_err[t, s, 0] = _angle_deg(p_gmr[el_link] - p_gmr[sh_link], h_upper)
            elbow_err[t, s, 1] = _angle_deg(p_ik[el_link] - p_ik[sh_link], h_upper)
            # B: and what does that cost at the hand?
            if wr_body in human:
                h_wr = np.asarray(human[wr_body][0], dtype=np.float64)
                wrist_err[t, s, 0] = np.linalg.norm(p_gmr[wr_link] - h_wr)
                wrist_err[t, s, 1] = np.linalg.norm(p_ik[wr_link] - h_wr)

    return dict(
        q_gmr=q_gmr,
        elbow_err_deg=elbow_err,
        wrist_err_m=wrist_err,
        solve_ms=solve_ms,
        human_height=human_height,
    )


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def smoothness(q: np.ndarray, valid: np.ndarray, dt: float) -> dict:
    """Peak/p99 commanded joint speed and twitch count over a 14-DoF path.

    Only steps between consecutive valid frames count -- a step across a
    tracking dropout is an artefact of the gap, not of the solver.
    """
    ok = valid[1:] & valid[:-1]
    if not ok.any():
        return dict(p99_vel=np.nan, max_vel=np.nan, twitch=0, mean_dq=np.nan)
    dq = np.max(np.abs(np.diff(q, axis=0)), axis=1)[ok]
    dq = dq[np.isfinite(dq)]
    if not len(dq):
        return dict(p99_vel=np.nan, max_vel=np.nan, twitch=0, mean_dq=np.nan)
    return dict(
        p99_vel=float(np.percentile(dq, 99) / dt),
        max_vel=float(dq.max() / dt),
        twitch=int((dq > _TWITCH_DQ).sum()),
        mean_dq=float(dq.mean()),
    )


def _stat(a: np.ndarray) -> tuple:
    a = a[np.isfinite(a)]
    if not len(a):
        return np.nan, np.nan
    return float(np.median(a)), float(np.percentile(a, 90))


def guard_duty(log: dict) -> dict:
    """How often each guard fired. Height-independent: a property of the
    recorded session, not of the retargeting."""
    cols = log["columns"]
    sc = log["scalars"]

    def col(name):
        return sc[:, cols.index(name)]

    return dict(
        clamp=float(np.maximum(col("clamped_l"), col("clamped_r")).mean()),
        rate_limited=float(col("rate_limited").mean()),
        jump_rejected=int(col("jump_rejected").sum()),
        ik_failed=int(col("ik_failed").sum()),
        n=len(sc),
    )


def session_dt(log: dict) -> float:
    cols = log["columns"]
    t = log["scalars"][:, cols.index("sim_time")]
    if len(t) < 2:
        return 0.02
    d = np.diff(t)
    d = d[d > 0]
    return float(np.median(d)) if len(d) else 0.02


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def report(log: dict, runs: list, dt: float) -> None:
    valid = log["valid"]
    g = guard_duty(log)

    print()
    print("=" * 78)
    print(f"session: {g['n']} engaged ticks, {int(valid.sum())} with body tracking, "
          f"dt={dt * 1e3:.1f} ms")
    print("=" * 78)

    print("\n[D] guards on the recorded IK -- how much of the jitter is the guard chain")
    print(f"  reach clamp duty     {g['clamp']:.3f}")
    print(f"  rate limiter duty    {g['rate_limited']:.3f}")
    print(f"  jump-guard rejects   {g['jump_rejected']} tick(s)")
    print(f"  ik failures          {g['ik_failed']} tick(s)")
    if g["rate_limited"] > 0.2 or g["jump_rejected"] > 0.02 * max(g["n"], 1):
        print("  -> a guard is active a large fraction of the time. Swapping the "
              "solver will NOT fix that on its own;")
        print("     the same guards sit downstream of GMR too. Tune them first.")

    print("\n[C] smoothness of the IK's own output (guards included / excluded)")
    for label, key in (("q_cmd (issued)", "q_cmd"), ("q_raw (unguarded)", "q_raw")):
        m = smoothness(log[key], valid, dt)
        print(f"  {label:20s} p99 {m['p99_vel']:6.2f} rad/s  max {m['max_vel']:6.2f}  "
              f"twitch {m['twitch']}")

    hdr = (f"\n{'height':>7} | {'A elbow err (deg)':>26} | {'B wrist err (m)':>24} | "
           f"{'C GMR smooth':>22} | {'E ms':>11}")
    print(hdr)
    print(f"{'':>7} | {'GMR med/p90':>12} {'IK med/p90':>13} | "
          f"{'GMR med/p90':>11} {'IK med/p90':>12} | {'p99':>7} {'max':>6} {'twitch':>7} | "
          f"{'mean':>5} {'p99':>5}")
    print("-" * 106)

    for r in runs:
        e_gmr = _stat(r["elbow_err_deg"][..., 0])
        e_ik = _stat(r["elbow_err_deg"][..., 1])
        w_gmr = _stat(r["wrist_err_m"][..., 0])
        w_ik = _stat(r["wrist_err_m"][..., 1])
        sm = smoothness(r["q_gmr"], valid, dt)
        ms = r["solve_ms"][np.isfinite(r["solve_ms"])]
        print(
            f"{r['human_height']:7.2f} | {e_gmr[0]:5.1f}/{e_gmr[1]:5.1f}    "
            f"{e_ik[0]:5.1f}/{e_ik[1]:5.1f}     | "
            f"{w_gmr[0]:5.3f}/{w_gmr[1]:5.3f} {w_ik[0]:5.3f}/{w_ik[1]:5.3f} | "
            f"{sm['p99_vel']:7.2f} {sm['max_vel']:6.2f} {sm['twitch']:7d} | "
            f"{ms.mean():5.2f} {np.percentile(ms, 99):5.2f}"
        )

    best = min(runs, key=lambda r: _stat(r["elbow_err_deg"][..., 0])[0])
    e_gmr = _stat(best["elbow_err_deg"][..., 0])[0]
    e_ik = _stat(best["elbow_err_deg"][..., 1])[0]
    w_gmr = _stat(best["wrist_err_m"][..., 0])[0]
    w_ik = _stat(best["wrist_err_m"][..., 1])[0]
    p99 = float(np.percentile(best["solve_ms"][np.isfinite(best["solve_ms"])], 99))

    print(f"\nverdict (best height {best['human_height']:.2f}):")
    print(f"  [A] upper-arm direction vs the operator: GMR {e_gmr:.1f} deg, "
          f"IK {e_ik:.1f} deg  ->  {e_ik - e_gmr:+.1f} deg")
    print(f"  [B] wrist position vs the operator:      GMR {w_gmr:.3f} m,   "
          f"IK {w_ik:.3f} m    ->  {(w_ik - w_gmr) * 100:+.1f} cm")
    print(f"  [E] retarget p99 {p99:.2f} ms against a {dt * 1e3:.0f} ms tick budget")
    print("\n  A is the go/no-go: it is the redundant DoF the 3-point input cannot "
          "determine at all.")
    print("  B is the price -- GMR follows the operator's skeleton, not the "
          "controller's metric position,")
    print("  so it is expected to lose there. Decide which one the task needs.")


def plot(ab_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(ab_path, allow_pickle=False) as d:
        q_cmd, q_gmr = d["q_cmd"], d["q_gmr"]
        elbow, wrist = d["elbow_err_deg"], d["wrist_err_m"]
        height = float(d["human_height"])

    fig, axes = plt.subplots(8, 2, figsize=(15, 20), sharex=True)
    for j, name in enumerate(ARM_JOINT_NAMES):
        ax = axes[j % 7, j // 7]  # left arm down column 0, right arm down column 1
        ax.plot(q_cmd[:, j], lw=0.9, label="IK (issued)")
        ax.plot(q_gmr[:, j], lw=0.9, label="GMR")
        ax.set_ylabel(name, fontsize=7)
        ax.tick_params(labelsize=6)
        if j == 0:
            ax.legend(fontsize=7)

    for s, side in enumerate(("left", "right")):
        ax = axes[7, s]
        ax.plot(elbow[:, s, 1], lw=0.9, label="IK")
        ax.plot(elbow[:, s, 0], lw=0.9, label="GMR")
        ax2 = ax.twinx()
        ax2.plot(wrist[:, s, 1], lw=0.6, ls=":", color="C0")
        ax2.plot(wrist[:, s, 0], lw=0.6, ls=":", color="C1")
        ax2.set_ylabel("wrist err (m, dotted)", fontsize=7)
        ax.set_ylabel(f"{side} elbow err (deg)", fontsize=7)
        ax.legend(fontsize=7)

    fig.suptitle(f"AMO 3-point IK vs GMR  --  {ab_path.name}  (human height {height:.2f} m)")
    fig.tight_layout()
    out = ab_path.with_suffix(".png")
    fig.savefig(out, dpi=110)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--input", type=Path, help="amo_ik_log_NNN.npz from SIM2REAL_AMO_IK_DEBUG=1")
    p.add_argument(
        "--actual-human-height",
        default="1.55,1.60,1.70",
        help="comma-separated heights to sweep. Set these BELOW your real "
        "height: PICO's skeleton estimate runs long, and TWIST2's own teleop "
        "script says so explicitly. Sweeping is free offline.",
    )
    p.add_argument("--out", type=Path, default=None, help="output npz (default: <input>_ab.npz)")
    p.add_argument("--plot-only", type=Path, default=None,
                   help="skip retargeting; plot an existing _ab.npz (needs matplotlib, "
                        "so run this one from the tele_benchmark env)")
    args = p.parse_args(argv)

    if args.plot_only:
        plot(args.plot_only)
        return 0
    if not args.input:
        p.error("--input is required (or use --plot-only)")

    heights = [float(h) for h in str(args.actual_human_height).split(",") if h.strip()]
    log = load_log(args.input)
    dt = session_dt(log)

    fk = RobotFK()
    runs = []
    for h in heights:
        print(f"retargeting at actual_human_height={h:.2f} ...", file=sys.stderr)
        runs.append(replay(log, h, fk))

    report(log, runs, dt)

    best = min(runs, key=lambda r: _stat(r["elbow_err_deg"][..., 0])[0])
    out = args.out or args.input.with_name(args.input.stem + "_ab.npz")
    np.savez_compressed(
        out,
        q_cmd=log["q_cmd"],
        q_raw=log["q_raw"],
        arm_joint_names=np.array(ARM_JOINT_NAMES),
        valid=log["valid"],
        dt=dt,
        heights=np.array(heights),
        **{k: best[k] for k in ("q_gmr", "elbow_err_deg", "wrist_err_m", "solve_ms", "human_height")},
    )
    print(f"\nwrote {out}  (best height {best['human_height']:.2f}; "
          f"plot it with --plot-only from an env that has matplotlib)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
