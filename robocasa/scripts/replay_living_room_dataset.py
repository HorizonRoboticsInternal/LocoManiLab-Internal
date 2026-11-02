"""Replay + health-check a collected living-room LeRobot dataset in sim.

Use this after collecting to confirm a dataset is good BEFORE training. For each
episode it does two things:

  1. HEALTH CHECK (no sim needed): per-field drift + fraction-of-frames-changing
     for action.motion_token, teleop.{left,right}_hand_joints and the arm joints
     of observation.state. It flags FROZEN fields -- the stale-controller-reference
     recording bug produced episodes whose motion_token / hand_joints were
     constant while the arm state still moved, so a frozen action field with a
     moving arm is the tell-tale sign and is reported in red.

  2. VISUAL REPLAY: feeds the recorded motion_token + hand joints through the
     SONIC whole-body decoder (the ``sonic_vla`` controller) frame-by-frame at
     the recorded fps, so you can watch the robot reproduce the demo.
     ``--render viewer`` opens a live window; ``--render video`` saves an mp4.
     NOTE: the dataset stores the robot obs/actions, not the object poses, so the
     scene objects are freshly sampled -- watch the ROBOT motion, not the objects.

Usage:
    # health-check every episode (fast, no sim):
    python -m robocasa.scripts.replay_living_room_dataset \
        --dataset outputs_sim/<backend>/<run> --check-only

    # health-check + save a replay mp4 per episode:
    MUJOCO_GL=egl python -m robocasa.scripts.replay_living_room_dataset \
        --dataset outputs_sim/<backend>/<run> --render video --video-dir outputs/replay

    # watch one episode live (needs a display):
    python -m robocasa.scripts.replay_living_room_dataset \
        --dataset outputs_sim/<backend>/<run> --episodes 0 --render viewer
"""

import argparse
import glob
import json
import pathlib
import time

import numpy as np
from termcolor import colored

import robosuite
from robosuite.controllers import load_composite_controller_config
from g1_bridge.sonic_utils.ckpt import (
    DEFAULT_SONIC_CKPT,
    SONIC_CKPT_CHOICES,
    apply_sonic_ckpt,
)

import robocasa  # noqa: F401  -- registers living-room envs

EGO_CAMERA = "robot0_head_camera"
REPLAY_CAMERA = "frontview"  # third-person view for the saved replay video
# observation.state arm-joint slices (43-DoF MuJoCo order) for the health x-check.
_ARM_SLICE = slice(15, 36)  # left arm + left hand + right arm (motion-bearing)


def episode_parquets(dataset):
    return sorted(glob.glob(f"{dataset}/data/chunk-*/*.parquet"))


def health_check(df):
    """Per-field drift + change-fraction for one episode dataframe."""

    def metrics(arr):
        arr = np.stack(arr).astype(np.float32)
        if arr.ndim == 1:
            arr = arr[:, None]
        drift = float(np.linalg.norm(arr[-1] - arr[0]))
        chg = float((np.linalg.norm(np.diff(arr, axis=0), axis=1) > 1e-4).mean())
        return drift, chg

    out = {}
    for key in (
        "action.motion_token",
        "teleop.left_hand_joints",
        "teleop.right_hand_joints",
    ):
        out[key] = metrics(df[key].values) if key in df.columns else (None, None)
    state = np.stack(df["observation.state"].values).astype(np.float32)
    arm = state[:, _ARM_SLICE]
    out["arm_state"] = (
        float(np.linalg.norm(arm[-1] - arm[0])),
        float((np.linalg.norm(np.diff(arm, axis=0), axis=1) > 1e-3).mean()),
    )
    # FROZEN = motion_token barely changes while the arm clearly moved (the bug).
    tok_chg = out["action.motion_token"][1]
    arm_chg = out["arm_state"][1]
    out["_frozen"] = (
        tok_chg is not None and tok_chg < 0.05 and arm_chg is not None and arm_chg > 0.3
    )
    return out


def print_health(name, h):
    def fmt(k):
        d, c = h[k]
        return "n/a" if d is None else f"drift={d:6.3f} chg={100 * c:4.0f}%"

    frozen = h["_frozen"]
    tag = (
        colored(" FROZEN!", "red", attrs=["bold"])
        if frozen
        else colored(" ok", "green")
    )
    color = "red" if frozen else "white"
    print(
        colored(
            f"  {name:<24} motion_token[{fmt('action.motion_token')}]  "
            f"Lhand[{fmt('teleop.left_hand_joints')}]  "
            f"Rhand[{fmt('teleop.right_hand_joints')}]  "
            f"arm[{fmt('arm_state')}]",
            color,
        )
        + tag
    )
    return frozen


def build_env(args, viewer):
    cfg_path = (
        pathlib.Path(robosuite.__file__).parent
        / "controllers/config/robots/default_g1_vla.json"
    )
    controller_config = load_composite_controller_config(controller=str(cfg_path))
    apply_sonic_ckpt(controller_config, getattr(args, "sonic_ckpt", DEFAULT_SONIC_CKPT))
    cams = [EGO_CAMERA] + ([] if viewer else [REPLAY_CAMERA])
    return robosuite.make(
        env_name=args.env,
        robots=args.robot,
        controller_configs=controller_config,
        has_renderer=viewer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=cams,
        camera_heights=480,
        camera_widths=640,
        render_camera=REPLAY_CAMERA,
        control_freq=args.control_freq,
        ignore_done=True,
        renderer="mjviewer" if viewer else "mujoco",
        seed=args.seed,
    )


def replay_episode(env, composite, df, viewer, save_video, fps, max_frames):
    """Drive the sonic_vla controller with the recorded actions, frame by frame."""
    tok = np.stack(df["action.motion_token"].values).astype(np.float32)
    lh = np.stack(df["teleop.left_hand_joints"].values).astype(np.float32)
    rh = np.stack(df["teleop.right_hand_joints"].values).astype(np.float32)
    n = min(len(tok), max_frames) if max_frames else len(tok)

    env.reset()
    composite = env.robots[0].composite_controller  # live controller (post-reset)
    zero_ac = np.zeros(env.action_dim)
    frames = []
    next_tick = time.monotonic()
    for t in range(n):
        composite.set_vla_action(tok[t], lh[t], rh[t])
        obs = env.step(zero_ac)[0]
        if viewer:
            env.render()
            next_tick += env.control_timestep
            s = next_tick - time.monotonic()
            time.sleep(s) if s > 0 else None
        elif save_video:
            frames.append(np.ascontiguousarray(obs[f"{REPLAY_CAMERA}_image"][::-1]))
    return frames


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        required=True,
        help="path to the LeRobot dataset dir (the one with meta/ " "data/ videos/).",
    )
    parser.add_argument("--env", default="TabletopPickPlace")
    parser.add_argument("--robot", default="G1FloatingBody")
    parser.add_argument(
        "--episodes",
        default="all",
        help="'all' or comma/space-separated episode indices (0-based).",
    )
    parser.add_argument(
        "--render",
        default="video",
        choices=["video", "viewer", "none"],
        help="'video': save mp4 per episode; 'viewer': live window; "
        "'none': health check only.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="health check only (alias for --render none).",
    )
    parser.add_argument("--video-dir", default="outputs/replay")
    parser.add_argument(
        "--control-freq",
        type=int,
        default=None,
        help="replay fps (default: the dataset's recorded fps).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="cap frames replayed per episode (0 = full episode).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sonic-ckpt",
        type=str,
        default=DEFAULT_SONIC_CKPT,
        choices=list(SONIC_CKPT_CHOICES),
        help="which SONIC encoder/decoder ONNX checkpoint to load: "
        "'old' (release) or 'new' (low-latency). Default 'new' -- only the "
        "low-latency export ships in the sim2real checkpoint tree.",
    )
    args = parser.parse_args()

    pqs = episode_parquets(args.dataset)
    if not pqs:
        raise SystemExit(f"no episodes under {args.dataset}/data/chunk-*/")
    if args.episodes != "all":
        idxs = [int(x) for x in args.episodes.replace(",", " ").split()]
        pqs = [pqs[i] for i in idxs]

    fps = args.control_freq
    if fps is None:
        info = json.load(open(f"{args.dataset}/meta/info.json"))
        fps = int(info.get("fps", 50))

    render = "none" if args.check_only else args.render
    viewer = render == "viewer"
    save_video = render == "video"

    # ---- pass 1: health check (always) ----
    import pandas as pd

    print(colored(f"\n=== health check: {len(pqs)} episode(s) @ {fps}fps ===", "cyan"))
    n_frozen = 0
    dfs = []
    for p in pqs:
        df = pd.read_parquet(p)
        dfs.append(df)
        n_frozen += int(print_health(pathlib.Path(p).name, health_check(df)))
    verdict = (
        colored(
            f"{n_frozen}/{len(pqs)} episodes FROZEN -- recording bug, do "
            "NOT train on this data.",
            "red",
            attrs=["bold"],
        )
        if n_frozen
        else colored(
            f"all {len(pqs)} episodes look healthy (motion_token varies).",
            "green",
            attrs=["bold"],
        )
    )
    print(colored("=== verdict: ", "cyan") + verdict)

    if render == "none":
        return

    # ---- pass 2: visual replay through the SONIC decoder ----
    args.control_freq = fps
    env = build_env(args, viewer)
    if save_video:
        pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    print(colored(f"\n=== replay ({render}) ===", "cyan"))
    for p, df in zip(pqs, dfs):
        name = pathlib.Path(p).stem
        print(colored(f"  replaying {name} ({len(df)} frames)...", "cyan"))
        frames = replay_episode(env, None, df, viewer, save_video, fps, args.max_frames)
        if save_video and frames:
            import imageio

            vp = pathlib.Path(args.video_dir) / f"{name}.mp4"
            imageio.mimwrite(vp, frames, fps=fps)
            print(colored(f"    saved {vp}", "cyan"))
    print(colored("=== done ===", "cyan"))


if __name__ == "__main__":
    main()
