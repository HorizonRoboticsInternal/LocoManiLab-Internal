"""Closed-loop in-sim validation of a trained GR00T REFERENCE-MOTION VLA policy.

Sibling of ``deploy_living_room_vla.py``. That script validates the *token* model
(``unitree_g1_sonic``: predicts the 64-D SONIC ``motion_token``, fed straight to the
decoder). THIS script validates the *reference-motion* model
(``unitree_g1_sonic_refmotion_joints_benchmark_smi``): the policy predicts the SONIC
encoder INPUT -- 24 SMPL joint positions (72-D) -- plus the 7+7 hand joints. At
deploy the predicted SMPL joints are pushed into the SONIC POSE ENCODER and the
FULL encoder->decoder is run (``sonic_pose_vla`` controller, default_g1_vla_pose.json).

Why a separate controller/script: the token path skips the encoder; the
reference-motion path must run it (the model output is the encoder's input, not
its output). anchor (root orientation) and wrist joints are NOT predicted by this
model, so they are left at the POSE controller's defaults (identity 6D / zeros) --
a close approximation for a near-stationary, forward-facing task.

Observation sent to the policy is identical to deploy_living_room_vla (nested
``video.ego_view`` + ``state.{left_leg,...,right_hand,projected_gravity}`` + prompt);
only the ACTION schema differs (smpl_joints instead of motion_token).

Usage (start the policy server first, in the Isaac-GR00T env, serving the
reference-motion model with embodiment
``unitree_g1_sonic_refmotion_joints_benchmark_smi`` on tcp://HOST:PORT), then:

    conda activate tele_benchmark
    python -m robocasa.scripts.deploy_living_room_vla_pose \
        --env TabletopPickPlace --host localhost --port 5550 \
        --prompt "pick up the mug and place it on the plate" \
        --fixed-scene --scene-seed 0 --episodes 10 --video-dir outputs/vla_eval_pose

Server-free smoke tests of the sim loop (no GR00T needed):
    # replay recorded smpl_joints from a collected/preprocessed episode parquet:
    ... --policy replay --replay-parquet <episode_000000.parquet>
    # zeros (robot just holds the SONIC default standing pose):
    ... --policy zero
"""

import argparse
import pathlib
import time

import numpy as np
from scipy.spatial.transform import Rotation as sRot
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
EGO_H, EGO_W = 240, 320
SMPL_DIM = 72  # 24 SMPL joints x 3
HAND_DIM = 7

# 43-DoF MuJoCo joint-vector slices -> the unitree_g1_sonic state groups
# (identical to G1_modality.json / the embodiment config; state schema is the
# SAME for the token and reference-motion models).
_STATE_SLICES = {
    "left_leg": (0, 6),
    "right_leg": (6, 12),
    "waist": (12, 15),
    "left_arm": (15, 22),
    "left_hand": (22, 29),
    "right_arm": (29, 36),
    "right_hand": (36, 43),
}


def projected_gravity(base_quat_wxyz):
    """Gravity in the robot base frame (matches the collection script)."""
    r = sRot.from_quat(np.asarray(base_quat_wxyz)[[1, 2, 3, 0]])  # wxyz -> xyzw
    return r.inv().apply([0.0, 0.0, -1.0]).astype(np.float32)


def build_observation(obs, prompt):
    """Assemble the GR00T observation dict from a robosuite obs.

    Identical to deploy_living_room_vla.build_observation -- the observation
    schema does not change between the token and reference-motion models. All
    state/video tensors carry leading [batch=1, time=1] dims.
    """
    q = np.asarray(obs["robot0_joint_pos"], dtype=np.float32)
    img = obs[f"{EGO_CAMERA}_image"][::-1]  # robosuite frames are vertically flipped
    img = np.ascontiguousarray(img).astype(np.uint8)

    state = {
        name: q[a:b][np.newaxis, np.newaxis] for name, (a, b) in _STATE_SLICES.items()
    }
    state["projected_gravity"] = projected_gravity(obs["robot0_base_quat"])[
        np.newaxis, np.newaxis
    ]
    return {
        "video": {"ego_view": img[np.newaxis, np.newaxis]},
        "state": state,
        "language": {"annotation.human.task_description": [[prompt]]},
    }


def _action_field(action, key):
    """Fetch ``key`` (or ``action.<key>``) from a policy action dict."""
    for k in (key, f"action.{key}"):
        if k in action and action[k] is not None:
            return np.asarray(action[k])
    raise KeyError(f"action field '{key}' not in {list(action.keys())}")


def _stat(a):
    a = np.asarray(a)
    return (
        f"shape={tuple(a.shape)} dtype={a.dtype} min={a.min():.4f} "
        f"max={a.max():.4f} mean={a.mean():.4f} std={a.std():.4f}"
    )


def debug_observation(obs_dict):
    print(colored("[debug] --- observation sent to policy ---", "magenta"))
    img = obs_dict["video"]["ego_view"]
    print(colored(f"[debug]   video.ego_view: {_stat(img)}", "magenta"))
    for k, v in obs_dict["state"].items():
        print(colored(f"[debug]   state.{k}: {_stat(v)}", "magenta"))
    print(
        colored(
            f"[debug]   prompt: {obs_dict['language']['annotation.human.task_description']}",
            "magenta",
        )
    )


def debug_action(action, prev_sj):
    print(colored(f"[debug]   action keys: {list(action.keys())}", "magenta"))
    for k in ("smpl_joints", "left_hand_joints", "right_hand_joints"):
        try:
            v = _action_field(action, k)
            line = f"[debug]   {k}: {_stat(v)}"
            if k == "smpl_joints":
                ch = _as_chunk(v, SMPL_DIM)  # [horizon, 72]
                # within-chunk horizon variation: does the chunk encode a TRAJECTORY
                # (smpl joints move across the 64 future steps) or is it FLAT?
                if len(ch) > 1:
                    hd = np.linalg.norm(np.diff(ch, axis=0), axis=1)
                    line += (
                        f"  | within-chunk per-step|Δ| med={np.median(hd):.4f}, "
                        f"smpl[0]->[-1]={np.linalg.norm(ch[-1]-ch[0]):.4f}"
                    )
                if prev_sj is not None:
                    line += f"  | Δ(prev applied)={np.linalg.norm(ch[0] - prev_sj):.4f}"
            print(colored(line, "magenta"))
        except Exception as e:
            print(colored(f"[debug]   {k}: MISSING ({e})", "red"))


def _as_chunk(arr, dim):
    """Normalize a policy action field to shape [horizon, dim]."""
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.squeeze(arr)
    if arr.ndim == 1:
        arr = arr.reshape(1, dim) if arr.shape[0] == dim else arr.reshape(-1, dim)
    return arr.reshape(-1, dim)


# --------------------------------------------------------------------------- #
# Policy backends
# --------------------------------------------------------------------------- #
class ServerPolicy:
    """Thin ZMQ client for the GR00T PolicyServer (mirrors gr00t PolicyClient's
    wire format: zmq REQ + msgpack_numpy)."""

    def __init__(self, host, port, timeout_ms=20000):
        import zmq
        import msgpack_numpy as mnp

        self._mnp = mnp
        self._zmq = zmq
        self.ctx = zmq.Context()
        self._connect(host, port, timeout_ms)
        self.host, self.port, self.timeout_ms = host, port, timeout_ms

    def _connect(self, host, port, timeout_ms):
        self.sock = self.ctx.socket(self._zmq.REQ)
        self.sock.setsockopt(self._zmq.RCVTIMEO, timeout_ms)
        self.sock.setsockopt(self._zmq.SNDTIMEO, timeout_ms)
        self.sock.connect(f"tcp://{host}:{port}")

    def _call(self, request):
        self.sock.send(self._mnp.packb(request))
        msg = self.sock.recv()
        if msg == b"ERROR":
            raise RuntimeError("policy server returned ERROR")
        return self._mnp.unpackb(msg, raw=False)

    def ping(self):
        try:
            self._call({"endpoint": "ping"})
            return True
        except Exception:
            self._connect(self.host, self.port, self.timeout_ms)
            return False

    def get_action(self, observation):
        resp = self._call(
            {
                "endpoint": "get_action",
                "data": {"observation": observation, "options": None},
            }
        )
        action, _info = resp[0], (resp[1] if len(resp) > 1 else {})
        return action


class ReplayPolicy:
    """Replays smpl_joints + hand joints from a recorded parquet (no server).

    Accepts either a collected parquet (``teleop.smpl_joints``) or a preprocessed
    one (``action.smpl_joints``)."""

    def __init__(self, parquet_path):
        import pandas as pd

        df = pd.read_parquet(parquet_path)

        def _col(*names):
            for n in names:
                if n in df.columns:
                    return np.stack(df[n].values).astype(np.float32)
            raise KeyError(f"none of {names} in parquet columns {list(df.columns)}")

        self.sj = _col("action.smpl_joints", "teleop.smpl_joints")
        self.lh = _col("teleop.left_hand_joints", "action.left_hand_joints")
        self.rh = _col("teleop.right_hand_joints", "action.right_hand_joints")
        self.i = 0

    def get_action(self, observation):
        j = min(self.i, len(self.sj) - 1)
        self.i += 1
        return {
            "smpl_joints": self.sj[j][None],
            "left_hand_joints": self.lh[j][None],
            "right_hand_joints": self.rh[j][None],
        }


class ZeroPolicy:
    """Zeros -> robot holds the SONIC default standing pose, hands open (sanity)."""

    def get_action(self, observation):
        return {
            "smpl_joints": np.zeros((1, SMPL_DIM), np.float32),
            "left_hand_joints": np.zeros((1, HAND_DIM), np.float32),
            "right_hand_joints": np.zeros((1, HAND_DIM), np.float32),
        }


def make_policy(args):
    if args.policy == "server":
        p = ServerPolicy(args.host, args.port)
        print(
            colored(
                f"[deploy] connecting to PolicyServer tcp://{args.host}:{args.port} ...",
                "cyan",
            )
        )
        print(
            colored(
                "[deploy] reachable"
                if p.ping()
                else "[deploy] WARNING: server not reachable (start gr00t PolicyServer)",
                "green" if p.ping() else "yellow",
            )
        )
        return p
    if args.policy == "replay":
        assert args.replay_parquet, "--policy replay needs --replay-parquet"
        return ReplayPolicy(args.replay_parquet)
    return ZeroPolicy()


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", default="TabletopPickPlace")
    parser.add_argument("--robot", default="G1FloatingBody")
    parser.add_argument(
        "--policy", default="server", choices=["server", "replay", "zero"]
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument(
        "--port",
        type=int,
        default=5550,
        help="gr00t PolicyServer port (run_gr00t_server.py default 5550).",
    )
    parser.add_argument("--replay-parquet", default=None)
    parser.add_argument(
        "--prompt",
        default=None,
        help="language instruction (default: the task's ep_meta lang).",
    )
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=600,
        help="max control steps per episode before timeout.",
    )
    parser.add_argument("--control-freq", type=int, default=50)
    # --- action-chunk handling, mirroring gear_sonic/run_vla_inference.py ---
    parser.add_argument(
        "--inference-rate",
        type=float,
        default=10.0,
        help="policy re-query rate (Hz). Control runs at "
        "--control-freq; the policy is re-queried every "
        "round(control_freq/inference_rate) steps, using the first "
        "actions of each fresh chunk. Real deploy uses 10Hz (TRT).",
    )
    parser.add_argument(
        "--action-horizon",
        type=int,
        default=64,
        help="action chunk length the model emits (must match "
        "training; only informational here -- the chunk is indexed "
        "as returned).",
    )
    parser.add_argument(
        "--smpl-ema-alpha",
        type=float,
        default=1.0,
        help="EMA blend at chunk boundaries (1.0 = snap to new "
        "prediction, <1 smooths smpl_joints + hand joints). Reduces "
        "motion discontinuity per chunk.",
    )
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument(
        "--fixed-scene",
        action="store_true",
        help="evaluate on the PERMANENTLY fixed deterministic scene "
        "(set by --scene-seed). Match how the data was collected.",
    )
    parser.add_argument(
        "--fixed-objects",
        action="store_true",
        help="same object instances (set by --scene-seed), randomized "
        "positions. Mutually exclusive with --fixed-scene.",
    )
    parser.add_argument(
        "--scene-seed",
        type=int,
        default=0,
        help="seed for --fixed-scene / --fixed-objects (match collection).",
    )
    parser.add_argument(
        "--render",
        default="video",
        choices=["video", "viewer"],
        help="'video' (default): headless, save an ego-view mp4 per "
        "episode (set --video-dir). 'viewer': open a live on-screen "
        "MuJoCo window to watch (needs a display; throttled to "
        "realtime). Both still render the ego camera the policy needs.",
    )
    parser.add_argument(
        "--video-dir",
        default=None,
        help="where to save per-episode ego mp4s (--render video).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--sonic-ckpt",
        type=str,
        default=DEFAULT_SONIC_CKPT,
        choices=list(SONIC_CKPT_CHOICES),
        help="which SONIC encoder/decoder ONNX checkpoint to load: "
        "'old' (release) or 'new' (low-latency). This pose path runs "
        "BOTH encoder and decoder, so the controller selects the "
        "matching encoder obs layout automatically. Default 'new' -- only "
        "the low-latency export ships in the sim2real checkpoint tree.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print model input/output diagnostics for the first "
        "few inferences (obs shapes/ranges, action keys, smpl_joints "
        "range + step-to-step delta) to debug a non-moving robot.",
    )
    args = parser.parse_args()
    # default video dir only matters for --render video
    if args.render == "video" and args.video_dir is None:
        args.video_dir = "outputs/vla_eval_pose"

    cfg_path = (
        pathlib.Path(robosuite.__file__).parent
        / "controllers/config/robots/default_g1_vla_pose.json"
    )
    controller_config = load_composite_controller_config(controller=str(cfg_path))
    apply_sonic_ckpt(controller_config, args.sonic_ckpt)
    print(colored(f"[deploy] SONIC ckpt: {args.sonic_ckpt}", "cyan"))

    scene_kwargs = {}
    if args.layout is not None:
        scene_kwargs["layout_ids"] = args.layout
    if args.style is not None:
        scene_kwargs["style_ids"] = args.style
    if args.fixed_scene:
        scene_kwargs["fixed_scene_seed"] = args.scene_seed
        print(
            colored(
                f"[deploy] FIXED scene (seed {args.scene_seed}) — matches "
                "fixed-scene collection.",
                "cyan",
            )
        )
    elif args.fixed_objects:
        scene_kwargs["fixed_objects_seed"] = args.scene_seed
        print(
            colored(
                f"[deploy] FIXED objects (seed {args.scene_seed}) — same "
                "mug/plate, randomized positions.",
                "cyan",
            )
        )

    viewer = args.render == "viewer"
    env = robosuite.make(
        env_name=args.env,
        robots=args.robot,
        controller_configs=controller_config,
        # both modes still render the ego camera offscreen (the policy's input);
        # 'viewer' additionally opens an on-screen mjviewer window to watch.
        has_renderer=viewer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[EGO_CAMERA],
        camera_heights=EGO_H,
        camera_widths=EGO_W,
        render_camera=None,
        control_freq=args.control_freq,
        ignore_done=True,
        renderer="mjviewer" if viewer else "mujoco",
        seed=args.seed,
        **scene_kwargs,
    )
    base = getattr(env, "unwrapped", env)
    print(
        colored(
            "[deploy] controller: sonic_pose_vla (reference-motion: predicted "
            "SMPL joints -> POSE encoder -> decoder)",
            "cyan",
        )
    )
    # NOTE: env.reset() rebuilds the composite controller, so we must RE-FETCH it
    # after every reset (below). Caching it once would leave set_vla_action writing
    # to a dead controller while the live one encodes a zero smpl ref -> the policy
    # never drives the robot (it just holds the SONIC default standing pose).
    policy = make_policy(args)

    save_video = (not viewer) and bool(args.video_dir)
    if save_video:
        pathlib.Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    print(
        colored(
            f"[deploy] render mode: {args.render}"
            + (
                f" (mp4 -> {args.video_dir})"
                if save_video
                else " (live window; close it or Ctrl-C to stop)"
                if viewer
                else ""
            ),
            "cyan",
        )
    )

    zero_ac = np.zeros(env.action_dim)
    replan_steps = max(
        1, int(round(args.control_freq / max(args.inference_rate, 1e-6)))
    )
    print(
        colored(
            f"[deploy] control {args.control_freq}Hz, inference {args.inference_rate}Hz "
            f"-> re-plan every {replan_steps} steps; smpl_ema_alpha={args.smpl_ema_alpha}",
            "cyan",
        )
    )
    n_success = 0
    ep_lengths = []

    for ep in range(args.episodes):
        obs = env.reset()
        composite = env.robots[0].composite_controller  # live controller (post-reset)
        prompt = args.prompt or base.get_ep_meta().get("lang", args.env)
        if hasattr(policy, "i"):
            policy.i = 0  # restart replay cursor
        frames = []
        chunk = None  # dict of [H, dim] arrays
        chunk_idx = 0
        new_chunk = False
        prev_sj = prev_lh = prev_rh = None
        success = False
        t0 = time.time()
        next_tick = time.monotonic()
        n_infer = 0
        dbg_ep = args.debug and ep == 0
        applied_smpl = []
        j_start = np.asarray(obs["robot0_joint_pos"], dtype=np.float32).copy()
        for step in range(args.max_steps):
            if chunk is None or chunk_idx >= replan_steps:
                obs_dict = build_observation(obs, prompt)
                dbg = args.debug and ep == 0 and n_infer < 5
                if dbg and n_infer == 0:
                    debug_observation(obs_dict)
                try:
                    action = policy.get_action(obs_dict)
                except Exception as e:
                    print(colored(f"[deploy] get_action FAILED: {e}", "red"))
                    raise
                if dbg:
                    debug_action(action, prev_sj)
                n_infer += 1
                sj = _as_chunk(_action_field(action, "smpl_joints"), SMPL_DIM)
                lh = _as_chunk(_action_field(action, "left_hand_joints"), HAND_DIM)
                rh = _as_chunk(_action_field(action, "right_hand_joints"), HAND_DIM)
                chunk = {
                    "sj": sj,
                    "lh": lh,
                    "rh": rh,
                    "len": min(len(sj), len(lh), len(rh)),
                }
                chunk_idx = 0
                new_chunk = True

            i = min(chunk_idx, chunk["len"] - 1)
            sjv = chunk["sj"][i].copy()
            lhj = chunk["lh"][i].copy()
            rhj = chunk["rh"][i].copy()
            # EMA-blend the first action of each new chunk with the last published
            # one to soften chunk-boundary discontinuities.
            if new_chunk and prev_sj is not None and args.smpl_ema_alpha < 1.0:
                a = args.smpl_ema_alpha
                sjv = a * sjv + (1 - a) * prev_sj
                lhj = a * lhj + (1 - a) * prev_lh
                rhj = a * rhj + (1 - a) * prev_rh
            new_chunk = False
            prev_sj, prev_lh, prev_rh = sjv.copy(), lhj.copy(), rhj.copy()

            composite.set_vla_action(
                smpl_joints=sjv, left_hand_joints=lhj, right_hand_joints=rhj
            )
            chunk_idx += 1
            if dbg_ep:
                applied_smpl.append(sjv.copy())

            obs = env.step(zero_ac)[0]
            if viewer:
                env.render()
                # throttle to realtime so the window is watchable
                next_tick += env.control_timestep
                sleep = next_tick - time.monotonic()
                if sleep > 0:
                    time.sleep(sleep)
                else:
                    next_tick = time.monotonic()
            elif save_video:
                frames.append(np.ascontiguousarray(obs[f"{EGO_CAMERA}_image"][::-1]))

            if bool(base._check_success()):
                success = True
                break

        n_success += int(success)
        ep_lengths.append(step + 1)
        dt = time.time() - t0
        print(
            colored(
                f"[deploy] ep {ep+1}/{args.episodes}: "
                f"{'SUCCESS' if success else 'timeout'} in {step+1} steps "
                f"({dt:.1f}s, {(step+1)/max(dt,1e-6):.0f} steps/s)",
                "green" if success else "yellow",
            )
        )

        if dbg_ep and applied_smpl:
            sjs = np.stack(applied_smpl)
            j_move = float(
                np.linalg.norm(
                    np.asarray(obs["robot0_joint_pos"], dtype=np.float32) - j_start
                )
            )
            sj_tstd = float(sjs.std(axis=0).mean())  # per-dim temporal std
            sj_drift = float(np.linalg.norm(sjs[-1] - sjs[0]))
            print(
                colored(
                    "[debug] --- episode diagnosis ---\n"
                    f"[debug]   applied smpl_joints temporal std (per-dim, over episode): "
                    f"{sj_tstd:.4f}\n"
                    f"[debug]   smpl drift |last-first|: {sj_drift:.4f}\n"
                    f"[debug]   joint movement |last-first|: {j_move:.3f}\n"
                    f"[debug]   -> smpl std ~0 => policy outputs a CONSTANT reference "
                    "(obs/model issue); smpl varies but joints don't move => "
                    "encode/decode/apply issue.",
                    "magenta",
                )
            )

        if save_video and frames:
            try:
                import imageio

                vp = (
                    pathlib.Path(args.video_dir) / f"{args.env}_ep{ep+1:03d}_"
                    f"{'success' if success else 'timeout'}.mp4"
                )
                imageio.mimwrite(vp, frames, fps=args.control_freq)
                print(colored(f"[deploy]   saved {vp}", "cyan"))
            except Exception as e:
                print(colored(f"[deploy]   video save failed: {e}", "yellow"))

    print(
        colored(
            f"\n[deploy] === {args.env}: success {n_success}/{args.episodes} "
            f"({100.0*n_success/max(args.episodes,1):.0f}%), "
            f"mean episode length {np.mean(ep_lengths):.0f} steps ===",
            "green",
        )
    )


if __name__ == "__main__":
    main()
