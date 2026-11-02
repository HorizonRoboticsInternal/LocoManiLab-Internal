"""Closed-loop in-sim validation of a trained GR00T ``unitree_g1_sonic`` VLA policy.

This is the INVERSE of ``collect_demos_g1.py``: instead of a human
teleoperating and recording (motion_token + hand joints), a trained GR00T policy
is queried every step and its predicted action drives the robot, so you can
measure task success rate in the same sim the data came from.

Architecture (mirrors ``gear_sonic/scripts/run_vla_inference.py``, sim-adapted):
  * the GR00T policy is served by a separate ``gr00t`` PolicyServer (TRT engine,
    started in the Isaac-GR00T env). This script is a thin ZMQ client to it
    (``--policy server --host H --port P``), so the heavy GR00T/TRT runtime stays
    in its own env while the sim runs in ``tele_benchmark``.
  * the env uses the ``sonic_vla`` G1 controller: the policy's 64-D ``motion_token``
    is fed to the SONIC whole-body DECODER (``composite.set_vla_action``) and the
    predicted 7+7 hand joints PD-drive the fingers. (See default_g1_vla.json.)

Observation sent to the policy mirrors run_vla_inference: nested
``video.ego_view`` (head cam), ``state.{left_leg,right_leg,waist,left_arm,
left_hand,right_arm,right_hand,projected_gravity}`` (slices of the 43-DoF joint
vector) and the language prompt.

Usage (start the policy server first, in the Isaac-GR00T env, serving your TRT
model on tcp://HOST:PORT), then:

    conda activate tele_benchmark
    python -m robocasa.scripts.deploy_living_room_vla \
        --env TabletopPickPlace --host localhost --port 5550 \
        --prompt "pick up the mug and place it on the plate" \
        --episodes 10 --max-steps 600 --video-dir outputs/vla_eval

Server-free smoke tests of the sim loop (no GR00T needed):
    # replay recorded tokens from a collected episode (open-loop sanity):
    ... --policy replay --replay-parquet <episode_000000.parquet>
    # zeros (robot just holds the SONIC default pose):
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

# 43-DoF MuJoCo joint-vector slices -> the unitree_g1_sonic state groups
# (identical to G1_modality.json / the embodiment config).
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
    """Assemble the GR00T observation dict from a robosuite obs (mirrors
    run_vla_inference.prepare_observation_from_sensors, sim-adapted).

    All state/video tensors carry leading [batch=1, time=1] dims, as the policy
    expects.
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


def debug_action(action, prev_tok):
    print(colored(f"[debug]   action keys: {list(action.keys())}", "magenta"))
    for k in ("motion_token", "left_hand_joints", "right_hand_joints"):
        try:
            v = _action_field(action, k)
            line = f"[debug]   {k}: {_stat(v)}"
            if k == "motion_token":
                ch = _as_chunk(v, 64)  # [horizon, 64]
                # within-chunk horizon variation: is the chunk a TRAJECTORY (the
                # tokens change across the 64 future steps) or FLAT? Recorded data
                # has per-step |Δ| ~0.06; near-0 here => the model predicts "hold".
                if len(ch) > 1:
                    hd = np.linalg.norm(np.diff(ch, axis=0), axis=1)
                    line += (
                        f"  | within-chunk per-step|Δ| med={np.median(hd):.4f} "
                        f"(recorded ~0.06), token[0]->[-1]={np.linalg.norm(ch[-1]-ch[0]):.4f}"
                    )
                if prev_tok is not None:
                    line += (
                        f"  | Δ(prev applied)={np.linalg.norm(ch[0] - prev_tok):.4f}"
                    )
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
    """Replays motion_token + hand joints from a recorded parquet (no server)."""

    def __init__(self, parquet_path):
        import pandas as pd

        df = pd.read_parquet(parquet_path)
        self.tok = np.stack(df["action.motion_token"].values).astype(np.float32)
        self.lh = np.stack(df["teleop.left_hand_joints"].values).astype(np.float32)
        self.rh = np.stack(df["teleop.right_hand_joints"].values).astype(np.float32)
        self.i = 0

    def get_action(self, observation):
        j = min(self.i, len(self.tok) - 1)
        self.i += 1
        return {
            "motion_token": self.tok[j][None],
            "left_hand_joints": self.lh[j][None],
            "right_hand_joints": self.rh[j][None],
        }


class ZeroPolicy:
    """Zeros -> robot holds the SONIC default pose, hands open (sanity)."""

    def get_action(self, observation):
        return {
            "motion_token": np.zeros((1, 64), np.float32),
            "left_hand_joints": np.zeros((1, 7), np.float32),
            "right_hand_joints": np.zeros((1, 7), np.float32),
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
        "--token-ema-alpha",
        type=float,
        default=1.0,
        help="EMA blend at chunk boundaries (1.0 = snap to new "
        "token, <1 smooths). Mirrors run_vla_inference "
        "TOKEN_EMA_ALPHA; reduces motion discontinuity per chunk.",
    )
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument(
        "--scene-id",
        type=int,
        default=None,
        help="pin an iTHOR FloorPlan number -- THE SCENE THE DATA WAS "
        "COLLECTED IN. Without it an iTHOR env picks a random FloorPlan of "
        "that room type on every reset, so the policy is tested in rooms it "
        "never trained in and the success rate is not comparable to a "
        "same-scene number. The dataset name's suffix IS the FloorPlan "
        "(IThorKitchenCloseOven_11 -> 11).",
    )
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
        "'old' (release) or 'new' (low-latency). The VLA token path "
        "only runs the decoder, which is identical across variants. "
        "Default 'new' -- only the low-latency export ships in the "
        "sim2real checkpoint tree.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="print model input/output diagnostics for the first "
        "few inferences (obs shapes/ranges, action keys, motion_token "
        "range + step-to-step delta) to debug a non-moving robot.",
    )
    args = parser.parse_args()
    # default video dir only matters for --render video
    if args.render == "video" and args.video_dir is None:
        args.video_dir = "outputs/vla_eval"

    cfg_path = (
        pathlib.Path(robosuite.__file__).parent
        / "controllers/config/robots/default_g1_vla.json"
    )
    controller_config = load_composite_controller_config(controller=str(cfg_path))
    apply_sonic_ckpt(controller_config, args.sonic_ckpt)
    print(colored(f"[deploy] SONIC ckpt: {args.sonic_ckpt}", "cyan"))

    scene_kwargs = {}
    if args.layout is not None:
        scene_kwargs["layout_ids"] = args.layout
    if args.style is not None:
        scene_kwargs["style_ids"] = args.style
    if args.scene_id is not None:
        # iTHOR envs only (IThorKitchenEnv.__init__); kitchen/living-room envs
        # take layout/style instead and would reject this.
        scene_kwargs["scene_ids"] = args.scene_id
        print(colored(f"[deploy] iTHOR scene pinned to FloorPlan{args.scene_id}", "cyan"))
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
    # NOTE: env.reset() rebuilds the composite controller, so we must RE-FETCH it
    # after every reset (below). Caching it here once would leave set_vla_action
    # writing to a dead controller while the live one decodes a zero token -> the
    # policy never drives the robot (it just holds the SONIC default pose).
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
    # Re-query the policy every this many control steps, then walk the fresh
    # chunk's first actions (mirrors the 50Hz-control / inference-rate-Hz-inference
    # decoupling of the real deploy; sim is non-realtime so we re-plan synchronously
    # -- no async thread / latency skip needed).
    replan_steps = max(
        1, int(round(args.control_freq / max(args.inference_rate, 1e-6)))
    )
    print(
        colored(
            f"[deploy] control {args.control_freq}Hz, inference {args.inference_rate}Hz "
            f"-> re-plan every {replan_steps} steps; token_ema_alpha={args.token_ema_alpha}",
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
        prev_tok = prev_lh = prev_rh = None
        success = False
        t0 = time.time()
        next_tick = time.monotonic()
        n_infer = 0
        dbg_ep = args.debug and ep == 0
        applied_tokens = []
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
                    debug_action(action, prev_tok)
                n_infer += 1
                mt = _as_chunk(_action_field(action, "motion_token"), 64)
                lh = _as_chunk(_action_field(action, "left_hand_joints"), 7)
                rh = _as_chunk(_action_field(action, "right_hand_joints"), 7)
                if np.abs(mt).max() > 1.25:
                    print(
                        colored(
                            f"[deploy] warn: |motion_token| max "
                            f"{np.abs(mt).max():.3f} > 1.25",
                            "yellow",
                        )
                    )
                chunk = {
                    "mt": mt,
                    "lh": lh,
                    "rh": rh,
                    "len": min(len(mt), len(lh), len(rh)),
                }
                chunk_idx = 0
                new_chunk = True

            i = min(chunk_idx, chunk["len"] - 1)
            tok = chunk["mt"][i].copy()
            lhj = chunk["lh"][i].copy()
            rhj = chunk["rh"][i].copy()
            # EMA-blend the first action of each new chunk with the last published
            # one to soften chunk-boundary discontinuities (TOKEN_EMA_ALPHA).
            if new_chunk and prev_tok is not None and args.token_ema_alpha < 1.0:
                a = args.token_ema_alpha
                tok = a * tok + (1 - a) * prev_tok
                lhj = a * lhj + (1 - a) * prev_lh
                rhj = a * rhj + (1 - a) * prev_rh
            new_chunk = False
            prev_tok, prev_lh, prev_rh = tok.copy(), lhj.copy(), rhj.copy()

            composite.set_vla_action(tok, lhj, rhj)
            chunk_idx += 1
            if dbg_ep:
                applied_tokens.append(tok.copy())

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

        if dbg_ep and applied_tokens:
            toks = np.stack(applied_tokens)
            j_move = float(
                np.linalg.norm(
                    np.asarray(obs["robot0_joint_pos"], dtype=np.float32) - j_start
                )
            )
            tok_tstd = float(toks.std(axis=0).mean())  # per-dim temporal std
            tok_drift = float(np.linalg.norm(toks[-1] - toks[0]))
            print(
                colored(
                    "[debug] --- episode diagnosis ---\n"
                    f"[debug]   applied motion_token temporal std (per-dim, over episode): "
                    f"{tok_tstd:.4f}   (recorded training data ~0.10)\n"
                    f"[debug]   token drift |last-first|: {tok_drift:.4f}\n"
                    f"[debug]   joint movement |last-first|: {j_move:.3f}\n"
                    f"[debug]   -> token std ~0 => policy outputs a CONSTANT token "
                    "(obs/model issue, NOT the decoder); token varies but joints don't "
                    "move => decode/apply issue.",
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
