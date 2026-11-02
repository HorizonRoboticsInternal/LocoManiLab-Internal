"""Replay a fixed-task collection run with randomized appearance -> augmented data.

For every appearance variant this replays each recorded episode ACTION-FOR-ACTION
through the same G1_BRIDGE backend that collected it, in a scene whose textures /
lights are randomized at the XML level (``scene_appearance_randomizer``) and whose
ego camera is perturbed post-compile (``robosuite`` ``CameraModder``). Appearance
never touches physics, and each episode starts from the EXACT recorded start pose
(the per-reset ``g1_xy_jitter`` in ``fixed_task_resets.jsonl``), so the replayed
trajectory is the recorded one and the source labels stay valid: the output run
copies the source action/state columns verbatim and only re-renders the videos.

An episode whose replay does NOT reach ``_check_success`` has diverged from the
recording -- its new video would not match the copied labels -- so it is dropped
from the output (``--keep-failed`` retains it for debugging). Kept episodes are
renumbered 0..N-1 (parquet ``episode_index``/``index`` columns, file names,
``meta/episodes*.jsonl``, ``meta/info.json`` totals); the source episode each one
came from is recorded in the ``augmentation`` block.

Usage (conda activate tele_benchmark; MUJOCO_GL=egl for headless):

    python -m robocasa.scripts.replay_augment \
        --source-run outputs_sim/sonic/IThorKitchenOpenOven_fixed_<ts> \
        --variants 4 --seed 0

One output run per variant: ``<out-base>/<env>_aug_s<variant_seed>_<ts>``.
"""

from __future__ import annotations

import argparse
import datetime
import json
import pathlib
import shutil
import sys
import time

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[2]


def _guard_imports() -> None:
    """Refuse to run against a robocasa/robosuite shadowing this checkout."""
    import robocasa
    import robosuite

    for mod in (robocasa, robosuite):
        path = pathlib.Path(mod.__file__).resolve()
        if _REPO not in path.parents:
            raise SystemExit(
                f"{mod.__name__} resolves to {path}, outside {_REPO} -- another "
                f"install is shadowing this repo. Put {_REPO} first on PYTHONPATH."
            )
        print(f"[aug] {mod.__name__}: {path}")


# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-run", required=True,
                   help="a fixed-task collection run dir (has fixed_task_manifest"
                        ".json / fixed_task_resets.jsonl / data / meta / videos)")
    p.add_argument("--out-base", default=None,
                   help="parent dir for the variant runs. Default is a SIBLING "
                        "lane named '<lane>_replay' (e.g. outputs_sim/sonic -> "
                        "outputs_sim/sonic_replay), so augmented runs never mix "
                        "with the originals they were replayed from")
    p.add_argument("--variants", type=int, default=1)
    p.add_argument("--seed", type=int, default=0,
                   help="base seed; variant v uses seed+v for its appearance")
    p.add_argument("--episodes", default="all", help="'all' or comma list of indices")
    p.add_argument("--keep-failed", action="store_true",
                   help="ALSO write episodes whose replay did not reach success "
                        "(their video does not match the copied labels -- debug "
                        "only, never train on them). Default: drop them.")

    p.add_argument("--viewer", action="store_true",
                   help="open the interactive MuJoCo window and step the replay "
                        "in it (needs a display; drop MUJOCO_GL=egl). Watching "
                        "is the point, so this also implies --no-write unless "
                        "you ask for output explicitly with --write.")
    p.add_argument("--write", dest="write", action="store_true", default=None,
                   help="write the output run even under --viewer")
    p.add_argument("--no-write", dest="write", action="store_false",
                   help="replay without producing an output run -- just report "
                        "which episodes still reach success")
    p.add_argument("--randomize", default="standard", metavar="SPEC",
                   help="what to randomize, in the SAME spec syntax as "
                        "`lab eval --randomize` and preview_randomization: "
                        "comma-separated tokens, each with an optional "
                        "=value, plus the presets off/mild/standard/stress "
                        "which expand in place and can be overridden by later "
                        "tokens (e.g. 'standard,textures=0.4'). Pass "
                        "--randomize help for the full token reference. "
                        "Augmentation draws from the 'train' half of the "
                        "texture pool by default, so an eval with the same "
                        "spec (which draws 'test') measures generalization.")

    p.add_argument("--reroll", default="0", metavar="N", type=str,
                   help="re-roll the texture/light randomization every N EPISODES: "
                        "0 (default) = never, one coherent room for the whole "
                        "output run; 1 = every episode, max visual diversity; "
                        "N = every N episodes. Camera, lens, robot tint and "
                        "distractor placement are per-episode regardless.")
    p.add_argument("--randomize-object-textures", action="store_true",
                   help="ALSO re-texture the task objects themselves (their "
                        "own model.xml textures; geometry and collision are "
                        "untouched, so the replayed physics is identical). "
                        "Off by default: the language instruction still names "
                        "the original object (\"apple\", \"basket\"), so this "
                        "trades instruction-appearance grounding for color/"
                        "texture robustness -- opt in deliberately.")

    p.add_argument("--distractor-groups",
                   default="cup,mug,banana,can,bowl,sponge,orange,lemon",
                   help="comma-separated object groups distractors are "
                        "sampled from (see kitchen_object_utils.OBJ_GROUPS)")
    p.add_argument("--distractor-radius", type=float, default=0.35,
                   help="max distance (m) from the chosen task object")
    p.add_argument("--distractor-clearance", type=float, default=0.12,
                   help="min distance (m) to every task object and other "
                        "distractors")
    p.add_argument("--distractor-corridor", type=float, default=0.15,
                   help="min distance (m) to the robot-spawn -> task-object "
                        "approach line")

    p.add_argument("--inference-backend", default=None,
                   help="whole-body controller runtime (onnx-gpu/onnx-cpu/...); "
                        "default: onnx-gpu when available")
    p.add_argument("--preroll-ticks", type=int, default=None,
                   help="feed the FIRST recorded action for N extra ticks "
                        "before capture starts, so the closed-loop controller "
                        "converges onto the recorded start pose instead of its "
                        "own settle stance. Default: 150 for the tracking-class "
                        "backends whose command space is a reference pose "
                        "(ref_qpos/smpl -- heft/scalebfm; measured 3/24 -> "
                        "21/24 kept on heft), 0 otherwise (holding sonic's "
                        "motion_token latent that long drifts: 10/10 -> 9/10).")
    p.add_argument("--tail-steps", type=int, default=150,
                   help="extra steps after the last recorded action, holding it, "
                        "so a success that lands on the final frames can register")
    p.add_argument("--settle-max", type=int, default=1000)
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
def parse_reroll(value: str) -> int | None:
    """Re-roll period in episodes: ``0`` -> None (never), ``N`` -> every N.

    One number covers every case, and 0 means "off" exactly like it does for
    every other knob in this pipeline (``--lens 0``, ``light-geom=0``,
    ``--cam-pos 0``, ``--jitter-g1 0``).
    """
    try:
        period = int(str(value).strip())
    except ValueError:
        raise SystemExit(
            f"--reroll {value!r}: expected a number of episodes -- 0 = one "
            f"appearance for the whole run, 1 = a fresh one every episode, "
            f"N = every N") from None
    if period < 0:
        raise SystemExit("--reroll must be >= 0")
    return period or None


def appearance_seed_for(period: int | None, variant_seed: int, ep: int) -> int:
    """The appearance seed episode ``ep`` draws from.

    Episodes sharing a bucket share a seed, so they share an appearance --
    that is what makes 'every N episodes' a block of N matching rooms rather
    than N unrelated ones.
    """
    if period is None:
        return variant_seed
    return variant_seed * 10000 + ep // period


def load_reset_jitters(run: pathlib.Path) -> dict[int, dict]:
    """episode index -> its reset entry. Episode k was recorded right after the
    reset whose ``saved_episodes`` == k; keep the LAST such entry (a discarded
    attempt inserts an extra reset with the same count)."""
    out: dict[int, dict] = {}
    with open(run / "fixed_task_resets.jsonl") as fh:
        for line in fh:
            entry = json.loads(line)
            out[int(entry["saved_episodes"])] = entry
    return out


def load_action_frames(parquet_path: pathlib.Path, backend: str) -> list[dict]:
    """Recorded per-tick actions, keyed the way ``set_vla_action`` expects.

    Matches ``locomanilab.runtime.sim_client.load_replay_frames`` without depending
    on the main CLI package: legacy sonic columns by name, blob-schema backends by slicing
    their packed action columns via ``g1_bridge.modality.layout_for``.
    """
    import pandas as pd
    from g1_bridge.modality import layout_for

    df = pd.read_parquet(parquet_path)
    groups = []
    blobs = getattr(layout_for(backend), "blobs", None) or []
    for blob in blobs:
        if blob.section != "action":
            continue
        offset = 0
        for g in blob.groups:
            groups.append((g.name, blob.feature_name, offset, offset + g.dim))
            offset += g.dim

    def col(name):
        for c in (f"action.{name}", f"teleop.{name}", name):
            if c in df.columns:
                return np.stack(df[c].values).astype(np.float32)
        for gname, feature, lo, hi in groups:
            if gname == name and feature in df.columns:
                return np.stack(df[feature].values).astype(np.float32)[:, lo:hi]
        return None

    # sonic's legacy schema is one column per signal; blob backends declare
    # their group names themselves.
    keys = [name for name, *_ in groups] or [
        "motion_token", "left_hand_joints", "right_hand_joints"]
    cols = {k: col(k) for k in keys}
    missing = [k for k, v in cols.items() if v is None]
    if missing:
        raise SystemExit(f"{parquet_path}: missing action columns {missing}")
    n = min(len(v) for v in cols.values())
    return [{k: v[i] for k, v in cols.items()} for i in range(n)]


# --------------------------------------------------------------------------- #
EGO_CAMERA = "robot0_head_camera"


def build_env(manifest: dict, spec, args):
    """The collection env, with the appearance randomizer installed on it.

    The randomizer rides robosuite's ``set_xml_processor`` seam, which runs on
    the merged model xml at every hard reset, so the caller can point it at a
    different appearance seed between episodes (``--reroll 1``)
    or leave it fixed for the whole variant. ``appearance_cfg`` of None
    installs nothing. The appearance profile is resolved from the env, so a
    bare ``lights`` gets the value that scene family actually wants. Returns
    (env, restore_fn, object_name_map, ep_meta, appearance_handle).
    """
    import robosuite
    from robosuite.environments.base import REGISTERED_ENVS

    import robocasa  # noqa: F401  -- registers the task envs
    from robocasa.g1_host.bridge import load_sim2real_controller_config

    env_name = manifest["env"]
    scene = manifest["scene"]
    robot = manifest["robot"]


    controller_config = load_sim2real_controller_config(
        backend=robot["backend"],
        pico_source="mock",  # nothing reads it on the replay path
        policy_config=robot.get("policy_config"),
        inference_backend=args.inference_backend or _default_inference_backend(),
    )
    try:
        env = robosuite.make(
            env_name=env_name,
            robots=robot["name"],
            controller_configs=controller_config,
            has_renderer=bool(args.viewer),
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=[EGO_CAMERA],
            camera_heights=240,
            camera_widths=320,
            render_camera=None,
            control_freq=50,
            ignore_done=True,
            # mjviewer, not the default opencv on-screen renderer: that one
            # goes through cv2.imshow, which a headless-built opencv refuses
            # ("rebuild with GTK+ support"). mjviewer is also what this repo's
            # own teleop/preview windows use, and env.step() pumps it, so no
            # render() call is needed in the replay loop.
            renderer="mjviewer" if args.viewer else "mujoco",
            seed=0,
            scene_ids=scene["scene_id"],
            fixed_scene_seed=scene["scene_seed"] if scene.get("fixed_scene") else None,
        )
    except Exception:
        raise
    # Pin the episode metadata the collection run recorded: scene name, robot
    # base pose, and -- crucially -- the OBJECT CFGS (their names, asset
    # instances and placement regions). A fresh env samples its own cfgs, which
    # may use different names and different asset instances, so the reset-log
    # object jitter could never be re-applied and the scene would not be the
    # recorded one. Every hard reset re-reads this via ``_setup_model``.
    name_map: dict[str, str] = {}
    ep_meta = manifest.get("task", {}).get("ep_meta")
    if ep_meta:
        ep_meta = json.loads(json.dumps(ep_meta))  # private copy
        name_map = _remap_object_names(ep_meta, getattr(env, "object_cfgs", []))
        env.set_ep_meta(ep_meta)

    from robocasa.utils.appearance import install_appearance, profile_for

    profile = profile_for(env)
    appearance = install_appearance(env, spec.appearance_config(profile), profile)
    print(spec.format_resolved(args.randomize, profile, seed=args.seed),
          flush=True)

    def restore():
        if appearance is not None:
            appearance.restore()

    return env, restore, name_map, ep_meta or {}, appearance


def _remap_object_names(ep_meta: dict, fresh_cfgs: list[dict]) -> dict[str, str]:
    """Rename recorded object cfgs to the names the CURRENT task class expects.

    Success checks key ``env.objects`` by the class's current names (e.g.
    ``item``/``receptacle``); a run collected before an object rename carries
    the old names. ``obj_groups`` (apple/bowl/...) is stable across renames, so
    match on it against the cfgs the make()-time fresh build sampled. Returns
    {recorded name -> current name} for remapping the reset-log jitter too.
    """
    old_cfgs = ep_meta.get("object_cfgs") or []
    remaining = list(fresh_cfgs)
    name_map: dict[str, str] = {}
    for cfg in old_cfgs:
        match = next((f for f in remaining
                      if str(f.get("obj_groups")) == str(cfg.get("obj_groups"))),
                     None)
        if match is None:
            continue
        remaining.remove(match)
        if match["name"] != cfg["name"]:
            name_map[cfg["name"]] = match["name"]
            cfg["name"] = match["name"]
    if name_map:
        print(f"[aug] object cfgs renamed to the current task class's names: "
              f"{name_map}")
    return name_map


def _default_inference_backend() -> str:
    try:
        import onnxruntime as ort

        if "CUDAExecutionProvider" in ort.get_available_providers():
            return "onnx-gpu"
    except Exception:
        pass
    return "onnx-cpu"


def apply_recorded_jitter(env, entry: dict, name_map: dict[str, str] | None = None) -> None:
    """Re-apply the exact XY offsets this episode's reset drew.

    The deterministic analogue of ``collect_fixed_task.apply_reset_pose_jitter``
    (``resets.jsonl`` records after-minus-before deltas on the same joints),
    including its bookkeeping so the env's own notion of the base pose agrees.
    """
    g1 = np.asarray(entry.get("g1_xy_jitter", (0.0, 0.0)), dtype=float)
    name_map = name_map or {}
    objects = {name_map.get(k, k): v
               for k, v in (entry.get("object_xy_jitter", {}) or {}).items()}

    joint = env._get_robot_free_joint()
    if joint is not None and np.any(g1 != 0.0):
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        qpos[:2] += g1
        env.sim.data.set_joint_qpos(joint, qpos)
    for name, delta in objects.items():
        obj = env.objects.get(name)
        obj_joint = obj.joints[0] if obj is not None and obj.joints else name
        try:
            qpos = env.sim.data.get_joint_qpos(obj_joint).copy()
        except Exception:
            print(f"[aug]   object joint {obj_joint!r} not found; skipped")
            continue
        qpos[:2] += np.asarray(delta, dtype=float)
        env.sim.data.set_joint_qpos(obj_joint, qpos)
    env.sim.forward()
    if hasattr(env, "_constrain_initial_jitter_candidate"):
        env._constrain_initial_jitter_candidate()
    if joint is not None and hasattr(env, "_robot_base_xy"):
        env.robot_base_pos[:2] = env._robot_base_xy()
        env.robot_base_ori[2] = env._robot_base_yaw()


def replay_episode(env, frames: list[dict], args,
                   image_fn=None) -> tuple[list[np.ndarray], dict]:
    """Feed the recorded actions tick-for-tick; capture the ego image the tick
    each action was recorded against (obs_i pairs with action_i, collection's
    frame semantics). Returns (captured images, report)."""
    zero = np.zeros(env.action_dim)
    composite = env.robots[0].composite_controller  # rebuilt by every reset

    settle = 0
    obs = None
    while not getattr(composite, "accepts_vla_action", True):
        obs = env.step(zero)[0]
        settle += 1
        if settle >= args.settle_max:
            raise RuntimeError(f"controller never accepted actions "
                               f"after {settle} settle steps")
    if obs is None:  # controller was ready immediately
        # force_update: the cached observation predates the jitter restore and
        # the distractor teleport -- without it, frame 0 shows the pre-restore
        # scene (visible as distractors popping in at frame 1).
        try:
            obs = env._get_observations(force_update=True)
        except TypeError:
            obs = env._get_observations()

    # Start-pose alignment: hold the episode's first action so the controller
    # tracks its way onto the recorded start pose before frame 0 is captured.
    # The first push carries the adapter's one-per-episode motion_first_frame
    # stamp, so the preroll and the episode form one continuous reference
    # stream -- exactly as if the stream had simply started a little earlier.
    for _ in range(max(0, args.preroll_ticks)):
        composite.set_vla_action(dict(frames[0]))
        obs = env.step(zero)[0]

    images: list[np.ndarray] = []
    success_step = None
    base = getattr(env, "unwrapped", env)
    # Watching wants wall-clock pacing: headless this loop runs as fast as the
    # physics allows, which makes a 443-frame episode flash past in the window.
    # Cap it at the control rate; when a step already costs more than its
    # budget this is a no-op.
    budget = env.control_timestep if getattr(args, "viewer", False) else 0.0
    for i, frame in enumerate(frames):
        tick = time.monotonic()
        img = np.ascontiguousarray(obs[f"{EGO_CAMERA}_image"][::-1])
        images.append(image_fn(img) if image_fn is not None else img)
        composite.set_vla_action(dict(frame))
        obs = env.step(zero)[0]
        if success_step is None and bool(base._check_success()):
            success_step = i + 1
        if budget:
            time.sleep(max(0.0, budget - (time.monotonic() - tick)))
    for t in range(args.tail_steps):  # hold the last action; no capture
        if success_step is not None:
            break
        obs = env.step(zero)[0]
        if bool(base._check_success()):
            success_step = len(frames) + t + 1
    report = dict(n_frames=len(frames), settle_steps=settle,
                  success=success_step is not None, success_step=success_step)
    return images, report


# --------------------------------------------------------------------------- #
# Output writing: only KEPT episodes land in the run, renumbered 0..N-1. The
# label columns are copied verbatim from the source parquet; only the
# episode_index / global index bookkeeping (and the file names) change.

def init_run_skeleton(source: pathlib.Path, out_run: pathlib.Path) -> None:
    (out_run / "data" / "chunk-000").mkdir(parents=True)
    (out_run / "meta").mkdir()
    for name in ("tasks.jsonl", "modality.json"):
        if (source / "meta" / name).exists():
            shutil.copy2(source / "meta" / name, out_run / "meta" / name)
    if (source / "fixed_task_manifest.json").exists():
        shutil.copy2(source / "fixed_task_manifest.json",
                     out_run / "fixed_task_manifest.json")


def write_reset_jitters(source: pathlib.Path, out_run: pathlib.Path,
                        kept: list[dict]) -> None:
    """Re-key ``fixed_task_resets.jsonl`` onto the OUTPUT episode numbering.

    Kept episodes are renumbered 0..N-1, and the sidecar is keyed by
    ``saved_episodes`` -- copying it verbatim would hand episode k the jitter
    recorded for a different episode as soon as anything was dropped, and the
    output run is itself replayable, so that error would compound.
    """
    src = source / "fixed_task_resets.jsonl"
    if not src.exists():
        return
    by_source = load_reset_jitters(source)
    lines = []
    for entry in kept:
        recorded = by_source.get(entry["source_episode"])
        if recorded is None:
            continue
        out = dict(recorded)
        out["saved_episodes"] = entry["new_index"]
        out["reset_index"] = entry["new_index"]
        out["source_episode"] = entry["source_episode"]
        lines.append(json.dumps(out))
    (out_run / "fixed_task_resets.jsonl").write_text(
        "\n".join(lines) + ("\n" if lines else ""))


def write_episode(source_parquet: pathlib.Path, out_run: pathlib.Path,
                  new_idx: int, frame_offset: int,
                  images: list[np.ndarray], fps: int) -> int:
    """One kept episode: reindexed parquet copy + freshly rendered video."""
    import imageio
    import pandas as pd

    df = pd.read_parquet(source_parquet)
    df["episode_index"] = np.int64(new_idx)
    df["index"] = np.arange(frame_offset, frame_offset + len(df), dtype=np.int64)
    df.to_parquet(out_run / "data" / "chunk-000" / f"episode_{new_idx:06d}.parquet")

    vdir = out_run / "videos" / "chunk-000" / "observation.images.ego_view"
    vdir.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(vdir / f"episode_{new_idx:06d}.mp4", fps=fps,
                            codec="libx264", pixelformat="yuv420p") as writer:
        for img in images:
            writer.append_data(img)
    return len(df)


def _uniform_index_stats(offset: int, n: int) -> dict:
    idx = np.arange(offset, offset + n, dtype=np.float64)
    return dict(min=[int(idx[0])], max=[int(idx[-1])],
                mean=[float(idx.mean())], std=[float(idx.std())], count=[n])


def write_meta(source: pathlib.Path, out_run: pathlib.Path,
               kept: list[dict]) -> None:
    """episodes.jsonl / episodes_stats.jsonl / info.json for the kept subset.

    Source lines are reused with the index bookkeeping fixed up; the per-column
    data stats stay valid because the data columns are verbatim copies (image
    stats are inherited and approximate -- noted in the augmentation block).
    """
    def load_jsonl(path):
        with open(path) as fh:
            return {int(json.loads(l)["episode_index"]): json.loads(l) for l in fh}

    src_eps = load_jsonl(source / "meta" / "episodes.jsonl")
    src_stats = load_jsonl(source / "meta" / "episodes_stats.jsonl")

    with open(out_run / "meta" / "episodes.jsonl", "w") as fh:
        for k in kept:
            entry = dict(src_eps[k["source_episode"]])
            entry["episode_index"] = k["new_index"]
            fh.write(json.dumps(entry) + "\n")

    with open(out_run / "meta" / "episodes_stats.jsonl", "w") as fh:
        for k in kept:
            entry = dict(src_stats[k["source_episode"]])
            entry["episode_index"] = k["new_index"]
            stats = dict(entry["stats"])
            n = k["n_frames"]
            stats["index"] = _uniform_index_stats(k["frame_offset"], n)
            stats["episode_index"] = dict(
                min=[k["new_index"]], max=[k["new_index"]],
                mean=[float(k["new_index"])], std=[0.0], count=[n])
            entry["stats"] = stats
            fh.write(json.dumps(entry) + "\n")

    info = json.loads((source / "meta" / "info.json").read_text())
    total_frames = sum(k["n_frames"] for k in kept)
    info["total_episodes"] = len(kept)
    info["total_frames"] = total_frames
    info["total_videos"] = len(kept)
    info["total_chunks"] = 1
    info["splits"] = {"train": f"0:{len(kept)}"}
    (out_run / "meta" / "info.json").write_text(json.dumps(info, indent=4))


def annotate_run(out_run: pathlib.Path, block: dict) -> None:
    info_path = out_run / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    target = info.get("collection_info")
    (target if isinstance(target, dict) else info)["augmentation"] = block
    info_path.write_text(json.dumps(info, indent=4))
    (out_run / "replay_augment_report.json").write_text(json.dumps(block, indent=2))


# --------------------------------------------------------------------------- #
def main(argv=None) -> int:
    args = parse_args(argv)
    _guard_imports()
    from robocasa.utils.appearance import AppearanceConfig
    from robocasa.utils.scene_appearance_randomizer import (
        cleanup_variant_xml, randomize_scene_xml)
    from robocasa.utils.scene_randomization import (
        LensWarp, build_distractor_cfgs, capture_baseline, parse_randomize_spec,
        randomize_camera, randomize_help, reset_to_baseline,
        teleport_distractors, tint_robot_geoms)

    if args.randomize.strip() in ("help", "?", "list"):
        print(randomize_help())
        return 0
    # Parse before touching the filesystem: a typo in the spec should fail on
    # the spot, not after a source run has been read and an env stood up.
    # Augmentation writes training data, so it draws from the 'train' half of
    # the texture pool; an eval with the same spec draws 'test'.
    try:
        spec = parse_randomize_spec(args.randomize, texture_split="train")
    except ValueError as exc:
        raise SystemExit(f"--randomize: {exc}") from None

    # Watching and producing are different intents: --viewer alone is "show me",
    # not "show me AND write another run I did not ask for".
    if args.write is None:
        args.write = not args.viewer

    reroll = parse_reroll(args.reroll)

    source = pathlib.Path(args.source_run).resolve()
    manifest = json.loads((source / "fixed_task_manifest.json").read_text())
    resets = load_reset_jitters(source)
    fps = json.loads((source / "meta" / "info.json").read_text())["fps"]
    parquets = sorted((source / "data" / "chunk-000").glob("episode_*.parquet"))
    episodes = (list(range(len(parquets))) if args.episodes == "all"
                else [int(x) for x in args.episodes.split(",")])
    # Augmented runs get their OWN lane rather than landing beside the source.
    # Mixing them into the collection lane makes `lab prepare <lane>/*` silently
    # merge originals and augmentations, and makes regenerating or deleting the
    # augmentations a per-directory job.
    out_base = (pathlib.Path(args.out_base) if args.out_base
                else source.parent.with_name(source.parent.name + "_replay"))

    if args.preroll_ticks is None:
        from g1_bridge.backends import get_backend_class

        space = get_backend_class(manifest["robot"]["backend"]).vla_action_space
        args.preroll_ticks = 150 if space in ("ref_qpos", "smpl") else 0

    print(f"[aug] source {source.name}: {len(episodes)} episode(s), "
          f"{args.variants} variant(s), scene {manifest['scene']['name']}, "
          f"preroll {args.preroll_ticks} tick(s)")

    total_dropped = 0
    for v in range(args.variants):
        variant_seed = args.seed + v
        stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        out_run = out_base / f"{manifest['env']}_aug_s{variant_seed}_{stamp}"
        # The appearance randomizer is installed ON THE ENV and re-runs at
        # every hard reset; one seed per variant, or per episode when the
        # scope says so.
        env = restore = handle = None
        appearance_report: dict = {}
        object_targets: list[tuple[pathlib.Path, pathlib.Path, int]] = []
        try:
            env, restore, name_map, ep_meta, handle = build_env(
                manifest, spec, args)
            handle.new_episode(variant_seed)
            appearance_report = handle.report

            def new_appearance(rng_seed: int) -> None:
                handle.new_episode(rng_seed)

            # Task-object re-texturing (opt-in): point each pinned object cfg
            # at a randomized sibling copy of its model.xml. create_obj loads
            # exactly the path in cfg["info"]["mjcf_path"] (its category
            # reverse-lookup uses the directory's model.xml), and only
            # <texture file>/<material rgba> change -- geometry, collision and
            # the reg_bbox sizing geom are byte-identical.
            if args.randomize_object_textures:
                obj_appearance = AppearanceConfig(
                    texture_frac=max(spec.textures or 0.0, 0.5),
                    texture_exclude=(),
                    light_jitter=0.0,
                    material_rgba_jitter=spec.rgba or 0.0,
                )
                for i, cfg in enumerate(ep_meta.get("object_cfgs") or []):
                    src = pathlib.Path((cfg.get("info") or {}).get("mjcf_path", ""))
                    if not src.is_file():
                        print(f"[aug] object cfg {cfg.get('name')!r}: mjcf_path "
                              f"missing, not re-textured")
                        continue
                    aug = src.with_name(src.stem + f".aug_s{variant_seed}.xml")
                    cfg["info"]["mjcf_path"] = str(aug)
                    object_targets.append((src, aug, i))

            def regen_objects(rng_seed: int) -> None:
                for src, aug, i in object_targets:
                    randomize_scene_xml(src, np.random.default_rng((rng_seed, i)),
                                        obj_appearance, out_xml=aug)

            if object_targets and reroll is None:
                regen_objects(variant_seed)

            # Distractors: take the no-distractor reference baseline first,
            # then append the extra cfgs (order preserves the task objects'
            # placement draws -- both initializers sample in cfg-list order).
            distractor_names: list[str] = []
            task_obj_names: list[str] = []
            ref_task_qpos: dict = {}
            if spec.distractors > 0 and ep_meta.get("object_cfgs"):
                env.reset()
                task_obj_names = [c["name"] for c in ep_meta["object_cfgs"]]
                ref_task_qpos = capture_baseline(env, task_obj_names)
                groups = [g for g in args.distractor_groups.split(",") if g]
                region_ids = [(c.get("placement") or {}).get("semantic_region_id")
                              for c in ep_meta["object_cfgs"]]
                region_ids = [r for r in region_ids if r]
                if not groups:
                    print("[aug] --distractors ignored: no groups")
                else:
                    # The distractor cfgs must use the SAME placement style as
                    # the task cfgs: injecting a semantic placement into a run
                    # whose task cfgs place via the plain surface sampler flips
                    # _build_placement_initializer into the semantic path for
                    # EVERY object -- a different code path, different draws,
                    # a deterministically shifted task placement. The initial
                    # spot barely matters anyway; teleport_distractors moves
                    # them after every reset.
                    cfgs = build_distractor_cfgs(
                        spec.distractors, groups,
                        np.random.default_rng(variant_seed + 777),
                        registries=tuple(getattr(env, "obj_registries",
                                                 ("objaverse", "lightwheel"))),
                        region_ids=region_ids or None)
                    ep_meta["object_cfgs"].extend(cfgs)
                    distractor_names.extend(c["name"] for c in cfgs)
                    print(f"[aug] distractors: {[c['obj_groups'] for c in cfgs]}")
            elif spec.distractors > 0:
                print("[aug] --distractors ignored: source run has no task "
                      "object cfgs")
            if args.write:
                init_run_skeleton(source, out_run)
            kept: list[dict] = []
            per_episode = {}
            frame_offset = 0
            for ep in episodes:
                frames = load_action_frames(parquets[ep],
                                            manifest["robot"]["backend"])
                appearance_seed = appearance_seed_for(reroll, variant_seed, ep)
                if reroll is not None:
                    new_appearance(appearance_seed)
                    if object_targets:
                        regen_objects(appearance_seed)
                if not reset_to_baseline(env, ref_task_qpos):
                    per_episode[str(ep)] = dict(
                        success=False, kept=False,
                        reason="baseline restore failed under distractors")
                    total_dropped += 1
                    print(f"[aug] s{variant_seed} ep{ep}: SKIPPED (baseline "
                          f"restore failed)", flush=True)
                    continue
                apply_recorded_jitter(env, resets.get(ep, {}), name_map)
                distractor_pos = (
                    teleport_distractors(
                        env, distractor_names, task_obj_names,
                        np.random.default_rng((variant_seed, ep, 555)),
                        radius=args.distractor_radius,
                        clearance=args.distractor_clearance,
                        corridor=args.distractor_corridor)
                    if distractor_names else None)
                cam = spec.camera or 0.0
                randomize_camera(env, [EGO_CAMERA], variant_seed * 1000 + ep,
                                 pos=0.01 * cam, rot=0.05 * cam, fovy=2.0 * cam)
                tint_robot_geoms(env, np.random.default_rng((variant_seed, ep, 99)),
                                 spec.robot or 0.0)
                lens = (LensWarp(240, 320, np.random.default_rng((variant_seed, ep, 77)),
                                 scale=spec.lens) if spec.lens else None)
                images, report = replay_episode(env, frames, args, image_fn=lens)
                if lens is not None:
                    report["lens"] = lens.describe()
                if len(images) != len(frames):
                    raise RuntimeError(
                        f"ep {ep}: captured {len(images)} frames for "
                        f"{len(frames)} actions")
                keep = report["success"] or args.keep_failed
                if keep:
                    new_idx = len(kept)
                    n = (write_episode(parquets[ep], out_run, new_idx,
                                       frame_offset, images, fps)
                         if args.write else report["n_frames"])
                    kept.append(dict(source_episode=ep, new_index=new_idx,
                                     n_frames=n, frame_offset=frame_offset))
                    frame_offset += n
                    report["new_index"] = new_idx
                else:
                    total_dropped += 1
                per_episode[str(ep)] = dict(
                    report, kept=keep, appearance_seed=appearance_seed,
                    **({"distractors": distractor_pos} if distractor_pos else {}),
                    # the RESOLVED appearance (which textures, final light
                    # params), so a run is auditable independently of the seed
                    **({"appearance": json.loads(json.dumps(appearance_report))}
                       if reroll is not None else {}))
                print(f"[aug] s{variant_seed} ep{ep}: "
                      f"{'SUCCESS @' + str(report['success_step']) if report['success'] else 'NO SUCCESS'}"
                      f"{'' if keep else '  -> DROPPED'}"
                      f"  ({report['n_frames']} frames, settle {report['settle_steps']})",
                      flush=True)
            if not args.write:
                ok = sum(1 for v in per_episode.values() if v.get("success"))
                print(f"[aug] s{variant_seed}: {ok}/{len(episodes)} episode(s) "
                      f"reached success; nothing written (--no-write)")
                continue
            if not kept:
                print(f"[aug] variant s{variant_seed}: nothing kept; removing "
                      f"{out_run}")
                shutil.rmtree(out_run)
                continue
            write_meta(source, out_run, kept)
            write_reset_jitters(source, out_run, kept)
            annotate_run(out_run, dict(
                source_run=str(source),
                variant_seed=variant_seed,
                randomize_spec=args.randomize,
                randomize_resolved=spec.describe(),
                appearance=dict(reroll=args.reroll,
                                profile=handle.profile.name,
                                texture_frac=handle.cfg.texture_frac,
                                texture_exclude=list(handle.cfg.texture_exclude),
                                texture_split=handle.cfg.texture_split,
                                light_jitter=handle.cfg.light_jitter,
                                light_geom_jitter=handle.cfg.light_geom_jitter,
                                extra_lights=handle.cfg.extra_lights,
                                material_rgba_jitter=handle.cfg.material_rgba_jitter,
                                material_pbr_jitter=handle.cfg.material_pbr_jitter,
                                light_intensity_jitter=handle.cfg.light_intensity_jitter,
                                light_shadow_toggle=handle.cfg.light_shadow_toggle,
                                robot_tint=spec.robot or 0.0,
                                lens=spec.lens or 0.0,
                                # variant-scope: one resolved appearance for the run
                                **({"resolved": json.loads(json.dumps(appearance_report))}
                                   if reroll is None else {})),
                camera=dict(scale=spec.camera or 0.0),
                preroll_ticks=args.preroll_ticks,
                randomize_object_textures=bool(object_targets),
                distractors=dict(count=len(distractor_names),
                                 groups=args.distractor_groups,
                                 radius=args.distractor_radius,
                                 clearance=args.distractor_clearance,
                                 corridor=args.distractor_corridor)
                if distractor_names else None,
                depth_not_regenerated=True,
                episodes_stats_note="image stats inherited from the source run; "
                                    "approximate for the re-rendered videos",
                per_episode=per_episode,
            ))
            print(f"[aug] variant s{variant_seed}: kept {len(kept)}/"
                  f"{len(episodes)} -> {out_run}")
        finally:
            if restore is not None:
                restore()
            if env is not None:
                env.close()
            for _src, aug, _i in object_targets:
                cleanup_variant_xml(aug)
    if total_dropped:
        print(f"[aug] {total_dropped} episode replay(s) diverged (no success) "
              f"and were dropped from the output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
