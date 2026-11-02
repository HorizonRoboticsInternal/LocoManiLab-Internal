"""h6: does the fine-tuned Kimodo still help once the tracker is the PORTED one?

Everything upstream of the simulator is imported from ``heft_work`` -- the same
goal sampler, the same prompts, the same actor, the same 6-step denoising chain
the weights were trained with. The ONLY thing swapped is the executor:

    mjlab + HEFT tracker      (training, exec_err median 0.480 -> 0.074)
    robosuite + ported HEFT   (here)

That makes the comparison an experiment rather than a re-implementation. If the
gap between base and fine-tuned survives the port, every downstream scene number
rests on something verified; if it does not, P2-P4 would have been built on an
assumption -- which is why this runs before them and not after.

Two numbers are reported, and the pair is the diagnostic:

    ref_err   the GENERATED reference's own wrist vs the goal (MuJoCo FK, no
              physics). Kimodo's aim. Should match heft_work closely, since the
              generation path is literally the same code -- if it does not, the
              difference is in how this script drives it, not in the port.
    exec_err  the ROBOT's wrist vs the goal, min over the 3 s rollout. What the
              task actually cares about, and where the port can show up.

``exec_err - ref_err`` is the execution gap: the thing the RL fine-tuning was
optimising away, and the thing the 1.9x standing residual could eat into.

    conda activate kimodo_rl
    python -m rl.kimodo.heft.scripts.h6_kimodo_empty --n 32 --tag base
    python -m rl.kimodo.heft.scripts.h6_kimodo_empty --n 32 --tag lora \\
        --ckpt rl/kimodo/_out/heft/ckpt/old_lr1e-5_nohands_kimodo_best.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

#: Kimodo's own reach envelope is measured against this nominal standing pelvis;
#: goals are sampled around it, so the robot must spawn at (approximately) the
#: same height for the goals to mean the same thing. Asserted at run time.
PELVIS_Z = 0.793


def resolve_actor_resources(heft_work: str | Path | None = None,
                            text_cache: str | Path | None = None) -> tuple[Path, Path]:
    """Resolve the external checkout and its existing text cache without loading models."""
    selected = heft_work or os.environ.get("HEFT_WORK")
    if not selected:
        raise ValueError("Set HEFT_WORK or pass --heft-work to the checkout containing kimodo_rl.")
    work = Path(selected).expanduser().resolve()
    if not (work / "kimodo_rl").is_dir():
        raise ValueError(f"No kimodo_rl package under {work}; select the correct HEFT_WORK checkout.")
    selected_cache = text_cache or os.environ.get("KIMODO_TEXT_CACHE") or work / "assets/text_cache.pt"
    cache = Path(selected_cache).expanduser().resolve()
    if not cache.is_file():
        raise ValueError(f"Kimodo text cache not found: {cache}. Set KIMODO_TEXT_CACHE "
                         "or pass --text-cache to the matching existing cache.")
    return work, cache


def build_actor(ckpt: str | None, steps: int, device: str = "cuda", *,
                heft_work: str | Path | None = None,
                text_cache: str | Path | None = None):
    """heft_work's actor, its generator, and its text cache.

    ``finetune_mode="full"`` skips LoRA injection, so the parameter names stay
    the base model's -- which is what ``save_native`` folded the adapters back
    into. Injecting LoRA first and loading afterwards is the documented trap:
    the folded names match nothing, ``strict=False`` reports success, and
    "after training" silently equals "before training".
    """
    try:
        work, cache_path = resolve_actor_resources(heft_work, text_cache)
    except ValueError as exc:
        raise SystemExit(f"[kimodo] {exc}") from exc
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))

    import torch

    from kimodo_rl.actor_kimodo import KimodoActor, KimodoActorConfig
    from kimodo_rl.gen import KimodoG1Generator
    from kimodo_rl.text_cache import TextCache

    cache = TextCache.load(str(cache_path))
    gen = KimodoG1Generator(load_text_encoder=False, device=device)
    actor = KimodoActor(gen, KimodoActorConfig(num_denoising_steps=steps,
                                               finetune_mode="full"),
                        text_cache=cache)
    if ckpt:
        sd = torch.load(ckpt, map_location="cpu")
        missing, unexpected = actor.model.denoiser.load_state_dict(sd["denoiser"],
                                                                  strict=False)
        print(f"[h6] loaded {os.path.basename(ckpt)} (mode={sd.get('finetune_mode')}) "
              f"missing={len(missing)} unexpected={len(unexpected)}")
        if missing or unexpected:
            # The guard that once reported missing=304: a silent no-op load.
            raise SystemExit(
                f"[h6] refusing to evaluate a partially loaded checkpoint: "
                f"{len(missing)} missing, {len(unexpected)} unexpected keys")
    else:
        print("[h6] base weights (no checkpoint)")
    return actor


def reference_wrist(qpos_30hz: np.ndarray, bridge) -> np.ndarray:
    """[T, 3] the REFERENCE's own right wrist in world, via FK. No physics."""
    r = bridge.from_qpos36(np.asarray(qpos_30hz, dtype=np.float32),
                           src_fps=30.0, resample_to=50.0)
    wi = list(r["body_names"]).index("right_wrist_yaw_link")
    return r["local_body_pos"][:, wi] + r["root_pos"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="save_native checkpoint; omit = base")
    ap.add_argument("--heft-work", help="checkout containing kimodo_rl (default: HEFT_WORK)")
    ap.add_argument("--text-cache", help="existing cache (default: KIMODO_TEXT_CACHE or HEFT_WORK/assets/text_cache.pt)")
    ap.add_argument("--tag", default="h6")
    ap.add_argument("--n", type=int, default=32, help="goals (one chunk each)")
    ap.add_argument("--seconds", type=float, default=3.0, help="chunk length; training used 3.0")
    ap.add_argument("--steps", type=int, default=6, help="denoising steps; training used 6")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--env", default="EmptyEnv")
    ap.add_argument("--batch", type=int, default=16, help="goals generated per GPU batch")
    ap.add_argument("--out", default="rl/kimodo/_out/heft/h6")
    args = ap.parse_args()

    try:
        work, cache_path = resolve_actor_resources(args.heft_work, args.text_cache)
    except ValueError as exc:
        ap.error(str(exc))
    if str(work) not in sys.path:
        sys.path.insert(0, str(work))
    os.makedirs(args.out, exist_ok=True)

    import torch

    from kimodo_rl.eef_goals import EefGoalSampler
    from kimodo_rl.prompts import prompt_for

    from rl.kimodo.heft.scene_track import KimodoHeftTracker
    from rl.kimodo.heft.tracker_const import DEFAULT_JOINT_POS
    from rl.kimodo.heft_bridge import HeftReference

    n_frames = int(round(args.seconds * 30))     # Kimodo's 30 Hz
    n_steps = int(round(args.seconds * 50))      # the tracker's 50 Hz
    print(f"[h6] tag={args.tag} n={args.n} chunk={args.seconds}s "
          f"({n_frames} gen frames -> {n_steps} ticks) denoising={args.steps}")

    # ---------------------------------------------------------------- scene
    tr = KimodoHeftTracker(env_name=args.env, seed=args.seed)
    root0_local = tr.reset()[:3].astype(np.float32)
    if abs(float(root0_local[2]) - PELVIS_Z) > 0.05:
        raise SystemExit(
            f"[h6] spawn pelvis is {root0_local[2]:.3f} m, not ~{PELVIS_Z}. The goal "
            "sampler places targets around the nominal standing height; a spawn "
            "that far off would shift the whole difficulty distribution.")

    # ---------------------------------------------------------------- goals
    # Sampled around the robot's ACTUAL spawn in the local frame, using the
    # training-side sampler so the difficulty distribution is the same one the
    # reported 0.074 was measured on.
    sampler = EefGoalSampler(seed=args.seed)
    root0 = np.tile(root0_local[None], (args.n, 1))
    goals = sampler.sample(args.n, root_pos=root0)                     # local frame
    local = EefGoalSampler.to_pelvis_frame(goals, root0)
    prompts = prompt_for(local)
    root_goals = list(EefGoalSampler.root_target(goals, root0))
    d_xy = EefGoalSampler.horizontal_distance(goals, root0)

    stand = np.concatenate([[0.0, 0.0, PELVIS_Z, 1.0, 0.0, 0.0, 0.0],
                            DEFAULT_JOINT_POS]).astype(np.float32)
    # The prefix training used: ONE standing frame tiled 5x, so its velocity is
    # identically zero. Kept deliberately -- see scene_track.prefix_frames.
    prefix = np.tile(stand, (5, 1))

    actor = build_actor(args.ckpt, args.steps, heft_work=work, text_cache=cache_path)
    bridge = HeftReference(["pelvis", "right_wrist_yaw_link"])

    rows, t0 = [], time.time()
    for lo in range(0, args.n, args.batch):
        hi = min(lo + args.batch, args.n)
        torch.manual_seed(args.seed + lo)          # same seeds across policies
        np.random.seed(args.seed + lo)
        with torch.no_grad():
            samples = actor.sample_actions_batch(
                prompts[lo:hi], n_frames, [prefix] * (hi - lo),
                [("RightHand", g) for g in goals[lo:hi]],
                root_goals=root_goals[lo:hi])

        for j, s in enumerate(samples):
            i = lo + j
            q = np.asarray(s["qpos_30hz"], dtype=np.float32)
            goal_l = goals[i].astype(np.float64)
            goal_w = tr.to_world(goal_l)

            # A drifted policy emits non-finite or absurd references. MuJoCo does
            # not raise on those; it poisons the state and every later number is
            # NaN. Score them as the failures they are instead of crashing.
            unusable = bool(not np.isfinite(q).all() or np.abs(q[:, 0:3]).max() > 50.0)
            if unusable:
                rows.append({"i": i, "d_xy": float(d_xy[i]), "unusable": True,
                             "fell": True, "ref_err": float("nan"),
                             "exec_err": float("nan"), "final_err": float("nan"),
                             "prompt": prompts[i]})
                continue

            ref_w = reference_wrist(q, bridge)
            ref_err = float(np.linalg.norm(ref_w - goal_l[None], axis=1).min())

            tr.reset()
            tr.load_chunk(q)
            res = tr.rollout(n_steps, track="right", stop_on_success=False)
            wrist_l = res["wrist_world"] - tr.to_world([0.0, 0.0, 0.0])[None]
            dists = np.linalg.norm(wrist_l - goal_l[None], axis=1)
            rows.append({
                "i": i, "d_xy": float(d_xy[i]), "unusable": False,
                "fell": bool(res["fell"]),
                "ref_err": ref_err,
                "exec_err": float(dists.min()),      # use_min_distance=True, as in training
                "final_err": float(dists[-1]),
                "prompt": prompts[i],
                "goal_world": [round(float(v), 4) for v in goal_w],
            })
            r = rows[-1]
            print(f"[h6] {i:3d}  d_xy {r['d_xy']:.2f}  ref {r['ref_err']:.3f}  "
                  f"exec {r['exec_err']:.3f}  gap {r['exec_err'] - r['ref_err']:+.3f}"
                  f"{'  FELL' if r['fell'] else ''}")

    tr.close()
    ok = [r for r in rows if not r["unusable"]]
    if not ok:
        raise SystemExit("[h6] every sample was unusable")
    ee = np.array([r["exec_err"] for r in ok])
    re_ = np.array([r["ref_err"] for r in ok])
    summary = {
        "tag": args.tag, "ckpt": args.ckpt, "n": len(rows), "usable": len(ok),
        "seconds": args.seconds, "denoising_steps": args.steps, "seed": args.seed,
        "env": args.env,
        "exec_err_median": round(float(np.median(ee)), 4),
        "exec_err_mean": round(float(ee.mean()), 4),
        "ref_err_median": round(float(np.median(re_)), 4),
        "gap_median": round(float(np.median(ee - re_)), 4),
        "fall_rate": round(float(np.mean([r["fell"] for r in rows])), 4),
        "secs": round(time.time() - t0, 1),
    }
    print("\n=== h6 summary ===")
    for k, v in summary.items():
        print(f"  {k:18s} {v}")
    print(_band_report(np.array([r["d_xy"] for r in ok]), ee, re_))

    path = os.path.join(args.out, f"{args.tag}.json")
    with open(path, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    print(f"[h6] saved {path}")
    return 0


def _band_report(d_xy, ee, re_, bands=((0.0, 0.6), (0.6, 1.0), (1.0, 1.6), (1.6, 9.0))):
    """Per-distance-band medians. A single number hides the only structure that
    matters: the execution gap grows with commanded travel, and an average over
    a distribution with a different mix is not comparable to heft_work's."""
    out = [f"{'d_xy band':>12}{'n':>5}{'ref_err':>10}{'exec_err':>10}{'gap':>8}"]
    for lo, hi in bands:
        m = (d_xy >= lo) & (d_xy < hi)
        if not m.any():
            continue
        out.append(f"{f'{lo:.1f}-{hi:.1f}':>12}{int(m.sum()):>5}"
                   f"{np.median(re_[m]):>10.3f}{np.median(ee[m]):>10.3f}"
                   f"{np.median(ee[m] - re_[m]):>+8.3f}")
    return "\n".join(out)


if __name__ == "__main__":
    raise SystemExit(main())
