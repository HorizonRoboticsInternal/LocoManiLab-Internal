"""Look at what ``--randomize`` actually does to a scene, without collecting.

Renders the SAME scene under a baseline plus N randomization seeds and tiles
the frames into one contact sheet, so the effect of a spec can be judged by
eye before it is used to augment data (``replay_augment``) or to perturb an
eval (``lab eval --randomize``). ``--viewer`` instead opens the interactive
window with one seed applied, to walk around the randomized room.

The randomization is the benchmark's own ``scene_randomization`` /
``appearance`` code driven by the shared spec parser, so what this shows is
what those two entry points do -- this adds no randomization of its own. It
works for every scene family the appearance profiles cover (MolmoSpaces rooms
and robocasa kitchens alike).

Usage (conda activate tele_benchmark; MUJOCO_GL=egl for the contact sheet):

    PYTHONPATH=$PWD MUJOCO_GL=egl python -m robocasa.scripts.preview_randomization \
        --env IThorKitchenOpenCabinet --scene-id 24 \
        --randomize textures,lights,camera,pbr --seeds 0,1,2,3 \
        --out /tmp/preview.png

    # walk around one randomized variant instead (needs a display)
    PYTHONPATH=$PWD python -m robocasa.scripts.preview_randomization \
        --env IThorKitchenOpenCabinet --scene-id 24 \
        --randomize all --seeds 3 --viewer
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
from termcolor import colored

from robocasa.utils.appearance import (
    UnsupportedScene,
    install_appearance,
    profile_for,
)
from robocasa.utils.g1_data_collection import (
    EGO_CAMERA,
    build_teleop_env,
    forward_scene_kwargs,
)
from robocasa.utils.scene_randomization import (
    LensWarp,
    capture_baseline,
    parse_randomize_spec,
    randomize_help,
    install_distractors,
    randomize_camera,
    teleport_distractors,
    tint_robot_geoms,
)

DISTRACTOR_GROUPS = ["cup", "mug", "banana", "can", "bowl", "sponge",
                     "orange", "lemon"]


def ego_frame(env, lens=None):
    """This episode's ego frame, as the dataset/policy would see it."""
    base = getattr(env, "unwrapped", env)
    try:
        obs = base._get_observations(force_update=True)
    except TypeError:
        obs = base._get_observations()
    img = np.asarray(obs[f"{EGO_CAMERA}_image"])[::-1]  # robosuite flips vertically
    return lens(img) if lens is not None else img


def contact_sheet(frames: list[tuple[str, np.ndarray]], out: pathlib.Path,
                  cols: int = 0) -> pathlib.Path:
    """Tile labelled frames into one PNG."""
    from PIL import Image, ImageDraw

    try:
        from PIL import ImageFont

        font = ImageFont.load_default(size=16)
    except Exception:
        font = None
    n = len(frames)
    cols = cols or min(n, 3)
    rows = (n + cols - 1) // cols
    h, w = frames[0][1].shape[:2]
    pad, bar = 6, 24
    sheet = Image.new("RGB", (cols * (w + pad) + pad,
                              rows * (h + bar + pad) + pad), (24, 24, 27))
    draw = ImageDraw.Draw(sheet)
    for i, (label, img) in enumerate(frames):
        r, c = divmod(i, cols)
        x, y = pad + c * (w + pad), pad + r * (h + bar + pad)
        sheet.paste(Image.fromarray(np.ascontiguousarray(img)), (x, y + bar))
        draw.text((x + 2, y + 4), label, fill=(230, 230, 235), font=font)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


def main():
    p = argparse.ArgumentParser(
        description="preview what --randomize does to a scene")
    p.add_argument("--env", required=True)
    p.add_argument("--scene-id", type=int, default=None,
                   help="pin the room (MolmoSpaces envs). Kitchens pin with "
                        "--layout/--style instead")
    p.add_argument("--robot", default="G1FloatingBody")
    p.add_argument("--backend", default="sonic",
                   help="only decides which controller config builds the env; "
                        "the policy never runs here")
    p.add_argument("--randomize", default="standard",
                   metavar="SPEC",
                   help="the same spec `lab eval --randomize` and "
                        "replay_augment take: presets off/mild/standard/stress "
                        "or comma-separated tokens, later tokens overriding "
                        "earlier ones. Pass 'help' for the full reference.")
    p.add_argument("--seeds", default="0,1,2,3",
                   help="comma list of --randomize-seed values to render")
    p.add_argument("--no-baseline", action="store_true",
                   help="skip the un-randomized reference frame")
    p.add_argument("--scene-seed", type=int, default=0)
    p.add_argument("--layout", type=int, default=None)
    p.add_argument("--style", type=int, default=None)
    p.add_argument("--out", default="randomize_preview.png")
    p.add_argument("--cols", type=int, default=0, help="contact sheet columns")
    p.add_argument("--viewer", action="store_true",
                   help="open the interactive window on the FIRST seed "
                        "instead of writing a contact sheet (needs a display)")
    args = p.parse_args()

    if args.randomize.strip() in ("help", "?", "list"):
        print(randomize_help())
        return
    try:
        # Same default as augmentation, so a preview shows what augmented
        # training data will look like; add texture-split=test to the spec to
        # see what an eval draws instead.
        spec = parse_randomize_spec(args.randomize, texture_split="train")
    except ValueError as exc:
        p.error(f"--randomize: {exc}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    if not seeds:
        p.error("--seeds needs at least one value")


    # Pin the scene so every tile differs ONLY by the randomization: without
    # this the env redraws its own layout/placements each reset and the
    # comparison confounds the two.
    scene_kwargs = forward_scene_kwargs(
        args.env, scene_id=args.scene_id, layout=args.layout, style=args.style,
        fixed_scene=True, scene_seed=args.scene_seed)

    from robocasa.g1_host.bridge import load_sim2real_controller_config

    controller_config = load_sim2real_controller_config(args.backend, "mock")
    env = build_teleop_env(args.env, args.robot, controller_config, scene_kwargs,
                           headless=not args.viewer)
    base = getattr(env, "unwrapped", env)
    restores = []
    handle = None
    if spec.edits_appearance:
        try:
            profile = profile_for(base)
        except UnsupportedScene as exc:
            p.error(f"--randomize: {exc}")
        handle = install_appearance(base, spec.appearance_config(profile),
                                    profile)
        restores.append(handle.restore)
        print(colored(spec.format_resolved(args.randomize, profile), "cyan"),
              flush=True)
        if spec.textures and spec.texture_split != "test":
            print(colored(
                f"[preview] texture pool split is {spec.texture_split!r} (what "
                f"augmentation writes). `lab eval --randomize` defaults to "
                f"'test', so an eval draws DIFFERENT images than these -- add "
                f"texture-split=test to the spec to preview what eval sees.",
                "yellow"), flush=True)

    try:
        distractor_names: list[str] = []
        task_names: list[str] = []
        if spec.distractors:
            env.reset()
            task_names = list(getattr(base, "objects", {}) or {})
            if not task_names:
                # articulation / approach tasks own no movable objects, and a
                # distractor is placed in an annulus AROUND one -- there is
                # nothing to anchor to.
                print(colored(
                    f"[preview] {args.env} has no movable task objects to "
                    f"anchor distractors on -- distractors skipped", "yellow"),
                    flush=True)
                spec.distractors = 0
            else:
                distractor_names, restore_objs, placed = install_distractors(
                    base, args.env, spec.distractors, DISTRACTOR_GROUPS,
                    np.random.default_rng(seeds[0] + 777),
                    registries=tuple(getattr(base, "obj_registries",
                                             ("objaverse", "lightwheel"))))
                spec.distractors = placed
                if restore_objs is not None:
                    restores.append(restore_objs)
                print(colored(f"[preview] distractors: {distractor_names}",
                              "cyan"), flush=True)
                # No physics runs here (one frame per seed, no stepping), so a
                # teleported distractor is rendered exactly where it was put
                # rather than where it would come to rest. In eval/replay the
                # episode steps on and it settles onto the support.
                print(colored("[preview] note: distractors are drawn at their "
                              "teleport pose -- this preview steps no physics, "
                              "so they can look airborne; in a real episode "
                              "they settle onto the surface", "yellow"),
                      flush=True)

        def apply(seed: int | None):
            """Render one variant; None = the un-randomized baseline."""
            if handle is not None:
                if seed is None:
                    handle.disable()
                else:
                    handle.new_episode(seed * 10000)
            env.reset()
            if handle is not None and seed is not None and handle.report:
                tex = handle.report.get("textures") or {}
                lights = handle.report.get("lights") or []
                print(f"[preview] seed {seed}: "
                      f"{tex.get('groups_swapped', 0)}/{tex.get('groups', 0)} "
                      f"surface groups swapped "
                      f"({len(tex.get('swapped') or {})} texture entries, "
                      f"pool split={tex.get('split')}), "
                      f"{len(lights)} lights", flush=True)
            lens = None
            if seed is not None:
                if spec.distractors and distractor_names:
                    teleport_distractors(base, distractor_names, task_names,
                                         np.random.default_rng((seed, 0, 555)))
                if spec.camera:
                    randomize_camera(base, [EGO_CAMERA], seed * 1000,
                                     pos=0.01 * spec.camera,
                                     rot=0.05 * spec.camera,
                                     fovy=2.0 * spec.camera)
                if spec.robot:
                    tint_robot_geoms(base, np.random.default_rng((seed, 0, 99)),
                                     spec.robot)
                if spec.lens:
                    h, w = base.camera_heights[0], base.camera_widths[0]
                    lens = LensWarp(h, w, np.random.default_rng((seed, 0, 77)),
                                    scale=spec.lens)
            return lens

        if args.viewer:
            apply(seeds[0])
            print(colored(
                f"[preview] seed {seeds[0]} applied — drive the G1 with the "
                f"keyboard, orbit with the mouse, close the window to quit.",
                "cyan"), flush=True)
            from robocasa.demos.demo_empty_teleop import run_sonic_viewer_loop

            run_sonic_viewer_loop(env, initial_reset=False)
            return

        frames = []
        if not args.no_baseline:
            frames.append(("baseline", ego_frame(env, apply(None))))
        for seed in seeds:
            frames.append((f"seed {seed}", ego_frame(env, apply(seed))))
        out = contact_sheet(frames, pathlib.Path(args.out).expanduser(),
                            cols=args.cols)
        print(colored(f"[preview] wrote {out}  ({len(frames)} frames, "
                      f"{frames[0][1].shape[1]}x{frames[0][1].shape[0]} each)",
                      "green"), flush=True)
    finally:
        try:
            env.close()
        except Exception:
            pass
        for restore in restores:
            try:
                restore()
            except Exception:
                pass


if __name__ == "__main__":
    main()
