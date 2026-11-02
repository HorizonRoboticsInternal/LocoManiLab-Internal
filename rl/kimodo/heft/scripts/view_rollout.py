"""Look at a recorded P2 rollout: interactive window, or mp4.

This script imports NO torch. That is the whole point of it existing separately
from ``h7_scene_rollout``: rendering and CUDA must not share a process
(DEBUG_LOG #11), so the rollout records states and this replays them.

    conda activate kimodo_rl
    S=rl/kimodo/_out/heft/h7/states/TabletopPickPlace_ep0.npz

    python -m rl.kimodo.heft.scripts.view_rollout --states $S              # window
    python -m rl.kimodo.heft.scripts.view_rollout --states $S --video a.mp4
    python -m rl.kimodo.heft.scripts.view_rollout --states $S --loop       # repeat

The interactive window is not a worse video -- it is the thing a video cannot be.
Orbit while paused, look under the table, check whether the palm is on the far
side of the mug or short of it. Every geometry bug found in this pipeline so far
looked fine from the one angle a fixed camera happened to use.

⚠️ The scene is rebuilt from ``env_name`` + ``seed`` stored in the npz, with a
SOFT reset. If the fingerprint does not match, this raises instead of drawing:
a hard reset resamples the objects, and replaying old qpos into a resampled
scene produces a sharp, plausible video of an episode that never happened.
"""

from __future__ import annotations

import argparse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--states", required=True, help="npz written by --save-states")
    ap.add_argument("--video", default=None, help="write an mp4 instead of opening a window")
    ap.add_argument("--env", default=None, help="override the scene name in the npz")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--model", default=None,
                    help="compiled .mjb written alongside the states. Use it for "
                         "any scene that resamples its objects on reset -- "
                         "rebuilding those from env+seed produces a DIFFERENT "
                         "room and the fingerprint check refuses to draw")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--stride", type=int, default=1, help="render every Nth state")
    ap.add_argument("--loop", action="store_true", help="repeat in the window")
    ap.add_argument("--size", type=int, nargs=2, default=[480, 640])
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-12.0)
    ap.add_argument("--distance", type=float, default=3.0)
    ap.add_argument("--cutaway", action="store_true",
                    help="hide the STATIC geometry (walls, floor) so a robot "
                         "wedged in a narrow room is visible at all. Costs the "
                         "one question these videos exist to answer -- is it "
                         "walking through the furniture -- so it is off by "
                         "default and belongs beside a normal render, not "
                         "instead of one")
    ap.add_argument("--ego", action="store_true",
                    help="render the robot's HEAD camera instead of a "
                         "third-person view -- what the VLA sees. Needs --model "
                         "and --video; the size defaults to the collection "
                         "resolution rather than this script's 480x640, because "
                         "an ego clip is meant to be comparable with the dataset")
    args = ap.parse_args()

    from rl.kimodo.heft.view import render_saved, replay_saved_live

    if args.ego:
        # ⚠️ Not a camera option on render_saved: that one drives a FREE camera
        # (azimuth/elevation/distance, plus the obstacle search that keeps it out
        # of walls), and none of those words mean anything for a camera bolted to
        # the head. Sharing the entry point would have meant three arguments that
        # are silently ignored in one of the two modes.
        import numpy as np

        from rl.kimodo.heft.export import render_ego
        from rl.kimodo.heft.render import write_mp4

        if not (args.model and args.video):
            raise SystemExit("[view] --ego needs both --model and --video")
        z = np.load(args.states, allow_pickle=True)
        size = tuple(args.size) if args.size != [480, 640] else None
        if size is None:
            from rl.kimodo.heft.export import _ensure_g1_bridge

            _ensure_g1_bridge()
            from g1_bridge.recording.features import EGO_H, EGO_W

            size = (EGO_H, EGO_W)
        frames = render_ego(args.model, z["qpos"][::args.stride], size)
        print(f"[view] wrote {write_mp4(frames, args.video, fps=args.fps)}")
        return 0

    if args.video:
        out = render_saved(args.states, args.video, env_name=args.env,
                           seed=args.seed, model_path=args.model,
                           size=tuple(args.size), fps=args.fps,
                           stride=args.stride, azimuth=args.azimuth,
                           elevation=args.elevation, distance=args.distance,
                           cutaway=args.cutaway)
        print(f"[view] wrote {out}")
    else:
        replay_saved_live(args.states, env_name=args.env, seed=args.seed,
                          fps=args.fps, loop=args.loop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
