"""Generic G1 teleop data collection — works for ANY registered task env
(kitchen / living-room / MolmoSpaces iTHOR rooms / future scene types).

Scene selection is forwarded to whatever the chosen env accepts (see
``g1_data_collection.forward_scene_kwargs``):
  * MolmoSpaces room tasks  ->  --scene-id <FloorPlan>
  * kitchen / living-room   ->  --layout / --style
  * any env                 ->  --fixed-scene --scene-seed N

Every --backend (sonic/amo/heft/scalebfm/almi) drives G1BridgeComposite
("G1_BRIDGE", robocasa/g1_host/bridge.py) -- sim2real is a required
dependency for collection. See add_collection_args (robocasa.utils.
g1_data_collection) for the full flag list, incl. --policy-config/
--gmr-viewer/--force-engage.

Usage (conda activate tele_benchmark; export MLSPACES_ASSETS_DIR=...):

    # interactive operator window (needs a display; use MUJOCO_GL=glfw)
    python -m robocasa.scripts.collect_demos_g1 \
        --env IThorKitchenApproachFridge --scene-id 1 --backend sonic \
        --pico-source real --fixed-scene --scene-seed 0

    # headless plumbing check (no window, no PICO)
    python -m robocasa.scripts.collect_demos_g1 \
        --env IThorKitchenApproachFridge --fixed-scene --scene-seed 0 \
        --pico-source mock --debug-record-steps 5 --no-viewer

    # amo / heft / scalebfm -- same flags, different --backend
    python -m robocasa.scripts.collect_demos_g1 \
        --env IThorKitchenApproachFridge --scene-id 1 --backend amo \
        --pico-source real --fixed-scene --scene-seed 0
"""

import argparse

from robocasa.utils.g1_data_collection import add_collection_args, collect_demos_main


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        type=str,
        required=True,
        help=(
            "environment name (task envs auto-save on success; EmptyEnv uses S "
            "for a manual episode boundary)"
        ),
    )
    add_collection_args(parser)
    args = parser.parse_args()
    collect_demos_main(args)


if __name__ == "__main__":
    main()
