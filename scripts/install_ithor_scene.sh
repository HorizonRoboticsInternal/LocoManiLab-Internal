#!/usr/bin/env bash
#
# install_ithor_scene.sh — download an iTHOR/MolmoSpaces FloorPlan (+ the shared
# objects/thor library, once) and symlink it into THIS repo's own asset tree.
#
# WHY THIS EXISTS
#   iTHOR scene meshes and the objects/thor library come from a completely
#   separate repo, github.com/allenai/molmospaces, with its own venv and its
#   own download cache -- NOT covered by `download_kitchen_assets`. RoboCasa
#   reads assets from ITS OWN `robocasa/models/assets/{scenes/ithor,objects/thor}`
#   tree, which molmospaces does not populate directly: bridging the two
#   requires symlinks (this used to be done by hand). See docs/molmospaces_scenes.md
#   for the full background and the one-time molmospaces venv setup this
#   script assumes is already done.
#
#   Known gotcha this script works around: `get_scenes("ithor", "train")`
#   returns train/val/test dicts keyed by the SAME integer sequence, but a
#   given FloorPlan number can live in any one of the three splits (e.g.
#   FloorPlan24 is under "val", not "train") -- this script checks all three.
#
# ONE-TIME PREREQUISITE (not done by this script):
#   git clone https://github.com/allenai/molmospaces.git ~/molmospaces
#   cd ~/molmospaces && uv venv --python 3.11 .venv
#   source .venv/bin/activate && uv pip install -e ".[mujoco]"
#   (if `torch` times out downloading, retry with `UV_HTTP_TIMEOUT=300`)
#
# Usage:
#   bash scripts/install_ithor_scene.sh 24 401 ...
#
# Env vars (all optional, sensible defaults):
#   MOLMOSPACES_REPO     molmospaces checkout             (default ~/molmospaces)
#   MLSPACES_ASSETS_DIR  molmospaces symlink resource dir  (default $MOLMOSPACES_REPO/sample_resources)
#   MLSPACES_CACHE_DIR   molmospaces download cache        (default ~/.cache/molmo-spaces-resources)
#
# After this script succeeds, any shell running `lab collect --env IThor* ...`
# on this machine ALSO needs `export MLSPACES_ASSETS_DIR=...` -- lab just
# shells out to this repo's scripts, and env vars propagate to that subprocess.
# Idempotent: rerunning for an already-installed FloorPlan is a cheap no-op.
set -euo pipefail

usage() {
  cat >&2 <<'USAGE'
Usage: scripts/install_ithor_scene.sh <FloorPlan-number> [<FloorPlan-number> ...]

  e.g.  scripts/install_ithor_scene.sh 24 401

  FloorPlan numbers encode the room type:
    1-30 kitchen | 201-230 living room | 301-330 bedroom | 401-430 bathroom
  `--env` and `--scene-id` must agree on room type when you later collect.

  Needs the molmospaces checkout + venv first (one-time, see header comment).
  Re-running for an already-installed FloorPlan is a cheap no-op.
USAGE
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
  usage
  [[ $# -lt 1 ]] && exit 1 || exit 0
fi

for arg in "$@"; do
  if ! [[ "$arg" =~ ^[0-9]+$ ]]; then
    echo "[ithor] not a FloorPlan number: '$arg'" >&2
    echo >&2
    usage
    exit 1
  fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MOLMOSPACES_REPO="${MOLMOSPACES_REPO:-$HOME/molmospaces}"
export MLSPACES_ASSETS_DIR="${MLSPACES_ASSETS_DIR:-$MOLMOSPACES_REPO/sample_resources}"
export MLSPACES_CACHE_DIR="${MLSPACES_CACHE_DIR:-$HOME/.cache/molmo-spaces-resources}"

if [[ ! -x "$MOLMOSPACES_REPO/.venv/bin/python" ]]; then
  echo "[ithor] molmospaces venv not found at $MOLMOSPACES_REPO/.venv -- set up per the header comment above / docs/molmospaces_scenes.md first." >&2
  exit 1
fi

mkdir -p "$MLSPACES_ASSETS_DIR" "$MLSPACES_CACHE_DIR"

echo "[ithor] downloading FloorPlan(s) $* via molmospaces (first run also pulls the shared objects/thor library, ~13GB)..."
"$MOLMOSPACES_REPO/.venv/bin/python" - "$@" <<'PYEOF'
import sys

from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path

scenes = get_scenes("ithor", "train")
for fp_arg in sys.argv[1:]:
    fp = int(fp_arg)
    path = None
    for split in ("train", "val", "test"):
        cand = scenes[split].get(fp)
        if cand is not None:
            path = cand
            print(f"[ithor] FloorPlan{fp}: found in split={split}")
            break
    if path is None:
        print(f"[ithor] FloorPlan{fp}: NOT FOUND in any split -- skipping", file=sys.stderr)
        continue
    install_scene_with_objects_and_grasps_from_path(path, exclude_thor=False)
PYEOF

echo "[ithor] linking into robocasa's own asset tree..."
ln -sfn "$MLSPACES_ASSETS_DIR/objects/thor" "$REPO_ROOT/robocasa/models/assets/objects/thor"

for fp in "$@"; do
  scene="FloorPlan${fp}_physics"
  src_dir="$MLSPACES_ASSETS_DIR/scenes/ithor/$scene"
  if [[ ! -d "$src_dir" ]]; then
    echo "[ithor] $scene: no asset dir at $src_dir (download above may have skipped it) -- not linking" >&2
    continue
  fi
  ln -sfn "$src_dir" "$REPO_ROOT/robocasa/models/assets/scenes/ithor/$scene"
  for suffix in _metadata.json _map.png; do
    src="$MLSPACES_ASSETS_DIR/scenes/ithor/${scene}${suffix}"
    [[ -e "$src" ]] && ln -sfn "$src" "$REPO_ROOT/robocasa/models/assets/scenes/ithor/${scene}${suffix}"
  done
  echo "[ithor] linked $scene"
done

echo "[ithor] done. Remember: any shell running \`lab collect --env IThor*\` needs:"
echo "  export MLSPACES_ASSETS_DIR=$MLSPACES_ASSETS_DIR"
