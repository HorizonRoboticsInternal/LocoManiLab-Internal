# Installation

This fork supports G1 only. Follow the [current README](../../README.md#installation)
to create a Python 3.11 simulator environment and install its bundled
`robosuite_src` fork. Installing a separate upstream robosuite checkout would
restore a different robot catalog and does not provide this fork's G1 changes.

Start with `EmptyEnv` and one selected controller. The G1 model and native floor
are bundled; no room, kitchen, iTHOR or Objaverse asset pack is needed for that
scene. A control session still needs the selected controller environment and
weights. Prepare those separately using the README's explicit asset plans.

For task/model inspection and CPU checks, see [G1 scope](../g1_scope.md#cpu-development-without-new-assets).
Those checks require no policy weights or additional scene downloads.

Scene-specific assets remain optional. Install only the packs referenced by
your chosen environment; installation does not download them automatically.
Keep downloaded assets, local macros, datasets and checkpoints outside public
source archives. Older manuals under `docs/_versions` describe upstream
multi-robot releases and are retained as historical reference.
