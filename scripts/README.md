# Project tools

Start with the repository's [installation and collection guide](../README.md).
The maintained collection entry points are `robocasa.scripts.collect_demos_g1`
and `robocasa.scripts.collect_fixed_task`. Model training, policy servers, and
multi-policy evaluation are documented in
[LocoManiLab](https://github.com/LocoManiLab/LocoManiLab/tree/main/docs).

| Tool | Purpose |
|---|---|
| `install_ithor_scene.sh` | Prepare a selected scene; first use still needs the shared THOR library |
| `browse_tasks_render.py` | Browse/render tasks in an already provisioned simulator |
| `convert_meshy_furniture.py`, `decimate_furniture.py` | Convert and simplify explicitly selected local furniture assets |
| `usd_pipeline/` | Reusable local USD/MJCF conversion, box-collision setup, and fixture diagnostics |
| `setup_xrobotoolkit_ubuntu.sh`, `test_xrt.py` | Explicit optional headset setup and input probe |
| `gmr_arm_ab.py` | Offline comparison of recorded AMO IK and GMR arm tracking |

Use the selected tool's instructions and an appropriate existing environment.
Asset preparation, headset setup, rendering, and model execution are separate
operations. Ordinary CPU development does not require them.

The one-off scene prototypes, old demo shortcuts, model-specific tmux launchers,
and private asset-upload script have been retired. Reusable upstream RoboCasa
tools under `robocasa/scripts/` and the bundled `robosuite_src/` remain intact.
