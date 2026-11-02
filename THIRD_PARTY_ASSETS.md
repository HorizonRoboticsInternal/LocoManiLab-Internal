# Third-party assets

The code in this repository is MIT-licensed ([LICENSE](LICENSE); the bundled robosuite fork
keeps [robosuite_src/LICENSE](robosuite_src/LICENSE)). Some asset files tracked in this
repository come from third-party asset releases, in original or adapted form, and keep
their own licenses. The MIT License does not cover them.

| Tracked files | Origin | License |
|---|---|---|
| `robocasa/models/assets/scenes/holodeck-objaverse-train/` (Holodeck scenes 117, 1018, 1197, 1664, 2025: scene files and per-scene assets) | Holodeck scenes from [MolmoSpaces](https://huggingface.co/datasets/allenai/molmospaces) (`mujoco/scenes/holodeck-objaverse-train`), some adapted | CC BY 4.0; the Objaverse objects these scenes load keep their per-object licenses (see below) |
| `robocasa/models/assets/scenes/ithor/FloorPlan*_*.xml` | iTHOR scenes from [AI2-THOR](https://ai2thor.allenai.org/) via MolmoSpaces (`mujoco/scenes/ithor`), some adapted | CC BY 4.0 |
| `robocasa/environments/ithor_kitchen/tasks/assets/Apple_25/` (mesh, collision meshes, texture; `model.xml` is a wrapper written for this repository) | iTHOR object `Apple_25` from AI2-THOR via MolmoSpaces (`objects/thor/Kitchen Objects/Apple/Prefabs/Apple_25/`) | CC BY 4.0 |
| `robocasa/models/assets/objects/lightwheel/glass_cup/GlassCup014/model_long_grip.xml`, `robocasa/models/assets/objects/lightwheel/teddy_bear/TeddyBear002/model.xml` | Lightwheel objects from [NVIDIA PhysicalAI Kitchen MJCF](https://huggingface.co/datasets/nvidia/PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF), adapted | CC BY 4.0 |
| `robocasa/models/assets/fixtures/cabinets/cabinet_panels/`, `robocasa/models/assets/fixtures/accessories/light_switches/white_wide_switch/`, `robocasa/models/assets/fixtures/accessories/outlets/simple_white/`, `robocasa/models/assets/fixtures/counters/*/visuals/marble*.png` | [RoboCasa](https://github.com/robocasa/robocasa) asset archives (including Lightwheel fixtures) | CC BY 4.0 |
| `robosuite_src/robosuite/models/assets/` except `robots/g1/` (robot descriptions, meshes, textures) | Upstream [robosuite](https://github.com/ARISE-Initiative/robosuite) | MIT ([robosuite_src/LICENSE](robosuite_src/LICENSE)) |
| `robosuite_src/robosuite/models/assets/robots/g1/meshes/` | Unitree G1 robot description meshes ([Unitree](https://github.com/unitreerobotics/unitree_ros)) | BSD-3-Clause (as published by Unitree) |

Asset files that are not tracked here (object libraries, textures, most scene packages) are
downloaded separately from their original hosts under those hosts' terms; LocoManiLab's
`scripts/hf/download.py assets` lists each file's source.

Objaverse objects keep the license chosen by each creator. Two objects loaded by Holodeck
scene 117 are CC BY-NC 4.0 (framed wall art `596fd518055a49a28d56c193cd087470` and
`f33af1b399a24448880930fef312ef3f`); their non-commercial terms apply wherever they are
used or rendered.

## Assets created for this project

The following directories under `robocasa/models/assets/objects/manual/` were made by the
LocoManiLab team and are covered by this repository's MIT License. The meshes and PBR
textures (hash-named OBJ files with `texture_pbr_20250901*.png` maps) were generated with
Tencent Hunyuan3D.

| Directory | Contents |
|---|---|
| `box/`, `desk/`, `green_desk/`, `handle/`, `shelf/` | Hunyuan3D-generated mesh and PBR maps, MJCF written for this repository |
| `cup/` | Hunyuan3D-generated mesh and PBR maps; `cup.xml` uses `cup_hollow.obj` (procedural hollow cup) and `cup_wood_dark.png` |
| `fridge/` | MJCF only; loads RoboCasa `Refrigerator032` fixture meshes (downloaded separately, CC BY 4.0) and `handle/` |
| `conveyor_belt/` | MJCF and script, no meshes |
| `open_cardboard_box/` | MJCF built from primitives, no meshes |
