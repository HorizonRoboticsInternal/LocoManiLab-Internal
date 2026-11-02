<h1 align="center">humanoid_benchmark</h1>

> This repository is a fork of [RoboCasa](https://github.com/robocasa/robocasa),
> modified for the [LocoManiLab](https://github.com/Gamma-Robot/LocoManiLab) benchmark.
> Licensed under the MIT License ([LICENSE](LICENSE)); see
> [THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md) for bundled third-party assets.

Simulation framework for **Unitree G1 whole-body teleoperation and data collection**,
built on [RoboCasa365](https://robocasa.ai) with a bundled fork of
[robosuite](https://github.com/ARISE-Initiative/robosuite) (no separate install).
PICO VR drives the robot through one of five whole-body policies; episodes are
recorded as LeRobot datasets.

Third-party asset files tracked here keep their own licenses: see
[THIRD_PARTY_ASSETS.md](THIRD_PARTY_ASSETS.md).

RoboCasa upstream: [home](https://robocasa.ai) · [docs](https://robocasa.ai/docs/introduction/overview.html) · [paper](https://robocasa.ai/assets/robocasa365_iclr26.pdf)

This fork supports **G1 only**, with `G1FloatingBody` as the default and the
`G1` / `G1FixedLowerBody` variants retained. Other robots' models, controller
configurations, grippers, demonstrations, and assets are not distributed.
Existing G1 tasks, scene definitions, calibrations, and shared framework code
remain in place. See [scope and compatibility](docs/g1_scope.md).

-------
## Installation

Start with **EmptyEnv + one selected controller**. EmptyEnv has a native MuJoCo
floor and the bundled G1 robot; it needs no kitchen, Objaverse, or iTHOR scene
pack. Package installation does not automatically download those packs or policy
weights. Python packages and source clones are separate explicit downloads.

For a new checkout, fetch only the maintained `main` branch:

```sh
git clone --single-branch --branch main https://github.com/LocoManiLab/humanoid_benchmark.git
cd humanoid_benchmark
```

Create a fresh simulator environment and install the bundled robosuite fork:

```sh
(
set -eu
conda create -c conda-forge -n tele_benchmark python=3.11
conda run -n tele_benchmark python -m pip install -e robosuite_src/
conda run -n tele_benchmark python -m pip install -e .
conda run -n tele_benchmark python -m robocasa.scripts.setup_macros
)
```

The last command writes the local `robocasa/macros_private.py`. Preserve an
existing environment/configuration rather than recreating it; use your own
compatible environment if it is already provisioned. For the remaining commands,
activate that environment with `conda activate tele_benchmark` and stay in the
`humanoid_benchmark` checkout. When this repository is used through LocoManiLab,
initialize its `sim` source profile instead of cloning a second copy: the main
repository's gitlinks select the integration revisions.

### G1 backend and selected weights

The shared controller/recording runtime is `g1_backend`, installed in the same
simulator environment. For a standalone checkout:

```sh
(
set -eu
git clone --single-branch --branch main https://github.com/LocoManiLab/g1_backend.git ../g1_backend
python -m pip install -e ../g1_backend --no-deps --no-build-isolation --ignore-requires-python
)
```

These compatibility flags preserve this simulator's dependency pins:
`--no-deps` avoids replacing its numpy/MuJoCo/LeRobot dependencies,
`--no-build-isolation` uses the environment's build tools, and
`--ignore-requires-python` permits the simulator's Python 3.11 environment to
install the runtime whose project metadata specifies Python 3.10. Keep the
runtime and simulator versions compatible. The distribution provides both
`g1_bridge` and `sim2real`; its directory name is not the import name.

For the SONIC `v1_1` example, inspect an offline plan, optionally query remote
sizes, then explicitly download only that selection:

```sh
# Offline static plan: no Hub requests, output directories, or downloads.
python robocasa/scripts/download_g1_checkpoints.py --family sonic_v1_1 --dest ../g1_backend/checkpoints --plan
# Optional Hub metadata query: exact file list/sizes, but no weight download.
python robocasa/scripts/download_g1_checkpoints.py --family sonic_v1_1 --dest ../g1_backend/checkpoints --dry-run
# Explicit download; the helper asks before transferring weights.
python robocasa/scripts/download_g1_checkpoints.py --family sonic_v1_1 --dest ../g1_backend/checkpoints
```

`--family` or `--all` is required. `--dryrun` is an alias for `--dry-run`;
both query the Hub, unlike offline `--plan`. The `sonic_v1_1` selection includes
its shared `planner_sonic.onnx` automatically, without fetching the low-latency
encoder/decoder or unrelated controller families. Other selections are `sonic`
(low latency), `amo`, `heft`, `scalebfm`, and `almi`. Select the controller and
checkpoint variant matching the data you intend to collect or replay. The
helper downloads weight files only and preserves Git-tracked policy manifests.

A first headset-free collection run needs no room pack:

```sh
python -m robocasa.scripts.collect_demos_g1 --env EmptyEnv --backend sonic \
    --policy-config ../g1_backend/checkpoints/sonic/v1_1/g1/policy.yaml --pico-source mock
```

This launches the simulator, rather than a dry-run; a working MuJoCo renderer and
the selected controller are still required. Real PICO teleoperation additionally
needs `xrobotoolkit_sdk` and the `roboticsservice` PC Service (port 60061). The
existing `scripts/test_xrt.py` probe can check an already provisioned headset.

### Assets by environment

| Environment/use | Required assets | Unrelated default downloads to skip |
| --- | --- | --- |
| `EmptyEnv` | Native floor XML, bundled G1 MJCF/STL files (G1 directory about 105 MiB in source), selected controller weights | All kitchen, Objaverse, and iTHOR scene packs |
| `TabletopPickPlace` | Shared floor/wall/counter textures and Objaverse mug/plate models; its counter XML is bundled | Generative textures/objects, unrelated fixtures and scenes |
| Other living-room/kitchen tasks | Their selected layout/style, furniture, textures, and object categories | Packs not referenced by the task; a full kitchen archive does not supply every custom living-room model |
| iTHOR FloorPlan tasks | Selected scene plus its THOR object dependencies and task assets | Unrelated kitchen/Objaverse packs; note the current installer still pulls the shared THOR library, about 13 GB on first use |

For the current tabletop recipe, preview the relevant archive categories:

```sh
python robocasa/scripts/download_kitchen_assets.py --type tex objs_objaverse --dry-run
```

This is an **offline category plan**. Remove `--dry-run` only when you explicitly
want those archives; it is not a per-object download. `--type` is required;
`--type all` explicitly selects the complete collection. Available categories are
`tex`, `tex_generative`, `fixtures_lw`, `objs_objaverse`, `objs_aigen`, and `objs_lw`.
Default tabletop mug/plate sampling uses Objaverse; a Lightwheel-only object pack
is not a replacement for that recipe.

For iTHOR, follow [scene installation](docs/molmospaces_scenes.md) and
`scripts/install_ithor_scene.sh` for your chosen FloorPlan. Selecting one scene
currently still requests the shared THOR object library; it does not imply a
small per-scene download. Other custom scenes may have additional asset terms
and local paths. Keep their provenance when transferring assets.

See [project tools](scripts/README.md) for optional asset preparation and
diagnostics. Use the generic/fixed collection entry points below for new data.

-------
## G1 data collection

For provisioned task assets, two entry points drive `G1BridgeComposite` (`G1_BRIDGE`):

```sh
# Fixed task: scene + initial task state pinned, small per-reset jitter.
python -m robocasa.scripts.collect_fixed_task \
    --env IThorKitchenOpenFridge --scene-seed 0 --backend sonic --pico-source real \
    --policy-config ../g1_backend/checkpoints/sonic/v1_1/g1/policy.yaml

# Generic: any registered env, scene selection forwarded to whatever it accepts.
python -m robocasa.scripts.collect_demos_g1 \
    --env IThorKitchenApproachFridge --scene-id 1 --backend sonic \
    --pico-source real --fixed-scene --scene-seed 0 \
    --policy-config ../g1_backend/checkpoints/sonic/v1_1/g1/policy.yaml
```

Data lands in `outputs_sim/<backend>/<env>_<timestamp>/` as a LeRobot dataset root
(`collect_fixed_task` adds a `_fixed` suffix); override with an absolute `--out`.
Hardware demos go to `outputs_real/` instead, from g1_backend's own collector —
one schema, two trees, because nothing inside a dataset says which it is.
The ego camera is kept as mp4 only — no PNG frames.

### Backends

| `--backend` | what drives the robot |
| --- | --- |
| `sonic` | SONIC whole-body tracking. PLANNER mode walks via a ported kPlanner; POSE mode tracks your upper body (SMPL/GR00T). |
| `amo` | Command-velocity legs/waist + PICO 3-point arm IK. |
| `heft` / `scalebfm` | Full-body GMR-retargeted tracking (launches a GMR subprocess). |
| `almi` | Locomotion commands plus GMR-retargeted upper-body tracking. |

Shared flags: `--policy-config <policy.yaml>` (override the backend's default
checkpoint), `--inference-backend {onnx-cpu,onnx-gpu,tensorrt,torch-jit}`,
`--max-episodes N`, `--out`, and scene selection — `--scene-id` (iThor FloorPlan),
`--layout`/`--style` (kitchen / living room), `--scene-seed`.

`collect_fixed_task` only: `--visualize-only` (run the viewer, record nothing),
`--g1-jitter-radius` (5 cm default), `--object-jitter-radius` (2 cm),
`--no-g1-jitter`, `--no-object-jitter`, `--g1-jitter-seed`.

`collect_demos_g1` only: `--fixed-scene` (identical scene every episode),
`--fixed-objects` (same instances, randomized poses), `--force-engage`.

### Operating it

PICO controller chords (`tracking` profile — `B+Y` and the left stick are muted):

| Chord | Action |
| --- | --- |
| `A+B+X+Y` | engage into IDLE / press again to ramp back to standing and hold |
| `A+X` | toggle POSE (tracks your body) ↔ IDLE (holds the last pose, motionless) |

IDLE and the stop ramp are **not** recorded — those frames have no operator command
behind them. Entering and leaving POSE is eased over `sim2real_pose_blend_s` so
re-entry never snaps; don't drop it far below 1.0 s on hardware.

Episodes **save automatically on the task-success rising edge**. Press `Q` in the
MuJoCo window to discard the in-progress episode and reset.

### Headless self-check (no display, no PICO)

```sh
MUJOCO_GL=egl python -m robocasa.scripts.collect_demos_g1 \
    --env EmptyEnv --backend sonic --pico-source mock \
    --policy-config ../g1_backend/checkpoints/sonic/v1_1/g1/policy.yaml
# Optional regression: provision the sonic and amo selections first.
MUJOCO_GL=egl python tests/test_standing_reference.py     # sonic + amo standing pose
# Optional WBT demo: provision the sonic (low_latency) selection first.
MUJOCO_GL=egl python -m robocasa.demos.demo_empty_teleop --robot G1FloatingBody --check
```

### Task environments

24 iThor room tasks are registered (`IThor{Kitchen,LivingRoom,Bedroom,Bathroom}*` —
`OpenFridge`, `CloseCabinet`, `CloseDrawer`, `CloseLaptop`, `ApproachFridge`, …).
List them with:

```py
from robosuite.environments.base import REGISTERED_ENVS
print(sorted(k for k in REGISTERED_ENVS if k.startswith("IThor")))
```

-------
## Checkpoints and path resolution

Policy checkpoints are **never** kept in this repo. The controller selections live
under `<g1_backend>/checkpoints/`: `sonic/low_latency/g1/`, `sonic/v1_1/g1/`,
`amo/`, `heft/pmg/`, `scalebfm/humanoid_transformer_m/`, and `almi/flat_dual/`.
The SONIC variants have different encoders/decoders and share the planner; use
the explicit selection described above instead of downloading every family.
They are published under `checkpoint/controller/` in the LocoManiLab release repository
[GammaRobotics/LocoManiLab](https://huggingface.co/datasets/GammaRobotics/LocoManiLab); use
`robocasa/scripts/download_g1_checkpoints.py` to fetch a selection.

Relative paths in the G1 controller configs are resolved against the `g1_backend`
repo root by `g1_bridge/g1_runtime.py:resolve_ckpt_path`, which checks
`$SIM2REAL_ROOT` and then the installed package.

> **Caveat:** `$SIM2REAL_ROOT` is honoured by `resolve_ckpt_path` but *not* by
> `g1_bridge/config.py`, which derives its own root from
> `sim2real.__file__`. Checkpoints placed anywhere other than beside the installed
> package will not be found by the `G1_BRIDGE` backends.

-------
## Keyboard locomotion (no PICO)

`demo_empty_teleop` runs the G1 under the `WHOLE_BODY_G1` composite (SONIC +
kinematic planner) with keyboard locomotion — useful for checking a scene without
the headset. `EmptyEnv` contains only an asset-free native MuJoCo ground plane;
`LivingRoomBasic` additionally needs its furniture assets. This optional keyboard
demo uses the `sonic` (low-latency) controller selection; a v1_1-only installation
should use the G1_BRIDGE collection command above, or explicitly provision that
additional selection before running this demo.

```sh
python -m robocasa.demos.demo_empty_teleop --robot G1FloatingBody
python -m robocasa.demos.demo_empty_teleop --robot G1FloatingBody --env LivingRoomBasic
```

There is no env action — the controller does everything internally, so the action
dim is a dummy `1`. The viewer opens in free-camera mode; `Tab` cycles the fixed
`FloorArena` cameras.

Focus the MuJoCo window first:

| Key | Action | | Key | Action |
| --- | --- | --- | --- | --- |
| `↑` `↓` | forward / back | | `-` `=` | target speed − / + |
| `←` `→` | strafe | | `;` `'` | target height − / + |
| `,` `.` | turn (yaw) | | `N` | re-init planner from current pose |
| `B` | brake | | `/` | print movement state + key help |
| `1`–`9`, `0` | locomotion mode | | `Q` | discard + reset while collecting |

In an `EmptyEnv` collection run, press `S` to save the current
free-teleoperation episode and reset. Task environments do not expose this
manual boundary; they save automatically when their success condition is reached.

| Mode | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | IDLE | SLOW_WALK | WALK | RUN | SQUAT | KNEEL_2 | KNEEL_1 | LYING | CRAWL | BOXING |

-------
## Optional research: Kimodo rollout

The normal collection workflow uses `g1_backend`. The separate Kimodo research
rollout requires an external `kimodo_rl` actor, matching tracker weights, and a
reference model; it is not a prerequisite for `--backend heft` or the benchmark.
See [the research resource guide](docs/kimodo_rollout.md) only if you intend to
use that experimental workflow. The standard setup does not fetch its resources.

-------
## Shared scene and object tools

The inherited scene, fixture, object, and task definitions remain available.
Prepare only the assets needed by the scene you select:

```sh
python -m robocasa.demos.demo_kitchen_scenes  # G1 viewer; local assets/controller required
python -m robocasa.demos.demo_objects         # browse objects (--obj_types aigen)
```

Use `collect_demos_g1` or `collect_fixed_task` for G1 recording. The retired
Panda demonstration collector/converter and upstream demonstration downloader
are not G1 workflows. Source definitions being retained does not certify that
every inherited kitchen task has been validated with G1.

The Gymnasium wrapper preserves `robocasa/<TaskName>` IDs and defaults to G1.
It exposes native observation keys and `env.action_spec` bounds. Whole-body
controllers use a dummy environment action and receive commands through their
backend; this is separate from the LocoManiLab policy-server interface. See the
[G1 scope guide](docs/g1_scope.md) before using it for policy learning.

> Run everything in the `tele_benchmark` conda env. On macOS use `mjpython`
> instead of `python` for any windowed (non-`--check`) command.
