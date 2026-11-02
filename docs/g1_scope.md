# G1 scope and compatibility

This maintained benchmark supports one G1 robot per environment. The default
is `G1FloatingBody`; `G1` and `G1FixedLowerBody` remain available. Use the
bundled `robosuite_src` fork rather than an independently installed upstream
robot catalog.

## What stays shared

The environment, camera, object, fixture, MJCF, generic controller, and
mobile/legged robot abstractions remain shared framework code. G1's fingers
are authored into its MJCF, so `NullGripper` is its intentional attachment.
The existing G1 model files, joint/actuator ordering, controller configurations,
scene assets, task definitions, success conditions, and calibration values
remain together.

All existing benchmark task registration names remain. Inherited kitchen task
definitions are preserved as source; they do not imply fresh validation of
every task with G1. Select and validate a task with its required local assets
before collecting data.

## Retired robot workflows

Other concrete robot models and their dedicated configurations, grippers,
mounts, assets, example environments, and demos have been removed. Optional
external robot catalogs and Mink examples are no longer imported automatically.
Unsupported robot names fail with the G1 choices rather than loading another
embodiment or looking for its assets.

The old Panda teleoperation collector, Panda dataset exporter/converter,
official Panda demonstration downloader, and embodiment-specific benchmark
scripts are retired. Generic HDF5 metadata/playback utilities and historical
dataset identity tags remain where shared consumers still use them. Keeping a
historical tag does not expose that robot as a supported simulator model.

Use `robocasa.scripts.collect_demos_g1` or `collect_fixed_task` for G1 data.
Both use `g1_backend` and preserve its controller-specific dataset contracts.

## Native Gymnasium interface

The existing `robocasa/<TaskName>` IDs expose the underlying G1 environment:

- Observation keys, shapes and dtypes come from the native environment.
- The action space comes from `env.action_spec`.
- Native reward, done and info values are forwarded. Task text is added to
  info; the wrapper does not evaluate success a second time.
- `create_env` defaults to G1 and its head camera. Use
  `render_offscreen=False` for checks that do not need camera observations.

`WHOLE_BODY_G1` and `G1_BRIDGE` expose a dummy environment action while their
backend accepts the actual reference/token/command stream. A learned model
must use the matching backend command contract. The Gym action Box is not a
replacement for LocoManiLab's policy-server protocol.

Task and model inspection only registers lightweight controller factories;
optional policy runtimes are imported when a selected controller is actually
constructed. Runtime execution still requires that controller's environment
and weights.

Importing the task catalog resolves optional asset paths without loading them.
The object loader reports a missing XML when that object is actually created;
select only the asset packs needed by the chosen task.

## CPU development without new assets

In a provisioned Python 3.11 simulator environment, run the focused checks:

```sh
CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 \
  python -m pytest -q tests/test_g1_distribution.py \
  tests/test_g1_gym_wrapper.py tests/test_empty_env.py \
  tests/test_task_asset_paths.py
```

These cases use the bundled G1 XML/meshes, fake Gym environments, and the
native floor definition. They do not run a policy, download assets, open a
camera/robot SDK, or render a scene. CPU checks do not establish controller
quality or task completion on hardware.

Use the [README](../README.md) for selected controller/scene setup. Historical
upstream manuals and `docs/_versions` describe the original multi-robot
projects; use current fork guides for supported commands. Upstream attribution
and license notices remain in this repository and in `robosuite_src`.
