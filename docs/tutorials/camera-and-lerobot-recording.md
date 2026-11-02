---
title: Camera and LeRobot Recording
sidebar_position: 4
---

This tutorial adds a camera feed to a policy run and records state/action/image episodes
in [LeRobot](https://github.com/huggingface/lerobot) v2.1 format, suitable for VLA
training data collection. It builds on the [Pico Teleoperation](./pico-teleoperation.md)
tutorial's setup.

`--record` alone (no flags below) keeps recording exactly as it was before this feature:
a single flat `.npz` dump for the whole process, no images. Everything here is opt-in.

## 1. Install the optional extras

```bash
uv sync --extra camera --extra lerobot
```

## 2. Start a camera server

```bash
uv run python -m sim2real.camera.server --cameras.ego_view.source realsense
```

Use `--cameras.ego_view.source cv2 --cameras.ego_view.device 0` for a plain USB webcam
instead of a RealSense. The server publishes on `tcp://*:5594` by default (`PORTS["camera"]`).

## 3. Run the policy with LeRobot recording enabled

```bash
uv run sim2real/rl_policy/tracking.py \
  --controller pico \
  --record --record-format lerobot \
  --camera-connect tcp://127.0.0.1:5594 \
  <other tracking args, see Pico Teleoperation>
```

Add `--hand-grip-connect tcp://127.0.0.1:5593` too if you want the PICO trigger-derived
hand-grip commands recorded as `action.hand_grip`.

Dataset creation is deferred until the first frame where every attached signal (camera,
hand-grip) has actually produced data, so the schema (including image shape) is
auto-detected rather than declared up front.

## 4. Record episodes

- **Keyboard**: `c` to start/stop, `x` to discard. This is the only way in.

The PICO headset has no recording gesture. `Y`/`Y+A` used to be one and were removed:
a bare rising edge on a button the operator's thumb rests near, for the one action
that silently changes what lands on disk, is a trap — see `g1_bridge/hosts/real.py`.

These are the same start/stop/discard semantics already used by downstream
data-collection forks (e.g. the `heft`/`AMO`/`sonic` controllers), so the same muscle
memory carries over.

## Output

A LeRobot v2.1 dataset directory (`meta/info.json`, `meta/modality.json`,
`meta/episodes.jsonl`, `data/`, `videos/`) at `--record-output`, or a timestamped
default under the current directory. `observation.*` features come straight from this
policy family's own `observation` groups (`policy_config["observation"]`) plus baseline
robot state (`joint_pos`/`joint_vel`/`root_quat_w`/`root_ang_vel_b`); `action.q_target`/
`action.raw` come from what the control loop already computes every step. No per-family
code is required for any of this -- only the camera/hand-grip fields need explicit flags,
since they live outside the `Observation` registry.
