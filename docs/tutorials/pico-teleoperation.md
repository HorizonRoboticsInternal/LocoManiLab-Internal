---
title: Pico Teleoperation
sidebar_position: 2
---

This tutorial uses the teleop publisher for live Pico / XR retargeting, its built-in mjviser server to inspect the retargeted G1 motion, and the root project tracking policy for execution.

## 1. Start the Pico retarget publisher

```bash
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py
```

Open the mjviser URL printed by the publisher and keep it open until the retargeted G1 motion looks correct.

## 2. Choose the execution backend

### Sim2Sim

Start the MuJoCo execution process:

```bash
uv run sim2real/sim_env/base_sim.py
```

In another terminal, start the tracking policy against the live motion stream:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/heft/pmg/policy.yaml \
  --motion-backend zmq \
  --controller pico
```

To also see the Dex3 hand move with real contact physics in this MuJoCo window
(no real Dex3 hand or DDS network needed -- see "Dex3 hand control" below),
add one flag to the Sim2Sim process:

```bash
uv run sim2real/sim_env/base_sim.py --with-dex3-hands
```

The tracking policy command doesn't change -- the sim reads the same
`hand_grip` stream directly, independent of any `--dex3-hand-dir` flag.

### Sim2Real

For hardware, first choose the deployment path in [Robot I/O](/reference/robot-io). The Pico-specific policy flags stay the same:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/heft/pmg/policy.yaml \
  --motion-backend zmq \
  --controller pico
```

Add only the robot I/O flag or bridge process required by the mode you chose.

## Pico Controls

- Press `A` to enter the init pose.
- Press `A` + `B` to enter policy mode.
- Press `X` to unpause the motion flow.

## Optional DH116S hand integration

The hand-grip stream can be consumed by an external DH116S adapter, but this
checkout does not ship `venv/dh116s` or a DH116S control script. Follow the SDK
provider's isolated setup for that hardware. The maintained in-repo hand path
is Unitree Dex3, described below.

## Dex3 hand control

If your G1 has Unitree's official Dex3 hand instead of a DH116S, hand control
runs **in the same tracking-policy process** -- no separate hand process to
launch. It reads the exact same hand-grip stream on TCP port `5593` (the left
trigger controls the left Dex3, the right trigger controls the right Dex3),
but talks to the hand over the same DDS bus used for arm/leg control
(`rt/dex3/{left,right}/cmd`/`state`), not a separate vendor SDK.

Unlike DH116S, Dex3 needs no separate SDK download or isolated environment:
`unitree_sdk2py` (which already ships the Dex3 `HandCmd_`/`HandState_` IDL
types) is the same package the `robot-g1` extra already installs for arm/leg
control. If you can already deploy to the real G1, you already have
everything Dex3 control needs.

Add a single flag, `--dex3-hand-dir double`, to the tracking policy command
from step 2 above -- the launch command itself doesn't change:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/heft/pmg/policy.yaml \
  --motion-backend zmq \
  --controller pico \
  --dex3-hand-dir double
```

That's the whole diff versus the command you already run without hand
control -- everything else defaults to the common case: the hand-grip stream
is read from the local Pico publisher (`tcp://127.0.0.1:5593`; pass
`--hand-grip-connect tcp://<publisher-ip>:5593` only if it runs on a different
machine), and the DDS domain/interface default to `0`/`eth0` (override with
`--dex3-domain-id`/`--dex3-interface` only if your G1's arm/leg control uses
something else). When `--record-format lerobot` recording is also enabled, the
hand's full 7-per-hand joint targets and measured angles
(`action.hand_joints`/`observation.hand_state`) are recorded automatically --
no need to also pass `--record-extra dex3`.

In Sim2Sim or on a dev machine with no real Dex3 hand (and no `eth0`/matching
DDS network interface -- `ChannelFactoryInitialize` will otherwise fail with
"no such interface"), add `--dex3-dry-run` to map grips to joint targets and
record them without touching `unitree_sdk2py`/DDS at all:

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/heft/pmg/policy.yaml \
  --motion-backend zmq \
  --controller pico \
  --dex3-hand-dir double \
  --dex3-dry-run
```

:::warning
The hand is enabled and driven toward the commanded grip from the moment the
tracking process starts. Clear the workspace around both hands before
launching it. Combining `--dex3-hand-dir` with `--robot-io inline` has not been
validated for DDS-participant conflicts with the debug-mode handshake in
`robot_io/g1_debug_mode.py` -- it's known-safe with the default `--robot-io
zmq`.
:::

Each tick's hand command is clamped against the hand's last measured joint
angles (`--dex3-max-delta-q`, default `0.25` rad) so a grip-stream glitch can't
snap the fingers. When the grip stream itself goes stale, the hand simply stops
receiving new commands and holds its last commanded pose.

For manual bring-up or a hardware smoke test *without* running the full
tracking policy, `scripts/dex3_control.py` is still available as a standalone
process, from the same
`robot-g1`-extra environment:

```bash
uv run --extra robot-g1 scripts/dex3_control.py \
  --hand-dir double --connect tcp://<pico-publisher-ip>:5593
```

Add `--dry-run` to validate the ZMQ stream and grip-to-joint mapping without
importing `unitree_sdk2py` or moving hardware.

### Seeing the Dex3 hand move in Sim2Sim

Everything above (`--dex3-hand-dir`, `--dex3-dry-run`) drives the real/dry-run
Dex3 *driver* -- it never touches the sim2sim MuJoCo model, which doesn't have
finger joints by default. To actually see (and physically collide) a Dex3 hand
in the Sim2Sim window, add `--with-dex3-hands` to `base_sim.py` instead (see
step 2 above):

```bash
uv run sim2real/sim_env/base_sim.py --with-dex3-hands
```

This splices Dex3 hand bodies/joints/collision geometry onto both wrists
(`sim2real.sim_env.utils.dex3_mjcf.add_dex3_hands`, replacing the decorative
`left_rubber_hand`/`right_rubber_hand` mesh) and drives them with local PD
control from the same `hand_grip` ZMQ stream `pico_retarget_pub.py` already
publishes -- independent of whatever the tracking policy process does with
`--dex3-hand-dir`/`--dex3-dry-run`. No real Dex3 hand, DDS, or network
interface is needed for this; it's pure MuJoCo physics.

:::warning
The PD gains default to the same real-hardware values `sim2real.utils.dex3`
uses (heft's Dex3HandBridge tuning) -- these reach the close pose within
about a second given the reference asset's actuator torque cap, but with
some overshoot/ringing (the real motor's own damping isn't reproduced by the
sim's PD). Override with `--dex3-kp`/`--dex3-kd`/`--dex3-thumb-kp`/
`--dex3-thumb-kd` if that's undesirable for your use case.
:::

## Notes

- `pico_retarget_pub.py` publishes the live motion stream consumed by the tracking policy and opens the retarget mjviser server.
- Hand-grip messages use their own ZMQ port and do not change the existing Pico button protocol. DH116S and Dex3 both consume that same `hand_grip` port/message format -- only the hand-side consumer differs (an external adapter for DH116S; in-process via `--dex3-hand-dir` for Dex3; `base_sim.py --with-dex3-hands` for a simulated hand).
- `sim2real/sim_env/base_sim.py` is the sim2sim execution backend.
- For real hardware, [Robot I/O](/reference/robot-io) lists the inline and bridge deployment modes.
- If the publisher and policy run on different machines, add `--motion-zmq-connect tcp://<publisher_ip>:28701`.

## Next Steps

- [Motion Recording](/tutorials/motion-recording)
