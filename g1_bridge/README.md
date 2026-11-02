# g1_bridge — the teleop layer, shared by the simulator and the real robot

`sim2real` (this repo's other top-level package) runs a whole-body policy against a
robot. `g1_bridge` is the layer above it: the operator's PICO input, the mode state
machine, the per-backend reference logic, and the LeRobot recording that turns a
teleop session into a training dataset.

It exists as its own package because **two different hosts drive it**:

| host | where | what it supplies |
| --- | --- | --- |
| `robocasa.g1_host.bridge.G1BridgeComposite` | humanoid_benchmark | MuJoCo state/torques, the rendered ego camera, the airborne-spawn settling window |
| `g1_bridge.hosts.real.RealRobotHost` | here | the robot's own state stream (`RobotIO`), a camera server, the arming state machine |

Everything between those two ends — `session.py`, `backends/`, `pico/`, `gmr_utils/`,
`reference/`, `modality.py`, `recording/` — is the same code on both paths. That is
the point: a dataset collected in sim and one collected on hardware carry the same
columns, in the same layout, produced by the same `teleop_sample()`.

## The host seam

A host implements five methods and a few optional attributes:

```python
read_joint_state()  -> (qpos[29], qvel[29])   # G1_JOINT_NAMES order
read_pelvis_state() -> (quat_wxyz, ang_vel)   # IMU on hardware
read_root_pos()     -> xyz                    # zeros on hardware (no odometry)
read_body_pose_w(name)                        # FK; hardware raises NotImplementedError
clock_ms()          -> int                    # sim time in sim, wall time on hardware
sim / robot_model / hand_mapping / qpos_addrs / qvel_addrs   # None off-sim
```

`TeleopSession` never touches anything else, and backends reach the host only through
the session (`Sim2RealContext`). New teleop logic goes in the session or a backend;
only host-specific behaviour goes in a host.

Inside, sim2real is always driven in **push mode**: the session builds a `RobotState`,
pushes it into a `PushRobotIO`, calls `tracking.step()` and reads the command back.
The real host owns a *second*, hardware `RobotIO` and pumps between the two, so the
numerically load-bearing part of the control path is byte-identical to the sim's.

## Real-robot collection

```bash
# terminal 1 — camera (match the control rate, or ~40% of frames repeat)
uv run python -m sim2real.camera.server --cameras.ego_view.source realsense --fps 50
# terminal 2 — robot bridge (only for --robot-io zmq, the default)
uv run python scripts/g1/real_bridge.py --interface eth0
# terminal 3 — collect
uv run python -m g1_bridge.collect_real --backend heft --pico-source real \
    --task-prompt "pick up the cup" --out /abs/path/to/dataset
```

Operator flow (collector terminal): the robot starts at zero torque and asks
for confirmation. `y`/Enter ramps it to the default stance. Once it is standing,
press `a` or the PICO `A+B+X+Y` engage chord to enter the tracking policy; both
paths refuse arming before the ramp finishes or while the robot is moving.
Arming captures the one-shot yaw alignment.

After arming, `c`/Enter starts or stops and saves an episode; `x` discards it.
`o` returns to zero torque and resets the confirmation gate, and `q` quits.
The separate keyboard pane accepts recording keys only. It cannot confirm startup
or arm the robot; use the collector terminal for those actions.

PICO `A+B+X+Y` toggles engagement and `A+X` toggles POSE; triggers drive the
hands. `e`/`p` in the collector terminal are keyboard fallbacks for the two mode
chords. A bare PICO `A` or `A+B` never arms this host, and the headset does not
start, save or discard recordings.

Frames are recorded only while the mode manager is engaged *and* a real reference
exists, so standing up, IDLE holds and stop ramps stay out of the dataset.

All five backends are supported: `sonic`, `heft`, `scalebfm`, `amo` and `almi`.
SONIC reads measured finger joints from the Dex3 driver. AMO records the previous
torso command, rather than an unobservable measured torso height. ALMI reads the
IMU orientation and uses commanded base height and GMR end-effector targets.
Choose the backend matching the policy's recorded action space; their action
representations are not interchangeable.

No hardware? The same code path runs against `sim2real/sim_env/base_sim.py` over ZMQ:
add `--robot-io zmq --pico-source mock --dex3-dry-run --arm-on-start --init-seconds 10
--force-engage --debug-record-steps 600`.

## Two recorders, on purpose

`sim2real/rl_policy/recording/` records a *policy-debugging* dataset: a family-independent
schema derived from whatever `observation` groups the policy.yaml declares
(`observation.joint_pos`, `action.q_target`, ...). `g1_bridge/recording/` records the
*training* dataset: the per-backend contract in `modality.py` (`observation.state(29)` /
`action.ref_qpos(36)` / `action.hands(14)` for heft). Different entry points, different
purposes; neither is going away.
