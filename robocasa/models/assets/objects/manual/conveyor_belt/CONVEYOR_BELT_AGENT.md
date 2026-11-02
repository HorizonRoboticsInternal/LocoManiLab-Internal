# Conveyor belt usage for agents

## Asset

Use `conveyor_belt_single.xml` for one reusable conveyor fixture. Its rack is 4.3 m
long with an 8 cm tabletop; the belt extends slightly beyond it, is 1.04 m
wide, and its visible top is at `z = 0.80`. The belt direction is world
`+x`. The emerald-green belt wraps around the top, underside, and both ends of
the frame. Sixteen visible green tread plates move across the top and produce
the physical friction effect. The support is a plain tabletop with four legs.

The file contains no floor, lights, robot, or payload. These must be added by
the caller. Payload bottoms should start near `z = 0.80` and inside
`x = [-2, 2]`, `y = [-0.5, 0.5]`.

## Loading

```python
import mujoco

model = mujoco.MjModel.from_xml_path("examples/conveyor_belt/conveyor_belt_single.xml")
data = mujoco.MjData(model)
```

For RoboCasa environments, instantiate `ConveyorBeltFixture` and merge it as a
complete `MujocoXML` model so its actuator section is preserved. Do not pass it
through `ManipulationTask.mujoco_objects`, which only merges object bodies and
assets. Raw MjSpec callers can attach this file directly.

## Control

All sixteen belt actuators must receive the same target velocity. Positive
values move objects along `+x`; negative values move them along `-x`; zero
stops the belt.

```python
belt_actuators = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"belt_drive_{i:02d}")
    for i in range(16)
]

def set_belt_speed(data, belt_actuators, speed):
    # speed is in m/s and is limited by the XML ctrlrange to [-1.5, 1.5].
    data.ctrl[belt_actuators] = speed
```

Call `set_belt_speed(..., 0.7)` before every `mj_step` while running, and
`set_belt_speed(..., 0.0)` when stopped.

## Required wrap logic

The visible tread plates must be wrapped after each simulation step or before
the next one. Without wrapping, they eventually leave the belt area.

```python
belt_joints = [
    mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"belt_slide_{i:02d}")
    for i in range(16)
]
belt_base_x = [model.body_pos[model.jnt_bodyid[j], 0] for j in belt_joints]

for joint_id, base_x in zip(belt_joints, belt_base_x):
    qpos_addr = model.jnt_qposadr[joint_id]
    world_x = base_x + data.qpos[qpos_addr]
    if world_x > 2.16:
        data.qpos[qpos_addr] -= 4.32
    elif world_x < -2.16:
        data.qpos[qpos_addr] += 4.32
```

Do not directly overwrite the payload's `qvel` or `qpos`. Payload movement is
intended to come from contact friction. Payloads with `freejoint` can rotate;
lock their rotation only when the task explicitly requires a non-rotating
object.

## Naming and multiple belts

The names `belt_slide_00` through `belt_slide_15` and
`belt_drive_00` through `belt_drive_15` are part of the control interface. If
multiple conveyors share one model, prefix every body, joint, and actuator
name, for example `conv2_belt_slide_00` and `conv2_belt_drive_00`.
