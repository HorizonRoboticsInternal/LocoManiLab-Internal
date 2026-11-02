# MuJoCo conveyor belt

Run from the repository root:

```bash
python robocasa/models/assets/objects/manual/conveyor_belt/run_conveyor_belt.py
```

Controls:

- `Space`: start/stop the belt
- `Left` / `Right`: decrease/increase belt speed
- `Esc` or `Q`: close the viewer

TeleLab environments:

- `TeleLabEnv` loads only the empty authored room.
- `TeleLabConveyor` merges `ConveyorBeltFixture` and the cleaned green desk
  into that room. Its default layout places the desk perpendicular to the
  conveyor at the left/near corner, with the robot inside the L facing the
  conveyor. The corner and robot offsets adapt to fixture scaling.
- `env.set_conveyor_speed(0.7)` moves payloads along world `+x`.
- In the G1 viewer, `U` / `O` decrease/increase speed and `P` stops it.

Example:

```python
import robocasa

env = robocasa.make(
    "TeleLabConveyor",
    robots="G1FloatingBody",
    conveyor_speed=0.3,
    conveyor_speed_step=0.1,
    conveyor_speed_range=(-1.0, 1.0),
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
)
env.reset()
env.set_conveyor_speed(0.7)
```

In the TeleLab teleop viewer, press `U` / `O` to decrease / increase the
configured speed by `conveyor_speed_step`, or `P` to stop. Commands are clamped
to `conveyor_speed_range`. The same values can be supplied on the command line:

```bash
python -m robocasa.demos.demo_empty_teleop \
    --env TeleLabConveyor \
    --conveyor-speed 0.3 \
    --conveyor-speed-step 0.1 \
    --conveyor-speed-range -1.0 1.0
```

The generic scene visualizer starts this environment at `0.3 m/s`. Its speed
is selected before launch and remains fixed while the viewer is open:

```bash
MUJOCO_GL=glfw python -m robocasa.scripts.visualize_scene \
    --env TeleLabConveyor --conveyor-speed 0.5
```

The conveyor is implemented once in `conveyor_belt_single.xml`. It contains no
room floor, lights, robot, or payloads. The standalone runner adds its demo
floor, light, and three test cubes at runtime.

The conveyor frame uses an 8 cm tabletop and four legs, leaving the space below
the belt open instead of filling it with a solid pedestal.

The model uses a plain four-legged support rack. Its emerald-green belt forms a
continuous capsule around the tabletop, underside, and both ends. Sixteen
transparent tread plates are attached to independent MuJoCo slide joints.
Payload motion therefore comes from contact friction, rather than the
controller directly overwriting payload velocity. The demo uses
`friction="0.45 0.005 0.001"`: sliding friction, torsional friction, and rolling
friction respectively. A free payload may rotate if the contact dynamics cause
it to; that is physically different from imposing a no-rotation constraint.
