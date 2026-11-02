"""Run the standalone conveyor-belt example.

Space: start/stop       Left/Right: change speed       Esc or Q: quit
"""

from pathlib import Path
import time

import mujoco
import mujoco.viewer


MODEL_PATH = Path(__file__).with_name("conveyor_belt_single.xml")
BeltState = dict[str, float | bool]


def build_model() -> mujoco.MjModel:
    """Load the fixture and add demo-only room elements and cubes."""
    spec = mujoco.MjSpec.from_file(str(MODEL_PATH))

    light = spec.worldbody.add_light()
    light.name = "demo_top_light"
    light.pos = [0.0, 0.0, 3.0]
    light.directional = True

    floor = spec.worldbody.add_geom()
    floor.name = "demo_floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [5.0, 5.0, 0.1]
    floor.rgba = [0.20, 0.21, 0.22, 1.0]
    floor.friction = [0.8, 0.03, 0.01]

    cube_specs = (
        ("test_cube_orange", [-1.45, -0.28, 0.92], [0.92, 0.28, 0.06, 1.0]),
        ("test_cube_blue", [-0.65, 0.00, 0.92], [0.08, 0.38, 0.90, 1.0]),
        ("test_cube_yellow", [0.15, 0.28, 0.92], [0.94, 0.72, 0.08, 1.0]),
    )
    for name, pos, rgba in cube_specs:
        body = spec.worldbody.add_body()
        body.name = name
        body.pos = pos
        body.add_freejoint().name = f"{name}_joint"
        geom = body.add_geom()
        geom.name = f"{name}_geom"
        geom.type = mujoco.mjtGeom.mjGEOM_BOX
        geom.size = [0.10, 0.10, 0.10]
        geom.mass = 0.8
        geom.rgba = rgba
        geom.friction = [0.8, 0.01, 0.001]
    return spec.compile()


def main() -> None:
    model = build_model()
    data = mujoco.MjData(model)
    belt_actuators = []
    belt_joints = []
    for i in range(model.nu):
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"belt_drive_{i:02d}"
        )
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, f"belt_slide_{i:02d}"
        )
        if actuator_id < 0 or joint_id < 0:
            break
        belt_actuators.append(actuator_id)
        belt_joints.append(joint_id)
    belt_base_x = [
        model.body_pos[model.jnt_bodyid[joint_id], 0] for joint_id in belt_joints
    ]
    spacing = (max(belt_base_x) - min(belt_base_x)) / (len(belt_base_x) - 1)
    wrap_limit = max(abs(min(belt_base_x)), abs(max(belt_base_x))) + spacing / 2
    state: BeltState = {"running": True, "speed": 0.7, "quit": False}

    def key_callback(key: int) -> None:
        if key in (256, 81):  # Escape or Q
            state["quit"] = True
        elif key == 32:  # Space
            state["running"] = not state["running"]
        elif key == 263:  # Left arrow
            state["speed"] = max(0.0, state["speed"] - 0.1)
        elif key == 262:  # Right arrow
            state["speed"] = min(2.0, state["speed"] + 0.1)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running() and not state["quit"]:
            speed = state["speed"] if state["running"] else 0.0
            for actuator_id in belt_actuators:
                data.ctrl[actuator_id] = speed
            # Wrap each slat so the moving contact surface is endless.
            for joint_id, base_x in zip(belt_joints, belt_base_x):
                qpos_addr = model.jnt_qposadr[joint_id]
                world_x = base_x + data.qpos[qpos_addr]
                if world_x > wrap_limit:
                    data.qpos[qpos_addr] -= 2 * wrap_limit
                elif world_x < -wrap_limit:
                    data.qpos[qpos_addr] += 2 * wrap_limit
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.001)


if __name__ == "__main__":
    main()
