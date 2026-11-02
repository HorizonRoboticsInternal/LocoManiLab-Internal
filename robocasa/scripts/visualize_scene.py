"""Generic scene/task visualizer — works for ANY registered env
(kitchen / living-room / MolmoSpaces iTHOR rooms / future scene types).

Modes:
  * default           : interactive MuJoCo operator window (orbit/zoom with the
    mouse, drive the G1 with the keyboard). Needs a display (MUJOCO_GL=glfw).
  * --probe           : calibration mode — keyboard-driven 3D probe marker for
    measuring world coordinates of tables, counters, etc.  No robot control;
    the viewer is purely for scene inspection + coordinate measurement.
  * --calibrate-spawn: move and rotate G1's base, then save its pose as an
    iTHOR scene calibration.  Auto-placement remains the fallback for scenes
    without a saved calibration.

Probe controls (--probe):
    W / S        — move probe +X / -X (world frame)
    A / D        — move probe +Y / -Y
    R / F        — move probe +Z / -Z
    1 / 2 / 3 / 4 — step size 0.001 / 0.010 / 0.100 / 1.000 m  (default: 0.01)
    Enter        — print probe coords in ithor_scene_specs.yaml-ready format
    Q            — quit

Spawn calibration controls (--calibrate-spawn):
    W / S, A / D — move G1 base in world-frame X / Y
    Z / X        — rotate yaw left / right
    1 / 2 / 3 / 4 — move step 0.01 / 0.05 / 0.10 / 0.25 m
    Enter        — write the current position and yaw to ithor_scene_specs.yaml
    N            — discard unsaved movement and switch to the next FloorPlan
    H            — toggle the main viewer between free view and G1 head camera

Task-layout calibration controls (--calibrate-task-layout):
    Enter        — save the current box / task-object poses to the task's YAML

Support inspection (--show-support):
    Cyan         — semantic support rectangles
    Blue         — manually calibrated scene-level support rectangles

Support calibration (--calibrate-support):
    W/S, A/D, R/F — move rectangle centre in X/Y/Z
    Z/X           — rotate rectangle yaw
    I/K, J/L      — grow/shrink local X/Y half-size
    Enter         — append this rectangle to ithor_support_specs.yaml

Scene selection is forwarded to whatever the env accepts (--scene-id for
MolmoSpaces rooms, --layout/--style for kitchen/living-room, --fixed-scene).

Usage:
    # interactive window
    MUJOCO_GL=glfw python -m robocasa.scripts.visualize_scene \
        --env IThorBedroomLiftBasket --scene-id 301

    # calibration probe
    MUJOCO_GL=glfw python -m robocasa.scripts.visualize_scene \
        --env IThorKitchenEnv --scene-id 1 --probe

    # calibrate a task-specific G1 spawn and persist it
    MUJOCO_GL=glfw python -m robocasa.scripts.visualize_scene \
        --env IThorKitchenApproachFridge --scene-id 1 --calibrate-spawn
"""

import argparse
import os

import numpy as np

from g1_bridge.backends import backend_names
from robocasa.utils.g1_data_collection import (
    EGO_CAMERA,
    build_teleop_env,
    forward_scene_kwargs,
)


# ---------------------------------------------------------------------------
# Probe (calibration marker) loop
# ---------------------------------------------------------------------------
def run_probe_viewer_loop(env):
    """Open the MuJoCo viewer with a keyboard-driven 3D calibration probe.

    A small red sphere is rendered at the probe position via the viewer's
    ``user_scn`` overlay (no physics interaction).  Keyboard controls move
    the probe in world-frame X/Y/Z and print coordinates to the terminal.

    Movement is *continuous*: press a direction key once to start moving,
    press Space to stop.  The step-size keys (1-4) set the movement speed.
    """
    import time as _time
    import mujoco

    env.reset()
    base = getattr(env, "unwrapped", env)
    scene_name = getattr(base, "scene_name", "?")
    print(f"[probe] env={type(base).__name__}  scene={scene_name}")
    print("[probe] Controls:")
    print("  W / S        — start moving +X / -X  (press Space to stop)")
    print("  A / D        — start moving +Y / -Y")
    print("  R / F        — start moving +Z / -Z")
    print(
        "  1 / 2 / 3 / 4 — speed: 0.001 / 0.010 / 0.100 / 1.000 m/s  [current: 0.010]"
    )
    print("  Space        — stop moving")
    print("  Enter        — print '>>> PROBE: center=(x, y, z)' for copy-paste")
    print("  Q            — quit")
    print()

    model = env.sim.model._model
    data = env.sim.data._data

    probe_pos = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    probe_vel = np.zeros(3, dtype=np.float64)  # world-frame velocity (m/s)
    speed_levels = {49: 0.001, 50: 0.010, 51: 0.100, 52: 1.000}  # GLFW key -> m/s
    cur_speed = 1.0  # m/s

    # last print time (throttled so terminal isn't flooded)
    last_print = _time.monotonic()

    # GLFW key codes
    _KEY_W, _KEY_S = 87, 83
    _KEY_A, _KEY_D = 65, 68
    _KEY_R, _KEY_F = 82, 70
    _KEY_SPACE = 32
    _KEY_ENTER = 257
    _KEY_Q = 81

    def key_cb(keycode: int) -> None:
        nonlocal cur_speed, probe_vel
        speed = cur_speed

        if keycode == _KEY_W:
            probe_vel[:] = (speed, 0.0, 0.0)
        elif keycode == _KEY_S:
            probe_vel[:] = (-speed, 0.0, 0.0)
        elif keycode == _KEY_A:
            probe_vel[:] = (0.0, speed, 0.0)
        elif keycode == _KEY_D:
            probe_vel[:] = (0.0, -speed, 0.0)
        elif keycode == _KEY_R:
            probe_vel[:] = (0.0, 0.0, speed)
        elif keycode == _KEY_F:
            probe_vel[:] = (0.0, 0.0, -speed)
        elif keycode == _KEY_SPACE:
            probe_vel[:] = 0.0
            print("probe: stopped")
        elif keycode in speed_levels:
            cur_speed = speed_levels[keycode]
            # update velocity magnitude while keeping direction
            mag = float(np.linalg.norm(probe_vel))
            if mag > 1e-9:
                probe_vel = (cur_speed / mag) * probe_vel
            print(f"[probe] speed = {cur_speed:.3f} m/s")
        elif keycode == _KEY_ENTER:
            print(
                f">>> PROBE: center=({probe_pos[0]:.3f}, {probe_pos[1]:.3f}, {probe_pos[2]:.3f})"
            )
        elif keycode == _KEY_Q:
            pass  # viewer.is_running() will become False
        # else: unknown key, ignore

    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=key_cb,
        show_left_ui=False,
        show_right_ui=False,
    )
    # MolmoSpaces structural / collision geoms are remapped to group 0. Keep
    # them hidden by default, matching robosuite's normal mjviewer behavior.
    viewer.opt.geomgroup[0] = 0
    viewer.cam.lookat = probe_pos.copy()
    viewer.cam.distance = 3.0

    # user_scn sphere geometry
    sphere_size = np.array([0.03, 0.0, 0.0], dtype=np.float64)
    sphere_rgba = np.array([1.0, 0.0, 0.0, 0.7], dtype=np.float32)
    eye = np.eye(3, dtype=np.float64).flatten()

    try:
        while viewer.is_running():
            # --- apply continuous probe velocity ---
            now = _time.monotonic()
            dt = now - last_print
            if dt <= 0:
                dt = 1.0 / 60.0  # fallback at ~60 Hz
            probe_pos += probe_vel * dt
            last_print = now
            # throttle coordinate prints to ~3 Hz
            if float(np.linalg.norm(probe_vel)) > 1e-9 and not hasattr(
                key_cb, "_last_vel_print"
            ):
                key_cb._last_vel_print = 0.0  # type: ignore[attr-defined]
            _lvp = getattr(key_cb, "_last_vel_print", 0.0)
            if float(np.linalg.norm(probe_vel)) > 1e-9:
                if now - _lvp > 0.3:
                    print(
                        f"probe: ({probe_pos[0]:.4f}, {probe_pos[1]:.4f}, "
                        f"{probe_pos[2]:.4f})  speed={cur_speed:.3f} m/s"
                    )
                    key_cb._last_vel_print = now  # type: ignore[attr-defined]

            scn = viewer.user_scn
            scn.ngeom = 0
            if scn.maxgeom > 0:
                mujoco.mjv_initGeom(
                    scn.geoms[0],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    sphere_size,
                    probe_pos,
                    eye,
                    sphere_rgba,
                )
                scn.ngeom = 1
                # small crosshair lines for precision
                if scn.maxgeom >= 7:
                    cross_rgba = np.array([1.0, 0.3, 0.3, 0.5], dtype=np.float32)
                    cross_len = np.float64(0.05)
                    for axis in range(3):
                        for sign in (-1, 1):
                            p0 = probe_pos.copy()
                            p1 = probe_pos.copy()
                            p0[axis] -= sign * cross_len
                            p1[axis] += sign * cross_len
                            idx = 1 + axis * 2 + (0 if sign == -1 else 1)
                            mujoco.mjv_initGeom(
                                scn.geoms[idx],
                                mujoco.mjtGeom.mjGEOM_CAPSULE,
                                np.array([0.003, cross_len, 0.0], dtype=np.float64),
                                (p0 + p1) / 2.0,
                                eye,
                                cross_rgba,
                            )
                            scn.ngeom = max(scn.ngeom, idx + 1)
            viewer.sync()
            # gently follow the probe with the camera
            viewer.cam.lookat = 0.9 * viewer.cam.lookat + 0.1 * probe_pos
    finally:
        viewer.close()


def _write_robot_pose_calibration(scene_name, task_name, pos, yaw, scope):
    """Merge one G1 spawn calibration into the iTHOR YAML registry."""
    import tempfile

    import yaml

    from robocasa.models.scenes.ithor_scene_specs import _SPEC_PATH, clear_specs_cache

    try:
        with open(_SPEC_PATH, "r") as f:
            specs = yaml.safe_load(f) or {}
    except FileNotFoundError:
        specs = {}

    if scope == "task":
        target = specs.setdefault("tasks", {}).setdefault(task_name, {})
    else:
        target = specs.setdefault("default", {})
    target.setdefault(scene_name, {})["robot_pose"] = {
        "pos": [round(float(value), 4) for value in pos],
        "yaw": round(float(yaw), 5),
    }

    directory = os.path.dirname(_SPEC_PATH)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".ithor_scene_specs.", suffix=".yaml", dir=directory
    )
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(specs, f, allow_unicode=True, sort_keys=False)
        os.replace(temp_path, _SPEC_PATH)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    clear_specs_cache()
    return _SPEC_PATH


def _write_task_object_pose_calibration(scene_name, task_name, poses):
    """Persist free-joint task-prop poses under one task and one FloorPlan."""
    import tempfile

    import yaml

    from robocasa.models.scenes.ithor_scene_specs import _SPEC_PATH, clear_specs_cache

    try:
        with open(_SPEC_PATH, "r") as f:
            specs = yaml.safe_load(f) or {}
    except FileNotFoundError:
        specs = {}

    target = (
        specs.setdefault("tasks", {})
        .setdefault(task_name, {})
        .setdefault(scene_name, {})
    )
    target["task_object_poses"] = {
        name: {
            "pos": [round(float(value), 5) for value in pose[:3]],
            "quat": [round(float(value), 7) for value in pose[3:]],
        }
        for name, pose in poses.items()
    }

    directory = os.path.dirname(_SPEC_PATH)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".ithor_scene_specs.", suffix=".yaml", dir=directory
    )
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(specs, f, allow_unicode=True, sort_keys=False)
        os.replace(temp_path, _SPEC_PATH)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    clear_specs_cache()
    return _SPEC_PATH


def _write_manual_support(scene_name, support):
    """Append one scene-level support rectangle to the dedicated YAML file."""
    import tempfile

    import yaml

    from robocasa.models.scenes.ithor_support_specs import (
        _SPEC_PATH,
        clear_support_specs_cache,
    )

    try:
        with open(_SPEC_PATH, "r") as f:
            specs = yaml.safe_load(f) or {}
    except FileNotFoundError:
        specs = {}
    supports = (
        specs.setdefault("scenes", {})
        .setdefault(scene_name, {})
        .setdefault("supports", [])
    )
    support["id"] = f"support_{len(supports) + 1}"
    supports.append(support)
    directory = os.path.dirname(_SPEC_PATH)
    os.makedirs(directory, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(
        prefix=".ithor_support_specs.", suffix=".yaml", dir=directory
    )
    try:
        with os.fdopen(fd, "w") as f:
            yaml.safe_dump(specs, f, allow_unicode=True, sort_keys=False)
        os.replace(temp_path, _SPEC_PATH)
    except Exception:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    clear_support_specs_cache()
    return _SPEC_PATH, support["id"]


def run_support_calibration_viewer_loop(env):
    """Interactively author oriented support rectangles for one FloorPlan."""
    import mujoco

    env.reset()
    base = getattr(env, "unwrapped", env)
    center = np.asarray(base.robot_base_pos, dtype=float).copy()
    center[2] = 0.80
    half_xy = np.array([0.30, 0.25], dtype=float)
    yaw = 0.0
    step, yaw_step = 0.05, np.deg2rad(5.0)
    size_step = 0.05
    saved = []
    model, data = env.sim.model._model, env.sim.data._data
    print(f"[support-calibration] scene={base.scene_name}")
    print("  W/S X +/-  A/D Y +/-  R/F Z +/-  Z/X yaw +/-")
    print("  I/K local-X half-size +/-  J/L local-Y half-size +/-")
    print("  1-4 step 0.01/0.05/0.10/0.25  Enter save  Space print")

    def describe():
        print(
            "[support-calibration] "
            f"center=({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) "
            f"half_xy=({half_xy[0]:.3f}, {half_xy[1]:.3f}) yaw={yaw:.5f}"
        )

    def support_hit(point):
        geomid = np.zeros(1, dtype=np.int32)
        distance = mujoco.mj_ray(
            # Start just above the marked plane, not above upper cabinets or
            # the ceiling; this identifies the body the rectangle is meant to
            # support.
            model,
            data,
            point + np.array([0.0, 0.0, 0.20]),
            np.array([0.0, 0.0, -1.0]),
            None,
            1,
            -1,
            geomid,
        )
        if distance < 0 or geomid[0] < 0:
            return None, None
        body_id = int(model.geom_bodyid[geomid[0]])
        return (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id),
            float(point[2] + 0.20 - distance),
        )

    def validate_support():
        c, s = np.cos(yaw), np.sin(yaw)
        rot = np.array([[c, -s], [s, c]])
        points = [center]
        for sx in (-1.0, 1.0):
            for sy in (-1.0, 1.0):
                point = center.copy()
                point[:2] += rot @ (half_xy * np.array([sx, sy]))
                points.append(point)
        hits = [support_hit(point) for point in points]
        body_name = hits[0][0]
        if body_name is None or any(body != body_name for body, _ in hits):
            return None, "centre and all four corners must hit the same support body"
        if any(abs(hit_z - center[2]) > 0.03 for _, hit_z in hits):
            return None, "support height differs from the marked Z by more than 3 cm"
        return body_name, None

    def key_cb(keycode):
        nonlocal yaw, step, size_step
        if keycode == 87:
            center[0] += step
        elif keycode == 83:
            center[0] -= step
        elif keycode == 65:
            center[1] += step
        elif keycode == 68:
            center[1] -= step
        elif keycode == 82:
            center[2] += step
        elif keycode == 70:
            center[2] -= step
        elif keycode == 90:
            yaw += yaw_step
        elif keycode == 88:
            yaw -= yaw_step
        elif keycode == 73:
            half_xy[0] += size_step
        elif keycode == 75:
            half_xy[0] = max(0.02, half_xy[0] - size_step)
        elif keycode == 74:
            half_xy[1] = max(0.02, half_xy[1] - size_step)
        elif keycode == 76:
            half_xy[1] += size_step
        elif keycode in {49, 50, 51, 52}:
            step = size_step = {49: 0.01, 50: 0.05, 51: 0.10, 52: 0.25}[keycode]
            print(f"[support-calibration] step={step:.2f} m")
            return
        elif keycode == 32:
            describe()
            return
        elif keycode == 257:
            body_name, error = validate_support()
            if error is not None:
                print(f"[support-calibration] not saved: {error}")
                return
            path, support_id = _write_manual_support(
                base.scene_name,
                {
                    "center": [round(float(v), 4) for v in center],
                    "half_xy": [round(float(v), 4) for v in half_xy],
                    "yaw": round(float(yaw), 5),
                    "body_name": body_name,
                },
            )
            saved.append((center.copy(), half_xy.copy(), yaw))
            print(f"[support-calibration] saved {support_id} ({body_name}) -> {path}")
            return
        else:
            return
        describe()

    viewer = mujoco.viewer.launch_passive(
        model, data, key_callback=key_cb, show_left_ui=False, show_right_ui=False
    )
    viewer.opt.geomgroup[0] = 0
    try:
        while viewer.is_running():
            scene = viewer.user_scn
            scene.ngeom = 0
            for pos, half, saved_yaw in [*saved, (center, half_xy, yaw)]:
                if scene.ngeom >= scene.maxgeom:
                    break
                c, s = np.cos(saved_yaw), np.sin(saved_yaw)
                mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                rgba = np.array([0.2, 0.7, 1.0, 0.25], dtype=np.float32)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_BOX,
                    np.array([half[0], half[1], 0.006]),
                    pos,
                    mat.ravel(),
                    rgba,
                )
                scene.ngeom += 1
            viewer.sync()
    finally:
        viewer.close()


def save_current_task_layout(env):
    """Save all dynamic task props from the current compiled simulation.

    A task opts in by exposing ``calibration_object_names``.  This avoids
    guessing which arbitrary scene objects should be written to YAML.
    """
    base = getattr(env, "unwrapped", env)
    names = tuple(getattr(base, "calibration_object_names", ()))
    if not names:
        print(
            f"[task-layout] {type(base).__name__} does not expose calibratable task props"
        )
        return None
    poses = {}
    for name in names:
        if name == "usd_box":
            obj = getattr(base, "usd_box", None)
        else:
            obj = base.objects.get(name)
        if obj is None or not obj.joints:
            print(f"[task-layout] cannot save {name}: no free joint")
            return None
        qpos = np.asarray(base.sim.data.get_joint_qpos(obj.joints[0]), dtype=float)
        if qpos.shape != (7,):
            print(
                f"[task-layout] cannot save {name}: expected a free joint, got {qpos.shape}"
            )
            return None
        poses[name] = qpos.copy()
    path = _write_task_object_pose_calibration(
        base.scene_name, type(base).__name__, poses
    )
    print(f"[task-layout] saved {', '.join(names)} -> {path}")
    return path


def draw_support_overlay(env):
    """Draw semantic support regions and manual support regions."""
    import mujoco

    base = getattr(env, "unwrapped", env)
    renderer = getattr(base, "viewer", None)
    viewer = getattr(renderer, "viewer", None)
    if viewer is None:
        return
    try:
        regions = list(base.get_region(kind="support", return_all=True))
    except Exception:
        return
    manual_count = sum(":manual_support:" in region.fixture_id for region in regions)
    if not getattr(base, "_printed_support_overlay", False):
        print(
            "[support-overlay] "
            f"semantic={len(regions) - manual_count} "
            f"manual={manual_count} drawn={len(regions)}"
        )
        base._printed_support_overlay = True
    scene = viewer.user_scn
    scene.ngeom = 0
    for region in regions:
        if scene.ngeom >= scene.maxgeom:
            break
        if ":manual_support:" in region.fixture_id:
            rgba = np.array([0.25, 0.35, 1.0, 0.38], dtype=np.float32)
        else:
            rgba = np.array([0.0, 0.75, 1.0, 0.24], dtype=np.float32)
        center, yaw = region.world_pose(base.sim)
        pos = np.asarray(center, dtype=float).copy()
        pos[2] += max(float(region.half_size[2]), 0.003) + 0.004
        half = np.asarray(region.half_size, dtype=float)
        c, s = np.cos(yaw), np.sin(yaw)
        mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_BOX,
            np.array([half[0], half[1], 0.006], dtype=float),
            pos,
            mat.ravel(),
            rgba,
        )
        scene.ngeom += 1
    viewer.sync()


def draw_navigation_success_overlay(env, reset_scene=True):
    """Draw the target fixture area used by navigation success checks."""
    import mujoco

    base = getattr(env, "unwrapped", env)
    renderer = getattr(base, "viewer", None)
    viewer = getattr(renderer, "viewer", None)
    fixture = getattr(base, "target_fixture", None)
    radius = getattr(base, "SUCCESS_DISTANCE", None)
    if viewer is None or fixture is None or radius is None:
        return

    scene = viewer.user_scn
    if reset_scene:
        scene.ngeom = 0

    if not getattr(base, "_printed_navigation_success_overlay", False):
        print(
            "[nav-success-overlay] "
            f"fixture={fixture.name} category={fixture.category} "
            f"source={fixture.source_category} radius={float(radius):.3f}"
        )
        base._printed_navigation_success_overlay = True

    body_ids = base._fixture_body_ids(fixture, include_siblings=True)
    geom_ids = base._collidable_geoms(body_ids)
    if len(geom_ids) > 10:
        indices = np.linspace(0, len(geom_ids) - 1, 10, dtype=int)
        geom_ids = [geom_ids[index] for index in indices]
    rgba = np.array([1.0, 0.1, 0.1, 0.28], dtype=np.float32)
    model, data = base.sim.model, base.sim.data
    for geom_id in geom_ids:
        if scene.ngeom >= scene.maxgeom:
            break
        geom_type = int(model.geom_type[geom_id])
        size = np.asarray(model.geom_size[geom_id], dtype=float).copy()
        pos = np.asarray(data.geom_xpos[geom_id], dtype=float).copy()
        mat = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
            size[:3] += float(radius)
        elif geom_type in (
            int(mujoco.mjtGeom.mjGEOM_SPHERE),
            int(mujoco.mjtGeom.mjGEOM_CAPSULE),
            int(mujoco.mjtGeom.mjGEOM_CYLINDER),
        ):
            size[0] += float(radius)
        else:
            geom_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
            size = np.array([float(model.geom_rbound[geom_id]) + float(radius), 0, 0])
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            geom_type,
            size,
            pos,
            mat.ravel(),
            rgba,
        )
        scene.ngeom += 1
    viewer.sync()


def draw_pnp_receptacle_overlay(env, reset_scene=True):
    """Draw task-specific success overlays."""
    import mujoco

    base = getattr(env, "unwrapped", env)
    receptacle_bounds = (
        ("drawer-pnp-receptacle-overlay", "_drawer_receptacle_bounds"),
        ("cabinet-pnp-receptacle-overlay", "_cabinet_receptacle_bounds"),
    )
    renderer = getattr(base, "viewer", None)
    viewer = getattr(renderer, "viewer", None)
    if viewer is None:
        return

    scene = viewer.user_scn
    if reset_scene:
        scene.ngeom = 0

    for label, attr_name in receptacle_bounds:
        bounds_fn = getattr(base, attr_name, None)
        if not callable(bounds_fn):
            continue
        bounds = bounds_fn()
        if bounds is not None and scene.ngeom < scene.maxgeom:
            center, half_size, yaw = bounds
            printed_attr = f"_printed_{attr_name[1:]}"
            if not getattr(base, printed_attr, False):
                print(
                    f"[{label}] "
                    f"center={[round(float(v), 3) for v in center]} "
                    f"half_size={[round(float(v), 3) for v in half_size]}"
                )
                setattr(base, printed_attr, True)
            c, s = np.cos(float(yaw)), np.sin(float(yaw))
            mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_BOX,
                np.asarray(half_size, dtype=float),
                np.asarray(center, dtype=float),
                mat.ravel(),
                np.array([1.0, 0.05, 0.05, 0.30], dtype=np.float32),
            )
            scene.ngeom += 1

    success_bounds_fn = getattr(base, "_success_region_bounds", None)
    if callable(success_bounds_fn):
        bounds = list(success_bounds_fn())
        if not getattr(base, "_printed_success_region_overlay", False):
            print(f"[success-region-overlay] collision_aabbs={len(bounds)}")
            base._printed_success_region_overlay = True
        rgba = np.array([1.0, 0.8, 0.05, 0.22], dtype=np.float32)
        for center, half, yaw in bounds:
            if scene.ngeom >= scene.maxgeom:
                break
            c, s = np.cos(float(yaw)), np.sin(float(yaw))
            mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_BOX,
                np.asarray(half, dtype=float),
                np.asarray(center, dtype=float),
                mat.ravel(),
                rgba,
            )
            scene.ngeom += 1
    viewer.sync()


def draw_drawer_pnp_receptacle_overlay(env, reset_scene=True):
    draw_pnp_receptacle_overlay(env, reset_scene=reset_scene)


def run_spawn_calibration_viewer_loop(env, scope):
    """Interactively place G1 and persist the selected base pose to YAML.

    Returns the next downloaded scene name when the user presses N, otherwise
    returns None after the viewer closes.
    """
    import mujoco

    env.reset()
    base = getattr(env, "unwrapped", env)
    scene_name = base.scene_name
    task_name = type(base).__name__
    xy = np.asarray(base.robot_base_pos[:2], dtype=float).copy()
    yaw = float(base.robot_base_ori[2])
    move_step, yaw_step = 0.05, np.deg2rad(5.0)
    step_levels = {49: 0.01, 50: 0.05, 51: 0.10, 52: 0.25}
    scene_names = sorted(
        base._list_downloaded_scenes(),
        key=lambda name: int(name.removeprefix("FloorPlan")),
    )
    next_scene = None
    switch_requested = False
    ego_camera_active = False

    print(f"[spawn-calibration] env={task_name} scene={scene_name} scope={scope}")
    print("  W/S: X +/-   A/D: Y +/-   Z/X: yaw +/-   1-4: move step")
    print(
        "  Enter: save YAML   Space: print pose   N: next scene   H: toggle G1 head camera"
    )

    def apply_pose():
        nonlocal xy, yaw
        if not base._set_robot_free_pose(xy, yaw):
            raise RuntimeError("The selected robot has no free base joint")
        base.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        base.robot_base_ori = np.array([0.0, 0.0, yaw])

    def describe_pose():
        collision_count = base.count_body_collisions()
        print(
            "[spawn-calibration] "
            f"pos=({xy[0]:.4f}, {xy[1]:.4f}, 0.0000) yaw={yaw:.5f} "
            f"collisions={collision_count}"
        )
        return collision_count

    model, data = env.sim.model._model, env.sim.data._data
    head_camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAMERA)
    viewer = None

    # GLFW key codes. This passive viewer callback owns only calibration keys.
    def key_cb(keycode: int) -> None:
        nonlocal xy, yaw, move_step, next_scene, switch_requested, ego_camera_active
        if keycode == 87:  # W
            xy[0] += move_step
        elif keycode == 83:  # S
            xy[0] -= move_step
        elif keycode == 65:  # A
            xy[1] += move_step
        elif keycode == 68:  # D
            xy[1] -= move_step
        elif keycode == 90:  # Z
            yaw += yaw_step
        elif keycode == 88:  # X
            yaw -= yaw_step
        elif keycode in step_levels:
            move_step = step_levels[keycode]
            print(f"[spawn-calibration] move step = {move_step:.3f} m")
            return
        elif keycode == 32:  # Space
            describe_pose()
            return
        elif keycode == 257:  # Enter
            collision_count = describe_pose()
            if collision_count:
                print("[spawn-calibration] not saved: choose a collision-free pose")
                return
            path = _write_robot_pose_calibration(
                scene_name, task_name, [xy[0], xy[1], 0.0], yaw, scope
            )
            print(f"[spawn-calibration] saved -> {path}")
            return
        elif keycode == 78:  # N
            if len(scene_names) <= 1:
                print("[spawn-calibration] no other downloaded scene in this room type")
                return
            index = scene_names.index(scene_name)
            next_scene = scene_names[(index + 1) % len(scene_names)]
            switch_requested = True
            print(
                f"[spawn-calibration] switching to {next_scene}; "
                "unsaved movement in the current scene is discarded"
            )
            return
        elif keycode == 72:  # H
            if head_camera_id < 0:
                print(f"[spawn-calibration] camera {EGO_CAMERA!r} is unavailable")
                return
            ego_camera_active = not ego_camera_active
            if ego_camera_active:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = head_camera_id
                print("[spawn-calibration] G1 head-camera view")
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                print("[spawn-calibration] free scene view")
            return
        else:
            return
        apply_pose()
        describe_pose()

    viewer = mujoco.viewer.launch_passive(
        model, data, key_callback=key_cb, show_left_ui=False, show_right_ui=False
    )
    # Hide the yellow / green collision and structural geometry by default.
    viewer.opt.geomgroup[0] = 0
    try:
        while viewer.is_running():
            viewer.sync()
            if switch_requested:
                break
    finally:
        viewer.close()
    return next_scene


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", type=str, required=True)
    p.add_argument("--robot", type=str, default="G1FloatingBody")
    p.add_argument("--scene-id", type=int, default=None)
    p.add_argument("--layout", type=int, default=None)
    p.add_argument("--style", type=int, default=None)
    p.add_argument("--fixed-scene", action="store_true")
    p.add_argument("--scene-seed", type=int, default=None)
    p.add_argument(
        "--conveyor-speed",
        type=float,
        default=0.3,
        help="initial TeleLabConveyor speed in m/s (default: 0.3)",
    )
    p.add_argument(
        "--backend",
        type=str,
        default="sonic",
        choices=backend_names(),  # live registry, not a literal list -- see collect_fixed_task
    )
    p.add_argument(
        "--probe",
        action="store_true",
        help="calibration mode: keyboard-driven 3D probe for measuring "
        "world coordinates",
    )
    p.add_argument(
        "--calibrate-spawn",
        action="store_true",
        help="interactively move/rotate G1 and save robot_pose to iTHOR YAML",
    )
    p.add_argument(
        "--calibrate-task-layout",
        action="store_true",
        help="Enter saves current dynamic task-prop poses to task-scoped iTHOR YAML",
    )
    p.add_argument(
        "--calibrate-support",
        action="store_true",
        help="interactively author scene-level support rectangles",
    )
    p.add_argument(
        "--show-support",
        action="store_true",
        help="draw authored and task-merged semantic support regions in the Viewer",
    )
    p.add_argument(
        "--calibration-scope",
        choices=("task", "default"),
        default="task",
        help="saved pose applies to this task only (default) or every iTHOR task in the scene",
    )
    args = p.parse_args()

    if (
        sum(
            (
                args.probe,
                args.calibrate_spawn,
                args.calibrate_task_layout,
                args.calibrate_support,
            )
        )
        > 1
    ):
        p.error("choose only one calibration mode")

    scene_kwargs = forward_scene_kwargs(
        args.env,
        scene_id=args.scene_id,
        layout=args.layout,
        style=args.style,
        fixed_scene=args.fixed_scene,
        scene_seed=args.scene_seed,
    )
    if args.env == "TeleLabConveyor":
        scene_kwargs["conveyor_speed"] = args.conveyor_speed

    def build_env(kwargs):
        from robocasa.g1_host.bridge import load_sim2real_controller_config

        controller_config = load_sim2real_controller_config(args.backend, "mock")
        return build_teleop_env(
            args.env,
            args.robot,
            controller_config,
            kwargs,
            headless=False,
        )

    env = build_env(scene_kwargs)

    # --- probe mode: 3D calibration marker, no robot control ---
    if args.probe:
        run_probe_viewer_loop(env)
        return

    if args.calibrate_spawn:
        while True:
            next_scene = run_spawn_calibration_viewer_loop(env, args.calibration_scope)
            env.close()
            if next_scene is None:
                break
            next_scene_id = int(next_scene.removeprefix("FloorPlan"))
            next_kwargs = dict(scene_kwargs)
            next_kwargs["scene_ids"] = next_scene_id
            env = build_env(next_kwargs)
        return

    if args.calibrate_support:
        run_support_calibration_viewer_loop(env)
        env.close()
        return

    # --- interactive teleop viewer ---
    env.reset()
    base = getattr(env, "unwrapped", env)
    initial_state = env.sim.get_state()
    initial_robot_base_pos = getattr(base, "robot_base_pos", np.zeros(3)).copy()
    initial_robot_base_ori = getattr(base, "robot_base_ori", np.zeros(3)).copy()
    print(
        f"[viz] {args.env}  lang='{base.get_ep_meta().get('lang')}' — "
        "drive the G1 with the keyboard; press Q to reset; close the window to quit."
    )
    from robocasa.demos.demo_empty_teleop import run_sonic_viewer_loop

    def reset_in_place(env):
        from robocasa.scripts.collect_fixed_task import (
            _refresh_after_restore,
            _reset_controllers,
        )

        base = getattr(env, "unwrapped", env)
        _reset_controllers(base)
        base.sim.set_state(initial_state)
        base.sim.forward()
        if hasattr(base, "_reset_task_progress"):
            base._reset_task_progress()
        if hasattr(base, "robot_base_pos"):
            base.robot_base_pos = initial_robot_base_pos.copy()
        if hasattr(base, "robot_base_ori"):
            base.robot_base_ori = initial_robot_base_ori.copy()
        _refresh_after_restore(base)
        return base._get_observations(force_update=True)

    env.reset = reset_in_place.__get__(env, type(env))

    def overlay(env):
        renderer = getattr(getattr(env, "unwrapped", env), "viewer", None)
        viewer = getattr(renderer, "viewer", None)
        if viewer is not None:
            viewer.user_scn.ngeom = 0
            if args.show_support:
                draw_support_overlay(env)
                draw_navigation_success_overlay(env, reset_scene=False)
                draw_drawer_pnp_receptacle_overlay(env, reset_scene=False)
            else:
                draw_navigation_success_overlay(env, reset_scene=False)
                draw_drawer_pnp_receptacle_overlay(env, reset_scene=False)

    if args.calibrate_task_layout:
        print(
            "[task-layout] Press Enter after arranging the box and task objects to save their poses."
        )

        def save_layout_key(keycode):
            if keycode == 257:  # GLFW Enter
                save_current_task_layout(env)
                return True
            return False

        run_sonic_viewer_loop(
            env,
            recorder=None,
            show_ego=True,
            initial_reset=False,
            extra_keypress_callback=save_layout_key,
            viewer_overlay_callback=overlay,
        )
    else:
        run_sonic_viewer_loop(
            env,
            recorder=None,
            show_ego=True,
            initial_reset=False,
            viewer_overlay_callback=overlay,
        )
    env.close()


if __name__ == "__main__":
    main()
