"""Collect fixed-task G1 teleop demos.

Usage (drives G1BridgeComposite -- sim2real is a required dependency for
collection, same as robocasa.scripts.collect_demos_g1):
    python -m robocasa.scripts.collect_fixed_task \
        --env IThorLivingRoomCloseLaptop --scene-seed 42 --scene-id 203 \
        --backend sonic --pico-source real

Data lands in ``<repo>/outputs_sim/<backend>/<env>_fixed_<timestamp>`` (the LeRobot
dataset root itself); pass an absolute ``--out`` to override.

The first reset is sampled from ``--scene-seed`` (and optional ``--scene-id``).
After that, each reset restores the fixed task state and samples a fresh valid
G1 / task-object XY jitter. The scene and task object identities stay
unchanged; object poses are perturbed slightly from the fixed baseline.
"""

import argparse
import json
import os
import pathlib
import types

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
from termcolor import colored

from robocasa.demos.demo_empty_teleop import run_sonic_viewer_loop
from robocasa.utils.scene_initial_jitter import apply_initial_pose_jitter
from robocasa.utils.g1_data_collection import (
    DEFAULT_OUT_BASE,
    EpisodeRecorder,
    _depth_millimeters,
    build_collection_info,
    build_frame,
    build_teleop_env,
    default_out_dir,
    forward_scene_kwargs,
    make_dataset,
)
from robocasa.g1_host.bridge import load_sim2real_controller_config
from g1_bridge.backends import backend_names
from g1_bridge.modality import G1_BACKEND_TELEOP_KEYS


MANIFEST_NAME = "fixed_task_manifest.json"
RESET_LOG_NAME = "fixed_task_resets.jsonl"
SIDECAR_SCHEMA_VERSION = 1


def _reset_controllers(env):
    """Reset controller-side caches without letting robot.reset change qpos."""
    for robot in env.robots:
        try:
            robot.reset(deterministic=True, rng=env.rng)
        except TypeError:
            robot.reset(deterministic=True)


def _refresh_after_restore(env):
    env.cur_time = 0
    env.timestep = 0
    env.done = False
    env._obs_cache = {}
    if hasattr(env.sim.data, "ctrl"):
        env.sim.data.ctrl[:] = 0
    for obs in env._observables.values():
        obs.reset()
    env.visualize(vis_settings={vis: False for vis in env._visualizations})
    env.update_state()
    if env.viewer is not None and env.renderer != "mujoco":
        env.viewer.reset()


def _robot_xy(env):
    joint = (
        env._get_robot_free_joint() if hasattr(env, "_get_robot_free_joint") else None
    )
    if joint is None:
        return np.zeros(2)
    return env.sim.data.get_joint_qpos(joint).copy()[:2]


def _movable_xy_poses(env):
    poses = {}
    for name, model in getattr(env, "objects", {}).items():
        joints = getattr(model, "joints", ())
        if not joints:
            continue
        poses[name] = env.sim.data.get_joint_qpos(joints[0]).copy()[:2]
    extra_joints = (
        env._initial_jitter_extra_joints()
        if hasattr(env, "_initial_jitter_extra_joints")
        else ()
    )
    for joint in extra_joints:
        if joint not in poses:
            poses[str(joint)] = env.sim.data.get_joint_qpos(joint).copy()[:2]
    return poses


def apply_reset_pose_jitter(
    env,
    g1_jitter_radius=0.0,
    object_jitter_radius=0.0,
    rng=None,
):
    """Apply reset-time XY jitter through the shared scene jitter utility."""
    before_robot = _robot_xy(env)
    before_objects = _movable_xy_poses(env)
    rng = np.random.default_rng() if rng is None else rng
    robot_ok = True
    object_ok = True
    if g1_jitter_radius > 0:
        robot_ok = apply_initial_pose_jitter(
            env,
            robot_radius=float(g1_jitter_radius),
            object_radius=0.0,
            rng=rng,
            include_robot=True,
            include_objects=False,
        )
    if object_jitter_radius > 0:
        object_ok = apply_initial_pose_jitter(
            env,
            robot_radius=0.0,
            object_radius=float(object_jitter_radius),
            rng=rng,
            include_robot=False,
            include_objects=True,
        )
    after_objects = _movable_xy_poses(env)
    return {
        "ok": bool(robot_ok and object_ok),
        "g1_ok": bool(robot_ok),
        "objects_ok": bool(object_ok),
        "g1": _robot_xy(env) - before_robot,
        "objects": {
            name: after_objects[name] - before_xy
            for name, before_xy in before_objects.items()
            if name in after_objects
        },
    }


def _supports_initial_pose_jitter(env):
    required = (
        "_get_robot_free_joint",
        "_initial_jitter_extra_joints",
        "_constrain_initial_jitter_candidate",
        "_initial_jitter_candidate_is_valid",
        "_semantic_placements_are_valid",
        "_body_tree_ids",
        "count_body_collisions",
    )
    return all(hasattr(env, name) for name in required)


def freeze_resets(
    env,
    g1_jitter_radius=0.0,
    object_jitter_radius=0.0,
    jitter_rng=None,
):
    """Restore the fixed task state on reset, then sample fresh XY jitter."""
    frozen_state = env.sim.get_state()
    frozen_robot_base_pos = getattr(env, "robot_base_pos", np.zeros(3)).copy()
    frozen_robot_base_ori = getattr(env, "robot_base_ori", np.zeros(3)).copy()
    original_reset = env.reset

    def fixed_reset(self):
        _reset_controllers(self)
        self.sim.set_state(frozen_state)
        self.sim.forward()
        if hasattr(self, "_reset_task_progress"):
            self._reset_task_progress()
        if hasattr(self, "robot_base_pos"):
            self.robot_base_pos = frozen_robot_base_pos.copy()
        if hasattr(self, "robot_base_ori"):
            self.robot_base_ori = frozen_robot_base_ori.copy()
        self._fixed_task_jitter = apply_reset_pose_jitter(
            self,
            g1_jitter_radius=float(g1_jitter_radius),
            object_jitter_radius=float(object_jitter_radius),
            rng=jitter_rng,
        )
        self._fixed_task_reset_count = getattr(self, "_fixed_task_reset_count", 0) + 1
        jitter = self._fixed_task_jitter.get("g1", np.zeros(2))
        status = "ok" if self._fixed_task_jitter.get("g1_ok", False) else "fallback"
        print(
            colored(
                f"[fixed-viz] reset #{self._fixed_task_reset_count}: "
                f"G1 XY jitter [{jitter[0]:+.3f}, {jitter[1]:+.3f}] m ({status})",
                "green" if status == "ok" else "yellow",
            ),
            flush=True,
        )
        if hasattr(self, "_fixed_task_reset_logger"):
            self._fixed_task_reset_logger(self)
        _refresh_after_restore(self)
        getter = (
            self.viewer._get_observations
            if getattr(self, "viewer_get_obs", False)
            else self._get_observations
        )
        return getter(force_update=True)

    env._unfrozen_reset = original_reset
    env.reset = types.MethodType(fixed_reset, env)
    return env


def build_controller_config(args):
    """The assembled G1_BRIDGE controller config.

    Built by the caller rather than inside ``build_fixed_env`` so the dataset --
    which must exist before robosuite.make starts the PICO SDK, see main() -- can
    record what actually ran in its ``collection_info``. This only assembles dicts;
    it touches neither MuJoCo nor the SDK.
    """
    return load_sim2real_controller_config(
        args.backend,
        args.pico_source,
        policy_config=args.policy_config,
        gmr_viewer=args.gmr_viewer,
        inference_backend=args.inference_backend,
    )


def build_fixed_env(args, controller_config=None):
    scene_kwargs = forward_scene_kwargs(
        args.env,
        scene_id=args.scene_id,
        layout=args.layout,
        style=args.style,
        robot_scale=args.robot_scale,
        fixed_scene=True,
        scene_seed=args.scene_seed,
    )
    controller_config = controller_config or build_controller_config(args)
    env = build_teleop_env(
        args.env,
        args.robot,
        controller_config,
        scene_kwargs,
        headless=args.no_viewer,
        control_freq=args.control_freq,
    )
    # No env.reset() before freeze_resets(): MujocoEnv.__init__ already ran
    # _load_model() + _reset_internal(), so the fixed task pose is sampled and
    # the sim state is snapshot-ready. A reset here would hard-destroy and
    # re-_load_model() -- with --scene-seed that re-seeds ithor_base's rng and
    # redraws the identical scene (a wasted reload), and without one it draws a
    # *different* scene than the one just built.
    jitter_rng = (
        None
        if args.g1_jitter_seed is None
        else np.random.default_rng(args.g1_jitter_seed)
    )
    jitter_radius = 0.0 if args.no_g1_jitter else float(args.g1_jitter_radius)
    object_jitter_radius = (
        0.0 if args.no_object_jitter else float(args.object_jitter_radius)
    )
    if (
        jitter_radius > 0 or object_jitter_radius > 0
    ) and not _supports_initial_pose_jitter(env):
        print(
            colored(
                "[fixed-viz] reset-time G1/object jitter disabled: "
                f"{args.env} does not implement the initial-jitter protocol.",
                "yellow",
            ),
            flush=True,
        )
        jitter_radius = 0.0
        object_jitter_radius = 0.0
    # What jitter this run ACTUALLY has, after the protocol check above may have
    # zeroed it. The sidecars and the dataset's collection_info both read this
    # rather than the requested radii, so neither claims jitter that never happened.
    env._fixed_task_jitter_config = {
        "g1_radius": jitter_radius,
        "object_radius": object_jitter_radius,
        "seed": args.g1_jitter_seed,
    }
    freeze_resets(
        env,
        g1_jitter_radius=jitter_radius,
        object_jitter_radius=object_jitter_radius,
        jitter_rng=jitter_rng,
    )
    return env


def _collect_out_root(args):
    if args.out:
        return pathlib.Path(args.out)
    return default_out_dir(args.env, args.backend, suffix="_fixed")


def jitter_settings(args, env=None):
    """``{g1_radius, object_radius, seed}`` -- effective when *env* is built, else asked.

    ``build_fixed_env`` zeroes both radii when the env does not implement the
    initial-jitter protocol and stashes the result on the env, so anything written
    after that point passes the env and reports what really happens.
    """
    effective = getattr(env, "_fixed_task_jitter_config", None)
    if effective:
        return dict(effective)
    return {
        "g1_radius": 0.0 if args.no_g1_jitter else float(args.g1_jitter_radius),
        "object_radius": (
            0.0 if args.no_object_jitter else float(args.object_jitter_radius)
        ),
        "seed": args.g1_jitter_seed,
    }


def fixed_task_collection_info(args, controller_config):
    """``collection_info`` for a fixed-task run: the shared keys plus this mode's own.

    The shared half (backend / pico_source / resolved policy_config / amo arm source)
    is exactly what ``collect_demos_g1`` records -- same function, so the two
    collectors cannot drift into describing themselves differently. On top of it go
    the settings that make a fixed-task dataset what it is: the scene it is pinned to
    and how far each reset is allowed to move things. Without them a merged dataset
    cannot tell "the same scene 200 times" from "200 sampled scenes", which is the
    difference between a task a policy can overfit and one it has to generalise over.

    The jitter radii here are the REQUESTED ones -- the env may still refuse
    (``_supports_initial_pose_jitter``), and ``record_effective_jitter`` corrects them
    once it has been built. Both cannot happen in one step: the dataset has to exist
    before robosuite.make (PICO SDK ordering, see make_dataset's docstring).
    """
    jitter = jitter_settings(args)
    info = build_collection_info(args, controller_config)
    info.update(
        {
            "fixed_task": True,
            "scene_id": args.scene_id,
            "scene_seed": args.scene_seed,
            "g1_jitter_radius": jitter["g1_radius"],
            "object_jitter_radius": jitter["object_radius"],
            "jitter_seed": jitter["seed"],
        }
    )
    return info


def record_effective_jitter(dataset, env):
    """Rewrite collection_info's jitter radii with the ones the env actually applied.

    A no-op in the normal case; it only bites when the env does not implement the
    initial-jitter protocol, where build_fixed_env silently zeroes both radii.
    """
    info = dataset.meta.info.get("collection_info")
    if not getattr(env, "_fixed_task_jitter_config", None) or not info:
        return
    effective = jitter_settings(args=None, env=env)
    updated = {
        "g1_jitter_radius": effective["g1_radius"],
        "object_jitter_radius": effective["object_radius"],
    }
    if all(info.get(k) == v for k, v in updated.items()):
        return
    from lerobot.datasets.utils import write_info

    info.update(updated)
    # Into meta.info and back through write_info: LeRobot rewrites info.json wholesale
    # from meta.info later, so a direct file write here would be clobbered.
    write_info(dataset.meta.info, dataset.meta.root)
    print(
        colored(f"[fixed-viz] collection_info jitter corrected to {updated}", "yellow"),
        flush=True,
    )


def _make_collect_dataset(args, controller_config):
    out_root = _collect_out_root(args)
    teleop_keys = G1_BACKEND_TELEOP_KEYS[args.backend]
    dataset = make_dataset(
        out_root,
        teleop_keys,
        fps=args.control_freq,
        collection_info=fixed_task_collection_info(args, controller_config),
    )
    return dataset, out_root


def _make_recorder(args, env, dataset):
    base = getattr(env, "unwrapped", env)
    lang = args.task_prompt or base.get_ep_meta().get("lang", args.env)
    teleop_keys = G1_BACKEND_TELEOP_KEYS[args.backend]
    return EpisodeRecorder(
        dataset,
        lang,
        teleop_keys,
        max_episodes=args.max_episodes,
        # Lets a backend take its per-episode origin (heft's episode-relative
        # reference root). Late-bound through the env: robosuite rebuilds the
        # composite on every env.reset(), so a reference captured now goes stale.
        on_episode_start=lambda: _session_of(env).on_episode_start(),
    )


def _session_of(env):
    """The live TeleopSession behind this env's composite controller."""
    return env.robots[0].composite_controller.session


def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


def _fixed_task_reset_record(args, env):
    base = getattr(env, "unwrapped", env)
    jitter = getattr(env, "_fixed_task_jitter", {})
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "reset_index": int(getattr(env, "_fixed_task_reset_count", 0)),
        "reason": getattr(env, "_fixed_task_reset_reason", "unknown"),
        "prev_episode_frames": getattr(
            env, "_fixed_task_reset_prev_episode_frames", None
        ),
        "saved_episodes": getattr(env, "_fixed_task_reset_saved_episodes", None),
        "scene_name": getattr(base, "scene_name", None),
        "scene_id": args.scene_id,
        "scene_seed": args.scene_seed,
        "g1_ok": bool(jitter.get("g1_ok", True)),
        "objects_ok": bool(jitter.get("objects_ok", True)),
        "g1_xy_jitter": _jsonable(jitter.get("g1", np.zeros(2))),
        "object_xy_jitter": _jsonable(jitter.get("objects", {})),
    }


def _mark_reset_context(env, reason, recorder=None):
    env._fixed_task_reset_reason = reason
    env._fixed_task_reset_prev_episode_frames = (
        0 if recorder is None else recorder.ep_frames
    )
    env._fixed_task_reset_saved_episodes = 0 if recorder is None else recorder.n_saved


def _install_reset_logger(args, env, out_root):
    path = pathlib.Path(out_root) / RESET_LOG_NAME

    def log_reset(current_env):
        with path.open("a") as f:
            f.write(json.dumps(_fixed_task_reset_record(args, current_env)) + "\n")

    env._fixed_task_reset_logger = log_reset
    _mark_reset_context(env, "initial", recorder=None)
    log_reset(env)
    print(colored(f"[fixed-viz] reset log at {path}", "green"), flush=True)
    return path


def _write_collect_manifest(args, env, out_root):
    base = getattr(env, "unwrapped", env)
    ep_meta = base.get_ep_meta() if hasattr(base, "get_ep_meta") else {}
    target_meta = {
        key: value for key, value in ep_meta.items() if key.startswith("target_")
    }
    manifest = {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "env": args.env,
        "files": {
            # the LeRobot dataset is this directory itself (no "lerobot/" level)
            "lerobot": ".",
            "manifest": MANIFEST_NAME,
            "reset_log": RESET_LOG_NAME,
        },
        "scene": {
            "name": getattr(base, "scene_name", None),
            "scene_id": args.scene_id,
            "scene_seed": args.scene_seed,
            "layout": args.layout,
            "style": args.style,
            "fixed_scene": True,
        },
        "robot": {
            "name": args.robot,
            "scale": args.robot_scale,
            "backend": args.backend,
            "pico_source": args.pico_source,
            "policy_config": args.policy_config,
        },
        # the radii this run actually applies, not the requested ones, so the manifest
        # and the dataset's collection_info agree -- two records of one setting that
        # disagree is worse than one record.
        "jitter": {**jitter_settings(args, env), "per_reset": True},
        "task": {
            "prompt": args.task_prompt or ep_meta.get("lang", args.env),
            "ep_meta": _jsonable(ep_meta),
            "target_meta": _jsonable(target_meta),
            "objects": sorted(getattr(base, "objects", {}).keys()),
            "jittered_objects": sorted(_movable_xy_poses(base).keys()),
        },
    }
    path = pathlib.Path(out_root) / MANIFEST_NAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(colored(f"[fixed-viz] manifest at {path}", "green"), flush=True)


def _debug_add_frames(env, recorder, steps):
    composite = env.robots[0].composite_controller
    zero = np.zeros(env.action_dim)
    n = 0
    for _ in range(steps):
        obs, _, _, _ = env.step(zero)
        teleop = composite.get_teleop_data_sample()
        frame = build_frame(
            obs,
            teleop,
            reward=0.0,
            done=False,
            task_idx=0,
            teleop_keys=recorder.teleop_keys,
            composite=composite,
        )
        if frame is not None:
            if recorder.add_frame(
                frame,
                task=recorder.lang,
                depth_mm=_depth_millimeters(obs, env.sim),
            ):
                n += 1
    return n


def _debug_record_fixed_episode(args, env, recorder):
    print(
        colored(
            f"[fixed-viz] debug: recording {args.debug_record_steps} steps", "yellow"
        )
    )
    n = _debug_add_frames(env, recorder, args.debug_record_steps)
    if n > 0:
        recorder.save(reason="debug")
        print(colored(f"[fixed-viz] debug episode saved ({n} frames).", "green"))
        _mark_reset_context(env, "success", recorder=recorder)
        env.reset()
        recorder.reset_episode()
    else:
        print(colored("[fixed-viz] debug: no frames recorded.", "red"))
    recorder.close()


def _debug_discard_fixed_episode(args, env, recorder):
    steps = args.debug_discard_after_steps
    print(colored(f"[fixed-viz] debug: recording {steps} steps then discard", "yellow"))
    n = _debug_add_frames(env, recorder, steps)
    _mark_reset_context(env, "discard", recorder=recorder)
    recorder.discard()
    env.reset()
    recorder.reset_episode()
    print(colored(f"[fixed-viz] debug discarded ({n} frames).", "yellow"))
    recorder.close()


def _check_fixed_resets(args, env):
    base = getattr(env, "unwrapped", env)
    scene_name = getattr(base, "scene_name", None)
    g1_offsets = []
    g1_ok_flags = []
    object_offsets = {}
    object_ok_flags = []
    object_keys = None
    for _ in range(args.check_resets):
        env.reset()
        current_scene = getattr(base, "scene_name", None)
        if current_scene != scene_name:
            raise AssertionError(
                f"scene changed across reset: {scene_name} -> {current_scene}"
            )
        jitter = getattr(env, "_fixed_task_jitter", {})
        offset = np.asarray(jitter.get("g1", np.zeros(2)), dtype=float)
        g1_offsets.append(offset)
        g1_ok_flags.append(bool(jitter.get("g1_ok", False)))
        objects = {
            name: np.asarray(value, dtype=float)
            for name, value in jitter.get("objects", {}).items()
        }
        keys = set(objects)
        if object_keys is None:
            object_keys = keys
        elif keys != object_keys:
            raise AssertionError(
                f"task object identities changed across reset: "
                f"{sorted(object_keys)} -> {sorted(keys)}"
            )
        for name, value in objects.items():
            object_offsets.setdefault(name, []).append(value)
        object_ok_flags.append(bool(jitter.get("objects_ok", False)))

    radius = 0.0 if args.no_g1_jitter else float(args.g1_jitter_radius)
    g1_norms = np.array([np.linalg.norm(v) for v in g1_offsets], dtype=float)
    if radius > 0 and np.any(g1_norms > radius + 1e-6):
        raise AssertionError(
            f"G1 jitter exceeded radius {radius}: max {float(g1_norms.max())}"
        )
    if radius > 0 and len(g1_offsets) > 1:
        unique = {tuple(np.round(v, 4)) for v in g1_offsets}
        if len(unique) <= 1:
            raise AssertionError("G1 jitter did not change across resets")
    if radius > 0 and not all(g1_ok_flags):
        raise AssertionError("at least one G1 jitter reset fell back")

    object_radius = 0.0 if args.no_object_jitter else float(args.object_jitter_radius)
    varied_objects = []
    if object_radius > 0 and object_offsets:
        if not all(object_ok_flags):
            raise AssertionError("at least one task-object jitter reset fell back")
        for name, values in object_offsets.items():
            norms = np.array([np.linalg.norm(v) for v in values], dtype=float)
            if np.any(norms > object_radius + 1e-6):
                raise AssertionError(
                    f"object jitter exceeded radius {object_radius} for {name}: "
                    f"max {float(norms.max())}"
                )
            if len({tuple(np.round(v, 4)) for v in values}) > 1:
                varied_objects.append(name)
        if not varied_objects and len(g1_offsets) > 1:
            raise AssertionError("task-object jitter did not change across resets")

    text = ", ".join(f"[{v[0]:+.3f},{v[1]:+.3f}]" for v in g1_offsets)
    object_text = ""
    if object_offsets:
        object_text = f", objects={len(object_offsets)}"
        if varied_objects:
            object_text += f", varied={len(varied_objects)}"
    print(
        colored(
            f"[fixed-viz] check passed: scene={scene_name}, "
            f"{len(g1_offsets)} reset(s), G1 offsets {text}{object_text}",
            "green",
        )
    )
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Collect fixed-task demos with scene and initial task state fixed."
    )
    parser.add_argument("--env", required=True, help="registered robosuite env name")
    parser.add_argument("--scene-seed", type=int, default=0)
    parser.add_argument("--scene-id", type=int, default=None)
    parser.add_argument("--layout", type=int, default=None)
    parser.add_argument("--style", type=int, default=None)
    parser.add_argument("--robot", type=str, default="G1FloatingBody")
    parser.add_argument("--robot-scale", type=float, default=None)
    parser.add_argument(
        "--g1-jitter-radius",
        type=float,
        default=0.05,
        help="sample a valid G1 root XY offset within this radius on every reset",
    )
    parser.add_argument(
        "--g1-jitter-seed",
        type=int,
        default=None,
        help="optional seed for reproducing reset-time G1/object jitter; scene-seed does not control it",
    )
    parser.add_argument(
        "--no-g1-jitter",
        action="store_true",
        help="disable the default 5cm G1 root XY jitter",
    )
    parser.add_argument(
        "--object-jitter-radius",
        type=float,
        default=0.02,
        help="sample valid task-object XY offsets within this radius on every reset",
    )
    parser.add_argument(
        "--no-object-jitter",
        action="store_true",
        help="disable the default 2cm task-object XY jitter",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="sonic",
        # the live registry, never a literal list: this script's --backend is a
        # second definition of the one in g1_bridge.recording.cli (which the
        # non-fixed collector and the real-robot collector share), and a
        # hand-copied choices list is exactly how it silently fell a backend
        # behind -- 'almi' was registered and runnable everywhere else while
        # this parser still rejected it.
        choices=backend_names(),
        help="drives G1BridgeComposite (robocasa/g1_host/bridge.py). "
        "'sonic' PLANNER mode walks via a ported kPlanner; POSE mode does "
        "upper-body SMPL/GR00T tracking.",
    )
    parser.add_argument(
        "--pico-source",
        type=str,
        default="real",
        choices=["mock", "real"],
        help="PICO data source for pico-* control modes: real device or mock (no hardware)",
    )
    parser.add_argument(
        "--policy-config",
        "--sonic-ckpt",
        dest="policy_config",
        type=str,
        default=None,
        help="explicit path to a sim2real policy.yaml, overriding --backend's default checkpoint.",
    )
    parser.add_argument(
        "--inference-backend",
        type=str,
        default=None,
        choices=["onnx-cpu", "onnx-gpu", "tensorrt", "torch-jit"],
        help="where the policy runs; overrides the controller JSON's "
        "sim2real_inference_backend (default onnx-cpu).",
    )
    parser.add_argument(
        "--gmr-viewer",
        action="store_true",
        help="heft/scalebfm only: keep the GMR subprocess's own viser web visualizer running.",
    )
    parser.add_argument("--control-freq", type=int, default=50)
    parser.add_argument(
        "--collect",
        action="store_true",
        default=True,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--visualize-only",
        dest="collect",
        action="store_false",
        help="do not record data; only run the fixed-task viewer",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="LeRobot dataset dir for collected data (default "
        f"{DEFAULT_OUT_BASE}/<backend>/<env>_fixed_<ts>)",
    )
    parser.add_argument("--max-episodes", type=int, default=1000)
    parser.add_argument(
        "--task-prompt",
        type=str,
        default=None,
        help="language instruction for collected episodes; defaults to env ep_meta",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="headless mode; mainly for --collect --debug-record-steps",
    )
    parser.add_argument(
        "--debug-record-steps",
        type=int,
        default=0,
        help="with --collect, record N zero-action steps, save once, and exit",
    )
    parser.add_argument(
        "--debug-discard-after-steps",
        type=int,
        default=0,
        help="with --collect, record N zero-action steps, discard once, reset, and exit",
    )
    parser.add_argument(
        "--check-resets",
        type=int,
        default=0,
        help="headless smoke check: reset N times and assert scene stays fixed while G1 jitter changes",
    )
    parser.add_argument(
        "--no-show-camera",
        action="store_true",
        help="do not open the live ego-camera preview window",
    )
    args = parser.parse_args()
    debug_modes = int(args.debug_record_steps > 0) + int(
        args.debug_discard_after_steps > 0
    )
    if debug_modes > 1:
        parser.error(
            "choose only one of --debug-record-steps or --debug-discard-after-steps"
        )
    if debug_modes > 0 and not args.collect:
        parser.error(
            "--debug-record-steps / --debug-discard-after-steps need collection enabled"
        )
    if args.check_resets > 0 or debug_modes > 0:
        args.no_viewer = True
    if args.check_resets > 0:
        args.collect = False

    dataset = None
    collect_out_root = None
    # Assembled here, before the dataset: it is what the dataset records as
    # collection_info, and build_fixed_env takes it rather than rebuilding it.
    controller_config = build_controller_config(args)
    if args.collect:
        dataset, collect_out_root = _make_collect_dataset(args, controller_config)
    env = build_fixed_env(args, controller_config)
    if dataset is not None:
        record_effective_jitter(dataset, env)
    base = getattr(env, "unwrapped", env)
    print(
        colored(
            f"[fixed-viz] env={args.env} scene={getattr(base, 'scene_name', '?')} "
            f"seed={args.scene_seed} scene_id={args.scene_id}",
            "green",
        )
    )
    jitter = getattr(env, "_fixed_task_jitter", {})
    g1_jitter = jitter.get("g1", np.zeros(2))
    print(
        colored(
            f"[fixed-viz] G1 XY jitter = [{g1_jitter[0]:+.3f}, {g1_jitter[1]:+.3f}] m",
            "green",
        )
    )
    object_jitters = jitter.get("objects", {})
    if object_jitters:
        text = ", ".join(
            f"{name}=[{offset[0]:+.3f},{offset[1]:+.3f}]"
            for name, offset in sorted(object_jitters.items())
        )
        print(colored(f"[fixed-viz] object XY jitter = {text} m", "green"))
    print(
        colored(
            "[fixed-viz] Q restores the exact first sampled task state: scene, "
            "task identities stay fixed; G1 and object XY jitter are resampled.",
            "green",
        )
    )
    if args.check_resets > 0:
        _check_fixed_resets(args, env)
        return
    if collect_out_root is not None:
        _write_collect_manifest(args, env, collect_out_root)
        _install_reset_logger(args, env, collect_out_root)
    recorder = _make_recorder(args, env, dataset) if dataset is not None else None
    if args.debug_record_steps > 0:
        _debug_record_fixed_episode(args, env, recorder)
        return
    if args.debug_discard_after_steps > 0:
        _debug_discard_fixed_episode(args, env, recorder)
        return
    if args.collect:
        print(
            colored(
                "[fixed-viz] collect mode: success saves; Q discards current "
                "episode and resets with fresh G1/object jitter.",
                "green",
            )
        )
    scene_keypress = getattr(env, "handle_scene_keypress", None)
    if hasattr(env, "scene_control_help"):
        print(env.scene_control_help())
    run_sonic_viewer_loop(
        env,
        recorder=recorder,
        show_ego=not args.no_show_camera,
        initial_reset=False,
        extra_keypress_callback=scene_keypress,
        reset_callback=_mark_reset_context if args.collect else None,
    )


if __name__ == "__main__":
    main()
