"""YAML-backed iTHOR scene calibration specs."""

from __future__ import annotations

import copy
import os

import numpy as np
import yaml


_MODEL_ASSET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets")
)
_SPEC_PATH = os.path.join(_MODEL_ASSET_ROOT, "scenes", "ithor_scene_specs.yaml")
_CACHE = None


def _load_specs():
    global _CACHE
    if _CACHE is None:
        try:
            with open(_SPEC_PATH, "r") as f:
                _CACHE = yaml.safe_load(f) or {}
        except FileNotFoundError:
            _CACHE = {}
    return _CACHE


def clear_specs_cache():
    """Force the next call to re-read ``ithor_scene_specs.yaml`` from disk."""
    global _CACHE
    _CACHE = None


def _merged_scene_spec(scene_name, task_name=None):
    specs = _load_specs()
    default = specs.get("default", {}).get(scene_name)
    task = (
        specs.get("tasks", {}).get(task_name, {}).get(scene_name) if task_name else None
    )
    if default is None and task is None:
        return None
    merged = copy.deepcopy(default or {})
    if task:
        merged.update(copy.deepcopy(task))
    return merged


def get_scene_surface(scene_name, task_name=None):
    spec = _merged_scene_spec(scene_name, task_name)
    if spec is None or "surface" not in spec:
        return None
    surface = spec["surface"]
    return {
        "center": np.array(surface["center"], dtype=float),
        "half_xy": np.array(surface["half_xy"], dtype=float),
    }


def get_scene_robot_pose(scene_name, task_name=None):
    spec = _merged_scene_spec(scene_name, task_name)
    if spec is None or "robot_pose" not in spec:
        return None
    pose = spec["robot_pose"]
    return np.array(pose["pos"], dtype=float), float(pose["yaw"])


def get_scene_robot_scale(scene_name, task_name=None):
    """Return the optional uniform G1 scale for a calibrated scene."""
    spec = _merged_scene_spec(scene_name, task_name)
    if spec is None:
        return 1.0
    scale = float(spec.get("robot_scale", 1.0))
    if scale <= 0.0:
        raise ValueError(f"{scene_name}: robot_scale must be positive, got {scale}")
    return scale


def get_scene_task_object_poses(scene_name, task_name=None):
    """Return manually calibrated dynamic task-prop poses, if present.

    The entries are deliberately task-scoped: a box layout for one task must
    never affect another task that happens to use the same Kitchen FloorPlan.
    Each value is a MuJoCo free-joint pose with ``pos`` (XYZ) and ``quat``
    (WXYZ).
    """
    spec = _merged_scene_spec(scene_name, task_name)
    if spec is None:
        return None
    poses = spec.get("task_object_poses")
    if not poses:
        return None
    result = {}
    for name, pose in poses.items():
        if not {"pos", "quat"}.issubset(pose):
            raise ValueError(
                f"{scene_name}/{task_name}: task pose {name!r} needs pos and quat"
            )
        pos = np.asarray(pose["pos"], dtype=float)
        quat = np.asarray(pose["quat"], dtype=float)
        if pos.shape != (3,) or quat.shape != (4,):
            raise ValueError(
                f"{scene_name}/{task_name}: task pose {name!r} must be XYZ + WXYZ"
            )
        result[name] = {"pos": pos, "quat": quat}
    return result
