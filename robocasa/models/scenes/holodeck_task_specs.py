"""Strict YAML-backed calibrations for Holodeck tasks."""

from __future__ import annotations

import copy
import os

import yaml


_SPEC_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../assets/scenes/holodeck_task_specs.yaml",
    )
)
_CACHE = None
_CACHE_MTIME_NS = None


def _load_specs():
    global _CACHE, _CACHE_MTIME_NS
    mtime_ns = os.stat(_SPEC_PATH).st_mtime_ns
    if _CACHE is None or _CACHE_MTIME_NS != mtime_ns:
        with open(_SPEC_PATH, "r") as file:
            _CACHE = yaml.safe_load(file) or {}
        _CACHE_MTIME_NS = mtime_ns
    return _CACHE


def clear_holodeck_task_specs_cache():
    """Force the next lookup to reload the YAML file."""
    global _CACHE, _CACHE_MTIME_NS
    _CACHE = None
    _CACHE_MTIME_NS = None


def get_holodeck_default_asset_layout(scene_name):
    """Return the configured default asset-layout task for a scene."""
    return (_load_specs().get("default_asset_layouts", {}).get(str(scene_name)))


def get_holodeck_task_spec(scene_name, task_name, required=()):
    """Return one explicit scene/task calibration with strict validation."""
    scene_specs = _load_specs().get("scenes", {}).get(str(scene_name))
    if scene_specs is None or task_name not in scene_specs:
        raise KeyError(
            f"No Holodeck calibration for {scene_name}/{task_name} in {_SPEC_PATH}"
        )
    spec = copy.deepcopy(scene_specs[task_name])
    # Keep the active calibration visible in teleop logs so edits to the
    # workspace YAML cannot be mistaken for changes to the baked XML scene.
    print(
        f"[holodeck] calibration {scene_name}/{task_name}: "
        f"robot_pose={spec.get('robot_pose')} "
        f"source={_SPEC_PATH}",
        flush=True,
    )
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"Incomplete Holodeck calibration for {scene_name}/{task_name}: "
            f"missing {missing}"
        )
    return spec


__all__ = [
    "clear_holodeck_task_specs_cache",
    "get_holodeck_default_asset_layout",
    "get_holodeck_task_spec",
]
