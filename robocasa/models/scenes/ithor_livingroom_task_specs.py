"""Strict scene-first calibrations for iTHOR living-room tasks."""

from __future__ import annotations

import copy
import os

import yaml


_SPEC_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../assets/scenes/ithor_livingroom_task_specs.yaml",
    )
)
_CACHE = None


def _load_specs():
    global _CACHE
    if _CACHE is None:
        with open(_SPEC_PATH, "r") as file:
            _CACHE = yaml.safe_load(file) or {}
    return _CACHE


def clear_ithor_livingroom_task_specs_cache():
    """Force the next lookup to reload the living-room calibration YAML."""
    global _CACHE
    _CACHE = None


def get_ithor_livingroom_task_spec(scene_name, task_name, required=()):
    """Return one explicit living-room scene/task calibration."""
    scene_specs = _load_specs().get("scenes", {}).get(str(scene_name))
    if scene_specs is None or task_name not in scene_specs:
        raise KeyError(
            f"No iTHOR living-room calibration for {scene_name}/{task_name} "
            f"in {_SPEC_PATH}"
        )
    spec = copy.deepcopy(scene_specs[task_name])
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"Incomplete iTHOR living-room calibration for "
            f"{scene_name}/{task_name}: missing {missing}"
        )
    return spec


def get_ithor_livingroom_task_specs(task_name, required=()):
    """Return all explicitly calibrated scenes for one living-room task."""

    specs = {}
    for scene_name in _load_specs().get("scenes", {}):
        try:
            specs[scene_name] = get_ithor_livingroom_task_spec(
                scene_name, task_name, required=required
            )
        except KeyError:
            continue
    return specs


__all__ = [
    "clear_ithor_livingroom_task_specs_cache",
    "get_ithor_livingroom_task_spec",
    "get_ithor_livingroom_task_specs",
]
