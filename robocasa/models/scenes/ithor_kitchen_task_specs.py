"""Strict scene-first calibrations for iTHOR Kitchen tasks."""

from __future__ import annotations

import copy
import os

import yaml


_SPEC_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../assets/scenes/ithor_kitchen_task_specs.yaml",
    )
)
_CACHE = None


def _load_specs():
    global _CACHE
    if _CACHE is None:
        with open(_SPEC_PATH, "r") as file:
            _CACHE = yaml.safe_load(file) or {}
    return _CACHE


def clear_ithor_kitchen_task_specs_cache():
    """Force the next lookup to reload the Kitchen task calibration YAML."""
    global _CACHE
    _CACHE = None


def get_ithor_kitchen_task_spec(scene_name, task_name, required=()):
    """Return one explicit Kitchen scene/task calibration.

    There are deliberately no scene defaults or cross-task fallbacks. A new
    scene is usable by a task only after that exact pair has been calibrated.
    """
    scene_specs = _load_specs().get("scenes", {}).get(str(scene_name))
    if scene_specs is None or task_name not in scene_specs:
        raise KeyError(
            f"No iTHOR Kitchen calibration for {scene_name}/{task_name} "
            f"in {_SPEC_PATH}"
        )
    spec = copy.deepcopy(scene_specs[task_name])
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"Incomplete iTHOR Kitchen calibration for {scene_name}/{task_name}: "
            f"missing {missing}"
        )
    return spec


def get_ithor_kitchen_task_specs(task_name, required=()):
    """Return all explicitly calibrated scenes for one Kitchen task."""
    specs = {}
    for scene_name in _load_specs().get("scenes", {}):
        try:
            specs[scene_name] = get_ithor_kitchen_task_spec(
                scene_name, task_name, required=required
            )
        except KeyError:
            continue
    return specs


__all__ = [
    "clear_ithor_kitchen_task_specs_cache",
    "get_ithor_kitchen_task_spec",
    "get_ithor_kitchen_task_specs",
]
