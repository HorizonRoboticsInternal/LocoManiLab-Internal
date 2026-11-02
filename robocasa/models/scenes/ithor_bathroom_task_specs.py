"""Strict YAML-backed calibrations for maintained iTHOR bathrooms."""

from __future__ import annotations

import copy
import os
import re

from robocasa.models.scenes.ithor_task_spec_utils import (
    load_scene_task_specs,
)


_SCENE_ID_RE = re.compile(r"^FloorPlan(\d+)$")
_SPEC_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../assets/scenes/ithor_bathroom_task_specs.yaml",
    )
)
_CACHE = None


def _load_specs():
    global _CACHE
    if _CACHE is None:
        _CACHE = load_scene_task_specs(_SPEC_PATH, "Bathroom")
    return _CACHE


def clear_ithor_bathroom_task_specs_cache():
    """Force the next lookup to reload the Bathroom task calibration YAML."""

    global _CACHE
    _CACHE = None


def get_ithor_bathroom_task_spec(scene_name, task_key, required=()):
    """Return one explicit scene/task calibration without fallbacks."""

    scene_specs = _load_specs().get(str(scene_name))
    if scene_specs is None or task_key not in scene_specs:
        raise KeyError(
            f"No iTHOR Bathroom calibration for {scene_name}/{task_key} "
            f"in {_SPEC_PATH}"
        )
    spec = copy.deepcopy(scene_specs[task_key])
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"Incomplete iTHOR Bathroom calibration for "
            f"{scene_name}/{task_key}: missing {missing}"
        )
    return spec


def get_supported_scene_ids(task_key):
    """Return sorted scene IDs that explicitly calibrate ``task_key``."""

    scene_ids = []
    for scene_name, tasks in _load_specs().items():
        if task_key not in tasks:
            continue
        match = _SCENE_ID_RE.fullmatch(scene_name)
        if match is None:
            raise ValueError(f"Invalid iTHOR Bathroom scene name: {scene_name}")
        scene_ids.append(int(match.group(1)))
    return tuple(sorted(scene_ids))


def get_ithor_bathroom_task_specs(task_key, required=()):
    """Return the complete explicit scene map for one task key."""

    return {
        f"FloorPlan{scene_id}": get_ithor_bathroom_task_spec(
            f"FloorPlan{scene_id}", task_key, required=required
        )
        for scene_id in get_supported_scene_ids(task_key)
    }


__all__ = [
    "clear_ithor_bathroom_task_specs_cache",
    "get_ithor_bathroom_task_spec",
    "get_ithor_bathroom_task_specs",
    "get_supported_scene_ids",
]
