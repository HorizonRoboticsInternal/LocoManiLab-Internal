"""Strict scene-first calibrations for maintained iTHOR Bedroom tasks."""

from __future__ import annotations

import copy
import os
import re

from robocasa.models.scenes.ithor_task_spec_utils import (
    load_scene_task_specs,
)


_SPEC_PATH = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../assets/scenes/ithor_bedroom_task_specs.yaml",
    )
)
_SCENE_ID_RE = re.compile(r"^FloorPlan(\d+)$")
_CACHE = None


def _load_specs():
    global _CACHE
    if _CACHE is None:
        _CACHE = load_scene_task_specs(_SPEC_PATH, "Bedroom")
    return _CACHE


def clear_ithor_bedroom_task_specs_cache():
    """Force the next lookup to reload the Bedroom calibration YAML."""

    global _CACHE
    _CACHE = None


def get_ithor_bedroom_task_spec(scene_name, task_key, required=()):
    """Return one exact Bedroom scene/task calibration with no fallback."""

    scene = _load_specs().get(str(scene_name))
    if scene is None or task_key not in scene:
        raise KeyError(
            f"No iTHOR Bedroom calibration for {scene_name}/{task_key} "
            f"in {_SPEC_PATH}"
        )
    spec = copy.deepcopy(scene[task_key])
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(
            f"Incomplete iTHOR Bedroom calibration for {scene_name}/{task_key}: "
            f"missing {missing}"
        )
    return spec


def get_supported_scene_ids(task_key):
    """Return sorted scene IDs that explicitly calibrate ``task_key``."""

    scene_ids = []
    for scene_name, tasks in _load_specs().items():
        if task_key not in tasks:
            continue
        match = _SCENE_ID_RE.fullmatch(str(scene_name))
        if match is None:
            raise ValueError(f"Invalid iTHOR Bedroom scene name: {scene_name}")
        scene_ids.append(int(match.group(1)))
    return tuple(sorted(scene_ids))


def get_ithor_bedroom_task_specs(task_key, required=()):
    """Return the complete explicit scene map for one task key."""

    return {
        f"FloorPlan{scene_id}": get_ithor_bedroom_task_spec(
            f"FloorPlan{scene_id}", task_key, required=required
        )
        for scene_id in get_supported_scene_ids(task_key)
    }


__all__ = [
    "clear_ithor_bedroom_task_specs_cache",
    "get_ithor_bedroom_task_spec",
    "get_ithor_bedroom_task_specs",
    "get_supported_scene_ids",
]
