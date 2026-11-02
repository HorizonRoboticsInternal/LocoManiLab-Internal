"""Scene-level, manually authored support rectangles for iTHOR rooms."""

from __future__ import annotations

import os

import numpy as np
import yaml


_SPEC_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets/scenes/ithor_support_specs.yaml")
)
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


def clear_support_specs_cache():
    global _CACHE
    _CACHE = None


def get_scene_manual_supports(scene_name):
    """Return validated scene-level support rectangles for ``scene_name``."""
    result = []
    for index, item in enumerate(
        _load_specs().get("scenes", {}).get(scene_name, {}).get("supports", ())
    ):
        center = np.asarray(item.get("center"), dtype=float)
        half_xy = np.asarray(item.get("half_xy"), dtype=float)
        if center.shape != (3,) or half_xy.shape != (2,) or np.any(half_xy <= 0.0):
            raise ValueError(f"{scene_name}: invalid manual support at index {index}")
        result.append(
            {
                "id": str(item.get("id", f"support_{index + 1}")),
                "center": center,
                "half_xy": half_xy,
                "yaw": float(item.get("yaw", 0.0)),
                "body_name": item.get("body_name"),
            }
        )
    return result
