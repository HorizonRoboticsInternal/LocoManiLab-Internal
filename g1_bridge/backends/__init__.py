"""Registry of ``G1_BRIDGE`` backends.

Adding a backend is a new module here plus one line in ``_BACKEND_MODULES`` --
no edit to ``composite.py``.

The registry maps to ``"module:ClassName"`` strings resolved on first use, not
to imported classes. That is load-bearing, not stylistic: the backend modules
pull in torch, pinocchio and onnxruntime, and both
``load_sim2real_controller_config`` (which runs before any env exists) and
``g1_bridge.modality`` (which promises to import without the sim2real
environment) need to reach this module cheaply.
"""

from __future__ import annotations

import importlib
from typing import Type

from g1_bridge.backends.base import (  # noqa: F401 — re-exported
    BackendResources,
    Sim2RealBackend,
    Sim2RealContext,
    TickInput,
)

_BACKEND_MODULES: dict[str, str] = {
    "holomotion": "g1_bridge.backends.holomotion_v14:HolomotionV14Backend",
    "sonic": "g1_bridge.backends.sonic:SonicBackend",
    "amo": "g1_bridge.backends.amo:AmoBackend",
    "heft": "g1_bridge.backends.generic:GenericTrackingBackend",
    "scalebfm": "g1_bridge.backends.generic:GenericTrackingBackend",
    "almi": "g1_bridge.backends.almi:AlmiBackend",
}

_CACHE: dict[str, Type[Sim2RealBackend]] = {}


def backend_names() -> tuple[str, ...]:
    """Every registered ``--backend`` value."""
    return tuple(_BACKEND_MODULES)


def has_backend(name: str) -> bool:
    return name in _BACKEND_MODULES


def get_backend_class(name: str) -> Type[Sim2RealBackend]:
    """Resolve a backend name to its class, importing the module on first use."""
    if name in _CACHE:
        return _CACHE[name]
    try:
        target = _BACKEND_MODULES[name]
    except KeyError:
        raise ValueError(
            f"unknown sim2real backend {name!r}; registered: "
            f"{sorted(_BACKEND_MODULES)}"
        ) from None
    module_name, _, class_name = target.partition(":")
    cls = getattr(importlib.import_module(module_name), class_name)
    # name is the registry key, so one class can serve several backends (the
    # GMR-tracking family differs only by checkpoint).
    if cls.name != name:
        cls = type(f"{class_name}_{name}", (cls,), {"name": name})
    _CACHE[name] = cls
    return cls


def make_backend(name: str) -> Sim2RealBackend:
    return get_backend_class(name)()
