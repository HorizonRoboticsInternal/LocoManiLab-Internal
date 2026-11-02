"""Recorder interface + small numpy helpers shared by all recording backends.

An ``EpisodeRecorderBase`` owns whatever buffering/writing its output format needs.
``BasePolicy`` only ever calls ``append_frame()``/``toggle()``/``discard()``/``close()``
on it -- it never inspects recorder internals, so new formats (npz, LeRobot, ...) can be
added without touching the control loop.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


def record_array(values: list[Any]) -> Any:
    """Stack a list of per-step values into one array, falling back to object dtype."""
    if not values:
        return np.empty((0,), dtype=np.float32)

    if any(value is None for value in values):
        return np.asarray(values, dtype=object)

    try:
        arrays = [np.asarray(value) for value in values]
        if arrays and all(array.shape == arrays[0].shape for array in arrays):
            if arrays[0].dtype.kind in "biufc?":
                return np.stack(arrays, axis=0)
    except Exception:
        pass

    try:
        return np.asarray(values)
    except Exception:
        return np.asarray(values, dtype=object)


def copy_array(value: Any, dtype: Any | None = np.float32) -> np.ndarray:
    if dtype is None:
        return np.asarray(value).copy()
    return np.asarray(value, dtype=dtype).copy()


class EpisodeRecorderBase(ABC):
    @abstractmethod
    def append_frame(self, frame: dict[str, Any]) -> None:
        """Append one control-loop tick's frame dict to the in-progress episode."""

    def toggle(self, *, source: str) -> None:
        """Start a new episode (if idle) or stop & save the in-progress one.

        No-op by default -- backends that don't support mid-run segmentation (e.g. the
        classic single-implicit-episode npz recorder) simply ignore this.
        """
        return None

    def discard(self, *, source: str) -> None:
        """Drop the in-progress episode without saving it. No-op by default."""
        return None

    @abstractmethod
    def close(self) -> None:
        """Flush/save whatever is left and release resources. Called once on exit."""
