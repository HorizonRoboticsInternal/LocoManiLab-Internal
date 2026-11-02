"""Pluggable camera capture backends for ``CameraServer``.

Mirrors the small name-registry pattern used by ``rl_policy.observations.Observation``
(``__init_subclass__`` auto-registration), but implemented locally so this package has
no dependency on ``rl_policy`` -- camera capture is meant to be usable standalone.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class CameraSource(ABC):
    """One named capture device. ``read()`` is called once per server tick."""

    registry: dict[str, type["CameraSource"]] = {}
    source_name: str | None = None

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if cls.source_name is not None:
            CameraSource.registry[cls.source_name] = cls

    @classmethod
    def resolve(cls, name: str) -> type["CameraSource"]:
        if name not in cls.registry:
            raise ValueError(
                f"Camera source {name!r} not found. Available: {sorted(cls.registry)}"
            )
        return cls.registry[name]

    @abstractmethod
    def read(self) -> np.ndarray | None:
        """Return the latest RGB frame (H, W, 3) uint8, or None if not yet available."""

    def close(self) -> None:
        return None


class DummyCameraSource(CameraSource):
    """Black frames for control-loop diagnostics without a physical camera."""

    source_name = "dummy"

    def __init__(self, width: int = 640, height: int = 480) -> None:
        self._frame = np.zeros((height, width, 3), dtype=np.uint8)

    def read(self) -> np.ndarray:
        return self._frame


class RealsenseCameraSource(CameraSource):
    """RGB stream from an Intel RealSense camera (depth is not published)."""

    source_name = "realsense"

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        serial_number: str | None = None,
    ) -> None:
        import pyrealsense2 as rs

        self._pipe = rs.pipeline()
        cfg = rs.config()
        if serial_number is not None:
            cfg.enable_device(serial_number)
        cfg.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        self._pipe.start(cfg)

    def read(self) -> np.ndarray | None:
        success, frames = self._pipe.try_wait_for_frames(timeout_ms=100)
        if not success:
            return None
        color_frame = frames.get_color_frame()
        if not color_frame:
            return None
        return np.asanyarray(color_frame.get_data())

    def close(self) -> None:
        self._pipe.stop()


class CV2CameraSource(CameraSource):
    """Any V4L2/UVC webcam reachable through OpenCV's ``VideoCapture``."""

    source_name = "cv2"

    def __init__(self, device: int | str = 0, width: int = 640, height: int = 480) -> None:
        import cv2

        self._cv2 = cv2
        self._cap = cv2.VideoCapture(device)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        ok, frame_bgr = self._cap.read()
        if not ok:
            return None
        return self._cv2.cvtColor(frame_bgr, self._cv2.COLOR_BGR2RGB)

    def close(self) -> None:
        self._cap.release()
