"""Generic camera server: opens named ``CameraSource``s and publishes frames over ZMQ.

Usage:
    uv run python -m sim2real.camera.server --cameras.ego_view.source realsense
    uv run python -m sim2real.camera.server --cameras.ego_view.source cv2 --cameras.ego_view.device 0

This is the supported camera publisher for ``--camera-connect``. Camera SDKs are
only needed for the selected source; the dummy source is available for CPU checks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import tyro
import zmq
from loguru import logger

from sim2real.camera.messages import encode_camera_frame
from sim2real.camera.sources import CameraSource
from sim2real.utils.common import PORTS


@dataclass
class CameraConfig:
    source: str = "realsense"
    """Camera backend name, see CameraSource.registry (e.g. 'realsense', 'cv2')."""

    device: int | str = 0
    """Only used by the 'cv2' source."""

    serial_number: str | None = None
    """Only used by the 'realsense' source, to pick a specific device."""

    width: int = 640
    height: int = 480


@dataclass
class CameraServerConfig:
    cameras: dict[str, CameraConfig] = field(default_factory=lambda: {"ego_view": CameraConfig()})
    port: int = PORTS["camera"]
    fps: float = 30.0
    jpeg_quality: int = 90


def _build_source(name: str, cfg: CameraConfig) -> CameraSource:
    source_cls = CameraSource.resolve(cfg.source)
    kwargs: dict = {"width": cfg.width, "height": cfg.height}
    if cfg.source == "realsense":
        kwargs["serial_number"] = cfg.serial_number
    elif cfg.source == "cv2":
        kwargs["device"] = cfg.device
    logger.info("Opening camera source {!r} ({}: {})", name, cfg.source, kwargs)
    return source_cls(**kwargs)


def main(config: CameraServerConfig) -> None:
    sources = {name: _build_source(name, cfg) for name, cfg in config.cameras.items()}

    context = zmq.Context.instance()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{config.port}")
    logger.info("Camera server publishing {} stream(s) on port {}", len(sources), config.port)

    period = 1.0 / config.fps
    try:
        while True:
            tick_start = time.time()
            images = {}
            timestamps = {}
            for name, source in sources.items():
                frame = source.read()
                if frame is None:
                    continue
                images[name] = frame
                timestamps[name] = tick_start
            if images:
                socket.send(encode_camera_frame(images, timestamps, config.jpeg_quality))
            elapsed = time.time() - tick_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        pass
    finally:
        for source in sources.values():
            source.close()
        socket.close(0)


if __name__ == "__main__":
    main(tyro.cli(CameraServerConfig))
