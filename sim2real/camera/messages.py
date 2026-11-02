"""Wire format for the generic camera PUB/SUB channel.

One message per publish tick: ``{"timestamps": {name: float}, "images": {name: jpeg_bytes}}``.
Images are JPEG-encoded before packing to keep messages small over the ZMQ PUB socket;
decode always returns RGB ``np.ndarray`` regardless of how the source produced the frame.
"""

from __future__ import annotations

import base64

import cv2
import msgpack
import msgpack_numpy as mnp
import numpy as np

mnp.patch()


def encode_camera_frame(
    images: dict[str, np.ndarray],
    timestamps: dict[str, float],
    jpeg_quality: int = 90,
) -> bytes:
    """Pack a set of named RGB frames (+ per-camera capture timestamps) into one message."""
    encoded_images: dict[str, bytes] = {}
    for name, image in images.items():
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
        if not ok:
            raise ValueError(f"Failed to JPEG-encode camera frame {name!r}")
        encoded_images[name] = buf.tobytes()

    return msgpack.packb(
        {"timestamps": timestamps, "images": encoded_images}, use_bin_type=True
    )


def _decode_image(value: bytes | str | np.ndarray) -> np.ndarray:
    """One frame, from whichever of the three shapes the wire carries. Returns RGB.

    ``str`` is base64-encoded JPEG, which is how the G1's own onboard server sends
    frames (GR00T-WholeBodyControl's ``ImageUtils.encode_image``, port 5555) --
    they used to land here as ``TypeError: a bytes-like object is required, not
    'str'``, several layers below anything that says "camera".

    The channel order differs with the shape, and that is not a guess: this
    server's own encoder writes ``cv2.imencode(".jpg", image)`` where ``image`` has
    ALREADY been converted to RGB by ``composed_camera.read``
    (``cvtColor(frame, COLOR_BGR2RGB)``). ``imencode`` treats its input as BGR, so
    those JPEGs carry RGB in BGR's slots -- decoding them and swapping again is
    what turned skin blue in recorded episodes. Frames sent as ``bytes`` come from
    ``encode_camera_frame`` above, which converts properly, so they are ordinary
    BGR JPEGs and do need the swap.
    """
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, str):
        mat = cv2.imdecode(
            np.frombuffer(base64.b64decode(value), dtype=np.uint8), cv2.IMREAD_COLOR
        )
        if mat is None:
            raise ValueError("Failed to decode base64 camera frame")
        return mat  # already RGB in BGR's slots -- see docstring
    mat = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if mat is None:
        raise ValueError("Failed to decode camera frame bytes")
    return mat[..., ::-1]  # BGR -> RGB


def decode_camera_frame(raw: bytes) -> dict[str, dict]:
    """Inverse of ``encode_camera_frame``: returns ``{"timestamps": {...}, "images": {...}}``."""
    payload = msgpack.unpackb(raw, raw=False)
    images = {name: _decode_image(value) for name, value in payload["images"].items()}
    return {"timestamps": payload["timestamps"], "images": images}
