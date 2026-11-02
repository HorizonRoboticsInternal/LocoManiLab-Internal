"""Camera window: see the frames, and see whether the link keeps up.

    uv run python -m g1_bridge.camera_monitor --connect tcp://127.0.0.1:5594

The heft fork's ``vla/camera_viewer.py`` in the shape this stack needs. It answers
the two questions you have at launch: *is the picture right* (you are about to
record it) and *is the link fast enough* (a camera slower than the control loop
fills a dataset with repeated frames and breaks nothing visibly).

Three display modes, because a collection box is not guaranteed a desktop:

``tk``    a native window (Tkinter + PIL). NOT cv2.imshow: the OpenCV in these
          environments is the headless build (``GUI: NONE``), where imshow exists
          and raises -- the same trap tele's own ego preview documents.
``http``  an MJPEG stream at http://<host>:<port>, for a headless box, an SSH
          session, or a second screen. Same idea as the mjviser window you
          already open for the simulator.
``none``  print-only: fps and frame size, nothing else.

``auto`` (the default) picks ``tk`` when a display is reachable, else ``http``.
Read-only either way: a plain ZMQ SUB alongside the collector's own, which costs
the publisher nothing (PUB fans out).
"""

from __future__ import annotations

import argparse
import io
import sys
import threading
import time

import numpy as np
from termcolor import colored

from sim2real.camera.client import CameraClient
from sim2real.utils.common import PORTS

DEFAULT_HTTP_PORT = 8081


class _Display:
    """Common surface: show one RGB frame, and say whether you are still alive."""

    alive = True

    def show(self, image: np.ndarray) -> None: ...

    def close(self) -> None: ...


class TkDisplay(_Display):
    """A native window, driven from the main thread's own loop."""

    def __init__(self, title: str):
        import tkinter as tk

        self._tk = tk
        self._root = tk.Tk()
        self._root.title(title)
        self._label = tk.Label(self._root)
        self._label.pack()
        self._root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._photo = None

    def _on_close(self):
        self.alive = False

    def show(self, image: np.ndarray) -> None:
        from PIL import Image, ImageTk

        self._photo = ImageTk.PhotoImage(Image.fromarray(image))
        self._label.configure(image=self._photo)
        try:
            self._root.update_idletasks()
            self._root.update()
        except Exception:  # noqa: BLE001 -- window closed under us
            self.alive = False

    def close(self) -> None:
        try:
            self._root.destroy()
        except Exception:  # noqa: BLE001
            pass


class HttpDisplay(_Display):
    """MJPEG over HTTP: open the printed URL in a browser.

    A single always-latest frame guarded by a lock, served as multipart/x-mixed-
    replace -- no buffering, so a slow viewer never delays the camera or the
    collector.
    """

    def __init__(self, port: int, title: str):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

        import cv2

        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):        # keep the pane readable
                pass

            def do_GET(self):
                if self.path not in ("/", "/stream.mjpg"):
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header(
                    "Content-Type", "multipart/x-mixed-replace; boundary=frame"
                )
                self.end_headers()
                try:
                    while outer.alive:
                        with outer._lock:
                            jpeg = outer._jpeg
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                        time.sleep(0.03)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self._cv2 = cv2
        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(
            colored(f"  {title}: http://localhost:{port}  (MJPEG stream)", "cyan"),
            flush=True,
        )

    def show(self, image: np.ndarray) -> None:
        ok, buf = self._cv2.imencode(".jpg", image[..., ::-1])   # RGB -> BGR
        if ok:
            with self._lock:
                self._jpeg = buf.tobytes()

    def close(self) -> None:
        self.alive = False
        self._server.shutdown()


def _make_display(mode: str, port: int, title: str) -> _Display | None:
    if mode == "none":
        return None
    if mode in ("auto", "tk"):
        try:
            return TkDisplay(title)
        except Exception as exc:  # noqa: BLE001 -- no display, no tkinter, ...
            if mode == "tk":
                raise
            print(
                colored(f"  no window ({exc}); serving the stream over HTTP instead", "yellow"),
                flush=True,
            )
    return HttpDisplay(port, title)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="g1_bridge.camera_monitor")
    parser.add_argument(
        "--connect",
        default=f"tcp://127.0.0.1:{PORTS['camera']}",
        help="camera server address (the robot's onboard server, or a local one).",
    )
    parser.add_argument(
        "--display",
        default="auto",
        choices=["auto", "tk", "http", "none"],
        help="auto = a native window if a display is reachable, else an MJPEG "
        "stream in the browser.",
    )
    parser.add_argument("--http-port", type=int, default=DEFAULT_HTTP_PORT)
    parser.add_argument(
        "--control-freq",
        type=float,
        default=50.0,
        help="the collector's rate; frames slower than this repeat in the dataset.",
    )
    parser.add_argument("--interval", type=float, default=2.0, help="seconds per report.")
    args = parser.parse_args(argv)

    client = CameraClient(args.connect)
    print(f"watching {args.connect} (Ctrl-C to stop)\n", flush=True)
    display = _make_display(args.display, args.http_port, f"G1 camera — {args.connect}")

    seen, last_stamp, window_start, waited = 0, None, time.monotonic(), 0.0
    shapes: dict[str, tuple] = {}
    try:
        while display is None or display.alive:
            message = client.read()
            images = (message or {}).get("images") or {}
            if images:
                name = "ego_view" if "ego_view" in images else sorted(images)[0]
                stamp = (message.get("timestamps") or {}).get(name)
                if stamp != last_stamp:
                    last_stamp = stamp
                    seen += 1
                    shapes[name] = images[name].shape
                    if display is not None:
                        display.show(np.ascontiguousarray(images[name]))
            time.sleep(0.005)

            elapsed = time.monotonic() - window_start
            if elapsed < args.interval:
                continue
            fps = seen / elapsed
            if seen == 0:
                waited += elapsed
                print(
                    colored(
                        f"  no frames for {waited:.0f}s — is the camera server "
                        f"running at {args.connect}?",
                        "red",
                    ),
                    flush=True,
                )
            else:
                waited = 0.0
                shape = ", ".join(f"{n} {tuple(s)}" for n, s in shapes.items())
                line = f"  {fps:5.1f} fps   {shape}"
                if fps < args.control_freq * 0.9:
                    dup = max(0.0, 1.0 - fps / args.control_freq)
                    line += colored(
                        f"   <- slower than the {args.control_freq:.0f} Hz control "
                        f"loop: ~{dup*100:.0f}% of recorded frames repeat",
                        "yellow",
                    )
                print(line, flush=True)
            seen, window_start = 0, time.monotonic()
    except KeyboardInterrupt:
        pass
    finally:
        if display is not None:
            display.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
