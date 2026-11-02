"""The collection arguments both hosts share.

Flag parity is not cosmetic here: the sim collector
(``robocasa.scripts.collect_demos_g1``) and the real one
(``g1_bridge.collect_real``) must agree on everything that decides *what ends up
in the dataset* -- which backend, which checkpoint, which PICO source, the task
prompt, the fps. Anything defined twice eventually drifts, and the drift shows up
as two datasets that look alike and are not.

argparse rather than tyro (the rest of this repo's CLIs) for exactly that reason:
the sim side is argparse, and one shared function beats two hand-synced ones.
"""

from __future__ import annotations

from g1_bridge.backends import backend_names
from g1_bridge.recording.features import FPS

#: Derived from the backend registry (the one place a backend is added), not a
#: second hand-maintained list. backends/__init__ resolves lazily, so this
#: import stays light. Kept as a module attribute because argparse `choices`
#: reads it here on both hosts.
BACKENDS = list(backend_names())


def add_shared_collection_args(parser, *, max_episodes: int = 1000):
    """Register the args that mean the same thing on every host.

    ``max_episodes`` is the one default that legitimately differs: a sim run is
    usually a batch you want to stop by itself, a teleop session on hardware ends
    when the operator says so. 0 means no limit.
    """
    parser.add_argument(
        "--backend",
        type=str,
        default="sonic",
        choices=BACKENDS,
        help="'sonic' = SONIC whole-body tracking (PLANNER mode walks via a "
        "ported kPlanner; POSE mode does upper-body SMPL/GR00T tracking). "
        "'amo' = command-velocity legs/waist + PICO 3-point arm-IK. "
        "'heft'/'scalebfm' = full-body GMR-retargeted tracking "
        "(launches a GMR subprocess).",
    )
    parser.add_argument(
        "--policy-config",
        type=str,
        default=None,
        help="explicit path to a sim2real policy.yaml, overriding --backend's "
        "default checkpoint (e.g. sonic's smpl/GMR-free variant instead of "
        "the g1 split-onnx default, a different heft variant, or scalebfm's "
        "humanoid_transformer_xl instead of _m).",
    )
    parser.add_argument(
        "--inference-backend",
        type=str,
        default=None,
        choices=["onnx-cpu", "onnx-gpu", "tensorrt", "torch-jit"],
        help="where the policy runs; overrides the controller JSON's "
        "sim2real_inference_backend (default onnx-cpu). 'onnx-gpu' needs a "
        "working CUDAExecutionProvider -- it raises at startup rather than "
        "falling back to CPU, so a failure here is loud, not silent.",
    )
    parser.add_argument(
        "--pico-source", type=str, default="real", choices=["mock", "real"]
    )
    parser.add_argument(
        "--amo-arm-source",
        type=str,
        default=None,
        choices=["gmr", "ik"],
        help="amo only: where the 8 shoulder/elbow + 6 wrist targets come "
        "from. 'gmr' retargets your 24-joint PICO body skeleton; 'ik' solves "
        "3-point (head + both controllers). Defaults to gmr with "
        "--pico-source real and ik otherwise.",
    )
    parser.add_argument(
        "--gmr-viewer",
        action="store_true",
        help="heft/scalebfm only: keep the GMR subprocess's own viser web "
        "visualizer (localhost:8080) running. Off by default -- it competes "
        "for CPU/GPU and has been observed to stall the retargeting loop.",
    )
    parser.add_argument(
        "--task-prompt",
        type=str,
        default=None,
        help="language instruction recorded with every episode.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="LeRobot dataset dir, created fresh. Use an ABSOLUTE path "
        "(a relative one silently lands in ~/.cache/huggingface/lerobot).",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=max_episodes,
        help="stop after N episodes; 0 = no limit"
        + (" (the default here)" if max_episodes == 0 else f" (default {max_episodes})"),
    )
    parser.add_argument("--control-freq", type=int, default=FPS)
    parser.add_argument(
        "--force-engage",
        action="store_true",
        help="debug-record-steps only: force ModeManager into POSE (tracking) "
        "before stepping, so a mock/no-PICO self-check still exercises the "
        "policy-driven path instead of just holding the default pose.",
    )
    parser.add_argument(
        "--debug-skip-readiness",
        action="store_true",
        help="debug-record-steps only: save frames even if no real reference "
        "(GMR payload, live PICO data, ...) ever arrived -- proves the "
        "plumbing constructs and steps without crashing, independent of "
        "whether a real GMR/PICO service is reachable.",
    )
    parser.add_argument(
        "--debug-record-steps",
        type=int,
        default=0,
        help="record N steps unconditionally then save + exit.",
    )
    return parser
