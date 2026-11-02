"""SONIC checkpoint (CKPT) selection — release vs low-latency vs v1.1.

The GR00T-SONIC repo ships three ONNX checkpoint variants of the whole-body
encoder/decoder:

  * ``old`` / ``release``      — the original ``model_encoder.onnx`` /
                                 ``model_decoder.onnx`` (sonic_release export).
  * ``new`` / ``low_latency``  — the low-latency export. **The default.**
  * ``v1_1``                   — the heading-normalized export
                                 (nvidia/GEAR-SONIC@main:sonic_v1_1), trained
                                 with wrist-pose augmentation for 3-point
                                 teleop and SONIC-backed VLA policies.

All live in the **sim2real repo's** checkpoint tree
(``<sim2real>/checkpoints/sonic/<variant>/g1/``), never in this repo.

The three share the SAME decoder input/output layout (994 -> 29), but their
encoder input layouts differ: release expects 1762D, low-latency 1247D and v1.1
1751D. The runtime SONIC controller inspects the loaded encoder ONNX input
dimension and selects the matching observation layout. This module is
the single source of truth for the on-disk path mapping plus a tiny helper that
rewrites the ``sonic_encoder_path`` / ``sonic_decoder_path`` fields of a loaded
WHOLE_BODY_G1 controller config so every data-collection and deployment
entrypoint can offer a uniform ``--sonic-ckpt`` switch.

Sharing a decoder *shape* is not sharing a decoder. Each variant's 64-D token
only means anything to the decoder it was exported with, and nothing checks
that at runtime — a policy trained on one variant's ``motion_token`` and
deployed against another produces a plausible action stream that tracks
nothing. Keep collection and deployment on the same variant.

The kinematic planner (``planner_sonic.onnx``) is NOT part of this switch — the
low-latency release does not change it — so ``planner_path`` is left untouched;
it ships alongside the low-latency g1 checkpoint (``SONIC_PLANNER_PATH``).

Paths are relative to the **sim2real repo root** and resolved at load time by
``g1_runtime.resolve_ckpt_path()`` (absolute paths are used as-is, otherwise
they are looked up under ``$SIM2REAL_ROOT`` / the installed sim2real package).
Nothing is ever read from a copy or symlink inside tele_humanoid_benchmark.
"""

from __future__ import annotations

from typing import Dict

# Where the sonic ONNX exports live inside the sim2real repo.
SONIC_CKPT_DIR = "checkpoints/sonic"

# Canonical ckpt name -> {encoder, decoder}, relative to the sim2real repo root.
SONIC_CKPT_PATHS: Dict[str, Dict[str, str]] = {
    "release": {
        "sonic_encoder_path": f"{SONIC_CKPT_DIR}/release/g1/model_encoder.onnx",
        "sonic_decoder_path": f"{SONIC_CKPT_DIR}/release/g1/model_decoder.onnx",
    },
    "low_latency": {
        "sonic_encoder_path": f"{SONIC_CKPT_DIR}/low_latency/g1/model_encoder.onnx",
        "sonic_decoder_path": f"{SONIC_CKPT_DIR}/low_latency/g1/model_decoder.onnx",
    },
    "v1_1": {
        "sonic_encoder_path": f"{SONIC_CKPT_DIR}/v1_1/g1/model_encoder.onnx",
        "sonic_decoder_path": f"{SONIC_CKPT_DIR}/v1_1/g1/model_decoder.onnx",
    },
}

# The kPlanner ONNX, shipped with the low-latency g1 checkpoint.
SONIC_PLANNER_PATH = f"{SONIC_CKPT_DIR}/low_latency/g1/planner_sonic.onnx"

# User-facing aliases -> canonical name. ``old``/``new`` are the friendly choices
# exposed on the command line; ``release``/``low_latency`` are accepted too.
# v1.1 has no friendly synonym -- it is a released version number, and inventing
# a third "newer" adjective would age as badly as "new" already has.
SONIC_CKPT_ALIASES: Dict[str, str] = {
    "old": "release",
    "release": "release",
    "new": "low_latency",
    "low_latency": "low_latency",
    "low-latency": "low_latency",
    "v1_1": "v1_1",
    "v1.1": "v1_1",
    "v11": "v1_1",
}

# Ordered choices for argparse (the friendly names). The default stays "new":
# every existing motion_token dataset and VLA checkpoint was produced against
# the low-latency export, and the tokens do not carry across variants, so
# changing the default would silently invalidate all of them. "old" is listed
# but not distributed -- it fails at construct time with a clear
# FileNotFoundError.
SONIC_CKPT_CHOICES = ("old", "new", "v1_1")
DEFAULT_SONIC_CKPT = "new"


def canonical_ckpt(ckpt: str) -> str:
    """Map a user-facing ckpt name/alias to its canonical key.

    Raises ``ValueError`` with the valid options on an unknown name.
    """
    key = SONIC_CKPT_ALIASES.get(str(ckpt).strip().lower())
    if key is None:
        raise ValueError(
            f"unknown SONIC ckpt {ckpt!r}; valid options: "
            f"{sorted(SONIC_CKPT_ALIASES)}"
        )
    return key


def apply_sonic_ckpt(controller_config: dict, ckpt: str) -> dict:
    """Override the SONIC encoder/decoder paths in ``controller_config`` in place.

    ``controller_config`` is a loaded WHOLE_BODY_G1 composite config dict (as
    returned by ``load_composite_controller_config``). Only configs that already
    carry ``sonic_encoder_path`` / ``sonic_decoder_path`` are modified; configs
    without those keys (e.g. a pure ALMI config) are returned unchanged so this
    is safe to call unconditionally.

    Returns the same dict for convenience.
    """
    key = canonical_ckpt(ckpt)
    paths = SONIC_CKPT_PATHS[key]
    spec = controller_config.get("composite_controller_specific_configs")
    if not isinstance(spec, dict):
        return controller_config
    for field, value in paths.items():
        if field in spec:
            spec[field] = value
    return controller_config
