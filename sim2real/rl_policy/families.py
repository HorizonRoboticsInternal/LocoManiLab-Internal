"""The checkpoint families shipped in ``checkpoints/``, by name.

A *family* is one whole-body policy: its default checkpoint, how it wants its
reference motion delivered, and -- the part that matters to anything driving it
from outside -- **what command space it consumes**.

That last field is the one worth reading carefully. A policy trained on data
collected through a given family can only be deployed through a family that
consumes the same thing, because the recorded action IS that command space.
So the useful question is never "which of N policies × M families", it is
"which command space", and there are only a few:

``ref_qpos``
    A 50 Hz reference pose (root position + orientation + joint angles), the
    shared contract most of these families were adapted onto. Anything
    producing ``ref_qpos`` drives *every* family in this group interchangeably
    -- which makes it the portable choice for new data.
``smpl``
    SMPL joint positions, fed to SONIC's encoder, which produces the latent
    token internally.
``motion_token``
    SONIC's 64-D latent directly, skipping the encoder and entering at the
    decoder. SONIC-only by construction.
``velocity_cmd``
    Torso velocity + limb targets. AMO is command-driven, not motion-tracking,
    so it shares nothing with the above.
``twist_ee``
    ALMI's split command: a 4-D twist (vx, vy, wz, base_height) for the legs
    plus, for the waist and arms, a retargeted 17-D joint reference and the
    matching torso-frame end-effector pose. Also command-driven, but the upper
    half is a pose reference rather than a limb target, so it is neither
    ``velocity_cmd`` nor ``ref_qpos``.

Paths are relative to the repo root (the parent of the ``sim2real`` package);
resolve them with :func:`policy_config_path`.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyFamily:
    name: str
    #: Default policy.yaml, relative to the repo root.
    policy_config: str
    #: What an external driver must supply. See the module docstring.
    command_space: str
    #: The ``motion_backend`` its reference stream uses, or None when the
    #: family takes commands rather than reference motion.
    motion_backend: str | None
    notes: str = ""


_FAMILIES: dict[str, PolicyFamily] = {
    f.name: f
    for f in (
        PolicyFamily(
            "sonic",
            "checkpoints/sonic/low_latency/g1/policy.yaml",
            command_space="motion_token",
            motion_backend="zmq",
            notes="Split encoder/decoder ONNX. The g1 variant takes the "
            "encoder's full input. This repository tracks manifests for "
            "low_latency/g1 and the heading-normalized v1_1/g1 export; "
            "select v1_1 with an "
            "explicit policy_config or --sonic-ckpt v1_1. "
            "Its 64-D token is NOT interchangeable with this default's -- see "
            "g1_bridge/sonic_utils/ckpt.py.",
        ),
        PolicyFamily(
            "heft",
            "checkpoints/heft/pmg/policy.yaml",
            command_space="ref_qpos",
            motion_backend="zmq",
            notes="The tracked default is the PMG manifest. Use an explicit "
            "policy_config for any separately supplied compatible export.",
        ),
        PolicyFamily(
            "scalebfm",
            "checkpoints/scalebfm/humanoid_transformer_m/policy.yaml",
            command_space="ref_qpos",
            motion_backend="zmq",
            notes="The tracked default is humanoid_transformer_m. Other "
            "compatible exports require their own explicit policy_config.",
        ),
        PolicyFamily(
            "amo",
            "checkpoints/amo/policy.yaml",
            command_space="velocity_cmd",
            motion_backend=None,
            notes="TorchScript. Drives 15 joints (legs+waist); arms are set "
            "by the caller after each step.",
        ),
        PolicyFamily(
            "almi",
            "checkpoints/almi/flat_dual/policy.yaml",
            command_space="twist_ee",
            motion_backend=None,
            notes="Dual-branch ONNX from mjlab-loco: lower(12) walks on a "
            "PICO-stick twist, upper(17) tracks GMR whole-body retargeting of "
            "the operator's waist and arms. Exported from the training .pt by "
            "scripts/export_almi_onnx.py.",
        ),
    )
}


def repo_root() -> pathlib.Path:
    """The directory holding ``checkpoints/`` -- the parent of this package.

    Resolved from this module's location, never from a working directory or a
    directory name, so the checkout can live anywhere.
    """
    return pathlib.Path(__file__).resolve().parents[2]


def family_names() -> tuple[str, ...]:
    return tuple(_FAMILIES)


def get_family(name: str) -> PolicyFamily:
    try:
        return _FAMILIES[name]
    except KeyError:
        raise ValueError(
            f"unknown policy family {name!r}; known: {sorted(_FAMILIES)}"
        ) from None


def policy_config_path(name: str, policy_config: str | None = None) -> pathlib.Path:
    """Absolute path to a family's policy.yaml.

    ``policy_config`` overrides the default (a sibling checkpoint of the same
    family, or one living outside this repo) and is used as-is once expanded.
    """
    if policy_config:
        return pathlib.Path(policy_config).expanduser()
    return repo_root() / get_family(name).policy_config


def families_for_command_space(command_space: str) -> tuple[str, ...]:
    """Every family a driver emitting ``command_space`` can drive.

    The cross-product question, answered once: this is why producing
    ``ref_qpos`` is worth more than producing anything else here.
    """
    return tuple(
        n for n, f in _FAMILIES.items() if f.command_space == command_space
    )
