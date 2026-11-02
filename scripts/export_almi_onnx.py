#!/usr/bin/env python3
"""Export an ALMI (mjlab-loco dual-policy) checkpoint to the two ONNX branches.

ALMI trains two independent rsl-rl actors that run in parallel every control
tick -- ``lower`` walks (12 leg joints) and ``upper`` drives waist + arms (17
joints) -- and saves them in one ``model_*.pt``. Two observation layouts have
shipped (see ``sim2real.rl_policy.observations.almi.LAYOUT_DIMS``), and the
checkpoint's own tensor shapes are what say which one it is::

    whole_body                                    half_body
    lower_actor_state_dict:
      obs_normalizer._mean/_std   (1, 100)        (1, 46)
      encoder_mlps.…history.{0,2} 500 -> 128 -> 64  230 -> 128 -> 64
      mlp.{0,2,4,6}               164 -> 512 -> 256 -> 128 -> 12
                                                  110 -> 512 -> 256 -> 128 -> 12
    upper_actor_state_dict:
      obs_normalizer._mean/_std   (1, 144)        (1, 105)
      mlp.{0,2,4,6}               144 -> …  -> 17   105 -> …  -> 17

This script does not take the layout as an argument: it reads the widths off the
checkpoint, matches them against ``LAYOUT_DIMS``, and prints the name to put in
the new ``policy.yaml``. A checkpoint matching no known layout is refused rather
than exported into something that would load and then be wrong.

There is no state-dict path in ``sim2real.rl_policy.inference`` -- every runtime
there loads ONNX or TorchScript -- so the checkpoint has to become ONNX before
the ALMI backend can run it. mjlab's own exporter is not usable here: it needs a
live env, and on this dual config ``get_base_metadata`` raises inside a
``try/except`` so the files it does write carry no metadata.

The forward pass reimplemented below is the one in mjlab's
``EncoderMLPModel.get_latent`` (only the flat 1-D group is normalized; the
history group bypasses the normalizer and goes straight into its encoder), and
matches ``deploy_mujoco/policies.py:_ActorState.__call__`` line for line. The
history input is exported **flattened** (5 x lower_obs), not as ``(1, 5, D)``
like mjlab exports it: ``sim2real.rl_policy.inference.onnx_module.ONNXModule``
reshapes every input to ``(1, -1)``, so a 3-D input would not survive the trip.

Usage::

    python -m scripts.export_almi_onnx \
        --checkpoint /path/to/model_10000.pt \
        --output-dir checkpoints/almi/flat_dual
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn

# Safe at module scope: sim2real wants torch imported first (see
# g1_bridge/_cuda_order.py) and the lines above have just done that. Imported
# rather than copied so the layout table has exactly one definition -- a second
# copy here is the drift that produces an ONNX the observation code cannot feed.
from sim2real.rl_policy.observations.almi import LAYOUT_DIMS

# EmpiricalNormalization's epsilon (rsl_rl/modules/normalization.py): the
# denominator is (std + EPS), not std, and 1e-2 is large enough that getting it
# wrong is visible in the action, not just in the last decimal.
NORMALIZER_EPS = 1.0e-2

LOWER_HISTORY_STEPS = 5
LOWER_ACTION_DIM = 12
UPPER_ACTION_DIM = 17

LOWER_INPUT_NAMES = ["obs", "lower_actor_history"]
UPPER_INPUT_NAMES = ["obs"]
OUTPUT_NAMES = ["action"]


def _linear_layer_indices(state: dict[str, torch.Tensor], prefix: str) -> list[int]:
    """Sorted indices of the ``nn.Linear`` layers under ``prefix``.

    rsl-rl's MLP interleaves Linear and ELU, so the indices are 0, 2, 4, ... and
    the layer count is not fixed by anything we control -- read it, don't assume.
    """
    indices = []
    for name, value in state.items():
        if not name.startswith(f"{prefix}.") or not name.endswith(".weight"):
            continue
        suffix = name[len(prefix) + 1 : -len(".weight")]
        if suffix.isdigit() and value.ndim == 2:
            indices.append(int(suffix))
    return sorted(indices)


def _build_mlp(state: dict[str, torch.Tensor], prefix: str) -> nn.Sequential:
    """Rebuild one Linear/ELU stack from its weights (no activation on the last)."""
    indices = _linear_layer_indices(state, prefix)
    if not indices:
        raise ValueError(f"No linear layers found under checkpoint prefix {prefix!r}")
    layers: list[nn.Module] = []
    for position, index in enumerate(indices):
        weight = state[f"{prefix}.{index}.weight"]
        bias = state[f"{prefix}.{index}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        with torch.no_grad():
            linear.weight.copy_(weight)
            linear.bias.copy_(bias)
        layers.append(linear)
        if position != len(indices) - 1:
            layers.append(nn.ELU())
    return nn.Sequential(*layers)


class UpperActor(nn.Module):
    """``normalize(obs) -> mlp`` -- the plain rsl-rl MLPModel actor."""

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer("obs_mean", state["obs_normalizer._mean"].reshape(1, -1))
        self.register_buffer("obs_std", state["obs_normalizer._std"].reshape(1, -1))
        self.mlp = _build_mlp(state, "mlp")
        self.obs_dim = int(self.obs_mean.shape[-1])

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.mlp((obs - self.obs_mean) / (self.obs_std + NORMALIZER_EPS))


class LowerActor(nn.Module):
    """``concat[normalize(obs), encoder(history)] -> mlp`` -- EncoderMLPModel.

    Only ``obs`` passes through the normalizer. ``history`` is the same 100-D
    vector stacked over 5 ticks (oldest -> newest) and enters raw, exactly as in
    ``EncoderMLPModel.get_latent``; normalizing it here would be a silent
    distribution shift that still produces plausible-looking actions.
    """

    def __init__(self, state: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.register_buffer("obs_mean", state["obs_normalizer._mean"].reshape(1, -1))
        self.register_buffer("obs_std", state["obs_normalizer._std"].reshape(1, -1))
        self.encoder = _build_mlp(state, "encoder_mlps.lower_actor_history")
        self.mlp = _build_mlp(state, "mlp")
        self.obs_dim = int(self.obs_mean.shape[-1])
        first_encoder = self.encoder[0]
        assert isinstance(first_encoder, nn.Linear)
        self.history_dim = int(first_encoder.in_features)

    def forward(self, obs: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        normalized = (obs - self.obs_mean) / (self.obs_std + NORMALIZER_EPS)
        latent = self.encoder(history.reshape(history.shape[0], -1))
        return self.mlp(torch.cat([normalized, latent], dim=-1))


@dataclass(frozen=True)
class Branch:
    name: str
    module: nn.Module
    input_names: list[str]
    example: tuple[torch.Tensor, ...]
    action_dim: int
    #: Which observation layout the source checkpoint was trained under -- the
    #: value policy.yaml's observation terms have to declare. Resolved once, in
    #: _load_branches, so the printed banner and the sidecar cannot disagree.
    layout: str


def _resolve_layout(lower_obs_dim: int, upper_obs_dim: int) -> str:
    """Which ALMI observation layout this checkpoint's widths belong to.

    Read off the weights, never taken as an argument: the layout is a property
    of the trained network, and an operator asked to name it is an operator who
    can name it wrong -- producing a policy.yaml whose observation terms build
    the wrong vector at the right width.
    """
    for name, dims in LAYOUT_DIMS.items():
        if dims == (lower_obs_dim, upper_obs_dim):
            return name
    raise SystemExit(
        f"checkpoint has lower_obs={lower_obs_dim}, upper_obs={upper_obs_dim}, "
        f"which is none of the known ALMI layouts {LAYOUT_DIMS}. If this is a "
        "new task, add it to sim2real/rl_policy/observations/almi.py (the table "
        "AND the vector it builds) before exporting."
    )


def _load_branches(checkpoint: Path) -> list[Branch]:
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
    for key in ("lower_actor_state_dict", "upper_actor_state_dict"):
        if key not in loaded:
            raise KeyError(
                f"{checkpoint} is not an ALMI dual-policy checkpoint; missing {key!r}. "
                f"Top-level keys: {sorted(loaded)}"
            )

    lower = LowerActor(loaded["lower_actor_state_dict"]).eval()
    upper = UpperActor(loaded["upper_actor_state_dict"]).eval()

    # The obs widths are what pick the layout, and the layout is what
    # checkpoints/almi/*/policy.yaml has to declare. A checkpoint from a
    # differently-shaped task would export fine and then be wrong at runtime, so
    # it is refused here instead.
    layout = _resolve_layout(lower.obs_dim, upper.obs_dim)
    _expect(lower.history_dim, LOWER_HISTORY_STEPS * lower.obs_dim, "lower history")
    _expect(_out_features(lower.mlp), LOWER_ACTION_DIM, "lower action")
    _expect(_out_features(upper.mlp), UPPER_ACTION_DIM, "upper action")
    print(
        f"[export] layout={layout} (lower_obs={lower.obs_dim}, "
        f"history={lower.history_dim}, upper_obs={upper.obs_dim}) -- put "
        f"`layout: {layout}` on every observation term in policy.yaml"
    )

    return [
        Branch(
            name="lower",
            module=lower,
            input_names=LOWER_INPUT_NAMES,
            example=(
                torch.zeros(1, lower.obs_dim),
                torch.zeros(1, lower.history_dim),
            ),
            action_dim=LOWER_ACTION_DIM,
            layout=layout,
        ),
        Branch(
            name="upper",
            module=upper,
            input_names=UPPER_INPUT_NAMES,
            example=(torch.zeros(1, upper.obs_dim),),
            action_dim=UPPER_ACTION_DIM,
            layout=layout,
        ),
    ]


def _out_features(mlp: nn.Sequential) -> int:
    last = mlp[-1]
    assert isinstance(last, nn.Linear)
    return int(last.out_features)


def _expect(actual: int, expected: int, what: str) -> None:
    if actual != expected:
        raise ValueError(f"Unexpected {what} dimension: got {actual}, expected {expected}")


def _random_inputs(branch: Branch, rng: np.random.Generator) -> tuple[torch.Tensor, ...]:
    return tuple(
        torch.from_numpy(rng.standard_normal(tuple(example.shape)).astype(np.float32))
        for example in branch.example
    )


def _verify(
    branch: Branch, onnx_path: Path, trials: int, rtol: float, atol: float, seed: int
) -> tuple[float, float]:
    """Run ``trials`` random inputs through both torch and ORT.

    Returns ``(max_abs_diff, max_rel_diff)``. The tolerance is the
    ``numpy.allclose`` pair rather than a bare ``atol``: standard-normal probe
    inputs are far outside the training distribution and drive the raw action to
    O(40), where float32 rounding alone is worth a few times 1e-5 absolutely but
    ~1e-6 relatively. Checked against a float64 reference, the ONNX graph is at
    least as accurate as the torch float32 module it was traced from.
    """
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    worst_abs = 0.0
    worst_rel = 0.0
    for _ in range(trials):
        inputs = _random_inputs(branch, rng)
        with torch.inference_mode():
            expected = branch.module(*inputs).detach().cpu().numpy()
        feed = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in zip(branch.input_names, inputs)
        }
        actual = session.run(None, feed)[0]
        diff = np.abs(expected - actual)
        worst_abs = max(worst_abs, float(np.max(diff)))
        worst_rel = max(worst_rel, float(np.max(diff / (np.abs(expected) + 1.0e-12))))
        if not np.allclose(expected, actual, rtol=rtol, atol=atol):
            raise AssertionError(
                f"{branch.name} ONNX disagrees with the torch reference: "
                f"max|diff|={float(np.max(diff)):.3e}, rtol={rtol:.1e}, atol={atol:.1e}"
            )
    return worst_abs, worst_rel


def export(
    checkpoint: Path,
    output_dir: Path,
    *,
    opset: int,
    trials: int,
    rtol: float,
    atol: float,
    seed: int,
) -> None:
    branches = _load_branches(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)

    for branch in branches:
        onnx_path = output_dir / f"{branch.name}.onnx"
        with torch.inference_mode():
            torch.onnx.export(
                branch.module,
                branch.example,
                str(onnx_path),
                input_names=branch.input_names,
                output_names=OUTPUT_NAMES,
                opset_version=opset,
                do_constant_folding=True,
                dynamic_axes={},
                dynamo=False,
            )
        # Sibling .json is documentation only (nothing imports it), matching
        # checkpoints/heft/pmg/policy.json and scalebfm's.
        (output_dir / f"{branch.name}.json").write_text(
            json.dumps(
                {
                    "source_checkpoint": str(checkpoint),
                    "layout": branch.layout,
                    "in_keys": branch.input_names,
                    "in_shapes": [list(t.shape) for t in branch.example],
                    "out_keys": OUTPUT_NAMES,
                    "out_shapes": [[1, branch.action_dim]],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        worst_abs, worst_rel = _verify(branch, onnx_path, trials, rtol, atol, seed)
        print(
            f"[export] {onnx_path}  inputs={branch.input_names} "
            f"-> action({branch.action_dim})  "
            f"max|diff|={worst_abs:.3e} max_rel={worst_rel:.3e} over {trials} trials"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="ALMI model_*.pt")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--rtol", type=float, default=1.0e-4)
    parser.add_argument("--atol", type=float, default=1.0e-5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    export(
        args.checkpoint.expanduser().resolve(),
        args.output_dir.expanduser().resolve(),
        opset=args.opset,
        trials=args.trials,
        rtol=args.rtol,
        atol=args.atol,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
