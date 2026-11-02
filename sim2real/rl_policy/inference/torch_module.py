"""TorchScript (``.pt``) inference backend, parallel to ``ONNXModule``.

Some checkpoints (e.g. AMO) ship as TorchScript modules exported via
``torch.jit.save``, not ONNX -- this wraps ``torch.jit.load`` behind the same
``__call__(input_dict) -> output_dict`` contract ``ONNXModule`` exposes, so
``BasePolicy``'s ``self.policy(...)`` closure doesn't need to care which
backend a given checkpoint uses.

Unlike ONNX graphs, TorchScript modules carry no metadata naming their
inputs/outputs -- ``input_keys``/``output_keys`` (in call-argument order) are
supplied by the caller instead of introspected.
"""

from __future__ import annotations

from typing import Dict, Sequence, Union

import numpy as np
import torch


def patch_jit_to_cpu(jit_module):
    """Rewrite hard-coded ``torch.device`` constants baked into a TorchScript
    graph (e.g. a module traced/scripted on a CUDA machine) to ``'cpu'``, so
    it can load and run on a CPU-only box. No-op if there's nothing to patch."""
    try:
        for node in jit_module.graph.nodes():
            if node.kind() == "prim::Constant" and "value" in node.attributeNames():
                val = node.output().toIValue()
                if isinstance(val, torch.device):
                    node.removeAttribute("value")
                    node.s_("value", "cpu")
    except Exception:
        pass
    return jit_module


class TorchJitModule:
    def __init__(
        self,
        path: str,
        input_keys: Sequence[Union[str, tuple]] = ("obs",),
        output_keys: Sequence[Union[str, tuple]] = ("action",),
        device: str = "cpu",
    ):
        self.device = device
        self.model = patch_jit_to_cpu(torch.jit.load(path, map_location=device))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.in_keys = list(input_keys)
        self.out_keys = list(output_keys)

    def __call__(
        self, input_dict: Dict[Union[str, tuple], np.ndarray]
    ) -> Dict[Union[str, tuple], np.ndarray]:
        args = []
        for key in self.in_keys:
            array = np.asarray(input_dict[key], dtype=np.float32).reshape(1, -1)
            args.append(torch.from_numpy(array).to(self.device))

        with torch.no_grad():
            raw_out = self.model(*args)

        outs = raw_out if isinstance(raw_out, (tuple, list)) else (raw_out,)
        if len(outs) != len(self.out_keys):
            raise ValueError(
                f"TorchJitModule({self.model!r}): model returned {len(outs)} "
                f"tensor(s) but output_keys has {len(self.out_keys)} entries "
                f"({self.out_keys!r})"
            )
        return {
            key: tensor.detach().cpu().numpy().reshape(-1)
            for key, tensor in zip(self.out_keys, outs)
        }
