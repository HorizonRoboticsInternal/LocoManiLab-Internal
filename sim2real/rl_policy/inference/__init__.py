from typing import Literal

from sim2real.rl_policy.inference.onnx_module import ONNXModule, Timer as Timer

InferenceBackend = Literal["onnx-gpu", "onnx-cpu", "tensorrt", "torch-jit"]
__all__ = ["InferenceBackend", "ONNXModule", "Timer", "build_inference_module"]


def build_inference_module(onnx_path: str, inference_backend: InferenceBackend, **kwargs):
    if inference_backend in {"onnx-cpu", "onnx-gpu"}:
        provider = "gpu" if inference_backend == "onnx-gpu" else "cpu"
        return ONNXModule(onnx_path, providers=provider)
    if inference_backend == "tensorrt":
        from sim2real.rl_policy.inference.tensorrt_module import TensorRTModule

        return TensorRTModule(onnx_path)
    if inference_backend == "torch-jit":
        from sim2real.rl_policy.inference.torch_module import TorchJitModule

        return TorchJitModule(onnx_path, **kwargs)
    raise ValueError(f"Unsupported inference backend: {inference_backend}")
