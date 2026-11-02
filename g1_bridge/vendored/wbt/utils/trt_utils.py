"""
TensorRT Execution Provider 工具
在 JetPack 6 / NVIDIA NX 上使用 TensorRT 加速 ONNX 推理

使用方式:
    from whole_body_teleop.utils.trt_utils import create_ort_session

    # 普通 CPU 推理 (原有行为, 不变)
    session = create_ort_session(model_path)

    # TensorRT 加速推理
    session = create_ort_session(model_path, use_trt=True)

注意事项:
    1. 首次使用 TRT 时会编译 engine, 大模型 (SONIC encoder ~48MB) 可能需要 2-5 分钟.
       编译结果缓存到 {model_dir}/trt_engines/, 后续启动秒级完成.
    2. TRT engine 与硬件绑定, 在不同设备 (如开发机 → NX) 间需要重新编译.
    3. 依赖 onnxruntime-gpu (JetPack 版本), 非标准 pip 包.
       安装参考: https://elinux.org/Jetson_Zoo#ONNX_Runtime
    4. FP16 默认开启, 在 Jetson 上通常提速 1.5-2x, 精度损失极小.
"""

import os
import onnxruntime as ort
from loguru import logger


_PROVIDERS_CACHE = None


def preferred_providers():
    """ONNX Runtime providers for SONIC sessions.

    Returns ``["CUDAExecutionProvider", "CPUExecutionProvider"]`` when the GPU
    provider is available (so SONIC runs on the GPU, with a CPU fallback if a
    given op/load fails), else ``["CPUExecutionProvider"]``. Cached after the
    first call.

    Also preloads the pip-installed CUDA/cuDNN libs (e.g. from torch's
    ``nvidia-*`` packages) via ``ort.preload_dlls()`` so the CUDA EP can find
    ``libcudnn.so.9`` — otherwise it silently falls back to CPU. Deliberately
    does NOT include TensorRT (it would auto-build engines, 2-5 min per model).
    """
    global _PROVIDERS_CACHE
    if _PROVIDERS_CACHE is not None:
        return _PROVIDERS_CACHE
    # Silence ORT's per-inference WARNING spam (the CUDA EP prints ~44
    # "ScatterND with reduction=='none'..." lines on EVERY encoder/decoder run;
    # at 50 Hz this floods stderr and the terminal I/O blocks the control loop
    # -> ~90 ms/step instead of ~18 ms). 3 = ERROR.
    try:
        ort.set_default_logger_severity(3)
    except Exception:
        pass
    if "CUDAExecutionProvider" in ort.get_available_providers():
        try:
            if hasattr(ort, "preload_dlls"):
                ort.preload_dlls()
        except Exception as e:  # pragma: no cover
            logger.warning("ort.preload_dlls() failed ({}); CUDA may fall back "
                           "to CPU", e)
        _PROVIDERS_CACHE = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        logger.info("SONIC ONNX providers: CUDA (GPU) with CPU fallback")
    else:
        _PROVIDERS_CACHE = ["CPUExecutionProvider"]
        logger.info("SONIC ONNX providers: CPU only (onnxruntime-gpu not "
                    "installed / no CUDA EP)")
    return _PROVIDERS_CACHE


def create_ort_session(
    model_path: str,
    use_trt: bool = False,
    trt_cache_dir: str = None,
    fp16: bool = True,
) -> ort.InferenceSession:
    """创建 ONNX Runtime InferenceSession, 可选 TensorRT 加速.

    Args:
        model_path:     ONNX 模型路径
        use_trt:        是否启用 TensorRT EP (默认 False, 与原行为完全一致)
        trt_cache_dir:  TRT engine 缓存目录. None 时自动使用模型同目录下的
                        trt_engines/ 子目录
        fp16:           是否启用 FP16 (TRT 模式下有效, 默认 True)

    Returns:
        ort.InferenceSession

    Raises:
        RuntimeError: TRT EP 不可用且 use_trt=True 时报错提示
    """
    if not use_trt:
        # Explicit providers (GPU if available) — without this, an
        # onnxruntime-gpu install defaults to trying TensorRT first (slow engine
        # build). preferred_providers() also preloads cuDNN + raises the log
        # severity. Per-session log_severity_level=3 silences the ScatterND
        # warning spam (see preferred_providers).
        providers = preferred_providers()
        so = ort.SessionOptions()
        so.log_severity_level = 3
        return ort.InferenceSession(
            model_path, sess_options=so, providers=providers
        )

    # --- TensorRT EP ---
    available_providers = ort.get_available_providers()
    if "TensorrtExecutionProvider" not in available_providers:
        logger.error(
            "TensorrtExecutionProvider 不可用! "
            "请确认已安装 JetPack 版 onnxruntime-gpu.\n"
            "参考: https://elinux.org/Jetson_Zoo#ONNX_Runtime\n"
            "当前可用 EP: {}",
            available_providers,
        )
        raise RuntimeError(
            "TensorrtExecutionProvider not available. "
            "Install onnxruntime-gpu for JetPack."
        )

    if trt_cache_dir is None:
        trt_cache_dir = os.path.join(
            os.path.dirname(os.path.abspath(model_path)), "trt_engines"
        )
    os.makedirs(trt_cache_dir, exist_ok=True)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    logger.info(
        "TensorRT EP 已启用: {} | FP16={} | 缓存目录={}",
        model_name,
        fp16,
        trt_cache_dir,
    )
    logger.info(
        "首次运行会编译 TRT engine (大模型可能需要 2-5 分钟), 后续启动秒级加载缓存."
    )

    trt_options = {
        "trt_engine_cache_enable": True,
        "trt_engine_cache_path": trt_cache_dir,
        "trt_fp16_enable": fp16,
        "trt_max_workspace_size": 1 << 30,  # 1 GB
    }

    providers = [
        ("TensorrtExecutionProvider", trt_options),
        ("CUDAExecutionProvider", {}),
        ("CPUExecutionProvider", {}),
    ]

    session = ort.InferenceSession(model_path, providers=providers)

    # 验证实际使用的 EP (TRT 可能对某些算子回退到 CUDA)
    actual_providers = session.get_providers()
    logger.info("{} 实际使用的 EP: {}", model_name, actual_providers)

    return session
