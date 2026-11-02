# Root Project

root project 用来跑 inference、tracking policy、MuJoCo simulation，以及 robot I/O。

## Setup

```bash
uv sync --extra inference-cpu
```

如果这台机器要跑 `--robot-io inline` 或 `scripts/g1/real_bridge_cpp.py`，还需要安装
G1 robot extra：

```bash
uv sync --extra inference-cpu --extra robot-g1
```

如果 `unitree_sdk2py` setup 找不到 `cyclonedds`，可以参考上游 FAQ：

[https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq](https://github.com/unitreerobotics/unitree_sdk2_python?tab=readme-ov-file#faq)

常见报错：

```text
Could not locate cyclonedds. Try to set CYCLONEDDS_HOME or CMAKE_PREFIX_PATH
```

先编译并安装 CycloneDDS：

```bash
cd ~
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install
cmake --build . --target install
export CYCLONEDDS_HOME="$HOME/cyclonedds/install"
```

然后重新执行环境 setup。

## 验证安装

运行无需权重、机器人 SDK 或硬件的 CPU 单元检查：

```bash
uv run --no-sync python -m unittest discover -s tests -p test_math_helpers.py
```

可选的推理检查需要先按[下载资产](/reference/artifacts)准备 HEFT 权重：

```bash
uv run --no-sync python scripts/test_policy_inference.py \
  --policy_config checkpoints/heft/pmg/policy.yaml \
  --inference_backend onnx-cpu --single
```

电机或校准脚本属于独立硬件操作，不作为安装检查。只有明确需要相应硬件流程时才运行。

## Next Steps

- [Offline Motion Tracking](../tutorials/offline-motion-tracking.md)
- [Robot I/O](/reference/robot-io)
