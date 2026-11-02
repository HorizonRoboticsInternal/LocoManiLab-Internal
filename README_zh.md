# g1_backend

供 [LocoManiLab](https://github.com/LocoManiLab/LocoManiLab) 和 [humanoid_benchmark](https://github.com/LocoManiLab/humanoid_benchmark) 共用的 G1 控制器与遥操作运行时。

[English](README.md) · [中文文档](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/intro.md) · [遥操作与录制接口](g1_bridge/README.md)

- `g1_bridge/`：仿真和真机采集共用的控制会话、PICO 输入、host 适配与训练数据录制。
- `sim2real/`：策略推理、参考动作、MuJoCo 执行与机器人 I/O。为保持兼容，保留现有 Python 包名。
- `checkpoints/`：随代码管理的 manifest；模型权重按需下载。`assets/g1_unified/` 是共用机器人描述。
- `venv/pico/`：隔离可选的 XRoboToolkit/GMR 依赖。机器人 SDK 与相机依赖也按需安装。

本仓库基于上游 [sim2real 部署栈](https://github.com/EGalahad/sim2real)，下方保留原项目引用。此 fork 的安装、资产和使用说明以本仓库为准。

## 从源码安装

独立环境需要 Python 3.10 与 [uv](https://docs.astral.sh/uv/)：

```bash
git clone https://github.com/LocoManiLab/g1_backend.git
cd g1_backend
uv sync --extra inference-cpu
```

运行时使用这个 editable checkout：manifest、机器人描述和启动脚本位于 Python 包旁边，不能只安装一个 wheel 后就移走源码。通过 LocoManiLab 使用时，请按它的 `lab` 安装和环境说明操作，避免再安装另一份。

通用 CPU 环境不需要机器人 SDK、PICO、相机依赖或权重。安装后可运行本地检查：

```bash
uv run --no-sync python -m unittest discover -s tests -p test_math_helpers.py
```

真机部署只安装所选 [Robot I/O 模式](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/robot_io.md) 所需的 extra。PICO 另外提供 [PC](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/teleop-x86-64.md) 与 [机载 Orin](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/getting-started/teleop-onboard-orin.md) 安装指南。

## 选择控制器和资产

共用运行时的五个控制器家族为 SONIC、HEFT、ScaleBFM、AMO、ALMI：

| 家族 / 变体 | 已跟踪的 manifest | 输入约定 |
| --- | --- | --- |
| SONIC | `checkpoints/sonic/low_latency/g1/policy.yaml` | 参考动作 / motion token；默认 SONIC 变体 |
| SONIC v1.1 | `checkpoints/sonic/v1_1/g1/policy.yaml` | 独立的 encoder/decoder 变体；token 与 low-latency 不通用 |
| HEFT | `checkpoints/heft/pmg/policy.yaml` | 参考动作 |
| ScaleBFM | `checkpoints/scalebfm/humanoid_transformer_m/policy.yaml` | 参考动作 |
| AMO | `checkpoints/amo/policy.yaml` | 控制器专用命令 |
| ALMI | `checkpoints/almi/flat_dual/policy.yaml` | 双分支控制器专用命令 |

家族、manifest 与数据/action schema 必须匹配，不能混用其他控制器的权重或录制动作。代码审查和上面的检查不需要权重。运行控制器时只下载所选家族，见[资产说明](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/artifacts.md)。本运行时不依赖独立 HEFT 仓库；这里的 HEFT 控制器继续保留。

仿真采集请按 [humanoid_benchmark](https://github.com/LocoManiLab/humanoid_benchmark) 或 LocoManiLab 采集指南操作；独立采集见 [g1_bridge](g1_bridge/README.md)，离线动作播放见[跟踪教程](docs/i18n/zh-Hans/docusaurus-plugin-content-docs/current/tutorials/offline-motion-tracking.md)。

## 许可证

g1_backend 以 [MIT 许可证](LICENSE) 发布，Copyright (c) 2026 Gamma Robotics。vendored 代码、机器人资产与控制器配置保留各自条款，见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) 及各目录下的许可证文件。下方引用用于署名，不代表许可证授权。

## 上游引用

保留原 sim2real 项目的引用：

```bibtex
@misc{sim2real2026,
  author       = {{RoboParty Lab Team}},
  title        = {sim2real: A Lightweight and Modular Sim2sim and Sim2real Deployment Stack},
  year         = {2026},
  howpublished = {\url{https://github.com/EGalahad/sim2real}},
  note         = {Documentation: \url{https://egalahad.github.io/sim2real/}}
}
```
