---
title: 下载资产
slug: /reference/artifacts
---

# 下载资产

新 clone 已经包含控制器 manifest 与共用机器人描述。源码检查、配置审查和纯 CPU 单元测试不需要模型下载。只为要运行的工作流安装原生依赖、下载权重。

## Manifest 与权重

`checkpoints/**/policy.yaml` 及相关 JSON/YAML 元数据随运行时代码管理，定义关节顺序、scale 和 observation 布局。权重单独存放在 LocoManiLab 发布仓库 [GammaRobotics/LocoManiLab](https://huggingface.co/datasets/GammaRobotics/LocoManiLab) 的 `checkpoint/controller/` 目录（与 `checkpoints/` 结构一致）。从仓库根目录**只下载选中的变体**到临时目录，再复制到 `checkpoints/`，例如 HEFT：

```bash
hf download GammaRobotics/LocoManiLab --repo-type dataset --local-dir /tmp/g1-weights \
  --include 'checkpoint/controller/heft/pmg/policy.onnx'
cp -r /tmp/g1-weights/checkpoint/controller/. checkpoints/
```

其他控制器使用以下过滤路径：

| 控制器 / 变体 | 权重路径，相对于 `checkpoints/`（写入 `--include` 时加前缀 `checkpoint/controller/`） |
| --- | --- |
| SONIC low-latency | `'sonic/low_latency/g1/model_encoder.onnx' 'sonic/low_latency/g1/model_decoder.onnx'` |
| SONIC v1.1 | `'sonic/v1_1/g1/model_encoder.onnx' 'sonic/v1_1/g1/model_decoder.onnx'` |
| HEFT PMG | `'heft/pmg/policy.onnx'` |
| ScaleBFM M | `'scalebfm/humanoid_transformer_m/policy.onnx'` |
| AMO | `'amo/amo_jit.pt' 'amo/adapter_jit.pt' 'amo/adapter_norm_stats.pt'` |
| ALMI | `'almi/flat_dual/lower.onnx' 'almi/flat_dual/upper.onnx'` |

只过滤权重文件。旧资产 revision 也包含 manifest，覆盖 checkout 会悄悄替换运行时代码所需的布局。自定义导出放到独立路径，并保留与之匹配的 manifest。

在 `humanoid_benchmark` checkout 中，可以用它的下载工具自动定位 backend 根目录、过滤权重并跳过已有文件：

```bash
python -m robocasa.scripts.download_g1_checkpoints --family heft
python -m robocasa.scripts.download_g1_checkpoints --family sonic
```

当前家族选项见该工具的 `--help`。普通安装不需要一次下载所有家族。

## SONIC 变体

| 变体 | Manifest 目录 | Encoder 约定 |
| --- | --- | --- |
| 默认 low-latency | `checkpoints/sonic/low_latency/g1` | 1247D，body-frame anchor，step1 参考 |
| v1.1 | `checkpoints/sonic/v1_1/g1` | 1751D，heading-normalized anchor，step5 参考 |

两者均使用 64D token 和 994→29 decoder 形状，但 token 含义与 decoder 权重不同。**采集和部署必须使用同一个变体。** 现有数据集继续默认使用 low-latency。

PLANNER 模式还需要共享的 `sonic/low_latency/g1/planner_sonic.onnx`，v1.1 也一样。仅当需要此模式时补充这个精确过滤路径：

```bash
hf download GammaRobotics/LocoManiLab --repo-type dataset --local-dir /tmp/g1-weights \
  --include 'checkpoint/controller/sonic/low_latency/g1/planner_sonic.onnx'
cp -r /tmp/g1-weights/checkpoint/controller/. checkpoints/
```

LocoManiLab 的 `lab eval --check` 会在评测前检查选中的运行时和资产，但不能让不同 SONIC 变体的 token 互相兼容。

## 可选的原生依赖

`third_party/wheels/` 和 `third_party/prebuilt/` 是可选的本地资产目录，不需要每个用户都下载。通用 CPU 推理不需要它们；只有选择真机相关 extra 时才安装 G1 依赖：

```bash
uv sync --extra inference-cpu --extra robot-g1
```

只有在受支持的 GPU 部署中才将 `inference-cpu` 换为 `inference-gpu`，不要同时安装两种 ONNX Runtime extra。JetPack 5 的 PICO 可能需要 `third_party/prebuilt/jetpack5-aarch64/` 下的平台库，见[机载 PICO 安装](/getting-started/teleop-onboard-orin)及其链接的源码构建指南；这与仿真数据采集是独立的。

不要提交下载的权重、原生二进制、凭据或私有训练输出。真机部署使用 `HF_HUB_OFFLINE=1` 与 `HF_HUB_DISABLE_TELEMETRY=1`；预先准备所选文件，再传到明确指定的主机与路径。机器专用同步和上传流程不放在本仓库。
