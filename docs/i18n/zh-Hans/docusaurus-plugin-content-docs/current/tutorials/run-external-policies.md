---
title: 运行外部控制器
slug: /tutorials/run-external-policies
---

# 运行外部控制器

已跟踪的 manifest 支持五个共用控制器家族。按照[下载资产](/reference/artifacts)只准备所选变体的权重。

| 家族 | 默认 manifest | 外部输入 |
| --- | --- | --- |
| SONIC | `checkpoints/sonic/low_latency/g1/policy.yaml` | SONIC motion token；bridge 构造匹配的 encoder 输入 |
| HEFT | `checkpoints/heft/pmg/policy.yaml` | `ref_qpos` 参考动作 |
| ScaleBFM | `checkpoints/scalebfm/humanoid_transformer_m/policy.yaml` | `ref_qpos` 参考动作 |
| AMO | `checkpoints/amo/policy.yaml` | `velocity_cmd` 命令 |
| ALMI | `checkpoints/almi/flat_dual/policy.yaml` | `twist_ee` 双分支命令 |

HEFT 与 ScaleBFM 共用参考动作约定。SONIC token、AMO 命令和 ALMI 命令需要匹配的 backend 与数据 schema，仅修改 `--policy-config` 不能相互转换。SONIC v1.1 是 `checkpoints/sonic/v1_1/g1/policy.yaml` 下的可选变体，其 token 与 low-latency 不通用。

独立 HEFT 仿真动作播放见[离线动作跟踪](/tutorials/offline-motion-tracking)。仿真采集使用 humanoid_benchmark/LocoManiLab；共用运行时 API 和真机采集见 checkout 中的 `g1_bridge/README.md`。

## 适配其他控制器

可选的 `.agents/skills/adapt-policy-to-sim2real/` helper 说明了如何追踪源 observation、关节顺序、action scale 与导出流程，同目录维护相应转换脚本。请自行提供匹配的 manifest 和导出权重；历史适配示例不代表此 fork 分发相应 checkpoint。

保持控制器专用输入名、归一化、历史顺序与关节布局，在部署前将导出结果与源实现作数值比较，不能只按 tensor 形状选择 checkpoint。

SONIC SMPL 专用 observation 与流见 [SONIC SMPL Input](/reference/sonic-smpl-input)。此类专用导出需要匹配的 manifest 和权重；本 checkout 分发的 SONIC manifest 仅包含 G1。
