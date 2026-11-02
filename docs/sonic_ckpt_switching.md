# SONIC 权重版本与数据兼容性

SONIC encoder/decoder 的权重和 Git 跟踪的配置位于独立的 `g1_backend` 仓库。
本仓负责仿真宿主；不要把权重放进 `humanoid_benchmark` 根目录或从另一台机器的路径猜配置。

## 采集：显式选择 policy.yaml

通用和固定任务采集都通过 `G1_BRIDGE` 使用后端配置：

| 选择 | 本地配置，相对 `g1_backend` 根 | 下载 family |
|---|---|---|
| Low latency | `checkpoints/sonic/low_latency/g1/policy.yaml` | `sonic` |
| v1.1 | `checkpoints/sonic/v1_1/g1/policy.yaml` | `sonic_v1_1` |

从 `humanoid_benchmark` 根目录，按 README 的相邻仓布局，先看离线计划：

```bash
python robocasa/scripts/download_g1_checkpoints.py \
  --family sonic_v1_1 --dest ../g1_backend/checkpoints --plan
```

`--plan` 完全离线；`--dry-run`（旧拼写 `--dryrun` 也接受）会查询 Hub 清单和大小，
但不传权重。要下载时去掉计划参数，脚本会询问确认。
`--family` 或 `--all` 必须显式提供；新安装不需要全部 family。

v1.1 同时需要共享的 `sonic/low_latency/g1/planner_sonic.onnx`，下载器会自动纳入，
无需再下载整套 low-latency encoder/decoder。`policy.yaml` 由后端源码提供，
下载器仅写权重文件。已准备好环境、渲染与权重后可启动：

```bash
python -m robocasa.scripts.collect_demos_g1 \
  --env EmptyEnv --backend sonic --pico-source mock \
  --policy-config ../g1_backend/checkpoints/sonic/v1_1/g1/policy.yaml
```

这是实际仿真启动，不是 dry-run。真实采集另需有效 PICO 输入及任务资产，
操作与保存方式见 [采集说明](g1_data_collection_pipeline.md)。
需要 low-latency 时，明确使用表中对应的 family 和配置；不要用含糊的版本昵称选数据语义。

## 模型、回放与可选 WBT 工具

SONIC 的不同 export 可能具有相同的 token 长度，但 token 只对其匹配的 decoder 有意义。
采集、数据转换、策略训练和回放必须保留版本来源，并使用匹配的动作字段/顺序；
不能仅通过张量维度判断可交换性。

LocoManiLab 的统一 CLI 由 `lab ... --sonic-ckpt v1_1` 解析版本并传入后端配置。
使用前按其 [仿真安装指南](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/simulation.md)
配置 workspace；训练与评测要求见 [policy 指南](https://github.com/LocoManiLab/LocoManiLab/tree/main/policies)。

本仓保留的 SONIC token/pose 客户端和记录回放工具见 [工作流说明](living_room_vla_pipeline.md)。
它们保留自身的 `--sonic-ckpt v1_1` 选择器，并通过后端的
[`g1_bridge/sonic_utils/ckpt.py`](https://github.com/LocoManiLab/g1_backend/blob/main/g1_bridge/sonic_utils/ckpt.py)
选择对应 encoder/decoder。可选 `demo_empty_teleop` 的 `WHOLE_BODY_G1` 配置也使用这套 helper；
它是独立 demo/rollout 路径，不是通用采集的默认控制器。

参数与路径已按源码核对；本页不表示已运行模型、仿真或硬件验证。
