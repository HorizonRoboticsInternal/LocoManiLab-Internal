# 场景采集、策略训练与仿真验证

客厅、厨房和 iTHOR 任务共用 G1 仿真采集入口；场景名称不再选择一套独立录制流程。
仿真环境和控制器按 [README](../README.md#installation) 安装，任务资产按需准备。
`TabletopPickPlace` 需要共享纹理和 Objaverse 杯盘资产；EmptyEnv 可先检查基础接线。

## 当前工作流

| 阶段 | 使用入口 |
|---|---|
| 仿真采集 | 本仓 `collect_demos_g1` 或 `collect_fixed_task`，见 [采集说明](g1_data_collection_pipeline.md) |
| 数据检查、合并与模型准备 | LocoManiLab 的 `lab validate` / `lab prepare`，见 [数据说明](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/data_collection.md) |
| 模型训练 | 对应 policy 的独立环境和本地 checkpoint，见 [policy 指南](https://github.com/LocoManiLab/LocoManiLab/tree/main/policies) 与 [训练说明](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/training.md) |
| 共享模型评测与回放 | LocoManiLab 的 `lab eval` / `lab replay`，见 [仿真验证](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/sim_validation.md) 与 [控制器契约](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/controller.md) |

采集示例：在已准备好任务资产和真实 PICO 的仿真环境中，从仓库根运行：

```bash
python -m robocasa.scripts.collect_demos_g1 \
  --env TabletopPickPlace --fixed-scene --scene-seed 0 \
  --backend sonic --pico-source real \
  --policy-config ../g1_backend/checkpoints/sonic/v1_1/g1/policy.yaml
```

这里假定 `g1_backend` 位于相邻目录；其他布局请填自己的配置绝对路径。
SONIC v1.1 的权重只按 `sonic_v1_1` family 准备，并包含共享 planner，
详见 [权重选择](sonic_ckpt_switching.md)。默认采集输出是
`outputs_sim/sonic/<env>_<timestamp>/` 数据集根；成功自动保存，`Q` 丢弃当前集。

先检查少量采集数据的帧、图像、动作、来源和控制器版本，再决定是否扩大采集或训练。
采集、训练和评测必须匹配动作字段、顺序和 SONIC export；相同的 token 维数不代表相同语义。

## 保留的独立 SONIC 工具

已有工作流可以继续使用以下本地工具；它们不负责安装、启动模型服务器或获取权重：

| 工具 | 用途与边界 |
|---|---|
| [`deploy_living_room_vla.py`](../robocasa/scripts/deploy_living_room_vla.py) | 已有兼容策略服务器/录制动作的 SONIC token 仿真客户端 |
| [`deploy_living_room_vla_pose.py`](../robocasa/scripts/deploy_living_room_vla_pose.py) | 对应 SMPL 参考动作的独立仿真客户端 |
| [`replay_living_room_dataset.py`](../robocasa/scripts/replay_living_room_dataset.py) | SONIC 记录检查和开环回放；不等于闭环模型评测 |

这些 SONIC 专用工具保留各自的 `--sonic-ckpt v1_1` 参数；采集入口使用的是
`--backend` 加 `--policy-config`，不要混用两套参数。在已安装的仿真环境中用
对应模块的 `--help` 核对输入要求。训练服务和仿真可以使用独立环境，
但传输协议和动作契约必须匹配。

本页是当前入口的说明，不构成对这些独立工具的新安装、训练效果或端到端运行验证声明。
