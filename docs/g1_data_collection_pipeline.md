# G1 仿真数据采集

仿真采集统一使用 `G1_BRIDGE` 和独立的 `g1_backend` 控制器/录制运行时。
先按 [README](../README.md#installation) 准备仿真环境、G1 后端及所选控制器权重；
再按任务准备场景资产。EmptyEnv 不需要厨房、Objaverse 或 iTHOR 场景包。

## 两个采集入口

| 入口 | 用途 |
|---|---|
| [`collect_demos_g1.py`](../robocasa/scripts/collect_demos_g1.py) | 通用采集，选择已注册的任务环境；场景参数转发给该环境 |
| [`collect_fixed_task.py`](../robocasa/scripts/collect_fixed_task.py) | 固定首轮采样的任务状态，在后续 reset 中施加受约束的位置抖动 |

两个入口都复用 [`g1_data_collection.py`](../robocasa/utils/g1_data_collection.py)：

```text
采集 CLI → g1_data_collection → G1_BRIDGE → g1_backend / TeleopSession
                        └── 仿真观测、成功信号和深度 → 共享录制层 → LeRobot 数据集
```

`humanoid_benchmark` 负责场景、MuJoCo 读写、reset 和任务成功信号。
`g1_backend` 提供 PICO 输入、模式管理、后端策略、重定向及共享录制结构。
不同控制器的数据含义可能不同，不能只看向量维度判断兼容。

## 明确选择控制器和权重

以下命令从 `humanoid_benchmark` 根目录、已按 README 安装的
`tele_benchmark` 环境运行，并假定 `g1_backend` 是相邻目录。
若使用其他安装布局，换成自己的明确路径。

SONIC v1.1 的离线计划：

```bash
python robocasa/scripts/download_g1_checkpoints.py \
  --family sonic_v1_1 --dest ../g1_backend/checkpoints --plan
```

`--plan` 不联网。需要查询远端清单/大小时改为 `--dry-run`；准备下载时去掉计划选项，
脚本会先询问。该选择包含 v1.1 encoder/decoder 及共用的
`sonic/low_latency/g1/planner_sonic.onnx`，不下载其他控制器或低延迟 encoder/decoder。
Git 中的 `policy.yaml` 来自后端源码，不由下载器覆盖。详见 [版本选择](sonic_ckpt_switching.md)。

## 最小启动与任务采集

无头显时，可先检查 EmptyEnv 和所选控制器能否启动：

```bash
python -m robocasa.scripts.collect_demos_g1 \
  --env EmptyEnv --backend sonic \
  --policy-config ../g1_backend/checkpoints/sonic/v1_1/g1/policy.yaml \
  --pico-source mock
```

这条命令会启动仿真，要求渲染环境及权重已准备好。Mock 不产生真实 PICO 按键，
不能用该启动结果代替遥操作、成功率或数据质量验证。EmptyEnv 没有任务成功分数，
在有有效录制帧时用 `S` 手动保存一集。

接入真实 PICO 后，先准备所选任务资产。固定场景和任务初态的示例：

```bash
python -m robocasa.scripts.collect_fixed_task \
  --env IThorKitchenOpenFridge --scene-seed 0 \
  --backend sonic --pico-source real \
  --policy-config ../g1_backend/checkpoints/sonic/v1_1/g1/policy.yaml
```

iTHOR 首次安装仍涉及共享 THOR 库，见 [场景安装](molmospaces_scenes.md)。
固定采集默认有 G1 和物体位置的小幅抖动；需要取消时分别使用
`--no-g1-jitter`、`--no-object-jitter`。通用入口则可用
`--fixed-scene --scene-seed 0` 或 `--fixed-objects`，两者不要同时选择。

| 选项 | 含义 |
|---|---|
| `--backend` | 明确选择控制器，如 `sonic`、`amo`、`heft`、`scalebfm`、`almi` |
| `--policy-config` | 所选后端的本地 `policy.yaml`，覆盖该后端默认值 |
| `--scene-id` | iTHOR FloorPlan；厨房/客厅布局使用适用的 `--layout`、`--style` |
| `--pico-source real` | 需要已安装的 SDK、PC Service 和真实头显输入 |
| `--out` | 新数据集的绝对路径；不要复用已有数据目录 |
| `--max-episodes` | 保存到指定集数后停止 |

需要其他选项时，在准备好的仿真环境中查看对应入口的 `--help`。
控制器操作以 [README](../README.md#g1-data-collection) 和当前 `g1_backend` 为准；
不要把旧 WBT demo 的操作模式当作采集配置。

## 保存与下游使用

任务成功时自动保存当前非空集，`Q` 丢弃当前集并 reset。默认输出根为
`outputs_sim/<backend>/<env>_<timestamp>/`；固定入口的环境名后增加 `_fixed`。
该目录本身就是 LeRobot 数据集根，包含 `meta/`、parquet 和视频；
仿真适配层还维护与接受帧对应的深度记录。无需再拼接 `lerobot/` 子目录。

后续使用 LocoManiLab 的 [数据工具](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/data_collection.md)、
[训练](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/training.md)、
[仿真验证](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/sim_validation.md) 和
[控制器契约](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/controller.md)。
入口与参数的静态核对不代表已完成模型训练、渲染、真实头显或硬件验证。
