# G1 仿真控制器职责

当前通用和固定任务采集使用 **`G1_BRIDGE`**，真正的控制器运行时在独立的
[`g1_backend`](https://github.com/LocoManiLab/g1_backend) 中。
不要把可选 WBT demo 的五模式配置当作采集默认入口。

## 采集调用路径

```text
collect_demos_g1 / collect_fixed_task
  → robocasa.utils.g1_data_collection
  → G1BridgeComposite（G1_BRIDGE）
  → g1_backend 的 TeleopSession / 选中 backend
  → MuJoCo 状态读写与任务成功信号 → 仿真录制适配层
```

| 层 | 负责什么 |
|---|---|
| [`g1_data_collection.py`](../robocasa/utils/g1_data_collection.py) | 组装仿真、观测/动作帧、成功保存和仿真深度记录；复用后端共享录制层 |
| [`g1_host/bridge.py`](../robocasa/g1_host/bridge.py) | robosuite composite API、MuJoCo 读写、每次 reset 的稳定阶段；把运行时交给 `TeleopSession` |
| [`g1_backend/g1_bridge/session.py`](https://github.com/LocoManiLab/g1_backend/blob/main/g1_bridge/session.py) | PICO 输入、模式管理、选中策略后端、重定向和参考流 |
| [`g1_host/mujoco_io.py`](../robocasa/g1_host/mujoco_io.py) | 只有仿真宿主能完成的关节/根状态与物理读写 |

采集时显式给出 `--backend` 和需要的 `--policy-config`。
SONIC 使用自身 encoder/decoder 及共享 planner；其他后端选择自己的策略配置，
有的还需要 GMR 或相应重定向资源。模式与可用输入由所选后端决定，
不能从另一个控制器的按键表推断当前行为。

操作说明以 [README](../README.md#g1-data-collection) 和当前后端版本为准。
数据记录只接受有效的控制器输出，任务成功触发保存；手动保存、丢弃和输出布局见
[采集说明](g1_data_collection_pipeline.md)。控制器版本与动作语义见 [权重选择](sonic_ckpt_switching.md)。

## 可选 WBT demo 与独立客户端

[`g1_host/wbt.py`](../robocasa/g1_host/wbt.py) 注册 `WHOLE_BODY_G1`，供
[`demo_empty_teleop.py`](../robocasa/demos/demo_empty_teleop.py)、部分 rollout 和保留的
SONIC token/pose 客户端使用。它与通用采集的 `G1_BRIDGE` 是不同的仿真宿主路径，
其配置、输入方式和 checkpoint 参数不能直接套给采集命令。
底层 WBT/SONIC 控制实现同样来自 `g1_backend`，不需要在本仓复制一套控制器。

保留客户端的用途见 [策略工作流](living_room_vla_pipeline.md)。
这一分层说明仅基于当前源码，不构成新一轮实时控制、渲染或硬件验证。
