# 相机 + LeRobot 录制

给所有基于 `BasePolicy` 的 policy family（HEFT、SONIC、TWIST2 等，PICO 遥操作已经通用）加上：
相机采集、真正的 LeRobot v2.1 episode 录制、以及录制过程的开始/停止/丢弃控制。默认行为
（`--record`，不加任何新 flag）和之前完全一样，一个扁平 npz，没有图像。

## 新增了什么

- `sim2real/camera/`：通用 ZMQ 相机 client/server（`messages.py`/`sources.py`/`server.py`/
  `client.py`），跟 `BasePolicy` 完全解耦，可以单独起来测试。
- `recording/base.py`：`EpisodeRecorderBase` 接口，`BasePolicy` 只认这一个抽象。
- `recording/npz_writer.py`：`NpzEpisodeRecorder`——原来的扁平 npz 逻辑原样搬过来，
  `--record` 不加别的 flag 时行为字节级不变。·
- `recording/lerobot_writer.py` + `recording/schema.py`：`--record-format lerobot` 时走
  这条路径。字段（`observation.*`）直接从每个 policy family 自己 `policy.yaml` 里已经声明
  的 `observation` 分组派生，不需要新写一行 per-family 代码；相机、手部抓握是独立的可选
  hook（`--camera-connect`/`--hand-grip-connect`），因为它们不在 `Observation` 注册表里。

## 完整使用方法

### 0. 前置：下载 checkpoint / 运行时资产

控制器配置清单随代码保留，运行时只准备所选控制器的权重与当前平台需要的原生依赖。
按 [`docs/artifacts.md`](../../../docs/artifacts.md) 选择对应文件，避免下载全部 family。
CPU 接口检查不需要模型权重；相机和机器人 SDK 仅在实际运行对应流程时安装。

下载的权重放在 `checkpoints/`，平台构建产物按资产指南放入本机 `third_party/` 目录。
保留 Git 跟踪的 `policy.yaml` 与 JSON/YAML 清单，不用下载包覆盖它们。

### 1. 环境安装

主环境（跑 policy/推理/仿真）：

```bash
uv sync --extra inference-cpu            # 本机/CPU 推理
# 或 G1 onboard：
uv sync --extra inference-gpu --extra robot-g1
```

这次新增的相机 + LeRobot 录制是可选 extra，按需再装：

```bash
uv sync --extra camera --extra lerobot
```

PICO 遥操作发布端（`teleop/pico_retarget_pub.py`）用的是独立的 `venv/pico` 项目（不是主
环境），按 [`AGENTS.md`](../../../AGENTS.md) 的约定：

```bash
uv sync --project venv/pico
```

### 2. 起 PICO 遥操作发布端

```bash
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py
```

打开它打印出来的 mjviser URL，确认重定向后的 G1 动作看起来正常，再往下走。

### 3.（可选）起相机 server

只有想录 `observation.images.*` 才需要这一步：

```bash
uv run python -m sim2real.camera.server --cameras.ego_view.source realsense
# 没有 RealSense、只有普通 USB 摄像头：
# uv run python -m sim2real.camera.server --cameras.ego_view.source cv2 --cameras.ego_view.device 0
```

默认发布在 `tcp://*:5594`（`PORTS["camera"]`）。

### 4. 起执行端（sim2sim 示例）

```bash
uv run sim2real/sim_env/base_sim.py --robot g1
```

### 5. 起 policy，开 PICO 遥操作 + LeRobot 录制

```bash
uv run sim2real/rl_policy/tracking.py \
  --robot g1 \
  --policy-config checkpoints/mimic-lite/32x8192-huge/policy.yaml \
  --motion-backend zmq \
  --controller pico \
  --record --record-format lerobot \
  --camera-connect tcp://127.0.0.1:5594 \
  --hand-grip-connect tcp://127.0.0.1:5593   # 可选，录 PICO trigger 派生的手部抓握量
```

不想要图像/LeRobot 格式、只想跟以前一样录一个扁平 npz：去掉
`--record-format lerobot --camera-connect ...`，单独一个 `--record` 行为和这个功能上线前
完全一致。

`--policy-config` 换成任意 family 的 checkpoint（`checkpoints/<family>/.../policy.yaml`）
都一样能用——PICO 输入和这次的录制控制是在 `BasePolicy` 基类里接的，对所有 family 通用，
不需要每个 family 单独适配。

### 6. PICO 遥操作+录制控制

沿用 [Pico Teleoperation](../../../docs/tutorials/pico-teleoperation.md) 已有的控制手势，
再加这次新增的 episode 控制：

| 手势 | 作用 |
| --- | --- |
| `A` | 进入 init pose |
| `A` + `B` | 进入 policy 模式 |
| `X` | 恢复/取消暂停动作流（`pico_retarget_pub.py` 自带） |
| `Y`（新增） | 开始 / 停止当前 episode 录制 |
| `Y` + `A`（新增） | 丢弃当前 episode（不保存） |

键盘等效：`c` 开始/停止，`x` 丢弃（跟 LocoManiLab 的 HEFT/AMO/SONIC 控制器约定一致）。

### 7. 输出

`--record-format lerobot` 时，`--record-output`（不传则是当前目录下一个带时间戳的默认路径）
下会生成一个 LeRobot v2.1 dataset 目录：

```text
meta/info.json
meta/modality.json
meta/episodes.jsonl
meta/stats.json
data/
videos/            # 只有接了 --camera-connect 才有
```

`observation.*` 字段直接来自这个 family 自己 `policy.yaml` 里声明的 `observation` 分组
（外加基线机器人状态），`action.q_target`/`action.raw` 来自控制循环本来就会算的量——都不
需要新写 per-family 代码。

更详细的分步教程（含中文版）：
[`docs/tutorials/camera-and-lerobot-recording.md`](../../../docs/tutorials/camera-and-lerobot-recording.md)。

## Controller 专属字段（`RecordingExtra`）

heft/sonic 各自还有一批只有自己知道怎么算的字段（heft 的 Dex3 手部测量/目标值；sonic 的正运动学末端位姿
`eef_state`、重建的全身动作 `action.wbc`、RL 策略内部 `action.motion_token`、以及一整套 `teleop.*`
遥操作原始元数据）。这些不适合塞进 `schema.py` 的通用派生逻辑（会长满 `if family == ...` 分支），所以
另开了一个注册表——照抄 `rl_policy/observations/` 已经验证过的模式（`__init_subclass__` 自动注册 +
按名字 `resolve()`）：

- `recording/extras/base.py` 的 `RecordingExtra`：`features()` 声明要加的 LeRobot 字段，
  `compute(frame)` 每帧算一次要合并进去的值，`close()` 释放自己开的资源（ZMQ socket、正运动学模型…）。
- 每个 controller 一个文件，命名 `<family>_extra.py`，比如 `recording/extras/heft_extra.py`/
  `recording/extras/sonic_extra.py`（跟 `observations/heft.py`/`observations/sonic.py` 对应）。
- 启用方式：`policy.yaml` 里加 `recording_extras: [name, ...]` 作为这个 family 的默认声明，或者
  `--record-extra name`（可多个）临时加，两者取并集。
- 字段冲突（两个来源声明了同一个 key）会在 `LeRobotRecorder` 里直接 `raise ValueError`，不会静默覆盖。

`recording/extras/_example_extra.py` 里的 `ExampleRecordingExtra`（`--record-extra example`）是纯演示用的
模板，不对应任何真实字段。

真实的两个：

- `extras/heft_extra.py` 的 `HeftHandsRecordingExtra`（`--record-extra heft-hands`）——从 heft 下游 fork 的
  `heft_debug` ZMQ topic（`deploy.py --publish-state-addr` 发布，端口默认 5558）读 Dex3 手部测量值/目标值，
  生成 `observation.hand_state`/`action.hand_joints`（各 14 维）。**只有跑在带这个状态发布 patch 的 heft
  fork 上才有真实数据**——纯上游 sim2real 不发布这个 topic，此时会退化成全零（跟原始 exporter 自己的
  `proprio.get(key, [0.0]*7)` 兜底行为一致，不会报错）。
- `extras/sonic_extra.py` 的 `SonicTeleopRecordingExtra`（`--record-extra sonic-teleop`）——从 sonic 自己的
  `pico_manager_thread_server`（`gear_sonic` 组件，不是 sim2real 自己的 `teleop/pico_retarget_pub.py`）
  发布的 `pose`/`planner`/`manager_state` topic（端口默认 5556）读一整套 `teleop.*` 原始遥操作元数据
  （SMPL joints/pose、body 朝向、手腕/手指关节、planner 移动指令、VR 三点追踪）。**同样只有那个 sonic 专属
  server 在跑才有真实数据**。`observation.eef_state`/`action.wbc`/`action.motion_token`/
  `observation.cpp_rotation_offset`/`observation.init_base_quat`/`teleop.delta_heading` 这几个字段
  **没有实现**——分别需要正运动学机器人模型或 sonic C++ 控制器自己的 `g1_debug` proprio topic，sim2real
  这边没有等价基础设施，模块 docstring 里写清楚了原因，没有拿假数据充数。

## 关键设计决定

- PICO 手势用 `Y`/`Y+A`，不用 `X`——`X` 已经被 `pico_retarget_pub.py` 自己用来做暂停切换，
  复用会导致一次按键在两个进程里触发两个不相关的动作。
- 相机 client 用 RCVHWM+丢到最新，而不是这个仓库其他地方统一用的 `zmq.CONFLATE`——
  CONFLATE 在订阅者晚加入时会永久收不到消息，这是已知坑。
- LeRobot dataset 的 `features`/`modality_config` 延迟到第一帧真正收到（相机/手部数据都
  到齐）才创建，图像 shape 自动探测，不需要额外声明。

## 验证与集成边界

- `tests/test_camera_wire_format.py`、`test_pico_episode_command.py`、
  `test_npz_recorder_backward_compat.py` 检查消息与录制接口；这些 CPU 检查不等同于
  真机控制、相机和 PICO 硬件验证。
- LocoManiLab 主仓编排工作流，`g1_bridge` 提供仿真与真机共用的采集和录制，
  `sim2real` 保留原生运行时与录制扩展。使用当前仓库的入口，不需要独立
  HeFT 研究源码仓库或旧同步命令。
- `BasePolicy` 的这些扩展仅作用于该原生控制循环；其他运行时仍需使用各自的录制接口。
