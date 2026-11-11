# HoloMotion / RoboOrchard VLA 迁入 LocoManiLab：调查与设计

调查及实现日期：2026-11-11。本文保留源码调查与设计推导；当前实现和验证状态见第 8 节及 [接入说明](holomotion_revo2.md)。已接通真实模型 EmptyEnv 调用，尚未完成正式任务行为验收。

## 结论

可行，但需要同时迁移控制器 runtime、机器人/相机 host、观测协议和参考窗口执行语义。只新增模型 manifest、把 151D 输出改名为现有 `ref_qpos`，无法保留现有链路。

建议首个闭环目标为 **Qwen3 + Revo2 + unified-v1 + HoloMotion v1.4.1 backpack**。保留 RoboOrchard 的 checkpoint 加载、输入处理、tokenizer 和 151D→87D 转换；把 HoloMotion 接入 `g1_backend`，把机器人和相机接入 `humanoid_benchmark`，由 LocoManiLab 统一启动、配置和记录评测。52D/Dex3 是另一条明确标记的兼容路线。

用户已明确目标：整体结构与 LocoManiLab 对齐，像 sonic 一样新增 holomotion controller。因此最终入口应是同级 `--backend holomotion`，复用既有模型服务、评测编排和 benchmark host；不采用包装 SIMPLE evaluator 的迁移路线。当前已按此边界实现首版，类名采用用户指定的 `HolomotionV14Backend`。

## 调查范围与版本

| 仓库 | 调查版本 |
|---|---|
| LocoManiLab | `306d62dc9a271f717d56a41be52fd777b244cb6d` |
| SIMPLE | `ab818408b10efa5ea150e1453ef54e6bf4d6abee` |
| RoboOrchard | `dae46e426e06e9813cfd07c40b6365084f385afd` |
| g1_backend | 项目 gitlink `25d262a56a6f034e679c0f4bc6be63585233ad20` |
| humanoid_benchmark | 项目 gitlink `2f96b798aa85e32444db582f4ca9bfdefe79c7f3` |

SIMPLE 存在第三方子模块工作区变化，RoboOrchard 存在未跟踪的临时目录；未修改它们。本调查以当前读取的项目源码为准，并未验证完整运行环境与这些 HEAD 完全可复现。

## 1. 原有链路实际如何工作

### 1.1 需要区分的三个适配方向

| 路线 | 模型输出 | 控制器输入 | 定位 |
|---|---|---|---|
| SIMPLE `vla_bridge.py` | decoupled-WBC 36D | 79D | 由固定腿姿和速度积分构造参考，源码明确称为 pipeline check |
| RoboOrchard `holomotion_52d_adapter.py` | 52D 物理动作 | 79D Dex3 reference | 旧的 canonical body/hand/root 契约 |
| RoboOrchard `holomotion_151d_adapter.py` | `[30,151]` unified-v1 | `[50,87]` Revo2 reference | 当前读取的 Revo2 server 强制采用的契约 |

36D bridge 不能当作现有模型能力的等价转换：它补造腿部参考、只构造 yaw 姿态，并不恢复训练时完整运动参考。

还存在命名滞后：`submit_96d_simple_evals.py` 中函数和选项说明仍称 `unified_v0/revo2_v0`，但它启动的 `serve_qwen3_151d_holomotion.py` 的 `validate_contract()` 要求 `action_contract=v1`、162D state 和 episode-relative root rotation。迁移必须以 checkpoint 的有效配置和 server 校验为准，不能从文件名或提交脚本标签推断版本。

### 1.2 Revo2 模型输入

SIMPLE 的 `HoloMotionV14VlaAgent._query()` 发送 HTTP `/act` 请求，包括 `image`、`instruction`、`state`、`history` 和 dataset。

| 字段 | 当前原生链路语义 | 新平台要提供的内容 |
|---|---|---|
| `image.observation.images.ego_view_left` | HWC、uint8 图像 | 对齐训练的左眼视图、光学参数、裁剪和颜色约定 |
| `instruction` | 非空任务文本 | 实际 episode 的训练兼容指令 |
| `state.observation.state` | 51D 实测关节 | body29 + left hand11 + right hand11，按名字映射 |
| `state.observation.base_pose` | xyz + wxyz，共 7D | pelvis/root 的世界位姿，不能使用 torso 替代 |
| `state.observation.base_vel` | 基座速度 | 通用请求携带；当前 151D `act()` 未直接消费 |
| `teleop.latest_obs` / `policy.mode` | 最近参考及控制模式 | 通用 agent 携带；不等于当前 VLM 必需输入 |
| `history` | session_id、episode_index、step_index、frame_index、首帧 reset | 明确区分仿真 step 与参考帧序号 |

`Revo2InputProcessor` 将 51D 实测状态变成 162D：17D upper-body state、64D FAAS hand slots、17D body mask、64D hand mask。身体用 π 缩放，手使用 tokenizer 的 q01/q99 和有效槽 mask。它还执行图像通道反转，并构造训练规定的 setup/state/prompt。首版应完整复用该 processor，避免在仿真端重新实现归一化、通道变换或 prompt 模板。

当前 server 维护每个 session/episode 的初始 heading 和最后一次请求 step；首个请求必须 step=0，后续 offset 必须在 1..38 内。简单地只调用本地 `driver.reset()` 不足以表达全部 episode 语义。

### 1.3 模型输出与参考转换

151D 分组为 `eef48 + body29 + hand64 + kinematics10`；tokenizer 契约要求每 chunk 318 action tokens，30 帧、30 Hz、raw physical 输出。当前 controller adapter 使用 body、hand 和 kinematics；eef48 不直接送入控制器，但仍属于模型/tokenizer 契约。

输出 87D 为：

```text
[0:29]   body joint positions
[29:58]  body joint velocities
[58:61]  root world xyz
[61:65]  root quaternion wxyz
[65:76]  left Revo2 targets
[76:87]  right Revo2 targets
```

转换不是普通 reshape：按 joint names 重排身体；FAAS 有效控制槽映射为每手 11 关节并补耦合、限幅；body/hand 插值；rot6d 转 quaternion 并 SLERP；恢复 episode initial heading；从本次观测的 root XY 出发积分速度，并处理 root 倾斜和高度变化；最后求身体参考速度。

unified-v1 监督时间是 t=0..29/30 秒，输出以 t=0 开始的 50 Hz 参考。不能把旧 52D 路线从 +1/model_hz 开始的时间约定套过来。

### 1.4 执行窗口与控制器

`Holomotion52DSimpleAgent` 虽然保留旧名字，也承担 Revo2 执行：

- controller/reference 均为 50 Hz，reference delay=0。
- 每次模型请求得到 50 帧参考；实际执行 16 步，即每 0.32 秒重规划。
- 每个 tick 控制器看到当前帧和未来 10 帧；第一次先装入 11 帧，以后补一帧。
- 每次重规划替换预测参考；断言 controller 实际消费的 frame index，拒绝意外 freeze。
- 手部目标使用当前参考帧，而非未来第 10 帧。
- 先在固定物理观测下完成 controller-only 初始化，然后第一步进入 motion；初始化调用不推进物理。
- invalid response 在专用 agent 中结束本 episode；通用 v14 agent 的重试/冻结行为与它不同。

`sim_node.py` 提供较清晰的 runtime 接缝：`feed_lowstate(q,dq,quat,gyro)`、`feed_reference(obs65,frame,stamp)`、`feed_pico()`、`run()`，返回 `action_target/action_kps/action_kds`。最终仍由 host 的 PD 执行，不是把 VLA 87D 直接写成 torque。

源码中的仿真时钟隔离 inference 墙钟延迟，否则一次慢模型请求可能触发 stale reference。迁移时建议通过 host clock 明确注入时间，避免依赖 SIMPLE 模块名的全局 time proxy。

## 2. LocoManiLab 的现状与缺口

现有链路是 `ModelPlugin._sim_server()` → server 进程 + `runtime/sim_client.py` → `sim2real.vla.VlaDriver` → `composite.set_vla_action()` → backend → PD。

| 层 | 已有能力 | HoloMotion 缺口 |
|---|---|---|
| 编排 | 独立模型环境、server 生命周期、episode/video/results | RoboOrchard 模型 manifest/launcher、契约预检 |
| 传输 | ZMQ REQ/REP、msgpack-numpy、get_action/reset | HTTP `/act` 的输入输出结构不同 |
| 观测 | video/state/language；默认 43D Dex3 分组 | Revo2 state51、base_pose7、history、左眼相机 |
| chunk driver | 同步/异步重规划、逐帧动作、可选 EMA | 当前+10帧的参考窗口交付与重规划原子替换 |
| backend registry | sonic/amo/heft/scalebfm/almi | 无 holomotion；需新增 runtime/backend/config/layout |
| host | 身体 29D、PD、hand loop、sim clock 接口 | 验证并扩展 11+11 Revo2 手部与本体资产 |
| benchmark | robosuite/robocasa 环境、任务成功判断 | 原 SIMPLE 任务/机器人/场景不能自动等价 |

`g1_backend` 的 `Sim2RealBackend` 已有 `build_tracking/on_load/step/vla_step/vla_reset` 等生命周期钩子；`step` 返回 body29 的 target/kp/kd。这是接 HoloMotion 的合理位置。但默认 `vla_hand_targets`、host hand-state 文档仍围绕 7+7 Dex3，不能只重写 backend 一处就假设 Revo2 完成。

`VlaDriver.step()` 对外只返回已经索引的单帧动作；底层有完整 chunk，但当前 public handoff 没有未来窗口。应新增明确的 reference-window 路径或窄接口扩展，而不是让单帧 adapter 猜未来动作。

当前仿真客户端默认 `robot0_head_camera`、320×240，并把 ego 图像上下翻转后传入。仅把 `ego_view` 重命名为 `ego_view_left` 无法保证与 HBVCAM 相机等价。

## 3. 推荐实现边界

### LocoManiLab

- 新增 `models/roboorchard.py` 和对应 manifest/guide，首版提供 eval；未实现的数据转换/训练阶段明确拒绝。
- 工作区配置保存 repo、模型环境、saved job、checkpoint、tokenizer、HoloMotion bundle 和 robot asset 位置；recipe 不包含机器路径。
- 增加轻量契约预检：模型 action_contract、robot/hand variant、controller bundle hash、30/50 Hz、16步/10帧窗口。
- 扩展 eval runtime 的观测、history/reset 和 reference-window 调度；CLI 主进程不 import torch、ONNX 或仿真模块。
- 结果记录软件 revision、模型/控制器/资产身份、任务设置、初始化来源、失败原因、参考索引和时钟。

### g1_backend

- 新增 HolomotionV14Backend、延迟导入注册项、controller 配置和 policy manifest。
- 复用/独立封装有明确 provenance 的 HoloMotion policy runtime；身体 q/dq、IMU 与参考帧按 joint names 对齐。
- 复用现有 host 的 target/kp/kd 和 clock 接口；每 episode 清空 reference queue、policy history、模式状态和上一 chunk。
- 为 reference window 定义独立 command space，例如拟议 `holomotion_reference_v1`，携带时间和 frame index；不要伪装成现有 `ref_qpos`。
- 明确 hand-layout descriptor，让 Revo2 使用 11+11 state/target，并覆盖 host 的读取、限幅、PD 和记录路径。
- 采集/训练不是首版前提，但录制 schema 应从首版保留 reference、实际状态和动作身份，便于回归。

### humanoid_benchmark

- 接入匹配的 G1/Revo2 本体、背包质量和惯量、碰撞体、传动与相机定义。
- 在 composite/host 层接 backend 输出、身体 PD 与 22D hand targets。
- 提供可靠的 pelvis xyz/wxyz、body q/dq、IMU、hand state 和图像。
- 按名称迁移初始化；不同 MuJoCo model 的 raw qpos/qvel snapshot 不能直接复制。
- 单独落实任务物体、目标、成功条件、时间预算和随机种子；新任务上的结果应标记为新 benchmark。

### RoboOrchard

首版保留现有 `serve_qwen3_151d_holomotion.py` 及其 decoder/adapter。用模型环境内的兼容 server 暴露 LocoManiLab 协议并调用同一 policy 对象，或用独立 ZMQ→HTTP bridge 过渡。前者少一跳，后者便于与原 server 对照；两者都必须显式映射 reset/history 与 metadata，不能仅转发动作数组。

不应复制训练框架到 LocoManiLab，也不应自动把任意 SONIC/AMO checkpoint 的输出转成 HoloMotion 参考。控制器兼容性属于 checkpoint 的动作语义。

## 4. 已确定的迁移路线

**不采用外层包装**：不以 LocoManiLab 启动 SIMPLE evaluator 作为交付。SIMPLE 只用作源码参考和数值/行为对照基准，不成为新平台的必需评测 host。

**采用同级 backend 接入**：按上述三仓库边界实现 controller、host、观测与窗口，新增与 SonicBackend 同级的 HolomotionV14Backend。保持现有 eval→VlaDriver→composite→backend→PD 的主链路，通过明确的 reference-window 能力扩展承载 HoloMotion 的未来参考，而不是另起一套评测循环。模型专属的 151D 解码转换留在模型 runtime；controller 通用的 65D reference 消费和 hand targets 执行归 backend/host。先用原 SIMPLE 链路作为对照基准，按下列验收逐层替换。

## 5. 分阶段验收

1. **冻结契约**：选定一套实际 saved job/checkpoint/tokenizer/source；记录 unified-v1、state162、318 tokens、Revo2、本体和 controller bundle hash。拒绝 v0/v1 混用。
2. **CPU adapter 等价**：对同一观测和已解码动作，比较原/新路径的归一化状态、87D reference、坐标/四元数、frame indices、16步重规划和 episode reset。
3. **controller-only**：用记录参考驱动同一个 HoloMotion 模型，比对 policy observation、raw ONNX output、target/kp/kd；不先加入 VLA。
4. **物理执行**：静止参考稳定性、移动/转身/抓手参考、跨 chunk 与跨 episode reset；检查无额外物理初始化步、无 frame freeze/off-by-one。
5. **模型开环**：固定图像/state/prompt 请求旧新 server；比较 processor 输入和 token/decoded reference。固定 seed，并记录数值误差容限。
6. **闭环小样本**：单任务、固定初始条件、少量 episodes；记录 success、fall、timeout、protocol error、tracking error、重规划次数及延迟。
7. **评测与包装**：固定 seeds/任务分布做回归；验证 wheel 在源码目录之外运行且包含 runtime/YAML；再扩大任务覆盖。

无 GPU 的 CPU 契约测试不代表 VLA 推理质量；controller-only ONNX 测试不代表物理稳定；仿真成功不代表硬件支持。

## 6. 已做验证与待确认项

执行了 RoboOrchard 既有 `tests/test_projects/test_holomotion_52d_adapter.py`，使用 SIMPLE 的现有 Python 环境并关闭外部 conftest 加载：**5 passed**。覆盖 52D/151D 转换、当前参考时间语义、真实 reference queue 的 16 步重规划和 session reset。测试用替身隔离 ONNX/physics，不能视为完整模型闭环。

初始调查没有修改实现；随后用户授权实施，已在 LocoManiLab 及所需子模块中实现，测试依赖隔离放在 tmp/。当时未修改 SIMPLE 或 RoboOrchard，也未进行 GPU 验证；后续真实模型验证与 AIDI 接入另见 `holomotion_revo2.md` 和 `aidi_roboorchard_eval.md`。

调查开始时两个运行时子模块均未初始化；现已取得两者锁定版本并完成源码核实和实现。初始实现阶段未更改 gitlink、未初始化不相关依赖；内部交付版本将两个子模块分别提交到依赖分支，并在父仓库锁定其提交。

实施前仍需明确：首个实际 checkpoint/artifact 集合、目标是旧任务等价复现还是新平台任务泛化、可用的 Revo2/相机/场景资产。默认建议优先原生 Revo2 unified-v1，先固定一组任务，分开报告接口等价与任务成功率。

## 7. 关键源码索引

LocoManiLab：`DESIGN.md`、`docs/controller.md`、`src/locomanilab/workflows/evaluation.py`、`src/locomanilab/common/sim.py`、`src/locomanilab/runtime/sim_client.py`、`src/locomanilab/plugin.py`。

g1_backend：`g1_bridge/backends/base.py`、`g1_bridge/backends/__init__.py`、`g1_bridge/session.py`、`g1_bridge/modality.py`、`sim2real/vla/{client,driver,observation,adapters}.py`。

SIMPLE：`src/simple/cli/eval_holomotion_v14.py`、`src/simple/agents/holomotion_v14_agent.py`、`src/simple/agents/holomotion_v14_vla_agent.py`、`src/simple/teleop/holomotion_v14/{sim_node,sim_clock,robot_variants,eval_setup,vla_bridge}.py`、同目录 `SOURCE.md`。

RoboOrchard：`projects/holobrain_internal/humanoid/scripts/{serve_qwen3_151d_holomotion,holomotion_151d_adapter,holomotion_52d_adapter,holomotion_52d_simple_agent,revo2_holomotion_eval_adapter,submit_96d_simple_evals}.py`、`tests/test_projects/test_holomotion_52d_adapter.py`。

HoloMotion vendored source 的来源 commit 和 bundle lock 必须保留；不同仓库、权重、机器人 mesh 和数据各自沿用原许可。本文不构成任何重新授权或公开分发判断。


## 8. 当前实现与验证结果

- `HolomotionV14Backend` 已注册为与 Sonic 同级的 backend；保留冻结 HoloMotion runtime、ONNX 输入构造与 reference queue，新增 host 时钟、具名关节映射、reset 和参考窗口消费。
- `VlaDriver` 增加独立 `holomotion_reference_v1` 契约；50 Hz、每 16 步更新、未来 10 帧、87D 数据校验，避免落入通用单帧/EMA 路径。
- benchmark 新增 `G1Revo2`，XML 与引用 meshes 自 SIMPLE 迁入，具名 11+11 手部 PD 使用源标定。修复 robosuite 合并时丢弃 `<extension>` 的问题；零质量腕部相机支架显式声明零惯性以兼容 MuJoCo 3.3.1。
- LocoManiLab 新增 roboorchard 模型插件和独立 ZMQ→HTTP runtime，复用原生 unified-v1 server；sim client 提供 51D 实测状态、7D base pose 和左头部相机。
- 顶层定向测试：97 passed、1 skipped；benchmark 分发/机器人与 HoloMotion driver/队列测试：16 passed。
- MuJoCo 3.3.1 + NumPy 2.2.5：一个代表性 EmptyEnv reset + 17 步仿真控制通过（替身模型，覆盖两次 chunk），未渲染相机。
- LocoManiLab、g1_backend、robosuite wheel 已构建并安装到源码目录以外，验证模型注册、server/config/vendor 资源和 G1Revo2 资源编译。
- public-source 检查、定向 lint、diff 检查、g1_backend 的离线 lock 一致性检查通过。

后续已使用用户指定 checkpoint 和本地 ONNX bundle 完成 EmptyEnv 连续调用；固定直立参考可站立，VLA 空场景输出导致摔倒，离线资产到位后，已执行三个抓取任务各一次：全部无运行错误，但均未成功，其中一次摔倒。不能声称真实任务端到端验收通过。仍需进行同输入 ONNX 输出对齐和真实任务 rollout；真机与遥操作采集不在本次已实现能力内。具体配置、入口和验收顺序见 [接入说明](holomotion_revo2.md)。
