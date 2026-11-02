# Pico Teleoperation

这个教程使用 teleop publisher 提供实时 Pico / XR retarget，用它内置的 mjviser server 检查 retarget 结果，再用 root project 的 tracking policy 做执行。

## 1. 启动 Pico retarget publisher

```bash
uv run --project venv/pico sim2real/teleop/pico_retarget_pub.py
```

打开 publisher 打印出来的 mjviser URL。先确认 viewer 里的 G1 retarget 动作是对的，再继续执行。

## 2. 选择执行后端

### Sim2Sim

启动 MuJoCo 执行进程：

```bash
uv run sim2real/sim_env/base_sim.py
```

在另一个终端，把 tracking policy 接到实时 motion stream：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/heft/pmg/policy.yaml \
  --motion-backend zmq \
  --controller pico
```

想在这个 MuJoCo 窗口里同时看到 Dex3 手指跟着真实物理接触张合（不需要真的
Dex3 手或 DDS 网络——见下面"Dex3 灵巧手控制"一节），给 Sim2Sim 进程加一个
flag：

```bash
uv run sim2real/sim_env/base_sim.py --with-dex3-hands
```

tracking policy 的命令不用变——仿真直接读同一个 `hand_grip` 数据流，和
`--dex3-hand-dir` 无关。

### Sim2Real

上真机前，先在 [Robot I/O](/reference/robot-io) 里选择部署路径。Pico 相关的 policy 参数保持一样：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/heft/pmg/policy.yaml \
  --motion-backend zmq \
  --controller pico
```

只额外加你选择的 robot I/O 模式真正需要的 flag 或 bridge 进程。

## Pico 按键

- 按 `A` 进入 init pose。
- 同时按 `A` + `B` 进入 policy mode。
- 按 `X` 解除 motion flow 暂停。

## 可选 DH116S 手部集成

hand-grip 流可供外部 DH116S 适配器消费，但本 checkout 不包含 `venv/dh116s` 或
DH116S 控制脚本。需要此硬件时，请按 SDK 提供方说明在隔离环境安装。本仓库维护的
手部路径是下方的 Unitree Dex3。

## Dex3 灵巧手控制

如果你的 G1 装的是 Unitree 官方 Dex3 灵巧手而不是 DH116S，手部控制**直接跑在
tracking policy 同一个进程里**——不需要另外起一个手部进程。它读取的是完全
相同的、TCP `5593` 端口上的手部 grip 数据流（左扳机控制左手 Dex3，右扳机
控制右手 Dex3），但走的是和臂/腿控制相同的 DDS 总线
（`rt/dex3/{left,right}/cmd`/`state`），不需要单独的厂商 SDK。

和 DH116S 不同，Dex3 不需要额外下载 SDK，也不需要单独的隔离环境：
`unitree_sdk2py`（本身就带 Dex3 的 `HandCmd_`/`HandState_` IDL 类型）就是
`robot-g1` extra 里给臂/腿控制装的那一份。只要你已经能往真机 G1 上部署，
Dex3 控制需要的东西就都已经有了。

在第 2 步的 tracking policy 命令里只加一个 flag，`--dex3-hand-dir
double`——启动命令本身不用变：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/heft/pmg/policy.yaml \
  --motion-backend zmq \
  --controller pico \
  --dex3-hand-dir double
```

和不控手时的命令相比就只差这一个 flag——其余都用常见场景的默认值：
手部 grip 数据流默认从本机的 Pico publisher 读取
（`tcp://127.0.0.1:5593`），只有 publisher 跑在别的机器上才需要加
`--hand-grip-connect tcp://<publisher-ip>:5593`；DDS domain/网口默认是
`0`/`eth0`，只有 G1 臂/腿控制用的不是这套默认值时才需要加
`--dex3-domain-id`/`--dex3-interface`。同时开启 `--record-format lerobot`
录制时，手部完整的 7×2 关节目标值/测量值
（`action.hand_joints`/`observation.hand_state`）会自动被记录——不需要
再额外加 `--record-extra dex3`。

在 Sim2Sim 或者没有真实 Dex3 手（也没有 `eth0`/对应 DDS 网口——否则
`ChannelFactoryInitialize` 会报"找不到网口"）的开发机上测试，加
`--dex3-dry-run`，把扳机值映射成关节角度并录制下来，但完全不碰
`unitree_sdk2py`/DDS：

```bash
uv run sim2real/rl_policy/tracking.py \
  --policy-config checkpoints/heft/pmg/policy.yaml \
  --motion-backend zmq \
  --controller pico \
  --dex3-hand-dir double \
  --dex3-dry-run
```

:::warning
tracking 进程一启动，手就会立刻被 enable 并按扳机抓握姿态驱动。启动前必须
清空双手周围空间。`--dex3-hand-dir` 和 `--robot-io inline` 一起用时，还没有
验证过是否会和 `robot_io/g1_debug_mode.py` 里的 debug-mode 握手产生 DDS
participant 冲突——默认的 `--robot-io zmq` 已确认安全。
:::

每个 tick 的手部指令都会按上一次测得的关节角度做限幅
（`--dex3-max-delta-q`，默认 `0.25` 弧度），避免 grip 数据流抖动导致手指
突然大幅度移动。grip stream 本身断流时，手只是不再收到新指令，保持最后一次
下发的姿态。

如果只是想做手部硬件的独立联调/冒烟测试，而不想启动完整的 tracking
policy，`scripts/dex3_control.py` 仍然可以单独跑，同样使用 `robot-g1` extra 环境：

```bash
uv run --extra robot-g1 scripts/dex3_control.py \
  --hand-dir double --connect tcp://<pico-publisher-ip>:5593
```

加 `--dry-run` 可以只验证 ZMQ 数据流和 grip→关节角度映射，不 import
`unitree_sdk2py`、不移动硬件。

### 在 Sim2Sim 里看到 Dex3 手动

上面这些（`--dex3-hand-dir`、`--dex3-dry-run`）驱动的都是真实/dry-run 的
Dex3 *driver*——完全不碰 sim2sim 的 MuJoCo 模型，而这个模型默认没有手指
关节。想真的在 Sim2Sim 窗口里看到（并且物理接触到）Dex3 手，改成给
`base_sim.py` 加 `--with-dex3-hands`（见上面第 2 步）：

```bash
uv run sim2real/sim_env/base_sim.py --with-dex3-hands
```

这会把 Dex3 手部 body/joint/碰撞几何拼接到两只手腕上
（`sim2real.sim_env.utils.dex3_mjcf.add_dex3_hands`，替换掉原来装饰性的
`left_rubber_hand`/`right_rubber_hand` mesh），并且用本地 PD 控制、从
`pico_retarget_pub.py` 已经在发布的同一个 `hand_grip` ZMQ 数据流驱动手指——
和 tracking policy 进程用不用 `--dex3-hand-dir`/`--dex3-dry-run` 完全无关。
不需要真实 Dex3 手、DDS，也不需要网口，纯 MuJoCo 物理。

:::warning
PD 增益默认就是 `sim2real.utils.dex3` 里给真机调的那套值（heft 的
Dex3HandBridge 参数）——在参考资产的执行器扭矩上限下，大概 1 秒左右能收拢到
闭合姿态，但会有一点过冲/回弹（真实电机自带的阻尼，仿真的 PD 没法完全复现）。
如果这个不合你的需求，用 `--dex3-kp`/`--dex3-kd`/`--dex3-thumb-kp`/
`--dex3-thumb-kd` 覆盖掉。
:::

## Notes

- `pico_retarget_pub.py` 发布实时 motion stream 给 tracking policy 使用，并自己创建 retarget mjviser server
- hand grip 使用独立的 ZMQ 端口，不会改变现有 Pico 按键协议。DH116S 和 Dex3 消费的是同一个 `hand_grip` 端口/消息格式，区别只在手侧的消费方式不同（DH116S 由外部适配器提供；Dex3 通过 `--dex3-hand-dir` 跑在 tracking 进程内；仿真手部通过 `base_sim.py --with-dex3-hands`）。
- `sim2real/sim_env/base_sim.py` 是 sim2sim 的执行后端
- 真机部署时，[Robot I/O](/reference/robot-io) 里列出了 inline 和 bridge 两类方式
- 如果 publisher 和 policy 跑在不同机器上，再加 `--motion-zmq-connect tcp://<publisher_ip>:28701`

## Next Steps

- [Motion Recording](./motion-recording.md)
