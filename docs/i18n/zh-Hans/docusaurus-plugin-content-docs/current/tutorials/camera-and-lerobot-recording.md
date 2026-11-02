# Camera and LeRobot Recording

这个教程给一次 policy 运行接上相机，并把 state/action/image 录制成
[LeRobot](https://github.com/huggingface/lerobot) v2.1 格式的 episode，适合用于 VLA
训练数据采集。它建立在 [Pico Teleoperation](./pico-teleoperation.md) 教程的基础配置上。

只用 `--record`（不加下面任何 flag）时，录制行为和这个功能上线之前完全一样：整个进程录成
一个扁平的 `.npz`，没有图像。下面的所有能力都是可选启用的。

## 1. 安装可选依赖

```bash
uv sync --extra camera --extra lerobot
```

## 2. 启动相机 server

```bash
uv run python -m sim2real.camera.server --cameras.ego_view.source realsense
```

如果用普通 USB 摄像头而不是 RealSense，改成
`--cameras.ego_view.source cv2 --cameras.ego_view.device 0`。server 默认发布在
`tcp://*:5594`（`PORTS["camera"]`）。

## 3. 开启 LeRobot 录制跑 policy

```bash
uv run sim2real/rl_policy/tracking.py \
  --controller pico \
  --record --record-format lerobot \
  --camera-connect tcp://127.0.0.1:5594 \
  <其他 tracking 参数，参见 Pico Teleoperation>
```

如果想把 PICO trigger 派生的手部抓握指令也录成 `action.hand_grip`，再加一条
`--hand-grip-connect tcp://127.0.0.1:5593`。

dataset 的创建会推迟到第一帧所有已启用的信号（相机、手部抓握）都真正产出数据为止，这样
schema（包括图像 shape）是自动探测出来的，不需要提前声明。

## 4. 录制 episode

- **键盘**：`c` 开始/停止，`x` 丢弃。只有这一条路。

Pico 头显上没有录制手势。`Y` / `Y+A` 曾经是，已经删掉了：操作员整场手都在摇杆上，`Y` 就在
A+B+X+Y engage 和弦里，蹭一下就开录、只在滚动日志里留一行绿字，再蹭一下就存盘。
把"悄悄改变落盘内容"的那个动作绑在一个裸上升沿上是个坑——见 `g1_bridge/hosts/real.py`。

这和下游数据采集 fork（比如 `heft`/`AMO`/`sonic` controller）已经在用的开始/停止/丢弃约定
完全一致，操作习惯可以直接复用。

## 输出

在 `--record-output` 指定的路径（或当前目录下一个带时间戳的默认路径）下会生成一个
LeRobot v2.1 dataset 目录（`meta/info.json`、`meta/modality.json`、
`meta/episodes.jsonl`、`data/`、`videos/`）。`observation.*` 字段直接来自这个 policy
family 自己声明的 `observation` 分组（`policy_config["observation"]`），再加上基线机器人
状态（`joint_pos`/`joint_vel`/`root_quat_w`/`root_ang_vel_b`）；`action.q_target`/
`action.raw` 来自控制循环本来就会算出来的量。这些都不需要写任何 per-family 代码——只有
相机/手部抓握字段需要显式加 flag，因为它们不在 `Observation` 注册表里。
