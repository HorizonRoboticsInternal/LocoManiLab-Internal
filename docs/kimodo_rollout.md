# 在 main 上跑 Kimodo rollout 采数据

**Kimodo**（运动扩散模型）生成一段参考动作 → **冻结的 HEFT tracker** 在 robosuite 里执行它 →
机器人在 robocasa 场景中完成任务 → **通过门控的回合**导出成 LeRobot 数据集。

这条线在 main 上**只有采集**。训练侧（DDPO/LoRA 微调）、24 个验证脚本（h0–h31）、
研究记录不在本仓库中。

```
Kimodo (CUDA, torch)  ──[36 维参考, 30 Hz]──>  heft_bridge 重采样到 50 Hz
                                                      │
                        robosuite env.step() ←── HEFT tracker (ONNX, CPU)
                                │
                    每 tick 记录 ──> 门控 ──> LeRobot 数据集 + 离线渲 ego 视频
```

---

## 1. 环境

### 1.1 你需要三个仓库

| | 是什么 | 怎么给到 |
|---|---|---|
| **本仓**（`tele_humanoid_benchmark`） | 场景、任务、rollout 代码 | `pip install -e robosuite_src/` + `pip install -e .` |
| **`heft_work`** | `kimodo_rl` 包:生成器 `gen.py`、`actor_kimodo`、`text_cache`、`eef_goals`、`prompts` | `export HEFT_WORK=/path/to/heft_work` |
| **`g1_backend`** | `g1_bridge` 包:数据集列定义、`bydmimic_utils` 关节序 | `export G1_BACKEND=/path/to/g1_backend`,并让它可 import |

⚠️ **`kimodo_rl` 是 import 进来的,不是重写的**。权重就是在那份代码上训出来的,
所以在这里跑出的数和训练时的数才可比。h7 会把 `$HEFT_WORK` 插进 `sys.path`;
路径不对会在启动时直接报错并告诉你该设什么,而不是等到 `import kimodo_rl` 才崩。

⚠️ **数据集的列定义以 `g1_bridge.recording.features` 为准**,和真机采集线同源。
`export.py` 会断言关节分组按**名字**对得上,对不上就报错。

### 1.2 Python 环境

用 `kimodo_rl` 那个 conda 环境（robosuite 从 `robosuite_src/` 装、mujoco 3.3.1、
onnxruntime、torch）。

⚠️ **额外四个包必须补,而且必须 `--no-deps`**:

```sh
pip install --no-deps "any4hdmi @ git+https://github.com/EGalahad/any4hdmi.git@main"
pip install --no-deps "mjhub @ git+https://github.com/EGalahad/mjhub.git@main"
pip install --no-deps tensordict pyvers
```

**为什么**:`import robocasa` 会拉起整条实机链——`robocasa/__init__.py` 无条件
import `robocasa.g1_host.wbt`,它在模块层 import `g1_bridge.sonic_utils.kplanner`,
那个包的 `__init__` 又 import `sim2real.rl_policy.observations` → `any4hdmi` →
`tensordict` → `pyvers`,`any4hdmi` 再要 `mjhub`。跑 heft 一个都用不上,
但少一个就 import 不进来。

`--no-deps` 是硬要求:这四个只是为了满足 import,**不该动环境里的 torch**
（装完确认 `torch.__version__` 没变）。

⚠️ `robocasa/__init__.py` 里那句注释写着 "Imports no sim2real: ... imported lazily
by its callers",**与实际不符**。这条线没有绕过它,只是补齐了依赖。

### 1.3 三样不在仓库里的产物

`rl/kimodo/_out/` 是 gitignored 的,下面这些**必须从别处拷贝或下载**:

| 产物 | 默认位置 | 大小 | 从哪来 |
|---|---|---|---|
| **HEFT tracker 策略** | `rl/kimodo/_out/heft/heft_policy_loc.onnx` | 24 MB | ⚠️ 见下 |
| tracker 参考输出（可选,自检用） | `rl/kimodo/_out/heft/heft_policy_ref.npz` | 0.45 MB | 同上 |
| **Kimodo 底模** | `~/.cache/huggingface/hub/models--nvidia--Kimodo-G1-RP-v1` | 1.1 GB | HuggingFace,首次自动下（要能联网/挂代理） |
| **文本编码缓存** | `$HEFT_WORK/assets/text_cache.pt` | 0.35 MB | heft_work 仓库自带 |
| Kimodo 微调权重（可选） | 任意 `.pt` | 1.1 GB / 个 | 训练产出;不给就跑底模 |

⚠️ **tracker ONNX 在 main 上没法自己生成**。它由 `rl.kimodo.heft.export_policy`
从 HEFT 的 `policy.pt` 导出,而那个脚本**属于验证侧、没有搬到 main**,
且要用 `motion_tracking/.venv` 的 torch 2.10 才读得动那个 pickle。
**在 main 上就是拷贝一份过来**,或者:

```sh
export HEFT_POLICY_ONNX=/path/to/heft_policy_loc.onnx
```

找不到时的报错会把这两条路都写出来。

### 1.4 场景资产

iThor 房型（`FloorPlan1–30` 厨房 / `201–230` 客厅 / `301–330` 卧室 / `401–430` 浴室）
是 gitignored 的下载物。缺哪一间,用到它的任务就建不起来——h7 会**重采样最多 6 次**
再放弃,每次都打印,一个建不起来的场景只赔掉一个回合而不是整轮。

### 1.5 一次性自检

```sh
export HEFT_WORK=/path/to/heft_work
export G1_BACKEND=/path/to/g1_backend
export PYTHONPATH=$G1_BACKEND:$PYTHONPATH

python -c "
from rl.kimodo.heft.scene_track import KimodoHeftTracker
tr = KimodoHeftTracker(env_name='IThorBathroomApproachToilet', seed=0, scene_ids=[401])
print('root =', tr.root_body, ' offset =', tr.root_offset, ' ctrl =', tr.composite._ctrl_type)
"
```

期望输出:

```
[scene_track] root=mobilebase0_support floor_z=0.0000 origin_xy=[...] sole_clearance=+0.0000 m
root = mobilebase0_support  offset = [0. 0. 0.793]  ctrl = heft_tracker
```

`root` **必须**是 `mobilebase0_support`、偏移 **0.793 m**。这一步在验证 robosuite 的 G1
和 HEFT 训练用的 `g1_5.xml` 是同一条运动链（容差 50 mm）;不通过会直接抛 `SystemExit`
并列出最接近的五个候选,**不会**退而求其次挑一个最像的。

⚠️ **`robot0_pelvis` 不是解剖学骨盆**。`G1FloatingBody` 挂在 `FloatingLeggedBase` 上,
那个 body 在地板高度,真正的骨盆是它上面 0.793 m 的 `mobilebase0_support`。
用错会让策略以为机器人一直蹲着 79 cm,而且**不报任何错**。

---

## 2. 跑一次

```sh
CK=rl/kimodo/_out/heft/ckpt/p41c_hands_kimodo_best.pt   # 不给 --ckpt 就是底模

python -m rl.kimodo.heft.scripts.h7_scene_rollout \
    --env IThorBathroomApproachToilet --ckpt $CK \
    --task-set 48 --episodes 10 \
    --save-data /path/to/dataset
```

单回合 ≈ 10–15 s（RTX 4090;3 s 一个 chunk,一个航点最多 4 个 chunk）。

### 2.1 最常用的开关

| 开关 | 作用 |
|---|---|
| `--env NAME` | 任务类名。`--list` 列出所有已接的 |
| `--ckpt PATH` / `--base` | 微调权重 / 显式跑底模。**两个一起给是错误**,不是优先级 |
| `--task-set 48` | 用 48 任务集的 adapter,并把场景**钉死**到该任务的 FloorPlan |
| `--episodes N` | 回合数 |
| `--seed S` | 种子。**它就是场景的身份**——`hard_reset` 下物体是按全局随机流重抽的 |
| `--save-data DIR` | 导出 LeRobot 数据集。**目录必须不存在** |
| `--gate usable\|success` | 导出门控,见 §4 |
| `--no-natural` | 把自然度从门里去掉,见 §4 |
| `--view` | 开实时窗口（R 重来,关窗停止） |
| `--show-success` | 窗口里画到达判定的圈 + 根的实际位置点 |
| `--ego-view` | 只渲第一视角视频,不导数据集 |
| `--video` | 渲第三视角视频 |
| `--out DIR` | 摘要和 state bundle 的落点（默认 `rl/kimodo/_out/heft/h7`） |

⚠️ **`--task-set 48` 要钉场景**。不钉的话每次 reset 重抽房型,那是所有测量里
**最大的噪声源**;两次跑出来的差可能全部来自抽到了不同的房间。

### 2.2 行为调节（有默认值,改之前先看理由）

| 开关 | 默认 | 说明 |
|---|---|---|
| `--seconds` | 3.0 | 一个 chunk 的时长。训练用的就是 3.0 |
| `--steps` | 6 | 去噪步数。训练用的就是 6 |
| `--max-chunks` | 4 | 一个航点最多几个 chunk |
| `--arrive-hold TICKS` | 50 | 到达后**站定**多久（1 s @ 50 Hz）,见 §5 |
| `--hop M` | — | 规划器的步长上限 |
| `--reach-leg M` | 0.35 | 超过这个距离就把"伸手"拆成"走过去 + 伸手" |
| `--root-gain G` | 1.0 | 把根的 ask 放大——tracker 的根平移是**开环**的,只送到约一半 |
| `--success-radius M` | — | 用"根到目标中心的半径"代替任务自己的面到面判据。⚠️ 只改 `arrived`,**不改 `success`** |

### 2.3 批量采集

```sh
python -m rl.kimodo.heft.scripts.h30_collect_sweep \
    --ckpt $CK --episodes 20 --out /path/to/datasets
```

- 每个任务**起一个子进程**:场景建不起来 / 卡死 / OOM 只赔一个任务,不赔整轮,
  而且任务之间**把 GPU 还回去**,一个任务的泄漏累积不到下一个;
- **可断点续跑**,按数据集目录判定:重跑会跳过已经采过的任务,
  也保证一个任务不会被采进两个目录、之后又被合并;
- **报告每跑完一个任务重写一次**,跑到一半就能看;
- 留 **25 GB 磁盘底线**,不够就停下来告诉你;
- `--timeout` 每任务超时,`--report-only` 中途看报告（安全）。

48 任务集:厨房 21 / 浴室 11 / 卧室 9 / 客厅 7,原子 33 / 复合 15,其中**导航类 10 个**。
3 个已知建不起来、6 个有已知问题,记在 `adapters48.CANNOT_BUILD` / `KNOWN_TROUBLE`,
启动时会打印。

---

## 3. 产出什么

```
rl/kimodo/_out/heft/h7/
    <tag>.json                        每回合的判定与逐航点日志
    states/<tag>_ep<N>.npz            仿真状态序列（离线重渲用）
    states/<tag>_ep<N>.mjb            编译好的场景（--keep-model / --save-data 时）
    states/<tag>_ep<N>.heft.npz       该回合的列数据 bundle
    states/<tag>_ep<N>.mp4            --video
    states/<tag>_ep<N>.ego.mp4        --ego-view
<--save-data DIR>/                    LeRobot v2.1 数据集
```

数据集是 **heft 契约**（列定义来自 `g1_bridge.recording.features`）:

| 列 | 形状 | 含义 |
|---|---|---|
| `action.ref_qpos` | 36 | Kimodo 生成的参考:根 3 + 四元数 4 + 29 关节,**dataset 关节序** |
| `action.hands` | 14 | 手指目标 = 抓取扳机 × 合拢目标 |
| `observation.state` | 29 | 机器人实际关节 |
| `observation.eef_state` | 14 | 手指实际关节 |
| `observation.root_orientation` | 4 | 根姿态四元数 |
| `observation.projected_gravity` | 3 | 投影重力 |
| `observation.images.ego_view` | 视频 240×320 | 第一视角,**离线**渲染 |

50 fps。一个 600 tick 的回合就是 600 帧。

⚠️ **三套关节序**:dataset（运动文件、Kimodo、`g1_5.xml`）/ canonical（策略的观测与动作）
/ sim。**永远按名字解析**——按位置索引过一次,1729 维观测里错了 1276 维,
而且看起来只像"建模难"。

⚠️ **ego view 必须离线渲**。一个进程里同时持有 EGL 离屏上下文和 CUDA 上下文
会在多卡机上死锁（`DEBUG_LOG` #11）,所以渲染在**子进程**里从记录的状态重放。

⚠️ **渲染只画 visual geom（group 1）**。MuJoCo 的 `MjvOption` 默认把 group 0
也画出来,而 robocasa 场景里 group 0 是**碰撞几何**——465 个不透明盒子盖住整个房间,
渲出来是一片橄榄色。

---

## 4. 门控:什么样的回合会被导出

**三层,别混**:

**① `success`** —— **任务自己的 `_check_success()`**,与这条管线无关。
例如 `IThorBathroomApproachToilet`:机器人几何与马桶几何的**面到面距离 ≤ 0.1 m**
（`mj_geomDistance`）,且**连续保持 10 个控制步**。

**② `usable`** —— 导出门,四项**全要**:

```python
success and not fell and physics_ok is True and natural_ok is True
```

⚠️ `physics_ok` 是 `None`（筛查没跑）算**不通过**——"没检查"和"检查过没问题"
不能导出同一批数据。

⚠️ **为什么不是"成功就导出"**:`IThorBathroomApproachSink` **成功率 100%、
可用率 20%**——它的判据允许机器人**肩膀插在水槽里**也算到达。按成功导出,
这些会带着"成功"标签进数据集。

**③ 两个筛查的阈值**:

| physics（P3-2） | 阈值 |
|---|---|
| 根下沉（取全程**最小值**,不是末帧） | 0.30 m |
| 倾角 | 45°,且**连续超 30 tick** 才算 |
| 根高绝对带 | 0.35–1.20 m |
| 穿模深度 | 0.02 m |
| 关节越界 | 0.05 rad |
| 不该碰的部位接触 | 5 tick |

| naturalness（DLJ,越大越平滑） | 门槛 | 实测活体 |
|---|---|---|
| `joint_dlj`（29 个指令关节） | ≥ **-16.8** | 中位 -15.3,p05 -15.8 |
| `wrist_dlj`（世界系手腕） | ≥ **-14.2** | 中位 -12.4,p05 -13.2 |

参照:往动作流里注入 σ=0.05 rad 噪声,`joint_dlj` 掉到 **-17.3**。

### 4.1 改门

| 开关 | 门变成什么 |
|---|---|
| （默认） | 四项全要 |
| `--no-natural` | 去掉自然度,其余不变 |
| `--gate success` | 只看任务判据 |

`--no-natural` **仍然测量并记录** DLJ（`natural_stats` / `natural_ok_rate` 照常）,
只是门不再问它。

⚠️ **两种门跑出来的 `usable_rate` 不可比**。摘要里因此有 `natural_gated` /
`physics_gated` 两个字段:同一个字段名在两种门下是两种测量,文件里必须写明是哪种。

### 4.2 被拦下来时

```
[h7] ep 0: NOT exported (naturalness (joint_dlj -16.67, wrist_dlj -14.27))
[h7] ⚠️ NOTHING TO EXPORT: 0 of 1 episodes passed the 'usable' gate, so
     /path/to/dataset was not created.
```

理由带着数字,能直接看出是差了 0.14 还是差了 3。逐回合的完整判定在
`rl/kimodo/_out/heft/h7/<tag>.json` 的 `episodes[i]` 里:
`physics_stats` / `physics_reasons` / `natural_stats` / `success_tick`。

⚠️ **单个回合的自然度说明不了什么**。门槛本身的不确定度和"差 0.1–0.3 nat"
是一个量级;要判断门槛该不该调,得跑够样本看分布。

---

## 5. 几个已经踩过的坑

**到达之后不会再往家具里挤**（2026-08-11 起）。停的时机是**瞬时**的"进入范围",
任务判据自带的 10 步保持改成**站着**攒（`--arrive-hold`,默认 50 tick = 1 s）。
之前那版把保持期走着过完,面到面一路压到 **0.000 m**,画面上就是贴着马桶。
（对应的内部问题记录不在本仓库中。）

**`--save-data` 的目录必须不存在**。LeRobot 拒绝写进已有的 root,而往别人的
数据集里追加不是这个参数的意思。

**中断采集不会留下一地编译好的场景**。`--save-data` 会留 `.mjb`（每个约 170 MB）,
`atexit` + SIGTERM→SystemExit 两半都在才保证杀掉的 sweep 会清干净——
实测曾经九个回合留了 1.1 GB。

**别用 `pkill -9` 杀正在用 CUDA 的进程**,用 SIGTERM。

**训练不渲染**。EGL + CUDA 同进程死锁,见 §3。

---

## 6. 这条线是怎么接到 main 上的

从内部研究分支搬过来时有四处必须适配,都验证过:

1. **HEFT 控制器接缝**。main 把 G1 teleop 栈搬去了 g1_bridge,`heft_tracker.py`
   和 `default_g1_heft.json` 都没了。控制器现在是
   `rl/kimodo/heft/tracker_controller.py`,由 `robocasa/g1_host/wbt.py` 的工厂
   **延迟 import**——它读 `rl.kimodo.heft.*`,放在 `robocasa/` 下会造出
   `robocasa → rl → robocasa` 的 import 环。
2. **自由关节改名**。统一 G1 模型把 `robot0_floating_base_joint` 叫成
   `robot0_pelvis`,按名字查返回 -1,第一次 reset 就崩。现在
   `model_patch.free_joint_id` **按关节类型**找——浮动基座有且只有一个 7 自由度关节。
3. **`g1_5.xml` 内联进仓**。`resolve_root_body` **每次 reset** 都要读它,
   而它原本在已删除的 `heft` 子模块里、还压着 23 MB 网格。
   `rl/kimodo/heft/assets/g1_5_kinematics.xml` 是去掉全部 geom 和 `<asset>` 的版本,
   在 `model_patch` 读的每一项上都相同:36 个 body 的 FK **0.0 m**、
   逐 body 质量 **0.0 kg**（总 33.3411）、30 个关节名同序。要用原件设 `HEFT_G1_5_XML`。
   ⚠️ 目录是 `assets/` 不是 `_assets/`——后者被 `rl/kimodo/.gitignore` 忽略且不限深度,
   放那里会**悄悄漏提交**,别人 clone 下来第一次 reset 就崩。
4. **两个任务改名**:`IThorLivingRoomPillowToSofa` → `BasketToSofa`,
   `IThorBathroomBarSoapShelfPnP` → `CanShelfPnP`。

**移植前后对照**（`IThorBathroomApproachToilet`,`--task-set 48`,同一权重）:

| | 分支（6 回合） | main（6 回合） |
|---|---|---|
| success | 1.00 | 1.00 |
| physics_ok | 1.00 | 1.00 |
| fall | 0.00 | 0.00 |
| natural_ok | 1.00 | **0.833** |
| 面到面最小（3 回合） | 0.050–0.057 m | 0.049–0.069 m |
| ≤0.02 m 的 tick | 0 | 0 |

⚠️ **`natural_ok` 差的是六个里的一个回合,这个数字什么都证明不了**——
既不能说退步也不能说持平。main 同时换了 G1 模型（geom 103→101,
damping/armature/frictionloss 有改动）又加了出生落地对齐
（`settle_robot_spawn`,实测每次下沉 0.03–0.04 m）,两件事都可能影响自然度。
**要下结论得两个独立种子各跑够样本**,现在没有。

`resolve_root_body` 在新模型上仍解析到 `mobilebase0_support`、偏移 0.793 m,
说明新模型的运动链仍在 g1_5 的 50 mm 容差内——**移植本身是成立的**。

---

## 7. 报错对照表

| 报错 | 原因 | 怎么办 |
|---|---|---|
| `No module named 'g1_bridge'` | g1_backend 不在 path 上 | `export PYTHONPATH=$G1_BACKEND:$PYTHONPATH` |
| `No module named 'any4hdmi' / 'tensordict' / 'pyvers' / 'mjhub'` | §1.2 那四个包没装 | 照 §1.2 装,**必须 `--no-deps`** |
| `[h7] no kimodo_rl package under ...` | `$HEFT_WORK` 没设或指错 | 指向 heft_work 的 checkout |
| `[heft_ckpt] no policy ONNX found` | tracker 权重不在 | 拷到 `rl/kimodo/_out/heft/`,或设 `$HEFT_POLICY_ONNX` |
| `Could not resolve model 'kimodo-g1-rp'` | Kimodo 底模没下 | 挂代理让 HuggingFace 能下,或拷 HF 缓存目录 |
| `[model_patch] expected exactly one free joint under 'robot0_'` | 机器人模型换了 | 说明这个 G1 变体不是这个端口认识的,别绕过去 |
| `[model_patch] no body frame reproduces g1_5's kinematic chain` | 机器人和 g1_5 不是同一条链 | **不要**放宽容差;先查是不是换了机器人变体 |
| `[scene_track] composite is 'sonic_planner', not 'heft_tracker'` | 加载了错的控制器配置 | 走 `KimodoHeftTracker`,它自己会加载 `default_g1_heft.json` |
| `--save-data ... already exists` | 目录已存在 | 换一个新目录 |
| `NOTHING TO EXPORT` | 门控全拦下了 | 看 §4.2 的理由,再决定调门还是调模型 |
| 场景反复 `will not build`,重采样 6 次 | 房型资产缺,或该任务本来就建不起来 | 查 `adapters48.CANNOT_BUILD`;缺资产就去下 |
| 渲染出来一片橄榄色方块 | 画了碰撞几何 | 已修;若自建渲染器,记得 `geomgroup[0]=0` |
