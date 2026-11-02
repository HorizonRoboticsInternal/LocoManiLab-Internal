# Kitchen 架构解析：任务、场景、资产、机器人的关系与数据采集闭环

> 本文从架构视角解释 RoboCasa / RoboCasa365 仓库 **Kitchen 部分**是如何围绕"任务"
> 组织起来的，并重点分析 **375 个注册环境、场景、资产、任务之间的关系**。
> 计数细节见配套文档 [`kitchen_tasks_and_assets_analysis.md`](./kitchen_tasks_and_assets_analysis.md)。

---

## 一、核心论点

整个 Kitchen 及其相关模块，是一个**以"任务"为中心、漏斗式装配的"仿真数据工厂 + 评测平台"**：

- **任务定义**是*配方层*——规定采什么、初始怎么摆、什么算成功；
- **场景 / 资产 / 机器人**是可组合的*素材层*——任务 `reset` 时被装配进 MuJoCo 场景；
- **遥操作与 rollout** 是*生产线*——产出轨迹数据；
- **dataset_scripts + registry** 是*打包出货*——加工成可训练数据集；
- 最终服务于上层的**策略训练与 benchmark**。

数据采集是这条流水线最关键的一段，但**不是唯一目的**——RoboCasa 上游框架的定位是
"a large-scale simulation framework for **training and benchmarking** generalist robots"，
即终极目标是训练并评测通用机器人策略，采集只是为此服务的中间产物。

---

## 二、整体分层架构

```
                    ┌─────────────── 任务 (Task / Env) ───────────────┐
                    │  environments/kitchen/  → 375 个注册环境          │
                    │  每个任务 = 一组 hook 方法：                       │
                    │   _setup_kitchen_references / _get_obj_cfgs       │
                    │   _setup_scene / get_ep_meta / _check_success     │
                    └───────────────────────────────────────────────────┘
        ┌──────────────────────┬──────────────────┬─────────────────────┐
        ▼                      ▼                  ▼                     ▼
   ┌─────────┐          ┌────────────┐     ┌───────────┐        ┌────────────┐
   │ 场景     │          │ 资产        │     │ 机器人     │        │ 控制器/相机 │
   │ scenes/ │          │ objects/   │     │ env_utils │        │ controller │
   │ 60×60   │          │ fixtures/  │     │ 偏移表     │        │ + cameras  │
   │ layout/ │          │ 198 类     │     │ G1/GR1/   │        │            │
   │ style   │          │ ~3400 实例 │     │ PandaOmron│        │            │
   └─────────┘          └────────────┘     └───────────┘        └────────────┘
        └──────────────────────┴──────────────────┴─────────────────────┘
                                    │ 三者是任务 reset 时被"装配"进场景的素材
                                    ▼
        ┌───────────────────────────────────────────────────────────────┐
        │              数据采集 / Rollout 管线（核心产出）                  │
        │  demos/demo_teleop · scripts/collect_demos · run_random_rollouts │
        └───────────────────────────────────────────────────────────────┘
                                    ▼
        ┌───────────────────────────────────────────────────────────────┐
        │              数据集后处理 + 训练/评测对接                          │
        │  dataset_scripts/  ·  dataset_registry  ·  robomimic/groot/lerobot │
        └───────────────────────────────────────────────────────────────┘
```

---

## 三、375 个注册环境是什么、从哪来

### 3.1 注册机制

- `Kitchen`（`environments/kitchen/kitchen.py`）是所有任务的基类，子类化自 robosuite 的 `ManipulationEnv`。
- 元类 `KitchenEnvMeta` 在**导入时**自动把每个 `Kitchen` 子类按类名登记进 `REGISTERED_KITCHEN_ENVS`，
  仅排除 5 个抽象基类（`MG_Robocasa_Env`、`PickPlace`、`ManipulateDoor`、`OpenDoor`、`CloseDoor`）。
- `environments/__init__.py` 把它暴露为 `ALL_KITCHEN_ENVIRONMENTS`。
- `robocasa/__init__.py` 是巨大的 import 清单——**只有被它 import 的任务模块，元类注册才会触发**。
  新增任务必须在此登记，否则不可发现。

### 3.2 375 = 74 Atomic + 301 Composite

| 类别 | 注册数 | 含义 | 组织方式 |
|---|---|---|---|
| **Atomic** | 74 | 单一技能原语（开/关、启/停、取放） | 15 个模块，按家电/技能族分组 |
| **Composite** | 301 | 多步真实活动（烹饪/清洗/整理流程） | 60 个活动目录，每目录 1–19 个任务 |

> "注册环境数 375"指的是**任务类的数量**，不是场景数。每个类本身只是"配方"——
> 它在被实例化（reset）时，才与具体的场景/资产/机器人组合成一个可运行的仿真实例。

---

## 四、四者关系：任务 × 场景 × 资产 × 机器人

这是理解整个系统的关键：**一个任务类 ≠ 一个固定场景**，而是一个"模板"，每次 reset 都会与
随机采样的场景、资产、（可配置的）机器人组合，产生一个不同的具体实例。组合是**乘法关系**。

```
   一个任务类 (e.g. BoilPot)
        × layout (60 种之一)
        × style  (60 种之一)
        × 采样物体实例 (从 obj_groups 命中的类别里随机抽 3D 变体)
        × 机器人本体 (PandaOmron / G1 / GR1 / ...)
   ───────────────────────────────────────────────
        = 一次具体的可运行仿真实例（每次 reset 都不同）
```

### 4.1 任务 → 资产：通过"语义组采样"而非硬编码

任务在 `_get_obj_cfgs()` 里**不写死具体 3D 模型**，而是声明语义组：

```python
# 伪代码示意
cfgs.append(dict(name="vegetable", obj_groups="vegetable", placement=...))
```

- `obj_groups` 指向 `OBJ_GROUPS`（230 个组）中的一个，组再展开成 `OBJ_CATEGORIES`（198 类）。
- reset 时 `sample_kitchen_object` 从命中的类别里随机抽一个 3D 实例（aigen / objaverse / lightwheel，
  合计约 3400 个变体）。
- **物体属性标记是任务可行性的门控**：只有 `cookable` 的类别能进 frying/baking，
  只有 `washable` 的能进 washing 类任务。任务通过"组 + 属性"间接选资产，从而获得物体级多样性。

### 4.2 任务 → 场景：通过 `FixtureType` 语义引用而非坐标

任务在 `_setup_kitchen_references()` 里用 `self.register_fixture_ref(name, dict(id=FixtureType.STOVE, ...))`
按**语义类型**而非具体坐标引用家电。

- 这样同一个任务能落到 60 个不同 layout 上——只要该 layout 含所需家电类型。
- 部分 layout 缺某类家电，于是有排除表：`OVEN_EXCLUDED_LAYOUTS`、`DOUBLE_CAB_EXCLUDED_LAYOUTS`
  `[32,41,59]`、`ISLAND_EXCLUDED_LAYOUTS`、`FREEZER_EXCLUDED_LAYOUTS` 等——
  这正是"任务 ↔ 场景"兼容性约束的体现。
- `style` 只换外观纹理（60 种），不改家电拓扑，因此对任务可行性透明，纯粹增加视觉多样性。

### 4.3 任务 → 机器人：通过 `env_utils` 解耦

- 支持的本体在 `env_utils._ROBOT_POS_OFFSETS`：`PandaOmron`（默认移动机械臂）、
  `GoogleRobot`、人形 `GR1*` / `G1*`（floating / fixed-lower-body 变体）。
- 同一任务可换不同机器人采集；基座偏移、相机自动适配
  （G1 用 `robot0_head_camera`，其余用 `robot0_eye_in_hand`）。
- 任务逻辑（成功判据、物体配置）与机器人本体**完全解耦**，所以"换机器人"几乎零成本。

### 4.4 `split`：把三者一起切，服务泛化采集

`split` 是贯穿 `create_env` 的核心概念，它**同时**控制 `obj_instance_split`、`layout_ids`、`style_ids`：

| split | layout/style id | 物体实例 | 用途 |
|---|---|---|---|
| `pretrain` | `-2`（layout/style 011–060，各 50） | 预训练划分 | 大规模预训练采集 |
| `target` | `1..10`（各 10） | held-out 划分 | 基准评测（未见过的场景+物体） |
| `all` | `-3` | 全部 | 不区分 |

> 这就是为什么文档说"2500+ 厨房场景"：60 layout × 60 style 的组合空间（受排除表裁剪），
> 再叠加物体实例的随机采样，单个任务类就能展开出海量不重复的实例。

---

## 五、数据采集闭环：各层如何服务于"采数据"

| 层 | 模块 | 在数据采集中的角色 |
|---|---|---|
| **任务** | `environments/kitchen/{atomic,composite}` | 定义"采什么"——目标、初始物体配置、成功判据 `_check_success`、语言指令 `ep_meta["lang"]` |
| **场景** | `models/scenes/`（60 layout × 60 style） | 提供环境多样性；`split` 把场景+物体切成 pretrain/target，直接服务"跨场景泛化"采集 |
| **资产** | `models/objects/`（198 类 / ~3400 实例）、`models/fixtures/` | reset 时按 `obj_groups` 随机采样；属性标记决定哪些物体进哪类任务 |
| **机器人** | `env_utils._ROBOT_POS_OFFSETS`（G1/GR1/PandaOmron/GoogleRobot） | 同一任务可换本体采集；基座偏移、相机自动适配 |
| **控制器/相机** | composite controller config、`camera_utils`、`gym_wrapper` 的 KeyConverter | 把机器人观测/动作映射成数据集 schema——决定"采出来的数据长什么样" |
| **采集入口** | `demos/demo_teleop`、`demo_empty_teleop`、`scripts/collect_demos`；G1 使用 `collect_demos_g1` / `collect_fixed_task` | 人工遥操作（PICO VR / SpaceMouse / 键盘）或随机 rollout 实际生成轨迹 |
| **数据集管线** | `dataset_scripts/`：`playback_dataset`、`dataset_states_to_obs`、`convert_hdf5_lerobot`、`generate_usd_trajectories` | 回放、从 state 重渲染观测、导出 LeRobot/USD |
| **数据集登记** | `dataset_registry.py`（317 任务）、`dataset_registry_utils` | 把每个任务映射到 human/mg/5x5 等数据集路径 + horizon |
| **训练/评测对接** | `utils/robomimic`、`groot_utils`、`lerobot_utils`、`eval_utils` | 数据的下游消费者（策略学习与评测） |

### 端到端数据流

```
 teleop 设备 (PICO VR / SpaceMouse / 键盘)
     │
     ▼   demo_teleop / collect_demos
 任务 reset → 装配 (layout × style × 采样物体 × 机器人)
     │
     ▼   记录 states + actions
 原始 HDF5 轨迹
     │
     ▼   dataset_states_to_obs  (从 state 重渲染多相机观测)
 带观测的 HDF5
     │
     ├──▶ convert_hdf5_lerobot   → LeRobot 数据集
     ├──▶ generate_usd_trajectories → USD
     └──▶ robomimic / groot_utils → 策略训练 → eval_utils 评测
```

---

## 六、为什么说"采集是核心环节，但不是唯一目的"

1. **存在评测/基准侧**：`tests/test_tasks_validity.py` 等用 rollout 验证任务可解；
   `eval_utils`、`docs/benchmarking` 面向 benchmark 而非采集。
2. **三种数据来源并行**：human 遥操作、`mg`（MimicGen 机器生成）、随机 rollout——
   说明采集只是手段，不同手段都为喂训练服务。
3. **多下游格式**：同时支持 robomimic、GR00T、LeRobot、USD——采集的终点是"对接各家训练框架"。

---

## 七、本分支的扩展方向

在上述既有框架之上，`main` 分支把**采集侧**从 PandaOmron 厨房扩展到人形机器人和新场景：

- **G1 + PICO VR 采集**：`collect_demos_g1.py` / `collect_fixed_task.py` 通过 `G1_BRIDGE` 使用独立 `g1_backend`，见 [采集说明](g1_data_collection_pipeline.md)。
- **living-room 扩展**：与其他场景共用上述采集入口；场景结构见 [客厅说明](living_room_scene.md)，环境层与记录层的差异见 [对比说明](living_room_vs_kitchen_comparison.md)。

这些都进一步印证主线：**围绕"任务"装配场景/资产/机器人，再通过遥操作/rollout 产出数据**。

---

## 八、一句话总结

> 375 个注册环境是 375 套"任务配方"，它们本身不绑定场景；运行时每个任务与
> 60×60 的 layout/style、约 3400 个物体实例、可切换的机器人本体做**乘法组合**，
> 在 reset 时装配成一个具体仿真实例。`split` 把场景与物体一起切成 pretrain/target
> 以服务泛化采集与评测。整套 Kitchen 体系就是一条**任务驱动的"仿真数据工厂 + 评测平台"**流水线，
> 数据采集是其最关键的一段，最终目标是训练并评测通用机器人策略。
