# Living Room vs Kitchen 链路对比与可扩展性评估

> 目的：在开始用 G1 大规模采集数据、并逐步扩充任务/场景/物体之前，系统梳理 living-room 链路相对
> kitchen 的**完整度差异**（资产数目、机器人本体之外的所有方面），评估"链路是否完整、是否好扩展"，
> 并给出按优先级排序的补齐建议。
>
> 结论先行：**核心骨架（注册 / 场景 / 物体复用 / 任务钩子）设计干净、对齐 kitchen、可扩展性好**；
> 与 kitchen 的主要差距集中在**观测/数据管线**与**split/基准集成**两块。对你当前"图像式 LeRobot 采集"
> 流程，**链路已可用**；若要走 robocasa 原生 state 数据集 / 回放 / 训练泛化，需补 3 处（见第 3 节）。

代码规模参考：`kitchen.py` 1782 行 vs `living_room.py` 380 行；差异主要在下面列的子系统。

> **更新 2026-05-31**：已实现 **#1（物体观测）** 和 **#3（split 接线）**，见 2.1 / 2.3 的 ✅ 标注。
> **#2（edit_model_xml）** 按需推迟（对当前 LeRobot 采集不阻塞，解释见 2.2）。

---

## 1. 已对齐 / 复用得很好的部分（骨架健康）

| 方面 | kitchen | living room | 状态 |
|---|---|---|---|
| 元类自动注册 | `KitchenEnvMeta` → `REGISTERED_KITCHEN_ENVS` | `LivingRoomEnvMeta` → `REGISTERED_LIVING_ROOM_ENVS` | ✅ 同构 |
| 基类可跑 + 任务子类 | `Kitchen` 基类可跑，任务子类化 | `LivingRoomEnv` 同样 | ✅ 同构 |
| 场景 = (layout, style) | scene_registry + KitchenArena + scene_builder | living_room_scene_registry + LivingRoomArena + living_room_scene_builder | ✅ 镜像 |
| 物体层 | sample_kitchen_object / create_obj / ObjectPositionSampler | **直接复用 kitchen 同一套** | ✅ 复用（省一大块） |
| fixture 查找 | `get_fixture` / `register_fixture_ref` | `get_fixture` / `register_furniture_ref` | ✅ 对应（简化版） |
| 任务钩子 | `_get_obj_cfgs` / `_setup_kitchen_references` / `_setup_scene` / `_check_success` / `get_ep_meta` | `_get_obj_cfgs` / `_setup_furniture_references` / `_setup_scene` / `_check_success` / `get_ep_meta` | ✅ 一一对应 |
| ep_meta 可复现 | 序列化 layout/style/object_cfgs | 同样序列化 | ✅ 镜像（可确定性重建一集） |
| 放置失败重试 | 有 | 有（20 次重试） | ✅ |

**扩展操作和 kitchen 一样简单**：加任务 = 子类化 + 在 `robocasa/__init__.py` import；加场景 = 写 layout/style yaml；
加家具 = fixture 类 + 资产 + 在 scene_builder 的 `FURNITURE` 注册类型。

---

## 2. 主要差异 / 缺口（资产数目、机器人之外）

### 2.1 ✅ 物体观测/传感器 —— **已修复（2026-05-31）**

- 已把 kitchen 的 `_setup_references` / `_setup_observables` / `_create_obj_sensors` 移植到
  `LivingRoomEnv`。现在每个可操作物体的 `<name>_pos` / `<name>_quat` / `<name>_to_<robot>eef_pos|quat`
  都进观测（验证：`CarryObjectToTable` 的 obs 含 `mug_pos/mug_quat/...`）。
- robot-agnostic：eef 相对量在双臂 G1 上退化为 identity，但绝对 `obj_pos/obj_quat` 对任意机器人都对；
  无物体的任务（如 `ApproachSofa`）不加传感器、不报错。
- 注意：**环境层**暴露物体观测，不代表录制层会自动写入对应数据列。G1 采集复用 `collect_demos_g1` / `collect_fixed_task` 的共享录制层，实际列以选中后端及数据集 `meta/info.json` 为准。

### 2.2 ⚠️ `edit_model_xml` —— **缺失**（数据回放可移植性）

- kitchen 覆写 `edit_model_xml`，把 model.xml 里的**绝对资产路径**改写成本机路径，让 demo 能在别的机器回放。
- living room **没有**这个覆写。
- 影响：
  - 你的 **LeRobot 采集直接存 obs（图像+state），不依赖 model.xml 回放，不受影响**。
  - 若用 robocasa 原生 `collect_demos`（存 states+model.xml）→ `playback_dataset` / `dataset_states_to_obs`
    跨机回放会因路径失效而炸。需补 `edit_model_xml`（可照搬 kitchen）。

### 2.3 ✅ Split 系统（pretrain/target/all）—— **已修复（2026-05-31）**

- `LivingRoomEnv.__init__` 现在接受 `obj_instance_split` / `obj_registries` / `layout_and_style_ids`
  （镜像 Kitchen，`layout_and_style_ids` 显式对优先），并把 `obj_instance_split` 接进 `sample_object`
  → **物体实例 train/test 划分可用**（验证：split=pretrain/target 正确设置）。
- `create_env` 加了 **living-room 分支**：遍历 `ALL_LIVING_ROOM_ENVIRONMENTS` 命中后用精简 kwarg（不传
  kitchen 专属的 generative_textures / randomize_cameras / translucent_robot），把 `split` 映射成
  `obj_instance_split`，相机按机器人自适应（G1→`robot0_head_camera`，否则 `robot0_eye_in_hand`），
  **不再 TypeError**。
- **设计取舍**：living room 目前只有 layout 1-4 / style 1-3，且任务常**绑定特定 layout**（如
  CarryObjectToTable 需 layout4 的两张 side_table）。因此 split 暂**只划分物体实例、不划分场景**；
  `create_env` 只在调用者显式传 `layout_ids/style_ids` 时才覆盖场景，否则**尊重任务固定的默认场景**
  （验证：split=all 仍得 layout4/style3）。等场景多了再加 scene 级 train/test 分组。

### 2.4 `create_env` 集成 —— living room 走的是 `robosuite.make`

- kitchen 的规范入口是 `create_env`（应用 split + 复合控制器配置）。
- living room 目前只能 `robosuite.make` 直建（采集脚本/demo 都这样），与 2.3 同源。

### 2.5 gym.make / 数据集注册 —— **部分缺失**

- gym 注册（`gym_wrapper.py:498`）遍历 `REGISTERED_ENVS`，**living-room env 其实也被注册了**；但
  `RoboCasaGymEnv` 的 `key_converter` **硬编码 `PandaOmronKeyConverter`**——`G1KeyConverter` 虽存在，
  但不按机器人类型自动路由。所以 `gym.make("robocasa/<LivingRoomTask>")` 结构上能建，但 obs/action 映射按
  PandaOmron 走，G1 不对。
- `dataset_registry.py` 里**没有 living-room 任务**（ATOMIC/COMPOSITE 都没有）。自采数据不依赖它，但官方基准
  结构上缺。

### 2.6 语言/指令系统 —— 简化

- kitchen 有 `_load_novel_instructions` + 较丰富的 lang 生成。
- living room 每个任务一个**固定 lang 字符串**（够采集用，泛化指令需扩）。

### 2.7 物体创建丰富度 —— 简化（已在代码注释说明）

- kitchen `_create_objects`：容器、辅助物体（aux）、堆叠、滑动 fixture 放置。
- living room `_create_objects`：每个 cfg 一个自由物体，**无容器/无 aux**。

### 2.8 fixture 状态随机化 / 放置区域 —— 较粗

- kitchen：随机门/抽屉开合、滑动 fixture、丰富的命名区域（台面/内腔等）。
- living room：`articulated.py` 提供铰接类，但 reset 期状态随机更简单；家具主要靠 **bbox 区域**采样，
  能在台面上放东西但区域较粗。

---

## 3. 可扩展性评估 + 按优先级的补齐建议

你的目标：**G1 采集数据 + 逐步扩充任务/场景/物体**。按你**当前的图像式 LeRobot 采集流程**，下面分清"现在就能做"
和"要补什么"。

### 现在即可做（链路对这条路是完整的）
- 加任务、加场景(yaml)、加家具、加可操作物体（复用 kitchen 物体库）—— 骨架已支持，操作和 kitchen 一致。
- G1 键盘/PICO 遥操 + 头相机 ego_view + LeRobot 落盘 —— 已打通（见 `g1_data_collection_pipeline.md`）。
- 任务成功判定（读 sim）+ 成功保存 demo —— 已支持。

### 建议补齐（按优先级）

**P1 — 若要让数据集含物体位姿 / 走 robocasa 原生回放（视你训练方案而定）**
1. 覆写 `LivingRoomEnv._setup_observables` + `_create_obj_sensors`（照搬 kitchen，把 `self.objects`
   的 pos/quat/to_eef 注册进 obs）。这样无论 LeRobot 还是原生 HDF5，物体 state 都进数据。
2. 覆写 `LivingRoomEnv.edit_model_xml`（照搬 kitchen 的路径改写），保证 demo 跨机回放/`dataset_states_to_obs`
   不炸。

**P2 — 训练/评测要泛化划分时**
3. `LivingRoomEnv.__init__` 增加 `obj_instance_split` / `obj_registries` / `layout_and_style_ids` kwargs
   并接进物体采样；定义 living-room 自己的 split 约定（如 layout/style 的 train/test 子集），让
   `create_env(split=...)` 能用。

**P3 — 与官方基准/gym 对齐（非阻塞）**
4. `RoboCasaGymEnv` 按机器人类型路由 key converter（G1→`G1KeyConverter`），并把 living-room 任务的相机/obs
   schema 补全。
5. living-room 任务进 `dataset_registry`（若要纳入标准数据集索引）。

**P4 — 丰富度（随任务复杂度逐步加）**
6. 物体容器/aux、家具命名区域（台面/格层）、铰接 fixture 的开合随机化、novel 指令生成。

### 一句话评估
- **骨架健康、可扩展性好**：注册/场景/物体复用/任务钩子都对齐 kitchen，加任务-场景-物体的成本低。
- **缺口集中在"观测与数据管线"和"split/基准集成"**，且**大多可照搬 kitchen 实现**。
- 对你眼下的图像式 G1 采集，**P1 的第 1 条（物体观测）最值得先做**（让数据集带上物体位姿，未来训练/调试更灵活），
  其余按需推进。

---

## 4. 关键文件对照表

| 角色 | kitchen | living room |
|---|---|---|
| 环境基类 + 元类 | `environments/kitchen/kitchen.py` | `environments/living_room/living_room.py` |
| 任务目录 | `environments/kitchen/{atomic,composite}/` | `environments/living_room/tasks/` |
| 场景注册 | `models/scenes/scene_registry.py` | `models/scenes/living_room_scene_registry.py` |
| 场景构建 | `models/scenes/scene_builder.py` | `models/scenes/living_room_scene_builder.py` |
| 竞技场 | `models/scenes/kitchen_arena.py` | `models/scenes/living_room_arena.py`（继承 `FloorArena`） |
| fixture/家具 | `models/fixtures/*.py`（含 FixtureType） | `models/fixtures/furniture.py` + `articulated.py`（FurnitureType） |
| 物体层 | `models/objects/*` + `utils/env_utils` | **复用 kitchen 的** |
| 创建入口 | `utils/env_utils.create_env` | 目前 `robosuite.make`（采集脚本/demo） |
| 数据集注册 | `utils/dataset_registry.py` | 暂无 |
