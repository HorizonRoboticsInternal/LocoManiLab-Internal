# 场景系统详解：arenas / scenes / fixtures / objects / furnitures 如何协作

本文回答两个问题：

1. **kitchen 的场景系统是怎么工作的** —— arena、scene、fixture、object 这几层各自负责什么、如何被调用、数据怎么流动。
2. **要把 living room 做成 kitchen 那样的系列场景，除了"任务"还缺什么** —— 逐项对照 + 哪些可复用、哪些要新建。

> 已有的相关文档：`docs/living_room_scene.md`（当前 living room 怎么搭）、`docs/furniture_workflow.md`（资产怎么转）。

---

## 1. 两类"东西"：fixture vs object

理解整个系统的前提，是分清场景里**两类截然不同的物体**：

| | **Fixture（固定物）** | **Object（可操作物）** |
|---|---|---|
| 例子 | 灶台、水槽、橱柜、墙、地板 | 苹果、锅、杯子、刀 |
| 是否可动 | **静态焊死**（无 free joint，可有门/抽屉关节） | **自由刚体**（free joint，可抓可推） |
| 谁来摆 | **arena**：按 layout+style **建场景时**就位 | **任务**：每次 `reset` 时**采样**到 fixture 表面 |
| 定义在 | `models/fixtures/` + layout/style yaml | `models/objects/` + 任务的 `_get_obj_cfgs()` |
| 语义引用 | `FixtureType` 枚举 | `OBJ_GROUPS` / `OBJ_CATEGORIES` |

> **living room 的 `FurnitureObject` ≈ kitchen 的 fixture**（静态、建场景时就位）。差别是：fixture 有一整套 layout/style/builder 系统来"自动布置"，而 furniture 目前是 `_get_furniture()` 里**硬编码**摆位。这正是后面要补的核心。

---

## 2. kitchen 场景系统的分层与调用链

### 2.1 各层职责

| 层 | 位置 | 职责 |
|---|---|---|
| **环境类** | `environments/kitchen/kitchen.py` (`Kitchen`) | 编排：建 arena → 取 fixtures → 建 objects → 合成模型 |
| **arena** | `models/scenes/kitchen_arena.py` (`KitchenArena`) | 读 layout+style，调 `create_fixtures()` 生成所有 fixture；持有房间外壳 XML |
| **arena 外壳 XML** | `models/assets/arenas/empty_kitchen_arena.xml` | 近乎空的 MJCF 骨架（灯光等），墙/地板其实由 layout 当 fixture 生成 |
| **scene 注册表** | `models/scenes/scene_registry.py` | `LayoutType`(60) / `StyleType`(60) 枚举 + `get_layout_path()` / `get_style_path()` → yaml 路径 |
| **layout yaml** | `models/assets/scenes/kitchen_layouts/**/*.yaml` | 房间几何（墙/地板）+ 有哪些 fixture、摆在哪（位置/对齐/分组） |
| **style yaml** | `models/assets/scenes/kitchen_styles/**/*.yaml` | 每个 fixture 类型选哪个具体资产实例 + 墙/地板贴图 |
| **scene builder** | `models/scenes/scene_builder.py` (`create_fixtures`) | **引擎**：把 layout+style 字典 → 实例化的 fixture 对象（含定位/对齐/堆叠/附属件） |
| **fixtures** | `models/fixtures/*.py` | fixture 的 Python 类（`Stove`/`Sink`/`Counter`…），包装 MJCF 资产；`FixtureType` 语义枚举；`fixture_utils` 查找 |
| **objects** | `models/objects/*.py` | `MJCFObject` + `OBJ_CATEGORIES` + `sample_kitchen_object()` |
| **放置采样器** | robosuite `ObjectPositionSampler`（经 `env_utils._get_placement_initializer`） | reset 时把 object 采样到 fixture 的 region 上 |

### 2.2 调用链（建场景）

```
Kitchen._load_model()
  └─ _setup_model()
       ├─ 选 layout_id / style_id（来自 ep_meta 或随机）
       ├─ KitchenArena(layout_id, style_id, ...)              # ← arena 是入口
       │     ├─ 加载 empty_kitchen_arena.xml（房间外壳）
       │     ├─ get_layout_path(layout_id) → 读 layout.yaml   # ← scene_registry
       │     ├─ get_style_path(style_id)   → 读 style.yaml
       │     └─ create_fixtures(layout, style)                # ← scene_builder 引擎
       │           ├─ 遍历 layout 的 group/fixture 配置
       │           ├─ load_style_config() 注入 style（选资产+贴图）
       │           ├─ FIXTURES[type] 把字符串→fixture 类(Stove/Sink/...)
       │           ├─ initialize_fixture() 实例化
       │           └─ 定位：align_to / stack_on / 绝对 pos / 附属件
       ├─ arena.get_fixture_cfgs() → self.fixtures            # name→fixture 对象
       └─ ManipulationTask(arena, robots, fixtures)           # 合成 MJCF
```

### 2.3 调用链（放任务物体，每次 reset）

```
Kitchen._create_objects()
  ├─ self.object_cfgs = self._get_obj_cfgs()        # ← 任务定义（子类覆写）
  │     # 每条：name / obj_groups / placement=dict(fixture=..., sample_region_kwargs=...)
  ├─ 对每条 cfg：sample_kitchen_object(obj_groups)  # ← 从 OBJ_CATEGORIES 采样具体实例
  │     └─ 得到 MJCFObject（带 free joint，可抓）
  └─ env_utils._get_placement_initializer(cfgs)     # 建 ObjectPositionSampler

Kitchen._reset_internal()  (每次 reset)
  └─ placement_initializer.sample()                 # 在指定 fixture 的 region 上采样位姿
       └─ 把每个 object 摆到 fixture 表面（带碰撞校验、重试）
```

要点：
- **fixture 在建场景时一次性就位**（layout 决定）；**object 每次 reset 重新采样**（任务决定，位置随机化）。
- object 落到哪，靠 `placement.fixture`（指定哪个 fixture）+ 该 fixture 暴露的 **region**（`reg_*` geom，如 counter 顶面）。`sample_region_kwargs` 选具体区域。
- 任务通过 `register_fixture_ref(name, dict(id=FixtureType.X))` 用**语义类型**拿到 layout 里随机选中的具体 fixture 实例。

---

## 3. living room 现状对照

当前 living room（见 `living_room_scene.md`）：

```
LivingRoomEnv._load_model()
  ├─ FloorArena()                    # 固定单一外壳（floor_arena.xml），无 layout/style
  ├─ robot 落位
  ├─ _get_furniture()                # 硬编码返回 3 个 FurnitureObject（pos/euler 写死）
  └─ ManipulationTask(arena, robots, furniture)
```

也就是说：**fixture 那一整套"layout+style+builder+registry+语义枚举"的自动化布置系统，living room 目前完全没有**——家具是手写摆的；object 层则一片空白。

---

## 4. 要做成 kitchen 那样，living room 还缺什么（除任务外）

按"建场景"到"放物体"的顺序，逐项列出。**标注了哪些可直接复用 kitchen 的代码、哪些需独立新建**（保持你要的解耦）。

| # | 缺的能力 | kitchen 对应 | living room 现状 | 建议做法 |
|---|---|---|---|---|
| 1 | **家具语义枚举** | `FixtureType` | 无 | **新建** `FurnitureType`（SOFA/TABLE/TV/SHELF…）+ `furniture_utils`（按类型查找、`register_*_ref`） |
| 2 | **可配置房间外壳** | `empty_kitchen_arena.xml` + 墙/地板当 fixture | 单一 `floor_arena.xml`（写死 6×6 房间） | **扩展**：把墙/地板尺寸做成可参数化，或多个 arena XML |
| 3 | **场景注册表** | `scene_registry.py`（LayoutType/StyleType + get_*_path） | 无 | **新建** living room 版枚举 + 路径解析 |
| 4 | **layout 描述** | layout yaml（几何 + fixture 布置） | 无（硬编码在 `_get_furniture`） | **新建** living room layout（yaml 或纯 Python data 都行，你之前倾向独立/data 化） |
| 5 | **style 描述** | style yaml（选资产 + 墙地贴图） | 无 | **新建** living room style（家具实例选择 + 墙纸/地板/地毯贴图） |
| 6 | **场景构建引擎** | `scene_builder.create_fixtures` | 无（手写 list） | **新建** `create_furniture(layout, style)`：把布置 data → `FurnitureObject` 列表，处理定位/对齐 |
| 7 | **arena 编排** | `KitchenArena` 读 layout+style 建 fixtures | `FloorArena` 只有地板 | **新建** `LivingRoomArena`：读 layout+style，调 `create_furniture`，`get_furniture_cfgs()` |
| 8 | **家具 region** | fixture 的 `reg_*`（供采样物体） | `FurnitureObject` 已有 `reg_bbox` ✓ | **基本就绪**；按需加 `reg_top` 等子区域（如沙发坐垫、茶几桌面） |
| 9 | **可操作物体** | `OBJ_CATEGORIES`/`sample_kitchen_object`/`MJCFObject` | 无 | **可复用** kitchen 的 objects 系统（与厨房无强绑定）；或建独立物体组 |
| 10 | **放置采样器接入** | `ObjectPositionSampler` + `_get_placement_initializer` | 无 | **可复用**：只要 furniture 像 fixture 一样暴露 region，采样器就能往家具上放物体 |
| 11 | **相机配置** | `CamUtils` + 每 layout 相机 | `floor_arena.xml` 4 个固定相机 | 够用；做策略/采集时再补 per-layout 相机 + key converter |
| 12 | **（任务）** | atomic/composite + 成功判定 + 语言 | 无 | 你已知，这里不展开 |

### 关键判断：哪些必须独立、哪些可直接复用

- **必须独立新建**（保持与 kitchen 解耦）：#1 `FurnitureType`、#3 注册表、#4 layout、#5 style、#6 builder、#7 arena。这些是"布置系统"，是 living room 自己的骨架。
- **可直接复用 kitchen 的**（它们本就与厨房无关）：#9 `MJCFObject`/`OBJ_CATEGORIES`/`sample_kitchen_object`、#10 `ObjectPositionSampler`。物体层不必重写——`FurnitureObject` 已带 `reg_bbox`，采样器能直接往上放物体。
- **基本就绪**：#8 region（已有 bbox）、#11 相机（已有 4 个）。

---

## 5. 建议的实现顺序（增量、每步可验证）

1. **`FurnitureType` + `furniture_utils`**（#1）——后面所有布置/任务都要靠它做语义引用。
2. **layout/style 数据 + `create_furniture` + `LivingRoomArena`**（#4/#5/#6/#7/#3）——一次把"自动布置"打通：`LivingRoomEnv._load_model` 改成 `LivingRoomArena(layout_id, style_id)`，不再硬编码 `_get_furniture`。先支持 1~2 个 layout、1~2 个 style。
3. **多 layout/style 扩展**（#2 房间可参数化）——形成"系列场景"。
4. **接入物体层**（#9/#10）——复用 kitchen 的采样器，往家具 region 上放可操作物体；给 `FurnitureObject` 按需加 `reg_top` 等子区域。
5. **任务**（#12）——atomic→composite。
6. **相机/obs/数据集**——做策略/采集时再补。

> 落地第 1、2 步后，living room 就具备了 kitchen 同构的"layout+style→自动建场景"能力，只是规模小；之后就是往 layout/style/资产里堆内容。

---

## 6. 一句话总结

kitchen 的精髓是 **"环境类只管编排，具体布置交给 `arena(layout, style) → builder → fixtures`，物体交给 `任务定义 → 采样器 → reset 时就位`"** 这套数据驱动的分层。living room 现在只有最外层环境类和最底层资产，**中间的"布置系统"（FurnitureType + 注册表 + layout/style + builder + arena）整段缺失**，需要独立搭一套同构的；而**物体采样那一层可以直接复用 kitchen 的**。
