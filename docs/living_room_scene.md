# Living Room 场景搭建详解

本文梳理 `LivingRoomBasic` 这个客厅场景**从代码到资产**是怎么一层层拼出来的：涉及哪些文件、各自负责什么、数据如何流动，以及如何增改家具/新建场景。

> 配套文档：
> - 资产怎么从 Meshy 转出来 → `docs/furniture_workflow.md`

---

## 1. 一句话概览

`LivingRoomBasic` 是一个**独立的 `ManipulationEnv` 子类**（不依赖 kitchen，也不依赖测试用的 `EmptyEnv`），运行时把 **`FloorArena`（地板+围墙）** + **机器人** + **三件焊死的家具网格** 合成一个 MuJoCo 模型。

```
robosuite.make("LivingRoomBasic")
        │
        ▼
LivingRoomEnv.__init__  ──▶  机器人由 robosuite 按 controller_config 创建
        │
        ▼
LivingRoomEnv._load_model:
   ├─ FloorArena()                         # 地板 + 4 面可碰撞墙 + 相机
   ├─ robot.set_base_xpos(robot_base_pos)  # 机器人落位
   ├─ _get_furniture()  ── LivingRoomBasic 返回 3 个 FurnitureObject
   │        └─ 每个 FurnitureObject 从 model.xml 加载网格、焊死、摆 pos/euler
   └─ ManipulationTask(arena, robots, objects)   # 合成最终 MJCF
```

---

## 2. 涉及的文件清单

| 层 | 文件 | 角色 |
|---|---|---|
| 注册 | `robocasa/__init__.py` | import living_room 模块 → 触发 EnvMeta 自动注册 `LivingRoomEnv`/`LivingRoomBasic` |
| 环境 | `robocasa/environments/living_room/living_room.py` | **核心**：`LivingRoomEnv`（场景骨架）+ `LivingRoomBasic`（家具布局） |
| 环境 | `robocasa/environments/living_room/__init__.py` | 导出两个类 |
| 场地 | `robocasa/models/arenas/floor_arena.py` | `FloorArena` 类，加载下面的 XML |
| 场地 | `robocasa/models/assets/arenas/floor_arena.xml` | 地板 + 4 面墙 + 4 个相机的 MJCF |
| 家具包装 | `robocasa/models/furniture/furniture.py` | `FurnitureObject`（静态网格家具）、`furniture_mjcf_path()` |
| 家具包装 | `robocasa/models/furniture/__init__.py` | 导出上述符号 |
| 家具资产 | `robocasa/models/assets/furniture/<类别>/<实例>/model.xml` | 每件家具的 MJCF（**gitignore，不入库**） |
| 家具资产 | `.../visual/{model.obj, model.mtl, texture.png}` | 网格 + 贴图 |
| 入口 | `robocasa/demos/demo_empty_teleop.py` | 遥操作 / 自检入口（`--env LivingRoomBasic`） |

> 注意区分两个相似路径：
> - **代码** `robocasa/models/furniture/` —— `FurnitureObject` 等 Python 类
> - **资产** `robocasa/models/assets/furniture/` —— 网格/贴图/`model.xml`（gitignore）

---

## 3. 逐层拆解

### 3.1 注册：`robocasa/__init__.py`

robosuite 的 `EnvMeta` 元类会**按类名自动注册**所有 `MujocoEnv` 子类。只要 living_room 模块被 import，`LivingRoomEnv`/`LivingRoomBasic` 就进了 `REGISTERED_ENVS`，从而 `robosuite.make("LivingRoomBasic")` 可用。

```python
# robocasa/__init__.py
from robocasa.environments.living_room.living_room import (
    LivingRoomEnv,
    LivingRoomBasic,
)
```

> 新建场景类后，**必须确保它被 import**（直接或通过 `living_room/__init__.py`），否则不会注册、`make` 时报 "环境不存在"。

### 3.2 场景骨架：`LivingRoomEnv`

`LivingRoomEnv(ManipulationEnv)` 是独立基类（结构和 kitchen 的 `Kitchen` 同级），自带：

- **`__init__(...)`**：标准 robosuite 环境参数 + 自定义 `robot_base_pos`（机器人在地面上的落位，默认 `(0,0,0)`）。
- **`_load_model()`**：场景装配的核心，三步——
  1. `FloorArena()` 建场地，`set_origin([0,0,0])`。
  2. `robot.robot_model.set_base_xpos(self.robot_base_pos)` 落位机器人。
  3. `self.objects = {obj.name: obj for obj in self._get_furniture()}`，再用 `ManipulationTask(arena, robots, objects)` 合成最终模型。
- **`_get_furniture()`**：钩子，基类返回 `[]`，**子类覆写**它来摆家具。
- **`reward()→0`、`_check_success()→False`、`get_ep_meta()`**：暂无任务，`get_ep_meta()["lang"]` 是给遥操作/采集脚本看的自然语言指令。

> `ManipulationTask` 是 robosuite 把"场地 + 机器人 + 物体列表"合并成一个 MJCF 模型树的工具，kitchen 也用它。

### 3.3 场地：`FloorArena` + `floor_arena.xml`

`FloorArena`（`floor_arena.py`）只是个壳，真正内容在 `floor_arena.xml`：

- **地板**：`<geom name="floor" type="plane" size="3 3 .125">` → **6m×6m**、可碰撞（`condim=3`）、在 **z=0**。
- **四面墙**：`wall_front/rear/left/right`，box，**1.5m 高**（half 0.75），位于 x/y=±3 处。**可碰撞**（默认 `contype/conaffinity=1`）——这是和 robosuite 原版 `EmptyArena`（墙纯视觉、机器人会穿墙）的关键区别。
- **相机**：`frontview` / `birdview` / `agentview` / `sideview`（viewer 里按 `Tab` 切换）。

> 坐标系约定（后面摆家具要用）：x、y ∈ **[-3, 3]**，地面 **z=0**，+Z 朝上。

### 3.4 家具布局：`LivingRoomBasic._get_furniture()`

`LivingRoomBasic` 覆写 `_get_furniture()`，返回 3 个 `FurnitureObject`：

```python
FurnitureObject(name="sofa",
    mjcf_path=furniture_mjcf_path("sofas", "Sofa001"),
    pos=(-2.2, 0.0, 0.552), euler=(0.0, 0.0, 1.5708))      # 靠 -x 墙，正面朝 +x
FurnitureObject(name="coffee_table",
    mjcf_path=furniture_mjcf_path("coffee_tables", "CoffeeTable001"),
    pos=(-0.8, 0.0, 0.32))                                  # 居中
FurnitureObject(name="tv_stand",
    mjcf_path=furniture_mjcf_path("tv_stands", "TVStand001"),
    pos=(2.3, 0.0, 0.391), euler=(0.0, 0.0, -1.5708))       # 靠 +x 墙，屏朝 -x
```

布局（沿 x 轴一条线，沙发和电视对望）：

```
  -x 墙                                              +x 墙
  [沙发] ──朝+x──▶   [茶几]      ◀──朝-x── [电视柜]
  x=-2.2           x=-0.8                  x=+2.3
   z=0.552         z=0.32                  z=0.391
```

- **`pos`**：物体中心的世界坐标。`z` = 该资产的**半高**（脚正好落在 z=0 地面）；这个值由转换脚本输出、也写在每件家具的注释里。
- **`euler`**（弧度，绕 body 的 Z 轴）：调朝向。网格"正面"沿其深度轴，旋转 0 / ±1.5708 / 3.1416 对齐到房间。

### 3.5 家具包装：`FurnitureObject`

`FurnitureObject`（`furniture.py`）继承 `MJCFObject`：

- 复用 `MJCFObject` 的 model.xml 加载、缩放、`reg_*` region 解析。
- 构造后**删除自动注入的 free joint** → **静态焊死**（障碍/支撑面，不会被推走或掉落）。
- 通过 `furniture_mjcf_path(category, instance)` 把路径解析到 `assets_root/furniture/<category>/<instance>/model.xml`。

### 3.6 家具资产：`model.xml`

每件家具一个目录，`model.xml` 是 robocasa 物体格式（详见 `furniture_workflow.md` 第 6 节）：

```
robocasa/models/assets/furniture/sofas/Sofa001/
├── model.xml            # mesh + 贴图材质 + reg_bbox + 碰撞盒
└── visual/{model.obj, model.mtl, texture.png}
```

- 视觉 geom：`group=1`、不碰撞，带 `euler="1.5708 0 0"` 把源网格 Y-up 转成 MuJoCo 的 Z-up，并平移到 body 原点居中。
- 碰撞 geom：`group=0`、参与物理（脚本默认整体包围盒）。
- `reg_bbox`：包围盒 region，供后续放置采样用。
- **整个 `assets/furniture/` 已 gitignore**——换机器需另行同步资产，否则 `make` 时会因找不到 mesh 报错。

---

## 4. 运行查看

```sh
conda activate tele_benchmark
cd /path/to/humanoid_benchmark

python -m robocasa.demos.demo_empty_teleop --env LivingRoomBasic                    # PandaOmron
python -m robocasa.demos.demo_empty_teleop --env LivingRoomBasic --robot G1FloatingBody
python -m robocasa.demos.demo_empty_teleop --env LivingRoomBasic --check            # 无屏自检
```

viewer 里按 `Tab` 切相机（frontview/birdview/agentview/sideview）。

---

## 5. 如何增改

### 改家具位置/朝向
直接编辑 `LivingRoomBasic._get_furniture()` 里的 `pos`/`euler`，跑 `--check` 或开窗口看。`pos.z` 用资产的半高（脚本输出）。

### 加一件新家具
1. 用 `scripts/convert_meshy_furniture.py` 把 Meshy 导出转成资产（见 `furniture_workflow.md`）。
2. 在 `_get_furniture()` 里追加一个 `FurnitureObject(...)`，`mjcf_path=furniture_mjcf_path("<类别>", "<实例>")`。

### 新建一个客厅布局（如 `LivingRoomCozy`）
```python
class LivingRoomCozy(LivingRoomEnv):
    def _get_furniture(self):
        return [ FurnitureObject(...), ... ]
    def get_ep_meta(self):
        ep = super().get_ep_meta(); ep["lang"] = "..."; return ep
```
然后在 `living_room/__init__.py` 和 `robocasa/__init__.py` 里 import 它以触发注册。

### 换场地（不同房型/尺寸）
改 `floor_arena.xml`（墙的位置/高度、地板尺寸、相机），或复制一份新 arena XML + 新建一个 `Arena` 子类，在 `LivingRoomEnv._load_model()` 里替换 `FloorArena()`。

---

## 6. 数据流总结（一图回顾）

```
demo_empty_teleop --env LivingRoomBasic
        │ robosuite.make
        ▼
REGISTERED_ENVS["LivingRoomBasic"]              ← robocasa/__init__.py import 触发注册
        │
        ▼  实例化 + reset → _load_model()
┌─────────────────────────────────────────────┐
│ FloorArena ── floor_arena.xml (地板+墙+相机)  │
│ robot     ── controller_config 创建 + 落位    │
│ furniture ── _get_furniture() 返回            │
│      FurnitureObject ── furniture_mjcf_path   │
│           └─ assets/furniture/.../model.xml   │
│                 └─ visual/model.obj + texture │
└──────────────┬──────────────────────────────┘
               ▼ ManipulationTask 合成
          最终 MuJoCo 模型 → 渲染 / 物理仿真
```
