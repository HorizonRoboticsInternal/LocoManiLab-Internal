# iTHOR (MolmoSpaces) 场景：大类 / 任务 / 调用脚本

> 把 [MolmoSpaces](https://github.com/allenai/molmospaces) 的 iTHOR 房间作为**背景**接入 RoboCasa，
> 按房间类型拆成**和 `kitchen`/`living_room` 同级的 4 个场景大类**，G1 在其中做操作任务。
> 运行环境:conda env **`tele_benchmark`**（仓库根目录）。通用采集/可视化脚本与 kitchen/living-room
> 共用同一套 G1 teleop 核心（`docs/g1_data_collection_pipeline.md`、`docs/living_room_vla_pipeline.md`）。

---

## 0. 前置:资产与环境变量

iTHOR 场景/物体资产由 molmospaces 仓库下载到本地缓存，RoboCasa 通过**软链目录**读取。每个新终端都要:

```sh
conda activate tele_benchmark
cd /path/to/humanoid_benchmark
export MLSPACES_ASSETS_DIR=~/molmospaces/sample_resources   # iTHOR 软链资源目录（必需）
export MUJOCO_GL=glfw     # 交互窗口用 glfw;无显示离屏渲染用 egl
```

> 首次加载某场景会在场景目录旁生成 `FloorPlanN_physics.robocasa.xml`（预处理产物，自动缓存）。

---

## 1. 场景大类、环境与任务

iTHOR 的 FloorPlan 编号**本身就编码房间类型**，所以拆成 4 个独立大类（目录在
`robocasa/environments/` 下，与 `kitchen`/`living_room` 同级）。每个大类有：自己的**可运行基类
env**（自由探索，无任务）+ 自己的**任务** + 自己的注册表。

| 大类目录 | 基类 env | 任务 | FloorPlan / scene-id | 已下载 |
|---|---|---|---|---|
| `ithor_kitchen` | `IThorKitchenEnv` | `IThorKitchenApproachFridge` | 1–30 | 1–5 |
| `ithor_living_room` | `IThorLivingRoomEnv` | `IThorLivingRoomLiftStatue` | 201–230 | 201–205 |
| `ithor_bedroom` | `IThorBedroomEnv` | `IThorBedroomLiftBasket` | 301–330 | 301–305 |
| `ithor_bathroom` | `IThorBathroomEnv` | `IThorBathroomApproachToilet` | 401–430 | 401–405 |

- **基类 env**（`IThor*Env`）= 自由探索该房间，无任务。
- **任务 env**（`IThor*` 任务类）= 在对应房间执行导航或物体操作，并定义任务专属成功条件。

> **env 决定房间类型，scene-id 决定该类型里的具体 FloorPlan**。两者不可混搭:给 living-room env
> 传厨房 id（如 `--scene-id 1`）会直接报错并提示正确范围。

查看全部已注册 iTHOR 环境:
```sh
python -c "import robocasa; from robocasa.environments import ALL_ITHOR_ENVIRONMENTS as A; print(sorted(A))"
```

---

## 2. 脚本与参数

两个通用脚本（对任意 env 通用，含 kitchen/living-room）：

### `visualize_scene` — 可视化
| 参数 | 说明 | 默认 |
|---|---|---|
| `--env` | 环境/任务名（必填） | — |
| `--scene-id` | 该房间类型里的 FloorPlan id（省略=随机） | None |
| `--probe` | 开交互窗口并显示 3D 探针，用于读取世界坐标 | 关 |
| `--calibrate-spawn` | 开交互窗口并保存 G1 初始位姿 | 关 |
| `--robot` | 机器人 | `G1FloatingBody` |
| `--fixed-scene` / `--scene-seed N` | 固定场景（每次 reset 一致） | 关 / 0 |

### `collect_demos_g1` — 数据采集
| 参数 | 说明 | 默认 |
|---|---|---|
| `--env` | 任务 env 名（必填，需有 `_check_success`） | — |
| `--scene-id` | FloorPlan id（省略=该类型随机） | None |
| `--g1-control` | `keyboard`（下肢移动，无需 PICO）/ `pico-manager` / `pico-teleop`（上肢 VR，需头显） | `keyboard` |
| `--pico-source` | `real` / `mock`（PICO 数据源） | `real` |
| `--fixed-scene` / `--scene-seed N` | 固定场景（采集/评测用同一组以对齐） | 关 / 0 |
| `--fixed-objects` | 固定物体实例、随机位置 | 关 |
| `--robot` | 机器人 | `G1FloatingBody` |
| `--out` | 输出目录（LeRobot 数据集根本身；建议绝对路径） | `<repo>/outputs/<backend>/<env>_<时间>` |
| `--control-freq` | 控制/录制频率 | 50 |
| `--max-episodes` | 最多采集 episode 数 | 1000 |
| `--no-viewer` | headless（不开窗，管线自检用） | 关 |
| `--no-show-camera` | 不开实时 ego 相机预览窗 | 关 |
| `--debug-record-steps N` | 录 N 步存一条就退出（管线自检） | 0 |

> **场景参数自动转发**:脚本按目标 env 的构造签名（沿 MRO 内省）只转发它接受的参数——iTHOR 房间走
> `--scene-id`，kitchen/living-room 走 `--layout/--style`，`--fixed-scene` 通用。不支持的自动跳过并提示。

---

## 3. 调用示例

### 可视化（交互窗口，鼠标旋转/缩放，键盘驱动 G1）
```sh
# 厨房导航任务（FloorPlan2）
MUJOCO_GL=glfw python -m robocasa.scripts.visualize_scene \
    --env IThorKitchenApproachFridge --scene-id 2

# 卧室自由探索（随机选 301-305 中一个）
MUJOCO_GL=glfw python -m robocasa.scripts.visualize_scene --env IThorBedroomEnv
```

### 数据采集（PICO VR 上肢遥操作，需头显）
```sh
# 厨房导航，固定 FloorPlan1
MUJOCO_GL=glfw python -m robocasa.scripts.collect_demos_g1 \
    --env IThorKitchenApproachFridge --scene-id 1 \
    --g1-control pico-manager --pico-source real \
    --fixed-scene --scene-seed 0

# 卫生间导航，随机选 401-405
MUJOCO_GL=glfw python -m robocasa.scripts.collect_demos_g1 \
    --env IThorBathroomApproachToilet --g1-control pico-manager --pico-source real
```

### 数据采集（键盘移动，无需硬件）/ 管线自检（headless）
```sh
# 键盘（下肢移动，适合导航类）
MUJOCO_GL=glfw python -m robocasa.scripts.collect_demos_g1 --env IThorLivingRoomApproachSofa --scene-id 202

# headless 自检（不开窗、不需 PICO，录 5 步存一条就退出）
MUJOCO_GL=egl python -m robocasa.scripts.collect_demos_g1 \
    --env IThorKitchenApproachFridge --scene-id 1 --fixed-scene --scene-seed 0 \
    --debug-record-steps 5 --no-viewer
```

输出:仓库根目录下 `outputs/<backend>/<env>_<时间>`（GR00T `unitree_g1_sonic` 格式），可直接进
`docs/living_room_vla_pipeline.md` 的预处理→训练→闭环验证（把 `--env` 换成 iTHOR 任务即可）。

> **怎么换场景/任务**:换房间 → 换 `--env`（如 `IThorKitchenApproachFridge`→`IThorBedroomLiftBasket`）;
> 换同类型里的具体场景 → 换 `--scene-id`（如厨房 `--scene-id 1`→`5`），或省略让它随机选。

---

## 4. 机器人站位（自动 + 校准）

- **自动**:机器人站开阔空地，**面朝场景自带的台面**（物体就放在该台面上，不额外加桌子）。射线检测
  含"屋顶净空"判据，保证站位在房间内（不会跑墙外/柜下）。
- **够不到的情况**:密集房间（如带中岛的厨房）里场景台面可能离机器人较远、够不到——这是不加额外桌子的
  取舍。需要精确可达站位时用下面的覆盖。
- **覆盖**（优先级从高到低）:`ep_meta` > `robot_base_pos` 构造参数 > 按场景校准 spec
  `robocasa/models/assets/scenes/ithor_scene_specs.yaml` > 自动。紧凑房间想精确指定就填校准表。
  同一 YAML 的 `robot_scale` 可按场景（或任务）设置 G1 的统一物理缩放；目前 `FloorPlan3` 为 `1.5`。

用 viewer 标定 G1 出生点（默认仅覆盖当前任务；自动采样仍是未标定场景的 fallback）：

```sh
MUJOCO_GL=glfw python -m robocasa.scripts.visualize_scene \
    --env IThorKitchenApproachFridge --scene-id 1 --calibrate-spawn
```

按 `W/S/A/D` 平移、`Z/X` 旋转、`1-4` 调整平移步长；按 Space 查看碰撞数，按 Enter
将无碰撞的当前位姿写入 `ithor_scene_specs.yaml`。按 `N` 会切换到同房型的下一个已下载
FloorPlan（当前未保存的移动会丢弃）。按 `H` 可在主 viewer 中切换自由场景视角与 G1 头部相机视角，
检查当前出生点和朝向下机器人实际看到的内容。如需让该站位用于此 FloorPlan 的所有 iTHOR 任务，
加 `--calibration-scope default`。

---

## 5. 下载更多场景

目前每类下了 5 个。下载更多（在 molmospaces 仓库的 venv 里，thor 物体库已装则只拉场景文件）:

```python
# 在 molmospaces 仓库（默认 ~/molmospaces，即 $MOLMOSPACES_REPO）下,其 .venv 里运行
from molmo_spaces.molmo_spaces_constants import get_scenes
from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path
idx = get_scenes("ithor", "train")["train"]
for fp in [6, 7, 206, 306, 406]:          # 想要的 FloorPlan 编号
    install_scene_with_objects_and_grasps_from_path(idx[fp], exclude_thor=False)
```
下载后 env 自动发现，直接 `--scene-id 6` 即可。**iTHOR 场景依赖 `objects/thor` 物体库(~3.2G)，须保留。**

---

## 6. 加新任务 / 新场景大类

**加一个新任务**（某个房间类型下）:在该大类的 `tasks/` 里新建模块，子类化对应房间 env，实现
`_get_obj_cfgs` / `_check_success` / `get_ep_meta["lang"]`，在该大类 `__init__.py` 导入即注册。
最省事:复用 `ithor_base.RoomSurfacePnP` mixin（场景自带台面 + 放进碗），只写 `_source_obj_cfg`，
例如 `robocasa/environments/ithor_kitchen/tasks/kitchen_pnp.py`。

**加一个新场景大类**:在 `robocasa/environments/` 下新建包，照 `ithor_kitchen` 结构写
`<cat>.py`（`make_ithor_registry()` + 基类 env 设 `ROOM_TYPE`）+ `tasks/`，在
`robocasa/environments/__init__.py` 和 `robocasa/__init__.py` 注册。通用采集/可视化脚本无需改动。

---

## 7. 实现要点（改代码时看）

- **`robocasa/environments/ithor_base.py`** — 共享基类 `IThorRoomEnv`（Arena 加载 / 站位 / 物体放置 /
  物体放在场景自带台面）+ `RoomSurfacePnP` 任务 mixin + `make_ithor_registry()`（每大类一份注册表+元类）。
- **`robocasa/environments/ithor_{kitchen,living_room,bedroom,bathroom}/`** — 4 个大类，各自基类 env + `tasks/`。
- **`robocasa/models/arenas/molmospaces_arena.py`** — 场景 MJCF → robosuite Arena。预处理三坑:
  ① 嵌套默认类内联展开;② geom group 约定相反（视觉 0→1、碰撞 3/4→0），否则材质被抹成纯色;
  ③ 资产相对路径 / 去 size cap / 补室内相机+顶灯。
- **`robocasa/models/scenes/molmospaces_scene_registry.py`** — FloorPlan→房间类型分类 + 按类型列场景。
- **`robocasa/models/assets/scenes/ithor_scene_specs.yaml`** — 按场景 / 任务校准放置面和站位（覆盖自动）。
- **`robocasa/utils/molmospaces_placement.py`** — 射线检测站位（含"屋顶净空"判据，确保在房间内）+ 台面检测。
- **`robocasa/utils/molmospaces_objects.py`** — objaverse 物体 → `MJCFObject`（自动缩放）。
- **`robocasa/utils/g1_data_collection.py`** — 与 kitchen/living-room 共用的 G1 采集核心。

---

## 8. MolmoSpaces 资产格式

资产下载在缓存目录、经软链暴露给 RoboCasa；`$MLSPACES_ASSETS_DIR` 下按类型组织。核心都是
**MuJoCo MJCF + obj/stl 网格 + png 贴图**:

| 类型 | 路径 | 文件格式 |
|---|---|---|
| **场景 scenes** | `scenes/ithor/FloorPlanN_physics.xml` | **MJCF**（整间房:墙/地/家具/可交互关节）;同名 `_physics.json`（THOR 场景定义）、`_physics_metadata.json`（物体清单）、`_physics_map.png`（占用图）、`FloorPlanN_physics/`（该场景专属 obj/mtl/png/meta） |
| **物体 objects / objaverse** | `objects/objaverse/<uid>/` | `<uid>.xml`（**MJCF**,自包含模型）+ `<uid>_visual.obj`+`.mtl`+`_visual_0.png`（视觉）+ `<uid>_colliderN.obj`（凸分解碰撞网格） |
| **物体 objects / thor** | `objects/thor/...` | THOR prefab 组织的 obj/mtl/png/meta（手工室内资产,被 iTHOR 场景引用） |
| **物体元数据** | `objects/objathor_metadata/` | `objects_metadata.json.gz`（标注）+ `clip_features.pkl`（文本检索特征） |
| **机器人 robots** | `robots/<名>/` | `*.xml`（**MJCF**,多种装配）+ `assets/*.stl/.obj/.png` + `*.urdf`（给规划器）+ `*.yml` |
| **抓取 grasps** | `grasps/<源>/<uid>/` | `<uid>_grasps_filtered.npz`（键 `transforms`,4×4 抓取位姿数组） |
| **benchmark** | `benchmarks/<名>/` | 纯 **JSON** 任务定义 |

> 关键:场景 MJCF 用相对路径 `../../objects/...` 引用物体,所以**场景依赖 objects 目录**;iTHOR 场景由
> thor prefab 搭建,**必须有 `objects/thor`(~3.2G)**。

## 9. 导入 RoboCasa 的流程

MolmoSpaces 和 robosuite 都是 MuJoCo MJCF,但 MJCF 约定不同,导入靠以下转换（实现见 §7）:

```
MolmoSpaces 场景 .xml
   │  ① MolmoSpacesArena 预处理（molmospaces_arena.py）:
   │     - 嵌套默认类 __VISUAL_MJT__ 等 → 内联展开成显式属性、删 <default>
   │       (robosuite 浅解析器否则 KeyError)
   │     - geom group 约定相反 → 重映射(视觉 0→1、碰撞 3/4→0)
   │       (否则视觉几何被当碰撞体重上色、material 被抹掉 → 渲染纯色)
   │     - 相对资产路径绝对化 / 去 legacy <size> / 补室内相机 + 顶灯
   ▼
robosuite Arena（= MujocoXML 包装器）
   │  ② ManipulationTask 合并 Arena + G1 机器人 + 物体
   │     (compiler/option 用 robosuite world 基座的,自动丢弃 MolmoSpaces 的 exotic option)
   ▼
可仿真的 MjModel
```

配套:
- **场景分类**（`molmospaces_scene_registry.py`）:按 FloorPlan 编号 → 4 个房间大类。
- **机器人站位**（`molmospaces_placement.py`）:射线检测找开阔空地 + 面朝场景台面（"屋顶净空"判据保证在房间内）。
- **物体导入**:RoboCasa 库物体直接用 `MJCFObject`;**objaverse 物体**（`molmospaces_objects.py`）从其 MJCF
  重建成 RoboCasa 格式 `MJCFObject`（嵌套 body + visual/collision class + bbox + 自动缩放）。

## 10. 脚本一览

**iTHOR / 通用工作流**（本接入新增/常用）:
| 脚本 | 作用 |
|---|---|
| `robocasa.scripts.visualize_scene` | 通用交互可视化，默认打开 viewer。任意 env。 |
| `robocasa.scripts.collect_demos_g1` | 通用 G1 采集；使用选中后端的 LeRobot 记录契约，见 [采集说明](g1_data_collection_pipeline.md)。 |
| `robocasa.scripts.collect_fixed_task` | 固定初态并施加受约束抖动的 G1 采集，共用 `G1_BRIDGE` 和录制核心。 |
| `robocasa.scripts.replay_living_room_dataset` | 保留的 SONIC 数据诊断与开环回放工具；诊断结果不等于训练质量保证。 |
| `robocasa.scripts.deploy_living_room_vla` | 保留的 SONIC token 仿真客户端，需要已有兼容策略服务器或录制动作。 |
| LocoManiLab `lab eval` / `lab replay` | 统一模型工作流，见 [主仓仿真验证说明](https://github.com/LocoManiLab/LocoManiLab/blob/main/docs/sim_validation.md)。 |

**MolmoSpaces 资产下载**（在 molmospaces 仓库 `scripts/assets/` + 其 `.venv`）:
| 脚本 | 作用 |
|---|---|
| `sample_assets.py` | 每类资产只取少量样例下载（省空间）。 |
| `hf_download.py` | 从 HuggingFace 批量下载整 source。 |
| `usda_downloader.py` | 单个 USDa 资产下载 / 场景扁平化（Isaac/Blender 用）。 |

> RoboCasa 自带的 kitchen 采集脚本 `robocasa.scripts.collect_demos`（PandaOmron，非 G1）与本工作流无关,
> 勿与 `collect_demos_g1` 混淆。
