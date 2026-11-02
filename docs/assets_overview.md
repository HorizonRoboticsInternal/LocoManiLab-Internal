# 资产文件夹说明 (`robocasa/models/assets/`)

`robocasa/models/assets/` 是整个 benchmark 的**仿真内容资源库**——所有出现在 MuJoCo 场景里的几何、纹理、布局都从这里加载。资产整体来自 NVIDIA 的 `PhysicalAI-Robotics-Manipulation-Objects-Kitchen-MJCF` 数据集。

按"功能角色 + 物理存储方式"可分为 **6 大类**。

## 总览（含磁盘占用 / 存储方式）

| 目录 | 占用 | 存储 | 类别 |
|---|---|---|---|
| `objects/` | **20 GB** | 软链接 → `humanoid_vla_benchmark/...` | 可操作物体 |
| `fixtures/` | **1.5 GB** | 本地 | 家具/电器（固定设施）|
| `generative_textures/` | 1.2 GB | 软链接 | AI 生成纹理 |
| `textures/` | 521 MB | 软链接 | 基础材质纹理 |
| `scenes/` | 1.2 MB | 本地 | 场景布局/风格配置 |
| `arenas/` | 12 KB | 本地 | 竞技场骨架 |
| `box_links/` | 60 KB | 本地 | 程序化盒体定义 |
| `groot_dataset_assets/` | 12 KB | 本地 | 机器人/数据集元数据 |
| `novel_instructions/` | 20 KB | 本地 | 语言指令变体 |

> **注**：`objects` / `textures` / `generative_textures` 的完整内容需按任务单独准备。开发环境可通过软链接复用已准备的资源，实际位置取决于本地安装布局。

---

## 1. 可操作物体 `objects/`（最大，20 GB）

机器人能抓取/放置/操作的**自由刚体**，按来源分三个子库：

- **`objaverse/`** — 主力库，按类别组织（`banana/`、`tomato/`、`jam/`、`bottled_water/`…），每个类别下有多个实例（`banana_1`、`banana_2`…）。
- **`aigen_objs/`** — AI 生成物体（`alcohol/`、`apple/`、`artichoke/`… 品类更全）。
- **`lightwheel/`** — 厨房专用件（`blender_jug`、`cheese_grater`、`colander`、`digital_scale`…）。

**单个物体的标准结构**（以 `objaverse/banana/banana_1` 为例）：

```
model.xml         # MJCF 定义（引用下面的网格）
visual/           # 视觉网格 (.obj/.mtl/.png，高精度)
collision/        # 碰撞网格（凸分解后的多个 .obj，用于物理）
```

全库统计：**~10.4 万个 .obj 网格、3500 个 xml、~6400 张贴图**——视觉/碰撞分离是 RoboCasa 物体的标志性格式。

---

## 2. 固定设施 `fixtures/`（1.5 GB，本地）

场景里的**家具与电器**（不被抓取，但可交互——开门、开火等）。23 个子类：

- **厨房电器**：`fridges`（843 文件，最多）、`microwaves`、`ovens`、`stoves`/`stovetops`、`dishwashers`、`toaster_ovens`/`toasters`、`coffee_machines`、`blenders`、`stand_mixers`、`electric_kettles`、`hoods`、`sinks`。
- **厨房结构件**：`cabinets`、`counters`、`handles`（门把手）、`windows`、`accessories`。
- **客厅家具**（living-room stack）：`sofas`、`coffee_tables`、`tv_stands`。
- **`fixture_registry/`** — **47 个 YAML 注册表**，把语义名（`fridge_french_door`、`stove_wide`、`flower_vase`…）映射到具体资产并配置默认参数。这是 fixtures 的"索引层"。

单个 fixture 结构与 object 类似：`model.xml + visual/(obj+mtl+texture) + preview.png`。

---

## 3. 纹理材质（两个软链接库）

- **`textures/`**（521 MB）— 基础 PBR 材质，按材料分类：`wood/`(25)、`tiles/`(17)、`flat/`(15)、`marble/`(10)、`bricks/`、`metals/`、`window_bkgs/`（窗外背景图）、`rotated/`(100，旋转变体)。含两个生成脚本 `generate_color_texture.py`、`rotate_textures.py`。
- **`generative_textures/`**（1.2 GB）— **AI 生成的场景表面纹理**，按用途分：`floor/`、`wall/`、`counter/`、`cabinet/`。用于场景风格（style）的多样化随机。

---

## 4. 场景配置 `scenes/`（1.2 MB，纯 YAML）

定义"房间长什么样"的**数据驱动配置**（不含网格，只是布局描述）。这是 RoboCasa 程序化场景系统的核心，且已扩展出客厅版本：

- `kitchen_layouts/`（`train/` + `test/`）— 厨房布局，fixtures 摆放位置。
- `kitchen_styles/`（`train/` + `test/`）— 厨房风格，材质/配色组合。
- `living_room_layouts/` — 客厅布局（`layout001~004`）。
- `living_room_styles/` — 客厅风格（`style001~003`）。

> layout 的 train/test 划分对应 `split` 机制（pretrain / target / all）。

---

## 5. 场景骨架 `arenas/`（12 KB）

最底层的 MJCF **空场景模板**，布局会被填充进来：

- `empty_kitchen_arena.xml` — 空厨房竞技场。
- `floor_arena.xml` — 仅地板（客厅/通用）。

---

## 6. 辅助 / 元数据（小文件）

- **`box_links/`** — `box_links_assets.json` / `box_links_ds.json`，程序化盒体连杆定义（用于柜门、抽屉等参数化几何）。
- **`groot_dataset_assets/`** — `PandaOmron_embodiment.json` / `_modality.json`，GR00T 数据集导出用的机器人本体与观测模态描述。
- **`novel_instructions/`** — `task_instruction_variants.csv`，任务自然语言指令的改写变体（语言泛化用）。

---

## 客厅自定义资产的可用性

**问题。** 客厅栈用到的几类资产是**本地从仓库外的源生成的**，既不入 git（被 `.gitignore` 的 `fixtures/**/*.{obj,mtl,png,glb,model.xml}` 规则忽略），也**不在** `download_kitchen_assets` 的下载包里：

| 目录 | 大小 | 源（仓库外） |
|---|---|---|
| `fixtures/sofas/Sofa001` | 47M | Meshy AI 导出 |
| `fixtures/coffee_tables/CoffeeTable001` | 54M | Meshy AI 导出 |
| `fixtures/tv_stands/TVStand001` | 28M | Meshy AI 导出 |
| `fixtures/articulated/{Asset124,Asset886,ElectricCooker001}` | 26M | Illusion USD（`scripts/usd_pipeline/` 生成）|

后果：合作者克隆仓库 + 跑 README 的 `download_kitchen_assets` 后，**依赖这些自定义家具的任务仍会因缺少 `model.xml` 报错**。能正常同步的有：所有 Python 代码、`scenes/` 下的 layout/style YAML、`fixture_registry/*.yaml`，以及 SideTable 复用的 lightwheel 凳子和 mug（后两者由官方下载包提供）。

这些资产需要从各自允许分发的来源单独准备，保留转换参数和许可说明；仓库不提供个人机器间的上传或同步入口。
需要家具导入流程时，参见 [家具工作流](furniture_workflow.md) 和 [USD 资产流程](usd_articulated_asset_pipeline.md)。
首次运行优先使用 README 中的 EmptyEnv；它不需要这些自定义家具。

---

## 一句话总结

资产分两层：

1. **几何/纹理实体**（`objects` 可抓物体、`fixtures` 家具电器、`textures`+`generative_textures` 材质）——占了 ~23 GB，大件靠软链接共享。
2. **配置/元数据**（`scenes` 布局风格、`arenas` 骨架、`box_links`/`groot`/`novel_instructions`）——KB 级 YAML/JSON/CSV。

前者是"画什么"，后者是"怎么组装和标注"。
