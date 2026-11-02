# Kitchen 场景：任务与资产分布分析

> 统计基准：`main` 分支当前代码与已下载资产（统计日期 2026-06-01）。
> 数据来源：`robocasa/environments/kitchen/`、`robocasa/models/`、`robocasa/utils/dataset_registry.py`，
> 计数通过遍历注册表（`ALL_KITCHEN_ENVIRONMENTS`）、AST 解析 `kitchen_objects.py` 与文件系统统计得到。

本文给出 RoboCasa / RoboCasa365 仓库中 **Kitchen 场景**的任务体系与资产分布的精确统计与内容分析。

---

## 一、任务总览

| 维度 | 数量 |
|---|---|
| **注册环境总数**（`ALL_KITCHEN_ENVIRONMENTS`） | **375** |
| ├ Atomic（原子技能） | 74 |
| └ Composite（复合活动） | 301 |
| Atomic 任务模块文件 | 15 个 `.py` |
| Composite 活动目录 | 60 个 |
| Composite 任务模块文件 | 301 个 `.py` |
| 带数据集的任务（`dataset_registry`） | 317（atomic 65 + composite 252） |

> Atomic 目录共有 78 个类定义，元类 `KitchenEnvMeta` 排除了 4 个抽象基类
> （`PickPlace`、`ManipulateDoor`、`OpenDoor`、`CloseDoor`，以及 robosuite 的 `MG_Robocasa_Env`），
> 故实际注册 74 个。

任务呈**金字塔结构**：74 个可组合的原子技能（"动词 × 家电"原语）→ 301 个多步复合活动。
复合任务量约为原子的 4 倍。

---

## 二、Atomic 原子任务（74 个，15 个模块）

按"家电 / 技能族"组织，每族包含 open/close、turn-on/off、pick-place 等动词变体。

| 模块 | 代表技能 |
|---|---|
| `kitchen_pick_place.py` | 20 个 PickPlace 变体（Counter ↔ Sink/Stove/Oven/Microwave/Cabinet/Drawer/Blender/StandMixer/ToasterOven/Fridge…）——**数量最大的一族** |
| `kitchen_doors.py` | Open/Close × Cabinet/Door/Fridge/Dishwasher/DropDownDoor + FridgeDrawer |
| `kitchen_drawer.py` | Open/Close Drawer、ManipulateDrawer |
| `kitchen_microwave.py` | TurnOn/Off、PressButton、Open/Close |
| `kitchen_oven.py` | Open/Close、Preheat、SlideRack |
| `kitchen_stove.py` | TurnOn/Off、LowerHeat、ManipulateKnob |
| `kitchen_sink.py` | TurnOn/Off Faucet、AdjustWaterTemperature、TurnSpout |
| `kitchen_coffee.py` | SetupMug、ServeMug、StartMachine、MakeIcedCoffee |
| `kitchen_blender.py` / `kitchen_stand_mixer.py` / `kitchen_electric_kettle.py` / `kitchen_toaster.py` / `kitchen_toaster_oven.py` | 各家电的开合 / 启停 / 温控 |
| `kitchen_navigate.py` | NavigateKitchen（移动底盘导航） |
| `g1_standing.py` | G1Standing（人形机器人专用站立） |

**技能动词分布**：以 Open/Close（门、抽屉、盖）、TurnOn/Off（启停）、PickPlace（取放）三类为主干，
覆盖厨房所有主要家电的可交互状态。

---

## 三、Composite 复合任务（301 个，60 类活动）

每个目录 = 一类真实烹饪 / 整理活动，内含 1–19 个具体任务。

### 按任务数排序的活动分布

| 任务数 | 活动 |
|---|---|
| 19 | washing_dishes |
| 13 | setting_the_table |
| 11 | washing_fruits_and_vegetables |
| 8 | restocking_supplies, managing_freezer_space, loading_fridge, frying, clearing_table, boiling |
| 7 | toasting_bread, sauteing_vegetables, portioning_meals, baking |
| 6 | serving_food, serving_beverages, sanitizing_surface, reheating_food, preparing_sandwiches, mixing_ingredients, microwaving_food, defrosting_food, chopping_food, brewing |
| 5 | tidying_cabinets_and_drawers, storing_leftovers, snack_preparation, making_smoothies, garnishing_dishes, broiling_fish, arranging_buffet |
| 4 | sanitizing_cutting_board, packing_lunches, organizing_recycling, filling_serving_dishes, adding_ice_to_beverages |
| 3 | steaming_vegetables, steaming_food, slow_cooking, slicing_meat, seasoning_food, preparing_marinade, plating_food, organizing_utensils, organizing_dishes_and_containers, mixing_and_blending, measuring_ingredients, making_toast, making_tea, making_juice, cleaning_sink, cleaning_appliances, arranging_condiments, arranging_cabinets |
| 2 | sorting_ingredients, preparing_hot_chocolate, meat_preparation, making_salads, loading_dishwasher, chopping_vegetables |
| 1 | simmering_sauces |

### 内容主题聚类

- **清洗 / 卫生**：washing_dishes、washing_fruits_and_vegetables、sanitizing_surface / cutting_board、cleaning_sink / appliances、loading_dishwasher
- **热加工**：boiling、frying、baking、broiling_fish、steaming_food / vegetables、sauteing_vegetables、simmering_sauces、slow_cooking、reheating_food、microwaving_food
- **备餐 / 装盘**：plating_food、serving_food / beverages、filling_serving_dishes、garnishing_dishes、portioning_meals、setting_the_table、arranging_buffet
- **整理 / 收纳**：tidying_cabinets_and_drawers、organizing_recycling / utensils / dishes_and_containers、loading_fridge、managing_freezer_space、storing_leftovers、restocking_supplies
- **预处理**：chopping_food / vegetables、slicing_meat、measuring_ingredients、seasoning_food、preparing_marinade、mixing_ingredients

---

## 四、物体资产（Objects）

| 维度 | 数量 |
|---|---|
| **物体语义类别**（`OBJ_CATEGORIES`） | **198** |
| **采样组**（`OBJ_GROUPS`，含 all/food/container/cookware 等聚合组） | 230 |
| 物体资产总磁盘占用 | **~20 GB** |

### 三套资产来源及实例数

一个类别可对应多个 3D 变体（instance）。

| 来源 | 覆盖类别 | 实例（变体）数 |
|---|---|---|
| aigen（AI 生成） | 153 | **1594** |
| lightwheel | 58 | 922 |
| objaverse | 84 | 914 |

### 类别按 type 分布（共 198 类）

| 数量 | type | | 数量 | type |
|---|---|---|---|---|
| 32 | vegetable | | 8 | condiment |
| 27 | receptacle | | 7 | bread_food |
| 19 | fruit | | 7 | utensil |
| 18 | meat | | 6 | dairy |
| 16 | packaged_food | | 5 | cleaner |
| 14 | tool | | 4 | stackable |
| 13 | drink | | 4 | spice |
| 12 | sweets | | 3 | alcohol |
| 9 | cooked_food | | 3 | pastry |
| | | | 2 | decoration |
| | | | 1 | iced_item / lid |

→ 以**蔬菜、容器 / 餐具、水果、肉类、包装食品**为主体，明显偏食材类。

### 物理属性标记（标 `True` 的类别数）

| 属性 | 类别数 | 用途 |
|---|---|---|
| graspable | 157 | 可抓取 |
| fridgable | 139 | 可放入冰箱 |
| washable | 137 | 可清洗 |
| freezable | 89 | 可冷冻 |
| microwavable | 63 | 可微波 |
| cookable | 62 | 可烹饪 |

这些属性驱动任务可行性：如仅 `cookable` 物体可进 frying / baking 任务，`washable` 进清洗任务。

---

## 五、场景：布局与风格

| 维度 | 总数 | pretrain / train | target / test |
|---|---|---|---|
| Kitchen 布局（`LayoutType`） | 60 | 50（layout011–060） | 10（layout001–010） |
| Kitchen 风格（`StyleType`） | 60 | 50 | 10 |

- 理论场景组合 = 60 layout × 60 style，即文档所称的 "2500+ 厨房场景"（受 `EXCLUDE_LAYOUTS` 与家电限制裁剪）。
- `split` 机制：`pretrain`（id `-2`）/ `target`（`1..10`）/ `all`（id `-3`），同时控制 layout、style、物体实例划分。
- 部分布局对特定家电有排除表：`OVEN_EXCLUDED_LAYOUTS`、`DOUBLE_CAB_EXCLUDED_LAYOUTS` `[32, 41, 59]`、
  `DINING_COUNTER_EXCLUDED_LAYOUTS`、`ISLAND_EXCLUDED_LAYOUTS`、`FREEZER_EXCLUDED_LAYOUTS`。

---

## 六、固定家具 / 家电资产（Fixtures，~1.6 GB）

每个家电族有大量纹理 / 型号变体（按 XML 计）：

| 数量 | fixture 族 | | 数量 | fixture 族 |
|---|---|---|---|---|
| 57 | handles | | 25 | electric_kettles |
| 50 | windows | | 25 | dishwashers |
| 50 | toasters | | 25 | blenders |
| 50 | toaster_ovens | | 24 | stovetops |
| 50 | sinks | | 18 | cabinets |
| 50 | microwaves | | 3 | articulated |
| 50 | fridges | | 2 | counters |
| 50 | coffee_machines | | 1 | tv_stands |
| 26 | stoves | | 1 | sofas |
| 25 | stand_mixers | | 1 | coffee_tables |
| 25 | ovens | | 25 | hoods |

> `sofas` / `tv_stands` / `coffee_tables` 为 living-room 扩展所用。

固定件 Python 类共 25 个模块（`stove.py`、`sink.py`、`fridge.py`、`microwave.py`、`oven.py`、
`dishwasher.py`、`coffee_machine.py`、`blender.py`、`stand_mixer.py`、`electric_kettle.py`、
`toaster.py`、`toaster_oven.py`、`cabinets.py`、`counter.py`、`hood.py`、`windows.py` 等），
通过 `FixtureType` 枚举进行语义引用。

---

## 七、其他资产

| 资产 | 数量 / 占用 |
|---|---|
| 基础纹理（png/jpg） | 222 |
| 生成式纹理（`generative_textures`） | 1.2 GB / 6 组 |
| box_links、groot_dataset_assets、novel_instructions | 小型辅助资产 |

---

## 八、关键洞察

1. **任务呈金字塔结构**：74 个原子技能（可组合的"动词 × 家电"原语）→ 301 个复合活动（多步真实烹饪流程），
   复合任务量是原子的约 4 倍。
2. **数据集覆盖率**：375 个注册环境中 317 个（约 85%）已有 HDF5 演示数据集（atomic 65 + composite 252），
   约 58 个环境暂无数据。
3. **资产以食材为核心**：198 物体类别中蔬果肉类占近半；aigen 提供最大规模（1594 变体）的程序化多样性，
   objaverse / lightwheel 补充真实扫描资产。物体总量 ~20 GB，是仓库最大资产块。
4. **场景多样性**：60 × 60 的 layout / style 矩阵 + train/test 5:1 划分，构成跨场景泛化基准的基础。
5. **人形 / 客厅扩展**：本仓库含 G1 / GR1 人形机器人支持（`g1_standing.py`、`env_utils._ROBOT_POS_OFFSETS`
   中的机器人基座偏移）及 living-room 场景骨架——这是相对原版 RoboCasa 的扩展方向。

---

## 附：复现统计的方法

```sh
# 注册环境数 / 数据集数
python -c "from robocasa.environments import ALL_KITCHEN_ENVIRONMENTS; print(len(ALL_KITCHEN_ENVIRONMENTS))"
python -c "from robocasa.utils.dataset_registry import ATOMIC_TASK_DATASETS, COMPOSITE_TASK_DATASETS; \
           print(len(ATOMIC_TASK_DATASETS), len(COMPOSITE_TASK_DATASETS))"

# Composite 活动目录与任务文件数
ls -d robocasa/environments/kitchen/composite/*/ | grep -v __pycache__ | wc -l
find robocasa/environments/kitchen/composite -name "*.py" ! -name "__init__.py" | wc -l

# 物体类别数（OBJ_CATEGORIES / OBJ_GROUPS）
python -c "from robocasa.models.objects.kitchen_objects import OBJ_CATEGORIES, OBJ_GROUPS; \
           print(len(OBJ_CATEGORIES), len(OBJ_GROUPS))"

# 物体实例变体数（按来源）
find robocasa/models/assets/objects/aigen_objs  -mindepth 2 -maxdepth 2 -type d | wc -l
find robocasa/models/assets/objects/objaverse   -mindepth 2 -maxdepth 2 -type d | wc -l
find robocasa/models/assets/objects/lightwheel  -mindepth 2 -maxdepth 2 -type d | wc -l

# 布局 / 风格数
grep -cE "LAYOUT[0-9]+ =" robocasa/models/scenes/scene_registry.py
grep -cE "STYLE[0-9]+ ="  robocasa/models/scenes/scene_registry.py
```
