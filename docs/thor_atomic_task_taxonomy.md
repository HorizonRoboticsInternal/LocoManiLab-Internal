# THOR 原子任务设计

本文档从底层技能出发，定义 iTHOR / ProcTHOR 任务的原子任务 taxonomy。原则是：

- **任务类型共用**：benchmark 对外使用统一的原子技能分类，便于统计难度和组合复合任务。
- **任务实例分场景落地**：iTHOR 和 ProcTHOR 不强行共用同一个环境类；每个场景族按自身 metadata、资产、空间结构做稳定实例。
- **先落地再抽象**：先保证每个原子任务在 1-2 个稳定场景里可视化和采集可靠，再把重复成功判据、语言模板、位姿逻辑抽成 mixin/helper。

## 设计层级

### 1. Atomic Skill

Atomic skill 是 benchmark 层面的统一动作语义，例如 `Approach`、`Lift`、`PickPlaceOnSurface`。这一层只定义：

- 输入：目标物体、目标家具、目标 surface、目标关节等。
- 初始状态：机器人相对 ref 的站位、物体初始位置、关节初始开合状态。
- 成功条件：距离、接触、关节 qpos、物体高度、是否释放等。
- 难度标签：是否需要下肢移动、是否需要双手、是否需要精细抓取、是否需要全身姿态变化。

### 2. Scene Adapter

Scene adapter 处理场景族差异：

- iTHOR：单房间、手工场景质量较高，适合近场桌面/家具任务和房间类别覆盖。
- ProcTHOR：多房间、物体更杂，适合跨房间移动、开门穿行、长距离搬运。
- RoboCasa Kitchen：已有 fixture 系统和 atomic 任务，可作为设计参照，不直接强塞给 THOR。

### 3. Task Instance

Task instance 是实际可运行的 `--env`：

- 绑定具体场景族、房间类别、目标物体和成功条件。
- 可以是同一个 atomic skill 的不同实例，例如 `IThorBedroomLiftBasket` 和 `IThorLivingRoomLiftStatue` 都属于 `LiftObject`。
- 对外记录 `atomic_skill`、`difficulty_level`、`task_family`、`scene_family`，方便后续采样和评估。

## 原子任务分类

| Atomic skill | 中文描述 | 核心成功条件 | iTHOR 第一批实例 | ProcTHOR 第一批实例 | 难度 |
|---|---|---|---|---|---|
| `ApproachFurniture` | 走到目标家具前 | 机器人 base 到目标家具距离小于阈值并保持若干步 | `IThorKitchenApproachFridge`, `IThorBedroomApproachBed`, `IThorLivingRoomApproachSofa`, `IThorBathroomApproachToilet` | 暂不优先，ProcTHOR 更适合跨房间导航 | L1 |
| `LiftObject` | 抓起并举高物体 | 物体高度超过初始高度/目标高度 | `IThorBedroomLiftBasket`, `IThorLivingRoomLiftStatue` | 可后续加 `ProcTHORLiftBottle` / `ProcTHORLiftBox` | L1 |
| `PickPlaceOnSurface` | 从一个 surface 拿起物体并放到另一个 surface | 物体 bbox 在目标 surface 范围内且已释放 | — | `ProcTHORTableToTable`, `ProcTHORContainerObjectToTable` 中可拆出 on-surface 子技能 | L1-L2 |
| `PlaceIntoReceptacle` | 把物体放进容器 | `check_obj_in_receptacle` 或 bbox 在容器内部，且已释放 | — | `ProcTHORTrashToBin` | L1-L2 |
| `OpenDrawer` | 拉开抽屉 | 目标 drawer joint qpos 超过阈值并保持若干步 | 暂不优先，iTHOR 抽屉资产/抓取点需要筛选 | `ProcTHOROpenDrawer` | L1-L2 |
| `OpenDoor` | 打开门 | 目标 hinge qpos 超过阈值 | iTHOR 可后续选 fridge/cabinet door | `ProcTHOROpenDoorWalkThrough` 的开门阶段 | L1-L2 |
| `CrossRoomCarry` | 跨房间搬运物体 | 物体到目标房间目标区域，且已释放 | 不适合 iTHOR 单房间 | `ProcTHORRoomToRoomCarry` | L3 |
| `WalkThroughDoor` | 开门并穿过门洞 | 门打开且机器人 base 进入门后区域 | 不适合普通 iTHOR 单房间 | `ProcTHOROpenDoorWalkThrough` | L3 |
| `SitOrPostureChange` | 坐下/弯腰/伸手等全身姿态任务 | 上身高度变化、base 到家具距离、保持若干步 | 可后续设计床边弯腰/桌边伸手 | 暂不优先，旧 `ProcTHORSitOnSofa` 可作为实验任务 | L2-L3 |

## 第一阶段任务集

第一阶段只追求覆盖“稳定、可采集、可解释”的原子技能，不追求数量。

### iTHOR

iTHOR 用于覆盖四类房间的近场任务：

| 房间 | 推荐 atomic skill | 推荐 env | 说明 |
|---|---|---|---|
| kitchen | `ApproachFurniture` | `IThorKitchenApproachFridge` | 适合冰箱相关导航任务。 |
| living room | `ApproachFurniture`, `LiftObject` | `IThorLivingRoomApproachSofa`, `IThorLivingRoomLiftStatue` | 适合 sofa / coffee table 附近任务。 |
| bedroom | `ApproachFurniture`, `LiftObject` | `IThorBedroomApproachBed`, `IThorBedroomLiftBasket` | basket lift 是 bedroom 的代表任务。 |
| bathroom | `ApproachFurniture` | `IThorBathroomApproachToilet` | 当前以 toilet 导航为主。 |

### ProcTHOR

ProcTHOR 用于补充多房间和长距离任务：

| 场景 | 推荐 atomic skill | 推荐 env | 说明 |
|---|---|---|---|
| `train_18` | `OpenDrawer`, `PickPlaceOnSurface` | `ProcTHOROpenDrawer`, `ProcTHORContainerObjectToTable`, `ProcTHORTableToTable` | 先保留人工标定，保证可视化和采集稳定。 |
| `train_4` | `OpenDoor`, `WalkThroughDoor` | `ProcTHOROpenDoorWalkThrough` | 作为全身移动和门交互样例。 |
| `train_0` | `CrossRoomCarry` | `ProcTHORRoomToRoomCarry` | 多房间搬运的核心样例。 |
| `train_18` | `PlaceIntoReceptacle` | `ProcTHORTrashToBin` | 可作为物体投放/清理任务。 |

## 难度分层

### L1：近场单技能

特点：

- 机器人基本不需要下肢移动，或只需要很小的 base 调整。
- 任务目标单一，成功条件直接。
- 适合最早采集和调控制器。

包含：

- `ApproachFurniture`
- `LiftObject`
- `PickPlaceOnSurface`
- `PlaceIntoReceptacle`
- `OpenDrawer`
- `OpenDoor`

### L2：短程组合

特点：

- 需要 pick + place 或 open + pick 等两步组合。
- 可以有小范围转身，但不强调跨房间移动。

包含：

- `DrawerObjectToTable`
- `PackBasket`

### L3：长距离/全身组合

特点：

- 需要明显下肢移动、跨房间、开门通过、长距离搬运。
- 数据采集难度明显高于桌面任务。

包含：

- `CrossRoomCarry`
- `WalkThroughDoor`
- `TrashToBin` 如果目标距离较远，也归入 L3。

### L4：长过程多阶段任务

暂不立即实现，等 L1-L3 稳定后再设计：

- 传送带分拣。
- 多目标清洁/整理桌面。
- 开柜、取物、移动、分类放置的长过程任务。

## 实施顺序

1. **冻结第一批 atomic task 列表**：先只维护上表任务，不再临时增加大量新 env。
2. **给每个 env 补 metadata**：至少包含 `atomic_skill`、`difficulty_level`、`scene_family`、`task_family`。
3. **逐个可视化检查**：每个 env 固定 scene-id 先看机器人初始位、目标物体位置、碰撞和尺度。
4. **只在稳定后抽 helper**：优先抽成功判据和 ep_meta，不优先抽机器人站位生成。
5. **从 atomic 组合 composite**：例如 `OpenDrawer + PickPlaceOnSurface` 组合成 drawer-to-table；`OpenDoor + CrossRoomCarry` 组合成跨房间任务。

## 文件组织

相似的原子任务可以写在同一个文件里，参考 RoboCasa kitchen atomic 的组织方式。文件按“技能族”分，而不是按单个 env 分。

### iTHOR

iTHOR 的具体任务实例仍然保留在各自房间目录；跨房间复用的原子任务实现放在 `ithor_atomic` 里。这样既能保留 bedroom / kitchen / bathroom 的场景语义，又不会把 `LiftObject`、`ApproachFurniture` 误写成某个房间专属能力。

```text
robocasa/environments/ithor_atomic/
  navigation.py        # RoomNavMixin / ApproachFurniture implementation
  pick_place.py        # RoomSurfacePnP / RoomPackIntoReceptacle implementation
  lift.py              # reserved for future shared LiftObject implementation

robocasa/environments/ithor_kitchen/tasks/
  atomic/               # 单一技能或同一无序谓词族
    approach_fridge.py
    articulation.py
    apple_on_plate.py
    pack_produce_into_box.py
  composite/            # 跨技能或要求阶段顺序
    approach_open_fridge.py
    drawer_pnp.py
    apple_cabinet_pnp.py

robocasa/environments/ithor_living_room/tasks/
  approach_sofa.py
  living_room_pnp.py
  lift_statue.py
  bring_can_to_sofa.py

robocasa/environments/ithor_bedroom/tasks/
  approach_bed.py
  bedroom_pnp.py
  lift_basket.py
  bring_candle_to_bed.py

robocasa/environments/ithor_bathroom/tasks/
  approach_toilet.py
  bathroom_pnp.py
  bring_soap_to_bin.py
```

`ithor_base.py` 只负责场景加载、surface、物体和通用 helper；`RoomSurfacePnP`、`RoomNavMixin`、`RoomPackIntoReceptacle`、`RoomLiftMixin` 等共享原子任务实现放在 `ithor_atomic`。具体任务类仍由各房间 `tasks/` 包导入和注册，避免 `ithor_atomic` 反向 import 具体任务导致循环依赖。

### ProcTHOR

ProcTHOR 更适合按技能族组织，因为它的任务通常跨房间或绑定具体 train scene：

```text
robocasa/environments/procthor/tasks/
  articulation.py      # OpenDrawer / OpenDoor
  pick_place.py        # TableToTable / DrawerObjectToTable
  receptacle.py        # TrashToBin
  navigation.py        # WalkThroughDoor / future room navigation
  carry.py             # RoomToRoomCarry
```

旧文件可以逐步迁移，不一次性大改。迁移顺序建议：

1. 先新建技能族文件，并把旧类原样搬进去，只改 import / registry。
2. 每搬一个文件跑一次 `visualize_scene` 看初始位姿和目标物。
3. 类名和 `--env` 名称保持不变，避免破坏已有采集脚本。
4. 等所有任务稳定后，再删除旧文件或保留 compatibility import。

## 当前不做的事情

- 不把 iTHOR / ProcTHOR baked room 强行转换成 RoboCasa kitchen fixture layout。
- 不要求所有 iTHOR 场景一开始都可采；先从稳定场景开始。
- 不用一个统一基类强行覆盖所有任务；共享 taxonomy 优先于共享代码。
- 不在 ProcTHOR 上自动选择机器人初始位置，除非经过可视化验证；当前优先保留人工标定。
