# 当前任务清单

最后更新：2026-10-16。

本文件统一维护当前 benchmark 的程序化客厅、iTHOR 各房间、ProcTHOR 多房间和 Holodeck 任务。沿用现有范围，不包含上游原始 RoboCasa Kitchen 任务库。

每个任务独占一行，任务名使用实际注册环境名。场景范围按当前代码与任务配置核对；列入表格表示存在实现与对应配置，不表示所有任务和场景都经过人工操作验证。

- iTHOR 的 scene-id 为 FloorPlan 后的数字，例如 `13` 对应 `FloorPlan13`。
- Holodeck 的 scene-id 为 train 后的数字，例如 `117` 对应 `train-117`。
- 程序化客厅使用 layout/style，不使用固定 scene-id；表中列出任务构造器的默认配置。
- ProcTHOR 使用 train 场景；未声明固定场景集合的任务标为“动态筛选”，具体场景须包含任务要求的家具、房间及可用空间。
- iTHOR 默认按任务目录和 `TASK_FAMILY` 区分原子/复合任务；本清单将厨房与客厅的部分多物体操作、搬运和投放任务归为复合任务。Holodeck 按 `TASK_FAMILY` 区分。程序化客厅和 ProcTHOR 缺少统一分类字段，按单一操作与包含搬运、导航或多目标操作的任务流程分组；此分类不改变代码。

## 基础环境与示例

基础环境用于场景浏览或自由遥操作，不计入原子/复合任务数量。

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `EmptyEnv` | 不适用 | 在空地上进行自由遥操作，不进行任务成功评分。 |
| `LivingRoomEnv` | 可用 layout/style | 加载程序化客厅，用于场景浏览和自由遥操作。 |
| `IThorKitchenEnv` | 已下载的厨房 FloorPlan | 加载 iTHOR 厨房，用于场景浏览和自由遥操作。 |
| `IThorLivingRoomEnv` | 已下载的客厅 FloorPlan | 加载 iTHOR 客厅，用于场景浏览和自由遥操作。 |
| `IThorBedroomEnv` | 已下载的卧室 FloorPlan | 加载 iTHOR 卧室，用于场景浏览和自由遥操作。 |
| `IThorBathroomEnv` | 已下载的浴室 FloorPlan | 加载 iTHOR 浴室，用于场景浏览和自由遥操作。 |
| `ProcTHOREnv` | 已下载的 ProcTHOR train 场景 | 加载 ProcTHOR 多房间住宅，用于场景浏览和自由遥操作。 |
| `HolodeckEnv` | 已下载的 Holodeck train 场景 | 加载 Holodeck 场景，用于场景浏览和自由遥操作。 |
| `TidyLivingRoom` | layout 1 / style 1（默认） | 整理沙发上的苹果；当前为示例任务，尚未实现成功判定。 |

## TeleLab

| 任务名 | 场景 | 任务描述 |
|---|---|---|
| `TeleLabApproachOpenFridge` | `1`（TeleLabMani） | 先走到冰箱操作位，再打开主冰箱门至至少 80%，并保持 1 秒。 |
| `TeleLabCarryCupFromShelfToDesk` | `1`（TeleLabMani） | 从第二层架子抬起杯子，走到书桌旁并放到桌面上；杯子须先抬离架面，机器人接近书桌后，杯底支撑在桌面上并稳定保持 1 秒。 |
| `TeleLabConveyorPlaceBoxOnDesk` | `1`（TeleLabConveyor） | 将以 0.1 m/s 运行的 2.5 m × 0.7 m、台面高 0.7 m 传送带上最左侧三个半尺寸盒子全部放到 1.2 m × 0.6 m、高 0.65 m 的旁边桌面并释放；盒子中心须在桌面范围内且与桌面接触，共同稳定保持 1 秒。 |

`ProcTHORPlaceToPlaceTask` 是共享实现基类，不作为独立 benchmark 任务列入下表。`HolodeckPlaceBoxFromDeskToFloor` 是 `HolodeckPlaceBoxOnFloor` 的代码别名，不重复计数。

## 程序化场景：客厅

共 12 个任务：7 个原子任务、5 个复合任务。

### 原子任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `ApproachSofa` | layout 3 / style 2（默认） | 走到沙发前并站定。 |
| `ApproachTable` | layout 8 / style 4（默认） | 走到桌前并站定。 |
| `TabletopPickPlace` | layout 7 / style 1（默认） | 将桌上的马克杯放到盘子上。 |
| `TabletopChargerSocket` | layout 7 / style 1（默认） | 拿起桌上的充电器并插入插座。 |
| `TabletopCylinderSocket` | layout 7 / style 1（默认） | 拿起桌上的圆柱并插入圆筒插槽。 |
| `StackCanOnPlate` | layout 10 / style 5（默认） | 将罐子放到盘子上。 |
| `CollectFruitInBowl` | layout 10 / style 5（默认） | 将边桌上的苹果放入茶几上的碗中。 |

### 复合任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `SquatPickPlaceTVStand` | layout 5 / style 1（默认） | 下蹲拿起茶几上的苹果，再将苹果放到身后的电视柜上。 |
| `StoreMugInCabinet` | layout 6 / style 1（默认） | 将桌上的马克杯搬到柜子旁，打开柜门并将杯子放入柜内。 |
| `CarryObjectToTable` | layout 4 / style 3（默认） | 将近处边桌上的马克杯搬到房间另一侧的边桌上。 |
| `FetchSnackFridge` | layout 8 / style 4（默认） | 拿起茶几上的苹果，走到冰箱旁，打开冰箱门并将苹果放入冰箱。 |
| `ClearCoffeeTable` | layout 9 / style 5（默认） | 将茶几上的苹果和罐子都搬到边桌上。 |

## iTHOR：厨房

共 80 个任务：31 个原子任务、49 个复合任务。

导航开关循环的终点位姿与开关阈值沿用对应原子任务配置；转身投放任务的垃圾桶布置在机器人身后约 0.6 米。Scene 6 没有现成垃圾桶，因此不支持两个转身投放任务。苹果香蕉交换只检查同时满足的最终状态，保留 1 秒保持要求。

### 原子任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `IThorKitchenSaltShakerStoolToSinkCounter` | 25 | 原生凳子放在机器人右侧，原生盐罐放在凳面；将盐罐完整放到面向水池时左边的柜台并连续保持 1 秒。 |
| `IThorKitchenCannedDrinkIntoPan` | 13 | 重新设计的 K05：新增可乐并放在锅右侧柜台前沿；将可乐放入灶台原生锅 Pan_13_29，并在锅内连续保持 1 秒。 |
| `IThorKitchenPanLeftToRightBurner` | 8 | 将原生锅 Pan_8_32 从面向灶台时的左前灶位移到右前灶位；重合面积占锅体水平投影面积至少 70%，连续保持 1 秒。机器人站在前排两灶位中间面向灶台。 |
| `IThorKitchenSpongeIntoOpenDrawer` | 8 | 海绵旁最上层抽屉 Drawer_8c0ca37e_5 初始打开 100%，机器人移近操作台并正对水槽，抽屉在机器人右侧；将原生海绵完整放入该抽屉并连续保持 1 秒。 |
| `IThorKitchenWineBottleOntoStool` | 3 | 重新设计的 K02：机器人面向台面酒瓶，凳子初始位于机器人右侧，酒瓶旁的杯子在本任务中移除；将酒瓶放到凳面并稳定保持 1 秒。 |
| `IThorKitchenEggIntoJar` | 3 | 将原生鸡蛋放入旁边的原生开口容器 Pot_3_22；鸡蛋完整位于罐体原生 receptacle 区域内连续保持 1 秒即成功。 |
| `IThorKitchenApproachDishwasher` | 6、7、13、25 | 走到洗碗机旁并保持 1 秒。 |
| `IThorKitchenApproachFridge` | 6、7、13、24、25 | 走到冰箱旁并保持 1 秒。 |
| `IThorKitchenApproachOven` | 6、7、13、24、25 | 走到烤箱旁并保持 1 秒。 |
| `IThorKitchenOpenFridge` | 6、7、13、24、25 | 打开冰箱门。 |
| `IThorKitchenCloseFridge` | 6、7、13、24、25 | 将冰箱门关闭至不超过 10% 开度，并保持 1 秒。 |
| `IThorKitchenOpenCabinet` | 6、7、13、24、25 | 将橱柜门打开至至少 50% 开度，并保持 1 秒。 |
| `IThorKitchenCloseCabinet` | 6、7、13、24、25 | 将橱柜门关闭至不超过 10% 开度，并保持 1 秒。 |
| `IThorKitchenOpenDrawer` | 6、7、13、24、25 | 将抽屉拉开至至少 80% 开度，并保持 1 秒。 |
| `IThorKitchenCloseDrawer` | 6、7、13、24、25 | 将抽屉关闭至不超过 20% 开度，并保持 1 秒。 |
| `IThorKitchenOpenOven` | 6、7、13、24、25 | 打开烤箱门。 |
| `IThorKitchenCloseOven` | 6、7、13、24、25 | 将烤箱门关闭至不超过 10% 开度，并保持 1 秒。 |
| `IThorKitchenAppleOnPlate` | 6、7、13、24、25 | 将苹果放到盘子上。 |
| `IThorKitchenBreadOnPlate` | 6、7、13、24、25 | 将面包放到盘子上。 |
| `IThorKitchenCheeseOnPlate` | 6、7、13、24、25 | 将奶酪放到盘子上。 |
| `IThorKitchenMugOnPlate` | 6、7、13、24、25 | 将马克杯放到盘子上。 |
| `IThorKitchenAppleIntoBox` | 6、7、13、24、25 | 将一个苹果放入盒子。 |
| `IThorKitchenCupIntoBox` | 6、7、13、24、25 | 将一个杯子放入盒子。 |
| `IThorKitchenStatueIntoBox` | 6、7、13、24、25 | 将一个小雕像放入盒子。 |
| `IThorKitchenBowlToSink` | 6、7、13、24、25 | 将水槽旁的碗放入水槽。 |
| `IThorKitchenCupToSink` | 6、7、13、24、25 | 将水槽旁的杯子放入水槽。 |
| `IThorKitchenStandCannedDrinkUpright` | 6、7、13、24、25 | 将横放在桌面的罐装饮料底朝下直立放置。 |
| `IThorKitchenPanTableToStove` | 6、7、13、24、25 | 将灶台旁桌面上的平底锅拿到灶台炉圈上平放。 |
| `IThorKitchenLiftEggplant` | 6、7、13、24、25 | 从桌面抓起茄子并相对初始高度抬高 0.3 米，保持 1 秒。 |
| `IThorKitchenLiftPlate` | 6、7、13、24、25 | 从桌面抓起盘子并相对初始高度抬高 0.3 米，保持 1 秒。 |
| `IThorKitchenLiftSaltShaker` | 6、7、13、24、25 | 从桌面抓起盐瓶并相对初始高度抬高 0.3 米，保持 1 秒。 |

### 复合任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `IThorKitchenCannedDrinkFromFridgeToTrash` | 25 | **已重新设计：**冰箱门初始完全打开，已验证易抓取的 `can_0` 可乐直立放在冰箱内部靠外位置，垃圾桶前的凳子已在初始化时移开；取出可乐，再将可乐放入垃圾桶并放开，可乐中心在桶内连续保持 1 秒。 |
| `IThorKitchenOpenDrawerThenPutCannedDrinkInside` | 13 | 可乐初始直立放在柜台内部、抽屉初始关闭；先将抽屉打开到至少 80%，将可乐完整放入并释放，再关闭到 20% 以下。可乐留在抽屉内且抽屉保持关闭 1 秒即成功。 |
| `IThorKitchenCannedDrinkIntoFridgeThenCloseDoor` | 8 | **已重新设计：**下层抽拉门初始打开 100%，可乐直立位于机器人面前；将可乐中心放入放宽后的下层冰箱空间并释放，允许可乐倾斜或跨越内部小分区，再将下层门关闭至不超过 10%。可乐保留在抽屉内且门保持关闭 1 秒即成功，不判定机器人站位。 |
| `IThorKitchenPanToRightBurnerThenEggIntoPan` | 8 | 鸡蛋初始位于右下角灶位右下侧的柜台；先将锅从左下角灶位移到右下角灶位并达到至少 70% 重合，再将鸡蛋放入锅内。最终锅保留在目标灶位且鸡蛋中心位于锅内，连续保持 1 秒。 |
| `IThorKitchenCupFromStoolToSink` | 3 | 杯子初始放在机器人面前的凳面；将杯子拿起，转身走到水槽边并放入水池。杯子中心在水池内连续保持 1 秒。 |
| `IThorKitchenBreadOffPlateThenAppleOnPlate` | 3 | **已重新设计：**盘子与盘中面包初始位于水槽右侧柜台，苹果位于机器人左手前方并避开正前方通道；先将面包放到右侧整块柜台，再将苹果放进盘子并释放。面包保留在柜台、苹果与盘子接触并连续保持 1 秒即成功。 |
| `IThorKitchenPackProduceIntoBox` | 6、7、13、24、25 | 将苹果和香蕉都放入盒子中。 |
| `IThorKitchenBananaBasketToTable` | 6、7、13、24、25 | 将装有香蕉的带把手篮子搬到目标桌面上。 |
| `IThorKitchenCoffeeDrinkToTrash` | 7、13、24、25 | 将咖啡饮料杯放入垃圾桶。 |
| `IThorKitchenCupStack` | 6、7、13、24、25 | 将两个杯子叠在一起。 |
| `IThorKitchenBendAndPutAppleOnPlate` | 6、7、13、24、25 | G1 站在空旷处，弯腰将面前 0.45 米高矮桌上的苹果放到同桌的盘子上；仅检查苹果与盘子接触且水平靠近盘子中心，连续保持 1 秒；不检查弯腰、顺序或释放。 |
| `IThorKitchenOpenCloseLowMicrowave` | 6、7、13、24、25 | G1 在空旷处下蹲，打开固定在落地木质侧板支撑的 0.25 米矮桌上的带把手微波炉，再关门；初始门关闭，打开至至少 40% 并保持 5 步，再关闭至不超过 10%，连续保持 1 秒；不检查下蹲角度或抓持。 |
| `IThorKitchenPutAppleInLowMicrowave` | 6、7、13、24、25 | G1 在空旷处下蹲，将微波炉顶上的苹果放进固定在落地木质侧板支撑的 0.25 米矮桌上的带把手微波炉并关门；初始门关闭，打开至至少 40% 并保持 5 步，单独提示并检查放入苹果阶段（保持门开度至少 40%、苹果中心在原生 receptacle 内 5 步），再关闭至不超过 10%，且苹果中心仍在原生 receptacle 内，连续保持 1 秒；不检查下蹲角度或抓持。 |
| `IThorKitchenOpenLowMicrowaveAndLiftApple` | 6、7、13、24、25 | G1 下蹲打开矮桌上的微波炉，从内部原始高度附近抓住苹果，再保持门打开并将苹果抬高至少 0.30 米，连续保持 1 秒。 |
| `IThorKitchenTakeAppleFromOpenLowMicrowaveToTop` | 6、7、13、24、25 | 微波炉门初始打开 80%；G1 下蹲抓起内部苹果，再将苹果放到微波炉顶部、释放并保持门打开，连续保持 1 秒。 |
| `IThorKitchenCloseThenOpenLowMicrowave` | 6、7、13、24、25 | 微波炉门初始打开 80%；G1 下蹲先将门关闭至不超过 10% 并保持 5 步，再重新打开至至少 40%，连续保持 1 秒。 |
| `IThorKitchenOpenLowMicrowaveAndPutAppleInside` | 6、7、13、24、25 | G1 下蹲先打开矮桌上的微波炉，再从顶部原始高度附近抓起苹果，将苹果完整放入微波炉内部、与微波炉接触并释放；最终保持门打开，连续保持 1 秒。 |
| `IThorKitchenOpenThenCloseCabinet` | 6、7、13、24、25 | 将同一橱柜门从初始 20% 开度打开至至少 80%，再关闭至不超过 20%，连续保持 1 秒。 |
| `IThorKitchenCloseThenOpenCabinet` | 6、7、13、24、25 | 将同一橱柜门从初始 80% 开度关闭至不超过 20%，再打开至至少 80%，连续保持 1 秒。 |
| `IThorKitchenApproachOpenFridge` | 6、7、13、24、25 | 先走到冰箱旁，再打开冰箱门。 |
| `IThorKitchenApproachOpenOven` | 6、7、13、24、25 | 先走到烤箱旁，再打开烤箱门。 |
| `IThorKitchenApproachTableAndStandCannedDrinkUpright` | 6、7、13、24、25 | 完成走到桌边和将罐装饮料直立放置；两者完成顺序不限。 |
| `IThorKitchenApproachStoveAndPutPan` | 6、7、13、24、25 | 完成走到灶台和将平底锅放到灶台上；两者完成顺序不限。 |
| `IThorKitchenApproachSinkAndPutBowl` | 6、7、13、24、25 | 完成走到水槽和将碗放入水槽；两者完成顺序不限。 |
| `IThorKitchenApproachSinkAndPutCup` | 6、7、13、24、25 | 完成走到水槽和将杯子放入水槽；两者完成顺序不限。 |
| `IThorKitchenApproachDrawerAndOpen` | 6、7、13、24、25 | 完成走到抽屉和打开抽屉；两者完成顺序不限。 |
| `IThorKitchenApproachDrawerAndClose` | 6、7、13、24、25 | 完成走到抽屉和关闭抽屉；两者完成顺序不限。 |
| `IThorKitchenApproachCabinetAndOpen` | 6、7、13、24、25 | 完成走到橱柜和打开柜门；两者完成顺序不限。 |
| `IThorKitchenApproachCabinetAndClose` | 6、7、13、24、25 | 完成走到橱柜和关闭柜门；两者完成顺序不限。 |
| `IThorKitchenApproachDrawerAndOpenClose` | 6、7、13、24、25 | 依次走到抽屉操作位姿、将同一抽屉打开至至少 80% 开度、关闭至不超过 20% 开度；打开和关闭各保持 1 秒。 |
| `IThorKitchenApproachCabinetAndOpenClose` | 6、7、13、24、25 | 依次走到橱柜操作位姿、将同一柜门打开至至少 50% 开度、关闭至不超过 10% 开度；打开和关闭各保持 1 秒。 |
| `IThorKitchenPickAppleAndTurnToTrash` | 7、13、24、25 | 将桌上的苹果放入身后垃圾桶的指定 receptacle 并释放，入桶状态连续保持 1 秒即成功。 |
| `IThorKitchenOpenDrawerAndTurnAppleToTrash` | 7、13、24、25 | 两阶段：先将指定抽屉打开至至少 80% 并保持 5 步；再将苹果放入指定垃圾桶 receptacle 并释放，连续保持 1 秒。 |
| `IThorKitchenDrawerPnP` | 6、7、13、24、25 | 打开抽屉，将苹果放入抽屉，再将抽屉关闭至不超过 20% 开度并保持 1 秒。 |
| `IThorKitchenAppleCabinetPnP` | 6、7、13、24、25 | 将桌上的苹果放入已打开的橱柜，再关闭柜门并保持 1 秒。 |
| `IThorKitchenTakeStatueFromCabinet` | 6、7、13、24、25 | 从已打开的橱柜中取出雕像并放入桌面目标区域，释放雕像，再关闭同一柜门并保持 1 秒。 |
| `IThorKitchenAppleBananaPlateSwap` | 6、7、13、24、25 | 将苹果从盘子移到桌面，同时将香蕉放到盘子上；苹果须离开盘子，两者最终状态同时保持 1 秒，操作顺序不限。 |
| `IThorKitchenOpenDrawerTakeApple` | 6、7、13、24、25 | 打开抽屉，将苹果取出并放到桌上，双手离开苹果并保持 1 秒。 |
| `IThorKitchenOvenThenFridge` | 6、7、13、24、25 | 先走到烤箱，再走到冰箱。 |
| `IThorKitchenFridgeThenOven` | 6、7、13、24、25 | 先走到冰箱，再走到烤箱。 |
| `IThorKitchenFridgeThenDishwasher` | 6、7、13、25 | 先走到冰箱，再走到洗碗机。 |
| `IThorKitchenDishwasherThenFridge` | 6、7、13、25 | 先走到洗碗机，再走到冰箱。 |
| `IThorKitchenOvenThenDishwasher` | 6、7、13、25 | 先走到烤箱，再走到洗碗机。 |
| `IThorKitchenDishwasherThenOven` | 6、7、13、25 | 先走到洗碗机，再走到烤箱。 |
| `IThorKitchenApproachTableAndLiftApple` | 6、7、13、24、25 | 先走到桌旁，再将苹果抬高 0.3 米，保持 1 秒。 |
| `IThorKitchenApproachTableAndLiftSaltShaker` | 6、7、13、24、25 | 先走到桌旁，再将盐瓶抬高 0.3 米，保持 1 秒。 |
| `IThorKitchenApproachTableAndLiftBowl` | 6、7、13、24、25 | 先走到桌旁，再将碗抬高 0.3 米，保持 1 秒。 |
| `IThorKitchenApproachTableAndPutAppleOnPlate` | 6、7、13、24、25 | 先走到桌旁，再将苹果放到盘子上。 |
| `IThorKitchenApproachTableAndPutBreadOnPlate` | 6、7、13、24、25 | 先走到桌旁，再将面包放到盘子上。 |
| `IThorKitchenApproachTableAndPutCheeseOnPlate` | 6、7、13、24、25 | 先走到桌旁，再将奶酪放到盘子上。 |
| `IThorKitchenApproachTableAndPutMugOnPlate` | 6、7、13、24、25 | 先走到桌旁，再将马克杯放到盘子上。 |
| `IThorKitchenStoreAppleInFridge` | 13、24、25 | 走到冰箱旁，依次打开冰箱门、将桌上的苹果放入冰箱、关闭冰箱门；拾取本身不作为独立阶段。 |
| `IThorKitchenOpenThenCloseDrawer` | 13、24、25 | 将同一抽屉从 20% 开度打开至至少 80%，再关闭至不超过 20%，最终状态保持 1 秒。 |
| `IThorKitchenBowlToSinkThenAppleOnPlate` | 13、24 | 先将碗放入水槽，再走到桌旁将苹果放到盘子上。 |
| `IThorKitchenApproachCabinetAndStoreStatue` | 7、13 | 走到已打开的橱柜旁，将桌上的雕像放入柜内，再关闭柜门。 |

## iTHOR：客厅

共 61 个任务：28 个原子任务、33 个复合任务。

最终条件需连续保持 1 秒。导航到桌后操作的七个复合任务要求先完成导航：抬升只检查高度，放篮子检查接触、水平包含和释放，不增加独立的抓持、底部高度、篮子姿态或位移条件。Scene 210 未配置到桌导航，因此不支持这七个任务。

### 原子任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `IThorLivingRoomBoxFromChairToFloor` | 201 | 重新设计的 L01：开口纸箱初始位于椅面，四周箱沿便于抓握；将纸箱拿到地面，纸箱接触地面、离手并放稳，连续保持 1 秒。 |
| `IThorLivingRoomLiftBowlFromCoffeeTable` | 201 | 站在茶几旁，将原生碗相对初始高度抬升至少 0.3 米并保持 1 秒。 |
| `IThorLivingRoomBoxFromSofaToFloor` | 205 | 重新设计的 L03：原生纸箱初始位于沙发坐垫；将纸箱放到地面、离手并保持稳定 1 秒。 |
| `IThorLivingRoomStatueChairToTable` | 205 | 将椅子上的 `Stone_Statue_4` 完整放到餐桌面上，连续保持 1 秒。 |
| `IThorLivingRoomStatueIntoBox` | 209 | 将咖啡桌上的 `Stone_Statue_4` 放入同桌完全打开的原生纸盒内，连续保持 1 秒。 |
| `IThorLivingRoomRemoteIntoBox` | 226 | 原生遥控器初始长边朝上竖立；将其放入同桌完全打开的原生纸箱内，连续保持 1 秒。 |
| `IThorLivingRoomApproachSofa` | 202、207、209、210、212、226 | 走到沙发旁。 |
| `IThorLivingRoomSitOnSofa` | 202、207、209、210、212、226 | 转身坐到沙发上，使下躯干与指定座面区域接触并保持 1 秒。 |
| `IThorLivingRoomApproachTable` | 202、207、209、212、226 | 走到桌旁。 |
| `IThorLivingRoomOpenDrawer` | 209、210、226 | 打开客厅抽屉。 |
| `IThorLivingRoomCloseDrawer` | 209、210、226 | 关闭指定的客厅抽屉。 |
| `IThorLivingRoomCloseLaptop` | 202、207、209、210、212、226 | 将笔记本电脑合上至不超过 20% 开度，并保持 1 秒。 |
| `IThorLivingRoomOpenLaptop` | 202、207、209、210、212、226 | 将笔记本电脑打开至至少 90% 开度，并保持 1 秒。 |
| `IThorLivingRoomLiftStatue` | 202、207、209、210、212、226 | 抓起竖直放置的雕像并相对初始稳定高度抬高 0.3 米，保持 1 秒。 |
| `IThorLivingRoomLiftWateringCan` | 202、207、209、210、212、226 | 抓起浇水壶并相对初始稳定高度抬高 0.3 米，保持 1 秒。 |
| `IThorLivingRoomLiftDumbbell` | 202、207、209、210、212、226 | 抓起哑铃并相对初始稳定高度抬高 0.3 米，保持 1 秒。 |
| `IThorLivingRoomLiftCupAmongObjects` | 202、207、209、210、212、226 | 桌上摆放杯子、苹果、罐头、鸡蛋和蜡烛，只需将杯子抬高 0.3 米并保持 1 秒。 |
| `IThorLivingRoomPourBlockFromBasket` | 202、207、209、210、212、226 | 拿起并倾斜篮子，将与 PushBlockToTargetRegion 相同的方块完整倒出篮外，并保持 1 秒。 |
| `IThorLivingRoomPourIceFromCup` | 202、207、209、210、212、226 | 拿起支撑台上的杯子并倾斜，将冰块完整倒出杯外，保持 1 秒。 |
| `IThorLivingRoomPushBasketball` | 202、207、209、210、212、226 | 将桌上的篮球推动至少 0.3 米。 |
| `IThorLivingRoomRemoteToBasket` | 202、207、209、210、212、226 | 将遥控器放入篮子。 |
| `IThorLivingRoomCandleToBasket` | 202、207、209、210、212、226 | 将蜡烛放入篮子中。 |
| `IThorLivingRoomTissueBoxToBasket` | 202、207、209、210、212、226 | 将纸巾盒放入篮子中。 |
| `IThorLivingRoomBookIntoBasket` | 202、207、209、210、212、226 | 将垫子上的书放入同一垫子上的篮子。 |
| `IThorLivingRoomStackBookOnBook` | 202、207、209、210、212、226 | 将垫子上的一本书叠到另一本书上。 |
| `IThorLivingRoomPushBlockToTargetRegion` | 202、207、209、210、212、226 | 不抬起小方块，将其沿垫子完整推入无碰撞的可视目标区域。 |
| `IThorLivingRoomOpenBoxFlap` | 202、207、209、210、212、226 | 固定纸箱基座，将指定盖片从 20% 开度打开到至少 80%，保持 1 秒。 |
| `IThorLivingRoomCloseBoxFlap` | 202、207、209、210、212、226 | 固定纸箱基座，将指定盖片从 80% 开度关闭至不超过 20%，保持 1 秒。 |

### 复合任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `IThorLivingRoomMugFromCoffeeTableToBoxOnSofa` | 201 | **已重新设计：**机器人初始站在咖啡桌旁，抓取跨度约 7.5 cm 的马克杯位于原生碗的位置，敞开的纸盒位于沙发座面靠前位置；拿起马克杯走到沙发旁，将马克杯完整放入纸盒并释放，连续保持 1 秒。 |
| `IThorLivingRoomMugFromSofaToCoffeeTable` | 201 | **已重新设计：**与 LC01 相同、抓取跨度约 7.5 cm 的马克杯初始位于沙发座面靠前且避开座垫接缝的位置，机器人站在沙发旁；拿起马克杯并走到咖啡桌旁，将马克杯释放后稳定放在桌面并保持 1 秒。 |
| `IThorLivingRoomCloseTopRightThenTopLeftDrawer` | 205 | **已重新设计：**电视柜右上、左上两个抽屉初始完全打开，下层两个抽屉关闭；机器人站在小圆桌前并面向右上抽屉。先关闭右上抽屉，再走到左边关闭左上抽屉；两个抽屉均关闭到 10% 以下并保持 1 秒。 |
| `IThorLivingRoomRemoteFromSideTableIntoBoxOnSofa` | 205 | **已修改：**原生遥控器初始竖直放在电视旁的小圆桌中央，敞开的原生纸箱位于长沙发上；拿起遥控器走到沙发旁并放入纸箱。遥控器中心和下端进入放宽后的纸箱范围即可，允许顶部露出箱口，连续保持 1 秒。 |
| `IThorLivingRoomStatueFromTableToTrashBin` | 209 | 缩小后的 `Stone_Statue_4` 初始位于桌面靠垃圾桶一侧，机器人从桌子靠垃圾桶的一侧面对雕像；拿起雕像走到垃圾桶旁并投入。雕像释放后，中心进入略微放宽的桶口范围并保持 1 秒即成功，允许露在桶外或搭在桶口。 |
| `IThorLivingRoomCannedDrinkFromTableIntoBoxOnArmchair` | 226 | 新增可乐初始直立放在咖啡桌上，敞开的原生纸箱位于旁边的单人扶手椅座面；拿起可乐走到扶手椅旁并释放在纸箱内，连续保持 1 秒。 |
| `IThorLivingRoomCoffeeDrinkToBin` | 202、207、209、210、212、226 | 将咖啡饮料杯放入垃圾桶。 |
| `IThorLivingRoomRemoteToBin` | 202、207、209、210、212、226 | 将桌上的遥控器放入身旁的垃圾桶。 |
| `IThorLivingRoomBasketToSofa` | 202、207、209、210、212、226 | 将篮子放到沙发上。 |
| `IThorLivingRoomDumbbellToSofa` | 202、207、209、210、212、226 | 将哑铃放到沙发上。 |
| `IThorLivingRoomStatueToSofa` | 202、207、209、210、212、226 | 将雕像放到沙发上。 |
| `IThorLivingRoomStackBooksAndPlaceMug` | 202、207、209、210、212、226 | 先把初始右侧的书叠到左侧的书上，再把 mug 放到两本书上；最终状态连续保持 1 秒。 |
| `IThorLivingRoomPackCandleStatueIntoBasket` | 202、207、209、212、226 | 将蜡烛和雕像都放入同一个篮子中，操作顺序不限。 |
| `IThorLivingRoomWalkAndOpenLowDrawer` | 209、210、226 | 先导航到指定低位抽屉旁（距离不超过 0.35 米并保持 5 步），再蹲下拉开；209 为柜子左下层、210 为柜子右下层、226 为茶几正面从上到下第二格。初始开度 20%，达到至少 80% 并保持 1 秒；蹲姿不设额外判定门槛。 |
| `IThorLivingRoomWalkAndCloseLowDrawer` | 209、210、226 | 先导航到指定低位抽屉旁并保持 5 步，再蹲下关闭抽屉；初始开度 80%，降至不超过 20% 并保持 1 秒。目标与 OpenLowDrawer 相同，226 为第二格。 |
| `IThorLivingRoomLowDrawerPnP` | 209、210、226 | G1 站在家具前，从顶面拿起苹果；将指定矮抽屉从 20% 拉开到至少 80% 并保持 5 步，再把苹果放入该抽屉的 receptacle 并关闭至不超过 20%，两条件同时保持 1 秒。226 为茶几正面第二格。 |
| `IThorLivingRoomWalkAndOpenDrawer` | 209、210、226 | 先走到指定抽屉旁，再打开同一抽屉。 |
| `IThorLivingRoomWalkAndOpenCloseDrawer` | 209、210、226 | 依次走到指定抽屉旁（距离不超过 0.35 米并保持 5 步）、将同一抽屉打开至至少 80%、关闭至不超过 20%；初始开度 20%，打开和关闭各保持 1 秒。 |
| `IThorLivingRoomWalkAndCloseDrawer` | 209、210、226 | 先走到指定抽屉旁，再关闭同一抽屉。 |
| `IThorLivingRoomWalkToTableAndPutCanInBasket` | 202、207、209、212、226 | 先走到指定桌旁，再将垫子上的罐子放入篮子。 |
| `IThorLivingRoomApproachTableAndLiftDumbbell` | 202、207、209、212、226 | 先走到桌旁，再将哑铃相对初始高度抬高至少 0.3 米并保持 1 秒。 |
| `IThorLivingRoomApproachTableAndLiftStatue` | 202、207、209、212、226 | 先走到桌旁，再将雕像相对初始高度抬高至少 0.3 米并保持 1 秒。 |
| `IThorLivingRoomApproachTableAndLiftWateringCan` | 202、207、209、212、226 | 先走到桌旁，再将浇水壶相对初始高度抬高至少 0.3 米并保持 1 秒。 |
| `IThorLivingRoomApproachTableAndPutRemoteInBasket` | 202、207、209、212、226 | 先走到桌旁，再将遥控器放入桌上的篮子并释放，保持 1 秒。 |
| `IThorLivingRoomApproachTableAndPutCandleInBasket` | 202、207、209、212、226 | 先走到桌旁，再将蜡烛放入桌上的篮子并释放，保持 1 秒。 |
| `IThorLivingRoomApproachTableAndPutTissueBoxInBasket` | 202、207、209、212、226 | 先走到桌旁，再将纸巾盒放入桌上的篮子并释放，保持 1 秒。 |
| `IThorLivingRoomApproachTableAndPutBookInBasket` | 202、207、209、212、226 | 先走到桌旁，再将书本放入桌上的篮子并释放，保持 1 秒。 |
| `IThorLivingRoomSofaThenTable` | 202、207、209、212、226 | 先走到指定沙发旁，再走到桌子旁。 |
| `IThorLivingRoomTableThenSofa` | 202、207、209、212、226 | 先走到指定桌子旁，再走到沙发旁。 |
| `IThorLivingRoomBinThenTable` | 202、207、209、212、226 | 先走到指定垃圾桶旁，再走到桌子旁。 |
| `IThorLivingRoomBinThenSofa` | 202、207、209、212、226 | 先走到指定垃圾桶旁，再走到沙发旁。 |
| `IThorLivingRoomTableThenBin` | 202、207、209、212、226 | 先走到指定桌子旁，再走到垃圾桶旁。 |
| `IThorLivingRoomSofaThenBin` | 202、207、209、212、226 | 先走到指定沙发旁，再走到垃圾桶旁。 |

## iTHOR：卧室

共 62 个任务：26 个原子任务、36 个复合任务。

最终条件需连续保持 1 秒；导航后操作要求导航完成前目标保持初态。主书桌放篮子任务集中在 Scene 329。抽屉与保险柜门的打开/关闭阈值为 80%/20%，其他机构使用各自场景配置。

新增六项三步日常整理任务使用独立判定：中间阶段连续 5 步，最终有效状态连续 1 秒；提前操作须恢复后在对应阶段重新完成。已完成操作的历史保留，最终重查物品入篮及关节关闭状态，允许中途掉出或重新打开后补救。新增入篮任务允许稳定叠放，释放要求物品受有效支撑且双手不再接触或抓持，不要求双手离开 18 cm。进入“接近床”阶段后须移动至少 0.30 m。该判定不影响旧任务。

### 原子任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `IThorBedroomApproachBed` | 302、306、310、323、329 | 走到床旁并保持 1 秒。 |
| `IThorBedroomApproachBedsideWindow` | 323 | 走到床旁边的窗户旁并保持 1 秒。 |
| `IThorBedroomApproachDesk` | 302、306、310、323、329 | 走到书桌旁并保持 1 秒。 |
| `IThorBedroomApproachTrashCan` | 315 | 走到原生垃圾桶附近，机器人碰撞几何到桶的距离不超过 0.35 米并连续保持 1 秒。 |
| `IThorBedroomCanIntoTrashCan` | 315 | 将书桌近边的罐子放入场景原位的原生垃圾桶，罐子接触垃圾桶底部并连续保持 1 秒，允许直立、横放或倾斜；使用 CanIntoBasket 的同款模型与缩放，不检查抓握或释放状态。 |
| `IThorBedroomNavigateToStorage` | 302、306、310、323、329 | 走到指定收纳家具旁并保持 1 秒：302 为保险柜，306 为洗衣篮，310 为双抽屉床头柜，323 为五斗柜，329 为地面纸箱。 |
| `IThorBedroomOpenDrawer` | 302、306、310、323、329 | 打开指定抽屉并保持 1 秒。 |
| `IThorBedroomCloseDrawer` | 302、306、310、323、329 | 关闭指定抽屉并保持 1 秒。 |
| `IThorBedroomOpenLaptop` | 302、306、310、323、329 | 打开笔记本电脑并保持 1 秒。 |
| `IThorBedroomCloseLaptop` | 302、306、310、323、329 | 合上笔记本电脑并保持 1 秒。 |
| `IThorBedroomCanIntoBasket` | 302、306、310、323、329 | 将主书桌上的罐子放入篮子并释放，保持 1 秒。 |
| `IThorBedroomOpenSafeDoor` | 302、323 | 打开保险柜门并保持 1 秒。 |
| `IThorBedroomCloseSafeDoor` | 302、323 | 关闭保险柜门并保持 1 秒。 |
| `IThorBedroomOpenHamperLid` | 302、306、310、323、329 | 打开洗衣篮盖并保持 1 秒。 |
| `IThorBedroomCloseHamperLid` | 302、306、310、323、329 | 关闭洗衣篮盖并保持 1 秒。 |
| `IThorBedroomOpenBoxFlap` | 302、306、310、323、329 | 打开纸箱翻盖并保持 1 秒。 |
| `IThorBedroomCloseBoxFlap` | 302、306、310、323、329 | 关闭纸箱翻盖并保持 1 秒。 |
| `IThorBedroomMugIntoBasket` | 302、306、310、323、329 | 将主书桌上的马克杯放入篮子并释放，保持 1 秒。 |
| `IThorBedroomBoxedFoodIntoBasket` | 302、306、310、323、329 | 将主书桌上的食品盒放入篮子并释放，保持 1 秒。 |
| `IThorBedroomOpenTopDrawer` | 302、306、323、329 | 打开顶部抽屉并保持 1 秒。 |
| `IThorBedroomCloseTopDrawer` | 302、306、323、329 | 关闭顶部抽屉并保持 1 秒。 |
| `IThorBedroomLiftBasket` | 302、306、310、323、329 | 抓住篮子并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBedroomLiftBasketFromFloor` | 302、306、310、323、329 | 从落地矮凳上抓起篮子，相对初始高度抬高至少 0.3 米并保持 1 秒。 |
| `IThorBedroomLiftBaseballBat` | 329 | 拿起棒球棒，使整根棒球棒离地连续保持 1 秒，不要求抓握状态；离地判定使用 5 毫米容差。 |
| `IThorBedroomStandPillowAtHeadboard` | 301 | 将床上平放的原生枕头立起靠在床头，枕面与床头平行（倾角不超过 20°），同时接触床垫和床头，连续保持 1 秒；与抓握状态无关。 |
| `IThorBedroomLiftStatueFromSofa` | 301 | 弯腰拿起书桌、窗户旁单人沙发上的竖直石质人物雕像（Stone_Statue_4，0.5 倍），物体中心相对 reset 高度抬升至少 0.30 米并连续保持 1 秒；与抓握状态无关；301 默认将原蓝色软座替换为该沙发。 |

Scene 301 两个物体状态任务的布局、成功条件与验证记录见 [任务说明](ithor_bedroom_301_objects.md)。复合任务见 [301 复合任务说明](ithor_bedroom_301_composites.md)。

Scene 315 垃圾桶原子任务的布局、成功条件与验证记录见 [任务说明](ithor_bedroom_315_trash_can.md)；新增复合任务见 [315 复合任务说明](ithor_bedroom_315_composites.md)。

### 复合任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `IThorBedroomApproachBedStandPillowThenDesk` | 301 | 先走到床旁，将平放的原生枕头立起靠在床头，再走到书桌旁；枕头须保持同时接触床垫与床头，最终在书桌旁连续保持 1 秒。 |
| `IThorBedroomOpenBoxThenStatueFromSofaIntoBox` | 301 | 先打开纸箱剩余直立翻盖，再从沙发取起竖直石质人物雕像（Stone_Statue_4，0.5 倍）并抬高至少 0.30 米，最后完整放入纸箱、落到箱底并松手，保持 1 秒；使用仅该任务加载的大纸箱。 |
| `IThorBedroomApproachTrashCanAndPutCanInTrashThenLiftBasket` | 315 | 先走到原生垃圾桶旁，将罐子放入桶底，再将书桌上的篮子相对 reset 高度抬高至少 0.30 米；不检查手部抓持，罐子保持接触桶底，最终连续保持 1 秒。 |
| `IThorBedroomApproachDeskAndPutMugThenCanInBasket` | 315 | 先走到书桌旁，再依次将杯子、罐子放入同一篮子并释放；篮子留在书桌上，两件物品同时入篮并连续保持 1 秒。 |
| `IThorBedroomBedThenDesk` | 302、306、310、323、329 | 先走到床旁，再走到书桌旁并保持 1 秒。 |
| `IThorBedroomDeskThenBed` | 302、306、310、323、329 | 先走到书桌旁，再走到床旁并保持 1 秒。 |
| `IThorBedroomBedThenStorage` | 302、306、310、323、329 | 先走到床旁，再走到指定收纳家具旁并保持 1 秒。 |
| `IThorBedroomStorageThenBed` | 302、306、310、323、329 | 先走到指定收纳家具旁，再走到床旁并保持 1 秒。 |
| `IThorBedroomDeskThenStorage` | 302、306、310、323、329 | 先走到书桌旁，再走到指定收纳家具旁并保持 1 秒。 |
| `IThorBedroomStorageThenDesk` | 302、306、310、323、329 | 先走到指定收纳家具旁，再走到书桌旁并保持 1 秒。 |
| `IThorBedroomApproachBedsideWindowThenOpenHamperLid` | 323 | 先走到床边窗户旁，再走到洗衣篮旁，最后打开篮盖并保持 1 秒；两段导航各连续到位 5 步，导航期间篮盖须保持初始开度。参数与验证见[说明](ithor_bedroom_323_window_hamper.md)。 |
| `IThorBedroomApproachDeskAndOpenLaptop` | 302、306、310、323、329 | 先走到书桌旁，再打开笔记本电脑并保持 1 秒。 |
| `IThorBedroomApproachDeskAndCloseLaptop` | 302、306、310、323、329 | 先走到书桌旁，再合上笔记本电脑并保持 1 秒。 |
| `IThorBedroomApproachDeskAndLiftBasket` | 302、306、310、323、329 | 先走到书桌旁，再抓起篮子并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBedroomApproachDeskAndPutCanInBasket` | 329 | 先走到主书桌旁，再将罐子放入篮子并释放，保持 1 秒。 |
| `IThorBedroomApproachDeskAndPutMugInBasket` | 329 | 先走到主书桌旁，再将马克杯放入篮子并释放，保持 1 秒。 |
| `IThorBedroomApproachDeskAndPutBoxedFoodInBasket` | 329 | 先走到主书桌旁，再将食品盒放入篮子并释放，保持 1 秒。 |
| `IThorBedroomLiftBasketFromFloorThenPlaceOnDeskAndPutBoxedFoodInBasket` | 329 | 复用 IThorBedroomLiftBasketFromFloor 的篮子与低凳：机器人距篮子约 0.45 米并面向篮子，原地右转 90° 面向主书桌；先将篮子相对初始高度抬高至少 0.3 米并保持 5 步，再直立接触主书桌且食品盒仍在桌上、篮子外保持 5 步，最后将食品盒放入并接触篮子、篮子直立留在书桌上保持 1 秒；各阶段均不检查抓握或释放状态。 |
| `IThorBedroomApproachStorageAndOpenSafeDoor` | 302、323 | 先走到保险柜旁，再打开保险柜门并保持 1 秒。 |
| `IThorBedroomApproachStorageAndCloseSafeDoor` | 302、323 | 先走到保险柜旁，再关闭保险柜门并保持 1 秒。 |
| `IThorBedroomApproachStorageAndOpenHamperLid` | 306 | 先走到指定收纳家具旁，再打开洗衣篮盖并保持 1 秒。 |
| `IThorBedroomApproachStorageAndCloseHamperLid` | 306 | 先走到指定收纳家具旁，再关闭洗衣篮盖并保持 1 秒。 |
| `IThorBedroomApproachStorageAndOpenDrawer` | 310、323 | 先走到指定收纳家具旁，再打开指定抽屉并保持 1 秒。 |
| `IThorBedroomApproachStorageAndCloseDrawer` | 310、323 | 先走到指定收纳家具旁，再关闭指定抽屉并保持 1 秒。 |
| `IThorBedroomApproachStorageAndOpenBoxFlap` | 329 | 先走到指定收纳家具旁，再打开纸箱翻盖并保持 1 秒。 |
| `IThorBedroomApproachStorageAndCloseBoxFlap` | 329 | 先走到指定收纳家具旁，再关闭纸箱翻盖并保持 1 秒。 |
| `IThorBedroomOpenThenCloseDrawer` | 302、306、310、323、329 | 先打开指定抽屉，再关闭并保持 1 秒。 |
| `IThorBedroomOpenThenCloseTopDrawer` | 302、306、323、329 | 先打开顶部抽屉，再关闭并保持 1 秒。 |
| `IThorBedroomOpenThenCloseSafeDoor` | 302、323 | 先打开保险柜门，再关闭并保持 1 秒。 |
| `IThorBedroomOpenThenCloseHamperLid` | 302、306、310、323、329 | 先打开洗衣篮盖，再关闭并保持 1 秒。 |
| `IThorBedroomOpenThenCloseBoxFlap` | 310、329 | 先打开纸箱翻盖，再关闭并保持 1 秒。 |
| `IThorBedroomOpenThenCloseLaptop` | 302、306、310、323、329 | 先打开笔记本电脑，再合上并保持 1 秒。 |
| `IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket` | 329 | 接近主书桌，先将罐子、再将食品盒放入同一篮子；两件均留在篮内并释放。 |
| `IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop` | 329 | 接近主书桌，将杯子放入篮子，再合上电脑；杯子保持入篮，电脑仍由桌面支撑。 |
| `IThorBedroomCloseLaptopThenApproachBedAndCloseTopDrawer` | 302、306、310、329 | 从电脑操作位出发，合上电脑，走到床旁，再关闭指定顶部床头抽屉；电脑和抽屉均保持关闭。 |
| `IThorBedroomApproachDeskAndCloseLaptopThenApproachBed` | 329 | 接近书桌，合上电脑，再走到床旁；电脑保持关闭。 |
| `IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk` | 302、306、310、323、329 | 接近洗衣篮，关闭盖子，再走到书桌旁；关盖后须移动至少 1 米，并在盖子保持关闭时于书桌旁连续保持 1 秒。 |
| `IThorBedroomPutCanThenBoxedFoodThenMugInBasket` | 329 | 从主书桌操作位出发，依次将罐子、食品盒、杯子放入同一篮子；仅此任务的篮子各维尺寸放大 20%，允许稳定叠放，三件均须留在篮内并释放。 |

## iTHOR：浴室

共 73 个任务：27 个原子任务、46 个复合任务。

最终条件需连续保持 1 秒；导航后操作要求导航完成前目标保持初态。导航到目标时，机器人与目标家具碰撞几何的距离不超过 0.35 米；连续导航的第一目标保持 5 个控制步，并要求再移动至少 0.3 米到下一目标。

### 原子任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `IThorBathroomApproachToilet` | 401、405、407、415、418 | 走到马桶旁并保持 1 秒。 |
| `IThorBathroomApproachSink` | 401、405、407、415、418 | 走到洗手池旁并保持 1 秒。 |
| `IThorBathroomNavigateToBathingArea` | 401、405、407、415、418 | 走到浴缸或淋浴区旁并保持 1 秒。 |
| `IThorBathroomOpenToiletLid` | 401、405、407、415、418 | 将马桶盖打开至至少 80% 开度并保持 1 秒。 |
| `IThorBathroomCloseToiletLid` | 401、405、407、415、418 | 将马桶盖关闭至不超过 20% 开度并保持 1 秒。 |
| `IThorBathroomOpenShowerDoor` | 407、418 | 将淋浴门打开至 30° 并保持 1 秒。 |
| `IThorBathroomCloseShowerDoor` | 407、418 | 将淋浴门关闭至不超过 2° 并保持 1 秒。 |
| `IThorBathroomOpenCabinetDoor` | 418 | 将洗手池下方柜门打开至至少 50% 开度并保持 1 秒。 |
| `IThorBathroomCloseCabinetDoor` | 418 | 将洗手池下方柜门关闭至不超过 10% 开度并保持 1 秒。 |
| `IThorBathroomOpenUpperCabinetDoor` | 422 | 将上方壁柜门从初始 20% 开度打开至至少 80%，并保持 1 秒。 |
| `IThorBathroomCloseUpperCabinetDoor` | 422 | 将上方壁柜门从初始 80% 开度关闭至不超过 20%，并保持 1 秒。 |
| `IThorBathroomOpenLowerDrawer` | 415 | 将低位抽屉打开至至少 80% 开度并保持 1 秒。 |
| `IThorBathroomCloseLowerDrawer` | 415 | 将低位抽屉关闭至不超过 20% 开度并保持 1 秒。 |
| `IThorBathroomApproachGarbageCan` | 401 | 走到垃圾桶旁并保持 1 秒。 |
| `IThorBathroomSoapToLowerShelf` | 401 | 将肥皂放到最下层架面并释放，保持支撑接触 1 秒。 |
| `IThorBathroomLiftSoapBottle` | 401、405、407、415、418 | 抓住皂液瓶并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomLiftSoapBottleFromCabinet` | 405 | 柜门初始全开，从柜内约 45 cm 高的置物桌上拿起原生 SoapBottle；相对初始高度抬高至少 0.30 米并保持 1 秒，与抓握和手部接触状态无关。 |
| `IThorBathroomLiftToiletPaperRoll` | 401、405、407、415、418 | 抓住完整卫生纸卷并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomLiftSprayBottle` | 401、405、407、415、418 | 抓住喷雾瓶并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomStandSoapBottleUpright` | 401、405、407、415、418 | 将横放的皂液瓶直立放回浴缸边支撑面并释放，保持 1 秒。 |
| `IThorBathroomLiftScrubBrush` | 401、405、407、415、418 | 抓住清洁刷并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomLiftCandleFromFloor` | 401 | 从落地矮凳上抓起蜡烛，相对初始高度抬高至少 0.3 米并保持 1 秒。 |
| `IThorBathroomPutSpongeInBathtub` | 401、405、407、415 | 将海绵放入浴缸并释放，保持与缸体接触且无非法穿透 1 秒。 |
| `IThorBathroomToiletPaperRollIntoBin` | 401 | 将完整卫生纸卷放入保持原位且直立的垃圾桶并释放，保持 1 秒。 |
| `IThorBathroomToiletPaperRollIntoDrawer` | 410 | 抽屉初始全开，将水槽旁台面上的完整纸卷放入水槽下方、最靠近马桶的上层抽屉；纸卷与该抽屉底部支撑面连续接触 1 秒即成功，不限制姿态或抓持状态，不要求完整包围盒位于内腔。 |
| `IThorBathroomTakeSoapBottleFromDrawer` | 410 | 从同一已打开的上层抽屉中抬起横放的原生肥皂瓶；相对 reset 后初始高度抬升至少 0.30 米并连续保持 1 秒即成功，与抓握或手部接触状态无关。 |

### 复合任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `IThorBathroomNavigateToSinkThenToilet` | 401、405、407、415、418 | 先走到洗手池旁，再移动至少 0.3 米到马桶旁并保持 1 秒。 |
| `IThorBathroomNavigateToToiletThenSink` | 401、405、407、415、418 | 先走到马桶旁，再移动至少 0.3 米到洗手池旁并保持 1 秒。 |
| `IThorBathroomNavigateToSinkThenBathingArea` | 401、405、407、415、418 | 先走到洗手池旁，再移动至少 0.3 米到洗浴区旁并保持 1 秒。 |
| `IThorBathroomNavigateToBathingAreaThenSink` | 401、405、407、415、418 | 先走到洗浴区旁，再移动至少 0.3 米到洗手池旁并保持 1 秒。 |
| `IThorBathroomNavigateToToiletThenBathingArea` | 401、405、407、415、418 | 先走到马桶旁，再移动至少 0.3 米到洗浴区旁并保持 1 秒。 |
| `IThorBathroomNavigateToBathingAreaThenToilet` | 401、405、407、415、418 | 先走到洗浴区旁，再移动至少 0.3 米到马桶旁并保持 1 秒。 |
| `IThorBathroomNavigateToGarbageCanThenSink` | 401、405、407、415、418 | 先走到垃圾桶旁，再移动至少 0.3 米到洗手池旁并保持 1 秒。 |
| `IThorBathroomNavigateToSinkThenGarbageCan` | 401、405、407、415、418 | 先走到洗手池旁，再移动至少 0.3 米到垃圾桶旁并保持 1 秒。 |
| `IThorBathroomNavigateToGarbageCanThenToilet` | 401、405、407、415、418 | 先走到垃圾桶旁，再移动至少 0.3 米到马桶旁并保持 1 秒。 |
| `IThorBathroomNavigateToToiletThenGarbageCan` | 401、405、407、415、418 | 先走到马桶旁，再移动至少 0.3 米到垃圾桶旁并保持 1 秒。 |
| `IThorBathroomNavigateToGarbageCanThenBathingArea` | 401、405、407、415、418 | 先走到垃圾桶旁，再移动至少 0.3 米到洗浴区旁并保持 1 秒。 |
| `IThorBathroomNavigateToBathingAreaThenGarbageCan` | 401、405、407、415、418 | 先走到洗浴区旁，再移动至少 0.3 米到垃圾桶旁并保持 1 秒。 |
| `IThorBathroomApproachToiletAndOpenToiletLid` | 401、405、407、415、418 | 先走到马桶旁，再将马桶盖打开至至少 80% 开度并保持 1 秒。 |
| `IThorBathroomApproachToiletAndCloseToiletLid` | 401、405、407、415、418 | 先走到马桶旁，再将马桶盖关闭至不超过 20% 开度并保持 1 秒。 |
| `IThorBathroomNavigateToBathingAreaAndOpenShowerDoor` | 407、418 | 先走到洗浴区旁，再将淋浴门打开至 30° 并保持 1 秒。 |
| `IThorBathroomNavigateToBathingAreaAndCloseShowerDoor` | 407、418 | 先走到洗浴区旁，再将淋浴门关闭至不超过 2° 并保持 1 秒。 |
| `IThorBathroomApproachSinkAndOpenCabinetDoor` | 405、407、418 | 先走到洗手池旁，再将洗手池下方柜门打开至至少 50% 开度并保持 1 秒。 |
| `IThorBathroomApproachSinkAndCloseCabinetDoor` | 405、407、418 | 先走到洗手池旁，再将洗手池下方柜门关闭至不超过 10% 开度并保持 1 秒。 |
| `IThorBathroomNavigateToBathingAreaAndLiftSoapBottle` | 401 | 先走到洗浴区旁，再抓住皂液瓶并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomApproachSinkAndLiftSoapBottle` | 405、407、418 | 先走到洗手池旁，再抓住皂液瓶并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomApproachSinkAndLiftToiletPaperRoll` | 405、407、418 | 先走到洗手池旁，再抓住完整卫生纸卷并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomApproachSinkAndLiftSprayBottle` | 405、407、418 | 先走到洗手池旁，再抓住喷雾瓶并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomNavigateToBathingAreaAndLiftScrubBrush` | 405、407、415 | 先走到洗浴区旁，再抓住清洁刷并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomApproachSinkAndLiftScrubBrush` | 418 | 先走到洗手池旁，再抓住清洁刷并相对初始高度抬高至少 0.3 米，保持 1 秒。 |
| `IThorBathroomNavigateToBathingAreaAndStandSoapBottleUpright` | 401 | 先走到浴缸旁，再将横放的皂液瓶直立放回支撑面并释放，保持 1 秒。 |
| `IThorBathroomNavigateToBathingAreaAndPutSpongeInBathtub` | 401、405、407、415 | 先走到浴缸旁，再将海绵放入缸内并释放，保持 1 秒。 |
| `IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin` | 401、405、407、415、418 | 先走到垃圾桶旁，再将完整卫生纸卷放入桶中并释放，保持 1 秒。 |
| `IThorBathroomOpenThenCloseToiletLid` | 401、405、407、415、418 | 先将马桶盖打开至至少 80% 开度，再关闭至不超过 20% 开度并保持 1 秒。 |
| `IThorBathroomCloseThenOpenToiletLid` | 401、405、407、415、418 | 先将马桶盖关闭至不超过 20% 开度，再打开至至少 80% 开度并保持 1 秒。 |
| `IThorBathroomOpenThenCloseShowerDoor` | 407、418 | 先将淋浴门打开至 30°，再关闭至不超过 2° 并保持 1 秒。 |
| `IThorBathroomCloseThenOpenShowerDoor` | 407、418 | 先将淋浴门关闭至不超过 2°，再打开至 30° 并保持 1 秒。 |
| `IThorBathroomOpenThenCloseCabinetDoor` | 405、407、418 | 先将洗手池下方柜门打开至至少 50% 开度，再关闭至不超过 10% 开度并保持 1 秒。 |
| `IThorBathroomCloseThenOpenCabinetDoor` | 405、407、418 | 先将洗手池下方柜门关闭至不超过 10% 开度，再打开至至少 50% 开度并保持 1 秒。 |
| `IThorBathroomOpenThenCloseLowerDrawer` | 415 | 先将低位抽屉打开至至少 80% 开度，再关闭至不超过 20% 开度并保持 1 秒。 |
| `IThorBathroomCloseThenOpenLowerDrawer` | 415 | 先将低位抽屉关闭至不超过 20% 开度，再打开至至少 80% 开度并保持 1 秒。 |
| `IThorBathroomApproachSinkAndPutSpongeInSinkThenLiftSoapBottle` | 401 | 走到水池旁并保持 5 步 → 海绵接触槽底 1 秒 → 池沿肥皂瓶相对初始高度抬升至少 0.30 米；第三阶段不要求抓握，抬升高度与海绵槽底接触同时保持 1 秒。仅此任务将原生肥皂瓶放在池沿。 |
| `IThorBathroomNavigateToBathingAreaAndStandSoapBottleUprightThenPutSpongeInBathtub` | 401 | 走到浴缸旁 → 摆正并释放皂液瓶 → 将海绵放入浴缸并释放；瓶子保持直立受支撑、海绵保持缸内接触，持续 1 秒。 |
| `IThorBathroomSoapToLowerShelfThenStandSoapBottleUprightAndPutSpongeInBathtub` | 401 | 肥皂块放到下层架 → 摆正皂液瓶 → 海绵放入浴缸；三个整理结果同时有效并保持 1 秒，允许在操作区之间行走。 |
| `IThorBathroomCloseToiletLidThenApproachSinkAndLiftSoapBottle` | 401、405、407、415、418 | 合上马桶盖 → 走到洗手池 → 从池旁置物凳抓起皂液瓶；盖子 ≤20%，瓶子抓持抬升 ≥0.30 米，保持 1 秒。 |
| `IThorBathroomApproachToiletAndOpenToiletLidThenLiftScrubBrush` | 401、405、407、415、418 | 走到马桶旁 → 打开马桶盖 → 抓起旁边置物凳上的清洁刷；盖子 ≥80%，刷子抓持抬升 ≥0.30 米，保持 1 秒。 |
| `IThorBathroomOpenUpperDrawerThenLiftToiletPaperRollAndCloseUpperDrawer` | 415 | 打开从上往下第二层抽屉至 ≥80% → 将纸卷相对 reset 初始高度抬升 ≥0.30 米并保持 5 个控制步 → 关闭同一抽屉至 ≤20%；最终纸卷完整位于抽屉外且无非法穿透，持续 1 秒，不要求保持抓持。 |
| `IThorBathroomOpenDrawerThenPutToiletPaperRollAndCloseDrawer` | 410 | 打开水池下方抽屉至 ≥80% → 将纸卷放到抽屉底面 → 关闭同一抽屉至 ≤20%；最终保持底面接触 1 秒，机器人初始位与纸卷入抽屉原子任务一致。 |
| `IThorBathroomTakeSoapBottleFromDrawerThenStandOnSinkAndApproachToilet` | 410 | 从抽屉取出肥皂瓶并相对初始高度抬升 ≥0.30 米 → 竖直放在水池台面并释放 → 走到马桶碰撞几何 0.35 米内；瓶子保持竖直受支撑，最终持续 1 秒。 |
| `IThorBathroomOpenCabinetThenPlaceSoapBottleOnCounterAndCloseCabinet` | 405 | 柜门初始打开 20%；打开至 ≥80% → 弯腰取柜内肥皂瓶并相对初始高度抬高 ≥0.30 米（第二阶段不检查抓握状态或历史） → 放在水池台面并松手 → 关闭同一柜门至 ≤10%；中间阶段保持 5 步，最终台面支撑、松手及关门状态保持 1 秒。 |

| `IThorBathroomOpenThenCloseUpperCabinetDoorAndApproachSink` | 422 | 上方柜门初始开度 20%；打开至 ≥80% → 关闭同一门至 ≤20% → 走到水池碰撞几何 0.35 米内；关门后须移动至少 0.30 米，最终关门与到达状态保持 1 秒。 |
| `IThorBathroomOpenDrawerThenLiftCandleFromStoolAndPutInDrawer` | 422 | 打开水池旁另一组柜体的抽屉（`Drawer_452cc903_0`）至 ≥80%（初始 20%）→ 将柜旁低凳上的蜡烛相对初始高度抬高 ≥0.30 米并保持 5 步（不检查抓握或抓取历史）→ 完整放入同一抽屉（不检查抓握、松手或手部接触）；最终抽屉保持打开、蜡烛接触真实底面且无非法穿透，持续 1 秒。 |

原有五个三阶段任务使用独立规格；柜内取喷雾瓶任务已删除，纸卷取物任务上移一层，其他任务未改变。中间阶段连续满足 5 个控制步，提前完成的操作需恢复非成功状态后重新完成；普通与固定重置均清空进度。布局、判定及验证边界见 [三阶段任务说明](bathroom_three_stage_routines.md)。

401 的水池三阶段任务独立深拷贝原子配置，仅在本任务中把原生肥皂瓶放到池沿；既有任务保持不变。见 [401 水池复合任务说明](ithor_bathroom_401_sink_routine.md)。

410 的两个新增任务各自深拷贝原子配置，仅支持 410；中间阶段保持 5 个控制步，最终保持 1 秒。肥皂瓶任务单独选用同一台面的较宽支撑区域并校准高度，不修改原子任务或共享语义区域。详见 [410 抽屉复合任务说明](ithor_bathroom_410_composites.md)。

405 新增开柜、柜内取瓶、台面放置、关柜四阶段任务，使用原子布局的独立规格副本及单独校准的柜门扫掠站位。详见 [405 柜内肥皂瓶复合任务说明](ithor_bathroom_405_cabinet_composite.md)。

422 的两个新增任务仅支持 422，使用独立规格；蜡烛与低凳沿用已有蜡烛原子资产，位置仅在该复合任务内覆盖。详见 [422 复合任务说明](ithor_bathroom_422_composites.md)。

## ProcTHOR：多房间住宅

共 8 个任务：3 个原子任务、5 个复合任务。

动态任务在场景装载时检查语义条件，不承诺所有已下载 scene-id 都适用。跨房间搬运食品盒的任务仅为 train_0 配置了显式出生点。

### 原子任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `ProcTHOROpenDoor` | 动态筛选（未限定 ID） | 打开指定的房间门。 |
| `ProcTHOROpenDrawer` | 动态筛选（未限定 ID） | 打开指定的斗柜抽屉。 |
| `ProcTHORCrouchPickCylinder` | 动态筛选（未限定 ID） | 原地下蹲，使头部相对初始高度下降至少 0.5 米；不要求拾取物体。 |

### 复合任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `ProcTHOROpenDoorWalkThrough` | 动态筛选（未限定 ID） | 打开连接两个房间的门，再穿过门进入目标房间。 |
| `ProcTHORRoomToRoomCarry` | 0（train_0） | 将食品盒从源房间搬到相邻房间的指定支撑面上。 |
| `ProcTHORContainerObjectToTable` | 动态筛选（未限定 ID） | 将抽屉或柜子中的瓶装饮料取出，搬到另一房间的桌上。 |
| `ProcTHORTableToTable` | 动态筛选（未限定 ID） | 将空罐从源房间的桌上搬到目标房间的桌上。 |
| `ProcTHORTrashToBin` | 动态筛选（未限定 ID） | 将源房间桌上的空罐搬到目标房间并放入垃圾桶。 |

## Holodeck：室内场景

按下表统计共 78 个任务：31 个原子任务、47 个复合任务。

场景范围按当前 `SUPPORTED_SCENE_IDS` 列出。部分取物与投放任务在代码中归类为复合任务，统一保留该分类。

### 原子任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `HolodeckOpenLaptop` | 117、1018、1197、1664、2025 | 打开指定的笔记本电脑。 |
| `HolodeckCloseLaptop` | 117、1018、1197、1664、2025 | 合上指定的笔记本电脑。 |
| `HolodeckOpenDrawer` | 117、1018、1197、1664、2025 | 打开指定的斗柜抽屉。 |
| `HolodeckCloseDrawer` | 117、1018、1197、1664、2025 | 关闭指定的斗柜抽屉。 |
| `HolodeckOpenCabinetDoor` | 117 | 将办公桌旁侧柜靠近书桌的右侧柜门从 20% 打开至至少 80%，连续保持 1 秒。 |
| `HolodeckLiftPlantFromLowCabinet` | 1477 | 抓住最右侧窗户对面墙边矮柜上的盆栽，相对初始高度抬升至少 0.30 米并保持抓持 1 秒。 |
| `HolodeckCubeInPlate` | 1477 | 魔方与目标盘子接触且不与咖啡桌接触，连续保持 1 秒即成功，不要求抓握、松手或完整位于盘内。 |
| `HolodeckStandCandleUpright` | 2025 | 将远离书桌的边桌上外侧倒下的蜡烛扶正；目标蜡烛与该桌面接触且竖直偏差不超过 10°，连续保持 1 秒即成功。 |
| `HolodeckPickAlarmClockFromSofa` | 2209 | 将粉红色沙发上的闹钟相对初始高度抬升至少 0.30 米并保持 1 秒，不要求抓握。 |
| `HolodeckPlaceAlarmClockInDrawer` | 2209 | 将书桌上的闹钟放入同一书桌初始打开的最上层抽屉；仅要求闹钟连续接触抽屉支撑面 1 秒，不要求抓握记录、松手、完整包围或静止。 |
| `HolodeckOpenDrawerThenTakeTissueBox` | 117、1018、1197、1664、2025 | 先打开指定抽屉，再取出纸巾盒并相对初始高度抬高超过 0.3 米，保持 1 秒。 |
| `HolodeckIceCubeInGlass` | 117、1018、1197、1664、2025 | 将冰块放入玻璃杯目标区域并保持 1 秒。 |
| `HolodeckOpenMugLid` | 117、1018、1197、1664、2025 | 取下罐盖，使其与罐体脱离接触并保持 1 秒。 |
| `HolodeckCloseMugLid` | 117、1018、1197、1664、2025 | 将罐盖放回闭合位置并保持 1 秒。 |
| `HolodeckSugarCubeInJar` | 117、1018、1197、1664、2025 | 将方糖完整放入已开盖的罐子目标范围并保持 1 秒。 |
| `HolodeckPickBookFromDresser` | 117、1018、1197、1664、2025 | 抓住边桌或斗柜上的书本，相对初始高度抬高超过 0.3 米并保持抓持 1 秒。 |
| `HolodeckMoveCupToCenter` | 117、1018、1197、1664、2025 | 将玻璃杯移入书桌中央目标区域、与书桌接触并保持 1 秒。 |
| `HolodeckMoveCupToCorner` | 117、1018、1197、1664、2025 | 将玻璃杯移入机器人一侧的任一桌角目标区域、与书桌接触并保持 1 秒。 |
| `HolodeckPickCupFromDesk` | 117、1018、1197、1664、2025 | 抓住书桌上的玻璃杯，相对初始高度抬高超过 0.3 米并保持抓持 1 秒。 |
| `HolodeckPickBoxFromFloor` | 117、1018、1197、1664、2025 | 拿起地上的纸箱，相对初始高度抬高至少 0.3 米并保持 1 秒。 |
| `HolodeckPlaceBoxOnFloor` | 117、1018、1197、1664、2025 | 将桌上的纸箱放到地面，保持与地面接触 1 秒。 |
| `HolodeckPlaceBoxOnDesk` | 117 | 将纸箱与书桌接触并保持 1 秒。 |
| `HolodeckPlaceCupOnPlacemat` | 117、1018、1197、1664、2025 | 将玻璃杯完整放入书桌上的餐垫目标区域并保持 1 秒。 |
| `HolodeckApproachDesk` | 117、1018、1197、1664、2025 | 走到指定书桌旁。 |
| `HolodeckApproachTrashCan` | 117、1018、1197、1664、2025 | 走到指定垃圾桶旁。 |
| `HolodeckApproachWindow` | 117、1018、1197、1664、2025 | 走到窗户排最右侧的指定窗户旁。 |
| `HolodeckPushChairToDesk` | 117、1018、1197、1664、2025 | 将指定办公椅推到书桌前。 |
| `HolodeckInsertFolderIntoShelf` | 1018、1197、2025 | 将文件夹插入书架中层目标区域，保持与书架接触 1 秒。 |
| `HolodeckRemoveFolderFromShelf` | 1018、1197、2025 | 将文件夹从书架取出，保持与书架脱离接触 1 秒。 |
| `HolodeckPullChairFromDesk` | 117、1018、1197、1664、2025 | 将指定办公椅从书桌下拉出，使椅子中心离开 reset 原位置至少 0.50 米，并保持 1 秒。 |
| `HolodeckThrowBottleIntoTrashCan` | 117、1018、1197、1664、2025 | 拿起垃圾桶旁边桌上的空饮料瓶，将其放入垃圾桶，保持与桶底接触 1 秒。 |


### 复合任务

| 任务名 | 适用 scene-id | 任务描述 |
|---|---|---|
| `HolodeckTakeTissueBoxFromOpenDrawer` | 117、1018、1197、1664、2025 | 从已打开的抽屉中取出纸巾盒，相对初始高度抬高超过 0.3 米并保持 1 秒。 |
| `HolodeckApproachTrashCanThenLiftPlantThenPlacePlantInTrashCan` | 1477 | 依次走到盆栽旁垃圾桶碰撞体 0.35 米内并保持 5 步、将边桌上的盆栽抬高至少 0.30 米并保持 1 秒、将盆体完整放入桶内并接触桶底保持 1 秒。提前举起需要在到达后放低并重新举起；各阶段不检查手部状态。垃圾桶和远端出生位姿仅作用于本任务，盆栽原子任务不变。 |
| `HolodeckPlaceAlarmClockThenCloseDrawerThenPickBox` | 2209 | 依次将闹钟放入初始打开的书桌最上层抽屉、将抽屉关闭至不超过 20% 开度、将地面纸箱相对 reset 高度抬升超过 0.30 米；各阶段保持 1 秒，后续阶段要求闹钟持续接触抽屉底面、抽屉保持关闭。纸箱只在此复合任务中添加。 |
| `HolodeckOpenDrawerThenPlaceAlarmClockThenApproachTrashCan` | 2209 | 依次将书桌最上层抽屉从 20% 打开至至少 80%、将闹钟放入抽屉、走到垃圾箱碰撞体 0.35 米内；各阶段保持 1 秒，导航阶段要求闹钟仍接触抽屉底面，提前放入闹钟或提前到达不能跳过阶段。垃圾箱复用 117 模型，只在此复合任务中添加。 |
| `HolodeckApproachWindowThenCubeInPlateThenOpenLaptop` | 1477 | 依次走到最右侧窗户附近、将魔方放入盘子并保持 1 秒、打开同桌上的笔记本电脑并保持 1 秒；魔方与盘子复用原子布局，电脑复用 117 的原生模型。 |
| `HolodeckApproachDeskThenOpenLaptop` | 117、1018、1197、1664、2025 | 先走到书桌旁，再打开桌上的指定笔记本电脑。 |
| `HolodeckApproachDeskThenPickCup` | 117、1018、1197、1664、2025 | 先走到书桌旁，再拿起桌上的玻璃杯。 |
| `HolodeckApproachDeskThenMoveCupToCenter` | 117、1018、1197、1664、2025 | 先走到书桌旁，再将玻璃杯移入书桌中央目标区域、与书桌接触并保持 1 秒。 |
| `HolodeckApproachDeskThenCloseDrawer` | 117、1197 | 先走到书桌旁，再关闭指定的斗柜抽屉。 |
| `HolodeckApproachDeskThenOpenJar` | 117、1018、1197、1664、2025 | 先走到书桌旁，再取下罐盖，使其与罐体脱离接触并保持 1 秒。 |
| `HolodeckSugarCubeThenCloseJar` | 117、1018、1197、1664、2025 | 先将方糖完整放入罐子目标范围并保持 1 秒，再盖好同一罐盖并保持 1 秒。 |
| `HolodeckOpenJarAddSugarThenCloseJar` | 117、1018、1197、1664、2025 | 依次取下罐盖、将方糖完整放入罐子目标范围、盖好同一罐盖；每个目标状态保持 1 秒。 |
| `HolodeckOpenDrawerTakeTissueBoxToDesk` | 117、1018、1197、1664、2025 | 依次打开指定抽屉、取出纸巾盒、将纸巾盒完整放到书桌上。 |
| `HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer` | 117、1018、1197、1664、2025 | 依次打开指定抽屉、将纸巾盒取出并放到书桌上、关闭同一抽屉。 |
| `HolodeckApproachDeskThenCloseLaptopMoveCup` | 117、1018、1197、1664、2025 | 依次走到书桌旁、合上笔记本电脑、将玻璃杯移到桌角并与书桌接触。 |
| `HolodeckRemoveFolderFromShelfToDesk` | 1018、1197、2025 | 先从书架取出文件夹，再将文件夹放到书桌上并保持接触 1 秒。 |
| `HolodeckIceCubeThenPlaceCup` | 117、1018、1197、1664、2025 | 先将冰块放入杯子，再将同一杯子放到书桌上的餐垫上。 |
| `HolodeckApproachDeskThenPullChair` | 117、1018、1197、1664、2025 | 先走到书桌旁，再将桌下的办公椅拉出。 |
| `HolodeckApproachWindowThenDeskThenPickCup` | 117、1018、1664、2025 | 依次走到指定窗户旁、走到书桌旁、拿起桌上的玻璃杯。 |
| `HolodeckPickBookFromSourceToDesk` | 117、1018 | 先将书本相对初始高度抬高超过 0.3 米并保持 1 秒，再放到书桌上并保持接触 1 秒。 |
| `HolodeckApproachDeskPullChairThenOpenLaptop` | 117、1018 | 依次走到书桌旁、拉出办公椅、打开笔记本电脑。 |
| `HolodeckMoveCupToCenterThenOpenLaptop` | 117、1018 | 先将玻璃杯移到书桌中央、与书桌接触，再打开笔记本电脑。 |
| `HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop` | 1018 | 依次从书架取出文件夹、将其放到书桌上、打开笔记本电脑。 |
| `HolodeckApproachDeskIceCubePlaceCupThenPickCup` | 117、1018、1197、1664、2025 | 依次走到书桌旁、将冰块放入杯子、将杯子放到餐垫上、拿起同一杯子。 |
| `HolodeckCloseLaptopCloseDrawerThenApproachWindow` | 117、1018 | 依次合上笔记本电脑、关闭指定抽屉、走到最右侧窗户附近。 |
| `HolodeckTakeTissueBoxThenPlaceCup` | 117、1018 | 先将已打开抽屉中的纸巾盒取出并放到书桌上，再将杯子放到餐垫上。 |
| `HolodeckApproachTrashCanThenThrowBottle` | 117、1018 | 先走到垃圾桶旁，再将边桌上的空饮料瓶放入桶内。 |
| `HolodeckApproachTrashCanThenPickBoxFromFloor` | 117、1018、1197、1664、2025 | 先走到垃圾桶旁，再拿起地上的纸箱。 |
| `HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk` | 117 | 依次走到垃圾桶旁、走到书桌旁、将地上的纸箱放到书桌上。 |
| `HolodeckCloseLaptopThenApproachTrashCanThenPickBox` | 117、1018、1197、1664、2025 | 依次合上笔记本电脑、走到垃圾桶旁、将地上的纸箱抬高超过 0.3 米。 |
| `HolodeckApproachWindowThenPickBox` | 117、1018、1197、1664、2025 | 先走到指定窗户旁，再将地上的纸箱抬高超过 0.3 米。 |
| `HolodeckPullChairThenApproachTrashCanThenPickBox` | 117 | 依次拉出桌下的办公椅、走到垃圾桶旁、举起地上的纸箱。 |
| `HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop` | 1018 | 依次从书架取出文件夹、将其放到书桌上、合上笔记本电脑。 |
| `HolodeckApproachWindowThenDesk` | 117、1018、1664、2025 | 先走到指定窗户旁，再走到书桌旁。 |
| `HolodeckApproachTrashCanThenThrowBottleThenApproachDesk` | 117、1018 | 依次走到垃圾桶旁、将空饮料瓶放入桶内、返回书桌旁。 |
| `HolodeckApproachDeskThenOpenLaptopThenApproachWindow` | 117、1018 | 依次走到书桌旁、打开笔记本电脑、走到指定窗户旁。 |
| `HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan` | 117、1018 | 依次走到书桌旁、合上笔记本电脑、走到垃圾桶旁。 |
| `HolodeckApproachTrashCanThenApproachDeskThenPullChair` | 117、1018 | 依次走到垃圾桶旁、走到书桌旁、拉出桌下的办公椅。 |
| `HolodeckPickBookFromSourceToDeskThenApproachTrashCan` | 117、1018 | 先拿起书本并放到书桌上，再走到垃圾桶旁。 |
| `HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan` | 117、1018 | 依次从已打开的抽屉取出纸巾盒并放到书桌上、关闭抽屉、走到垃圾桶旁。 |
| `HolodeckApproachDeskThenIceCubeInGlass` | 117、1018、1197、1664、2025 | 先走到书桌旁，再将冰块放入玻璃杯中。 |
| `HolodeckCloseJarThenApproachDeskThenApproachTrashCan` | 117、1018、1664 | 先盖好罐盖并保持 1 秒，再依次走到书桌旁和垃圾桶旁。 |
| `HolodeckApproachTrashCanThenApproachDeskThenOpenJar` | 117、1018 | 依次走到垃圾桶旁、走到罐子所在书桌旁、取下罐盖并保持脱离接触 1 秒。 |
| `HolodeckIceCubeThenApproachWindowThenPickBox` | 117、1018、1664、2025 | 依次将冰块放入杯子、走到指定窗户旁、将地上的纸箱抬高至少 0.3 米；每个目标状态保持 1 秒。 |
| `HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer` | 117、1197 | 依次走到书桌旁、将桌上的纸箱放到地面、关闭指定抽屉；每个目标状态保持 1 秒。 |
| `HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat` | 117、1018、1664、2025 | 先拿起杯子并抬高至少 0.3 米，再移动至少 0.15 米到目标书桌旁，将杯子放到餐垫上；每个目标状态保持 1 秒。 |
| `HolodeckPlaceBoxOnFloorThenPlaceBoxOnDesk` | 117、1018、1197、1664、2025 | 先将桌上的纸箱放到地面，再将地上的纸箱放回书桌；每个目标状态保持 1 秒。 |

## 已有验证记录

以下记录从原有房间任务文档合并，用于保留验证范围；本次文档整理未重新运行任务测试或仿真。

- iTHOR 卧室/浴室：原记录包含 31 条导航路径的 reset 与短时稳定检查；2026-09-05 的 6 条浴室原子初始化路径完成了定向检查；2026-09-06 的 35 个浴室复合任务、87 个任务与场景组合完成了 headless 检查。均未完成 GUI、人工全流程或采集验收。
- iTHOR 客厅：2026-09-08 的七个到桌操作复合任务已有 15 项定向测试通过；Scene 209 的哑铃抬升与书本入篮各完成一次显式 reset，机器人无碰撞、脚底在地板内、初始不成功。其余场景的完整操作可达性及人工抓放未验证。
- iTHOR 厨房：本轮新增四个导航循环/转身投放任务已有 8 项定向测试通过；使用 `G1FloatingBody` 在 Scene 13 对三种初始化实现各完成一次显式 reset，机器人无碰撞、脚底在地板内、初始不成功。桌面与抽屉苹果距机器人约 0.459/0.506 米，垃圾桶入口约 0.600 米；各进行一次 100 物理步的物体稳定检查，位移约 3.4 毫米。四个投放场景的垃圾桶布局未检测到超过 3 毫米的非地板穿插。其余场景的机器人 reset、人工抓取/转身/投放和采集未验证。

- Holodeck 1477（2026-10-14）：新增 `HolodeckApproachWindowThenCubeInPlateThenOpenLaptop`，仅支持 1477。静态确认三层包导出、Holodeck/robosuite 注册、最右侧窗户 stable ID，以及 117 笔记本原生 body/关节/密度配置原样复制；魔方与盘子的初始化直接继承 `HolodeckCubeInPlate`。4 项定向测试覆盖 20/50 Hz 正常顺序、提前摆放/开盖和最终保持中断。1 项 headless reset 与 100 物理步稳定检查通过，使用 `G1FloatingBody`，初始不成功、机器人无碰撞、脚印位于房间多边形内、电脑稳定且保持合盖。首次 floor 检查使用了不适用于该导出平面地板的语义 Floor helper，改用房间多边形后仅重跑失败用例。命令：`MUJOCO_GL=egl python -m pytest -q tests/test_holodeck_1477_cube_laptop_workflow.py`，失败修复后的定向复测增加 `::test_one_headless_reset_and_tabletop_stability`。GUI、人工全流程和数据采集未验证。
- iTHOR 浴室 405（2026-10-14）：新增 `IThorBathroomLiftSoapBottleFromCabinet`，仅支持 405。柜门初始全开，原生 SoapBottle 放在柜内固定小桌上；实测支撑面高约 0.448 米，瓶子根部初始高约 0.564 米，距机器人约 0.687 米。成功仅检查相对 reset 高度抬高至少 0.30 米并连续保持 1 秒，与抓握和手部接触无关。注册与去重、10/20 Hz 保持及中断单元测试通过；一次代表性 headless reset 检查通过，机器人无碰撞、脚印在 Floor，0.5 秒静置后瓶子位移约 0.11 毫米，无抓握的高度正例和高度不足反例通过。场景检查命令：`MUJOCO_GL=egl python -m pytest -q tests/test_ithor_bathroom_cabinet_lift.py::test_405_reset_support_stability_and_physical_lift -s`。已有任务规格经静态差异检查保持一致；GUI、人工抓取和数据采集未验证。
- iTHOR 浴室 422（2026-10-15）：新增独立原子任务 `IThorBathroomOpenUpperCabinetDoor`，精确绑定上方壁柜 `FloorPlan422:Cabinet_a6b630aa_2` 的门铰链，初始开度 20%，达到至少 80% 并连续保持 1 秒成功。一次 G1 headless reset 通过：初始不成功、机器人碰撞数为 0、完整脚印在 Floor，20%→80% 扫掠无机器人碰撞；初始把手交互点高度约 1.562 米。注册/去重及 10/20 Hz 阈值、保持和中断测试共 3 项通过，命令：`MUJOCO_GL=egl python -m pytest -q tests/test_ithor_bathroom_open_upper_cabinet_422.py`。语法与 `git diff --check` 通过，原有已提交的 84 项浴室任务规格保持一致；GUI、人工开门和数据采集未验证。启动命令：`python -m robocasa.scripts.visualize_scene --env IThorBathroomOpenUpperCabinetDoor --scene-id 422 --onscreen`。
- iTHOR 浴室 422（2026-10-15）：新增独立原子任务 `IThorBathroomCloseUpperCabinetDoor`，复用打开任务的上方壁柜关节与站位；初始开度 80%，关闭至不超过 20% 并连续保持 1 秒成功。3 项注册/去重、精确目标、10/20 Hz 阈值与保持中断测试通过，命令：`MUJOCO_GL=egl python -m pytest -q tests/test_ithor_bathroom_close_upper_cabinet_422.py`。一次 G1 headless reset 通过，初始不成功、机器人碰撞数为 0、脚印在 Floor，80%→20% 扫掠无机器人碰撞；语法与 `git diff --check` 通过。与本轮修改前工作区快照对比，原有 89 项浴室场景/任务规格及全部已有 articulation 类保持一致。GUI、人工关门和数据采集未验证，无已知阻塞。启动命令：`python -m robocasa.scripts.visualize_scene --env IThorBathroomCloseUpperCabinetDoor --scene-id 422 --onscreen`。

- Holodeck 2209（2026-10-16）：新增 `HolodeckPlaceAlarmClockThenCloseDrawerThenPickBox` 和 `HolodeckOpenDrawerThenPlaceAlarmClockThenApproachTrashCan`，原子任务及共享场景保持不变。7 个定向测试最终通过，覆盖注册、20/50 Hz 顺序与保持中断、两个初始化分支的 headless reset 和自由物体短时稳定性。初始化失败修复后只复测失败用例；GUI、人工全流程和采集未验证。设计、场景内新增资产与命令见 `docs/holodeck_2209_alarm_clock_workflows.md`。
