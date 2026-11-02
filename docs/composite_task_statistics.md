# 复合任务分布统计

统计规则：

所有房间、场景和任务族的复合任务统计统一按以下顺序执行：

1. **先判定原子任务**：盘点当前注册任务和自然语言指令，将每个单一物理动作归入导航、站立铰链或轨道开合、站立拿取与放置、需要弯腰或下蹲全身协调四类；先固定原子任务名称、类别和别名去重口径。
2. **提交人工确认**：把原子任务清单、类别边界以及别名/重复任务处理方式交由人工确认。确认前不开始复合任务拆分，也不生成最终统计数字；若人工修改分类，必须以确认后的版本重新计算。
3. **再拆分复合任务**：仅使用已确认的原子任务表，根据复合任务的自然语言指令按执行顺序拆分；记录每个复合任务包含的原子动作类别和出现次数。隐含的跨工作面移动、开合环或低位全身操作，按已确认的统计口径记录，并在对应章节说明。
4. **最后统计出现次数**：对拆分结果分别应用两种标准：标准 A 在单个复合任务内对同一类别去重后计数；标准 B 保留同一复合任务内同一类别的重复次数后累计。汇总前检查每个复合任务均已拆分，且类别总数可以由明细复算。

## kitchen

共 61 个纳入统计的任务：25 个原子任务、36 个复合任务。

### 原子任务统计
按四类原子动作归类，共 25 个任务；该清单已完成人工确认，以下复合任务拆分和 A/B 统计均以此版本为准。虽然 `IThorKitchenPackProduceIntoBox` 和 `IThorKitchenCupStack` 位于 `tasks/atomic/` 目录，但它们的自然语言指令分别包含两个物体放入和两个杯子叠放，已按多阶段动作归入下面的复合任务清单。

- 导航（3）
    - `IThorKitchenApproachFridge`
    - `IThorKitchenApproachOven`
    - `IThorKitchenApproachDishwasher`
- 站立铰链或轨道开合类（8）
    - `IThorKitchenOpenFridge`
    - `IThorKitchenCloseFridge`
    - `IThorKitchenOpenCabinet`
    - `IThorKitchenCloseCabinet`
    - `IThorKitchenOpenDrawer`
    - `IThorKitchenCloseDrawer`
    - `IThorKitchenOpenOven`
    - `IThorKitchenCloseOven`
- 站立拿取与放置类（14）
    - `IThorKitchenAppleOnPlate`
    - `IThorKitchenBreadOnPlate`
    - `IThorKitchenCheeseOnPlate`
    - `IThorKitchenMugOnPlate`
    - `IThorKitchenAppleIntoBox`
    - `IThorKitchenCupIntoBox`
    - `IThorKitchenStatueIntoBox`
    - `IThorKitchenBowlToSink`
    - `IThorKitchenCupToSink`
    - `IThorKitchenStandCannedDrinkUpright`
    - `IThorKitchenPanTableToStove`
    - `IThorKitchenLiftEggplant`
    - `IThorKitchenLiftPlate`
    - `IThorKitchenLiftSaltShaker`
- 需要弯腰或下蹲全身协调（0）

合计：3 + 8 + 14 + 0 = 25 个原子任务。

### 复合任务统计

#### 原子动作拆分
拆分记法为“类别×次数”。下列类别顺序固定为：导航、站立铰链或轨道开合、站立拿取与放置、需要弯腰或下蹲全身协调。跨工作面搬运按一次导航计数；同一关节的开→关按一个开合环计数；`BendAndPutAppleOnPlate` 的低位操作只计入全身协调类；`AppleBananaPlateSwap`、`PackProduceIntoBox` 和 `CupStack` 的两个独立物体分别计作两次站立拿取与放置。

- `IThorKitchenBendAndPutAppleOnPlate`：需要弯腰或下蹲全身协调×1
- `IThorKitchenApproachOpenFridge`：导航×1 + 站立铰链或轨道开合×1
- `IThorKitchenApproachOpenOven`：导航×1 + 站立铰链或轨道开合×1
- `IThorKitchenBananaBasketToTable`：站立拿取与放置×1
- `IThorKitchenCoffeeDrinkToTrash`：站立拿取与放置×1
- `IThorKitchenApproachTableAndStandCannedDrinkUpright`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachStoveAndPutPan`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachSinkAndPutBowl`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachSinkAndPutCup`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachDrawerAndOpen`：导航×1 + 站立铰链或轨道开合×1
- `IThorKitchenApproachDrawerAndClose`：导航×1 + 站立铰链或轨道开合×1
- `IThorKitchenApproachCabinetAndOpen`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorKitchenApproachCabinetAndClose`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorKitchenApproachDrawerAndOpenClose`：导航×1 + 站立铰链或轨道开合×2
- `IThorKitchenApproachCabinetAndOpenClose`：导航×1 + 需要弯腰或下蹲全身协调×2
- `IThorKitchenPickAppleAndTurnToTrash`：导航×1 + 站立拿取与放置×1
- `IThorKitchenOpenDrawerAndTurnAppleToTrash`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `IThorKitchenDrawerPnP`：站立铰链或轨道开合×1 + 站立拿取与放置×1
- `IThorKitchenAppleCabinetPnP`需要弯腰或下蹲全身协调×2 + 站立拿取与放置×1
- `IThorKitchenAppleBananaPlateSwap`：站立拿取与放置×2
- `IThorKitchenOpenDrawerTakeApple`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `IThorKitchenOvenThenFridge`：导航×2
- `IThorKitchenFridgeThenOven`：导航×2
- `IThorKitchenFridgeThenDishwasher`：导航×2
- `IThorKitchenDishwasherThenFridge`：导航×2
- `IThorKitchenOvenThenDishwasher`：导航×2
- `IThorKitchenDishwasherThenOven`：导航×2
- `IThorKitchenApproachTableAndLiftApple`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachTableAndLiftSaltShaker`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachTableAndLiftBowl`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachTableAndPutAppleOnPlate`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachTableAndPutBreadOnPlate`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachTableAndPutCheeseOnPlate`：导航×1 + 站立拿取与放置×1
- `IThorKitchenApproachTableAndPutMugOnPlate`：导航×1 + 站立拿取与放置×1
- `IThorKitchenPackProduceIntoBox`：站立拿取与放置×2
- `IThorKitchenCupStack`：站立拿取与放置×2
- `IThorKitchenOpenCloseLowMicrowave`：需要弯腰或下蹲全身协调×2
- `IThorKitchenPutAppleInLowMicrowave`：需要弯腰或下蹲全身协调×2
- `IThorKitchenOpenLowMicrowaveAndLiftApple`：需要弯腰或下蹲全身协调×2
- `IThorKitchenTakeAppleFromOpenLowMicrowaveToTop`：需要弯腰或下蹲全身协调×2
- `IThorKitchenCloseThenOpenLowMicrowave`：需要弯腰或下蹲全身协调×2
- `IThorKitchenOpenLowMicrowaveAndPutAppleInside`：需要弯腰或下蹲全身协调×2


#### 四类原子动作出现次数

标准 A：同一复合任务中，同一类别出现多次时只计一次。标准 B：保留同一复合任务中同一类别的重复出现次数。统计对象为上面列出的 36 个复合任务。

| 统计标准 | 导航 | 站立铰链或轨道开合类 | 站立拿取与放置类 | 需要弯腰或下蹲全身协调 | 合计 |
|---|---:|---:|---:|---:|---:|
| 标准 A（同类去重） | 28 | 12 | 21 | 1 | 62 |
| 标准 B（重复累计） | 34 | 12 | 24 | 1 | 71 |

## living room

共 48 个任务：21 个原子任务、27 个复合任务。

### 原子任务统计

- 导航（2）
    - `IThorLivingRoomApproachSofa`
    - `IThorLivingRoomApproachTable`
- 站立铰链或轨道开合类（6）
    - `IThorLivingRoomOpenDrawer`
    - `IThorLivingRoomCloseDrawer`
    - `IThorLivingRoomOpenLaptop`
    - `IThorLivingRoomCloseLaptop`
    - `IThorLivingRoomOpenBoxFlap`
    - `IThorLivingRoomCloseBoxFlap`
- 站立拿取与放置类（7）
    - `IThorLivingRoomPourBlockFromBasket`
    - `IThorLivingRoomPourIceFromCup`
    - `IThorLivingRoomLiftCupAmongObjects`
    - `IThorLivingRoomStackBookOnBook`
    - `IThorLivingRoomCandleToBasket`
    - `IThorLivingRoomBookIntoBasket`
    - `IThorLivingRoomPushBlockToTargetRegion`
- 需要弯腰或下蹲全身协调（6）
    - `IThorLivingRoomLiftDumbbell`
    - `IThorLivingRoomLiftStatue`
    - `IThorLivingRoomLiftWateringCan`
    - `IThorLivingRoomPushBasketball`
    - `IThorLivingRoomRemoteToBasket`
    - `IThorLivingRoomTissueBoxToBasket`

合计：2 + 6 + 7 + 6 = 21 个原子任务。

### 复合任务统计

#### 原子动作拆分

拆分记法为“类别×次数”。低抽屉任务包含低位操作，计入需要弯腰或下蹲全身协调类；同一抽屉的打开和关闭按一个开合环计数；两个独立物体分别计作两次站立拿取与放置。

- `IThorLivingRoomCoffeeDrinkToBin`：导航×1 + 站立拿取与放置×1
- `IThorLivingRoomRemoteToBin`：站立拿取与放置×1
- `IThorLivingRoomBasketToSofa`：站立拿取与放置×1
- `IThorLivingRoomDumbbellToSofa`：站立拿取与放置×1
- `IThorLivingRoomStatueToSofa`：站立拿取与放置×1
- `IThorLivingRoomPackCandleStatueIntoBasket`：站立拿取与放置×2
- `IThorLivingRoomWalkAndOpenDrawer`：导航×1 + 站立铰链或轨道开合×1
- `IThorLivingRoomWalkAndCloseDrawer`：导航×1 + 站立铰链或轨道开合×1 + 需要弯腰或下蹲全身协调×1
- `IThorLivingRoomWalkAndOpenCloseDrawer`：导航×1 + 站立铰链或轨道开合×2
- `IThorLivingRoomWalkToTableAndPutCanInBasket`：导航×1 + 站立拿取与放置×1
- `IThorLivingRoomTableThenSofa`：导航×2
- `IThorLivingRoomSofaThenTable`：导航×2
- `IThorLivingRoomBinThenTable`：导航×2
- `IThorLivingRoomTableThenBin`：导航×2
- `IThorLivingRoomBinThenSofa`：导航×2
- `IThorLivingRoomSofaThenBin`：导航×2
- `IThorLivingRoomApproachTableAndLiftDumbbell`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorLivingRoomApproachTableAndLiftStatue`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorLivingRoomApproachTableAndLiftWateringCan`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorLivingRoomApproachTableAndPutRemoteInBasket`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorLivingRoomApproachTableAndPutCandleInBasket`：导航×1 + 站立拿取与放置×1
- `IThorLivingRoomApproachTableAndPutTissueBoxInBasket`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorLivingRoomApproachTableAndPutBookInBasket`：导航×1 + 站立拿取与放置×1
- `IThorLivingRoomWalkAndOpenLowDrawer`：导航×1 + 站立铰链或轨道开合×1 + 需要弯腰或下蹲全身协调×1
- `IThorLivingRoomWalkAndCloseLowDrawer`：导航×1 + 站立铰链或轨道开合×1 + 需要弯腰或下蹲全身协调×1
- `IThorLivingRoomLowDrawerPnP`：站立铰链或轨道开合×1 + 站立拿取与放置×1 + 需要弯腰或下蹲全身协调×1
- `IThorLivingRoomStackBooksAndPlaceMug`：站立拿取与放置×2

#### 四类原子动作出现次数

标准 A：同一复合任务中，同一类别出现多次时只计一次。标准 B：保留同一复合任务中同一类别的重复出现次数。统计对象为上面列出的 27 个复合任务。

| 统计标准 | 导航 | 站立铰链或轨道开合类 | 站立拿取与放置类 | 需要弯腰或下蹲全身协调 | 合计 |
|---|---:|---:|---:|---:|---:|
| 标准 A（同类去重） | 20 | 6 | 11 | 9 | 46 |
| 标准 B（重复累计） | 26 | 7 | 13 | 9 | 55 |

## bedroom
### 原子任务统计
共 52 个纳入统计的任务：20 个原子任务、32 个复合任务。

按四类原子动作归类，共 20 个任务；以下复合任务拆分和 A/B 统计均以此清单为准。

- 导航（3）
    - `IThorBedroomApproachBed`
    - `IThorBedroomApproachDesk`
    - `IThorBedroomNavigateToStorage`
- 站立铰链或轨道开合类（6）
    - `IThorBedroomOpenLaptop`
    - `IThorBedroomCloseLaptop`
    - `IThorBedroomOpenSafeDoor`
    - `IThorBedroomCloseSafeDoor`
    - `IThorBedroomOpenHamperLid`
    - `IThorBedroomCloseHamperLid`
- 站立拿取与放置类（4）
    - `IThorBedroomCanIntoBasket`
    - `IThorBedroomMugIntoBasket`
    - `IThorBedroomBoxedFoodIntoBasket`
    - `IThorBedroomLiftBasket`
- 需要弯腰或下蹲全身协调（7）
    - `IThorBedroomLiftBasketFromFloor`
    - `IThorBedroomOpenBoxFlap`
    - `IThorBedroomCloseBoxFlap`
    - `IThorBedroomOpenDrawer`
    - `IThorBedroomCloseDrawer`
    - `IThorBedroomOpenTopDrawer`
    - `IThorBedroomCloseTopDrawer`

合计：3 + 6 + 4 + 7 = 20 个原子任务。

### 复合任务统计

#### 原子动作拆分

拆分记法为“类别×次数”。类别顺序固定为：导航、站立铰链或轨道开合、站立拿取与放置、需要弯腰或下蹲全身协调。跨工作面移动按一次导航计数；`OpenThenClose*` 任务中的打开和关闭分别计为两次开合动作；`IThorBedroomLiftBasketFromFloor` 仅作为原子任务出现，复合任务列表中没有对应的低位搬运复合任务。

- `IThorBedroomBedThenDesk`：导航×2
- `IThorBedroomDeskThenBed`：导航×2
- `IThorBedroomBedThenStorage`：导航×2
- `IThorBedroomStorageThenBed`：导航×2
- `IThorBedroomDeskThenStorage`：导航×2
- `IThorBedroomStorageThenDesk`：导航×2
- `IThorBedroomApproachDeskAndOpenLaptop`：导航×1 + 站立铰链或轨道开合×1
- `IThorBedroomApproachDeskAndCloseLaptop`：导航×1 + 站立铰链或轨道开合×1
- `IThorBedroomApproachDeskAndLiftBasket`：导航×1 + 站立拿取与放置×1
- `IThorBedroomApproachDeskAndPutCanInBasket`：导航×1 + 站立拿取与放置×1
- `IThorBedroomApproachDeskAndPutMugInBasket`：导航×1 + 站立拿取与放置×1
- `IThorBedroomApproachDeskAndPutBoxedFoodInBasket`：导航×1 + 站立拿取与放置×1
- `IThorBedroomApproachStorageAndOpenSafeDoor`：导航×1 + 站立铰链或轨道开合×1
- `IThorBedroomApproachStorageAndCloseSafeDoor`：导航×1 + 站立铰链或轨道开合×1
- `IThorBedroomApproachStorageAndOpenHamperLid`：导航×1 + 站立铰链或轨道开合×1
- `IThorBedroomApproachStorageAndCloseHamperLid`：导航×1 + 站立铰链或轨道开合×1
- `IThorBedroomApproachStorageAndOpenDrawer`：导航×1 + 站立铰链或轨道开合×1
- `IThorBedroomApproachStorageAndCloseDrawer`：导航×1 + 站立铰链或轨道开合×1
- `IThorBedroomApproachStorageAndOpenBoxFlap`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorBedroomApproachStorageAndCloseBoxFlap`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorBedroomOpenThenCloseDrawer`：需要弯腰或下蹲全身协调×2
- `IThorBedroomOpenThenCloseTopDrawer`：需要弯腰或下蹲全身协调×2
- `IThorBedroomOpenThenCloseSafeDoor`：站立铰链或轨道开合×2
- `IThorBedroomOpenThenCloseHamperLid`：站立铰链或轨道开合×2
- `IThorBedroomOpenThenCloseBoxFlap`：需要弯腰或下蹲全身协调×2
- `IThorBedroomOpenThenCloseLaptop`：站立铰链或轨道开合×2
- `IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket`：导航×1 + 站立拿取与放置×2
- `IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `IThorBedroomCloseLaptopThenApproachBedAndCloseTopDrawer`：导航×1 + 站立铰链或轨道开合×2
- `IThorBedroomApproachDeskAndCloseLaptopThenApproachBed`：导航×2 + 站立铰链或轨道开合×1
- `IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk`：导航×2 + 站立铰链或轨道开合×1
- `IThorBedroomPutCanThenBoxedFoodThenMugInBasket`：站立拿取与放置×3

#### 四类原子动作出现次数

标准 A：同一复合任务中，同一类别出现多次时只计一次。标准 B：保留同一复合任务内同一类别的重复出现次数。统计对象为上面列出的 32 个复合任务。

| 统计标准 | 导航 | 站立铰链或轨道开合类 | 站立拿取与放置类 | 需要弯腰或下蹲全身协调 | 合计 |
|---|---:|---:|---:|---:|---:|
| 标准 A（同类去重） | 25 | 15 | 7 | 5 | 52 |
| 标准 B（重复累计） | 33 | 19 | 10 | 8 | 70 |

## bathroom
### 原子任务统计
共 62 个纳入统计的任务：21 个原子任务、41 个复合任务。

按四类原子动作归类，共 21 个任务；以下复合任务拆分和 A/B 统计均以此清单为准。

- 导航（4）
    - `IThorBathroomApproachToilet`
    - `IThorBathroomApproachSink`
    - `IThorBathroomNavigateToBathingArea`
    - `IThorBathroomApproachGarbageCan`
- 站立铰链或轨道开合类（4）
    - `IThorBathroomOpenToiletLid`
    - `IThorBathroomCloseToiletLid`
    - `IThorBathroomOpenShowerDoor`
    - `IThorBathroomCloseShowerDoor`
- 站立拿取与放置类（8）
    - `IThorBathroomLiftSoapBottle`
    - `IThorBathroomLiftToiletPaperRoll`
    - `IThorBathroomLiftSprayBottle`
    - `IThorBathroomToiletPaperRollIntoBin`
    - `IThorBathroomStandSoapBottleUpright`
    - `IThorBathroomLiftScrubBrush`
    - `IThorBathroomPutSpongeInBathtub`
    - `IThorBathroomSoapToLowerShelf`
- 需要弯腰或下蹲全身协调（5）
    - `IThorBathroomLiftCandleFromFloor`
    - `IThorBathroomOpenLowerDrawer`
    - `IThorBathroomCloseLowerDrawer`
    - `IThorBathroomOpenCabinetDoor`
    - `IThorBathroomCloseCabinetDoor`

合计：4 + 4 + 8 + 5 = 21 个原子任务。

### 复合任务统计

#### 原子动作拆分

拆分记法为“类别×次数”。类别顺序固定为：导航、站立铰链或轨道开合、站立拿取与放置、需要弯腰或下蹲全身协调。跨工作面移动按一次导航计数；开→关或关→开复合任务中的两个方向分别计为两次开合动作；独立物体或目标分别计作一次站立拿取与放置；涉及 lower drawer 的操作计入需要弯腰或下蹲全身协调类。`IThorBathroomLiftCandleFromFloor` 的低位操作只作为原子任务出现，复合任务中没有对应的低位搬运任务。

- `IThorBathroomNavigateToSinkThenToilet`：导航×2
- `IThorBathroomNavigateToToiletThenSink`：导航×2
- `IThorBathroomNavigateToSinkThenBathingArea`：导航×2
- `IThorBathroomNavigateToBathingAreaThenSink`：导航×2
- `IThorBathroomNavigateToToiletThenBathingArea`：导航×2
- `IThorBathroomNavigateToBathingAreaThenToilet`：导航×2
- `IThorBathroomNavigateToGarbageCanThenSink`：导航×2
- `IThorBathroomNavigateToSinkThenGarbageCan`：导航×2
- `IThorBathroomNavigateToGarbageCanThenToilet`：导航×2
- `IThorBathroomNavigateToToiletThenGarbageCan`：导航×2
- `IThorBathroomNavigateToGarbageCanThenBathingArea`：导航×2
- `IThorBathroomNavigateToBathingAreaThenGarbageCan`：导航×2
- `IThorBathroomApproachToiletAndOpenToiletLid`：导航×1 + 站立铰链或轨道开合×1
- `IThorBathroomApproachToiletAndCloseToiletLid`：导航×1 + 站立铰链或轨道开合×1
- `IThorBathroomNavigateToBathingAreaAndOpenShowerDoor`：导航×1 + 站立铰链或轨道开合×1
- `IThorBathroomNavigateToBathingAreaAndCloseShowerDoor`：导航×1 + 站立铰链或轨道开合×1
- `IThorBathroomApproachSinkAndOpenCabinetDoor`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorBathroomApproachSinkAndCloseCabinetDoor`：导航×1 + 需要弯腰或下蹲全身协调×1
- `IThorBathroomNavigateToBathingAreaAndLiftSoapBottle`：导航×1 + 站立拿取与放置×1
- `IThorBathroomApproachSinkAndLiftSoapBottle`：导航×1 + 站立拿取与放置×1
- `IThorBathroomApproachSinkAndLiftToiletPaperRoll`：导航×1 + 站立拿取与放置×1
- `IThorBathroomApproachSinkAndLiftSprayBottle`：导航×1 + 站立拿取与放置×1
- `IThorBathroomNavigateToBathingAreaAndLiftScrubBrush`：导航×1 + 站立拿取与放置×1
- `IThorBathroomApproachSinkAndLiftScrubBrush`：导航×1 + 站立拿取与放置×1
- `IThorBathroomNavigateToBathingAreaAndStandSoapBottleUpright`：导航×1 + 站立拿取与放置×1
- `IThorBathroomNavigateToBathingAreaAndPutSpongeInBathtub`：导航×1 + 站立拿取与放置×1
- `IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin`：导航×1 + 站立拿取与放置×1
- `IThorBathroomOpenThenCloseToiletLid`：站立铰链或轨道开合×2
- `IThorBathroomCloseThenOpenToiletLid`：站立铰链或轨道开合×2
- `IThorBathroomOpenThenCloseShowerDoor`：站立铰链或轨道开合×2
- `IThorBathroomCloseThenOpenShowerDoor`：站立铰链或轨道开合×2
- `IThorBathroomOpenThenCloseCabinetDoor`：需要弯腰或下蹲全身协调×2
- `IThorBathroomCloseThenOpenCabinetDoor`：需要弯腰或下蹲全身协调×2
- `IThorBathroomOpenThenCloseLowerDrawer`：需要弯腰或下蹲全身协调×2
- `IThorBathroomCloseThenOpenLowerDrawer`：需要弯腰或下蹲全身协调×2
- `IThorBathroomNavigateToBathingAreaAndStandSoapBottleUprightThenPutSpongeInBathtub`：导航×1 + 站立拿取与放置×2
- `IThorBathroomSoapToLowerShelfThenStandSoapBottleUprightAndPutSpongeInBathtub`：站立拿取与放置×3
- `IThorBathroomCloseToiletLidThenApproachSinkAndLiftSoapBottle`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `IThorBathroomApproachToiletAndOpenToiletLidThenLiftScrubBrush`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `IThorBathroomApproachSinkAndOpenCabinetDoorThenLiftSprayBottle`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `IThorBathroomOpenUpperDrawerThenLiftToiletPaperRollAndCloseUpperDrawer`：站立铰链或轨道开合×1 + 站立拿取与放置×1

#### 四类原子动作出现次数

标准 A：同一复合任务中，同一类别出现多次时只计一次。标准 B：保留同一复合任务中同一类别的重复出现次数。统计对象为上面列出的 41 个复合任务。

| 统计标准 | 导航 | 站立铰链或轨道开合类 | 站立拿取与放置类 | 需要弯腰或下蹲全身协调 | 合计 |
|---|---:|---:|---:|---:|---:|
| 标准 A（同类去重） | 31 | 12 | 15 | 6 | 64 |
| 标准 B（重复累计） | 43 | 16 | 18 | 10 | 87 |

## office
### 原子任务统计

统计口径：按单一物理动作统计，共 25 个原子任务；同一物理动作的开、关方向分别计数。`HolodeckPlaceBoxFromDeskToFloor` 是 `HolodeckPlaceBoxOnFloor` 的代码别名，不重复计数。虽然 `HolodeckTakeTissueBoxFromOpenDrawer` 和 `HolodeckThrowBottleIntoTrashCan` 的实现元数据沿用复合任务族，它们各自只有一个取放动作，因此按原子任务计入；`HolodeckOpenDrawerThenTakeTissueBox` 则保留为复合任务，因为它明确包含“开抽屉→取纸巾盒”两个阶段。

- 导航（3）
    - `HolodeckApproachDesk`
    - `HolodeckApproachTrashCan`
    - `HolodeckApproachWindow`
- 站立铰链或轨道开合类（4）
    - `HolodeckOpenLaptop`
    - `HolodeckCloseLaptop`
    - `HolodeckOpenDrawer`
    - `HolodeckCloseDrawer`
- 站立拿取与放置类（13）
    - `HolodeckIceCubeInGlass`
    - `HolodeckOpenMugLid`
    - `HolodeckCloseMugLid`
    - `HolodeckSugarCubeInJar`
    - `HolodeckMoveCupToCenter`
    - `HolodeckMoveCupToCorner`
    - `HolodeckPickBookFromDresser`
    - `HolodeckPickCupFromDesk`
    - `HolodeckPlaceCupOnPlacemat`
    - `HolodeckInsertFolderIntoShelf`
    - `HolodeckRemoveFolderFromShelf`
    - `HolodeckTakeTissueBoxFromOpenDrawer`
    - `HolodeckThrowBottleIntoTrashCan`
- 需要弯腰或下蹲全身协调（5）
    - `HolodeckPickBoxFromFloor`
    - `HolodeckPlaceBoxOnFloor`
    - `HolodeckPlaceBoxOnDesk`
    - `HolodeckPushChairToDesk`
    - `HolodeckPullChairFromDesk`

第四类按需要改变身体高度或利用躯干、下肢施力的低位/家具操作归类，因此包含桌面纸箱放置、地面纸箱和桌下椅子推拉。

合计：3 + 4 + 13 + 5 = 25 个原子任务。


### 复合任务统计

#### 原子动作拆分

拆分符号与口径：`导航`按自然语言中的走到/返回，以及跨工作面搬运所需的一次移动计数；`站立铰链或轨道开合`按一个关节的开合环计数；每个独立物体目标计一次`站立拿取与放置`；地面/桌面纸箱和椅子推拉计入`需要弯腰或下蹲全身协调`。下面先列出每个复合任务中的动作出现次数，汇总时再分别应用标准 A、标准 B。

- `HolodeckApproachDeskThenOpenLaptop`：导航×1 + 站立铰链或轨道开合×1
- `HolodeckOpenDrawerThenTakeTissueBox`：站立铰链或轨道开合×1 + 站立拿取与放置×1
- `HolodeckOpenDrawerTakeTissueBoxToDesk`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `HolodeckOpenDrawerTakeTissueBoxToDeskThenCloseDrawer`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `HolodeckSugarCubeThenCloseJar`：站立拿取与放置×2
- `HolodeckOpenJarAddSugarThenCloseJar`：站立拿取与放置×3
- `HolodeckApproachDeskThenCloseLaptopMoveCup`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `HolodeckRemoveFolderFromShelfToDesk`：导航×1 + 站立拿取与放置×1
- `HolodeckIceCubeThenPlaceCup`：站立拿取与放置×2
- `HolodeckApproachDeskThenPullChair`：导航×1 + 需要弯腰或下蹲全身协调×1
- `HolodeckApproachWindowThenDeskThenPickCup`：导航×2 + 站立拿取与放置×1
- `HolodeckPickBookFromSourceToDesk`：导航×1 + 站立拿取与放置×1
- `HolodeckApproachDeskPullChairThenOpenLaptop`：导航×1 + 站立铰链或轨道开合×1 + 需要弯腰或下蹲全身协调×1
- `HolodeckMoveCupToCenterThenOpenLaptop`：站立铰链或轨道开合×1 + 站立拿取与放置×1
- `HolodeckRemoveFolderFromShelfToDeskThenOpenLaptop`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `HolodeckApproachDeskIceCubePlaceCupThenPickCup`：导航×1 + 站立拿取与放置×3
- `HolodeckCloseLaptopCloseDrawerThenApproachWindow`：导航×1 + 站立铰链或轨道开合×2
- `HolodeckTakeTissueBoxThenPlaceCup`：导航×1 + 站立拿取与放置×2
- `HolodeckApproachTrashCanThenThrowBottle`：导航×1 + 站立拿取与放置×1
- `HolodeckRemoveFolderFromShelfToDeskThenCloseLaptop`：导航×1 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `HolodeckApproachWindowThenDesk`：导航×2
- `HolodeckApproachTrashCanThenThrowBottleThenApproachDesk`：导航×2 + 站立拿取与放置×1
- `HolodeckApproachDeskThenOpenLaptopThenApproachWindow`：导航×2 + 站立铰链或轨道开合×1
- `HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan`：导航×2 + 站立铰链或轨道开合×1
- `HolodeckApproachTrashCanThenApproachDeskThenPullChair`：导航×2 + 需要弯腰或下蹲全身协调×1
- `HolodeckPickBookFromSourceToDeskThenApproachTrashCan`：导航×2 + 站立拿取与放置×1
- `HolodeckTakeTissueBoxThenCloseDrawerThenApproachTrashCan`：导航×2 + 站立铰链或轨道开合×1 + 站立拿取与放置×1
- `HolodeckApproachDeskThenIceCubeInGlass`：导航×1 + 站立拿取与放置×1
- `HolodeckCloseJarThenApproachDeskThenApproachTrashCan`：导航×2 + 站立拿取与放置×1
- `HolodeckApproachTrashCanThenApproachDeskThenOpenJar`：导航×2 + 站立拿取与放置×1
- `HolodeckApproachTrashCanThenPickBoxFromFloor`：导航×1 + 需要弯腰或下蹲全身协调×1
- `HolodeckApproachTrashCanThenPickBoxAndPlaceOnDesk`：导航×2 + 需要弯腰或下蹲全身协调×2
- `HolodeckCloseLaptopThenApproachTrashCanThenPickBox`：导航×1 + 站立铰链或轨道开合×1 + 需要弯腰或下蹲全身协调×1
- `HolodeckApproachWindowThenPickBox`：导航×1 + 需要弯腰或下蹲全身协调×1
- `HolodeckPullChairThenApproachTrashCanThenPickBox`：导航×1 + 需要弯腰或下蹲全身协调×2
- `HolodeckIceCubeThenApproachWindowThenPickBox`：导航×1 + 站立拿取与放置×1 + 需要弯腰或下蹲全身协调×1
- `HolodeckApproachDeskPlaceBoxOnFloorThenCloseDrawer`：导航×1 + 站立铰链或轨道开合×1 + 需要弯腰或下蹲全身协调×1
- `HolodeckPickCupThenApproachDeskThenPlaceCupOnPlacemat`：导航×1 + 站立拿取与放置×2
- `HolodeckPlaceBoxOnFloorThenPickBook`：站立拿取与放置×1 + 需要弯腰或下蹲全身协调×1
- `HolodeckApproachDeskThenPickCup`：导航×1 + 站立拿取与放置×1
- `HolodeckApproachDeskThenMoveCupToCenter`：导航×1 + 站立拿取与放置×1
- `HolodeckApproachDeskThenCloseDrawer`：导航×1 + 站立铰链或轨道开合×1
- `HolodeckApproachDeskThenOpenJar`：导航×1 + 站立拿取与放置×1

#### 四类原子动作出现次数

标准 A：同一复合任务中，同一类别出现多次时只计一次。标准 B：保留同一复合任务中同一类别的重复出现次数。统计对象为当前列出的 43 个复合任务。

| 统计标准 | 导航 | 站立铰链或轨道开合类 | 站立拿取与放置类 | 需要弯腰或下蹲全身协调 | 合计 |
|---|---:|---:|---:|---:|---:|
| 标准 A（同类去重） | 37 | 16 | 28 | 11 | 92 |
| 标准 B（重复累计） | 48 | 17 | 36 | 13 | 114 |



# 汇总统计
## 已设计复合任务

下表各单元格按“标准 A / 标准 B”填写复合任务中对应类别的出现次数；标准 A 对同一复合任务内同类动作去重，标准 B 保留重复次数。

| 房间 | 导航 | 站立铰链或轨道开合 | 站立拿取与放置 | 需要弯腰或下蹲全身协调 |复合任务数|
|---|---:|---:|---:|---:|---|
| kitchen | 28 / 34 | 12 / 12 | 21 / 24 | 1 / 1 |36|
| living room | 20 / 26 | 6 / 7 | 11 / 13 | 9 / 9 |27|
| bedroom | 25 / 33 | 15 / 19 | 7 / 10 | 5 / 8 |32|
| bathroom | 31 / 43 | 12 / 16 | 15 / 18 | 6 / 10 |41|
| office | 37 / 48 | 16 / 17 | 28 / 36 | 11 / 13 |43|
| 合计 | 141 / 184 | 61 / 71 | 82 / 101 | 32 / 41 | 179 |

## 已采集复合任务
- 已采集列表（截至 9/18）：
IThorKitchenBendAndPutAppleOnPlate
IThorKitchenApproachOpenFridge
IThorKitchenApproachOpenOven
IThorKitchenOpenDrawerTakeApple
IThorKitchenDrawerPnP
IThorKitchenAppleCabinetPnP
IThorKitchenAppleBananaPlateSwap
IThorKitchenPackProduceIntoBox
IThorKitchenBananaBasketToTable
IThorKitchenCoffeeDrinkToTrash
IThorKitchenCupStack
IThorKitchenOvenThenFridge
IThorKitchenFridgeThenOven
IThorKitchenApproachTableAndLiftApple
IThorKitchenApproachTableAndLiftSaltShaker
IThorKitchenApproachTableAndLiftBowl
IThorKitchenApproachTableAndPutMugOnPlate
IThorKitchenApproachTableAndPutAppleOnPlate
IThorKitchenApproachTableAndPutBreadOnPlate
IThorKitchenApproachTableAndPutCheeseOnPlate
IThorKitchenApproachTableAndStandCannedDrinkUpright
IThorKitchenApproachSinkAndPutBowl
IThorKitchenApproachStoveAndPutPan
IThorKitchenApproachSinkAndPutCup
IThorKitchenApproachDrawerAndOpen
IThorKitchenApproachDrawerAndClose
IThorKitchenApproachCabinetAndOpen
IThorKitchenApproachCabinetAndClose
IThorKitchenApproachDrawerAndOpenClose
IThorKitchenApproachCabinetAndOpenClose
IThorKitchenPickAppleAndTurnToTrash
IThorKitchenOpenDrawerAndTurnAppleToTrash
IThorKitchenOpenCloseLowMicrowave
IThorKitchenCloseThenOpenLowMicrowave

IThorLivingRoomCoffeeDrinkToBin
IThorLivingRoomRemoteToBin
IThorLivingRoomBasketToSofa
IThorLivingRoomDumbbellToSofa
IThorLivingRoomStatueToSofa
IThorLivingRoomPackCandleStatueIntoBasket
IThorLivingRoomWalkAndOpenDrawer
IThorLivingRoomWalkAndCloseDrawer
IThorLivingRoomWalkAndOpenCloseDrawer
IThorLivingRoomWalkToTableAndPutCanInBasket
IThorLivingRoomApproachTableAndLiftDumbbell
IThorLivingRoomApproachTableAndLiftStatue
IThorLivingRoomApproachTableAndLiftWateringCan
IThorLivingRoomWalkAndOpenLowDrawer
IThorLivingRoomWalkAndCloseLowDrawer
IThorLivingRoomApproachTableAndPutRemoteInBasket
IThorLivingRoomApproachTableAndPutCandleInBasket
IThorLivingRoomApproachTableAndPutTissueBoxInBasket
IThorLivingRoomApproachTableAndPutBookInBasket
IThorLivingRoomLowDrawerPnP


IThorBedroomApproachDeskAndOpenLaptop
IThorBedroomApproachDeskAndCloseLaptop
IThorBedroomApproachDeskAndLiftBasket
IThorBedroomApproachDeskAndPutCanInBasket
IThorBedroomApproachDeskAndPutMugInBasket
IThorBedroomApproachDeskAndPutBoxedFoodInBasket
IThorBedroomApproachStorageAndOpenSafeDoor
IThorBedroomApproachStorageAndCloseSafeDoor
IThorBedroomApproachStorageAndOpenHamperLid
IThorBedroomApproachStorageAndCloseHamperLid
IThorBedroomApproachStorageAndOpenDrawer
IThorBedroomApproachStorageAndCloseDrawer
IThorBedroomApproachStorageAndOpenBoxFlap
IThorBedroomApproachStorageAndCloseBoxFlap
IThorBedroomOpenThenCloseDrawer
IThorBedroomOpenThenCloseTopDrawer
IThorBedroomOpenThenCloseSafeDoor
IThorBedroomOpenThenCloseHamperLid
IThorBedroomOpenThenCloseBoxFlap
IThorBedroomOpenThenCloseLaptop
IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket
IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop
IThorBedroomCloseLaptopThenApproachBedAndCloseTopDrawer
IThorBedroomApproachDeskAndCloseLaptopThenApproachBed
IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk
IThorBedroomPutCanThenBoxedFoodThenMugInBasket




IThorBathroomApproachToiletAndOpenToiletLid
IThorBathroomApproachToiletAndCloseToiletLid
IThorBathroomNavigateToBathingAreaAndOpenShowerDoor
IThorBathroomNavigateToBathingAreaAndCloseShowerDoor
IThorBathroomApproachSinkAndOpenCabinetDoor
IThorBathroomApproachSinkAndCloseCabinetDoor
IThorBathroomNavigateToBathingAreaAndLiftSoapBottle
IThorBathroomApproachSinkAndLiftSoapBottle
IThorBathroomApproachSinkAndLiftToiletPaperRoll
IThorBathroomApproachSinkAndLiftSprayBottle
IThorBathroomNavigateToBathingAreaAndLiftScrubBrush
IThorBathroomApproachSinkAndLiftScrubBrush
IThorBathroomNavigateToBathingAreaAndStandSoapBottleUpright
IThorBathroomNavigateToBathingAreaAndPutSpongeInBathtub
IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin
IThorBathroomOpenThenCloseToiletLid
IThorBathroomCloseThenOpenToiletLid
IThorBathroomOpenThenCloseShowerDoor
IThorBathroomCloseThenOpenShowerDoor
IThorBathroomOpenThenCloseCabinetDoor
IThorBathroomCloseThenOpenCabinetDoor
IThorBathroomOpenThenCloseLowerDrawer
IThorBathroomCloseThenOpenLowerDrawer
IThorBathroomNavigateToBathingAreaAndStandSoapBottleUprightThenPutSpongeInBathtub
IThorBathroomSoapToLowerShelfThenStandSoapBottleUprightAndPutSpongeInBathtub
IThorBathroomCloseToiletLidThenApproachSinkAndLiftSoapBottle
IThorBathroomApproachToiletAndOpenToiletLidThenLiftScrubBrush
IThorBathroomOpenUpperDrawerThenLiftToiletPaperRollAndCloseUpperDrawer


HolodeckApproachDeskThenOpenLaptop
HolodeckOpenDrawerThenTakeTissueBox
HolodeckOpenDrawerTakeTissueBoxToDesk
HolodeckSugarCubeThenCloseJar
HolodeckOpenJarAddSugarThenCloseJar
HolodeckApproachDeskThenCloseLaptopMoveCup
HolodeckApproachTrashCanThenThrowBottle
HolodeckApproachDeskThenIceCubeInGlass
HolodeckApproachDeskThenPickCup
HolodeckApproachDeskThenMoveCupToCenter
HolodeckApproachDeskThenCloseDrawer
HolodeckApproachDeskThenOpenJar
HolodeckRemoveFolderFromShelfToDesk
HolodeckIceCubeThenPlaceCup
HolodeckApproachDeskThenPullChair
HolodeckApproachWindowThenDeskThenPickCup
HolodeckApproachDeskIceCubePlaceCupThenPickCup
HolodeckApproachDeskThenCloseLaptopThenApproachTrashCan
HolodeckApproachTrashCanThenApproachDeskThenPullChair
HolodeckApproachTrashCanThenPickBoxFromFloor
HolodeckCloseLaptopThenApproachTrashCanThenPickBox
HolodeckApproachWindowThenPickBox
HolodeckIceCubeThenApproachWindowThenPickBox


### 统计情况

已采集任务按任务名去重。当前清单共 131 个任务：kitchen 34 个、living room 20 个、bedroom 26 个、bathroom 28 个、office 23 个。已设计复合任务共 179 个；其中 31 个为尚未采集的纯导航复合任务，因此仍未采集的非纯导航复合任务为 `179 - 131 - 31 = 17` 个。

标准 A 对同一复合任务内同类动作去重；标准 B 保留同一复合任务内同类动作的重复次数。

#### 标准 A

| 房间 | 导航 | 站立铰链或轨道开合 | 站立拿取与放置 | 需要弯腰或下蹲全身协调 | 复合任务数 |
|---|---:|---:|---:|---:|---:|
| kitchen | 24 | 8 | 21 | 7 | 34 |
| living room | 14 | 6 | 10 | 9 | 20 |
| bedroom | 19 | 15 | 7 | 5 | 26 |
| bathroom | 18 | 11 | 14 | 6 | 28 |
| office | 19 | 7 | 15 | 6 | 23 |
| 合计 | 94 | 47 | 67 | 33 | 131 |

#### 标准 B

| 房间 | 导航 | 站立铰链或轨道开合 | 站立拿取与放置 | 需要弯腰或下蹲全身协调 | 复合任务数 |
|---|---:|---:|---:|---:|---:|
| kitchen | 26 | 9 | 24 | 11 | 34 |
| living room | 14 | 7 | 11 | 9 | 20 |
| bedroom | 21 | 19 | 10 | 8 | 26 |
| bathroom | 18 | 15 | 17 | 10 | 28 |
| office | 22 | 7 | 21 | 6 | 23 |
| 合计 | 101 | 57 | 83 | 44 | 131 |
