# 所选房间复合任务设计清单

沿用原子任务设计文档所选的房间，共设计 12 个复合任务。任务内容由你逐项填写。

## 待设计任务

| 编号 | 房间 | Scene ID | 场景类型 | 任务 ID | 任务描述 | 成功条件 |
|---|---|---|---|---|---|---|
| KC01 | Kitchen | 3 | 新场景（已重新设计） | `IThorKitchenBreadOffPlateThenAppleOnPlate` | 将盘中的面包放到右侧柜台，再将盘子左边的苹果放进盘子 | 按顺序完成；面包保留在右侧柜台，苹果放在盘中并释放，保持 1 秒 |
| KC02 | Kitchen | 3 | 新场景 | `IThorKitchenCupFromStoolToSink` | 将凳子上的杯子拿起，转身走到水槽边，再将杯子放进水池 | 杯子中心进入水池并保持 1 秒 |
| KC03 | Kitchen | 8 | 新场景 | `IThorKitchenPanToRightBurnerThenEggIntoPan` | 将锅从左下角灶位移到右下角灶位，再将旁边的鸡蛋放进锅里 | 按顺序完成；锅与右下角灶位至少重合 70%，鸡蛋中心在锅内，保持 1 秒 |
| KC04 | Kitchen | 8 | 新场景（已重新设计） | `IThorKitchenCannedDrinkIntoFridgeThenCloseDoor` | 将面前的可乐放入已打开的下层冰箱，然后关门 | 可乐中心进入放宽后的下层冰箱空间并释放，允许可乐倾斜或跨分区；再将下层门关闭到 10% 以下，保持 1 秒；不判定机器人站位 |
| KC05 | Kitchen | 13 | 训练场景 | `IThorKitchenOpenDrawerThenPutCannedDrinkInside` | 打开抽屉，将柜台上的可乐放进抽屉，再关上抽屉 | 按顺序完成；可乐留在抽屉内，抽屉关闭到 20% 以下并保持 1 秒 |
| KC06 | Kitchen | 25 | 已有未训练场景（已重新设计） | `IThorKitchenCannedDrinkFromFridgeToTrash` | 从完全打开的冰箱中取出可乐，再将可乐放进垃圾桶 | 按顺序完成；可乐中心进入垃圾桶、放开可乐并保持 1 秒 |
| LC01 | Living Room | 201 | 新场景（已重新设计） | `IThorLivingRoomMugFromCoffeeTableToBoxOnSofa` | 拿起咖啡桌上的马克杯，走到沙发旁，将马克杯放进沙发上的敞开纸盒 | 马克杯完整位于纸盒内、已释放并保持 1 秒 |
| LC02 | Living Room | 201 | 新场景（已重新设计） | `IThorLivingRoomMugFromSofaToCoffeeTable` | 从沙发上拿起马克杯，走到咖啡桌旁，将马克杯放在咖啡桌上 | 马克杯离开沙发后释放并稳定放在咖啡桌上，保持 1 秒 |
| LC03 | Living Room | 205 | 新场景（已重新设计） | `IThorLivingRoomCloseTopRightThenTopLeftDrawer` | 先关闭电视柜右上抽屉，再走到左边关闭左上抽屉 | 必须按右上、左上的顺序关闭；两个抽屉均关闭到 10% 以下并保持 1 秒 |
| LC04 | Living Room | 205 | 新场景（已修改） | `IThorLivingRoomRemoteFromSideTableIntoBoxOnSofa` | 从电视旁的小圆桌上拿起竖直遥控器，走到长沙发旁，将遥控器放进沙发上的敞开纸箱 | 遥控器中心和下端进入放宽后的纸箱范围即可，允许竖直遥控器露出箱口；纸箱保持在长沙发上，连续保持 1 秒 |
| LC05 | Living Room | 209 | 训练场景 | `IThorLivingRoomStatueFromTableToTrashBin` | 从桌面靠垃圾桶一侧拿起雕像，走到垃圾桶旁，将雕像丢进垃圾桶 | 雕像释放后，中心进入略微放宽的垃圾桶开口范围并保持 1 秒；允许雕像露在桶外或搭在桶口 |
| LC06 | Living Room | 226 | 已有未训练场景 | `IThorLivingRoomCannedDrinkFromTableIntoBoxOnArmchair` | 从咖啡桌上拿起可乐，走到旁边的单人小沙发，将可乐放进座面上的敞开纸箱 | 可乐离开桌面并释放在纸箱内，纸箱保持位于单人小沙发上，连续保持 1 秒 |

## 设计注意事项

- 每个任务由两个或更多连续动作组成。
- 任务成功后需要连续维持 1 秒。
- 初始状态不能已经满足成功条件。
- 抓取物优先选择有效抓取跨度大于约 7 cm 的物体。
- 无需移动的任务中，机器人应站定后能够完成全部操作。
- 机器人和物体的初始化只在当前任务中生效，不改变同场景其他任务。
- 设计与实现时参考 [task_design_checklist.md](task_design_checklist.md)。

## KC01 初始化（已重新设计）

- 任务 ID 已由 `IThorKitchenBreadOffPlateThenPlateToSink` 更新为 `IThorKitchenBreadOffPlateThenAppleOnPlate`。
- 盘子与盘中的面包初始放在水槽右侧柜台，苹果初始放在机器人左手前方的柜台上，靠近机器人并避开正前方操作通道，与盘子留出足够间距。
- 先将面包放到右侧整块柜台，再将苹果放进盘子；机器人面向操作区站定，全程无需行走。
- 面包保留在右侧柜台、苹果与盘子接触且已释放，连续保持 1 秒即成功；不显示黄色判定区域。

## KC02 初始化

- 凳子初始位于机器人面前，杯子稳定放在凳面上。
- 机器人将杯子从凳面拿起，再转身走到水槽边；不要求抬高 30 cm。
- 杯子中心进入整个水池范围并连续保持 1 秒即成功；不显示黄色判定区域。

## KC03 初始化

- 锅初始位于左下角灶位，鸡蛋放在右下角灶位右下侧的柜台。
- 机器人靠近灶台并正对两个下方灶位站定，全程无需行走。
- 不显示黄色判定区域。

## KC04 初始化（已重新设计）

- 任务 ID 已由 `IThorKitchenEggIntoFridgeThenCloseDoor` 更新为 `IThorKitchenCannedDrinkIntoFridgeThenCloseDoor`。
- 机器人面向柜台站立，可乐直立位于机器人正前方。
- 上方冰箱门保持关闭，最下面的抽拉式冰箱门初始打开 100%。
- 将可乐放入打开的下层冰箱抽屉并释放；可乐允许倾斜或跨越内部小分区，再将下层门关闭到 10% 以下。
- 机器人可自行移动到冰箱旁，站位和移动过程不参与成功判定；全程不显示黄色判定区域。

## KC05 初始化

- 可乐初始直立放在机器人面前、距柜台边缘较远的位置，目标抽屉初始完全关闭。
- 机器人面向可乐与抽屉站定，全程无需行走。
- 先将抽屉打开到 80% 以上，将可乐完整放入并释放，再把抽屉关闭到 20% 以下。
- 全程不显示黄色判定区域。

## KC06 初始化（已重新设计）

- 任务 ID 已由 `IThorKitchenAppleFromFridgeToTrash` 更新为 `IThorKitchenCannedDrinkFromFridgeToTrash`。
- 使用其他任务中已经验证过、抓取直径约 7 cm 的 `can_0` 可乐；可乐初始直立放在完全打开的冰箱内部靠外位置。
- 仅当可乐中心真正离开冰箱容纳空间时才进入投放阶段。
- 原本位于垃圾桶前的凳子在初始化时移到一旁；移动凳子不属于任务步骤。
- 机器人初始站在已打开冰箱的前侧，直接面向靠外的可乐；取出可乐后，再走到垃圾桶旁完成投放。
- 可乐中心进入垃圾桶、双手放开并连续保持 1 秒即成功。
- 全程不显示黄色判定区域。

## LC01 初始化（已重新设计）

- 任务 ID 已由 `IThorLivingRoomBowlFromCoffeeTableToBoxOnSofa` 更新为 `IThorLivingRoomMugFromCoffeeTableToBoxOnSofa`。
- 仅在 LC01 中移除咖啡桌上的原生碗，将抓取跨度已标定为约 7.5 cm 的马克杯放到原碗位置；机器人初始站在咖啡桌旁并面向马克杯。
- 顶部完全敞开的纸盒初始稳定放在沙发座面靠前位置。
- 机器人拿起马克杯后走到沙发旁，将马克杯完整放进纸盒并释放；全程不显示黄色判定区域。

## LC02 初始化（已重新设计）

- 任务 ID 已由 `IThorLivingRoomBowlFromSofaToCoffeeTable` 更新为 `IThorLivingRoomMugFromSofaToCoffeeTable`。
- 仅在 LC02 中移除原生碗，将与 LC01 相同、抓取跨度约 7.5 cm 的马克杯稳定放在沙发座面靠前、避开两块座垫接缝的位置；机器人初始站在沙发旁并面向马克杯。
- 机器人拿起马克杯后转身走到咖啡桌旁，将马克杯放下并释放。
- 马克杯离开沙发后，稳定放在咖啡桌上并连续保持 1 秒即成功；全程不显示黄色判定区域。

## LC03 初始化（已重新设计）

- 任务 ID 已由 `IThorLivingRoomWineBottleFromDiningTableToSideTable` 更新为 `IThorLivingRoomCloseTopRightThenTopLeftDrawer`。
- 电视柜最上层的右、左两个抽屉初始完全打开，下层两个抽屉初始关闭。
- 机器人初始站在电视旁的小圆桌前，面向右上抽屉。
- 必须先将右上抽屉关闭到 10% 以下，并保持左上抽屉打开；再走到左边，将左上抽屉关闭到 10% 以下。两个抽屉均关闭并连续保持 1 秒即成功。

## LC04 初始化（已修改）

- 原生遥控器初始竖直放在电视旁的小圆桌中央，机器人初始站在小圆桌前并面向遥控器。
- 原生纸箱初始放在长沙发座面，四个箱盖完全打开。
- 机器人拿起遥控器后走到长沙发旁，将遥控器放进敞开的纸箱。
- 判定范围已放宽：遥控器中心和下端进入纸箱范围即可，允许竖直或倾斜放置且顶部露出箱口；纸箱保持接触长沙发，连续保持 1 秒即成功。全程不显示黄色判定区域。

## LC05 初始化

- 使用缩小后的 `Stone_Statue_4`；雕像初始直立放在桌面靠近垃圾桶的一侧，机器人初始站在桌子靠垃圾桶的一侧并面向雕像。
- 机器人拿起雕像后走到垃圾桶旁，将雕像丢进垃圾桶。
- 判定范围仅比垃圾桶原始开口略大：雕像释放后，中心进入该范围并连续保持 1 秒即成功，允许雕像露在桶外或横搭在桶口。
- 全程不显示黄色判定区域。

## LC06 初始化

- 新增一罐可乐，初始直立放在咖啡桌上；机器人初始站在桌旁并面向可乐。
- 原生纸箱移到旁边的单人小沙发座面中央，四个箱盖完全打开。
- 机器人拿起可乐后走到单人小沙发旁，将可乐放进敞开的纸箱。
- 可乐释放在纸箱内且纸箱保持接触单人小沙发，连续保持 1 秒即成功；全程不显示黄色判定区域。

## 鼠标施力测试

KC01：

```bash
conda activate tele_benchmark

python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenBreadOffPlateThenAppleOnPlate \
  --scene-id 3 \
  --scene-seed 0
```

KC02：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenCupFromStoolToSink \
  --scene-id 3 \
  --scene-seed 0
```

KC03：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenPanToRightBurnerThenEggIntoPan \
  --scene-id 8 \
  --scene-seed 0
```

KC04：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenCannedDrinkIntoFridgeThenCloseDoor \
  --scene-id 8 \
  --scene-seed 0
```

KC05：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenOpenDrawerThenPutCannedDrinkInside \
  --scene-id 13 \
  --scene-seed 0
```

KC06：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenCannedDrinkFromFridgeToTrash \
  --scene-id 25 \
  --scene-seed 0
```

LC01：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomMugFromCoffeeTableToBoxOnSofa \
  --scene-id 201 \
  --scene-seed 0
```

LC02：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomMugFromSofaToCoffeeTable \
  --scene-id 201 \
  --scene-seed 0
```

LC03：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomCloseTopRightThenTopLeftDrawer \
  --scene-id 205 \
  --scene-seed 0
```

LC04：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomRemoteFromSideTableIntoBoxOnSofa \
  --scene-id 205 \
  --scene-seed 0
```

LC05：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomStatueFromTableToTrashBin \
  --scene-id 209 \
  --scene-seed 0
```

LC06：

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomCannedDrinkFromTableIntoBoxOnArmchair \
  --scene-id 226 \
  --scene-seed 0
```

其余任务实现后，将 `--env` 和 `--scene-id` 改成表中对应值：

```bash
python -m robocasa.scripts.visualize_scene \
  --env <COMPOSITE_TASK_ID> \
  --scene-id <SCENE_ID> \
  --scene-seed 0
```

鼠标左键拖动物体或家具施力；完成任务并维持成功条件 1 秒后，终端应显示 `success`。

## 后续采集命令模板

```bash
export COLLECT_ENV=<COMPOSITE_TASK_ID>
export SCENE_ID=<SCENE_ID>

lab collect --sim \
  --env "$COLLECT_ENV" \
  --scene-id "$SCENE_ID" \
  --fixed-task \
  --scene-seed 0 \
  --backend sonic \
  --sim-conda-env kimodo_rl \
  --pico-source mock \
  --inference-backend onnx-gpu
```

开发、测试和输出都放在 `/path/to/humanoid_benchmark`，不要在采集目录切换分支或写入调试结果。
