# 房间任务设计清单

任务由你填写。每行对应一个任务，同一 Scene ID 的两行分别设计不同任务，共 12 个。

## 已完成任务

| 编号 | 房间 | Scene ID | 场景类型 | 任务 ID | 任务描述 | 成功条件 |
|---|---|---|---|---|---|---|
| K01 | Kitchen | 3 | 新场景 | `IThorKitchenEggIntoJar` | 将鸡蛋放到罐子中 | 鸡蛋在指定罐体内连续保持 1 秒即成功 |
| K02 | Kitchen | 3 | 新场景（已重新设计） | `IThorKitchenWineBottleOntoStool` | 将酒瓶放到凳子上 | 酒瓶稳定放在凳面并连续保持 1 秒 |
| K03 | Kitchen | 8 | 新场景 | `IThorKitchenSpongeIntoOpenDrawer` | 将海绵放进旁边预先打开的最上层抽屉 | 海绵完整位于指定抽屉内部，连续保持 1 秒 |
| K04 | Kitchen | 8 | 新场景 | `IThorKitchenPanLeftToRightBurner` | 将锅从左下角灶位移到右下角灶位 | 锅体与右前灶位水平投影重合至少 70%，连续保持 1 秒 |
| K05 | Kitchen | 13 | 训练场景（已重新设计） | `IThorKitchenCannedDrinkIntoPan` | 将可乐放进锅里 | 可乐位于锅内并连续保持 1 秒 |
| K06 | Kitchen | 25 | 已有未训练场景 | `IThorKitchenSaltShakerStoolToSinkCounter` | 将盐罐从机器人旁边的凳子上拿到水池左边柜台 | 盐罐完整放在指定柜台上，连续保持 1 秒 |
| L01 | Living Room | 201 | 新场景（已重新设计） | `IThorLivingRoomBoxFromChairToFloor` | 将椅子上的开口纸箱拿下来放到地上 | 纸箱接触地面、离手并放稳，连续保持 1 秒 |
| L02 | Living Room | 201 | 新场景 | `IThorLivingRoomLiftBowlFromCoffeeTable` | 站在茶几旁，将茶几上的碗举起来 | 碗相对初始高度抬升至少 30 cm，连续保持 1 秒 |
| L03 | Living Room | 205 | 新场景（已重新设计） | `IThorLivingRoomBoxFromSofaToFloor` | 将沙发上的纸箱放到地上 | 纸箱接触地面、离手并稳定，连续保持 1 秒 |
| L04 | Living Room | 205 | 新场景（已修改） | `IThorLivingRoomStatueChairToTable` | 将椅子上的雕像放到餐桌上 | 雕像完整放在餐桌面上，连续保持 1 秒 |
| L05 | Living Room | 209 | 训练场景（已修改） | `IThorLivingRoomStatueIntoBox` | 将桌上的雕像放进桌上的纸盒 | 雕像位于指定纸盒内部，连续保持 1 秒 |
| L06 | Living Room | 226 | 已有未训练场景（已修改） | `IThorLivingRoomRemoteIntoBox` | 将桌上的遥控器放进桌上的纸箱 | 遥控器位于指定纸箱内部，连续保持 1 秒 |

K06 已选择 25；L06 已选择 226。

## K01 初始化与判定

状态：**设计完成**（2026-10-15，用户确认鼠标施力放入后成功）。

- 任务 ID：`IThorKitchenEggIntoJar`，仅支持 Scene 3。
- 使用鸡蛋旁边的原生开口容器 `Pot_3_22` 和鸡蛋 `Egg_3_21`，保留两者原有 XY 位置，不使用远处的糖罐。
- 机器人出生位置 `[-0.9, -1.3185, 0.18]`，朝向 `yaw = π`（面对 -X）。沿台面方向位于鸡蛋与罐子中间；地面接触校准后自由基座 Z 约为 `0.1682`。出生到台面前沿水平距离约 43 cm，到鸡蛋中心约 72 cm。
- 原生鸡蛋在本任务内按约 `1.095` 倍缩放，使有效抓持跨度约 7.5 cm，并将长轴转为平行台面边缘；支撑高度自动贴合原台面。原始场景文件和其他任务不变。
- 判定：鸡蛋全部碰撞顶点位于罐体原生 receptacle 区域的并集内，连续保持 1 秒，中途离开则清零。不额外要求接触罐体、双手释放、罐体保持原位置或固定倾斜角度。
- 反例：停在罐外或鸡蛋未完整进入罐内不能成功；放在罐内稍偏离中心仍可成功。
- 验证：初始化阶段已检查原生物体 0.5 秒稳定性；本次修正单独验证偏心放入、罐外反例、1 秒保持与中断清零。最终出生零碰撞、Floor 脚印和初始不成功检查通过；本次 6 项定向测试通过，并复用真实场景模型验证偏心 2.5 cm 放入保持 1 秒成功、罐外失败。GUI 鼠标施力测试已由用户确认放入后成功；机器人实际抓放与正式采集尚未验证。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenEggIntoJar --scene-id 3 --scene-seed 0
```

## K02 初始化与判定

状态：**已重新设计**。原任务 `IThorKitchenPullStoolFromUnderTable` 已替换。

- 新任务 ID：`IThorKitchenWineBottleOntoStool`，仅支持 Scene 3。
- 初始时机器人面向右侧台面的原生酒瓶，凳子位于机器人右侧；酒瓶旁的原生杯子仅在本任务中移除。
- 任务：将酒瓶从台面拿起并放到凳子上。
- 成功：酒瓶位于凳面范围内、接触凳面且已经放稳，连续保持 1 秒。
- 当前状态：布局与判定已实现，等待 GUI 鼠标施力验证。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenWineBottleOntoStool --scene-id 3 --scene-seed 0
```

## K03 初始化与判定

状态：**设计完成**（2026-10-15，用户确认鼠标施力放入测试成功）。

- 任务 ID：`IThorKitchenSpongeIntoOpenDrawer`，仅支持 Scene 8。
- 使用原生海绵 `Dish_Sponge_1_25`，移到机器人正前方台面，中心 XY 为 `[0.05, -1.45]`，距机器人水平约 40 cm；长轴平行台面边缘，底部贴合真实台面。保留尺寸和物理参数，有效抓持跨度约 7.36 cm。
- 目标为海绵旁最上层抽屉 `Drawer_8c0ca37e_5`；初始打开 100%，滑动关节位置为 `0.245 m`，初始速度为零。
- 机器人出生位置 `[0.05, -1.05, 0.05]`，朝向 `yaw = -π/2`（正对水槽），移近操作台，打开的抽屉位于机器人右侧，高度自动校准到地面。
- 成功：海绵完整位于指定抽屉的原生内部区域，连续保持 1 秒，离开则清零。判定区域随抽屉移动；不增加开抽屉、关抽屉或松手步骤。
- 反例：海绵仍在台面、未完整放入、放入旁边其他抽屉，均不能成功。
- 验证：原有 5 项几何与计时测试、放入正例及错误抽屉反例仍适用；本轮 headless 检查确认 100% 开度、面向水槽且抽屉在右侧的近端站位零碰撞、脚印在 Floor 内、初始不成功；海绵前置约 40 cm，台面接触和 0.5 秒稳定性检查通过。鼠标施力测试已由用户确认成功；机器人实际抓放与正式采集尚未验证。K01、K02 实现和配置不变。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenSpongeIntoOpenDrawer --scene-id 8 --scene-seed 0
```

## K04 初始化与判定

状态：**设计完成**（2026-10-15，用户确认鼠标施力测试通过）。

- 任务 ID：`IThorKitchenPanLeftToRightBurner`，仅支持 Scene 8。
- 使用左前灶位上的原生锅 `Pan_8_32`，保留原始位置、尺寸和物理参数。左下角、右下角指面向灶台时的左前、右前灶位。
- 机器人站在两个前排灶位中间，面向灶台；出生位置 `[-1.245, -0.95, 0.05]`，朝向 `yaw = -π/2`，高度自动贴地。
- 成功：重合面积 ÷ 锅体水平投影面积达到 70%，连续保持 1 秒；低于 70% 则重新计时。锅体使用原生碰撞轮廓的凸包，目标为右前炉圈支撑区域。
- 不再要求中心距离 5 cm、锅底高度、接触或倾斜角度；不要求抬高 30 cm 或额外松手。
- 验证：5 项定向测试通过；真实场景验证 70% 重合保持 1 秒成功、69% 失败并清零，初始不成功。初始化与站位未改；用户已确认鼠标施力测试通过，机器人实际抓放尚未验证。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenPanLeftToRightBurner --scene-id 8 --scene-seed 0
```

## K05 初始化与判定

状态：**已重新设计**。原任务 `IThorKitchenPanStoveToLeftCounter` 已替换。

- 新任务 ID：`IThorKitchenCannedDrinkIntoPan`，仅支持 Scene 13。
- 锅保持在灶台原位；新增可乐放在锅右侧柜台的前沿内侧，机器人靠近柜台并保持面向灶台。
- 任务：将可乐拿起并放进锅里。
- 成功：可乐中心位于锅的原生内部区域，连续保持 1 秒；中途离开则重新计时。
- 当前状态：布局与判定已实现，等待 GUI 鼠标施力验证。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenCannedDrinkIntoPan --scene-id 13 --scene-seed 0
```

## K06 初始化与判定

状态：**设计完成**（2026-10-15，用户确认鼠标施力测试成功）。

- 任务 ID：`IThorKitchenSaltShakerStoolToSinkCounter`，仅支持 Scene 25。
- 原生凳子放在机器人右侧约 46 cm，原生盐罐稳放在凳面中央；机器人面向水池，出生位置 `[-1.0, 0.92, 0.05]`。机器人到柜台近边约 35 cm，固定底盘即可够到盐罐和目标柜台。
- 目标为面向水池时水池左边的柜台。成功条件：盐罐完整位于该柜台范围内，盐罐底部到达柜台高度，连续保持 1 秒；不要求额外松手或保持直立。
- 自动检查：4 项定向测试通过；真实场景初始化零碰撞、脚印在 Floor，凳子和盐罐稳定；柜台正例、悬空反例和 1 秒计时通过。鼠标施力测试已由用户确认成功；机器人实际抓放与采集未验证。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenSaltShakerStoolToSinkCounter --scene-id 25 --scene-seed 0
```

## L01 初始化与判定

状态：**已重新设计**。原任务 `IThorLivingRoomPullChairFromUnderTable` 已替换。

- 任务 ID：`IThorLivingRoomBoxFromChairToFloor`，仅支持 Scene 201。
- 原生椅子 `Chair_201_1_28` 初始从桌下向后拉出并旋转 180°，使椅背一侧朝向餐桌；新增一个顶部完全开口的纸箱，稳定放在椅面上，四周箱沿可直接抓握。
- 机器人站在椅子斜前方并靠近椅面操作，正对纸箱，同时留出腿部安全距离。
- 任务：将开口纸箱从椅子上拿下来，放到旁边地面。
- 成功：纸箱接触房间地面、机器人已经离手且纸箱放稳，连续保持 1 秒；中途离地、再次抓住或继续明显移动会重新计时。
- 当前状态：布局与判定已实现，等待 GUI 鼠标施力验证。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomBoxFromChairToFloor --scene-id 201 --scene-seed 0
```

## L02 初始化与判定

状态：**设计完成**（2026-10-16，用户确认移近机器人后鼠标施力测试通过）。

- 任务 ID：`IThorLivingRoomLiftBowlFromCoffeeTable`，仅支持 Scene 201。
- 使用原生茶几 `Coffee_Table_201_1_0` 上的原生碗 `Bowl_1_8`。机器人站在茶几左侧，出生位置 `[-2.85, 4.79, 0.05]`，朝向 `yaw = -0.0074`，面向碗且无需移动。
- 成功：碗相对本轮稳定初始高度抬升至少 30 cm，连续保持 1 秒；高度不足或中途落下会清零。不增加抓持、松手、碗姿态或手部接触条件。
- 自动检查：4 项定向测试通过；机器人移近后距碗中心约 40 cm，最终配置复核为出生零碰撞、脚印在 Floor 内且初始不成功；30 cm 抬升与 1 秒成功判定保持不变。GUI 鼠标施力测试已由用户确认通过；机器人实际操作与正式采集尚未验证。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomLiftBowlFromCoffeeTable --scene-id 201 --scene-seed 0
```

## L03 初始化与判定

状态：**已重新设计**。原任务 `IThorLivingRoomPushChairUnderTable` 和中间方案 `IThorLivingRoomBowlFromChairToTable` 已替换。

- 任务 ID：`IThorLivingRoomBoxFromSofaToFloor`，仅支持 Scene 205。
- 使用场景中的原生纸箱 `Box_5`，初始放在长沙发靠机器人一侧的坐垫上。
- 机器人站在沙发正前方并面向纸箱，已向沙发靠近 15 cm，可直接抓取；餐桌及桌上物体保持场景原状。
- 成功：纸箱接触地面、已经离手且基本静止，连续保持 1 秒；中途移动或离地则重新计时。
- 当前状态：布局与判定已实现，等待 GUI 鼠标施力验证。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomBoxFromSofaToFloor --scene-id 205 --scene-seed 0
```

## L04 初始化与判定

状态：**已修改**（2026-10-16：原生雕像替换为 `Stone_Statue_4`；等待重新进行 GUI 鼠标施力测试）。

- 任务 ID：`IThorLivingRoomStatueChairToTable`，仅支持 Scene 205。
- 使用任务资产 `Stone_Statue_4`（缩放 0.5），替换场景原生 `Waving_Statue_3_23`，并将其稳放在餐桌左侧原生椅子 `Chair_002_1_1` 的座面中央。椅子侧向紧贴桌子，扶手与左侧桌沿平行，并保留约 1 cm 防穿模间隙。
- 机器人出生位置 `[-2.55, 1.90, 0.05]`，位于桌子左上角与椅子前方形成的 90° 空间内，朝向 `yaw = -0.85`，可同时操作椅子上的雕像和最近的桌面，无需移动。
- 成功：雕像全部碰撞几何位于餐桌水平范围内，雕像底部到达桌面高度，连续保持 1 秒；中途离开则清零。不增加松手、直立或手部接触条件。
- `Stone_Statue_4` 缩放后的水平碰撞尺寸约 7.5 × 6.8 cm、高约 29.3 cm。自动检查确认雕像与椅面接触、机器人出生零碰撞且初始不成功；将雕像放到餐桌后成功判定成立。该任务 4 项定向测试通过；更换资产后需要重新进行 GUI 鼠标施力测试。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomStatueChairToTable --scene-id 205 --scene-seed 0
```

## L05 初始化与判定

状态：**已修改**（2026-10-16：原生雕像替换为 `Stone_Statue_4`；等待重新进行 GUI 鼠标施力测试）。

- 任务 ID：`IThorLivingRoomStatueIntoBox`，仅支持 Scene 209。
- 使用任务资产 `Stone_Statue_4`（缩放 0.5）替换场景原生 `Towel_Statue_1_22`，纸盒继续使用原生 `Box_9_20`。纸盒中心为 `[-2.60, -2.25]`，雕像中心为 `[-2.60, -2.75]`，两者沿咖啡桌靠机器人一侧并排放置，底部贴合真实桌面；纸盒四片盖片初始完全展开。
- 机器人出生位置 `[-3.10, -2.50, 0.05]`，朝向 `yaw = 0`，站在雕像与纸盒中间的桌前；更换雕像后向后调整 5 cm，避免与咖啡桌接触，仍可无需移动完成操作。
- 成功：雕像全部碰撞几何的水平投影位于纸盒内腔范围，且雕像底部进入盒底至盒口之间，连续保持 1 秒；中途离开则清零。不增加松手、直立或手部接触条件。
- 原生纸盒内腔约 26.4 × 20.3 cm，`Stone_Statue_4` 缩放后的水平碰撞尺寸约 7.5 × 6.8 cm，可以完整放入。自动检查确认机器人出生零碰撞且初始不成功，将雕像放入盒底后成功判定成立。该任务 4 项定向测试通过；更换资产后需要重新进行 GUI 鼠标施力测试。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomStatueIntoBox --scene-id 209 --scene-seed 0
```

## L06 初始化与判定

状态：**修改完成**（2026-10-16：遥控器改为长边朝上竖立，用户确认可用）。

- 任务 ID：`IThorLivingRoomRemoteIntoBox`，仅支持 Scene 226。
- 使用原生遥控器 `Remote_3_2` 和原生纸箱 `Box_26_0`。纸箱中心为 `[-0.90, -0.48]`，遥控器中心为 `[-0.35, -0.52]`，两者沿咖啡桌长边左右并排，底部贴合真实桌面；纸箱四片盖片初始完全展开。
- 遥控器初始长轴竖直朝上，按键面朝向机器人，便于从两侧夹住遥控器中段；放入纸箱时不限制最终朝向。
- 机器人出生位置 `[-0.70, -1.00, 0.05]`，朝向 `yaw = π/2`，站在遥控器与纸箱中间的桌前，到遥控器约 0.59 m、到纸箱约 0.56 m，无需移动。
- 成功：遥控器全部碰撞几何的水平投影位于纸箱内腔范围，且遥控器底部进入盒底至盒口之间，连续保持 1 秒；中途离开则清零。不增加松手、朝向或手部接触条件。
- 遥控器竖立后高约 21.9 cm，桌面占地约 6.8 × 2.2 cm。自动检查确认机器人出生零碰撞且初始不成功；遥控器模拟 100 步后位移约 0.08 mm并保持直立，放入纸箱后成功判定成立。该任务 2 项定向测试通过；用户已确认修改后的布局可用。

鼠标施力测试此任务：

```bash
conda activate tele_benchmark
python -m robocasa.scripts.visualize_scene \
  --env IThorLivingRoomRemoteIntoBox --scene-id 226 --scene-seed 0
```

## 任务设计注意事项

所有任务满足成功条件后，必须连续维持 **1 秒** 才算成功；中途不满足则重新计时。

| 动作 | 初始状态 → 成功状态 |
|---|---|
| 抓起 | 相对物品原高度抬高 **超过 30 cm** |
| 打开抽屉 | 开度 20% → 至少 80% |
| 关闭抽屉 | 开度 80% → 不超过 20% |
| 打开橱柜 | 开度 10% → 至少 50% |
| 关闭橱柜 | 开度 50% → 不超过 10% |
| 打开冰箱 | 开度 10% → 至少 50% |
| 关闭冰箱 | 开度 30% → 不超过 10% |
| 打开烤箱 | 开度 10% → 至少 50% |
| 关闭烤箱 | 开度 20% → 不超过 10% |
| 开门 | 0° → 至少 30° |
| 关门 | 30° → 0° |
| 打开电脑 | 开度 20% → 至少 90% |
| 合上电脑 | 开度 90% → 不超过 20% |

- 门任务：机器人在基本不需要移动的情况下可以完成。
- 抓起放下：物品实际抓取部位的直径大于约 **7 cm**，便于抓取。
- 抓起放下：机器人出生位置到桌面的水平距离约 **40 cm**，初始化不与桌子碰撞，不需要移动即可操作。

开发与调试均在 `/path/to/humanoid_benchmark` 下进行，运行使用 `--sim-conda-env kimodo_rl`，输出也写入该目录。

## 鼠标施力测试

后续查看场景默认使用鼠标施力模式，不加 `--probe`。以 Kitchen 3 为例：

```bash
conda activate tele_benchmark

python -m robocasa.scripts.visualize_scene \
  --env IThorKitchenEnv \
  --scene-id 3 \
  --scene-seed 0
```

先点击窗口使其获得焦点，再进行操作：

| 鼠标操作 | 功能 |
|---|---|
| 左键双击门板、抽屉或物体 | 选中要测试的部件 |
| 按住 Ctrl + 右键拖动 | 对选中部件施加拉力，测试拉门、拉抽屉、移动物体 |
| 按住 Ctrl + 左键拖动 | 对选中部件施加旋转力 |
| 左键拖动（不按 Ctrl） | 旋转视角 |
| 右键拖动（不按 Ctrl） | 平移视角 |
| 鼠标滚轮 | 缩放视角 |

门和抽屉按已有关节运动，固定家具不能自由拖动。鼠标施力用于探索物理交互，不代表机器人已经能完成任务；拖动后的状态不会自动保存为任务初始布局。

操作参考：[MuJoCo 官方交互说明](https://github.com/google-deepmind/mujoco/blob/main/doc/programming/samples.rst)。
