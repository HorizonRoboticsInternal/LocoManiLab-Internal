# iTHOR Living Room：叠书后放 mug

环境：`IThorLivingRoomStackBooksAndPlaceMug`。支持 FloorPlan202、207、209、210、212、226。

## 布局与资产

- G1 面向工作面；左右按 G1 初始朝向定义，左书为 `base_book`，右书为 `moving_book`，mug 初始位于两书之间。两书平放，中心间距 0.38 m，不随机交换左右。
- 复用叠书原子任务的命名支撑家具和可见工作面，工作面高度 0.75 m。202 的本任务工作面半宽为 0.56 m，为三件物体留出空间；210 使用原有 Ottoman 上的工作面，其余使用茶几上的工作面。
- 两书复用现有 `Book_3` 刚体适配器和 0.65 scale；mug 复用 `ithor_atomic/task_assets.py` 的标准 mug 模型与 scale。没有修改共享资产或已有任务参数。
- 新任务 YAML 独立保存六场景标定；语言直接定义在任务类中。相比现有叠书原子任务，新增 mug 放置阶段及强制执行顺序，属于独立复合任务。

## 成功条件

1. 右书在左书上接触、基本对齐并松手，左书仍受工作面支撑，mug 尚未接触右书；连续保持 0.25 s 后进入下一阶段。
2. 保持书堆完整，将 mug 直立放在上层书上并松手。要求实际接触、mug 底部与书的实际碰撞顶面高度差不超过 1 cm、完整水平包围盒位于上层书内（5 mm 容差），连续保持 1.0 s。

最终时长按 `control_freq` 换算，20 Hz 为 20 个连续控制步。mug 条件中断即清零；书堆散开则回到第一阶段。先把 mug 放在右书上再整体叠书无法跳过第一阶段。

书的 authored bbox 顶面与实际碰撞顶面存在约 1.5 cm 偏差，因此新任务的 mug 高度条件使用碰撞几何计算。已有叠书任务保持原逻辑。

## 已执行验证

- `pytest -q tests/test_ithor_living_room_stack_books_and_mug.py`：9 项通过；覆盖六场景注册、初始左右关系与工作面范围、顺序、连续保持与清零、接触缺失、悬空、侧碰、边缘悬出、未松手、倒置等反例。
- FloorPlan202 代表性 headless `make/reset`：三件物体均有工作面接触，初始未成功，机器人无身体碰撞且脚底区域在地板上。固定机器人后的短时物理检查，通过受控设置物体位姿验证两书可堆叠，mug 可稳定受上层书支撑，并满足最终物理谓词。首次检查发现上述 bbox 高度偏差，修复后仅重跑该代表场景并通过。
- 语法编译与 `git diff --check`。

尚未进行 GUI/遥操作完整成功轨迹；其余五场景只做配置和几何布局检查，未逐场景运行 MuJoCo。

查看示例：

```bash
python -m robocasa.scripts.visualize_scene --env IThorLivingRoomStackBooksAndPlaceMug --scene-id 202 --onscreen
```
