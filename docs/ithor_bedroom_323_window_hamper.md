# Scene 323：先到床边窗户，再到衣篮并开盖

新增环境：`IThorBedroomApproachBedsideWindowThenOpenHamperLid`，仅支持 Scene 323，类别为 composite。

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorBedroomApproachBedsideWindowThenOpenHamperLid \
  --scene-id 323 --onscreen
```

语言指令：`walk to the window beside the bed, then walk to the laundry hamper and open its lid`。

按以下顺序完成三个阶段：

1. 到床边窗户旁，机器人与窗户碰撞几何的最短距离不超过 0.35 米，连续保持 5 个控制步。
2. 到洗衣篮旁，与洗衣篮碰撞几何的最短距离不超过 0.35 米，连续保持 5 个控制步。
3. 在衣篮旁打开指定篮盖，开度达到原子任务的 80% 阈值，连续保持 1 秒；保持步数按实际控制频率换算。

前两个阶段都要求篮盖保持 reset 开度，允许误差为总行程的 3%。提前开盖无法获得导航阶段完成记录，须先恢复 reset 开度再继续。导航条件中断清零当前停留计数；完成的导航阶段保留。最终开盖条件或衣篮距离条件中断时清零最终保持计数。读取成功状态不会推进阶段，同一实例 reset 会清空全部阶段。

## 复用与隔离

窗户目标为 `FloorPlan323:Window_89624b90_14`。复用 `IThorBedroomApproachBedsideWindow` 的出生 XY `[1.50, 1.35]`、朝向、最小出生距离 4 米、地面高度标定与 0.35 米距离阈值，不删除场景家具。

衣篮复用 `IThorBedroomOpenHamperLid` 的 Scene 323 规格以及其底层 mixin：同一资产、位置、缩放、地面支撑、基座 weld、关节物理、初始 30° 开度和 80% 目标阈值。精确目标为 `FloorPlan323:spawned:task_laundry_hamper`，关节为 `task_laundry_hamper_Laundry_Hamper_1_1_laundry_hamper_lid_1_joint_0`。沿用原子任务的近端操作位 `[0.15, -1.95]`。

新任务使用深拷贝规格和独立状态机。未修改原子任务、已有复合任务、共享初始化函数、YAML 或资产。注册目录的计数同步为当前实际任务数；这也修正了新增前已落后于公开 catalog 的场景计数，不改变任何任务行为。

## 任务设计检查结论（2026-10-15）

- 任务定义：[x] Scene 323，三阶段复合任务，最终产生可观察的指定篮盖开度变化。
- 重复与注册：[x] 新类只注册一次，包导出、支持场景和目录一致；卧室静态注册/物理签名审计通过，无重复任务。
- 物理可行性：[x] 复用精确窗户与篮盖 fixture/joint。诊断摆放得到窗边站位约 `[-0.596, -3.116]`，距离约 0.211 米；原子衣篮操作位距离约 0.157 米。两处站位无机器人碰撞，脚印在 Floor；初始开度到成功开度的 10 点关节扫掠未产生机器人碰撞。这些是静态几何证据，未执行连续行走或手部操纵。
- 初始化与成功：[x] G1 headless 构建/reset 可执行，初始不成功、碰撞数为 0、完整脚印在 Floor；出生距窗户约 4.85 米。默认初始化噪声会造成双脚毫米级高差：诊断中至少一脚接地，另一脚最大约 9 毫米间隙；未修改原有站姿或控制器。
- 复合顺序：[x] 正确顺序、先到衣篮、提前开盖、导航中断、最终状态中断、重复读取和同实例进度重置均有定向覆盖；同一 headless 实例通过诊断摆放依序触发三个阶段，最终需要完整 20 步（20 Hz）。
- 兼容性：[x] 对比本轮修改前的工作区快照，原有 61 个卧室任务的规格、支持场景、目标状态和阶段顺序签名保持一致；未实例化旧任务进行回归。
- 自动验证：[x] `MUJOCO_GL=egl python -m pytest -q tests/test_ithor_bedroom_window_hamper.py` 首轮 6 项通过；修正公开任务继承及目录统计后，以 `-k atomic_calibrations` 定向复测剩余 1 项通过。改动模块 `py_compile`、`git diff --check` 通过。headless 诊断的双脚同时贴地强断言不适用于带默认姿态噪声的初始化；排除这一额外断言后，接地、场景距离及三阶段物理谓词诊断通过。共进行 3 次 headless 构建/reset 检查，后续目标诊断均复用最后一个实例。
- 未验证项：[ ] GUI、人工完整行走及开盖、控制器动态平衡、数据采集未验证。未新增物品抓放或随机布局，相关检查 N/A。

未解决问题与风险：无已知任务逻辑或注册阻塞；静态目标摆放和关节扫掠不能替代真实遥操作可达性或动态平衡证据。

最终结论：可实现；自动验收范围如上，人工操作与数据采集未验证。
