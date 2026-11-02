# Scene 301：枕头立起与沙发雕像抬升

两个原子任务仅支持 `scene-id 301`。2026-10-18 按用户要求，将原沙发篮球抬升任务改为竖直石质人物雕像抬升；保留原有 301 默认沙发。枕头尺寸与位置按下文的 80% 等比缩放配置更新。

| 环境名 | 初态与布局 | 最终条件 |
|---|---|---|
| `IThorBedroomStandPillowAtHeadboard` | 枕头平放于床垫靠机器人一侧并贴近床头；中心 XY 为 `[-0.25, 1.6774]`，机器人 XY 为 `[0.30, 1.48]`，朝向床 | 枕面平行于真实床头，允许偏转 20°，同时接触床垫与床头，连续保持 1 秒 |
| `IThorBedroomLiftStatueFromSofa` | 石质人物雕像竖直置于默认单人沙发，中心 XY `[0.93, -0.75]`；机器人 XY `[0.93, -0.30]`，yaw `-π/2` | 雕像语义中心相对 reset 高度抬升至少 `0.30 m`，连续保持 1 秒；与抓握状态无关 |

两个任务均不检查机器人抓握、手部接触或松手状态。按实际 `control_freq` 换算 1 秒所需控制步数；每个控制步更新一次，条件中断立即清零，读取 `_check_success()` 不会推进计时。雕像抬升任务的“弯腰”是操作指令，不增加机器人躯干姿态的成功门槛。

## 2026-10-18：两个沙发任务改用雕像

- 两个正式入口改为 `IThorBedroomLiftStatueFromSofa`、`IThorBedroomOpenBoxThenStatueFromSofaIntoBox`；旧篮球类名不再注册。两个任务均复用客厅原子任务 `IThorLivingRoomLiftStatue` 的 `Stone_Statue_4.xml`、`LivingRoomStatueObject` 适配器及 `0.5` 倍尺寸；目标身份为 `FloorPlan301:spawned:task_sofa_statue`。客厅任务和共享资产均不修改。
- 源雕像中心 XY `[0.93, -0.75]`，四元数 `[√0.5, √0.5, 0, 0]`，绕 X 轴旋转 90°，使局部 Y 长轴竖直，站立于同一沙发座面。实际碰撞体范围约 `0.075 × 0.068 × 0.293 m`；底座横向跨度约 `7.5 cm`。两个任务只在各自加载副本中移除原篮球 `FloorPlan301:Basketball_1_7`，原生狗形雕像保留在场景原位，不修改场景 XML、metadata 或共享预处理。
- 抬升阈值、连续保持时长和不要求抓握的既有语义不变。复合任务的阶段名、语言指令与完整入箱边界改为雕像语义，详见复合任务说明。
- 下文 2026-10-15/16 的篮球质量、旧坐标和旧验证均为历史记录，不适用于当前雕像任务；当前配置以本节和任务代码为准。

## 资产与支撑

- 枕头为 `FloorPlan301:pillow_1_5`，床为 `FloorPlan301:Bed_1_3`。使用床的 `collision_1` 判定真实床头接触、`collision_2` 判定床垫接触，不用床中心距离替代“靠床头”。枕头局部 Y 轴为枕面法向；在此前局部 Y 轴 `0.54` 倍的薄枕头基础上，三个方向统一缩放至此前的 `80%`，相对原生资产的倍率为 `(0.8, 0.432, 0.8)`，实际碰撞尺寸约 `0.45660 × 0.06020 × 0.28341 m`；原子任务与 `IThorBedroomApproachBedStandPillowThenDesk` 共用同一缩放实现及 `[-0.25, 1.6774]` 初始位置。枕头仍按大件双手操作处理。
- 历史篮球配置（当前两个任务已替换）：篮球为 `FloorPlan301:Basketball_1_7`，保留原生直径 `0.24 m`，按双手托举大球处理，不缩为小型单手抓持物。2026-10-16 按用户要求，将篮球质量从约 `3.62 kg` 调低至约 `0.30 kg`；原子任务与对应开箱投球复合任务共用 `_LightweightBasketballMixin` 的质量设定。质量在任务模型编译前覆盖，转动惯量由 MuJoCo 根据球形碰撞体重新计算；源场景 XML、其他场景和客厅推篮球任务不改动。
- 默认替换：移除原蓝色软座 `FloorPlan301:Dog_Bed_2_1_19` 的场景实体，使用原生尺寸的 `Armchair_201_1` 单人沙发（带靠背和扶手），新语义 ID 为 `FloorPlan301:Armchair_201_1_19`、类别 `Sofa`。替换在场景预处理与语义注册时统一生效，适用于所有加载 301 的任务和可视化入口，无需额外开关；其他场景不变，下载的原始 XML/metadata 不改写。沙发固定于 `[0.93, -0.99, 0.461151517]`，朝房间内部，底部落地，向内让出墙面净空；座面碰撞体顶部约 `0.46379 m`，篮球 reset 中心高度约 `0.58296 m`。 沙发的支撑区域 site 设为透明并归入辅助标记组，保留语义用途，消除座面上的红色方块。
- 两条初始化路径均复用 `FreeFixturePlacementMixin` 的真实支撑接触探测，reset 返回时目标已有浅接触且速度清零。初始化没有随机扰动。
- 机器人根节点到源物中心水平距离：枕头约 `0.573 m`，篮球 `0.67 m`。源物均在家具近边；中心距离只作布局诊断，不能冒充完整抓取可达性证据。

## 2026-10-16：两个篮球任务共用轻量配置

- 改动范围：`IThorBedroomLiftBasketballFromSofa` 与 `IThorBedroomOpenBoxThenBasketballFromSofaIntoBox`，仅 Scene 301 的篮球质量；尺寸、摩擦、布局、阶段顺序和成功条件沿用各任务当前配置。覆盖在加载的模型副本上实施，其他任务和资产文件不受影响。
- 定向单元检查：`MUJOCO_GL=egl PYTHONPATH=. python -m pytest -q tests/test_ithor_bedroom_301_objects.py -k basketball_mass`，结果 `1 passed, 11 deselected`。验证两个入口应用相同质量、其他 XML 属性不变、源模型未被修改。
- 代表性 headless 检查：原子任务一次 `make` / `reset`，实际总质量 `0.30 kg`，主惯量三轴均为 `0.001728 kg·m²`；机器人零碰撞、完整脚印在 Floor、篮球接触沙发、初始不成功。固定机器人推进 150 物理步，篮球 XY 无漂移，Z 变化约 `+0.815 mm`，保持沙发接触。本地诊断结果 `PASS`；复合任务复用同一质量初始化路径，未另跑仿真。
- 语法编译及 `git diff --check` 通过。未运行 GUI、人工抓取、完整控制轨迹或数据采集；无已知自动验收阻塞。

## 启动

```bash
python -m robocasa.scripts.visualize_scene \
  --env IThorBedroomStandPillowAtHeadboard --scene-id 301 --onscreen

python -m robocasa.scripts.visualize_scene \
  --env IThorBedroomLiftStatueFromSofa --scene-id 301 --onscreen
```

以上为人工检查入口，本轮没有运行 GUI。

## 2026-10-16 枕头尺寸与位置调整检查结论

- 改动：仅 `IThorBedroomStandPillowAtHeadboard` 的 301 初始化沿枕头局部 Y 轴缩放 `0.54`，同步视觉与碰撞网格；枕头中心 XY 调整到 `[-0.25, 1.642]`，平放于床垫并贴近床头。相关复合任务显式保留原始 `[-0.25, 1.48]` 布局及原生厚度，成功谓词保持原样。
- Headless 历史验证：首次检查用保守语义包围盒核对厚度而失败，改用实际碰撞网格顶点后仅复查本任务一次并通过；实测碰撞尺寸 `0.57075 × 0.07525 × 0.35426 m`，床头间隙约 `1.39 mm`、床垫接触距离约 `0.062 mm`。机器人零碰撞、脚印在 Floor、初始谓词与成功均为假；固定机器人推进 30 物理步后水平漂移约 `1.27 mm`、Z 下降约 `2.21 mm`，仍有合法支撑。
- 定向测试：在 `tele_benchmark` 环境运行 `MUJOCO_GL=egl PYTHONPATH=. python -m pytest -q tests/test_ithor_bedroom_301_objects.py::test_pillow_requires_upright_mattress_and_headboard_contact tests/test_ithor_bedroom_301_composites.py::test_catalog_and_native_specs_are_isolated`：`2 passed`。修改模块的 `py_compile` 与 `git diff --check` 通过。
- 未验证项：GUI、人工抓取、完整操作轨迹及数据采集。
- 风险：无已知自动验证阻塞；未验证人工操作可达性。最终结论：可实现。

## 2026-10-15 历史任务设计检查结论

- 改动：301 默认将蓝色软座替换为原生尺寸单人沙发；同步碰撞、视觉与语义注册。篮球任务改用沙发座面和新站位，30 cm、连续 1 秒、与抓握无关的成功谓词保持不变。枕头任务自身规格和谓词不变，共享 301 的默认家具替换。
- 自动验证：篮球任务一次代表性 headless reset，发现墙面净空与机器人站位问题并修正后仅复查受影响路径；机器人零场景碰撞、脚印在 Floor、初始不成功，支撑接触有效。固定机器人推进 150 物理步后篮球无水平漂移，Z 变化约 +0.815 mm，并通过实际高度正例。
- 默认替换检查：定向测试验证 301 预处理默认移除原软座且仅加入一件沙发、资产路径有效、XML 与语义 ID 一致、其他家具和其他场景不变。固定家具使用几何距离复查地面支撑与墙面/家具净空，避免固定实体间不生成 contact 导致漏检。
- 自动检查命令：`MUJOCO_GL=egl PYTHONPATH=. python -m pytest -q tests/test_ithor_bedroom_301_objects.py`：11 passed（运行环境 `tele_benchmark`）。覆盖默认家具替换、碰撞净空、平放枕头、缺少床垫接触、穿透、篮球不足 30 cm、计时中断与 10/20/30 Hz 换算；改动模块的 `py_compile` 和 `git diff --check` 通过。
- 未验证项：未运行 GUI、人工抓取、完整手臂轨迹或数据采集；几何距离和稳定性检查不等同于完整机器人操作可达性验证。

已知限制：卧室原有复合任务数量与 `EXPECTED_TASK_MATRIX` 存在历史差异；本轮未修改旧场景的预期复合数量。

## 2026-10-18 任务设计检查结论

- 改动：仅 Scene 301 两个沙发任务改用客厅已有的竖直 `Stone_Statue_4`，共用 `0.5` 倍尺寸、适配器与任务内初始化；抬升和保持条件、机器人/纸箱摆位及顺序不变。
- 自动验证：两个 301 测试文件筛选 `-k 'statue or containment or future_goal or catalog or registration'`，10 passed；根节点局部坐标修正后仅复查 `-k statue_replacement`，1 passed。修改模块语法编译与 `git diff --check` 通过。
- Headless：初次分别检查原子与复合入口；资产转换的根节点偏移导致语义边界与真实碰撞体不一致，修正后只复查复合代表路径。`G1FloatingBody` 下 reset、零机器人碰撞、完整 Floor 脚印、初始失败、雕像竖直且浅接触沙发、30 物理步稳定及真实碰撞体完整入箱正例通过。当前语义中心约 `[0.93, -0.75, 0.609] m`，短时 Z 位移约 `+0.35 mm`。
- 隔离：其余 62 个卧室任务的规格与支持场景对比一致；客厅任务代码、复用适配器、客厅规格和雕像资产文件的前后哈希一致。原生狗形雕像未移除。
- 未验证项：GUI、人工抓取/完整投放轨迹、数据采集和 `G1` 操作可用性；headless 使用 `G1FloatingBody`，诊断放置不代表机器人完成操作。
- 风险：无已知自动验证阻塞；人工可达性仍未验证。最终结论：可实现。

## 2026-10-18 枕头整体缩放至 80% 检查结论

- 改动：301 原子枕头任务及对应的到床、立枕头、到书桌复合任务，共用此前薄枕头等比缩小至 `80%` 后的模型、碰撞体和靠床头布局。成功条件、机器人出生点和阶段顺序保持原样；缩放仅作用于这两个任务的场景副本。
- 历史自动验证：共享枕头初始化路径做一次代表性原子任务 `make/reset`，通过。实测碰撞尺寸 `0.456598 × 0.060200 × 0.283409 m`，床头间隙 `1.358 mm`、床垫距离 `0.538 mm`；机器人零碰撞、完整脚印在 Floor、初始不成功。固定机器人推进 30 物理步后，水平漂移约 `1.03 mm`、Z 下降约 `2.16 mm`，仍保持合法支撑。
- 定向验证：`tests/test_ithor_bedroom_301_composites.py::test_catalog_and_native_specs_are_isolated`、`::test_pillow_requires_bed_then_new_reorientation_then_desk` 及 `tests/test_ithor_bedroom_301_objects.py::test_pillow_requires_upright_mattress_and_headboard_contact`，`3 passed`；覆盖共用尺寸/缩放入口、规格隔离、复合顺序和接触谓词。修改模块的 `py_compile` 与 `git diff --check` 通过。
- 未验证项：复合任务未单独运行物理 reset；GUI、人工抓取、完整操作轨迹和数据采集未验证。
- 风险：无已知自动验证阻塞；人工操作可达性未验证。最终结论：可实现。
