# Illusion USD 铰接资产 → RoboCasa Fixture 工作流

这份文档记录了把 **Illusion 风格的 USD 铰接资产**（带 `joint_info.yaml` 描述 PhysX 关节的二进制 USDC）转成 **robocasa 可用 Fixture** 并接入仿真环境的完整流程。

已经用 **`ElectricCooker001`**（来自 `Illusion_Assets/1d2a906e43eecc174d2b67e28c17c2b6/`）走通了端到端，下面所有路径、文件名、参数值都是这个已落地的工作示例；处理新资产时把它当模板复制改即可。

全程在 `tele_benchmark` conda 环境、仓库根目录 `/path/to/humanoid_benchmark` 下操作。

---

## 0. 总览

```
源资产                      转换/重写脚本 (scripts/usd_pipeline/)        接入仿真
USD/USDC + joint_info.yaml  ─▶ usd_to_mjcf_fixture.py        ─▶  Fixture 子类
(Y-up, cm, mesh + 关节)       └ 生成 model.xml + visuals/*.obj    └ scene_builder.FIXTURES
                              └ 默认每 mesh 一个凸包碰撞           └ fixture_registry/*.yaml
                                                                  └ 任务里 set_pos/set_euler
                            ─▶ set_box_collision.py box_configs/<Asset>.json
                              └ 用非互锁 box 替换凸包碰撞 (box 尺寸存 JSON)
                            ─▶ 手动调关节参数 (damping/frictionloss/inertial)
                            ─▶ fixture_joint_diag.py 验证
```

> **脚本都在 `scripts/usd_pipeline/`**：`usd_to_mjcf_fixture.py`（转换）、
> `set_box_collision.py`（通用 box 碰撞，读 `box_configs/<Asset>.json`）、
> `fixture_joint_diag.py`（诊断）。
> 每个资产的 box 尺寸是 `box_configs/<Asset>.json` 一个数据文件，不再每个资产复制一份脚本。

最关键的观察：**MuJoCo 不可能用一组凸包同时表达"闭合密封 + 开门无锁"**。原生 robocasa 的烤箱/微波炉用**手摆的 box 集合**做碰撞而不是网格——这也是这条流水线的核心手工步骤。

---

## 1. 环境依赖

`tele_benchmark` 里默认没装 USD 工具链。一次性装好（可逆，全 pip）：

```sh
conda activate tele_benchmark
pip install usd-core trimesh coacd
```

- `usd-core`：解析二进制 USDC（无 USD CLI 工具时唯一可行的方式）
- `trimesh`：加载/写 OBJ
- `coacd`：凸分解（备选方案，**不推荐**作为主流程，原因见 §3）

---

## 2. 源资产格式速查

每个 Illusion 资产包是一个独立目录，至少包含：

```
1d2a906e43eecc174d2b67e28c17c2b6/
├── instance.usd          # 二进制 USDC：mesh + 材质 + PhysX 关节
├── joint_info.yaml       # 关节人话描述（最关键的元信息）
├── metadata.json         # 材质/贴图清单
└── Materials/
    ├── *.mdl             # Omniverse MDL 材质（忽略）
    └── Textures/*.png    # 颜色/法线/金属度图（仅 BaseColor 会用上）
```

`joint_info.yaml` 形如：

```yaml
joints:
- name: /Root/Instance/joint_revolute_001
  type: PhysicsRevoluteJoint
  body0: /Root/Instance/Group_00      # 父刚体（base）
  body1: /Root/Instance/Group_200     # 子刚体（lid/door）
  axis: {x: 1.0, y: 0.0, z: 0.0}
  motion_range: {min: 0.0, max: 90.0}
```

USD 元信息：**Y-up、单位 cm**。一个 Xform Group 下挂若干 Mesh 子件；每个 Group 就是一个刚体。

---

## 3. Pipeline 步骤

下面每一步都给出命令、产出物、以及"如果出错该看哪"。

### Step 1 — USD → MJCF 自动转换

```sh
OUT=robocasa/models/assets/fixtures/articulated/<AssetName>
mkdir -p "$OUT"
python scripts/usd_pipeline/usd_to_mjcf_fixture.py \
    /path/to/Illusion_Assets/<asset_hash> \
    "$OUT" --name <semantic_name>
```

`scripts/usd_pipeline/usd_to_mjcf_fixture.py` 做的事：

1. 用 `pxr.Usd` 打开 USDC，读 `joint_info.yaml`
2. 每个 USD Mesh → 一个带 UV/法线的 OBJ，写到 `<OUT>/visuals/`
3. 解析每个 mesh 绑定的 Material：BaseColor 贴图存在就用贴图，否则用纯色
4. **坐标系**：Y-up cm → Z-up m（绕 X 转 -90°、所有点 × `metersPerUnit`）
5. 关节 pivot/axis 转到 MuJoCo 帧（同样的旋转与缩放）
6. 生成 `model.xml`：
   - 父 Group 的 mesh 放在 `<body name="object">`
   - 子 Group 放在 `<body name="<Group_path_tail>">`，里面一个 hinge `<joint>`
   - **每个 mesh 默认配一个 type=mesh 的凸包碰撞 geom**（class="collision"）
   - 全包围盒 `reg_main` + 缩到 55% 的 `reg_int`

**这一步基本完全自动**。验证：

```sh
python scripts/usd_pipeline/fixture_joint_diag.py "$OUT/model.xml"
```

可以编译并显示关节信息就 OK。但**这里的碰撞通常不能直接用**，往下走 Step 2。

#### Step 1b — 朝向：把"立正"焊进资产（`--extra-rot`）

RoboCasa 的约定是**"立正/朝向"焊进资产本身，`set_euler` 只留给摆放时的偏航（yaw）**：

- kitchen 家电（microwave/oven/fridge）：朝向直接在 OBJ 顶点里，根 body 无 euler/quat
- meshy 家具（CoffeeTable/Sofa）：`euler="1.5708 0 0"` 写在 visual geom 上
- robosuite 的 `set_euler` 只设根 body 的 euler，`rot` 属性只读 `euler[2]`（z 偏航）——**设计意图就是 yaw**

所以**不要**在 task 里用 `set_euler((±π/2,0,0))` 来"立正"（那是把立正和摆放混在一起，还会和真正的 yaw 冲突）。正确做法是转换时用 `--extra-rot-x/y/z`（度）把朝向焊进资产：

```sh
# 例：让 Asset124 的盖子朝上（绕 X 多转 -90°）
python scripts/usd_pipeline/usd_to_mjcf_fixture.py "$SRC" "$OUT" --name asset124 --extra-rot-x -90
```

转换器会把这个额外旋转施加到**顶点、法线、关节 pivot、关节 axis、region 盒**上，输出的资产天生就是目标朝向。这样：
- 关节/碰撞框/网格自动保持一致，不用手改
- task 里 `set_euler` / `z_rot` 完全留给"朝向房间哪个方向"

**怎么定 `--extra-rot` 的值**：先不带参数转换一次，用 `fixture_joint_diag.py` 看 pivot 在哪条边（顶/底/前），或渲染看一眼；想要的朝向和当前差多少就转多少。常见就是绕 X 转 ±90°。注意：改了 `--extra-rot` 后所有 AABB 都在新帧里，Step 2 的 box 坐标要按新帧重测。

### Step 2 — 碰撞重写（关键手工步骤）

**默认凸包碰撞几乎一定会卡铰链。** 原因：

- 一个 mesh = 一个凸包。凸包会填平 mesh 的凹陷
- 门（lid/panel）和本体在密封边缘几何上互相嵌套
- 凸包把这块凹腔填平后，门和本体在闭合处**深度重叠**（实测电饭煲闭合时门↔本体 53 对接触、最大穿透 4.2 cm），任何力都掰不开

走过的 3 种方案对照：

| 方案 | 闭合穿透 | 开门是否卡 | 备注 |
|---|---|---|---|
| 单凸包（自动） | -4.2 cm | 卡死，20 N·m 都开不动 | **不可用** |
| CoACD 凸分解（threshold=0.1, 165 块） | -1.1 cm（接触 53→6 沿开合下降） | 闭合处仍有几何互锁，需要 >20 N·m | 改善有限、文件膨胀 |
| **手摆 box（推荐）** | 0 | 全程 0–90° 0 接触 | 与原生 robocasa 同风格 |

推荐流程是**保留 visual mesh，删掉所有 mesh 碰撞 geom，按本体形状摆几个 box**。

#### 2a. 实空腔本体（电视/烤箱/微波炉等大块外壳）

用 `scripts/usd_pipeline/set_box_collision.py`（通用脚本）+ 一个 `box_configs/<Asset>.json`（每个资产的 box 尺寸数据，复制 `ElectricCooker001.json` 改）：

```sh
python scripts/usd_pipeline/set_box_collision.py box_configs/<Asset>.json
```

- 输入：JSON 里的 `asset`（资产目录）+ `body`/`door` 两组 box（local Z-up 米坐标）
- 操作：删除所有 `<mesh ... name="..._col"/>` asset 项 + 删除 base/door 上所有 `class="collision"` geom，按 JSON 添加 body+door box
- 注意点：
  - **本体 box 的前面（面向门那一侧）要内缩到铰链轴之后**，否则门一转就扫进 body box 卡死（电饭煲实测：本体 box 前面 y=-0.06 在铰链 y=-0.049 前方时，门转到 ~0.09 rad 就锁死；内缩到 y=-0.04 后全程 0 接触）
  - **门 box 不要伸到 pivot 之下**，免得勾住下方的 body
  - 门 box 厚度尽量贴 mesh 真实厚度

#### 2b. 空心容器（电饭煲那种"开门后要能往里放东西"）

整体壳体改成 **5 个壁的 box**（后壁/左壁/右壁/底壁/顶壁），前面留开口。`box_configs/ElectricCooker001.json` 里就是这么做的（5 块壁见 `body_back / body_left / body_right / body_floor / body_top`）。

关键尺寸（电饭煲坐标系，参考用）：

```
interior cavity local extents: x ±0.068, y[-0.04, +0.057], z ±0.083
back wall:  pos (0,    0.0805, 0)   size (0.123, 0.0235, 0.151)
left wall:  pos (-0.0955, 0.0085, 0) size (0.0275, 0.0485, 0.151)
right wall: pos (+0.0955, 0.0085, 0) size (0.0275, 0.0485, 0.151)
floor:      pos (0,    0.0085, -0.117) size (0.068, 0.0485, 0.034)
top:        pos (0,    0.0085, +0.117) size (0.068, 0.0485, 0.034)
```

新资产怎么定尺寸（半自动）：

```sh
# 用 trimesh 算每组 Group 的 AABB
python -c "
import trimesh, numpy as np
for n in ['Component_X','Component_Y']:  # 按 group 列出
    m = trimesh.load(f'visuals/{n}.obj', force='mesh')
    print(n, 'min', m.vertices.min(0).round(3), 'max', m.vertices.max(0).round(3))
"
```

拿这些范围拼壁体，再用 `fixture_joint_diag.py` 校验（见 §4）。

### Step 3 — 关节参数调优

转换脚本默认给关节 `damping=1`、`frictionloss=1`、`stiffness=0.05`、`armature=0.01`（对齐 oven 风格）。**这套默认值在 Step 2 把碰撞改对之前都不会立刻奏效**，所以先 Step 2、再这一步。

常见的几个手工调整：

#### 3a. 关节轴方向

`usd_to_mjcf_fixture.py` 可能算出 `axis="-1 0 0"`，但门会向**本体内部**扫（开角越大穿透越大）。诊断脚本会标 `[WARN] contacts grow with positive angle`。修法：把 axis 改回正向：

```xml
<joint axis="1.000000 0.000000 0.000000" .../>
```

实际判断：开 30° 时门↔本体接触数 > 闭合时，几乎一定是反了。

#### 3b. 质量 / 惯性

转换出来的 mesh 假设实心 → 一片薄板算成几公斤，重力力矩巨大，鼠标拖不动。给铰接子体加显式 `<inertial>`：

```xml
<body name="Group_200">
  <joint .../>
  <inertial pos="0.0 -0.082 0.0105" mass="0.5"
            diaginertia="0.0029 0.0049 0.0021"/>
  ...
</body>
```

经验值：薄门面板 0.3–0.5 kg、小盖子 0.1–0.3 kg。CoM 放在 mesh AABB 中心，diaginertia 用 `(1/12) m (b²+c²)` 等长方体公式估算即可。

#### 3c. "放哪停哪" 手感（重要）

光设 `frictionloss` 不够——MuJoCo 的 dof 摩擦约束默认是软的，约束力达不到 `frictionloss` 值，门在重力下会缓慢漂移。**必须开 noslip 迭代**才会真正饱和：

```xml
<option timestep="0.002" gravity="0.0 0.0 -9.81" noslip_iterations="10"/>
```

并设 `frictionloss > |gravity_max|`（用诊断脚本测）。电饭煲实际值：`damping=0.1 frictionloss=0.7`，门 0.5 kg、最大重力力矩 0.66 N·m。

#### 3d. ⚠️ noslip_iterations 在场景里会被丢

robosuite 合并场景模型时**丢掉 fixture 的 `<option>` 块**，只保留 arena 的 option。所以光在 model.xml 写 noslip 不够，在场景里仍然漂移。

修法：在用到这个 fixture 的任务里加一个 `_setup_scene` 覆写（参考 `robocasa/environments/living_room/tasks/tidy_living_room.py`）：

```python
def _setup_scene(self):
    super()._setup_scene()
    self.sim.model.opt.noslip_iterations = max(
        int(self.sim.model.opt.noslip_iterations), 10
    )
```

或者，如果整套 fixture 都需要它，可以塞到 `LivingRoomEnv._setup_scene` / `Kitchen._setup_scene` 一次性解决。

### Step 4 — 验证

```sh
python scripts/usd_pipeline/fixture_joint_diag.py "$OUT/model.xml"
```

诊断脚本逐铰接关节打印：

1. 轴/枢轴/范围/damping/frictionloss
2. **child↔parent 接触数 vs 角度**：穿透 < 5mm 即合格；若数量随角度增长 → 轴反了
3. **重力力矩 profile**：和 frictionloss 对比，告知能否静态保持
4. **stays-where-placed**：5 个角度位置静置 8 秒，drift 应该 = 0
5. **手感**：拖开/拖关需要的最小力矩

电饭煲的输出对照（应该长这样）：

```
contacts vs angle: 全 0
gravity max |0.66|  vs frictionloss 0.70  →  [OK]
stays-where-placed drift +0.000 全部
open ~0.2 N·m   close ~1.3 N·m
```

### Step 5 — 注册成 Fixture 类

四处改动（参考电饭煲的提交记录）：

1. **写一个 Fixture 子类**（如果通用单铰接的就直接复用 `ArticulatedObject`）：
   `robocasa/models/fixtures/articulated.py` 里的 `ArticulatedObject` 是一个最小通用类，自动发现所有 hinge 当作 door joint，提供 `is_open / is_closed / open_door / close_door`。

2. **暴露类名**：在 `robocasa/models/fixtures/__init__.py` 加一行 `from ... import ArticulatedObject`

3. **加入场景构建注册表**：`robocasa/models/scenes/scene_builder.py` 的 `FIXTURES = dict(...)` 里加 `<type_name>=ArticulatedObject`

4. **写 fixture_registry yaml**：`robocasa/models/assets/fixtures/fixture_registry/<type_name>.yaml`，把 xml 路径列出来：

```yaml
default:
  xml: fixtures/articulated/<AssetName>
<AssetName>:
  xml: fixtures/articulated/<AssetName>
```

### Step 6 — 接入任务场景

任务里在 `_setup_model` 钩子里实例化并摆位（电饭煲示例：`robocasa/environments/living_room/tasks/tidy_living_room.py`）：

```python
def _setup_model(self):
    super()._setup_model()
    ct = self.furniture["coffee_table"]
    top_z = ct.pos[2] + ct._regions["bbox"]["pz"][2]

    fxt = ArticulatedObject(name="<instance_name>", rng=self.rng)
    # 资产可能需要绕 X 转 90° 才直立（USD Y-up 残留），实测后定
    fxt.set_euler((-1.5708, 0.0, 0.0))
    half_height = fxt._regions["main"]["py"][1]
    fxt.set_pos((ct.pos[0], ct.pos[1], top_z + half_height))

    self.furniture["<instance_name>"] = fxt
    self.fixtures = self.furniture
    self.<instance_name> = fxt

def _setup_scene(self):
    super()._setup_scene()
    self.sim.model.opt.noslip_iterations = max(
        int(self.sim.model.opt.noslip_iterations), 10
    )
```

调试时 headless 跑通一次：

```sh
python -m robocasa.demos.demo_empty_teleop --env <TaskName> --check
```

`--check` 会跑 300 步不开窗口，主要捕捉 model 编译错误、碰撞数组越界、Python 异常。

可视化拖拽测试：

```sh
python -m robocasa.demos.demo_empty_teleop --env <TaskName>
```

**注意**：默认 PandaOmron 走的 `collect_human_trajectory` 在用户第一次按动作键之前**根本不调 `env.step`**，所以鼠标 Ctrl+拖拽看起来"没反应"。先随便按一下方向键，让循环开始 step，之后再拖。G1+SONIC 没有这个限制（无条件每帧 step）。

---

## 4. 经常踩的坑（按出现频率）

| 现象 | 真因 | 解 |
|---|---|---|
| 门掰不动、20 N·m 都不行 | mesh 凸包碰撞 base↔lid 互嵌 | Step 2：换 box 碰撞 |
| 开门时门"扎进"本体（穿模越来越深） | joint axis 符号反了 | model.xml 改 `axis="1 0 0"`（或反过来） |
| 关节自由摆动、放哪都不停 | frictionloss 没生效 | 加 `<option noslip_iterations="10"/>` **并** 在任务里手动设 sim.model.opt.noslip_iterations |
| 装好的 fixture 在场景里漂 | 同上 — robosuite 合并时丢了 fixture 的 `<option>` | 在任务的 `_setup_scene` 里写 sim.model.opt |
| 鼠标 Ctrl+拖拽 PandaOmron 没反应、G1 OK | `collect_human_trajectory` 等首次输入才 step | 先按一下方向键唤醒；或换无条件 step 的循环 |
| 关节稳但开门极费力 | mesh 实心估算出几公斤门，重力力矩 >2 N·m | 加 `<inertial mass=0.3 .../>` |
| 资产侧躺/倒立、盖子朝向不对 | 资产建模方向与默认 Y-up→Z-up 不一致 | 转换时加 `--extra-rot-x/y/z`（度）把朝向焊进资产，**别**用 set_euler 立正（见 Step 1b） |
| 门关闭后能进物体 | body 碰撞前壁没补 | 见 §2a，本体 box 前面如果内缩，2 cm 缝可以不补；介意就再加个薄 box |
| 苹果生成进电饭煲里 | 任务里的物体采样器不知道有这个 fixture | 把物体放到别的家具上（e.g. sofa）；或缩小采样区域 |
| 转换时 MuJoCo 报 `qhull error` / `mesh volume is too small` | 资产里有完全平面的 mesh（某轴 extent=0） | 已自动处理：converter 给这种 mesh 加 `inertia="shell"` 并跳过它的碰撞 geom |

---

## 5. 文件清单（电饭煲示例）

按本流水线产出 / 修改的文件：

```
scripts/usd_pipeline/
  usd_to_mjcf_fixture.py            # Step 1 通用转换器
  set_box_collision.py             # Step 2 通用 box 碰撞（读 box_configs/<Asset>.json）
  box_configs/<Asset>.json          # Step 2 每个资产的 box 尺寸（数据，非脚本）
  fixture_joint_diag.py             # Step 4 通用诊断工具
  
robocasa/models/assets/fixtures/articulated/ElectricCooker001/
  model.xml                          # 最终的 MJCF
  model.xml.singlehull.bak           # 自动转换出来的原始版（可参考）
  visuals/*.obj  *.png               # 12 个可视 mesh + 2 张贴图
  visuals/collision/                 # CoACD 产出（box 方案后已无用）

robocasa/models/fixtures/articulated.py
  └ ArticulatedObject 通用类
robocasa/models/fixtures/__init__.py
  └ 导出 ArticulatedObject
robocasa/models/scenes/scene_builder.py
  └ FIXTURES 里加 electric_cooker=ArticulatedObject
robocasa/models/assets/fixtures/fixture_registry/electric_cooker.yaml
  └ 注册 xml 路径

robocasa/environments/living_room/tasks/tidy_living_room.py
  └ _setup_model 实例化 + _setup_scene 设 noslip_iterations
```

---

## 6. 新资产快速复制单（cheat-sheet）

```sh
# 0. 环境
conda activate tele_benchmark

# 1. 转换（先不带朝向参数转一次看朝向）
SRC=/path/to/Illusion_Assets/<hash>
OUT=robocasa/models/assets/fixtures/articulated/<AssetName>
mkdir -p "$OUT"
python scripts/usd_pipeline/usd_to_mjcf_fixture.py "$SRC" "$OUT" --name <semantic>
#   朝向不对就加 --extra-rot-x/y/z（度）重转，把"立正"焊进资产（见 Step 1b）
#   例：python scripts/usd_pipeline/usd_to_mjcf_fixture.py "$SRC" "$OUT" --name <semantic> --extra-rot-x -90

# 2. 诊断（看 axis/碰撞穿透问题）
python scripts/usd_pipeline/fixture_joint_diag.py "$OUT/model.xml"

# 3. 改碰撞：复制 box_configs/ElectricCooker001.json 成 box_configs/<Asset>.json
#    用 trimesh 算每组 mesh 的 AABB，填 JSON 里的 body/door box 位置/尺寸
#    cp "$OUT/model.xml" "$OUT/model.xml.singlehull.bak"
python scripts/usd_pipeline/set_box_collision.py box_configs/<Asset>.json

# 4. 再诊断（看 stays-where-placed/手感）
python scripts/usd_pipeline/fixture_joint_diag.py "$OUT/model.xml"
# 根据输出微调 model.xml 里的 axis / damping / frictionloss / <inertial>

# 5. 注册（4 处改动，见 §Step 5）

# 6. 任务里用
# - _setup_model: ArticulatedObject(name=...) + set_pos
#   （朝向已焊进资产，set_euler 只在需要改房间朝向/yaw 时用）
# - _setup_scene: sim.model.opt.noslip_iterations = max(..., 10)

# 7. 验
python -m robocasa.demos.demo_empty_teleop --env <Task> --check
python -m robocasa.demos.demo_empty_teleop --env <Task>   # 按一下方向键再 Ctrl+拖
```
