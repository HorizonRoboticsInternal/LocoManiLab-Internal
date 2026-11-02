# 客厅家具资产转换 & 场景搭建工作流

这份文档记录了把 **Meshy AI 生成的网格** 转成 **robocasa 可用家具资产**、再 **摆进客厅场景** 的完整流程，方便以后自己复现。

全程在 `tele_benchmark` conda 环境、仓库根目录 `/path/to/humanoid_benchmark` 下操作。

---

## 0. 总览（一张图看懂）

```
Meshy AI 生成               转换脚本                         接入场景
(.obj+.mtl+.png)  ──▶  convert_meshy_furniture.py  ──▶  living_room.py
                       └ 写出 robocasa 资产目录          └ FurnitureObject(...)
                         + model.xml + preview.png         摆进 LivingRoomBasic
                                                       ──▶  跑 demo 查看
```

涉及的文件：

| 角色 | 路径 |
|---|---|
| 转换脚本 | `scripts/convert_meshy_furniture.py` |
| 资产输出目录（**已 gitignore，不入库**） | `robocasa/models/assets/furniture/<类别>/<实例>/` |
| 家具 Python 包装类 | `robocasa/models/furniture/furniture.py`（`FurnitureObject`） |
| 场景定义 | `robocasa/environments/living_room/living_room.py`（`LivingRoomBasic`） |
| 查看入口 | `robocasa/demos/demo_empty_teleop.py` |

---

## 1. 在 Meshy 生成资产

在 Meshy AI 里生成单件家具，导出 **OBJ with texture**，会得到一个目录，里面三个文件：

```
Meshy_AI_xxx_texture.obj   # 网格（顶点 + UV + 面）
Meshy_AI_xxx_texture.mtl   # 材质（引用贴图）
Meshy_AI_xxx_texture.png   # 贴图（UV atlas）
```

经验：
- **一次只生成一件家具**（沙发 / 茶几 / 电视柜…）。整间房会融合成一个不可拆分的网格。
- 桌上的摆件（书、杯子、盆栽）会和家具**焊死在同一网格**，无法单独抓取，只能当装饰。
- Meshy 输出会被**归一化到 [-1, 1]**（不是真实米制），且是 **Y-up**（Blender 约定）。这两点转换脚本都会处理。

---

## 2. 用脚本转换

```sh
conda activate tele_benchmark
cd /path/to/humanoid_benchmark

python scripts/convert_meshy_furniture.py \
    --src "/path/to/Downloads/<Meshy导出目录>" \
    --category sofas --name Sofa001 \
    --target-size 2.0 --render
```

### 脚本做了什么（对应早先的手动步骤）

1. **找文件**：从 `--src` 目录自动找 `.obj/.mtl/.png`。
2. **ASCII 重命名 + 修引用**：拷到 `<资产>/visual/` 下改名 `model.obj` / `model.mtl` / `texture.png`，并修正 obj 的 `mtllib`、mtl 的 `map_Kd` 引用（MuJoCo 不认中文文件名）。
3. **算包围盒(AABB)**：扫描 obj 顶点得到各轴跨度，用于定 scale 和 `reg_bbox`。
4. **Y-up → Z-up**：MuJoCo 是 Z 朝上，给可见 geom 加 `euler="1.5708 0 0"`（绕 X 转 +90°）。
5. **重新居中**：把网格包围盒中心平移到 body 原点。
6. **缩放**：按 `--target-size`（最长边的真实米数）或 `--scale` 计算并写入 mesh `scale`。
7. **写 `model.xml`**：可见网格 + 贴图材质 + `reg_bbox` region + 近似碰撞盒。
8. **`--render`**：渲一张 `preview.png` 验证图（放在资产目录里）。

### 常用参数

| 参数 | 说明 |
|---|---|
| `--src` | Meshy 导出目录（自动找 obj/mtl/png） |
| `--obj` / `--texture` / `--mtl` | 显式指定单个文件（覆盖 `--src` 的自动查找） |
| `--category` / `--name` | 输出到 `furniture/<category>/<name>/`，如 `sofas/Sofa001` |
| `--target-size` | **设最长边的真实米数**（Meshy 归一化资产用这个最省事） |
| `--scale` | 或直接给缩放系数（与 `--target-size` 二选一） |
| `--up` | 源朝向：`y`（Meshy/Blender 默认）/ `z` |
| `--collision` | `box`（整体包围盒，默认）/ `none`（纯视觉无碰撞） |
| `--render` | 输出 `preview.png` |
| `--force` | 覆盖已存在的资产目录 |

### 怎么选 `--target-size`

按家具最长边的现实尺寸（米）填：

| 家具 | 最长边 ≈ |
|---|---|
| 双人沙发 | 1.8 – 2.2 |
| 茶几 | 1.0 – 1.2 |
| 电视柜 | 1.4 – 1.8 |
| 单人扶手椅 | 0.8 – 0.9 |
| 书架 | 0.8 – 1.2 |

脚本输出末尾会打印**最终尺寸**和**放置高度 z**，对照一下是否合理。

### 脚本输出示例（沙发）

```
[aabb] source spans (x,y,z) = (2.000, 1.105, 1.130)
[scale] target-size 2.0 m -> scale 1.0000
[done] final extents (m): 2.000 x 1.130 x 1.105
[done] place the body at z=0.552 to rest it on the floor.   <-- 记下这个 z
[use ] FurnitureObject(name='sofa001', mjcf_path=furniture_mjcf_path('sofas', 'Sofa001'), pos=(x, y, 0.552))
```

---

## 3. 验证转换结果

打开资产目录里的 `preview.png`：

```
robocasa/models/assets/furniture/<category>/<name>/preview.png
```

对照旁边的 **0.5m 蓝色参考立方体** 看尺度，确认：
- 朝向对不对（家具正立、腿朝下）？不对就调 `--up`。
- 尺寸合不合理？不对就调 `--target-size` 重跑（加 `--force`）。

---

## 4. 摆进客厅场景

编辑 `robocasa/environments/living_room/living_room.py`，在 `LivingRoomBasic._get_furniture()` 里加一个 `FurnitureObject`：

```python
from robocasa.models.furniture import FurnitureObject, furniture_mjcf_path

furniture.append(
    FurnitureObject(
        name="sofa",
        mjcf_path=furniture_mjcf_path("sofas", "Sofa001"),
        pos=(-2.2, 0.0, 0.552),     # z = 脚本打印的放置高度
        euler=(0.0, 0.0, 1.5708),   # 绕 Z 旋转摆朝向（见下）
    )
)
```

### 坐标系与布局约定

- 地板是 `FloorArena`，x、y 范围约 **[-3, 3]**，地面在 **z = 0**，四周有碰撞墙。
- `pos` 是物体**中心**的世界坐标。`z` 用脚本打印的放置高度（= 半高），脚就落在地面。
- `euler`（弧度）绕 body 的 Z 轴旋转摆朝向。网格的“正面”一般沿其**深度轴**朝某个方向，旋转 90° 的倍数对齐到房间。

当前客厅布局（沿 x 轴一条线）：

```
  -x 墙                                              +x 墙
  [沙发] ──朝+x──▶   [茶几]      ◀──朝-x── [电视柜]
  x=-2.2           x=-0.8                  x=+2.3
```

- 沙发：`euler=(0,0, 1.5708)` → 宽边沿 y、正面朝 +x。
- 电视柜：`euler=(0,0,-1.5708)` → 宽边沿 y、电视屏朝 -x（对着沙发）。
- 茶几：无旋转，居中。

**朝向调不对怎么办**：旋转就 0 / ±1.5708 / 3.1416 四个值挨个试，渲染一眼就能看出来（见第 5 步）。

### FurnitureObject 说明

`robocasa/models/furniture/furniture.py` 里的 `FurnitureObject`：
- 继承 `MJCFObject`，复用其 model.xml 加载、缩放、`reg_*` region 解析。
- 构造后**删掉自动注入的 free joint** → **静态焊死**（障碍 / 支撑面，不会被推走或掉落）。
- 参数：`name`、`mjcf_path`、`pos`、`euler`、`scale`（额外缩放，一般不用）。

---

## 5. 查看 / 验证场景

**带显示器**（交互遥操作，能走动）：

```sh
python -m robocasa.demos.demo_empty_teleop --env LivingRoomBasic                    # PandaOmron 键盘
python -m robocasa.demos.demo_empty_teleop --env LivingRoomBasic --robot G1FloatingBody   # G1+SONIC
```
窗口里按 `Tab` 切相机，鼠标拖拽看细节。

**无显示器**（只确认能构建、不报错）：

```sh
python -m robocasa.demos.demo_empty_teleop --env LivingRoomBasic --check
```

**离屏渲染存图**（自己写脚本看布局，可参考下面片段）：

```python
import numpy as np, robosuite, robocasa, imageio.v2 as imageio, mujoco
from robosuite.controllers import load_composite_controller_config
cfg = load_composite_controller_config(robot="PandaOmron")
env = robosuite.make(env_name="LivingRoomBasic", robots="PandaOmron", controller_configs=cfg,
                     has_renderer=False, has_offscreen_renderer=True, use_camera_obs=False,
                     ignore_done=True, camera_names="birdview", camera_heights=256, camera_widths=256)
env.reset()
for _ in range(20): env.step(np.zeros(env.action_dim))
opt = mujoco.MjvOption(); opt.geomgroup[0] = 0   # 隐藏碰撞 group，否则红盒子盖住网格
r = mujoco.Renderer(env.sim.model._model, height=480, width=640)   # 注意 <=640 (framebuffer 上限)
cam = mujoco.MjvCamera(); cam.lookat[:] = [0, 0, 0.5]; cam.distance = 7.2; cam.azimuth = 120; cam.elevation = -22
r.update_scene(env.sim.data._data, camera=cam, scene_option=opt)
imageio.imwrite("/tmp/scene.png", r.render())
```
用 `MUJOCO_GL=egl python ...` 跑（无显示器）。

> **坑提醒**：离屏渲染若看到**红色半透明盒子盖住家具**，那是碰撞几何（group 0）。env 实际的 mjviewer 会自动隐藏 group 0，所以遥操作时看到的是正常纹理；自己渲染时记得 `opt.geomgroup[0]=0`。
> framebuffer 默认 640，渲染宽 >640 会报错。

---

## 6. robocasa 资产格式（model.xml）速查

转换脚本生成的 `model.xml` 结构（也是手写参照）：

```xml
<mujoco model="Sofa001">
  <asset>
    <mesh file="visual/model.obj" name="Sofa001_vis" scale="1.0 1.0 1.0"/>   <!-- scale 缩放 -->
    <texture file="visual/texture.png" name="Sofa001_tex" type="2d"/>
    <material name="Sofa001_mat" texture="Sofa001_tex" .../>
  </asset>
  <worldbody>
    <body>
      <body name="object">                                   <!-- 固定约定名 "object" -->
        <geom mesh="..._vis" class="visual" material="..._mat" euler="1.5708 0 0" pos="..."/>  <!-- 可见、不碰撞、Z-up 旋转 + 居中 -->
        <geom class="region" name="reg_bbox" type="box" .../>  <!-- 包围盒，放置采样用 -->
        <geom class="collision" type="box" .../>               <!-- 碰撞近似 -->
      </body>
    </body>
  </worldbody>
</mujoco>
```

- **visual geom**：`contype=0 conaffinity=0 group=1` → 看得见、不参与物理。
- **collision geom**：`group=0` → 看不见、参与物理。脚本默认用**整体包围盒**。
- **reg_bbox**：`class="region"`，供后续“在沙发/茶几上采样放物体”用。

参考实例：仓库里 `objects/lightwheel/stool/Stool013/`（凸分解碰撞）、`objects/objaverse/cereal/cereal_6/`（box 碰撞）。

---

## 7. 手动精修（可选）

脚本默认碰撞是**整体包围盒**，对沙发/电视柜这种会“填实”内部空腔（够当障碍和支撑面，但没法把东西塞进电视柜格子）。需要更精细时，手动编辑该资产的 `model.xml`：

- 把单个碰撞 box 拆成多个（如桌面板 + 四条腿 + 不挡开放格）。
- 复杂凹形（沙发坐垫凹陷）可用 trimesh + V-HACD 做凸分解，参考 `Stool013` 的多段 `collision_mesh`。

改完直接重跑 demo 查看，无需重新转换。

---

## 8. Git 说明

- `robocasa/models/assets/furniture/` 整个目录已加入 `.gitignore`，**资产（含 50MB 网格、贴图、preview）不入库**。
- 入库的只有**代码**：`scripts/convert_meshy_furniture.py`、`robocasa/models/furniture/*.py`、`robocasa/environments/living_room/*.py`。
- 换机器/协作时，资产需要另行同步（拷贝、软链接，或日后接 Git LFS）。

---

## 9. 快速复现清单（TL;DR）

```sh
# 1. Meshy 生成单件家具，导出 OBJ with texture
# 2. 转换
python scripts/convert_meshy_furniture.py \
    --src "/path/to/Meshy_export" --category <类别> --name <实例> \
    --target-size <最长边米数> --render
# 3. 看 robocasa/models/assets/furniture/<类别>/<实例>/preview.png 验证尺度/朝向
# 4. 在 living_room.py 的 _get_furniture() 加 FurnitureObject(...)，pos.z 用脚本打印的 z，euler 调朝向
# 5. 查看
python -m robocasa.demos.demo_empty_teleop --env LivingRoomBasic
```
