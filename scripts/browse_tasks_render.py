"""Render an ego + top-down browse sheet for iTHOR tasks.

For each (task, scene) pair this writes two images plus an index.html gallery:

    <out>/<Env>_<scene>_<seed>_1.png    G1 head camera (first-person)
    <out>/<Env>_<scene>_<seed>_2.png    room top-down, robot marked
    <out>/index.html                    filterable browse page

The top-down view marks the robot with a red circle and a yellow heading arrow,
projected from world coordinates through the birdview camera matrix.

Example:
    python scripts/browse_tasks_render.py --out browse_task_0804 \
        --scene-map kitchen=13,living_room=209,bedroom=329,bathroom=401
"""

import argparse
import html
import json
import os
import pathlib
import sys
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")

# This repo must win over any editable `robocasa` install pointing at a
# different checkout, otherwise newer tasks silently fail to register.
_REPO_ROOT = str(pathlib.Path(__file__).resolve().parent.parent)
if sys.path and sys.path[0] != _REPO_ROOT:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

W, H = 640, 480
EGO_CAMERA = "robot0_head_camera"
# The arena's stock `birdview` sits just under the ceiling with a 45 deg fovy,
# so it frames a couple of floor tiles and clips the robot away entirely. We
# add our own top-down camera, raised and widened to cover the whole room.
TOP_CAMERA = "browse_topdown"

ROOM_OF_PREFIX = (
    ("IThorKitchen", "kitchen"),
    ("IThorLivingRoom", "living_room"),
    ("IThorBedroom", "bedroom"),
    ("IThorBathroom", "bathroom"),
)

TASKS = {
    "kitchen": [
        "IThorKitchenEnv",
        "IThorKitchenApproachFridge",
        "IThorKitchenApproachOpenFridge",
        "IThorKitchenOpenFridge",
        "IThorKitchenCloseFridge",
        "IThorKitchenOpenCabinet",
        "IThorKitchenCloseCabinet",
        "IThorKitchenOpenDrawer",
        "IThorKitchenCloseDrawer",
        "IThorKitchenOpenOven",
        "IThorKitchenCloseOven",
        "IThorKitchenApproachOpenOven",
        "IThorKitchenDrawerPnP",
        "IThorKitchenAppleCabinetPnP",
        "IThorKitchenAppleOnPlate",
        "IThorKitchenAppleBananaPlateSwap",
        "IThorKitchenPackProduceIntoBox",
        "IThorKitchenBananaBasketToTable",
        "IThorKitchenCoffeeDrinkToTrash",
        "IThorKitchenCupStack",
        "IThorKitchenOpenDrawerTakeApple",
        "IThorKitchenSpongeToSink",
    ],
    "living_room": [
        "IThorLivingRoomEnv",
        "IThorLivingRoomApproachSofa",
        "IThorLivingRoomApproachTable",
        "IThorLivingRoomOpenDrawer",
        "IThorLivingRoomCloseLaptop",
        "IThorLivingRoomLiftStatue",
        "IThorLivingRoomCoffeeDrinkToBin",
        "IThorLivingRoomPillowToSofa",
    ],
    "bedroom": [
        "IThorBedroomEnv",
        "IThorBedroomApproachBed",
        "IThorBedroomApproachDesk",
        "IThorBedroomCanIntoBasket",
        "IThorBedroomCloseLaptop",
        "IThorBedroomCloseTopDrawer",
        "IThorBedroomLiftBasket",
        "IThorBedroomLiftBasketFromFloor",
        "IThorBedroomOpenTopDrawer",
        "IThorBedroomBedThenDesk",
        "IThorBedroomDeskThenBed",
        "IThorBedroomBedThenStorage",
        "IThorBedroomStorageThenBed",
        "IThorBedroomDeskThenStorage",
        "IThorBedroomStorageThenDesk",
        "IThorBedroomApproachDeskAndOpenLaptop",
        "IThorBedroomApproachDeskAndCloseLaptop",
        "IThorBedroomApproachDeskAndLiftBasket",
        "IThorBedroomApproachDeskAndPutCanInBasket",
        "IThorBedroomApproachDeskAndPutMugInBasket",
        "IThorBedroomApproachDeskAndPutBoxedFoodInBasket",
        "IThorBedroomApproachStorageAndOpenSafeDoor",
        "IThorBedroomApproachStorageAndCloseSafeDoor",
        "IThorBedroomApproachStorageAndOpenHamperLid",
        "IThorBedroomApproachStorageAndCloseHamperLid",
        "IThorBedroomApproachStorageAndOpenDrawer",
        "IThorBedroomApproachStorageAndCloseDrawer",
        "IThorBedroomApproachStorageAndOpenBoxFlap",
        "IThorBedroomApproachStorageAndCloseBoxFlap",
        "IThorBedroomOpenThenCloseDrawer",
        "IThorBedroomOpenThenCloseTopDrawer",
        "IThorBedroomOpenThenCloseSafeDoor",
        "IThorBedroomOpenThenCloseHamperLid",
        "IThorBedroomOpenThenCloseBoxFlap",
        "IThorBedroomOpenThenCloseLaptop",
    ],
    "bathroom": [
        "IThorBathroomEnv",
        "IThorBathroomApproachGarbageCan",
        "IThorBathroomApproachSink",
        "IThorBathroomApproachToilet",
        "IThorBathroomCloseToiletLid",
        "IThorBathroomApproachGarbageCanAndPutToiletPaperRollIntoBin",
        "IThorBathroomLiftCandleFromFloor",
        "IThorBathroomOpenToiletLid",
        "IThorBathroomPutSpongeInBathtub",
        "IThorBathroomSoapToLowerShelf",
        "IThorBathroomToiletPaperRollIntoBin",
        "IThorBathroomLiftScrubBrush",
        "IThorBathroomStandSoapBottleUpright",
        "IThorBathroomApproachToiletAndOpenToiletLid",
    ],
}


def room_of(env_name):
    for prefix, room in ROOM_OF_PREFIX:
        if env_name.startswith(prefix):
            return room
    return "other"


def _install_topdown_camera(env_name):
    """Patch the arena so every built scene carries a room-framing top-down cam.

    Done as a wrapper around the arena's camera setup because the camera must
    exist in the MJCF before robosuite compiles the model and allocates the
    offscreen render buffers.
    """
    import xml.etree.ElementTree as ET

    import mujoco
    from robosuite.utils.mjcf_utils import new_element

    from robocasa.models.arenas.molmospaces_arena import MolmoSpacesArena

    if getattr(MolmoSpacesArena, "_browse_topdown_patched", False):
        return
    original = MolmoSpacesArena._add_default_cameras

    def patched(self):
        original(self)
        if any(c.get("name") == TOP_CAMERA for c in self.worldbody.findall("./camera")):
            return
        model = mujoco.MjModel.from_xml_string(
            ET.tostring(self.root, encoding="unicode")
        )
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        xyz = data.geom_xpos
        lo, hi = xyz.min(0), xyz.max(0)
        cx, cy = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2)
        span = max(float(hi[0] - lo[0]), float(hi[1] - lo[1]))

        # Height so a 45 deg vertical fov covers `span` with a little margin,
        # looking straight down (-z). xyaxes: right=+x, up=+y  =>  forward=-z,
        # which puts world +y at the top of the image.
        fovy = 45.0
        height = (span * 0.62) / np.tan(np.radians(fovy / 2))
        self.worldbody.append(
            new_element(
                tag="camera",
                name=TOP_CAMERA,
                mode="fixed",
                pos=f"{cx:.5f} {cy:.5f} {height:.5f}",
                xyaxes="1 0 0 0 1 0",
                fovy=f"{fovy:.1f}",
            )
        )

    MolmoSpacesArena._add_default_cameras = patched
    MolmoSpacesArena._browse_topdown_patched = True


def build_env(env_name, scene_id, seed):
    """Build the env with the G1 robot, both cameras on, offscreen only."""
    import robosuite

    from robocasa.utils.g1_data_collection import forward_scene_kwargs

    _install_topdown_camera(env_name)

    scene_kwargs = forward_scene_kwargs(
        env_name,
        scene_id=scene_id,
        fixed_scene=True,
        scene_seed=seed,
        warn=False,
    )
    return robosuite.make(
        env_name=env_name,
        robots="G1FloatingBody",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=[EGO_CAMERA, TOP_CAMERA],
        camera_heights=H,
        camera_widths=W,
        render_camera=None,
        ignore_done=True,
        control_freq=20,
        renderer="mujoco",
        **scene_kwargs,
    )


def hide_ceiling(sim):
    """Make ceiling geoms fully transparent so the top-down sees the room.

    MolmoSpaces rooms are fully enclosed; a camera above the room would
    otherwise render nothing but the ceiling's underside.
    """
    model = sim.model
    hidden = 0
    for i in range(model.ngeom):
        name = model.geom_id2name(i) or ""
        body = model.body_id2name(int(model.geom_bodyid[i])) or ""
        # Names may carry a scene prefix ("fp329:ceiling_..."), so strip it
        # before matching rather than anchoring at the start of the raw name.
        stems = (name.rpartition(":")[2], body.rpartition(":")[2])
        if any(s.startswith(("ceiling", "lightfixture")) for s in stems):
            model.geom_rgba[i][3] = 0.0
            hidden += 1
    sim.forward()
    return hidden


def world_to_pixel(sim, camera_name, points):
    """Project Nx3 world points into (col,row) pixels of camera_name."""
    from robosuite.utils import camera_utils as CU

    mat = CU.get_camera_transform_matrix(
        sim=sim, camera_name=camera_name, camera_height=H, camera_width=W
    )
    px = CU.project_points_from_world_to_camera(
        points=np.asarray(points, dtype=float).reshape(-1, 3),
        world_to_camera_transform=mat,
        camera_height=H,
        camera_width=W,
    )
    # project_points_... returns (row, col); flip to (x, y) for drawing
    return [(int(c), int(r)) for r, c in px]


def annotate_topdown(img, sim, base_xy, yaw):
    """Draw the robot position circle + heading arrow onto the top-down image."""
    from PIL import Image, ImageDraw

    out = Image.fromarray(img).convert("RGB")
    draw = ImageDraw.Draw(out)

    z = 0.9  # roughly torso height, so the marker sits on the robot
    tip = (base_xy[0] + 0.6 * np.cos(yaw), base_xy[1] + 0.6 * np.sin(yaw), z)
    try:
        (px, py), (tx, ty) = world_to_pixel(
            sim, TOP_CAMERA, [(base_xy[0], base_xy[1], z), tip]
        )
    except Exception:
        return np.asarray(out)

    r = 13
    draw.ellipse([px - r, py - r, px + r, py + r], outline=(255, 40, 40), width=4)
    draw.line([px, py, tx, ty], fill=(255, 205, 0), width=5)
    # arrow head
    ang = np.arctan2(ty - py, tx - px)
    for s in (+1, -1):
        a = ang + s * 0.5
        draw.line(
            [tx, ty, int(tx - 15 * np.cos(a)), int(ty - 15 * np.sin(a))],
            fill=(255, 205, 0),
            width=5,
        )
    return np.asarray(out)


def render_one(env_name, scene_id, seed, out_dir):
    """Render one (task, scene). Returns a record dict, or raises."""
    from PIL import Image

    env = build_env(env_name, scene_id, seed)
    try:
        obs = env.reset()
        base = getattr(env, "unwrapped", env)

        ego = obs[f"{EGO_CAMERA}_image"][::-1]

        # Re-render the top-down with the ceiling hidden (the obs from reset()
        # was captured with the room still sealed).
        hide_ceiling(base.sim)
        top = base.sim.render(width=W, height=H, camera_name=TOP_CAMERA, depth=False)[
            ::-1
        ]

        xy = base._robot_base_xy()
        yaw = base._robot_base_yaw()
        top = annotate_topdown(top, base.sim, xy, yaw)

        stem = f"{env_name}_{scene_id}_{seed}"
        Image.fromarray(ego).save(out_dir / f"{stem}_1.png")
        Image.fromarray(top).save(out_dir / f"{stem}_2.png")

        try:
            lang = base.get_ep_meta().get("lang", "")
        except Exception:
            lang = ""
        return {
            "env": env_name,
            "room": room_of(env_name),
            "scene": scene_id,
            "seed": seed,
            "lang": lang,
            "scene_name": getattr(base, "scene_name", None),
            "ego": f"{stem}_1.png",
            "top": f"{stem}_2.png",
        }
    finally:
        try:
            env.close()
        except Exception:
            pass


def render_in_subprocess(env_name, scene_id, seed, out_dir, timeout=900):
    """Render one task in a fresh interpreter.

    MuJoCo/EGL state accumulates across repeated env construction and takes the
    whole process down partway through a long sweep, so each task gets its own.
    """
    import subprocess

    proc = subprocess.run(
        [
            sys.executable,
            os.path.abspath(__file__),
            "--out",
            str(out_dir),
            "--seed",
            str(seed),
            "--one",
            f"{env_name},{scene_id}",
        ],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("__REC__"):
            return json.loads(line[len("__REC__") :])

    tail = (proc.stderr or proc.stdout or "").strip().splitlines()
    detail = " | ".join(tail[-3:]) if tail else "no output"
    raise RuntimeError(f"worker exited {proc.returncode}: {detail}")


PAGE_CSS = """
:root{--bg:#fff;--fg:#16181d;--muted:#6b7280;--line:#e5e7eb;--card:#f8fafc;--accent:#b45309}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--fg:#e8eaed;--muted:#9aa3af;--line:#252a33;--card:#161a21;--accent:#f59e0b}}
:root[data-theme=dark]{--bg:#0f1115;--fg:#e8eaed;--muted:#9aa3af;--line:#252a33;--card:#161a21;--accent:#f59e0b}
:root[data-theme=light]{--bg:#fff;--fg:#16181d;--muted:#6b7280;--line:#e5e7eb;--card:#f8fafc;--accent:#b45309}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.55 ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
header{padding:24px 20px 12px;border-bottom:1px solid var(--line);position:sticky;top:0;
 background:var(--bg);z-index:5}
h1{margin:0 0 4px;font-size:19px;letter-spacing:-.01em}
.sub{color:var(--muted);font-size:13px}
.filters{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
button.f{font:inherit;font-size:13px;padding:5px 12px;border-radius:999px;cursor:pointer;
 border:1px solid var(--line);background:var(--card);color:var(--fg)}
button.f[aria-pressed=true]{background:var(--accent);border-color:var(--accent);color:#fff}
main{padding:8px 20px 60px;max-width:1400px;margin:0 auto}
section{margin:28px 0 0;scroll-margin-top:120px}
.thead{display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;
 padding-bottom:8px;border-bottom:1px solid var(--line)}
.thead h2{margin:0;font-size:16px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.tag{font-size:11px;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
 border:1px solid var(--line);border-radius:4px;padding:1px 6px}
.lang{color:var(--muted);font-size:13px;font-style:italic}
.grid{display:grid;gap:16px;margin-top:14px;
 grid-template-columns:repeat(auto-fill,minmax(330px,1fr))}
.card{border:1px solid var(--line);border-radius:10px;overflow:hidden;background:var(--card)}
.card .cap{padding:6px 10px;font-size:12px;color:var(--muted);
 font-family:ui-monospace,SFMono-Regular,Menlo,monospace;border-bottom:1px solid var(--line)}
.imgs{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line)}
.imgs img{width:100%;height:auto;display:block;background:#000;cursor:zoom-in}
.fail{padding:10px;color:#b91c1c;font-size:12px;font-family:ui-monospace,monospace;
 white-space:pre-wrap;word-break:break-word}
dialog{border:none;padding:0;background:transparent;max-width:96vw;max-height:96vh}
dialog::backdrop{background:rgba(0,0,0,.85)}
dialog img{max-width:96vw;max-height:90vh;display:block}
dialog .dcap{color:#ddd;font:13px ui-monospace,monospace;padding:8px 2px;text-align:center}
"""

PAGE_JS = """
const dlg=document.querySelector('dialog'),dimg=dlg.querySelector('img'),
      dcap=dlg.querySelector('.dcap');
document.querySelectorAll('.imgs img').forEach(im=>im.addEventListener('click',()=>{
  dimg.src=im.src;dcap.textContent=im.alt;dlg.showModal();}));
dlg.addEventListener('click',()=>dlg.close());
document.querySelectorAll('button.f').forEach(b=>b.addEventListener('click',()=>{
  const room=b.dataset.room;
  document.querySelectorAll('button.f').forEach(o=>
    o.setAttribute('aria-pressed',String(o===b)));
  document.querySelectorAll('section').forEach(s=>{
    s.style.display=(room==='all'||s.dataset.room===room)?'':'none';});
}));
"""


def write_index(records, failures, out_dir, scene_map):
    by_env = {}
    for rec in records:
        by_env.setdefault(rec["env"], []).append(rec)
    for env_name, fail in failures:
        by_env.setdefault(env_name, [])

    rooms = sorted({room_of(e) for e in by_env})
    parts = [
        "<title>iTHOR task browse — ego + top-down</title>",
        f"<style>{PAGE_CSS}</style>",
        "<header><h1>iTHOR 任务浏览 — 第一视角 + 房间俯视</h1>",
        f'<div class="sub">{len(by_env)} 个任务 · {len(records)} 组渲染成功'
        f"{f' · {len(failures)} 个失败' if failures else ''}"
        " · 每组左为 G1 第一视角，右为俯视（红圈=机器人位置，黄箭头=朝向）</div>",
        '<div class="filters">',
        '<button class="f" data-room="all" aria-pressed="true">全部</button>',
    ]
    for room in rooms:
        parts.append(
            f'<button class="f" data-room="{html.escape(room)}" '
            f'aria-pressed="false">{html.escape(room)}</button>'
        )
    parts.append("</div></header><main>")

    fail_map = dict(failures)
    for env_name in sorted(by_env, key=lambda e: (room_of(e), e)):
        recs = sorted(by_env[env_name], key=lambda r: (r["scene"], r["seed"]))
        room = room_of(env_name)
        lang = recs[0]["lang"] if recs else ""
        parts.append(f'<section data-room="{html.escape(room)}"><div class="thead">')
        parts.append(f"<h2>{html.escape(env_name)}</h2>")
        parts.append(f'<span class="tag">{html.escape(room)}</span>')
        parts.append(f'<span class="tag">{len(recs)} scenes</span>')
        if lang:
            parts.append(f'<span class="lang">“{html.escape(lang)}”</span>')
        parts.append("</div><div class=grid>")
        if not recs and env_name in fail_map:
            parts.append(
                f'<div class=card><div class="fail">'
                f"{html.escape(fail_map[env_name])}</div></div>"
            )
        for rec in recs:
            cap = f"{rec['scene_name'] or rec['scene']} · seed {rec['seed']}"
            parts.append(
                f'<div class=card><div class="cap">{html.escape(cap)}</div>'
                f'<div class="imgs">'
                f'<img loading=lazy src="{rec["ego"]}" alt="{html.escape(rec["ego"])}">'
                f'<img loading=lazy src="{rec["top"]}" alt="{html.escape(rec["top"])}">'
                f"</div></div>"
            )
        parts.append("</div></section>")

    parts.append("</main><dialog><img><div class=dcap></div></dialog>")
    parts.append(f"<script>{PAGE_JS}</script>")
    (out_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="browse_task_0804")
    ap.add_argument(
        "--scene-map",
        default="kitchen=13,living_room=209,bedroom=329,bathroom=401",
        help="room=scene_id pairs, comma separated",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--envs", nargs="*", default=None, help="restrict to these envs")
    ap.add_argument("--rooms", nargs="*", default=None, help="restrict to these rooms")
    ap.add_argument(
        "--in-process",
        action="store_true",
        help="render in this process instead of one subprocess per task "
        "(MuJoCo/EGL resources accumulate across envs and eventually crash the "
        "process, so the default isolates each task)",
    )
    ap.add_argument("--one", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    scene_map = {}
    for pair in args.scene_map.split(","):
        room, _, sid = pair.partition("=")
        scene_map[room.strip()] = int(sid)

    out_dir = pathlib.Path(args.out)
    if not out_dir.is_absolute():
        out_dir = pathlib.Path(__file__).resolve().parent.parent / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Worker mode: render exactly one (env, scene) and report it as JSON.
    if args.one:
        env_name, scene_id = args.one.split(",")
        rec = render_one(env_name, int(scene_id), args.seed, out_dir)
        print("__REC__" + json.dumps(rec), flush=True)
        return

    todo = []
    for room, envs in TASKS.items():
        if args.rooms and room not in args.rooms:
            continue
        if room not in scene_map:
            print(f"[browse] no scene id for room {room}, skipping")
            continue
        for env_name in envs:
            if args.envs and env_name not in args.envs:
                continue
            todo.append((env_name, scene_map[room]))

    records, failures = [], []
    for i, (env_name, scene_id) in enumerate(todo, 1):
        print(f"[browse] ({i}/{len(todo)}) {env_name} scene={scene_id}", flush=True)
        try:
            if args.in_process:
                records.append(render_one(env_name, scene_id, args.seed, out_dir))
            else:
                records.append(
                    render_in_subprocess(env_name, scene_id, args.seed, out_dir)
                )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[browse]   FAILED {msg}", flush=True)
            failures.append((env_name, msg))

    write_index(records, failures, out_dir, scene_map)
    (out_dir / "records.json").write_text(
        json.dumps({"records": records, "failures": failures}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[browse] done: {len(records)} ok, {len(failures)} failed -> {out_dir}")
    if failures:
        for env_name, msg in failures:
            print(f"  FAIL {env_name}: {msg}")


if __name__ == "__main__":
    main()
