"""Reusable scene-randomization primitives for replay augmentation AND eval.

Everything here is deterministic given the rngs/seeds the caller passes, and
physics-safe by construction where it claims to be: appearance edits never
touch geometry, and distractor handling comes with the baseline capture /
restore needed to keep the task objects' placement bit-identical to a run
without distractors (adding cfgs perturbs the env's own rng consumption -- see
``reset_to_baseline``).

The scene-family-independent half lives here (camera, lens, robot tint,
distractors). Textures / lights / material shading live in
``robocasa.utils.appearance``, which installs itself on the env through
robosuite's ``set_xml_processor`` seam and therefore covers kitchens as well as
MolmoSpaces rooms.

Used by ``robocasa.scripts.replay_augment`` (replay-time data augmentation)
and by LocoManiLab's sim-eval client (eval-time robustness randomization).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ROBOT_REF_KEY = "__robot_free_joint__"


# --------------------------------------------------------------------------- #
# Camera

def randomize_camera(env, camera_names: list[str], rng_seed: int, *,
                     pos: float = 0.01, rot: float = 0.05, fovy: float = 2.0):
    """Perturb cameras in the compiled model (per episode; the model is
    rebuilt by every hard reset, so this never accumulates)."""
    if not (pos > 0 or rot > 0 or fovy > 0):
        return None
    from robosuite.utils.mjmod import CameraModder

    modder = CameraModder(
        sim=env.sim,
        camera_names=list(camera_names),
        randomize_position=pos > 0,
        randomize_rotation=rot > 0,
        randomize_fovy=fovy > 0,
        position_perturbation_size=pos,
        rotation_perturbation_size=rot,
        fovy_perturbation_size=fovy,
        random_state=np.random.RandomState(rng_seed),
    )
    modder.randomize()
    return modder


# --------------------------------------------------------------------------- #
# Lens (image-space camera intrinsics)

class LensWarp:
    """Image-space camera-intrinsics randomization for a pinhole render.

    MuJoCo renders an ideal pinhole (the G1 head camera is a plain ``fovy``
    camera), so distortion coefficients and a principal-point offset cannot
    be expressed in the model; they are applied to every rendered frame
    instead: Brown radial distortion (k1, k2 -- barrel/fisheye-like or
    pincushion), a principal-point shift in pixels and a zoom (which also
    hides the border the distortion would otherwise expose). Built once per
    episode; ``scale`` multiplies every range (0 = identity).
    """

    def __init__(self, height: int, width: int, rng, scale: float = 1.0) -> None:
        self.height, self.width = int(height), int(width)
        s = float(scale)
        self.k1 = float(rng.uniform(-0.25, 0.25)) * s
        self.k2 = float(rng.uniform(-0.05, 0.05)) * s
        self.shift = (rng.uniform(-8.0, 8.0, 2) * s).astype(np.float32)  # px
        self.zoom = 1.0 + float(rng.uniform(0.0, 0.12)) * s
        self._maps = self._build_maps() if s > 0 else None

    def _build_maps(self):
        h, w = self.height, self.width
        cx, cy = w / 2.0 + self.shift[0], h / 2.0 + self.shift[1]
        f = 0.5 * w  # normalization focal; the coefficients are relative to it
        u, v = np.meshgrid(np.arange(w, dtype=np.float32),
                           np.arange(h, dtype=np.float32))
        x = (u - cx) / f / self.zoom
        y = (v - cy) / f / self.zoom
        r2 = x * x + y * y
        factor = 1.0 + self.k1 * r2 + self.k2 * r2 * r2
        map_x = (x * factor * f + w / 2.0).astype(np.float32)
        map_y = (y * factor * f + h / 2.0).astype(np.float32)
        return map_x, map_y

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self._maps is None:
            return img
        import cv2

        map_x, map_y = self._maps
        return cv2.remap(np.ascontiguousarray(img), map_x, map_y,
                         interpolation=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT)

    def describe(self) -> dict:
        return dict(k1=round(self.k1, 4), k2=round(self.k2, 4),
                    principal_shift_px=[round(float(v), 2) for v in self.shift],
                    zoom=round(self.zoom, 4))


# --------------------------------------------------------------------------- #
# Robot appearance

def tint_robot_geoms(env, rng, amount: float, prefix: str = "robot0_") -> int:
    """Jitter the rgba tint of the robot's own geoms (post-compile; the model
    is rebuilt by every hard reset). Render-only. Returns geoms touched.

    The policy sees its own hands/arms in the ego view, so this is the
    MuJoCo-side analogue of randomizing the robot's shader parameters."""
    if amount <= 0:
        return 0
    model = env.sim.model
    touched = 0
    for gid in range(model.ngeom):
        name = model.geom_id2name(gid) or ""
        body = model.body_id2name(int(model.geom_bodyid[gid])) or ""
        if not (name.startswith(prefix) or body.startswith(prefix)):
            continue
        rgba = np.array(model.geom_rgba[gid], dtype=np.float32)
        rgba[:3] = np.clip(rgba[:3] + rng.uniform(-amount, amount, 3), 0.0, 1.0)
        model.geom_rgba[gid] = rgba
        touched += 1
    return touched


# --------------------------------------------------------------------------- #
# Distractors

def pick_distractor_instance(group: str, rng,
                             registries: tuple[str, ...] = ("objaverse",
                                                            "lightwheel")
                             ) -> str | None:
    """A concrete model.xml for ``group``, chosen with OUR rng.

    Pinning the instance via cfg["info"] makes create_obj take its
    deterministic branch: sampling the instance with env.rng instead would
    consume draws BETWEEN object creation and placement sampling and shift
    every placement that follows. Only registries the env searches are valid
    (the pinned-xml reverse lookup raises ValueError for any other).
    """
    from robocasa.models.objects.kitchen_object_utils import (
        OBJ_CATEGORIES, OBJ_GROUPS)

    paths = []
    for cat in OBJ_GROUPS.get(group) or []:
        for reg, entry in (OBJ_CATEGORIES.get(cat) or {}).items():
            if reg in registries:
                paths.extend(getattr(entry, "mjcf_paths", []) or [])
    if not paths:
        return None
    return sorted(paths)[int(rng.integers(len(paths)))]


def _dist_to_segment(p, a, b) -> float:
    p, a, b = (np.asarray(x, dtype=float) for x in (p, a, b))
    ab = b - a
    denom = float(ab @ ab)
    t = 0.0 if denom < 1e-12 else float(np.clip((p - a) @ ab / denom, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _semantic_region_of(env, name):
    for cfg in getattr(env, "object_cfgs", []):
        if cfg.get("name") == name:
            region = cfg.get("placement", {}).get("region")
            if type(region).__name__ == "SemanticRegion":
                return region
    return None


def _inside_region(region, xy, margin: float) -> bool:
    try:
        return bool(region.contains_xy(np.asarray(xy, dtype=float), margin=margin))
    except Exception:
        return True  # bounds unavailable: fall back to the distance checks


def _support_region_at(env, qpos):
    """The support region an object at ``qpos`` is resting on, or None."""
    try:
        regions = [r for r in env.get_region(return_all=True)
                   if getattr(r, "kind", None) == "support"]
    except Exception:
        return None
    for region in regions:
        try:
            center, _ = region.world_pose()
            if abs(float(qpos[2]) - float(center[2])) > 0.4:
                continue
            if region.contains_xy(np.asarray(qpos[:2], dtype=float)):
                return region
        except Exception:
            continue
    return None


def task_object_qpos(env, names: list[str]) -> dict[str, np.ndarray]:
    out = {}
    for n in names:
        obj = env.objects.get(n)
        if obj is not None and obj.joints:
            out[n] = env.sim.data.get_joint_qpos(obj.joints[0]).copy()
    return out


def robot_base_xy(env) -> np.ndarray:
    """The robot's base position in the world, for any scene family.

    MolmoSpaces room envs publish ``robot_base_pos``; a robocasa kitchen does
    not, so fall back to the robot's own base body in the compiled model.
    Returns the origin when even that is unavailable -- the corridor check it
    feeds is a refinement, not a correctness requirement.
    """
    pos = getattr(env, "robot_base_pos", None)
    if pos is not None:
        return np.asarray(pos[:2], dtype=float)
    for robot in getattr(env, "robots", None) or []:
        body = getattr(getattr(robot, "robot_model", None), "root_body", None)
        if body:
            try:
                return np.asarray(
                    env.sim.data.get_body_xpos(body)[:2], dtype=float).copy()
            except Exception:
                pass
    return np.zeros(2, dtype=float)


def _robot_free_joint(env):
    """The robot's free joint, or None.

    MolmoSpaces room envs expose ``_get_robot_free_joint``; use it when it is
    there. Otherwise do the same lookup here -- it is not family-specific, just
    "does the robot's root body carry a free joint" -- so that a robocasa
    kitchen gets the same baseline guard. Without it, adding distractor cfgs
    moves the robot's spawn as well as the objects, and a distractor run stops
    being comparable to the distractor-free one.
    """
    getter = getattr(env, "_get_robot_free_joint", None)
    if getter is not None:
        return getter()
    robots = getattr(env, "robots", None) or []
    if not robots:
        return None
    try:
        model = env.sim.model
        body_id = model.body_name2id(robots[0].robot_model.root_body)
        joint_id = model.body_jntadr[body_id]
        if joint_id >= 0 and model.jnt_type[joint_id] == 0:  # 0 = mjJNT_FREE
            return model.joint_names[joint_id]
    except Exception:
        return None
    return None


def capture_baseline(env, task_names: list[str]) -> dict:
    """Task-object AND robot free-joint qpos of a distractor-free reset."""
    ref = task_object_qpos(env, task_names)
    joint = _robot_free_joint(env)
    if joint is not None:
        ref[ROBOT_REF_KEY] = env.sim.data.get_joint_qpos(joint).copy()
    return ref


def baseline_matches(env, ref_qpos: dict, atol: float = 1e-6) -> bool:
    now = task_object_qpos(env, [n for n in ref_qpos if n != ROBOT_REF_KEY])
    if not all(n in now and np.allclose(now[n], q, atol=atol)
               for n, q in ref_qpos.items() if n != ROBOT_REF_KEY):
        return False
    joint = _robot_free_joint(env)
    if ROBOT_REF_KEY in ref_qpos and joint is not None:
        return bool(np.allclose(env.sim.data.get_joint_qpos(joint),
                                ref_qpos[ROBOT_REF_KEY], atol=atol))
    return True


def restore_baseline(env, ref_qpos: dict) -> bool:
    """Put the task objects and the robot back on the reference baseline.

    Needed under distractors: an unplaced or ill-placed extra object changes
    how many draws the env's own INITIAL_POSE_JITTER validity loop consumes,
    shifting every jittered pose deterministically -- retrying resets cannot
    help under a fixed seed, restoring absolute poses can. Returns whether the
    baseline holds afterwards.
    """
    if not ref_qpos:
        return True
    if baseline_matches(env, ref_qpos):
        return True
    for name, qpos in ref_qpos.items():
        if name == ROBOT_REF_KEY:
            joint = _robot_free_joint(env)
        else:
            obj = env.objects.get(name)
            joint = obj.joints[0] if obj is not None and obj.joints else None
        if joint is not None:
            env.sim.data.set_joint_qpos(joint, qpos.copy())
    env.sim.forward()
    if hasattr(env, "_robot_base_xy"):
        env.robot_base_pos[:2] = env._robot_base_xy()
        env.robot_base_ori[2] = env._robot_base_yaw()
    return baseline_matches(env, ref_qpos)


def reset_to_baseline(env, ref_qpos: dict) -> bool:
    """reset() + ``restore_baseline``."""
    env.reset()
    if restore_baseline(env, ref_qpos):
        return True
    print("[scene-rnd]   baseline restore FAILED")
    return False


def teleport_distractors(env, distractor_names: list[str],
                         task_names: list[str], rng, *,
                         radius: float = 0.35, clearance: float = 0.12,
                         corridor: float = 0.15) -> dict:
    """Move each distractor into an annulus around a task object, keeping
    clear of every task object, other distractors, the robot's approach
    corridor and the support region's edge. A distractor with no valid spot is
    parked below the floor (an unplaced one at the world origin would
    intersect the scene). Returns {name: [x, y]} of the placed ones."""
    task_qpos = task_object_qpos(env, task_names)
    task_xy = {n: q[:2].copy() for n, q in task_qpos.items()}
    if not task_xy:
        return {}
    base_xy = robot_base_xy(env)
    placed: dict[str, np.ndarray] = {}
    for dname in distractor_names:
        obj = env.objects.get(dname)
        if obj is None or not obj.joints:
            continue
        joint = obj.joints[0]
        qpos = env.sim.data.get_joint_qpos(joint).copy()
        target = list(task_xy)[int(rng.integers(len(task_xy)))]
        center = task_xy[target]
        region = (_semantic_region_of(env, target)
                  or _support_region_at(env, task_qpos[target]))
        spot = None
        for _ in range(40):
            r = rng.uniform(clearance, radius)
            th = rng.uniform(0.0, 2.0 * np.pi)
            cand = center + r * np.array([np.cos(th), np.sin(th)])
            if any(np.linalg.norm(cand - xy) < clearance
                   for xy in list(task_xy.values()) + list(placed.values())):
                continue
            if _dist_to_segment(cand, base_xy, center) < corridor:
                continue
            if region is not None and not _inside_region(region, cand, margin=0.03):
                continue
            spot = cand
            break
        if spot is None:
            print(f"[scene-rnd]   {dname}: no clear spot near {target!r}; "
                  f"parked out of the scene")
            qpos[2] = -5.0
            env.sim.data.set_joint_qpos(joint, qpos)
            continue
        qpos[:2] = spot
        # Rest on the target's surface: reuse its resting height (the
        # distractor's own z is meaningless when the sampler never placed it).
        qpos[2] = float(task_qpos[target][2]) + 0.02
        env.sim.data.set_joint_qpos(joint, qpos)
        placed[dname] = spot
    env.sim.forward()
    return {k: [round(float(v[0]), 4), round(float(v[1]), 4)]
            for k, v in placed.items()}


def build_distractor_cfgs(count: int, groups: list[str], rng,
                          registries: tuple[str, ...],
                          region_ids: list[str] | None = None,
                          regions: list | None = None) -> list[dict]:
    """Distractor object cfgs, safe to APPEND to an existing cfg list.

    * Instances are pinned via ``info`` (no env-rng consumption).
    * Placement style must MATCH the task cfgs': a semantic placement mixed
      into a plain surface-sampled run flips ``_build_placement_initializer``
      into a different code path for every object and deterministically shifts
      the task placement. Pass ``region_ids`` (strings, replay/pinned-meta
      style) or ``regions`` (resolved SemanticRegion objects, eval style) when
      the task cfgs place semantically; neither -> plain ``placement={}``.
    """
    cfgs = []
    for i in range(count):
        group = groups[int(rng.integers(len(groups)))]
        instance = pick_distractor_instance(group, rng, registries=registries)
        if instance is None:
            print(f"[scene-rnd] distractor group {group!r} has no instances; "
                  f"skipped")
            continue
        if region_ids:
            placement = dict(
                semantic_region_id=region_ids[int(rng.integers(len(region_ids)))],
                strategy="uniform", margin=0.05)
        elif regions:
            placement = dict(region=regions[int(rng.integers(len(regions)))],
                             strategy="uniform", margin=0.05)
        else:
            placement = {}
        cfgs.append(dict(name=f"distractor_{i}", obj_groups=group,
                         info=dict(mjcf_path=instance), placement=placement))
    return cfgs


def _distractor_placement(cfgs: list[dict]) -> dict:
    """A placement for an appended distractor, in the style the task uses.

    Mixing styles is what breaks placement: a semantic placement dropped into a
    fixture-sampled run flips ``_build_placement_initializer`` onto a different
    code path for EVERY object and deterministically shifts the task placement,
    and an empty placement in a kitchen gives the sampler nothing to sample on
    (``PlacementError: Cannot initialize placement``). So copy what is already
    there:

    * MolmoSpaces rooms place into a ``SemanticRegion`` -> reuse the region;
    * robocasa kitchens place onto a ``fixture`` with a sampling window ->
      reuse one of those windows, preferring the task's OWN distractor cfg
      (``distr_*``), which is already "some object loose near the workspace";
    * anything else -> an empty placement, the previous behaviour.
    """
    placements = [c.get("placement") or {} for c in cfgs]
    regions = [p["region"] for p in placements
               if type(p.get("region")).__name__ == "SemanticRegion"]
    if regions:
        return dict(region=regions[0], strategy="uniform", margin=0.05)
    fixtured = [(c, p) for c, p in zip(cfgs, placements) if p.get("fixture") is not None]
    if fixtured:
        preferred = [p for c, p in fixtured if str(c.get("name", "")).startswith("distr")]
        # Shallow copy: the fixture entry is a live Fixture object rebuilt by
        # every reset and must be shared, not duplicated.
        return dict((preferred or [p for _c, p in fixtured])[0])
    return {}


def wrap_obj_cfgs_with_distractors(env_name: str, count: int,
                                   groups: list[str], rng,
                                   registries: tuple[str, ...]):
    """Class-level wrap of ``_get_obj_cfgs`` appending distractor cfgs.

    The eval-side injection point: eval envs sample their own cfgs (no pinned
    ep_meta), so the append happens where they are generated. The placement
    style follows the generated cfgs (region objects when they are semantic).
    Returns ``(names, restore)``.
    """
    from robosuite.environments.base import REGISTERED_ENVS

    cls = REGISTERED_ENVS[env_name]
    original = cls._get_obj_cfgs
    # Identities chosen ONCE with the caller's rng: stable across resets.
    fixed = build_distractor_cfgs(count, groups, rng, registries)
    names = [c["name"] for c in fixed]

    def wrapped(self):
        cfgs = list(original(self))
        out = []
        for c in fixed:
            c = dict(c)
            c["placement"] = _distractor_placement(cfgs)
            out.append(c)
        return cfgs + out

    cls._get_obj_cfgs = wrapped

    def restore() -> None:
        cls._get_obj_cfgs = original

    return names, restore


def install_distractors(env, env_name: str, count: int, groups: list[str], rng,
                        registries: tuple[str, ...]):
    """Wrap in up to ``count`` distractors, backing off until they FIT.

    How many extra objects a room can take is a property of the room, not of
    the spec: a bathroom or a bedroom whose task region is already tight
    rejects three and raises out of ``env.reset()``. Asking for three and
    getting two is the useful outcome; losing the whole eval to a
    ``RandomizationError`` is not. Returns ``(names, restore, placed)`` with
    ``placed`` the count that actually took, possibly 0.
    """
    for n in range(int(count), 0, -1):
        names, restore = wrap_obj_cfgs_with_distractors(
            env_name, n, groups, rng, registries)
        try:
            env.reset()
            if n != count:
                print(f"[scene-rnd] this scene took {n} distractor(s), not "
                      f"{count}", flush=True)
            return names, restore, n
        except Exception as exc:
            restore()
            print(f"[scene-rnd] {n} distractor(s) do not fit here "
                  f"({type(exc).__name__}: {str(exc)[:120]}); "
                  f"trying {n - 1}", flush=True)
    env.reset()  # leave the env on a clean, distractor-free model
    return [], None, 0


# --------------------------------------------------------------------------- #
# Spec parsing
#
# The compact ``textures=0.7,lights,camera`` form every entry point takes. It
# lives here so the preview tool, eval and replay agree on token names and
# defaults by construction rather than by three hand-kept copies.

@dataclass
class RandomizeSpec:
    """A parsed ``--randomize`` spec. ``None`` means "this token was absent"."""

    textures: float | None = None
    #: 'train' (augmentation) / 'test' (held-out eval) / 'all'. The caller
    #: supplies the default because it differs per entry point.
    texture_split: str = "all"
    lights: float | None = None
    #: True when ``lights`` came from the token's own default rather than an
    #: explicit ``lights=<value>``, so the profile may substitute its own.
    lights_defaulted: bool = False
    light_geom: float | None = None
    camera: float | None = None
    lens: float | None = None
    pbr: float | None = None
    robot: float | None = None
    #: Material COLOUR jitter. Separate from ``pbr`` (which is shading only):
    #: a material's rgba modulates its texture, so this tints textured
    #: surfaces as well as flat ones. Off unless asked for.
    rgba: float | None = None
    #: How many extra spot lights ``light-geom`` may add (each with p=0.5,
    #: capped by the profile's light budget).
    extra_lights: int | None = None
    distractors: int = 0
    #: ``None`` = "the profile's default", which differs per scene family:
    #: MolmoSpaces rooms name fixtures after the real appliance so "Oven" keeps
    #: the usual task landmark recognizable, while kitchen textures are named
    #: after the pool CATEGORY (cabinet/counter/floor/wall) and "Oven" would
    #: match nothing. An explicit ``exclude=`` always wins.
    exclude: tuple[str, ...] | None = None

    @property
    def edits_appearance(self) -> bool:
        """Whether anything here is applied by rewriting the model xml, i.e.
        needs an appearance profile for the env's scene family."""
        return bool(self.textures or self.lights or self.light_geom or self.pbr)

    #: Historical name from when this was MolmoSpaces-only.
    edits_scene_xml = edits_appearance

    def appearance_config(self, profile=None):
        """The :class:`AppearanceConfig` for the xml-level tokens.

        ``profile`` supplies the family-specific default for ``exclude`` when
        the spec did not set one.
        """
        from robocasa.utils.appearance import AppearanceConfig

        exclude = self.exclude
        if exclude is None:
            exclude = getattr(profile, "default_exclude", ("Oven",))
        lights = self.lights or 0.0
        if self.lights is not None and self.lights_defaulted:
            lights = getattr(profile, "default_light_jitter", 0.15)
        return AppearanceConfig(
            texture_frac=self.textures or 0.0,
            texture_exclude=tuple(exclude),
            texture_split=self.texture_split,
            light_jitter=lights,
            light_geom_jitter=self.light_geom or 0.0,
            extra_lights=(self.extra_lights if self.extra_lights is not None
                          else (1 if self.light_geom else 0)),
            # The intensity swing tracks the colour jitter, so a family that
            # gets a gentler default gets a gentler swing with it.
            light_intensity_jitter=2.0 * lights if self.lights else 0.0,
            light_shadow_toggle=0.5 if self.lights else 0.0,
            material_rgba_jitter=self.rgba or 0.0,
            material_pbr_jitter=self.pbr or 0.0,
        )

    def format_resolved(self, spec_str: str = "", profile=None,
                        seed: int | None = None) -> str:
        """The two-line banner every entry point prints before it starts.

        What matters here is the values AFTER the scene family resolved them:
        a bare ``lights`` is 0.15 in one room and 0.08 in another, and
        ``exclude`` defaults differently too, so echoing the spec string alone
        would not tell you what actually ran.
        """
        cfg = self.appearance_config(profile)
        parts = []
        if self.textures:
            parts.append(f"textures={cfg.texture_frac:g}({cfg.texture_split})")
        if self.lights:
            parts.append(f"lights={cfg.light_jitter:g}")
        if self.light_geom:
            parts.append(f"light-geom={self.light_geom:g}"
                         f"(+{cfg.extra_lights} spot)")
        for name, value in (("camera", self.camera), ("lens", self.lens),
                            ("pbr", self.pbr), ("rgba", self.rgba),
                            ("robot", self.robot)):
            if value:
                parts.append(f"{name}={value:g}")
        if self.distractors:
            parts.append(f"distractors={self.distractors}")
        head = f"[randomize] spec={spec_str or '(none)'}"
        if profile is not None:
            head += f"  profile={getattr(profile, 'name', '?')}"
        if seed is not None:
            head += f"  seed={seed}"
        body = "  ".join(parts) or "nothing enabled"
        if cfg.texture_exclude:
            body += f"  exclude={list(cfg.texture_exclude)}"
        return f"{head}\n            {body}"

    def describe(self) -> dict:
        out = dict(
            textures=self.textures,
            texture_split=self.texture_split if self.textures else None,
            lights=self.lights,
            light_geom=self.light_geom,
            camera=self.camera,
            lens=self.lens,
            pbr=self.pbr,
            robot=self.robot,
            rgba=self.rgba,
            extra_lights=self.extra_lights,
            distractors=self.distractors or None,
            exclude=None if self.exclude is None else list(self.exclude),
        )
        return {k: v for k, v in out.items() if v is not None}


#: Named recipes, usable anywhere a token is. A preset expands in place and
#: later tokens override it, so ``standard,textures=0.4`` is "the standard
#: recipe but gentler textures". Deliberately NOT named light/medium/heavy:
#: ``light`` one letter away from the ``lights`` token is a trap.
PRESETS: dict[str, str] = {
    "off": "",
    "mild": "textures=0.5,lights",
    "standard": "textures,lights,camera,pbr",
    "stress": "all,lens,robot",
}

#: Every token, with a one-line description. Single source for --randomize help
#: and for the docs table.
TOKEN_HELP: list[tuple[str, str]] = [
    ("textures[=frac]", "replace that fraction of the scene's SURFACES"),
    ("texture-split=train|test|all", "which half of the texture pool to draw from"),
    ("lights[=jitter]", "light colour + intensity + shadows + direction/position"),
    ("light-geom[=scale]", "light geometry only (0 keeps shadows put)"),
    ("extra-lights[=N]", "how many extra spot lights light-geom may add"),
    ("camera[=scale]", "ego extrinsics: pos +-1cm, rot +-0.05rad, fovy +-2deg"),
    ("lens[=scale]", "image-space intrinsics: distortion, principal point, zoom"),
    ("pbr[=jitter]", "material shininess / specular"),
    ("rgba[=jitter]", "material colour tint (modulates textures too)"),
    ("robot[=tint]", "the robot's own geoms"),
    ("distractors[=N]", "extra objects near the task objects"),
    ("exclude=A|B", "substrings of texture names/paths never swapped"),
    ("all", "= textures,lights,camera,pbr,distractors"),
]


def randomize_help() -> str:
    """The token/preset reference, for ``--randomize help``."""
    width = max(len(t) for t, _ in TOKEN_HELP)
    lines = ["tokens (comma separated, '=value' optional -- omitted takes the default):"]
    lines += [f"  {t:<{width}}  {d}" for t, d in TOKEN_HELP]
    lines += ["", "presets (expand in place; later tokens override them):"]
    width = max(len(n) for n in PRESETS)
    lines += [f"  {n:<{width}}  {v or '(nothing -- an explicit baseline)'}"
              for n, v in PRESETS.items()]
    lines += ["",
              "Defaults marked 'per scene family' resolve against the env: "
              "lights is 0.15 in a MolmoSpaces room and 0.08 in a kitchen or "
              "living room (one light instead of five), and exclude defaults "
              "to 'Oven' only where fixtures are named after appliances.",
              "Examples:  standard          standard,lens          "
              "standard,textures=0.4          all,distractors=2"]
    return "\n".join(lines)


def _expand_presets(spec: str) -> list[str]:
    """Flatten ``spec`` into tokens, expanding preset names in place."""
    out: list[str] = []
    for tok in (t.strip() for t in str(spec).split(",")):
        if not tok:
            continue
        if tok in PRESETS:
            out.extend(t for t in PRESETS[tok].split(",") if t.strip())
        else:
            out.append(tok)
    return out


def parse_randomize_spec(spec: str, *, texture_split: str = "all") -> RandomizeSpec:
    """Parse ``textures=0.5,lights,camera,distractors=2`` into a RandomizeSpec.

    Tokens are comma separated and may carry ``=value`` (omitted = the token's
    own default). ``texture_split`` is the caller's default for the
    ``texture-split=`` token: eval passes 'test' (held-out), augmentation
    passes 'train'. Raises ValueError on an unknown token.
    """
    out = RandomizeSpec(texture_split=texture_split)
    for tok in _expand_presets(spec):
        key, _, val = tok.partition("=")
        if key == "all":
            out.textures, out.lights = 0.7, 0.15
            out.lights_defaulted = True
            out.camera, out.distractors, out.pbr = 1.0, 3, 0.2
        elif key == "textures":
            out.textures = float(val) if val else 0.7
        elif key == "texture-split":
            if val not in ("train", "test", "all"):
                raise ValueError("texture-split= must be train|test|all")
            out.texture_split = val
        elif key == "lights":
            out.lights = float(val) if val else 0.15
            out.lights_defaulted = not val
        elif key == "light-geom":
            out.light_geom = float(val) if val else 1.0
        elif key == "camera":
            out.camera = float(val) if val else 1.0
        elif key == "pbr":
            out.pbr = float(val) if val else 0.2
        elif key == "robot":
            out.robot = float(val) if val else 0.15
        elif key == "rgba":
            out.rgba = float(val) if val else 0.15
        elif key == "extra-lights":
            out.extra_lights = int(val) if val else 1
        elif key == "lens":
            out.lens = float(val) if val else 1.0
        elif key == "distractors":
            out.distractors = int(val) if val else 3
        elif key == "exclude":
            out.exclude = tuple(s for s in val.split("|") if s)
        else:
            raise ValueError(
                f"unknown token {tok!r}. Tokens: "
                + ", ".join(t for t, _ in TOKEN_HELP)
                + ". Presets: " + ", ".join(PRESETS)
                + ". Pass 'help' as the whole spec for the full reference.")
    if out.lights is not None and out.light_geom is None:
        out.light_geom = 1.0  # 'lights' means color AND geometry (shadows)
    return out
