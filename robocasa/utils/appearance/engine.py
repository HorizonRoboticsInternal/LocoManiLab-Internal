"""The one appearance randomizer: an XML string in, an XML string out.

Only ``<asset><texture file=...>``, ``<asset><material ...>``, ``<light>``
(plus optionally new ones) and ``<visual><headlight>`` are touched -- nothing
that could move a body or change a contact, so an episode replayed in a
randomized scene is physically identical to the original.

It runs on the MERGED model xml (``env.model.get_xml()``), through robosuite's
own ``set_xml_processor`` seam, for every scene family. Two properties make
that safe and location-independent:

* ``MujocoXML.resolve_asset_dependency`` has already absolutized every asset
  ``file``, so rewriting the string needs no temp file next to a source xml;
* robot / task-object assets are identifiable by naming prefix and are skipped
  (``ownership.py``), which is a stronger guarantee than the old "we only ever
  opened the room's own xml" -- it holds even for kitchens, whose room has no
  xml of its own.

``report`` is filled with the RESOLVED choices (which texture went where, the
final light parameters) so a run's appearance can be audited or reproduced
independently of the seed.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from lxml import etree as ET

from .config import (
    LIGHT_DIR_JITTER_RAD,
    LIGHT_POS_JITTER_M,
    AppearanceConfig,
    texture_pool,
)
from .ownership import is_protected
from .profiles import THOR_PROFILE, AppearanceProfile


def _excluded(needles: tuple[str, ...], *haystacks: str | None) -> bool:
    return any(n.lower() in h.lower() for n in needles for h in haystacks if h)


def _jitter_color(value: str, rng: np.random.Generator, half_range: float,
                  lo: float = 0.0, keep_tail: int = 0) -> str:
    """Jitter the leading channels of a space-separated color attribute.

    ``keep_tail`` channels at the end (e.g. alpha) pass through unchanged.
    """
    parts = [float(x) for x in value.split()]
    n = len(parts) - keep_tail
    jittered = [
        float(np.clip(p + rng.uniform(-half_range, half_range), lo, 1.0))
        for p in parts[:n]
    ] + parts[n:]
    return " ".join(f"{p:.6g}" for p in jittered)


def _fmt(vec) -> str:
    return " ".join(f"{float(v):.6g}" for v in vec)


def _rotate_small(vec: np.ndarray, rng: np.random.Generator,
                  max_angle: float) -> np.ndarray:
    """Rotate ``vec`` about a random axis by a random angle in [-max, max]."""
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis) + 1e-9
    angle = rng.uniform(-max_angle, max_angle)
    c, s = np.cos(angle), np.sin(angle)
    out = (vec * c + np.cross(axis, vec) * s + axis * (axis @ vec) * (1.0 - c))
    return out / (np.linalg.norm(out) + 1e-9)


# --------------------------------------------------------------------------- #
# Textures

def _swap_textures(root, rng, cfg, profile, protected, report, pool=None) -> None:
    materials_of = defaultdict(list)
    for m in root.iter("material"):
        if m.get("texture"):
            materials_of[m.get("texture")].append(m)

    groups: dict = {}
    categories: dict = {}
    index = 0
    for tex in root.iter("texture"):
        name, path = tex.get("name"), tex.get("file")
        if not path:
            continue
        # Groups are keyed and iterated in document order. For MolmoSpaces
        # scenes that order is the same in the room's own xml and in the
        # merged model (verified: FloorPlan13, 44/44 textures, same sequence),
        # which is what keeps this identical to the pre-merge implementation.
        this_index, index = index, index + 1
        if is_protected(name, protected):
            continue
        if _excluded(cfg.texture_exclude, name, path):
            continue
        found = profile.group_of(tex, materials_of.get(name, []), this_index)
        if found is None:
            continue
        key, category = found
        groups.setdefault(key, []).append(tex)
        categories[key] = category

    keys = list(groups)
    n_swap = int(round(cfg.texture_frac * len(keys)))
    picked = (rng.choice(len(keys), size=n_swap, replace=False)
              if n_swap else np.array([], dtype=int))
    pools: dict = {}
    swapped = {}
    for i in picked:
        key = keys[int(i)]
        category = categories[key]
        if category not in pools:
            pools[category] = list(pool) if pool is not None else texture_pool(
                split=cfg.texture_split,
                **({} if category is None else {"categories": (category,)}))
        choices = pools[category]
        # Absolute path: the swapped file lives outside the scene's asset
        # tree, so the xml-relative convention the other entries use cannot
        # express it.
        new = str(choices[int(rng.integers(len(choices)))])
        fixup = (profile.fixups or {}).get(category) or {}
        for tex in groups[key]:
            tex.set("file", new)
            for attr, value in (fixup.get("texture") or {}).items():
                tex.set(attr, value)
            for m in materials_of.get(tex.get("name"), []):
                for attr, value in (fixup.get("material") or {}).items():
                    m.set(attr, value)
            swapped[tex.get("name")] = new.rsplit("/", 1)[-1]
    report["textures"] = dict(split=cfg.texture_split, profile=profile.name,
                              groups=len(keys), groups_swapped=len(picked),
                              exclude=list(cfg.texture_exclude),
                              swapped=swapped)


# --------------------------------------------------------------------------- #
# Materials / lights (no scene-family knowledge needed)

def _randomize_materials(root, rng, cfg, protected, report) -> None:
    if not (cfg.material_rgba_jitter > 0 or cfg.material_pbr_jitter > 0):
        return
    excluded = cfg.texture_exclude + cfg.material_exclude
    touched = 0
    for m in root.iter("material"):
        if is_protected(m.get("name"), protected):
            continue
        if _excluded(excluded, m.get("name")):
            continue
        if cfg.material_rgba_jitter > 0 and m.get("rgba") is not None:
            m.set("rgba", _jitter_color(m.get("rgba"), rng,
                                        cfg.material_rgba_jitter, keep_tail=1))
        if cfg.material_pbr_jitter > 0:
            for attr in ("shininess", "specular"):
                cur = float(m.get(attr, 0.5))
                jittered = np.clip(
                    cur + rng.uniform(-cfg.material_pbr_jitter,
                                      cfg.material_pbr_jitter), 0.0, 1.0)
                m.set(attr, f"{float(jittered):.4g}")
        touched += 1
    report["materials_touched"] = touched


def _randomize_lights(root, rng, cfg, profile, report) -> None:
    lights = list(root.iter("light"))
    # -- color -------------------------------------------------------------- #
    if cfg.light_jitter > 0:
        for light in lights:
            for attr in ("diffuse", "ambient", "specular"):
                if light.get(attr) is not None:
                    light.set(attr, _jitter_color(light.get(attr), rng,
                                                  cfg.light_jitter, lo=0.05))
        headlight = root.find("visual/headlight")
        if headlight is not None:
            for attr in ("ambient", "diffuse", "specular"):
                if headlight.get(attr) is not None:
                    headlight.set(attr, _jitter_color(headlight.get(attr), rng,
                                                      cfg.light_jitter, lo=0.05))
    # -- intensity (hue-preserving scale) & shadows ------------------------- #
    if cfg.light_intensity_jitter > 0:
        x = float(cfg.light_intensity_jitter)
        for light in lights:
            k = rng.uniform(1.0 - x, 1.0 + x)
            for attr in ("diffuse", "specular"):
                if light.get(attr) is not None:
                    vals = np.clip([float(v) * k for v in light.get(attr).split()],
                                   0.02, 1.0)
                    light.set(attr, _fmt(vals))
    if cfg.light_shadow_toggle > 0:
        for light in lights:
            if rng.random() < cfg.light_shadow_toggle:
                light.set("castshadow", "true" if rng.random() < 0.7 else "false")
    # -- geometry ----------------------------------------------------------- #
    s = float(cfg.light_geom_jitter)
    if s > 0:
        for light in lights:
            if light.get("dir") is not None:
                d = np.array([float(x) for x in light.get("dir").split()])
                light.set("dir", _fmt(_rotate_small(d, rng,
                                                    LIGHT_DIR_JITTER_RAD * s)))
            if light.get("pos") is not None:
                p = np.array([float(x) for x in light.get("pos").split()])
                p += rng.uniform(-LIGHT_POS_JITTER_M * s, LIGHT_POS_JITTER_M * s, 3)
                p[2] = max(p[2], 0.5)
                light.set("pos", _fmt(p))
    # -- extra spot lights -------------------------------------------------- #
    added = 0
    worldbody = root.find("worldbody")
    if cfg.extra_lights > 0 and worldbody is not None:
        anchor = np.array([0.0, 0.0, 2.0])
        for light in lights:
            if light.get("pos") is not None:
                anchor = np.array([float(x) for x in light.get("pos").split()])
                break
        room = min(cfg.extra_lights, max(0, profile.max_lights - len(lights)))
        for _ in range(room):
            if rng.random() >= 0.5:
                continue
            pos = np.array([anchor[0] + rng.uniform(-2.0, 2.0),
                            anchor[1] + rng.uniform(-2.0, 2.0),
                            rng.uniform(1.8, 3.0)])
            direction = _rotate_small(np.array([0.0, 0.0, -1.0]), rng, 0.6)
            diffuse = rng.uniform(0.15, 0.55, 3)
            ET.SubElement(worldbody, "light", dict(
                name=f"aug_light_{added}", pos=_fmt(pos), dir=_fmt(direction),
                directional="false", diffuse=_fmt(diffuse),
                specular=_fmt(diffuse * 0.3), castshadow="true"))
            added += 1
    report["lights"] = [dict(pos=l.get("pos"), dir=l.get("dir"),
                             diffuse=l.get("diffuse"),
                             castshadow=l.get("castshadow", "true"))
                        for l in root.iter("light")]
    report["extra_lights_added"] = added


# --------------------------------------------------------------------------- #
def randomize_xml(xml: str, rng: np.random.Generator,
                  cfg: AppearanceConfig = AppearanceConfig(),
                  profile: AppearanceProfile = THOR_PROFILE,
                  protected: tuple[str, ...] = (),
                  report: dict | None = None,
                  pool: list | None = None) -> str:
    """Return an appearance-randomized copy of ``xml``.

    ``protected`` are naming prefixes whose assets are never touched (see
    ``ownership.protected_prefixes``). ``pool`` overrides the texture library
    for this call (a legacy escape hatch; normally the profile's per-category
    pools are used).
    """
    report = {} if report is None else report
    # lxml, not stdlib ElementTree: it preserves the comments MJCF files carry
    # (stdlib drops them, which would silently rewrite parts of the model this
    # randomizer has no business touching) and parses the ~1.6 MB merged xml
    # faster. Encode first -- lxml refuses a str that carries an encoding
    # declaration, which a scene file on disk may well have.
    root = ET.fromstring(xml.encode("utf-8") if isinstance(xml, str) else xml)
    _swap_textures(root, rng, cfg, profile, protected, report, pool)
    _randomize_materials(root, rng, cfg, protected, report)
    _randomize_lights(root, rng, cfg, profile, report)
    return ET.tostring(root).decode("utf-8")
