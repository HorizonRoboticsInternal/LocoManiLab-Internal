"""What to randomize (:class:`AppearanceConfig`) and where replacement
textures come from (the generative-textures pool)."""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path

#: Texture pool: the robocasa AI-generated texture library.
GENERATIVE_TEXTURES_DIR = (
    Path(__file__).resolve().parents[2] / "models" / "assets" / "generative_textures"
)
#: The pool's categories. These double as :data:`SurfaceKind` values: a kitchen
#: texture is classified by WHICH of these directories its file came from, so a
#: swap always draws a wall texture for a wall (see ``profiles.py``).
POOL_CATEGORIES = ("cabinet", "counter", "floor", "wall")
#: Fraction of the pool that belongs to the ``train`` split (rest = ``test``).
TRAIN_SPLIT_FRACTION = 0.8
#: Budget on lights in one scene. MuJoCo 3.3 renders up to ``mjMAXLIGHT`` = 100,
#: so this is a LOOK constraint, not a renderer one: past a handful of overhead
#: lights a room washes out and the shadow variation the jitter is there to
#: produce disappears. It has to clear what a scene already ships with --
#: a MolmoSpaces room arrives with five (its own ceiling light plus the 2x2
#: overhead grid ``MolmoSpacesArena`` adds for even rendering), a kitchen with
#: one -- or ``light-geom``'s extra spots can never be placed.
MAX_SCENE_LIGHTS = 8

#: Light geometry jitter magnitudes at ``light_geom_jitter == 1.0``.
LIGHT_DIR_JITTER_RAD = 0.35
LIGHT_POS_JITTER_M = 0.3


@dataclass(frozen=True)
class AppearanceConfig:
    """What to randomize, and by how much. All effects are render-only."""

    #: Fraction of (non-excluded, non-protected) texture GROUPS whose image
    #: file is swapped for a random one from the pool. What a group is comes
    #: from the profile: one texture element for MolmoSpaces scenes, one
    #: surface category (wall / floor / counter / cabinet) for kitchens.
    texture_frac: float = 0.7
    #: Case-insensitive substrings; a texture whose ``name`` OR ``file``
    #: contains any of them is never touched. The default keeps task-critical
    #: fixtures recognizable. (Task OBJECTS and the robot are excluded
    #: structurally -- see ``ownership.py`` -- not through this list.)
    texture_exclude: tuple[str, ...] = ("Oven",)
    #: Which half of the pool to draw from: ``train`` for augmenting training
    #: data, ``test`` for held-out eval textures, ``all`` for everything. The
    #: split is a deterministic hash of the file name, identical everywhere.
    texture_split: str = "all"
    #: Per-channel jitter half-range for <light diffuse/ambient/specular> and
    #: the headlight.
    light_jitter: float = 0.0
    #: Scale of the light GEOMETRY jitter: at 1.0 each light's direction is
    #: rotated by up to LIGHT_DIR_JITTER_RAD and its position moved by up to
    #: LIGHT_POS_JITTER_M per axis. 0 leaves shadows where they were.
    light_geom_jitter: float = 0.0
    #: Up to this many additional spot lights, each present with probability
    #: 0.5, at random positions above the scene (capped by MAX_SCENE_LIGHTS).
    extra_lights: int = 0
    #: Multiplicative INTENSITY jitter, separate from hue: each light's
    #: diffuse+specular are scaled by a factor in [1-x, 1+x] (then clamped).
    light_intensity_jitter: float = 0.0
    #: Probability that a light's ``castshadow`` flag is re-sampled (on with
    #: p=0.7 when re-sampled). Toggling shadows is one of the cheapest, most
    #: visible lighting variations a rasterizer offers.
    light_shadow_toggle: float = 0.0
    #: Per-channel jitter half-range for existing <material rgba> colors (alpha
    #: kept). Material colors modulate their texture, so this also tints
    #: textured surfaces. 0 disables.
    material_rgba_jitter: float = 0.0
    #: Jitter half-range for material shininess / specular (MuJoCo's shading
    #: knobs, default 0.5 each), clamped to [0, 1]. 0 disables.
    material_pbr_jitter: float = 0.0
    #: Extra substrings protecting materials from rgba / shading jitter (same
    #: semantics as texture_exclude, matched on the material name).
    material_exclude: tuple[str, ...] = field(default_factory=tuple)


def texture_split_of(path: str | os.PathLike) -> str:
    """``train`` or ``test``, from a deterministic hash of the file name."""
    digest = hashlib.md5(Path(path).name.encode("utf-8")).hexdigest()
    return "train" if (int(digest, 16) % 1000) < TRAIN_SPLIT_FRACTION * 1000 else "test"


def texture_pool(root: Path | None = None, split: str = "all",
                 categories: tuple[str, ...] = POOL_CATEGORIES) -> list[Path]:
    """The generative-textures library (one split, some categories of it),
    sorted for determinism."""
    if split not in ("train", "test", "all"):
        raise ValueError(f"texture_split must be train|test|all, got {split!r}")
    root = GENERATIVE_TEXTURES_DIR if root is None else Path(root)
    pool = [p for cat in categories for p in sorted((root / cat).glob("*.png"))]
    if split != "all":
        pool = [p for p in pool if texture_split_of(p) == split]
    if not pool:
        raise FileNotFoundError(
            f"no textures found under {root} (split {split}, categories "
            f"{list(categories)})")
    return pool
