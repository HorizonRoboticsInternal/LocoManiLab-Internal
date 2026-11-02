"""Per-scene-family knowledge, reduced to one pure function.

Lights and materials need no family knowledge at all -- a ``<light>`` is a
``<light>``. Textures do: to swap one you must know WHAT SURFACE it is on, so
you draw a wall texture for a wall and not a countertop for it.

A profile answers exactly that, per texture element:

    group_of(tex, mats, index) -> (group_key, category) | None

``None`` means "leave this one alone". Textures sharing a ``group_key`` are
swapped together, to the SAME new file -- that is what makes a kitchen's
sixteen wall panels stay one wall instead of becoming sixteen different walls.
``category`` picks the pool directory to draw from (``None`` = draw from all).

MolmoSpaces scenes are 1:1 -- each surface owns its own texture element (44
textures / 44 materials in FloorPlan13) -- so every element is its own group
and the category is unconstrained, which is the behaviour this randomizer has
always had for iTHOR.

RoboCasa kitchens are the opposite: fixtures are separate merged models, and
one texture is shared by every panel of a surface (all wall panels point at one
file, all cabinetry at another). Which file that is depends on the STYLE and
comes from either texture library -- ``textures/bricks/red_bricks.png`` for
style 1, ``generative_textures/wall/tex044.png`` for a generative style -- so
the path cannot classify it; the MATERIAL NAMES can, and those rules are
robocasa's own, ported from ``robocasa.utils.texture_swap`` (which is what the
built-in ``generative_textures="100p"`` flag uses). Grouping by surface then
reproduces exactly the unit that flag swaps, while adding the train/test split,
per-surface independence and seed control it lacks.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import MAX_SCENE_LIGHTS, POOL_CATEGORIES


class UnsupportedScene(ValueError):
    """No appearance profile knows how to randomize this env's scene."""


#: Sentinel kind: this material VETOES its texture (a cabinet handle shares the
#: cabinet's metal texture; re-texturing it would repaint every handle).
SKIP = "__skip__"

#: Attributes a swapped texture (and the materials using it) needs for the
#: replacement to be laid out the way the pool's images expect. Mirrors what
#: ``texture_swap.replace_floor_texture`` / ``replace_wall_texture`` set.
KITCHEN_FIXUPS = {
    "floor": dict(texture=dict(type="2d"), material=dict(texrepeat="2 2")),
    "wall": dict(texture=dict(type="2d"), material=dict(texrepeat="3 3")),
}

#: A ``FloorArena`` floor ships ``texrepeat="10 10"``, tuned for the tight
#: repeating tiles the living-room styles use. The pool's floor images are
#: room-scale boards, and at 10 they collapse into noise (rendered both ways to
#: check), so a swapped floor is relaid at the same 2 the kitchen uses. The
#: wall is already at 3, which is where the kitchen puts it too.
LIVING_ROOM_FIXUPS = {
    "floor": dict(texture=dict(type="2d"), material=dict(texrepeat="2 2")),
    "wall": dict(texture=dict(type="2d"), material=dict(texrepeat="3 3")),
}


@dataclass(frozen=True)
class AppearanceProfile:
    name: str
    #: (tex_elem, materials_using_it, document_index) -> (key, category) | None
    group_of: Callable[..., tuple | None]
    #: Default texture/material protection for this family's naming scheme.
    default_exclude: tuple[str, ...]
    #: Per-category attribute fixups applied when a texture is swapped.
    fixups: dict | None = None
    #: What a bare ``lights`` token means for this family. It has to differ:
    #: the jitter is per-light, so its VISIBLE effect scales with how many
    #: lights the scene has. A MolmoSpaces room has five (its own ceiling light
    #: plus the 2x2 overhead grid the arena adds) and averages them out; a
    #: kitchen has exactly one, where the same numbers read as a heavy colour
    #: cast. Passing ``lights=<value>`` explicitly always wins.
    default_light_jitter: float = 0.15
    max_lights: int = MAX_SCENE_LIGHTS


def _thor_group(tex, mats, index):
    """One group per texture element, any category."""
    return (index, None)


def _kitchen_kind_of_material(name: str | None):
    """Which surface a kitchen material is, by robocasa's own naming rules.

    Ported from ``robocasa.utils.texture_swap``: the four ``replace_*``
    functions each walk the materials and match on the same substrings. The
    order matters -- a floor is named ``floor_room_wall_mat``, so "floor" has
    to be tested before "wall".
    """
    if not name:
        return None
    n = name.lower()
    if "handle" in n or "transparent" in n:
        return SKIP
    if "counter_top" in n:
        return "counter"
    if "floor" in n:
        return None if "backing" in n else "floor"
    if "wall" in n:
        return None if "backing" in n else "wall"
    if "counter_base" in n or "housing" in n:
        return "cabinet"
    if ("stack" in n or "cab" in n or "shelves" in n or "bottom" in n
            or ("top" in n and "counter" not in n and "stove" not in n)):
        return "cabinet"
    return None


#: The two surfaces a ``FloorArena`` owns, by the names it gives them
#: (``robocasa/models/assets/arenas/floor_arena.xml`` plus
#: ``FloorArena.set_floor_texture`` / ``set_wall_texture``). Exact names, not
#: substrings: this arena is generated, not authored per scene.
_FLOOR_ARENA_SURFACES = {
    "floorplane": "floor",
    "walls_mat": "wall",
}


def _living_room_group(tex, mats, index):
    """Group by the FloorArena surface a texture is on.

    A living room is built by ``LivingRoomArena(FloorArena)`` -- a floor plus a
    four-wall shell, textured by the STYLE -- not by kitchen's wall fixtures,
    so the kitchen classifier's names do not apply here. Furniture (sofas,
    tables, TV stands) carries its own baked mesh textures and is left alone,
    exactly as kitchen appliances are.

    A surface whose style declares no image texture never becomes a candidate:
    ``walls_mat`` is a solid colour until ``set_wall_texture`` runs, and
    ``texplane`` is a builtin checker until ``set_floor_texture`` runs -- the
    engine only considers textures that have a ``file``.
    """
    kinds = {_FLOOR_ARENA_SURFACES.get(m.get("name")) for m in mats}
    kinds.discard(None)
    if len(kinds) != 1:
        return None
    kind = kinds.pop()
    return (kind, kind)


def _kitchen_group(tex, mats, index):
    """Group by surface, decided by the materials that use this texture.

    A texture is claimed only when every material using it agrees on one
    surface and none of them vetoes (``SKIP``). Anything ambiguous or unknown
    -- appliance meshes, the plain-colour backings behind walls and floors --
    is left alone, which is also what the built-in flag does.
    """
    kinds = {_kitchen_kind_of_material(m.get("name")) for m in mats}
    if SKIP in kinds:
        return None
    kinds.discard(None)
    if len(kinds) != 1:
        return None
    kind = kinds.pop()
    return (kind, kind) if kind in POOL_CATEGORIES else None


THOR_PROFILE = AppearanceProfile(
    name="molmospaces",
    group_of=_thor_group,
    # iTHOR names appliances after the real thing; the oven is the usual task
    # landmark, so it stays recognizable by default.
    default_exclude=("Oven",),
)

KITCHEN_PROFILE = AppearanceProfile(
    name="kitchen",
    group_of=_kitchen_group,
    fixups=KITCHEN_FIXUPS,
    default_light_jitter=0.08,
    # Kitchen textures are named after the POOL category, not the fixture, so
    # "Oven" would match nothing. Appliances are already left alone (their
    # baked mesh textures are not in the pool), so nothing needs excluding by
    # default -- pass exclude=cabinet|counter|floor|wall to pin a surface.
    default_exclude=(),
)


LIVING_ROOM_PROFILE = AppearanceProfile(
    name="living_room",
    group_of=_living_room_group,
    # Surfaces are named floor/wall here too, so "Oven" would match nothing.
    default_exclude=(),
    fixups=LIVING_ROOM_FIXUPS,
    # One directional light, same as a kitchen.
    default_light_jitter=0.08,
)


def profile_for(env_or_cls) -> AppearanceProfile:
    """Pick the profile for an env instance (or class). Raises
    :class:`UnsupportedScene` when no family matches."""
    cls = env_or_cls if isinstance(env_or_cls, type) else type(env_or_cls)
    # MolmoSpaces families (iTHOR / ProcTHOR / Holodeck) are exactly the ones
    # that load their room from a scene xml.
    if hasattr(cls, "_get_scene_xml"):
        return THOR_PROFILE
    from robocasa.environments.kitchen.kitchen import Kitchen
    from robocasa.environments.living_room.living_room import LivingRoomEnv

    if issubclass(cls, Kitchen):
        return KITCHEN_PROFILE
    if issubclass(cls, LivingRoomEnv):
        return LIVING_ROOM_PROFILE
    raise UnsupportedScene(
        f"{cls.__name__} is not a MolmoSpaces-style env (no _get_scene_xml), "
        f"a robocasa Kitchen or a LivingRoomEnv, so no appearance profile "
        f"applies; drop textures/lights/pbr from the randomize spec")
