"""File-level appearance randomization, and the compatibility surface for the
names this module used to own.

The implementation moved to :mod:`robocasa.utils.appearance`, which runs on the
MERGED model xml through robosuite's ``set_xml_processor`` seam and therefore
works for every scene family (MolmoSpaces rooms AND robocasa kitchens, which
have no room xml of their own). Use ``appearance.install_appearance`` for an
env.

What stays here is the FILE-in / FILE-out form, which is still the right shape
for the one case that is not an env: re-texturing a task object's own
``model.xml`` before it is loaded (``replay_augment --randomize-object-textures``).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from robocasa.utils.appearance import (  # noqa: F401  -- re-exported
    GENERATIVE_TEXTURES_DIR,
    MAX_SCENE_LIGHTS,
    THOR_PROFILE,
    AppearanceConfig,
    randomize_xml,
    texture_pool,
    texture_split_of,
)
from robocasa.utils.appearance.config import (  # noqa: F401  -- re-exported
    LIGHT_DIR_JITTER_RAD,
    LIGHT_POS_JITTER_M,
    POOL_CATEGORIES as _POOL_CATEGORIES,
    TRAIN_SPLIT_FRACTION,
)


def randomize_scene_xml(
    src_xml: str | os.PathLike,
    rng: np.random.Generator,
    cfg: AppearanceConfig = AppearanceConfig(),
    out_xml: str | os.PathLike | None = None,
    pool: list[Path] | None = None,
    report: dict | None = None,
) -> Path:
    """Write an appearance-randomized sibling copy of ``src_xml``; return it.

    The copy lands NEXT TO the source because a standalone MJCF resolves its
    mesh paths against its own directory. ``pool`` is accepted for backwards
    compatibility and, when given, replaces the texture library for this call.
    """
    src_xml = Path(src_xml).resolve()
    if out_xml is None:
        out_xml = src_xml.with_name(src_xml.stem + ".aug.xml")
    out_xml = Path(out_xml)
    if out_xml.parent != src_xml.parent:
        raise ValueError(
            f"out_xml must live next to the source xml (relative mesh paths "
            f"resolve against it): {out_xml} vs {src_xml.parent}"
        )
    randomized = randomize_xml(src_xml.read_text(), rng, cfg,
                               profile=THOR_PROFILE, pool=pool, report=report)
    out_xml.write_text(randomized)
    return out_xml


def cleanup_variant_xml(variant_xml: str | os.PathLike) -> None:
    """Remove a variant xml and the ``.robocasa.xml`` preprocess_scene_xml
    writes next to it. Missing files are fine."""
    variant_xml = Path(variant_xml)
    companion = variant_xml.with_name(variant_xml.stem + ".robocasa.xml")
    for p in (variant_xml, companion):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
