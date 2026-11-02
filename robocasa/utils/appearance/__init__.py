"""Appearance randomization (textures / lights / material shading).

ONE implementation for every scene family. The seam is robosuite's
``set_xml_processor`` (``install.py``); the engine is a pure
xml-string -> xml-string transform (``engine.py``); what the robot and the
task objects own is identified structurally and never touched
(``ownership.py``); and the only per-family knowledge is a single classifier
function that says which surface a texture is on (``profiles.py``).

Physics is never touched: only ``<texture file>``, ``<material>`` shading,
``<light>`` and ``<visual><headlight>`` change.
"""

from .config import (  # noqa: F401
    GENERATIVE_TEXTURES_DIR,
    MAX_SCENE_LIGHTS,
    POOL_CATEGORIES,
    AppearanceConfig,
    texture_pool,
    texture_split_of,
)
from .engine import randomize_xml  # noqa: F401
from .install import AppearanceHandle, install_appearance  # noqa: F401
from .ownership import is_protected, protected_prefixes  # noqa: F401
from .profiles import (  # noqa: F401
    KITCHEN_PROFILE,
    LIVING_ROOM_PROFILE,
    THOR_PROFILE,
    AppearanceProfile,
    UnsupportedScene,
    profile_for,
)

__all__ = [
    "AppearanceConfig", "AppearanceHandle", "AppearanceProfile",
    "GENERATIVE_TEXTURES_DIR", "KITCHEN_PROFILE", "LIVING_ROOM_PROFILE",
    "MAX_SCENE_LIGHTS",
    "POOL_CATEGORIES", "THOR_PROFILE", "UnsupportedScene", "install_appearance",
    "is_protected", "profile_for", "protected_prefixes", "randomize_xml",
    "texture_pool", "texture_split_of",
]
