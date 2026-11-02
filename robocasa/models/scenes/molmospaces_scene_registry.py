"""Registry and classification for MolmoSpaces scenes.

MolmoSpaces ships 120 hand-crafted iTHOR scenes named ``FloorPlanN``. iTHOR's
fixed numbering encodes the room type, so we expose them grouped by type rather
than as one undifferentiated pile -- each type gets its own environment base
class (``IThorKitchenEnv``, ``IThorLivingRoomEnv``, ...).

    FloorPlan   1- 30  -> kitchen
    FloorPlan 201-230  -> living_room
    FloorPlan 301-330  -> bedroom
    FloorPlan 401-430  -> bathroom

This module does NOT import the ``molmo_spaces`` package (it lives in a separate
env): it only needs the scene .xml files on disk, found under RoboCasa's asset
root (``scenes/ithor`` and ``scenes/procthor-objaverse-train``).
"""

import os
import re
from enum import Enum


class RoomType(str, Enum):
    KITCHEN = "kitchen"
    LIVING_ROOM = "living_room"
    BEDROOM = "bedroom"
    BATHROOM = "bathroom"
    MULTI_ROOM = "multi_room"


# (inclusive_low, inclusive_high) FloorPlan id ranges per iTHOR room type.
_ROOM_TYPE_RANGES = {
    RoomType.KITCHEN: (1, 30),
    RoomType.LIVING_ROOM: (201, 230),
    RoomType.BEDROOM: (301, 330),
    RoomType.BATHROOM: (401, 430),
}

_FLOORPLAN_RE = re.compile(r"FloorPlan(\d+)")
_PROCTHOR_RE = re.compile(r"(?:train_|ProcTHORTrain)(\d+)")


def classify_floorplan(name_or_id):
    """Return the RoomType for a FloorPlan id (int) or name ('FloorPlan12').

    Raises ValueError if the id is outside the known iTHOR ranges.
    """
    if isinstance(name_or_id, int):
        n = name_or_id
    else:
        m = _FLOORPLAN_RE.search(str(name_or_id))
        if not m:
            raise ValueError(f"not a FloorPlan name: {name_or_id!r}")
        n = int(m.group(1))
    for room_type, (lo, hi) in _ROOM_TYPE_RANGES.items():
        if lo <= n <= hi:
            return room_type
    raise ValueError(f"FloorPlan{n} is outside known iTHOR room-type ranges")


def get_ithor_scenes_dir():
    """Directory holding the MolmoSpaces iTHOR scene .xml files."""
    import robocasa

    return os.path.join(robocasa.models.assets_root, "scenes", "ithor")


def get_procthor_scenes_dir():
    """Directory holding MolmoSpaces ProcTHOR-Objaverse train scene .xml files."""
    import robocasa

    return os.path.join(
        robocasa.models.assets_root, "scenes", "procthor-objaverse-train"
    )


def list_scenes(room_type=None, downloaded_only=True):
    """List available iTHOR scenes, optionally filtered by room type.

    Args:
        room_type (RoomType or str or None): restrict to this room type.
        downloaded_only (bool): only return scenes whose .xml exists on disk.
            When False, returns the full theoretical id set for the type(s).
    Returns:
        dict[str, str]: scene name ('FloorPlan12') -> absolute .xml path. The
        path may not exist if downloaded_only=False.
    """
    scenes_dir = get_ithor_scenes_dir()
    types = [RoomType(room_type)] if room_type is not None else list(_ROOM_TYPE_RANGES)

    out = {}
    if downloaded_only:
        if not os.path.isdir(scenes_dir):
            return out
        for fn in os.listdir(scenes_dir):
            # the real scene file, not the *.robocasa.xml cleaned copy
            m = re.fullmatch(r"(FloorPlan\d+)_physics\.xml", fn)
            if not m:
                continue
            name = m.group(1)
            try:
                rt = classify_floorplan(name)
            except ValueError:
                continue
            if rt in types:
                out[name] = os.path.join(scenes_dir, fn)
    else:
        for rt in types:
            lo, hi = _ROOM_TYPE_RANGES[rt]
            for n in range(lo, hi + 1):
                name = f"FloorPlan{n}"
                out[name] = os.path.join(scenes_dir, f"{name}_physics.xml")
    return out


def get_scene_xml(name_or_id):
    """Return the absolute .xml path for a FloorPlan scene (id or name)."""
    if isinstance(name_or_id, int):
        name = f"FloorPlan{name_or_id}"
    else:
        m = _FLOORPLAN_RE.search(str(name_or_id))
        name = f"FloorPlan{m.group(1)}" if m else str(name_or_id)
    return os.path.join(get_ithor_scenes_dir(), f"{name}_physics.xml")


def list_procthor_scenes(downloaded_only=True):
    """List available ProcTHOR-Objaverse train scenes.

    Returns:
        dict[str, str]: scene name (``"train_0"``) -> absolute .xml path.
    """
    scenes_dir = get_procthor_scenes_dir()
    out = {}
    if not os.path.isdir(scenes_dir):
        return out
    for fn in os.listdir(scenes_dir):
        m = re.fullmatch(r"(train_\d+)\.xml", fn)
        if not m:
            continue
        name = m.group(1)
        out[name] = os.path.join(scenes_dir, fn)
    return out


def normalize_procthor_scene_name(name_or_id):
    """Normalize an int / ``train_N`` / ``ProcTHORTrainN`` to ``train_N``."""
    if isinstance(name_or_id, int):
        return f"train_{name_or_id}"
    m = _PROCTHOR_RE.search(str(name_or_id))
    if not m:
        raise ValueError(f"not a ProcTHOR train scene name/id: {name_or_id!r}")
    return f"train_{int(m.group(1))}"


def get_procthor_scene_xml(name_or_id):
    """Return the absolute .xml path for a ProcTHOR train scene."""
    name = normalize_procthor_scene_name(name_or_id)
    return os.path.join(get_procthor_scenes_dir(), f"{name}.xml")
