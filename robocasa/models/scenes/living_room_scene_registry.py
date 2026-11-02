"""
Living-room layout/style registry -- the living-room analogue of
``scene_registry.py``.

Mirrors the kitchen API (``LayoutType`` / ``StyleType`` enums +
``get_layout_path`` / ``get_style_path`` + ``unpack_*``) so living-room scenes
are selected and resolved exactly like kitchen scenes, but reads from
``assets/scenes/living_room_layouts`` / ``living_room_styles`` and stays
independent of the kitchen registry.

Currently sparse (one style, two layouts) -- add enum entries + matching YAML
files to grow it.
"""

import re
from enum import IntEnum

import numpy as np
from robosuite.utils.mjcf_utils import xml_path_completion

import robocasa


class LivingRoomLayoutType(IntEnum):
    """Available living-room layouts (peer of kitchen ``LayoutType``)."""

    LAYOUT001 = 1
    LAYOUT002 = 2
    LAYOUT003 = 3
    LAYOUT004 = 4
    LAYOUT005 = 5
    LAYOUT006 = 6
    LAYOUT007 = 7
    LAYOUT008 = 8
    LAYOUT009 = 9
    LAYOUT010 = 10

    # negative values correspond to groups (see LAYOUT_GROUPS_TO_IDS)
    ALL = -3


LAYOUT_GROUPS_TO_IDS = {
    -3: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],  # all
}


class LivingRoomStyleType(IntEnum):
    """Available living-room styles (peer of kitchen ``StyleType``)."""

    STYLE001 = 1
    STYLE002 = 2
    STYLE003 = 3
    STYLE004 = 4
    STYLE005 = 5

    # negative values correspond to groups
    ALL = -3


STYLE_GROUPS_TO_IDS = {
    -3: [1, 2, 3, 4, 5],  # all
}


def get_layout_path(layout_id):
    """Yaml path for a living-room layout id (int or ``LivingRoomLayoutType``)."""
    if isinstance(layout_id, (int, np.int64, np.int32)):
        int_to_name = dict(
            map(lambda item: (item.value, item.name.lower()), LivingRoomLayoutType)
        )
        layout_name = int_to_name[int(layout_id)]
    elif isinstance(layout_id, LivingRoomLayoutType):
        layout_name = layout_id.name.lower()
    else:
        raise ValueError
    return xml_path_completion(
        f"scenes/living_room_layouts/{layout_name}.yaml",
        root=robocasa.models.assets_root,
    )


def get_style_path(style_id):
    """Yaml path for a living-room style id (int or ``LivingRoomStyleType``)."""
    if isinstance(style_id, (int, np.int64, np.int32)):
        int_to_name = dict(
            map(lambda item: (item.value, item.name.lower()), LivingRoomStyleType)
        )
        style_name = int_to_name[int(style_id)]
    elif isinstance(style_id, LivingRoomStyleType):
        style_name = style_id.name.lower()
    else:
        raise ValueError
    return xml_path_completion(
        f"scenes/living_room_styles/{style_name}.yaml",
        root=robocasa.models.assets_root,
    )


def unpack_layout_ids(layout_ids):
    if layout_ids is None:
        layout_ids = LivingRoomLayoutType.ALL
    if not isinstance(layout_ids, list):
        layout_ids = [layout_ids]

    all_ids = []
    for id in layout_ids:
        if isinstance(id, dict):
            all_ids.append(id)
            continue
        id = int(id)
        all_ids += LAYOUT_GROUPS_TO_IDS[id] if id < 0 else [id]
    return all_ids


def unpack_style_ids(style_ids):
    if style_ids is None:
        style_ids = LivingRoomStyleType.ALL
    if not isinstance(style_ids, list):
        style_ids = [style_ids]

    all_ids = []
    for id in style_ids:
        if isinstance(id, dict):
            all_ids.append(id)
            continue
        id = int(id)
        all_ids += STYLE_GROUPS_TO_IDS[id] if id < 0 else [id]
    return all_ids
