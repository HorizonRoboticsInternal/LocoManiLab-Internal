"""
Living-room arena -- the living-room analogue of ``KitchenArena``.

Reads a (layout, style) pair, applies the style's floor texture, and builds the
furniture via ``create_furniture``. Subclasses ``FloorArena`` (collidable floor
+ perimeter walls) rather than kitchen's empty-arena-shell-plus-wall-fixtures
approach, since living-room walls are a fixed shell for now.
"""

import yaml

from robocasa.models.arenas import FloorArena
from robocasa.models.scenes.living_room_scene_registry import (
    get_layout_path,
    get_style_path,
)
from robocasa.models.scenes.living_room_scene_builder import create_furniture


class LivingRoomArena(FloorArena):
    """Floor + walls + furniture for a given (layout, style).

    Args:
        layout_id (int or LivingRoomLayoutType or dict): layout to load. A dict
            is treated as an already-parsed layout config.
        style_id (int or LivingRoomStyleType or dict): style to load.
        rng (np.random.Generator): forwarded to ``create_furniture``.
    """

    def __init__(self, layout_id, style_id, rng=None):
        if isinstance(layout_id, dict):
            layout_config = layout_id
        else:
            with open(get_layout_path(layout_id), "r") as f:
                layout_config = yaml.safe_load(f)

        if isinstance(style_id, dict):
            style_config = style_id
        else:
            with open(get_style_path(style_id), "r") as f:
                style_config = yaml.safe_load(f)

        super().__init__(
            floor_texture=style_config.get("floor_texture"),
            wall_texture=style_config.get("wall_texture"),
        )

        self.furniture = create_furniture(
            layout_config=layout_config,
            style_config=style_config,
            rng=rng,
        )

    def get_furniture_cfgs(self):
        """Config data for all furniture (mirrors ``get_fixture_cfgs``)."""
        cfgs = []
        for name, fxtr in self.furniture.items():
            cfgs.append(dict(name=name, model=fxtr, type="fixture"))
        return cfgs
