import os
import xml.etree.ElementTree as ET

import robocasa.models
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion


class FloorArena(Arena):
    """
    A reusable empty arena: a collidable floor enclosed by four collidable walls.

    Unlike robosuite's ``EmptyArena`` (whose walls are visual-only), the walls
    here participate in collision, so robots cannot drive/pass through them.
    Intended as the base arena for kitchen-free custom environments.

    Available cameras: ``frontview``, ``birdview``, ``agentview``, ``sideview``.

    Args:
        floor_texture (str): optional path (relative to the asset root) to an
            image texture for the floor. If given, replaces the default checker.
        wall_texture (str): optional path (relative to the asset root) to an
            image texture for the walls. If given, replaces the solid color.
    """

    def __init__(self, floor_texture=None, wall_texture=None):
        super().__init__(
            xml_path_completion(
                "arenas/floor_arena.xml", root=robocasa.models.assets_root
            )
        )
        if floor_texture is not None:
            self.set_floor_texture(floor_texture)
        if wall_texture is not None:
            self.set_wall_texture(wall_texture)

    def set_floor_texture(self, floor_texture):
        """Point the floor material at an image texture (relative to asset root)."""
        abs_path = xml_path_completion(floor_texture, root=robocasa.models.assets_root)
        if not os.path.exists(abs_path):
            raise FileNotFoundError("floor texture not found: {}".format(abs_path))
        tex = self.asset.find("./texture[@name='texplane']")
        tex.set("type", "2d")
        tex.set("file", abs_path)
        for attr in ("builtin", "rgb1", "rgb2", "width", "height"):
            tex.attrib.pop(attr, None)

    def set_wall_texture(self, wall_texture):
        """Point the wall material at an image texture (relative to asset root).

        The default ``walls_mat`` is a solid color; this adds a 2d texture and
        switches the material to use it (mirrors how the kitchen ``Wall`` fixture
        textures its walls).
        """
        abs_path = xml_path_completion(wall_texture, root=robocasa.models.assets_root)
        if not os.path.exists(abs_path):
            raise FileNotFoundError("wall texture not found: {}".format(abs_path))
        if self.asset.find("./texture[@name='wallplane']") is None:
            ET.SubElement(
                self.asset,
                "texture",
                {"name": "wallplane", "type": "2d", "file": abs_path},
            )
        mat = self.asset.find("./material[@name='walls_mat']")
        mat.set("texture", "wallplane")
        mat.set("texrepeat", "3 3")
        mat.set("texuniform", "true")
        mat.attrib.pop("rgba", None)
