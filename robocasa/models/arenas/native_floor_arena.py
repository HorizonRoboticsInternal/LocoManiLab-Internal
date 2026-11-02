import robocasa.models
from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import xml_path_completion


class NativeFloorArena(Arena):
    """Asset-free arena containing one native MuJoCo ground plane.

    The XML uses only MuJoCo builtin textures plus fixed lights and cameras. It
    deliberately contains no walls, meshes, furniture, or file-backed assets,
    making it the smallest useful scene for whole-body teleoperation checks.
    """

    def __init__(self):
        super().__init__(
            xml_path_completion(
                "arenas/native_floor_arena.xml", root=robocasa.models.assets_root
            )
        )
