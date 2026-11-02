"""Task-free G1 environment for the empty TeleLab room."""

from pathlib import Path

from robocasa.environments.empty.empty_env import EmptyEnv
from robocasa.models.arenas.telelab_arena import TeleLabArena


_ASSET_ROOT = Path(__file__).resolve().parents[2] / "models" / "assets"
_DEFAULT_SCENE_XML = _ASSET_ROOT / "scenes" / "teleai-lab" / "empty_scene.xml"


class TeleLabEnv(EmptyEnv):
    """Load the empty TeleLab room and a separately controlled robot.

    The base environment contains only the authored floor, walls, and lights.
    Scene-specific equipment belongs in subclasses such as
    :class:`TeleLabConveyor`.

    Args:
        scene_xml: Absolute path, or a path relative to RoboCasa's assets
            directory, of the empty lab scene to load.
    """

    def __init__(
        self,
        *args,
        scene_xml=_DEFAULT_SCENE_XML,
        robot_base_pos=(0.0, -1.35, 0.0),
        robot_base_ori=(0.0, 0.0, 0.0),
        **kwargs,
    ):
        scene_path = Path(scene_xml).expanduser()
        if not scene_path.is_absolute():
            scene_path = _ASSET_ROOT / scene_path
        self.scene_xml = scene_path.resolve()
        super().__init__(
            *args,
            robot_base_pos=robot_base_pos,
            robot_base_ori=robot_base_ori,
            **kwargs,
        )

    def _make_arena(self):
        return TeleLabArena(self.scene_xml)

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def get_ep_meta(self):
        return {
            "lang": "freely explore the empty TeleLab real-to-sim room",
            "scene_xml": str(self.scene_xml),
        }
