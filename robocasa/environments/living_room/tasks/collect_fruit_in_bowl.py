"""
CollectFruitInBowl: gather an apple into a bowl across the room.

An apple starts on the left side table; a bowl sits on the coffee table.
The robot must pick the apple, carry it across the room, and place it *in*
the bowl — not just on the table.  Success uses ``check_obj_in_receptacle``
(contact + centering), which requires actual grasping.

Scene: layout 10 (symmetric, two named side tables) + style 5 (dark wood /
warm white walls).
"""

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures import FurnitureType
from robocasa.utils import object_utils as OU


class CollectFruitInBowl(LivingRoomEnv):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 10)
        kwargs.setdefault("style_ids", 5)
        kwargs.setdefault("robot_base_pos", (0.0, -0.5, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_furniture_references(self):
        self.origin_table = self.register_furniture_ref(
            "origin_table", dict(id="left_table")
        )
        self.coffee_table = self.register_furniture_ref(
            "coffee_table", dict(id=FurnitureType.COFFEE_TABLE)
        )

    def _get_obj_cfgs(self):
        return [
            dict(
                name="apple",
                obj_groups="apple",
                placement=dict(
                    fixture=self.origin_table,
                    size=(0.15, 0.15),
                ),
            ),
            dict(
                name="bowl",
                obj_groups="bowl",
                placement=dict(
                    fixture=self.coffee_table,
                    size=(0.30, 0.30),
                ),
            ),
        ]

    def _check_success(self):
        return bool(OU.check_obj_in_receptacle(self, "apple", "bowl"))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "pick up the apple from the left side table and place it in "
            "the bowl on the coffee table"
        )
        return ep_meta
