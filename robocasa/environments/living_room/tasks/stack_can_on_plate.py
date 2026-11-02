"""
StackCanOnPlate: precision receptacle placement task.

A can starts on the coffee table; a plate is already on the right side table.
The robot must pick up the can, carry it across the room, and place it onto
the plate.  Success uses contact + centering (``check_obj_in_receptacle``),
which requires actual grasping.

Scene: layout 10 (symmetric, two named side tables) + style 5 (dark wood /
warm white walls).
"""

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures import FurnitureType
from robocasa.utils import object_utils as OU


class StackCanOnPlate(LivingRoomEnv):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 10)
        kwargs.setdefault("style_ids", 5)
        kwargs.setdefault("robot_base_pos", (0.0, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_furniture_references(self):
        self.coffee_table = self.register_furniture_ref(
            "coffee_table", dict(id=FurnitureType.COFFEE_TABLE)
        )
        self.target_table = self.register_furniture_ref(
            "target_table", dict(id="right_table")
        )

    def _get_obj_cfgs(self):
        return [
            dict(
                name="can",
                obj_groups="can",
                placement=dict(
                    fixture=self.coffee_table,
                    size=(0.15, 0.15),
                ),
            ),
            dict(
                name="plate",
                obj_groups="plate",
                placement=dict(
                    fixture=self.target_table,
                    size=(0.25, 0.25),
                ),
            ),
        ]

    def _check_success(self):
        return bool(OU.check_obj_in_receptacle(self, "can", "plate"))

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "pick up the can from the coffee table and place it on the "
            "plate on the right side table"
        )
        return ep_meta
