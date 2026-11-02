"""
FetchSnackFridge: pick a snack from the coffee table and store it in the fridge.

The robot picks an apple off the coffee table (in front of it), navigates through
the dense layout-007 scene (past the stove and dishwasher), opens the fridge door,
and places the apple on a shelf inside the fridge.

Success is the conjunction of:
  1. the fridge door is open (fridge.is_open), and
  2. the apple is inside the fridge interior (OU.obj_inside_of with partial_check).

Scene: layout 7 (dense clutter with fridge, stove, dishwasher) + style 4.
"""

import numpy as np

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures import FurnitureType
import robocasa.utils.object_utils as OU


class FetchSnackFridge(LivingRoomEnv):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 8)
        kwargs.setdefault("style_ids", 4)
        # Spawn south-east of the coffee table, facing the table.
        # The fridge is at (2.5, -2.5) — the robot must walk around the
        # coffee table then past the stove/dishwasher corridor to reach it.
        kwargs.setdefault("robot_base_pos", (-1.2, 1.2, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_furniture_references(self):
        # Coffee table by FurnitureType (used for object placement)
        self.coffee_table = self.register_furniture_ref(
            "coffee_table", dict(id=FurnitureType.COFFEE_TABLE)
        )
        # Fridge by name (Fridge is a Fixture subclass, not FurnitureType)
        self.fridge = self.register_furniture_ref("fridge", dict(id="big_fridge"))

    def _get_obj_cfgs(self):
        # Apple on the coffee table top
        return [
            dict(
                name="apple",
                obj_groups="apple",
                placement=dict(
                    fixture=self.coffee_table,
                    size=(0.15, 0.35),
                ),
            )
        ]

    def _setup_scene(self):
        # The fridge hinge needs noslip iterations to hold the door at the
        # opened angle (frictionloss > gravity torque). Robosuite's scene
        # merge drops the fixture's <option>, so set it after sim is built.
        super()._setup_scene()
        self.sim.model.opt.noslip_iterations = max(
            int(self.sim.model.opt.noslip_iterations), 10
        )

    def fridge_door_open(self):
        return bool(self.fridge.is_open(env=self))

    def apple_inside_fridge(self):
        return bool(OU.obj_inside_of(self, "apple", self.fridge, partial_check=True))

    def _check_success(self):
        return self.fridge_door_open() and self.apple_inside_fridge()

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "pick up the apple from the coffee table, walk to the fridge, "
            "open the fridge door, and place the apple on a shelf inside"
        )
        return ep_meta
