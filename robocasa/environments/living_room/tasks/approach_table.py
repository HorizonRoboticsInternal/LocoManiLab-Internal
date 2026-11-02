"""
DeskScene: a living-room scene with a desk (origin_table), a sofa, and two
objects (water bottle + potato) on the desk.

Scene: layout 7 + style 4. The task has no success condition (free-roam demo).
"""

import numpy as np

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures import FurnitureType


class ApproachTable(LivingRoomEnv):
    HOLD_STEPS = 10
    SUCCESS_RADIUS = 1.0

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 8)
        kwargs.setdefault("style_ids", 4)
        kwargs.setdefault("robot_base_pos", (-2.2, 0.0, 0.0))
        self._in_range_steps = 0
        super().__init__(*args, **kwargs)

    def _setup_furniture_references(self):
        self.origin_table = self.register_furniture_ref(
            "origin_table", dict(id="origin_table")
        )

    def _robot_base_xy(self):
        """Current planar position of the robot's base (works for any robot)."""
        root_body = self.robots[0].robot_model.root_body
        return np.array(self.sim.data.get_body_xpos(root_body))[:2]

    def success_distance(self):
        """Current planar distance from the robot base to the origin table center."""
        return float(np.linalg.norm(self._robot_base_xy() - self.origin_table.pos[:2]))

    def _post_action(self, action):
        # advance the hold counter exactly once per env step
        ret = super()._post_action(action)
        if self.success_distance() < self.SUCCESS_RADIUS:
            self._in_range_steps += 1
        else:
            self._in_range_steps = 0
        return ret

    def _setup_scene(self):
        super()._setup_scene()
        # reset the hold counter at the start of every episode
        self._in_range_steps = 0

    def _check_success(self):
        # free-roam demo — no success condition
        return self._in_range_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "Walk over to the table and stand in front of it"
        return ep_meta
