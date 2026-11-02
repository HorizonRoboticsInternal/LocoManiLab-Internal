"""
ApproachSofa: a navigation task for the living room.

The first *complete* living-room task (unlike the placeholder TidyLivingRoom, it
has a real success condition). It is locomotion-only on purpose, so the G1 +
SONIC whole-body controller (keyboard locomotion) can do it: the robot spawns
across the room and must walk over to the sofa. Success is purely a distance
check on the robot's base, so it works for any robot (PandaOmron, G1, ...).

Scene: layout 3 + style 2 (marble floor / white-brick walls) -- visually
distinct from TidyLivingRoom. The sofa sits against the -x wall facing the room;
coffee table and TV are off to the side leaving a clear lane to walk down.
"""

import numpy as np

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures import FurnitureType


class ApproachSofa(LivingRoomEnv):
    # success when the robot base is within this planar distance of the sofa
    # center (sofa is against a wall, so the only approachable side is the front)
    SUCCESS_RADIUS = 1.0
    # ...and stays within range this many consecutive steps (so merely passing
    # through the zone does not count -- the robot must stop and stand there)
    HOLD_STEPS = 10

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 3)
        kwargs.setdefault("style_ids", 2)
        # spawn the robot across the room (+x side) so it has to walk to the sofa
        kwargs.setdefault("robot_base_pos", (1.8, -0.8, 0.0))
        self._in_range_steps = 0
        super().__init__(*args, **kwargs)

    def _setup_furniture_references(self):
        # the destination: the sofa (claimed by semantic type)
        self.sofa = self.register_furniture_ref(
            "sofa", dict(id=FurnitureType.SOFA)
        )

    def _setup_scene(self):
        super()._setup_scene()
        # reset the hold counter at the start of every episode
        self._in_range_steps = 0

    def _robot_base_xy(self):
        """Current planar position of the robot's base (works for any robot)."""
        root_body = self.robots[0].robot_model.root_body
        return np.array(self.sim.data.get_body_xpos(root_body))[:2]

    def success_distance(self):
        """Current planar distance from the robot base to the sofa center."""
        return float(np.linalg.norm(self._robot_base_xy() - self.sofa.pos[:2]))

    def _post_action(self, action):
        # advance the hold counter exactly once per env step
        ret = super()._post_action(action)
        if self.success_distance() < self.SUCCESS_RADIUS:
            self._in_range_steps += 1
        else:
            self._in_range_steps = 0
        return ret

    def _check_success(self):
        # reached + held the sofa front for HOLD_STEPS consecutive steps
        return self._in_range_steps >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "walk over to the sofa and stand in front of it"
        return ep_meta
