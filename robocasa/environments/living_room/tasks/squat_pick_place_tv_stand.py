"""SquatPickPlaceTVStand: squat to pick the apple off the (low) coffee table in
front of the robot and place it on the TV stand behind the robot.

Success is judged purely on the apple's final position (it reaches the TV-stand
top), the same object-position-only style as CarryObjectToTable -- the task is
complete and real even before grasping (PICO VR) is wired up; the same task
becomes completable then with no changes here.

Scene: layout 5 -- a low coffee table in front (apple start) + a TV stand behind
(target). Only two (decimated) furniture pieces, so it renders fast.
"""

import numpy as np

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures import FurnitureType


class SquatPickPlaceTVStand(LivingRoomEnv):
    # success when the apple is within this 3D distance of the TV-stand top center
    TARGET_RADIUS = 0.30

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 5)
        kwargs.setdefault("style_ids", 1)
        # robot at the origin, facing +x (toward the coffee table); the TV stand
        # is behind it at -x.
        kwargs.setdefault("robot_base_pos", (0.0, 0.0, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_furniture_references(self):
        self.coffee_table = self.register_furniture_ref(
            "coffee_table", dict(id=FurnitureType.COFFEE_TABLE)
        )
        self.tv_stand = self.register_furniture_ref(
            "tv_stand", dict(id=FurnitureType.TV_STAND)
        )

    def _get_obj_cfgs(self):
        # apple on the (low) coffee-table top -> robot squats to pick it up
        return [
            dict(
                name="apple",
                obj_groups="apple",
                placement=dict(fixture=self.coffee_table, size=(0.2, 0.2)),
            )
        ]

    def _target_point(self):
        """Center of the TV-stand top, in world coords."""
        bbox = self.tv_stand._regions.get("bbox", None)
        top_rel = bbox["pz"][2] if bbox is not None else 0.4
        p = self.tv_stand.pos
        return np.array([p[0], p[1], p[2] + top_rel])

    def _obj_pos(self):
        return np.array(
            self.sim.data.get_joint_qpos(self.objects["apple"].joints[0])[:3]
        )

    def success_distance(self):
        """3D distance from the apple to the TV-stand top center (for live readout)."""
        return float(np.linalg.norm(self._obj_pos() - self._target_point()))

    def _check_success(self):
        return bool(self.success_distance() < self.TARGET_RADIUS)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "squat down, pick up the apple from the coffee table, and place it "
            "on the tv stand behind you"
        )
        return ep_meta
