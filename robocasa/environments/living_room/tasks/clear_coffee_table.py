"""
ClearCoffeeTable: multi-object sequential carry task.

An apple and a can both start on the coffee table (on opposite halves).
The robot must transport BOTH items to the side table across the room.
Success checks object-to-surface distance for each item independently.

Scene: layout 9 (L-shaped corner, obstacle-rich) + style 5 (dark wood /
warm white walls).  The diagonal sofa blocks the direct path between the
coffee table and the side table.
"""

import numpy as np

from robocasa.environments.living_room.living_room import LivingRoomEnv
from robocasa.models.fixtures import FurnitureType


class ClearCoffeeTable(LivingRoomEnv):
    TARGET_RADIUS = 0.25

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 9)
        kwargs.setdefault("style_ids", 5)
        kwargs.setdefault("robot_base_pos", (0.5, 0.5, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_furniture_references(self):
        self.coffee_table = self.register_furniture_ref(
            "coffee_table", dict(id=FurnitureType.COFFEE_TABLE)
        )
        self.side_table = self.register_furniture_ref(
            "side_table", dict(id="side_table")
        )

    def _get_obj_cfgs(self):
        return [
            dict(
                name="apple",
                obj_groups="apple",
                placement=dict(
                    fixture=self.coffee_table,
                    size=(0.30, 0.15),
                    offset=(0.0, -0.10),
                ),
            ),
            dict(
                name="can",
                obj_groups="can",
                placement=dict(
                    fixture=self.coffee_table,
                    size=(0.15, 0.15),
                    offset=(0.0, 0.10),
                ),
            ),
        ]

    def _fixture_top_center(self, fixture):
        """World-frame center of the fixture's top surface."""
        bbox = fixture._regions.get("bbox", None)
        top_z = bbox["pz"][2] if bbox is not None else 0.4
        p = fixture.pos
        return np.array([p[0], p[1], p[2] + top_z])

    def _obj_pos(self, name):
        return np.array(self.sim.data.get_joint_qpos(self.objects[name].joints[0])[:3])

    def _obj_distance(self, name, fixture):
        return float(
            np.linalg.norm(self._obj_pos(name) - self._fixture_top_center(fixture))
        )

    def _check_success(self):
        return bool(
            self._obj_distance("apple", self.side_table) < self.TARGET_RADIUS
            and self._obj_distance("can", self.side_table) < self.TARGET_RADIUS
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "clear the coffee table by moving the apple and the can to "
            "the side table"
        )
        return ep_meta
