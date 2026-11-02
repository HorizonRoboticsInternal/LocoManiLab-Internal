"""
CarryObjectToTable: a pick-and-carry task for the living room.

The object (a mug) starts on one side table and must end up on a second side
table at the far corner of the room. Success is judged purely on the *object's*
final position (it reaches the target table top), independent of how it got
there -- so the task definition is complete and real even though G1+SONIC
(keyboard, locomotion-only) cannot grasp yet. When PICO VR upper-body teleop is
wired up later, the same task becomes completable with no changes here.

Scene: layout 4 + style 3 (red-wood floor / light-gray walls). Two side tables
(reused lightwheel stool meshes) at opposite corners + a sofa for context.
"""

import numpy as np

from robocasa.environments.living_room.living_room import LivingRoomEnv


class CarryObjectToTable(LivingRoomEnv):
    # success when the object is within this 3D distance of the target table-top
    # center (i.e. it has been placed on the far table near its middle)
    TARGET_RADIUS = 0.35

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("layout_ids", 4)
        kwargs.setdefault("style_ids", 3)
        # spawn the robot near the origin table (where the object starts)
        kwargs.setdefault("robot_base_pos", (-0.5, -1.0, 0.0))
        super().__init__(*args, **kwargs)

    def _setup_furniture_references(self):
        # origin (object starts here) and target (object must reach here) tables,
        # looked up by name (both are FurnitureType.SIDE_TABLE)
        self.origin_table = self.register_furniture_ref(
            "origin_table", dict(id="origin_table")
        )
        self.target_table = self.register_furniture_ref(
            "target_table", dict(id="target_table")
        )

    def _get_obj_cfgs(self):
        # spawn a mug on the origin table top (reuses kitchen's object + sampler)
        return [
            dict(
                name="mug",
                obj_groups="mug",
                placement=dict(fixture=self.origin_table, size=(0.2, 0.2)),
            )
        ]

    def _target_point(self):
        """Center of the target table top, in world coords."""
        bbox = self.target_table._regions.get("bbox", None)
        top_rel = bbox["pz"][2] if bbox is not None else 0.4
        p = self.target_table.pos
        return np.array([p[0], p[1], p[2] + top_rel])

    def _obj_pos(self):
        return np.array(self.sim.data.get_joint_qpos(self.objects["mug"].joints[0])[:3])

    def success_distance(self):
        """3D distance from the object to the target table-top center."""
        return float(np.linalg.norm(self._obj_pos() - self._target_point()))

    def _check_success(self):
        # the mug has been carried onto the far table
        return bool(self.success_distance() < self.TARGET_RADIUS)

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = (
            "pick up the mug from the near side table and carry it to the side "
            "table across the room"
        )
        return ep_meta
