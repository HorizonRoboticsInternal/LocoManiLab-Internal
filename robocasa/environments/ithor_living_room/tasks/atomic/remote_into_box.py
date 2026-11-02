"""Scene 226: place the native remote into the open native cardboard box."""

from robocasa.environments.ithor_living_room.tasks.atomic.statue_into_box import (
    IThorLivingRoomStatueIntoBox,
)
from robocasa.environments.ithor_living_room.ithor_living_room import (
    IThorLivingRoomEnv,
)


class IThorLivingRoomRemoteIntoBox(IThorLivingRoomStatueIntoBox):
    SUPPORTED_SCENE_IDS = (226,)
    SPEC_TASK = "remote_into_box"
    ATOMIC_SKILL = "PickPlaceRemote"

    BOX_BODY = "box_b1c47d67056cd714cb582cc351224939_1_0_0"
    BOX_JOINT = BOX_BODY + "_jntfree_0"
    BOX_BASE_BODY = "box_b1c47d67056cd714cb582cc351224939_1_4_0"
    BOX_BOTTOM_GEOM = BOX_BASE_BODY + "_collision_7"
    BOX_WALL_X_NEG = BOX_BASE_BODY + "_collision_8"
    BOX_WALL_X_POS = BOX_BASE_BODY + "_collision_9"
    BOX_WALL_Y_NEG = BOX_BASE_BODY + "_collision_10"
    BOX_WALL_Y_POS = BOX_BASE_BODY + "_collision_11"
    FLAP_JOINTS = (
        "box_b1c47d67056cd714cb582cc351224939_1_1_0_box26flapouter1joint0_1",
        "box_b1c47d67056cd714cb582cc351224939_1_2_0_box26flapouter2joint0_2",
        "box_b1c47d67056cd714cb582cc351224939_1_3_0_box26flapinner2joint0_3",
        "box_b1c47d67056cd714cb582cc351224939_1_5_0_box26flapinner1joint0_4",
    )
    OBJECT_BODY = "remotecontrol_bbeb7464bb8b20b11df78329c1bd9460_1_0_0"
    OBJECT_JOINT = OBJECT_BODY + "_jntfree_0"

    def _create_objects(self):
        """This task uses only the native Scene 226 remote and box."""
        self.objects = {}
        self.object_cfgs = []

    def _setup_baked_scene_fixtures(self):
        IThorLivingRoomEnv._setup_baked_scene_fixtures(self)

    def _setup_references(self):
        IThorLivingRoomEnv._setup_references(self)
        box_root = self.sim.model.body_name2id(self.BOX_BODY)
        object_root = self.sim.model.body_name2id(self.OBJECT_BODY)
        self._object_joint = self.OBJECT_JOINT
        self._box_geom_ids = self._collidable_geoms(self._body_tree_ids([box_root]))
        self._object_geom_ids = self._collidable_geoms(
            self._body_tree_ids([object_root])
        )
        self._box_bottom_geom_id = self.sim.model.geom_name2id(self.BOX_BOTTOM_GEOM)
        self._box_wall_geom_ids = {
            "y_pos": self.sim.model.geom_name2id(self.BOX_WALL_Y_POS),
            "y_neg": self.sim.model.geom_name2id(self.BOX_WALL_Y_NEG),
            "x_neg": self.sim.model.geom_name2id(self.BOX_WALL_X_NEG),
            "x_pos": self.sim.model.geom_name2id(self.BOX_WALL_X_POS),
        }

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.pop("replaced_scene_entity", None)
        meta.update(
            lang="place the remote control into the open cardboard box on the coffee table",
            task_id=type(self).__name__,
            target_entity="FloorPlan226:Remote_3_2",
            target_receptacle="FloorPlan226:Box_26_0",
            support="FloorPlan226:Coffee_Table_203_1_13",
            initial_orientation="standing upright with the long axis vertical",
            success_definition=(
                "the remote footprint and bottom remain inside the cardboard box "
                "continuously for 1 second"
            ),
        )
        return meta


__all__ = ["IThorLivingRoomRemoteIntoBox"]
