"""Scene 315 navigation and can placement at the native wastebasket."""

import numpy as np

from robocasa.environments.ithor_atomic.navigation import ExactSceneNavigation
from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_specs,
    get_supported_scene_ids,
)
from robocasa.utils import env_utils as EnvUtils

from .assets import CAN_MODEL, CAN_SCALE, asset_xml_path


class BedroomTrashCanLayoutMixin:
    """Use the native bin in its original scene pose in both tasks."""

    INITIAL_POSE_JITTER = False

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.bin_fixture = self.require_fixture(spec["fixture"])
        self.bin_region = self.bin_fixture.regions[spec["bin_region"]]
        self.fixture_refs["trash_can"] = self.bin_fixture
        self.region_refs["trash_can_interior"] = self.bin_region

    def _calibrate_robot_free_joint_height(self):
        """Ground G1 before checking its clearance from scene fixtures."""
        joint = self._get_robot_free_joint()
        if joint is None:
            return False
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[2] = float(self.scene_task_spec()["robot_root_height"])
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.forward()
        address = self.sim.model.get_joint_qpos_addr(joint)[0]
        if EnvUtils._snap_freejoint_to_ground(self, address) is None:
            return False
        self._robot_free_joint_z = float(self.sim.data.get_joint_qpos(joint)[2])
        self._robot_default_free_joint_z = self._robot_free_joint_z
        return self.count_body_collisions() == 0

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["target_fixture_id"] = self.bin_fixture.stable_id
        meta["requires_grasp"] = False
        return meta


class IThorBedroomApproachTrashCan(
    BedroomTrashCanLayoutMixin, ExactSceneNavigation, IThorBedroomEnv
):
    """Walk within 35 cm of the native bin and remain there for one second."""

    SPEC_TASK = "approach_trash_can"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = {
        scene: dict(spec, lang="walk to the trash can")
        for scene, spec in get_ithor_bedroom_task_specs(SPEC_TASK).items()
    }


class IThorBedroomCanIntoTrashCan(
    BedroomTrashCanLayoutMixin,
    SupportContactPlacementMixin,
    SceneSpecMixin,
    IThorBedroomEnv,
):
    """Keep the can touching the bin bottom for one second, in any orientation."""

    SPEC_TASK = "can_into_trash_can"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = {
        scene: dict(spec, lang="put the can into the trash can")
        for scene, spec in get_ithor_bedroom_task_specs(SPEC_TASK).items()
    }
    TASK_FAMILY = "receptacle"
    ATOMIC_SKILL = "PlaceIntoReceptacle"
    DIFFICULTY_LEVEL = "L1"
    SOURCE_NAME, SOURCE_GROUP = "item", "can"
    SOURCE_MJCF_PATH = asset_xml_path(CAN_MODEL)
    SOURCE_SCALE = CAN_SCALE

    def _get_scene_robot_pose_calibration(self):
        pose = self.scene_task_spec()["robot_pose"]
        return np.array([*pose["xy"], 0.0]), float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.bin_bottom_geoms = tuple(spec["bin_bottom_geoms"])
        self.source_fixture = self.require_fixture(spec["source_fixture"])
        self.source_region = self.source_fixture.regions[spec["source_region"]]
        self.fixture_refs["source_surface"] = self.source_fixture
        self.region_refs["source_surface"] = self.source_region

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups=self.SOURCE_GROUP,
                info={"mjcf_path": self.SOURCE_MJCF_PATH},
                object_scale=self.SOURCE_SCALE,
                placement=dict(
                    region=self.source_region, strategy="center", rotation=0.0
                ),
            )
        ]

    def _setup_scene(self):
        super()._setup_scene()
        self._reset_task_progress()
        position, yaw = self._get_scene_robot_pose_calibration()
        if not (
            self.robot_footprint_is_on_floor(position[:2], radius=0.25)
            and self._set_robot_free_pose(position[:2], yaw)
            and self._calibrate_robot_free_joint_height()
            and self.count_body_collisions() == 0
        ):
            raise RuntimeError(f"{self.scene_name}: invalid can manipulation pose")
        self.robot_base_pos = position
        self.robot_base_ori = np.array([0.0, 0.0, yaw])

    def _apply_object_placements(self, placements):
        super()._apply_object_placements(placements)
        if not self.place_object_on_support_contact(
            self.SOURCE_NAME,
            self.source_region,
            fixture=self.source_fixture,
            xy=self.scene_task_spec()["source_xy"],
            inward_fractions=(0.0,),
            probe_step=0.001,
        ):
            raise RuntimeError(f"{self.scene_name}: can has no contacted desk pose")

    def _reset_task_progress(self):
        self._success_hold = 0

    def _can_contacts_trash_can_bottom(self):
        return self.check_contact(
            self.objects[self.SOURCE_NAME].contact_geoms, self.bin_bottom_geoms
        )

    def _post_action(self, action):
        self._success_hold = (
            self._success_hold + 1 if self._can_contacts_trash_can_bottom() else 0
        )
        return super()._post_action(action)

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = self.scene_task_spec()["lang"]
        meta["source_model"] = CAN_MODEL
        meta["source_scale"] = self.SOURCE_SCALE
        meta["source_position"] = self.sim.data.get_body_xpos(
            self.objects[self.SOURCE_NAME].root_body
        ).tolist()
        meta["target_contact_geoms"] = list(self.bin_bottom_geoms)
        meta["success_requires_release"] = False
        meta["success_definition"] = (
            "the can must remain in contact with the trash can bottom "
            f"for {self.SUCCESS_HOLD_SECONDS:.1f} seconds, "
            "in any orientation, independent of grasp or release"
        )
        return meta


__all__ = ["IThorBedroomApproachTrashCan", "IThorBedroomCanIntoTrashCan"]
