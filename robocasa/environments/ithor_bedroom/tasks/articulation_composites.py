"""Bedroom composites built from existing navigation and articulation atomics."""

from __future__ import annotations

import copy

import numpy as np

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
)

from .articulation import _BedroomTopDrawerMixin
from .articulation_base import (
    BedroomArticulationMixin,
    BedroomTwentyEightyArticulationMixin,
)
from .box_flap import BedroomBoxFlapMixin
from .hamper_lid import _FloorHamperMixin
from .laptop import BedroomLaptopMixin
from .safe_door import _BedsideSafeMixin, _RaisedSafeMixin


def _action_specs(scene_ids, action_task, lang, navigation_task=None):
    specs = {}
    for scene_id in scene_ids:
        scene_name = f"FloorPlan{scene_id}"
        action = get_ithor_bedroom_task_spec(
            scene_name,
            action_task,
            required=("joint", "target", "lang"),
        )
        spec = copy.deepcopy(action)
        spec["lang"] = lang
        if navigation_task is not None:
            navigation = get_ithor_bedroom_task_spec(
                scene_name,
                navigation_task,
                required=(
                    "fixture",
                    "start_pose",
                    "minimum_start_distance",
                    "remove_fixtures",
                ),
            )
            spec["navigation"] = copy.deepcopy(navigation)
        specs[scene_name] = spec
    return specs


def _box_flap_approach_specs(scene_ids, lang):
    """Build private navigation setups for the floor-box composites.

    The generic ``approach_storage`` atomics point at a different fixture in
    four scenes. Reuse each scene's collision-calibrated remote spawn, but
    gate this composite on the relocated box flap itself. This keeps the
    atomic navigation configs and every other composite unchanged.
    """

    start_task_by_scene = {
        302: "approach_hamper_lid",
        306: "approach_bed",
        310: "approach_bed",
        323: "approach_hamper_lid",
        329: "approach_storage",
    }
    specs = _action_specs(scene_ids, "box_flap", lang)
    for scene_id in scene_ids:
        scene_name = f"FloorPlan{scene_id}"
        navigation = get_ithor_bedroom_task_spec(
            scene_name,
            start_task_by_scene[scene_id],
            required=(
                "start_pose",
                "minimum_start_distance",
                "remove_fixtures",
            ),
        )
        # The BoxFlap mixin separately removes only the action's calibrated
        # obstructions. The approach setup must not remove the box target.
        navigation.update(
            fixture=specs[scene_name]["fixture"],
            remove_fixtures=[],
        )
        if scene_id == 310:
            # FloorPlan310 reuses the bed's safe remote spawn. Unlike its
            # source navigation task, this composite must initially face the
            # relocated box at the calibrated floor position.
            start_xy = np.asarray(navigation["start_pose"]["xy"], dtype=float)
            box_xy = np.asarray(specs[scene_name]["box_floor_xy"], dtype=float)
            navigation["start_pose"]["yaw"] = float(
                np.arctan2(box_xy[1] - start_xy[1], box_xy[0] - start_xy[0])
            )
        specs[scene_name]["navigation"] = navigation
    return specs


class _ApproachThenArticulationMixin(CompositeStageReporterMixin):
    """Keep an articulation at reset until its exact navigation gate is met."""

    ATOMIC_SKILL = "NavigateThenArticulate"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "navigate_then_articulate"
    COMPOSITE_ORDERING = ("approach_fixture", "operate_joint")
    APPROACH_HOLD_STEPS = 5
    HOLD_STEPS = 5
    SUCCESS_DISTANCE = 0.35
    INITIAL_JOINT_FRACTION_TOLERANCE = 0.03
    INITIAL_ROBOT_JITTER_RADIUS = 0.0

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        navigation = self.scene_task_spec()["navigation"]
        self.approach_fixture = self.get_fixture(id=navigation["fixture"])
        if (
            self.approach_fixture is None
            or self.approach_fixture.stable_id != navigation["fixture"]
        ):
            raise LookupError(
                f"{self.scene_name}: missing exact approach fixture "
                f"{navigation['fixture']}"
            )
        removal_ids = navigation["remove_fixtures"]
        if not isinstance(removal_ids, (list, tuple)) or isinstance(removal_ids, str):
            raise ValueError("navigation remove_fixtures must be a list")
        removals = []
        for stable_id in removal_ids:
            fixture = self.get_fixture(id=stable_id)
            if fixture is None or fixture.stable_id != stable_id:
                raise LookupError(
                    f"{self.scene_name}: missing approach removal {stable_id}"
                )
            if fixture.stable_id == self.target_fixture.stable_id:
                raise ValueError("an articulation target cannot be removed")
            if fixture.stable_id == self.approach_fixture.stable_id:
                raise ValueError("an approach target cannot be removed")
            removals.append(fixture)
        if removals:
            self.remove_semantic_fixtures(removals)
        self.removed_approach_fixture_ids = list(removal_ids)
        pose = navigation["start_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if xy.shape != (2,) or not np.all(np.isfinite(xy)):
            raise ValueError(f"{self.scene_name}: invalid approach start pose")
        self.set_robot_base_pose([xy[0], xy[1], 0.0], yaw)

    def _setup_scene(self):
        super()._setup_scene()
        navigation = self.scene_task_spec()["navigation"]
        pose = navigation["start_pose"]
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        if not (
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
        ):
            raise RuntimeError(f"{self.scene_name}: invalid approach start pose")
        self.sim.forward()
        if self.count_body_collisions():
            raise RuntimeError(f"{self.scene_name}: approach start collides")
        distance = self.robot_fixture_collision_distance(
            self.approach_fixture, max_distance=5.0
        )
        minimum = min(0.70, float(navigation["minimum_start_distance"]))
        if not np.isfinite(distance) or distance < minimum:
            raise RuntimeError(
                f"{self.scene_name}: approach starts only {distance:.3f} m "
                "from its exact approach fixture"
            )
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._approach_start_distance = float(distance)

    def _reset_internal(self):
        super()._reset_internal()
        self._approach_initial_fraction = float(
            self.target_semantic_joint.fraction(self.sim)
        )
        self._approach_stage = 0
        self._approach_hold = 0
        self._operation_hold = 0

    def _near_articulation(self):
        distance = self.robot_fixture_collision_distance(
            self.target_fixture, max_distance=1.0
        )
        return bool(np.isfinite(distance) and distance <= self.SUCCESS_DISTANCE)

    def _initial_articulation_intact(self):
        fraction = float(self.target_semantic_joint.fraction(self.sim))
        return bool(
            abs(fraction - self._approach_initial_fraction)
            <= self.INITIAL_JOINT_FRACTION_TOLERANCE
            and not self._raw_success()
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        near = self._near_articulation()
        if self._approach_stage == 0:
            ready = near and self._initial_articulation_intact()
            self._approach_hold = self._approach_hold + 1 if ready else 0
            if self._approach_hold >= self.APPROACH_HOLD_STEPS:
                self._approach_stage = 1
                self._report_stage_complete(1, 2, next_stage="operate_the_named_joint")
            self._operation_hold = 0
            return ret
        valid = near and self._raw_success()
        self._operation_hold = self._operation_hold + 1 if valid else 0
        return ret

    def _check_success(self):
        return self._approach_stage == 1 and self._operation_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            {
                "lang": self.SCENE_SPECS[self.scene_name]["lang"],
                "start_distance": self._approach_start_distance,
                "approach_distance": self.SUCCESS_DISTANCE,
                "removed_approach_fixtures": list(self.removed_approach_fixture_ids),
                "stage_order": [
                    "approach the exact fixture while its joint stays at reset",
                    "operate the named joint while remaining near the fixture",
                ],
            }
        )
        return ep_meta


class _OpenThenCloseArticulationMixin(CompositeStageReporterMixin):
    """Open one exact joint, then return it to its closed threshold."""

    ATOMIC_SKILL = "OpenThenCloseArticulation"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "composite"
    COMPOSITE_KIND = "articulation_cycle"
    COMPOSITE_ORDERING = ("open_joint", "close_joint")
    FIRST_HOLD_STEPS = 5
    HOLD_STEPS = 5
    BEHAVIOR = "open"

    def _reset_internal(self):
        super()._reset_internal()
        self._cycle_stage = 0
        self._cycle_open_hold = 0
        self._cycle_close_hold = 0

    def _joint_matches(self, behavior):
        fractions = [joint.fraction(self.sim) for joint in self._success_joints()]
        if behavior == "open":
            return all(fraction >= self.SUCCESS_OPEN_FRACTION for fraction in fractions)
        return all(fraction <= self.SUCCESS_CLOSE_FRACTION for fraction in fractions)

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._cycle_stage == 0:
            opened = self._joint_matches("open")
            self._cycle_open_hold = self._cycle_open_hold + 1 if opened else 0
            if self._cycle_open_hold >= self.FIRST_HOLD_STEPS:
                self._cycle_stage = 1
                self._report_stage_complete(1, 2, next_stage="close_the_joint")
            self._cycle_close_hold = 0
            return ret
        closed = self._joint_matches("close")
        self._cycle_close_hold = self._cycle_close_hold + 1 if closed else 0
        return ret

    def _check_success(self):
        return self._cycle_stage == 1 and self._cycle_close_hold >= self.HOLD_STEPS

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            {
                "lang": self.SCENE_SPECS[self.scene_name]["lang"],
                "stage_order": [
                    "open the named joint to its atomic-task threshold",
                    "close the same joint to its atomic-task threshold",
                ],
            }
        )
        return ep_meta


class _ResettableOpenThenCloseArticulationMixin(_OpenThenCloseArticulationMixin):
    """Clear cycle progress when viewers restore a fixed physical state."""

    def _reset_task_progress(self):
        # Fixed-scene collection and visualization bypass _reset_internal().
        self._success_hold = 0
        self._cycle_stage = 0
        self._cycle_open_hold = 0
        self._cycle_close_hold = 0


def _approach_desk_laptop_specs(scene_ids, lang):
    """Use FloorPlan329's Laptop_13 only in the four native-laptop composites."""

    specs = _action_specs(scene_ids, "laptop", lang, navigation_task="approach_desk")
    reference = get_ithor_bedroom_task_spec("FloorPlan329", "laptop")
    for scene_id in (302, 306, 310, 323):
        scene_name = f"FloorPlan{scene_id}"
        spec = specs[scene_name]
        native_fixture = spec["fixture"]
        surface_task = "grasp_workspace" if scene_id in (302, 323) else "workspace"
        surface = get_ithor_bedroom_task_spec(
            scene_name, surface_task, required=("fixture", "region")
        )
        spawn_name = "approach_desk_laptop"
        spec.update(
            fixture_mode="spawn_asset",
            replace_fixture=native_fixture,
            fixture=f"{scene_name}:spawned:{spawn_name}",
            asset=reference["asset"],
            asset_id=reference["asset_id"],
            spawn_name=spawn_name,
            spawn_root_pos=copy.deepcopy(reference["spawn_root_pos"]),
            spawn_quat=copy.deepcopy(reference["spawn_quat"]),
            spawn_scale=reference["spawn_scale"],
            surface_fixture=surface["fixture"],
            surface_region=surface["region"],
            joint=f"{spawn_name}_Laptop_13_laptop_lid_13_joint_0",
        )
    return specs


class _ApproachDeskLaptopMixin(BedroomLaptopMixin):
    """Replace native laptops only in these two approach composites."""

    def _setup_task_fixtures(self):
        native_id = self.scene_task_spec().get("replace_fixture")
        if native_id is not None:
            native = self.get_fixture(id=native_id)
            if native is None or native.stable_id != native_id:
                raise LookupError(
                    f"{self.scene_name}: missing native laptop {native_id}"
                )
            self.remove_semantic_fixtures((native,))
        super()._setup_task_fixtures()

    def _workspace_spec(self):
        workspace = super()._workspace_spec()
        if self.scene_name in ("FloorPlan302", "FloorPlan323"):
            desk = get_ithor_bedroom_task_spec(
                self.scene_name,
                "grasp_workspace",
                required=("fixture", "region", "seating", "remove_fixtures"),
            )
            workspace.update(
                fixture=desk["fixture"],
                region=desk["region"],
                laptop_seating=desk["seating"],
                remove_fixtures=desk["remove_fixtures"],
            )
        native_id = self.scene_task_spec().get("replace_fixture")
        if native_id is not None:
            workspace["remove_fixtures"] = [
                fixture_id
                for fixture_id in workspace.get("remove_fixtures", ())
                if fixture_id != native_id
            ]
        return workspace


class IThorBedroomApproachDeskAndOpenLaptop(
    _ApproachThenArticulationMixin,
    _ApproachDeskLaptopMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "laptop"
    SCENE_SPECS = _approach_desk_laptop_specs(
        (302, 306, 310, 323, 329), "walk to the desk and open the laptop"
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    BEHAVIOR = "open"


class IThorBedroomApproachDeskAndCloseLaptop(
    _ApproachThenArticulationMixin,
    _ApproachDeskLaptopMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "laptop"
    SCENE_SPECS = _approach_desk_laptop_specs(
        (302, 306, 310, 323, 329), "walk to the desk and close the laptop"
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    BEHAVIOR = "close"


class IThorBedroomApproachStorageAndOpenSafeDoor(
    _ApproachThenArticulationMixin,
    BedroomTwentyEightyArticulationMixin,
    _BedsideSafeMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "open_safe_door"
    SCENE_SPECS = _action_specs(
        (302, 323),
        "open_safe_door",
        "walk to the safe and open its door",
        navigation_task="approach_storage",
    )
    # FloorPlan323's generic storage target is the dresser, while this
    # composite relocates and operates its calibrated safe. Keep the remote
    # spawn but gate the approach on that exact safe instead.
    SCENE_SPECS["FloorPlan323"]["navigation"].update(
        fixture=SCENE_SPECS["FloorPlan323"]["fixture"],
        start_pose={"xy": [0.30, -2.85], "yaw": 2.400676},
        minimum_start_distance=2.75,
        lang="walk to the safe",
    )
    SUPPORTED_SCENE_IDS = (302, 323)
    BEHAVIOR = "open"


class IThorBedroomApproachStorageAndCloseSafeDoor(
    _ApproachThenArticulationMixin,
    BedroomTwentyEightyArticulationMixin,
    _RaisedSafeMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "safe_door"
    SCENE_SPECS = _action_specs(
        (302, 323),
        "safe_door",
        "walk to the safe and close its door",
        navigation_task="approach_storage",
    )
    # Keep the raised calibration pose clear of the bed.
    SCENE_SPECS["FloorPlan302"]["navigation"]["start_pose"] = {
        "xy": [0.40, -0.80],
        "yaw": 2.018316,
    }
    SCENE_SPECS["FloorPlan302"]["navigation"]["minimum_start_distance"] = 0.60
    SCENE_SPECS["FloorPlan302"]["navigation"]["remove_fixtures"] = [
        "FloorPlan302:Tennis_Racquet_1_13",
    ]
    # FloorPlan323's generic storage target is the dresser, while this
    # composite raises and operates its calibrated safe on a pedestal.
    SCENE_SPECS["FloorPlan323"]["navigation"].update(
        fixture=SCENE_SPECS["FloorPlan323"]["fixture"],
        start_pose={"xy": [0.30, -2.85], "yaw": 1.368820},
        minimum_start_distance=3.70,
        lang="walk to the safe",
    )
    SUPPORTED_SCENE_IDS = (302, 323)
    BEHAVIOR = "close"


class _ApproachHamperLidMixin(_ApproachThenArticulationMixin):
    """Require arrival on the operable side of a spawned hamper lid."""

    HAMPER_OPERATION_RADIUS = 0.30

    def _near_articulation(self):
        pose = self.scene_task_spec()["navigation"].get("operation_pose")
        joint_state_only = self.scene_name in (
            "FloorPlan302",
            "FloorPlan310",
            "FloorPlan323",
            "FloorPlan329",
        )
        far_side_arrival = self.scene_name in ("FloorPlan323", "FloorPlan329")

        # In these spawned-hamper scenes, only the first stage is navigation.
        # Once it is complete, joint state alone determines the second-stage
        # result; do not retain a robot-distance or operation-pose gate.
        if joint_state_only and self._approach_stage == 1:
            return True

        if not super()._near_articulation():
            return False
        if far_side_arrival:
            return True
        if pose is None:
            return True
        return bool(
            np.linalg.norm(self._robot_base_xy() - np.asarray(pose["xy"]))
            <= self.HAMPER_OPERATION_RADIUS
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        pose = self.scene_task_spec()["navigation"].get("operation_pose")
        if pose is not None:
            meta["hamper_operation_pose"] = copy.deepcopy(pose)
        return meta


class IThorBedroomApproachStorageAndOpenHamperLid(
    _ApproachHamperLidMixin,
    _FloorHamperMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "hamper_lid"
    SCENE_SPECS = _action_specs(
        (302, 306, 310, 323, 329),
        "hamper_lid",
        "walk to the laundry hamper and open its lid",
        navigation_task="approach_hamper_lid",
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    BEHAVIOR = "open"

    def _setup_scene(self):
        super()._setup_scene()
        if self._fixture_mode() == "spawn_asset":
            # The spawned lid is also used by the operable atomic tasks.
            # Preserve its original joint friction / constraint softness;
            # only the native 306 lid needs the navigation calibration.
            return
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        self.sim.model.dof_solref[dof_address] = (0.002, 1.0)
        self.sim.model.dof_solimp[dof_address] = (0.9999, 0.9999, 0.001, 0.5, 2.0)
        self.sim.data.qvel[dof_address] = 0.0
        self.sim.forward()


class IThorBedroomApproachStorageAndCloseHamperLid(
    _ApproachHamperLidMixin,
    _FloorHamperMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "hamper_lid"
    SCENE_SPECS = _action_specs(
        (302, 306, 310, 323, 329),
        "hamper_lid",
        "walk to the laundry hamper and close its lid",
        navigation_task="approach_hamper_lid",
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    BEHAVIOR = "close"
    # Retain the calibrated full-open navigation start for native 306 only.
    # Spawned lids use the atomic task's 85% aperture, away from the stop.
    OPEN_INIT_FRACTION = 1.0

    def _initial_joint_qpos(self, joint):
        if self._fixture_mode() != "spawn_asset":
            return super()._initial_joint_qpos(joint)
        return float(joint.closed_qpos + 0.85 * (joint.open_qpos - joint.closed_qpos))


class IThorBedroomApproachStorageAndOpenDrawer(
    _ApproachThenArticulationMixin,
    BedroomTwentyEightyArticulationMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "drawer"
    SCENE_SPECS = {
        **_action_specs(
            (310,),
            "drawer",
            "walk to the bedroom's only two-drawer bedside table and open "
            "its drawer",
            navigation_task="approach_storage",
        ),
        **_action_specs(
            (323,),
            "drawer",
            "walk to the dresser and open its drawer",
            navigation_task="approach_storage",
        ),
    }
    SUPPORTED_SCENE_IDS = (310, 323)
    BEHAVIOR = "open"


class IThorBedroomApproachStorageAndCloseDrawer(
    _ApproachThenArticulationMixin,
    BedroomTwentyEightyArticulationMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "drawer"
    SCENE_SPECS = {
        **_action_specs(
            (310,),
            "drawer",
            "walk to the bedroom's only two-drawer bedside table and close "
            "its drawer",
            navigation_task="approach_storage",
        ),
        **_action_specs(
            (323,),
            "drawer",
            "walk to the dresser and close its drawer",
            navigation_task="approach_storage",
        ),
    }
    SUPPORTED_SCENE_IDS = (310, 323)
    BEHAVIOR = "close"


class IThorBedroomApproachStorageAndOpenBoxFlap(
    _ApproachThenArticulationMixin,
    BedroomBoxFlapMixin,
    IThorBedroomEnv,
):
    SCENE_SPECS = _box_flap_approach_specs(
        (302, 306, 310, 323, 329),
        "walk to the cardboard box on the floor and open its flap",
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    BEHAVIOR = "open"


class IThorBedroomApproachStorageAndCloseBoxFlap(
    _ApproachThenArticulationMixin,
    BedroomBoxFlapMixin,
    IThorBedroomEnv,
):
    SCENE_SPECS = _box_flap_approach_specs(
        (302, 306, 310, 323, 329),
        "walk to the cardboard box on the floor and close its flap",
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    BEHAVIOR = "close"

    def _initial_articulation_intact(self):
        if self.scene_name != "FloorPlan310":
            return super()._initial_articulation_intact()
        # This scene's flap passively opens from its 0.85 reset fraction to
        # 1.0 before the robot can reach the box. Permit only opening drift;
        # closing before the approach stage must still invalidate the gate.
        fraction = float(self.target_semantic_joint.fraction(self.sim))
        return bool(
            fraction
            >= self._approach_initial_fraction - self.INITIAL_JOINT_FRACTION_TOLERANCE
            and not self._raw_success()
        )


class IThorBedroomOpenThenCloseDrawer(
    _ResettableOpenThenCloseArticulationMixin,
    BedroomTwentyEightyArticulationMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "drawer"
    SCENE_SPECS = _action_specs(
        (302, 306, 310, 323, 329),
        "drawer",
        "open the drawer, then close it",
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)


class IThorBedroomOpenThenCloseTopDrawer(
    _ResettableOpenThenCloseArticulationMixin,
    _BedroomTopDrawerMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "top_drawer"
    SCENE_SPECS = _action_specs(
        (302, 306, 323, 329),
        "top_drawer",
        "open the top drawer, then close it",
    )
    SUPPORTED_SCENE_IDS = (302, 306, 323, 329)


class IThorBedroomOpenThenCloseSafeDoor(
    _ResettableOpenThenCloseArticulationMixin,
    BedroomTwentyEightyArticulationMixin,
    _BedsideSafeMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "open_safe_door"
    SCENE_SPECS = _action_specs(
        (302, 323),
        "open_safe_door",
        "open the safe door, then close it",
    )
    SUPPORTED_SCENE_IDS = (302, 323)


class IThorBedroomOpenThenCloseHamperLid(
    _ResettableOpenThenCloseArticulationMixin,
    _FloorHamperMixin,
    BedroomArticulationMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "hamper_lid"
    SCENE_SPECS = _action_specs(
        (302, 306, 310, 323, 329),
        "hamper_lid",
        "open the laundry hamper lid, then close it",
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)

    def _setup_scene(self):
        super()._setup_scene()
        if self._fixture_mode() == "spawn_asset":
            # Spawned lids use the operable joint physics from the atomics.
            return
        joint_id = self.sim.model.joint_name2id(self.target_joint)
        dof_address = int(self.sim.model.jnt_dofadr[joint_id])
        # Prevent gravity creep from completing the closing stage after the
        # operator opens and releases the lid. Preserve atomic joint physics.
        self.sim.model.dof_solref[dof_address] = (0.002, 1.0)
        self.sim.model.dof_solimp[dof_address] = (0.9999, 0.9999, 0.001, 0.5, 2.0)
        self.sim.data.qvel[dof_address] = 0.0
        self.sim.forward()


class IThorBedroomOpenThenCloseBoxFlap(
    _ResettableOpenThenCloseArticulationMixin,
    BedroomBoxFlapMixin,
    IThorBedroomEnv,
):
    SCENE_SPECS = _action_specs(
        (302, 306, 310, 323, 329),
        "box_flap",
        "open the box flap, then close it",
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)

    def _configure_articulation_joint(self, joint):
        super()._configure_articulation_joint(joint)
        joint_id = self.sim.model.joint_name2id(joint.name)
        # The opening atomic stops at the upright start pose. A full cycle
        # must also allow the same flap to return to its visual closed endpoint.
        self.sim.model.jnt_range[joint_id] = sorted(
            (joint.closed_qpos, joint.open_qpos)
        )


class IThorBedroomOpenThenCloseLaptop(
    _OpenThenCloseArticulationMixin,
    BedroomLaptopMixin,
    IThorBedroomEnv,
):
    SPEC_TASK = "laptop"
    SCENE_SPECS = _action_specs(
        (302, 306, 310, 323, 329),
        "laptop",
        "open the laptop, then close it",
    )
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)


__all__ = [
    "IThorBedroomApproachDeskAndCloseLaptop",
    "IThorBedroomApproachDeskAndOpenLaptop",
    "IThorBedroomApproachStorageAndCloseBoxFlap",
    "IThorBedroomApproachStorageAndCloseDrawer",
    "IThorBedroomApproachStorageAndCloseHamperLid",
    "IThorBedroomApproachStorageAndCloseSafeDoor",
    "IThorBedroomApproachStorageAndOpenBoxFlap",
    "IThorBedroomApproachStorageAndOpenDrawer",
    "IThorBedroomApproachStorageAndOpenHamperLid",
    "IThorBedroomApproachStorageAndOpenSafeDoor",
    "IThorBedroomOpenThenCloseBoxFlap",
    "IThorBedroomOpenThenCloseDrawer",
    "IThorBedroomOpenThenCloseHamperLid",
    "IThorBedroomOpenThenCloseLaptop",
    "IThorBedroomOpenThenCloseSafeDoor",
    "IThorBedroomOpenThenCloseTopDrawer",
]
