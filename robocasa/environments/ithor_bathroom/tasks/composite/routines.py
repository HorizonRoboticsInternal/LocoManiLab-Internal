"""Independent three-stage routines over unchanged Bathroom skill predicates."""

from copy import deepcopy
from dataclasses import replace

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_atomic.task_assets import SOAP_MODEL, SOAP_SCALE, asset_xml_path
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin
from ..bathtub_tasks import _BathroomSpongeInBathtubTask
from ..lift import _BathroomNativeLiftTask
from ..object_manipulation import _NativeBathroomFixtureMixin
from ..reorientation import _BathroomNativeUprightTask
from ..shelf_pnp import _BathroomShelfPnPMixin
from .brush_support import _BathroomScrubBrushLiftTask
from .routine_specs import (
    _TOILET_ROUTINE_SCENES, drawer_specs, tidy_specs, toilet_specs,
)


class _SkillBinding:
    """Own skill-local fields; delegate only simulator services to the environment.

    These objects are not environments. In particular their lifecycle and
    counters never participate in the environment's reward/reset MRO.
    """

    def __init__(self, env, spec):
        self.env = env
        self.spec = deepcopy(spec)
        self.fixture_refs = {}
        self.region_refs = {}

    def __getattr__(self, name):
        return getattr(self.env, name)

    def scene_task_spec(self):
        return deepcopy(self.spec)


class _BindingEnd:
    def _setup_task_fixtures(self):
        pass

    def _reset_internal(self):
        pass

    def _is_scene_valid(self):
        return True


class _Bottle(_SkillBinding, _BathroomNativeUprightTask, _BindingEnd):
    pass


class _Lift(_SkillBinding, _BathroomNativeLiftTask, _BindingEnd):
    pass


class _StoolLift(_SkillBinding, _BathroomScrubBrushLiftTask, _BindingEnd):
    pass


class _Sponge(_SkillBinding, _BathroomSpongeInBathtubTask, _BindingEnd):
    pass


class _ShelfSoap(_SkillBinding, _BathroomShelfPnPMixin, _BindingEnd):
    SOURCE_NAME = "bar_soap"
    SOURCE_GROUP = asset_xml_path(SOAP_MODEL)
    SOURCE_LANG = "bar of soap"
    SOURCE_SCALE = SOAP_SCALE
    SOURCE_ROTATION = 0.0


class _BathroomRoutineMixin(
    _NativeBathroomFixtureMixin, SupportContactPlacementMixin,
    SceneSpecMixin, CompositeStageReporterMixin,
):
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    ATOMIC_SKILL = "ThreeSkillRoutine"
    COMPOSITE_KIND = "ordered_bathroom_routine"
    INITIAL_POSE_JITTER = False
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    INTERMEDIATE_HOLD_STEPS = 5
    SUCCESS_DISTANCE = 0.35
    MIN_LIFT_DELTA = 0.30
    GRASP_HEIGHT_ALLOWANCE = 0.10
    PERSISTENT_STAGES = ()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self._skills = {}
        self._interior = None
        for name, item in spec["items"].items():
            if "interior_region" in item:
                fixture = self.get_fixture(id=item["surface_fixture"])
                self._interior = deepcopy(fixture.regions[item["interior_region"]])
                # Semantic bounds are world-axis projections, NOT local Z.
                floor = replace(self._interior, kind="support",
                    stable_id=item["surface_region"],
                    _world_offset=self._interior._world_offset +
                    np.array([0., 0., -self._interior.half_size[2]]))
                fixture.regions[floor.stable_id] = floor
            cls = {"bottle": _Bottle, "sponge": _Sponge, "soap": _ShelfSoap}.get(name, _Lift)
            if "source_stool" in item:
                cls = _StoolLift
            skill = cls(self, item)
            skill._setup_task_fixtures()
            if not skill._is_scene_valid():
                raise RuntimeError(f"{self.scene_name}: invalid routine item {name}")
            self._skills[name] = skill
            self.fixture_refs.update({f"{name}:{k}": v for k, v in skill.fixture_refs.items()})
            self.region_refs.update({f"{name}:{k}": v for k, v in skill.region_refs.items()})
        art = spec.get("articulation")
        self._routine_joint = None
        if art:
            self._joint_fixture = self.get_fixture(id=art["fixture"])
            self._routine_joint = next(j for j in self._joint_fixture.joints.values()
                                       if j.name == art["joint"])
            self.fixture_refs["routine_articulation"] = self._joint_fixture
            if art.get("anchor_fixture"):
                root = self.find_scene_body(self._joint_fixture.body_name)
                free = [j for j in root.findall("./joint") if j.get("type") == "free"]
                if len(free) != 1:
                    raise RuntimeError("routine drawer cabinet needs one free base joint")
                root.remove(free[0])
        nav = spec["navigation"]
        self._navigation_fixture = self.get_fixture(id=nav["fixture"]) if nav else None
        if nav:
            self.fixture_refs["routine_navigation"] = self._navigation_fixture
        self._routine_start_pose = spec["start_pose"]
        if self._routine_start_pose is None:
            # The shelf skill sets precisely the atomic shelf stand pose.
            self._routine_start_pose = dict(xy=self.robot_base_pos[:2].tolist(),
                                           yaw=float(self.robot_base_ori[2]))
        pose = self._routine_start_pose
        self.set_robot_base_pose([*pose["xy"], 0.], pose["yaw"])

    def _get_obj_cfgs(self):
        return [cfg for skill in self._skills.values() for cfg in skill._get_obj_cfgs()]

    def _setup_scene(self):
        super()._setup_scene()
        art = self.scene_task_spec().get("articulation")
        if art:
            joint = self._routine_joint
            jid = self.sim.model.joint_name2id(joint.name)
            dof = int(self.sim.model.jnt_dofadr[jid])
            toilet = "toilet_lid" in " ".join(self.COMPOSITE_ORDERING)
            self.sim.model.dof_damping[dof] = max(self.sim.model.dof_damping[dof], 0.5)
            self.sim.model.dof_frictionloss[dof] = max(
                self.sim.model.dof_frictionloss[dof], 0.6 if toilet else 0.2)
            if toilet:
                stop = joint.closed_qpos + .20 * (joint.open_qpos - joint.closed_qpos)
                self.sim.model.jnt_range[jid] = sorted([stop, joint.open_qpos])
                self.sim.model.dof_solref[dof] = [0.002, 1.0]
                self.sim.model.dof_solimp[dof] = [0.9999, 0.9999, 0.001, 0.5, 2.0]
            self.sim.data.set_joint_qpos(joint.name, joint.closed_qpos +
                art["init_open_fraction"] * (joint.open_qpos - joint.closed_qpos))
            self.sim.data.qvel[dof] = 0.
            self.sim.forward()
        for skill in self._skills.values():
            layout = skill.spec.get("source_stool")
            if not layout:
                continue
            eid = skill._stool_base_weld_id()
            self.sim.data.eq_active[eid] = 0
            self.sim.forward()
            if not self._place_fixture_on_floor(skill.task_stool_fixture, layout["xy"],
                    contract_name="routine_stool_floor", probe_above=layout["floor_probe_above"],
                    probe_below=layout["floor_probe_below"], probe_step=layout["floor_probe_step"]):
                raise RuntimeError(f"{self.scene_name}: routine stool has no Floor support")
            bid = self.sim.model.body_name2id(skill.task_stool.root_body)
            position = self.sim.data.body_xpos[bid]
            rotation = self.sim.data.body_xmat[bid].reshape(3, 3)
            quaternion = self.sim.data.body_xquat[bid]
            self.sim.model.eq_data[eid, 3:6] = -rotation.T @ position
            self.sim.model.eq_data[eid, 6:10] = np.r_[quaternion[0], -quaternion[1:]]
            self._zero_fixture_velocity(skill.task_stool_fixture)
            self.sim.data.eq_active[eid] = 1
        self._set_exact_robot_pose(self._routine_start_pose, "routine spawn")

    def _reset_internal(self):
        super()._reset_internal()
        for skill in self._skills.values():
            # Only explicitly requested skill reset hooks run, not their reward
            # or robot-placement lifecycle. All counters remain skill-local.
            skill._reset_internal()
        if self.count_body_collisions() or not self.robot_footprint_is_on_floor(self._robot_base_xy(), radius=.25):
            raise RuntimeError(f"{self.scene_name}: unsafe routine spawn")
        self._robot_free_joint_z = float(self.sim.data.get_joint_qpos(self._get_robot_free_joint())[2])
        self._robot_default_free_joint_z = self._robot_free_joint_z
        if self.COMPOSITE_ORDERING[0].startswith("approach_"):
            distance = self.robot_fixture_collision_distance(self._navigation_fixture, max_distance=10.)
            if distance < self.SUCCESS_DISTANCE + .15:
                raise RuntimeError(f"{self.scene_name}: routine starts too near navigation goal")
        self._reset_task_progress()

    def _reset_task_progress(self):
        self._routine_stage = 0
        self._routine_hold = 0
        self._routine_success = False
        self._grasped_at_source = False
        self._initial_heights = {
            name: float(self.sim.data.body_xpos[self.sim.model.body_name2id(skill.source_fixture.body_name), 2])
            for name, skill in self._skills.items() if "source_fixture" in skill.__dict__}
        for skill in self._skills.values():
            skill._success_hold = skill._success_hold_steps = 0
            skill._grasped_near_source = False
        self._enter_stage()

    def _enter_stage(self):
        self._routine_hold = 0
        self._stage_touched = False
        self._stage_armed = not self._stage_geometry(self.COMPOSITE_ORDERING[self._routine_stage])
        self._grasped_at_source = False

    def _source_height(self):
        source = self._skills["source"].source_fixture
        return float(self.sim.data.body_xpos[self.sim.model.body_name2id(source.body_name), 2])

    def _outside_storage(self):
        if self._interior is None:
            return True
        spec = self._skills["source"].spec
        points = self._fixture_bbox_points(self._skills["source"].source_fixture)
        direction = spec["exit_direction"]
        plane = spec["exit_x"]
        if plane is None:
            # The moving drawer front extends 14.5 cm beyond its receptacle centre.
            plane = float(self._interior.world_pose(self.sim)[0][0]) - .145
        return bool(np.all(direction * (points[:, 0] - plane) >= .01))

    def _source_grasped(self):
        return self._g1_hand_grasping_fixture(self._skills["source"].source_fixture)

    def _source_grasp_is_at_start(self):
        source = self._skills["source"].source_fixture
        return self._interior is None or self._fixture_bbox_inside_region_xy(source, self._interior)

    def _source_clear(self):
        hands = [f"robot0_{side}_wrist_yaw_link" for side in ("left", "right")]
        return not self._fixture_has_illegal_penetration(
            self._skills["source"].source_fixture, hands)

    def _stage_geometry(self, stage):
        if stage.startswith("approach_"):
            distance = self.robot_fixture_collision_distance(self._navigation_fixture, max_distance=10.)
            return bool(np.isfinite(distance) and distance <= self.SUCCESS_DISTANCE)
        if stage.startswith(("open_", "close_")):
            art = self.scene_task_spec()["articulation"]
            fraction = self._routine_joint.fraction(self.sim)
            return bool(fraction >= art["success_open_fraction"] if stage.startswith("open_")
                        else fraction <= art["success_close_fraction"])
        if stage.startswith("lift_"):
            return self._source_height() >= self._initial_heights["source"] + self.MIN_LIFT_DELTA
        if stage == "stand_soap_bottle_upright":
            skill = self._skills["bottle"]
            return skill._fixture_is_upright(skill.source_fixture, skill._target_upright_axis)
        if stage == "soap_to_lower_shelf":
            return self._skills["soap"]._object_center_is_on_target_level()
        if stage == "put_sponge_in_bathtub":
            skill = self._skills["sponge"]
            points = self.object_bbox_points("sponge")
            center = (points.min(axis=0) + points.max(axis=0)) / 2
            return bool(np.all(abs(center[:2] - skill.basin_center_xy) <= skill.basin_half_xy + skill.BASIN_XY_TOLERANCE)
                        and points[:, 2].min() >= skill.basin_z_range[0] - skill.BASIN_Z_TOLERANCE
                        and points[:, 2].max() <= skill.basin_z_range[1] + skill.BASIN_Z_TOLERANCE)
        raise ValueError(f"unknown routine stage {stage}")

    def _stage_goal(self, stage):
        if stage == "stand_soap_bottle_upright":
            return self._skills["bottle"]._raw_success()
        if stage == "soap_to_lower_shelf":
            return self._skills["soap"]._object_is_on_target_shelf_and_released()
        if stage == "put_sponge_in_bathtub":
            return self._skills["sponge"]._raw_success()
        if stage.startswith("lift_"):
            return bool(self._stage_geometry(stage) and self._grasped_at_source
                        and self._source_grasped() and self._outside_storage() and self._source_clear())
        return self._stage_geometry(stage)

    def _stage_contact(self, stage):
        if stage.startswith(("open_", "close_")):
            jid = self.sim.model.joint_name2id(self._routine_joint.name)
            targets = self._geom_names_for_bodies(self._body_tree_ids([self.sim.model.jnt_bodyid[jid]]))
            hands = self._geom_names_for_bodies(self._body_ids_matching(("left_hand", "right_hand")))
            return bool(hands and targets and self.check_contact(hands, targets))
        if stage.startswith("lift_"):
            return self._source_grasped()
        name = {"stand_soap_bottle_upright": "bottle", "soap_to_lower_shelf": "soap",
                "put_sponge_in_bathtub": "sponge"}[stage]
        skill = self._skills[name]
        if name == "bottle":
            return not self._fixture_released(skill.source_fixture)
        return any(self.g1_hand_object_contact(side, skill.SOURCE_NAME) for side in ("left", "right"))

    def _final_invariants(self):
        return all(self._stage_goal(stage) for stage in self.PERSISTENT_STAGES)

    def _advance_routine(self):
        stage = self.COMPOSITE_ORDERING[self._routine_stage]
        navigation = stage.startswith("approach_")
        if not navigation:
            if not self._stage_geometry(stage):
                self._stage_armed = True
            if self._stage_armed and self._stage_contact(stage):
                self._stage_touched = True
                if stage.startswith("lift_") and self._source_height() <= self._initial_heights["source"] + self.GRASP_HEIGHT_ALLOWANCE:
                    # For storage, credit the initial grasp only inside its
                    # named compartment, not an item removed out of order.
                    if self._source_grasp_is_at_start():
                        self._grasped_at_source = True
        valid = self._stage_goal(stage) and (navigation or self._stage_touched)
        final = self._routine_stage == 2
        if final:
            valid = valid and self._final_invariants()
        self._routine_hold = self._routine_hold + 1 if valid else 0
        required = max(1, int(np.ceil(self.control_freq))) if final else self.INTERMEDIATE_HOLD_STEPS
        was_success = self._routine_success
        self._routine_success = bool(final and self._routine_hold >= required)
        if self._routine_success and not was_success:
            self._report_stage_complete(3, 3)
        elif not final and self._routine_hold >= required:
            self._routine_stage += 1
            self._enter_stage()
            self._report_stage_complete(self._routine_stage, 3, self.COMPOSITE_ORDERING[self._routine_stage])

    def _post_action(self, action):
        self._advance_routine()
        return super()._post_action(action)

    def _check_success(self):
        return bool(getattr(self, "_routine_success", False))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        spec = self.scene_task_spec()
        meta.update(lang=spec["lang"], composite_stages=list(self.COMPOSITE_ORDERING),
                    composite_ordering=list(self.COMPOSITE_ORDERING),
                    routine_stage=getattr(self, "_routine_stage", 0),
                    routine_layout=spec, intermediate_hold_steps=5,
                    final_hold_seconds=1., navigation_distance=.35, minimum_lift_delta=.30,
                    persistent_stages=list(self.PERSISTENT_STAGES),
                    final_grasp_required="source" in self._skills,
                    final_lift_height_required=self.COMPOSITE_ORDERING[-1].startswith("lift_"),
                    storage_exit_required=self._interior is not None,
                    initial_source_root_heights=dict(self._initial_heights),
                    allow_walking_between_operations=True)
        if "articulation" in spec:
            for key in ("init_open_fraction", "success_open_fraction", "success_close_fraction"):
                meta[key] = spec["articulation"][key]
        return meta


class IThorBathroomNavigateToBathingAreaAndStandSoapBottleUprightThenPutSpongeInBathtub(_BathroomRoutineMixin, IThorBathroomEnv):
    SPEC_TASK = "approach_bath_stand_bottle_put_sponge"
    SUPPORTED_SCENE_IDS = (401,)
    SCENE_SPECS = tidy_specs()
    COMPOSITE_ORDERING = ("approach_bathing_area", "stand_soap_bottle_upright", "put_sponge_in_bathtub")
    PERSISTENT_STAGES = ("stand_soap_bottle_upright",)


class IThorBathroomSoapToLowerShelfThenStandSoapBottleUprightAndPutSpongeInBathtub(_BathroomRoutineMixin, IThorBathroomEnv):
    SPEC_TASK = "shelf_soap_stand_bottle_put_sponge"
    SUPPORTED_SCENE_IDS = (401,)
    SCENE_SPECS = tidy_specs(shelf=True)
    COMPOSITE_ORDERING = ("soap_to_lower_shelf", "stand_soap_bottle_upright", "put_sponge_in_bathtub")
    PERSISTENT_STAGES = ("soap_to_lower_shelf", "stand_soap_bottle_upright")


class IThorBathroomCloseToiletLidThenApproachSinkAndLiftSoapBottle(_BathroomRoutineMixin, IThorBathroomEnv):
    SPEC_TASK = "close_lid_approach_sink_lift_soap"
    SUPPORTED_SCENE_IDS = _TOILET_ROUTINE_SCENES
    SCENE_SPECS = toilet_specs(close=True)
    COMPOSITE_ORDERING = ("close_toilet_lid", "approach_sink", "lift_soap_bottle")
    PERSISTENT_STAGES = ("close_toilet_lid",)


class IThorBathroomApproachToiletAndOpenToiletLidThenLiftScrubBrush(_BathroomRoutineMixin, IThorBathroomEnv):
    SPEC_TASK = "approach_toilet_open_lid_lift_brush"
    SUPPORTED_SCENE_IDS = _TOILET_ROUTINE_SCENES
    SCENE_SPECS = toilet_specs(close=False)
    COMPOSITE_ORDERING = ("approach_toilet", "open_toilet_lid", "lift_scrub_brush")
    PERSISTENT_STAGES = ("open_toilet_lid",)


class IThorBathroomOpenUpperDrawerThenLiftToiletPaperRollAndCloseUpperDrawer(_BathroomRoutineMixin, IThorBathroomEnv):
    SPEC_TASK = "open_upper_drawer_lift_paper_close_upper_drawer"
    SUPPORTED_SCENE_IDS = (415,)
    SCENE_SPECS = drawer_specs()
    COMPOSITE_ORDERING = ("open_upper_drawer", "lift_toilet_paper_roll", "close_upper_drawer")

    def _stage_contact(self, stage):
        # This routine advances solely from each stage's target state.  The
        # final invariants still enforce that the roll is clear of the drawer
        # when the whole routine succeeds.
        return True

    def _stage_goal(self, stage):
        # This task's lift stage is credited solely from the measured source
        # height: the roll must rise 0.30 m above its reset baseline.
        # Storage-exit and penetration checks remain enforced by
        # _final_invariants after the drawer is closed.
        if stage == "lift_toilet_paper_roll":
            return bool(self._stage_geometry(stage))
        return super()._stage_goal(stage)

    def _final_invariants(self):
        # The roll may be released after the lift stage.  It must still be
        # completely outside the drawer and free of illegal wrist penetration.
        return bool(super()._final_invariants() and self._outside_storage()
                    and self._source_clear())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["final_grasp_required"] = False
        return meta

    def _source_grasp_is_at_start(self):
        # Closing fingers can tip or pull the roll partly over the compartment's
        # conservative XY patch. Requiring EVERY corner inside loses legitimate
        # grasps that cannot be credited while held aloft. Its centre must still be
        # inside the moving drawer, near the original height; outside re-grasps
        # and picking an already raised / dropped roll cannot establish credit.
        points = self._fixture_bbox_points(self._skills["source"].source_fixture)
        center = (points.min(axis=0) + points.max(axis=0)) / 2
        return bool(self._interior.contains_xy(center[:2], self.sim)
                    and abs(self._source_height() - self._initial_heights["source"]) <= self.GRASP_HEIGHT_ALLOWANCE)

TASK_CLASSES = (
    IThorBathroomNavigateToBathingAreaAndStandSoapBottleUprightThenPutSpongeInBathtub,
    IThorBathroomSoapToLowerShelfThenStandSoapBottleUprightAndPutSpongeInBathtub,
    IThorBathroomCloseToiletLidThenApproachSinkAndLiftSoapBottle,
    IThorBathroomApproachToiletAndOpenToiletLidThenLiftScrubBrush,
    IThorBathroomOpenUpperDrawerThenLiftToiletPaperRollAndCloseUpperDrawer,
)
__all__ = [cls.__name__ for cls in TASK_CLASSES]
