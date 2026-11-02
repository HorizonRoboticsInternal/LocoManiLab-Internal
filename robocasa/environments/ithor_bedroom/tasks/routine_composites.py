"""Three-skill Bedroom routines, isolated from the published two-step tasks."""

from copy import deepcopy

import mujoco
import numpy as np

from robocasa.environments.ithor_atomic.pick_place import SemanticSurfaceTask
from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.environments.ithor_composite.progress import CompositeStageReporterMixin
from robocasa.models.scenes.ithor_bedroom_task_specs import get_ithor_bedroom_task_spec
from robocasa.utils import env_utils as EnvUtils

from .articulation_base import BedroomArticulationMixin
from .assets import (
    BASKET_MODEL,
    BASKET_SCALE,
    BOXED_FOOD_MODEL,
    BOXED_FOOD_SCALE,
    CAN_MODEL,
    CAN_SCALE,
    MUG_MODEL,
    MUG_SCALE,
    asset_xml_path,
)
from .composite_base import BedroomCompositeObjectMixin
from .hamper_lid import _FloorHamperMixin
from .laptop import BedroomLaptopMixin


_ASSETS = {
    "basket": (BASKET_MODEL, BASKET_SCALE),
    "can": (CAN_MODEL, CAN_SCALE),
    "boxed_food": (BOXED_FOOD_MODEL, BOXED_FOOD_SCALE),
    "mug": (MUG_MODEL, MUG_SCALE),
}


def _spec(scene, action, stages, lang, *, far=True, basket_asset=None):
    """Copy calibrations, including exact physical stage targets, for dedup."""
    name = f"FloorPlan{scene}"
    spec = deepcopy(get_ithor_bedroom_task_spec(name, action))
    spec["lang"] = lang
    workspace = (
        get_ithor_bedroom_task_spec(name, "grasp_workspace")
        if scene == 329 or any(stage.startswith("put_") for stage in stages)
        else None
    )
    targets = {}
    for stage in stages:
        if stage == "approach_bed":
            targets[stage] = get_ithor_bedroom_task_spec(name, "approach_bed")
        elif stage == "approach_desk":
            targets[stage] = (
                workspace
                if workspace is not None
                else get_ithor_bedroom_task_spec(name, stage)
            )
        elif stage == "approach_storage":
            targets[stage] = get_ithor_bedroom_task_spec(name, "approach_storage")
        elif stage == "close_top_drawer":
            # 310 has a real upper drawer (joint_1), calibrated as "drawer"
            # by the atomic task. Only this routine calls it "top drawer".
            targets[stage] = get_ithor_bedroom_task_spec(
                name, "drawer" if scene == 310 else "top_drawer"
            )
        elif stage.startswith("put_"):
            item = stage.removeprefix("put_")
            targets[stage] = dict(
                item=item,
                asset=_ASSETS[item],
                basket=basket_asset or _ASSETS["basket"],
                fixture=workspace["fixture"],
                region=workspace["region"],
            )
        else:
            targets[stage] = {
                key: spec[key]
                for key in ("joint", "fixture", "asset_id")
                if key in spec
            }
    spec["routine_targets"] = targets
    spec["routine_workspace"] = workspace
    spec["routine_start"] = (
        get_ithor_bedroom_task_spec(
            name,
            "approach_storage"
            if stages[0] == "approach_storage"
            else "secondary_workspace",
        )
        if far
        else None
    )
    return {name: spec}


def _scene_specs(scenes, *args, **kwargs):
    return {
        name: spec
        for scene in scenes
        for name, spec in _spec(scene, *args, **kwargs).items()
    }


class _ThreeStageRoutineMixin(CompositeStageReporterMixin):
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L3"
    ATOMIC_SKILL = "ThreeSkillRoutine"
    COMPOSITE_KIND = "ordered_bedroom_routine"
    APPROACH_HOLD_STEPS = 5
    SUCCESS_DISTANCE = 0.35
    BED_MIN_DISPLACEMENT = 0.30
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    ITEM_NAMES = ()

    def _reset_internal(self):
        super()._reset_internal()
        self._initialize_routine_layout()
        start = self.scene_task_spec()["routine_start"]
        if start is not None:
            pose = start.get("start_pose", start.get("far_start_pose"))
            xy = np.asarray(pose["xy"], dtype=float)
            yaw = float(pose["yaw"])
            if hasattr(self, "_robot_free_joint_z"):
                del self._robot_free_joint_z
            if not (
                self.robot_footprint_is_on_floor(xy, radius=0.25)
                and self._set_robot_free_pose(xy, yaw)
                and self._calibrate_robot_free_joint_height()
                and self.count_body_collisions() == 0
            ):
                raise RuntimeError(f"{self.scene_name}: invalid routine far spawn")
            self.robot_base_pos = np.r_[xy, 0.0]
            self.robot_base_ori = np.array([0.0, 0.0, yaw])
            target = self.get_fixture(id=start["fixture"])
            distance = self.robot_fixture_collision_distance(target, max_distance=10.0)
            if distance < float(start["minimum_start_distance"]):
                raise RuntimeError(
                    f"{self.scene_name}: routine starts too near its target"
                )
            # This far pose is applied after the parent's ground snap. Reseat
            # the controller stance at the final XY, then retain its measured Z.
            EnvUtils.settle_robot_spawn(self)
        self.sim.forward()
        root_joint = self._get_robot_free_joint()
        self._robot_free_joint_z = float(self.sim.data.get_joint_qpos(root_joint)[2])
        self._robot_default_free_joint_z = self._robot_free_joint_z
        if self.count_body_collisions() or not self.robot_footprint_is_on_floor(
            self._robot_base_xy(), radius=0.25
        ):
            raise RuntimeError(
                f"{self.scene_name}: routine reset has an unsafe robot pose"
            )
        self._reset_task_progress()

    def _reset_task_progress(self):
        reset = getattr(super(), "_reset_task_progress", None)
        if reset is not None:
            reset()
        if hasattr(self, "_success_hold"):
            self._success_hold = 0
        self._routine_stage = 0
        self._routine_hold = 0
        self._routine_success = False
        self._enter_routine_stage()

    def _enter_routine_stage(self):
        self._routine_hold = 0
        self._routine_touched = False
        self._routine_entry_xy = self._robot_base_xy().copy()
        stage = self.COMPOSITE_ORDERING[self._routine_stage]
        # A precompleted operation must leave the goal region before it can
        # earn fresh credit. Merely touching an already placed item cannot arm it.
        self._routine_armed = not self._routine_geometry(stage)

    def _routine_geometry(self, stage):
        if stage.startswith("approach_"):
            spec = self.scene_task_spec()["routine_targets"][stage]
            fixture = self.get_fixture(id=spec["fixture"])
            near = (
                self.robot_fixture_collision_distance(fixture, max_distance=10.0)
                <= self.SUCCESS_DISTANCE
            )
            if stage == "approach_bed":
                near = (
                    near
                    and np.linalg.norm(self._robot_base_xy() - self._routine_entry_xy)
                    >= self.BED_MIN_DISPLACEMENT
                )
            return bool(near)
        if stage.startswith("put_"):
            return self._inside_routine_basket(stage.removeprefix("put_"))
        joint, threshold = self._routine_joints[stage]
        return bool(joint.fraction(self.sim) <= threshold)

    def _routine_goal(self, stage):
        if not self._routine_geometry(stage):
            return False
        if stage.startswith("put_"):
            return stage.removeprefix("put_") in self._supported_basket_items()
        if stage == "close_laptop":
            return self._laptop_support_contact_is_valid()
        return True

    def _routine_stage_requires_contact(self, stage):
        """Whether an operation stage must be touched before it can advance."""
        return not stage.startswith("approach_")

    def _routine_contact(self, stage):
        if stage.startswith("put_"):
            return self._hands_touch_routine_object(stage.removeprefix("put_"))
        joint, _ = self._routine_joints[stage]
        jid = self.sim.model.joint_name2id(joint.name)
        moving = self._body_tree_ids([int(self.sim.model.jnt_bodyid[jid])])
        return self._hands_touch_bodies(moving)

    def _hands_touch_bodies(self, bodies):
        hand_roots = [
            self.sim.model.body_name2id(f"robot0_{side}_wrist_yaw_link")
            for side in ("left", "right")
        ]
        hand_geoms = self._geom_names_for_bodies(self._body_tree_ids(hand_roots))
        target_geoms = self._geom_names_for_bodies(bodies)
        return bool(
            hand_geoms and target_geoms and self.check_contact(hand_geoms, target_geoms)
        )

    def _hands_touch_routine_object(self, name):
        return self._hands_touch_bodies(self._body_tree_ids([self.obj_body_id[name]]))

    def _advance_routine(self):
        stage = self.COMPOSITE_ORDERING[self._routine_stage]
        navigation = stage.startswith("approach_")
        if not navigation:
            if not self._routine_geometry(stage):
                self._routine_armed = True
            if self._routine_armed and self._routine_contact(stage):
                self._routine_touched = True
        requires_contact = self._routine_stage_requires_contact(stage)
        valid = self._routine_goal(stage) and (
            navigation or not requires_contact or self._routine_touched
        )
        final = self._routine_stage == 2
        if final:
            valid = valid and all(
                self._routine_goal(previous)
                for previous in self.COMPOSITE_ORDERING[:2]
                if not previous.startswith("approach_")
            )
        self._routine_hold = self._routine_hold + 1 if valid else 0
        required = max(1, int(np.ceil(float(self.control_freq)))) if final else 5
        was_success = self._routine_success
        self._routine_success = bool(final and self._routine_hold >= required)
        if self._routine_success and not was_success:
            self._report_stage_complete(3, 3)
        elif not final and self._routine_hold >= required:
            self._routine_stage += 1
            self._enter_routine_stage()
            self._report_stage_complete(
                self._routine_stage, 3, self.COMPOSITE_ORDERING[self._routine_stage]
            )

    def _post_action(self, action):
        # Run before reward/done observes success. Queries never advance time.
        self._advance_routine()
        return super()._post_action(action)

    def _check_success(self):
        return bool(getattr(self, "_routine_success", False))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=self.scene_task_spec()["lang"],
            composite_stages=list(self.COMPOSITE_ORDERING),
            composite_ordering=list(self.COMPOSITE_ORDERING),
            routine_targets=self.scene_task_spec()["routine_targets"],
            routine_stage=getattr(self, "_routine_stage", 0),
            intermediate_hold_steps=5,
            final_hold_seconds=1.0,
            release_definition="supported, neither hand touching or grasping the item",
            navigation_distance=0.35,
            bed_entry_displacement=0.30,
        )
        return meta


class _RoutineBasketMixin(BedroomCompositeObjectMixin):
    OBJECT_EDGE_MARGIN = 0.025
    OBJECT_GAP = 0.03

    def _get_obj_cfgs(self):
        return [
            dict(
                name=name,
                obj_groups=name,
                object_scale=_ASSETS[name][1],
                info={"mjcf_path": asset_xml_path(_ASSETS[name][0])},
            )
            for name in ("basket", *self.ITEM_NAMES)
        ]

    def _composite_object_released(self, name):
        return not self._hands_touch_routine_object(
            name
        ) and not self.g1_hand_grasping_object(name)

    def _inside_routine_basket(self, name):
        basket = self.objects["basket"]
        origin = self.sim.data.get_body_xpos(basket.root_body)
        matrix = self.sim.data.get_body_xmat(basket.root_body).reshape(3, 3)
        bounds = (self.object_bbox_points("basket") - origin) @ matrix
        points = (self.object_bbox_points(name) - origin) @ matrix
        lower, upper = bounds.min(axis=0), bounds.max(axis=0)
        center = (points.min(axis=0) + points.max(axis=0)) / 2.0
        normalized = (center[:2] - (lower[:2] + upper[:2]) / 2.0) / (
            (upper[:2] - lower[:2]) * 0.41
        )
        bottom_tolerance = 0.035 if name == "mug" else 0.005
        return bool(
            np.sum(normalized**2) <= 1.0
            and lower[2] - bottom_tolerance
            <= points[:, 2].min()
            <= upper[2] - self.RECEPTACLE_RIM_CLEARANCE
        )

    def _load_bearing_contact(self, upper, lower):
        """An upward, shallow, compressive contact; side-wall touch is insufficient."""
        upper_bodies = self._body_tree_ids([self.obj_body_id[upper]])
        lower_bodies = self._body_tree_ids([self.obj_body_id[lower]])
        supported = False
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            b1, b2 = (
                int(self.sim.model.geom_bodyid[g])
                for g in (contact.geom1, contact.geom2)
            )
            sign = (
                1
                if b2 in upper_bodies and b1 in lower_bodies
                else -1
                if b1 in upper_bodies and b2 in lower_bodies
                else 0
            )
            if not sign:
                continue
            if float(contact.dist) < -0.003:
                return False
            normal_z = sign * float(contact.frame[2])
            if normal_z < 0.35:
                continue
            force = np.zeros(6)
            mujoco.mj_contactForce(
                self.sim.model._model, self.sim.data._data, index, force
            )
            if force[0] > 1e-5:
                supported = True
        return supported

    def _supported_basket_items(self):
        if not self._receptacle_upright_released_on_region(
            "basket", self.work_surface_region
        ) or not self._routine_no_penetration("basket"):
            return set()
        candidates = {
            name
            for name in self.ITEM_NAMES
            if self._inside_routine_basket(name)
            and self._composite_object_released(name)
            and self._routine_no_penetration(name)
        }
        supported = {"basket"}
        for _ in self.ITEM_NAMES:
            new = {
                name
                for name in candidates - supported
                if any(self._load_bearing_contact(name, lower) for lower in supported)
            }
            if not new:
                break
            supported.update(new)
        return supported - {"basket"}

    def _routine_no_penetration(self, name):
        # No broad fixture-sibling allowance: other task objects are allowed
        # to support this item only through shallow contacts as well.
        return not self.illegal_contact_pairs_for_bodies(
            [self.obj_body_id[name]], tolerance=0.003
        )

    def _layout_basket_row(self, *, laptop=False):
        region = self.work_surface_region
        center, outward, lateral = self._surface_layout_frame(
            region,
            operator_xy=region.world_pose(self.sim)[0][:2]
            + 2.0
            * np.asarray(
                get_ithor_bedroom_task_spec(self.scene_name, "grasp_workspace")[
                    "approach_direction"
                ]
            ),
        )
        names = ("basket", *self.ITEM_NAMES)
        for i, name in enumerate(names):
            self._set_object_bottom_at(name, center[:2], 2.5 + i * 0.2)
            self.sim.forward()
            self._orient_object_long_axis_crosswise(name, region)
            self.sim.forward()
        widths = [
            2 * self._object_projected_half_extent(name, lateral) for name in names
        ]
        depths = [self._object_projected_half_extent(name, outward) for name in names]
        if laptop:
            points, _ = self._body_geom_points(self.target_fixture.body_names)
            laptop_bounds = points[:, :2] @ lateral
            widths.append(float(np.ptp(laptop_bounds)))
        total = sum(widths) + self.OBJECT_GAP * (len(widths) - 1)
        if total + 2 * self.OBJECT_EDGE_MARGIN > 2 * self.support_region_radial_extent(
            region, lateral, sim=self.sim
        ):
            raise RuntimeError("routine objects do not fit the original table")
        row_depth = (
            self.support_region_radial_extent(region, outward, sim=self.sim)
            - max(depths)
            - self.OBJECT_EDGE_MARGIN
        )
        cursor = -total / 2
        if laptop:
            # Move the already calibrated laptop before probing the basket;
            # its old centered pose would obstruct the new left-hand row.
            root = self._laptop_free_joint()
            qpos = self.sim.data.get_joint_qpos(root).copy()
            desired = float(center[:2] @ lateral) + total / 2 - widths[-1] / 2
            qpos[:2] += lateral * (
                desired - (laptop_bounds.min() + laptop_bounds.max()) / 2
            )
            self.sim.data.set_joint_qpos(root, qpos)
            self.sim.forward()
            if self._probe_laptop_support_contact(root, qpos, center[2]) is None:
                raise RuntimeError("routine laptop lacks contacted desk support")
        for name, width in zip(names, widths):
            xy = center[:2] + outward * row_depth + lateral * (cursor + width / 2)
            if not self.place_object_on_support_contact(
                name,
                region,
                fixture=self.work_surface_fixture,
                xy=xy,
                inward_fractions=(0.0,),
                probe_below=0.08,
                probe_step=0.001,
                penetration_tolerance=0.002,
            ):
                raise RuntimeError(f"{self.scene_name}: cannot support routine {name}")
            cursor += width + self.OBJECT_GAP
        self.sim.forward()


class _RoutineTableMixin(_RoutineBasketMixin, SemanticSurfaceTask):
    WORK_SURFACE_FIXTURES = {
        f"FloorPlan{scene}": get_ithor_bedroom_task_spec(
            f"FloorPlan{scene}", "grasp_workspace"
        )["fixture"]
        for scene in (302, 306, 310, 323, 329)
    }
    WORK_SURFACE_REGIONS = {
        f"FloorPlan{scene}": get_ithor_bedroom_task_spec(
            f"FloorPlan{scene}", "grasp_workspace"
        )["region"]
        for scene in (302, 306, 310, 323, 329)
    }
    STAND_DISTANCES = (0.40, 0.48, 0.58)
    INITIAL_ROBOT_JITTER_RADIUS = 0.0

    def _approach_directions(self):
        return (
            np.asarray(
                get_ithor_bedroom_task_spec(self.scene_name, "grasp_workspace")[
                    "approach_direction"
                ]
            ),
        )

    def _initialize_routine_layout(self):
        self._layout_basket_row()


class _RoutineLaptopMixin(BedroomLaptopMixin):
    BEHAVIOR = "close"
    SPEC_TASK = "laptop"
    LAPTOP_EDGE_CLEARANCE = 0.025

    def _workspace_spec(self):
        spec = super()._workspace_spec()
        if self.scene_name == "FloorPlan329":
            spec["laptop_long_edge_approach_hint"] = [0.0, -1.0]
        return spec

    def _initialize_routine_layout(self):
        self._routine_joints = {"close_laptop": (self.target_semantic_joint, 0.12)}
        if "close_top_drawer" in self.COMPOSITE_ORDERING:
            spec = self.scene_task_spec()["routine_targets"]["close_top_drawer"]
            fixture = self.get_fixture(id=spec["fixture"])
            joint = next(j for j in fixture.joints.values() if j.name == spec["joint"])
            self.sim.data.set_joint_qpos(
                joint.name,
                joint.closed_qpos + 0.8 * (joint.open_qpos - joint.closed_qpos),
            )
            jid = self.sim.model.joint_name2id(joint.name)
            self.sim.data.qvel[int(self.sim.model.jnt_dofadr[jid])] = 0.0
            self._routine_joints["close_top_drawer"] = (joint, 0.20)
            self.sim.forward()


class _RoutineMugLaptopMixin(_RoutineBasketMixin, _RoutineLaptopMixin):
    def _initialize_routine_layout(self):
        super()._initialize_routine_layout()
        self.work_surface_region = self.laptop_work_surface_region
        self.work_surface_fixture = self.laptop_work_surface_fixture
        # The parent's near-desk controller stance can obstruct the basket
        # probe. Move to the final far XY before laying out the objects; the
        # routine reset still performs the final ground and safety checks.
        pose = self.scene_task_spec()["routine_start"]["far_start_pose"]
        if not self._set_robot_free_pose(pose["xy"], float(pose["yaw"])):
            raise RuntimeError(
                f"{self.scene_name}: cannot move robot clear of routine basket layout"
            )
        self._layout_basket_row(laptop=True)
        self.target_fixture_xy = self.target_fixture.world_center(self.sim)[:2].copy()
        self.target_joint_pos = self._joint_interaction_pos(self.target_semantic_joint)
        self.target_face_pos = self._joint_target_pos(self.target_semantic_joint)


class _RoutineHamperMixin(_FloorHamperMixin, BedroomArticulationMixin):
    BEHAVIOR = "close"
    SPEC_TASK = "hamper_lid"
    OPEN_INIT_FRACTION = 1.0

    def _initialize_routine_layout(self):
        self._routine_joints = {"close_hamper": (self.target_semantic_joint, 0.12)}


class IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket(
    _ThreeStageRoutineMixin, _RoutineTableMixin, IThorBedroomEnv
):
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    SPEC_TASK = "can_into_basket"
    ITEM_NAMES = ("can", "boxed_food")
    COMPOSITE_ORDERING = ("approach_desk", "put_can", "put_boxed_food")
    SCENE_SPECS = _scene_specs(
        SUPPORTED_SCENE_IDS,
        SPEC_TASK,
        COMPOSITE_ORDERING,
        "Approach the desk, put the can in the basket, then put the boxed food in the same basket.",
    )
    # Scene 323 uses the exact navigation atomic spawn, not the secondary
    # workspace pose; other scenes retain their original start calibration.
    SCENE_SPECS["FloorPlan323"]["routine_start"] = get_ithor_bedroom_task_spec(
        "FloorPlan323", "approach_desk"
    )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        if self.scene_name not in ("FloorPlan306", "FloorPlan323"):
            return
        # This routine does not inherit BedroomSurfacePlacementMixin, so its
        # workspace cleanup would otherwise be skipped. Keep removals local
        # to these two newly supported scenes and reuse the atomic IDs.
        workspace = get_ithor_bedroom_task_spec(self.scene_name, "grasp_workspace")
        removal_ids = tuple(
            dict.fromkeys(
                (*workspace.get("seating", ()), *workspace.get("remove_fixtures", ()))
            )
        )
        fixtures = [self.get_fixture(id=stable_id) for stable_id in removal_ids]
        missing = [
            stable_id
            for stable_id, fixture in zip(removal_ids, fixtures)
            if fixture is None
        ]
        if missing:
            raise ValueError(
                f"{self.scene_name}: unknown exact workspace removals {missing}"
            )
        self.remove_semantic_fixtures(fixtures)


class IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop(
    _ThreeStageRoutineMixin, _RoutineMugLaptopMixin, IThorBedroomEnv
):
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    ITEM_NAMES = ("mug",)
    COMPOSITE_ORDERING = ("approach_desk", "put_mug", "close_laptop")
    SCENE_SPECS = _scene_specs(
        SUPPORTED_SCENE_IDS,
        "laptop",
        COMPOSITE_ORDERING,
        "Approach the desk, put the mug in the basket, then close the laptop.",
    )
    # Keep the secondary-workspace staging pose for object placement; the
    # actual 323 spawn and first navigation target come from the desk atomic.
    SCENE_SPECS["FloorPlan323"]["routine_start"][
        "start_pose"
    ] = get_ithor_bedroom_task_spec("FloorPlan323", "approach_desk")["start_pose"]
    SCENE_SPECS["FloorPlan323"]["routine_start"][
        "minimum_start_distance"
    ] = get_ithor_bedroom_task_spec("FloorPlan323", "approach_desk")[
        "minimum_start_distance"
    ]

    def _workspace_spec(self):
        if self.scene_name not in ("FloorPlan302", "FloorPlan323"):
            return super()._workspace_spec()
        # The native laptop sits on a small side table in these scenes; this
        # composite moves it onto the same main desk as the basket and mug.
        workspace = get_ithor_bedroom_task_spec(self.scene_name, "grasp_workspace")
        workspace["laptop_seating"] = workspace.get("seating", ())
        return workspace


class IThorBedroomCloseLaptopThenApproachBedAndCloseTopDrawer(
    _ThreeStageRoutineMixin, _RoutineLaptopMixin, IThorBedroomEnv
):
    SUPPORTED_SCENE_IDS = (302, 306, 310, 329)
    # The near spawn must also clear the desk after SONIC poses the arms and
    # seats the feet. The parent's 0.28 m candidate only clears its MJCF pose.
    LAPTOP_STAND_EDGE_DISTANCES = (0.40, 0.42, 0.50, 0.60, 0.72)
    COMPOSITE_ORDERING = ("close_laptop", "approach_bed", "close_top_drawer")
    SCENE_SPECS = _scene_specs(
        SUPPORTED_SCENE_IDS,
        "laptop",
        COMPOSITE_ORDERING,
        "Close the laptop, approach the bed, then close the top bedside drawer.",
        far=False,
    )

    def _finalize_articulation_robot_pose(self, distance):
        if self.scene_name == "FloorPlan302":
            # The MJCF stance clears the doorway, but SONIC's left middle
            # finger intersects its frame. Evaluate candidate poses with the
            # actual controller stance before the parent's final ground snap.
            EnvUtils._apply_controller_spawn_pose(self)
        return super()._finalize_articulation_robot_pose(distance)


class IThorBedroomApproachDeskAndCloseLaptopThenApproachBed(
    _ThreeStageRoutineMixin, _RoutineLaptopMixin, IThorBedroomEnv
):
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    COMPOSITE_ORDERING = ("approach_desk", "close_laptop", "approach_bed")
    SCENE_SPECS = _scene_specs(
        SUPPORTED_SCENE_IDS,
        "laptop",
        COMPOSITE_ORDERING,
        "Approach the desk, close the laptop, then approach the bed.",
    )
    SCENE_SPECS["FloorPlan323"]["routine_start"] = get_ithor_bedroom_task_spec(
        "FloorPlan323", "approach_desk"
    )

    def _workspace_spec(self):
        if self.scene_name not in ("FloorPlan302", "FloorPlan323"):
            return super()._workspace_spec()
        # The atomic laptop workspace is a bedside table in these two scenes,
        # metres away from the named desk. Stage the laptop on the actual desk
        # for this approach-desk routine alone.
        workspace = get_ithor_bedroom_task_spec(self.scene_name, "grasp_workspace")
        workspace["laptop_seating"] = workspace.get("seating", ())
        return workspace


def _hamper_desk_scene_specs(scene_ids, stages, lang):
    specs = _scene_specs(scene_ids, "hamper_lid", stages, lang)
    for scene_id in scene_ids:
        if scene_id == 306:
            continue  # Preserve the native hamper routine and its existing spawn.
        name = f"FloorPlan{scene_id}"
        spec = specs[name]
        navigation = get_ithor_bedroom_task_spec(name, "approach_hamper_lid")
        spec["routine_start"] = deepcopy(navigation)
        spec["routine_targets"]["approach_storage"] = deepcopy(navigation)
        spec["routine_targets"]["approach_desk"] = get_ithor_bedroom_task_spec(
            name, "approach_desk"
        )
    return specs


class IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk(
    _ThreeStageRoutineMixin, _RoutineHamperMixin, IThorBedroomEnv
):
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    FINAL_NAV_MIN_DISPLACEMENT = 1.0
    COMPOSITE_ORDERING = ("approach_storage", "close_hamper", "approach_desk")
    SCENE_SPECS = _hamper_desk_scene_specs(
        SUPPORTED_SCENE_IDS,
        COMPOSITE_ORDERING,
        "Approach the laundry hamper, close its lid, then walk over to the desk, keeping the lid closed.",
    )

    def _routine_geometry(self, stage):
        valid = super()._routine_geometry(stage)
        # Restore the evaluation rule only for the native scene306 routine.
        if self.scene_name == "FloorPlan306" and stage == "approach_desk":
            valid = (
                valid
                and np.linalg.norm(self._robot_base_xy() - self._routine_entry_xy)
                >= self.FINAL_NAV_MIN_DISPLACEMENT
            )
        return bool(valid)

    def _initial_joint_qpos(self, joint):
        if self._fixture_mode() != "spawn_asset":
            return super()._initial_joint_qpos(joint)
        # Match the atomic close task; a fully open spawned lid touches floor.
        return float(joint.closed_qpos + 0.85 * (joint.open_qpos - joint.closed_qpos))

    def _routine_stage_requires_contact(self, stage):
        # For the spawned hamper scenes, the close stage is gated only by the
        # lid state, matching the two-stage hamper composites. Preserve the
        # native 306 contact requirement and all other routine stages.
        if (
            stage == "close_hamper"
            and getattr(self, "scene_name", "FloorPlan306") != "FloorPlan306"
        ):
            return False
        return super()._routine_stage_requires_contact(stage)

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.pop("bed_entry_displacement", None)
        if self.scene_name == "FloorPlan306":
            meta["final_navigation_min_displacement"] = self.FINAL_NAV_MIN_DISPLACEMENT
        return meta


class IThorBedroomPutCanThenBoxedFoodThenMugInBasket(
    _ThreeStageRoutineMixin, _RoutineTableMixin, IThorBedroomEnv
):
    SUPPORTED_SCENE_IDS = (302, 306, 310, 323, 329)
    SPEC_TASK = "can_into_basket"
    ITEM_NAMES = ("can", "boxed_food", "mug")
    # Explicit task-only basket variant; the canonical basket and all sources
    # keep their shared sizes in every other task.
    BASKET_ASSET = (BASKET_MODEL, BASKET_SCALE * 1.20)
    # Leave room for the grounded controller stance before probing the basket;
    # all three sources and the basket remain reachable from this near pose.
    # Distances are measured from the table edge, not the source centers.
    STAND_DISTANCES = (0.30, 0.32, 0.40)
    ROW_FACING_STAND_DISTANCES = (0.40, 0.48, 0.58, 0.68, 0.80)
    COMPOSITE_ORDERING = ("put_can", "put_boxed_food", "put_mug")
    SCENE_SPECS = _scene_specs(
        SUPPORTED_SCENE_IDS,
        SPEC_TASK,
        COMPOSITE_ORDERING,
        "Put the can, then the boxed food, then the mug in the same larger basket. Stable stacking is allowed.",
        far=False,
        basket_asset=BASKET_ASSET,
    )

    def _place_robot_at_surface(self):
        if self.scene_name in (
            "FloorPlan302",
            "FloorPlan306",
            "FloorPlan310",
            "FloorPlan323",
        ):
            # The older near-edge 0.30 m stance intersects these desks once
            # the controller seats the SONIC arms. Keep 329 unchanged.
            self.STAND_DISTANCES = (0.40, 0.48, 0.58, 0.68, 0.80)
        return super()._place_robot_at_surface()

    def _initialize_routine_layout(self):
        super()._initialize_routine_layout()
        if self.scene_name != "FloorPlan302":
            return
        # FloorPlan302's default surface approach reaches the north side of
        # the desk, while this routine lays its basket-and-source row on the
        # east edge. Move to that row's front, rather than merely rotating at
        # the unrelated table side. This is isolated to this task and scene.
        workspace = get_ithor_bedroom_task_spec(self.scene_name, "grasp_workspace")
        region = self.work_surface_region
        center, outward, _ = self._surface_layout_frame(
            region,
            operator_xy=(
                region.world_pose(self.sim)[0][:2]
                + 2.0 * np.asarray(workspace["approach_direction"])
            ),
        )
        extent = self.support_region_radial_extent(region, outward, sim=self.sim)
        row_xy = np.mean(
            [self.object_pos(name)[:2] for name in ("basket", *self.ITEM_NAMES)],
            axis=0,
        )
        for distance in self.ROW_FACING_STAND_DISTANCES:
            robot_xy = center[:2] + outward * (extent + distance)
            yaw = float(np.arctan2(row_xy[1] - robot_xy[1], row_xy[0] - robot_xy[0]))
            if hasattr(self, "_robot_free_joint_z"):
                del self._robot_free_joint_z
            if not (
                self.robot_footprint_is_on_floor(robot_xy, radius=0.25)
                and self._set_robot_free_pose(robot_xy, yaw)
                and self._calibrate_robot_free_joint_height()
                and self.count_body_collisions() == 0
            ):
                continue
            self.robot_base_pos = np.array([robot_xy[0], robot_xy[1], 0.0])
            self.robot_base_ori = np.array([0.0, 0.0, yaw])
            return
        raise RuntimeError(
            f"{self.scene_name}: cannot stand in front of routine object row"
        )

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        if self.scene_name == "FloorPlan310":
            # Only the chair obstructs this routine's controller stance;
            # preserve the other native desk objects and shared calibrations.
            self.remove_semantic_fixtures(
                [self.get_fixture(id="FloorPlan310:Chair_204_2_16")]
            )
            return
        if self.scene_name not in ("FloorPlan302", "FloorPlan306", "FloorPlan323"):
            return
        # The table-only mixin does not apply grasp_workspace removals.
        # Clear only these new layouts using the atomic task's exact IDs.
        workspace = get_ithor_bedroom_task_spec(self.scene_name, "grasp_workspace")
        # The 302 floor racket lies under this near-desk stance: without
        # removing it the hands stop ground snapping 2.5 cm above the floor.
        floor_obstructions = (
            ("FloorPlan302:Tennis_Racquet_1_13",)
            if self.scene_name == "FloorPlan302"
            else ()
        )
        removal_ids = tuple(
            dict.fromkeys(
                (
                    *workspace.get("seating", ()),
                    *workspace.get("remove_fixtures", ()),
                    *floor_obstructions,
                )
            )
        )
        fixtures = [self.get_fixture(id=stable_id) for stable_id in removal_ids]
        missing = [
            stable_id
            for stable_id, fixture in zip(removal_ids, fixtures)
            if fixture is None
        ]
        if missing:
            raise ValueError(
                f"{self.scene_name}: unknown exact workspace removals {missing}"
            )
        self.remove_semantic_fixtures(fixtures)

    def _get_obj_cfgs(self):
        configs = super()._get_obj_cfgs()
        for config in configs:
            if config["name"] == "basket":
                config["object_scale"] = self.BASKET_ASSET[1]
        return configs

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["basket_variant"] = "three_item_large"
        return meta


TASK_CLASSES = (
    IThorBedroomApproachDeskAndPutCanThenBoxedFoodInBasket,
    IThorBedroomApproachDeskAndPutMugInBasketThenCloseLaptop,
    IThorBedroomCloseLaptopThenApproachBedAndCloseTopDrawer,
    IThorBedroomApproachDeskAndCloseLaptopThenApproachBed,
    IThorBedroomApproachStorageAndCloseHamperLidThenApproachDesk,
    IThorBedroomPutCanThenBoxedFoodThenMugInBasket,
)
__all__ = [cls.__name__ for cls in TASK_CLASSES]
