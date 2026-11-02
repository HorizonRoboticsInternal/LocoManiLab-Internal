"""Move the calibrated Holodeck glass cup across its desk."""

import os

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.holodeck.tasks.ice_cube_in_glass import (
    HolodeckIceCubeInGlass,
)
from robocasa.models.objects.objects import MJCFObject
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec
from robosuite.utils.mjcf_utils import new_site


class HolodeckMoveCupToCenter(HolodeckEnv):
    """Move the desk cup into the calibrated central target region."""

    ATOMIC_SKILL = "PickPlaceOnSurface"
    TASK_FAMILY = "pick_place"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "move_cup_to_center"
    CALIBRATION_SPEC_TASK = SPEC_TASK
    CUP_ASSET = HolodeckIceCubeInGlass.CUP_ASSET
    CALIBRATED_REPLACED_BODIES = {
        "train_117": "holodeck117_table_lamp",
        "train_1018": "tablelamp_ecbe5eedc41fd2b8bc6ea19f402eb6a9_1_0_0",
    }
    CALIBRATED_LAPTOP_BODIES = {
        "train_117": "laptop_6ae82082b761fc8537b74b4d805f7cb1_1_0_0",
        "train_1664": "laptop_f09fcf086ebe9e3ab2d97bff50b833ef_1_0_0",
        "train_2025": "laptop_63fbcd680e0db93dfb0454951f2114f3_1_0_0",
    }
    HOLD_SECONDS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    USE_DEFAULT_ASSET_LAYOUT = False
    TARGET_REGION_SITE_NAME = "cup_center_target_region"
    SHOW_TARGET_REGION = False
    NOISE_FREE_ROBOT_SPAWN = True
    INITIAL_CUP_DISTANCE = 0.45
    INITIAL_MIN_TARGET_CLEARANCE = 0.10
    INITIAL_DESK_COLLISION_DISTANCE = 0.10
    INITIAL_DESK_DISTANCE_TOLERANCE = 0.01
    INITIAL_REQUIRE_LATERAL_TARGET_OFFSET = True
    INITIAL_REQUIRE_RIGHT_TARGET_OFFSET = True
    INITIAL_MAX_LONGITUDINAL_TARGET_OFFSET = 1e-3
    INITIAL_LAYOUT_VALIDATION_DISABLED_SCENES = {
        "train_117",
        "train_1018",
        "train_1197",
        "train_1664",
        "train_2025",
    }
    VALIDATE_INITIAL_LAYOUT = True

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.CALIBRATION_SPEC_TASK,
            required=(
                "desk_body",
                "cup_scale",
                "cup_pose",
                "initial_cup_base_xy",
                "robot_pose",
                "target_base_xy",
                "target_tolerance",
                "tabletop_bounds_xy",
                "support_z",
                "support_z_tolerance",
                "upright_local_axis",
                "min_upright_cosine",
                "max_linear_speed",
                "max_angular_speed",
            ),
        )
        remove_scene_bodies = tuple(self.task_spec.get("remove_scene_bodies", ()))
        if remove_scene_bodies:
            removed = self.remove_scene_bodies(remove_scene_bodies)
            missing = set(remove_scene_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured task bodies are missing: "
                    f"{sorted(missing)}"
                )
        # Holodeck Objaverse exports may leave the desk objectType/category
        # empty, so resolve the calibrated surface by its exact baked body.
        # Prefer the semantic record because corner-target geometry needs its
        # local bounds; the raw ProcTHOR fixture record is only a fallback.
        desk_body = self.task_spec["desk_body"]
        semantic_fixtures = getattr(self.semantic_scene, "fixtures", {}).values()
        self.desk_fixture = next(
            (
                fixture
                for fixture in semantic_fixtures
                if fixture.body_name == desk_body
            ),
            None,
        )
        if self.desk_fixture is None:
            self.desk_fixture = next(
                (
                    fixture
                    for fixture in getattr(self, "fixtures", {}).values()
                    if fixture.body_name == desk_body
                ),
                None,
            )
        if self.desk_fixture is None:
            desks = self.get_fixture(category="desk", return_all=True) or ()
            self.desk_fixture = next(
                (fixture for fixture in desks if fixture.body_name == desk_body),
                None,
            )
        if self.desk_fixture is not None:
            self.fixture_refs["cup_surface"] = self.desk_fixture
            pose = self.task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )
            if self.SHOW_TARGET_REGION:
                target = np.asarray(self.task_spec["target_base_xy"], dtype=float)
                tolerance = np.asarray(self.task_spec["target_tolerance"], dtype=float)
                self.mujoco_arena.worldbody.append(
                    new_site(
                        name=(
                            f"{self.TARGET_REGION_SITE_NAME}_"
                            f"{self.scene_name}"
                        ),
                        type="box",
                        pos=[
                            float(target[0]),
                            float(target[1]),
                            float(self.task_spec["support_z"]) + 0.012,
                        ],
                        size=[float(tolerance[0]), float(tolerance[1]), 0.01],
                        rgba=[0.1, 0.9, 0.2, 0.35],
                        group=0,
                    )
                )

    def _is_scene_valid(self):
        return self.desk_fixture is not None

    def _setup_baked_scene_fixtures(self):
        replaced_body = self.CALIBRATED_REPLACED_BODIES.get(self.scene_name)
        if replaced_body is not None:
            removed = self.remove_scene_bodies((replaced_body,))
            if replaced_body not in removed:
                raise RuntimeError(
                    f"{self.scene_name}: configured source body is missing: "
                    f"{replaced_body}"
                )
        laptop_body = self.CALIBRATED_LAPTOP_BODIES.get(self.scene_name)
        if laptop_body is not None:
            removed = self.remove_scene_bodies((laptop_body,))
            if laptop_body not in removed:
                raise RuntimeError(
                    f"{self.scene_name}: configured laptop body is missing: "
                    f"{laptop_body}"
                )
        super()._setup_baked_scene_fixtures()

    def _get_obj_cfgs(self):
        return []

    def _create_objects(self):
        super()._create_objects()
        cup = MJCFObject(
            name="cup",
            mjcf_path=os.path.join(robocasa.models.assets_root, self.CUP_ASSET),
            scale=float(self.task_spec["cup_scale"]),
        )
        self.objects[cup.name] = cup
        self.model.merge_objects([cup])

    def _setup_scene(self):
        super()._setup_scene()
        self._set_initial_cup_state()
        self.sim.forward()
        self._cup_was_grasped = False
        self._success_hold_steps = 0

    def _reset_internal(self):
        super()._reset_internal()
        self._set_initial_cup_state()
        self.sim.forward()
        validate_layout = self.VALIDATE_INITIAL_LAYOUT and not (
            self.__class__ in (HolodeckMoveCupToCenter, HolodeckMoveCupToCorner)
            and getattr(self, "scene_name", None)
            in self.INITIAL_LAYOUT_VALIDATION_DISABLED_SCENES
        )
        if validate_layout and not self._initial_layout_valid():
            raise RuntimeError(
                f"{self.scene_name}: cup reset layout is invalid; "
                f"{self._initial_layout_failure_details()}"
            )
        self._cup_was_grasped = False
        self._success_hold_steps = 0

    def _set_initial_cup_state(self):
        pose = dict(self.task_spec["cup_pose"])
        pose["pos"] = [
            *self._initial_cup_base_xy(),
            float(self.task_spec["cup_pose"]["pos"][2]),
        ]
        self._place_cup(pose)

    def _initial_cup_base_xy(self):
        explicit_initial = self.task_spec.get("initial_cup_base_xy")
        initial = np.asarray(explicit_initial, dtype=float)
        if initial.shape != (2,):
            raise ValueError(
                f"{self.scene_name}: initial_cup_base_xy must contain XY"
            )
        return initial

    def _desk_target_frame(self):
        """Return the desk-local long and short directions in world XY."""
        local_half_size = getattr(self.desk_fixture, "local_half_size", None)
        if local_half_size is None:
            support = self.desk_fixture.get_region("support", required=False)
            if support is None:
                desk_yaw = float(self.desk_fixture.yaw)
                half_size = np.asarray(self.desk_fixture.half_size[:2], dtype=float)
            else:
                _, desk_yaw = support.world_pose()
                half_size = np.asarray(support.half_size[:2], dtype=float)
        else:
            desk_yaw = float(self.desk_fixture.yaw)
            half_size = np.asarray(local_half_size[:2], dtype=float)
        long_axis = int(np.argmax(half_size))
        long_direction = np.array(
            [
                np.cos(desk_yaw + long_axis * np.pi / 2),
                np.sin(desk_yaw + long_axis * np.pi / 2),
            ],
            dtype=float,
        )
        long_direction /= np.linalg.norm(long_direction)
        short_direction = np.array([-long_direction[1], long_direction[0]], dtype=float)
        return long_direction, short_direction

    def _initial_layout_valid(self):
        """Validate the calibrated source pose and robot standoff at reset."""
        desk_fixture = getattr(self, "desk_fixture", None)
        if desk_fixture is None:
            # Scene validity is checked separately before a runnable episode is
            # accepted. Keep lightweight unit-test doubles usable as well.
            return True

        source = self._initial_cup_base_xy()
        target = self._target_base_xy()
        tolerance = self._target_tolerance()
        separation = source - target
        outside = np.maximum(np.abs(separation) - tolerance, 0.0)
        if (
            np.linalg.norm(outside)
            < self.INITIAL_MIN_TARGET_CLEARANCE
        ):
            return False

        robot_xy = self._robot_base_xy()
        robot_to_cup = source - robot_xy
        if not np.isclose(
            np.linalg.norm(robot_to_cup),
            self.INITIAL_CUP_DISTANCE,
            atol=1e-3,
        ):
            return False

        robot_yaw = self._robot_base_yaw()
        view_axis = np.array([np.cos(robot_yaw), np.sin(robot_yaw)])
        lateral_axis = np.array([-view_axis[1], view_axis[0]])
        lateral_offset = float(np.dot(separation, lateral_axis))
        longitudinal_offset = float(np.dot(separation, view_axis))
        if self.INITIAL_REQUIRE_LATERAL_TARGET_OFFSET:
            if (
                abs(longitudinal_offset)
                > self.INITIAL_MAX_LONGITUDINAL_TARGET_OFFSET
            ):
                return False
            if self.INITIAL_REQUIRE_RIGHT_TARGET_OFFSET:
                # Positive right is the negative lateral axis in the robot's
                # viewing frame. Keep the cup on that side of the target.
                if -lateral_offset <= 0.0:
                    return False
            elif abs(lateral_offset) <= 0.0:
                return False
        expected_yaw = np.arctan2(robot_to_cup[1], robot_to_cup[0])
        yaw_error = np.arctan2(
            np.sin(robot_yaw - expected_yaw),
            np.cos(robot_yaw - expected_yaw),
        )
        if abs(yaw_error) > 1e-3:
            return False

        # The 10 cm calibration is defined for G1FloatingBody. Other G1
        # variants expose different fixed-base collision trees, so their raw
        # geom distance is not comparable and must not reject the layout.
        if self._uses_g1_collision_model():
            desk_distance = self.robot_fixture_collision_distance(desk_fixture)
            if not np.isfinite(desk_distance) or desk_distance < (
                self.INITIAL_DESK_COLLISION_DISTANCE
                - self.INITIAL_DESK_DISTANCE_TOLERANCE
            ):
                return False
        return bool(
            self.count_body_collisions() == 0
            and abs(self._cup_bottom_z() - float(self.task_spec["support_z"]))
            <= float(self.task_spec["support_z_tolerance"])
            and self._cup_is_upright()
        )

    def _uses_g1_collision_model(self):
        robots = getattr(self, "robots", ())
        robot_model = getattr(robots[0], "robot_model", None) if robots else None
        model_name = type(robot_model).__name__ if robot_model is not None else ""
        return model_name == "G1FloatingBody"

    def _initial_layout_failure_details(self):
        """Return measured reset values to make calibration failures actionable."""
        source = self._initial_cup_base_xy()
        robot = self._robot_base_xy()
        target = self._target_base_xy()
        details = {
            "robot": [float(value) for value in robot],
            "cup": [float(value) for value in source],
            "target": [float(value) for value in target],
            "cup_distance": float(np.linalg.norm(source - robot)),
            "collisions": int(self.count_body_collisions()),
        }
        if getattr(self, "desk_fixture", None) is not None:
            details["desk_distance"] = float(
                self.robot_fixture_collision_distance(self.desk_fixture)
            )
        return ", ".join(f"{key}={value}" for key, value in details.items())

    def _place_cup(self, pose):
        joint = self.objects["cup"].joints[0]
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[:3] = np.asarray(pose["pos"], dtype=float)
        qpos[3:7] = np.asarray(pose["quat"], dtype=float)
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.data.set_joint_qvel(joint, np.zeros(6))

    def _cup_body_ids(self):
        return self._body_tree_ids([self.obj_body_id["cup"]])

    def _desk_body_ids(self):
        return self._fixture_body_ids(self.desk_fixture, include_siblings=True)

    def _cup_base_center(self):
        return self.object_pos("cup")

    def _cup_bottom_z(self):
        points = self.object_bbox_points("cup")
        return float(points[:, 2].min()) if len(points) else -np.inf

    def _cup_is_upright(self):
        body_id = self.obj_body_id["cup"]
        root_mat = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(
            3, 3
        )
        local_up = np.asarray(self.task_spec["upright_local_axis"], dtype=float)
        world_up = root_mat @ local_up
        return float(world_up[2]) >= float(self.task_spec["min_upright_cosine"])

    def _cup_contacts_desk(self):
        cup_bodies = self._cup_body_ids()
        desk_bodies = self._desk_body_ids()
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in cup_bodies and body2 in desk_bodies) or (
                body2 in cup_bodies and body1 in desk_bodies
            ):
                return True
        return False

    def _cup_released(self):
        cup_bodies = self._cup_body_ids()
        robot_root = self.sim.model.body_name2id(self.robots[0].robot_model.root_body)
        robot_bodies = self._body_tree_ids([robot_root])
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            body1 = int(self.sim.model.geom_bodyid[contact.geom1])
            body2 = int(self.sim.model.geom_bodyid[contact.geom2])
            if (body1 in cup_bodies and body2 in robot_bodies) or (
                body2 in cup_bodies and body1 in robot_bodies
            ):
                return False
        return True

    def _cup_is_grasped(self):
        cup_bodies = self._cup_body_ids()
        for side in ("left", "right"):
            hand_bodies = self._body_ids_matching((f"{side}_hand",))
            contacted = False
            for index in range(self.sim.data.ncon):
                contact = self.sim.data.contact[index]
                body1 = int(self.sim.model.geom_bodyid[contact.geom1])
                body2 = int(self.sim.model.geom_bodyid[contact.geom2])
                if (body1 in cup_bodies and body2 in hand_bodies) or (
                    body2 in cup_bodies and body1 in hand_bodies
                ):
                    contacted = True
                    break
            if contacted and self.g1_hand_closed(side):
                return True
        return False

    def _cup_in_target_state(self):
        """Require the cup to be inside the target region and resting on the desk."""
        return bool(
            self._cup_in_target_region(self._cup_base_center())
            and self._cup_contacts_desk()
        )

    def _cup_in_target_region(self, base_center):
        target = self._target_base_xy()
        tolerance = self._target_tolerance()
        return np.all(np.abs(base_center[:2] - target) <= tolerance)

    def _target_base_xy(self):
        return np.asarray(self.task_spec["target_base_xy"], dtype=float)

    def _target_tolerance(self):
        return np.asarray(self.task_spec["target_tolerance"], dtype=float)

    @property
    def hold_steps(self):
        return max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self._cup_in_target_state():
            self._success_hold_steps += 1
        else:
            self._success_hold_steps = 0
        return ret

    def _check_success(self):
        return self._success_hold_steps >= self.hold_steps

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "move the glass cup to the center"
        return meta


class HolodeckMoveCupToCorner(HolodeckMoveCupToCenter):
    """Move the desk cup into either robot-side desk-corner target region."""

    SPEC_TASK = "move_cup_to_corner"
    CALIBRATION_SPEC_TASK = SPEC_TASK
    # The target half-size is expressed in the desk-local long/short basis.
    CORNER_TARGET_TOLERANCE = np.asarray((0.12, 0.12), dtype=float)
    INITIAL_CUP_TOWARD_ROBOT = 0.05
    INITIAL_MIN_TARGET_CLEARANCE = INITIAL_CUP_TOWARD_ROBOT
    INITIAL_REQUIRE_LATERAL_TARGET_OFFSET = False
    TARGET_REGION_SITE_NAME = "cup_corner_target_region"
    SHOW_TARGET_REGION = False
    SHOW_CORNER_TARGET_REGION = False
    VALIDATE_INITIAL_LAYOUT = True
    INITIAL_REQUIRE_RIGHT_TARGET_OFFSET = False

    def _get_corner_target_base_xys(self):
        long_direction, short_direction = self._desk_target_frame()

        explicit_targets = self.task_spec.get("corner_target_base_xys")
        if explicit_targets is not None:
            targets = np.asarray(explicit_targets, dtype=float)
            if targets.shape != (2, 2):
                raise ValueError(
                    f"{self.scene_name}: corner_target_base_xys must contain "
                    "two XY targets"
                )
            return targets

        tabletop_bounds = np.asarray(
            self.task_spec.get("tabletop_bounds_xy"), dtype=float
        )
        if tabletop_bounds.shape != (2, 2) or np.any(
            tabletop_bounds[1] <= tabletop_bounds[0]
        ):
            raise ValueError(
                f"{self.scene_name}: tabletop_bounds_xy must contain valid XY bounds"
            )

        tabletop_center = tabletop_bounds.mean(axis=0)
        tabletop_corners = np.asarray(
            (
                [tabletop_bounds[0, 0], tabletop_bounds[0, 1]],
                [tabletop_bounds[0, 0], tabletop_bounds[1, 1]],
                [tabletop_bounds[1, 0], tabletop_bounds[0, 1]],
                [tabletop_bounds[1, 0], tabletop_bounds[1, 1]],
            ),
            dtype=float,
        )
        tabletop_local = np.column_stack(
            (
                (tabletop_corners - tabletop_center) @ long_direction,
                (tabletop_corners - tabletop_center) @ short_direction,
            )
        )
        long_min, long_max = tabletop_local[:, 0].min(), tabletop_local[:, 0].max()
        short_min, short_max = tabletop_local[:, 1].min(), tabletop_local[:, 1].max()

        robot = np.asarray(self.task_spec["robot_pose"]["pos"], dtype=float)[:2]
        robot_side = float(np.dot(robot - tabletop_center, short_direction))
        if abs(robot_side) <= 1e-8:
            raise ValueError(f"{self.scene_name}: robot is on the desk long axis")
        robot_side = np.sign(robot_side)

        target_long_half, target_short_half = self.CORNER_TARGET_TOLERANCE
        if (
            long_max - long_min <= 2.0 * target_long_half
            or short_max - short_min <= 2.0 * target_short_half
        ):
            raise ValueError(
                f"{self.scene_name}: tabletop is smaller than the corner target"
            )

        near_short_center = (
            short_max - target_short_half
            if robot_side > 0
            else short_min + target_short_half
        )
        long_centers = (long_max - target_long_half, long_min + target_long_half)
        return np.asarray(
            [
                tabletop_center
                + long_center * long_direction
                + near_short_center * short_direction
                for long_center in long_centers
            ],
            dtype=float,
        )

    def _corner_target_frame(self):
        return self._desk_target_frame()

    def _right_corner_target_index(self):
        """Return the corner on the robot's right in its initial viewing frame."""
        robot = np.asarray(self.task_spec["robot_pose"]["pos"], dtype=float)[:2]
        yaw = float(self.task_spec["robot_pose"]["yaw"])
        left_direction = np.array([-np.sin(yaw), np.cos(yaw)], dtype=float)
        offsets = self._corner_target_base_xys - robot
        return int(np.argmin(offsets @ left_direction))

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self._corner_target_base_xys = self._get_corner_target_base_xys()
        if self.SHOW_CORNER_TARGET_REGION:
            right_index = self._right_corner_target_index()
            for index, target in enumerate(self._corner_target_base_xys):
                side = "right" if index == right_index else "left"
                long_direction, _ = self._corner_target_frame()
                target_yaw = float(np.arctan2(long_direction[1], long_direction[0]))
                self.mujoco_arena.worldbody.append(
                    new_site(
                        name=f"{self.TARGET_REGION_SITE_NAME}_{self.scene_name}_{side}",
                        type="box",
                        pos=[
                            *target,
                            float(self.task_spec["support_z"]) + 0.012,
                        ],
                        size=[*self.CORNER_TARGET_TOLERANCE, 0.01],
                        quat=[
                            float(np.cos(target_yaw / 2)),
                            0.0,
                            0.0,
                            float(np.sin(target_yaw / 2)),
                        ],
                        rgba=[0.1, 0.9, 0.2, 0.35],
                        group=0,
                    )
                )

    def _target_base_xy(self):
        return self._corner_target_base_xys[self._right_corner_target_index()]

    def _target_tolerance(self):
        return self.CORNER_TARGET_TOLERANCE

    def _cup_in_target_region(self, base_center):
        long_direction, short_direction = self._corner_target_frame()
        return any(
            np.all(
                np.abs(
                    [
                        np.dot(base_center[:2] - target, long_direction),
                        np.dot(base_center[:2] - target, short_direction),
                    ]
                )
                <= self.CORNER_TARGET_TOLERANCE
            )
            for target in self._corner_target_base_xys
        )

    def _initial_layout_valid(self):
        """Check the right-corner source offset in addition to the base layout."""
        if not super()._initial_layout_valid():
            return False

        source = self._initial_cup_base_xy()
        target = self._target_base_xy()
        yaw = self._robot_base_yaw()
        view_direction = np.array([np.cos(yaw), np.sin(yaw)], dtype=float)
        left_direction = np.array([-view_direction[1], view_direction[0]])
        offset = source - target
        expected_left_offset = (
            float(self.CORNER_TARGET_TOLERANCE[0])
            + self.INITIAL_CUP_TOWARD_ROBOT
        )
        if not np.isclose(
            np.dot(offset, left_direction), expected_left_offset, atol=1e-3
        ):
            return False

        if self._uses_g1_collision_model():
            desk_distance = self.robot_fixture_collision_distance(self.desk_fixture)
            if not np.isfinite(desk_distance) or not np.isclose(
                desk_distance,
                self.INITIAL_DESK_COLLISION_DISTANCE,
                atol=self.INITIAL_DESK_DISTANCE_TOLERANCE,
            ):
                return False
        return True

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = (
            "move the glass cup to either robot-side "
            "desk corner"
        )
        meta["initial_cup_base_xy"] = [
            float(value) for value in self._initial_cup_base_xy()
        ]
        meta["target_cup_base_xy"] = [
            [float(value) for value in target]
            for target in self._corner_target_base_xys
        ]
        return meta


__all__ = [
    "HolodeckMoveCupToCenter",
    "HolodeckMoveCupToCorner",
]
