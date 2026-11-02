"""Compact shared base for MolmoSpaces / iTHOR room environments."""
import os
import re
import xml.etree.ElementTree as ET
from copy import deepcopy
import mujoco
import numpy as np
from robosuite.environments.base import EnvMeta, REGISTERED_ENVS
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils import transform_utils as T
from robosuite.utils.errors import RandomizationError
from robosuite.utils.mjcf_utils import (
    array_to_string,
    get_elements,
    scale_mjcf_model,
    string_to_array,
)
from robosuite.utils.placement_samplers import (
    SequentialCompositeSampler,
    UniformRandomSampler,
)

import robocasa.utils.env_utils as EnvUtils
from robocasa.environments.thor_atomic_metadata import add_atomic_task_metadata
from robocasa.models.arenas.molmospaces_arena import MolmoSpacesArena
from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
from robocasa.models.scenes.ithor_scene_specs import (
    get_scene_robot_pose,
    get_scene_robot_scale,
    get_scene_surface,
)
from robocasa.models.scenes.ithor_support_specs import get_scene_manual_supports
from robocasa.models.scenes.molmospaces_scene_registry import (
    RoomType,
    _ROOM_TYPE_RANGES,
    classify_floorplan,
    get_ithor_scenes_dir,
    get_scene_xml,
    list_scenes,
)
from robocasa.models.scenes.thor_semantics import (
    SemanticFixture,
    SemanticJoint,
    SemanticRegion,
    build_semantic_registry,
    semantic_fixture_from_robocasa_fixture,
)
from robocasa.utils.errors import PlacementError
from robocasa.utils.mjcf_xml_utils import (
    mjcf_xml_bounds,
    rotated_bounds,
)
from robocasa.utils.molmospaces_placement import analyze_scene
from robocasa.utils.scene_initial_jitter import apply_initial_pose_jitter


def make_ithor_registry():
    registry = {}

    class _IThorRoomEnvMeta(EnvMeta):
        def __new__(meta, name, bases, class_dict):
            cls = super().__new__(meta, name, bases, class_dict)
            registry[cls.__name__] = cls
            return cls

    return registry, _IThorRoomEnvMeta


def control_steps_for_duration(duration_seconds, control_freq):
    """Convert a wall-clock task hold duration to environment control steps."""

    duration_seconds = float(duration_seconds)
    control_freq = float(control_freq)
    if (
        not np.isfinite(duration_seconds)
        or duration_seconds <= 0.0
        or not np.isfinite(control_freq)
        or control_freq <= 0.0
    ):
        raise ValueError(
            "duration_seconds and control_freq must be finite and positive"
        )
    return max(1, int(round(duration_seconds * control_freq)))


def make_room_env_category(room_type):
    registry, meta = make_ithor_registry()
    room_name = room_type.value.replace("_", " ").title().replace(" ", "")
    success_hold_seconds = (
        1.0 if room_type in (RoomType.BEDROOM, RoomType.BATHROOM) else None
    )

    class RoomEnv(IThorRoomEnv, metaclass=meta):
        ROOM_TYPE = room_type
        SUCCESS_HOLD_SECONDS = success_hold_seconds

    new_name = f"IThor{room_name}Env"
    del registry["RoomEnv"]
    registry[new_name] = RoomEnv
    del REGISTERED_ENVS["RoomEnv"]
    REGISTERED_ENVS[new_name] = RoomEnv
    RoomEnv.__name__ = new_name
    RoomEnv.__qualname__ = new_name
    RoomEnv.REGISTRY = registry
    RoomEnv.Meta = meta
    return registry, RoomEnv


class IThorRoomEnv(ManipulationEnv):
    ROOM_TYPE = None
    SCENE_FAMILY = "iTHOR"
    SCENE_ID_ARG = "--scene-id"
    REQUIRE_SCENE_CALIBRATION = True
    INITIAL_POSE_JITTER = True
    INITIAL_ROBOT_JITTER_RADIUS = 0.10
    INITIAL_OBJECT_JITTER_RADIUS = 0.10
    IGNORE_OBJECT_PLACEMENT_FAILURE = False
    SUCCESS_HOLD_SECONDS = None

    def __init__(
        self,
        robots,
        scene_ids=None,
        robot_base_pos=None,
        robot_scale=None,
        obj_registries=("objaverse", "lightwheel"),
        obj_instance_split=None,
        fixed_scene_seed=None,
        **kwargs,
    ):
        assert self.ROOM_TYPE is not None, f"{type(self).__name__} needs ROOM_TYPE"
        self._fixed_scene_seed = fixed_scene_seed
        self._robot_scale_override = (
            getattr(self, "ROBOT_SCALE", None) if robot_scale is None else robot_scale
        )
        self._robot_base_pos_override = (
            None if robot_base_pos is None else np.array(robot_base_pos)
        )
        self.robot_base_pos = np.zeros(3)
        self.robot_base_ori = np.zeros(3)
        self.obj_registries = obj_registries
        self.obj_instance_split = obj_instance_split
        downloaded = self._list_downloaded_scenes()
        self.scene_names = (
            sorted(downloaded.keys())
            if scene_ids is None
            else self._scene_ids_to_names(scene_ids, downloaded)
        )
        if not self.scene_names:
            raise RuntimeError(f"no downloaded {self.ROOM_TYPE.value} iTHOR scenes")

        self.furniture = {}
        self.fixtures = {}
        self.fixture_cfgs = []
        self.fixture_refs = {}
        self.region_refs = {}
        self.semantic_scene = None
        self.objects = {}
        self.object_cfgs = []
        self.placement_initializer = None
        self.object_placements = {}
        self.surface = None
        defaults = dict(
            env_configuration="default",
            controller_configs=None,
            gripper_types="default",
            base_types="default",
            initialization_noise="default",
            use_camera_obs=False,
            has_renderer=False,
            has_offscreen_renderer=True,
            render_camera="frontview",
            render_collision_mesh=False,
            render_visual_mesh=True,
            render_gpu_device_id=-1,
            control_freq=20,
            lite_physics=True,
            horizon=1000,
            ignore_done=True,
            hard_reset=True,
            camera_names="agentview",
            camera_heights=256,
            camera_widths=256,
            camera_depths=False,
            camera_segmentations=None,
            renderer="mjviewer",
            renderer_config=None,
            seed=None,
        )
        defaults.update(kwargs)
        super().__init__(robots=robots, **defaults)
        self._configure_success_hold_duration()

    def reset(self):
        observations = super().reset()
        # A hard reset rebuilds the task model and may reload a task-local
        # fallback HOLD_STEPS value from its scene specification.
        self._configure_success_hold_duration()
        return observations

    def _configure_success_hold_duration(self):
        if self.SUCCESS_HOLD_SECONDS is None:
            return
        required_steps = control_steps_for_duration(
            self.SUCCESS_HOLD_SECONDS,
            self.control_freq,
        )
        # HOLD_STEPS is the final-success threshold used by Bedroom and
        # Bathroom atomic tasks and their explicit composite state machines.
        self.HOLD_STEPS = required_steps
        # Ordered multi-stage tasks use this fallback for their final stage.
        self.SUCCESS_HOLD_STEPS = required_steps

    def _list_downloaded_scenes(self):
        return list_scenes(self.ROOM_TYPE)

    def _scene_ids_to_names(self, scene_ids, downloaded):
        ids = [scene_ids] if isinstance(scene_ids, int) else list(scene_ids)
        for scene_id in ids:
            room_type = classify_floorplan(scene_id)
            if room_type != self.ROOM_TYPE:
                lo, hi = _ROOM_TYPE_RANGES[self.ROOM_TYPE]
                raise ValueError(
                    f"{self.SCENE_ID_ARG} {scene_id} is {room_type.value}, "
                    f"but {type(self).__name__} expects [{lo}-{hi}]"
                )
        return [f"FloorPlan{scene_id}" for scene_id in ids]

    def _get_scene_xml(self, scene_name):
        return get_scene_xml(scene_name)

    def _load_model(self):
        if self._fixed_scene_seed is not None:
            self.rng = np.random.default_rng(self._fixed_scene_seed)
        super()._load_model()
        last_error = None
        last_rejection = None
        for _ in range(20):
            self._setup_model()
            self._apply_scene_robot_scale()
            if not self._is_scene_valid():
                last_rejection = getattr(self, "_scene_invalid_reason", None)
                if hasattr(self, "sim"):
                    self._destroy_sim()
                continue
            for robot in self.robots:
                robot.robot_model.set_base_xpos(self.robot_base_pos)
                robot.robot_model.set_base_ori(self.robot_base_ori)
            self.model = ManipulationTask(
                mujoco_arena=self.mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=list(self.furniture.values()),
            )
            self._create_objects()
            if not self.object_cfgs:
                self.placement_initializer = None
                self.object_placements = {}
                return
            try:
                self.placement_initializer = self._build_placement_initializer()
                self.object_placements = self.placement_initializer.sample()
                return
            except (PlacementError, RandomizationError) as exc:
                if self.IGNORE_OBJECT_PLACEMENT_FAILURE:
                    self.placement_initializer = None
                    self.object_placements = {}
                    return
                last_error = exc
        detail = (
            f"{type(last_error).__name__}: {last_error}"
            if last_error is not None
            else str(last_rejection or "no placement exception was recorded")
        )
        raise RuntimeError(f"Could not place iTHOR objects after 20 attempts; {detail}")

    def _apply_scene_robot_scale(self):
        target = (
            self._get_scene_robot_scale_calibration()
            if self._robot_scale_override is None
            else float(self._robot_scale_override)
        )
        if target <= 0.0:
            raise ValueError(f"{type(self).__name__}: robot_scale must be positive")
        for robot in self.robots:
            model = robot.robot_model
            current = float(getattr(model, "_ithor_scene_scale", 1.0))
            ratio = target / current
            if np.isclose(ratio, 1.0):
                continue
            scale_mjcf_model(
                obj=model.worldbody,
                asset_root=model.asset,
                scale=ratio,
                get_elements_func=get_elements,
                worldbody=model.worldbody,
            )
            for camera in model.worldbody.iter("camera"):
                if camera.get("pos") is not None:
                    camera.set(
                        "pos",
                        array_to_string(string_to_array(camera.get("pos")) * ratio),
                    )
            model._ithor_scene_scale = target

    def _get_scene_surface_calibration(self):
        """Return the legacy scene calibration unless a task overrides it."""
        return get_scene_surface(self.scene_name, type(self).__name__)

    def _get_scene_robot_pose_calibration(self):
        """Return the legacy scene calibration unless a task overrides it."""
        return get_scene_robot_pose(self.scene_name, type(self).__name__)

    def _get_scene_robot_scale_calibration(self):
        """Return the legacy scene calibration unless a task overrides it."""
        return get_scene_robot_scale(self.scene_name, type(self).__name__)

    def _build_placement_initializer(self):
        semantic = any(
            isinstance(
                cfg.get("placement", {}).get(key), (SemanticFixture, SemanticRegion)
            )
            for cfg in self.object_cfgs
            for key in ("fixture", "region")
        )
        if semantic:
            return self._build_semantic_placement_initializer()
        if any(
            cfg.get("placement", {}).get("fixture") is not None
            for cfg in self.object_cfgs
        ):
            return EnvUtils._get_placement_initializer(self, self.object_cfgs)
        if self.surface is None:
            raise PlacementError("no placement surface found in scene")
        center, half_xy = self.surface
        sampler = SequentialCompositeSampler(name="SceneSampler")
        for cfg in self.object_cfgs:
            margin = cfg.get("placement", {}).get("margin", 0.0)
            half = np.maximum(np.asarray(half_xy, dtype=float) - margin, 0.0)
            sampler.append_sampler(
                UniformRandomSampler(
                    name=f"{cfg['name']}_Sampler",
                    mujoco_objects=self.objects[cfg["name"]],
                    x_range=[-half[0], half[0]],
                    y_range=[-half[1], half[1]],
                    rotation=cfg.get("placement", {}).get("rotation"),
                    rotation_axis="z",
                    ensure_object_boundary_in_range=False,
                    ensure_valid_placement=True,
                    reference_pos=center,
                    z_offset=0.01,
                )
            )
        return sampler

    def _build_semantic_placement_initializer(self):
        sampler = SequentialCompositeSampler(name="THORSemanticRegionSampler")
        for cfg in self.object_cfgs:
            placement = cfg.get("placement", {})
            fixture = placement.get("fixture")
            region = placement.get("region")
            if isinstance(region, str):
                region = fixture.get_region(region)
            elif region is None and isinstance(fixture, SemanticFixture):
                region = fixture.get_region("support")
            if not isinstance(region, SemanticRegion):
                raise PlacementError(f"{cfg['name']}: invalid semantic region")
            center, _ = region.world_pose()
            margin = float(placement.get("margin", 0.03))
            half = np.maximum(region.half_size[:2] - margin, 0.0)
            if np.any(half < 0.015):
                raise PlacementError(f"{cfg['name']}: semantic region is too small")
            strategy = placement.get("strategy")
            if strategy == "center":
                x_range = y_range = [0.0, 0.0]
                ensure_boundary = False
            elif strategy == "edge":
                target_xy, _ = self.support_region_edge_pose(
                    region,
                    self._object_half_xy(cfg["name"]),
                    self.robot_base_pos[:2],
                    margin=margin,
                    edge_fraction=placement.get("edge_fraction", 1.0),
                    lateral_offset=placement.get("edge_lateral_offset"),
                    sim=self.sim if getattr(self, "sim", None) is not None else None,
                )
                delta = np.asarray(target_xy, dtype=float)[:2] - center[:2]
                x_range = [float(delta[0]), float(delta[0])]
                y_range = [float(delta[1]), float(delta[1])]
                ensure_boundary = False
            else:
                safe = float(np.min(half)) if strategy != "uniform" else None
                x_range = (
                    [-half[0], half[0]] if strategy == "uniform" else [-safe, safe]
                )
                y_range = (
                    [-half[1], half[1]] if strategy == "uniform" else [-safe, safe]
                )
                if placement.get("near_robot_half", False) and hasattr(
                    self, "_near_robot_half_ranges"
                ):
                    x_range, y_range = self._near_robot_half_ranges(region, half)
                ensure_boundary = True
            sampler.append_sampler(
                UniformRandomSampler(
                    name=f"{cfg['name']}_SemanticSampler",
                    mujoco_objects=self.objects[cfg["name"]],
                    x_range=x_range,
                    y_range=y_range,
                    rotation=placement.get("rotation"),
                    rotation_axis=placement.get("rotation_axis", "z"),
                    ensure_object_boundary_in_range=ensure_boundary,
                    ensure_valid_placement=True,
                    reference_pos=center,
                    z_offset=float(placement.get("z_offset", 0.01)),
                )
            )
        return sampler

    def _setup_model(self):
        self.fixture_refs = {}
        self.region_refs = {}
        for attr in ("_robot_free_joint_z", "_robot_default_free_joint_z"):
            if hasattr(self, attr):
                delattr(self, attr)
        self.scene_name = self._ep_meta.get(
            "scene_name", self.scene_names[self.rng.choice(len(self.scene_names))]
        )
        self.mujoco_arena = MolmoSpacesArena(self._get_scene_xml(self.scene_name))
        self.mujoco_arena.set_origin([0, 0, 0])
        self.furniture = {}
        self.fixtures = {}
        self.fixture_cfgs = []
        self._setup_baked_scene_fixtures()

        info = analyze_scene(self.mujoco_arena, rng=self.rng)
        self.room_center = info["room_center"]
        self.semantic_scene = self._build_semantic_scene(info)
        if (
            self.renderer == "mjviewer"
            and self.render_camera is None
            and self.renderer_config is None
        ):
            half = info.get("room_half_extent", np.array([2.0, 2.0]))
            self.renderer_config = {
                "cam_config": {
                    "lookat": [
                        float(self.room_center[0]),
                        float(self.room_center[1]),
                        1.0,
                    ],
                    "distance": max(3.0, 1.6 * float(np.max(half))),
                    "azimuth": 180,
                    "elevation": -25,
                }
            }

        surface = self._get_scene_surface_calibration()
        self.surface = (
            (surface["center"], surface["half_xy"])
            if surface is not None
            else info.get("surface")
        )
        if self.surface is None:
            raise RuntimeError(f"No surface found for {self.scene_name}")

        if "robot_base_pos" in self._ep_meta:
            self.robot_base_pos = np.array(self._ep_meta["robot_base_pos"])
            self.robot_base_ori = np.array(
                self._ep_meta.get("robot_base_ori", [0, 0, 0])
            )
        elif self._robot_base_pos_override is not None:
            self.robot_base_pos = self._robot_base_pos_override
            self.robot_base_ori = np.zeros(3)
        else:
            calibrated = self._get_scene_robot_pose_calibration()
            if calibrated is None:
                self.robot_base_pos = info["base_pos"]
                self.robot_base_ori = np.array([0.0, 0.0, info["base_yaw"]])
            else:
                self.robot_base_pos, yaw = calibrated
                self.robot_base_ori = np.array([0.0, 0.0, yaw])
        self._setup_task_fixtures()

    def _create_objects(self):
        self.objects = {}
        self.object_cfgs = deepcopy(
            self._ep_meta.get("object_cfgs", self._get_obj_cfgs())
        )
        for cfg in self.object_cfgs:
            placement = cfg.get("placement", {})
            fixture_id = placement.pop("semantic_fixture_id", None)
            region_id = placement.pop("semantic_region_id", None)
            if fixture_id is not None:
                placement["fixture"] = self.get_fixture(id=fixture_id)
            if region_id is not None:
                placement["region"] = next(
                    (
                        r
                        for r in self.get_region(return_all=True)
                        if r.stable_id == region_id
                    ),
                    None,
                )
        for index, cfg in enumerate(self.object_cfgs):
            cfg["type"] = "object"
            cfg.setdefault("name", f"obj_{index + 1}")
            if "objaverse_uid" in cfg:
                from robocasa.utils.molmospaces_objects import make_objaverse_object

                model = make_objaverse_object(
                    cfg["objaverse_uid"],
                    cfg["name"],
                    scale=cfg.get("scale", 1.0),
                    fit_size=cfg.get("fit_size", 0.15),
                )
            else:
                model, info = EnvUtils.create_obj(self, cfg)
                cfg["info"] = info
            self.objects[model.name] = model
            self.model.merge_objects([model])

    def _build_semantic_scene(self, scene_info):
        registry = build_semantic_registry(
            self.mujoco_arena,
            self.scene_name,
            os.path.join(
                get_ithor_scenes_dir(), f"{self.scene_name}_physics_metadata.json"
            ),
            surfaces=scene_info.get("surfaces", []),
            scene_center=scene_info.get("room_center"),
            override_path=os.path.join(
                os.path.dirname(get_ithor_scenes_dir()), "thor_semantic_overrides.yaml"
            ),
        )
        for spec in get_scene_manual_supports(self.scene_name):
            support_id = f"{self.scene_name}:manual_support:{spec['id']}"
            body_name = spec["body_name"]
            region = SemanticRegion(
                stable_id=f"{support_id}:region:support",
                fixture_id=support_id,
                kind="support",
                parent_body=None,
                local_center=np.zeros(3),
                half_size=np.array([spec["half_xy"][0], spec["half_xy"][1], 0.02]),
                _world_center=spec["center"],
                _world_yaw=spec["yaw"],
            )
            registry.add_fixture(
                SemanticFixture(
                    stable_id=support_id,
                    name=f"manual_support_{spec['id']}",
                    category="manual_support",
                    source_category="ManualSupport",
                    room_id=None,
                    body_name=body_name,
                    body_names=[body_name] if body_name else [],
                    site_names=[],
                    asset_id=None,
                    object_id=spec["id"],
                    center=spec["center"],
                    half_size=np.array([spec["half_xy"][0], spec["half_xy"][1], 0.02]),
                    yaw=spec["yaw"],
                    front=np.array([np.cos(spec["yaw"]), np.sin(spec["yaw"])]),
                    capabilities={"support", "manual"},
                    regions={region.stable_id: region},
                )
            )
        return registry

    def _reset_internal(self):
        if self.sim is None:
            return
        super()._reset_internal()
        from termcolor import colored

        print(
            colored(
                f"[ithor] {type(self).__name__}  {self.scene_name}",
                "cyan",
                attrs=["bold"],
            ),
            flush=True,
        )
        self._restore_spawn_cleared_geoms()
        self._setup_scene()
        self._before_object_placement()
        if not self.deterministic_reset and self.placement_initializer is not None:
            semantic = any(
                isinstance(cfg.get("placement", {}).get("region"), SemanticRegion)
                for cfg in self.object_cfgs
            )
            semantic_attempts = int(getattr(self, "SEMANTIC_PLACEMENT_ATTEMPTS", 12))
            for attempt in range(semantic_attempts if semantic else 1):
                if attempt:
                    self.placement_initializer = self._build_placement_initializer()
                    self.object_placements = self.placement_initializer.sample()
                self._apply_object_placements(self.object_placements)
                self.sim.forward()
                if not semantic or self._semantic_placements_are_valid():
                    break
            else:
                raise RuntimeError(
                    f"{self.scene_name}: no collision-free semantic object placement after {semantic_attempts} attempts"
                )
        if not self.deterministic_reset and self.INITIAL_POSE_JITTER:
            apply_initial_pose_jitter(
                self,
                robot_radius=self.INITIAL_ROBOT_JITTER_RADIUS,
                object_radius=self.INITIAL_OBJECT_JITTER_RADIUS,
            )
        self._clear_robot_spawn_collisions()
        # This family places the robot at model-build time (set_base_xpos), so it
        # never went through set_robot_base's spawn-pose + ground-snap pair and the
        # robot used to drop and sag into the floor on the first steps. Runs last:
        # the snap measures against the scene, so objects have to be placed and
        # spawn collisions cleared first.
        EnvUtils.settle_robot_spawn(self)

    def _apply_object_placements(self, placements):
        for obj_pos, obj_quat, obj in placements.values():
            self.sim.data.set_joint_qpos(
                obj.joints[0], np.concatenate([obj_pos, obj_quat])
            )

    def _body_tree_ids(self, root_ids):
        ids = {int(i) for i in root_ids if i is not None and int(i) >= 0}
        changed = True
        while changed:
            changed = False
            for body_id in range(1, self.sim.model.nbody):
                if (
                    int(self.sim.model.body_parentid[body_id]) in ids
                    and body_id not in ids
                ):
                    ids.add(body_id)
                    changed = True
        return ids

    def _fixture_body_ids(self, fixture, include_siblings=False):
        body_names = [
            name
            for name in getattr(fixture, "body_names", ())
            if name in self.sim.model.body_names
        ]
        if include_siblings:
            prefixes = set()
            for name in body_names:
                parts = name.rsplit("_", 3)
                prefixes.add(
                    parts[0]
                    if len(parts) == 4 and all(p.isdigit() for p in parts[1:])
                    else name
                )
            body_names.extend(
                name
                for name in self.sim.model.body_names
                if name
                and any(
                    name == prefix or name.startswith(prefix + "_")
                    for prefix in prefixes
                )
            )
        return self._body_tree_ids(
            [self.sim.model.body_name2id(name) for name in body_names]
        )

    def _object_has_illegal_penetration(self, name, region, tolerance=0.002):
        fixture = self.get_fixture(id=region.fixture_id)
        return bool(
            self.illegal_contact_pairs_for_bodies(
                [self.obj_body_id[name]],
                self._fixture_body_ids(fixture, include_siblings=True),
                tolerance,
            )
        )

    def _semantic_placements_are_valid(self):
        for cfg in self.object_cfgs:
            region = cfg.get("placement", {}).get("region")
            if not isinstance(region, SemanticRegion):
                continue
            contained = (
                self.object_on_region(cfg["name"], region, require_contact=False)
                if region.kind == "support"
                else self.object_in_region(cfg["name"], region)
            )
            if not contained or self._object_has_illegal_penetration(
                cfg["name"], region
            ):
                return False
        return True

    def _setup_references(self):
        super()._setup_references()
        if self.semantic_scene is not None:
            self.semantic_scene.bind(self.sim)
        self.obj_body_id = {
            name: self.sim.model.body_name2id(model.root_body)
            for name, model in self.objects.items()
        }

    def _setup_task_fixtures(self):
        pass

    def _initial_jitter_extra_joints(self):
        return ()

    def _initial_jitter_allowed_contact_bodies(self, name):
        del name
        return set()

    def _constrain_initial_jitter_candidate(self):
        pass

    def _initial_jitter_candidate_is_valid(self):
        return True

    def _setup_baked_scene_fixtures(self):
        pass

    def _is_scene_valid(self):
        return True

    def _get_obj_cfgs(self):
        return []

    def _setup_scene(self):
        pass

    def _before_object_placement(self):
        pass

    def get_fixture(
        self,
        id=None,
        category=None,
        capabilities=(),
        room_id=None,
        ref=None,
        relation=None,
        return_all=False,
        **kwargs,
    ):
        del kwargs
        from robocasa.models.fixtures import Fixture

        if isinstance(id, (Fixture, SemanticFixture)):
            return id
        if self.semantic_scene is not None and (
            category is not None
            or capabilities
            or room_id is not None
            or relation is not None
            or isinstance(id, str)
        ):
            result = self.semantic_scene.get_fixture(
                id=id,
                category=category,
                capabilities=capabilities,
                room_id=room_id,
                ref=ref,
                relation=relation,
                return_all=return_all,
            )
            if result is not None and (not return_all or result):
                return result
        if isinstance(id, str) and id in self.fixtures:
            return self.fixtures[id]
        matches = [name for name in self.fixtures if isinstance(id, str) and id in name]
        return (
            [self.fixtures[name] for name in matches]
            if return_all
            else (self.fixtures[matches[0]] if matches else None)
        )

    def register_fixture_ref(self, ref_name, query):
        serialized = self._ep_meta.get("semantic_refs", {}).get("fixtures", {})
        fixture = (
            self.get_fixture(id=serialized[ref_name])
            if ref_name in serialized
            else self.fixture_refs.get(ref_name)
        )
        fixture = fixture or self.get_fixture(**dict(query))
        if fixture is None:
            raise LookupError(f"No semantic fixture for {ref_name!r}: {query}")
        self.fixture_refs[ref_name] = fixture
        return fixture

    def get_region(
        self,
        fixture=None,
        kind=None,
        min_size=None,
        ref=None,
        relation=None,
        return_all=False,
    ):
        if isinstance(fixture, SemanticRegion):
            return fixture
        if self.semantic_scene is None:
            return [] if return_all else None
        return self.semantic_scene.get_region(
            fixture=fixture,
            kind=kind,
            min_size=min_size,
            ref=ref,
            relation=relation,
            return_all=return_all,
        )

    def register_region_ref(self, ref_name, query):
        serialized = self._ep_meta.get("semantic_refs", {}).get("regions", {})
        if ref_name in serialized and self.semantic_scene is not None:
            region = next(
                (
                    r
                    for r in self.semantic_scene.get_region(return_all=True)
                    if r.stable_id == serialized[ref_name]
                ),
                None,
            )
        else:
            region = self.region_refs.get(ref_name) or self.get_region(**dict(query))
        if region is None:
            raise LookupError(f"No semantic region for {ref_name!r}: {query}")
        self.region_refs[ref_name] = region
        return region

    def require_fixture(self, id, description=None):
        fixture = self.get_fixture(id)
        if fixture is None:
            raise RuntimeError(f"Could not find fixture {description or id}")
        return fixture

    def set_robot_base_pose(self, pos, yaw=0.0):
        self.robot_base_pos = np.asarray(pos, dtype=float)
        self.robot_base_ori = np.array([0.0, 0.0, float(yaw)])

    def set_robot_base_from_fixture(self, fixture, distance=0.8):
        fixture = self.get_fixture(id=fixture)
        front = np.asarray(fixture.front, dtype=float)
        pos = fixture.center.copy()
        pos[:2] += front * (fixture.radial_extent(front) + float(distance))
        pos[2] = 0.0
        yaw = float(np.arctan2(fixture.center[1] - pos[1], fixture.center[0] - pos[0]))
        self.set_robot_base_pose(pos, yaw)
        return pos, yaw

    def robot_fixture_collision_distance(self, fixture, max_distance=2.0):
        fixture = self.get_fixture(id=fixture)
        fixture_bodies = self._body_tree_ids(
            [
                self.sim.model.body_name2id(b)
                for b in fixture.body_names
                if b in self.sim.model.body_names
            ]
        )
        robot_bodies = self._body_tree_ids(
            [self.sim.model.body_name2id(self.robots[0].robot_model.root_body)]
        )
        fixture_geoms = self._collidable_geoms(fixture_bodies)
        robot_geoms = self._collidable_geoms(robot_bodies)
        if not fixture_geoms or not robot_geoms:
            return np.inf
        fromto = np.empty(6, dtype=np.float64)
        distances = [
            float(
                mujoco.mj_geomDistance(
                    self.sim.model._model,
                    self.sim.data._data,
                    rg,
                    fg,
                    float(max_distance),
                    fromto,
                )
            )
            for rg in robot_geoms
            for fg in fixture_geoms
        ]
        distances = [d for d in distances if d >= 0]
        return min(distances) if distances else np.inf

    def _collidable_geoms(self, body_ids):
        return [
            geom_id
            for geom_id in range(self.sim.model.ngeom)
            if int(self.sim.model.geom_bodyid[geom_id]) in body_ids
            and (
                int(self.sim.model.geom_contype[geom_id]) != 0
                or int(self.sim.model.geom_conaffinity[geom_id]) != 0
            )
        ]

    def _set_robot_free_pose(self, xy, yaw):
        joint = self._get_robot_free_joint()
        if joint is None:
            return False
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[:2] = np.asarray(xy, dtype=float)
        qpos[2] = getattr(self, "_robot_free_joint_z", qpos[2])
        qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.forward()
        return True

    def place_robot_at_fixture(
        self,
        fixture,
        distances=(0.7, 0.85, 1.0, 1.2, 1.5),
        lateral_offsets=(0.0, 0.25, -0.25),
        max_yaw_offset_deg=30,
        approach_directions=None,
        require_floor_footprint=False,
        footprint_radius=0.28,
        min_fixture_distance=None,
    ):
        fixture = self.get_fixture(id=fixture)
        target = fixture.world_center(self.sim)[:2]
        directions = approach_directions or [fixture.front]
        angles = [float(np.arctan2(d[1], d[0])) for d in directions]
        if approach_directions is None:
            base = angles[0]
            angles += [base + np.deg2rad(o) for o in (45, -45, 90, -90, 135, -135, 180)]
        yaw_offsets = [0.0] + [
            np.deg2rad(d) * s
            for d in range(10, int(max_yaw_offset_deg) + 1, 10)
            for s in (1, -1)
        ]
        for angle in angles:
            direction = np.array([np.cos(angle), np.sin(angle)])
            right = np.array([-direction[1], direction[0]])
            boundary = fixture.radial_extent(direction)
            for distance in distances:
                for lateral in lateral_offsets:
                    xy = (
                        target
                        + direction * (boundary + float(distance))
                        + right * lateral
                    )
                    if (
                        require_floor_footprint
                        and not self.robot_footprint_is_on_floor(
                            xy, radius=footprint_radius
                        )
                    ):
                        continue
                    for yaw_offset in yaw_offsets:
                        yaw = (
                            float(np.arctan2(target[1] - xy[1], target[0] - xy[0]))
                            + yaw_offset
                        )
                        if (
                            self._set_robot_free_pose(xy, yaw)
                            and self._calibrate_robot_free_joint_height()
                        ):
                            if min_fixture_distance is not None:
                                actual_distance = self.robot_fixture_collision_distance(
                                    fixture,
                                    max_distance=float(min_fixture_distance) + 0.25,
                                )
                                if not np.isfinite(
                                    actual_distance
                                ) or actual_distance < float(min_fixture_distance):
                                    continue
                            self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
                            self.robot_base_ori = np.array([0.0, 0.0, yaw])
                            return True
        return False

    def xy_is_on_floor(self, xy, margin=0.0):
        """Return whether an XY point lies inside a semantic floor footprint."""
        xy = np.asarray(xy, dtype=float)[:2]
        for fixture in self.semantic_scene.fixtures.values():
            category = str(fixture.category).strip().lower()
            suffix = category.removeprefix("floor_")
            is_hashed_floor = bool(
                category.startswith("floor_")
                and suffix
                and all(character in "0123456789abcdef" for character in suffix)
            )
            if category != "floor" and not is_hashed_floor:
                continue
            center = np.asarray(fixture.center[:2], dtype=float)
            yaw = float(fixture.yaw)
            c, s = np.cos(-yaw), np.sin(-yaw)
            delta = xy - center
            local = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
            half = np.asarray(fixture.half_size[:2], dtype=float) - float(margin)
            if np.all(half > 0.0) and np.all(np.abs(local) <= half):
                return True
        return False

    def robot_footprint_is_on_floor(self, xy, radius=0.28):
        """Require the base center and eight footprint samples to remain indoors."""
        offsets = [np.zeros(2)]
        for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
            offsets.append(float(radius) * np.array([np.cos(angle), np.sin(angle)]))
        return all(self.xy_is_on_floor(np.asarray(xy) + offset) for offset in offsets)

    def _calibrate_robot_free_joint_height(self, max_raise=0.25, step=0.01):
        if hasattr(self, "_robot_free_joint_z"):
            if self.count_body_collisions() == 0:
                return True
            delattr(self, "_robot_free_joint_z")
        joint = self._get_robot_free_joint()
        if joint is None:
            return False
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        base_z = getattr(self, "_robot_default_free_joint_z", float(qpos[2]))
        self._robot_default_free_joint_z = base_z
        for raise_by in np.arange(0.01, max_raise + 0.5 * step, step):
            qpos[2] = base_z + float(raise_by)
            self.sim.data.set_joint_qpos(joint, qpos)
            self.sim.forward()
            if self.count_body_collisions() == 0:
                self._robot_free_joint_z = float(qpos[2])
                return True
        qpos[2] = base_z
        self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.forward()
        return False

    def sample_object(self, groups, **kwargs):
        kwargs.setdefault("obj_registries", self.obj_registries)
        kwargs.setdefault("split", self.obj_instance_split)
        return sample_kitchen_object(groups, rng=self.rng, **kwargs)

    def _object_half_xy(self, name):
        if name not in getattr(self, "objects", {}):
            return np.zeros(2, dtype=float)
        obj = self.objects[name]
        try:
            bbox = obj._regions["bbox"]["elem"]
            return string_to_array(bbox.get("size"))[:2]
        except Exception:
            radius = float(obj.horizontal_radius)
            return np.array([radius, radius], dtype=float)

    def _object_bbox_center_xy(self, name):
        obj = self.objects[name]
        try:
            bbox = obj._regions["bbox"]["elem"]
            return string_to_array(bbox.get("pos"))[:2]
        except Exception:
            return np.zeros(2, dtype=float)

    def manual_support_regions(self):
        return [
            r
            for r in self.get_region(kind="support", return_all=True)
            if ":manual_support:" in r.fixture_id
        ]

    def _support_matches_requirements(self, region, fixture, requirements):
        center, _ = region.world_pose()
        lo, hi = requirements.get("height_range", (-np.inf, np.inf))
        if not float(lo) <= center[2] <= float(hi):
            return False
        min_half = requirements.get("min_half_xy")
        if min_half is not None and np.any(
            np.asarray(region.half_size[:2]) < np.asarray(min_half, dtype=float)
        ):
            return False
        words = tuple(str(w).lower() for w in requirements.get("body_keywords", ()))
        text = self.semantic_fixture_text(fixture)
        return not words or any(word in text for word in words)

    def semantic_fixture_text(self, fixture):
        return " ".join(
            [
                fixture.name,
                fixture.category,
                fixture.source_category,
                fixture.body_name or "",
                *fixture.body_names,
            ]
        ).lower()

    def support_region_radial_extent(self, region, direction, sim=None):
        _, yaw = region.world_pose(sim)
        direction = np.asarray(direction, dtype=float)[:2]
        norm = np.linalg.norm(direction)
        if norm < 1e-8:
            return float(np.max(region.half_size[:2]))
        direction = direction / norm
        c, s = np.cos(-yaw), np.sin(-yaw)
        local = np.array(
            [c * direction[0] - s * direction[1], s * direction[0] + c * direction[1]]
        )
        return float(np.dot(np.abs(local), region.half_size[:2]))

    def support_region_long_edge_approach_directions(
        self,
        region,
        toward_xy=None,
        sim=None,
    ):
        """Return both normals of a support's long edges, preferred side first."""

        center, yaw = region.world_pose(sim)
        short_axis = int(np.argmin(np.asarray(region.half_size[:2], dtype=float)))
        local_normal = np.zeros(2, dtype=float)
        local_normal[short_axis] = 1.0
        c, s = np.cos(yaw), np.sin(yaw)
        normal = np.array(
            [
                c * local_normal[0] - s * local_normal[1],
                s * local_normal[0] + c * local_normal[1],
            ],
            dtype=float,
        )
        directions = [normal, -normal]
        if toward_xy is not None:
            toward = np.asarray(toward_xy, dtype=float)[:2] - center[:2]
            directions.sort(key=lambda direction: -float(np.dot(direction, toward)))
        return tuple(directions)

    def support_region_edge_pose(
        self,
        region,
        object_half_xy,
        toward_xy,
        margin=0.03,
        edge_fraction=1.0,
        sim=None,
        lateral_fraction=None,
        edge_direction=None,
        lateral_offset=None,
    ):
        center, yaw = region.world_pose(sim)
        relative = np.asarray(toward_xy, dtype=float)[:2] - center[:2]
        c, s = np.cos(yaw), np.sin(yaw)
        local = np.array(
            [c * relative[0] + s * relative[1], -s * relative[0] + c * relative[1]]
        )
        safe_half = np.maximum(
            np.asarray(region.half_size[:2], dtype=float)
            - np.asarray(object_half_xy, dtype=float)[:2]
            - float(margin),
            0.02,
        )
        edge_local = local
        if edge_direction is not None:
            edge_direction = np.asarray(edge_direction, dtype=float)[:2]
            norm = float(np.linalg.norm(edge_direction))
            if norm < 1e-8:
                raise ValueError("edge_direction must be non-zero")
            edge_direction = edge_direction / norm
            edge_local = np.array(
                [
                    c * edge_direction[0] + s * edge_direction[1],
                    -s * edge_direction[0] + c * edge_direction[1],
                ]
            )
        edge_axis = int(np.argmax(np.abs(edge_local) / safe_half))
        edge_sign = np.sign(edge_local[edge_axis])
        if edge_sign == 0:
            edge_sign = 1.0
        other_axis = 1 - edge_axis
        local[edge_axis] = edge_sign * float(edge_fraction) * safe_half[edge_axis]
        lateral_limit = safe_half[other_axis]
        if lateral_fraction is not None:
            lateral_fraction = float(lateral_fraction)
            if lateral_fraction < 0.0:
                raise ValueError("lateral_fraction must be non-negative")
            lateral_limit *= lateral_fraction
        requested_lateral = (
            local[other_axis] if lateral_offset is None else float(lateral_offset)
        )
        local[other_axis] = np.clip(
            requested_lateral,
            -lateral_limit,
            lateral_limit,
        )
        xy = center[:2] + np.array(
            [c * local[0] - s * local[1], s * local[0] + c * local[1]]
        )
        front_local = np.zeros(2)
        front_local[edge_axis] = edge_sign
        front = np.array(
            [
                c * front_local[0] - s * front_local[1],
                s * front_local[0] + c * front_local[1],
            ]
        )
        return xy, front / max(np.linalg.norm(front), 1e-6)

    _STRUCTURAL_CLUTTER_BLOCKLIST = (
        "wall",
        "floor",
        "ceiling",
        "room",
        "table",
        "desk",
        "sidetable",
        "nightstand",
        "dresser",
        "chestofdrawers",
        "bed",
        "counter",
        "island",
        "cabinet",
        "drawer",
        "shelf",
        "fridge",
        "refrigerator",
        "oven",
        "stove",
        "sink",
        "door",
        "window",
    )

    def _fixture_is_structural_clutter_blocked(self, fixture):
        structural = {
            "".join(ch for ch in token.lower() if ch.isalnum() or ch == "_")
            for token in self._STRUCTURAL_CLUTTER_BLOCKLIST
        }
        category_values = (
            getattr(fixture, "category", ""),
            getattr(fixture, "source_category", ""),
        )
        if any(
            "".join(ch for ch in str(value).lower() if ch.isalnum() or ch == "_")
            in structural
            for value in category_values
        ):
            return True
        tokens = set(re.split(r"[^a-z0-9]+", self.semantic_fixture_text(fixture)))
        return bool(tokens & structural)

    def region_xy_overlaps_circle(self, region, xy, radius, sim=None, margin=0.02):
        center, yaw = region.world_pose(sim)
        half = np.asarray(region.half_size[:2], dtype=float) + float(margin)
        c, s = np.cos(-yaw), np.sin(-yaw)
        delta = np.asarray(xy, dtype=float)[:2] - center[:2]
        local = np.array([c * delta[0] - s * delta[1], s * delta[0] + c * delta[1]])
        return bool(np.all(np.abs(local) <= half + float(radius)))

    def semantic_fixture_overlaps_region(
        self, fixture, region, z_below=0.08, z_above=0.80
    ):
        center, _ = region.world_pose()
        bottom = float(fixture.center[2] - fixture.half_size[2])
        top = float(fixture.center[2] + fixture.half_size[2])
        if top < center[2] - float(z_below) or bottom > center[2] + float(z_above):
            return False
        return self.semantic_fixture_xy_overlaps_region(fixture, region)

    def semantic_fixture_is_on_support(
        self,
        fixture,
        region,
        z_below=0.18,
        z_above=0.80,
    ):
        if self._fixture_is_structural_clutter_blocked(fixture):
            return False
        center, _ = region.world_pose()
        bottom = float(fixture.center[2] - fixture.half_size[2])
        if (
            not float(center[2]) - float(z_below)
            <= bottom
            <= float(center[2]) + float(z_above)
        ):
            return False
        return self.semantic_fixture_xy_overlaps_region(fixture, region)

    def visible_freejoint_roots_on_support(
        self,
        region,
        fixture,
        protected_body_roots=(),
        max_radius=None,
        z_below=0.18,
        z_above=0.80,
        visible_only=True,
    ):
        if getattr(self, "sim", None) is None:
            return []
        support_center, _ = region.world_pose(self.sim)
        protected = list(protected_body_roots)
        protected.append(
            self.sim.model.body_name2id(self.robots[0].robot_model.root_body)
        )
        if fixture is not None:
            protected.extend(
                self.sim.model.body_name2id(body_name)
                for body_name in getattr(fixture, "body_names", ())
                if body_name in self.sim.model.body_names
            )
        skip = self._body_tree_ids(protected)
        roots = []
        for root in {
            int(self.sim.model.body_rootid[i]) for i in range(1, self.sim.model.nbody)
        }:
            bodies = self._body_tree_ids([root])
            if bodies & skip or not self._body_has_free_joint(bodies):
                continue
            geoms = self._geom_ids_for_bodies(bodies)
            if not geoms:
                continue
            if visible_only and not any(
                float(self.sim.model.geom_rgba[g, 3]) > 0.01 for g in geoms
            ):
                continue
            radius = self._geom_radius(geoms)
            xy = np.mean([self.sim.data.geom_xpos[g][:2] for g in geoms], axis=0)
            low_z = min(
                float(self.sim.data.geom_xpos[g][2])
                - float(self.sim.model.geom_rbound[g])
                for g in geoms
            )
            if (
                (max_radius is None or radius <= float(max_radius))
                and self.region_xy_overlaps_circle(region, xy, radius, self.sim)
                and support_center[2] - z_below <= low_z <= support_center[2] + z_above
            ):
                roots.append(root)
        return roots

    def clear_support_region_clutter(
        self,
        region,
        fixture=None,
        protected_fixtures=(),
        protected_body_roots=(),
        max_radius=None,
        z_below=0.18,
        z_above=0.80,
        clear_semantic=True,
        clear_freejoints=True,
    ):
        fixture = fixture or self.get_fixture(id=region.fixture_id)
        removed = set()
        if clear_semantic:
            protected = tuple(
                item
                for item in (fixture, *tuple(protected_fixtures))
                if item is not None
            )
            removed = self.clear_semantic_region_clutter(
                region,
                protected_fixtures=protected,
                z_below=z_below,
                z_above=z_above,
                tabletop_only=True,
            )
        if clear_freejoints and getattr(self, "sim", None) is not None:
            self.clear_support_freejoint_clutter(
                region,
                fixture,
                protected_body_roots=protected_body_roots,
                max_radius=max_radius,
                z_below=z_below,
                z_above=z_above,
                restore_previous=False,
            )
        return removed

    def semantic_fixture_xy_overlaps_region(self, fixture, region, margin=0.02):
        fixture_half = np.asarray(fixture.half_size[:2], dtype=float)
        radius = float(np.linalg.norm(fixture_half))
        return self.region_xy_overlaps_circle(
            region, fixture.center[:2], radius, margin=margin
        )

    def semantic_fixture_is_tabletop_clutter(
        self,
        fixture,
        region,
        z_tol=0.12,
    ):
        return self.semantic_fixture_is_on_support(
            fixture,
            region,
            z_below=z_tol,
            z_above=z_tol,
        )

    def clear_semantic_region_clutter(
        self,
        region,
        protected_fixtures=(),
        z_below=0.12,
        z_above=0.80,
        tabletop_only=True,
    ):
        protected = {
            id(fixture) for fixture in protected_fixtures if fixture is not None
        }
        host = self.get_fixture(id=region.fixture_id)
        if host is not None:
            protected.add(id(host))
        clutter = [
            fixture
            for fixture in self.semantic_scene.fixtures.values()
            if id(fixture) not in protected
            and fixture.body_name is not None
            and (
                self.semantic_fixture_is_tabletop_clutter(
                    fixture,
                    region,
                    z_tol=z_below,
                )
                if tabletop_only
                else self.semantic_fixture_overlaps_region(
                    fixture, region, z_below=z_below, z_above=z_above
                )
            )
        ]
        return self.remove_semantic_fixtures(clutter)

    def remove_semantic_fixtures_near(self, tokens, xy, radius):
        xy = np.asarray(xy, dtype=float)[:2]
        tokens = tuple(str(token).lower() for token in tokens)
        fixtures = [
            fixture
            for fixture in self.semantic_scene.fixtures.values()
            if fixture.body_name is not None
            and any(token in self.semantic_fixture_text(fixture) for token in tokens)
            and float(np.linalg.norm(fixture.center[:2] - xy)) <= float(radius)
        ]
        return self.remove_semantic_fixtures(fixtures)

    def select_manual_support(self, requirements=None, target_xy=None, fit_points=()):
        requirements = requirements or {}
        target_xy = np.asarray(
            target_xy if target_xy is not None else self.robot_base_pos[:2], dtype=float
        )[:2]
        forward = np.array(
            [
                np.cos(float(self.robot_base_ori[2])),
                np.sin(float(self.robot_base_ori[2])),
            ]
        )
        robot_xy = np.asarray(self.robot_base_pos[:2], dtype=float)
        candidates = []
        self._last_support_rejections = []
        for region in self.manual_support_regions():
            fixture = self.get_fixture(id=region.fixture_id)
            if fixture is None or not self._support_matches_requirements(
                region, fixture, requirements
            ):
                self._last_support_rejections.append((region.stable_id, "requirements"))
                continue
            bad = next(
                (
                    label
                    for label, point, margin in fit_points
                    if not region.contains_xy(point, margin=margin)
                ),
                None,
            )
            if bad is not None:
                self._last_support_rejections.append(
                    (region.stable_id, f"{bad} outside support")
                )
                continue
            center, _ = region.world_pose()
            to_center = center[:2] - robot_xy
            candidates.append(
                (
                    0 if np.dot(to_center, forward) > 0 else 1,
                    -float(np.dot(to_center, forward)),
                    float(np.linalg.norm(center[:2] - target_xy)),
                    region.stable_id,
                    region,
                    fixture,
                )
            )
        return (None, None) if not candidates else min(candidates)[4:]

    def clear_support_freejoint_clutter(
        self,
        region,
        fixture,
        protected_body_roots=(),
        max_radius=None,
        z_below=0.12,
        z_above=0.80,
        restore_previous=True,
    ):
        if restore_previous:
            for geom_id, state in getattr(
                self, "_support_clutter_geom_state", {}
            ).items():
                (
                    self.sim.model.geom_contype[geom_id],
                    self.sim.model.geom_conaffinity[geom_id],
                ) = state[:2]
                self.sim.model.geom_rgba[geom_id, 3] = state[2]
            self._support_clutter_geom_state = {}
        for root in self.visible_freejoint_roots_on_support(
            region,
            fixture,
            protected_body_roots=protected_body_roots,
            max_radius=max_radius,
            z_below=z_below,
            z_above=z_above,
            visible_only=False,
        ):
            self._hide_geoms(self._geom_ids_for_bodies(self._body_tree_ids([root])))
        self.sim.forward()

    def _body_has_free_joint(self, body_ids):
        for body_id in body_ids:
            start = int(self.sim.model.body_jntadr[body_id])
            for joint_id in range(
                start, start + int(self.sim.model.body_jntnum[body_id])
            ):
                if int(self.sim.model.jnt_type[joint_id]) == int(
                    mujoco.mjtJoint.mjJNT_FREE
                ):
                    return True
        return False

    def _geom_ids_for_bodies(self, body_ids):
        return [
            g
            for g in range(self.sim.model.ngeom)
            if int(self.sim.model.geom_bodyid[g]) in body_ids
        ]

    def _geom_radius(self, geom_ids):
        return max(float(self.sim.model.geom_rbound[g]) for g in geom_ids)

    def _hide_geoms(self, geom_ids):
        if not hasattr(self, "_support_clutter_geom_state"):
            self._support_clutter_geom_state = {}
        for geom_id in geom_ids:
            if geom_id not in self._support_clutter_geom_state:
                self._support_clutter_geom_state[geom_id] = (
                    int(self.sim.model.geom_contype[geom_id]),
                    int(self.sim.model.geom_conaffinity[geom_id]),
                    float(self.sim.model.geom_rgba[geom_id, 3]),
                )
            self.sim.model.geom_contype[geom_id] = 0
            self.sim.model.geom_conaffinity[geom_id] = 0
            self.sim.model.geom_rgba[geom_id, 3] = 0.0

    def illegal_contact_pairs_for_bodies(
        self, body_roots, allowed_body_roots=(), tolerance=0.002
    ):
        def expand_roots(roots):
            roots = {int(root) for root in roots}
            expanded = self._body_tree_ids(roots - {0})
            if 0 in roots:
                expanded.add(0)
            return expanded

        query = expand_roots(body_roots)
        allowed = expand_roots(allowed_body_roots)
        pairs = set()
        for index in range(self.sim.data.ncon):
            contact = self.sim.data.contact[index]
            if float(contact.dist) >= -float(tolerance):
                continue
            geoms = (int(contact.geom1), int(contact.geom2))
            bodies = tuple(int(self.sim.model.geom_bodyid[g]) for g in geoms)
            if bodies[0] in query and bodies[1] not in query | allowed:
                pairs.add(tuple(sorted(geoms)))
            elif bodies[1] in query and bodies[0] not in query | allowed:
                pairs.add(tuple(sorted(geoms)))
        return pairs

    def register_task_body_contact_contract(
        self,
        name,
        subject_body_roots,
        support_body_roots,
    ):
        """Register reset/stability contact required by this task.

        Roots may be compiled body names or integer body IDs. Body ID zero is
        intentionally valid for MuJoCo's physical world/floor body.
        """

        if not isinstance(name, str) or not name:
            raise ValueError("body contact contract name must be nonempty")

        def normalized(value, label):
            if isinstance(value, (str, int, np.integer)):
                roots = (value,)
            else:
                try:
                    roots = tuple(value)
                except TypeError as exc:
                    raise TypeError(
                        f"{label} body roots must be a root or iterable"
                    ) from exc
            if not roots or any(
                not isinstance(root, (str, int, np.integer)) for root in roots
            ):
                raise ValueError(
                    f"{label} body roots must contain names or integer IDs"
                )
            return tuple(
                int(root) if isinstance(root, (int, np.integer)) else root
                for root in roots
            )

        contracts = getattr(self, "_task_body_contact_contracts", None)
        if not isinstance(contracts, dict):
            contracts = {}
            self._task_body_contact_contracts = contracts
        contracts[name] = {
            "subject_body_roots": normalized(subject_body_roots, "subject"),
            "support_body_roots": normalized(support_body_roots, "support"),
        }

    def find_scene_body(self, body_name):
        return self.mujoco_arena.worldbody.find(f".//body[@name='{body_name}']")

    def add_scene_body_geom(self, body_name, **geom_attrs):
        body = self.find_scene_body(body_name)
        if body is None:
            raise RuntimeError(f"Missing scene body: {body_name}")
        name = geom_attrs.get("name")
        if name and body.find(f"./geom[@name='{name}']") is not None:
            return None
        return ET.SubElement(body, "geom", **geom_attrs)

    def remove_scene_bodies(self, body_names):
        body_names = {name for name in body_names if name}
        removed = set()
        pending = [self.mujoco_arena.worldbody]
        while pending:
            parent = pending.pop()
            for body in list(parent.findall("./body")):
                if body.get("name") in body_names:
                    removed.update(
                        child.get("name")
                        for child in body.iter("body")
                        if child.get("name")
                    )
                    parent.remove(body)
                else:
                    pending.append(body)
        contact = self.mujoco_arena.root.find("contact")
        if contact is not None:
            for pair in list(contact):
                if pair.get("body1") in removed or pair.get("body2") in removed:
                    contact.remove(pair)
        return removed

    def remove_semantic_fixtures(self, fixtures):
        return self.remove_scene_bodies(
            {f.body_name for f in fixtures if f is not None and f.body_name is not None}
        )

    def remove_semantic_fixture(self, fixture):
        body = self.mujoco_arena.worldbody.find(f"./body[@name='{fixture.body_name}']")
        if body is None:
            raise RuntimeError(
                f"Cannot remove non-root scene fixture: {fixture.body_name}"
            )
        blueprint = deepcopy(body)
        names = {
            child.get("name") for child in blueprint.iter("body") if child.get("name")
        }
        contact = self.mujoco_arena.root.find("contact")
        saved = []
        if contact is not None:
            for item in list(contact):
                if item.get("body1") in names or item.get("body2") in names:
                    saved.append(deepcopy(item))
                    contact.remove(item)
        self.mujoco_arena.worldbody.remove(body)
        return {"body": blueprint, "contacts": saved}

    def _prefix_xml_names(self, item, prefix, name_map):
        for elem in item.iter():
            elem_class = elem.get("class")
            if elem.tag == "geom" and elem_class == "__VISUAL_MJT__":
                elem.attrib.setdefault("contype", "0")
                elem.attrib.setdefault("conaffinity", "0")
                elem.attrib.setdefault("group", "1")
            elif elem.tag == "geom" and elem_class in {
                "__STRUCTURAL_MJT__",
                "__DYNAMIC_MJT__",
            }:
                elem.attrib.setdefault("conaffinity", "15")
                elem.attrib.setdefault("group", "4")
                elem.attrib.setdefault("friction", "0.9 0.9 0.001")
                elem.attrib.setdefault("solref", "0.025")
                elem.attrib.setdefault("solimp", "0.998 0.998")
            elem.attrib.pop("class", None)
            name = elem.get("name")
            if name:
                mapped = f"{prefix}_{re.sub(r'[^a-zA-Z0-9_]+', '_', name)}"
                name_map[name] = mapped
                elem.set("name", mapped)
        for elem in item.iter():
            for attr in ("body1", "body2", "mesh", "material", "texture"):
                value = elem.get(attr)
                if value in name_map:
                    elem.set(attr, name_map[value])

    def _append_xml_assets(self, source_root, source_dir, prefix):
        name_map = {}
        asset = self.mujoco_arena.root.find("asset")
        if asset is None:
            asset = ET.SubElement(self.mujoco_arena.root, "asset")
        for elem in source_root.findall("./asset/*"):
            copied = deepcopy(elem)
            self._prefix_xml_names(copied, prefix, name_map)
            file_attr = copied.get("file")
            if file_attr and not os.path.isabs(file_attr):
                copied.set("file", os.path.abspath(os.path.join(source_dir, file_attr)))
            asset.append(copied)
        return name_map

    def _ensure_xml_body_inertials(self, body):
        for item in body.iter("body"):
            if item.find("./joint") is None or item.find("./inertial") is not None:
                continue
            item.insert(
                0,
                ET.Element(
                    "inertial",
                    {
                        "pos": "0 0 0",
                        "mass": "0.05",
                        "diaginertia": "0.0001 0.0001 0.0001",
                    },
                ),
            )

    def _unique_scene_fixture_prefix(self, root_body, name):
        base_name = name or root_body.get("name") or "scene_fixture"
        base_prefix = re.sub(r"[^a-zA-Z0-9_]+", "_", base_name)
        root_body_name = re.sub(r"[^a-zA-Z0-9_]+", "_", root_body.get("name") or "root")
        prefix = base_prefix
        suffix = 1
        while self.find_scene_body(f"{prefix}_{root_body_name}") is not None:
            suffix += 1
            prefix = f"{base_prefix}_{suffix}"
        return prefix

    def _append_xml_fixture_model(
        self,
        source_root,
        source_dir,
        prefix,
        pos,
        quat,
        ensure_inertials=True,
    ):
        asset_map = self._append_xml_assets(source_root, source_dir, prefix)
        body = deepcopy(source_root.find("./worldbody/body"))
        name_map = dict(asset_map)
        self._prefix_xml_names(body, prefix, name_map)
        if ensure_inertials:
            self._ensure_xml_body_inertials(body)
        body.set("pos", " ".join(str(float(v)) for v in pos))
        if quat is not None:
            body.set("quat", " ".join(str(float(v)) for v in quat))
        self.mujoco_arena.worldbody.append(body)

        contact = self.mujoco_arena.root.find("contact")
        if contact is None:
            contact = ET.SubElement(self.mujoco_arena.root, "contact")
        for item in source_root.findall("./contact/*"):
            copied = deepcopy(item)
            self._prefix_xml_names(copied, prefix, name_map)
            contact.append(copied)
        return body

    def _semantic_joints_from_body(self, body, stable_id):
        joints = {}
        for joint in body.iter("joint"):
            joint_name = joint.get("name")
            if not joint_name or joint.get("type") == "free" or "jntfree" in joint_name:
                continue
            lo, hi = (0.0, 0.0)
            if joint.get("range"):
                lo, hi = (float(v) for v in string_to_array(joint.get("range"))[:2])
            role = "lid" if "lid" in joint_name.lower() else "articulation"
            joints[role] = SemanticJoint(
                stable_id=f"{stable_id}:joint:{role}",
                name=joint_name,
                role=role,
                range=(lo, hi),
                closed_qpos=0.0,
                open_qpos=hi if abs(hi) >= abs(lo) else lo,
            )
        return joints

    def register_semantic_fixture_model(
        self,
        fixture,
        name=None,
        category=None,
        source_category=None,
        asset_id=None,
        region_kinds=None,
    ):
        stable_name = name or fixture.name
        semantic = semantic_fixture_from_robocasa_fixture(
            fixture=fixture,
            stable_id=f"{self.scene_name}:spawned:{stable_name}",
            category=category or stable_name,
            source_category=source_category,
            asset_id=asset_id,
            region_kinds=region_kinds,
        )
        if self.semantic_scene is not None:
            self.semantic_scene.add_fixture(semantic)
        self.fixture_refs[stable_name] = semantic
        for region in semantic.regions.values():
            self.region_refs[f"{stable_name}_{region.kind}"] = region
        return semantic

    def create_scene_fixture(
        self,
        blueprint=None,
        pos=None,
        quat=None,
        xml_path=None,
        name=None,
        category=None,
        source_category=None,
        asset_id=None,
        front=None,
        half_size=None,
        ensure_inertials=True,
    ):
        if xml_path is not None:
            return self._create_scene_fixture_from_xml(
                xml_path=xml_path,
                pos=pos,
                quat=quat,
                name=name,
                category=category,
                source_category=source_category,
                asset_id=asset_id,
                front=front,
                half_size=half_size,
                ensure_inertials=ensure_inertials,
            )

        body = deepcopy(blueprint["body"])
        body.set("pos", " ".join(str(float(v)) for v in pos))
        if quat is not None:
            body.set("quat", " ".join(str(float(v)) for v in quat))
        self.mujoco_arena.worldbody.append(body)
        contact = self.mujoco_arena.root.find("contact")
        if contact is not None:
            contact.extend(deepcopy(blueprint.get("contacts", [])))
        return body

    def _semantic_fixture_from_xml_body(
        self,
        xml_path,
        body,
        pos,
        quat=None,
        name=None,
        category=None,
        source_category=None,
        asset_id=None,
        front=None,
        half_size=None,
    ):
        local_center, xml_half_size, _ = mjcf_xml_bounds(xml_path)
        half_size = np.asarray(half_size if half_size is not None else xml_half_size)
        quat_arr = np.asarray(quat if quat is not None else [1.0, 0.0, 0.0, 0.0])
        rotated_center, rotated_half_size, _ = rotated_bounds(
            local_center, half_size, quat_arr
        )
        center = np.asarray(pos, dtype=float) + rotated_center
        category = category or "object"
        source_category = source_category or category
        fixture_name = name or body.get("name")
        stable_id = f"{self.scene_name}:spawned:{fixture_name}"

        fixture = SemanticFixture(
            stable_id=stable_id,
            name=fixture_name,
            category=str(category).lower(),
            source_category=source_category,
            room_id=None,
            body_name=body.get("name"),
            body_names=[b.get("name") for b in body.iter("body") if b.get("name")],
            site_names=[s.get("name") for s in body.iter("site") if s.get("name")],
            asset_id=asset_id,
            object_id=fixture_name,
            center=center,
            half_size=rotated_half_size,
            yaw=0.0,
            front=np.asarray(front if front is not None else [1.0, 0.0], dtype=float),
            capabilities={"articulated", "openable"},
            joints=self._semantic_joints_from_body(body, stable_id),
            source={"xml_path": os.path.abspath(xml_path)},
            local_center=local_center,
            local_half_size=half_size,
        )
        if self.semantic_scene is not None:
            self.semantic_scene.add_fixture(fixture)
        return fixture

    def _create_scene_fixture_from_xml(
        self,
        xml_path,
        pos,
        quat=None,
        name=None,
        category=None,
        source_category=None,
        asset_id=None,
        front=None,
        half_size=None,
        ensure_inertials=True,
    ):
        source_root = ET.parse(xml_path).getroot()
        root_body = source_root.find("./worldbody/body")
        if root_body is None:
            raise RuntimeError(f"{xml_path}: no root body")

        prefix = self._unique_scene_fixture_prefix(root_body, name)
        body = self._append_xml_fixture_model(
            source_root,
            os.path.dirname(os.path.abspath(xml_path)),
            prefix,
            pos,
            quat,
            ensure_inertials=ensure_inertials,
        )
        return self._semantic_fixture_from_xml_body(
            xml_path=xml_path,
            body=body,
            pos=pos,
            quat=quat,
            name=prefix,
            category=category,
            source_category=source_category,
            asset_id=asset_id,
            front=front,
            half_size=half_size,
        )

    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        return False

    def object_pos(self, name):
        return np.array(self.sim.data.body_xpos[self.obj_body_id[name]])

    def object_bbox_points(self, name):
        body_id = self.obj_body_id[name]
        pos = np.array(self.sim.data.body_xpos[body_id])
        quat = T.convert_quat(self.sim.data.body_xquat[body_id], to="xyzw")
        return np.array(self.objects[name].get_bbox_points(trans=pos, rot=quat))

    def _body_geom_points(self, body_names):
        roots = [
            self.sim.model.body_name2id(b)
            for b in body_names
            if b in self.sim.model.body_names
        ]
        bodies = self._body_tree_ids(roots)
        points = []
        signs = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, -1],
                [1, 1, 1],
            ],
            dtype=float,
        )
        for geom_id in range(self.sim.model.ngeom):
            if int(self.sim.model.geom_bodyid[geom_id]) not in bodies:
                continue
            aabb = np.asarray(self.sim.model.geom_aabb[geom_id], dtype=float)
            rot = np.asarray(self.sim.data.geom_xmat[geom_id], dtype=float).reshape(
                3, 3
            )
            center = np.asarray(self.sim.data.geom_xpos[geom_id], dtype=float)
            points.append(center)
            points.extend(center + rot @ (aabb[:3] + aabb[3:] * sign) for sign in signs)
        return np.asarray(points), bodies

    def object_bottom_z(self, name):
        return float(np.min(self.object_bbox_points(name)[:, 2]))

    def _object_region_bounds(self, name, region):
        center, yaw = region.world_pose(self.sim)
        points = self.object_bbox_points(name)
        c, s = np.cos(-yaw), np.sin(-yaw)
        delta = points[:, :2] - center[:2]
        local = np.column_stack(
            (c * delta[:, 0] - s * delta[:, 1], s * delta[:, 0] + c * delta[:, 1])
        )
        return center, local, float(points[:, 2].min()), float(points[:, 2].max())

    def object_on_region(
        self, name, region, z_tol=0.08, require_contact=True, require_bounds=True
    ):
        center, local, bottom, _ = self._object_region_bounds(name, region)
        if not (
            (not require_bounds or np.all(np.abs(local) <= region.half_size[:2] + 1e-4))
            and abs(bottom - float(center[2])) <= z_tol
        ):
            return False
        if not require_contact:
            return True
        fixture = self.get_fixture(id=region.fixture_id)
        fixture_bodies = self._fixture_body_ids(fixture, include_siblings=True)
        object_bodies = self._body_tree_ids([self.obj_body_id[name]])
        for i in range(self.sim.data.ncon):
            body1 = int(self.sim.model.geom_bodyid[self.sim.data.contact[i].geom1])
            body2 = int(self.sim.model.geom_bodyid[self.sim.data.contact[i].geom2])
            if (body1 in object_bodies and body2 in fixture_bodies) or (
                body2 in object_bodies and body1 in fixture_bodies
            ):
                return True
        return False

    def object_in_region(self, name, region, rim_tol=0.03, xy_tol=0.0):
        center, local, bottom, top = self._object_region_bounds(name, region)
        return bool(
            np.all(np.abs(local) <= region.half_size[:2] + float(xy_tol))
            and bottom >= float(center[2] - region.half_size[2]) - rim_tol
            and top <= float(center[2] + region.half_size[2]) + rim_tol
        )

    def eef_positions(self):
        site_ids = self.robots[0].eef_site_id
        site_ids = site_ids.values() if isinstance(site_ids, dict) else [site_ids]
        return [np.array(self.sim.data.site_xpos[s]) for s in site_ids]

    def _mj_name(self, obj_type, obj_id):
        try:
            return mujoco.mj_id2name(self.sim.model, obj_type, int(obj_id)) or ""
        except Exception:
            names_attr = {
                mujoco.mjtObj.mjOBJ_BODY: "body_names",
                mujoco.mjtObj.mjOBJ_GEOM: "geom_names",
            }.get(obj_type)
            names = getattr(self.sim.model, names_attr, ()) if names_attr else ()
            return (names[int(obj_id)] or "") if 0 <= int(obj_id) < len(names) else ""

    def _body_ids_matching(self, tokens):
        tokens = tuple(str(token).lower() for token in tokens)
        return {
            body_id
            for body_id in range(self.sim.model.nbody)
            if any(
                token in self._mj_name(mujoco.mjtObj.mjOBJ_BODY, body_id).lower()
                for token in tokens
            )
        }

    def _geom_names_for_bodies(self, body_ids):
        return [
            name
            for geom_id in self._geom_ids_for_bodies(body_ids)
            for name in (self._mj_name(mujoco.mjtObj.mjOBJ_GEOM, geom_id),)
            if name
        ]

    def g1_hand_closed(self, side, threshold=0.20, min_closed_joints=3):
        prefix = self.robots[0].robot_model.naming_prefix
        joint_bases = (
            "thumb_1",
            "thumb_2",
            "index_0",
            "index_1",
            "middle_0",
            "middle_1",
        )
        fractions = []
        for base in joint_bases:
            joint_name = f"{prefix}{side}_hand_{base}_joint"
            try:
                joint_id = self.sim.model.joint_name2id(joint_name)
            except Exception:
                continue
            qpos_addr = int(self.sim.model.jnt_qposadr[joint_id])
            joint_range = np.asarray(self.sim.model.jnt_range[joint_id], dtype=float)
            close_mag = float(np.max(np.abs(joint_range)))
            if close_mag <= 1e-6:
                continue
            fractions.append(
                min(abs(float(self.sim.data.qpos[qpos_addr])) / close_mag, 1.0)
            )
        if not fractions:
            return False
        closed = [fraction >= float(threshold) for fraction in fractions]
        return bool(
            sum(closed) >= int(min_closed_joints)
            and np.mean(fractions) >= float(threshold)
        )

    def g1_hand_object_contact(self, side, obj_name):
        obj = self.objects[obj_name]
        obj_body_id = self.sim.model.body_name2id(obj.root_body)
        obj_geoms = self._geom_names_for_bodies(self._body_tree_ids([obj_body_id]))
        hand_bodies = self._body_ids_matching((f"{side}_hand",))
        hand_geoms = self._geom_names_for_bodies(hand_bodies)
        return bool(
            hand_geoms and obj_geoms and self.check_contact(hand_geoms, obj_geoms)
        )

    def g1_hand_grasping_object(self, obj_name, threshold=0.20, min_closed_joints=3):
        for side in ("left", "right"):
            if self.g1_hand_closed(
                side, threshold=threshold, min_closed_joints=min_closed_joints
            ) and self.g1_hand_object_contact(side, obj_name):
                return True
        return False

    def object_released(self, name, th=0.25):
        pos = self.object_pos(name)
        return all(np.linalg.norm(eef - pos) > th for eef in self.eef_positions())

    def _robot_base_xy(self):
        return np.array(
            self.sim.data.get_body_xpos(self.robots[0].robot_model.root_body)
        )[:2]

    def _robot_base_yaw(self):
        body_id = self.sim.model.body_name2id(self.robots[0].robot_model.root_body)
        mat = np.asarray(self.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
        return float(np.arctan2(mat[1, 0], mat[0, 0]))

    def _get_robot_free_joint(self):
        try:
            body_id = self.sim.model.body_name2id(self.robots[0].robot_model.root_body)
        except Exception:
            return None
        joint_id = self.sim.model.body_jntadr[body_id]
        return (
            self.sim.model.joint_names[joint_id]
            if joint_id >= 0 and self.sim.model.jnt_type[joint_id] == 0
            else None
        )

    def _is_floor_geom(self, geom_id):
        body_id = int(self.sim.model.geom_bodyid[geom_id])
        text = f"{self.sim.model.geom_id2name(geom_id) or ''} {self.sim.model.body_id2name(body_id) or ''}".lower()
        return "floor" in text

    def count_body_collisions(self):
        count = 0
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            z1, z2 = float(self.sim.data.geom_xpos[contact.geom1][2]), float(
                self.sim.data.geom_xpos[contact.geom2][2]
            )
            if (
                self._is_floor_geom(contact.geom1)
                or self._is_floor_geom(contact.geom2)
                or (z1 < 0.02 and z2 < 0.02)
            ):
                continue
            names = (
                self.sim.model.geom_id2name(contact.geom1) or "",
                self.sim.model.geom_id2name(contact.geom2) or "",
            )
            robot = [name.startswith("robot0_") for name in names]
            count += int(any(robot) and not all(robot))
        return count

    def _restore_spawn_cleared_geoms(self):
        if getattr(self, "_spawn_cleared_model", None) is not self.sim.model:
            self._spawn_cleared_geom_state = {}
            self._spawn_cleared_model = self.sim.model
            return
        for geom_id, (contype, conaffinity, rgba) in getattr(
            self, "_spawn_cleared_geom_state", {}
        ).items():
            self.sim.model.geom_contype[geom_id] = contype
            self.sim.model.geom_conaffinity[geom_id] = conaffinity
            self.sim.model.geom_rgba[geom_id] = rgba
        self._spawn_cleared_geom_state = {}
        self.sim.forward()

    def _spawn_clearable_fixture_roots(self):
        roots = set()
        if self.semantic_scene is None:
            return roots
        for fixture in self.semantic_scene.fixtures.values():
            if "stool" not in fixture.category.lower():
                continue
            for body_name in fixture.body_names:
                if body_name not in self.sim.model.body_names:
                    continue
                root = self.sim.model.body_name2id(body_name)
                while int(self.sim.model.body_parentid[root]) != 0:
                    root = int(self.sim.model.body_parentid[root])
                roots.add(root)
        return roots

    def _clear_robot_spawn_collisions(self):
        clearable = self._spawn_clearable_fixture_roots()
        if not clearable:
            return
        robot_bodies = self._body_tree_ids(
            [self.sim.model.body_name2id(self.robots[0].robot_model.root_body)]
        )
        roots = set()
        for i in range(self.sim.data.ncon):
            contact = self.sim.data.contact[i]
            bodies = (
                int(self.sim.model.geom_bodyid[contact.geom1]),
                int(self.sim.model.geom_bodyid[contact.geom2]),
            )
            if bodies[0] in robot_bodies or bodies[1] in robot_bodies:
                other = bodies[1] if bodies[0] in robot_bodies else bodies[0]
                while int(self.sim.model.body_parentid[other]) != 0:
                    other = int(self.sim.model.body_parentid[other])
                if other in clearable:
                    roots.add(other)
        if not roots:
            return
        state = getattr(self, "_spawn_cleared_geom_state", {})
        for geom_id in self._geom_ids_for_bodies(self._body_tree_ids(roots)):
            state.setdefault(
                geom_id,
                (
                    int(self.sim.model.geom_contype[geom_id]),
                    int(self.sim.model.geom_conaffinity[geom_id]),
                    self.sim.model.geom_rgba[geom_id].copy(),
                ),
            )
            self.sim.model.geom_contype[geom_id] = 0
            self.sim.model.geom_conaffinity[geom_id] = 0
            self.sim.model.geom_rgba[geom_id, 3] = 0.0
        self._spawn_cleared_geom_state = state
        self._spawn_cleared_model = self.sim.model
        self.sim.forward()

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = f"explore the {self.ROOM_TYPE.value.replace('_', ' ')}"
        if hasattr(self, "scene_name"):
            meta["scene_name"] = self.scene_name
            meta["robot_base_pos"] = [float(v) for v in self.robot_base_pos]
            meta["robot_base_ori"] = [float(v) for v in self.robot_base_ori]
        cfgs = deepcopy(self.object_cfgs)
        for cfg in cfgs:
            placement = cfg.get("placement", {})
            fixture = placement.pop("fixture", None)
            region = placement.pop("region", None)
            if isinstance(fixture, SemanticFixture):
                placement["semantic_fixture_id"] = fixture.stable_id
            elif fixture is not None:
                placement["fixture"] = fixture
            if isinstance(region, SemanticRegion):
                placement["semantic_region_id"] = region.stable_id
            elif region is not None:
                placement["region"] = region
        meta["object_cfgs"] = cfgs
        meta["semantic_refs"] = {
            "fixtures": {
                name: fixture.stable_id
                for name, fixture in self.fixture_refs.items()
                if isinstance(fixture, SemanticFixture)
            },
            "regions": {
                name: region.stable_id
                for name, region in self.region_refs.items()
                if isinstance(region, SemanticRegion)
            },
        }
        success_hold_seconds = getattr(self, "SUCCESS_HOLD_SECONDS", None)
        if success_hold_seconds is not None:
            meta["success_hold_seconds"] = float(success_hold_seconds)
            meta["success_hold_steps"] = int(self.HOLD_STEPS)
        return add_atomic_task_metadata(self, meta)


(REGISTERED_ITHOR_KITCHEN_ENVS, IThorKitchenEnv) = make_room_env_category(
    RoomType.KITCHEN
)
(REGISTERED_ITHOR_LIVING_ROOM_ENVS, IThorLivingRoomEnv) = make_room_env_category(
    RoomType.LIVING_ROOM
)
(REGISTERED_ITHOR_BEDROOM_ENVS, IThorBedroomEnv) = make_room_env_category(
    RoomType.BEDROOM
)
(REGISTERED_ITHOR_BATHROOM_ENVS, IThorBathroomEnv) = make_room_env_category(
    RoomType.BATHROOM
)
