"""
Independent living-room environment stack.

``LivingRoomEnv`` is its own ``ManipulationEnv`` subclass (a peer of the kitchen
``Kitchen`` class) with its own ``LivingRoomEnvMeta`` metaclass that
auto-registers every living-room env into ``REGISTERED_LIVING_ROOM_ENVS`` (the
peer of ``REGISTERED_KITCHEN_ENVS``). Exactly like kitchen, the base env is
itself runnable (free-roam furnished room, no task) and every other env is a
*task* subclass -- there are no scene-only classes; a "scene" is a
(layout, style) choice. It depends on neither the kitchen ``Kitchen`` env nor
its layout/style/``KitchenArena`` machinery, but it mirrors kitchen's *flow* so
the two feel the same and share the lower layers:

    pick a task (env name = task class)
        -> reset -> _load_model:
             1. _setup_model: choose (layout_id, style_id) -> build FloorArena
                + furniture fixtures from that layout/style  (scene structure)
             2. _setup_furniture_references: task claims furniture by FurnitureType
             3. _create_objects: task's _get_obj_cfgs -> sample objects -> place
                them onto furniture regions (reuses kitchen's object + sampler)
             4. robot base placement

Shared / reused (kitchen-agnostic) building blocks:
    - fixture classes under ``models/fixtures`` (furniture = static fixtures),
    - ``models/objects`` + ``utils.env_utils`` object creation and the
      ``ObjectPositionSampler`` placement machinery,
    - textures under ``models/assets/textures`` (the floor texture).

Independent (living-room only): the layout/style registry + builder + arena
under ``models/scenes/living_room_*`` (peers of kitchen's ``scene_registry`` /
``scene_builder`` / ``KitchenArena``), plus ``FurnitureType``. Tasks subclass
``LivingRoomEnv`` and live under ``tasks/``.
"""

import os

import numpy as np

import robosuite.utils.transform_utils as T
from robosuite.environments.base import EnvMeta
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.observables import Observable, sensor

import robocasa.utils.env_utils as EnvUtils
from robocasa.models.fixtures import Fixture, FurnitureType, furniture_is_type
from robocasa.models.objects.kitchen_object_utils import sample_kitchen_object
from robocasa.models.scenes.living_room_arena import LivingRoomArena
from robocasa.models.scenes.living_room_scene_registry import (
    unpack_layout_ids,
    unpack_style_ids,
)
from robocasa.utils.errors import PlacementError
from robosuite.utils.errors import RandomizationError


# registry of all living-room envs (the base + every task), mirroring kitchen's
# REGISTERED_KITCHEN_ENVS. Exposed as ALL_LIVING_ROOM_ENVIRONMENTS.
REGISTERED_LIVING_ROOM_ENVS = {}

# abstract intermediate bases that group tasks but are not runnable themselves
# (none yet; add names here as task families grow, e.g. a future "Navigate").
_ABSTRACT_LIVING_ROOM_ENVS = []


def register_living_room_env(target_class):
    REGISTERED_LIVING_ROOM_ENVS[target_class.__name__] = target_class


class LivingRoomEnvMeta(EnvMeta):
    """Metaclass that auto-registers living-room envs (peer of KitchenEnvMeta)."""

    def __new__(meta, name, bases, class_dict):
        cls = super().__new__(meta, name, bases, class_dict)
        if cls.__name__ not in _ABSTRACT_LIVING_ROOM_ENVS:
            register_living_room_env(cls)
        return cls


class LivingRoomEnv(ManipulationEnv, metaclass=LivingRoomEnvMeta):
    """Base living-room env, mirroring ``Kitchen``.

    Like ``Kitchen``, this is both the base class for tasks *and* a runnable env
    on its own: with no task it builds a furnished room from a random
    (layout, style) -- a "free-roam" scene. Concrete tasks subclass it (under
    ``tasks/``) and add objects + a success condition. There are deliberately no
    scene-only subclasses: a "scene" is a (layout, style) choice, selected via
    ``layout_ids`` / ``style_ids`` at make() time, not a class.

    Args:
        robots: G1 robot variant (e.g. "G1FloatingBody").
        layout_ids ((list of) int): allowed layout ids (see the scene registry /
            ``assets/scenes/living_room_layouts``). None means all; one is chosen
            per reset (random unless fixed in ep_meta).
        style_ids ((list of) int): allowed style ids (``living_room_styles``).
        robot_base_pos (3-tuple): where to place the robot base on the floor.
        All other args mirror the standard robosuite environment constructor.
    """

    def __init__(
        self,
        robots,
        layout_ids=None,
        style_ids=None,
        layout_and_style_ids=None,
        obj_registries=("objaverse", "lightwheel"),
        obj_instance_split=None,
        env_configuration="default",
        controller_configs=None,
        gripper_types="default",
        base_types="default",
        initialization_noise="default",
        robot_base_pos=(0.0, 0.0, 0.0),
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
        fixed_scene_seed=None,
        fixed_objects_seed=None,
    ):
        # Scene-determinism modes (None = fully randomized per reset, the default):
        #  * fixed_scene_seed: re-seed at the start of every scene build, so EVERY
        #    reset is the IDENTICAL scene -- same layout/style, same object
        #    instances, AND same placements. Fully fixed.
        #  * fixed_objects_seed: fix the layout/style + object INSTANCES (same mug,
        #    same plate every reset), but RANDOMIZE their placements. Light
        #    variation for generalization while staying learnable from few demos.
        # Set at most one. fixed_scene_seed takes precedence if both are given.
        self._fixed_scene_seed = fixed_scene_seed
        self._fixed_objects_seed = (
            None if fixed_scene_seed is not None else fixed_objects_seed
        )

        self.robot_base_pos = np.array(robot_base_pos)

        # object-sampling config (reuses kitchen's object registries + the
        # train/test instance split machinery in kitchen_object_utils).
        self.obj_registries = obj_registries
        self.obj_instance_split = obj_instance_split

        # build the set of (layout, style) combos this env may sample from.
        # ``layout_and_style_ids`` (explicit pairs) takes precedence, mirroring
        # Kitchen; otherwise take the cross product of layout_ids x style_ids.
        if layout_and_style_ids is not None:
            assert (
                layout_ids is None and style_ids is None
            ), "set either layout_and_style_ids OR layout_ids/style_ids, not both"
            self.layout_and_style_ids = [
                (int(l), int(s)) for (l, s) in layout_and_style_ids
            ]
            self.layout_ids = sorted({l for l, s in self.layout_and_style_ids})
            self.style_ids = sorted({s for l, s in self.layout_and_style_ids})
        else:
            self.layout_ids = unpack_layout_ids(layout_ids)
            self.style_ids = unpack_style_ids(style_ids)
            self.layout_and_style_ids = [
                (l, s) for l in self.layout_ids for s in self.style_ids
            ]

        # populated during _load_model
        self.furniture = {}
        self.fixtures = {}
        self.furniture_refs = {}
        self.objects = {}
        self.object_cfgs = []
        self.placement_initializer = None
        self.object_placements = {}

        super().__init__(
            robots=robots,
            env_configuration=env_configuration,
            controller_configs=controller_configs,
            base_types=base_types,
            gripper_types=gripper_types,
            initialization_noise=initialization_noise,
            use_camera_obs=use_camera_obs,
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            render_camera=render_camera,
            render_collision_mesh=render_collision_mesh,
            render_visual_mesh=render_visual_mesh,
            render_gpu_device_id=render_gpu_device_id,
            control_freq=control_freq,
            lite_physics=lite_physics,
            horizon=horizon,
            ignore_done=ignore_done,
            hard_reset=hard_reset,
            camera_names=camera_names,
            camera_heights=camera_heights,
            camera_widths=camera_widths,
            camera_depths=camera_depths,
            camera_segmentations=camera_segmentations,
            renderer=renderer,
            renderer_config=renderer_config,
            seed=seed,
        )

    # ------------------------------------------------------------------ #
    # Scene assembly (mirrors Kitchen._load_model / _setup_model)
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Build the scene: arena + furniture + robot + manipulable objects.

        Object placement can fail stochastically, so the build (layout/style
        choice -> furniture -> objects -> sample) is retried a few times. The sim
        is not built yet here (the base env builds it after ``_load_model``
        returns), so retrying just rebuilds the model tree -- nothing to destroy.
        """
        # Permanently-fixed-scene mode: re-seed the RNG so the whole build below
        # (layout/style choice, object sampling, placement) is deterministic and
        # IDENTICAL on every reset. Done here because _load_model runs on the
        # initial build AND every hard reset.
        if self._fixed_scene_seed is not None:
            self.rng = np.random.default_rng(self._fixed_scene_seed)

        super()._load_model()

        for robot in self.robots:
            robot.robot_model.set_base_xpos(self.robot_base_pos)

        for _ in range(20):
            self.furniture_refs = {}
            # Fixed-objects mode: re-seed before layout + object-instance sampling
            # so the layout/style and the chosen mug/plate INSTANCES are identical
            # every reset (re-applied per attempt so retries keep the same objects).
            if self._fixed_objects_seed is not None:
                self.rng = np.random.default_rng(self._fixed_objects_seed)
            self._setup_model()

            self.model = ManipulationTask(
                mujoco_arena=self.mujoco_arena,
                mujoco_robots=[robot.robot_model for robot in self.robots],
                mujoco_objects=list(self.furniture.values()),
            )

            # task claims furniture by semantic type
            self._setup_furniture_references()

            # create + place manipulable objects (reuses kitchen's object layer)
            self._create_objects()
            if len(self.object_cfgs) == 0:
                self.placement_initializer = None
                self.object_placements = {}
                return
            # Fixed-objects mode: re-randomize before placement so only the object
            # POSITIONS vary between resets (instances stay fixed above).
            if self._fixed_objects_seed is not None:
                self.rng = np.random.default_rng()
            try:
                self.placement_initializer = EnvUtils._get_placement_initializer(
                    self, self.object_cfgs
                )
                self.object_placements = self.placement_initializer.sample()
                return
            except (PlacementError, RandomizationError):
                continue

        raise RuntimeError("Could not place living-room objects after 20 attempts")

    def _setup_model(self):
        """Choose layout + style and build the arena + furniture for them.

        Furniture comes from the arena (driven by the chosen layout/style), the
        same way kitchen fixtures come from ``KitchenArena`` -- not from a
        per-scene hook.
        """
        if "layout_id" in self._ep_meta and "style_id" in self._ep_meta:
            self.layout_id = self._ep_meta["layout_id"]
            self.style_id = self._ep_meta["style_id"]
        else:
            idx = self.rng.choice(len(self.layout_and_style_ids))
            self.layout_id, self.style_id = self.layout_and_style_ids[idx]

        self.mujoco_arena = LivingRoomArena(self.layout_id, self.style_id, rng=self.rng)
        self.mujoco_arena.set_origin([0, 0, 0])

        self.furniture = self.mujoco_arena.furniture
        # kitchen's object/placement helpers look up via ``env.fixtures``
        self.fixtures = self.furniture

    def _create_objects(self):
        """Create manipulable objects and merge them into the model.

        Simplified from ``Kitchen._create_objects`` (no container/auxiliary
        logic): each cfg from ``_get_obj_cfgs`` becomes one free object, reusing
        kitchen's ``create_obj`` (so living-room tasks can use kitchen objects).
        """
        self.objects = {}
        if "object_cfgs" in self._ep_meta:
            self.object_cfgs = self._ep_meta["object_cfgs"]
        else:
            self.object_cfgs = self._get_obj_cfgs()

        for obj_num, cfg in enumerate(self.object_cfgs):
            cfg["type"] = "object"
            if "name" not in cfg:
                cfg["name"] = "obj_{}".format(obj_num + 1)
            model, info = EnvUtils.create_obj(self, cfg)
            cfg["info"] = info
            self.objects[model.name] = model
            self.model.merge_objects([model])

    def _reset_internal(self):
        super()._reset_internal()

        self._setup_scene()

        if not self.deterministic_reset and self.placement_initializer is not None:
            for obj_pos, obj_quat, obj in self.object_placements.values():
                self.sim.data.set_joint_qpos(
                    obj.joints[0],
                    np.concatenate([np.array(obj_pos), np.array(obj_quat)]),
                )

    # ------------------------------------------------------------------ #
    # Object observables (ported from Kitchen) -- so the manipulable objects'
    # poses appear in the observation / dataset, not just in _check_success.
    # ------------------------------------------------------------------ #

    def _setup_references(self):
        super()._setup_references()
        self.obj_body_id = {}
        for name, model in self.objects.items():
            self.obj_body_id[name] = self.sim.model.body_name2id(model.root_body)

    def _setup_observables(self):
        """Add absolute + eef-relative pose observables for every object.

        Ported from ``Kitchen._setup_observables``. Robot-agnostic: the
        eef-relative sensors degrade to identity when the robot exposes no
        single-arm ``{prefix}eef_pos`` (e.g. the bimanual G1), while the
        absolute ``obj_pos`` / ``obj_quat`` work for any robot.
        """
        observables = super()._setup_observables()

        pf = self.robots[0].robot_model.naming_prefix
        modality = "object"

        @sensor(modality=modality)
        def world_pose_in_gripper(obs_cache):
            return (
                T.pose_inv(
                    T.pose2mat((obs_cache[f"{pf}eef_pos"], obs_cache[f"{pf}eef_quat"]))
                )
                if f"{pf}eef_pos" in obs_cache and f"{pf}eef_quat" in obs_cache
                else np.eye(4)
            )

        sensors = [world_pose_in_gripper]
        names = ["world_pose_in_gripper"]
        actives = [False]

        for obj_name in self.obj_body_id:
            obj_sensors, obj_sensor_names = self._create_obj_sensors(
                obj_name=obj_name, modality=modality
            )
            sensors += obj_sensors
            names += obj_sensor_names
            actives += [True] * len(obj_sensors)

        for name, s, active in zip(names, sensors, actives):
            observables[name] = Observable(
                name=name,
                sensor=s,
                sampling_rate=self.control_freq,
                active=active,
            )

        return observables

    def _create_obj_sensors(self, obj_name, modality="object"):
        """Per-object pos / quat / eef-relative sensors (ported from Kitchen)."""
        pf = self.robots[0].robot_model.naming_prefix

        @sensor(modality=modality)
        def obj_pos(obs_cache):
            return np.array(self.sim.data.body_xpos[self.obj_body_id[obj_name]])

        @sensor(modality=modality)
        def obj_quat(obs_cache):
            return T.convert_quat(
                self.sim.data.body_xquat[self.obj_body_id[obj_name]], to="xyzw"
            )

        @sensor(modality=modality)
        def obj_to_eef_pos(obs_cache):
            if any(
                name not in obs_cache
                for name in [
                    f"{obj_name}_pos",
                    f"{obj_name}_quat",
                    "world_pose_in_gripper",
                ]
            ):
                return np.zeros(3)
            obj_pose = T.pose2mat(
                (obs_cache[f"{obj_name}_pos"], obs_cache[f"{obj_name}_quat"])
            )
            rel_pose = T.pose_in_A_to_pose_in_B(
                obj_pose, obs_cache["world_pose_in_gripper"]
            )
            rel_pos, rel_quat = T.mat2pose(rel_pose)
            obs_cache[f"{obj_name}_to_{pf}eef_quat"] = rel_quat
            return rel_pos

        @sensor(modality=modality)
        def obj_to_eef_quat(obs_cache):
            return (
                obs_cache[f"{obj_name}_to_{pf}eef_quat"]
                if f"{obj_name}_to_{pf}eef_quat" in obs_cache
                else np.zeros(4)
            )

        sensors = [obj_pos, obj_quat, obj_to_eef_pos, obj_to_eef_quat]
        names = [
            f"{obj_name}_pos",
            f"{obj_name}_quat",
            f"{obj_name}_to_{pf}eef_pos",
            f"{obj_name}_to_{pf}eef_quat",
        ]
        return sensors, names

    # ------------------------------------------------------------------ #
    # Fixture lookup (mirrors Kitchen.get_fixture / register_fixture_ref)
    # ------------------------------------------------------------------ #

    def get_fixture(self, id, ref=None, full_name_check=False, **kwargs):
        """Look up furniture by ``Fixture`` instance, name, or ``FurnitureType``.

        Simplified from ``Kitchen.get_fixture`` (no counter/island/ref logic).
        When several pieces match a type, one is chosen at random.
        """
        if isinstance(id, Fixture):
            return id
        if isinstance(id, str) and id in self.fixtures:
            return self.fixtures[id]

        if isinstance(id, FurnitureType):
            matches = [
                name for name, f in self.fixtures.items() if furniture_is_type(f, id)
            ]
        elif isinstance(id, str):
            if full_name_check:
                matches = [name for name in self.fixtures if name == id]
            else:
                matches = [name for name in self.fixtures if id in name]
        else:
            matches = []

        if len(matches) == 0:
            return None
        return self.fixtures[self.rng.choice(matches)]

    def sample_object(self, groups, **kwargs):
        """Sample a kitchen object instance (reused as living-room clutter).

        Thin wrapper over ``sample_kitchen_object`` matching the signature
        ``EnvUtils.create_obj`` expects, so kitchen's object layer works here.
        """
        if (
            isinstance(groups, str)
            and groups.endswith(".xml")
            and os.path.exists(groups)
        ):
            scale = kwargs.get("object_scale", None) or 1.0
            return (
                dict(mjcf_path=groups, scale=scale),
                dict(
                    mjcf_path=groups,
                    cat=os.path.basename(os.path.dirname(os.path.dirname(groups))),
                    groups=[],
                    groups_containing_sampled_obj=[],
                    split=None,
                ),
            )
        kwargs.setdefault("obj_registries", self.obj_registries)
        kwargs.setdefault("split", self.obj_instance_split)
        return sample_kitchen_object(groups, rng=self.rng, **kwargs)

    def register_furniture_ref(self, ref_name, fn_kwargs):
        """Resolve and cache a furniture reference for the task to use.

        ``fn_kwargs`` are forwarded to ``get_fixture`` (e.g.
        ``dict(id=FurnitureType.COFFEE_TABLE)``). Returns the matched fixture.
        """
        if ref_name not in self.furniture_refs:
            self.furniture_refs[ref_name] = self.get_fixture(**fn_kwargs)
        return self.furniture_refs[ref_name]

    # ------------------------------------------------------------------ #
    # Extension hooks (override in concrete scenes / tasks). Mirror Kitchen.
    # ------------------------------------------------------------------ #

    def _get_obj_cfgs(self):
        """Return manipulable-object placement configs. Override per task.

        Same schema as kitchen's ``_get_obj_cfgs`` (name / obj_groups /
        placement=dict(fixture=..., sample_region_kwargs=..., size=..., ...)).
        """
        return []

    def _setup_furniture_references(self):
        """Resolve furniture refs by type for the task. Override per task.

        Analogous to kitchen's ``_setup_kitchen_references``; use
        ``self.register_furniture_ref(name, dict(id=FurnitureType.X))``.
        """
        pass

    def _setup_scene(self):
        """Per-reset scene setup hook (e.g. extra sites). Override as needed."""
        pass

    def reward(self, action=None):
        """No reward shaping by default."""
        return 0.0

    def _check_success(self):
        """Task success condition. Override per task."""
        return False

    def get_ep_meta(self):
        """Episode metadata. ``lang`` is the natural-language instruction.

        Serializes the chosen layout/style + object cfgs so an episode can be
        rebuilt deterministically (mirrors kitchen's ``get_ep_meta``).
        """
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "move around the living room"
        if hasattr(self, "layout_id"):
            ep_meta["layout_id"] = int(self.layout_id)
            ep_meta["style_id"] = int(self.style_id)
        ep_meta["object_cfgs"] = self.object_cfgs
        return ep_meta
