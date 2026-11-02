"""Walk to the calibrated desk in a Holodeck scene."""

import os

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckApproachDesk(HolodeckEnv):
    """Navigate from a calibrated far start to the scene's desk."""

    ATOMIC_SKILL = "ApproachDesk"
    TASK_FAMILY = "navigation"
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    SPEC_TASK = "approach_desk"
    HOLD_STEPS = 10
    SUCCESS_RADIUS = 1.0
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    USE_DEFAULT_ASSET_LAYOUT = False
    LAPTOP_ASSET = (
        "objects/thor/Living Room and Bedroom Objects/Laptop_Full/"
        "Prefabs/Laptop_26/Laptop_26.xml"
    )

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _setup_baked_scene_fixtures(self):
        """Apply task-local laptop calibration before semantic parsing."""
        spec = get_holodeck_task_spec(self.scene_name, self.SPEC_TASK)
        remove_scene_bodies = tuple(spec.get("remove_scene_bodies", ()))
        source_body = spec.get("source_body")
        if source_body:
            remove_scene_bodies += (source_body,)
        if remove_scene_bodies:
            removed = self.remove_scene_bodies(remove_scene_bodies)
            missing = set(remove_scene_bodies) - removed
            if missing:
                raise RuntimeError(
                    f"{self.scene_name}: configured scene bodies are missing: "
                    f"{sorted(missing)}"
                )
        laptop_body_name = spec.get("laptop_body")
        laptop_pose = spec.get("laptop_pose")
        if (
            spec.get("fixture_mode") != "spawn_asset"
            and (laptop_body_name is None) != (laptop_pose is None)
        ):
            raise ValueError(
                f"Incomplete Holodeck laptop calibration for {self.scene_name}: "
                "laptop_body and laptop_pose must be provided together"
            )
        if laptop_body_name is not None:
            laptop_body = self.find_scene_body(laptop_body_name)
            if laptop_body is None:
                raise RuntimeError(
                    f"{self.scene_name}: configured laptop body is missing: "
                    f"{laptop_body_name}"
                )
            laptop_body.set(
                "pos",
                " ".join(f"{float(value):.9f}" for value in laptop_pose["pos"]),
            )
            laptop_body.set(
                "quat",
                " ".join(f"{float(value):.9f}" for value in laptop_pose["quat"]),
            )
        super()._setup_baked_scene_fixtures()

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.task_spec = get_holodeck_task_spec(
            self.scene_name,
            self.SPEC_TASK,
            required=("desk_body", "robot_pose", "success_radius"),
        )
        if self.task_spec.get("fixture_mode") == "spawn_asset":
            support = self.task_spec.get("laptop_support_geom")
            if support is not None:
                self.add_scene_body_geom(
                    self.task_spec["desk_body"],
                    name=f"{self.scene_name}_laptop_support",
                    type="box",
                    pos=" ".join(str(float(value)) for value in support["pos"]),
                    size=" ".join(str(float(value)) for value in support["size"]),
                    contype="8",
                    conaffinity="15",
                    group="0",
                    friction="0.9 0.9 0.001",
                )
            pose = self.task_spec["laptop_pose"]
            self.laptop_fixture = self.create_scene_fixture(
                xml_path=os.path.join(robocasa.models.assets_root, self.LAPTOP_ASSET),
                pos=np.asarray(pose["pos"], dtype=float),
                quat=np.asarray(pose["quat"], dtype=float),
                name="spawned_laptop",
                category="Laptop",
                source_category="Laptop",
                asset_id="Laptop_26",
                front=np.asarray(self.task_spec.get("laptop_front", [0.0, -1.0])),
            )
        desks = self.get_fixture(category="desk", return_all=True)
        self.desk_fixture = next(
            (
                fixture
                for fixture in desks
                if fixture.body_name == self.task_spec["desk_body"]
            ),
            None,
        )
        if self.desk_fixture is not None:
            self.fixture_refs["approach_desk"] = self.desk_fixture
            pose = self.task_spec["robot_pose"]
            self.set_robot_base_pose(
                np.asarray(pose["pos"], dtype=float), float(pose["yaw"])
            )

    def _is_scene_valid(self):
        return self.desk_fixture is not None

    def _robot_base_xy(self):
        root_body = self.robots[0].robot_model.root_body
        return np.asarray(self.sim.data.get_body_xpos(root_body), dtype=float)[:2]

    def success_distance(self):
        desk_xy = np.asarray(self.desk_fixture.center[:2], dtype=float)
        return float(np.linalg.norm(self._robot_base_xy() - desk_xy))

    def _setup_scene(self):
        super()._setup_scene()
        self._reset_task_progress()

    def _reset_internal(self):
        super()._reset_internal()
        self._reset_task_progress()

    def _reset_task_progress(self):
        parent = getattr(super(), "_reset_task_progress", None)
        if callable(parent):
            parent()
        self._in_range_steps = 0

    def _post_action(self, action):
        ret = super()._post_action(action)
        if self.success_distance() < self.success_radius:
            self._in_range_steps += 1
        else:
            self._in_range_steps = 0
        return ret

    @property
    def success_radius(self):
        return float(self.task_spec.get("success_radius", self.SUCCESS_RADIUS))

    def _check_success(self):
        return self._in_range_steps >= self.HOLD_STEPS

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["lang"] = "walk to the desk and stand nearby"
        return meta


__all__ = ["HolodeckApproachDesk"]
