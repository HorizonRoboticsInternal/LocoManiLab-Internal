"""Scene 1477 window, Rubik's cube, and scene-117 laptop workflow."""

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from robocasa.environments.holodeck.holodeck import get_holodeck_scenes_dir
from robocasa.environments.holodeck.tasks.approach_window import HolodeckApproachWindow
from robocasa.environments.holodeck.tasks.cube_in_plate import HolodeckCubeInPlate
from robocasa.environments.holodeck.tasks.laptop import HolodeckOpenLaptop
from robocasa.environments.holodeck.tasks.multi_hop.base import (
    HolodeckAtomicTask,
    HolodeckMultiHopComposite,
)
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class HolodeckApproachWindowThenCubeInPlateThenOpenLaptop(
    HolodeckMultiHopComposite,
    HolodeckCubeInPlate,
):
    """Reach the rightmost window, place the cube, then open the laptop."""

    ABSTRACT = False
    ATOMIC_SKILL = "ApproachWindowThenCubeInPlateThenOpenLaptop"
    TASK_FAMILY = "composite"
    DIFFICULTY_LEVEL = "L4"
    SUCCESS_DEBUG_ENABLED = False
    SUPPORTED_SCENE_IDS = (1477,)
    # Keep the inherited cube initializer bound to its unchanged atomic spec.
    SPEC_TASK = HolodeckCubeInPlate.SPEC_TASK
    WORKFLOW_SPEC_TASK = "approach_window_then_cube_in_plate_then_open_laptop"
    LAPTOP_PREFIX = "cube_workflow"
    ATOMIC_TASKS = (
        HolodeckAtomicTask(
            "approach_rightmost_window",
            "_window_reached",
            task_cls=HolodeckApproachWindow,
            hold_steps=HolodeckApproachWindow.HOLD_STEPS,
            metadata={"skill": "navigation", "target": "rightmost_window"},
        ),
        HolodeckAtomicTask(
            "cube_in_plate",
            "_cube_placed_after_window",
            task_cls=HolodeckCubeInPlate,
            on_enter="_begin_cube_stage",
            metadata={"skill": "place", "object": "rubiks_cube", "target": "plate"},
        ),
        HolodeckAtomicTask(
            "open_laptop",
            "_laptop_opened_after_cube",
            task_cls=HolodeckOpenLaptop,
            on_enter="_begin_laptop_stage",
            metadata={"skill": "articulation", "target": "laptop_lid"},
        ),
    )

    def _setup_baked_scene_fixtures(self):
        self.workflow_spec = get_holodeck_task_spec(
            self.scene_name,
            self.WORKFLOW_SPEC_TASK,
            required=("window_body", "robot_pose", "success_radius", "laptop_pose"),
        )
        self.laptop_spec = get_holodeck_task_spec("train_117", "open_laptop")
        self._add_scene_117_laptop()
        super()._setup_baked_scene_fixtures()

    def _add_scene_117_laptop(self):
        """Copy the actual 117 body and only its referenced visual assets."""
        source_path = Path(get_holodeck_scenes_dir()) / "train_117.robocasa.xml"
        source = ET.parse(source_path).getroot()
        body = source.find(
            f"./worldbody/body[@name='{self.laptop_spec['target_body']}']"
        )
        if body is None:
            raise RuntimeError("train_117: configured source laptop is missing")
        assets = {item.get("name"): item for item in source.findall("./asset/*")}
        needed = set()
        pending = [body]
        while pending:
            for element in pending.pop().iter():
                for attribute in ("mesh", "material", "texture"):
                    name = element.get(attribute)
                    if name and name not in needed:
                        needed.add(name)
                        pending.append(assets[name])
        laptop_model = ET.Element("mujoco")
        asset = ET.SubElement(laptop_model, "asset")
        for item in source.findall("./asset/*"):
            if item.get("name") in needed:
                asset.append(deepcopy(item))
        ET.SubElement(laptop_model, "worldbody").append(deepcopy(body))
        pose = self.workflow_spec["laptop_pose"]
        self._append_xml_fixture_model(
            laptop_model,
            str(source_path.parent),
            self.LAPTOP_PREFIX,
            pose["pos"],
            pose["quat"],
            ensure_inertials=False,
        )
        self.laptop_body = f"{self.LAPTOP_PREFIX}_{self.laptop_spec['target_body']}"
        self.laptop_joint = f"{self.LAPTOP_PREFIX}_{self.laptop_spec['target_joint']}"

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self.window_body = self.workflow_spec["window_body"]
        self.window_fixture = next(
            (f for f in self.fixtures.values() if f.body_name == self.window_body),
            None,
        )
        if self.window_fixture is not None:
            self.fixture_refs["approach_window"] = self.window_fixture
        pose = self.workflow_spec["robot_pose"]
        self.set_robot_base_pose(np.asarray(pose["pos"]), float(pose["yaw"]))

    def _reset_task_progress(self):
        # Restore the imported free base and closed lid on every fixed-scene reset.
        body_id = self.sim.model.body_name2id(self.laptop_body)
        joint_id = int(self.sim.model.body_jntadr[body_id])
        joint_name = self.sim.model.joint_id2name(joint_id)
        pose = self.workflow_spec["laptop_pose"]
        self.sim.data.set_joint_qpos(joint_name, [*pose["pos"], *pose["quat"]])
        self.sim.data.set_joint_qvel(joint_name, np.zeros(6))
        self.sim.data.set_joint_qpos(self.laptop_joint, self.laptop_spec["closed_qpos"])
        self.sim.data.set_joint_qvel(self.laptop_joint, 0.0)
        self.sim.forward()
        self._cube_stage_armed = False
        self._laptop_stage_armed = False
        super()._reset_task_progress()

    def _validate_initial_composite_state(self):
        super()._validate_initial_composite_state()
        if self.count_body_collisions():
            raise RuntimeError(
                f"{self.scene_name}: composite robot starts in collision"
            )
        # Holodeck uses a plane and room polygons rather than a Floor fixture.
        base = self.sim.data.get_body_xpos(self.robots[0].robot_model.root_body)[:2]
        samples = [base] + [
            base + 0.28 * np.array([np.cos(angle), np.sin(angle)])
            for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False)
        ]
        if any(self.room_of_point([*point, 0.0]) is None for point in samples):
            raise RuntimeError(
                f"{self.scene_name}: composite robot footprint leaves floor"
            )

    def _window_reached(self):
        return bool(
            HolodeckApproachWindow.success_distance(self)
            <= float(self.workflow_spec["success_radius"])
        )

    def _begin_cube_stage(self):
        self._cube_stage_armed = not self._cube_placed()

    def _cube_placed_after_window(self):
        placed = self._cube_placed()
        if not placed:
            self._cube_stage_armed = True
        return bool(self._cube_stage_armed and placed)

    def _laptop_open_fraction(self):
        closed = float(self.laptop_spec["closed_qpos"])
        opened = float(self.laptop_spec["open_qpos"])
        qpos = float(self.sim.data.get_joint_qpos(self.laptop_joint))
        return (qpos - closed) / (opened - closed)

    def _begin_laptop_stage(self):
        self._laptop_stage_armed = self._laptop_open_fraction() <= 0.05

    def _laptop_opened_after_cube(self):
        fraction = self._laptop_open_fraction()
        if fraction <= 0.05:
            self._laptop_stage_armed = True
        return bool(
            self._laptop_stage_armed
            and fraction >= HolodeckOpenLaptop.SUCCESS_OPEN_FRACTION
            and self._cube_placed()
        )

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta.update(
            lang=(
                "walk to the rightmost window, put the Rubik's cube into the "
                "plate, then open the laptop on the same table"
            ),
            window_body=self.window_body,
            cube_body=self.task_spec["target_body"],
            plate_body=self.task_spec["plate_body"],
            table_body=self.task_spec["table_body"],
            laptop_body=self.laptop_body,
            laptop_joint=self.laptop_joint,
            laptop_source_scene="train_117",
            success_definition=(
                "reach the rightmost window, then keep the cube in contact with "
                "the plate and out of contact with the table for 1 second, then "
                "open the scene-117 laptop lid to at least 95% for 1 second "
                "while the cube stays in the plate; early placement or opening "
                "must be undone before completing that stage"
            ),
        )
        return meta


__all__ = ["HolodeckApproachWindowThenCubeInPlateThenOpenLaptop"]
