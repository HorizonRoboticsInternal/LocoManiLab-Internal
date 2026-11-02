"""Private helpers for manipulating exact dynamic Bathroom scene fixtures."""

from __future__ import annotations

import numpy as np

from robocasa.environments.ithor_atomic.free_fixture import (
    FreeFixturePlacementMixin,
)
from robocasa.environments.ithor_atomic.pick_place import (
    resolve_task_support_region,
)
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin


class _NativeBathroomFixtureMixin(FreeFixturePlacementMixin):
    """Treat one exact free-joint scene fixture as a manipulable task object."""

    CONTACT_PENETRATION_TOLERANCE = 0.003
    UPRIGHT_COSINE = float(np.cos(np.deg2rad(20.0)))

    def _exact_fixture(self, stable_id):
        fixture = self.get_fixture(id=stable_id)
        return (
            fixture
            if fixture is not None and fixture.stable_id == stable_id
            else None
        )

    def _remove_exact_task_fixtures(self, stable_ids, *, forbidden=()):
        """Remove a strict task-local list without category-based fallback."""
        if not isinstance(stable_ids, (list, tuple)):
            raise ValueError("remove_fixtures must be a list of stable IDs")
        stable_ids = tuple(stable_ids)
        if not all(isinstance(stable_id, str) for stable_id in stable_ids):
            raise ValueError("remove_fixtures must contain stable ID strings")
        if len(stable_ids) != len(set(stable_ids)):
            raise ValueError("remove_fixtures must contain unique stable IDs")
        overlap = set(stable_ids) & set(forbidden)
        if overlap:
            raise ValueError(
                "task source or support cannot be removed: "
                f"{sorted(overlap)}"
            )

        fixtures = tuple(self._exact_fixture(stable_id) for stable_id in stable_ids)
        missing = tuple(
            stable_id
            for stable_id, fixture in zip(stable_ids, fixtures)
            if fixture is None
        )
        if missing:
            raise LookupError(
                f"{self.scene_name}: missing exact task-local fixtures "
                f"{list(missing)}"
            )

        expected_bodies = {
            fixture.body_name
            for fixture in fixtures
            if fixture.body_name is not None
        }
        removed_bodies = self.remove_semantic_fixtures(fixtures)
        if not expected_bodies.issubset(removed_bodies):
            raise RuntimeError(
                f"{self.scene_name}: failed to remove exact task-local "
                f"fixtures {list(stable_ids)}"
            )
        return fixtures

    def _fixture_released(self, fixture):
        """Return whether neither G1 hand still touches the fixture."""
        return not any(
            self._g1_hand_contacts_fixture(side, fixture)
            for side in ("left", "right")
        )

    def _g1_hand_contacts_fixture(self, side, fixture):
        hand_bodies = self._body_ids_matching((f"{side}_hand",))
        hand_geoms = self._geom_names_for_bodies(hand_bodies)
        fixture_geoms = self._geom_names_for_bodies(
            self._fixture_body_ids(fixture, include_siblings=True)
        )
        return bool(
            hand_geoms
            and fixture_geoms
            and self.check_contact(hand_geoms, fixture_geoms)
        )

    def _g1_hand_grasping_fixture(self, fixture):
        return any(
            self.g1_hand_closed(side)
            and self._g1_hand_contacts_fixture(side, fixture)
            for side in ("left", "right")
        )

    def _upright_axis(self, fixture):
        body_id = self.sim.model.body_name2id(fixture.body_name)
        matrix = np.asarray(
            self.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        axis = int(np.argmax(np.abs(matrix[2, :])))
        sign = 1.0 if matrix[2, axis] >= 0.0 else -1.0
        return axis, sign

    def _fixture_is_upright(self, fixture, upright_axis):
        axis, sign = upright_axis
        body_id = self.sim.model.body_name2id(fixture.body_name)
        matrix = np.asarray(
            self.sim.data.body_xmat[body_id], dtype=float
        ).reshape(3, 3)
        return float(sign * matrix[2, int(axis)]) >= self.UPRIGHT_COSINE

    def _set_exact_robot_pose(self, pose, task_label):
        xy = np.asarray(pose["xy"], dtype=float)
        yaw = float(pose["yaw"])
        if xy.shape != (2,):
            raise ValueError(f"{task_label}: robot_pose.xy must have two values")
        if hasattr(self, "_robot_free_joint_z"):
            delattr(self, "_robot_free_joint_z")
        if not (
            self.robot_footprint_is_on_floor(xy, radius=0.25)
            and self._set_robot_free_pose(xy, yaw)
            and self._calibrate_robot_free_joint_height()
            and self.count_body_collisions() == 0
        ):
            raise RuntimeError(
                f"{self.scene_name}: exact {task_label} robot pose is invalid"
            )
        self.robot_base_pos = np.array([xy[0], xy[1], 0.0])
        self.robot_base_ori = np.array([0.0, 0.0, yaw])


class _BathroomWorkspaceMixin(
    _NativeBathroomFixtureMixin,
    SceneSpecMixin,
):
    """Resolve one exact task surface and one exact collision-free robot pose."""

    INITIAL_POSE_JITTER = False
    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.work_surface_fixture = self._exact_fixture(spec["surface_fixture"])
        raw_region = None
        if self.work_surface_fixture is not None:
            raw_region = next(
                (
                    region
                    for region in self.get_region(
                        fixture=self.work_surface_fixture,
                        return_all=True,
                    )
                    if region.stable_id == spec["surface_region"]
                ),
                None,
            )
        self.work_surface_region = (
            resolve_task_support_region(raw_region)
            if raw_region is not None
            else None
        )
        if self.work_surface_fixture is not None:
            self.fixture_refs["task_surface"] = self.work_surface_fixture
        if self.work_surface_region is not None:
            self.region_refs["task_surface"] = self.work_surface_region
        pose = spec["robot_pose"]
        self.set_robot_base_pose(
            [float(pose["xy"][0]), float(pose["xy"][1]), 0.0],
            float(pose["yaw"]),
        )

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.work_surface_fixture is not None
            and self.work_surface_region is not None
        )

    def _setup_scene(self):
        super()._setup_scene()
        self._set_exact_robot_pose(
            self.scene_task_spec()["robot_pose"], "Bathroom task"
        )

    def _record_source_horizontal_distance(self, source_fixture):
        """Record the task-specific near-pose-to-source reach distance.

        Composite tasks replace the near manipulation pose with a far
        navigation start during scene setup.  The relevant reach measurement
        is therefore always relative to the explicit near ``robot_pose`` in
        the task spec, rather than the robot's current world position.
        """

        spec = self.scene_task_spec()
        reference_xy = np.asarray(spec["robot_pose"]["xy"], dtype=float)
        source_xy = np.asarray(
            source_fixture.world_center(self.sim), dtype=float
        )[:2]
        if reference_xy.shape != (2,) or not np.all(np.isfinite(reference_xy)):
            raise ValueError("robot_pose.xy must contain two finite values")
        if source_xy.shape != (2,) or not np.all(np.isfinite(source_xy)):
            raise RuntimeError("source fixture has no finite world XY center")

        self._source_horizontal_distance = float(
            np.linalg.norm(reference_xy - source_xy)
        )
        return self._source_horizontal_distance


class _BathroomNativeObjectTask(_BathroomWorkspaceMixin):
    """Stage one exact free-joint scene object on a calibrated work surface."""

    SPEC_TASK = None

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.source_fixture = self._exact_fixture(spec["source_fixture"])
        self._setup_exact_rearrangements(spec)
        if self.source_fixture is not None:
            self.fixture_refs["source_object"] = self.source_fixture

    def _setup_exact_rearrangements(self, spec):
        remove_values = spec.get("remove_fixtures", ())
        remove_ids = tuple(remove_values)

        auxiliary_values = spec.get("auxiliary_placements", ())
        if not isinstance(auxiliary_values, (list, tuple)):
            raise ValueError("auxiliary_placements must be a list")
        auxiliary_specs = tuple(auxiliary_values)
        self._auxiliary_placements = []
        seen_auxiliary_ids = set()
        for index, placement in enumerate(auxiliary_specs):
            if not isinstance(placement, dict) or set(placement) != {
                "fixture",
                "xy",
            }:
                raise ValueError(
                    "each auxiliary placement requires exactly fixture and xy"
                )
            stable_id = placement["fixture"]
            xy = np.asarray(placement["xy"], dtype=float)
            if xy.shape != (2,) or not np.all(np.isfinite(xy)):
                raise ValueError(
                    "auxiliary placement xy must have two finite values"
                )
            if not isinstance(stable_id, str):
                raise ValueError("auxiliary fixture must be a stable ID string")
            if stable_id in seen_auxiliary_ids:
                raise ValueError(
                    f"duplicate auxiliary fixture placement: {stable_id}"
                )
            seen_auxiliary_ids.add(stable_id)
            fixture = self._exact_fixture(stable_id)
            if fixture is not None:
                self._auxiliary_placements.append((fixture, xy))
                self.fixture_refs[f"native_auxiliary_{index}"] = fixture
        resolved_auxiliary_ids = {
            fixture.stable_id for fixture, _ in self._auxiliary_placements
        }
        self._missing_auxiliary_fixture_ids = tuple(
            placement["fixture"]
            for placement in auxiliary_specs
            if placement["fixture"] not in resolved_auxiliary_ids
        )

        source_id = spec["source_fixture"]
        surface_id = spec["surface_fixture"]
        auxiliary_ids = resolved_auxiliary_ids
        if source_id in remove_ids or source_id in auxiliary_ids:
            raise ValueError("the task source cannot be removed or auxiliary")
        if surface_id in remove_ids or surface_id in auxiliary_ids:
            raise ValueError("the task surface cannot be removed or auxiliary")
        if set(remove_ids) & auxiliary_ids:
            raise ValueError("a removed fixture cannot be placed as auxiliary")

        self._removed_fixtures = self._remove_exact_task_fixtures(
            remove_values,
            forbidden=(source_id, surface_id, *auxiliary_ids),
        )
        self._removed_fixture_ids = remove_ids

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.source_fixture is not None
            and not self._missing_auxiliary_fixture_ids
        )

    def _get_obj_cfgs(self):
        return []

    def _park_source_for_auxiliary_placement(self):
        joint = self._fixture_free_joint_name(self.source_fixture)
        qpos = self.sim.data.get_joint_qpos(joint).copy()
        qpos[2] += 2.0
        self.sim.data.set_joint_qpos(joint, qpos)
        self._zero_fixture_velocity(self.source_fixture)
        self.sim.forward()

    def _reset_internal(self):
        super()._reset_internal()
        spec = self.scene_task_spec()
        if self._auxiliary_placements:
            self._park_source_for_auxiliary_placement()
        for index, (fixture, xy) in enumerate(self._auxiliary_placements):
            if not self._place_fixture_on_support(
                fixture,
                self.work_surface_region,
                self.work_surface_fixture,
                xy,
                contract_name=f"native_auxiliary_{index}_support",
            ):
                raise RuntimeError(
                    f"{self.scene_name}: no stable contacted pose for "
                    f"auxiliary fixture {fixture.stable_id}"
                )
        if "source_quat" in spec:
            self._set_fixture_quaternion(
                self.source_fixture,
                spec["source_quat"],
            )
        if not self._place_fixture_on_support(
            self.source_fixture,
            self.work_surface_region,
            self.work_surface_fixture,
            spec["source_xy"],
            contract_name="native_source_support",
            probe_above=float(spec.get("support_probe_above", 0.08)),
            probe_below=float(spec.get("support_probe_below", 0.30)),
        ):
            raise RuntimeError(
                f"{self.scene_name}: no stable contacted pose for "
                f"{self.source_fixture.stable_id}"
            )
        if self.count_body_collisions():
            raise RuntimeError(
                f"{self.scene_name}: native object staging collides with G1"
            )
        self._record_source_horizontal_distance(self.source_fixture)

    def _native_object_ep_meta(self):
        metadata = {
            "source_fixture": self.source_fixture.stable_id,
            "source_surface": self.work_surface_region.stable_id,
            "removed_fixtures": list(self._removed_fixture_ids),
            "auxiliary_fixtures": [
                fixture.stable_id for fixture, _ in self._auxiliary_placements
            ],
        }
        if hasattr(self, "_source_horizontal_distance"):
            metadata["source_horizontal_distance"] = float(
                self._source_horizontal_distance
            )
        return metadata


__all__ = []
