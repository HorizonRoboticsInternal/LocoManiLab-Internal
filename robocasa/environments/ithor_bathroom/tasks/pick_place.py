"""Calibrated Bathroom pick-and-place tasks for exact scene geometry."""

from __future__ import annotations

from copy import deepcopy

import numpy as np

from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)

from .object_manipulation import _BathroomWorkspaceMixin


class _BathroomNativePickPlaceTask(_BathroomWorkspaceMixin):
    """Private base for exact native source objects and target geometry."""

    ATOMIC_SKILL = "PickPlace"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "pick_place"
    HOLD_STEPS = 5

    def _get_obj_cfgs(self):
        return []

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = (
            self._success_hold + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS


class _BathroomToiletPaperRollIntoBinTask(_BathroomNativePickPlaceTask):
    """Put an exact native full toilet-paper roll into the exact bin."""

    SPEC_TASK = "toilet_paper_roll_into_bin"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "source_fixture",
            "bin_fixture",
            "bin_region",
            "bin_region_z_offset",
            "surface_fixture",
            "surface_region",
            "source_xy",
            "source_quat",
            "bin_xy",
            "robot_pose",
            "lang",
        ),
    )
    SOURCE_SCALE = 1.0
    BIN_CENTER_XY_TOLERANCE = 0.01
    BIN_CENTER_Z_TOLERANCE = 0.01
    BIN_MAX_RESET_DISPLACEMENT = 0.10

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.source_fixture = self._exact_fixture(spec["source_fixture"])
        self.bin_fixture = self._exact_fixture(spec["bin_fixture"])
        self.bin_region = None
        if self.bin_fixture is not None:
            raw_bin_region = next(
                (
                    region
                    for region in self.get_region(
                        fixture=self.bin_fixture,
                        kind="receptacle",
                        return_all=True,
                    )
                    if region.stable_id == spec["bin_region"]
                ),
                None,
            )
            if raw_bin_region is not None:
                self.bin_region = deepcopy(raw_bin_region)
                self.bin_region._world_offset = np.asarray(
                    self.bin_region._world_offset, dtype=float
                ) + np.array(
                    [0.0, 0.0, float(spec["bin_region_z_offset"])],
                    dtype=float,
                )
        if self.source_fixture is not None:
            self.fixture_refs["toilet_paper_roll"] = self.source_fixture
        if self.bin_fixture is not None:
            self.fixture_refs["garbage_bin"] = self.bin_fixture
        if self.bin_region is not None:
            self.region_refs["garbage_bin"] = self.bin_region

    def _is_scene_valid(self):
        return bool(
            super()._is_scene_valid()
            and self.source_fixture is not None
            and self.bin_fixture is not None
            and self.bin_region is not None
        )

    def _reset_internal(self):
        super()._reset_internal()
        spec = self.scene_task_spec()
        self._bin_upright_axis = self._upright_axis(self.bin_fixture)
        self._set_fixture_quaternion(
            self.source_fixture, spec["source_quat"]
        )
        if not self._place_fixture_on_floor(
            self.bin_fixture,
            spec["bin_xy"],
            contract_name="native_bin_floor",
        ):
            raise RuntimeError(
                f"{self.scene_name}: no stable floor pose for "
                f"{self.bin_fixture.stable_id}"
            )
        if not self._place_fixture_on_support(
            self.source_fixture,
            self.work_surface_region,
            self.work_surface_fixture,
            spec["source_xy"],
            contract_name="toilet_roll_source_support",
        ):
            raise RuntimeError(
                f"{self.scene_name}: no stable contacted pose for "
                f"{self.source_fixture.stable_id}"
            )
        if self.count_body_collisions():
            raise RuntimeError(
                f"{self.scene_name}: bin task staging collides with G1"
            )
        self._record_source_horizontal_distance(self.source_fixture)
        self._bin_reset_center = np.asarray(
            self.bin_fixture.world_center(self.sim), dtype=float
        ).copy()
        self._success_hold = 0

    def _source_inside_bin(self):
        """Accept any roll orientation whose semantic center is in the bin."""
        source_center = np.asarray(
            self.source_fixture.world_center(self.sim), dtype=float
        )
        center, _ = self.bin_region.world_pose(self.sim)
        lower_z = float(center[2] - self.bin_region.half_size[2])
        upper_z = float(center[2] + self.bin_region.half_size[2])
        return bool(
            self.bin_region.contains_xy(
                source_center,
                self.sim,
                margin=-self.BIN_CENTER_XY_TOLERANCE,
            )
            and lower_z - self.BIN_CENTER_Z_TOLERANCE
            <= float(source_center[2])
            <= upper_z + self.BIN_CENTER_Z_TOLERANCE
        )

    def _bin_stays_at_reset_pose(self):
        center = np.asarray(self.bin_fixture.world_center(self.sim), dtype=float)
        return bool(
            np.linalg.norm(center - self._bin_reset_center)
            <= self.BIN_MAX_RESET_DISPLACEMENT
        )

    def _raw_success(self):
        return bool(
            self._source_inside_bin()
            and self._fixture_contacts_fixture(
                self.source_fixture, self.bin_fixture
            )
            and self._fixture_released(self.source_fixture)
            and self._fixture_is_upright(
                self.bin_fixture, self._bin_upright_axis
            )
            and self._bin_stays_at_reset_pose()
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            {
                "lang": self.scene_task_spec()["lang"],
                "source_fixture": self.source_fixture.stable_id,
                "target_fixture": self.bin_fixture.stable_id,
                "source_horizontal_distance": float(
                    self._source_horizontal_distance
                ),
                "success_definition": (
                    "the full toilet-paper roll center must be inside and the "
                    "roll must contact the upright bin without touching either "
                    "G1 hand while the bin remains within "
                    f"{self.BIN_MAX_RESET_DISPLACEMENT:.2f} m of its reset pose "
                    f"for {self.HOLD_STEPS} control steps"
                ),
            }
        )
        return ep_meta


class IThorBathroomToiletPaperRollIntoBin(
    _BathroomToiletPaperRollIntoBinTask,
    IThorBathroomEnv,
):
    """Put an exact native full toilet-paper roll into the exact bin."""


__all__ = ["IThorBathroomToiletPaperRollIntoBin"]
