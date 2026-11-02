"""Reorient exact native Bathroom objects on calibrated work surfaces."""

from __future__ import annotations

import numpy as np

from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)

from .object_manipulation import _BathroomNativeObjectTask


class _BathroomNativeUprightTask(_BathroomNativeObjectTask):
    """Stand one named native object upright and release it on its surface."""

    ATOMIC_SKILL = "ReorientObject"
    DIFFICULTY_LEVEL = "L1"
    TASK_FAMILY = "reorientation"
    HOLD_STEPS = 5
    SUPPORT_PLANE_TOLERANCE = 0.03

    SOURCE_LABEL = None

    def _reset_internal(self):
        super()._reset_internal()
        spec = self.scene_task_spec()
        self._target_upright_axis = (
            int(spec["upright_axis"]),
            float(spec["upright_sign"]),
        )
        if self._fixture_is_upright(
            self.source_fixture,
            self._target_upright_axis,
        ):
            raise RuntimeError(
                f"{self.scene_name}: upright task source starts upright"
            )
        self._success_hold = 0

    def _raw_success(self):
        support_bodies = self._fixture_body_ids(
            self.work_surface_fixture,
            include_siblings=True,
        )
        contact_distances = self._fixture_contact_distances(
            self.source_fixture,
            support_bodies,
        )
        return bool(
            self._fixture_is_upright(
                self.source_fixture,
                self._target_upright_axis,
            )
            and self._fixture_bbox_inside_region_xy(
                self.source_fixture,
                self.work_surface_region,
            )
            and self._fixture_bottom_is_on_support_plane()
            and contact_distances
            and min(contact_distances)
            >= -self.CONTACT_PENETRATION_TOLERANCE
            and not self._fixture_has_illegal_penetration(
                self.source_fixture,
                self.work_surface_fixture.body_names,
            )
            and self._fixture_released(self.source_fixture)
        )

    def _fixture_bottom_is_on_support_plane(self):
        support_z = float(
            self.work_surface_region.world_pose(self.sim)[0][2]
        )
        bottom_z = float(
            np.min(self._fixture_bbox_points(self.source_fixture)[:, 2])
        )
        return bool(
            abs(bottom_z - support_z) <= self.SUPPORT_PLANE_TOLERANCE
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold = (
            self._success_hold + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        return self._success_hold >= self.HOLD_STEPS

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(self._native_object_ep_meta())
        ep_meta.update(
            {
                "lang": self.scene_task_spec()["lang"],
                "success_definition": (
                    f"the named {self.SOURCE_LABEL} must be upright, fully "
                    "inside and resting on top of the named work-surface "
                    "region, shallowly touching that surface, free of "
                    "unrelated penetration, and released "
                    f"for {self.HOLD_STEPS} control steps"
                ),
            }
        )
        return ep_meta


class IThorBathroomStandSoapBottleUpright(
    _BathroomNativeUprightTask,
    IThorBathroomEnv,
):
    """Stand the exact native soap bottle upright."""

    SPEC_TASK = "stand_soap_bottle_upright"
    SOURCE_LABEL = "soap bottle"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "source_fixture",
            "surface_fixture",
            "surface_region",
            "source_xy",
            "source_quat",
            "upright_axis",
            "upright_sign",
            "remove_fixtures",
            "auxiliary_placements",
            "robot_pose",
            "lang",
        ),
    )


__all__ = ["IThorBathroomStandSoapBottleUpright"]
