"""Scene-calibrated articulation tasks for iTHOR bedrooms."""

import numpy as np

from robocasa.environments.ithor_bedroom.ithor_bedroom import IThorBedroomEnv
from robocasa.models.scenes.ithor_bedroom_task_specs import (
    get_ithor_bedroom_task_spec,
    get_ithor_bedroom_task_specs,
    get_supported_scene_ids,
)

from .articulation_base import (
    BedroomArticulationMixin,
    BedroomTwentyEightyArticulationMixin,
)


_TOP_DRAWER_REQUIRED_FIELDS = ("fixture", "joint", "target", "lang")


def _load_bedroom_top_drawer_scene_specs():
    """Load exact alternate-fixture drawer specs and reject task overlap."""

    specs = get_ithor_bedroom_task_specs(
        "top_drawer", required=_TOP_DRAWER_REQUIRED_FIELDS
    )
    for scene_name, spec in specs.items():
        primary = get_ithor_bedroom_task_spec(
            scene_name, "drawer", required=("fixture", "joint")
        )
        if spec["fixture"] == primary["fixture"]:
            raise ValueError(
                f"{scene_name}/top_drawer must use a different fixture from drawer"
            )
        if spec["joint"] == primary["joint"]:
            raise ValueError(
                f"{scene_name}/top_drawer must use a different joint from drawer"
            )
    return specs


class _BedroomTopDrawerMixin(
    BedroomTwentyEightyArticulationMixin,
    BedroomArticulationMixin,
):
    """Enforce that TopDrawer is the highest drawer on another furniture item."""

    MIN_PRIMARY_DRAWER_SEPARATION = 1.0
    SUCCESS_FRACTION_TOLERANCE = 1e-9

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        primary_spec = get_ithor_bedroom_task_spec(
            self.scene_name, "drawer", required=("fixture", "joint")
        )
        self.primary_drawer_fixture = self.get_fixture(
            id=primary_spec["fixture"]
        )
        if self.target_fixture is None or self.primary_drawer_fixture is None:
            raise RuntimeError(
                f"{self.scene_name}: missing exact TopDrawer fixture reference"
            )
        if self.target_fixture.stable_id == self.primary_drawer_fixture.stable_id:
            raise RuntimeError(
                f"{self.scene_name}: TopDrawer reuses the normal Drawer fixture"
            )
        self.primary_drawer_separation = float(
            np.linalg.norm(
                np.asarray(self.target_fixture.center[:2], dtype=float)
                - np.asarray(self.primary_drawer_fixture.center[:2], dtype=float)
            )
        )
        if self.primary_drawer_separation < self.MIN_PRIMARY_DRAWER_SEPARATION:
            raise RuntimeError(
                f"{self.scene_name}: TopDrawer fixture is only "
                f"{self.primary_drawer_separation:.3f} m from the normal Drawer "
                "fixture"
            )
        self.fixture_refs["primary_drawer_reference"] = (
            self.primary_drawer_fixture
        )

    def _setup_scene(self):
        super()._setup_scene()
        slide_joints = [
            joint
            for joint in self.target_fixture.joints.values()
            if self._joint_type_name(joint) == "slide"
        ]
        if not slide_joints:
            raise RuntimeError(
                f"{self.scene_name}: TopDrawer fixture has no slide joint"
            )
        target_z = self._joint_world_z(self.target_semantic_joint)
        highest_z = max(self._joint_world_z(joint) for joint in slide_joints)
        if target_z < highest_z - 1e-6:
            raise RuntimeError(
                f"{self.scene_name}: exact TopDrawer joint is not the highest "
                "drawer joint"
            )
        self.target_joint_level = "top"

    def _raw_success(self):
        fraction = self.target_semantic_joint.fraction(self.sim)
        tolerance = self.SUCCESS_FRACTION_TOLERANCE
        if self.BEHAVIOR == "close":
            return fraction <= self.SUCCESS_CLOSE_FRACTION + tolerance
        return fraction >= self.SUCCESS_OPEN_FRACTION - tolerance

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["primary_drawer_fixture"] = (
            self.primary_drawer_fixture.stable_id
        )
        ep_meta["drawer_fixture_separation"] = self.primary_drawer_separation
        return ep_meta


class IThorBedroomOpenTopDrawer(_BedroomTopDrawerMixin, IThorBedroomEnv):
    """Open the top drawer on a second, spatially distinct furniture item."""

    ATOMIC_SKILL = "OpenDrawer"
    SPEC_TASK = "top_drawer"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = _load_bedroom_top_drawer_scene_specs()
    BEHAVIOR = "open"
    OPEN_STAND_DISTANCE = 0.55


class IThorBedroomCloseTopDrawer(_BedroomTopDrawerMixin, IThorBedroomEnv):
    """Close the top drawer on a second, spatially distinct furniture item."""

    ATOMIC_SKILL = "CloseDrawer"
    SPEC_TASK = "top_drawer"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = _load_bedroom_top_drawer_scene_specs()
    BEHAVIOR = "close"
    CLOSE_STAND_DISTANCE = 0.55


__all__ = [
    "IThorBedroomCloseTopDrawer",
    "IThorBedroomOpenTopDrawer",
]
