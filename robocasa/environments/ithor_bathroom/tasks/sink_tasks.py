"""Put the canonical sponge into FloorPlan401's sink."""

import numpy as np

from robocasa.environments.ithor_atomic.pick_place import SupportContactPlacementMixin
from robocasa.environments.ithor_atomic.task_assets import (
    SPONGE_MODEL,
    SPONGE_SCALE,
    asset_xml_path,
)
from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.environments.ithor_bathroom.ithor_bathroom import IThorBathroomEnv
from robocasa.models.scenes.ithor_bathroom_task_specs import (
    get_ithor_bathroom_task_specs,
    get_supported_scene_ids,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion


class IThorBathroomPutSpongeInSink(
    SupportContactPlacementMixin, SceneSpecMixin, IThorBathroomEnv
):
    """Succeed after one second of sponge contact with the sink bottom."""

    ATOMIC_SKILL = "PickPlace"
    TASK_FAMILY = "receptacle"
    SPEC_TASK = "put_sponge_in_sink"
    SUPPORTED_SCENE_IDS = get_supported_scene_ids(SPEC_TASK)
    SCENE_SPECS = get_ithor_bathroom_task_specs(
        SPEC_TASK,
        required=(
            "sink_fixture",
            "sink_bottom_geoms",
            "source_region",
            "sink_region",
            "robot_pose",
            "lang",
        ),
    )
    SOURCE_NAME = "sponge"
    SOURCE_SCALE = SPONGE_SCALE
    INITIAL_POSE_JITTER = False

    def _get_scene_robot_pose_calibration(self):
        pose = self.scene_task_spec()["robot_pose"]
        return np.asarray(pose["pos"], dtype=float), float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self.scene_task_spec()
        self.sink_fixture = self.get_fixture(id=spec["sink_fixture"])
        if (
            self.sink_fixture is None
            or self.sink_fixture.stable_id != spec["sink_fixture"]
        ):
            raise ValueError(f"{self.scene_name}: configured sink fixture is missing")
        self.sink_bottom_geoms = tuple(spec["sink_bottom_geoms"])
        self.source_region = self._make_region("support", spec["source_region"])
        self.sink_region = self._make_region("receptacle", spec["sink_region"])
        self.fixture_refs["sink"] = self.sink_fixture
        self.region_refs["sink_rim"] = self.source_region
        self.region_refs["sink_receptacle"] = self.sink_region

    def _make_region(self, kind, spec):
        return SemanticRegion(
            stable_id=f"{self.sink_fixture.stable_id}:region:{kind}:sponge_in_sink",
            fixture_id=self.sink_fixture.stable_id,
            kind=kind,
            parent_body=None,
            local_center=np.zeros(3),
            half_size=np.asarray(spec["half_size"], dtype=float),
            _world_center=np.asarray(spec["center"], dtype=float),
            _world_yaw=0.0,
        )

    def _get_obj_cfgs(self):
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups=asset_xml_path(SPONGE_MODEL),
                object_scale=self.SOURCE_SCALE,
                placement=dict(
                    region=self.source_region,
                    rotation=0.0,
                    strategy="center",
                    z_offset=0.0,
                ),
            )
        ]

    def _reset_internal(self):
        self._success_hold_steps = 0
        super()._reset_internal()
        if not self.place_object_on_support_contact(
            self.SOURCE_NAME,
            self.source_region,
            fixture=self.sink_fixture,
            inward_fractions=(0.0,),
            probe_above=0.01,
            probe_below=0.02,
            probe_step=0.001,
        ):
            raise RuntimeError(
                f"{self.scene_name}: sponge has no shallow sink-rim support"
            )

    def _raw_success(self):
        return self.check_contact(
            self.objects[self.SOURCE_NAME].contact_geoms, self.sink_bottom_geoms
        )

    def _post_action(self, action):
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return super()._post_action(action)

    def _check_success(self):
        return self._success_hold_steps >= self.HOLD_STEPS

    def reward(self, action=None):
        return float(self._check_success())

    def _success_region_bounds(self):
        center, yaw = self.sink_region.world_pose(self.sim)
        return [(center, self.sink_region.half_size.copy(), yaw)]

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta.update(
            lang=self.scene_task_spec()["lang"],
            source_object=self.SOURCE_NAME,
            source_model=SPONGE_MODEL,
            source_scale=self.SOURCE_SCALE,
            source_support_region=self.source_region.stable_id,
            target_fixture=self.sink_fixture.stable_id,
            target_contact_geoms=list(self.sink_bottom_geoms),
            success_requires_release=False,
            success_definition=(
                "the sponge must contact the sink bottom continuously for 1.0 s, "
                "independent of grasp or hand contact"
            ),
        )
        return ep_meta


__all__ = ["IThorBathroomPutSpongeInSink"]
