"""iTHOR kitchen: put a bowl into the sink."""

import os

import numpy as np
from robosuite.utils.mjcf_utils import new_site
from robosuite.utils.placement_samplers import SequentialCompositeSampler

from robocasa.environments.ithor_atomic.pick_place import (
    SupportContactPlacementMixin,
)
from robocasa.environments.ithor_kitchen.ithor_kitchen import IThorKitchenEnv
from robocasa.models.scenes.ithor_kitchen_task_specs import (
    get_ithor_kitchen_task_spec,
)
from robocasa.models.scenes.thor_semantics import SemanticRegion


class IThorKitchenBowlToSink(SupportContactPlacementMixin, IThorKitchenEnv):
    ATOMIC_SKILL = "BowlToSink"
    DIFFICULTY_LEVEL = "L2"
    TASK_FAMILY = "receptacle"

    SOURCE_NAME = "bowl"
    SOURCE_LANG = "bowl"
    SOURCE_POS_KEY = "object_pos"
    SOURCE_XML = "models/assets/objects/objaverse/bowl/bowl_1/model.xml"
    SOURCE_SCALE = 0.65
    SUPPORTED_SCENE_IDS = (6, 7, 13, 24, 25)
    SPEC_TASK = "bowl_to_sink"
    SINK_XY_TOL = 0.03
    SINK_Z_TOL = 0.04
    TARGET_REGION_SITE_NAME = "bowl_to_sink_target_region"

    def __init__(self, *args, scene_ids=None, **kwargs):
        if scene_ids is None:
            scene_ids = self.SUPPORTED_SCENE_IDS
        super().__init__(*args, scene_ids=scene_ids, **kwargs)

    def _get_task_spec(self):
        if getattr(self, "_task_spec_scene", None) != self.scene_name:
            self.task_spec = get_ithor_kitchen_task_spec(
                self.scene_name,
                self.SPEC_TASK,
                required=(
                    "robot_pose",
                    self.SOURCE_POS_KEY,
                    "sink_fixture",
                    "sink_region",
                ),
            )
            self._task_spec_scene = self.scene_name
        return self.task_spec

    def _get_scene_robot_pose_calibration(self):
        pose = self._get_task_spec()["robot_pose"]
        pos = np.asarray(pose["pos"], dtype=float)
        if pos.shape != (3,) or "yaw" not in pose:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: robot_pose needs XYZ + yaw"
            )
        return pos, float(pose["yaw"])

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        spec = self._get_task_spec()
        self.sink_fixture = self.get_fixture(id=spec["sink_fixture"])
        if self.sink_fixture is None:
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: missing configured sink fixture"
            )
        self.sink_region = self._make_region(
            "sink", spec["sink_region"], self.sink_fixture.stable_id, "receptacle"
        )
        if bool(spec.get("show_target_region", False)):
            center, yaw = self.sink_region.world_pose()
            self.mujoco_arena.worldbody.append(
                new_site(
                    name=self.TARGET_REGION_SITE_NAME,
                    type="box",
                    pos=center,
                    size=self.sink_region.half_size,
                    quat=[np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)],
                    rgba=[0.1, 0.9, 0.2, 0.30],
                    group=0,
                )
            )

        self.fixture_refs["sink"] = self.sink_fixture
        self.region_refs["sink_receptacle"] = self.sink_region

    def _is_scene_valid(self):
        return bool(
            IThorKitchenEnv._is_scene_valid(self)
            and self.sink_fixture is not None
            and self.sink_region is not None
        )

    def _get_obj_cfgs(self):
        source_xml = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "../../../..", self.SOURCE_XML)
        )
        return [
            dict(
                name=self.SOURCE_NAME,
                obj_groups=source_xml,
                object_scale=self.SOURCE_SCALE,
            )
        ]

    def _build_placement_initializer(self):
        return SequentialCompositeSampler(name=f"Manual{self.SOURCE_NAME}ToSinkSampler")

    def _semantic_placements_are_valid(self):
        return True

    def _reset_internal(self):
        self._success_hold_steps = 0
        super()._reset_internal()
        self.success_hold_seconds = float(self._get_task_spec()["hold_seconds"])
        if self.SOURCE_NAME in self.objects:
            self._place_source_object()

    def _place_source_object(self):
        spec = self._get_task_spec()
        pos = np.asarray(spec[self.SOURCE_POS_KEY], dtype=float)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}: "
                f"{self.SOURCE_POS_KEY} must be finite [x, y, bottom_z]"
            )
        self._set_task_object_bottom_at(self.SOURCE_NAME, pos[:2], pos[2])

    def _success_region_bounds(self):
        center, yaw = self.sink_region.world_pose(self.sim)
        half_size = np.asarray(self.sink_region.half_size, dtype=float).copy()
        half_size[:2] += self.SINK_XY_TOL
        half_size[2] += self.SINK_Z_TOL
        return [(center, half_size, yaw)]

    def _raw_success(self):
        return bool(
            self.object_in_region(
                self.SOURCE_NAME,
                self.sink_region,
                rim_tol=self.SINK_Z_TOL,
                xy_tol=self.SINK_XY_TOL,
            )
            and self.object_released(self.SOURCE_NAME)
        )

    def _post_action(self, action):
        ret = super()._post_action(action)
        self._success_hold_steps = (
            self._success_hold_steps + 1 if self._raw_success() else 0
        )
        return ret

    def _check_success(self):
        required_steps = max(
            1, int(round(self.success_hold_seconds * self.control_freq))
        )
        return self._success_hold_steps >= required_steps

    def _make_region(self, label, spec, fixture_id, kind):
        center = np.asarray(spec.get("center"), dtype=float)
        half_size = np.asarray(spec.get("half_size"), dtype=float)
        yaw = float(spec.get("yaw", 0.0))
        if center.shape != (3,) or half_size.shape != (3,) or np.any(half_size <= 0):
            raise ValueError(
                f"{self.scene_name}/{self.SPEC_TASK}/{label}_region: "
                "expected center XYZ and positive half_size XYZ"
            )
        return SemanticRegion(
            stable_id=f"{fixture_id}:region:{kind}:task_calibrated",
            fixture_id=fixture_id,
            kind=kind,
            parent_body=None,
            local_center=np.zeros(3),
            half_size=half_size,
            _world_center=center,
            _world_yaw=yaw,
        )

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        sink_center, _ = self.sink_region.world_pose(self.sim)
        ep_meta.update(
            sink_receptacle=[float(v) for v in sink_center],
            lang=(
                f"pick up the {self.SOURCE_LANG} beside the sink and put it into "
                "the sink"
            ),
        )
        return ep_meta


__all__ = ["IThorKitchenBowlToSink"]
