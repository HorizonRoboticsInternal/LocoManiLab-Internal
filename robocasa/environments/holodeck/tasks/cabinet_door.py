"""Atomic cabinet-door articulation task for the Holodeck office scene."""

from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class _HolodeckCabinetDoorMixin:
    """Open the calibrated office credenza door to its target fraction."""

    TASK_FAMILY = "articulation"
    TARGET_CATEGORIES = ("Dresser",)
    TARGET_LANG = "office desk cabinet door"
    TARGET_JOINT_TYPES = ("hinge",)
    SUCCESS_JOINT_SELECTION = "target"
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    ARTICULATION_STAND_POSE_JITTER = False
    HOLD_SECONDS = 1.0
    SUPPORTED_SCENE_IDS = (117,)
    USE_DEFAULT_ASSET_LAYOUT = False
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_holodeck_task_spec)

    def _reset_internal(self):
        super()._reset_internal()
        # A normal MuJoCo reset restores the compiled XML qpos. Reapply the
        # calibrated 20% open state for every episode.
        self.sim.data.set_joint_qpos(
            self.target_joint, self._initial_joint_qpos(self.target_semantic_joint)
        )
        self.sim.data.set_joint_qvel(self.target_joint, 0.0)
        self.sim.forward()

    def _configure_articulation_task(self):
        super()._configure_articulation_task()
        self.HOLD_STEPS = max(1, int(round(self.HOLD_SECONDS * self.control_freq)))

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        spec = self._get_articulation_task_spec()
        if spec is not None and spec.get("lang"):
            meta["lang"] = spec["lang"]
        meta["success_definition"] = (
            "the specified office desk cabinet door joint reaches open fraction "
            f"{self.SUCCESS_OPEN_FRACTION:.2f} and remains there for "
            f"{self.HOLD_SECONDS:.1f} seconds"
        )
        return meta


class HolodeckOpenCabinetDoor(
    _HolodeckCabinetDoorMixin, RoomOpenArticulation, HolodeckEnv
):
    """Open the office desk's nearest credenza cabinet door."""

    SPEC_TASK = "open_cabinet_door"
    ATOMIC_SKILL = "OpenCabinetDoor"
    BEHAVIOR = "open"
    OPEN_STAND_DISTANCE = 0.35


__all__ = ["HolodeckOpenCabinetDoor"]
