"""Atomic drawer articulation tasks for Holodeck office scenes."""

from robocasa.environments.ithor_atomic.articulation import RoomOpenArticulation
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.scenes.holodeck_task_specs import get_holodeck_task_spec


class _HolodeckDrawerMixin:
    """Open or close one calibrated, physically sliding office drawer.

    Success is defined by the target joint state alone.
    """

    TASK_FAMILY = "articulation"
    TARGET_CATEGORIES = ("Dresser",)
    TARGET_LANG = "dresser drawer"
    TARGET_JOINT_TYPES = ("slide",)
    SUCCESS_JOINT_SELECTION = "target"
    INITIAL_ROBOT_JITTER_RADIUS = 0.0
    INITIAL_OBJECT_JITTER_RADIUS = 0.0
    ARTICULATION_STAND_POSE_JITTER = False
    HOLD_SECONDS = 1.0
    SUPPORTED_SCENE_IDS = (117, 1018, 1197, 1664, 2025)
    ARTICULATION_TASK_SPEC_LOADER = staticmethod(get_holodeck_task_spec)

    def _reset_internal(self):
        super()._reset_internal()
        # RoomOpenArticulation applies the calibrated initial qpos during
        # setup, but a normal reset only restores the compiled XML state.
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
        target_fraction = (
            self.SUCCESS_CLOSE_FRACTION
            if self.BEHAVIOR == "close"
            else self.SUCCESS_OPEN_FRACTION
        )
        meta["success_definition"] = (
            f"the specified drawer joint reaches {self.BEHAVIOR} fraction "
            f"{target_fraction:.2f}; no grasp or other action state is required"
        )
        return meta

class HolodeckOpenDrawer(_HolodeckDrawerMixin, RoomOpenArticulation, HolodeckEnv):
    """Open the calibrated office drawer to its scene-specific success threshold."""

    SPEC_TASK = "open_drawer"
    ATOMIC_SKILL = "OpenDrawer"
    BEHAVIOR = "open"
    OPEN_STAND_DISTANCE = 0.35


class HolodeckCloseDrawer(_HolodeckDrawerMixin, RoomOpenArticulation, HolodeckEnv):
    """Move the calibrated office drawer from 80% to 20% open."""

    SPEC_TASK = "close_drawer"
    ATOMIC_SKILL = "CloseDrawer"
    BEHAVIOR = "close"
    CLOSE_STAND_DISTANCE = 0.25


__all__ = ["HolodeckOpenDrawer", "HolodeckCloseDrawer"]
