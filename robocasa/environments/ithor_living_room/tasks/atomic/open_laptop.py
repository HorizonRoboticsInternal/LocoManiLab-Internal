"""iTHOR living-room open-laptop task."""

from robocasa.environments.ithor_living_room.tasks.atomic.close_laptop import (
    IThorLivingRoomCloseLaptop,
)


class IThorLivingRoomOpenLaptop(IThorLivingRoomCloseLaptop):
    """Open a fixed-base laptop lid from 20 to 90 percent for one second."""

    ATOMIC_SKILL = "OpenLaptop"
    BEHAVIOR = "open"
    SPEC_TASK = "open_laptop"

    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        body = self.find_scene_body(self.target_fixture.body_name)
        root_joint = body.find("joint[@type='free']")
        if root_joint is None:
            raise RuntimeError(f"{self.scene_name}: laptop base has no free joint")
        # Keep the calibrated base pose fixed while leaving the child lid hinge free.
        body.remove(root_joint)

    def _move_laptop_to_fixed_pose(self):
        """The static base already has its calibrated pose in the model."""

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["laptop_base_fixed"] = True
        return ep_meta


__all__ = ["IThorLivingRoomOpenLaptop"]
