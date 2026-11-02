from robocasa.environments.kitchen.kitchen import *


class G1Standing(Kitchen):
    """G1 spawned in a kitchen and held in its default standing pose.

    Step 1 scaffold: empty objects, no success condition, no language goal.
    Used to verify that the G1 robot + WHOLE_BODY_G1 composite controller +
    wbt DefaultPosController stack loads and steps without errors.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _setup_kitchen_references(self):
        super()._setup_kitchen_references()
        self.counter = self.register_fixture_ref(
            "counter", dict(id=FixtureType.COUNTER)
        )
        self.init_robot_base_ref = self.counter

    def get_ep_meta(self):
        ep_meta = super().get_ep_meta()
        ep_meta["lang"] = "stand still in the kitchen"
        return ep_meta

    def _get_obj_cfgs(self):
        return []

    def _check_success(self):
        return False
