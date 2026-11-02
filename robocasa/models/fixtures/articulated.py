from robocasa.models.fixtures import Fixture


class ArticulatedObject(Fixture):
    """
    Generic articulated fixture converted from a USD asset. Supports any number
    of hinge joints (e.g. a lid or door); every hinge joint discovered in the
    MJCF is treated as a door joint for open/close queries.

    Args:
        xml (str): path to mjcf xml file
        name (str): name of the object
    """

    def __init__(
        self,
        xml="fixtures/articulated/ElectricCooker001",
        name="articulated_object",
        *args,
        **kwargs,
    ):
        super().__init__(
            xml=xml, name=name, duplicate_collision_geoms=False, *args, **kwargs
        )

    @property
    def door_joint_names(self):
        # all hinge joints found in the model (names already include naming_prefix)
        return list(self._joint_infos.keys())

    def is_open(self, env, th=0.90):
        return super().is_open(env, self.door_joint_names, th=th)

    def is_closed(self, env, th=0.005):
        return super().is_closed(env, self.door_joint_names, th=th)

    def open_door(self, env, min=0.90, max=1.0):
        self.set_joint_state(
            min=min, max=max, env=env, joint_names=self.door_joint_names
        )

    def close_door(self, env, min=0.0, max=0.0):
        self.set_joint_state(
            min=min, max=max, env=env, joint_names=self.door_joint_names
        )

    def get_reset_region_names(self):
        return ("int",)

    @property
    def nat_lang(self):
        return "electric cooker"
