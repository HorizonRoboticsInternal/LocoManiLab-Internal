"""Task-free TeleLab environment with an actuated conveyor and green desk."""

import math
from pathlib import Path

import numpy as np
from robosuite.models.base import MujocoXML
from robosuite.utils.mjcf_utils import (
    array_to_string,
    get_elements,
    scale_mjcf_model,
    string_to_array,
)

from robocasa.environments.telelab.telelab_env import TeleLabEnv
from robocasa.models.fixtures.conveyor_belt import ConveyorBeltFixture
from robocasa.models.objects.objects import MJCFObject


_ASSET_ROOT = Path(__file__).resolve().parents[2] / "models" / "assets"
_GREEN_DESK_XML = (
    _ASSET_ROOT / "objects" / "manual" / "green_desk" / "green_desk.xml"
)
_DEFAULT_CONVEYOR_POS = (0.0, 0.45, 0.0)
_DEFAULT_DESK_QUAT = (0.5, 0.5, 0.5, 0.5)
_DEFAULT_ROBOT_BASE_YAW = math.pi / 2.0
_BOX_SPECS = (
    ("box_1", -1.72),
    ("box_2", -1.03),
    ("box_3", -0.34),
    ("box_4", 0.35),
    ("box_5", 1.04),
    ("box_6", 1.73),
)


class TeleLabConveyor(TeleLabEnv):
    """TeleLab room with a controllable conveyor and an L-shaped green desk.

    Conveyor commands are independent from the robot action vector. Positive
    speed moves payloads along world ``+x`` and negative speed along ``-x``.

    Args:
        conveyor_pos: Translation of the conveyor's floor-aligned authored pose.
        conveyor_quat: Optional root quaternion in MuJoCo ``w x y z`` order.
        conveyor_speed: Initial belt velocity in metres per second.
        conveyor_speed_step: Increment used by :meth:`handle_scene_keypress`.
        conveyor_speed_range: Minimum and maximum commanded belt velocity in
            metres per second. The range must include zero.
        desk_pos: Floor-aligned position of the green desk. By default it is
            derived from the fixture scales and placed perpendicular to the
            conveyor's left/near corner.
        desk_quat: MuJoCo ``w x y z`` orientation that stands the desk upright
            and turns its long edge perpendicular to the conveyor.
    """

    KEY_CONVEYOR_SLOWER = ord("U")
    KEY_CONVEYOR_STOP = ord("P")
    KEY_CONVEYOR_FASTER = ord("O")
    ACTUATOR_SPEED_RANGE = (-1.5, 1.5)
    CONVEYOR_POS = _DEFAULT_CONVEYOR_POS
    DESK_QUAT = _DEFAULT_DESK_QUAT
    ROBOT_BASE_YAW = _DEFAULT_ROBOT_BASE_YAW
    L_CORNER_GAP_Y = 0.05
    DESK_SHIFT_X = 0.0
    ROBOT_INNER_CLEARANCE_X = 0.46
    ROBOT_INNER_CLEARANCE_Y = 0.35
    # Subclasses may narrow the authored payload set for a task-specific
    # episode without changing the general-purpose scene.
    BOX_SPECS = _BOX_SPECS
    # Optional task-specific translation in the conveyor's local frame.  The
    # default scene keeps payloads on the belt centreline.
    BOX_POSITION_OFFSET = (0.0, 0.0, 0.0)
    # Optional per-box replacement for BOX_POSITION_OFFSET.  Keeping this
    # empty preserves the common offset behavior for existing tasks.
    BOX_POSITION_OFFSET_OVERRIDES = {}
    BOX_SCALE = 1.0
    # Task subclasses may override individual payload properties without
    # changing the general-purpose conveyor scene.
    BOX_SCALE_OVERRIDES = {}
    BOX_MASS_OVERRIDES = {}
    BOX_QUAT_OVERRIDES = {}
    BOX_MASS_LIMIT_KG = None
    BOX_BOTTOM_HEIGHT = 0.555

    def __init__(
        self,
        *args,
        conveyor_pos=_DEFAULT_CONVEYOR_POS,
        conveyor_quat=None,
        conveyor_speed=0.0,
        conveyor_speed_step=0.01,
        conveyor_speed_range=(-1.5, 1.5),
        conveyor_scale=(1.0, 1.0, 1.0),
        desk_pos=None,
        desk_quat=_DEFAULT_DESK_QUAT,
        desk_scale=(1.0, 1.0, 1.0),
        robot_base_pos=None,
        robot_base_ori=None,
        **kwargs,
    ):
        self.conveyor_pos = np.asarray(conveyor_pos, dtype=float)
        if self.conveyor_pos.shape != (3,):
            raise ValueError("conveyor_pos must contain three values")
        self.conveyor_quat = conveyor_quat
        self.conveyor_scale = self._validate_scale(
            conveyor_scale, "conveyor_scale"
        )
        self.conveyor_speed_range = np.asarray(conveyor_speed_range, dtype=float)
        if self.conveyor_speed_range.shape != (2,):
            raise ValueError("conveyor_speed_range must contain two values")
        if not np.all(np.isfinite(self.conveyor_speed_range)):
            raise ValueError("conveyor_speed_range values must be finite")
        if not self.conveyor_speed_range[0] < self.conveyor_speed_range[1]:
            raise ValueError("conveyor_speed_range minimum must be below maximum")
        if not self.conveyor_speed_range[0] <= 0.0 <= self.conveyor_speed_range[1]:
            raise ValueError("conveyor_speed_range must include zero")
        actuator_minimum, actuator_maximum = self.ACTUATOR_SPEED_RANGE
        if (
            self.conveyor_speed_range[0] < actuator_minimum
            or self.conveyor_speed_range[1] > actuator_maximum
        ):
            raise ValueError(
                "conveyor_speed_range must stay within the actuator range "
                f"[{actuator_minimum}, {actuator_maximum}]"
            )

        self.conveyor_speed = float(conveyor_speed)
        if not np.isfinite(self.conveyor_speed):
            raise ValueError("conveyor_speed must be finite")
        self.conveyor_speed_step = float(conveyor_speed_step)
        if not np.isfinite(self.conveyor_speed_step) or self.conveyor_speed_step <= 0:
            raise ValueError("conveyor_speed_step must be finite and positive")

        self.desk_scale = self._validate_scale(desk_scale, "desk_scale")
        default_desk_pos, default_robot_base_pos = self._default_l_layout()
        self.desk_pos = np.asarray(
            default_desk_pos if desk_pos is None else desk_pos, dtype=float
        )
        if self.desk_pos.shape != (3,) or not np.all(np.isfinite(self.desk_pos)):
            raise ValueError("desk_pos must contain three finite values")
        self.desk_quat = np.asarray(desk_quat, dtype=float)
        if self.desk_quat.shape != (4,) or not np.all(np.isfinite(self.desk_quat)):
            raise ValueError("desk_quat must contain four finite values")
        desk_quat_norm = np.linalg.norm(self.desk_quat)
        if desk_quat_norm <= 0:
            raise ValueError("desk_quat must be non-zero")
        self.desk_quat /= desk_quat_norm
        self.robot_base_pos = np.asarray(
            default_robot_base_pos if robot_base_pos is None else robot_base_pos,
            dtype=float,
        )
        if robot_base_ori is None:
            robot_base_ori = (0.0, 0.0, self.ROBOT_BASE_YAW)
        self.robot_base_ori = np.asarray(robot_base_ori, dtype=float)

        self.conveyor = None
        self.desk = None
        self._scene_actuator_ids = {}
        self._scene_actuator_commands = {}
        self._belt_actuator_names = []
        self._belt_joint_ids = []
        self._belt_base_x = []
        self._belt_wrap_limit = None
        self.objects = {}
        self.boxes = self.objects
        super().__init__(
            *args,
            robot_base_pos=self.robot_base_pos,
            robot_base_ori=self.robot_base_ori,
            **kwargs,
        )

    @staticmethod
    def _validate_scale(scale, name):
        scale = np.asarray(scale, dtype=float)
        if scale.shape != (3,) or not np.all(np.isfinite(scale)) or np.any(scale <= 0):
            raise ValueError(f"{name} must contain three positive finite values")
        return scale

    def _default_l_layout(self):
        """Place the desk and robot around the conveyor's inner L corner."""
        conveyor_left = self.conveyor_pos[0] - 2.15 * self.conveyor_scale[0]
        conveyor_near = self.conveyor_pos[1] - 0.60 * self.conveyor_scale[1]

        # With DESK_QUAT, local Z points along world X and local X along world Y.
        desk_left_offset = (0.422 - 0.032) * self.desk_scale[2]
        desk_width = 2 * 0.422 * self.desk_scale[2]
        desk_far_offset = (0.075 + 0.795) * self.desk_scale[0]
        # Align the desk's left long edge with the conveyor's left end, then
        # apply any task-specific shift toward the robot.
        desk_left = conveyor_left + self.DESK_SHIFT_X
        desk_right = desk_left + desk_width
        desk_far = conveyor_near - self.L_CORNER_GAP_Y
        desk_pos = np.array(
            [
                desk_left + desk_left_offset,
                desk_far - desk_far_offset,
                0.0,
            ]
        )
        robot_pos = np.array(
            [
                conveyor_left + desk_width + self.ROBOT_INNER_CLEARANCE_X,
                conveyor_near - self.ROBOT_INNER_CLEARANCE_Y,
                0.0,
            ]
        )
        return desk_pos, robot_pos

    def _load_model(self):
        super()._load_model()
        self.conveyor = ConveyorBeltFixture(
            pos=self.conveyor_pos,
            quat=self.conveyor_quat,
            scale=self.conveyor_scale,
        )
        # Full-model merge is required: unlike merge_objects(), this includes
        # the fixture's actuator section as well as its body and assets.
        self.model.merge(self.conveyor)
        self.desk = MujocoXML(str(_GREEN_DESK_XML))
        if not np.allclose(self.desk_scale, 1.0):
            scale_mjcf_model(
                obj=self.desk.worldbody,
                asset_root=self.desk.asset,
                scale=self.desk_scale,
                get_elements_func=get_elements,
            )
        desk_body = self.desk.worldbody.find("./body[@name='green_desk']")
        if desk_body is None:
            raise ValueError(f"{_GREEN_DESK_XML}: missing root body 'green_desk'")
        desk_body.set("pos", array_to_string(self.desk_pos))
        desk_body.set("quat", array_to_string(self.desk_quat))
        self.model.merge(self.desk)
        self._load_parcel_boxes()

    def _load_parcel_boxes(self):
        """Place all six dynamic manual parcel boxes on the conveyor surface."""
        self.objects = {}
        self.boxes = self.objects
        conveyor_quat = (
            np.asarray(self.conveyor_quat, dtype=float)
            if self.conveyor_quat is not None
            else np.array([1.0, 0.0, 0.0, 0.0])
        )
        conveyor_quat = conveyor_quat / np.linalg.norm(conveyor_quat)
        conveyor_rotation = self._quat_to_matrix(conveyor_quat)

        # The authored conveyor root is z=0.25 and the tread top is z=0.798.
        conveyor_root = self.conveyor_pos + np.array(
            [0.0, 0.0, 0.25 * self.conveyor_scale[2]]
        )
        box_bottom_height = self.BOX_BOTTOM_HEIGHT * self.conveyor_scale[2]

        for box_name, local_x in self.BOX_SPECS:
            xml_path = (
                _ASSET_ROOT
                / "objects"
                / "manual"
                / "box"
                / box_name
                / f"{box_name}.xml"
            )
            box_object = MJCFObject(
                name=box_name,
                mjcf_path=str(xml_path),
                scale=self._box_scale(box_name),
                density=100,
                friction=(0.8, 0.03, 0.01),
            )
            body = box_object.get_obj()

            box_quat = self._box_quat(box_name, conveyor_quat)
            box_mass = self.BOX_MASS_OVERRIDES.get(box_name)
            if box_mass is not None:
                box_mass = float(box_mass)
                if not np.isfinite(box_mass) or box_mass <= 0:
                    raise ValueError(
                        f"BOX_MASS_OVERRIDES[{box_name!r}] must be positive and finite"
                    )
                mass_limit = self.BOX_MASS_LIMIT_KG
                if mass_limit is not None and box_mass > float(mass_limit):
                    raise ValueError(
                        f"BOX_MASS_OVERRIDES[{box_name!r}] exceeds the configured "
                        f"mass limit of {mass_limit} kg"
                    )
                for geom in body.iter("geom"):
                    if geom.get("group") != "1":
                        geom.set("mass", str(box_mass))

            # BOX_BOTTOM_HEIGHT denotes the desired bottom support height.
            # Account for a task-specific root rotation before placing the body.
            bbox_geom = next(
                geom
                for geom in body.iter("geom")
                if geom.get("name", "").endswith("reg_bbox")
            )
            bbox_center = string_to_array(bbox_geom.get("pos", "0 0 0"))
            bbox_half_size = string_to_array(bbox_geom.get("size", "0 0 0"))
            bbox_corners = np.asarray(
                [
                    bbox_center + np.asarray(signs, dtype=float) * bbox_half_size
                    for signs in (
                        (-1, -1, -1), (-1, -1, 1), (-1, 1, -1), (-1, 1, 1),
                        (1, -1, -1), (1, -1, 1), (1, 1, -1), (1, 1, 1),
                    )
                ]
            )
            local_box_quat = self.BOX_QUAT_OVERRIDES.get(box_name)
            if local_box_quat is None:
                local_box_quat = (1.0, 0.0, 0.0, 0.0)
            bbox_min_z = float(
                np.min(
                    (
                        self._quat_to_matrix(
                            np.asarray(local_box_quat, dtype=float)
                        )
                        @ bbox_corners.T
                    )[2]
                )
            )

            position_offset = self.BOX_POSITION_OFFSET_OVERRIDES.get(
                box_name, self.BOX_POSITION_OFFSET
            )
            local_pos = np.array(
                [local_x, 0.0, box_bottom_height - bbox_min_z], dtype=float
            ) + np.asarray(position_offset, dtype=float)
            body.set(
                "pos",
                array_to_string(conveyor_root + conveyor_rotation @ local_pos),
            )
            body.set("quat", array_to_string(box_quat))

            self.objects[box_name] = box_object
            self.model.merge_assets(box_object)
            self.model.worldbody.append(box_object.get_obj())
            self.model.mujoco_objects.append(box_object)

    def _box_scale(self, box_name):
        return self.BOX_SCALE_OVERRIDES.get(box_name, self.BOX_SCALE)

    def _box_quat(self, box_name, conveyor_quat):
        local_quat = self.BOX_QUAT_OVERRIDES.get(box_name)
        if local_quat is None:
            return conveyor_quat
        return self._quat_multiply(conveyor_quat, np.asarray(local_quat, dtype=float))

    @staticmethod
    def _quat_multiply(first, second):
        """Multiply MuJoCo-order (w, x, y, z) quaternions."""
        w1, x1, y1, z1 = first
        w2, x2, y2, z2 = second
        return np.asarray(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )

    @staticmethod
    def _quat_to_matrix(quat):
        """Convert a MuJoCo-order quaternion to a 3x3 rotation matrix."""
        w, x, y, z = quat
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )

    def _setup_references(self):
        super()._setup_references()
        robot_actuator_names = {
            name for robot in self.robots for name in robot.robot_model.actuators
        }
        self._scene_actuator_ids = {
            self.sim.model.actuator(actuator_id).name: actuator_id
            for actuator_id in range(self.sim.model.nu)
            if self.sim.model.actuator(actuator_id).name not in robot_actuator_names
        }
        self._scene_actuator_commands = {name: 0.0 for name in self._scene_actuator_ids}
        self._belt_actuator_names = sorted(
            name
            for name in self._scene_actuator_ids
            if name.startswith(ConveyorBeltFixture.ACTUATOR_PREFIX)
        )
        self._belt_joint_ids = []
        self._belt_base_x = []
        for name in self._belt_actuator_names:
            actuator_id = self._scene_actuator_ids[name]
            joint_id = int(self.sim.model.actuator_trnid[actuator_id, 0])
            self._belt_joint_ids.append(joint_id)
            body_id = int(self.sim.model.jnt_bodyid[joint_id])
            self._belt_base_x.append(float(self.sim.model.body_pos[body_id, 0]))

        if len(self._belt_base_x) > 1:
            spacing = (max(self._belt_base_x) - min(self._belt_base_x)) / (
                len(self._belt_base_x) - 1
            )
            self._belt_wrap_limit = (
                max(abs(min(self._belt_base_x)), abs(max(self._belt_base_x)))
                + spacing / 2
            )
        else:
            self._belt_wrap_limit = None
        self.set_conveyor_speed(self.conveyor_speed)

    @property
    def scene_actuator_names(self):
        """Names controlled outside the robot action vector."""
        return tuple(self._scene_actuator_ids)

    def set_scene_actuator(self, name, value):
        """Set one scene actuator command, clamped to its control range."""
        if name not in self._scene_actuator_ids:
            available = ", ".join(self.scene_actuator_names) or "<none>"
            raise KeyError(f"unknown scene actuator {name!r}; available: {available}")
        actuator_id = self._scene_actuator_ids[name]
        command = float(value)
        if self.sim.model.actuator_ctrllimited[actuator_id]:
            command = float(
                np.clip(command, *self.sim.model.actuator_ctrlrange[actuator_id])
            )
        self._scene_actuator_commands[name] = command
        self.sim.data.ctrl[actuator_id] = command
        return command

    def set_conveyor_speed(self, speed):
        """Set every belt actuator without changing the robot action space."""
        speed = float(speed)
        if not np.isfinite(speed):
            raise ValueError("conveyor speed must be finite")
        speed = float(np.clip(speed, *self.conveyor_speed_range))
        if not self._belt_actuator_names:
            if speed != 0.0:
                raise RuntimeError("TeleLabConveyor has no belt_drive_* actuators")
            self.conveyor_speed = 0.0
            return self.conveyor_speed
        commands = [
            self.set_scene_actuator(name, speed) for name in self._belt_actuator_names
        ]
        self.conveyor_speed = commands[0]
        return self.conveyor_speed

    def adjust_conveyor_speed(self, delta):
        return self.set_conveyor_speed(self.conveyor_speed + float(delta))

    def handle_scene_keypress(self, keycode):
        """Handle U/O/P conveyor keys; return True when consumed."""
        keycode = ord(chr(keycode).upper()) if 0 <= keycode < 256 else keycode
        if keycode == self.KEY_CONVEYOR_SLOWER:
            self.adjust_conveyor_speed(-self.conveyor_speed_step)
        elif keycode == self.KEY_CONVEYOR_STOP:
            self.set_conveyor_speed(0.0)
        elif keycode == self.KEY_CONVEYOR_FASTER:
            self.adjust_conveyor_speed(self.conveyor_speed_step)
        else:
            return False
        print(f"[telelab] conveyor speed: {self.conveyor_speed:+.2f} m/s")
        return True

    def scene_control_help(self):
        minimum, maximum = self.conveyor_speed_range
        return (
            "TeleLab scene controls: U/O = conveyor slower/faster, P = stop "
            f"(step {self.conveyor_speed_step:g} m/s, range "
            f"[{minimum:g}, {maximum:g}] m/s)"
        )

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step)
        for name, command in self._scene_actuator_commands.items():
            self.sim.data.ctrl[self._scene_actuator_ids[name]] = command
        self._wrap_conveyor_slats()

    def _wrap_conveyor_slats(self):
        if self._belt_wrap_limit is None:
            return
        period = 2 * self._belt_wrap_limit
        for joint_id, base_x in zip(self._belt_joint_ids, self._belt_base_x):
            qpos_addr = int(self.sim.model.jnt_qposadr[joint_id])
            world_x = base_x + self.sim.data.qpos[qpos_addr]
            if world_x > self._belt_wrap_limit:
                self.sim.data.qpos[qpos_addr] -= period
            elif world_x < -self._belt_wrap_limit:
                self.sim.data.qpos[qpos_addr] += period

    def get_ep_meta(self):
        return {
            "lang": "freely explore the TeleLab conveyor scene",
            "scene_xml": str(self.scene_xml),
            "desk_pos": self.desk_pos.tolist(),
            "desk_quat": self.desk_quat.tolist(),
            "conveyor_speed": self.conveyor_speed,
            "conveyor_speed_step": self.conveyor_speed_step,
            "conveyor_speed_range": self.conveyor_speed_range.tolist(),
        }
