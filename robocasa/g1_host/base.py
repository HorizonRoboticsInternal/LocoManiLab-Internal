"""Common robosuite-facing shell for the G1 whole-body composite controllers.

These controllers do not use robosuite's part-controller machinery at all: they
take no env action, compute a 29-joint PD target from a learned policy, and
write ``sim.data.ctrl`` directly. That leaves an identical set of obligations
towards robosuite -- stub out the action API, rebuild the MuJoCo address maps on
every ``env.reset()``, read sim state, write torques -- which each composite
used to satisfy with its own copy.

``G1CompositeBase`` owns exactly that shell, delegating the actual MuJoCo work
to :mod:`g1_runtime`.

It deliberately does **not** own ``run_controller``. The two subclasses' control
loops share only ~20 of their 110/210 lines, and four of the differences are
load-bearing: WHOLE_BODY_G1 polls PICO only in ``sonic_manager`` mode and acts
on the resulting ``CalibrationEvent``, swaps between three preloaded SONIC
controllers while refreshing the inactive ones' references, overrides the arm
target in OFF, and drives its hands from joint-range-derived close poses with
its own gains -- where G1_BRIDGE has one policy call and sim2real's
hardware-tuned dex3 gains. A shared skeleton would need six hooks and would put
the untestable VLA deploy path behind it.

Same reasoning for ``reset``, ``handle_keypress``, and anything gripper-related.
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np

from robosuite.controllers.composite.composite_controller import CompositeController
from g1_bridge.g1_runtime import G1_JOINT_NAMES
from robocasa.g1_host.mujoco_io import (
    build_freejoint_mapping,
    build_joint_mapping,
    read_body_pose_w,
    read_joint_state,
    read_pelvis_state,
    read_root_pos,
    write_body_torques,
)


class G1CompositeBase(CompositeController):
    """Address maps, sim readers, torque write, and the robosuite API stubs."""

    #: Joint order this controller drives. Subclasses may narrow it.
    JOINT_NAMES = G1_JOINT_NAMES

    def __init__(self, sim, robot_model, grippers):
        super().__init__(sim, robot_model, grippers)
        # Per-joint sim.data addresses / ctrl indices, None where this
        # particular G1 variant does not carry the joint or its actuator.
        self._qpos_addrs = None
        self._qvel_addrs = None
        self._actuator_ids = None
        # Root free joint, absent on fixed-base variants.
        self._free_qpos_addr = None
        self._free_qvel_addr = None

    # ------------------------------------------------------------------ #
    # Lifecycle. Both helpers are meant to be called from the subclass's
    # load_controller_config, which robosuite re-invokes on EVERY env.reset()
    # (Robot.reset() -> _load_controller()). sim/sim.model are genuinely new
    # objects each reset, so the address maps MUST be rebuilt every call --
    # unlike ONNX sessions and hardware connections, which must NOT be.
    # ------------------------------------------------------------------ #
    def _init_robosuite_state(self, part_controller_config):
        """Satisfy the attributes robosuite expects without building any part
        controllers -- these composites drive every joint themselves."""
        self.part_controller_config = part_controller_config or {}
        self.part_controllers = OrderedDict()
        self._action_split_indexes = OrderedDict()

    def _build_mappings(self):
        mapping = build_joint_mapping(self.sim, self.robot_model, self.JOINT_NAMES)
        self._qpos_addrs = mapping.qpos_addrs
        self._qvel_addrs = mapping.qvel_addrs
        self._actuator_ids = mapping.actuator_ids
        qpos_addr, qvel_addr = build_freejoint_mapping(self.sim, self.robot_model)
        # Assigned only when found, preserving the original behaviour of both
        # composites: a model whose root body is missing keeps whatever the
        # previous load left behind rather than reverting to None.
        if qpos_addr is not None:
            self._free_qpos_addr = qpos_addr
            self._free_qvel_addr = qvel_addr

    # Public read-only views of the address maps, for collaborators that need
    # raw sim addresses (e.g. g1_bridge backends writing qpos directly).
    @property
    def qpos_addrs(self):
        return self._qpos_addrs

    @property
    def qvel_addrs(self):
        return self._qvel_addrs

    @property
    def actuator_ids(self):
        return self._actuator_ids

    @property
    def free_qpos_addr(self):
        return self._free_qpos_addr

    @property
    def free_qvel_addr(self):
        return self._free_qvel_addr

    # ------------------------------------------------------------------ #
    # Sim state
    # ------------------------------------------------------------------ #
    def _read_joint_state(self):
        return read_joint_state(self.sim, self._qpos_addrs, self._qvel_addrs)

    def _read_pelvis_state(self):
        """Return (quaternion[wxyz], body-local angular velocity)."""
        return read_pelvis_state(self.sim, self._free_qpos_addr, self._free_qvel_addr)

    def _read_root_pos(self):
        physical_root = getattr(self.robot_model, "physical_root_body", None)
        if physical_root is not None:
            bid = self.sim.model.body_name2id(physical_root)
            return np.asarray(self.sim.data.body_xpos[bid], dtype=np.float32).copy()
        return read_root_pos(self.sim, self._free_qpos_addr)

    def _read_body_pose_w(self, body_name):
        return read_body_pose_w(self.sim, self.robot_model, body_name)

    def _write_body_torques(self, target_q, kps, kds, qpos, qvel):
        return write_body_torques(
            self.sim, self._actuator_ids, target_q, kps, kds, qpos, qvel
        )

    # Public names for the readers above. Collaborators outside this package
    # (g1_bridge's backends, via the Sim2RealContext Protocol) read sim
    # state through these rather than the underscore-prefixed originals.
    read_joint_state = _read_joint_state
    read_pelvis_state = _read_pelvis_state
    read_root_pos = _read_root_pos
    read_body_pose_w = _read_body_pose_w

    # ------------------------------------------------------------------ #
    # robosuite action API. No env-driven action: a single dummy dim keeps
    # robosuite/robocasa able to build a Box action space.
    # ------------------------------------------------------------------ #
    def setup_action_split_idx(self):
        pass

    def set_goal(self, all_action):
        pass

    def update_state(self):
        return

    @property
    def action_limits(self):
        return np.array([0.0]), np.array([0.0])

    def create_action_vector(self, action_dict):
        return np.zeros(1)

    def create_action_dict_from_action_vector(self, action_vector):
        return {}
