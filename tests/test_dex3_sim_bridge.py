from __future__ import annotations

import mujoco
import numpy as np
import pytest
import zmq

from sim2real.sim_env.utils.bridge import Dex3SimHandBridge
from sim2real.config.robots import get_robot_cfg
from sim2real.utils.common import HandGripMessage
from sim2real.utils.dex3 import DEX3_JOINT_NAMES, grip_to_joint_targets

_HANDFUL_MJCF = get_robot_cfg("g1").mjcf_path_with_hands
pytestmark = pytest.mark.skipif(
    _HANDFUL_MJCF is None
    or not get_robot_cfg("g1").resolve_mjcf_path(with_hands=True).is_file(),
    reason="hand-ful G1 MJCF (RobotCfg.mjcf_path_with_hands) not present",
)

_STUB_RUBBER_HAND_VERTS = list(
    (np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64) * 0.01).flatten()
)

_STUB_XML = """
<mujoco model="stub_g1">
  <compiler angle="radian"/>
  <worldbody>
    <body name="pelvis">
      <freejoint/>
      <geom type="box" size="0.1 0.1 0.1"/>
      <body name="left_wrist_yaw_link" pos="0.5 0.2 0.3">
        <joint name="left_wrist_yaw_joint" axis="0 0 1" range="-1.61443 1.61443"/>
        <geom type="box" size="0.02 0.02 0.02"/>
        <geom name="left_rubber_hand_geom" pos="0.0415 0.003 0" quat="1 0 0 0"
              type="mesh" mesh="left_rubber_hand" contype="0" conaffinity="0"/>
      </body>
      <body name="right_wrist_yaw_link" pos="0.5 -0.2 0.3">
        <joint name="right_wrist_yaw_joint" axis="0 0 1" range="-1.61443 1.61443"/>
        <geom type="box" size="0.02 0.02 0.02"/>
        <geom name="right_rubber_hand_geom" pos="0.0415 -0.003 0" quat="1 0 0 0"
              type="mesh" mesh="right_rubber_hand" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _build_stub_model_with_hands() -> mujoco.MjModel:
    """The real hand-ful G1 model, which already carries the 14 Dex3 joints.

    Used to be the hand-less stub plus a runtime splice; the splice is gone
    because the hands are authored into the model on disk.
    """
    return mujoco.MjModel.from_xml_path(
        str(get_robot_cfg("g1").resolve_mjcf_path(with_hands=True))
    )


def test_bridge_raises_without_hand_joints_in_model() -> None:
    spec = mujoco.MjSpec.from_string(_STUB_XML)
    for name in ("left_rubber_hand", "right_rubber_hand"):
        spec.add_mesh(name=name, uservert=_STUB_RUBBER_HAND_VERTS)
    model = spec.compile()
    with pytest.raises(ValueError, match="No Dex3 hand joints"):
        Dex3SimHandBridge(model, hand_grip_connect="tcp://127.0.0.1:0")


def test_bridge_resolves_both_hands_and_applies_torques_from_grip_stream() -> None:
    model = _build_stub_model_with_hands()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")

    bridge = Dex3SimHandBridge(model, hand_grip_connect=f"tcp://127.0.0.1:{port}")
    try:
        assert set(bridge.act_adrs) == {"left", "right"}
        assert len(bridge.act_adrs["left"]) == len(DEX3_JOINT_NAMES)

        torques = np.zeros(model.nu)
        message = HandGripMessage(timestamp_ns=1, left_grip=1.0, right_grip=0.0)
        for _ in range(200):
            pub.send(message.to_bytes())
            bridge.apply_torques(data, torques)
            if bridge.last_grips["left"] == 1.0:
                break

        assert bridge.last_grips == {"left": 1.0, "right": 0.0}
        # left hand should be commanded away from its at-rest pose; right hand
        # (grip=0) should stay at zero torque since target == measured == 0.
        assert np.any(torques[bridge.act_adrs["left"]] != 0.0)
        np.testing.assert_allclose(torques[bridge.act_adrs["right"]], 0.0)
    finally:
        bridge.close()
        pub.close(0)


def test_bridge_default_gains_reach_close_pose_within_one_second() -> None:
    """Regression test: the real-hardware gains (kp=1.5) only crawl toward the
    close pose under the reference asset's actuator ctrlrange cap (+/-0.7 Nm)
    -- still short of a -1.5 rad target after a full second, and not fully
    converged even after ten. Dex3SimHandBridge's own default gains
    must be strong enough (given that same cap) to get there quickly.
    """
    model = _build_stub_model_with_hands()
    data = mujoco.MjData(model)
    model.opt.timestep = 0.005
    mujoco.mj_forward(model, data)

    joint_names_mujoco = [model.joint(i).name for i in range(model.njnt)]
    qpos_adrs = [
        model.jnt_qposadr[joint_names_mujoco.index(f"left_hand_{n}_joint")]
        for n in DEX3_JOINT_NAMES
    ]

    ctx = zmq.Context.instance()
    pub = ctx.socket(zmq.PUB)
    port = pub.bind_to_random_port("tcp://127.0.0.1")
    bridge = Dex3SimHandBridge(model, hand_grip_connect=f"tcp://127.0.0.1:{port}")
    try:
        message = HandGripMessage(timestamp_ns=1, left_grip=1.0, right_grip=0.0)
        torques = np.zeros(model.nu)
        for _ in range(200):  # 1 simulated second
            pub.send(message.to_bytes())
            bridge.apply_torques(data, torques)
            data.ctrl[:] = torques
            mujoco.mj_step(model, data)

        target = grip_to_joint_targets(1.0, "left")
        np.testing.assert_allclose(data.qpos[qpos_adrs], target, atol=0.1)
    finally:
        bridge.close()
        pub.close(0)
