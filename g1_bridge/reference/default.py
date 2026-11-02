"""The reference frame the composite holds while disengaged, built the way
sim2real's own publisher builds its paused frame.

In a real sim2real session ``pico_retarget_pub`` is the single reference source
and it publishes continuously, including while the operator has it paused. The
paused frame is not "wherever the robot happens to be" -- it is
``_capture_paused_qpos()`` -> ``_pose_arrays_from_qpos()``:

    qpos      = RobotCfg.default_qpos            (a real standing pose)
    qpos[:2]  = the reference's current xy       (stay where you are)
    qpos[3:7] = yaw_quat(current root quat)      (keep heading, drop roll/pitch)
    -> mujoco.mj_forward on RobotCfg's own MJCF -> body_pos_w / body_quat_w

with one deliberate deviation: the joint angles come from the *running policy's*
``default_dof_angles`` when the caller supplies them, not from
``RobotCfg.default_qpos``. See :meth:`DefaultReferenceFrame.standing_joints`.

This module reproduces that, so the bridge's disengaged reference is the same
object the real stack would be publishing. Three things fall out of using the
same recipe rather than an approximation of it:

* **It is a standing pose.** The previous synthetic frame echoed the robot's own
  live body poses, so whatever sag the policy had drifted into got frozen in as
  the thing to hold, and the robot had no reason to stand back up.
* **The layout matches the live stream exactly** -- same MJCF, so the same 33
  bodies led by ``world``, and the same joint order. A layout change makes
  ``RealtimeMotionBuffer._set_layout_locked`` drop its last transformed frame and
  reset the segment transform, which silently turns the ``motion_first_frame``
  stitch into a no-op precisely at the OFF->live handover that needs it.
* **Only yaw is inherited**, so a reference pose is never tilted by the robot's
  own roll/pitch -- which is what makes it a target worth tracking rather than a
  mirror of the current error.
"""

from __future__ import annotations

import mujoco
import numpy as np

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from sim2real.utils.math import yaw_quat


class DefaultReferenceFrame:
    """Forward kinematics of ``RobotCfg.default_qpos`` at a given xy + heading.

    One MJCF load and one ``MjData`` per robot, cached process-wide by
    :func:`default_reference_frame` -- ``payload()`` runs at policy rate.
    """

    def __init__(self, robot_cfg):
        self._model = mujoco.MjModel.from_xml_path(str(robot_cfg.resolve_mjcf_path()))
        self._data = mujoco.MjData(self._model)
        self._default_qpos = np.asarray(robot_cfg.default_qpos, dtype=np.float64).copy()
        if self._default_qpos.shape[0] != self._model.nq:
            raise ValueError(
                f"RobotCfg.default_qpos has {self._default_qpos.shape[0]} entries but "
                f"{robot_cfg.resolve_mjcf_path()} has nq={self._model.nq}"
            )

        # Joint order is the RobotCfg's, matching what the publisher reports and
        # what StateProcessor/the observations index by.
        self.joint_names = list(robot_cfg.joint_names)
        self._joint_qpos_indices = []
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Failed to resolve joint name in MJCF: {name}")
            self._joint_qpos_indices.append(int(self._model.jnt_qposadr[joint_id]))

        # The flat-qpos convention every consumer here shares: a 7-DoF free
        # root followed by the RobotCfg joints in order. The publisher's own
        # "qpos" key uses it (RealtimeMotionBuffer auto-slices a 36-long qpos
        # as [7:7+njoints]), and so does the blend in reference_blend.py. If a
        # future MJCF broke it, payload_from_qpos would silently reorder the
        # joints instead of failing, so pin it here.
        expected = list(range(7, 7 + len(self.joint_names)))
        if self._joint_qpos_indices != expected:
            raise ValueError(
                f"{robot_cfg.resolve_mjcf_path()} does not use the "
                f"free-root-then-joints qpos layout this module assumes: "
                f"joint qpos addresses are {self._joint_qpos_indices}, "
                f"expected {expected}"
            )

        # Every named body in the model, in model order -- the publisher's
        # _resolve_publish_body_names does exactly this, which is why its payload
        # leads with `world`.
        self.body_names = []
        self._body_ids = []
        for body_id in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if name:
                self.body_names.append(str(name))
                self._body_ids.append(int(body_id))

    @property
    def default_root_height(self) -> float:
        """Root z of the robot cfg's standing pose -- the height to stand back
        up to when a reference has drifted down."""
        return float(self._default_qpos[2])

    def standing_joints(self, joint_pos=None) -> np.ndarray:
        """The 29 joint angles to stand in.

        ``joint_pos`` should be the running policy's own ``default_dof_angles``
        -- the pose it is centred on, since it commands
        ``q_target = default_dof_angles + action * action_scale``. Handing it
        ``RobotCfg.default_qpos`` instead (which is what this module used to do
        unconditionally) asks the policy to hold a pose it does not stand in:
        for the sonic checkpoint that is knees 15 deg straighter, hips 6 deg
        straighter and elbows 34 deg more flexed than its own default, and
        near-straight knees is exactly where a G1's balance authority runs out
        -- it reads as a persistent sway in place. Every policy family differs
        here (sonic knee 0.669, heft 0.5, RobotCfg 0.4), so there is no single
        constant that would work.

        Falls back to the robot cfg's own default when the caller has nothing
        better, which keeps this usable before a policy exists.
        """
        if joint_pos is None:
            return self._default_qpos[7:].copy()
        joints = np.asarray(joint_pos, dtype=np.float64).reshape(-1)
        expected = len(self.joint_names)
        if joints.shape[0] != expected:
            raise ValueError(
                f"standing joint_pos has {joints.shape[0]} entries, expected "
                f"{expected} (one per RobotCfg joint, same order)"
            )
        return joints

    def default_qpos_at(self, root_pos, root_quat_wxyz, joint_pos=None) -> np.ndarray:
        """The standing pose as a flat qpos, at ``root_pos``'s xy and
        ``root_quat_wxyz``'s yaw.

        The blend anchor for both ends of a session: what an episode's first
        POSE entry ramps *from*, and what the OFF chord ramps *back to*. Same
        recipe (and therefore the same pose) as :meth:`payload`, which is what
        the settling window has been pushing all along.
        """
        qpos = self._default_qpos.copy()
        qpos[:2] = np.asarray(root_pos, dtype=np.float64).reshape(-1)[:2]
        qpos[3:7] = np.asarray(
            yaw_quat(np.asarray(root_quat_wxyz, dtype=np.float32).reshape(1, 4)),
            dtype=np.float64,
        ).reshape(4)
        qpos[7:] = self.standing_joints(joint_pos)
        return qpos

    def payload(self, root_pos, root_quat_wxyz, joint_pos=None) -> dict:
        """A push_payload()-shaped default frame standing at ``root_pos``'s xy,
        facing ``root_quat_wxyz``'s yaw."""
        return self.payload_from_qpos(
            self.default_qpos_at(root_pos, root_quat_wxyz, joint_pos)
        )

    def payload_from_qpos(self, qpos) -> dict:
        """FK an arbitrary qpos into the same push_payload()-shaped frame.

        This is what makes an interpolated reference safe to push: the live GMR
        stream's ``body_pos_w``/``body_quat_w`` are themselves pure FK of its
        published qpos (``pico_retarget_pub._pose_arrays_from_qpos`` is their
        only producer, on every branch), so a blend done in qpos space and
        re-FK'd here lands on exactly the live arrays when the blend reaches
        its endpoint -- no seam at the hand-off. Lerping the 33 body positions
        directly would instead foreshorten limbs mid-blend, worst precisely
        when the two poses differ most.
        """
        qpos = np.asarray(qpos, dtype=np.float64).reshape(-1)
        if qpos.shape[0] != self._model.nq:
            raise ValueError(
                f"qpos has {qpos.shape[0]} entries but the MJCF has "
                f"nq={self._model.nq}"
            )
        self._data.qpos[:] = qpos
        mujoco.mj_forward(self._model, self._data)
        return {
            "joint_pos": np.asarray(
                self._data.qpos[self._joint_qpos_indices], dtype=np.float32
            ).tolist(),
            "body_pos_w": np.asarray(
                self._data.xpos[self._body_ids], dtype=np.float32
            ).tolist(),
            "body_quat_w": np.asarray(
                self._data.xquat[self._body_ids], dtype=np.float32
            ).tolist(),
            "joint_names": list(self.joint_names),
            "body_names": list(self.body_names),
        }


_CACHE: dict[str, DefaultReferenceFrame] = {}


def default_reference_frame(robot_cfg) -> DefaultReferenceFrame:
    """Cached per robot: the MJCF load is far too expensive for a 50 Hz tick,
    and robosuite rebuilds the composite on every ``env.reset()``."""
    key = str(getattr(robot_cfg, "name", "")) or str(robot_cfg.resolve_mjcf_path())
    if key not in _CACHE:
        _CACHE[key] = DefaultReferenceFrame(robot_cfg)
    return _CACHE[key]
