"""VR_3PT: the mode numbering, the button chords, and the mode_id=1 encoder.

Three things fail silently here and so are pinned:

* ``ManagerMode``'s VALUES. They are written verbatim into
  ``teleop.stream_mode``; until 2026-08-26 ours put VR_3PT at 4, which is
  upstream's POSE_PAUSE, so the same number meant different things in our
  datasets and NVlabs'. Nothing about the column says which wrote it.
* Which 12 of the 29 IsaacLab joints the teleop block calls "legs". Picking up
  shoulder/waist joints instead yields a perfectly well-formed encoder input
  that simply does not walk.
* Which blocks stay zero. An all-zero block is a valid input the graph will
  happily turn into a token.
"""

from __future__ import annotations

import unittest

import numpy as np

import g1_bridge._cuda_order  # noqa: F401 — MUST precede any sim2real import
from g1_bridge.pico.mode_manager import CalibrationEvent, ManagerMode, ModeManager
from g1_bridge.sonic_utils.encoder_obs import (
    ENCODER_DIM,
    ENCODER_DIM_V1_1,
    MODE_TELEOP,
    sonic_g1_low_latency_encoder_input,
    sonic_g1_v1_1_encoder_input,
)
from test_sonic_v1_1_encoder import (
    ISAAC_JOINT_NAMES,
    N_FRAMES,
    build,
    make_env,
    quat_from_euler,
)

#: gear_sonic/scripts/pico_manager_thread_server.py:126-132, copied out by
#: hand. Deriving it from our own enum would pass whatever we happened to say.
UPSTREAM_STREAM_MODE = {
    "OFF": 0,
    "POSE": 1,
    "PLANNER": 2,
    "PLANNER_FROZEN_UPPER_BODY": 3,
    "POSE_PAUSE": 4,
    "PLANNER_VR_3PT": 5,
}

#: The teleop block's slots, longhand from the layout tables.
LOWER_POS = (650, 770)
LOWER_VEL = (770, 890)
VR_POS = (890, 899)
VR_ORN = (899, 911)
ANCHOR_CUR = (644, 650)


def _buttons(**kw):
    out = {k: False for k in ("a", "b", "x", "y", "left_axis_click", "left_menu")}
    out.update(kw)
    return out


class ModeNumberingTest(unittest.TestCase):
    def test_values_match_upstream_stream_mode(self) -> None:
        ours = {
            "OFF": ManagerMode.OFF,
            "POSE": ManagerMode.POSE,
            "PLANNER": ManagerMode.PLANNER,
            "PLANNER_FROZEN_UPPER_BODY": ManagerMode.PLANNER_FROZEN_UPPER,
            "POSE_PAUSE": ManagerMode.POSE_PAUSE,
            "PLANNER_VR_3PT": ManagerMode.PLANNER_VR_3PT,
        }
        for name, want in UPSTREAM_STREAM_MODE.items():
            with self.subTest(mode=name):
                self.assertEqual(int(ours[name]), want)

    def test_legacy_spelling_is_an_alias_that_moved(self) -> None:
        # Callers that still say VR_3PT (tele's wbt.py) must land on 5, not on
        # the 4 the name used to mean.
        self.assertIs(ManagerMode.VR_3PT, ManagerMode.PLANNER_VR_3PT)
        self.assertEqual(int(ManagerMode.VR_3PT), 5)


class PosePauseTest(unittest.TestCase):
    def setUp(self) -> None:
        self.mm = ModeManager(initial_mode=ManagerMode.POSE, verbose=False)

    def test_held_menu_button_pauses_and_release_resumes(self) -> None:
        mode, _ = self.mm.tick(_buttons(left_menu=True))
        self.assertIs(mode, ManagerMode.POSE_PAUSE)
        # Still held -> stays paused. A rising-edge implementation would fall
        # straight back to POSE here.
        mode, _ = self.mm.tick(_buttons(left_menu=True))
        self.assertIs(mode, ManagerMode.POSE_PAUSE)
        mode, _ = self.mm.tick(_buttons())
        self.assertIs(mode, ManagerMode.POSE)

    def test_emergency_stop_still_wins_while_paused(self) -> None:
        self.mm.tick(_buttons(left_menu=True))
        mode, _ = self.mm.tick(
            _buttons(a=True, b=True, x=True, y=True, left_menu=True)
        )
        self.assertIs(mode, ManagerMode.OFF)

    def test_tracking_profile_has_no_pause(self) -> None:
        # heft/scalebfm listen to start+ax only; POSE there means "tracking
        # your body", and a menu press must not silently freeze it.
        mm = ModeManager(
            initial_mode=ManagerMode.POSE, verbose=False, profile="tracking"
        )
        mode, _ = mm.tick(_buttons(left_menu=True))
        self.assertIs(mode, ManagerMode.POSE)


class Vr3ptTransitionTest(unittest.TestCase):
    def test_l_click_enters_from_either_parent_and_returns_there(self) -> None:
        for parent in (ManagerMode.PLANNER, ManagerMode.PLANNER_FROZEN_UPPER):
            with self.subTest(parent=parent.name):
                mm = ModeManager(initial_mode=parent, verbose=False)
                mode, event = mm.tick(_buttons(left_axis_click=True))
                self.assertIs(mode, ManagerMode.PLANNER_VR_3PT)
                self.assertIs(event, CalibrationEvent.CALIB_WRISTS)
                mm.tick(_buttons())  # release, so the next click is a new edge
                mode, _ = mm.tick(_buttons(left_axis_click=True))
                self.assertIs(mode, parent)

    def test_both_chain_exits_go_to_pose(self) -> None:
        for chord in ({"a": True, "x": True}, {"b": True, "y": True}):
            with self.subTest(chord=sorted(chord)):
                mm = ModeManager(
                    initial_mode=ManagerMode.PLANNER_VR_3PT, verbose=False
                )
                mode, _ = mm.tick(_buttons(**chord))
                self.assertIs(mode, ManagerMode.POSE)


class TeleopEncoderTest(unittest.TestCase):
    """The mode_id=1 encoder input, on both shipping checkpoint variants."""

    def setUp(self) -> None:
        self.env = make_env(quat_from_euler(0.1, -0.05, 0.7))
        self.cases = (
            (sonic_g1_low_latency_encoder_input, ENCODER_DIM),
            (sonic_g1_v1_1_encoder_input, ENCODER_DIM_V1_1),
        )
        self.vr_pos = np.arange(9, dtype=np.float32) + 1.0
        self.vr_orn = np.arange(12, dtype=np.float32) + 100.0

    def _teleop_obs(self, cls):
        obs = build(cls, self.env)
        obs.set_vr_3point(self.vr_pos, self.vr_orn)
        obs.set_encoder_mode("teleop")
        return obs, obs.compute()

    def test_leg_indices_really_are_the_legs(self) -> None:
        """Independent of the table itself: name every joint the block selects
        and require the documented MuJoCo leg order."""
        from g1_bridge.sonic_utils.encoder_obs import _LEG_IDX

        got = [ISAAC_JOINT_NAMES[i] for i in _LEG_IDX]
        want = [
            "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
            "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
            "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
            "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        ]
        self.assertEqual(got, want)

    def test_mode_id_and_slot_contents(self) -> None:
        from g1_bridge.sonic_utils.encoder_obs import _LEG_IDX

        for cls, dim in self.cases:
            with self.subTest(cls=cls.__name__):
                obs, out = self._teleop_obs(cls)
                self.assertEqual(out.shape, (dim,))
                self.assertEqual(out[0], MODE_TELEOP)

                pos = obs._motion._select(obs._motion.ref_joint_pos_future)[0]
                vel = obs._motion._select(obs._motion.ref_joint_vel_future)[0]
                np.testing.assert_allclose(
                    out[slice(*LOWER_POS)],
                    pos[:N_FRAMES, _LEG_IDX].reshape(-1),
                    atol=0,
                )
                np.testing.assert_allclose(
                    out[slice(*LOWER_VEL)],
                    vel[:N_FRAMES, _LEG_IDX].reshape(-1),
                    atol=0,
                )
                np.testing.assert_allclose(out[slice(*VR_POS)], self.vr_pos)
                np.testing.assert_allclose(out[slice(*VR_ORN)], self.vr_orn)
                # The anchor comes from the same object g1 mode uses, so v1_1
                # gets its heading frame and low_latency its body frame for
                # free -- compare against that mode's own output.
                obs.set_encoder_mode("g1")
                np.testing.assert_allclose(
                    out[slice(*ANCHOR_CUR)], obs.compute()[slice(*ANCHOR_CUR)], atol=0
                )

    def test_unused_blocks_stay_zero(self) -> None:
        for cls, dim in self.cases:
            with self.subTest(cls=cls.__name__):
                _obs, out = self._teleop_obs(cls)
                # g1's own blocks (everything between the mode id and the
                # current-frame anchor) and every smpl block past 911.
                np.testing.assert_array_equal(out[4 : ANCHOR_CUR[0]], 0)
                np.testing.assert_array_equal(out[VR_ORN[1] : dim], 0)

    def test_quats_default_to_identity_not_zero(self) -> None:
        """Before any PICO sample arrives. An all-zero quaternion is not a
        rotation and the encoder never saw one in training."""
        for cls, _dim in self.cases:
            with self.subTest(cls=cls.__name__):
                obs = build(cls, self.env)
                obs.set_encoder_mode("teleop")
                orn = obs.compute()[slice(*VR_ORN)].reshape(3, 4)
                np.testing.assert_allclose(orn, np.tile([1.0, 0, 0, 0], (3, 1)))


if __name__ == "__main__":
    unittest.main()
