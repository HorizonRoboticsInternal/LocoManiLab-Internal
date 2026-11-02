"""Frame-convention tests for the AMO VR arm-teleop path.

Every bug this file guards against came from the same failure mode: two pieces
of code disagreeing about what a rotation matrix *means* — which frame it maps
from, which it maps to, and whether it composes on the left or the right. That
disagreement is invisible at runtime (nothing crashes, nothing is NaN, the
numbers stay unit-norm) and shows up only as a robot whose wrist sits at a
joint limit while the operator's hand is level.

So the assertions here are deliberately about *meaning*, not about plumbing:
they pin the axis conventions to literal expected vectors, they check
invariance to the things the operator should be free to vary (where they
stand, which way they face, what pose the robot was in), and they check the
final target against real G1 forward kinematics rather than against another
copy of the same algebra.

Run: ``python tests/test_amo_vr_frames.py``  (no sim, no PICO, no GPU)
"""

import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation as sRot

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pinocchio as pin  # noqa: E402
from g1_bridge.vendored.amo.g1_arm_ik import (  # noqa: E402
    _URDF,
    G1ArmIK,
)
from g1_bridge.pico.vr_source import (  # noqa: E402
    _Q_UNITY_TO_ROBOT,
    _Q_YUP2ZUP,
    head_relative_pose,
    pose7_to_robot,
    pose_is_valid,
    robot_pose7_to_mat,
)


def _rz(deg):
    return sRot.from_euler("z", deg, degrees=True).as_matrix()


def _sdk_pose(pos, rot):
    """Build an SDK-layout pose7 (scalar-LAST) from robot-agnostic A-frame
    position + rotation matrix."""
    return np.concatenate(
        [np.asarray(pos, float), sRot.from_matrix(rot).as_quat(scalar_first=False)]
    )


def _robot_pose7(pos, rot):
    """Build a robot-axes pose7 (scalar-FIRST) directly."""
    return np.concatenate(
        [np.asarray(pos, float), sRot.from_matrix(rot).as_quat(scalar_first=True)]
    )


def _ang(R):
    """Geodesic magnitude of a rotation, in radians."""
    return float(np.linalg.norm(sRot.from_matrix(R).as_rotvec()))


# --------------------------------------------------------------------------- #
def test_conventions_are_pinned():
    """The two Unity->robot matrices in this repo are the same convention off
    by a 90 deg yaw. Nothing enforces that but this test, and the SMPL path
    only works because its root keypoint offset supplies the missing yaw."""
    print("F1 convention lock")
    assert np.allclose(_rz(90) @ _Q_UNITY_TO_ROBOT, _Q_YUP2ZUP, atol=1e-12), (
        "Rz(90) @ _Q_UNITY_TO_ROBOT no longer equals _Q_YUP2ZUP -- the SMPL "
        "path and the direct-pose path have drifted apart"
    )
    # Psi0's grd_yup2grd_zup, verbatim (constants_vuer.py:17, upper-left 3x3).
    psi0 = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0.0]])
    assert np.allclose(_Q_YUP2ZUP, psi0, atol=1e-12)
    assert abs(np.linalg.det(_Q_YUP2ZUP) - 1.0) < 1e-12, "must be a proper rotation"
    assert np.allclose(_Q_YUP2ZUP @ _Q_YUP2ZUP.T, np.eye(3), atol=1e-12)
    print("PASS: frame matrices match Psi0 and each other")


def test_axis_mapping():
    """Pin the axis mapping to literal vectors. If the SDK ever changes
    handedness this is the test that catches it, not a confused operator."""
    print("F2 axis mapping")
    cases = [
        ("SDK forward (0,0,-1)", [0, 0, -1], [1, 0, 0]),
        ("SDK up      (0,1,0)", [0, 1, 0], [0, 0, 1]),
        ("SDK left    (-1,0,0)", [-1, 0, 0], [0, 1, 0]),
    ]
    for name, a, expect in cases:
        got = _Q_YUP2ZUP @ np.array(a, dtype=float)
        assert np.allclose(got, expect, atol=1e-12), f"{name}: got {got}, want {expect}"
        print(f"  ok  {name} -> robot {np.array(expect)}")

    # A pure position pose must convert without touching the rotation.
    out = pose7_to_robot(_sdk_pose([0, 1, -2], np.eye(3)))
    assert np.allclose(out[:3], [2, 0, 1], atol=1e-9), out[:3]
    assert np.allclose(out[3:], [1, 0, 0, 0], atol=1e-9), out[3:]

    # A rotation about SDK "up" must become a rotation about robot +Z, with the
    # same sign -- the similarity transform is what guarantees this.
    yaw_a = sRot.from_euler("y", 35, degrees=True).as_matrix()
    R = robot_pose7_to_mat(pose7_to_robot(_sdk_pose([0, 0, 0], yaw_a)))[:3, :3]
    rv = sRot.from_matrix(R).as_rotvec(degrees=True)
    assert np.allclose(rv, [0, 0, 35], atol=1e-9), rv
    print("PASS: SDK axes map to robot axes as documented")


def test_head_relative_is_pose_invariant():
    """The invariance the whole absolute design rests on: the operator may
    stand anywhere and face any way; the same body-relative gesture must give
    the same head-relative pose."""
    print("F4b operator-placement invariance")
    dev_in_head = np.eye(4)
    dev_in_head[:3, :3] = sRot.from_euler(
        "xyz", [10, -20, 30], degrees=True
    ).as_matrix()
    dev_in_head[:3, 3] = [0.25, 0.15, -0.40]

    rng = np.random.default_rng(0)
    ref = None
    for _ in range(20):
        R_head = sRot.random(random_state=rng.integers(1 << 30)).as_matrix()
        p_head = rng.uniform(-3, 3, 3)
        R_dev = R_head @ dev_in_head[:3, :3]
        p_dev = p_head + R_head @ dev_in_head[:3, 3]
        got = head_relative_pose(
            _robot_pose7(p_head, R_head), _robot_pose7(p_dev, R_dev)
        )
        if ref is None:
            ref = got
        assert np.allclose(
            got, ref, atol=1e-9
        ), "head-relative pose changed when only the operator's world placement did"
    assert np.allclose(ref, dev_in_head, atol=1e-9), "should recover the input exactly"
    print("PASS: head-relative pose is invariant to where the operator stands/faces")


def test_yaw_only_reference_drops_head_pitch():
    """`amo_vr_ori_ref="head_yaw"` exists so looking down does not rotate the
    arm targets. Check it actually does that, and that it still tracks yaw."""
    print("F4d yaw-only reference")
    R_dev = sRot.from_euler("z", 20, degrees=True).as_matrix()
    p_dev = np.array([0.3, 0.0, -0.2])
    base = None
    for pitch in (0, 15, -25):
        R_head = sRot.from_euler("y", pitch, degrees=True).as_matrix()
        got = head_relative_pose(
            _robot_pose7([0, 0, 1.5], R_head), _robot_pose7(p_dev, R_dev), yaw_only=True
        )
        if base is None:
            base = got
        assert np.allclose(got, base, atol=1e-9), "head pitch leaked through yaw_only"
    # ...but a real yaw must still be tracked.
    turned = head_relative_pose(
        _robot_pose7([0, 0, 1.5], _rz(40)), _robot_pose7(p_dev, R_dev), yaw_only=True
    )
    assert not np.allclose(turned, base, atol=1e-3), "yaw_only must still follow yaw"
    print("PASS: yaw_only drops head pitch/roll and keeps yaw")


def test_pose_validity_guard():
    """Every way the SDK says 'no data' must be caught before the IK sees it."""
    print("F5c validity guard")
    good = _robot_pose7([0.2, 0.1, -0.3], np.eye(3))
    assert pose_is_valid(good)
    assert not pose_is_valid(np.zeros(7)), "all-zero pose must be rejected"
    assert not pose_is_valid(
        np.concatenate([[0.1, 0.2, 0.3], np.zeros(4)])
    ), "zero quaternion must be rejected"
    bad = good.copy()
    bad[2] = np.nan
    assert not pose_is_valid(bad), "NaN must be rejected"
    bad = good.copy()
    bad[:3] = 0.0
    assert not pose_is_valid(bad), "exactly-zero position must be rejected"
    print("PASS: zero / NaN / zero-quat poses are all rejected")


def test_reference_pose_is_the_gesture_we_tell_the_operator():
    """The operator will be told to assume a specific physical gesture, and the
    calibration bakes whatever they do into M. If the instruction does not
    match q=0, every wrist target is offset by the difference -- which is the
    exact bug being fixed. So assert the gesture description against the URDF.
    """
    print("F5a reference pose == q=0")
    ik = G1ArmIK()
    m, d = ik.model, ik.data
    pin.framesForwardKinematics(m, d, np.zeros(ik.nq))

    for side, fid, sgn in (("left", ik.L_id, +1), ("right", ik.R_id, -1)):
        R = d.oMf[fid].rotation
        assert _ang(R) < 1e-3, f"{side} EE rotation at q=0 is not identity: {R}"

    # Upper arm hangs vertically; forearm points forward and level.
    sh = d.oMi[m.getJointId("left_shoulder_pitch_joint")].translation
    el = d.oMi[m.getJointId("left_elbow_joint")].translation
    wr = d.oMi[m.getJointId("left_wrist_yaw_joint")].translation
    upper, fore = el - sh, wr - el
    assert upper[2] < -0.15 and abs(upper[0]) < 0.05, f"upper arm not vertical: {upper}"
    assert fore[0] > 0.15 and abs(fore[2]) < 0.05, f"forearm not level+forward: {fore}"
    print(
        f"  ok  upper arm {np.round(upper, 3)} (down), forearm {np.round(fore, 3)} (fwd)"
    )

    # Palms face each other -- NOT palms down. The two gripper fingers are
    # stacked vertically and the thumb (the opposing digit, i.e. the palm
    # normal) points toward the midline on both sides.
    full = pin.buildModelFromUrdf(_URDF)
    fd = full.createData()
    pin.framesForwardKinematics(full, fd, np.zeros(full.nq))

    def _in_palm(side, link):
        palm = fd.oMf[full.getFrameId(f"{side}_hand_palm_link")]
        p = fd.oMf[full.getFrameId(f"{side}_hand_{link}_link")].translation
        return palm.rotation.T @ (p - palm.translation)

    for side, thumb_sign in (("left", -1), ("right", +1)):
        idx, mid = _in_palm(side, "index_1"), _in_palm(side, "middle_1")
        thumb = _in_palm(side, "thumb_2")
        assert idx[0] > 0.1 and mid[0] > 0.1, f"{side} fingers do not point forward"
        sep = idx - mid
        assert abs(sep[2]) > 0.9 * np.linalg.norm(sep), (
            f"{side} fingers are not stacked vertically: {sep} -- the reference "
            "gesture description ('palms facing each other') is wrong"
        )
        assert (
            np.sign(thumb[1]) == thumb_sign and abs(thumb[1]) > 0.05
        ), f"{side} thumb does not point toward the midline: {thumb}"
        print(f"  ok  {side}: fingers fwd, stacked vertically, thumb toward midline")
    print("PASS: q=0 is 'forearms forward, PALMS FACING EACH OTHER' as documented")


def test_absolute_target_matches_fk_at_the_reference_gesture():
    """End to end for the orientation channel: an operator who assumes the
    reference gesture must command exactly FK(q=0), and a known body-frame
    perturbation of their hand must transport through the remap correctly.

    This is the direct regression for the reported symptom -- with a target of
    FK(q=0) the IK solution has every wrist joint at zero, i.e. maximally far
    from the limits the wrist was previously pinned against."""
    print("F5b absolute target vs G1 FK")
    ik = G1ArmIK()
    pin.framesForwardKinematics(ik.model, ik.data, np.zeros(ik.nq))
    ee_ref = {
        s: ik.data.oMf[f].homogeneous.copy()
        for s, f in (("left", ik.L_id), ("right", ik.R_id))
    }

    # An arbitrary "true" controller->EE remap and an arbitrary operator stance.
    M_true = {
        "left": sRot.from_euler("xyz", [90, 0, 0], degrees=True).as_matrix(),
        "right": sRot.from_euler("xyz", [-90, 0, 0], degrees=True).as_matrix(),
    }
    R_head = sRot.from_euler("xyz", [4, -7, 25], degrees=True).as_matrix()
    p_head = np.array([1.3, -0.4, 1.62])

    # The controller pose a perfect operator in the reference gesture produces.
    def ctrl_pose(side, extra_body_rot=np.eye(3)):
        R_rel = ee_ref[side][:3, :3] @ M_true[side].T @ extra_body_rot
        p_rel = ee_ref[side][:3, 3] - np.array([0.05, 0.0, 0.55])
        return _robot_pose7(p_head + R_head @ p_rel, R_head @ R_rel)

    head7 = _robot_pose7(p_head, R_head)
    M_rec, d_rec = {}, {}
    for side in ("left", "right"):
        rel = head_relative_pose(head7, ctrl_pose(side))
        M_rec[side] = rel[:3, :3].T @ ee_ref[side][:3, :3]
        d_rec[side] = ee_ref[side][:3, 3] - rel[:3, 3]
        assert _ang(M_rec[side].T @ M_true[side]) < 1e-9, "M not recovered"

        # The commanded target at the gesture must BE FK(q=0).
        R_tgt = rel[:3, :3] @ M_rec[side]
        p_tgt = rel[:3, 3] + d_rec[side]
        assert _ang(R_tgt.T @ ee_ref[side][:3, :3]) < 1e-9, "orientation != FK(q=0)"
        assert np.allclose(p_tgt, ee_ref[side][:3, 3], atol=1e-9), "position != FK(q=0)"
        print(f"  ok  {side}: calibration gesture commands exactly FK(q=0)")

    # A known body-frame hand rotation must transport as M^T dR M.
    for side in ("left", "right"):
        dR = sRot.from_euler("y", 30, degrees=True).as_matrix()
        rel = head_relative_pose(head7, ctrl_pose(side, extra_body_rot=dR))
        R_tgt = rel[:3, :3] @ M_rec[side]
        want = ee_ref[side][:3, :3] @ (M_rec[side].T @ dR @ M_rec[side])
        assert _ang(R_tgt.T @ want) < 1e-9, f"{side} body-frame perturbation mis-mapped"
    print("  ok  body-frame hand rotations transport as M^T dR M")

    # And the whole thing is invariant to where the operator stood at capture.
    rng = np.random.default_rng(7)
    ref_target = None
    for _ in range(20):
        Rh = sRot.random(random_state=rng.integers(1 << 30)).as_matrix()
        ph = rng.uniform(-2, 2, 3)
        h7 = _robot_pose7(ph, Rh)
        rel0 = head_relative_pose(
            h7,
            _robot_pose7(
                ph + Rh @ (ee_ref["left"][:3, 3] - np.array([0.05, 0, 0.55])),
                Rh @ (ee_ref["left"][:3, :3] @ M_true["left"].T),
            ),
        )
        M = rel0[:3, :3].T @ ee_ref["left"][:3, :3]
        dR = sRot.from_euler("x", 30, degrees=True).as_matrix()
        rel = head_relative_pose(
            h7,
            _robot_pose7(
                ph + Rh @ (ee_ref["left"][:3, 3] - np.array([0.05, 0, 0.55])),
                Rh @ (ee_ref["left"][:3, :3] @ M_true["left"].T @ dR),
            ),
        )
        tgt = rel[:3, :3] @ M
        if ref_target is None:
            ref_target = tgt
        assert _ang(tgt.T @ ref_target) < 1e-9, (
            "commanded EE rotation depends on the operator's stance at calibration "
            "-- this is exactly the bug being fixed"
        )
    print("PASS: absolute target reproduces FK(q=0) and is stance-invariant")


def test_reference_gesture_target_is_far_from_wrist_limits():
    """The literal inverse of the reported symptom: solve the IK for the
    reference gesture and for +/-30 deg perturbations about each EE axis, and
    require every wrist joint to stay well clear of its limit."""
    print("F5d wrist headroom at the reference gesture")
    ik = G1ArmIK()
    pin.framesForwardKinematics(ik.model, ik.data, np.zeros(ik.nq))
    tL = ik.data.oMf[ik.L_id].homogeneous.copy()
    tR = ik.data.oMf[ik.R_id].homogeneous.copy()
    wrist_idx = [4, 5, 6, 11, 12, 13]

    worst = 1e9
    for axis in ("x", "y", "z"):
        for deg in (0, 30, -30):
            dR = sRot.from_euler(axis, deg, degrees=True).as_matrix()
            a, b = tL.copy(), tR.copy()
            a[:3, :3] = a[:3, :3] @ dR
            b[:3, :3] = b[:3, :3] @ dR
            ik.prime_filter(np.zeros(ik.nq))
            q, _ = ik.solve_ik(a, b, current_q=np.zeros(ik.nq))
            margin = float(
                np.min(
                    np.minimum(
                        q[wrist_idx] - ik.lower[wrist_idx],
                        ik.upper[wrist_idx] - q[wrist_idx],
                    )
                )
            )
            worst = min(worst, margin)
            assert margin > 0.05, (
                f"wrist within {margin:.3f} rad of a limit for a {deg} deg "
                f"rotation about {axis} -- this is the reported symptom"
            )
    print(f"PASS: worst wrist limit margin over all perturbations: {worst:.3f} rad")


def test_all_sources_emit_the_device_pose_schema():
    """Every PicoVRSource must carry the new keys, and the synthetic ones must
    carry a VALID gesture -- that is what makes `--pico-source mock` exercise
    this path instead of skipping it."""
    print("F8 source schema")
    from g1_bridge.pico.vr_source import (
        DEVICE_POSE_KEYS,
        KeyboardPicoVRSource,
        MockPicoVRSource,
        empty_device_poses,
    )

    def _check(name, sample, expect_valid):
        for k in DEVICE_POSE_KEYS:
            assert k in sample, f"{name} is missing {k}"
        for k in ("vr_head_pose", "vr_left_ctrl_pose", "vr_right_ctrl_pose"):
            v = np.asarray(sample[k])
            assert v.shape == (7,) and v.dtype == np.float32, f"{name}.{k}: {v.shape}"
            assert abs(np.linalg.norm(v[3:]) - 1.0) < 1e-5, f"{name}.{k} quat not unit"
        valid = np.asarray(sample["vr_pose_valid"])
        assert valid.shape == (3,) and valid.dtype == np.uint8
        assert np.array_equal(valid, expect_valid), f"{name}.vr_pose_valid = {valid}"
        print(f"  ok  {name}: schema + valid={valid}")

    for cls in (MockPicoVRSource, KeyboardPicoVRSource):
        src = cls()
        src.start()
        _check(cls.__name__, src.get_latest(), np.ones(3, dtype=np.uint8))
        src.stop()

    _check("empty_device_poses", empty_device_poses(), np.zeros(3, dtype=np.uint8))
    print("PASS: all sources carry the device-pose schema")


def test_mock_source_lands_on_the_reference_pose():
    """End to end through the mock: the synthetic gesture, the head-relative
    composition and the default remap must agree on FK(q=0). If any one of the
    three drifts, this fails -- which is the point of tying them to one set of
    constants."""
    print("F8b mock -> FK(q=0)")
    from g1_bridge.pico.vr_source import (
        MockPicoVRSource,
        REF_GESTURE_CTRL_ROT,
    )

    ik = G1ArmIK()
    pin.framesForwardKinematics(ik.model, ik.data, np.zeros(ik.nq))

    src = MockPicoVRSource()
    src.start()
    s = src.get_latest()
    src.stop()

    for side, fid, key in (
        ("left", ik.L_id, "vr_left_ctrl_pose"),
        ("right", ik.R_id, "vr_right_ctrl_pose"),
    ):
        rel = head_relative_pose(s["vr_head_pose"], s[key])
        # The default remap is the transpose of the gesture rotation, so the
        # commanded orientation at the gesture is exactly the EE reference.
        M = REF_GESTURE_CTRL_ROT[side].T
        R_tgt = rel[:3, :3] @ M
        # 1e-3 rad, not 1e-9: the EE frame at q=0 is identity only to ~2e-4 in
        # the URDF (its joint axes are not exactly axis-aligned). That is model
        # precision, not algebra error -- the exact-equality checks live in F5b,
        # which compares against FK rather than against literal identity.
        err = _ang(R_tgt.T @ ik.data.oMf[fid].rotation)
        assert (
            err < 1e-3
        ), f"{side}: mock gesture is {err:.2e} rad off the FK(q=0) orientation"
        print(f"  ok  {side}: mock gesture commands the q=0 EE orientation")
    print("PASS: mock source drives the absolute path to the reference pose")


class _Harness:
    """The real AmoBackend arm path, driven tick by tick against a lagging
    plant. Deliberately the real object rather than a reimplementation: the
    engage blend, the jump guard and the rate limiter interact, and the failure
    this guards against (the blend's step being vetoed by the guard, which then
    auto-disengages after 0.5 s) only exists in that interaction."""

    def __init__(self, tmpdir, **vr_overrides):
        from g1_bridge.g1_runtime import G1_JOINT_NAMES
        from g1_bridge.backends import make_backend
        from g1_bridge.backends.amo import (
            _AMO_DEFAULT_DOF_POS,
            _AMO_JOINT_NAMES,
        )

        os.environ["TELE_VR_ARM_CALIB"] = os.path.join(tmpdir, "vr_arm_calib.json")
        if os.path.exists(os.environ["TELE_VR_ARM_CALIB"]):
            os.remove(os.environ["TELE_VR_ARM_CALIB"])

        self.b = make_backend("amo")
        self.b._vr = dict(self.b._vr, **vr_overrides)
        self.b._build_amo_joint_mapping()
        self.b._amo_torso_control_mode = "joystick"
        self.b._init_amo_arm_ik()
        self.b._init_wrist_calibration()

        self.qpos = np.zeros(len(G1_JOINT_NAMES))
        for name, v in zip(_AMO_JOINT_NAMES, _AMO_DEFAULT_DOF_POS):
            self.qpos[G1_JOINT_NAMES.index(name)] = v
        self.t = 0.0

    def tick(self, device=None, **btn):
        from g1_bridge.pico.vr_source import (
            MockPicoVRSource,
            synthetic_device_poses,
        )

        if not hasattr(self, "_base"):
            src = MockPicoVRSource()
            src.start()
            self._base = src.get_latest()
            src.stop()
        s = dict(self._base)
        s.update(device if device is not None else synthetic_device_poses())
        s["buttons"] = dict({"left_axis": (0, 0), "right_axis": (0, 0)}, **btn)
        prev = (
            None
            if self.b._amo_prev_cmd_q14 is None
            else self.b._amo_prev_cmd_q14.copy()
        )
        self.b._tick_amo_arms(self.qpos, s["buttons"], s, self.t)
        self.t += 0.02
        for i, j in enumerate(self.b._amo_arm_sim_indices):
            self.qpos[j] += 0.35 * (self.b._amo_arm_action[i] - self.qpos[j])
        for i, j in enumerate(self.b._amo_wrist_sim_indices):
            self.qpos[j] += 0.35 * (self.b._amo_wrist_targets[i] - self.qpos[j])
        if prev is None or self.b._amo_prev_cmd_q14 is None:
            return 0.0
        return float(np.max(np.abs(self.b._amo_prev_cmd_q14 - prev)))

    def calibrate(self):
        for _ in range(60):
            self.tick(left_menu=True, right_menu=True)
        self.tick()  # release
        return self.b._calib.source != "default"

    def engage(self):
        self.tick(right_axis_click=True)
        return self.b._amo_arm_engaged


def test_engage_blend_never_trips_the_jump_guard():
    """The regression for the interaction that would otherwise make absolute
    mode unusable: a fixed remap means the first absolute target is not the
    arm's current pose, and an unramped step there is vetoed by the jump guard
    25 times running, which auto-disengages the operator half a second in."""
    print("F6 engage blend")
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        h = _Harness(tmp)
        assert h.calibrate(), "calibration did not capture"
        assert h.engage(), "engage did not latch"
        deltas = [h.tick() for _ in range(120)]  # ride the 1.5 s blend and settle
        d = np.array(deltas)
        budget = 3.0 * 0.02  # _AMO_ARM_VEL_LIMIT * _POLICY_DT
        assert h.b._amo_arm_engaged, "auto-disengaged during the engage blend"
        assert (
            d.max() <= budget + 1e-9
        ), f"engage produced {d.max():.4f} rad in one tick, budget {budget:.4f}"
        assert int((d > 0.1).sum()) == 0, "twitch during engage"
        # The blend, not the limiter, should be doing the work.
        assert d.max() < 0.9 * budget, (
            f"the rate limiter is absorbing the engage step ({d.max():.4f} vs "
            f"{budget:.4f}) -- the blend is not sized correctly"
        )
        q14 = h.b._read_amo_arm_q14(h.qpos)
        wrist = np.abs(q14[[4, 5, 6, 11, 12, 13]])
        print(
            f"  ok  max dq {d.max():.4f} rad/tick, wrists settle at {wrist.max():.3f} rad"
        )
        assert wrist.max() < 0.3, (
            f"wrists settled at {wrist.max():.3f} rad from zero for the reference "
            "gesture -- they should be near the middle of their range"
        )
    print("PASS: engage ramps smoothly and lands with the wrists near neutral")


def test_pose_dropout_freezes_then_disengages():
    """A lost controller must freeze the arm (not drift, not slam), survive a
    short outage, and hand control back with a named reason after a long one."""
    print("F7 pose dropout")
    import tempfile

    from g1_bridge.pico.vr_source import empty_device_poses

    with tempfile.TemporaryDirectory() as tmp:
        h = _Harness(tmp)
        assert h.calibrate() and h.engage()
        for _ in range(120):
            h.tick()

        invalid = empty_device_poses()
        frozen = h.b._amo_prev_cmd_q14.copy()
        for _ in range(10):  # 0.2 s < amo_vr_pose_hold_s
            h.tick(device=invalid)
        assert h.b._amo_arm_engaged, "a 0.2 s outage should not disengage"
        drift = float(np.max(np.abs(h.b._amo_prev_cmd_q14 - frozen)))
        assert drift < 0.02, f"target drifted {drift:.4f} rad during the outage"
        print(f"  ok  short outage: held (drift {drift:.5f} rad), still engaged")

        for _ in range(40):  # push past the hold timeout
            h.tick(device=invalid)
        assert not h.b._amo_arm_engaged, "a long outage must disengage"
        print("  ok  long outage: auto-disengaged")
    print("PASS: dropout freezes the target, then releases the operator")


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    tests = [
        test_conventions_are_pinned,
        test_axis_mapping,
        test_head_relative_is_pose_invariant,
        test_yaw_only_reference_drops_head_pitch,
        test_pose_validity_guard,
        test_reference_pose_is_the_gesture_we_tell_the_operator,
        test_absolute_target_matches_fk_at_the_reference_gesture,
        test_reference_gesture_target_is_far_from_wrist_limits,
        test_all_sources_emit_the_device_pose_schema,
        test_mock_source_lands_on_the_reference_pose,
        test_engage_blend_never_trips_the_jump_guard,
        test_pose_dropout_freezes_then_disengages,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}")
        print()
    if failed:
        print(f"{failed}/{len(tests)} checks FAILED")
        sys.exit(1)
    print("PASS: all AMO VR frame-convention checks")
