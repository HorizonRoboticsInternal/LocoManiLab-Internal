# VR frame conventions

Two separate bugs in the AMO arm teleop came from the same root: code
disagreeing about what a rotation matrix *meant* — which frame it maps from,
which it maps to, and whether it composes on the left or the right. Nothing
crashes when that goes wrong. The quaternions stay unit-norm, the numbers stay
finite, and the only symptom is a robot whose wrist sits at a joint limit while
the operator's hand is level.

So: every transform in the VR path is listed here with its from-frame, its
to-frame, and its multiplication side. If you add one, add a row.

## Frames

| symbol | definition |
|---|---|
| `A` | SDK / OpenXR world. Right-handed, **+Y up, −Z forward, +X right**. What every `xrt.get_*_pose()` returns. (The "Unity, left-handed" comment in `pico_vr_source.py` is a misnomer — the conversion matrix has det = +1.) |
| `R` | Robot. Right-handed, **+X forward, +Y left, +Z up**. The `pelvis` frame of `g1_body29_hand14.urdf`. |
| `H` | Operator head, or its yaw-only projection (`amo_vr_ori_ref`). |
| `D` | Device (controller) body frame **re-expressed in robot axes** — i.e. `Q·(grip frame)`, not the raw grip frame. |
| `E` | Robot end-effector, `L_ee` / `R_ee` = `{left,right}_wrist_yaw_joint` + `SE3(I, [0.05, 0, 0])`. |

## Left vs right multiplication

- **Left** multiply = change of **reference** frame ("in which frame is this rotation expressed").
- **Right** multiply = change of **body** frame ("which physical axes of the object am I naming").

Getting these backwards is the single most common way to produce a mapping that
looks plausible, tracks small motions, and is wrong in a way that depends on
something invisible (the operator's stance at calibration).

## The transforms

| # | transform | from → to | side | where |
|---|---|---|---|---|
| 1 | `_Q_YUP2ZUP` (`Q`) | A-coords → R-coords | — | `pico_vr_source.py` |
| 2 | `p_R = Q p_A` | position | — | `pose7_to_robot` |
| 3 | `R_R = Q R_A Qᵀ` | A→A rotation → R→R rotation | **similarity** | `pose7_to_robot` |
| 4 | `R_rel = R_headᵀ R_dev` | reference A → H | **left** | `head_relative_pose` |
| 5 | `p_rel = R_headᵀ (p_dev − p_head)` | position, A → H | — | `head_relative_pose` |
| 6 | `M` (per hand) | body E → D | **right** | `_WristCalibration.target` |
| 7 | `R_target = R_rel M` | → EE orientation in R | — | `_WristCalibration.target` |
| 8 | `p_target = k·p_rel + d` | → EE position in R | — | `_WristCalibration.target` |

Row 3 is a **similarity transform, not a single multiply**. That is what makes
the result a rotation between two frames that have both been re-expressed in
robot axes, which is why the device body frame downstream is `D = Q·(grip)` and
not the raw grip frame. `M` is measured against exactly that quantity, so the
two are consistent by construction. Collapsing row 3 to `Q R_A` breaks the
calibration silently.

## Two conversion matrices, and why

```
_Q_UNITY_TO_ROBOT = [[-1,0,0],[0,0,1],[0,1,0]]     # SMPL path only
_Q_YUP2ZUP        = [[0,0,-1],[-1,0,0],[0,1,0]]    # direct-pose path
Rz(+90°) @ _Q_UNITY_TO_ROBOT == _Q_YUP2ZUP         # asserted in tests
```

They are the same convention off by a 90° yaw. The SMPL path's missing yaw is
supplied by `_KEYPOINT_ROTATION_OFFSETS[0] = Rz(−90°)` on the root keypoint,
which cancels during the root-relative step — so `_Q_UNITY_TO_ROBOT` is only
correct *bundled with those offsets*. Using it on a headset or controller pose
yaws everything 90°. `tests/test_amo_vr_frames.py::test_conventions_are_pinned`
asserts the identity so the two cannot drift apart.

## The two kinds of VR data are not interchangeable

| | `vr_3point_local_target` / `_orn_target` | `vr_head_pose` / `vr_*_ctrl_pose` |
|---|---|---|
| source | PICO SMPL body-tracking fit | headset + controller 6-DoF |
| frame | relative to the operator's pose at `ThreePointPose` calibration | metric, robot axes |
| anchored to a robot link? | **no** (`_capture_calibration` references identity, with a live `TODO(phase2): use real G1 FK`) | yes, via the wrist calibration |
| use for | SONIC's policy encoder, VR_3PT mode, datasets | kinematics / IK targets |

Feeding the 3-point set to an IK solver is what produced both bugs. Measured:
the same physical 30° hand rotation commanded `[-30,0,0]`, `[-15,0,-25.98]` or
`[-24.57,-17.21,0]` (rotvec, degrees) depending only on the operator's stance
when PICO calibration fired; a 0.20 m hand motion likewise came out as
`[-0.20,0,0]`, `[-0.141,0.141,0]` or `[-0.173,0,0.10]` depending on their neck
pose. Do not "reuse" it because it is already there.

## The reference gesture

One physical pose, used for three things: what the operator assumes when
calibrating, what the mock/keyboard sources synthesize, and the built-in
default calibration. It corresponds to the G1's arms at `q14 = 0`:

> Stand upright, look level. Upper arms hanging vertically at your sides,
> elbows bent 90°, forearms horizontal pointing straight forward,
> **palms facing each other** (thumbs up, as if holding a box between your
> hands). Hold both menu buttons for 1 s.

**Palms facing each other, not palms down.** Measured on the URDF at `q=0`: the
two gripper fingers are stacked *vertically* and both thumbs — the opposing
digit, i.e. the palm normal — point toward the midline. Telling the operator
"palms down" would bake a 90° error into every wrist target, which is the exact
bug this path exists to fix. `test_reference_pose_is_the_gesture_we_tell_the_operator`
asserts the description against the URDF so the instruction cannot drift from
the robot.

## Useful measured constants

| quantity | value |
|---|---|
| `L_ee`/`R_ee` rotation at `q14 = 0` | identity (to ~2e-4 rad) |
| `L_ee`/`R_ee` position at `q14 = 0` | `[0.250, ±0.149, 0.095]` |
| shoulder-pitch origin | `[0, ±0.100, 0.292]` |
| `|L_ee − shoulder|` at rest (`_Q_REF`) | 0.365 m |
| `|L_ee − shoulder|` maximum | 0.460 m |
| wrist limits | roll ±1.972, pitch ±1.614, yaw ±1.614 rad |

Read these from the model at runtime rather than copying them — see
`_init_amo_arm_ik` and `_init_wrist_calibration`, which both do.
