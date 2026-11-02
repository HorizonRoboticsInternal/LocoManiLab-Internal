# The unified G1 model — one geometry lineage for every controller

Two files, one geometry. Everything here is tracked in git, meshes included —
nothing to fetch, see "Meshes".

| file | shape | meshes | who loads it |
|---|---|---|---|
| `robot.xml` | 43 DOF (29 body + 14 Dex3), `nq=50`, `nbody=47`, 34.395 kg | 49 STLs | **physics**: robosuite/robocasa (copied into `robosuite_src/.../robots/g1/robot.xml`) |
| `robot_fk.xml` | 29 DOF, no hands, `nq=36`, `nbody=38`, 33.341 kg | **none** | **FK / reference**: `RobotCfg.mjcf_path`, i.e. `DefaultReferenceFrame`, `RealtimeMotionBuffer`, sim2real's own sim2sim |

It has to be two files, and the split is enforced by code, not taste:

* `DefaultReferenceFrame.__init__` (`g1_bridge/reference/default.py:56`) asserts
  `nq == RobotCfg.qpos_size` — 7 + 29 = **36**. Hand joints make that 50 and it
  raises outright. The reference frame is also what the real robot uses, where
  the 14 hand DOF are driven separately and have no place in the tracked frame.
* `RealtimeMotionBuffer`'s default frame looks up `left/right_toe_link`, which
  the hand-ful file does not carry (it would fall back to zeros, silently).

What makes this "unified" is that the two agree **exactly** where they overlap:
same 29 joints in the same order, and FK world positions equal to **0.000 mm**
across random poses. That was the whole problem before — the FK model
(`hf://elijahgalahad/g1_xmls`) and the physics model (sonic's deploy lineage)
disagreed by up to 10.4 mm and 1.78 kg.

## Why one file, and why this one

Measured across the four controllers' native upstream MJCFs (mujoco-compiled,
same joint angles, per-body world positions):

| | heft | amo | scalebfm | sonic |
|---|---|---|---|---|
| FK vs heft | — | **0.000 mm** | **0.000 mm** | 10.4 mm |
| shared-joint geometry diff | — | 0 items | 4 items (wrist_yaw mass only) | 10 items |

heft, amo and scalebfm are the *same robot in three assembly states*:

* `scalebfm` = heft's 29-DOF skeleton + the official Dex3 hands (the hand
  bodies/joints are byte-identical to sonic's, and `wrist_yaw` gains
  +0.2028 kg ×2 of palm/actuator mass — physically the right thing to happen
  when you bolt a hand on).
* `amo` = heft's skeleton with the 6 wrist joints removed (the wrist *links*
  and their masses are identical; AMO's policy is 23-DOF and drives the wrists
  from IK instead).

So `scalebfm`'s file is their maximal common model, and that is what this
directory holds. sonic is the one true outlier (10.4 mm at the waist, +1.78 kg
of torso); it was moved onto this model deliberately — see the parity note
below.

## What this file does NOT carry

Per-joint `armature` / `damping` / `frictionloss` are **not** geometry: each
controller trained under its own values, and they differ enough to matter
(injecting only these three, from the same initial state, a 0.5 s free fall
diverges by up to 1.27 rad). They now live next to `joint_kp`/`joint_kd` in each
controller's `checkpoints/<backend>/**/policy.yaml`, and are applied on top of
this model at load time:

* compile-time — `sim2real.sim_env.utils.mjcf.apply_joint_dynamics`
* runtime (already-compiled `MjModel`, e.g. robosuite) —
  `apply_joint_dynamics_runtime`

Both routes were verified to produce identical `dof_*` arrays, and injection
leaves FK bit-identical (`max |Δxpos| = 0.000000 mm`).

## Provenance

`robot.xml` — copied verbatim from ScaleBFM's
`ScaleBridge/scalebridge/data/robot/g1_29dof/g1_29dof_dex3.xml`
(`zengweishuai/ScaleBFM`, main), then stripped of its bundled scene (a second
`asset`/`worldbody` pair plus `visual`/`statistic`): both `_temp_scene_with_floor`
and robosuite add their own ground, and a second `floor` geom makes MuJoCo reject
the model on a duplicate name. Total mass 34.395 kg, which matches NVIDIA's own
*training* URDF for sonic
(`GR00T-WholeBodyControl/gear_sonic/data/assets/robot_description/urdf/g1/main.urdf`,
34.394 kg) to within 1 gram — i.e. this is the unitree official
URDF/description lineage, not the unitree_mujoco deploy lineage that sonic's
`gear_sonic_deploy/g1/*.xml` (36.165 kg) belongs to.

`robot_fk.xml` — copied verbatim from
`Axellwppr/motion_tracking@sim2real:sim2real/config/g1/assets/g1.xml` (heft's
native model), which is the same lineage minus the hands and with the toe links
present.

## Meshes

`meshes/` holds exactly the 49 STLs `robot.xml` references — 31MB of the
official unitree description, tracked in git. There is nothing to download and
no "copy them from a full checkout" step; a fresh clone compiles both models.

Two things changed to make that affordable, because the number that made it look
unaffordable was wrong:

* **`robot_fk.xml` carries no meshes at all.** Its consumers only ever FK it, and
  FK needs the kinematic tree, not the visual shells. Every body has an explicit
  `<inertial>`, so dropping the 35 `class="visual"` mesh geoms leaves `xpos`,
  `xquat`, `site_xpos`, `body_mass`, `body_inertia`, `body_ipos`, `body_iquat`,
  `dof_armature` and `jnt_range` **bit-identical** (`max |Δxpos| = 0.000000000 mm`
  over 1000 random poses, `nq/nv/nbody/njnt/nsite` and every body/joint/site name
  unchanged). This is the model the robot itself loads, so it is the one that
  most wanted to stop depending on an out-of-band asset.
* **The other 118 files are gone.** The "~106MB" was the full
  `g1_29dof_with_hand` drop: Dex1, the force-sensor shells, backpack, d455, the
  5010 wrists, the rubber hands. No XML in this repo references any of them.

A viewer that wants to see a rendered robot should load `robot.xml`, which keeps
its visuals. Re-adding visuals to `robot_fk.xml` means re-adding ~4MB of meshes
for every clone including the one on the G1 — the `class="visual"` default is
still declared there if you ever need to.

These STLs are the unitree official description lineage and have not changed;
the same files also ship, byte-identical, inside `tele_humanoid_benchmark`
(`robosuite_src/.../robots/g1/meshes`, 49 of the 49) and `heft`
(`sim2real/config/g1/assets/meshes`), both likewise tracked in git.

## sonic parity note

Moving sonic onto this model changes its physics (waist −10 mm, torso −1.78 kg,
`armature` 0.010 → the per-motor values). Its previously recorded sim success
rates are therefore **not comparable across that change** — rerun all four
backends' baselines together after adopting this model.
