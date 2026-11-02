"""HEFT's frozen G1 motion-tracking policy as a wbt controller.

Structural peer of ``g1_bridge/vendored/wbt/controllers/wbt_controller_sonic.py``:
same
``step(robot) -> (target_q, kps, kds)`` contract, same MuJoCo joint order on the
way out, so ``WholeBodyG1Composite.run_controller`` drives it with the torque
loop it already has and nothing downstream changes.

**Why go through the composite at all**, when ``rl/kimodo/heft/heft_driver.py``
already drives the robot directly: the driver bypasses ``env.step()``, which is
fine for verifying the tracker in a bare arena but wrong for anything with a
task. ``_post_action`` hold counters, ``_check_success``, camera observations and
the recorder all live on that path. And ``EpisodeRecorder.record_step`` reads
``composite.get_teleop_data_sample()["target_q_29"]``; with no composite in the
loop that is ``None``, ``build_frame`` returns ``None``, and **every frame is
silently dropped** -- an empty dataset with no error.

Three things this has to get right:

  * **Joint order out is MuJoCo/dataset, not canonical.** ``run_controller``
    computes ``kps * (target_q - qpos)`` against ``_read_joint_state()``, which
    is MuJoCo-ordered, and SONIC's ``step`` ends with
    ``target_q_isaac[isaaclab_to_mujoco_reindex]`` for exactly this reason. The
    policy speaks canonical; the permutation happens here, on the way out.
  * **Gains are HEFT's, not the vendored table.** ``bydmimic_utils.kps/kds`` are
    SONIC's: hip_pitch differs by 2.47x and wrist_pitch/yaw by 1.95x. Wrong gains
    there do not topple the robot, they make it track slightly worse.
  * **One policy tick per control step.** ``run_controller`` runs every 0.002 s
    substep and gates the policy behind ``need_policy``; this class is only ever
    called from inside that gate, so the history push and the reference cursor
    advance exactly once per 50 Hz tick. ``n_ticks`` is exposed so a caller can
    assert the 1:10 ratio rather than trust it.

The observation needs no root position: HEFT's deployed policy reads none (root
position and linear velocity are privileged, teacher-only), and ``TargetRootZ``
comes from the *reference*. So the state dict ``run_controller`` already builds
-- joint pos/vel, pelvis quaternion, body-frame angular velocity -- is complete.
That is why this controller needs no floor datum and no scene knowledge.
"""

from __future__ import annotations

import numpy as np

from rl.kimodo.heft.g1_actuators import gain_arrays
from rl.kimodo.heft.heft_ckpt import HeftPolicy
from rl.kimodo.heft.heft_obs import TrackerObs
from rl.kimodo.heft.tracker_const import (
    CANON_TO_DS,
    DATASET_JOINT_NAMES,
    DEFAULT_JOINT_POS,
    DEFAULT_JOINT_POS_CANON,
    DS_TO_CANON,
    N_JOINTS,
)


class HeftTrackerController:
    """Frozen HEFT tracker, wrapped to the wbt controller contract."""

    def __init__(self, onnx_path=None, policy: HeftPolicy | None = None,
                 threads: int = 1):
        self.policy = policy if policy is not None else HeftPolicy(onnx_path,
                                                                   threads=threads)
        self.obs = TrackerObs(num_envs=1)
        kp, kd, _, _ = gain_arrays(DATASET_JOINT_NAMES)   # MuJoCo order, on purpose
        self.kps = kp.astype(np.float32)
        self.kds = kd.astype(np.float32)
        self._default_ds = DEFAULT_JOINT_POS.astype(np.float32)
        self.n_ticks = 0
        self.n_ticks_no_ref = 0
        self._has_reference = False
        self._replay = None
        self._replay = None
        self.reset()

    # ------------------------------------------------------------------ #
    def reset(self) -> None:
        """Clear per-episode state, keeping the loaded session.

        ``_make_wbt_controller`` caches controllers across ``env.reset()`` and
        calls this instead of rebuilding, so the ONNX session survives while the
        history buffers and the reference cursor do not.

        **The reference is dropped, not merely rewound.** ``reset_buffers`` puts
        the cursor back to 0 but leaves ``obs.ref`` in place, so without this the
        next episode silently tracks the PREVIOUS episode's chunk from its first
        frame -- the robot moves, plausibly, to the wrong motion, and nothing
        reports anything. A new episode has no reference until it is given one.
        """
        self.obs.reset_buffers()
        self.obs.ref = None
        self.n_ticks = 0
        self.n_ticks_no_ref = 0
        self._has_reference = False

    def set_replay(self, targets_ds) -> None:
        """Drive from a recorded ``target_q_29`` stream instead of the policy.

        ``targets_ds`` is [T, 29] in MuJoCo/dataset order -- exactly the column
        ``EpisodeRecorder`` writes as ``action.wbc``. Pass None to go back to the
        policy. Past the end the last target is held, so a replay that outlives
        its recording holds a pose rather than reading out of bounds.
        """
        self._replay = (None if targets_ds is None
                        else np.asarray(targets_ds, dtype=np.float64).reshape(-1, N_JOINTS))
        self.n_ticks = 0

    def set_reference(self, ref_qpos36: np.ndarray, reboot: bool = True) -> None:
        """``ref_qpos36`` [T, 36] at 50 Hz, dataset joint order, scene-local frame.

        Does not touch the history buffers: at a chunk boundary the continuation
        was generated from the robot's realized state, so clearing history would
        erase the drift the whole project measures.
        """
        ref = np.asarray(ref_qpos36, dtype=np.float64)
        self.obs.set_reference(ref[None] if ref.ndim == 2 else ref, reboot=reboot)
        self._has_reference = True

    @property
    def reference_step(self) -> int:
        return int(self.obs.idx[0]) if self._has_reference else -1

    @property
    def reference_len(self) -> int:
        return int(self.obs.T) if self._has_reference else 0

    @property
    def reference_done(self) -> bool:
        """True once the cursor has reached the last frame of the chunk."""
        return self._has_reference and self.reference_step >= self.reference_len - 1

    # ------------------------------------------------------------------ #
    def step(self, robot) -> tuple:
        """``(target_q, kps, kds)``, all 29-dim in MuJoCo/dataset joint order.

        Before a reference is set, the default pose is commanded -- not because
        that holds the robot up (it does NOT: measured 0.72 m of collapse over
        200 ticks, because HEFT's gains are deliberately soft and the gravity
        compensation lives in the policy's delta, see plan.md's note on the
        open-loop null hypothesis) but because no observation can be assembled
        from a reference that does not exist, and crashing here would break
        robosuite's own reset-time stepping.

        **So: never step the sim between ``env.reset()`` and
        ``set_heft_reference``.** A warm-up step, which is exactly what the
        SONIC-era ``KimodoSonicTracker.reset`` does to lazy-init the kPlanner,
        knocks the robot down before tracking starts. ``n_ticks_no_ref`` counts
        these so a harness can assert it is zero instead of trusting the order.
        """
        st = robot.get_state()
        qpos_ds = np.asarray(st["joint_pos"], dtype=np.float64)
        qvel_ds = np.asarray(st["joint_vel"], dtype=np.float64)
        quat = np.asarray(st["quaternion"], dtype=np.float64)      # wxyz
        # Body frame already: MuJoCo's freejoint qvel is [linear_world, angular_local]
        # and _read_pelvis_state slices the angular part. mjlab reports world and
        # rotates it; doing that again here corrupts 27 of the 1729 dims.
        gyro = np.asarray(st["gyroscope"], dtype=np.float64)

        # Replay: hand back the recorded targets instead of running the policy.
        # This is what makes an episode's `action.wbc` column checkable -- P3-4
        # asks whether the recorded actions REPRODUCE the episode, and they only
        # answer that if they can be played back as the actions. Checked before
        # the reference guard because a replay needs no reference at all: the
        # actions already encode everything the policy would have produced.
        if self._replay is not None:
            k = min(self.n_ticks, len(self._replay) - 1)
            self.n_ticks += 1
            return self._replay[k].astype(np.float32), self.kps, self.kds

        if not self._has_reference:
            self.n_ticks_no_ref += 1
            return self._default_ds.copy(), self.kps, self.kds

        qc, qdc = qpos_ds[DS_TO_CANON], qvel_ds[DS_TO_CANON]
        self.obs.push_history(quat, gyro, qc, qdc)
        # root_pos is unused by the policy; pass zeros rather than inventing a datum.
        v = self.obs.observe(np.zeros(3), quat, qc)
        delta, raw = self.policy(v)

        target_canon = DEFAULT_JOINT_POS_CANON + delta[0]
        target_ds = target_canon[CANON_TO_DS].astype(np.float32)

        self.obs.push_action(raw[0])
        self.obs.advance()
        self.n_ticks += 1

        if not np.isfinite(target_ds).all():
            # MuJoCo does not raise on a non-finite ctrl; it poisons the state,
            # every later reward is NaN and the step rate collapses ~10x.
            raise RuntimeError("[heft_tracker] non-finite joint target")
        return target_ds, self.kps, self.kds

    # ------------------------------------------------------------------ #
    def realized_qpos36(self, root_pos_local, root_quat_wxyz, joint_pos_ds
                        ) -> np.ndarray:
        """[36] in the layout Kimodo continuation expects, dataset joint order.

        Root pose is supplied by the caller because this controller deliberately
        knows nothing about the scene's floor datum or the fact that robosuite's
        anatomical pelvis is ``mobilebase0_support`` rather than the body named
        ``pelvis``. See ``rl/kimodo/heft/model_patch.resolve_root_body``.
        """
        return np.concatenate([
            np.asarray(root_pos_local, dtype=np.float64).reshape(3),
            np.asarray(root_quat_wxyz, dtype=np.float64).reshape(4),
            np.asarray(joint_pos_ds, dtype=np.float64).reshape(N_JOINTS),
        ])

    def __repr__(self) -> str:
        return (f"HeftTrackerController({self.policy.path.name}, "
                f"ref {self.reference_step}/{self.reference_len}, "
                f"{self.n_ticks} ticks)")
