"""Reusable base for serial multi-hop Holodeck tasks.

Holodeck scenes are baked XML exports.  Concrete composite tasks declare an
ordered list of atomic stages, optional fixed-pose task assets, and a robot
spawn pose; the base class owns the serial success state machine.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any

import numpy as np

import robocasa
from robocasa.environments.holodeck.holodeck import HolodeckEnv
from robocasa.models.objects.objects import MJCFObject


StageCallback = Callable[["HolodeckMultiHopComposite"], Any]


@dataclass(frozen=True)
class HolodeckAtomicTask:
    """One ordered atomic stage.

    ``task_cls`` records the existing Holodeck atomic-task environment whose
    calibrated primitive is reused by this stage. It is descriptive rather
    than an environment to instantiate: every stage runs in the one shared
    scene and simulation owned by the composite. ``success`` and callbacks may
    be method names or callables. Method names are resolved on the composite
    environment; callables receive that environment as their only argument.
    """

    name: str
    success: str | StageCallback
    task_cls: type | str | None = None
    hold_steps: int | None = None
    on_enter: str | StageCallback | None = None
    on_complete: str | StageCallback | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class HolodeckMultiHopComposite(HolodeckEnv):
    """Base class for ordered, multi-stage Holodeck tasks.

    A subclass can be configured declaratively, for example::

        ATOMIC_TASKS = (
            HolodeckAtomicTask("reach_desk", "_desk_reached"),
            HolodeckAtomicTask("place_mug", "_mug_placed", hold_steps=5),
        )
        ROBOT_BASE_POSES = {
            "train_117": {"pos": [1.0, 2.0, 0.0], "yaw": 1.57},
        }
        ASSET_CONFIGS = {
            "train_117": (
                {
                    "name": "mug",
                    "mjcf_path": "objects/lightwheel/mug/Mug001/model.xml",
                    "scale": 1.0,
                    "pose": {
                        "pos": [1.2, 2.1, 0.8],
                        "quat": [1.0, 0.0, 0.0, 0.0],
                    },
                },
            ),
        }

    ``ROBOT_BASE_POSE`` is a scene-independent fallback.  Asset configs are
    task-local MJCF objects, placed after the normal Holodeck reset.
    """

    ABSTRACT = True
    TASK_FAMILY = "composite"
    ATOMIC_TASKS: Sequence[HolodeckAtomicTask | Mapping[str, Any]] = ()
    SUCCESS_HOLD_STEPS = 5
    SUCCESS_DEBUG_ENABLED = False
    SUCCESS_DEBUG_INTERVAL_SECONDS = 5.0
    ROBOT_BASE_POSE: Mapping[str, Any] | Sequence[float] | None = None
    ROBOT_BASE_POSES: Mapping[str, Mapping[str, Any] | Sequence[float]] = {}
    ASSET_CONFIGS: (
        Mapping[str, Sequence[Mapping[str, Any]]] | Sequence[Mapping[str, Any]]
    ) = ()

    def __init__(self, *args, **kwargs):
        self._atomic_tasks = self._normalize_atomic_tasks(self._get_atomic_tasks())
        self._validate_atomic_tasks(self._atomic_tasks)
        self._stage_index = 0
        self._stage_hold_steps = 0
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # Declarative configuration
    # ------------------------------------------------------------------
    def _get_atomic_tasks(self):
        """Return the ordered stage declarations for this task instance."""
        return self.ATOMIC_TASKS

    @staticmethod
    def _normalize_atomic_tasks(tasks):
        normalized = []
        for task in tasks:
            if isinstance(task, HolodeckAtomicTask):
                normalized.append(task)
                continue
            if not isinstance(task, Mapping):
                raise TypeError(
                    "ATOMIC_TASKS entries must be HolodeckAtomicTask or mappings"
                )
            config = dict(task)
            if "success" not in config and "success_condition" in config:
                config["success"] = config.pop("success_condition")
            normalized.append(HolodeckAtomicTask(**config))
        return tuple(normalized)

    @staticmethod
    def _validate_atomic_tasks(tasks):
        if not tasks:
            raise ValueError(
                "HolodeckMultiHopComposite subclasses must declare ATOMIC_TASKS"
            )
        names = [task.name for task in tasks]
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("every atomic task must have a non-empty string name")
        if len(names) != len(set(names)):
            raise ValueError(f"atomic task names must be unique: {names}")
        for task in tasks:
            if not isinstance(task.success, str) and not callable(task.success):
                raise TypeError(
                    f"{task.name}: success must be a method name or callable"
                )
            if task.task_cls is not None and not isinstance(task.task_cls, (str, type)):
                raise TypeError(
                    f"{task.name}: task_cls must be an environment class or name"
                )
            for callback_name, callback in (
                ("on_enter", task.on_enter),
                ("on_complete", task.on_complete),
            ):
                if (
                    callback is not None
                    and not isinstance(callback, str)
                    and not callable(callback)
                ):
                    raise TypeError(
                        f"{task.name}: {callback_name} must be a method name or callable"
                    )
            if task.hold_steps is not None and int(task.hold_steps) < 1:
                raise ValueError(f"{task.name}: hold_steps must be positive")

    def _get_robot_base_pose_config(self):
        """Return a pose mapping or a ``(position, yaw)`` pair."""
        scene_poses = self.ROBOT_BASE_POSES or {}
        config = scene_poses.get(self.scene_name)
        if config is None:
            config = scene_poses.get("default")
        if config is None:
            config = self.ROBOT_BASE_POSE
        return config

    def _configure_robot_spawn(self):
        config = self._get_robot_base_pose_config()
        if config is None:
            return
        if isinstance(config, Mapping):
            if "pos" not in config:
                raise ValueError(
                    f"{type(self).__name__}: robot pose config needs 'pos'"
                )
            pos = config["pos"]
            yaw = config.get("yaw", 0.0)
        else:
            if len(config) != 2:
                raise ValueError(
                    f"{type(self).__name__}: robot pose must be (pos, yaw)"
                )
            pos, yaw = config
        self.set_robot_base_pose(np.asarray(pos, dtype=float), float(yaw))

    def _get_asset_configs(self):
        """Return fixed-pose task assets for the current Holodeck scene."""
        configs = self.ASSET_CONFIGS
        if isinstance(configs, Mapping):
            configs = configs.get(self.scene_name, configs.get("default", ()))
        return tuple(deepcopy(config) for config in configs)

    # ------------------------------------------------------------------
    # Holodeck lifecycle
    # ------------------------------------------------------------------
    def _setup_task_fixtures(self):
        super()._setup_task_fixtures()
        self._configure_robot_spawn()
        self._validate_asset_configs(self._get_asset_configs())

    def _get_obj_cfgs(self):
        # These assets have explicit poses and must not be sampled onto the
        # automatically detected scene surface.
        return []

    def _create_objects(self):
        super()._create_objects()
        for config in self._get_asset_configs():
            name = config["name"]
            if name in self.objects:
                raise ValueError(f"duplicate Holodeck asset name: {name!r}")
            path = config.get("mjcf_path", config.get("asset"))
            if not os.path.isabs(path):
                path = os.path.join(robocasa.models.assets_root, path)
            asset = MJCFObject(
                name=name,
                mjcf_path=path,
                scale=float(config.get("scale", 1.0)),
            )
            self.objects[name] = asset
            self.model.merge_objects([asset])

    def _reset_internal(self):
        super()._reset_internal()
        self._place_configured_assets()
        self.sim.forward()
        self._reset_task_progress()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        """Reject a reset that already satisfies the first stage.

        Concrete tasks can extend this hook for additional collision, floor,
        support, or object-layout checks while retaining this guard.
        """
        if self.current_atomic_task is not None and self._stage_success(
            self.current_atomic_task
        ):
            raise RuntimeError(
                f"{self.scene_name}: composite task starts with completed stage "
                f"{self.current_atomic_task.name!r}"
            )

    def _validate_asset_configs(self, configs):
        names = []
        for config in configs:
            if not isinstance(config, Mapping):
                raise TypeError("ASSET_CONFIGS entries must be mappings")
            name = config.get("name")
            if not isinstance(name, str) or not name:
                raise ValueError("every Holodeck asset config needs a name")
            if name in names:
                raise ValueError(f"duplicate Holodeck asset name: {name!r}")
            names.append(name)
            if not config.get("mjcf_path", config.get("asset")):
                raise ValueError(f"{name}: asset config needs 'mjcf_path'")
            if "pose" not in config:
                raise ValueError(f"{name}: fixed-pose asset config needs 'pose'")
            self._parse_asset_pose(config["pose"], name)

    @staticmethod
    def _parse_asset_pose(pose, name):
        if isinstance(pose, Mapping):
            if "pos" not in pose or "quat" not in pose:
                raise ValueError(f"{name}: pose needs 'pos' and 'quat'")
            pos, quat = pose["pos"], pose["quat"]
        else:
            if len(pose) != 7:
                raise ValueError(
                    f"{name}: pose must be {{pos, quat}} or a 7-element sequence"
                )
            pos, quat = pose[:3], pose[3:]
        if len(pos) != 3 or len(quat) != 4:
            raise ValueError(f"{name}: pose needs 3 position and 4 quaternion values")
        quat = np.asarray(quat, dtype=float)
        if not np.isfinite(quat).all() or np.linalg.norm(quat) <= 1e-8:
            raise ValueError(f"{name}: quaternion must be finite and non-zero")
        return np.asarray(pos, dtype=float), quat

    def _place_configured_assets(self):
        for config in self._get_asset_configs():
            pos, quat = self._parse_asset_pose(config["pose"], config["name"])
            obj = self.objects[config["name"]]
            qpos = self.sim.data.get_joint_qpos(obj.joints[0]).copy()
            qpos[:3] = pos
            qpos[3:7] = quat
            self.sim.data.set_joint_qpos(obj.joints[0], qpos)
            qvel_addr = self.sim.model.get_joint_qvel_addr(obj.joints[0])
            if isinstance(qvel_addr, tuple):
                self.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
            else:
                self.sim.data.qvel[qvel_addr] = 0.0

    # ------------------------------------------------------------------
    # Ordered success state machine
    # ------------------------------------------------------------------
    @property
    def current_atomic_task(self):
        if self._stage_index >= len(self._atomic_tasks):
            return None
        return self._atomic_tasks[self._stage_index]

    @property
    def completed_atomic_tasks(self):
        return self._atomic_tasks[: self._stage_index]

    @property
    def stage_index(self):
        return self._stage_index

    def _reset_composite_progress(self):
        self._stage_index = 0
        self._stage_hold_steps = 0
        # These counters are also used by atomic primitives mixed into
        # composites. ``reset()`` normally skips ``_setup_scene()``, so clear
        # them here to prevent the previous episode from leaking success.
        self._in_range_steps = 0
        self._success_hold = 0
        self._success_hold_steps = 0
        self._success_debug_next_log_time = float(self.SUCCESS_DEBUG_INTERVAL_SECONDS)
        self._success_debug_reported = False

    def _reset_task_progress(self):
        """Clear episode-local state for both full and fixed-scene resets."""
        self._reset_composite_progress()

    def _resolve_callback(self, callback):
        if isinstance(callback, str):
            resolved = getattr(self, callback)
            return resolved() if callable(resolved) else resolved
        return callback(self)

    def _enter_stage(self, task):
        if task is not None and task.on_enter is not None:
            self._resolve_callback(task.on_enter)

    def _complete_stage(self, task):
        if task.on_complete is not None:
            self._resolve_callback(task.on_complete)
        self._stage_index += 1
        self._stage_hold_steps = 0
        self._enter_stage(self.current_atomic_task)

    def _stage_success(self, task):
        return bool(self._resolve_callback(task.success))

    def _success_debug_details(self, task):
        """Return optional details for the active stage debug log."""
        del task
        return ""

    def _required_stage_hold_steps(self, task):
        if task.hold_steps is not None:
            return max(1, int(task.hold_steps))
        hold_seconds = getattr(task.task_cls, "HOLD_SECONDS", None)
        if hold_seconds is not None:
            return max(1, int(round(float(hold_seconds) * self.control_freq)))
        return max(1, int(self.SUCCESS_HOLD_STEPS))

    def _log_success_conditions(self):
        """Print ordered stage conditions at most once per debug interval.

        Completed stages are reported from the state machine instead of being
        reevaluated, while only the active stage predicate is called. This
        keeps diagnostic logging from changing bookkeeping in future stages.
        """
        sim_time = float(self.sim.data.time)
        interval = max(0.1, float(self.SUCCESS_DEBUG_INTERVAL_SECONDS))
        next_log_time = getattr(self, "_success_debug_next_log_time", interval)
        task_success = bool(self._check_success())
        first_success = task_success and not getattr(
            self, "_success_debug_reported", False
        )
        if sim_time < next_log_time and not first_success:
            return
        while next_log_time <= sim_time:
            next_log_time += interval
        self._success_debug_next_log_time = next_log_time
        if task_success:
            self._success_debug_reported = True

        stage_names = {
            "open_drawer": "打开抽屉",
            "take_tissue_box": "取出纸巾盒",
            "place_tissue_box_on_desk": "将纸巾盒放到书桌",
            "close_drawer": "关闭抽屉",
            "approach_window": "走到窗边",
            "approach_desk": "走到书桌旁",
            "approach_trash_can": "走到垃圾桶旁",
            "throw_bottle": "投瓶到垃圾桶",
            "pick_cup": "拿起玻璃杯",
        }
        statuses = []
        for index, task in enumerate(self._atomic_tasks):
            label = stage_names.get(task.name, task.name)
            if index < self.stage_index:
                status = "已完成"
            elif index == self.stage_index:
                status = "当前满足" if self._stage_success(task) else "未满足"
            else:
                status = "待完成"
            statuses.append(f"{label}={status}")

        active = self.current_atomic_task
        hold_target = self._required_stage_hold_steps(active) if active else 0
        # Temporarily hide the verbose per-condition dump while keeping the
        # periodic stage and overall success log enabled.
        # success_details = self._success_debug_details(active) if active else ""
        # details_text = f"；条件详情：{success_details}" if success_details else ""
        details_text = ""
        print(
            f"[{type(self).__name__}][{self.scene_name}] "
            f"时间={sim_time:.2f}s，当前阶段="
            f"{stage_names.get(active.name, active.name) if active else '全部完成'}，"
            f"保持={self._stage_hold_steps}/{hold_target}；"
            f"阶段状态：{', '.join(statuses)}{details_text}；"
            f"任务={'成功' if task_success else '未成功'}",
            flush=True,
        )

    def _post_action(self, action):
        _, done, info = super()._post_action(action)
        task = self.current_atomic_task
        if task is not None:
            if self._stage_success(task):
                self._stage_hold_steps += 1
            else:
                self._stage_hold_steps = 0
            hold_steps = self._required_stage_hold_steps(task)
            if self._stage_hold_steps >= hold_steps:
                self._complete_stage(task)

        # ``super()._post_action`` computes reward before this state machine
        # advances. Publish the post-transition result so callers observe the
        # final-stage completion in the same env.step() that caused it.
        success = bool(self._check_success())
        if self.SUCCESS_DEBUG_ENABLED:
            self._log_success_conditions()
        info["task_success"] = success
        # Keep the standard success key in sync for direct Robosuite callers;
        # GymWrapper also exposes this key, but direct env.step() does not.
        info["success"] = success
        info["completed_atomic_tasks"] = [
            task.name for task in self.completed_atomic_tasks
        ]
        return float(success), done, info

    def _check_success(self):
        return self.current_atomic_task is None

    def reward(self, action=None):
        return float(self._check_success())

    def get_ep_meta(self):
        meta = super().get_ep_meta()
        meta["task_family"] = self.TASK_FAMILY
        meta["atomic_tasks"] = [
            {
                "name": task.name,
                "task_class": (
                    task.task_cls
                    if isinstance(task.task_cls, str)
                    else (task.task_cls.__name__ if task.task_cls is not None else None)
                ),
                **dict(task.metadata),
            }
            for task in self._atomic_tasks
        ]
        meta["asset_configs"] = self._get_asset_configs()
        meta["completed_atomic_tasks"] = [
            task.name for task in self.completed_atomic_tasks
        ]
        meta["stage_index"] = self.stage_index
        meta["stage_hold_steps"] = self._stage_hold_steps
        meta["active_atomic_task"] = (
            self.current_atomic_task.name
            if self.current_atomic_task is not None
            else None
        )
        meta["success_definition"] = (
            "atomic tasks must be completed in declared order; each stage must "
            f"hold for at least {self.SUCCESS_HOLD_STEPS} control steps by default"
        )
        return meta


__all__ = ["HolodeckAtomicTask", "HolodeckMultiHopComposite"]
