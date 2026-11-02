"""Room-independent ordered-stage state machine for iTHOR tasks."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from robocasa.environments.ithor_composite.progress import (
    CompositeStageReporterMixin,
)


StageCallback = Callable[["IThorMultiStageCompositeMixin"], Any]


@dataclass(frozen=True)
class IThorAtomicTask:
    """One ordered atomic stage inside an iTHOR composite task."""

    name: str
    success: str | StageCallback
    task_cls: type | str | None = None
    hold_steps: int | None = None
    on_enter: str | StageCallback | None = None
    on_complete: str | StageCallback | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class IThorMultiStageCompositeMixin(CompositeStageReporterMixin):
    """Ordered stage engine shared by concrete iTHOR room bases."""

    TASK_FAMILY = "composite"
    ATOMIC_TASKS: Sequence[IThorAtomicTask | Mapping[str, Any]] = ()
    SUCCESS_HOLD_STEPS = 5
    PRINT_STAGE_MESSAGES = True

    def __init__(self, *args, **kwargs):
        self._atomic_tasks = self._normalize_atomic_tasks(self._get_atomic_tasks())
        self._validate_atomic_tasks(self._atomic_tasks)
        self._stage_index = 0
        self._stage_hold_steps = 0
        super().__init__(*args, **kwargs)

    def _get_atomic_tasks(self):
        """Return the ordered stage declarations for this task instance."""

        return self.ATOMIC_TASKS

    @staticmethod
    def _normalize_atomic_tasks(tasks):
        normalized = []
        for task in tasks:
            if isinstance(task, IThorAtomicTask):
                normalized.append(task)
                continue
            if not isinstance(task, Mapping):
                raise TypeError(
                    "ATOMIC_TASKS entries must be IThorAtomicTask or mappings"
                )
            config = dict(task)
            if "success" not in config and "success_condition" in config:
                config["success"] = config.pop("success_condition")
            normalized.append(IThorAtomicTask(**config))
        return tuple(normalized)

    @staticmethod
    def _validate_atomic_tasks(tasks):
        if not tasks:
            raise ValueError(
                "IThorMultiStageCompositeMixin consumers must declare ATOMIC_TASKS"
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
                        f"{task.name}: {callback_name} must be a method name or "
                        "callable"
                    )
            if task.hold_steps is not None and int(task.hold_steps) < 1:
                raise ValueError(f"{task.name}: hold_steps must be positive")

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

    def _reset_task_progress(self):
        """Reset ordered progress through the standard iTHOR task hook."""

        self._reset_composite_progress()

    def _resolve_stage_callback(self, callback):
        if isinstance(callback, str):
            resolved = getattr(self, callback)
            return resolved() if callable(resolved) else resolved
        return callback(self)

    def _enter_stage(self, task):
        if task is not None and task.on_enter is not None:
            self._resolve_stage_callback(task.on_enter)

    def _stage_success(self, task):
        return bool(self._resolve_stage_callback(task.success))

    def _required_stage_hold_steps(self, task):
        """Return the consecutive control steps required for one stage."""

        return int(task.hold_steps or self.SUCCESS_HOLD_STEPS)

    def _print_stage_completion(self, task, completed_index):
        if not self.PRINT_STAGE_MESSAGES:
            return
        total = len(self._atomic_tasks)
        next_task = self.current_atomic_task
        self._report_stage_complete(
            completed_index,
            total,
            next_stage=next_task.name if next_task is not None else None,
        )

    def _complete_stage(self, task):
        if task.on_complete is not None:
            self._resolve_stage_callback(task.on_complete)
        self._stage_index += 1
        self._stage_hold_steps = 0
        self._print_stage_completion(task, self._stage_index)
        self._enter_stage(self.current_atomic_task)

    def _reset_internal(self):
        # Clear the previous episode before any parent reset / scene setup hook
        # can observe a stale completed-stage state.
        self._reset_task_progress()
        super()._reset_internal()
        self._enter_stage(self.current_atomic_task)
        self._validate_initial_composite_state()

    def _validate_initial_composite_state(self):
        """Reject a reset that already satisfies the first ordered stage."""

        task = self.current_atomic_task
        if task is not None and self._stage_success(task):
            raise RuntimeError(
                f"{self.scene_name}: composite task starts with completed stage "
                f"{task.name!r}"
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

        # The parent computes reward before this state machine advances. Return
        # the post-transition result so the final stage succeeds in this step.
        success = bool(self._check_success())
        info["task_success"] = success
        info["success"] = success
        info["stage_index"] = self.stage_index
        info["stage_hold_steps"] = self._stage_hold_steps
        info["active_atomic_task"] = (
            self.current_atomic_task.name
            if self.current_atomic_task is not None
            else None
        )
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
        meta["atomic_tasks"] = [
            {
                "name": task.name,
                "task_class": (
                    task.task_cls
                    if isinstance(task.task_cls, str)
                    else (task.task_cls.__name__ if task.task_cls is not None else None)
                ),
                "hold_steps": self._required_stage_hold_steps(task),
                **dict(task.metadata),
            }
            for task in self._atomic_tasks
        ]
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


__all__ = ["IThorAtomicTask", "IThorMultiStageCompositeMixin"]
