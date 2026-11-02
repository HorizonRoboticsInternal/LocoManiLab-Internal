"""Generic ProcTHOR task-spec loading.

These specs sit above task-local fixtures: a task spec can name scene-specific
spawn points, target regions, fixture config files, and other calibrated task
data without embedding that data in Python task classes.
"""

from __future__ import annotations

import os

from robocasa.models.scenes.procthor_yaml import config_path
from robocasa.models.scenes.procthor_yaml import load_yaml_config


_MODEL_ASSET_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../assets")
)
_TASK_SPEC_ROOT = os.path.join(_MODEL_ASSET_ROOT, "scenes", "procthor_task_specs")


def task_spec_config_path(name):
    return config_path(_TASK_SPEC_ROOT, name)


def load_task_spec(name, **variables):
    return load_yaml_config(_TASK_SPEC_ROOT, name, **variables)
