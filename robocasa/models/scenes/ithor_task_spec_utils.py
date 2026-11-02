"""Strict shared loading for scene-calibrated iTHOR task catalogs."""

import re

import yaml


_SCENE_NAME_RE = re.compile(r"^FloorPlan\d+$")


class _StrictSafeLoader(yaml.SafeLoader):
    """Safe YAML loader that rejects duplicate mapping keys."""


def _construct_unique_mapping(loader, node, deep=False):
    loader.flatten_mapping(node)
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        try:
            duplicate = key in mapping
        except TypeError as exc:
            raise ValueError(
                f"Unhashable YAML mapping key at {key_node.start_mark}"
            ) from exc
        if duplicate:
            raise ValueError(
                f"Duplicate YAML key {key!r} at {key_node.start_mark}"
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def load_scene_task_specs(path, room_label):
    """Return a validated ``FloorPlan -> task -> spec`` mapping."""

    with open(path, "r") as file:
        document = yaml.load(file, Loader=_StrictSafeLoader) or {}
    scenes = document.get("scenes") if isinstance(document, dict) else None
    if (
        not isinstance(document, dict)
        or set(document) != {"scenes"}
        or not isinstance(scenes, dict)
        or not scenes
        or any(
            _SCENE_NAME_RE.fullmatch(str(scene_name)) is None
            or not isinstance(task_specs, dict)
            or not task_specs
            or any(
                not isinstance(task_key, str) or not isinstance(spec, dict)
                for task_key, spec in task_specs.items()
            )
            for scene_name, task_specs in scenes.items()
        )
    ):
        raise ValueError(f"Invalid iTHOR {room_label} task specs in {path}")
    return scenes


__all__ = ["load_scene_task_specs"]
