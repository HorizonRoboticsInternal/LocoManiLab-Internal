"""Catalog discovery and static validation for public iTHOR room tasks."""

from dataclasses import dataclass
import importlib

from robosuite.environments.base import REGISTERED_ENVS

from robocasa.environments.ithor_atomic.task_specs import SceneSpecMixin
from robocasa.models.scenes.molmospaces_scene_registry import classify_floorplan


BEDROOM_PACKAGE = "robocasa.environments.ithor_bedroom"
BATHROOM_PACKAGE = "robocasa.environments.ithor_bathroom"
EXPECTED_TASK_COUNTS = {BEDROOM_PACKAGE: 64, BATHROOM_PACKAGE: 73}
EXPECTED_TASK_MATRIX = {
    BEDROOM_PACKAGE: {
        301: {"atomic": 2, "composite": 2},
        302: {"atomic": 20, "composite": 30},
        306: {"atomic": 18, "composite": 27},
        310: {"atomic": 16, "composite": 28},
        315: {"atomic": 2, "composite": 2},
        323: {"atomic": 21, "composite": 32},
        329: {"atomic": 19, "composite": 28},
    },
    BATHROOM_PACKAGE: {
        401: {"atomic": 16, "composite": 25},
        405: {"atomic": 17, "composite": 29},
        407: {"atomic": 18, "composite": 32},
        410: {"atomic": 2, "composite": 2},
        415: {"atomic": 15, "composite": 24},
        418: {"atomic": 16, "composite": 31},
        422: {"atomic": 2, "composite": 2},
    },
}


@dataclass(frozen=True)
class TaskCatalogEntry:
    package: str
    name: str
    cls: type
    scene_ids: tuple[int, ...]
    family: str


def discover_task_catalog(package_names):
    """Discover public task classes from each room's canonical task package."""

    entries = []
    definition_issues = []
    for package_name in package_names:
        room_package = importlib.import_module(package_name)
        tasks = importlib.import_module(f"{package_name}.tasks")
        task_classes = getattr(tasks, "TASK_CLASSES", None)
        if not isinstance(task_classes, tuple):
            definition_issues.append(f"{tasks.__name__}: TASK_CLASSES must be a tuple")
            continue
        if not task_classes:
            definition_issues.append(
                f"{tasks.__name__}: TASK_CLASSES must not be empty"
            )
        if any(not isinstance(cls, type) for cls in task_classes):
            definition_issues.append(
                f"{tasks.__name__}: TASK_CLASSES contains a non-class value"
            )
            continue
        if len({id(cls) for cls in task_classes}) != len(task_classes):
            definition_issues.append(
                f"{tasks.__name__}: TASK_CLASSES contains duplicate classes"
            )
        expected_names = [cls.__name__ for cls in task_classes]
        if len(set(expected_names)) != len(expected_names):
            definition_issues.append(
                f"{tasks.__name__}: TASK_CLASSES contains duplicate names"
            )
        root_task_classes = getattr(room_package, "TASK_CLASSES", None)
        if root_task_classes is not task_classes:
            definition_issues.append(
                f"{package_name}: root TASK_CLASSES must reference "
                f"{tasks.__name__}.TASK_CLASSES"
            )
        exported_names = getattr(tasks, "__all__", None)
        if not isinstance(exported_names, (list, tuple)):
            definition_issues.append(
                f"{tasks.__name__}: __all__ must be an ordered list or tuple"
            )
        elif list(exported_names) != expected_names:
            definition_issues.append(
                f"{tasks.__name__}: __all__ must exactly follow TASK_CLASSES "
                f"(expected={expected_names}, actual={list(exported_names)})"
            )
        root_exports = list(getattr(room_package, "__all__", ()))
        root_task_exports = [
            name for name in root_exports if name in set(expected_names)
        ]
        if root_task_exports != expected_names:
            definition_issues.append(
                f"{package_name}: root task exports must follow TASK_CLASSES "
                f"(expected={expected_names}, actual={root_task_exports})"
            )
        for cls in task_classes:
            name = cls.__name__
            if getattr(tasks, name, None) is not cls:
                definition_issues.append(
                    f"{tasks.__name__}: {name} is not bound to its TASK_CLASSES class"
                )
            if getattr(room_package, name, None) is not cls:
                definition_issues.append(
                    f"{package_name}: {name} is not re-exported as its catalog class"
                )
            scene_ids = tuple(
                int(scene_id)
                for scene_id in (getattr(cls, "SUPPORTED_SCENE_IDS", ()) or ())
            )
            entries.append(
                TaskCatalogEntry(
                    package=package_name,
                    name=name,
                    cls=cls,
                    scene_ids=scene_ids,
                    family=str(getattr(cls, "TASK_FAMILY", "unknown")),
                )
            )
    if definition_issues:
        raise ValueError("invalid task catalog:\n" + "\n".join(definition_issues))
    return entries


def _freeze_signature_value(value):
    if isinstance(value, dict):
        return tuple(
            (key, _freeze_signature_value(item)) for key, item in sorted(value.items())
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_signature_value(item) for item in value)
    try:
        hash(value)
    except TypeError:
        return repr(value)
    return value


def _task_signature(entry, scene_id):
    """Return a language-independent exact physical task signature."""

    scene_name = f"FloorPlan{scene_id}"
    raw_spec = getattr(entry.cls, "SCENE_SPECS", {}).get(scene_name, {}) or {}
    physical_spec = {key: value for key, value in raw_spec.items() if key != "lang"}
    attribute_names = (
        "BEHAVIOR",
        "COMPOSITE_KIND",
        "COMPOSITE_ORDERING",
        "TARGET_STABLE_ID",
        "TARGET_CATEGORY",
        "TARGET_CATEGORIES",
        "TARGET_ASSET_ID",
        "TARGET_JOINT_ROLES",
        "TARGET_JOINT_LEVELS",
        "SOURCE_NAME",
        "SOURCE_CATEGORY",
        "SOURCE_GROUP",
        "SOURCE_MJCF_PATH",
        "SOURCE_MODEL",
        "SOURCE_XML",
        "TARGET_NAME",
        "TARGET_GROUP",
        "TARGET_MJCF_PATH",
        "TARGET_MODEL",
        "TARGET_XML",
        "RECEPTACLE_NAME",
        "RECEPTACLE_GROUP",
        "RECEPTACLE_MJCF_PATH",
        "RECEPTACLE_MODEL",
        "RECEPTACLE_XML",
        "ITEM_NAMES",
        "CUP_NAMES",
        "WORK_SURFACE_FIXTURES",
        "WORK_SURFACE_REGIONS",
    )
    attributes = []
    for name in attribute_names:
        value = getattr(entry.cls, name, None)
        if value is None or (isinstance(value, (list, tuple, dict)) and not value):
            continue
        attributes.append((name, _freeze_signature_value(value)))
    if not physical_spec and not attributes:
        return None
    return (
        entry.family,
        scene_id,
        _freeze_signature_value(physical_spec),
        tuple(attributes),
    )


def validate_task_catalog(entries):
    """Return all catalog, registration, calibration, and duplicate issues."""

    issues = []
    exported = {}
    registries = {}
    package_names = {entry.package for entry in entries}
    for entry in entries:
        if entry.name in exported:
            issues.append(
                f"duplicate public task name {entry.name}: "
                f"{exported[entry.name].package}, {entry.package}"
            )
        exported[entry.name] = entry
        if not isinstance(entry.cls, type):
            issues.append(f"{entry.name}: public export is not a class")
            continue
        if entry.cls.__name__ != entry.name:
            issues.append(f"{entry.name}: export points to class {entry.cls.__name__}")
        if not entry.cls.__module__.startswith(f"{entry.package}.tasks"):
            issues.append(
                f"{entry.name}: class is defined outside {entry.package}.tasks"
            )
        if entry.family == "unknown":
            issues.append(f"{entry.name}: missing TASK_FAMILY")
        if REGISTERED_ENVS.get(entry.name) is not entry.cls:
            issues.append(f"{entry.name}: not registered in robosuite")
        room_registry = getattr(entry.cls, "REGISTRY", None)
        if not isinstance(room_registry, dict):
            issues.append(f"{entry.name}: not registered in its room registry")
        else:
            registries.setdefault(id(room_registry), room_registry)
            if room_registry.get(entry.name) is not entry.cls:
                issues.append(f"{entry.name}: not registered in its room registry")

        raw_scene_ids = getattr(entry.cls, "SUPPORTED_SCENE_IDS", None)
        if not isinstance(raw_scene_ids, tuple):
            issues.append(f"{entry.name}: SUPPORTED_SCENE_IDS must be a tuple")
        if len(set(entry.scene_ids)) != len(entry.scene_ids):
            issues.append(f"{entry.name}: SUPPORTED_SCENE_IDS contains duplicates")
        if not entry.scene_ids:
            issues.append(f"{entry.name}: missing SUPPORTED_SCENE_IDS")
        specs = getattr(entry.cls, "SCENE_SPECS", None)
        uses_scene_specs = issubclass(entry.cls, SceneSpecMixin)
        if uses_scene_specs and not isinstance(specs, dict):
            issues.append(f"{entry.name}: SCENE_SPECS must be a dict")
        if isinstance(specs, dict) and (uses_scene_specs or specs):
            expected_names = [f"FloorPlan{scene_id}" for scene_id in entry.scene_ids]
            actual_names = list(specs)
            if expected_names != actual_names:
                issues.append(
                    f"{entry.name}: scene specs must follow supported scenes "
                    f"(expected={expected_names}, actual={actual_names})"
                )
            for scene_name, spec in specs.items():
                if not isinstance(spec, dict):
                    issues.append(f"{entry.name}/{scene_name}: spec is not a dict")
                elif uses_scene_specs and not spec.get("lang"):
                    issues.append(f"{entry.name}/{scene_name}: missing task language")
        room_type = getattr(entry.cls, "ROOM_TYPE", None)
        for scene_id in entry.scene_ids:
            if room_type is not None and classify_floorplan(scene_id) != room_type:
                issues.append(
                    f"{entry.name}: scene {scene_id} is outside its room type"
                )

    public_names = set(exported)
    final_classes = {entry.cls: entry for entry in entries}
    for entry in entries:
        inherited_finals = [
            final_classes[base]
            for base in entry.cls.__mro__[1:]
            if base in final_classes
        ]
        for inherited in inherited_finals:
            issues.append(
                f"{entry.name}: public task must not inherit public task "
                f"{inherited.name}"
            )
    reported_hidden = set()
    for room_registry in registries.values():
        for name, cls in room_registry.items():
            module = getattr(cls, "__module__", "")
            in_task_package = any(
                module.startswith(f"{package_name}.tasks")
                for package_name in package_names
            )
            if not in_task_package:
                continue
            if name.startswith("_"):
                issues.append(f"{name}: private task class is registered")
            elif name not in public_names:
                issues.append(f"{name}: registered task is hidden from tasks.__all__")
            if name.startswith("_") or name not in public_names:
                reported_hidden.add((name, id(cls)))

    # Room metaclass registries are canonical, but the global robosuite registry
    # is independently mutable. Reverse-scan both so an obsolete alias cannot
    # remain constructible while being absent from the public room catalog.
    for name, cls in REGISTERED_ENVS.items():
        module = getattr(cls, "__module__", "")
        if not any(
            module.startswith(f"{package_name}.tasks") for package_name in package_names
        ):
            continue
        if (name.startswith("_") or name not in public_names) and (
            name,
            id(cls),
        ) not in reported_hidden:
            issues.append(
                f"{name}: globally registered task is hidden from tasks.__all__"
            )

    signatures = {}
    for entry in entries:
        for scene_id in entry.scene_ids:
            signature = _task_signature(entry, scene_id)
            if signature is None:
                issues.append(
                    f"{entry.name}:{scene_id}: missing physical task signature"
                )
                continue
            previous = signatures.get(signature)
            if previous is not None:
                issues.append(
                    f"duplicate physical task: {previous} == "
                    f"{entry.name}:{scene_id} -> {signature}"
                )
            else:
                signatures[signature] = f"{entry.name}:{scene_id}"
    counts_by_package = {}
    for entry in entries:
        counts_by_package[entry.package] = counts_by_package.get(entry.package, 0) + 1
    for package, expected in EXPECTED_TASK_COUNTS.items():
        if package in counts_by_package and counts_by_package[package] != expected:
            issues.append(
                f"{package}: expected {expected} catalog tasks, "
                f"found {counts_by_package[package]}"
            )
    matrix = task_counts(entries)
    for package, expected in EXPECTED_TASK_MATRIX.items():
        if package in matrix and matrix[package] != expected:
            issues.append(
                f"{package}: task matrix changed "
                f"(expected={expected}, actual={matrix[package]})"
            )
    return issues


def task_counts(entries):
    counts = {}
    for entry in entries:
        bucket = "composite" if entry.family == "composite" else "atomic"
        for scene_id in entry.scene_ids:
            room = counts.setdefault(entry.package, {})
            scene = room.setdefault(scene_id, {"atomic": 0, "composite": 0})
            scene[bucket] += 1
    return counts


__all__ = [
    "BATHROOM_PACKAGE",
    "BEDROOM_PACKAGE",
    "EXPECTED_TASK_COUNTS",
    "EXPECTED_TASK_MATRIX",
    "TaskCatalogEntry",
    "discover_task_catalog",
    "task_counts",
    "validate_task_catalog",
]
