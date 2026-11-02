"""Shared metadata tags for iTHOR / ProcTHOR atomic tasks.

The tags are intentionally lightweight: they describe the benchmark taxonomy
without changing scene construction, object placement, or success checks.
"""


def add_atomic_task_metadata(env, ep_meta):
    """Attach THOR atomic-task taxonomy fields to episode metadata."""
    atomic_skill = getattr(env, "ATOMIC_SKILL", None)
    if atomic_skill is not None:
        ep_meta["atomic_skill"] = atomic_skill

    difficulty_level = getattr(env, "DIFFICULTY_LEVEL", None)
    if difficulty_level is not None:
        ep_meta["difficulty_level"] = difficulty_level

    task_family = getattr(env, "TASK_FAMILY", None)
    if task_family is not None:
        ep_meta["task_family"] = task_family

    scene_family = getattr(env, "SCENE_FAMILY", None)
    if scene_family is not None:
        ep_meta["scene_family"] = scene_family

    return ep_meta
