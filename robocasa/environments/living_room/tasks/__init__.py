"""
Living-room tasks.

Task classes subclass ``LivingRoomEnv`` (see ``../living_room.py``) and override
the task hooks: ``_get_furniture()`` (scene), ``_get_obj_cfgs()`` (manipulable
objects), ``_setup_furniture_references()`` (semantic refs), ``_setup_scene()``
(per-reset setup), ``_check_success()`` and ``get_ep_meta()`` (success + language).

This mirrors the kitchen ``atomic/`` + ``composite/`` split. Add single-skill
primitives and multi-step activities here as the benchmark grows, then import
them in ``robocasa/__init__.py`` so ``EnvMeta`` registers them.
"""
