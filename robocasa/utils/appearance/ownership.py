"""Which assets in a MERGED model xml belong to the SCENE, and which belong to
the robot or the task objects.

The appearance randomizer runs on ``env.model.get_xml()`` -- the whole compiled
model, not just the room -- so "don't re-texture the apple the policy is being
asked to pick up, or the robot holding it" has to be a rule, not an accident of
which file we happened to open.

The rule is structural. ``MujocoXMLModel.__init__`` calls
``add_prefix(root=self.root, prefix=self.naming_prefix)`` over its WHOLE tree,
``<asset>`` included (robosuite ``models/base.py``), so every texture,
material and mesh a robot / gripper / task object contributes is named
``robot0_...`` / ``gripper0_right_...`` / ``obj_...``. An ``Arena`` (both
``KitchenArena`` and ``MolmoSpacesArena``) is a plain ``MujocoXML`` and gets no
prefix, and so do robocasa's kitchen fixtures -- which is exactly right: walls,
floors and cabinets ARE the scene and should be re-textured.

Measured on this repo (PickPlaceCounterToCabinet / IThorKitchenApproachFridge,
G1FloatingBody): kitchen 96 merged textures -> 5 protected (``obj_image0``,
``distr_cab_*``, ``distr_counter_*``); iTHOR 44 merged textures -> 0 protected.
"""

from __future__ import annotations


def protected_prefixes(env) -> tuple[str, ...]:
    """Naming prefixes of everything in the merged xml that is NOT scene.

    Robots, their grippers/hands, and the task objects (``env.objects`` --
    which for a robocasa kitchen holds ONLY the task objects; the fixtures live
    in ``env.fixtures`` and in ``env.model.mujoco_objects``, so neither of
    those is a valid source here).
    """
    out: set[str] = set()
    for robot in getattr(env, "robots", None) or []:
        model = getattr(robot, "robot_model", None)
        prefix = getattr(model, "naming_prefix", None)
        if prefix:
            out.add(prefix)
        grippers = getattr(robot, "gripper", None) or {}
        for gripper in (grippers.values() if hasattr(grippers, "values")
                        else grippers):
            prefix = getattr(gripper, "naming_prefix", None)
            if prefix:
                out.add(prefix)
    for obj in (getattr(env, "objects", None) or {}).values():
        prefix = getattr(obj, "naming_prefix", None)
        if prefix:
            out.add(prefix)
    return tuple(sorted(out))


def is_protected(name: str | None, prefixes: tuple[str, ...]) -> bool:
    """Whether an asset name belongs to the robot or a task object."""
    if not name or not prefixes:
        return False
    return name.startswith(prefixes)
