from __future__ import annotations

import os
import shutil
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import mujoco
import numpy as np
from loguru import logger

from sim2real.config.robots.base import RobotCfg
from sim2real.utils.strings import resolve_matching_names_values


VIEWER_VISUAL_XML = """\
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
"""

VIEWER_ASSET_XML = """\
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
"""

VIEWER_LIGHT_XML = """\
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
"""

VIEWER_FLOOR_XML = """\
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""

VIEWER_WORLDBODY_XML = VIEWER_LIGHT_XML + VIEWER_FLOOR_XML


def _resolve_scene_xml_path(scene_xml: str | Path) -> Path:
    scene_path = Path(scene_xml).expanduser()
    if not scene_path.is_absolute():
        scene_path = scene_path.absolute()
    if not scene_path.is_file():
        raise FileNotFoundError(f"Extra scene MJCF not found: {scene_path}")
    return scene_path


def _scene_defines_ground(scene_path: Path) -> bool:
    """Whether the extra scene brings its own ground plane.

    If it does, the built-in ``floor`` geom is skipped so the two do not
    collide on the ``floor`` name (MuJoCo rejects duplicate geom names).
    """
    return 'type="plane"' in scene_path.read_text(encoding="utf-8")


def _inject_floor_scene_xml(xml_text: str, scene_path: Path | None = None) -> str:
    if "<visual>" not in xml_text:
        insertion_point = xml_text.find("<asset>")
        if insertion_point < 0:
            raise ValueError("Expected <asset> block in MJCF")
        xml_text = xml_text[:insertion_point] + VIEWER_VISUAL_XML + xml_text[insertion_point:]

    asset_close = xml_text.find("</asset>")
    if asset_close < 0:
        raise ValueError("Expected </asset> block in MJCF")
    xml_text = xml_text[:asset_close] + VIEWER_ASSET_XML + xml_text[asset_close:]

    worldbody_xml = VIEWER_WORLDBODY_XML
    if scene_path is not None and _scene_defines_ground(scene_path):
        worldbody_xml = VIEWER_LIGHT_XML

    worldbody_close = xml_text.find("</worldbody>")
    if worldbody_close < 0:
        raise ValueError("Expected </worldbody> block in MJCF")
    xml_text = xml_text[:worldbody_close] + worldbody_xml + xml_text[worldbody_close:]

    if scene_path is not None:
        mujoco_close = xml_text.rfind("</mujoco>")
        if mujoco_close < 0:
            raise ValueError("Expected </mujoco> root element in MJCF")
        include_xml = f'  <include file="{scene_path}"/>\n'
        xml_text = xml_text[:mujoco_close] + include_xml + xml_text[mujoco_close:]

    return xml_text


@contextmanager
def _temp_scene_with_floor(mjcf_path: Path, scene_path: Path | None = None) -> Path:
    source_path = Path(mjcf_path).expanduser()
    if not source_path.is_absolute():
        source_path = source_path.absolute()
    xml_text = source_path.read_text(encoding="utf-8")
    viewer_xml = _inject_floor_scene_xml(xml_text, scene_path)

    temp_path: Path | None = None
    staging_dir: Path | None = None
    try:
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".xml",
                prefix=".sim_scene_",
                dir=source_path.parent,
                delete=False,
                encoding="utf-8",
            ) as tmp:
                tmp.write(viewer_xml)
                temp_path = Path(tmp.name)
        except OSError:
            staging_dir = Path(tempfile.mkdtemp(prefix=".sim_scene_"))
            for child in source_path.parent.iterdir():
                if child.name == source_path.name:
                    continue
                os.symlink(child, staging_dir / child.name, target_is_directory=child.is_dir())
            temp_path = staging_dir / source_path.name
            temp_path.write_text(viewer_xml, encoding="utf-8")

        yield temp_path
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink()
            except FileNotFoundError:
                pass
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)


def _add_joint_motor_actuator(
    spec: mujoco.MjSpec,
    *,
    joint_name: str,
    effort_limit: float,
    gear: float = 1.0,
) -> None:
    actuator = spec.add_actuator(name=joint_name, target=joint_name)
    actuator.trntype = mujoco.mjtTrn.mjTRN_JOINT
    actuator.dyntype = mujoco.mjtDyn.mjDYN_NONE
    actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
    actuator.biastype = mujoco.mjtBias.mjBIAS_NONE
    actuator.gear[0] = float(gear)
    actuator.forcelimited = True
    actuator.forcerange[:] = np.array([-effort_limit, effort_limit], dtype=np.float64)
    actuator.ctrllimited = True
    actuator.ctrlrange[:] = np.array([-effort_limit, effort_limit], dtype=np.float64)
    # NOTE: armature/frictionloss deliberately NOT set here. They used to be,
    # which made the override silently conditional on the joint happening to
    # lack an actuator -- true for the old hand-less MJCF (nu=0), false for the
    # unified model (nu=43), so switching models would have quietly dropped
    # every per-controller value. See apply_joint_dynamics.


#: The three per-joint quantities that are a *controller's* property rather than
#: the robot's geometry: each backend trained under its own values, and they
#: differ enough to change behaviour outright (injecting only these, a 0.5 s free
#: fall from one state diverges by up to 1.27 rad between backends). They live in
#: ``checkpoints/<backend>/**/policy.yaml`` next to ``joint_kp``/``joint_kd``.
JOINT_DYNAMICS_FIELDS = ("armature", "damping", "frictionloss")

#: Optional lower bound on injected ``armature``, off by default.
#:
#: Kept as a knob because a too-small joint inertia genuinely does hurt MuJoCo's
#: contact solve, but it is NOT what causes the spawn-time foot penetration on
#: this benchmark: measured on IThorKitchenCloseFridge with the pre-existing
#: model and no injection at all, every backend already penetrates by 7.8-10.5mm
#: during the settling window (amo -10.54, heft -8.74, scalebfm -8.34, sonic
#: -7.75). Flooring armature at 0.01 changes that by under 2mm. The cause is the
#: spawn pose itself -- see the note in `_apply_controller_spawn_pose`.
#:
#: Default None so controllers get exactly the values they trained with.
SIM_ARMATURE_FLOOR = None


def robot_cfg_joint_dynamics(robot_cfg: RobotCfg) -> dict[str, Mapping[str, float]]:
    """``RobotCfg``'s own per-joint dynamics, as a fallback source.

    ``joint_armature``/``joint_frictionloss`` there are computed from the motor
    type (reflected inertia through each joint's two-stage planetary gearing),
    so they are the right values for any checkpoint that does not carry its own
    -- e.g. one exported before ``policy.yaml`` grew these keys, or one obtained
    from elsewhere. Without a fallback such a checkpoint silently inherits
    whatever the shared geometry MJCF happens to declare, which is a different
    controller's numbers.

    No entry for ``damping``: ``RobotCfg`` has no equivalent, so a config with no
    ``joint_damping`` keeps the MJCF's value.
    """
    return {
        "armature": robot_cfg.joint_armature,
        "frictionloss": robot_cfg.joint_frictionloss,
    }


def _resolve_joint_dynamics(
    joint_dynamics: Mapping[str, Any],
    joint_names: Sequence[str],
    armature_floor: float | None = SIM_ARMATURE_FLOOR,
    fallback: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """``{field: {joint_name: value}}`` for whichever fields the config defines.

    Keys are regexes resolved the same way ``joint_kp``/``joint_kd`` are
    (``resolve_matching_names_values``), so a config written for one robot's
    joint set degrades to "leave the rest at the XML's value" rather than
    erroring -- but note that helper rejects a joint matching two patterns, so
    the patterns must be mutually exclusive (no ``.*`` catch-all plus overrides).
    """
    resolved: dict[str, dict[str, float]] = {}
    wanted = set(joint_names)
    for field in JOINT_DYNAMICS_FIELDS:
        table = joint_dynamics.get(f"joint_{field}")
        if not table:
            # No value from this checkpoint: fall back to the robot cfg's own
            # motor-derived table rather than leaving the shared geometry MJCF's
            # numbers in place, which belong to whichever controller that file
            # came from. Keys there are exact joint names, not patterns.
            spare = (fallback or {}).get(field)
            if not spare:
                continue
            resolved[field] = {n: float(v) for n, v in spare.items() if n in wanted}
            if resolved[field]:
                logger.info(
                    "joint_{} absent from the policy config; using RobotCfg's own "
                    "values for {} joint(s)", field, len(resolved[field]),
                )
            else:
                del resolved[field]
                continue
        else:
            indices, _names, values = resolve_matching_names_values(
                table, list(joint_names), preserve_order=True, strict=False
            )
            resolved[field] = {joint_names[i]: float(v) for i, v in zip(indices, values)}
        if field == "armature" and armature_floor is not None:
            raised = {n: v for n, v in resolved[field].items() if v < armature_floor}
            if raised:
                resolved[field].update({n: armature_floor for n in raised})
                logger.info(
                    "armature floored at {} for {} joint(s) (sim contact stability, "
                    "see SIM_ARMATURE_FLOOR): {}",
                    armature_floor, len(raised),
                    {n: round(v, 6) for n, v in list(raised.items())[:4]},
                )
        untouched = [n for n in joint_names if n not in resolved[field]]
        if untouched:
            logger.warning(
                "joint_{} covers {}/{} joints; the rest keep the MJCF's own value: {}",
                field,
                len(resolved[field]),
                len(joint_names),
                untouched,
            )
    return resolved


def apply_joint_dynamics(
    spec: mujoco.MjSpec,
    joint_dynamics: Mapping[str, Any],
    joint_names: Sequence[str],
    *,
    armature_floor: float | None = SIM_ARMATURE_FLOOR,
    fallback: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, int]:
    """Override armature/damping/frictionloss on an uncompiled spec, in place.

    Unconditional by design (contrast the old behaviour described in
    ``_add_joint_motor_actuator``). Returns ``{field: joints_written}``.

    ``armature_floor=None`` writes the controller's values verbatim; see
    ``SIM_ARMATURE_FLOOR`` for why the default is not that.
    """
    written: dict[str, int] = {}
    for field, table in _resolve_joint_dynamics(
        joint_dynamics, joint_names, armature_floor, fallback
    ).items():
        for joint_name, value in table.items():
            setattr(spec.joint(joint_name), field, value)
        written[field] = len(table)
    if written:
        logger.info("Applied per-controller joint dynamics to sim model: {}", written)
    return written


def apply_joint_dynamics_runtime(
    model: mujoco.MjModel,
    joint_dynamics: Mapping[str, Any],
    joint_names: Sequence[str],
    *,
    name_in_model: Callable[[str], str] = lambda name: name,
    armature_floor: float | None = SIM_ARMATURE_FLOOR,
    fallback: Mapping[str, Mapping[str, float]] | None = None,
) -> dict[str, int]:
    """Same override, but onto an already-compiled model.

    This is the form a host that does not own the compile step needs -- robosuite
    builds the scene itself, so by the time the backend is known there is no
    ``MjSpec`` left to edit. ``dof_armature``/``dof_damping``/``dof_frictionloss``
    are plain model arrays, so writing them takes effect without a recompile;
    verified to produce byte-identical arrays to the spec route.

    ``joint_names`` are the CANONICAL names the policy config's patterns are
    written against; ``name_in_model`` maps each to its name in this particular
    model. The two must stay separate: a host that prefixes names (robosuite ->
    ``robot0_``) would otherwise break every literal pattern key, since
    ``waist_pitch_joint`` does not fullmatch ``robot0_waist_pitch_joint`` even
    though ``.*_knee_joint`` happens to.
    """
    written: dict[str, int] = {}
    for field, table in _resolve_joint_dynamics(
        joint_dynamics, joint_names, armature_floor, fallback
    ).items():
        array = getattr(model, f"dof_{field}")
        count = 0
        absent: list[str] = []
        for joint_name, value in table.items():
            model_name = name_in_model(joint_name)
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, model_name)
            if joint_id < 0:
                absent.append(model_name)
                continue
            array[int(model.jnt_dofadr[joint_id])] = value
            count += 1
        if absent:
            # Loud, because the failure is otherwise invisible: a host whose model
            # prefixes joint names (robosuite -> "robot0_...") silently matched
            # nothing and every controller quietly inherited the MJCF's values.
            # Callers must pass names as they exist in THIS model.
            logger.warning(
                "joint_{}: {}/{} joints not found in the model (pass names as they "
                "exist in it, e.g. with the host's naming prefix): {}",
                field, len(absent), len(table), absent[:4],
            )
        written[field] = count
    if written and not any(written.values()):
        raise ValueError(
            "apply_joint_dynamics_runtime matched 0 joints -- the policy config "
            "defines joint dynamics but none of the names exist in this model. "
            "Fix the caller's joint_names rather than running with silently "
            "unmodified dynamics."
        )
    if written:
        logger.info("Applied per-controller joint dynamics to compiled model: {}", written)
    return written


def ensure_joint_motor_actuators(
    spec: mujoco.MjSpec,
    robot_cfg: RobotCfg,
) -> list[str]:
    joint_names_in_spec = {joint.name for joint in spec.joints}
    actuator_names_in_spec = {actuator.name for actuator in spec.actuators}
    added_joint_names: list[str] = []

    for joint_name in robot_cfg.joint_names:
        if joint_name not in joint_names_in_spec:
            continue
        # Actuator naming is not consistent across the G1 MJCFs -- several drop
        # the "_joint" suffix their joints carry. Checking only the exact name
        # made this add a SECOND actuator to every already-actuated joint (58 on
        # a 29-DOF model), so accept either spelling, the same way robocasa's
        # build_joint_mapping and Dex3SimHandBridge do.
        if any(
            candidate in actuator_names_in_spec
            for candidate in (joint_name, joint_name.replace("_joint", ""))
        ):
            continue

        effort_limit = robot_cfg.joint_effort_limit.get(joint_name)
        if effort_limit is None:
            raise KeyError(
                f"Missing joint_effort_limit for joint {joint_name!r}; "
                "cannot auto-create MuJoCo motor actuator."
            )
        _add_joint_motor_actuator(
            spec,
            joint_name=joint_name,
            effort_limit=float(effort_limit),
        )
        actuator_names_in_spec.add(joint_name)
        added_joint_names.append(joint_name)

    return added_joint_names


def load_sim_model(
    robot_cfg: RobotCfg,
    *,
    ground_rgb: tuple[float, float, float] = (0.2, 0.3, 0.4),
    with_dex3_hands: bool = False,
    scene_xml: str | Path | None = None,
    joint_dynamics: Mapping[str, Any] | None = None,
) -> mujoco.MjModel:
    """Compile the shared robot model, optionally specialised for one controller.

    ``joint_dynamics`` is that controller's policy config (or any mapping holding
    ``joint_armature``/``joint_damping``/``joint_frictionloss``). Pass it whenever
    a policy is in the loop: the geometry is shared across backends but these
    three are not. Omit it for a bare viewer/smoke run, which then gets whatever
    the MJCF itself declares.
    """
    # with_dex3_hands selects the hand-ful MJCF rather than splicing hands onto
    # the hand-less one at runtime: both files ship the same official Dex3 bodies,
    # so the splice was re-deriving something already on disk -- and it needed a
    # separate ~100MB reference asset on every machine to do it.
    mjcf_path = robot_cfg.resolve_mjcf_path(with_hands=with_dex3_hands)
    if ground_rgb != (0.2, 0.3, 0.4):
        logger.warning("load_sim_model currently ignores non-default ground_rgb={}", ground_rgb)
    scene_path = _resolve_scene_xml_path(scene_xml) if scene_xml else None
    if scene_path is not None:
        logger.info("Including extra scene MJCF in sim model: {}", scene_path)
    with _temp_scene_with_floor(mjcf_path, scene_path) as scene_mjcf_path:
        spec = mujoco.MjSpec.from_file(str(scene_mjcf_path))
        if joint_dynamics:
            apply_joint_dynamics(
                spec, joint_dynamics, robot_cfg.joint_names,
                fallback=robot_cfg_joint_dynamics(robot_cfg),
            )
        added_joint_names = ensure_joint_motor_actuators(spec, robot_cfg)
        if added_joint_names:
            logger.info(
                "Added {} missing motor actuators to sim model: {}",
                len(added_joint_names),
                ", ".join(added_joint_names),
            )
        return spec.compile()
