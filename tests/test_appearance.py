"""Pure-XML tests for robocasa.utils.appearance (no simulator needed).

Covers the two things the unified randomizer has to get right and the old
scene-xml-only implementation never had to: it runs on the MERGED model, so
(a) robot / task-object assets must be left alone, and (b) a kitchen's shared
surface textures must be swapped per SURFACE, not per xml element.
"""

import xml.etree.ElementTree as ET

import numpy as np
import pytest

from robocasa.utils.appearance import (
    KITCHEN_PROFILE,
    LIVING_ROOM_PROFILE,
    THOR_PROFILE,
    AppearanceConfig,
    UnsupportedScene,
    is_protected,
    profile_for,
    randomize_xml,
)
from robocasa.utils.appearance.config import GENERATIVE_TEXTURES_DIR

GEN = GENERATIVE_TEXTURES_DIR


def _kitchen_xml():
    """A merged-model shape, with robocasa's real naming: four wall panels and
    their backings, a floor, one shared counter-top texture, cube + 2d cabinet
    entries, a handle texture shared across cabinetry (must be vetoed), an
    appliance mesh texture, and a task object + robot asset."""
    walls = "".join(
        f'<texture name="wall_{i}_room_wall" type="2d" '
        f'file="/assets/textures/bricks/red_bricks.png"/>'
        f'<material name="wall_{i}_room_wall_mat" texture="wall_{i}_room_wall" '
        f'texrepeat="3 3"/>' for i in range(4))
    return f"""<mujoco><asset>
      {walls}
      <texture name="wall_backing_room_wall" type="2d" file="/assets/flat/gray.png"/>
      <material name="wall_backing_room_wall_mat" texture="wall_backing_room_wall"/>
      <texture name="floor_room_wall" type="2d" file="/assets/tiles/concrete.png"/>
      <material name="floor_room_wall_mat" texture="floor_room_wall" texrepeat="0.5 0.5"/>
      <texture name="counter/tex008_2d" type="2d" file="{GEN}/counter/tex008.png"/>
      <material name="counter_main_group_counter_top" texture="counter/tex008_2d"/>
      <texture name="flat/gray" type="cube" file="/assets/flat/gray.png"/>
      <material name="counter_main_group_counter_base" texture="flat/gray"/>
      <material name="cab_1_main_group_mat" texture="flat/gray"/>
      <texture name="flat/gray_2d" type="2d" file="/assets/flat/gray.png"/>
      <material name="cab_1_main_group_door_mat" texture="flat/gray_2d"/>
      <texture name="Handle_metals/bright_metal" type="2d" file="/assets/metal.png"/>
      <material name="cab_1_main_group_door_handle_mat" texture="Handle_metals/bright_metal"/>
      <texture name="sink_left_group_1_Body001" type="2d" file="/assets/sinks/T.png"/>
      <material name="sink_left_group_1_Body001" texture="sink_left_group_1_Body001"/>
      <texture name="obj_image0" type="2d" file="/assets/objects/apple.png"/>
      <texture name="robot0_skin" type="2d" file="/assets/robots/g1.png"/>
      <material name="obj_Scene" texture="obj_image0" rgba="1 0 0 1"/>
      <material name="robot0_mat" texture="robot0_skin"/>
    </asset><worldbody>
      <light pos="2 -2 2" dir="0 0 -1" diffuse="0.3 0.3 0.3"/>
      <body name="obj_main"><geom name="obj_g" size="0.03" pos="1 2 3"/></body>
    </worldbody></mujoco>"""


def _thor_xml():
    return """<mujoco><asset>
      <texture name="Drywall3" type="2d" file="/thor/WallDrywallWhite.png"/>
      <texture name="FloorPlan13" type="2d" file="/thor/FloorPlan13.png"/>
      <texture name="Oven_Mat" type="2d" file="/thor/Oven.png"/>
      <texture name="Apple1_Mat5" type="2d" file="/thor/Apple1.png"/>
      <material name="material_Drywall3" texture="Drywall3"/>
    </asset><worldbody>
      <light pos="1 -1 1.5" dir="-0.5 0.5 -0.5" diffuse="0.5 0.5 0.5"/>
    </worldbody></mujoco>"""


def _tex(xml):
    return {t.get("name"): t.get("file")
            for t in ET.fromstring(xml).iter("texture")}


PROTECTED = ("gripper0_left_", "obj_", "robot0_")


# --------------------------------------------------------------------------- #
# Ownership

def test_is_protected_matches_only_prefixes():
    assert is_protected("obj_image0", PROTECTED)
    assert is_protected("robot0_skin", PROTECTED)
    assert not is_protected("wall_0_room_wall", PROTECTED)
    assert not is_protected(None, PROTECTED)
    assert not is_protected("obj_image0", ())


def test_robot_and_task_object_assets_are_never_touched():
    cfg = AppearanceConfig(texture_frac=1.0, texture_exclude=(),
                           material_rgba_jitter=0.5, material_pbr_jitter=0.5)
    before = _tex(_kitchen_xml())
    out = randomize_xml(_kitchen_xml(), np.random.default_rng(0), cfg,
                        KITCHEN_PROFILE, PROTECTED)
    after = _tex(out)
    for name in ("obj_image0", "robot0_skin"):
        assert after[name] == before[name], f"{name} was re-textured"
    mats = {m.get("name"): m for m in ET.fromstring(out).iter("material")}
    assert mats["obj_Scene"].get("rgba") == "1 0 0 1"
    assert mats["obj_Scene"].get("shininess") is None
    assert mats["wall_0_room_wall_mat"].get("shininess") is not None


def test_geometry_is_untouched():
    cfg = AppearanceConfig(texture_frac=1.0, texture_exclude=(),
                           light_jitter=0.3, light_geom_jitter=1.0,
                           material_pbr_jitter=0.3)
    out = randomize_xml(_kitchen_xml(), np.random.default_rng(3), cfg,
                        KITCHEN_PROFILE, PROTECTED)
    geom = next(ET.fromstring(out).iter("geom"))
    assert geom.get("pos") == "1 2 3" and geom.get("size") == "0.03"


# --------------------------------------------------------------------------- #
# Kitchen classifier

def test_kitchen_groups_by_surface_category():
    cfg = AppearanceConfig(texture_frac=1.0, texture_exclude=())
    report: dict = {}
    out = randomize_xml(_kitchen_xml(), np.random.default_rng(1), cfg,
                        KITCHEN_PROFILE, PROTECTED, report)
    after = _tex(out)
    # four wall panels -> ONE new wall texture, not four different ones
    walls = {after[f"wall_{i}_room_wall"] for i in range(4)}
    assert len(walls) == 1
    # cube and 2d cabinet entries are one surface, so they share a draw
    assert after["flat/gray"] == after["flat/gray_2d"]
    # ... and every replacement comes from its own category directory
    assert f"{GEN}/wall/" in walls.pop()
    assert f"{GEN}/floor/" in after["floor_room_wall"]
    assert f"{GEN}/counter/" in after["counter/tex008_2d"]
    assert f"{GEN}/cabinet/" in after["flat/gray"]
    assert report["textures"]["groups"] == 4  # wall/floor/counter/cabinet


def test_kitchen_leaves_appliance_backing_and_handle_textures_alone():
    cfg = AppearanceConfig(texture_frac=1.0, texture_exclude=())
    before, out = _tex(_kitchen_xml()), randomize_xml(
        _kitchen_xml(), np.random.default_rng(2), cfg, KITCHEN_PROFILE, PROTECTED)
    after = _tex(out)
    for name in ("sink_left_group_1_Body001", "wall_backing_room_wall",
                 "Handle_metals/bright_metal"):
        assert after[name] == before[name], name


def test_kitchen_fixups_relay_the_new_textures():
    cfg = AppearanceConfig(texture_frac=1.0, texture_exclude=())
    out = randomize_xml(_kitchen_xml(), np.random.default_rng(1), cfg,
                        KITCHEN_PROFILE, PROTECTED)
    mats = {m.get("name"): m for m in ET.fromstring(out).iter("material")}
    assert mats["floor_room_wall_mat"].get("texrepeat") == "2 2"
    assert mats["wall_0_room_wall_mat"].get("texrepeat") == "3 3"
    # untouched surfaces keep whatever the style set
    assert mats["wall_backing_room_wall_mat"].get("texrepeat") is None


def test_kitchen_exclude_pins_a_surface():
    cfg = AppearanceConfig(texture_frac=1.0, texture_exclude=("floor",))
    before, out = _tex(_kitchen_xml()), randomize_xml(
        _kitchen_xml(), np.random.default_rng(4), cfg, KITCHEN_PROFILE, PROTECTED)
    after = _tex(out)
    assert after["floor_room_wall"] == before["floor_room_wall"]
    assert after["wall_0_room_wall"] != before["wall_0_room_wall"]


# --------------------------------------------------------------------------- #
# MolmoSpaces classifier

def test_thor_swaps_per_element_and_honours_exclude():
    cfg = AppearanceConfig(texture_frac=1.0, texture_exclude=("Oven",))
    report: dict = {}
    out = randomize_xml(_thor_xml(), np.random.default_rng(0), cfg,
                        THOR_PROFILE, (), report)
    after = _tex(out)
    assert after["Oven_Mat"] == "/thor/Oven.png"
    assert report["textures"]["groups"] == 3
    swapped = {after[n] for n in ("Drywall3", "FloorPlan13", "Apple1_Mat5")}
    assert all(str(GEN) in s for s in swapped)
    # independent draws, not one shared group
    assert len(swapped) > 1


# --------------------------------------------------------------------------- #
# Determinism / profile selection

@pytest.mark.parametrize("profile,xml", [(KITCHEN_PROFILE, _kitchen_xml()),
                                         (THOR_PROFILE, _thor_xml())])
def test_same_seed_same_xml(profile, xml):
    cfg = AppearanceConfig(texture_frac=0.7, texture_exclude=(),
                           light_jitter=0.2, light_geom_jitter=1.0,
                           extra_lights=2, material_pbr_jitter=0.2)
    a = randomize_xml(xml, np.random.default_rng(11), cfg, profile, PROTECTED)
    b = randomize_xml(xml, np.random.default_rng(11), cfg, profile, PROTECTED)
    c = randomize_xml(xml, np.random.default_rng(12), cfg, profile, PROTECTED)
    assert a == b and a != c


def test_extra_lights_fit_under_the_budget():
    cfg = AppearanceConfig(texture_frac=0.0, light_geom_jitter=1.0,
                           extra_lights=3)
    out = randomize_xml(_thor_xml(), np.random.default_rng(0), cfg,
                        THOR_PROFILE, ())
    lights = list(ET.fromstring(out).iter("light"))
    assert 1 <= len(lights) <= THOR_PROFILE.max_lights


def test_profile_for_rejects_an_unknown_family():
    class NotAScene:
        pass

    with pytest.raises(UnsupportedScene):
        profile_for(NotAScene())


def test_profile_defaults_differ_by_family():
    assert THOR_PROFILE.default_exclude == ("Oven",)
    assert KITCHEN_PROFILE.default_exclude == ()


# --------------------------------------------------------------------------- #
# Living room (FloorArena)

def _living_room_xml():
    """A FloorArena as LivingRoomArena leaves it: one floor plane and one wall
    shell, both textured by the style, plus a furniture mesh texture."""
    return """<mujoco><asset>
      <texture type="skybox" builtin="gradient" width="256" height="256"/>
      <texture name="texplane" type="2d" file="/assets/textures/wood-tiles.png"/>
      <texture name="wallplane" type="2d" file="/assets/textures/cream-plaster.png"/>
      <texture name="sofa_main_T_BC001" type="2d" file="/assets/fixtures/sofas/T.png"/>
      <material name="floorplane" texture="texplane" texrepeat="10 10" texuniform="true"/>
      <material name="walls_mat" texture="wallplane" texrepeat="3 3"/>
      <material name="sofa_main_T_BC001" texture="sofa_main_T_BC001"/>
    </asset><worldbody>
      <light pos="1 1 1.5" dir="-0.2 -0.2 -1" specular="0.3 0.3 0.3"/>
    </worldbody></mujoco>"""


def test_living_room_swaps_only_floor_and_wall():
    cfg = AppearanceConfig(texture_frac=1.0, texture_exclude=())
    report: dict = {}
    before = _tex(_living_room_xml())
    out = randomize_xml(_living_room_xml(), np.random.default_rng(0), cfg,
                        LIVING_ROOM_PROFILE, (), report)
    after = _tex(out)
    assert report["textures"]["groups"] == 2
    assert f"{GEN}/floor/" in after["texplane"]
    assert f"{GEN}/wall/" in after["wallplane"]
    # furniture meshes and the builtin skybox stay put
    assert after["sofa_main_T_BC001"] == before["sofa_main_T_BC001"]


def test_living_room_relays_the_floor():
    """A FloorArena floor ships texrepeat 10, tuned for tight tiles; a
    room-scale pool image has to be relaid or it renders as noise."""
    cfg = AppearanceConfig(texture_frac=1.0, texture_exclude=())
    out = randomize_xml(_living_room_xml(), np.random.default_rng(0), cfg,
                        LIVING_ROOM_PROFILE, ())
    mats = {m.get("name"): m for m in ET.fromstring(out).iter("material")}
    assert mats["floorplane"].get("texrepeat") == "2 2"
    assert mats["walls_mat"].get("texrepeat") == "3 3"


# --------------------------------------------------------------------------- #
# Profile-dependent defaults

def test_bare_lights_token_takes_the_family_default():
    from robocasa.utils.scene_randomization import parse_randomize_spec

    bare = parse_randomize_spec("lights")
    assert bare.appearance_config(THOR_PROFILE).light_jitter == 0.15
    assert bare.appearance_config(KITCHEN_PROFILE).light_jitter == 0.08
    assert bare.appearance_config(LIVING_ROOM_PROFILE).light_jitter == 0.08
    # ... and the intensity swing follows it
    assert bare.appearance_config(THOR_PROFILE).light_intensity_jitter == 0.3


def test_explicit_lights_value_beats_the_family_default():
    from robocasa.utils.scene_randomization import parse_randomize_spec

    spec = parse_randomize_spec("lights=0.15")
    assert spec.appearance_config(KITCHEN_PROFILE).light_jitter == 0.15
    assert parse_randomize_spec("all").appearance_config(
        KITCHEN_PROFILE).light_jitter == 0.08


# --------------------------------------------------------------------------- #
# Spec ergonomics: presets, new tokens, the resolved banner

def test_presets_expand_in_place_and_lose_to_later_tokens():
    from robocasa.utils.scene_randomization import PRESETS, parse_randomize_spec

    assert set(PRESETS) == {"off", "mild", "standard", "stress"}
    std = parse_randomize_spec("standard")
    assert (std.textures, std.lights, std.camera, std.pbr) == (0.7, 0.15, 1.0, 0.2)
    assert std.lens is None and std.distractors == 0
    # a preset is a token, so it composes and can be overridden
    assert parse_randomize_spec("standard,lens").lens == 1.0
    assert parse_randomize_spec("standard,textures=0.4").textures == 0.4
    assert parse_randomize_spec("textures=0.4,standard").textures == 0.7  # order wins
    assert parse_randomize_spec("off").describe() == {}
    stress = parse_randomize_spec("stress")
    assert stress.distractors == 3 and stress.robot == 0.15 and stress.lens == 1.0


def test_new_tokens_cover_what_the_replay_flags_used_to():
    from robocasa.utils.scene_randomization import parse_randomize_spec

    cfg = parse_randomize_spec("rgba=0.3,light-geom,extra-lights=3").appearance_config(
        THOR_PROFILE)
    assert cfg.material_rgba_jitter == 0.3
    assert cfg.extra_lights == 3
    # light-geom alone still implies a single extra spot
    assert parse_randomize_spec("light-geom").appearance_config(
        THOR_PROFILE).extra_lights == 1


def test_resolved_banner_shows_the_family_resolved_values():
    from robocasa.utils.scene_randomization import parse_randomize_spec

    spec = parse_randomize_spec("standard", texture_split="train")
    thor = spec.format_resolved("standard", THOR_PROFILE, seed=0)
    kitchen = spec.format_resolved("standard", KITCHEN_PROFILE, seed=0)
    assert "profile=molmospaces" in thor and "lights=0.15" in thor
    assert "exclude=['Oven']" in thor
    assert "profile=kitchen" in kitchen and "lights=0.08" in kitchen
    assert "exclude=" not in kitchen          # kitchen protects nothing by default
    assert "textures=0.7(train)" in thor
    assert "nothing enabled" in parse_randomize_spec("off").format_resolved("off")


def test_help_text_lists_every_token_and_preset():
    from robocasa.utils.scene_randomization import (
        PRESETS, TOKEN_HELP, randomize_help)

    text = randomize_help()
    for token, _ in TOKEN_HELP:
        assert token in text
    for name in PRESETS:
        assert name in text


def test_reroll_period_buckets_episodes():
    from robocasa.scripts.replay_augment import appearance_seed_for, parse_reroll

    # one number covers every case; 0 is "off", as it is for every other knob
    assert parse_reroll("0") is None
    assert parse_reroll("1") == 1
    assert parse_reroll("4") == 4
    seeds = lambda scope: [appearance_seed_for(parse_reroll(scope), 3, e)
                           for e in range(8)]
    assert len(set(seeds("0"))) == 1                # one room for the whole run
    assert len(set(seeds("1"))) == 8                # a fresh one every episode
    assert seeds("4") == [30000] * 4 + [30001] * 4  # blocks of four
    with pytest.raises(SystemExit):
        parse_reroll("every-other")
    with pytest.raises(SystemExit):
        parse_reroll("-1")
