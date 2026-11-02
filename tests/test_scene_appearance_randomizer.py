"""Pure-XML tests for scene_appearance_randomizer (no simulator needed)."""

import os

import numpy as np
import pytest
from lxml import etree as ET

from robocasa.models.scenes.molmospaces_scene_registry import get_scene_xml
from robocasa.utils.scene_appearance_randomizer import (
    MAX_SCENE_LIGHTS,
    AppearanceConfig,
    cleanup_variant_xml,
    randomize_scene_xml,
    texture_pool,
    texture_split_of,
)

SCENE = 7  # FloorPlan7 -- the OpenOven collection scene, present in the repo


@pytest.fixture()
def scene_xml():
    return get_scene_xml(SCENE)


def _textures(path):
    return {t.get("name"): t.get("file")
            for t in ET.parse(str(path)).getroot().iter("texture")}


def _lights(path):
    return [dict(l.attrib) for l in ET.parse(str(path)).getroot().iter("light")]


def test_pool_nonempty_and_split_disjoint():
    pool = texture_pool()
    assert len(pool) > 100
    assert all(p.suffix == ".png" for p in pool)
    train, test = texture_pool(split="train"), texture_pool(split="test")
    assert set(train).isdisjoint(test)
    assert len(train) + len(test) == len(pool)
    assert 0.7 < len(train) / len(pool) < 0.9
    # deterministic, name-based: same answer for the same file everywhere
    assert all(texture_split_of(p) == "train" for p in train)


def test_swap_count_exclusions_and_split(scene_xml):
    rng = np.random.default_rng(0)
    cfg = AppearanceConfig(texture_frac=0.7, texture_exclude=("Oven",),
                           texture_split="test")
    report = {}
    out = randomize_scene_xml(scene_xml, rng, cfg, report=report)
    try:
        before, after = _textures(scene_xml), _textures(out)
        assert set(before) == set(after)
        excluded = {n for n, f in before.items()
                    if "oven" in n.lower() or (f and "oven" in f.lower())}
        assert excluded, "FloorPlan7 should have oven textures to protect"
        for name in excluded:
            assert after[name] == before[name], f"excluded texture {name} was swapped"
        candidates = [n for n in before if n not in excluded]
        changed = [n for n in candidates if after[n] != before[n]]
        assert len(changed) == round(0.7 * len(candidates))
        for name in changed:
            assert os.path.isabs(after[name]) and os.path.exists(after[name])
            assert texture_split_of(after[name]) == "test"
        assert set(report["textures"]["swapped"]) == set(changed)
    finally:
        cleanup_variant_xml(out)


def test_deterministic_given_seed(scene_xml):
    cfg = AppearanceConfig(texture_frac=0.5, light_jitter=0.2,
                           light_geom_jitter=1.0, extra_lights=2,
                           material_rgba_jitter=0.1, material_pbr_jitter=0.2)
    outs = []
    for tag in ("a", "b"):
        out = randomize_scene_xml(
            scene_xml, np.random.default_rng(123), cfg,
            out_xml=str(scene_xml).replace(".xml", f".aug_{tag}.xml"))
        outs.append(out)
    try:
        assert outs[0].read_bytes() == outs[1].read_bytes()
    finally:
        for o in outs:
            cleanup_variant_xml(o)


def test_light_intensity_and_shadow_toggle(scene_xml):
    cfg = AppearanceConfig(texture_frac=0.0, light_intensity_jitter=0.3,
                           light_shadow_toggle=1.0)
    report = {}
    out = randomize_scene_xml(scene_xml, np.random.default_rng(3), cfg,
                              report=report)
    try:
        before, after = _lights(scene_xml), _lights(out)
        d0 = np.array([float(x) for x in before[0]["diffuse"].split()])
        d1 = np.array([float(x) for x in after[0]["diffuse"].split()])
        # hue preserved: a pure scale of the original color
        ratio = d1 / d0
        assert np.allclose(ratio, ratio[0], atol=1e-3) and 0.7 <= ratio[0] <= 1.3
        assert after[0]["castshadow"] in ("true", "false")
        assert report["lights"][0]["castshadow"] == after[0]["castshadow"]
    finally:
        cleanup_variant_xml(out)


def test_light_geometry_and_extra_lights(scene_xml):
    cfg = AppearanceConfig(texture_frac=0.0, light_geom_jitter=1.0,
                           extra_lights=10)
    report = {}
    out = randomize_scene_xml(scene_xml, np.random.default_rng(5), cfg,
                              report=report)
    try:
        before, after = _lights(scene_xml), _lights(out)
        # the original light moved / turned, and stays unit-length
        d0 = np.array([float(x) for x in before[0]["dir"].split()])
        d1 = np.array([float(x) for x in after[0]["dir"].split()])
        assert not np.allclose(d0, d1)
        assert abs(np.linalg.norm(d1) - 1.0) < 1e-4
        assert before[0]["pos"] != after[0]["pos"]
        # extra lights are capped by the renderer-safe ceiling
        assert len(after) <= MAX_SCENE_LIGHTS
        assert len(after) - len(before) == report["extra_lights_added"]
        for extra in after[len(before):]:
            assert extra["directional"] == "false"
            assert float(extra["pos"].split()[2]) >= 1.8
    finally:
        cleanup_variant_xml(out)


def test_material_and_light_color_clamped_physics_untouched(scene_xml):
    rng = np.random.default_rng(7)
    cfg = AppearanceConfig(texture_frac=0.0, light_jitter=1.0,
                           material_rgba_jitter=0.2, material_pbr_jitter=0.5)
    out = randomize_scene_xml(scene_xml, rng, cfg)
    try:
        root = ET.parse(str(out)).getroot()
        for light in root.iter("light"):
            for attr in ("diffuse", "ambient", "specular"):
                if light.get(attr):
                    vals = [float(x) for x in light.get(attr).split()]
                    assert all(0.05 <= v <= 1.0 for v in vals)
        src_root = ET.parse(str(scene_xml)).getroot()
        src_alpha = {m.get("name"): m.get("rgba").split()[-1]
                     for m in src_root.iter("material") if m.get("rgba")}
        for m in root.iter("material"):
            if m.get("rgba") and m.get("name") in src_alpha:
                assert m.get("rgba").split()[-1] == src_alpha[m.get("name")]
            for attr in ("shininess", "specular"):
                if m.get(attr) is not None:
                    assert 0.0 <= float(m.get(attr)) <= 1.0
        # shading attrs were actually written on non-excluded materials
        assert any(m.get("shininess") is not None for m in root.iter("material"))
        # nothing physical differs: bodies, geoms, joints, meshes
        for tag in ("body", "geom", "joint", "mesh"):
            src = [e.attrib for e in src_root.iter(tag)]
            dst = [e.attrib for e in root.iter(tag)]
            assert src == dst, f"<{tag}> changed"
    finally:
        cleanup_variant_xml(out)


def test_out_xml_must_be_sibling(scene_xml, tmp_path):
    with pytest.raises(ValueError):
        randomize_scene_xml(scene_xml, np.random.default_rng(0),
                            AppearanceConfig(), out_xml=tmp_path / "x.xml")
