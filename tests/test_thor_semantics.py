import numpy as np

from robocasa.models.scenes.thor_semantics import (
    SemanticFixture,
    SemanticRegion,
    SemanticSceneRegistry,
)


def _fixture(name, category, center):
    stable_id = f"scene:{name}"
    support = SemanticRegion(
        stable_id=f"{stable_id}:support",
        fixture_id=stable_id,
        kind="support",
        parent_body=name,
        local_center=np.zeros(3),
        half_size=np.array([0.3, 0.2, 0.02]),
        _world_center=np.asarray(center, dtype=float),
    )
    return SemanticFixture(
        stable_id=stable_id,
        name=name,
        category=category,
        source_category=category,
        room_id="1",
        body_name=name,
        body_names=[name],
        site_names=[],
        asset_id=None,
        object_id=name,
        center=np.asarray(center, dtype=float),
        half_size=np.array([0.3, 0.2, 0.4]),
        yaw=0.0,
        front=np.array([1.0, 0.0]),
        capabilities={"support"},
        regions={support.stable_id: support},
    )


def test_semantic_fixture_and_region_queries():
    table = _fixture("table", "table", [0.0, 0.0, 0.8])
    counter = _fixture("counter", "countertop", [1.0, 0.0, 0.9])
    registry = SemanticSceneRegistry("scene", [table, counter])
    assert registry.get_fixture(category="table") is table
    assert registry.get_fixture(id=counter.stable_id) is counter
    assert len(registry.get_region(kind="support", return_all=True)) == 2
    assert registry.audit() == []


def test_region_contains_xy_uses_region_extent():
    fixture = _fixture("table", "table", [0.0, 0.0, 0.8])
    region = fixture.get_region("support")
    assert region.contains_xy([0.2, 0.1])
    assert not region.contains_xy([0.5, 0.1])


def test_fixture_radial_extent_uses_projected_boundary():
    fixture = _fixture("table", "table", [0.0, 0.0, 0.8])
    assert np.isclose(fixture.radial_extent([1.0, 0.0]), 0.3)
    assert np.isclose(fixture.radial_extent([0.0, 1.0]), 0.2)
    expected = (0.3 + 0.2) / np.sqrt(2)
    assert np.isclose(fixture.radial_extent([1.0, 1.0]), expected)


def test_fixture_long_edge_and_normal_follow_local_obb():
    fixture = _fixture("table", "table", [0.0, 0.0, 0.8])
    fixture.local_half_size = np.array([0.6, 0.2, 0.4])
    fixture.yaw = np.pi / 2
    assert np.allclose(fixture.long_edge_direction(), [0.0, 1.0])
    assert np.allclose(fixture.long_edge_normal(), [-1.0, 0.0])
