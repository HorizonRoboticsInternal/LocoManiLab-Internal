"""Shelf fixture model."""

from robocasa.models.fixtures.fixture import Fixture


class Shelf(Fixture):
    """A static shelf loaded from a fixture-compatible MJCF file."""

    def __init__(self, xml, name="shelf", **kwargs):
        super().__init__(
            xml=xml,
            name=name,
            joints=None,
            duplicate_collision_geoms=False,
            **kwargs,
        )


__all__ = ["Shelf"]
