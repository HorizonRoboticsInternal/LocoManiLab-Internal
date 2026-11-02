"""Robot selection for the G1 benchmark's public entry points."""

G1_ROBOTS = ("G1", "G1FixedLowerBody", "G1FloatingBody")
DEFAULT_ROBOT = "G1FloatingBody"
DEFAULT_CAMERA = "robot0_head_camera"


def validate_g1_robots(robots):
    """Return one supported robot name without changing the caller's value."""
    if isinstance(robots, str):
        names = [robots]
    elif isinstance(robots, (list, tuple)):
        names = list(robots)
    else:
        names = []
    if len(names) != 1 or names[0] not in G1_ROBOTS:
        raise ValueError(
            "humanoid_benchmark supports one G1 robot per environment; "
            f"choose one of {G1_ROBOTS}, got {robots!r}"
        )
    return names[0]
