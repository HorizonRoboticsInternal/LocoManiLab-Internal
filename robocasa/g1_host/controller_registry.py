"""Register G1 controller names without importing optional policy runtimes.

Constructing a selected controller imports its implementation and preserves its
normal robosuite registration. Merely listing tasks or inspecting robot models
does not need Torch, ONNX Runtime, SDKs, or policy weights.
"""

from robosuite.controllers.composite.composite_controller import (
    REGISTERED_COMPOSITE_CONTROLLERS_DICT,
    register_composite_controller,
)


class _WholeBodyG1Factory:
    name = "WHOLE_BODY_G1"

    def __new__(cls, *args, **kwargs):
        from robocasa.g1_host.wbt import WholeBodyG1Composite

        return WholeBodyG1Composite(*args, **kwargs)


class _G1BridgeFactory:
    name = "G1_BRIDGE"

    def __new__(cls, *args, **kwargs):
        from robocasa.g1_host.bridge import G1BridgeComposite

        return G1BridgeComposite(*args, **kwargs)


def register_g1_controller_factories():
    """Keep real implementations already registered by an explicit import."""
    for factory in (_WholeBodyG1Factory, _G1BridgeFactory):
        if factory.name not in REGISTERED_COMPOSITE_CONTROLLERS_DICT:
            register_composite_controller(factory)
