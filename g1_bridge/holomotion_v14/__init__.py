"""Frozen HoloMotion v1.4 runtime; imported only inside the controller process."""
from pathlib import Path
import sys
VENDOR_DIR = Path(__file__).parent / "vendor"
ROBOT_CONFIG = Path(__file__).parent / "config" / "g1_29dof_holomotion.yaml"
def ensure_vendor_path():
    for name in ("humanoid_policy", "holomotion_policy_core"):
        module = sys.modules.get(name)
        if module is not None:
            source = getattr(module, "__file__", None)
            if source is None or not Path(source).resolve().is_relative_to(VENDOR_DIR.resolve()):
                raise RuntimeError(f"Conflicting {name} runtime already imported")
    path = str(VENDOR_DIR)
    if path not in sys.path:
        sys.path.insert(0, path)
