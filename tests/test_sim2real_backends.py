"""Backend dispatch tests that need no sim, no checkpoint and no GPU.

Possible because Sim2RealBackend takes a Sim2RealContext *Protocol* rather than
the composite itself -- a fake with three attributes is a valid context.

This covers the sonic smpl flavor specifically, which cannot be exercised
end-to-end here: checkpoints/sonic/low_latency/smpl/ ships policy.yaml but no
policy.onnx, so the golden-trace harness can only reach the split-onnx flavor.

Run: python tests/test_sim2real_backends.py
"""
import numpy as np
from g1_bridge.pico.mode_manager import ManagerMode
from g1_bridge.backends.sonic import SonicBackend, FLAVOR_SMPL, FLAVOR_SPLIT_ONNX
from g1_bridge.backends.base import TickInput


class FakeCtx:
    """Only what SonicBackend.step touches."""

    def __init__(self):
        self.held = 0
        self.pico_source = None
        self.tracking = None

    def hold_pose(self):
        self.held += 1
        return (np.zeros(29), np.ones(29), np.ones(29))


def tick(mode):
    return TickInput(
        qpos=np.zeros(29),
        qvel=np.zeros(29),
        sim_time=0.0,
        mode=mode,
        buttons=None,
        sample=None,
    )


b = SonicBackend()
ctx = FakeCtx()
b._flavor = FLAVOR_SMPL
for m in (ManagerMode.OFF, ManagerMode.PLANNER, ManagerMode.VR_3PT):
    b.step(ctx, tick(m))
assert ctx.held == 3, f"smpl flavor should hold in non-POSE modes, held={ctx.held}"

# PLANNER with no kplanner armed must not reach the planner path
b._flavor = FLAVOR_SPLIT_ONNX
b._kplanner = None
assert b.handle_keypress(ctx, ord("W")) is False, "keys must be inert with no kPlanner"
print("PASS: smpl flavor holds in OFF/PLANNER/VR_3PT; keypress inert without kPlanner")


# --------------------------------------------------------------------------- #
# Reference blending. Both of sonic's channels are unreachable without a PICO
# (POSE needs live SMPL; PLANNER needs a 739MB kPlanner ONNX), so they are
# exercised directly here against hand-built inputs.
# --------------------------------------------------------------------------- #
import types
from sim2real.config.robots.g1 import G1_CFG
from g1_bridge.reference.default import default_reference_frame

BLEND_S = 0.2
STEP = 0.02


class BlendCtx(FakeCtx):
    def __init__(self):
        super().__init__()
        self.tracking = types.SimpleNamespace(robot_cfg=G1_CFG)
        self.sim = types.SimpleNamespace(data=types.SimpleNamespace(time=0.0))
        self.settle_qpos = None
        self.pico_source = None

    def read_root_pos(self):
        return np.array([0.4, -0.2, 0.79])

    def read_pelvis_state(self):
        return np.array([1.0, 0.0, 0.0, 0.0]), np.zeros(3)

    def standing_joints(self):
        return np.asarray(G1_CFG.default_qpos)[7:]


def kplanner_payload(joints):
    return {
        "joint_pos": np.asarray(joints, np.float32).tolist(),
        "body_pos_w": [[1.0, 2.0, 0.8]],
        "body_quat_w": [[1.0, 0.0, 0.0, 0.0]],
        "joint_names": list(G1_CFG.joint_names),
        "body_names": ["pelvis"],
    }


# --- kPlanner (qpos) channel ------------------------------------------------
sb = SonicBackend()
sb._blend_s = BLEND_S
sb._planner_blender = __import__(
    "g1_bridge.reference.blend", fromlist=["PoseBlender"]
).PoseBlender(BLEND_S)
sb._planner_blend_pending = True
sb._planner_anchor_qpos = None
bctx = BlendCtx()
anchor_joints = default_reference_frame(G1_CFG).default_qpos_at(
    bctx.read_root_pos(), bctx.read_pelvis_state()[0], bctx.standing_joints()
)[7:]
gait = np.asarray(G1_CFG.default_qpos)[7:] + 0.6

first = sb._blend_planner_payload(bctx, kplanner_payload(gait), 0.0)
assert first.get("motion_first_frame") is True, "PLANNER entry must open a new segment"
assert len(first["body_names"]) == 1, (
    "the blended frame must keep kPlanner's 1-body layout -- a layout change "
    "wipes the buffer's frames and segment transform"
)
assert np.allclose(
    first["joint_pos"], anchor_joints, atol=1e-5
), "PLANNER entry must start at the held pose, not the gait's first frame"

out = first
for i in range(1, int(BLEND_S / STEP) + 1):
    out = sb._blend_planner_payload(bctx, kplanner_payload(gait), i * STEP)
assert np.allclose(out["joint_pos"], gait, atol=1e-5), "blend must reach the gait frame"
assert out.get("motion_first_frame") is None, "only the entry frame is stamped"
steps = np.abs(np.diff([anchor_joints] + [gait], axis=0)).max()
print(
    f"PASS: kPlanner reference ramps {steps:.3f} rad in over {BLEND_S}s, "
    "stamped once, 1-body layout preserved"
)

# blend_s=0 disables it entirely
sb0 = SonicBackend()
sb0._blend_s = 0.0
sb0._planner_blend_pending = True
assert (
    sb0._blend_planner_payload(bctx, kplanner_payload(gait), 0.0)["joint_pos"]
    == kplanner_payload(gait)["joint_pos"]
), "blend_s=0 must pass the payload straight through"

# --- SMPL (POSE) channel ----------------------------------------------------
stand_j = np.zeros((24, 3), np.float32)
stand_w = np.zeros(6, np.float32)
live_j = np.ones((24, 3), np.float32)
live_w = np.ones(6, np.float32)
anchor6 = np.array([1, 0, 0, 0, 1, 0], np.float32)

sb2 = SonicBackend()
sb2._blend_s = BLEND_S
sb2._standing_smpl = (stand_j, stand_w)
sb2._pose_blend_pending = True
c2 = BlendCtx()
j, w, a = sb2._blend_pose_reference(c2, live_j, live_w, anchor6)
assert np.allclose(j, stand_j), "POSE entry must start at the standing pose"
assert np.allclose(a, anchor6), "anchor6 passes through untouched"
c2.sim.data.time = BLEND_S * 0.5
j, w, _ = sb2._blend_pose_reference(c2, live_j, live_w, anchor6)
assert np.allclose(j, 0.5 * live_j, atol=1e-6), "halfway through the ramp"
c2.sim.data.time = BLEND_S * 1.5
j, w, _ = sb2._blend_pose_reference(c2, live_j, live_w, anchor6)
assert np.allclose(j, live_j), "past the ramp the operator's pose passes through"

# no captured standing pose -> no blend, exactly the old behaviour
sb3 = SonicBackend()
sb3._blend_s = BLEND_S
sb3._standing_smpl = None
sb3._pose_blend_pending = True
j, w, _ = sb3._blend_pose_reference(BlendCtx(), live_j, live_w, anchor6)
assert np.allclose(j, live_j), "without a standing anchor the ramp is skipped"
print(
    "PASS: POSE reference ramps from the captured standing pose; "
    "degrades to pass-through when none was captured"
)


# --------------------------------------------------------------------------- #
# PICO gait chords (A+B / X+Y) in PLANNER mode.
#
# This path is only reachable with a real headset in PLANNER mode, and only on
# the tick where the gait actually CHANGES -- which is why a missing
# @staticmethod on _locomotion_name sat here undetected and then raised
# TypeError mid-session, after two demos had already been recorded.
# --------------------------------------------------------------------------- #
class GaitCtx(FakeCtx):
    def __init__(self):
        super().__init__()
        self.sim = types.SimpleNamespace(data=types.SimpleNamespace(time=0.0))


def chord(**pressed):
    b = {k: False for k in ("a", "b", "x", "y")}
    b.update(pressed)
    b["left_axis"] = (0.0, 0.0)
    b["right_axis"] = (0.0, 0.0)
    return b


gb = SonicBackend()
gb._movement_state = __import__(
    "g1_bridge.sonic_utils.kplanner", fromlist=["MovementState"]
).MovementState()
# _tick_pico_planner_input no-ops without a planner; a sentinel is enough,
# nothing on this path calls into it.
gb._kplanner = object()
gctx = GaitCtx()

# name lookup itself: called both bound and unbound
assert SonicBackend._locomotion_name(1) == "SLOW_WALK", "unbound lookup"
assert gb._locomotion_name(1) == "SLOW_WALK", "bound lookup (needs @staticmethod)"
assert "UNKNOWN" in SonicBackend._locomotion_name(
    9999
), "unknown id is reported, not raised"

# drive the chords the way the operator does: press, release, press...
start = gb._selected_gait
t = 0.0
for i in range(6):
    for keys in (chord(a=True, b=True), chord()):
        t += SonicBackend._PICO_INPUT_DT
        gctx.sim.data.time = t
        gb._tick_pico_planner_input(gctx, keys)
up = gb._selected_gait
assert up > start, f"A+B should raise the gait ({start} -> {up})"
for i in range(6):
    for keys in (chord(x=True, y=True), chord()):
        t += SonicBackend._PICO_INPUT_DT
        gctx.sim.data.time = t
        gb._tick_pico_planner_input(gctx, keys)
assert (
    gb._selected_gait < up
), f"X+Y should lower the gait ({up} -> {gb._selected_gait})"
print(
    f"PASS: A+B / X+Y gait chords step the gait ({start} -> {up} -> "
    f"{gb._selected_gait}) without raising"
)
