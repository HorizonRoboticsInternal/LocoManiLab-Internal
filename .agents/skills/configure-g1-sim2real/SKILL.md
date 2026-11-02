---
name: configure-g1-sim2real
description: Install, repair, and verify sim2real on G1 robot computers. Use for root uv environments and inference extras, robot-g1 dependencies, CycloneDDS builds, G1 network interfaces, Hugging Face offline assets, ONNX Runtime compatibility, inline RobotIO or MotionSwitcher DDS failures, deployment checkouts, isolated PICO dependencies, and G1 deployment bring-up.
---

# Configure G1 sim2real

Configure the G1 host from current evidence. Do not infer its OS, network
interface, Python environment, or GPU runtime from its SSH alias.

## Operating rules

1. Treat the local checkout as the source of truth. Make source edits locally,
   then deploy a reviewed Git revision. Never transfer `.venv` directories.
2. Run `scripts/inspect_host.sh` before installing or repairing anything.
3. Keep these environments separate:
   - root project: policy, simulation, ZMQ RobotIO, and optional G1 inline SDK;
   - `venv/pico`: XRoboToolkit, GMR, SMPL-X, and PICO publisher;
4. Default G1 deployment to CPU inference unless the user explicitly requests
   GPU inference. A listed CUDA provider is not proof that its shared libraries
   load.
5. Never store sudo, Wi-Fi, user, or SDK credentials in this skill or the
   repository. Let the operator enter secrets interactively.
6. Do not start a policy, switch modes, home a hand, or send motor commands
   during installation unless the user explicitly requests a hardware test.
7. Use `bash -lc` for remote commands, but explicitly source the sim2real
   environment file. Do not assume variables appended to `.bashrc` are visible
   to noninteractive shells.

## Workflow

### 1. Inspect and choose a profile

Run locally against the checkout:

```bash
ssh <host> 'bash -s -- "$HOME/sim2real"' \
  < .agents/skills/configure-g1-sim2real/scripts/inspect_host.sh
```

Record the OS, architecture, Python, uv, disk space, interfaces,
`CYCLONEDDS_HOME`, HF settings, native library paths, and existing environments.
Read `references/g1-setup-runbook.md` before modifying a live host.

Choose exactly one root profile:

- generic/ZMQ CPU: `--extra inference-cpu`
- G1 inline CPU: `--extra inference-cpu --extra robot-g1`
- G1 inline GPU: `--extra inference-gpu --extra robot-g1`

Do not install both inference extras in one environment.

### 2. Prepare the deployment checkout

Use a clean checkout of the reviewed revision on the selected host. For a new
checkout, run on that host:

```bash
git clone https://github.com/LocoManiLab/g1_backend.git "$HOME/sim2real"
cd "$HOME/sim2real"
git checkout <reviewed-commit>
```

For an existing checkout, inspect its status before fetching or switching.
Preserve local changes, environments and host configuration. The project no
longer ships personal robot/workspace synchronization scripts. Transfer only
explicitly selected weights or native artifacts to matching paths separately;
verify their destination and size before deployment.

### 3. Configure native and shell dependencies

For `robot-g1`, install CycloneDDS first if the required installation is
missing. Put persistent variables in `~/.config/sim2real/env.sh`, source it
from `~/.profile`, and source it explicitly in automation:

```bash
source "$HOME/.config/sim2real/env.sh"
```

Verify the variables with a fresh `bash -lc`; checking the current interactive
shell is insufficient.

### 4. Install the selected root profile

Run from `~/sim2real`:

```bash
source "$HOME/.config/sim2real/env.sh"
uv sync --extra inference-cpu --extra robot-g1
```

Replace `inference-cpu` only when the selected profile requires another
backend. Do not use the legacy `--group g1`.

If Git or package downloads hang, use a bounded HTTP(S) proxy tunnel or seed
the required source/cache from the local machine. Do not wait indefinitely.

### 5. Prepare deploy-time assets

Resolve required Hugging Face assets once while online, then run deployments
with `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`. Validate with the
actual `mjhub.resolve_asset_reference()` call rather than cache directory
names.

Treat an HF stall, an ONNX model-format error, a missing CUDA library, a DDS
interface error, and missing low state as different failures. Use the failure
map in the runbook.

### 6. Install isolated peripherals only when requested

Use:

```bash
uv sync --project venv/pico
```

Keep XRoboToolkit/GMR native dependencies out of root. Verify PICO with imports before any hardware action. DH116S is an optional
third-party integration; this checkout does not ship a `venv/dh116s` project
or a DH116S control entry point. Follow the SDK provider instructions in an
isolated environment if that hardware is needed.

### 7. Verify without commanding hardware

Run the verifier inside the selected root environment:

```bash
uv run --no-sync python \
  .agents/skills/configure-g1-sim2real/scripts/verify_install.py \
  --profile g1-cpu \
  --asset hf://elijahgalahad/g1_xmls@main/g1-mode_13_15.xml
```

For a policy artifact, add `--onnx <path>`. The verifier imports dependencies,
resolves optional cached assets, and loads the ONNX session; it never creates
RobotIO or sends commands.

When the skill files have not been synced yet, stream the verifier instead of
assuming its remote path exists:

```bash
ssh <host> 'bash -lc '"'"'
  cd "$HOME/sim2real"
  source "$HOME/.config/sim2real/env.sh"
  uv run --no-sync python - --profile g1-cpu
'"'"'' < .agents/skills/configure-g1-sim2real/scripts/verify_install.py
```

Only after this passes, and only with user authorization, test inline startup
using the real robot interface. Confirm no old `tracking.py` or
`real_bridge.py` process is active first.

## Completion report

Report:

- host, OS, architecture, Python, uv, and selected install profile;
- exact sync and install commands used;
- installed, skipped, and failed components;
- environment variables visible in a fresh login shell;
- import, HF resolver, and ONNX load results;
- whether any real hardware action was intentionally not tested;
- a directly runnable next command.

Do not claim readiness from a successful `uv sync` alone.
