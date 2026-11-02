#!/usr/bin/env bash
# XRoboToolkit (PICO VR) setup helper for Ubuntu.
#
# Installs:
#   1. PC Service from a prebuilt .deb (provided by you, e.g. ~/Downloads/XRoboToolkit*.deb)
#   2. xrobotoolkit_sdk Python binding into the *currently active* conda env
#
# Usage:
#   conda activate robocasa
#   bash scripts/setup_xrobotoolkit_ubuntu.sh [path/to/XRoboToolkit.deb]
#
# If no .deb path is given, the script searches ~/Downloads for one matching
# XRoboToolkit*.deb.  Pass --skip-deb if the PC Service is already installed.
#
# The script is idempotent: rerunning is safe.  Each step prints a banner; if
# something fails the script exits early with a clear message — no silent
# breakage.

set -euo pipefail

# ---------- helpers -----------------------------------------------------------
log()  { printf '\n\033[1;36m[xrt-setup]\033[0m %s\n' "$*"; }
warn() { printf '\n\033[1;33m[xrt-setup WARN]\033[0m %s\n' "$*"; }
die()  { printf '\n\033[1;31m[xrt-setup ERROR]\033[0m %s\n' "$*" >&2; exit 1; }

# ---------- arg parsing -------------------------------------------------------
SKIP_DEB=0
DEB_PATH=""
PYBIND_REPO_URL="https://github.com/XR-Robotics/XRoboToolkit-PC-Service-Pybind.git"
PC_SERVICE_REPO_URL="https://github.com/XR-Robotics/XRoboToolkit-PC-Service.git"
WORK_ROOT="${HOME}/xrt_setup_work"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-deb) SKIP_DEB=1; shift ;;
    --work-root) WORK_ROOT="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,18p' "$0"; exit 0 ;;
    *)
      DEB_PATH="$1"; shift ;;
  esac
done

# ---------- 0. sanity: OS + env ----------------------------------------------
log "Step 0: environment sanity check"

if ! command -v dpkg >/dev/null; then
  die "dpkg not found — this script is for Debian/Ubuntu only."
fi

if [[ -r /etc/os-release ]]; then
  . /etc/os-release
  echo "  OS:    ${PRETTY_NAME:-unknown}"
fi
echo "  Arch:  $(uname -m)"
echo "  User:  $(whoami)"

if [[ -z "${CONDA_DEFAULT_ENV:-}" ]]; then
  warn "No conda env is active.  The Python binding will install into the system Python."
  warn "Press Ctrl-C now to abort, or wait 5 seconds to continue..."
  sleep 5
else
  echo "  Conda env: ${CONDA_DEFAULT_ENV}"
  echo "  Python:    $(which python)"
fi

mkdir -p "$WORK_ROOT"
cd "$WORK_ROOT"
echo "  Work dir:  $WORK_ROOT"

# ---------- 1. install PC Service from .deb ----------------------------------
if [[ $SKIP_DEB -eq 1 ]]; then
  log "Step 1: SKIPPED (--skip-deb)"
else
  log "Step 1: install PC Service (.deb)"

  if [[ -z "$DEB_PATH" ]]; then
    # auto-discover in ~/Downloads
    CANDIDATES=( "$HOME"/Downloads/XRoboToolkit*.deb )
    if [[ ! -e "${CANDIDATES[0]}" ]]; then
      die "No .deb path given and none found at ~/Downloads/XRoboToolkit*.deb"
    fi
    DEB_PATH="${CANDIDATES[0]}"
  fi

  [[ -f "$DEB_PATH" ]] || die ".deb not found at: $DEB_PATH"
  echo "  .deb: $DEB_PATH"

  echo "  ---- dpkg metadata ----"
  dpkg-deb -I "$DEB_PATH" | sed 's/^/  /'
  echo "  -----------------------"

  log "Installing .deb (will prompt for sudo password)..."
  if ! sudo dpkg -i "$DEB_PATH"; then
    warn "dpkg -i failed — attempting to fix missing deps with apt-get install -f"
    sudo apt-get update
    sudo apt-get install -f -y
    sudo dpkg -i "$DEB_PATH" || die "dpkg -i failed even after dep fix"
  fi

  log "Locating installed files..."
  SDK_SO=$(find /opt /usr/lib /usr/local -name 'libPXREARobotSDK.so' 2>/dev/null | head -1 || true)
  SDK_HDR=$(find /opt /usr/include /usr/local/include -name 'PXREARobotSDK.h' 2>/dev/null | head -1 || true)
  RUN_SH=$(find /opt /usr/local -name 'runService.sh' 2>/dev/null | head -1 || true)

  echo "  libPXREARobotSDK.so : ${SDK_SO:-NOT FOUND}"
  echo "  PXREARobotSDK.h     : ${SDK_HDR:-NOT FOUND}"
  echo "  runService.sh       : ${RUN_SH:-NOT FOUND}"

  [[ -n "$SDK_SO" ]] || warn ".so not found — Python binding build may fail."
  [[ -n "$RUN_SH" ]] || warn "runService.sh not found — you won't be able to start the daemon."
fi

# ---------- 2. clone + build Python binding ----------------------------------
log "Step 2: build & install xrobotoolkit_sdk Python binding"

PYBIND_DIR="$WORK_ROOT/XRoboToolkit-PC-Service-Pybind"
if [[ ! -d "$PYBIND_DIR" ]]; then
  git clone "$PYBIND_REPO_URL" "$PYBIND_DIR"
else
  echo "  Pybind repo already cloned, pulling..."
  git -C "$PYBIND_DIR" pull --ff-only || warn "git pull failed; using existing checkout"
fi

cd "$PYBIND_DIR"
mkdir -p lib include

# Try the shortcut path: reuse the .so / headers from the installed .deb.
# If anything is missing we fall back to the upstream setup_ubuntu.sh which
# clones PC-Service and builds the SDK from source.
SHORTCUT_OK=1
SDK_SO=$(find /opt /usr/lib /usr/local -name 'libPXREARobotSDK.so' 2>/dev/null | head -1 || true)
SDK_HDR=$(find /opt /usr/include /usr/local/include -name 'PXREARobotSDK.h' 2>/dev/null | head -1 || true)
NLOHMANN_DIR=$(find /opt /usr/include /usr/local/include -type d -name 'nlohmann' 2>/dev/null | head -1 || true)

if [[ -z "$SDK_SO" || -z "$SDK_HDR" ]]; then
  SHORTCUT_OK=0
fi

if [[ $SHORTCUT_OK -eq 1 ]]; then
  log "  Using SDK from installed .deb (shortcut path)"
  cp -f "$SDK_SO"  lib/
  cp -f "$SDK_HDR" include/
  if [[ -n "$NLOHMANN_DIR" ]]; then
    rm -rf include/nlohmann
    cp -r "$NLOHMANN_DIR" include/nlohmann
  else
    warn "nlohmann/ headers not found system-wide — pulling from upstream PC-Service repo"
    mkdir -p tmp; cd tmp
    [[ -d XRoboToolkit-PC-Service ]] || git clone --depth 1 "$PC_SERVICE_REPO_URL"
    cp -r XRoboToolkit-PC-Service/RoboticsService/PXREARobotSDK/nlohmann ../include/nlohmann
    cd ..
  fi
else
  warn "SDK .so or header not found from .deb — running upstream setup_ubuntu.sh"
  warn "(this clones PC-Service and recompiles the SDK from source)"
  bash setup_ubuntu.sh
  log "Step 2 done via upstream script — verifying import below"
fi

# When we used the shortcut, we still need to install pybind11 + run setup.py.
if [[ $SHORTCUT_OK -eq 1 ]]; then
  log "  Installing build deps + Python module"
  if [[ -n "${CONDA_DEFAULT_ENV:-}" ]]; then
    conda install -c conda-forge libstdcxx-ng pybind11 -y
  else
    pip install pybind11
  fi
  pip uninstall -y xrobotoolkit_sdk || true
  python setup.py install
fi

# ---------- 3. verify ---------------------------------------------------------
log "Step 3: verify import"

python - <<'PY'
import sys
try:
    import xrobotoolkit_sdk as xrt
except Exception as e:
    print("[FAIL] import xrobotoolkit_sdk:", e)
    sys.exit(1)
print("[OK]  import xrobotoolkit_sdk:", xrt.__file__)

# Don't call xrt.init() here — that needs the PC Service running + PICO
# connected, which is a separate step.  Just confirming the .so loads.
PY

log "Done."

cat <<'EONOTE'

Next steps:

  1. Plug in your PICO Neo3 (USB or WiFi same subnet), turn it on, enable
     Body Tracking in the XRoboToolkit app on the headset.

  2. Start the PC Service in another terminal:

         bash /opt/apps/roboticsservice/runService.sh   # or whatever path was
                                                        # printed in Step 1

  3. Sanity-test in Python (PC Service must be running, PICO connected):

         python -c "
         import xrobotoolkit_sdk as xrt, numpy as np, time
         xrt.init()
         time.sleep(1.0)
         print('available =', xrt.is_body_data_available())
         print('shape     =', np.array(xrt.get_body_joints_pose()).shape)
         "

     Expect shape (24, 7) (24 SMPL joints × xyz+quat).

  4. Once that prints data, you're ready to fill in RealPicoVRSource in
        robosuite_src/robosuite/controllers/composite/pico_vr_source.py
     and flip pico_source to "real" in
        robosuite_src/robosuite/controllers/config/robots/default_g1_teleop.json

EONOTE
