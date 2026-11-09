#!/usr/bin/env bash
#
# One-time setup for running backex on a Raspberry Pi 5 (Raspberry Pi OS
# Bookworm or newer). Installs system camera/OpenCV packages via apt, creates
# a Python venv that can still see them, and installs the remaining Python
# dependencies (preferring lightweight tflite-runtime over full tensorflow
# where possible).
#
# Usage:
#   chmod +x scripts/setup_pi.sh
#   ./scripts/setup_pi.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="$REPO_ROOT/venv"
PIWHEELS_INDEX="https://www.piwheels.org/simple"

log()  { printf '\n\033[1;34m==> %s\033[0m\n' "$1"; }
warn() { printf '\033[1;33m[warn] %s\033[0m\n' "$1"; }
die()  { printf '\033[1;31m[error] %s\033[0m\n' "$1"; exit 1; }

# --- 1. Sanity checks -------------------------------------------------------

if [[ "$(uname -s)" != "Linux" ]]; then
    die "This script is meant to run on Raspberry Pi OS, not $(uname -s)."
fi

if ! grep -qi "raspbian\|raspberry pi os" /etc/os-release 2>/dev/null; then
    warn "Could not confirm this is Raspberry Pi OS. Continuing anyway."
fi

ARCH="$(uname -m)"
if [[ "$ARCH" != "aarch64" ]]; then
    warn "Expected aarch64 (64-bit Raspberry Pi OS) on a Pi 5, got '$ARCH'. Continuing anyway."
fi

log "Detected: $(cat /etc/os-release 2>/dev/null | grep PRETTY_NAME | cut -d= -f2 || echo unknown) on $ARCH"

# --- 2. System packages (apt) ------------------------------------------------
# picamera2 depends on libcamera bindings that must come from apt -- a plain
# `pip install picamera2` in an isolated venv will not work correctly.

log "Installing system packages (sudo required)..."
sudo apt update
sudo apt install -y \
    python3-picamera2 \
    python3-libcamera \
    python3-opencv \
    python3-venv \
    python3-pip \
    rpicam-apps \
    libopenblas-dev

# --- 3. Confirm the camera is actually detected -----------------------------

log "Checking for a connected camera..."
if command -v rpicam-hello >/dev/null 2>&1; then
    if rpicam-hello --list-cameras 2>&1 | grep -qi "no cameras available"; then
        warn "No camera detected by rpicam-hello. Check the ribbon cable / camera port before running the app."
    else
        rpicam-hello --list-cameras || true
    fi
else
    warn "rpicam-hello not found -- skipping camera detection check."
fi

# --- 4. Python virtual environment -------------------------------------------
# --system-site-packages is required so the venv can see the apt-installed
# picamera2 / libcamera / opencv bindings above.

log "Creating virtual environment at $VENV_DIR (with access to system packages)..."
if [[ -d "$VENV_DIR" ]]; then
    warn "Venv already exists at $VENV_DIR -- reusing it."
else
    python3 -m venv --system-site-packages "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
pip install --upgrade pip

# --- 5. Python dependencies ---------------------------------------------------

log "Installing common Python dependencies..."
pip install numpy scikit-learn joblib pytest

log "Installing TFLite inference runtime (preferred over full TensorFlow on a Pi)..."
if pip install --extra-index-url "$PIWHEELS_INDEX" tflite-runtime; then
    echo "tflite-runtime installed successfully."
else
    warn "tflite-runtime wheel not available for this platform/Python version."
    warn "Falling back to full 'tensorflow' (heavier, slower to install and run on a Pi)."
    pip install --extra-index-url "$PIWHEELS_INDEX" tensorflow
fi

# --- 6. Verify everything imports correctly ----------------------------------

log "Verifying installed packages import correctly..."
python3 - <<'PYEOF'
import importlib
import sys

checks = ["cv2", "numpy", "sklearn", "joblib", "picamera2"]
optional = {"tflite_runtime.interpreter": "tflite-runtime", "tensorflow": "tensorflow"}

failed = []
for mod in checks:
    try:
        importlib.import_module(mod)
        print(f"  [ok] {mod}")
    except ImportError as e:
        failed.append(mod)
        print(f"  [FAIL] {mod}: {e}")

inference_backend_ok = False
for mod, label in optional.items():
    try:
        importlib.import_module(mod)
        print(f"  [ok] {label} ({mod})")
        inference_backend_ok = True
        break
    except ImportError:
        continue

if not inference_backend_ok:
    print("  [FAIL] neither tflite-runtime nor tensorflow is importable")
    failed.append("inference backend")

if failed:
    print(f"\nSetup finished with problems in: {', '.join(failed)}")
    sys.exit(1)
else:
    print("\nAll required packages import correctly.")
PYEOF

log "Setup complete."
echo "Activate the environment with:"
echo "  source $VENV_DIR/bin/activate"
echo "Then run the app with:"
echo "  python3 src/realtime_detector.py"
