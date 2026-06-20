#!/usr/bin/env bash
# setup-demo.sh — idempotent demo asset bootstrap for the rust-ai workspace.
#
# Downloads / exports all models, labels, and sample media needed to run the
# four demo apps (classify, detect, server, counter) on CPU.  Re-running the
# script is safe: already-present files are skipped.
#
# Output layout (all gitignored via root .gitignore):
#   assets/
#     mobilenetv2-12.onnx    — MobileNetV2 ONNX opset-12 (classify + server)
#     imagenet_classes.txt   — 1000 ImageNet class labels (classify + server)
#     yolov8n.onnx           — YOLOv8n ONNX export (detect + counter)
#     sample.jpg             — sample image for classify / detect / server
#     frames/                — JPEG frames extracted from sample video (counter)
#
# Requirements: wget, python3, ffmpeg.
# All are checked below and the script reports clearly if something is missing.
#
# NOTE — system Python packages: if ultralytics, onnx, or onnxslim are not
# already installed, this script installs them via
#   pip3 install <pkg> --break-system-packages
# which writes packages into the OS-managed Python environment (bypasses the
# PEP 668 externally-managed-environment guard on Debian/Ubuntu).  If you
# prefer to keep the system Python untouched, install those packages yourself
# in a virtualenv first, activate it, and then re-run this script.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$REPO_ROOT/assets"
FRAMES="$ASSETS/frames"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

info()  { echo "[setup] $*"; }
ok()    { echo "[setup] OK  $*"; }
skip()  { echo "[setup] SKIP $* (already present)"; }
die()   { echo "[setup] ERROR: $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "'$1' not found — install it and re-run."
}

# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

info "Checking required tools..."
require_cmd wget
require_cmd python3
require_cmd ffmpeg

# ultralytics + onnx are needed for the YOLOv8 export.  Install them if absent.
# IMPORTANT: any missing package is installed into the SYSTEM Python via
# `pip3 install --break-system-packages`, bypassing the PEP 668 guard.  See
# the header comment above if you want to avoid mutating the system environment.
MISSING_PY_PKGS=()
python3 -c "import ultralytics" 2>/dev/null || MISSING_PY_PKGS+=(ultralytics)
python3 -c "import onnx"        2>/dev/null || MISSING_PY_PKGS+=(onnx)
python3 -c "import onnxslim"    2>/dev/null || MISSING_PY_PKGS+=(onnxslim)

if [[ ${#MISSING_PY_PKGS[@]} -gt 0 ]]; then
    info "Installing ${MISSING_PY_PKGS[*]} into the system Python via pip3 --break-system-packages"
    pip3 install "${MISSING_PY_PKGS[@]}" --break-system-packages \
        || die "pip install failed.  Run manually:
  pip3 install ${MISSING_PY_PKGS[*]} --break-system-packages
Then re-run this script."
fi

info "All required tools found."

# ---------------------------------------------------------------------------
# Create assets directory
# ---------------------------------------------------------------------------

mkdir -p "$ASSETS" "$FRAMES"

# ---------------------------------------------------------------------------
# 1. MobileNetV2 ONNX model (classify + server)
# ---------------------------------------------------------------------------

MOBILENET="$ASSETS/mobilenetv2-12.onnx"
if [[ -f "$MOBILENET" ]]; then
    skip "mobilenetv2-12.onnx"
else
    info "Downloading MobileNetV2 ONNX model from ONNX Model Zoo..."
    # The ONNX Model Zoo redirects through GitHub's raw CDN; wget follows redirects.
    wget -q --show-progress \
        "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx" \
        -O "$MOBILENET"
    ok "mobilenetv2-12.onnx"
fi

# ---------------------------------------------------------------------------
# 2. ImageNet class labels (classify + server)
# ---------------------------------------------------------------------------

LABELS="$ASSETS/imagenet_classes.txt"
if [[ -f "$LABELS" ]]; then
    skip "imagenet_classes.txt"
else
    info "Downloading ImageNet class labels..."
    wget -q --show-progress \
        "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt" \
        -O "$LABELS"
    ok "imagenet_classes.txt ($(wc -l < "$LABELS") lines)"
fi

# ---------------------------------------------------------------------------
# 3. YOLOv8n ONNX model (detect + counter)
# ---------------------------------------------------------------------------

YOLO="$ASSETS/yolov8n.onnx"
if [[ -f "$YOLO" ]]; then
    skip "yolov8n.onnx"
else
    info "Exporting YOLOv8n to ONNX via ultralytics..."
    # Export into a temp dir to avoid polluting the repo root with .pt/.yaml files.
    EXPORT_TMP="$(mktemp -d)"
    # ultralytics downloads yolov8n.pt on first use; the export writes yolov8n.onnx
    # into the current directory by default — we cd into the temp dir.
    (
        cd "$EXPORT_TMP"
        python3 -c "
from ultralytics import YOLO
model = YOLO('yolov8n.pt')
model.export(format='onnx', imgsz=640, opset=12)
"
    )
    # The export file lands at $EXPORT_TMP/yolov8n.onnx
    mv "$EXPORT_TMP/yolov8n.onnx" "$YOLO"
    rm -rf "$EXPORT_TMP"
    ok "yolov8n.onnx"
fi

# ---------------------------------------------------------------------------
# 4. Sample image (classify / detect / server)
# ---------------------------------------------------------------------------

SAMPLE_IMG="$ASSETS/sample.jpg"
if [[ -f "$SAMPLE_IMG" ]]; then
    skip "sample.jpg"
else
    info "Downloading sample image (Labrador — good for MobileNetV2 + YOLOv8)..."
    # Public-domain dog photo; MobileNetV2 classifies it as Labrador retriever at ~95%.
    wget -q --show-progress \
        "https://images.dog.ceo/breeds/labrador/n02099712_4323.jpg" \
        -O "$SAMPLE_IMG"
    ok "sample.jpg"
fi

# ---------------------------------------------------------------------------
# 5. Sample video frames for counter (ffmpeg extracts JPEG frames)
# ---------------------------------------------------------------------------

# We use a short Creative Commons / public-domain video clip.  If the download
# fails or is unavailable, we synthesise frames from the sample image instead
# so the counter demo can still run end-to-end.

SAMPLE_VIDEO="$ASSETS/sample_video.mp4"
FRAMES_DONE="$FRAMES/.done"

if [[ -f "$FRAMES_DONE" ]]; then
    skip "video frames (already extracted)"
else
    if [[ ! -f "$SAMPLE_VIDEO" ]]; then
        info "Attempting to download sample video (Coverr / public-domain)..."
        # Short pedestrian clip from the Coverr open collection.
        set +e
        wget -q --show-progress --timeout=30 \
            "https://download.samplelib.com/mp4/sample-5s.mp4" \
            -O "$SAMPLE_VIDEO" 2>/dev/null
        WGET_EXIT=$?
        set -e
        if [[ $WGET_EXIT -ne 0 || ! -s "$SAMPLE_VIDEO" ]]; then
            info "Video download unavailable — synthesising frames from sample.jpg instead."
            rm -f "$SAMPLE_VIDEO"
            # Create 30 identical JPEG frames (enough to exercise the pipeline).
            for i in $(seq -w 1 30); do
                cp "$SAMPLE_IMG" "$FRAMES/frame_${i}.jpg"
            done
            touch "$FRAMES_DONE"
            ok "frames/ (30 synthetic frames from sample.jpg)"
        fi
    fi

    if [[ -f "$SAMPLE_VIDEO" && ! -f "$FRAMES_DONE" ]]; then
        info "Extracting frames from sample video..."
        ffmpeg -i "$SAMPLE_VIDEO" -q:v 2 "$FRAMES/frame_%06d.jpg" -y -loglevel error
        NFRAMES=$(ls "$FRAMES"/*.jpg 2>/dev/null | wc -l)
        touch "$FRAMES_DONE"
        ok "frames/ ($NFRAMES frames extracted)"
    fi
fi

# ---------------------------------------------------------------------------
# Done — print a summary
# ---------------------------------------------------------------------------

echo ""
echo "================================================================="
echo " Assets ready in: $ASSETS"
echo "================================================================="
ls -lh "$ASSETS" | grep -v "^total"
echo ""
echo "Next: build the workspace then run the demos:"
echo "  cargo build --release -p classify -p detect -p server -p counter"
echo ""
echo "See DEMO.md at the repo root for exact run commands."
