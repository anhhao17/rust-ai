#!/usr/bin/env bash
# deploy-to-target.sh — Sync a crate to the Jetson, build it natively, and
# optionally run the resulting binary.
#
# WHY build natively rather than cross-compile?
#   The `ort` crate links against onnxruntime.so and (optionally) the CUDA /
#   TensorRT shared libraries, all of which must be the aarch64 variants
#   present on the Jetson.  Attempting to cross-compile from x86 would require
#   replicating the entire CUDA + TensorRT sysroot, which is fragile and
#   unsupported by the ort build scripts.  Syncing source and building on-device
#   is simpler, fully reproducible, and leverages the exact SDK already
#   installed on the target.
#
# Usage:
#   ./scripts/deploy-to-target.sh [--build-only] [-- <binary-args>...]
#
# Environment variables (all optional — defaults shown):
#   DEVICE    SSH target             default: root@192.168.7.149
#   CRATE     Cargo crate directory  default: detect
#   REMOTE_DIR Remote base path      default: /root/rust-ai/<CRATE>
#
# Examples:
#   # Build only:
#   ./scripts/deploy-to-target.sh --build-only
#
#   # Build then run with arguments forwarded to the binary:
#   ./scripts/deploy-to-target.sh -- \
#       --model /root/models/yolov8n.onnx \
#       image --input /root/images/test.jpg
#
#   # Override device and crate:
#   DEVICE=jetson@10.0.0.5 CRATE=classify \
#       ./scripts/deploy-to-target.sh --build-only

set -euo pipefail

# ---------------------------------------------------------------------------
# Configuration — override via env vars before invoking the script.
# ---------------------------------------------------------------------------

DEVICE="${DEVICE:-root@192.168.7.149}"
CRATE="${CRATE:-detect}"

# Resolve the repo root so the script works when called from any directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CRATE_DIR="${REPO_ROOT}/${CRATE}"
REMOTE_DIR="${REMOTE_DIR:-/root/rust-ai/${CRATE}}"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

BUILD_ONLY=false
BINARY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-only)
            BUILD_ONLY=true
            shift
            ;;
        --)
            shift
            BINARY_ARGS=("$@")
            break
            ;;
        -h|--help)
            sed -n '2,/^set -euo/p' "${BASH_SOURCE[0]}" | grep '^#' | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1" >&2
            echo "Usage: $0 [--build-only] [-- <binary-args>...]" >&2
            exit 1
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

log()  { echo "==> $*"; }
warn() { echo "WARN: $*" >&2; }
die()  { echo "ERROR: $*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Preflight checks
# ---------------------------------------------------------------------------

log "Deploying crate '${CRATE}' to ${DEVICE}:${REMOTE_DIR}"

[[ -d "${CRATE_DIR}" ]] \
    || die "crate directory not found: ${CRATE_DIR}"

[[ -f "${CRATE_DIR}/Cargo.toml" ]] \
    || die "no Cargo.toml in: ${CRATE_DIR}"

# Quick reachability check — times out fast so CI doesn't hang.
log "Checking device reachability..."
ssh -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=no \
    "${DEVICE}" "uname -m" \
    || die "cannot reach ${DEVICE} — is it powered on and on the network?"

# ---------------------------------------------------------------------------
# Sync source to device
# ---------------------------------------------------------------------------

log "Syncing ${CRATE_DIR}/ → ${DEVICE}:${REMOTE_DIR}/"

# We sync only the crate source directory (not the whole repo) so the remote
# build directory stays self-contained.  We exclude:
#   target/   — compiled artefacts (rebuilt on-device; sending them wastes bandwidth)
#   .git/     — not needed on the device
# --delete removes files on the remote that no longer exist locally, keeping the
# remote tree in lockstep with the local source.
rsync \
    --archive \
    --compress \
    --delete \
    --exclude="target/" \
    --exclude=".git/" \
    --info=progress2 \
    "${CRATE_DIR}/" \
    "${DEVICE}:${REMOTE_DIR}/"

log "Sync complete."

# ---------------------------------------------------------------------------
# Build on device (native aarch64)
# ---------------------------------------------------------------------------

log "Building ${CRATE} on ${DEVICE} (cargo build --release)..."

# We pass --release so the Jetson CUDA kernels get full optimisation.
# ORT will detect the CUDA execution provider at runtime if the Jetson
# CUDA/cuDNN stack is installed; otherwise it silently falls back to CPU.
ssh -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=no \
    "${DEVICE}" \
    "cd '${REMOTE_DIR}' && cargo build --release 2>&1"

log "Build complete."

# ---------------------------------------------------------------------------
# Optional: run the binary on the device
# ---------------------------------------------------------------------------

if [[ "${BUILD_ONLY}" == true ]]; then
    log "Done (build-only mode)."
    exit 0
fi

if [[ ${#BINARY_ARGS[@]} -eq 0 ]]; then
    log "No binary arguments supplied — skipping run step."
    log "To run the binary, add '-- <args>' after the script options."
    exit 0
fi

log "Running ${CRATE} on ${DEVICE} with args: ${BINARY_ARGS[*]}"

# shellcheck disable=SC2029  # intentional: expand BINARY_ARGS on this side
ssh -o BatchMode=yes \
    -o ConnectTimeout=10 \
    -o StrictHostKeyChecking=no \
    "${DEVICE}" \
    "'${REMOTE_DIR}/target/release/${CRATE}' ${BINARY_ARGS[*]}"

log "Run complete."
