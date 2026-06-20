# detect

Real-time object detector using YOLOv8 and ONNX Runtime, with bounding-box
overlay saved to JPEG files.

## Requirements

- Rust toolchain (edition 2024 / Rust 1.85+)
- A YOLOv8n ONNX model file
- Linux system with V4L2 support (for live camera mode)

The binary prefers the ONNX Runtime CUDA execution provider and falls back to
CPU automatically, so it works on any development machine.

## Obtaining the YOLOv8n ONNX model

### From Ultralytics (recommended)

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
```

This produces `yolov8n.onnx` in the current directory.

### Pre-exported model

Download a community-hosted export:

```bash
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
```

Then export as above, or use any YOLOv8n ONNX with input shape
`[1, 3, 640, 640]` and output shape `[1, 84, 8400]`.

## Building

```bash
cd detect
cargo build --release
```

## Running

### Single image (good for testing without a camera)

```bash
./target/release/detect --model yolov8n.onnx image --input photo.jpg
```

Annotated output is saved alongside the input as `annotated_photo.jpg`.
Specify a custom output path with `--output`.

### Live camera (V4L2, e.g. /dev/video0)

```bash
./target/release/detect --model yolov8n.onnx camera --device 0 --output-dir ./frames
```

Annotated frames are written to `./frames/frame_000000.jpg`, `frame_000001.jpg`, …

Use `--max-frames 100` to stop after 100 frames; omit it (or pass 0) to run until
interrupted with Ctrl-C.

### Common options

| Flag | Default | Description |
|------|---------|-------------|
| `--conf` | 0.25 | Minimum class confidence to report |
| `--nms-iou` | 0.45 | IoU threshold for NMS |

## Running tests

Tests cover preprocessing (letterbox, normalisation) and postprocessing (IoU,
NMS, coordinate inversion) without requiring a model file or GPU:

```bash
cargo test
```

## Notes for Jetson Orin NX

When CUDA drivers are installed, ONNX Runtime selects the CUDA execution
provider automatically.  For further acceleration, add the TensorRT execution
provider in `build_session` (before the CUDA provider entry).

For a CSI camera (e.g. IMX219), expose it as a V4L2 device with:

```bash
sudo modprobe nvargus
v4l2-ctl --list-devices
```

Then pass the corresponding device index to `--device`.

## Deploying to the Jetson

Use the repo-level helper script to sync this crate's source to the Jetson and
build it natively there.  The script is at `scripts/deploy-to-target.sh`.

### Why build on-device?

`ort` links against `onnxruntime.so` and the CUDA / TensorRT shared libraries.
Those are aarch64 binaries installed by the Jetson SDK; cross-compiling from
x86 would require replicating the entire CUDA sysroot, which is fragile and
unsupported.  Syncing source and running `cargo build --release` on the Jetson
is the simplest and most reliable approach.

### Prerequisites

- Passwordless SSH to the device (add your public key to `~/.ssh/authorized_keys` on the Jetson).
- Rust toolchain installed on the Jetson (`curl https://sh.rustup.rs | sh`).
- `rsync` available on both the host and the device.

### Build only

```bash
# From the repo root:
./scripts/deploy-to-target.sh --build-only

# Override the target device:
DEVICE=jetson@10.0.0.5 ./scripts/deploy-to-target.sh --build-only
```

Default `DEVICE` is `root@192.168.7.149`.

### Build and run in one step

Pass binary arguments after `--`:

```bash
./scripts/deploy-to-target.sh -- \
    --model /root/models/yolov8n.onnx \
    image --input /root/images/dog.jpg
```

### Full option reference

| Environment variable | Default | Purpose |
|----------------------|---------|---------|
| `DEVICE` | `root@192.168.7.149` | SSH target (`user@host`) |
| `CRATE` | `detect` | Cargo crate directory name |
| `REMOTE_DIR` | `/root/rust-ai/<CRATE>` | Remote path to sync into |

All three can also be changed for other crates in the repo:

```bash
CRATE=classify ./scripts/deploy-to-target.sh --build-only
```
