# counter — People Counter / Smart Camera

Detects people in a video or image sequence, tracks them across frames, counts
line crossings, and publishes a **live annotated MJPEG video stream** and count
stats to a small web dashboard.

**Teaching point:** combining a YOLO object-detection model with stateful logic
(tracking + counting) and real-time video output.

---

## Obtaining the YOLOv8 model

The same model used by the `detect` crate works here — see
[detect/README.md](../detect/README.md) for the download command.  In short:

```bash
pip install ultralytics
yolo export model=yolov8n.pt format=onnx imgsz=640
# → produces yolov8n.onnx in the current directory
```

---

## Running

```bash
cargo build --release
```

### Image-frame directory

```bash
# Extract frames from a video first (if needed)
ffmpeg -i video.mp4 -q:v 2 frames/frame_%06d.jpg

./target/release/counter \
  --model yolov8n.onnx \
  --source frames/ \
  --line-x1 384 --line-y1 0 --line-x2 384 --line-y2 576 \
  --bind 0.0.0.0:3000
```

### Video file (decoded via ffmpeg)

```bash
./target/release/counter \
  --model yolov8n.onnx \
  --source assets/walk.avi \
  --fps 10 \
  --line-x1 384 --line-y1 0 --line-x2 384 --line-y2 576
```

Add `--no-loop` to stop after the first playthrough instead of looping.

### RTSP / HTTP(S) / HLS network stream

```bash
./target/release/counter \
  --model yolov8n.onnx \
  --source rtsp://192.168.1.100:554/live \
  --fps 15 \
  --line-x1 640 --line-y1 0 --line-x2 640 --line-y2 720 \
  --bind 0.0.0.0:3000
```

Live streams never loop and run until interrupted (Ctrl-C).  HTTP(S) and HLS
URLs (e.g. `http://…/stream.m3u8`) are also supported.

### V4L2 camera (requires `--features camera`)

```bash
cargo build --release --features camera

./target/release/counter \
  --model yolov8n.onnx \
  --source 0 \
  --line-x1 640 --line-y1 0 --line-x2 640 --line-y2 480
```

`0` means `/dev/video0`.  Use `camera:1` for the second device.

---

## `--source` argument

The `--source` value is auto-detected in this order:

| Pattern | Behaviour |
|---------|-----------|
| Existing **directory** | Sorted JPEG/PNG images; loops by default |
| Bare integer or `camera:N` | V4L2 device N (requires `camera` feature) |
| `rtsp://`, `rtsps://`, `http://`, `https://`, `hls://` prefix | ffmpeg network decode |
| Existing file with video extension (`.mp4`, `.avi`, `.mkv`, …) | ffmpeg file decode |
| Existing file with image extension (`.jpg`, `.jpeg`, `.png`) | Single-frame list |

---

## Additional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--fps N` | 0 (uncapped) | Cap inference output to N frames per second.  Useful for file playback at natural speed or for reducing CPU load. |
| `--loop-source` / `--no-loop` | on for finite sources | Loop the source after it is exhausted.  Automatically disabled for live sources (camera, network stream). |
| `--conf F` | 0.25 | Minimum detection confidence (0–1). |
| `--nms-iou F` | 0.45 | IoU threshold for non-maximum suppression. |
| `--bind ADDR` | 0.0.0.0:3000 | TCP address for the dashboard HTTP server. |
| `--line-x1 / --line-y1 / --line-x2 / --line-y2` | — | Counting-line endpoints in pixel space. |

---

## Dashboard & API

| URL             | Description                                                |
|-----------------|------------------------------------------------------------|
| `GET /`         | HTML page: live MJPEG stream + count cards (auto-refresh)  |
| `GET /stream`   | Raw MJPEG stream (`multipart/x-mixed-replace`)             |
| `GET /count`    | `{"entered": N, "left": N, "net": N}` JSON                 |
| `GET /health`   | `{"status": "ok"}` liveness probe                          |

The `/stream` endpoint serves the browser's `<img src="/stream">` tag directly.
Each frame is a JPEG annotated with:
- cyan bounding boxes around each tracked person
- track ID label above each box (e.g. `#7`)
- red counting line
- counts overlay (In / Out / Net) in the top-left corner

---

## Cargo features

| Feature  | Default | Description |
|----------|---------|-------------|
| `camera` | off     | Enables live V4L2 camera capture (Jetson / Linux only).  Not required for file-based input or CI. |

```bash
# Build with camera support
cargo build --features camera
```

---

## Bundled assets

| File | Notes |
|------|-------|
| `assets/walk-frames/` | 795 JPEG frames from a pedestrian scene (768×576) |
| `assets/walk.avi` | Source video (msmpeg4v3, 10 fps) |
| `assets/DejaVuSans.ttf` | Font embedded at compile-time via `include_bytes!` |
| `assets/DejaVuSans-LICENSE.txt` | Bitstream Vera / DejaVu permissive licence |

---

## How it works

1. **Detection** — each frame is letterbox-resized to 640×640 and fed to
   YOLOv8n via ONNX Runtime.  Only the *person* class (COCO class 0) is
   decoded.  CUDA is registered first; CPU is the automatic fallback.

2. **Tracking** — a greedy IoU-based multi-object tracker assigns stable IDs
   across frames.  Tracks that go unmatched for more than 5 consecutive frames
   are pruned.

3. **Counting** — a virtual counting line is defined at startup.  When a
   track's centroid crosses the line the direction (entered vs. left) is
   determined by the sign of the cross product of the line vector and the
   displacement vector.

4. **Annotation** — `draw.rs` uses `imageproc` (drawing primitives) and an
   embedded DejaVuSans font (`ab_glyph`) to burn boxes, labels, the line, and
   count text into each RGBA frame.

5. **Multi-source decode** — `source.rs` classifies the `--source` argument
   and dispatches to the appropriate reader: a directory walker, an ffmpeg
   subprocess (`ffmpeg.rs`), or a V4L2 camera.  ffmpeg decodes any container
   or protocol it supports and pipes JPEG frames to the inference loop via an
   OS thread + `std::sync::mpsc`.

6. **MJPEG stream** — annotated frames are JPEG-encoded and published to a
   `tokio::sync::watch` channel.  The `/stream` handler subscribes, assembles
   `multipart/x-mixed-replace` parts, and streams them as an infinite
   `axum::Body::from_stream`.  A slow browser client simply misses intermediate
   frames (correct MJPEG semantics — no frame queue needed).

7. **Dashboard** — the HTML page loads the MJPEG stream via a plain
   `<img src="/stream">` tag, and polls `/count` every second for the numeric
   stats.  Both run concurrently via `tokio::spawn`.
