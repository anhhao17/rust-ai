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

# Directory of JPEG/PNG frames (e.g. extracted from a video with ffmpeg)
ffmpeg -i video.mp4 -q:v 2 frames/frame_%06d.jpg

./target/release/counter \
  --model yolov8n.onnx \
  --input frames/ \
  --line-x1 384 --line-y1 0 --line-x2 384 --line-y2 576 \
  --bind 0.0.0.0:3000
```

Then open **http://localhost:3000/** — you'll see the live annotated video with
bounding boxes, track IDs, counting line, and the count stats below it.

The counting line is defined by two pixel-space points.  The example above
places a vertical line at x=384 for a 768×576 scene.  For a 1280×720 scene a
horizontal mid-line would be `--line-x1 0 --line-y1 360 --line-x2 1280 --line-y2 360`.

The frame sequence loops automatically when exhausted so the demo plays
continuously without intervention.

### Using the bundled pedestrian frames

```bash
./target/release/counter \
  --model assets/yolov8n.onnx \
  --input assets/walk-frames/ \
  --line-x1 384 --line-y1 0 --line-x2 384 --line-y2 576 \
  --bind 127.0.0.1:3010
```

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

5. **MJPEG stream** — annotated frames are JPEG-encoded and published to a
   `tokio::sync::watch` channel.  The `/stream` handler subscribes, assembles
   `multipart/x-mixed-replace` parts, and streams them as an infinite
   `axum::Body::from_stream`.  A slow browser client simply misses intermediate
   frames (correct MJPEG semantics — no frame queue needed).

6. **Dashboard** — the HTML page loads the MJPEG stream via a plain
   `<img src="/stream">` tag, and polls `/count` every second for the numeric
   stats.  Both run concurrently via `tokio::spawn`.
