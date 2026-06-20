# counter — People Counter / Smart Camera

Detects people in a video or image sequence, tracks them across frames, counts
line crossings, and publishes a live count to a small web dashboard.

**Teaching point:** combining a YOLO object-detection model with stateful logic
(tracking + counting).

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

# Single image
./target/release/counter \
  --model yolov8n.onnx \
  --input photo.jpg

# Directory of JPEG/PNG frames (e.g. extracted from a video with ffmpeg)
ffmpeg -i video.mp4 -q:v 2 frames/frame_%06d.jpg

./target/release/counter \
  --model yolov8n.onnx \
  --input frames/ \
  --line-x1 0 --line-y1 360 --line-x2 1280 --line-y2 360 \
  --bind 0.0.0.0:3000
```

The counting line is defined by two points in image-pixel space.  The example
above places a horizontal line at y=360 spanning the full 1280-pixel width,
which is a sensible default for a 1280×720 scene.

---

## Dashboard

Once running, open **http://localhost:3000/** in a browser.  The page polls
`/count` every second and displays:

| Field   | Meaning                        |
|---------|--------------------------------|
| Entered | People who crossed inside→out  |
| Left    | People who crossed out→inside  |
| Net     | `entered − left`               |

### API

```
GET /count   → {"entered": N, "left": N, "net": N}
GET /health  → {"status": "ok"}
```

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

4. **Dashboard** — an axum HTTP server runs concurrently with the inference
   loop via `tokio::spawn`.  The live tally is shared as
   `Arc<Mutex<CountTally>>`.
