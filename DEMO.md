# Demo run guide

Four apps, each verified running on CPU (no CUDA GPU required).

The CUDA execution-provider registration prints a one-line error on startup
(`libcublasLt.so.12: cannot open shared object file`) and then falls back to
CPU transparently — this is expected on a dev box with no GPU.

---

## Quick start

```bash
# 1. Download / export all assets (idempotent — safe to re-run)
bash scripts/setup-demo.sh

# 2. Build all four crates in release mode
cargo build --release -p classify -p detect -p server -p counter
```

First-run compile takes a few minutes.  Subsequent runs are fast.

---

## 1. classify

Classifies a single image with MobileNetV2 and prints the top-3 ImageNet labels.

```bash
cargo run --release -p classify -- \
    --model  assets/mobilenetv2-12.onnx \
    --labels assets/imagenet_classes.txt \
    --image  assets/sample.jpg
```

Expected output (Labrador photo):

```
Top-3 predictions:
  1.  95.25%  Labrador retriever
  2.   1.78%  kuvasz
  3.   1.11%  golden retriever
```

---

## 2. detect

Runs YOLOv8n on a single image and writes an annotated JPEG with bounding boxes.

```bash
cargo run --release -p detect -- \
    --model assets/yolov8n.onnx \
    image --input assets/sample.jpg
```

Expected output:

```
Detections in assets/sample.jpg:
  [0] class=16 (dog) conf=85.4%  box=[166,49,500,372]
  [1] class=0 (person) conf=46.6%  box=[0,255,204,372]
  [2] class=77 (teddy bear) conf=26.5%  box=[62,225,208,369]
Saved annotated image to assets/annotated_sample.jpg
```

The annotated image is written to `assets/annotated_sample.jpg`.

---

## 3. server

Axum REST server: POST an image, get top-5 predictions as JSON.

**Terminal 1 — start the server:**

```bash
cargo run --release -p server -- \
    --model  assets/mobilenetv2-12.onnx \
    --labels assets/imagenet_classes.txt \
    --bind   0.0.0.0:3000
```

The server logs `listening addr=0.0.0.0:3000` when ready.

**Terminal 2 — test it:**

```bash
# Liveness probe
curl http://localhost:3000/health
# {"status":"ok","model_loaded":true}

# Classify an image
curl -X POST http://localhost:3000/classify \
     -F image=@assets/sample.jpg
```

Expected classify response:

```json
{
  "predictions": [
    {"label": "Labrador retriever",    "confidence": 0.9524825},
    {"label": "kuvasz",                "confidence": 0.017803796},
    {"label": "golden retriever",      "confidence": 0.011086132},
    {"label": "Rhodesian ridgeback",   "confidence": 0.0018225348},
    {"label": "Chesapeake Bay retriever", "confidence": 0.0011392175}
  ]
}
```

Stop the server with Ctrl-C.

---

## 4. counter

YOLOv8 person detection + tracking + line-crossing counting on a video,
with a live web dashboard.

`setup-demo.sh` extracts `assets/sample_video.mp4` into `assets/frames/`
(171 JPEG frames at 30 fps) automatically.

**Terminal 1 — run the counter:**

```bash
cargo run --release -p counter -- \
    --model  assets/yolov8n.onnx \
    --input  assets/frames/ \
    --line-x1 0   --line-y1 240 \
    --line-x2 640 --line-y2 240 \
    --bind   0.0.0.0:3000
```

The counter processes all frames in a few seconds, then stays alive serving
the dashboard until you press Ctrl-C.

**Terminal 2 — query the API:**

```bash
curl http://localhost:3000/health
# {"status":"ok"}

curl http://localhost:3000/count
# {"entered":0,"left":0,"net":0}
```

**Browser — live dashboard:**

Open `http://localhost:3000/` — the page polls `/count` every second and
displays Entered / Left / Net counts.

Note: the bundled sample video is a generic 5-second clip; detections appear
in frames 155-165 (1 person) but the person does not cross the counting line,
so counts remain 0.  To see non-zero counts, drop a video with pedestrians
crossing the frame into `assets/` and re-run `setup-demo.sh` to regenerate
frames, or supply your own frame directory.

---

## Supplying your own video (counter)

```bash
# Extract frames from any MP4
ffmpeg -i /path/to/your_video.mp4 -q:v 2 assets/frames/frame_%06d.jpg -y

# Run counter (adjust line coordinates to match your video resolution)
cargo run --release -p counter -- \
    --model  assets/yolov8n.onnx \
    --input  assets/frames/ \
    --line-x1 0    --line-y1 360 \
    --line-x2 1280 --line-y2 360 \
    --bind   0.0.0.0:3000
```
