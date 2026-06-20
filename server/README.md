# server

Edge inference REST server for MobileNetV2 image classification.

Loads a MobileNetV2 ONNX model once at startup and serves inference over HTTP.
The CUDA execution provider is registered automatically and falls back to CPU
when no CUDA hardware is present, so the binary runs on any development machine.

## Endpoints

| Method | Path        | Description                              |
|--------|-------------|------------------------------------------|
| GET    | `/health`   | Liveness probe — returns 200 immediately |
| POST   | `/classify` | Run inference on an uploaded image       |

### `GET /health`

Returns HTTP 200 with a JSON body indicating whether the model is loaded.

```bash
curl http://localhost:3000/health
```

Example response:

```json
{"status":"ok","model_loaded":true}
```

### `POST /classify`

Accepts a `multipart/form-data` request with a single field named `image`
containing the raw image bytes (JPEG, PNG, WebP, BMP, etc.).

```bash
curl -X POST http://localhost:3000/classify \
     -F image=@/path/to/photo.jpg
```

Example response (HTTP 200):

```json
{
  "predictions": [
    {"label": "golden retriever",    "confidence": 0.8241},
    {"label": "Labrador retriever",  "confidence": 0.0913},
    {"label": "kuvasz",              "confidence": 0.0322},
    {"label": "clumber",             "confidence": 0.0128},
    {"label": "Sussex spaniel",      "confidence": 0.0097}
  ]
}
```

Error responses use a JSON body with `error` (machine-readable code) and
`message` (human-readable description):

```json
{"error":"bad_request","message":"failed to decode image bytes: ..."}
```

| HTTP status | `error` code       | When                                         |
|-------------|--------------------|----------------------------------------------|
| 400         | `bad_request`      | Missing/empty `image` field, undecodable bytes |
| 503         | `model_not_loaded` | Server started without a model file          |
| 500         | `inference_error`  | Unexpected ORT runtime failure               |

## Requirements

- Rust toolchain (edition 2024 / Rust 1.85+)
- A MobileNetV2 ONNX model file
- An ImageNet class labels text file

## Obtaining the model and labels

These are identical to the assets used by the `classify` crate; see
`../classify/README.md` for details.  Quick reference:

```bash
# Model (ONNX Model Zoo, opset-12)
wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx \
     -O mobilenetv2.onnx

# Labels (1000 ImageNet classes, one per line)
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

Alternatively, export from PyTorch/torchvision:

```python
import torch, torchvision
model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()
dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(model, dummy, "mobilenetv2.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=12)
```

## Building

```bash
cd server
cargo build --release
```

## Running

```bash
./target/release/server \
    --model mobilenetv2.onnx \
    --labels imagenet_classes.txt \
    --bind 0.0.0.0:3000
```

All options can also be set via environment variables:

```bash
MODEL_PATH=mobilenetv2.onnx \
LABELS_PATH=imagenet_classes.txt \
BIND_ADDR=0.0.0.0:3000 \
./target/release/server
```

## Running tests

Tests cover the HTTP layer (health, error paths) and pure processing logic
(softmax, top-k, normalization) and require no model file or GPU:

```bash
cargo test
```

## Notes for Jetson Orin NX deployment

When running on the Jetson with CUDA drivers installed, ONNX Runtime selects
the CUDA execution provider automatically.  For maximum throughput, build with
`--release` and consider adding the TensorRT execution provider in
`model::build_session` before the CUDA entry.

The inference session is protected by a `Mutex` (one request at a time) — a
conservative choice for the single-GPU Jetson target.  If profiling shows
lock contention as a bottleneck, a `deadpool` of sessions is the recommended
next step; see the comment in `src/model.rs`.
