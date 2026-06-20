# classify

Command-line image classifier using MobileNetV2 and ONNX Runtime.

## Requirements

- Rust toolchain (edition 2024 / Rust 1.85+)
- A MobileNetV2 ONNX model file
- ImageNet class labels text file

The binary uses the ONNX Runtime CUDA execution provider when CUDA is available
and falls back to the CPU automatically, so it works on any development machine.

## Obtaining the model

Download the ONNX opset-12 MobileNetV2 model from the official ONNX Model Zoo:

```bash
wget https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx \
     -O mobilenetv2.onnx
```

Alternatively, export it from PyTorch/torchvision:

```python
import torch, torchvision
model = torchvision.models.mobilenet_v2(weights="IMAGENET1K_V1").eval()
dummy = torch.zeros(1, 3, 224, 224)
torch.onnx.export(model, dummy, "mobilenetv2.onnx",
                  input_names=["input"], output_names=["output"],
                  opset_version=12)
```

## Obtaining the class labels

```bash
wget https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt
```

This file contains 1000 lines, one ImageNet class label per line, matching the
output index order of MobileNetV2.

## Building

```bash
cd classify
cargo build --release
```

## Running

```bash
./target/release/classify \
    --model mobilenetv2.onnx \
    --labels imagenet_classes.txt \
    --image path/to/photo.jpg
```

Example output:

```
Top-3 predictions:
  1.  82.41%  golden retriever
  2.   9.13%  Labrador retriever
  3.   3.22%  kuvasz
```

## Running tests

Tests cover pre/post-processing logic only and require no model file or GPU:

```bash
cargo test
```

## Notes for Jetson Orin NX deployment

When running on the Jetson with CUDA drivers installed, ONNX Runtime will
automatically select the CUDA execution provider.  For maximum throughput,
build with `--release` and consider enabling the TensorRT execution provider in
`build_session` (add `TensorRTExecutionProvider::default().build()` before CUDA).
