# Orin NX — Rust + Embedded AI Learning Roadmap

**Goal:** Build on existing Yocto/embedded Linux skills to learn Rust and on-device AI, using a Jetson Orin NX board.

---

## The Core Pattern

The Orin NX has an Ampere GPU, so heavy AI inference wants CUDA/TensorRT. Rust doesn't replace that. The productive split is:

- **Rust owns the system layer** — camera capture, GPIO/sensors, networking, concurrency, application logic.
- **Inference runs through a Rust ML crate** that talks to the GPU.

A typical project is: *Rust app captures data → runs a model → acts on the result.*

### Key Rust ML Crates

| Crate | What it is | Best for |
|-------|-----------|----------|
| `ort` | ONNX Runtime bindings (CUDA + TensorRT execution providers) | Vision; near-native perf on Jetson |
| `candle` | Pure Rust ML from HuggingFace, CUDA support | LLMs and audio; easiest to embed |
| `tch-rs` | libtorch bindings | Powerful but heavy |

---

## Project Ideas

### Starter — learn Rust + basic inference

1. **Image classifier CLI**
   Load a MobileNet/ResNet ONNX model with `ort`, classify an image from the command line.
   *Teaches:* `clap` for args, Rust `Result`/error handling, the inference loop. Tiny but complete.

2. **Real-time object detector**
   YOLO ONNX model on a USB/CSI camera feed, draw bounding boxes.
   *Teaches:* the video pipeline, frames-per-second, basic async.

### Intermediate — systems + AI together

3. **Edge inference server**
   A small `axum` web server exposing a REST endpoint that runs a model and returns JSON.
   *Teaches:* Rust web/serialization, managing GPU state across requests. Genuinely useful.

4. **People counter / smart camera**
   Detection + simple tracking; counts people entering/leaving, pushes a live count to a small dashboard.
   *Teaches:* combining models with stateful logic.

5. **Wake-word or keyword spotting**
   Capture audio in Rust, run a small audio model (Whisper-tiny via `candle` works on Orin NX).
   *Teaches:* low-latency edge audio — a strong embedded skill.

### Ambitious — portfolio-level

6. **Multi-model pipeline**
   Chain models: detect → crop → OCR or classify, with Rust managing concurrency and backpressure so the GPU stays busy without stalling.
   *Where Rust's strengths really show.*

7. **Local LLM assistant**
   Run a quantized small LLM (via `candle`) on the Orin NX with a Rust frontend. The 16GB Orin NX handles a small quantized model comfortably. A fully offline assistant.

8. **Rust ROS2 perception node**
   Use `r2r` to write a perception node. Pairs naturally with the Orin's role in robots/drones.

---

## The Differentiator (uses all three skills)

Take any project above and **bake the Rust binary into a custom Yocto image as a systemd service**, using a `cargo-bitbake`-generated recipe.

This single project exercises Yocto + Rust + AI at once — almost nobody can do all three well. That's a real differentiator.

---

## Suggested Path

1. **Project #1** (image classifier CLI) — get Rust basics + `ort` inference working.
2. **Project #2** (object detector) — add the camera/video pipeline.
3. Then branch based on interest:
   - Lean systems/backend → **Project #3** (edge inference server)
   - AI itself excites you → **Project #7** (local LLM assistant)
4. Eventually: wrap your favorite build into a **Yocto systemd service** (the differentiator).

---

## Next Step

Pick a direction — **computer vision**, **audio/LLM**, or **robotics** — and define a concrete first build:
- exact model (e.g. specific ONNX file)
- crate setup (`ort` vs `candle`)
- cargo dependencies
- the minimal end-to-end "hello world" inference loop