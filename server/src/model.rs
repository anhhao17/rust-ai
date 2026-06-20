//! ONNX Runtime session wrapper for MobileNetV2 inference.
//!
//! This module owns the session lifecycle: build once at startup, then serve
//! classification requests for the lifetime of the process.
//!
//! # Concurrency
//!
//! `ort::Session::run` is NOT guaranteed to be `Sync` across the ONNX Runtime
//! versions in use (rc.12 of the 2.0 series).  Rather than relying on an
//! undocumented thread-safety promise, we wrap the session in a `Mutex`.  Each
//! incoming request acquires the lock, runs inference, then releases it.
//!
//! Throughput trade-off: serialized inference is the conservative default
//! for the Jetson Orin NX target (single GPU, bounded concurrency anyway).
//! If a future profiling run shows lock contention is a bottleneck, consider:
//!   - A `deadpool` of sessions, or
//!   - Session cloning once `Session: Clone` lands in a stable ORT release.

use std::path::Path;
use std::sync::Mutex;

use anyhow::{Context, anyhow};
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};

use crate::postprocess;
use crate::preprocess;
use vision_core::session::build_session;

/// ImageNet input width expected by MobileNetV2.
pub const INPUT_WIDTH: u32 = 224;

/// ImageNet input height expected by MobileNetV2.
pub const INPUT_HEIGHT: u32 = 224;

/// How many top predictions to return per request.
pub const DEFAULT_TOP_K: usize = 5;

/// A loaded MobileNetV2 ONNX model ready for inference.
///
/// The session is wrapped in a `Mutex` to serialize concurrent inference calls
/// (see module-level doc for the reasoning).  The labels vector is read-only
/// after construction and therefore needs no lock.
pub struct Classifier {
    session: Mutex<Session>,
    labels: Vec<String>,
}

impl Classifier {
    /// Loads the ONNX model from `model_path` and the class labels from
    /// `labels_path`, returning a `Classifier` ready for inference.
    ///
    /// Tries CUDA first; falls back to CPU silently when no CUDA hardware or
    /// drivers are present, so this works on any development machine.
    ///
    /// # Errors
    ///
    /// Returns an error if the model file is missing, the ORT session cannot be
    /// built, or the labels file is missing/empty.
    pub fn load(model_path: &Path, labels_path: &Path) -> anyhow::Result<Self> {
        let session = build_session(model_path)?;
        let labels = load_labels(labels_path)?;
        Ok(Self {
            session: Mutex::new(session),
            labels,
        })
    }

    /// Runs MobileNetV2 inference on `image_bytes` and returns the top-`top_k`
    /// predicted `(label, confidence)` pairs sorted by descending confidence.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `image_bytes` cannot be decoded as a recognized image format,
    /// - the ORT session fails to run inference, or
    /// - the output tensor cannot be extracted.
    pub fn classify(&self, image_bytes: &[u8], top_k: usize) -> anyhow::Result<Vec<(String, f32)>> {
        let tensor = preprocess::decode_and_preprocess(image_bytes, INPUT_WIDTH, INPUT_HEIGHT)
            .context("image preprocessing failed")?;

        let logits = self.run_inference(&tensor)?;

        let top = postprocess::top_k_softmax(&logits, top_k);

        let predictions = top
            .into_iter()
            .map(|(idx, confidence)| {
                let label = self
                    .labels
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| format!("<unknown class {idx}>"));
                (label, confidence)
            })
            .collect();

        Ok(predictions)
    }

    /// Acquires the session lock and runs a single inference pass on `tensor`.
    ///
    /// Returns the raw logits as a flat `Vec<f32>`.
    fn run_inference(&self, tensor: &Array4<f32>) -> anyhow::Result<Vec<f32>> {
        // Safety: Mutex::lock only returns Err if a thread panicked while
        // holding the lock.  We treat that as an unrecoverable server error.
        let mut session = self
            .session
            .lock()
            .map_err(|_| anyhow!("inference session mutex was poisoned"))?;

        let input_ref = TensorRef::from_array_view(tensor.view())
            .context("failed to create input tensor view")?;

        let outputs = session
            .run(ort::inputs!["input" => input_ref])
            .context("ORT session run failed")?;

        let (_shape, logits) = outputs[0]
            .try_extract_tensor::<f32>()
            .context("failed to extract output tensor")?;

        Ok(logits.to_vec())
    }
}

/// Loads the text file at `path`, returning one label string per line.
fn load_labels(path: &Path) -> anyhow::Result<Vec<String>> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("failed to read labels file: {}", path.display()))?;
    let labels: Vec<String> = raw.lines().map(str::to_owned).collect();
    if labels.is_empty() {
        anyhow::bail!("labels file is empty: {}", path.display());
    }
    Ok(labels)
}
