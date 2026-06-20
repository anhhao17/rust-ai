//! ONNX Runtime session construction with CUDA + CPU fallback.
//!
//! Every inference crate in the workspace builds sessions the same way: register
//! the CUDA execution provider first, then fall back to CPU automatically when
//! CUDA hardware or drivers are absent.  This module centralises that logic so
//! it lives in one place and the four consumer crates stay lean.

use anyhow::{Context, anyhow};
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::Session,
    session::builder::GraphOptimizationLevel,
};

/// ORT graph-optimization level applied to every session.
///
/// Level 3 enables all available passes (constant folding, node fusion, etc.),
/// which is appropriate for a fixed-model production deployment.
const OPT_LEVEL: GraphOptimizationLevel = GraphOptimizationLevel::Level3;

/// Builds an ONNX Runtime [`Session`] from a model file, preferring CUDA and
/// falling back to CPU automatically.
///
/// The CUDA execution provider is registered first.  If CUDA hardware or
/// drivers are unavailable, ORT silently selects the CPU provider instead, so
/// the binary runs on any development machine without modification.
///
/// Graph optimisation is set to [`GraphOptimizationLevel::Level3`] (all
/// available passes).
///
/// # Errors
///
/// Returns `Err` if:
/// - the ORT session builder cannot be created,
/// - registering the execution providers fails, or
/// - `model_path` does not exist or cannot be parsed as a valid ONNX model.
pub fn build_session(model_path: &std::path::Path) -> anyhow::Result<Session> {
    let builder =
        Session::builder().map_err(|e| anyhow!("failed to create ORT session builder: {e}"))?;

    let builder = builder
        .with_optimization_level(OPT_LEVEL)
        .map_err(|e| anyhow!("failed to set optimization level: {}", e.message()))?;

    let mut builder = builder
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])
        .map_err(|e| anyhow!("failed to register execution providers: {}", e.message()))?;

    builder
        .commit_from_file(model_path)
        .with_context(|| format!("failed to load model: {}", model_path.display()))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    /// Loading a nonexistent model path must return Err, not panic.
    #[test]
    fn build_session_nonexistent_path_returns_err() {
        let result = build_session(Path::new("/nonexistent/path/model.onnx"));
        assert!(
            result.is_err(),
            "expected Err for a nonexistent model path, got Ok"
        );
    }
}
