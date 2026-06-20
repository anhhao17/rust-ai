//! `counter` — people counter with YOLOv8 detection, multi-object tracking,
//! line-crossing counting, and a live dashboard.
//!
//! # What it does
//!
//! 1. Loads a YOLOv8n ONNX model (person class only).
//! 2. Reads frames from a video file (or image directory) via the `image` crate.
//! 3. Runs per-frame person detection → multi-object tracking → line-crossing
//!    counting.
//! 4. Publishes the live count on a small axum dashboard.
//!
//! # Usage
//!
//! ```text
//! counter --model yolov8n.onnx --input frames/ \
//!         --line-x1 0 --line-y1 360 --line-x2 1280 --line-y2 360
//! ```
//!
//! Dashboard: http://localhost:3000/
//! Count API:  http://localhost:3000/count
//!
//! # Cargo features
//!
//! - `camera` (default: **off**) — enables live V4L2 camera capture.  Not
//!   needed (and not compiled) when running on an x86 dev box or in CI.
//!
//! # On-Jetson / CUDA note
//!
//! The session builder registers CUDA first then falls back to CPU, matching
//! the pattern in the `detect` and `server` crates.  No GPU or model file is
//! required for `cargo build` or `cargo test`.

use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::{Context, Result, anyhow};
use clap::Parser;
use ndarray::Array4;
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::Session,
    session::builder::GraphOptimizationLevel,
    value::TensorRef,
};
use tracing_subscriber::EnvFilter;

mod dashboard;
mod line_counter;
mod postprocess;
mod preprocess;
mod tracker;

use dashboard::CountState;
use line_counter::{CountTally, CountingLine, LineCounter};
use tracker::Tracker;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Default confidence threshold for person detections.
const DEFAULT_CONF_THRESHOLD: f32 = 0.25;

/// Default IoU threshold for non-maximum suppression.
const DEFAULT_NMS_IOU_THRESHOLD: f32 = 0.45;

/// Default HTTP bind address for the dashboard.
const DEFAULT_BIND_ADDR: &str = "0.0.0.0:3000";

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

/// CLI arguments for the people counter binary.
#[derive(Parser, Debug)]
#[command(
    name = "counter",
    version,
    about = "People counter — YOLOv8 detection + tracking + line-crossing + live dashboard"
)]
struct Args {
    /// Path to the YOLOv8n ONNX model file.
    #[arg(long, value_name = "FILE")]
    model: PathBuf,

    /// Input: path to a single image file or a directory of JPEG/PNG images.
    #[arg(long, value_name = "PATH")]
    input: PathBuf,

    /// Counting-line start point — x coordinate (pixels).
    #[arg(long, default_value_t = 0.0)]
    line_x1: f32,

    /// Counting-line start point — y coordinate (pixels).
    #[arg(long, default_value_t = 360.0)]
    line_y1: f32,

    /// Counting-line end point — x coordinate (pixels).
    #[arg(long, default_value_t = 1280.0)]
    line_x2: f32,

    /// Counting-line end point — y coordinate (pixels).
    #[arg(long, default_value_t = 360.0)]
    line_y2: f32,

    /// Minimum class confidence score (0–1).
    #[arg(long, default_value_t = DEFAULT_CONF_THRESHOLD)]
    conf: f32,

    /// IoU threshold for non-maximum suppression (0–1).
    #[arg(long, default_value_t = DEFAULT_NMS_IOU_THRESHOLD)]
    nms_iou: f32,

    /// HTTP address for the dashboard (e.g. 0.0.0.0:3000).
    #[arg(long, default_value = DEFAULT_BIND_ADDR)]
    bind: SocketAddr,
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("counter=info".parse()?))
        .init();

    let args = Args::parse();

    let counting_line = CountingLine {
        start: (args.line_x1, args.line_y1),
        end: (args.line_x2, args.line_y2),
    };

    // Start with a zero count; the dashboard reflects this even before any
    // frames are processed.
    let count_state: CountState = Arc::new(Mutex::new(CountTally::default()));

    // Spawn the dashboard server in a background task so inference runs
    // concurrently with the HTTP server without blocking either.
    let dashboard_state = Arc::clone(&count_state);
    let bind_addr = args.bind;
    tokio::spawn(async move {
        if let Err(err) = run_dashboard(dashboard_state, bind_addr).await {
            tracing::error!("dashboard error: {err:#}");
        }
    });

    tracing::info!(bind = %args.bind, "dashboard started");

    // Load the model.  Missing model file → clear error and exit, no panic.
    let mut session = build_ort_session(&args.model)?;

    // Collect frames from the input path.
    let frame_paths = collect_frame_paths(&args.input)?;
    tracing::info!(frames = frame_paths.len(), input = %args.input.display(), "processing input");

    let mut tracker = Tracker::new();
    let mut line_counter = LineCounter::new(counting_line);

    for (frame_idx, frame_path) in frame_paths.iter().enumerate() {
        let img = image::open(frame_path)
            .with_context(|| format!("failed to open frame: {}", frame_path.display()))?;

        let (orig_w, orig_h) = (img.width(), img.height());
        let (tensor, params) = preprocess::letterbox_and_normalise(&img);

        let raw_output = run_ort_inference(&mut session, tensor)?;

        let detections = postprocess::decode_persons(
            &raw_output,
            &params,
            args.conf,
            args.nms_iou,
            orig_w,
            orig_h,
        )?;

        let tracks = tracker.update(&detections);
        line_counter.update(tracks);

        let tally = line_counter.tally();

        // Publish updated tally to dashboard state.
        match count_state.lock() {
            Ok(mut state) => *state = tally,
            Err(_) => tracing::warn!("count state mutex poisoned — skipping update"),
        }

        tracing::info!(
            frame = frame_idx,
            persons = detections.len(),
            tracks = tracks.len(),
            entered = tally.entered,
            left = tally.left,
            net = tally.net(),
        );
    }

    tracing::info!("input exhausted — dashboard remains live; press Ctrl-C to exit");

    // Keep the process alive so the dashboard can still be queried.
    tokio::signal::ctrl_c()
        .await
        .context("failed to listen for Ctrl-C")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Starts the axum dashboard server on `bind_addr`.
///
/// # Errors
///
/// Returns `Err` if the TCP listener fails to bind or if axum exits unexpectedly.
async fn run_dashboard(state: CountState, bind_addr: SocketAddr) -> Result<()> {
    let app = dashboard::build_router(state);
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .with_context(|| format!("failed to bind dashboard to {bind_addr}"))?;

    axum::serve(listener, app)
        .await
        .context("dashboard server exited with error")?;

    Ok(())
}

/// Builds an ORT session with CUDA preferred, falling back to CPU.
///
/// # Errors
///
/// Returns `Err` if the model file does not exist or cannot be parsed.
fn build_ort_session(model_path: &PathBuf) -> Result<Session> {
    let builder =
        Session::builder().map_err(|e| anyhow!("failed to create ORT session builder: {e}"))?;

    let builder = builder
        .with_optimization_level(GraphOptimizationLevel::Level3)
        .map_err(|e| anyhow!("failed to set optimization level: {}", e.message()))?;

    let mut builder = builder
        .with_execution_providers([
            CUDAExecutionProvider::default().build(),
            CPUExecutionProvider::default().build(),
        ])
        .map_err(|e| anyhow!("failed to register execution providers: {}", e.message()))?;

    let session = builder
        .commit_from_file(model_path)
        .with_context(|| format!("failed to load model: {}", model_path.display()))?;

    Ok(session)
}

/// Runs ORT inference on one NCHW tensor and returns the flat output slice.
///
/// # Errors
///
/// Returns `Err` if the ORT session run fails or the output tensor cannot be
/// extracted.
fn run_ort_inference(session: &mut Session, tensor: Array4<f32>) -> Result<Vec<f32>> {
    let input_ref = TensorRef::from_array_view(tensor.view())
        .with_context(|| "failed to create input tensor view")?;

    let outputs = session
        .run(ort::inputs!["images" => input_ref])
        .with_context(|| "ORT session run failed")?;

    let (_shape, logits) = outputs[0]
        .try_extract_tensor::<f32>()
        .with_context(|| "failed to extract output tensor")?;

    Ok(logits.to_vec())
}

/// Returns a sorted list of image file paths from `input_path`.
///
/// If `input_path` is a file, returns `[input_path]`.  If it is a directory,
/// returns all `.jpg`, `.jpeg`, and `.png` files sorted by name.
///
/// # Errors
///
/// Returns `Err` if `input_path` does not exist or is not a file/directory.
fn collect_frame_paths(input_path: &PathBuf) -> Result<Vec<PathBuf>> {
    if input_path.is_file() {
        return Ok(vec![input_path.clone()]);
    }

    if !input_path.is_dir() {
        return Err(anyhow!(
            "input path does not exist or is neither a file nor a directory: {}",
            input_path.display()
        ));
    }

    let mut paths: Vec<PathBuf> = std::fs::read_dir(input_path)
        .with_context(|| format!("failed to read input directory: {}", input_path.display()))?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|p| {
            matches!(
                p.extension().and_then(|e| e.to_str()),
                Some("jpg") | Some("jpeg") | Some("png")
            )
        })
        .collect();

    paths.sort();

    if paths.is_empty() {
        return Err(anyhow!(
            "no JPEG/PNG images found in: {}",
            input_path.display()
        ));
    }

    Ok(paths)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    // Helper: create a temporary directory with the given filenames.
    fn temp_dir_with_files(names: &[&str]) -> TempDir {
        let dir = tempfile::tempdir().unwrap();
        for name in names {
            fs::write(dir.path().join(name), b"").unwrap();
        }
        dir
    }

    #[test]
    fn collect_frame_paths_returns_single_file() {
        let dir = temp_dir_with_files(&["frame.jpg"]);
        let path = dir.path().join("frame.jpg");
        let result = collect_frame_paths(&path).unwrap();
        assert_eq!(result, vec![path]);
    }

    #[test]
    fn collect_frame_paths_filters_non_image_files() {
        let dir = temp_dir_with_files(&["a.jpg", "b.png", "c.txt", "d.onnx"]);
        let path = dir.path().to_path_buf();
        let result = collect_frame_paths(&path).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn collect_frame_paths_returns_sorted_order() {
        let dir = temp_dir_with_files(&["c.jpg", "a.jpg", "b.jpg"]);
        let path = dir.path().to_path_buf();
        let result = collect_frame_paths(&path).unwrap();
        let names: Vec<String> = result
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
            .collect();
        assert_eq!(names, vec!["a.jpg", "b.jpg", "c.jpg"]);
    }

    #[test]
    fn collect_frame_paths_errors_on_empty_directory() {
        let dir = temp_dir_with_files(&[]);
        let result = collect_frame_paths(&dir.path().to_path_buf());
        assert!(result.is_err(), "empty directory should produce Err");
    }

    #[test]
    fn collect_frame_paths_errors_on_nonexistent_path() {
        let path = PathBuf::from("/tmp/counter_test_nonexistent_xyz");
        let result = collect_frame_paths(&path);
        assert!(result.is_err());
    }
}
