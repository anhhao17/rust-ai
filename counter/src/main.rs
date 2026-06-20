//! `counter` — people counter with YOLOv8 detection, multi-object tracking,
//! line-crossing counting, annotated MJPEG video stream, and a live dashboard.
//!
//! # Sources (`--source`)
//!
//! | Value | Behaviour |
//! |-------|-----------|
//! | `/path/to/dir` | Sorted JPEG/PNG images (loops by default) |
//! | `/path/to/video.mp4` (or .avi, .mkv, …) | ffmpeg file decode |
//! | `rtsp://…` / `http://…` / `https://…` / `hls://…` | ffmpeg network stream |
//! | `0` / `camera:0` | V4L2 camera (requires `--features camera`) |
//!
//! # Usage
//!
//! ```text
//! counter --model yolov8n.onnx --source assets/walk-frames/ \
//!         --line-x1 384 --line-y1 0 --line-x2 384 --line-y2 576
//!
//! counter --model yolov8n.onnx --source assets/walk.avi --no-loop
//!
//! counter --model yolov8n.onnx --source rtsp://192.168.1.1:554/live \
//!         --fps 15 --bind 0.0.0.0:3000
//! ```
//!
//! Dashboard: http://localhost:3000/
//! Stream:    http://localhost:3000/stream
//! Count API: http://localhost:3000/count
//!
//! # Cargo features
//!
//! - `camera` (default: **off**) — enables live V4L2 camera capture.
//!
//! # On-Jetson / CUDA note
//!
//! The session builder registers CUDA first then falls back to CPU.  No GPU or
//! model file is required for `cargo build` or `cargo test`.

use std::{
    net::SocketAddr,
    path::PathBuf,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, Result, anyhow};
use bytes::Bytes;
use clap::Parser;
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};
use tracing_subscriber::EnvFilter;
use vision_core::session::build_session;

mod dashboard;
mod draw;
mod ffmpeg;
mod line_counter;
mod postprocess;
mod preprocess;
mod source;
mod tracker;

use dashboard::SharedAppState;
use line_counter::{CountingLine, LineCounter};
use source::{SourceKind, detect_source_kind};
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
    about = "People counter — YOLOv8 detection + tracking + line-crossing + MJPEG live dashboard"
)]
struct Args {
    /// Path to the YOLOv8n ONNX model file.
    #[arg(long, value_name = "FILE")]
    model: PathBuf,

    /// Video source: image directory, video file, RTSP/HTTP/HLS URL, or
    /// camera index (e.g. `0` or `camera:0`; requires the `camera` feature).
    #[arg(long, value_name = "SOURCE")]
    source: String,

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

    /// Loop the source after it is exhausted (default: on for finite sources).
    /// Automatically disabled for live sources (camera, network stream).
    /// Pass `--no-loop` to play a file once and then keep the server alive.
    #[arg(long, default_value_t = true, action = clap::ArgAction::Set)]
    loop_source: bool,

    /// Cap output to at most N frames per second.  0 = uncapped (default).
    /// Useful for file playback at a natural rate and for reducing CPU load on
    /// fast inference hardware.
    #[arg(long, default_value_t = 0, value_name = "N")]
    fps: u32,
}

// ---------------------------------------------------------------------------
// Pipeline context
// ---------------------------------------------------------------------------

/// Shared inference state threaded through every source runner.
///
/// Grouping these into one struct keeps the per-source entry points to a
/// manageable number of arguments and makes it easy to add new fields later
/// without changing every call site.
///
/// The `'static` bound on `FontRef` is satisfied because the font bytes are
/// embedded via `include_bytes!` and therefore live for the duration of the
/// process.
struct PipelineCtx {
    session: Session,
    tracker: Tracker,
    line_counter: LineCounter,
    counting_line: CountingLine,
    font: ab_glyph::FontRef<'static>,
    app_state: SharedAppState,
    frame_tx: dashboard::FrameTx,
    conf: f32,
    nms_iou: f32,
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

    // Detect source type early so we can fail fast before starting the server.
    let source_kind = detect_source_kind(&args.source)
        .with_context(|| format!("invalid --source '{}'", args.source))?;

    // Live sources cannot loop — override any explicit --loop-source=true.
    let should_loop = args.loop_source && source_kind.is_finite();

    tracing::info!(
        source = %args.source,
        kind = ?source_kind,
        loop_source = should_loop,
        fps_cap = args.fps,
        "source detected"
    );

    // Validate that camera sources have the feature compiled in.
    if matches!(source_kind, SourceKind::Camera(_)) {
        #[cfg(not(feature = "camera"))]
        return Err(anyhow!(
            "camera source requires the `camera` cargo feature; \
             rebuild with `--features camera`"
        ));
    }

    // Build shared state and the frame-publishing channel.
    let (app_state, frame_tx) = dashboard::new_state();

    // Spawn the dashboard server in a background task.
    let dashboard_state: SharedAppState = Arc::clone(&app_state);
    let bind_addr = args.bind;
    tokio::spawn(async move {
        if let Err(err) = run_dashboard(dashboard_state, bind_addr).await {
            tracing::error!("dashboard error: {err:#}");
        }
    });

    tracing::info!(bind = %args.bind, "dashboard started — open http://{}/", args.bind);

    let font = draw::load_font().context("failed to load embedded font")?;
    let session = build_session(&args.model)?;

    let mut ctx = PipelineCtx {
        session,
        tracker: Tracker::new(),
        line_counter: LineCounter::new(counting_line),
        counting_line,
        font,
        app_state,
        frame_tx,
        conf: args.conf,
        nms_iou: args.nms_iou,
    };

    let min_frame_interval = fps_cap_interval(args.fps);

    // Dispatch to the appropriate source reader.
    match source_kind {
        SourceKind::FrameDir(ref dir_path) => {
            run_frame_dir_loop(dir_path, should_loop, min_frame_interval, &mut ctx).await?;
        }
        SourceKind::VideoFile(ref file_path) => {
            run_ffmpeg_loop(
                file_path.to_str().unwrap_or_default(),
                should_loop,
                min_frame_interval,
                &mut ctx,
            )
            .await?;
        }
        SourceKind::NetworkStream(ref url) => {
            // Network sources never loop; `should_loop` was already set to false.
            run_ffmpeg_loop(url, should_loop, min_frame_interval, &mut ctx).await?;
        }
        SourceKind::Camera(_device_index) => {
            #[cfg(feature = "camera")]
            run_camera_loop(_device_index, min_frame_interval, &mut ctx).await?;

            #[cfg(not(feature = "camera"))]
            // Unreachable: we already returned Err above for Camera without the feature.
            // This arm is present to satisfy the exhaustiveness check.
            unreachable!("camera source without camera feature");
        }
    }

    tracing::info!("source exhausted — dashboard remains live; press Ctrl-C to exit");

    // Keep the process alive so the dashboard can still be queried.
    tokio::signal::ctrl_c()
        .await
        .context("failed to listen for Ctrl-C")?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Source runners
// ---------------------------------------------------------------------------

/// Runs the inference loop over a directory of image files (or a single image
/// file).  Loops when `should_loop` is true.
async fn run_frame_dir_loop(
    dir_path: &std::path::Path,
    should_loop: bool,
    min_frame_interval: Option<Duration>,
    ctx: &mut PipelineCtx,
) -> Result<()> {
    let frame_paths = collect_frame_paths(dir_path)?;
    tracing::info!(
        frames = frame_paths.len(),
        source = %dir_path.display(),
        loop_source = should_loop,
        "starting frame-dir inference loop"
    );

    let mut frame_idx: u64 = 0;

    loop {
        for frame_path in &frame_paths {
            let img = image::open(frame_path)
                .with_context(|| format!("failed to open frame: {}", frame_path.display()))?;

            process_frame(img, frame_idx, min_frame_interval, ctx).await?;

            frame_idx += 1;
        }

        if !should_loop {
            break;
        }

        tracing::debug!(
            "frame-dir sequence exhausted after {} frames — looping",
            frame_paths.len()
        );
    }

    Ok(())
}

/// Runs the inference loop reading frames from an ffmpeg subprocess.
/// Used for both local video files and network streams.
async fn run_ffmpeg_loop(
    source: &str,
    should_loop: bool,
    min_frame_interval: Option<Duration>,
    ctx: &mut PipelineCtx,
) -> Result<()> {
    let mut frame_idx: u64 = 0;

    loop {
        tracing::info!(source, "spawning ffmpeg");

        let mut reader = ffmpeg::FfmpegReader::spawn(source)?;

        while let Some(frame_result) = reader.recv() {
            match frame_result {
                Ok(img) => {
                    process_frame(img, frame_idx, min_frame_interval, ctx).await?;
                    frame_idx += 1;
                }
                Err(err) => {
                    // A single bad frame is logged and skipped; we do not abort.
                    tracing::warn!("frame decode error (frame {frame_idx}): {err:#}");
                }
            }
        }

        // Wait for ffmpeg to clean up.
        let status = reader.wait()?;
        tracing::info!(exit_code = ?status.code(), "ffmpeg exited");

        if !should_loop {
            break;
        }

        tracing::debug!(frame_idx, "ffmpeg source exhausted — looping");
    }

    Ok(())
}

/// Camera capture loop — compiled only when the `camera` feature is enabled.
#[cfg(feature = "camera")]
async fn run_camera_loop(
    device: u32,
    min_frame_interval: Option<Duration>,
    ctx: &mut PipelineCtx,
) -> Result<()> {
    use image::DynamicImage;
    use nokhwa::{
        Camera,
        pixel_format::RgbFormat,
        utils::{CameraIndex, RequestedFormat, RequestedFormatType},
    };

    let mut camera = Camera::new(
        CameraIndex::Index(device),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )
    .with_context(|| format!("failed to open camera device {device}"))?;

    camera
        .open_stream()
        .context("failed to open camera stream")?;

    tracing::info!(device, "camera stream opened");

    let mut frame_idx: u64 = 0;
    loop {
        let buffer = camera.frame().context("failed to capture camera frame")?;
        let rgb_buf = buffer
            .decode_image::<RgbFormat>()
            .context("failed to decode camera frame")?;
        let img = DynamicImage::ImageRgb8(rgb_buf);

        process_frame(img, frame_idx, min_frame_interval, ctx).await?;
        frame_idx += 1;
    }
}

// ---------------------------------------------------------------------------
// Shared per-frame processing
// ---------------------------------------------------------------------------

/// Runs detection → tracking → counting → annotation → publish on one frame.
///
/// All source runners call this function so the inference pipeline is defined
/// exactly once.
async fn process_frame(
    img: image::DynamicImage,
    frame_idx: u64,
    min_frame_interval: Option<Duration>,
    ctx: &mut PipelineCtx,
) -> Result<()> {
    let frame_start = Instant::now();

    let (orig_w, orig_h) = (img.width(), img.height());
    let (tensor, params) = preprocess::letterbox_and_normalise(&img);
    let raw_output = run_ort_inference(&mut ctx.session, tensor)?;

    let detections =
        postprocess::decode_persons(&raw_output, &params, ctx.conf, ctx.nms_iou, orig_w, orig_h)?;

    let tracks = ctx.tracker.update(&detections);
    ctx.line_counter.update(tracks);
    let tally = ctx.line_counter.tally();

    // Publish count.
    match ctx.app_state.count.lock() {
        Ok(mut state) => *state = tally,
        Err(_) => tracing::warn!("count state mutex poisoned — skipping update"),
    }

    // Annotate and publish JPEG frame.
    let annotated = draw::annotate_frame(&img, tracks, ctx.counting_line, tally, &ctx.font);
    match draw::encode_jpeg(&annotated, dashboard::STREAM_JPEG_QUALITY) {
        Ok(jpeg_bytes) => {
            let _ = ctx.frame_tx.send(Some(Bytes::from(jpeg_bytes)));
        }
        Err(err) => tracing::warn!("failed to JPEG-encode frame {frame_idx}: {err:#}"),
    }

    tracing::debug!(
        frame = frame_idx,
        persons = detections.len(),
        tracks = tracks.len(),
        entered = tally.entered,
        left = tally.left,
        net = tally.net(),
    );

    // Yield so the dashboard task can serve requests.
    tokio::task::yield_now().await;

    // FPS cap: sleep for the remainder of the minimum inter-frame interval.
    if let Some(interval) = min_frame_interval {
        let elapsed = frame_start.elapsed();
        if elapsed < interval {
            tokio::time::sleep(interval - elapsed).await;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns the minimum inter-frame interval for an FPS cap, or `None` for
/// uncapped.
///
/// `fps == 0` means uncapped.
fn fps_cap_interval(fps: u32) -> Option<Duration> {
    if fps == 0 {
        None
    } else {
        Some(Duration::from_secs_f64(1.0 / fps as f64))
    }
}

/// Starts the axum dashboard server on `bind_addr`.
///
/// # Errors
///
/// Returns `Err` if the TCP listener fails to bind or if axum exits unexpectedly.
async fn run_dashboard(state: SharedAppState, bind_addr: SocketAddr) -> Result<()> {
    let app = dashboard::build_router(state);
    let listener = tokio::net::TcpListener::bind(bind_addr)
        .await
        .with_context(|| format!("failed to bind dashboard to {bind_addr}"))?;

    axum::serve(listener, app)
        .await
        .context("dashboard server exited with error")?;

    Ok(())
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
fn collect_frame_paths(input_path: &std::path::Path) -> Result<Vec<PathBuf>> {
    if input_path.is_file() {
        return Ok(vec![input_path.to_path_buf()]);
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
        let result = collect_frame_paths(dir.path()).unwrap();
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn collect_frame_paths_returns_sorted_order() {
        let dir = temp_dir_with_files(&["c.jpg", "a.jpg", "b.jpg"]);
        let result = collect_frame_paths(dir.path()).unwrap();
        let names: Vec<String> = result
            .iter()
            .map(|p| p.file_name().unwrap().to_str().unwrap().to_owned())
            .collect();
        assert_eq!(names, vec!["a.jpg", "b.jpg", "c.jpg"]);
    }

    #[test]
    fn collect_frame_paths_errors_on_empty_directory() {
        let dir = temp_dir_with_files(&[]);
        let result = collect_frame_paths(dir.path());
        assert!(result.is_err(), "empty directory should produce Err");
    }

    #[test]
    fn collect_frame_paths_errors_on_nonexistent_path() {
        let path = PathBuf::from("/tmp/counter_test_nonexistent_xyz");
        let result = collect_frame_paths(&path);
        assert!(result.is_err());
    }

    #[test]
    fn fps_cap_zero_means_uncapped() {
        assert!(fps_cap_interval(0).is_none());
    }

    #[test]
    fn fps_cap_30_gives_correct_interval() {
        let interval = fps_cap_interval(30).unwrap();
        // 1/30 s ≈ 33.3 ms
        assert!((interval.as_secs_f64() - 1.0 / 30.0).abs() < 1e-6);
    }

    #[test]
    fn fps_cap_1_gives_one_second_interval() {
        let interval = fps_cap_interval(1).unwrap();
        assert!((interval.as_secs_f64() - 1.0).abs() < 1e-6);
    }
}
