//! `detect` — YOLOv8 real-time object detector CLI.
//!
//! Loads a YOLOv8n ONNX model via ONNX Runtime, reads frames from a V4L2
//! camera or a single image file, runs object detection, and saves annotated
//! output images.
//!
//! # Usage — single image
//! ```text
//! detect --model yolov8n.onnx image --input photo.jpg
//! detect --model yolov8n.onnx image --input photo.jpg --output annotated.jpg
//! ```
//!
//! # Usage — live camera
//! ```text
//! detect --model yolov8n.onnx camera --device 0 --output-dir ./frames
//! ```

use anyhow::{Context, Result, anyhow};
use clap::{Parser, Subcommand};
use image::DynamicImage;
use ndarray::Array4;
use nokhwa::{
    Camera,
    pixel_format::RgbFormat,
    utils::{CameraIndex, RequestedFormat, RequestedFormatType},
};
use ort::{
    execution_providers::{CPUExecutionProvider, CUDAExecutionProvider},
    session::Session,
    session::builder::GraphOptimizationLevel,
    value::TensorRef,
};
use std::path::PathBuf;

mod coco_labels;
mod draw;
mod postprocess;
mod preprocess;

/// Default confidence threshold for filtering detections.
const DEFAULT_CONF_THRESHOLD: f32 = 0.25;

/// Default IoU threshold for non-maximum suppression.
const DEFAULT_NMS_IOU_THRESHOLD: f32 = 0.45;

/// CLI arguments for the detect binary.
#[derive(Parser, Debug)]
#[command(
    name = "detect",
    version,
    about = "YOLOv8 real-time object detector with bounding-box overlay"
)]
struct Args {
    /// Path to the YOLOv8n ONNX model file.
    #[arg(long, value_name = "FILE")]
    model: PathBuf,

    /// Minimum class confidence score to report a detection (0–1).
    #[arg(long, default_value_t = DEFAULT_CONF_THRESHOLD)]
    conf: f32,

    /// IoU threshold for non-maximum suppression (0–1).
    #[arg(long, default_value_t = DEFAULT_NMS_IOU_THRESHOLD)]
    nms_iou: f32,

    #[command(subcommand)]
    source: Source,
}

/// Input source subcommand.
#[derive(Subcommand, Debug)]
enum Source {
    /// Run detection on a single image file and save the annotated result.
    Image {
        /// Input image path.
        #[arg(long, value_name = "FILE")]
        input: PathBuf,
        /// Output image path (defaults to `annotated_<input>`).
        #[arg(long, value_name = "FILE")]
        output: Option<PathBuf>,
    },
    /// Capture frames from a V4L2 camera and save annotated JPEG files.
    Camera {
        /// V4L2 device index (0 = /dev/video0).
        #[arg(long, default_value_t = 0)]
        device: u32,
        /// Directory to write annotated frames into.
        #[arg(long, value_name = "DIR", default_value = "frames")]
        output_dir: PathBuf,
        /// Maximum number of frames to capture (0 = run until interrupted).
        #[arg(long, default_value_t = 0)]
        max_frames: u64,
    },
}

fn main() -> Result<()> {
    let args = Args::parse();

    let mut session = build_session(&args.model)?;

    match &args.source {
        Source::Image { input, output } => {
            run_on_image(
                &mut session,
                input,
                output.as_deref(),
                args.conf,
                args.nms_iou,
            )?;
        }
        Source::Camera {
            device,
            output_dir,
            max_frames,
        } => {
            run_on_camera(
                &mut session,
                *device,
                output_dir,
                *max_frames,
                args.conf,
                args.nms_iou,
            )?;
        }
    }

    Ok(())
}

/// Builds an ORT session with CUDA preferred, CPU as fallback.
fn build_session(model_path: &PathBuf) -> Result<Session> {
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

/// Detects objects in a single image and saves the annotated result.
fn run_on_image(
    session: &mut Session,
    input_path: &PathBuf,
    output_path: Option<&std::path::Path>,
    conf: f32,
    nms_iou: f32,
) -> Result<()> {
    let img = image::open(input_path)
        .with_context(|| format!("failed to open image: {}", input_path.display()))?;

    let detections = detect_in_frame(session, &img, conf, nms_iou)?;

    println!("Detections in {}:", input_path.display());
    for (i, det) in detections.iter().enumerate() {
        println!(
            "  [{i}] class={} ({}) conf={:.1}%  box=[{:.0},{:.0},{:.0},{:.0}]",
            det.class_id,
            coco_labels::label(det.class_id),
            det.confidence * 100.0,
            det.x1,
            det.y1,
            det.x2,
            det.y2,
        );
    }

    let annotated = draw::draw_detections(&img, &detections);

    let out = match output_path {
        Some(p) => p.to_path_buf(),
        None => {
            let stem = input_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("image");
            let ext = input_path
                .extension()
                .and_then(|s| s.to_str())
                .unwrap_or("jpg");
            input_path
                .parent()
                .unwrap_or(std::path::Path::new("."))
                .join(format!("annotated_{stem}.{ext}"))
        }
    };

    annotated
        .save(&out)
        .with_context(|| format!("failed to save annotated image: {}", out.display()))?;
    println!("Saved annotated image to {}", out.display());

    Ok(())
}

/// Captures frames from a V4L2 camera, runs detection on each, and saves
/// annotated JPEG files into `output_dir`.
fn run_on_camera(
    session: &mut Session,
    device: u32,
    output_dir: &PathBuf,
    max_frames: u64,
    conf: f32,
    nms_iou: f32,
) -> Result<()> {
    std::fs::create_dir_all(output_dir)
        .with_context(|| format!("failed to create output dir: {}", output_dir.display()))?;

    let mut camera = Camera::new(
        CameraIndex::Index(device),
        RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate),
    )
    .with_context(|| format!("failed to open camera device {device}"))?;

    camera
        .open_stream()
        .with_context(|| "failed to open camera stream")?;

    let run_forever = max_frames == 0;
    let mut frame_idx: u64 = 0;

    loop {
        if !run_forever && frame_idx >= max_frames {
            break;
        }

        let buffer = camera
            .frame()
            .with_context(|| "failed to capture camera frame")?;

        let rgb_buf = buffer
            .decode_image::<RgbFormat>()
            .with_context(|| "failed to decode frame to RGB")?;

        let img = DynamicImage::ImageRgb8(rgb_buf);
        let detections = detect_in_frame(session, &img, conf, nms_iou)?;

        println!("Frame {frame_idx}: {} detection(s)", detections.len());
        for det in &detections {
            println!(
                "  {} ({:.1}%)  [{:.0},{:.0},{:.0},{:.0}]",
                coco_labels::label(det.class_id),
                det.confidence * 100.0,
                det.x1,
                det.y1,
                det.x2,
                det.y2,
            );
        }

        let annotated = draw::draw_detections(&img, &detections);
        let out_path = output_dir.join(format!("frame_{frame_idx:06}.jpg"));
        annotated
            .save(&out_path)
            .with_context(|| format!("failed to save frame: {}", out_path.display()))?;

        frame_idx += 1;
    }

    Ok(())
}

/// Runs the full detection pipeline on a single frame: preprocess → infer → postprocess.
fn detect_in_frame(
    session: &mut Session,
    img: &DynamicImage,
    conf: f32,
    nms_iou: f32,
) -> Result<Vec<postprocess::Detection>> {
    let (orig_w, orig_h) = (img.width(), img.height());
    let (tensor, params) = preprocess::letterbox_and_normalise(img);

    let logits = run_inference(session, tensor)?;

    let detections =
        postprocess::decode_yolov8_output(&logits, &params, conf, nms_iou, orig_w, orig_h)?;

    Ok(detections)
}

/// Runs ORT inference on `tensor` and returns the flat raw output.
fn run_inference(session: &mut Session, tensor: Array4<f32>) -> Result<Vec<f32>> {
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
