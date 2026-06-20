//! `classify` — MobileNetV2 image classifier CLI.
//!
//! Loads a MobileNetV2 ONNX model via ONNX Runtime, pre-processes an input
//! image with ImageNet normalization, runs inference, and prints the top-3
//! predicted ImageNet labels with their confidence scores.
//!
//! # Usage
//! ```text
//! classify --model mobilenetv2.onnx --labels imagenet_classes.txt --image dog.jpg
//! ```

use anyhow::{Context, Result};
use clap::Parser;
use ndarray::Array4;
use ort::{session::Session, value::TensorRef};
use std::{fs, path::PathBuf};

mod postprocess;
mod preprocess;

use vision_core::session::build_session;

/// ImageNet input width expected by MobileNetV2.
const INPUT_WIDTH: u32 = 224;

/// ImageNet input height expected by MobileNetV2.
const INPUT_HEIGHT: u32 = 224;

/// How many top predictions to display.
const TOP_K: usize = 3;

/// CLI arguments for the classify binary.
#[derive(Parser, Debug)]
#[command(
    name = "classify",
    version,
    about = "Classify an image using a MobileNetV2 ONNX model"
)]
struct Args {
    /// Path to the MobileNetV2 ONNX model file.
    #[arg(long, value_name = "FILE")]
    model: PathBuf,

    /// Path to the ImageNet class labels file (one label per line).
    #[arg(long, value_name = "FILE")]
    labels: PathBuf,

    /// Path to the input image file.
    #[arg(long, value_name = "FILE")]
    image: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let labels = load_labels(&args.labels)?;
    let mut session = build_session(&args.model)?;

    let tensor = preprocess::load_and_preprocess(&args.image, INPUT_WIDTH, INPUT_HEIGHT)?;
    let outputs = run_inference(&mut session, &tensor)?;
    let top = postprocess::top_k_softmax(&outputs, TOP_K);

    print_results(&top, &labels);

    Ok(())
}

/// Loads the text file at `path`, returning one label string per line.
fn load_labels(path: &PathBuf) -> Result<Vec<String>> {
    let raw = fs::read_to_string(path)
        .with_context(|| format!("failed to read labels file: {}", path.display()))?;
    let labels: Vec<String> = raw.lines().map(str::to_owned).collect();
    if labels.is_empty() {
        anyhow::bail!("labels file is empty: {}", path.display());
    }
    Ok(labels)
}

/// Runs inference on `tensor` and returns the raw logits as a flat `Vec<f32>`.
fn run_inference(session: &mut Session, tensor: &Array4<f32>) -> Result<Vec<f32>> {
    let input_ref = TensorRef::from_array_view(tensor.view())
        .with_context(|| "failed to create input tensor view")?;

    let outputs = session
        .run(ort::inputs!["input" => input_ref])
        .with_context(|| "ORT session run failed")?;

    let (_shape, logits) = outputs[0]
        .try_extract_tensor::<f32>()
        .with_context(|| "failed to extract output tensor")?;

    Ok(logits.to_vec())
}

/// Prints the top-k `(class_index, confidence)` pairs with their label strings.
fn print_results(top: &[(usize, f32)], labels: &[String]) {
    println!("Top-{} predictions:", top.len());
    for (rank, &(class_idx, confidence)) in top.iter().enumerate() {
        let label = labels
            .get(class_idx)
            .map(String::as_str)
            .unwrap_or("<unknown>");
        println!("  {}. {:>6.2}%  {}", rank + 1, confidence * 100.0, label);
    }
}
