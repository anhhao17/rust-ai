//! `server` — edge inference REST server for MobileNetV2 image classification.
//!
//! Loads a MobileNetV2 ONNX model once at startup, then serves inference
//! requests over HTTP.  Two endpoints are available:
//!
//! - `GET /health` — liveness probe; responds immediately without running
//!   inference.
//! - `POST /classify` — accepts a multipart upload with an `image` field and
//!   returns the top-5 ImageNet predictions as JSON.
//!
//! # Configuration
//!
//! All options can be set via CLI flags **or** environment variables (env vars
//! take precedence when a flag is omitted):
//!
//! | Flag            | Env var        | Default                  |
//! |-----------------|----------------|--------------------------|
//! | `--model`       | `MODEL_PATH`   | `mobilenetv2.onnx`       |
//! | `--labels`      | `LABELS_PATH`  | `imagenet_classes.txt`   |
//! | `--bind`        | `BIND_ADDR`    | `0.0.0.0:3000`           |
//!
//! # Example
//!
//! ```text
//! server --model /models/mobilenetv2.onnx --labels /models/labels.txt --bind 0.0.0.0:3000
//! ```

use std::{net::SocketAddr, path::PathBuf, sync::Arc};

use anyhow::{Context, Result};
use axum::{
    Router,
    routing::{get, post},
};
use clap::Parser;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

mod handlers;
mod model;
mod postprocess;
mod preprocess;

use handlers::AppState;

/// Default model file path (relative to the working directory or absolute).
const DEFAULT_MODEL_PATH: &str = "mobilenetv2.onnx";

/// Default class labels file path.
const DEFAULT_LABELS_PATH: &str = "imagenet_classes.txt";

/// Default bind address for the HTTP listener.
const DEFAULT_BIND_ADDR: &str = "0.0.0.0:3000";

/// CLI arguments for the edge inference server.
#[derive(Parser, Debug)]
#[command(
    name = "server",
    version,
    about = "MobileNetV2 edge inference REST server"
)]
struct Args {
    /// Path to the MobileNetV2 ONNX model file.
    #[arg(
        long,
        value_name = "FILE",
        env = "MODEL_PATH",
        default_value = DEFAULT_MODEL_PATH
    )]
    model: PathBuf,

    /// Path to the ImageNet class labels file (one label per line).
    #[arg(
        long,
        value_name = "FILE",
        env = "LABELS_PATH",
        default_value = DEFAULT_LABELS_PATH
    )]
    labels: PathBuf,

    /// Address and port to bind the HTTP server to.
    #[arg(
        long,
        value_name = "ADDR:PORT",
        env = "BIND_ADDR",
        default_value = DEFAULT_BIND_ADDR
    )]
    bind: SocketAddr,
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging.  RUST_LOG controls the filter level;
    // default directive surfaces INFO-level messages from this crate without
    // flooding the output with tokio/hyper internals.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env().add_directive("server=info".parse()?))
        .init();

    let args = Args::parse();

    tracing::info!(
        model = %args.model.display(),
        labels = %args.labels.display(),
        bind = %args.bind,
        "loading model"
    );

    // Load the model.  If the file is missing this returns a clear error
    // message and exits — no panics, no partial startup.
    let classifier = model::Classifier::load(&args.model, &args.labels).with_context(|| {
        format!(
            "failed to start server: could not load model from '{}'",
            args.model.display()
        )
    })?;

    let state: AppState = Arc::new(Some(classifier));
    let app = build_router(state);

    tracing::info!(addr = %args.bind, "listening");

    let listener = tokio::net::TcpListener::bind(args.bind)
        .await
        .with_context(|| format!("failed to bind to {}", args.bind))?;

    axum::serve(listener, app)
        .await
        .context("server exited with error")?;

    Ok(())
}

/// Builds and returns the axum `Router` with all routes and middleware wired up.
///
/// Extracted into a standalone function so integration tests can construct a
/// router with a custom state without going through `main`.
pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(handlers::health))
        .route("/classify", post(handlers::classify))
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}
