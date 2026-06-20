//! Axum dashboard for the live people counter.
//!
//! Endpoints:
//! - `GET /`       — HTML dashboard showing the live video stream and count stats.
//! - `GET /stream` — MJPEG video stream (`multipart/x-mixed-replace`).
//! - `GET /count`  — live count as JSON (`{"entered":N,"left":N,"net":N}`).
//! - `GET /health` — liveness probe; always returns HTTP 200.
//!
//! Shared state is [`AppState`], which bundles two things:
//! - a `Mutex<CountTally>` for the numeric count (polled by `/count`)
//! - a `tokio::sync::watch` channel for annotated JPEG frames (streamed by `/stream`)
//!
//! The processing loop writes to both; the HTTP handlers read from both.
//! Using `watch` (not `broadcast`) means the stream endpoint always serves the
//! *latest* frame — a slow browser client simply drops intermediate frames,
//! which is correct MJPEG semantics.

use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderValue, StatusCode, header},
    response::{Html, IntoResponse, Response},
    routing::get,
};
use bytes::Bytes;
use futures_util::stream;
use serde::Serialize;
use tokio::sync::watch;

use crate::line_counter::CountTally;

// ---------------------------------------------------------------------------
// MJPEG framing constants
// ---------------------------------------------------------------------------

/// Boundary token used to delimit MJPEG frames (arbitrary printable string).
const MJPEG_BOUNDARY: &str = "mjpeg_frame";

/// JPEG quality level used when encoding annotated frames for streaming.
/// 75 gives good visual quality at a moderate bandwidth cost.
pub const STREAM_JPEG_QUALITY: u8 = 75;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// The latest annotated JPEG frame.  `None` until the first frame is produced.
pub type FrameBytes = Option<Bytes>;

/// Sender half of the frame watch channel.  The processing loop calls
/// `frame_tx.send(...)` after encoding each annotated frame.
pub type FrameTx = watch::Sender<FrameBytes>;

/// Receiver half of the frame watch channel.  Each `/stream` connection clones
/// this and subscribes independently.
pub type FrameRx = watch::Receiver<FrameBytes>;

/// All shared state passed into every axum handler via `State`.
#[derive(Clone)]
pub struct AppState {
    /// Live count tally updated by the processing loop.
    pub count: Arc<Mutex<CountTally>>,
    /// Latest annotated JPEG frame (watch receiver).
    pub frame_rx: FrameRx,
}

/// Convenience alias used in `main` and tests.
pub type SharedAppState = Arc<AppState>;

/// Creates the shared [`AppState`] and returns it together with the sender half
/// of the frame channel so the processing loop can publish new frames.
pub fn new_state() -> (SharedAppState, FrameTx) {
    let (frame_tx, frame_rx) = watch::channel(None);
    let state = Arc::new(AppState {
        count: Arc::new(Mutex::new(CountTally::default())),
        frame_rx,
    });
    (state, frame_tx)
}

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// JSON body returned by `GET /count`.
#[derive(Debug, Serialize)]
pub struct CountResponse {
    /// Total people who entered.
    pub entered: u64,
    /// Total people who left.
    pub left: u64,
    /// Net occupancy (entered − left).
    pub net: i64,
}

/// JSON body returned by `GET /health`.
#[derive(Debug, Serialize)]
pub struct HealthResponse {
    /// Always `"ok"`.
    pub status: &'static str,
}

// ---------------------------------------------------------------------------
// Router builder
// ---------------------------------------------------------------------------

/// Builds and returns the axum `Router` for the dashboard.
///
/// Extracted from `main` so integration tests can construct the router without
/// spawning a TCP listener.
pub fn build_router(state: SharedAppState) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/stream", get(stream_handler))
        .route("/count", get(count_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /` — HTML dashboard page with embedded video stream and count stats.
async fn index_handler() -> impl IntoResponse {
    Html(DASHBOARD_HTML)
}

/// `GET /stream` — MJPEG streaming endpoint.
///
/// Returns a `multipart/x-mixed-replace` response.  Each part is one JPEG
/// frame delimited by `--mjpeg_frame`.  The browser's `<img src="/stream">`
/// tag renders this as live video.
///
/// When no frame has been published yet the stream blocks until one arrives.
/// Between frames the handler waits for the watch channel to be updated; a
/// slow consumer just misses intermediate frames — that is correct MJPEG
/// semantics (no frame queue needed).
async fn stream_handler(State(state): State<SharedAppState>) -> Response {
    let rx = state.frame_rx.clone();

    // Build a stream of Result<Bytes, Infallible> chunks.  Each iteration
    // waits for the watch channel to change, then emits one MJPEG part.
    // Body::from_stream requires Result items; we use Infallible as the error
    // type because all errors are handled internally.
    use std::convert::Infallible;
    let frame_stream = stream::unfold(rx.clone(), move |mut rx| async move {
        // Wait for the next frame change.
        // `changed()` returns Err only when the sender is dropped (binary exiting).
        if rx.changed().await.is_err() {
            return None; // sender gone — close the stream gracefully
        }

        let frame_bytes: Option<Bytes> = rx.borrow().clone();

        let jpeg = match frame_bytes {
            Some(bytes) => bytes,
            // No frame available yet — yield an empty chunk and keep waiting
            // so the browser connection stays alive.
            None => return Some((Ok::<Bytes, Infallible>(Bytes::new()), rx)),
        };

        // Assemble the MJPEG part header + body as a single Bytes chunk.
        let part_header = format!(
            "--{MJPEG_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {}\r\n\r\n",
            jpeg.len()
        );
        let mut part = Vec::with_capacity(part_header.len() + jpeg.len() + 2);
        part.extend_from_slice(part_header.as_bytes());
        part.extend_from_slice(&jpeg);
        part.extend_from_slice(b"\r\n");

        Some((Ok::<Bytes, Infallible>(Bytes::from(part)), rx))
    });

    let body = Body::from_stream(frame_stream);

    Response::builder()
        .status(StatusCode::OK)
        .header(
            header::CONTENT_TYPE,
            HeaderValue::from_static("multipart/x-mixed-replace; boundary=mjpeg_frame"),
        )
        // Disable buffering proxies from accumulating all frames before sending.
        .header(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"))
        .header(header::PRAGMA, HeaderValue::from_static("no-cache"))
        .body(body)
        .unwrap_or_else(|_| {
            // Response builder only fails on invalid header values, which can't
            // happen with our static literals above.
            Response::new(Body::empty())
        })
}

/// `GET /count` — returns the live count as JSON.
///
/// # Errors
///
/// Returns HTTP 500 if the mutex is poisoned (the writer thread panicked).
async fn count_handler(State(state): State<SharedAppState>) -> impl IntoResponse {
    match state.count.lock() {
        Ok(tally) => {
            let response = CountResponse {
                entered: tally.entered,
                left: tally.left,
                net: tally.net(),
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(_) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": "state lock poisoned"})),
        )
            .into_response(),
    }
}

/// `GET /health` — liveness probe.
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(HealthResponse { status: "ok" }))
}

// ---------------------------------------------------------------------------
// Dashboard HTML (inline — no runtime static-file dependency on the Jetson)
// ---------------------------------------------------------------------------

const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>People Counter</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      display: flex;
      flex-direction: column;
      align-items: center;
      padding: 1.5rem;
      min-height: 100vh;
    }
    h1 { font-size: 1.5rem; margin-bottom: 1rem; }
    .stream-container {
      width: 100%;
      max-width: 900px;
      background: #1e293b;
      border-radius: 12px;
      overflow: hidden;
      margin-bottom: 1.5rem;
      box-shadow: 0 4px 24px rgba(0,0,0,0.5);
    }
    .stream-container img {
      width: 100%;
      display: block;
    }
    .stream-placeholder {
      width: 100%;
      aspect-ratio: 16/9;
      display: flex;
      align-items: center;
      justify-content: center;
      color: #475569;
      font-size: 0.9rem;
    }
    .cards {
      display: flex;
      gap: 1.5rem;
      flex-wrap: wrap;
      justify-content: center;
    }
    .card {
      background: #1e293b;
      border-radius: 12px;
      padding: 1.2rem 2rem;
      text-align: center;
      min-width: 130px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .card .label {
      font-size: 0.75rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #94a3b8;
      margin-bottom: 0.4rem;
    }
    .card .value { font-size: 2.5rem; font-weight: 700; }
    .entered { color: #34d399; }
    .left    { color: #f87171; }
    .net     { color: #60a5fa; }
    .status  { margin-top: 1rem; font-size: 0.75rem; color: #64748b; }
  </style>
</head>
<body>
  <h1>People Counter — Live View</h1>

  <div class="stream-container">
    <img src="/stream" alt="Live camera stream"
         onerror="this.style.display='none';document.getElementById('placeholder').style.display='flex';" />
    <div class="stream-placeholder" id="placeholder" style="display:none;">
      Waiting for stream…
    </div>
  </div>

  <div class="cards">
    <div class="card">
      <div class="label">Entered</div>
      <div class="value entered" id="entered">–</div>
    </div>
    <div class="card">
      <div class="label">Left</div>
      <div class="value left" id="left">–</div>
    </div>
    <div class="card">
      <div class="label">Net (inside)</div>
      <div class="value net" id="net">–</div>
    </div>
  </div>
  <div class="status" id="status">connecting…</div>

  <script>
    async function refreshCount() {
      try {
        const resp = await fetch('/count');
        if (!resp.ok) throw new Error('HTTP ' + resp.status);
        const data = await resp.json();
        document.getElementById('entered').textContent = data.entered;
        document.getElementById('left').textContent    = data.left;
        document.getElementById('net').textContent     = data.net;
        document.getElementById('status').textContent  =
          'Last updated: ' + new Date().toLocaleTimeString();
      } catch (err) {
        document.getElementById('status').textContent = 'Error: ' + err.message;
      }
    }
    refreshCount();
    setInterval(refreshCount, 1000);
  </script>
</body>
</html>
"#;

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::Body, http::Request};
    use tower::util::ServiceExt; // for `oneshot`

    fn test_router_with_state(entered: u64, left: u64) -> (axum::Router, FrameTx) {
        let (state, tx) = new_state();
        // Pre-populate the count so the /count tests can check values.
        *state.count.lock().unwrap() = CountTally { entered, left };
        (build_router(state), tx)
    }

    // --- /health ---

    #[tokio::test]
    async fn health_endpoint_returns_200() {
        let (app, _tx) = test_router_with_state(0, 0);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    // --- /count ---

    #[tokio::test]
    async fn count_endpoint_returns_zeros_on_fresh_state() {
        let (app, _tx) = test_router_with_state(0, 0);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/count")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["entered"], 0);
        assert_eq!(json["left"], 0);
        assert_eq!(json["net"], 0);
    }

    #[tokio::test]
    async fn count_endpoint_reflects_updated_state() {
        let (app, _tx) = test_router_with_state(7, 3);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/count")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(json["entered"], 7);
        assert_eq!(json["left"], 3);
        assert_eq!(json["net"], 4);
    }

    // --- / ---

    #[tokio::test]
    async fn index_endpoint_returns_html_with_stream_tag() {
        let (app, _tx) = test_router_with_state(0, 0);
        let resp = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let html = std::str::from_utf8(&body).unwrap();
        assert!(html.contains("People Counter"));
        assert!(
            html.contains("/stream"),
            "dashboard HTML must reference /stream"
        );
        assert!(
            html.contains("/count"),
            "dashboard HTML must reference /count"
        );
    }

    // --- /stream ---

    #[tokio::test]
    async fn stream_endpoint_returns_200_with_multipart_content_type() {
        let (app, _tx) = test_router_with_state(0, 0);
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let content_type = resp
            .headers()
            .get(header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("");

        assert!(
            content_type.contains("multipart/x-mixed-replace"),
            "expected multipart/x-mixed-replace content-type, got: {content_type}"
        );
        assert!(
            content_type.contains(MJPEG_BOUNDARY),
            "content-type must include the boundary token, got: {content_type}"
        );
    }

    #[tokio::test]
    async fn stream_publishes_jpeg_frame_bytes() {
        let (state, tx) = new_state();
        let app = build_router(Arc::clone(&state));

        // Publish a synthetic 2-byte "JPEG" (real frames would be larger; we
        // only need to confirm the boundary + bytes flow through the channel).
        let fake_jpeg = Bytes::from_static(b"\xFF\xD8"); // JPEG magic bytes
        tx.send(Some(fake_jpeg.clone())).unwrap();

        // Hit /stream and collect a small amount of the response body.
        let resp = app
            .oneshot(
                Request::builder()
                    .uri("/stream")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        // Read a bounded amount — the stream is infinite while the sender is
        // alive, so we just verify the first chunk contains the MJPEG preamble.
        use axum::body::to_bytes;
        // Drop the sender so the stream terminates cleanly, then read.
        drop(tx);
        let body = to_bytes(resp.into_body(), 4096).await.unwrap();
        let body_str = String::from_utf8_lossy(&body);

        assert!(
            body_str.contains(&format!("--{MJPEG_BOUNDARY}")),
            "response body must contain the MJPEG boundary"
        );
        assert!(
            body_str.contains("Content-Type: image/jpeg"),
            "response body must contain per-frame content-type"
        );
    }
}
