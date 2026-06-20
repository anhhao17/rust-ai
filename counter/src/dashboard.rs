//! Minimal axum dashboard for the live people count.
//!
//! Endpoints:
//! - `GET /count`  — returns `{"entered":N,"left":N,"net":N}` JSON.
//! - `GET /`       — serves a tiny static HTML page with auto-refreshing count.
//! - `GET /health` — liveness probe; always returns HTTP 200.
//!
//! The count state is shared as `Arc<Mutex<CountTally>>` so the inference
//! loop can update it from a separate thread while the axum task serves it.

use std::sync::{Arc, Mutex};

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::get,
};
use serde::Serialize;

use crate::line_counter::CountTally;

// ---------------------------------------------------------------------------
// Shared state
// ---------------------------------------------------------------------------

/// Thread-safe handle to the live count — passed into every axum handler via `State`.
pub type CountState = Arc<Mutex<CountTally>>;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

/// JSON body returned by `GET /count`.
#[derive(Debug, Serialize)]
pub struct CountResponse {
    /// Total people who entered (crossed inside → outside direction).
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
pub fn build_router(state: CountState) -> Router {
    Router::new()
        .route("/", get(index_handler))
        .route("/count", get(count_handler))
        .route("/health", get(health_handler))
        .with_state(state)
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

/// `GET /` — serves the static dashboard HTML page.
async fn index_handler() -> impl IntoResponse {
    Html(DASHBOARD_HTML)
}

/// `GET /count` — returns the live count as JSON.
///
/// # Errors
///
/// Returns HTTP 500 if the mutex is poisoned (should never happen in practice).
async fn count_handler(State(state): State<CountState>) -> impl IntoResponse {
    match state.lock() {
        Ok(tally) => {
            let response = CountResponse {
                entered: tally.entered,
                left: tally.left,
                net: tally.net(),
            };
            (StatusCode::OK, Json(response)).into_response()
        }
        Err(_) => {
            // Mutex poisoning means the writer thread panicked — this should
            // not happen in normal operation, but we return 500 rather than
            // crashing the dashboard server.
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({"error": "state lock poisoned"})),
            )
                .into_response()
        }
    }
}

/// `GET /health` — liveness probe.
async fn health_handler() -> impl IntoResponse {
    (StatusCode::OK, Json(HealthResponse { status: "ok" }))
}

// ---------------------------------------------------------------------------
// Static HTML
// ---------------------------------------------------------------------------

/// Inline dashboard page — polls `/count` every second and updates the display.
///
/// Kept inline rather than as a separate asset file so the binary is
/// self-contained (no `static/` directory required at runtime on the Jetson).
const DASHBOARD_HTML: &str = r#"<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>People Counter</title>
  <style>
    body {
      font-family: system-ui, sans-serif;
      background: #0f172a;
      color: #e2e8f0;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
    }
    h1 { font-size: 1.8rem; margin-bottom: 2rem; }
    .cards {
      display: flex;
      gap: 2rem;
      flex-wrap: wrap;
      justify-content: center;
    }
    .card {
      background: #1e293b;
      border-radius: 12px;
      padding: 1.5rem 2.5rem;
      text-align: center;
      min-width: 140px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    }
    .card .label {
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.1em;
      color: #94a3b8;
      margin-bottom: 0.5rem;
    }
    .card .value {
      font-size: 3rem;
      font-weight: 700;
    }
    .entered { color: #34d399; }
    .left    { color: #f87171; }
    .net     { color: #60a5fa; }
    .status  { margin-top: 2rem; font-size: 0.8rem; color: #64748b; }
  </style>
</head>
<body>
  <h1>People Counter</h1>
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
    async function refresh() {
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
    refresh();
    setInterval(refresh, 1000);
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
    use axum::{
        Router,
        body::Body,
        http::{Request, StatusCode},
    };
    use tower::util::ServiceExt; // for `oneshot`

    fn test_router(entered: u64, left: u64) -> Router {
        let state: CountState = Arc::new(Mutex::new(CountTally { entered, left }));
        build_router(state)
    }

    // --- /health ---

    #[tokio::test]
    async fn health_endpoint_returns_200() {
        let app = test_router(0, 0);
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
        let app = test_router(0, 0);
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
        let app = test_router(7, 3);
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
    async fn index_endpoint_returns_html() {
        let app = test_router(0, 0);
        let resp = app
            .oneshot(Request::builder().uri("/").body(Body::empty()).unwrap())
            .await
            .unwrap();

        assert_eq!(resp.status(), StatusCode::OK);

        let body = axum::body::to_bytes(resp.into_body(), usize::MAX)
            .await
            .unwrap();
        let html = std::str::from_utf8(&body).unwrap();

        // Sanity-check that the dashboard HTML contains key identifiers.
        assert!(html.contains("People Counter"));
        assert!(html.contains("/count"));
    }
}
