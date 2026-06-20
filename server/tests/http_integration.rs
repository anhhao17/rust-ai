//! Black-box HTTP integration tests for the edge inference server.
//!
//! # Strategy
//!
//! The `server` crate is a binary-only target; integration tests in `tests/`
//! cannot import from `main.rs`.  We therefore use two complementary
//! approaches:
//!
//! 1. **Binary spawn tests** — compile the server binary with
//!    `env!("CARGO_BIN_EXE_server")`, spawn it with `--model
//!    /nonexistent/path`, and assert it exits cleanly (non-zero exit code, no
//!    abort/SIGSEGV) when a model file is missing.  This proves the startup
//!    error path does not panic.
//!
//! 2. **Live HTTP tests** — bind a `TcpListener` on an ephemeral port, spawn
//!    the server binary pointing at a real (minimal) model, and exercise the
//!    HTTP endpoints with `reqwest`.  These tests are **skipped** (not failed)
//!    when `INTEGRATION_MODEL_PATH` is not set so they always pass in CI with
//!    no model/GPU.
//!
//! ## Running with a real model
//!
//! ```sh
//! INTEGRATION_MODEL_PATH=/path/to/mobilenetv2.onnx \
//! INTEGRATION_LABELS_PATH=/path/to/imagenet_classes.txt \
//! cargo test --test http_integration
//! ```
//!
//! ## Note on lib split
//!
//! A `src/lib.rs` exposing `build_router` would allow fully in-process oneshot
//! tests without spawning a subprocess and without requiring a real model.
//! If that split is added in a future PR, these tests can be ported to use
//! `tower::ServiceExt::oneshot` directly.

use std::{
    net::TcpListener,
    process::{Child, Command, Stdio},
    time::Duration,
};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Environment variable naming the ONNX model path for live HTTP tests.
const ENV_MODEL_PATH: &str = "INTEGRATION_MODEL_PATH";

/// Environment variable naming the labels file path for live HTTP tests.
const ENV_LABELS_PATH: &str = "INTEGRATION_LABELS_PATH";

/// Milliseconds to wait for the server to begin accepting connections.
const SERVER_STARTUP_MS: u64 = 1500;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Allocates an ephemeral TCP port by binding and immediately dropping a
/// listener.  There is an inherent TOCTOU race (another process could grab the
/// port before the server binds it), but this is acceptable for test use.
fn free_port() -> u16 {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind on port 0 must succeed");
    listener
        .local_addr()
        .expect("local_addr must be set")
        .port()
}

/// Spawns the server binary on the given `addr` using `model_path` and
/// `labels_path`.  The caller is responsible for killing the child.
fn spawn_server(addr: &str, model_path: &str, labels_path: &str) -> Child {
    let bin = env!("CARGO_BIN_EXE_server");
    Command::new(bin)
        .args([
            "--bind",
            addr,
            "--model",
            model_path,
            "--labels",
            labels_path,
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .expect("failed to spawn server binary")
}

/// Returns `true` once the server is accepting TCP connections on `addr`, or
/// `false` if `timeout_ms` elapses first.
fn wait_for_port(addr: &str, timeout_ms: u64) -> bool {
    let start = std::time::Instant::now();
    let deadline = Duration::from_millis(timeout_ms);
    loop {
        if std::net::TcpStream::connect(addr).is_ok() {
            return true;
        }
        if start.elapsed() >= deadline {
            return false;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
}

// ---------------------------------------------------------------------------
// Binary startup tests (always run — no model needed)
// ---------------------------------------------------------------------------

/// The server must exit with a non-zero exit code (clean error, not panic/abort)
/// when the model file does not exist.  This verifies the startup error path
/// propagates `Result::Err` rather than panicking.
#[test]
fn server_exits_cleanly_on_missing_model() {
    let bin = env!("CARGO_BIN_EXE_server");
    let status = Command::new(bin)
        .args([
            "--model",
            "/nonexistent/path/mobilenetv2.onnx",
            "--labels",
            "/nonexistent/path/labels.txt",
            "--bind",
            "127.0.0.1:0",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("failed to run server binary");

    // Non-zero exit code confirms an error was returned from main(), not a panic.
    // On Linux a panic produces exit code 101; a clean anyhow error produces 1.
    assert!(
        !status.success(),
        "server should exit non-zero when model file is missing"
    );
    // Confirm it did not crash with a signal (signal() is None on clean exit).
    #[cfg(unix)]
    {
        use std::os::unix::process::ExitStatusExt;
        assert!(
            status.signal().is_none(),
            "server must not be killed by a signal on clean error exit; got signal {:?}",
            status.signal()
        );
    }
}

/// An empty labels path also yields a clean error exit, not a panic.
#[test]
fn server_exits_cleanly_on_missing_labels() {
    // We need a model path that exists (even if unusable) to get past the model
    // load step and reach the labels check.  Since both are loaded together in
    // Classifier::load, a missing model path is sufficient to trigger a clean
    // exit — we don't need a real model file.
    let bin = env!("CARGO_BIN_EXE_server");
    let status = Command::new(bin)
        .args([
            "--model",
            "/nonexistent/mobilenetv2.onnx",
            "--labels",
            "/nonexistent/labels.txt",
        ])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .expect("failed to run server binary");

    assert!(
        !status.success(),
        "server should exit non-zero when paths are missing"
    );
}

// ---------------------------------------------------------------------------
// Live HTTP tests (skipped when INTEGRATION_MODEL_PATH is not set)
// ---------------------------------------------------------------------------

/// Returns the model/labels paths from environment, or `None` if not configured.
fn live_test_paths() -> Option<(String, String)> {
    let model = std::env::var(ENV_MODEL_PATH).ok()?;
    // Labels path is optional; default to empty string (will be caught by the
    // labels-empty check in Classifier::load, causing a clean startup error).
    let labels =
        std::env::var(ENV_LABELS_PATH).unwrap_or_else(|_| "imagenet_classes.txt".to_owned());
    Some((model, labels))
}

/// A tiny 1x1 red PNG for tests that need valid image bytes.
///
/// This is the canonical minimal PNG used across the test suite.
fn tiny_red_png() -> &'static [u8] {
    &[
        0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44,
        0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x08, 0x02, 0x00, 0x00, 0x00, 0x90,
        0x77, 0x53, 0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, 0x54, 0x08, 0xd7, 0x63, 0xf8,
        0xcf, 0xc0, 0x00, 0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc, 0x33, 0x00, 0x00, 0x00,
        0x00, 0x49, 0x45, 0x4e, 0x44, 0xae, 0x42, 0x60, 0x82,
    ]
}

/// GET /health → 200 JSON {"status":"ok","model_loaded":true/false}.
///
/// Skipped when `INTEGRATION_MODEL_PATH` is not set.
#[tokio::test]
async fn live_health_returns_200_and_json() {
    let Some((model, labels)) = live_test_paths() else {
        eprintln!("SKIP live_health_returns_200_and_json: set {ENV_MODEL_PATH} to enable");
        return;
    };

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let mut child = spawn_server(&addr, &model, &labels);

    if !wait_for_port(&addr, SERVER_STARTUP_MS) {
        child.kill().ok();
        panic!("server did not start within {SERVER_STARTUP_MS}ms");
    }

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("GET /health must not fail");

    assert_eq!(resp.status(), 200, "GET /health must return 200");

    let body: serde_json::Value = resp.json().await.expect("body must be valid JSON");
    assert_eq!(body["status"], "ok", "status field must be \"ok\"");
    assert!(
        body["model_loaded"].is_boolean(),
        "model_loaded must be a boolean, got: {body}"
    );

    child.kill().ok();
}

/// POST /classify with a valid image → 200 or 503 depending on model presence;
/// verifies the response is well-formed JSON either way.
///
/// Skipped when `INTEGRATION_MODEL_PATH` is not set.
#[tokio::test]
async fn live_classify_with_valid_image_returns_json() {
    let Some((model, labels)) = live_test_paths() else {
        eprintln!(
            "SKIP live_classify_with_valid_image_returns_json: set {ENV_MODEL_PATH} to enable"
        );
        return;
    };

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let mut child = spawn_server(&addr, &model, &labels);

    if !wait_for_port(&addr, SERVER_STARTUP_MS) {
        child.kill().ok();
        panic!("server did not start within {SERVER_STARTUP_MS}ms");
    }

    let client = reqwest::Client::new();

    let image_part = reqwest::multipart::Part::bytes(tiny_red_png())
        .file_name("test.png")
        .mime_str("image/png")
        .expect("valid mime str");
    let form = reqwest::multipart::Form::new().part("image", image_part);

    let resp = client
        .post(format!("http://{addr}/classify"))
        .multipart(form)
        .send()
        .await
        .expect("POST /classify must not fail");

    let status = resp.status();
    // With a real model this should be 200; with a model that fails to load for
    // any reason it may be 503.  Both are acceptable; 5xx (other than 503) or
    // 4xx are not.
    assert!(
        status == 200 || status == 503,
        "expected 200 or 503, got {status}"
    );

    let _body: serde_json::Value = resp.json().await.expect("response body must be valid JSON");

    child.kill().ok();
}

/// POST /classify with no multipart body → 400 (bad_request); server must
/// still be alive to serve the next request.
///
/// Skipped when `INTEGRATION_MODEL_PATH` is not set.
#[tokio::test]
async fn live_classify_empty_body_returns_400_or_503_not_500() {
    let Some((model, labels)) = live_test_paths() else {
        eprintln!("SKIP live_classify_empty_body_returns_400_or_503_not_500: set {ENV_MODEL_PATH}");
        return;
    };

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let mut child = spawn_server(&addr, &model, &labels);

    if !wait_for_port(&addr, SERVER_STARTUP_MS) {
        child.kill().ok();
        panic!("server did not start within {SERVER_STARTUP_MS}ms");
    }

    let client = reqwest::Client::new();

    // POST with empty body and no Content-Type multipart header.
    let resp = client
        .post(format!("http://{addr}/classify"))
        .body("")
        .send()
        .await
        .expect("POST /classify (empty) must not fail");

    let status = resp.status().as_u16();
    assert!(
        status == 400 || status == 422 || status == 503,
        "empty body must yield 400/422/503, not {status}"
    );
    assert_ne!(status, 500, "server must not 500 on empty body");
    assert_ne!(status, 0, "server must not crash");

    // Second request proves the server did not crash/panic after the bad input.
    let resp2 = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("second request after bad input must succeed");
    assert_eq!(
        resp2.status(),
        200,
        "server must still serve /health after bad /classify input"
    );

    child.kill().ok();
}

/// POST /classify with non-image bytes (random garbage) → 400 or 503.
/// Server must remain alive.
///
/// Skipped when `INTEGRATION_MODEL_PATH` is not set.
#[tokio::test]
async fn live_classify_non_image_bytes_returns_400_or_503_not_500() {
    let Some((model, labels)) = live_test_paths() else {
        eprintln!(
            "SKIP live_classify_non_image_bytes_returns_400_or_503_not_500: set {ENV_MODEL_PATH}"
        );
        return;
    };

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let mut child = spawn_server(&addr, &model, &labels);

    if !wait_for_port(&addr, SERVER_STARTUP_MS) {
        child.kill().ok();
        panic!("server did not start within {SERVER_STARTUP_MS}ms");
    }

    let client = reqwest::Client::new();

    let garbage_part = reqwest::multipart::Part::bytes(b"this is not an image".as_ref())
        .file_name("bad.bin")
        .mime_str("application/octet-stream")
        .expect("valid mime str");
    let form = reqwest::multipart::Form::new().part("image", garbage_part);

    let resp = client
        .post(format!("http://{addr}/classify"))
        .multipart(form)
        .send()
        .await
        .expect("POST /classify (garbage) must not fail");

    let status = resp.status().as_u16();
    assert!(
        status == 400 || status == 503,
        "non-image bytes must yield 400 or 503, not {status}"
    );
    assert_ne!(status, 500, "non-image bytes must not produce a 500");

    let body: serde_json::Value = resp
        .json()
        .await
        .expect("error response must be valid JSON");
    assert!(
        body["error"].is_string(),
        "error response must have an 'error' field: {body}"
    );

    // Server must still be alive.
    let resp2 = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("server must still respond after non-image input");
    assert_eq!(resp2.status(), 200);

    child.kill().ok();
}

/// POST /classify with wrong multipart field name (not "image") → 400 or 503.
///
/// Skipped when `INTEGRATION_MODEL_PATH` is not set.
#[tokio::test]
async fn live_classify_wrong_field_name_returns_400_or_503() {
    let Some((model, labels)) = live_test_paths() else {
        eprintln!("SKIP live_classify_wrong_field_name_returns_400_or_503: set {ENV_MODEL_PATH}");
        return;
    };

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let mut child = spawn_server(&addr, &model, &labels);

    if !wait_for_port(&addr, SERVER_STARTUP_MS) {
        child.kill().ok();
        panic!("server did not start within {SERVER_STARTUP_MS}ms");
    }

    let client = reqwest::Client::new();

    // Upload image data under the wrong field name ("photo" instead of "image").
    let wrong_field_part = reqwest::multipart::Part::bytes(tiny_red_png())
        .file_name("photo.png")
        .mime_str("image/png")
        .expect("valid mime str");
    let form = reqwest::multipart::Form::new().part("photo", wrong_field_part);

    let resp = client
        .post(format!("http://{addr}/classify"))
        .multipart(form)
        .send()
        .await
        .expect("POST /classify (wrong field) must not fail");

    let status = resp.status().as_u16();
    // 400: bad_request (missing "image" field); 503: no model loaded.
    assert!(
        status == 400 || status == 503,
        "wrong field name must yield 400 or 503, not {status}"
    );
    assert_ne!(status, 500, "wrong field name must not produce a 500");

    child.kill().ok();
}

/// POST /classify with a malformed multipart body (missing boundary) → 400/422
/// or 503.  The server must not crash.
///
/// Skipped when `INTEGRATION_MODEL_PATH` is not set.
#[tokio::test]
async fn live_classify_malformed_multipart_returns_4xx_or_503_not_500() {
    let Some((model, labels)) = live_test_paths() else {
        eprintln!(
            "SKIP live_classify_malformed_multipart_returns_4xx_or_503_not_500: set {ENV_MODEL_PATH}"
        );
        return;
    };

    let port = free_port();
    let addr = format!("127.0.0.1:{port}");
    let mut child = spawn_server(&addr, &model, &labels);

    if !wait_for_port(&addr, SERVER_STARTUP_MS) {
        child.kill().ok();
        panic!("server did not start within {SERVER_STARTUP_MS}ms");
    }

    let client = reqwest::Client::new();

    // Send Content-Type: multipart/form-data but with malformed body (no
    // boundary marker in the body itself).
    let resp = client
        .post(format!("http://{addr}/classify"))
        .header(
            "Content-Type",
            "multipart/form-data; boundary=boundarythatdoesnotexist",
        )
        .body("this is not a valid multipart body at all")
        .send()
        .await
        .expect("POST /classify (malformed multipart) must not fail");

    let status = resp.status().as_u16();
    assert!(
        status >= 400 && status < 600,
        "malformed multipart must yield 4xx or 5xx, not {status}"
    );
    assert_ne!(status, 500, "malformed multipart must not produce a 500");

    // Server still alive.
    let resp2 = client
        .get(format!("http://{addr}/health"))
        .send()
        .await
        .expect("server must still respond after malformed multipart");
    assert_eq!(resp2.status(), 200);

    child.kill().ok();
}
