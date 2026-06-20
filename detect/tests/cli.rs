//! CLI black-box tests for the `detect` binary.
//!
//! All tests run the compiled binary via `assert_cmd` and require no ONNX
//! model file, GPU, or camera. They exercise argument parsing and graceful
//! error handling only.

use assert_cmd::Command;
use predicates::prelude::*;

// ── helpers ──────────────────────────────────────────────────────────────────

/// Returns an `assert_cmd::Command` pointed at the `detect` binary.
fn detect() -> Command {
    Command::cargo_bin("detect").expect("detect binary not found — run `cargo build` first")
}

// ── help flags ───────────────────────────────────────────────────────────────

/// `detect --help` exits 0 and mentions both subcommands.
#[test]
fn help_flag_exits_zero_and_lists_subcommands() {
    detect()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicate::str::contains("image"))
        .stdout(predicate::str::contains("camera"));
}

/// `detect image --help` exits 0 and mentions the `--input` argument.
#[test]
fn image_subcommand_help_exits_zero() {
    detect()
        .args(["--model", "dummy.onnx", "image", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("input"));
}

/// `detect camera --help` exits 0 and mentions `--device`.
#[test]
fn camera_subcommand_help_exits_zero() {
    detect()
        .args(["--model", "dummy.onnx", "camera", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("device"));
}

// ── missing / invalid arguments ──────────────────────────────────────────────

/// Invoking `detect` with no arguments whatsoever fails with a clap usage error.
#[test]
fn no_args_fails_with_usage_error() {
    detect()
        .assert()
        .failure()
        .stderr(predicate::str::contains("Usage").or(predicate::str::contains("error")));
}

/// `detect --model foo.onnx` with no subcommand fails (clap requires a subcommand).
#[test]
fn missing_subcommand_fails() {
    detect()
        .args(["--model", "foo.onnx"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error").or(predicate::str::contains("required")));
}

/// An unrecognised subcommand name fails with a clap error on stderr.
#[test]
fn unknown_subcommand_fails_with_clap_error() {
    detect()
        .args(["--model", "foo.onnx", "video"])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("error")
                .or(predicate::str::contains("unrecognized"))
                .or(predicate::str::contains("invalid")),
        );
}

/// `detect image` with `--input` but no `--model` flag fails with a clap error.
#[test]
fn image_subcommand_missing_model_fails() {
    detect()
        .args(["image", "--input", "photo.jpg"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error").or(predicate::str::contains("required")));
}

/// `detect --model foo.onnx image` with no `--input` fails with a clap error.
#[test]
fn image_subcommand_missing_input_fails() {
    detect()
        .args(["--model", "foo.onnx", "image"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("error").or(predicate::str::contains("required")));
}

// ── graceful error on missing files ──────────────────────────────────────────

/// Passing a nonexistent model path causes a non-zero exit WITHOUT a panic.
///
/// `build_session` calls `commit_from_file`, which should propagate an error
/// through `anyhow` and print a human-readable message rather than aborting.
#[test]
fn nonexistent_model_path_fails_gracefully_no_panic() {
    detect()
        .args([
            "--model",
            "/nonexistent/path/to/model.onnx",
            "image",
            "--input",
            "photo.jpg",
        ])
        .assert()
        .failure()
        // Must NOT contain a Rust panic message.
        .stderr(predicate::str::contains("panicked at").not());
}

/// Passing a nonexistent `--input` image path causes a non-zero exit without a panic.
///
/// The model path also doesn't exist, so the binary will fail at model-load
/// time, but either way it must exit non-zero and cleanly.
#[test]
fn nonexistent_input_image_fails_gracefully_no_panic() {
    detect()
        .args([
            "--model",
            "/nonexistent/model.onnx",
            "image",
            "--input",
            "/nonexistent/image.jpg",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("panicked at").not());
}

/// A path that exists but is not a valid ONNX file fails gracefully (no panic).
///
/// Uses this test file itself as the "model" — guaranteed to exist, guaranteed
/// to be unreadable as ONNX.
#[test]
fn invalid_onnx_file_fails_gracefully_no_panic() {
    let not_an_onnx = std::env!("CARGO_MANIFEST_DIR").to_string() + "/tests/cli.rs";
    detect()
        .args([
            "--model",
            &not_an_onnx,
            "image",
            "--input",
            "/nonexistent/image.jpg",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("panicked at").not());
}
