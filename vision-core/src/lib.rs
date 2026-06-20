//! `vision-core` — shared utilities for the embedded vision pipeline.
//!
//! This crate houses the cross-cutting logic that every inference crate in the
//! workspace needs, so that behavior is defined and tested in one place rather
//! than duplicated across `classify`, `detect`, `server`, and `counter`.
//!
//! # Modules
//!
//! - [`session`] — ONNX Runtime session construction with CUDA + CPU fallback.

pub mod session;
