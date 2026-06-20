//! Parameterized logic tests for the `detect` crate's pure pre- and
//! post-processing functions.
//!
//! Because `detect` is a binary crate (no `lib.rs`), these integration tests
//! include the relevant source modules directly via `#[path]` attributes.
//! No ONNX model, GPU, or camera is required — all tests are pure computation.

// rstest expands parameterized test functions that clippy counts as having "too
// many arguments" (one per #[case] parameter).  This is inherent to the macro
// and not a real code-quality issue; silence the lint for this file.
#![allow(clippy::too_many_arguments)]

use rstest::rstest;

// Pull in the pure modules under test.  The `coco_labels` module is also
// included because `postprocess` does not depend on it at the type level, but
// `preprocess` and `postprocess` are self-contained.
#[path = "../src/preprocess.rs"]
mod preprocess;

#[path = "../src/postprocess.rs"]
mod postprocess;

use postprocess::{Detection, decode_yolov8_output};
use preprocess::{LetterboxParams, letterbox_and_normalise};

// ── helpers ───────────────────────────────────────────────────────────────────

fn make_det(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32, class_id: usize) -> Detection {
    Detection {
        x1,
        y1,
        x2,
        y2,
        confidence: conf,
        class_id,
    }
}

/// Build a raw tensor of the correct size filled with zeros (no detections).
fn zero_raw() -> Vec<f32> {
    vec![0.0_f32; 84 * 8400]
}

/// Build a raw tensor with exactly one synthetic detection injected at `anchor`.
fn raw_with_one_detection(
    anchor: usize,
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    class_id: usize,
    conf: f32,
) -> Vec<f32> {
    let n = 8400_usize;
    let mut raw = vec![0.0_f32; 84 * n];
    raw[anchor] = cx;
    raw[n + anchor] = cy;
    raw[2 * n + anchor] = w;
    raw[3 * n + anchor] = h;
    raw[(4 + class_id) * n + anchor] = conf;
    raw
}

/// Absolute difference within half a pixel.
fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < 0.5
}

// ── decode_yolov8_output: wrong-length input returns Err ─────────────────────

/// Any tensor length other than 84×8400 must return Err, not panic.
#[rstest]
#[case(0)]
#[case(1)]
#[case(84 * 8400 - 1)]
#[case(84 * 8400 + 1)]
#[case(84 * 8400 * 2)]
fn wrong_length_returns_err(#[case] len: usize) {
    let params = LetterboxParams {
        scale: 1.0,
        pad_x: 0,
        pad_y: 0,
    };
    let raw = vec![0.0_f32; len];
    let result = decode_yolov8_output(&raw, &params, 0.25, 0.45, 640, 640);
    assert!(result.is_err(), "expected Err for len={len}, got Ok");
}

// ── decode_yolov8_output: correct length → Ok ────────────────────────────────

#[test]
fn correct_length_all_zeros_returns_ok_empty() {
    let params = LetterboxParams {
        scale: 1.0,
        pad_x: 0,
        pad_y: 0,
    };
    let result = decode_yolov8_output(&zero_raw(), &params, 0.25, 0.45, 640, 640);
    assert!(result.is_ok());
    assert!(result.unwrap().is_empty());
}

/// A single strong detection above threshold survives filtering.
#[test]
fn one_high_confidence_detection_survives_threshold() {
    let params = LetterboxParams {
        scale: 1.0,
        pad_x: 0,
        pad_y: 0,
    };
    let raw = raw_with_one_detection(0, 320.0, 320.0, 100.0, 100.0, 0, 0.9);
    let result = decode_yolov8_output(&raw, &params, 0.25, 0.45, 640, 640).unwrap();
    assert_eq!(result.len(), 1);
    assert!((result[0].confidence - 0.9).abs() < 1e-5);
    assert_eq!(result[0].class_id, 0);
}

/// A detection below the confidence threshold is filtered out.
#[test]
fn low_confidence_detection_is_filtered() {
    let params = LetterboxParams {
        scale: 1.0,
        pad_x: 0,
        pad_y: 0,
    };
    let raw = raw_with_one_detection(0, 320.0, 320.0, 100.0, 100.0, 0, 0.10);
    let result = decode_yolov8_output(&raw, &params, 0.25, 0.45, 640, 640).unwrap();
    assert!(result.is_empty(), "expected no detections below threshold");
}

// ── IoU: parameterized cases ──────────────────────────────────────────────────

/// IoU between two identical boxes must be 1.0.
#[rstest]
#[case(0.0, 0.0, 10.0, 10.0)]
#[case(100.0, 50.0, 200.0, 150.0)]
#[case(0.0, 0.0, 1.0, 1.0)]
fn iou_identical_boxes_is_one(#[case] x1: f32, #[case] y1: f32, #[case] x2: f32, #[case] y2: f32) {
    let d = make_det(x1, y1, x2, y2, 1.0, 0);
    let iou = d.iou(&d.clone());
    assert!((iou - 1.0).abs() < 1e-5, "IoU={iou}, expected 1.0");
}

/// IoU between fully disjoint boxes must be 0.0.
#[allow(clippy::too_many_arguments)]
#[rstest]
#[case(0.0, 0.0, 10.0, 10.0, 20.0, 20.0, 30.0, 30.0)]
#[case(0.0, 0.0, 5.0, 5.0, 6.0, 6.0, 11.0, 11.0)]
#[case(0.0, 0.0, 10.0, 10.0, 100.0, 0.0, 110.0, 10.0)]
fn iou_disjoint_boxes_is_zero(
    #[case] ax1: f32,
    #[case] ay1: f32,
    #[case] ax2: f32,
    #[case] ay2: f32,
    #[case] bx1: f32,
    #[case] by1: f32,
    #[case] bx2: f32,
    #[case] by2: f32,
) {
    let a = make_det(ax1, ay1, ax2, ay2, 1.0, 0);
    let b = make_det(bx1, by1, bx2, by2, 1.0, 0);
    assert!(a.iou(&b).abs() < 1e-5, "IoU={}, expected 0.0", a.iou(&b));
}

/// 50% overlap: intersection=50, union=150 → IoU≈0.333.
#[test]
fn iou_half_overlap_equals_one_third() {
    let a = make_det(0.0, 0.0, 10.0, 10.0, 1.0, 0);
    let b = make_det(5.0, 0.0, 15.0, 10.0, 1.0, 0);
    let iou = a.iou(&b);
    assert!((iou - 1.0 / 3.0).abs() < 1e-4, "IoU={iou}");
}

/// Zero-area (degenerate) boxes must not cause a panic.
#[allow(clippy::too_many_arguments)]
#[rstest]
#[case(5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 10.0, 10.0)] // point vs box
#[case(5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0)] // point vs point
fn iou_zero_area_does_not_panic(
    #[case] ax1: f32,
    #[case] ay1: f32,
    #[case] ax2: f32,
    #[case] ay2: f32,
    #[case] bx1: f32,
    #[case] by1: f32,
    #[case] bx2: f32,
    #[case] by2: f32,
) {
    let a = make_det(ax1, ay1, ax2, ay2, 1.0, 0);
    let b = make_det(bx1, by1, bx2, by2, 1.0, 0);
    let _ = a.iou(&b); // must not panic
}

// ── letterbox preprocessing: output shape and padding ────────────────────────

/// Output tensor shape is always [1, 3, 640, 640] regardless of input size.
#[rstest]
#[case(640, 640)]
#[case(1280, 720)]
#[case(320, 240)]
#[case(100, 100)]
#[case(1920, 1080)]
#[case(480, 640)] // portrait
fn letterbox_output_shape_is_always_model_size(#[case] w: u32, #[case] h: u32) {
    use image::{DynamicImage, RgbImage};
    let img = DynamicImage::ImageRgb8(RgbImage::new(w, h));
    let (tensor, _) = letterbox_and_normalise(&img);
    assert_eq!(
        tensor.shape(),
        &[1, 3, 640, 640],
        "shape mismatch for {w}×{h}"
    );
}

/// 1280×720 landscape: scale=0.5, pad_x=0, pad_y=140.
#[test]
fn letterbox_1280x720_params_are_correct() {
    use image::{DynamicImage, RgbImage};
    let img = DynamicImage::ImageRgb8(RgbImage::new(1280, 720));
    let (_, params) = letterbox_and_normalise(&img);
    assert!((params.scale - 0.5).abs() < 0.01, "scale={}", params.scale);
    assert_eq!(params.pad_x, 0);
    assert_eq!(params.pad_y, 140);
}

/// 480×640 portrait: scale=1.0, pad_x=80, pad_y=0.
///
/// scale = min(640/480, 640/640) = min(1.333, 1.0) = 1.0;
/// scaled_w=480 → pad_x=(640−480)/2=80; scaled_h=640 → pad_y=0.
#[test]
fn letterbox_480x640_portrait_params_are_correct() {
    use image::{DynamicImage, RgbImage};
    let img = DynamicImage::ImageRgb8(RgbImage::new(480, 640));
    let (_, params) = letterbox_and_normalise(&img);
    assert!((params.scale - 1.0).abs() < 0.01, "scale={}", params.scale);
    assert_eq!(params.pad_x, 80);
    assert_eq!(params.pad_y, 0);
}

/// Square inputs have no padding; scale = 640 / side.
#[rstest]
#[case(640, 1.0)]
#[case(320, 2.0)]
#[case(1280, 0.5)]
fn letterbox_square_has_no_padding(#[case] side: u32, #[case] expected_scale: f32) {
    use image::{DynamicImage, RgbImage};
    let img = DynamicImage::ImageRgb8(RgbImage::new(side, side));
    let (_, params) = letterbox_and_normalise(&img);
    assert!(
        (params.scale - expected_scale).abs() < 0.01,
        "scale={}",
        params.scale
    );
    assert_eq!(params.pad_x, 0);
    assert_eq!(params.pad_y, 0);
}

// ── coordinate inversion round-trip ──────────────────────────────────────────

/// Injecting a known model-space box into a tensor and decoding it should
/// yield corners within ±0.5 px of the manual inversion formula.
#[allow(clippy::too_many_arguments)]
#[rstest]
// (orig_w, orig_h, cx, cy, bw, bh, scale, pad_x, pad_y)
#[case(640, 640, 320.0, 320.0, 100.0, 80.0, 1.0, 0, 0)]
#[case(1280, 720, 320.0, 320.0, 100.0, 80.0, 0.5, 0, 140)]
#[case(480, 640, 320.0, 320.0, 100.0, 80.0, 1.0, 80, 0)]
fn coordinate_round_trip(
    #[case] orig_w: u32,
    #[case] orig_h: u32,
    #[case] cx: f32,
    #[case] cy: f32,
    #[case] bw: f32,
    #[case] bh: f32,
    #[case] scale: f32,
    #[case] pad_x: u32,
    #[case] pad_y: u32,
) {
    let params = LetterboxParams {
        scale,
        pad_x,
        pad_y,
    };
    let raw = raw_with_one_detection(0, cx, cy, bw, bh, 0, 0.9);
    let dets = decode_yolov8_output(&raw, &params, 0.5, 0.45, orig_w, orig_h).unwrap();
    assert_eq!(dets.len(), 1, "expected exactly one detection");

    let det = &dets[0];
    let ex1 = ((cx - bw / 2.0 - pad_x as f32) / scale).max(0.0);
    let ey1 = ((cy - bh / 2.0 - pad_y as f32) / scale).max(0.0);
    let ex2 = ((cx + bw / 2.0 - pad_x as f32) / scale).min(orig_w as f32);
    let ey2 = ((cy + bh / 2.0 - pad_y as f32) / scale).min(orig_h as f32);

    assert!(approx_eq(det.x1, ex1), "x1={} expected≈{}", det.x1, ex1);
    assert!(approx_eq(det.y1, ey1), "y1={} expected≈{}", det.y1, ey1);
    assert!(approx_eq(det.x2, ex2), "x2={} expected≈{}", det.x2, ex2);
    assert!(approx_eq(det.y2, ey2), "y2={} expected≈{}", det.y2, ey2);
}

// ── NMS edge cases ────────────────────────────────────────────────────────────

/// Two heavily-overlapping same-class boxes: only the higher-confidence one survives.
#[test]
fn nms_overlapping_same_class_suppresses_lower_confidence() {
    let params = LetterboxParams {
        scale: 1.0,
        pad_x: 0,
        pad_y: 0,
    };
    let n = 8400_usize;
    // Anchor 0: class 0, conf 0.9; anchor 1: class 0, conf 0.8 — nearly same box.
    let mut raw = raw_with_one_detection(0, 320.0, 320.0, 100.0, 100.0, 0, 0.9);
    let raw2 = raw_with_one_detection(1, 322.0, 322.0, 100.0, 100.0, 0, 0.8);
    raw[1] = raw2[1];
    raw[n + 1] = raw2[n + 1];
    raw[2 * n + 1] = raw2[2 * n + 1];
    raw[3 * n + 1] = raw2[3 * n + 1];
    raw[4 * n + 1] = raw2[4 * n + 1];

    let result = decode_yolov8_output(&raw, &params, 0.5, 0.45, 640, 640).unwrap();
    assert_eq!(
        result.len(),
        1,
        "expected NMS to suppress the lower-confidence duplicate"
    );
    assert!(
        (result[0].confidence - 0.9).abs() < 1e-5,
        "survivor should be the 0.9-conf box"
    );
}

/// Two distant same-class boxes must both survive NMS.
#[test]
fn nms_distant_same_class_both_survive() {
    let params = LetterboxParams {
        scale: 1.0,
        pad_x: 0,
        pad_y: 0,
    };
    let n = 8400_usize;
    let mut raw = raw_with_one_detection(0, 50.0, 50.0, 40.0, 40.0, 0, 0.9);
    let raw2 = raw_with_one_detection(1, 580.0, 580.0, 40.0, 40.0, 0, 0.8);
    raw[1] = raw2[1];
    raw[n + 1] = raw2[n + 1];
    raw[2 * n + 1] = raw2[2 * n + 1];
    raw[3 * n + 1] = raw2[3 * n + 1];
    raw[4 * n + 1] = raw2[4 * n + 1];

    let result = decode_yolov8_output(&raw, &params, 0.5, 0.45, 640, 640).unwrap();
    assert_eq!(
        result.len(),
        2,
        "distant same-class boxes must both survive NMS"
    );
}

/// Heavily-overlapping boxes of *different* classes must both survive
/// (class-aware NMS does not suppress across classes).
#[test]
fn nms_overlapping_different_classes_both_survive() {
    let params = LetterboxParams {
        scale: 1.0,
        pad_x: 0,
        pad_y: 0,
    };
    let n = 8400_usize;
    // anchor 0: class 0, conf 0.9; anchor 1: class 1, conf 0.85 — nearly same box.
    let mut raw = raw_with_one_detection(0, 320.0, 320.0, 100.0, 100.0, 0, 0.9);
    let raw2 = raw_with_one_detection(1, 322.0, 322.0, 100.0, 100.0, 1, 0.85);
    raw[1] = raw2[1];
    raw[n + 1] = raw2[n + 1];
    raw[2 * n + 1] = raw2[2 * n + 1];
    raw[3 * n + 1] = raw2[3 * n + 1];
    // class 1 confidence is at row (4+1)=5.
    raw[5 * n + 1] = raw2[5 * n + 1];

    let result = decode_yolov8_output(&raw, &params, 0.5, 0.45, 640, 640).unwrap();
    assert_eq!(
        result.len(),
        2,
        "different-class boxes must both survive NMS even when overlapping"
    );
}

/// Empty detections list passes NMS cleanly.
#[test]
fn nms_empty_input_returns_empty() {
    let params = LetterboxParams {
        scale: 1.0,
        pad_x: 0,
        pad_y: 0,
    };
    // Threshold of 1.1 guarantees nothing passes (confidences are ≤1.0).
    let result = decode_yolov8_output(&zero_raw(), &params, 1.1, 0.45, 640, 640).unwrap();
    assert!(result.is_empty());
}
