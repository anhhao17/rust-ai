//! Post-processing for YOLOv8 detection outputs.
//!
//! YOLOv8 ONNX output shape: `[1, 84, 8400]`
//! - Rows 0–3 : cx, cy, w, h (in model-input coordinates, 0–640)
//! - Rows 4–83: per-class confidence scores (no separate objectness)
//!
//! Pipeline:
//! 1. Iterate over the 8400 anchors.
//! 2. Filter out detections whose best class confidence < `conf_threshold`.
//! 3. Keep only detections whose best class is *person* (COCO class 0).
//! 4. Convert cx/cy/w/h → x1/y1/x2/y2 in original-image pixels.
//! 5. Apply greedy NMS to suppress duplicate boxes.
//!
//! # Note on code duplication
//! // TODO: counter/ and detect/ now share this YOLOv8 decode/NMS logic.
//! //       Extract into a shared `vision-core` lib crate in a follow-up PR.

use anyhow::{Result, anyhow};

use crate::preprocess::LetterboxParams;

/// COCO class index for "person".  Keeping this as a named constant makes the
/// filtering step self-documenting and easy to change (e.g. counting vehicles).
pub const COCO_PERSON_CLASS_ID: usize = 0;

/// Number of COCO classes in the standard YOLOv8n model.
const NUM_COCO_CLASSES: usize = 80;

/// Number of bounding-box coordinate values per anchor (cx, cy, w, h).
const BOX_COORD_COUNT: usize = 4;

/// Number of detection anchors from a 640-input YOLOv8 model.
const DETECTION_ANCHOR_COUNT: usize = 8400;

/// A bounding box for a single detected person in original-image coordinates.
#[derive(Debug, Clone, PartialEq)]
pub struct PersonDetection {
    /// Top-left x coordinate in original-image pixels.
    pub x1: f32,
    /// Top-left y coordinate in original-image pixels.
    pub y1: f32,
    /// Bottom-right x coordinate in original-image pixels.
    pub x2: f32,
    /// Bottom-right y coordinate in original-image pixels.
    pub y2: f32,
    /// Class confidence score in \[0, 1\].
    pub confidence: f32,
}

impl PersonDetection {
    /// Returns the centroid of this bounding box as `(cx, cy)`.
    pub fn centroid(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    /// Returns the intersection-over-union of this box and `other`.
    pub fn iou(&self, other: &PersonDetection) -> f32 {
        let inter_x1 = self.x1.max(other.x1);
        let inter_y1 = self.y1.max(other.y1);
        let inter_x2 = self.x2.min(other.x2);
        let inter_y2 = self.y2.min(other.y2);

        let inter_w = (inter_x2 - inter_x1).max(0.0);
        let inter_h = (inter_y2 - inter_y1).max(0.0);
        let inter_area = inter_w * inter_h;

        let area_self = (self.x2 - self.x1) * (self.y2 - self.y1);
        let area_other = (other.x2 - other.x1) * (other.y2 - other.y1);
        let union_area = area_self + area_other - inter_area;

        if union_area <= 0.0 {
            0.0
        } else {
            inter_area / union_area
        }
    }
}

/// Decodes the raw YOLOv8 output logits into a filtered, NMS-reduced list of
/// [`PersonDetection`]s in original-image coordinates.
///
/// `raw` must be the flat slice from the `[1, 84, 8400]` output tensor in
/// row-major order.  Only detections whose best class is *person* (class 0)
/// and whose confidence meets `conf_threshold` are returned.
///
/// # Errors
///
/// Returns `Err` if `raw` has an unexpected length — guards against loading a
/// different model variant without crashing.
pub fn decode_persons(
    raw: &[f32],
    params: &LetterboxParams,
    conf_threshold: f32,
    nms_iou_threshold: f32,
    orig_w: u32,
    orig_h: u32,
) -> Result<Vec<PersonDetection>> {
    let expected_len = (BOX_COORD_COUNT + NUM_COCO_CLASSES) * DETECTION_ANCHOR_COUNT;
    if raw.len() != expected_len {
        return Err(anyhow!(
            "unexpected YOLOv8 output size: got {}, expected {} \
             ({}×{} anchors — wrong imgsz or model variant?)",
            raw.len(),
            expected_len,
            BOX_COORD_COUNT + NUM_COCO_CLASSES,
            DETECTION_ANCHOR_COUNT,
        ));
    }

    let mut candidates: Vec<PersonDetection> = Vec::new();

    for anchor_idx in 0..DETECTION_ANCHOR_COUNT {
        // The model output is stored in [channel, anchor] order (row-major).
        let cx = raw[anchor_idx];
        let cy = raw[DETECTION_ANCHOR_COUNT + anchor_idx];
        let w = raw[2 * DETECTION_ANCHOR_COUNT + anchor_idx];
        let h = raw[3 * DETECTION_ANCHOR_COUNT + anchor_idx];

        // Only look at the "person" class score — we don't need to scan all 80.
        let person_confidence =
            raw[(BOX_COORD_COUNT + COCO_PERSON_CLASS_ID) * DETECTION_ANCHOR_COUNT + anchor_idx];

        if person_confidence < conf_threshold {
            continue;
        }

        let (x1, y1, x2, y2) = model_box_to_image_coords(cx, cy, w, h, params, orig_w, orig_h);

        candidates.push(PersonDetection {
            x1,
            y1,
            x2,
            y2,
            confidence: person_confidence,
        });
    }

    Ok(non_max_suppression(candidates, nms_iou_threshold))
}

/// Converts a center-format bounding box from model-input space (0–640) to
/// corner-format coordinates in the original image, accounting for letterbox
/// padding and scaling.
fn model_box_to_image_coords(
    cx: f32,
    cy: f32,
    w: f32,
    h: f32,
    params: &LetterboxParams,
    orig_w: u32,
    orig_h: u32,
) -> (f32, f32, f32, f32) {
    let x1 = ((cx - w / 2.0 - params.pad_x as f32) / params.scale).max(0.0);
    let y1 = ((cy - h / 2.0 - params.pad_y as f32) / params.scale).max(0.0);
    let x2 = ((cx + w / 2.0 - params.pad_x as f32) / params.scale).min(orig_w as f32);
    let y2 = ((cy + h / 2.0 - params.pad_y as f32) / params.scale).min(orig_h as f32);

    (x1, y1, x2, y2)
}

/// Applies greedy non-maximum suppression.
///
/// Detections are sorted by confidence (descending) and a box is suppressed
/// when its IoU with a higher-confidence box exceeds `iou_threshold`.
fn non_max_suppression(
    mut detections: Vec<PersonDetection>,
    iou_threshold: f32,
) -> Vec<PersonDetection> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<PersonDetection> = Vec::with_capacity(detections.len());

    for candidate in detections {
        let suppressed = kept
            .iter()
            .any(|existing| existing.iou(&candidate) > iou_threshold);

        if !suppressed {
            kept.push(candidate);
        }
    }

    kept
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preprocess::LetterboxParams;
    use approx::assert_abs_diff_eq;

    fn identity_params() -> LetterboxParams {
        LetterboxParams {
            scale: 1.0,
            pad_x: 0,
            pad_y: 0,
        }
    }

    fn make_detection(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32) -> PersonDetection {
        PersonDetection {
            x1,
            y1,
            x2,
            y2,
            confidence: conf,
        }
    }

    // --- IoU ---

    #[test]
    fn iou_identical_boxes_returns_one() {
        let det = make_detection(0.0, 0.0, 10.0, 10.0, 1.0);
        assert_abs_diff_eq!(det.iou(&det.clone()), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn iou_non_overlapping_boxes_returns_zero() {
        let a = make_detection(0.0, 0.0, 10.0, 10.0, 1.0);
        let b = make_detection(20.0, 20.0, 30.0, 30.0, 1.0);
        assert_abs_diff_eq!(a.iou(&b), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn iou_half_overlap_is_correct() {
        // a: [0,0]–[10,10], b: [5,0]–[15,10] → intersection=50, union=150
        let a = make_detection(0.0, 0.0, 10.0, 10.0, 1.0);
        let b = make_detection(5.0, 0.0, 15.0, 10.0, 1.0);
        assert_abs_diff_eq!(a.iou(&b), 50.0 / 150.0, epsilon = 1e-4);
    }

    // --- centroid ---

    #[test]
    fn centroid_returns_midpoint_of_bounding_box() {
        let det = make_detection(10.0, 20.0, 50.0, 80.0, 0.9);
        let (cx, cy) = det.centroid();
        assert_abs_diff_eq!(cx, 30.0, epsilon = 1e-4);
        assert_abs_diff_eq!(cy, 50.0, epsilon = 1e-4);
    }

    // --- NMS ---

    #[test]
    fn nms_keeps_highest_confidence_when_boxes_overlap_heavily() {
        let detections = vec![
            make_detection(0.0, 0.0, 10.0, 10.0, 0.9),
            make_detection(1.0, 1.0, 11.0, 11.0, 0.8), // nearly identical — should be suppressed
        ];
        let result = non_max_suppression(detections, 0.5);
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0].confidence, 0.9, epsilon = 1e-6);
    }

    #[test]
    fn nms_keeps_both_boxes_when_far_apart() {
        let detections = vec![
            make_detection(0.0, 0.0, 10.0, 10.0, 0.9),
            make_detection(50.0, 50.0, 60.0, 60.0, 0.8),
        ];
        let result = non_max_suppression(detections, 0.5);
        assert_eq!(result.len(), 2);
    }

    // --- decode_persons size validation ---

    #[test]
    fn wrong_length_raw_returns_err_not_panic() {
        let params = identity_params();
        let too_short =
            vec![0.0_f32; (BOX_COORD_COUNT + NUM_COCO_CLASSES) * DETECTION_ANCHOR_COUNT - 1];
        let result = decode_persons(&too_short, &params, 0.25, 0.45, 640, 640);
        assert!(result.is_err(), "expected Err for wrong-length input");
    }

    #[test]
    fn correct_length_zeros_returns_empty_detections() {
        let params = identity_params();
        let zeros = vec![0.0_f32; (BOX_COORD_COUNT + NUM_COCO_CLASSES) * DETECTION_ANCHOR_COUNT];
        let result = decode_persons(&zeros, &params, 0.25, 0.45, 640, 640);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    // --- coordinate inversion ---

    #[test]
    fn box_coords_invert_identity_letterbox_correctly() {
        let params = identity_params();
        // Center box at (100, 100) with 20×20 → corners at (90, 90)–(110, 110).
        let (x1, y1, x2, y2) =
            model_box_to_image_coords(100.0, 100.0, 20.0, 20.0, &params, 640, 640);
        assert_abs_diff_eq!(x1, 90.0, epsilon = 1e-4);
        assert_abs_diff_eq!(y1, 90.0, epsilon = 1e-4);
        assert_abs_diff_eq!(x2, 110.0, epsilon = 1e-4);
        assert_abs_diff_eq!(y2, 110.0, epsilon = 1e-4);
    }
}
