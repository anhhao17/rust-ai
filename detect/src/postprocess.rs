//! Post-processing for YOLOv8 detection outputs.
//!
//! YOLOv8 ONNX output shape: `[1, 84, 8400]`
//! - Rows 0–3: cx, cy, w, h (in model-input coordinates, 0–640)
//! - Rows 4–83: class confidence scores (no separate objectness)
//!
//! Pipeline:
//! 1. Transpose to `[8400, 84]` for ergonomic iteration.
//! 2. Filter detections whose max class confidence ≥ `conf_threshold`.
//! 3. Convert cx/cy/w/h → x1/y1/x2/y2 in original image coordinates.
//! 4. Apply class-aware NMS (non-maximum suppression).

use crate::preprocess::LetterboxParams;

/// Number of COCO classes in YOLOv8.
const NUM_CLASSES: usize = 80;

/// Number of bounding-box coordinate values per detection.
const BOX_COORDS: usize = 4;

/// Expected number of raw detections from a 640-input YOLOv8 model.
const NUM_ANCHORS: usize = 8400;

/// A single detected object with its bounding box and class.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Top-left x coordinate in original image pixels.
    pub x1: f32,
    /// Top-left y coordinate in original image pixels.
    pub y1: f32,
    /// Bottom-right x coordinate in original image pixels.
    pub x2: f32,
    /// Bottom-right y coordinate in original image pixels.
    pub y2: f32,
    /// Class confidence score (0–1).
    pub confidence: f32,
    /// COCO class index (0–79).
    pub class_id: usize,
}

impl Detection {
    /// Returns the intersection-over-union of this detection and `other`.
    pub fn iou(&self, other: &Detection) -> f32 {
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
/// [`Detection`]s in original-image coordinates.
///
/// `raw` must be the flat slice from the `[1, 84, 8400]` output tensor
/// (row-major order: all 8400 values for cx, then cy, w, h, class0…class79).
pub fn decode_yolov8_output(
    raw: &[f32],
    params: &LetterboxParams,
    conf_threshold: f32,
    nms_iou_threshold: f32,
    orig_w: u32,
    orig_h: u32,
) -> Vec<Detection> {
    assert_eq!(
        raw.len(),
        (BOX_COORDS + NUM_CLASSES) * NUM_ANCHORS,
        "unexpected output tensor size"
    );

    // Transpose [84, 8400] → candidates filtered by confidence.
    // Row layout: cx(8400), cy(8400), w(8400), h(8400), class0(8400), …, class79(8400).
    let mut candidates: Vec<Detection> = Vec::new();

    for anchor in 0..NUM_ANCHORS {
        // cx, cy, w, h are in model-input space (0–640).
        let cx = raw[anchor];
        let cy = raw[NUM_ANCHORS + anchor];
        let w = raw[2 * NUM_ANCHORS + anchor];
        let h = raw[3 * NUM_ANCHORS + anchor];

        // Find the best class.
        let (class_id, confidence) = (0..NUM_CLASSES)
            .map(|c| (c, raw[(BOX_COORDS + c) * NUM_ANCHORS + anchor]))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or((0, 0.0));

        if confidence < conf_threshold {
            continue;
        }

        // Map from model-input coords to original-image coords.
        let (x1, y1, x2, y2) = model_box_to_image_coords(cx, cy, w, h, params, orig_w, orig_h);

        candidates.push(Detection {
            x1,
            y1,
            x2,
            y2,
            confidence,
            class_id,
        });
    }

    non_max_suppression(candidates, nms_iou_threshold)
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
    // Remove padding offset, then invert the uniform scale.
    let x1 = ((cx - w / 2.0 - params.pad_x as f32) / params.scale).max(0.0);
    let y1 = ((cy - h / 2.0 - params.pad_y as f32) / params.scale).max(0.0);
    let x2 = ((cx + w / 2.0 - params.pad_x as f32) / params.scale).min(orig_w as f32);
    let y2 = ((cy + h / 2.0 - params.pad_y as f32) / params.scale).min(orig_h as f32);

    (x1, y1, x2, y2)
}

/// Applies greedy class-aware non-maximum suppression.
///
/// Detections are sorted by confidence (descending) and a box is suppressed
/// when its IoU with a higher-confidence box of the *same class* exceeds
/// `iou_threshold`.
fn non_max_suppression(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut kept: Vec<Detection> = Vec::with_capacity(detections.len());

    for candidate in detections {
        let suppressed = kept.iter().any(|existing| {
            existing.class_id == candidate.class_id && existing.iou(&candidate) > iou_threshold
        });
        if !suppressed {
            kept.push(candidate);
        }
    }

    kept
}

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

    fn make_detection(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32, class_id: usize) -> Detection {
        Detection {
            x1,
            y1,
            x2,
            y2,
            confidence: conf,
            class_id,
        }
    }

    // --- IoU tests ---

    #[test]
    fn iou_identical_boxes_returns_one() {
        let d = make_detection(0.0, 0.0, 10.0, 10.0, 1.0, 0);
        assert_abs_diff_eq!(d.iou(&d.clone()), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn iou_non_overlapping_boxes_returns_zero() {
        let a = make_detection(0.0, 0.0, 10.0, 10.0, 1.0, 0);
        let b = make_detection(20.0, 20.0, 30.0, 30.0, 1.0, 0);
        assert_abs_diff_eq!(a.iou(&b), 0.0, epsilon = 1e-6);
    }

    #[test]
    fn iou_half_overlap_is_correct() {
        // a: [0,0]–[10,10], b: [5,0]–[15,10] → intersection=50, union=150
        let a = make_detection(0.0, 0.0, 10.0, 10.0, 1.0, 0);
        let b = make_detection(5.0, 0.0, 15.0, 10.0, 1.0, 0);
        assert_abs_diff_eq!(a.iou(&b), 50.0 / 150.0, epsilon = 1e-4);
    }

    // --- NMS tests ---

    #[test]
    fn nms_keeps_highest_confidence_of_overlapping_same_class() {
        let detections = vec![
            make_detection(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            make_detection(1.0, 1.0, 11.0, 11.0, 0.8, 0), // high overlap with first
        ];
        let result = non_max_suppression(detections, 0.5);
        assert_eq!(result.len(), 1);
        assert_abs_diff_eq!(result[0].confidence, 0.9, epsilon = 1e-6);
    }

    #[test]
    fn nms_keeps_both_boxes_when_different_classes() {
        let detections = vec![
            make_detection(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            make_detection(1.0, 1.0, 11.0, 11.0, 0.8, 1), // same area, different class
        ];
        let result = non_max_suppression(detections, 0.5);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn nms_keeps_non_overlapping_same_class_boxes() {
        let detections = vec![
            make_detection(0.0, 0.0, 10.0, 10.0, 0.9, 0),
            make_detection(50.0, 50.0, 60.0, 60.0, 0.8, 0), // far away
        ];
        let result = non_max_suppression(detections, 0.5);
        assert_eq!(result.len(), 2);
    }

    // --- model_box_to_image_coords tests ---

    #[test]
    fn box_coords_invert_letterbox_correctly_for_identity() {
        let params = identity_params();
        // Center box at (100,100) with 20×20 → corners at (90,90)–(110,110)
        let (x1, y1, x2, y2) =
            model_box_to_image_coords(100.0, 100.0, 20.0, 20.0, &params, 640, 640);
        assert_abs_diff_eq!(x1, 90.0, epsilon = 1e-4);
        assert_abs_diff_eq!(y1, 90.0, epsilon = 1e-4);
        assert_abs_diff_eq!(x2, 110.0, epsilon = 1e-4);
        assert_abs_diff_eq!(y2, 110.0, epsilon = 1e-4);
    }

    #[test]
    fn box_coords_invert_letterbox_with_scale_and_padding() {
        // Simulate 1280×720 input letterboxed to 640×640:
        //   scale=0.5, pad_x=0, pad_y=140
        let params = LetterboxParams {
            scale: 0.5,
            pad_x: 0,
            pad_y: 140,
        };
        // A model-space box at cx=320, cy=320, w=100, h=100 should map to:
        //   x1 = (320-50-0)/0.5 = 540, x2 = (320+50-0)/0.5 = 740
        //   y1 = (320-50-140)/0.5 = 260, y2 = (320+50-140)/0.5 = 460
        let (x1, y1, x2, y2) =
            model_box_to_image_coords(320.0, 320.0, 100.0, 100.0, &params, 1280, 720);
        assert_abs_diff_eq!(x1, 540.0, epsilon = 1.0);
        assert_abs_diff_eq!(y1, 260.0, epsilon = 1.0);
        assert_abs_diff_eq!(x2, 740.0, epsilon = 1.0);
        assert_abs_diff_eq!(y2, 460.0, epsilon = 1.0);
    }
}
