//! Bounding-box rendering on images.
//!
//! Draws coloured hollow rectangles over detected objects.  Labels are printed
//! to stdout by the caller rather than embedded in the image, which avoids a
//! font-file dependency at build time.

use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;

use crate::postprocess::Detection;

/// Colour palette for up to 20 distinct classes; cycles for larger class IDs.
const COLOURS: [[u8; 4]; 20] = [
    [255, 0, 0, 255],
    [0, 255, 0, 255],
    [0, 0, 255, 255],
    [255, 255, 0, 255],
    [0, 255, 255, 255],
    [255, 0, 255, 255],
    [255, 128, 0, 255],
    [128, 0, 255, 255],
    [0, 128, 255, 255],
    [255, 0, 128, 255],
    [0, 255, 128, 255],
    [128, 255, 0, 255],
    [200, 100, 50, 255],
    [50, 200, 100, 255],
    [100, 50, 200, 255],
    [180, 180, 0, 255],
    [0, 180, 180, 255],
    [180, 0, 180, 255],
    [90, 90, 90, 255],
    [220, 220, 220, 255],
];

/// Draws a coloured bounding box for each detection and returns the annotated
/// image.  The border is drawn with a 2-pixel thickness by drawing two
/// concentric rectangles.
pub fn draw_detections(img: &DynamicImage, detections: &[Detection]) -> DynamicImage {
    let mut rgba: RgbaImage = img.to_rgba8();

    for det in detections {
        let [r, g, b, a] = COLOURS[det.class_id % COLOURS.len()];
        let colour = Rgba([r, g, b, a]);

        let x = det.x1.max(0.0) as i32;
        let y = det.y1.max(0.0) as i32;
        let w = (det.x2 - det.x1).max(1.0) as u32;
        let h = (det.y2 - det.y1).max(1.0) as u32;

        // Outer rectangle.
        draw_hollow_rect_mut(&mut rgba, Rect::at(x, y).of_size(w, h), colour);
        // Inner rectangle for a thicker border (shrink by 1px on each side).
        if w > 2 && h > 2 {
            draw_hollow_rect_mut(
                &mut rgba,
                Rect::at(x + 1, y + 1).of_size(w - 2, h - 2),
                colour,
            );
        }
    }

    DynamicImage::ImageRgba8(rgba)
}
