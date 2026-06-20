//! Image preprocessing for YOLOv8 / COCO models.
//!
//! Applies letterbox resizing to fit the image into a 640×640 square
//! without distortion, then scales pixel values to [0, 1] and lays out
//! the result as an NCHW `Array4<f32>` (batch=1).

use image::{DynamicImage, GenericImageView, RgbImage, imageops::FilterType};
use ndarray::Array4;

/// Target input dimension for YOLOv8.
pub const MODEL_SIZE: u32 = 640;

/// Metadata returned alongside the preprocessed tensor so that
/// postprocessing can map detections back to the original image.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxParams {
    /// Uniform scale factor applied to fit the image inside MODEL_SIZE×MODEL_SIZE.
    pub scale: f32,
    /// Horizontal padding (pixels) added on each side of the scaled image.
    pub pad_x: u32,
    /// Vertical padding (pixels) added on each side of the scaled image.
    pub pad_y: u32,
}

/// Letterbox-resizes `img` to fit inside [`MODEL_SIZE`]×[`MODEL_SIZE`],
/// normalises pixels to [0, 1], and returns the tensor plus the scaling
/// parameters needed to invert the transform in postprocessing.
pub fn letterbox_and_normalise(img: &DynamicImage) -> (Array4<f32>, LetterboxParams) {
    let (orig_w, orig_h) = img.dimensions();

    let scale = (MODEL_SIZE as f32 / orig_w as f32).min(MODEL_SIZE as f32 / orig_h as f32);
    let scaled_w = (orig_w as f32 * scale).round() as u32;
    let scaled_h = (orig_h as f32 * scale).round() as u32;

    let pad_x = (MODEL_SIZE - scaled_w) / 2;
    let pad_y = (MODEL_SIZE - scaled_h) / 2;

    let params = LetterboxParams {
        scale,
        pad_x,
        pad_y,
    };

    // Resize the source image.
    let resized = img
        .resize_exact(scaled_w, scaled_h, FilterType::Lanczos3)
        .to_rgb8();

    // Build the padded canvas (grey fill = 114 per ultralytics convention).
    let mut canvas = RgbImage::from_pixel(MODEL_SIZE, MODEL_SIZE, image::Rgb([114u8, 114, 114]));
    image::imageops::overlay(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    // Convert to NCHW f32 tensor, scaling to [0, 1].
    let tensor = rgb_image_to_nchw(&canvas);

    (tensor, params)
}

/// Converts an `RgbImage` into a normalised NCHW `Array4<f32>` (batch=1).
fn rgb_image_to_nchw(img: &RgbImage) -> Array4<f32> {
    let w = img.width() as usize;
    let h = img.height() as usize;
    let mut tensor = Array4::<f32>::zeros([1, 3, h, w]);

    for y in 0..h {
        for x in 0..w {
            let pixel = img.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                tensor[[0, c, y, x]] = pixel[c] as f32 / 255.0;
            }
        }
    }

    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    fn solid_rgb_image(w: u32, h: u32, r: u8, g: u8, b: u8) -> DynamicImage {
        let mut img = RgbImage::new(w, h);
        for px in img.pixels_mut() {
            *px = image::Rgb([r, g, b]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn output_tensor_shape_is_always_model_size() {
        let img = solid_rgb_image(1280, 720, 0, 0, 0);
        let (tensor, _params) = letterbox_and_normalise(&img);
        assert_eq!(
            tensor.shape(),
            &[1, 3, MODEL_SIZE as usize, MODEL_SIZE as usize]
        );
    }

    #[test]
    fn scale_and_pad_values_are_consistent_for_landscape_input() {
        let img = solid_rgb_image(1280, 720, 0, 0, 0);
        let (_tensor, params) = letterbox_and_normalise(&img);

        // Width is the limiting dimension for 1280×720 → 640×360.
        assert!((params.scale - 0.5).abs() < 0.01, "scale={}", params.scale);
        // Padded height: (640 - 360) / 2 = 140
        assert_eq!(params.pad_y, 140);
        // No horizontal padding since width fills MODEL_SIZE exactly.
        assert_eq!(params.pad_x, 0);
    }

    #[test]
    fn pixel_values_are_normalised_to_0_1() {
        let img = solid_rgb_image(640, 640, 255, 0, 128);
        let (tensor, _params) = letterbox_and_normalise(&img);

        // Red channel should be 1.0 at a non-padded pixel.
        let r = tensor[[0, 0, 0, 0]];
        assert!((r - 1.0_f32).abs() < 0.01, "red={r}");
        // Green channel should be 0.0.
        let g = tensor[[0, 1, 0, 0]];
        assert!(g.abs() < 0.01, "green={g}");
    }

    #[test]
    fn square_image_has_no_padding() {
        let img = solid_rgb_image(640, 640, 0, 0, 0);
        let (_tensor, params) = letterbox_and_normalise(&img);
        assert_eq!(params.pad_x, 0);
        assert_eq!(params.pad_y, 0);
        assert!((params.scale - 1.0).abs() < 0.01);
    }
}
