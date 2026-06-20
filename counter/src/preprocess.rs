//! Image pre-processing for YOLOv8 person detection.
//!
//! Applies letterbox resizing to fit any input into a 640×640 square
//! without distortion, scales pixel values to \[0, 1\], and lays the result
//! out as an NCHW `Array4<f32>` (batch = 1).
//!
//! # Note on code duplication
//! // TODO: counter/ and detect/ now share this letterbox/preprocess logic.
//! //       Extract it into a shared `vision-core` lib crate in a follow-up PR
//! //       rather than inlining the same code in both binaries.

use image::{DynamicImage, GenericImageView, RgbImage, imageops::FilterType};
use ndarray::Array4;

/// Model input resolution used by all YOLOv8 variants in this project.
pub const MODEL_INPUT_SIZE: u32 = 640;

/// Ultralytics letterbox grey-fill value — matches the training pipeline.
const LETTERBOX_FILL_VALUE: u8 = 114;

/// Metadata returned alongside the pre-processed tensor so post-processing
/// can map model-space bounding boxes back to original-image coordinates.
#[derive(Debug, Clone, Copy)]
pub struct LetterboxParams {
    /// Uniform scale factor applied to fit the image inside
    /// [`MODEL_INPUT_SIZE`] × [`MODEL_INPUT_SIZE`].
    pub scale: f32,
    /// Horizontal padding (pixels) added on each side of the scaled image.
    pub pad_x: u32,
    /// Vertical padding (pixels) added on each side of the scaled image.
    pub pad_y: u32,
}

/// Letterbox-resizes `img` to fit inside [`MODEL_INPUT_SIZE`] × [`MODEL_INPUT_SIZE`],
/// normalises pixels to \[0, 1\], and returns the NCHW tensor plus the
/// [`LetterboxParams`] needed to invert the transform in post-processing.
pub fn letterbox_and_normalise(img: &DynamicImage) -> (Array4<f32>, LetterboxParams) {
    let (orig_w, orig_h) = img.dimensions();

    // Choose the uniform scale that fits both dimensions inside the model square.
    let scale =
        (MODEL_INPUT_SIZE as f32 / orig_w as f32).min(MODEL_INPUT_SIZE as f32 / orig_h as f32);

    let scaled_w = (orig_w as f32 * scale).round() as u32;
    let scaled_h = (orig_h as f32 * scale).round() as u32;

    // Integer-divide the remainder so we pad equal amounts on each side.
    let pad_x = (MODEL_INPUT_SIZE - scaled_w) / 2;
    let pad_y = (MODEL_INPUT_SIZE - scaled_h) / 2;

    let params = LetterboxParams {
        scale,
        pad_x,
        pad_y,
    };

    let resized = img
        .resize_exact(scaled_w, scaled_h, FilterType::Lanczos3)
        .to_rgb8();

    // Build the padded canvas with the standard Ultralytics fill colour.
    let fill = image::Rgb([
        LETTERBOX_FILL_VALUE,
        LETTERBOX_FILL_VALUE,
        LETTERBOX_FILL_VALUE,
    ]);
    let mut canvas = RgbImage::from_pixel(MODEL_INPUT_SIZE, MODEL_INPUT_SIZE, fill);
    image::imageops::overlay(&mut canvas, &resized, pad_x as i64, pad_y as i64);

    let tensor = rgb_image_to_nchw(&canvas);

    (tensor, params)
}

/// Converts an `RgbImage` into a normalised NCHW `Array4<f32>` (batch = 1).
///
/// Each channel value is divided by 255 to place it in \[0, 1\].
fn rgb_image_to_nchw(img: &RgbImage) -> Array4<f32> {
    let width = img.width() as usize;
    let height = img.height() as usize;
    let mut tensor = Array4::<f32>::zeros([1, 3, height, width]);

    for row in 0..height {
        for col in 0..width {
            let pixel = img.get_pixel(col as u32, row as u32);
            for channel in 0..3 {
                tensor[[0, channel, row, col]] = pixel[channel] as f32 / 255.0;
            }
        }
    }

    tensor
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    fn solid_rgb_image(width: u32, height: u32, r: u8, g: u8, b: u8) -> DynamicImage {
        let mut img = RgbImage::new(width, height);
        for px in img.pixels_mut() {
            *px = image::Rgb([r, g, b]);
        }
        DynamicImage::ImageRgb8(img)
    }

    #[test]
    fn output_tensor_shape_is_always_model_input_size() {
        let img = solid_rgb_image(1280, 720, 0, 0, 0);
        let (tensor, _params) = letterbox_and_normalise(&img);
        assert_eq!(
            tensor.shape(),
            &[1, 3, MODEL_INPUT_SIZE as usize, MODEL_INPUT_SIZE as usize]
        );
    }

    #[test]
    fn landscape_image_produces_correct_scale_and_vertical_padding() {
        // 1280×720 → scale=0.5 fits width; scaled height=360; pad_y=(640-360)/2=140.
        let img = solid_rgb_image(1280, 720, 0, 0, 0);
        let (_tensor, params) = letterbox_and_normalise(&img);

        assert!(
            (params.scale - 0.5).abs() < 0.01,
            "expected scale≈0.5, got {}",
            params.scale
        );
        assert_eq!(
            params.pad_y, 140,
            "expected pad_y=140, got {}",
            params.pad_y
        );
        assert_eq!(params.pad_x, 0, "expected pad_x=0, got {}", params.pad_x);
    }

    #[test]
    fn square_image_has_no_padding_and_scale_one() {
        let img = solid_rgb_image(640, 640, 0, 0, 0);
        let (_tensor, params) = letterbox_and_normalise(&img);
        assert_eq!(params.pad_x, 0);
        assert_eq!(params.pad_y, 0);
        assert!((params.scale - 1.0).abs() < 0.01);
    }

    #[test]
    fn pixel_values_are_normalised_to_0_1() {
        // A fully red square image — red channel should land near 1.0,
        // green near 0.0 at any non-padded pixel.
        let img = solid_rgb_image(640, 640, 255, 0, 0);
        let (tensor, _) = letterbox_and_normalise(&img);

        let red_channel_value = tensor[[0, 0, 0, 0]];
        let green_channel_value = tensor[[0, 1, 0, 0]];

        assert!(
            (red_channel_value - 1.0_f32).abs() < 0.01,
            "red={red_channel_value}"
        );
        assert!(
            green_channel_value.abs() < 0.01,
            "green={green_channel_value}"
        );
    }
}
