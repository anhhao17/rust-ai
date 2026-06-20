//! Image preprocessing for MobileNetV2 / ImageNet models.
//!
//! Applies the standard ImageNet pipeline:
//! 1. Decode image from disk.
//! 2. Resize to the target dimensions with Lanczos3 filtering.
//! 3. Convert pixels to `f32` in [0, 1].
//! 4. Normalize each channel with ImageNet mean and standard deviation.
//! 5. Lay out as an NCHW `Array4<f32>` (batch=1).

use anyhow::{Context, Result};
use image::{DynamicImage, imageops::FilterType};
use ndarray::Array4;
use std::path::PathBuf;

/// ImageNet per-channel mean values (RGB order).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet per-channel standard deviation values (RGB order).
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Loads an image from `path`, resizes it to `(width, height)`, normalizes it
/// with ImageNet statistics, and returns it as an NCHW `Array4<f32>` with a
/// batch dimension of 1.
pub fn load_and_preprocess(path: &PathBuf, width: u32, height: u32) -> Result<Array4<f32>> {
    let img =
        image::open(path).with_context(|| format!("failed to open image: {}", path.display()))?;

    let resized = resize_image(img, width, height);
    let tensor = image_to_nchw(resized, width, height);
    Ok(tensor)
}

/// Resizes `img` to `(width, height)` using Lanczos3, converting to RGB8.
fn resize_image(img: DynamicImage, width: u32, height: u32) -> DynamicImage {
    img.resize_exact(width, height, FilterType::Lanczos3)
}

/// Converts a `DynamicImage` into a normalized NCHW `Array4<f32>`.
///
/// Pixel values are scaled to [0, 1] then shifted by the ImageNet mean and
/// divided by the ImageNet standard deviation per channel.
fn image_to_nchw(img: DynamicImage, width: u32, height: u32) -> Array4<f32> {
    let rgb = img.to_rgb8();
    let (w, h) = (width as usize, height as usize);

    // Shape: [batch=1, channels=3, height, width]
    let mut tensor = Array4::<f32>::zeros([1, 3, h, w]);

    for y in 0..h {
        for x in 0..w {
            let pixel = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let normalized = normalize_pixel(pixel[c], c);
                tensor[[0, c, y, x]] = normalized;
            }
        }
    }

    tensor
}

/// Scales `raw_byte` to [0, 1] then applies ImageNet mean/std normalization
/// for the given channel index `c` (0=R, 1=G, 2=B).
#[inline]
pub fn normalize_pixel(raw_byte: u8, c: usize) -> f32 {
    let scaled = raw_byte as f32 / 255.0;
    (scaled - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn normalize_pixel_zero_maps_to_negative_mean_over_std() {
        // pixel value 0 → scaled = 0.0 → (0 - mean) / std
        for (c, &mean) in IMAGENET_MEAN.iter().enumerate() {
            let expected = -mean / IMAGENET_STD[c];
            assert_abs_diff_eq!(normalize_pixel(0, c), expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn normalize_pixel_max_maps_to_one_minus_mean_over_std() {
        // pixel value 255 → scaled = 1.0 → (1 - mean) / std
        for (c, &mean) in IMAGENET_MEAN.iter().enumerate() {
            let expected = (1.0 - mean) / IMAGENET_STD[c];
            assert_abs_diff_eq!(normalize_pixel(255, c), expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn normalize_pixel_mean_value_maps_to_zero() {
        // A pixel value that equals the channel mean should normalize to ~0.
        // We approximate mean*255 and round to the nearest u8.
        for (c, &mean) in IMAGENET_MEAN.iter().enumerate() {
            let approx_mean_byte = (mean * 255.0).round() as u8;
            let result = normalize_pixel(approx_mean_byte, c);
            // Rounding introduces up to 0.5/255 / std error, which is < 0.012.
            assert!(
                result.abs() < 0.012,
                "channel {c}: expected ~0, got {result}"
            );
        }
    }
}
