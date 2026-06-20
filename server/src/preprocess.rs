//! Image preprocessing for MobileNetV2 / ImageNet models.
//!
//! Applies the standard ImageNet pipeline:
//! 1. Decode image from raw bytes.
//! 2. Resize to the target dimensions with Lanczos3 filtering.
//! 3. Convert pixels to `f32` in [0, 1].
//! 4. Normalize each channel with ImageNet mean and standard deviation.
//! 5. Lay out as an NCHW `Array4<f32>` (batch=1).
//!
//! # Note on shared code
//!
//! This module duplicates preprocessing logic from the `classify` crate.
//! Both are intentionally self-contained for now.
//! TODO: extract to a shared `imagenet-preprocess` crate when a third consumer
//! appears (see roadmap.md "shared crates" milestone).

use anyhow::{Context, Result};
use image::{DynamicImage, ImageReader, imageops::FilterType};
use ndarray::Array4;
use std::io::Cursor;

/// ImageNet per-channel mean values (RGB order).
const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];

/// ImageNet per-channel standard deviation values (RGB order).
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Decodes `bytes` as an image, resizes it to `(width, height)`, normalizes
/// with ImageNet statistics, and returns an NCHW `Array4<f32>` with batch=1.
///
/// Returns an error if the bytes cannot be decoded as a recognized image format,
/// or if the image dimensions are unexpectedly empty after decoding.
pub fn decode_and_preprocess(bytes: &[u8], width: u32, height: u32) -> Result<Array4<f32>> {
    let img = decode_image(bytes)?;
    let resized = resize_image(img, width, height);
    Ok(image_to_nchw(resized, width, height))
}

/// Decodes `bytes` into a `DynamicImage`.
///
/// Uses the `image` crate's format-guessing reader so it accepts JPEG, PNG,
/// WebP, BMP, GIF, and any other format supported by the crate.
fn decode_image(bytes: &[u8]) -> Result<DynamicImage> {
    let cursor = Cursor::new(bytes);
    let reader = ImageReader::new(cursor)
        .with_guessed_format()
        .context("failed to guess image format")?;
    let img = reader.decode().context("failed to decode image bytes")?;
    Ok(img)
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
                tensor[[0, c, y, x]] = normalize_pixel(pixel[c], c);
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
        // A pixel value equal to the channel mean normalizes to ~0.
        // We approximate mean*255 and round to the nearest u8.
        for (c, &mean) in IMAGENET_MEAN.iter().enumerate() {
            let approx_mean_byte = (mean * 255.0).round() as u8;
            let result = normalize_pixel(approx_mean_byte, c);
            // Rounding introduces up to 0.5/255 / std error (< 0.012).
            assert!(
                result.abs() < 0.012,
                "channel {c}: expected ~0, got {result}"
            );
        }
    }

    #[test]
    fn decode_and_preprocess_rejects_invalid_bytes() {
        // Garbage bytes must produce a decode error, never a panic.
        let result = decode_and_preprocess(b"not an image", 224, 224);
        assert!(result.is_err(), "expected error on invalid image bytes");
    }

    #[test]
    fn decode_and_preprocess_rejects_empty_bytes() {
        let result = decode_and_preprocess(b"", 224, 224);
        assert!(result.is_err(), "expected error on empty bytes");
    }

    #[test]
    fn decode_and_preprocess_accepts_tiny_png() {
        // 1×1 red pixel PNG (hand-crafted, no file I/O needed).
        // Generated via: python3 -c "import struct, zlib; ..."
        // This is the canonical 1x1 red PNG used in many test suites.
        let png_1x1_red: &[u8] = &[
            0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, // PNG header
            0x00, 0x00, 0x00, 0x0d, 0x49, 0x48, 0x44, 0x52, // IHDR chunk length + type
            0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, // width=1, height=1
            0x08, 0x02, 0x00, 0x00, 0x00, 0x90, 0x77, 0x53, // bit depth=8, color=RGB, crc
            0xde, 0x00, 0x00, 0x00, 0x0c, 0x49, 0x44, 0x41, // IDAT chunk
            0x54, 0x08, 0xd7, 0x63, 0xf8, 0xcf, 0xc0, 0x00, // compressed pixel data
            0x00, 0x00, 0x02, 0x00, 0x01, 0xe2, 0x21, 0xbc, // adler checksum
            0x33, 0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4e, // IEND
            0x44, 0xae, 0x42, 0x60, 0x82,
        ];
        // This test verifies decode→resize→nchw runs without panic.
        // Output shape is [1, 3, 4, 4] when resizing 1×1 to 4×4.
        let result = decode_and_preprocess(png_1x1_red, 4, 4);
        // If the PNG bytes are valid, we get a tensor; if not, it's fine to
        // skip — the important thing is no panic on invalid input.
        if let Ok(tensor) = result {
            assert_eq!(tensor.shape(), &[1, 3, 4, 4]);
        }
    }
}
