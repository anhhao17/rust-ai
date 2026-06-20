//! Frame annotation for the people counter.
//!
//! Draws the following overlays onto each processed frame:
//! - a bounding box per tracked person (cyan fill on outer+inner rect for thickness)
//! - a track-ID label above each box (e.g. "#7")
//! - the configured counting line (red)
//! - a live counts overlay in the top-left corner (entered / left / net)
//!
//! Text is rasterised using an embedded DejaVuSans TTF font so the binary has
//! no runtime font-path dependency on the Jetson or in CI.
//!
//! # Note on future refactor
//! // TODO: vision-core PR-D — fold this draw module into a shared
//! //       `vision-core::draw` module once counter/ and detect/ both need
//! //       per-frame annotation with text labels.

use ab_glyph::{FontRef, PxScale};
use anyhow::{Context, Result};
use image::{DynamicImage, Rgba, RgbaImage};
use imageproc::{
    drawing::{draw_hollow_rect_mut, draw_line_segment_mut, draw_text_mut},
    rect::Rect,
};

use crate::line_counter::{CountTally, CountingLine};
use crate::tracker::Track;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Colour used to draw track bounding boxes (cyan).
const TRACK_BOX_COLOUR: Rgba<u8> = Rgba([0, 255, 255, 255]);

/// Colour used to draw the counting line (red).
const COUNTING_LINE_COLOUR: Rgba<u8> = Rgba([255, 50, 50, 255]);

/// Colour used for text labels on track boxes (white with full opacity).
const LABEL_COLOUR: Rgba<u8> = Rgba([255, 255, 255, 255]);

/// Colour used for the counts overlay text (yellow).
const OVERLAY_COLOUR: Rgba<u8> = Rgba([255, 230, 0, 255]);

/// Font size (pixels) used for track-ID labels.
const LABEL_FONT_SIZE: f32 = 18.0;

/// Font size (pixels) used for the counts overlay text.
const OVERLAY_FONT_SIZE: f32 = 22.0;

/// Vertical margin between consecutive overlay text lines (pixels).
const OVERLAY_LINE_SPACING: i32 = 26;

/// Top-left x offset for the overlay text block.
const OVERLAY_X: i32 = 8;

/// Top-left y offset for the overlay text block.
const OVERLAY_Y: i32 = 8;

/// The embedded DejaVuSans font, included at compile time.
/// Embedding avoids a runtime font-path lookup on the Jetson or in CI.
const FONT_BYTES: &[u8] = include_bytes!("../assets/DejaVuSans.ttf");

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Loads the embedded DejaVuSans font.
///
/// # Errors
///
/// Returns `Err` if the embedded font bytes are malformed (should never happen
/// in practice — the bytes are checked at compile time by `include_bytes!`).
pub fn load_font() -> Result<FontRef<'static>> {
    FontRef::try_from_slice(FONT_BYTES).context("failed to parse embedded DejaVuSans font")
}

/// Draws all overlays onto `img` and returns the annotated [`DynamicImage`].
///
/// Overlays drawn (in order, so later draws appear on top):
/// 1. Bounding box + track-ID label for each active track.
/// 2. The counting line.
/// 3. Counts overlay (entered / left / net) in the top-left corner.
pub fn annotate_frame(
    img: &DynamicImage,
    tracks: &[Track],
    counting_line: CountingLine,
    tally: CountTally,
    font: &FontRef<'static>,
) -> DynamicImage {
    let mut rgba: RgbaImage = img.to_rgba8();

    for track in tracks {
        draw_track_box(&mut rgba, track, font);
    }

    draw_counting_line(&mut rgba, counting_line);
    draw_counts_overlay(&mut rgba, tally, font);

    DynamicImage::ImageRgba8(rgba)
}

/// Encodes a [`DynamicImage`] as JPEG bytes at the given quality level.
///
/// Quality ranges from 1 (worst) to 100 (best).  75 is a good balance for
/// streaming: visually acceptable while keeping frame sizes moderate.
///
/// # Errors
///
/// Returns `Err` if the JPEG encoder fails (out-of-memory or I/O error on the
/// in-memory buffer — extremely unlikely).
pub fn encode_jpeg(img: &DynamicImage, quality: u8) -> Result<Vec<u8>> {
    let mut buf: Vec<u8> = Vec::new();
    let mut encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    encoder
        .encode_image(img)
        .context("failed to JPEG-encode annotated frame")?;
    Ok(buf)
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

/// Draws a cyan bounding box and a track-ID label for one track.
fn draw_track_box(canvas: &mut RgbaImage, track: &Track, font: &FontRef<'static>) {
    let x = track.bbox.x1.max(0.0) as i32;
    let y = track.bbox.y1.max(0.0) as i32;
    let w = (track.bbox.x2 - track.bbox.x1).max(1.0) as u32;
    let h = (track.bbox.y2 - track.bbox.y1).max(1.0) as u32;

    // Two concentric rectangles give a ~2-pixel border thickness.
    draw_hollow_rect_mut(canvas, Rect::at(x, y).of_size(w, h), TRACK_BOX_COLOUR);
    if w > 2 && h > 2 {
        draw_hollow_rect_mut(
            canvas,
            Rect::at(x + 1, y + 1).of_size(w - 2, h - 2),
            TRACK_BOX_COLOUR,
        );
    }

    // Label: "#<id>" positioned just above the box (clamped to image top).
    let label = format!("#{}", track.id);
    let label_y = (y - LABEL_FONT_SIZE as i32 - 2).max(0);
    draw_text_mut(
        canvas,
        LABEL_COLOUR,
        x,
        label_y,
        PxScale::from(LABEL_FONT_SIZE),
        font,
        &label,
    );
}

/// Draws the counting line across the frame.
fn draw_counting_line(canvas: &mut RgbaImage, line: CountingLine) {
    // draw_line_segment_mut takes (f32, f32) tuples.
    draw_line_segment_mut(canvas, line.start, line.end, COUNTING_LINE_COLOUR);
}

/// Burns the live count (entered / left / net) into the top-left corner.
fn draw_counts_overlay(canvas: &mut RgbaImage, tally: CountTally, font: &FontRef<'static>) {
    let lines = [
        format!("In:  {}", tally.entered),
        format!("Out: {}", tally.left),
        format!("Net: {}", tally.net()),
    ];

    for (i, line) in lines.iter().enumerate() {
        let y = OVERLAY_Y + i as i32 * OVERLAY_LINE_SPACING;
        draw_text_mut(
            canvas,
            OVERLAY_COLOUR,
            OVERLAY_X,
            y,
            PxScale::from(OVERLAY_FONT_SIZE),
            font,
            line,
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocess::PersonDetection;
    use crate::tracker::Track;
    use image::{DynamicImage, RgbaImage};

    fn blank_frame(width: u32, height: u32) -> DynamicImage {
        DynamicImage::ImageRgba8(RgbaImage::new(width, height))
    }

    fn make_track(id: u64, x1: f32, y1: f32, x2: f32, y2: f32) -> Track {
        Track {
            id,
            bbox: PersonDetection {
                x1,
                y1,
                x2,
                y2,
                confidence: 0.9,
            },
            missed_frames: 0,
            frame_count: 1,
        }
    }

    #[test]
    fn load_font_succeeds() {
        let result = load_font();
        assert!(result.is_ok(), "embedded font failed to load: {:?}", result);
    }

    #[test]
    fn annotate_frame_returns_same_dimensions() {
        let font = load_font().unwrap();
        let frame = blank_frame(640, 480);
        let tracks = vec![make_track(1, 10.0, 20.0, 100.0, 200.0)];
        let line = CountingLine {
            start: (0.0, 240.0),
            end: (640.0, 240.0),
        };
        let tally = CountTally {
            entered: 3,
            left: 1,
        };

        let annotated = annotate_frame(&frame, &tracks, line, tally, &font);

        assert_eq!(annotated.width(), 640);
        assert_eq!(annotated.height(), 480);
    }

    #[test]
    fn annotate_frame_with_empty_tracks_does_not_panic() {
        let font = load_font().unwrap();
        let frame = blank_frame(320, 240);
        let line = CountingLine {
            start: (0.0, 120.0),
            end: (320.0, 120.0),
        };
        let tally = CountTally::default();

        let annotated = annotate_frame(&frame, &[], line, tally, &font);
        assert_eq!(annotated.width(), 320);
    }

    #[test]
    fn encode_jpeg_produces_non_empty_bytes() {
        let frame = blank_frame(64, 64);
        let result = encode_jpeg(&frame, 75);
        assert!(result.is_ok(), "JPEG encoding failed: {:?}", result);
        let bytes = result.unwrap();
        // JPEG magic bytes: 0xFF 0xD8
        assert!(bytes.len() > 2, "JPEG output too short");
        assert_eq!(bytes[0], 0xFF);
        assert_eq!(bytes[1], 0xD8);
    }

    #[test]
    fn encode_jpeg_quality_affects_output_size() {
        let font = load_font().unwrap();
        let frame = blank_frame(320, 240);
        // Put something on the frame so quality actually matters.
        let tracks = vec![
            make_track(1, 10.0, 10.0, 100.0, 100.0),
            make_track(2, 200.0, 100.0, 300.0, 200.0),
        ];
        let line = CountingLine {
            start: (0.0, 120.0),
            end: (320.0, 120.0),
        };
        let annotated = annotate_frame(
            &frame,
            &tracks,
            line,
            CountTally {
                entered: 2,
                left: 0,
            },
            &font,
        );

        let high_quality = encode_jpeg(&annotated, 95).unwrap();
        let low_quality = encode_jpeg(&annotated, 10).unwrap();

        assert!(
            high_quality.len() > low_quality.len(),
            "high quality ({} bytes) should be larger than low quality ({} bytes)",
            high_quality.len(),
            low_quality.len(),
        );
    }
}
