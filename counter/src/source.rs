//! Video source detection and classification for the `--source` argument.
//!
//! The `--source` value is interpreted as follows (evaluated in order):
//!
//! | Pattern | Source kind |
//! |---------|-------------|
//! | Existing directory on disk | [`SourceKind::FrameDir`] — sorted JPEG/PNG files |
//! | Integer or `camera:N` string | [`SourceKind::Camera`] — V4L2 device N (feature-gated) |
//! | String starting with `rtsp://`, `http://`, `https://`, `hls://` | [`SourceKind::NetworkStream`] — ffmpeg network decode |
//! | Existing file with a video extension | [`SourceKind::VideoFile`] — ffmpeg file decode |
//! | Existing image file (jpg/jpeg/png) | [`SourceKind::FrameDir`] — single-image list |
//!
//! Any other value produces a clear `Err` rather than a panic.

use std::path::Path;

use anyhow::{Result, anyhow};

/// Video extensions that ffmpeg can decode.  This list covers the most common
/// containers; ffmpeg will also accept less common extensions — the filter is
/// conservative on purpose to avoid treating e.g. `.onnx` files as videos.
const VIDEO_EXTENSIONS: &[&str] = &[
    "mp4", "avi", "mkv", "mov", "webm", "flv", "ts", "m4v", "wmv", "mpg", "mpeg", "3gp",
];

/// URL scheme prefixes that indicate a network / streaming source.
const NETWORK_PREFIXES: &[&str] = &["rtsp://", "rtsps://", "http://", "https://", "hls://"];

/// The kind of video source the counter will read from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SourceKind {
    /// A directory of sorted JPEG/PNG image files, or a single image file.
    /// Finite; the caller decides whether to loop.
    FrameDir(std::path::PathBuf),

    /// A local video file decoded via ffmpeg (mp4, avi, mkv, …).
    /// Finite; the caller decides whether to loop.
    VideoFile(std::path::PathBuf),

    /// An RTSP, HTTP, HTTPS, or HLS stream decoded via ffmpeg.
    /// Potentially infinite; looping is disabled for live sources.
    NetworkStream(String),

    /// A V4L2 camera device (Linux/Jetson only; requires the `camera` feature).
    /// Infinite live source; looping is not applicable.
    Camera(u32),
}

impl SourceKind {
    /// Returns `true` for sources that have a natural end (file or frame dir).
    ///
    /// Live sources (camera, network stream) return `false`.
    pub fn is_finite(&self) -> bool {
        matches!(self, SourceKind::FrameDir(_) | SourceKind::VideoFile(_))
    }
}

/// Parses the `--source` argument string and returns the appropriate
/// [`SourceKind`].
///
/// Detection rules (applied in order):
/// 1. If the value is a path to an existing **directory** → `FrameDir`.
/// 2. If the value is a bare integer or `camera:<N>` → `Camera(N)`.
/// 3. If the value starts with a known network prefix → `NetworkStream`.
/// 4. If the value is a path to an existing file with a **video extension**
///    → `VideoFile`.
/// 5. If the value is a path to an existing file with an **image extension**
///    → `FrameDir` (single-image list).
/// 6. Otherwise → `Err`.
///
/// # Errors
///
/// Returns `Err` with a descriptive message when the value does not match any
/// recognised pattern.
pub fn detect_source_kind(source: &str) -> Result<SourceKind> {
    let path = Path::new(source);

    // Rule 1: existing directory.
    if path.is_dir() {
        return Ok(SourceKind::FrameDir(path.to_path_buf()));
    }

    // Rule 2: bare integer or `camera:<N>` → camera device index.
    if let Some(index) = parse_camera_index(source) {
        return Ok(SourceKind::Camera(index));
    }

    // Rule 3: network URL prefix.
    let lower = source.to_ascii_lowercase();
    if NETWORK_PREFIXES.iter().any(|pfx| lower.starts_with(pfx)) {
        return Ok(SourceKind::NetworkStream(source.to_owned()));
    }

    // Rule 4 & 5: existing file — distinguish video vs image by extension.
    if path.is_file() {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();

        if VIDEO_EXTENSIONS.contains(&ext.as_str()) {
            return Ok(SourceKind::VideoFile(path.to_path_buf()));
        }

        // Treat single image files as a one-element frame directory.
        if matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
            return Ok(SourceKind::FrameDir(path.to_path_buf()));
        }

        return Err(anyhow!(
            "file '{}' has unrecognised extension '{}'; \
             supported video extensions: {}, image extensions: jpg/jpeg/png",
            source,
            ext,
            VIDEO_EXTENSIONS.join(", ")
        ));
    }

    // Nothing matched.
    Err(anyhow!(
        "cannot determine source type for '{}': \
         not an existing directory, file, camera index, or recognised URL \
         (valid prefixes: {})",
        source,
        NETWORK_PREFIXES.join(", ")
    ))
}

/// Attempts to parse `value` as a V4L2 camera device index.
///
/// Accepts:
/// - A bare non-negative integer string (e.g. `"0"`, `"2"`).
/// - A `camera:<N>` prefix (e.g. `"camera:0"`).
///
/// Returns `None` for anything else.
fn parse_camera_index(value: &str) -> Option<u32> {
    if let Some(rest) = value.strip_prefix("camera:") {
        return rest.parse::<u32>().ok();
    }
    // Only treat bare integers as camera indices when they look like device IDs
    // (small non-negative numbers).  Checking is_file/is_dir first (done in the
    // caller) prevents collisions with numeric filenames that exist on disk.
    value.parse::<u32>().ok()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    // --- parse_camera_index ---

    #[test]
    fn bare_zero_is_camera_index_zero() {
        assert_eq!(parse_camera_index("0"), Some(0));
    }

    #[test]
    fn bare_integer_is_camera_index() {
        assert_eq!(parse_camera_index("3"), Some(3));
    }

    #[test]
    fn camera_prefix_parses_index() {
        assert_eq!(parse_camera_index("camera:2"), Some(2));
    }

    #[test]
    fn non_integer_returns_none() {
        assert_eq!(parse_camera_index("rtsp://host/stream"), None);
        assert_eq!(parse_camera_index("video.mp4"), None);
        assert_eq!(parse_camera_index("camera:abc"), None);
    }

    // --- detect_source_kind with a real directory ---

    #[test]
    fn existing_directory_is_frame_dir() {
        let dir = tempdir().unwrap();
        let result = detect_source_kind(dir.path().to_str().unwrap()).unwrap();
        assert!(matches!(result, SourceKind::FrameDir(_)));
    }

    // --- detect_source_kind with real files ---

    #[test]
    fn mp4_file_is_video_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("clip.mp4");
        fs::write(&path, b"fake").unwrap();
        let result = detect_source_kind(path.to_str().unwrap()).unwrap();
        assert!(matches!(result, SourceKind::VideoFile(_)));
    }

    #[test]
    fn avi_file_is_video_file() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("walk.avi");
        fs::write(&path, b"fake").unwrap();
        let result = detect_source_kind(path.to_str().unwrap()).unwrap();
        assert!(matches!(result, SourceKind::VideoFile(_)));
    }

    #[test]
    fn jpg_file_is_frame_dir() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("frame.jpg");
        fs::write(&path, b"fake").unwrap();
        let result = detect_source_kind(path.to_str().unwrap()).unwrap();
        assert!(matches!(result, SourceKind::FrameDir(_)));
    }

    // --- detect_source_kind with network URLs ---

    #[test]
    fn rtsp_url_is_network_stream() {
        let result = detect_source_kind("rtsp://192.168.1.1:554/stream").unwrap();
        assert!(matches!(result, SourceKind::NetworkStream(_)));
    }

    #[test]
    fn http_url_is_network_stream() {
        let result = detect_source_kind("http://example.com/live.m3u8").unwrap();
        assert!(matches!(result, SourceKind::NetworkStream(_)));
    }

    #[test]
    fn https_url_is_network_stream() {
        let result = detect_source_kind("https://example.com/stream.m3u8").unwrap();
        assert!(matches!(result, SourceKind::NetworkStream(_)));
    }

    #[test]
    fn hls_url_is_network_stream() {
        let result = detect_source_kind("hls://example.com/index.m3u8").unwrap();
        assert!(matches!(result, SourceKind::NetworkStream(_)));
    }

    // --- detect_source_kind error cases ---

    #[test]
    fn nonexistent_path_returns_err() {
        let result = detect_source_kind("/tmp/counter_nonexistent_xyz_12345");
        assert!(result.is_err(), "nonexistent path should fail");
    }

    #[test]
    fn unknown_extension_returns_err() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("model.onnx");
        fs::write(&path, b"fake").unwrap();
        let result = detect_source_kind(path.to_str().unwrap());
        assert!(
            result.is_err(),
            ".onnx should not be recognised as a source"
        );
    }

    // --- SourceKind::is_finite ---

    #[test]
    fn frame_dir_is_finite() {
        let kind = SourceKind::FrameDir(std::path::PathBuf::from("/tmp"));
        assert!(kind.is_finite());
    }

    #[test]
    fn video_file_is_finite() {
        let kind = SourceKind::VideoFile(std::path::PathBuf::from("clip.mp4"));
        assert!(kind.is_finite());
    }

    #[test]
    fn network_stream_is_not_finite() {
        let kind = SourceKind::NetworkStream("rtsp://host/s".into());
        assert!(!kind.is_finite());
    }

    #[test]
    fn camera_is_not_finite() {
        let kind = SourceKind::Camera(0);
        assert!(!kind.is_finite());
    }
}
