//! ffmpeg-based video frame reader.
//!
//! Spawns an `ffmpeg` subprocess that decodes any source (local file, RTSP,
//! HTTP/HLS, …) and writes decoded frames to its stdout as a stream of JPEG
//! images (`-f image2pipe -vcodec mjpeg`).  A background thread reads the
//! stdout pipe and splits the byte stream on JPEG SOI markers (`\xFF\xD8`),
//! sending complete [`image::DynamicImage`] frames through a channel.
//!
//! ## Why `-f image2pipe -vcodec mjpeg`?
//!
//! This format avoids needing to probe the video dimensions beforehand
//! (required for rawvideo) and avoids per-frame file I/O (required for
//! image2pipe with PNG).  JPEG is fast to decode on CPU, the in-process
//! `image` crate can decode it directly, and the SOI/EOI byte markers make
//! frame splitting unambiguous.
//!
//! ## Error handling
//!
//! All errors surface as `Result` values — the channel closes on the first
//! unrecoverable error and the caller's `recv()` loop terminates naturally.
//! No `unwrap` or `panic` in non-test code.

use std::{
    io::{BufReader, Read},
    process::{Child, ChildStdout, Command, Stdio},
    sync::mpsc,
    thread,
};

use anyhow::{Context, Result, anyhow};
use image::DynamicImage;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum JPEG frame size accepted from the ffmpeg pipe (16 MiB).
///
/// Frames larger than this indicate a pipeline error (e.g. ffmpeg printing
/// error messages to stdout) rather than a real video frame.
const MAX_JPEG_FRAME_BYTES: usize = 16 * 1024 * 1024;

/// Pipe-read buffer size.  Large enough to amortise syscall overhead without
/// wasting memory.
const READ_BUF_BYTES: usize = 64 * 1024;

/// JPEG Start-of-Image marker bytes.
const JPEG_SOI: [u8; 2] = [0xFF, 0xD8];

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// A running ffmpeg decode session.
///
/// Drop this to signal that no more frames are needed; the background thread
/// will notice the channel closed and exit.  If you need to wait for ffmpeg to
/// finish, call [`FfmpegReader::wait`].
pub struct FfmpegReader {
    /// Handle to the ffmpeg child process.
    child: Child,
    /// Receives decoded frames from the background reader thread.
    frame_rx: mpsc::Receiver<Result<DynamicImage>>,
}

impl FfmpegReader {
    /// Spawns ffmpeg to read from `source` and starts the background frame-
    /// splitting thread.
    ///
    /// `source` may be any value ffmpeg accepts as an input (`-i`): a local
    /// file path, an RTSP URL, an HTTP/HLS URL, etc.
    ///
    /// # Errors
    ///
    /// Returns `Err` if ffmpeg is not installed or the subprocess cannot be
    /// spawned.  Source-open errors (e.g. network unreachable, file not found)
    /// are reported as `Err` items on the first call to [`recv`][Self::recv].
    pub fn spawn(source: &str) -> Result<Self> {
        let mut child = Command::new("ffmpeg")
            .args([
                "-nostdin", // don't read from stdin (we pipe stdout)
                "-loglevel",
                "error", // suppress progress output; errors go to stderr
                "-i",
                source, // input source (file, URL, …)
                "-f",
                "image2pipe", // mux format: stream of images to a pipe
                "-vcodec",
                "mjpeg", // encode each decoded frame as JPEG
                "-q:v",
                "3",      // JPEG quality (1=best, 31=worst; 3 is high quality)
                "pipe:1", // output to stdout
            ])
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit()) // let ffmpeg errors appear in the terminal
            .spawn()
            .with_context(|| {
                format!(
                    "failed to spawn ffmpeg — is it installed? \
                     (tried to open source: '{source}')"
                )
            })?;

        let stdout: ChildStdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow!("ffmpeg stdout pipe was not captured"))?;

        let (frame_tx, frame_rx) = mpsc::channel();

        // Spawn a dedicated OS thread for blocking I/O.  Mixing blocking reads
        // with async tasks on a tokio thread would stall the executor.
        thread::Builder::new()
            .name("ffmpeg-reader".to_owned())
            .spawn(move || {
                read_frames_from_pipe(BufReader::with_capacity(READ_BUF_BYTES, stdout), frame_tx);
            })
            .context("failed to spawn ffmpeg-reader thread")?;

        Ok(Self { child, frame_rx })
    }

    /// Receives the next decoded frame.
    ///
    /// Returns:
    /// - `Ok(Some(frame))` — a successfully decoded frame.
    /// - `Ok(None)` — the stream ended (ffmpeg exited normally).
    /// - `Err(e)` — a decode or I/O error on the current frame; subsequent
    ///   calls may still yield frames (e.g. a single corrupt frame in a file).
    pub fn recv(&self) -> Option<Result<DynamicImage>> {
        self.frame_rx.recv().ok()
    }

    /// Waits for the ffmpeg subprocess to exit and returns its exit status.
    ///
    /// Call this after the frame loop to avoid zombie processes.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the wait syscall fails.
    pub fn wait(&mut self) -> Result<std::process::ExitStatus> {
        self.child
            .wait()
            .context("failed to wait for ffmpeg process")
    }
}

impl Drop for FfmpegReader {
    /// Kills the ffmpeg subprocess when the reader is dropped so it doesn't
    /// keep running after the caller stops consuming frames.
    fn drop(&mut self) {
        // Ignore errors here — the process may have already exited.
        let _ = self.child.kill();
    }
}

// ---------------------------------------------------------------------------
// Frame-splitting logic (pure, testable)
// ---------------------------------------------------------------------------

/// Reads bytes from `reader`, splits them on JPEG SOI markers, and sends
/// each complete JPEG to `tx`.
///
/// The function exits when `reader` reaches EOF or `tx` is closed (i.e. the
/// [`FfmpegReader`] was dropped).
fn read_frames_from_pipe(mut reader: impl Read, tx: mpsc::Sender<Result<DynamicImage>>) {
    let mut buf = Vec::with_capacity(READ_BUF_BYTES);
    let mut read_buf = vec![0u8; READ_BUF_BYTES];

    loop {
        let bytes_read = match reader.read(&mut read_buf) {
            Ok(0) => break, // EOF — ffmpeg exited
            Ok(n) => n,
            Err(err) => {
                // Send the I/O error and stop reading.
                let _ = tx.send(Err(
                    anyhow::Error::from(err).context("ffmpeg pipe read error")
                ));
                break;
            }
        };

        buf.extend_from_slice(&read_buf[..bytes_read]);

        // Extract all complete JPEG frames from the accumulated buffer.
        // A frame starts at SOI (\xFF\xD8) and ends just before the next SOI.
        // We can't use EOI (\xFF\xD9) as the end because some JPEG streams
        // omit it or embed it in EXIF markers; SOI is always unambiguous.
        loop {
            // Find the start of the *second* SOI — that marks the end of the
            // current frame.
            let next_soi = buf
                .windows(JPEG_SOI.len())
                .skip(1) // skip the very first byte to avoid matching at index 0
                .position(|w| w == JPEG_SOI)
                .map(|pos| pos + 1); // adjust for the skip(1) offset

            match next_soi {
                Some(end) => {
                    // We have a complete frame: buf[0..end].
                    let frame_bytes = buf[..end].to_vec();
                    buf.drain(..end);

                    if frame_bytes.len() > MAX_JPEG_FRAME_BYTES {
                        // Overly large "frame" — likely an ffmpeg error message
                        // leaking into stdout.  Log and skip.
                        let _ = tx.send(Err(anyhow!(
                            "oversized frame ({} bytes) from ffmpeg pipe — skipping",
                            frame_bytes.len()
                        )));
                        continue;
                    }

                    // Only try to decode frames that start with SOI.
                    if frame_bytes.starts_with(&JPEG_SOI) {
                        let decoded = image::load_from_memory_with_format(
                            &frame_bytes,
                            image::ImageFormat::Jpeg,
                        )
                        .map_err(|e| anyhow!("JPEG decode error: {e}"));

                        if tx.send(decoded).is_err() {
                            // Receiver dropped — stop reading.
                            return;
                        }
                    }
                }
                None => break, // need more data
            }
        }
    }

    // Flush the last (possibly incomplete) frame if it starts with SOI and
    // has a reasonable size.  This handles the final frame of a finite file.
    if buf.len() >= 2 && buf.starts_with(&JPEG_SOI) && buf.len() <= MAX_JPEG_FRAME_BYTES {
        let decoded = image::load_from_memory_with_format(&buf, image::ImageFormat::Jpeg)
            .map_err(|e| anyhow!("JPEG decode error (last frame): {e}"));
        let _ = tx.send(decoded);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    /// Builds a minimal valid JPEG from two 1×1 pixel solid-colour frames.
    /// We use actual JPEG bytes generated at test time rather than hard-coded
    /// constants so the test is not brittle against encoder changes.
    fn make_jpeg_bytes(r: u8, g: u8, b: u8) -> Vec<u8> {
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([r, g, b]));
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        let mut buf = Vec::new();
        let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90);
        enc.encode_image(&dyn_img).unwrap();
        buf
    }

    #[test]
    fn read_frames_splits_two_concatenated_jpegs() {
        let frame1 = make_jpeg_bytes(255, 0, 0);
        let frame2 = make_jpeg_bytes(0, 255, 0);

        // Concatenate them as ffmpeg would write to the pipe.
        let mut pipe_data = frame1.clone();
        pipe_data.extend_from_slice(&frame2);

        let (tx, rx) = mpsc::channel();
        read_frames_from_pipe(std::io::Cursor::new(pipe_data), tx);

        // We should receive exactly 2 decoded frames.
        let result1 = rx.recv().expect("expected frame 1");
        assert!(result1.is_ok(), "frame 1 should decode OK");
        let result2 = rx.recv().expect("expected frame 2");
        assert!(result2.is_ok(), "frame 2 should decode OK");
        // No third frame.
        assert!(rx.recv().is_err(), "expected channel closed after 2 frames");
    }

    #[test]
    fn read_frames_handles_empty_pipe() {
        let (tx, rx) = mpsc::channel();
        read_frames_from_pipe(std::io::Cursor::new(b""), tx);
        // Channel should close immediately with no frames.
        assert!(rx.recv().is_err(), "empty pipe should yield no frames");
    }

    #[test]
    fn read_frames_handles_garbage_data() {
        // Non-JPEG data — should produce no decodable frames (the garbage
        // won't start with SOI so we never attempt to decode it).
        let (tx, rx) = mpsc::channel();
        read_frames_from_pipe(std::io::Cursor::new(b"not a jpeg at all"), tx);
        assert!(
            rx.recv().is_err(),
            "garbage data should yield no decoded frames"
        );
    }

    #[test]
    fn read_frames_single_jpeg_at_eof() {
        // A single JPEG with no following SOI — flushed as the last frame.
        let frame = make_jpeg_bytes(128, 128, 128);
        let (tx, rx) = mpsc::channel();
        read_frames_from_pipe(std::io::Cursor::new(frame), tx);
        let result = rx.recv().expect("expected one frame from single JPEG");
        assert!(result.is_ok(), "single JPEG should decode OK: {result:?}");
    }

    #[test]
    fn ffmpeg_reader_spawn_fails_with_bad_source() {
        // A nonexistent source should either fail to spawn (if ffmpeg isn't
        // installed) or return quickly with an error frame.  Either way, the
        // call must not panic.
        let reader_result = FfmpegReader::spawn("/nonexistent_video_xyz.mp4");
        match reader_result {
            Err(_) => {} // ffmpeg not installed or spawn failed — ok
            Ok(mut reader) => {
                // ffmpeg spawned but should exit quickly with an error because
                // the file doesn't exist.  Drain the frame channel.
                while let Some(_frame) = reader.recv() {
                    // just consume
                }
                // Wait to avoid a zombie.
                let _ = reader.wait();
            }
        }
    }
}
