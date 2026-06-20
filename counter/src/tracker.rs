//! Multi-object tracker for person detections.
//!
//! Uses greedy IoU-based association: each new frame's detections are matched
//! to existing tracks by picking the highest-IoU pair above a threshold.
//! Unmatched detections spawn new tracks; unmatched tracks accumulate missed
//! frames and are removed after [`MAX_MISSED_FRAMES`] consecutive misses.
//!
//! This is intentionally simple — the goal is stable IDs for line-crossing
//! counting, not perfect re-identification.  A Kalman-filter / Hungarian
//! assignment upgrade can replace `greedy_match` without touching the public
//! API.
//!
//! All logic is pure (no I/O, no GPU) so it is fully unit-testable.

use crate::postprocess::PersonDetection;

/// Number of consecutive frames a track may go unmatched before it is pruned.
const MAX_MISSED_FRAMES: u32 = 5;

/// Minimum IoU for a detection to be associated with an existing track.
const ASSOCIATION_IOU_THRESHOLD: f32 = 0.30;

/// A single person track with a stable ID and its most-recent bounding box.
#[derive(Debug, Clone)]
pub struct Track {
    /// Monotonically-increasing, never-reused track identifier.
    pub id: u64,
    /// Most recently associated bounding box (original-image pixels).
    pub bbox: PersonDetection,
    /// Number of consecutive frames this track was not matched to a detection.
    pub missed_frames: u32,
    /// Total frames this track has been alive (including missed frames).
    pub frame_count: u32,
}

impl Track {
    /// Returns the centroid of the track's current bounding box as `(cx, cy)`.
    pub fn centroid(&self) -> (f32, f32) {
        self.bbox.centroid()
    }
}

/// Stateful multi-object tracker.
#[derive(Debug)]
pub struct Tracker {
    /// Currently active tracks.
    active_tracks: Vec<Track>,
    /// Counter used to assign unique IDs; incremented on every new track.
    next_track_id: u64,
}

impl Tracker {
    /// Creates a new empty tracker.
    pub fn new() -> Self {
        Self {
            active_tracks: Vec::new(),
            next_track_id: 1,
        }
    }

    /// Processes one frame's detections, updates track states, and returns the
    /// current list of active tracks (including newly-created ones).
    ///
    /// Tracks that exceed [`MAX_MISSED_FRAMES`] without a match are dropped
    /// from the returned list and will not appear in future frames.
    pub fn update(&mut self, detections: &[PersonDetection]) -> &[Track] {
        // Associate detections to existing tracks by greedy IoU matching.
        let (matched_pairs, unmatched_detection_indices, unmatched_track_indices) =
            greedy_match(&self.active_tracks, detections, ASSOCIATION_IOU_THRESHOLD);

        // Apply matched pairs: update track bboxes and reset missed count.
        for (track_idx, det_idx) in &matched_pairs {
            let track = &mut self.active_tracks[*track_idx];
            track.bbox = detections[*det_idx].clone();
            track.missed_frames = 0;
            track.frame_count += 1;
        }

        // Increment missed counter for unmatched existing tracks.
        for track_idx in &unmatched_track_indices {
            self.active_tracks[*track_idx].missed_frames += 1;
            self.active_tracks[*track_idx].frame_count += 1;
        }

        // Spawn new tracks for unmatched detections.
        for det_idx in &unmatched_detection_indices {
            let id = self.next_track_id;
            self.next_track_id += 1;
            self.active_tracks.push(Track {
                id,
                bbox: detections[*det_idx].clone(),
                missed_frames: 0,
                frame_count: 1,
            });
        }

        // Prune tracks that have been missing too long.
        self.active_tracks
            .retain(|t| t.missed_frames <= MAX_MISSED_FRAMES);

        &self.active_tracks
    }

    /// Returns the current active tracks without processing a new frame.
    ///
    /// Useful when a caller wants a snapshot of tracked people between frame
    /// updates (e.g. a concurrent dashboard that reads track positions).
    // The method is part of the public API; the compiler warns because nothing
    // in this binary crate calls it directly, but it is called in tests and
    // will be used by camera-capture code once that feature is enabled.
    #[allow(dead_code)]
    pub fn active_tracks(&self) -> &[Track] {
        &self.active_tracks
    }
}

impl Default for Tracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Associates `detections` to `tracks` via greedy IoU matching.
///
/// Returns three lists:
/// - `matched_pairs`: `(track_idx, detection_idx)` pairs above the threshold.
/// - `unmatched_detection_indices`: detections that had no suitable track.
/// - `unmatched_track_indices`: tracks that had no suitable detection.
///
/// The greedy strategy picks the globally-highest IoU pair first, removes both
/// participants, and repeats.  This is O(N·M) per frame which is fine for the
/// small person counts expected in a single-camera scene.
fn greedy_match(
    tracks: &[Track],
    detections: &[PersonDetection],
    iou_threshold: f32,
) -> (Vec<(usize, usize)>, Vec<usize>, Vec<usize>) {
    // Build the full IoU matrix.
    let mut iou_matrix: Vec<Vec<f32>> = tracks
        .iter()
        .map(|track| detections.iter().map(|det| track.bbox.iou(det)).collect())
        .collect();

    let mut matched_pairs: Vec<(usize, usize)> = Vec::new();
    let mut matched_tracks: Vec<bool> = vec![false; tracks.len()];
    let mut matched_detections: Vec<bool> = vec![false; detections.len()];

    // Repeat until no more valid pairs can be found.
    loop {
        // Find the (track, detection) pair with the highest IoU.
        let best = iou_matrix
            .iter()
            .enumerate()
            .flat_map(|(t_idx, row)| {
                row.iter()
                    .enumerate()
                    .map(move |(d_idx, &iou)| (t_idx, d_idx, iou))
            })
            .filter(|&(t_idx, d_idx, iou)| {
                iou >= iou_threshold && !matched_tracks[t_idx] && !matched_detections[d_idx]
            })
            .max_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        match best {
            Some((t_idx, d_idx, _)) => {
                matched_pairs.push((t_idx, d_idx));
                matched_tracks[t_idx] = true;
                matched_detections[d_idx] = true;

                // Zero out the row/column so these participants aren't picked again.
                for row in iou_matrix.iter_mut() {
                    row[d_idx] = 0.0;
                }
                for cell in iou_matrix[t_idx].iter_mut() {
                    *cell = 0.0;
                }
            }
            None => break,
        }
    }

    let unmatched_detection_indices = (0..detections.len())
        .filter(|&d| !matched_detections[d])
        .collect();

    let unmatched_track_indices = (0..tracks.len()).filter(|&t| !matched_tracks[t]).collect();

    (
        matched_pairs,
        unmatched_detection_indices,
        unmatched_track_indices,
    )
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    fn make_det(x1: f32, y1: f32, x2: f32, y2: f32) -> PersonDetection {
        PersonDetection {
            x1,
            y1,
            x2,
            y2,
            confidence: 0.9,
        }
    }

    // --- track creation ---

    #[test]
    fn first_frame_detections_create_new_tracks_with_sequential_ids() {
        let mut tracker = Tracker::new();
        let dets = vec![
            make_det(0.0, 0.0, 10.0, 10.0),
            make_det(50.0, 50.0, 60.0, 60.0),
        ];
        let tracks = tracker.update(&dets);
        assert_eq!(tracks.len(), 2);
        // IDs should be 1 and 2 in order.
        let ids: Vec<u64> = tracks.iter().map(|t| t.id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&2));
    }

    #[test]
    fn empty_frame_creates_no_tracks() {
        let mut tracker = Tracker::new();
        let tracks = tracker.update(&[]);
        assert!(tracks.is_empty());
    }

    // --- ID stability ---

    #[test]
    fn overlapping_detections_preserve_track_ids_across_frames() {
        let mut tracker = Tracker::new();

        // Frame 1: single person at [0,0,10,10].
        let frame1 = vec![make_det(0.0, 0.0, 10.0, 10.0)];
        let tracks_f1 = tracker.update(&frame1);
        let original_id = tracks_f1[0].id;

        // Frame 2: same person moved slightly — should associate with the same track.
        let frame2 = vec![make_det(1.0, 1.0, 11.0, 11.0)];
        let tracks_f2 = tracker.update(&frame2);

        assert_eq!(tracks_f2.len(), 1);
        assert_eq!(
            tracks_f2[0].id, original_id,
            "track ID should be stable across frames"
        );
    }

    #[test]
    fn non_overlapping_second_frame_detection_gets_new_id() {
        let mut tracker = Tracker::new();

        // Frame 1: person A at top-left.
        let frame1 = vec![make_det(0.0, 0.0, 10.0, 10.0)];
        let tracks_f1 = tracker.update(&frame1);
        let original_id = tracks_f1[0].id;

        // Frame 2: person A is gone; an entirely different person appears far away.
        let frame2 = vec![make_det(500.0, 500.0, 510.0, 510.0)];
        let tracks_f2 = tracker.update(&frame2);

        // After one missed frame the original track is still alive (below the prune threshold).
        // The new detection spawns a fresh track.
        let new_track = tracks_f2.iter().find(|t| t.id != original_id);
        assert!(
            new_track.is_some(),
            "expected a new track for the far-away detection"
        );
    }

    // --- aging / removal ---

    #[test]
    fn track_is_pruned_after_max_missed_frames() {
        let mut tracker = Tracker::new();

        // Frame 1: create a track.
        tracker.update(&[make_det(0.0, 0.0, 10.0, 10.0)]);

        // Subsequent empty frames age the track until it is pruned.
        for frame_num in 0..=MAX_MISSED_FRAMES {
            let tracks = tracker.update(&[]);
            if frame_num < MAX_MISSED_FRAMES {
                // Track should still be alive while within the window.
                assert_eq!(
                    tracks.len(),
                    1,
                    "track should survive at missed_frames={frame_num}"
                );
            } else {
                // After MAX_MISSED_FRAMES consecutive misses the track is removed.
                assert_eq!(
                    tracks.len(),
                    0,
                    "track should be pruned after {MAX_MISSED_FRAMES} misses"
                );
            }
        }
    }

    // --- greedy_match ---

    #[test]
    fn greedy_match_correctly_pairs_high_iou_boxes() {
        let tracks = vec![Track {
            id: 1,
            bbox: make_det(0.0, 0.0, 10.0, 10.0),
            missed_frames: 0,
            frame_count: 1,
        }];
        let detections = vec![
            make_det(0.5, 0.5, 10.5, 10.5),       // high IoU with track 0
            make_det(100.0, 100.0, 110.0, 110.0), // far away
        ];
        let (matched, unmatched_dets, unmatched_tracks) =
            greedy_match(&tracks, &detections, ASSOCIATION_IOU_THRESHOLD);

        assert_eq!(matched, vec![(0, 0)], "track 0 should match detection 0");
        assert_eq!(
            unmatched_dets,
            vec![1],
            "detection 1 is far away — unmatched"
        );
        assert!(unmatched_tracks.is_empty(), "all tracks matched");
    }

    #[test]
    fn greedy_match_returns_all_unmatched_when_below_threshold() {
        let tracks = vec![Track {
            id: 1,
            bbox: make_det(0.0, 0.0, 10.0, 10.0),
            missed_frames: 0,
            frame_count: 1,
        }];
        // Detection is far away — IoU well below threshold.
        let detections = vec![make_det(200.0, 200.0, 210.0, 210.0)];
        let (matched, unmatched_dets, unmatched_tracks) =
            greedy_match(&tracks, &detections, ASSOCIATION_IOU_THRESHOLD);

        assert!(matched.is_empty());
        assert_eq!(unmatched_dets, vec![0]);
        assert_eq!(unmatched_tracks, vec![0]);
    }

    // --- centroid helper ---

    #[test]
    fn track_centroid_returns_midpoint() {
        let track = Track {
            id: 1,
            bbox: make_det(10.0, 20.0, 30.0, 60.0),
            missed_frames: 0,
            frame_count: 1,
        };
        let (cx, cy) = track.centroid();
        assert_abs_diff_eq!(cx, 20.0, epsilon = 1e-4);
        assert_abs_diff_eq!(cy, 40.0, epsilon = 1e-4);
    }
}
