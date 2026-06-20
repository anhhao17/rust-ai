//! Line-crossing counter for person tracks.
//!
//! A [`CountingLine`] is defined by two points in image space.  On each frame
//! update the counter checks which tracks have moved from one side of the line
//! to the other since the last frame.  Crossing direction determines whether
//! the movement is counted as "entered" or "left".
//!
//! Crossing direction convention
//! ------------------------------
//! Given a directed line from `start` to `end`, the left-hand side of the
//! line is considered the *inside* zone.  A centroid moving from the right
//! side to the left side increments `entered`; the reverse increments `left`.
//!
//! Formally, the signed side is `cross_product(line_vec, centroid - start)`
//! (the z-component of the 3-D cross product).  Positive z = left side
//! (entered); negative z = right side (left).
//!
//! All logic is pure — no I/O, no GPU — so it is fully unit-testable.

use std::collections::HashMap;

use crate::tracker::Track;

/// A line segment in image space that defines the virtual counting boundary.
#[derive(Debug, Clone, Copy)]
pub struct CountingLine {
    /// First endpoint.
    pub start: (f32, f32),
    /// Second endpoint.
    pub end: (f32, f32),
}

/// Signed side of the counting line — positive = "inside", negative = "outside".
///
/// We store the raw sign (+1 / 0 / -1) so we can detect direction of change.
type Side = i8;

/// Running tally maintained by [`LineCounter`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CountTally {
    /// Number of crossings from outside → inside.
    pub entered: u64,
    /// Number of crossings from inside → outside.
    pub left: u64,
}

impl CountTally {
    /// Net occupancy count: `entered - left`.
    pub fn net(&self) -> i64 {
        self.entered as i64 - self.left as i64
    }
}

/// Stateful line-crossing counter.
///
/// Tracks the last-known side of each active track and emits `entered`/`left`
/// events when a track's centroid crosses the counting line.
#[derive(Debug)]
pub struct LineCounter {
    /// The virtual counting line.
    line: CountingLine,
    /// Maps track ID → signed side on the previous frame.
    previous_sides: HashMap<u64, Side>,
    /// Cumulative tally.
    tally: CountTally,
}

impl LineCounter {
    /// Creates a new counter for the given counting line.
    pub fn new(line: CountingLine) -> Self {
        Self {
            line,
            previous_sides: HashMap::new(),
            tally: CountTally::default(),
        }
    }

    /// Updates the counter with the current active tracks.
    ///
    /// For each track whose centroid has crossed the line since the last call,
    /// `entered` or `left` is incremented.  Tracks that disappeared since the
    /// last frame are removed from state (no double-counting on re-appearance).
    pub fn update(&mut self, tracks: &[Track]) {
        let mut current_ids: std::collections::HashSet<u64> =
            std::collections::HashSet::with_capacity(tracks.len());

        for track in tracks {
            current_ids.insert(track.id);

            let current_side = signed_side(self.line, track.centroid());

            if let Some(&previous_side) = self.previous_sides.get(&track.id) {
                // A crossing is only registered when both sides are non-zero
                // (centroid not on the line) and the sign has flipped.
                if previous_side != 0 && current_side != 0 && previous_side != current_side {
                    if current_side > 0 {
                        self.tally.entered += 1;
                    } else {
                        self.tally.left += 1;
                    }
                }
            }

            self.previous_sides.insert(track.id, current_side);
        }

        // Remove stale state for tracks that are no longer active.
        // This prevents a re-appearing track from immediately re-triggering a
        // crossing event based on stale side information.
        self.previous_sides.retain(|id, _| current_ids.contains(id));
    }

    /// Returns the current cumulative [`CountTally`].
    pub fn tally(&self) -> CountTally {
        self.tally
    }
}

/// Returns the signed side of `point` relative to `line`.
///
/// Positive → left-hand side ("entered"); negative → right-hand side ("left");
/// zero → exactly on the line.
///
/// The value is the z-component of the cross product of the line vector and
/// the vector from `line.start` to `point`.  This is a standard computational-
/// geometry primitive for half-plane tests.
fn signed_side(line: CountingLine, point: (f32, f32)) -> Side {
    let (lx, ly) = (line.end.0 - line.start.0, line.end.1 - line.start.1);
    let (px, py) = (point.0 - line.start.0, point.1 - line.start.1);

    // z-component of cross product: lx*py - ly*px
    let cross = lx * py - ly * px;

    if cross > 0.0 {
        1
    } else if cross < 0.0 {
        -1
    } else {
        0
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::postprocess::PersonDetection;

    /// Horizontal counting line at y=100, running left to right.
    fn horizontal_line() -> CountingLine {
        CountingLine {
            start: (0.0, 100.0),
            end: (1000.0, 100.0),
        }
    }

    fn make_track(id: u64, cx: f32, cy: f32) -> Track {
        use crate::tracker::Track;
        Track {
            id,
            bbox: PersonDetection {
                x1: cx - 5.0,
                y1: cy - 10.0,
                x2: cx + 5.0,
                y2: cy + 10.0,
                confidence: 0.9,
            },
            missed_frames: 0,
            frame_count: 1,
        }
    }

    // --- signed_side ---

    #[test]
    fn signed_side_above_horizontal_line_is_negative() {
        // For a left-to-right line at y=100, a point at y=50 is above the line
        // (right-hand side when looking along the line direction → negative).
        let line = horizontal_line();
        assert_eq!(signed_side(line, (500.0, 50.0)), -1);
    }

    #[test]
    fn signed_side_below_horizontal_line_is_positive() {
        let line = horizontal_line();
        assert_eq!(signed_side(line, (500.0, 150.0)), 1);
    }

    #[test]
    fn signed_side_on_line_is_zero() {
        let line = horizontal_line();
        assert_eq!(signed_side(line, (500.0, 100.0)), 0);
    }

    // --- crossing detection ---

    #[test]
    fn crossing_from_above_to_below_counts_as_entered() {
        let mut counter = LineCounter::new(horizontal_line());

        // Frame 1: track above the line (y=50 → side=-1).
        let frame1 = vec![make_track(1, 100.0, 50.0)];
        counter.update(&frame1);
        assert_eq!(counter.tally().entered, 0, "no crossing yet");

        // Frame 2: track below the line (y=150 → side=+1) → entered.
        let frame2 = vec![make_track(1, 100.0, 150.0)];
        counter.update(&frame2);
        assert_eq!(counter.tally().entered, 1);
        assert_eq!(counter.tally().left, 0);
    }

    #[test]
    fn crossing_from_below_to_above_counts_as_left() {
        let mut counter = LineCounter::new(horizontal_line());

        let frame1 = vec![make_track(1, 100.0, 150.0)]; // below
        counter.update(&frame1);

        let frame2 = vec![make_track(1, 100.0, 50.0)]; // above
        counter.update(&frame2);

        assert_eq!(counter.tally().left, 1);
        assert_eq!(counter.tally().entered, 0);
    }

    #[test]
    fn no_double_count_on_same_side_consecutive_frames() {
        let mut counter = LineCounter::new(horizontal_line());

        let frame1 = vec![make_track(1, 100.0, 50.0)]; // above
        counter.update(&frame1);
        let frame2 = vec![make_track(1, 100.0, 60.0)]; // still above
        counter.update(&frame2);

        assert_eq!(counter.tally().entered, 0);
        assert_eq!(counter.tally().left, 0);
    }

    #[test]
    fn multiple_tracks_counted_independently() {
        let mut counter = LineCounter::new(horizontal_line());

        // Frame 1: two tracks, both above.
        let frame1 = vec![make_track(1, 100.0, 50.0), make_track(2, 200.0, 50.0)];
        counter.update(&frame1);

        // Frame 2: both cross to below.
        let frame2 = vec![make_track(1, 100.0, 150.0), make_track(2, 200.0, 150.0)];
        counter.update(&frame2);

        assert_eq!(counter.tally().entered, 2);
        assert_eq!(counter.tally().left, 0);
    }

    #[test]
    fn disappearing_and_reappearing_track_does_not_double_count() {
        let mut counter = LineCounter::new(horizontal_line());

        // Frame 1: track appears below the line.
        counter.update(&[make_track(1, 100.0, 150.0)]);
        // Frame 2: track disappears — state for ID 1 is cleared.
        counter.update(&[]);
        // Frame 3: track re-appears above the line.  Without the stale-state
        // cleanup this would look like a crossing from the remembered side.
        counter.update(&[make_track(1, 100.0, 50.0)]);
        // Frame 4: track crosses back below — only ONE crossing expected.
        counter.update(&[make_track(1, 100.0, 150.0)]);

        assert_eq!(
            counter.tally().entered,
            1,
            "only one genuine crossing should be counted"
        );
    }

    // --- net count ---

    #[test]
    fn net_count_reflects_entered_minus_left() {
        let tally = CountTally {
            entered: 5,
            left: 2,
        };
        assert_eq!(tally.net(), 3);
    }

    #[test]
    fn net_count_can_be_negative_when_more_leave_than_enter() {
        let tally = CountTally {
            entered: 1,
            left: 3,
        };
        assert_eq!(tally.net(), -2);
    }
}
