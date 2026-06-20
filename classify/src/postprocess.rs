//! Post-processing for classification model outputs.
//!
//! Provides softmax activation and top-k selection over a flat logit vector.

/// Applies softmax to a slice of logits and returns a probability `Vec<f32>`.
///
/// Numerically stable: subtracts the maximum logit before exponentiation.
pub fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }

    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    let exps: Vec<f32> = logits.iter().map(|&v| (v - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();

    exps.iter().map(|&e| e / sum).collect()
}

/// Applies softmax to `logits` and returns the top-`k` `(class_index, probability)` pairs,
/// sorted in descending order of probability.
///
/// If `k` exceeds the number of classes, all classes are returned.
pub fn top_k_softmax(logits: &[f32], k: usize) -> Vec<(usize, f32)> {
    let probs = softmax(logits);

    let mut indexed: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();

    // Partial sort: bring the top-k elements to the front.
    let take = k.min(indexed.len());
    indexed.select_nth_unstable_by(take.saturating_sub(1), |a, b| {
        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut top = indexed[..take].to_vec();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn softmax_uniform_logits_produce_equal_probs() {
        let logits = vec![1.0_f32; 4];
        let probs = softmax(&logits);
        for p in &probs {
            assert_abs_diff_eq!(*p, 0.25, epsilon = 1e-6);
        }
    }

    #[test]
    fn softmax_output_sums_to_one() {
        let logits = vec![2.0_f32, 1.0, 0.1, -1.0, 3.5];
        let probs = softmax(&logits);
        let total: f32 = probs.iter().sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-6);
    }

    #[test]
    fn softmax_large_dominant_logit_concentrates_probability() {
        // A very large logit relative to the others should get probability close to 1.
        let logits = vec![100.0_f32, 0.0, 0.0];
        let probs = softmax(&logits);
        assert!(
            probs[0] > 0.999,
            "expected dominant class prob > 0.999, got {}",
            probs[0]
        );
    }

    #[test]
    fn softmax_empty_input_returns_empty() {
        let probs = softmax(&[]);
        assert!(probs.is_empty());
    }

    #[test]
    fn softmax_numerical_stability_with_large_values() {
        // Without the max-subtraction trick this would overflow to NaN.
        let logits = vec![1000.0_f32, 1001.0, 1002.0];
        let probs = softmax(&logits);
        let total: f32 = probs.iter().sum();
        assert_abs_diff_eq!(total, 1.0, epsilon = 1e-6);
        assert!(
            probs.iter().all(|p| p.is_finite()),
            "expected all finite probabilities"
        );
    }

    #[test]
    fn top_k_returns_correct_number_of_results() {
        let logits = vec![0.1_f32, 5.0, 2.0, 1.0, 3.0];
        let top = top_k_softmax(&logits, 3);
        assert_eq!(top.len(), 3);
    }

    #[test]
    fn top_k_results_are_sorted_descending() {
        let logits = vec![0.1_f32, 5.0, 2.0, 1.0, 3.0];
        let top = top_k_softmax(&logits, 3);
        for window in top.windows(2) {
            assert!(
                window[0].1 >= window[1].1,
                "results not sorted: {} < {}",
                window[0].1,
                window[1].1
            );
        }
    }

    #[test]
    fn top_k_selects_highest_probability_class() {
        // Logit index 1 is dominant; it must appear first.
        let logits = vec![0.1_f32, 10.0, 0.1];
        let top = top_k_softmax(&logits, 1);
        assert_eq!(top[0].0, 1, "expected class 1 to be top prediction");
    }

    #[test]
    fn top_k_capped_at_number_of_classes() {
        let logits = vec![1.0_f32, 2.0];
        let top = top_k_softmax(&logits, 100);
        assert_eq!(top.len(), 2);
    }
}
