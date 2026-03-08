// Shared math utilities used by feature_eval.rs and cross_section.rs.

// ── Ranking helper ─────────────────────────────────────────────────────────────

/// Compute fractional percentile ranks matching pandas `rank(axis=1, pct=True)`.
///
/// Uses average rank for tied values (matches pandas default `method='average'`).
/// For N values with no ties, rank of k-th smallest (1-indexed) is k/N.
pub fn rank_pct(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![1.0]; // single value: rank = 1/1 = 1.0 (matches pandas)
    }
    // argsort: indices that would sort values ascending
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks = vec![0.0f64; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        // Find the end of the tie group
        while j < n && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        // Average of 1-indexed ranks i+1 through j, divided by n
        // avg_rank (1-indexed) = (i+1 + j) / 2  [sum of arithmetic series / count]
        let avg_rank = (i + j + 1) as f64 / 2.0;
        for &k in &idx[i..j] {
            ranks[k] = avg_rank / n as f64;
        }
        i = j;
    }
    ranks
}

// ── Pearson correlation helper ─────────────────────────────────────────────────

/// Compute Pearson correlation of two equal-length slices.
///
/// Returns `f64::NAN` if either slice is empty.
/// Returns `0.0` if denominator is zero (all-identical values).
pub fn pearson_corr(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return f64::NAN;
    }
    debug_assert_eq!(x.len(), y.len(), "pearson_corr: slices must be equal length");

    let mean_x = x.iter().sum::<f64>() / n as f64;
    let mean_y = y.iter().sum::<f64>() / n as f64;

    let mut num = 0.0_f64;
    let mut ss_x = 0.0_f64;
    let mut ss_y = 0.0_f64;
    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        num += dx * dy;
        ss_x += dx * dx;
        ss_y += dy * dy;
    }
    let denom = ss_x.sqrt() * ss_y.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        num / denom
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rank_pct_empty() {
        assert!(rank_pct(&[]).is_empty());
    }

    #[test]
    fn test_rank_pct_single() {
        let r = rank_pct(&[42.0]);
        assert_eq!(r.len(), 1);
        assert!((r[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_rank_pct_no_ties() {
        // [3, 1, 4, 2] → pct = [0.75, 0.25, 1.0, 0.5]
        let r = rank_pct(&[3.0, 1.0, 4.0, 2.0]);
        assert!((r[0] - 0.75).abs() < 1e-12, "r[0]={}", r[0]);
        assert!((r[1] - 0.25).abs() < 1e-12, "r[1]={}", r[1]);
        assert!((r[2] - 1.00).abs() < 1e-12, "r[2]={}", r[2]);
        assert!((r[3] - 0.50).abs() < 1e-12, "r[3]={}", r[3]);
    }

    #[test]
    fn test_rank_pct_ties() {
        // [1, 1, 3] → avg tied ranks = 1.5/3=0.5, 3/3=1.0
        let r = rank_pct(&[1.0, 1.0, 3.0]);
        assert!((r[0] - 0.5).abs() < 1e-12, "r[0]={}", r[0]);
        assert!((r[1] - 0.5).abs() < 1e-12, "r[1]={}", r[1]);
        assert!((r[2] - 1.0).abs() < 1e-12, "r[2]={}", r[2]);
    }

    #[test]
    fn test_pearson_corr_empty_is_nan() {
        assert!(pearson_corr(&[], &[]).is_nan());
    }

    #[test]
    fn test_pearson_corr_perfect_positive() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [2.0, 4.0, 6.0, 8.0, 10.0];
        let r = pearson_corr(&x, &y);
        assert!((r - 1.0).abs() < 1e-10, "r={}", r);
    }

    #[test]
    fn test_pearson_corr_all_same_returns_zero() {
        let x = [1.0, 1.0, 1.0];
        let y = [2.0, 3.0, 4.0];
        assert_eq!(pearson_corr(&x, &y), 0.0);
    }
}
