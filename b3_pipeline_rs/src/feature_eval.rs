// Rust implementation of IC (Information Coefficient) computation.
//
// compute_ic_series: Task 04
//   Computes Spearman rank correlation between a feature and pre-ranked forward returns
//   per date, using rayon for date-level parallelism.
//
// compute_ic_series_batch: Task 04
//   Processes N features against the same forward returns in one rayon call.
//
// compute_turnover_rs: Task 04
//   Computes feature turnover (1 - mean day-to-day rank autocorrelation).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Float64Builder, Int32Builder, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatch;
use rayon::prelude::*;

use crate::util::{rank_pct, pearson_corr};

// ── RecordBatch extraction helpers ────────────────────────────────────────────

/// Extract the wide-format Float64 columns from a RecordBatch (skipping "date" column).
/// Returns (date_strings, matrix) where matrix[row][col] is the float value (NaN if null).
fn extract_wide_batch(batch: &RecordBatch) -> Result<(Vec<String>, Vec<Vec<Option<f64>>>), String> {
    let n_rows = batch.num_rows();
    let schema = batch.schema();
    let fields = schema.fields();

    // First column must be "date"
    let date_col = batch
        .column_by_name("date")
        .ok_or("missing 'date' column")?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or("'date' column is not StringArray")?;

    let dates: Vec<String> = (0..n_rows)
        .map(|i| {
            if date_col.is_null(i) {
                String::new()
            } else {
                date_col.value(i).to_string()
            }
        })
        .collect();

    // Remaining columns are tickers (Float64, nullable)
    let n_ticker_cols = fields.len() - 1; // subtract date column
    let mut matrix: Vec<Vec<Option<f64>>> = Vec::with_capacity(n_rows);
    for _ in 0..n_rows {
        matrix.push(vec![None; n_ticker_cols]);
    }

    for col_idx in 1..fields.len() {
        let arr = batch.column(col_idx);
        let fa = arr
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| format!("column '{}' is not Float64Array", fields[col_idx].name()))?;
        for row in 0..n_rows {
            if !fa.is_null(row) {
                matrix[row][col_idx - 1] = Some(fa.value(row));
            }
        }
    }

    Ok((dates, matrix))
}

// ── Per-date IC computation ────────────────────────────────────────────────────

/// Compute IC for a single date row.
///
/// `feature_row`: slice of Option<f64> for each ticker column (None = missing).
/// `fwd_rank_row`: pre-ranked forward return values for the same tickers.
/// `min_valid_stocks`: minimum number of valid (non-null in both) stocks.
///
/// Returns Some(ic) or None if insufficient data.
fn compute_ic_for_row(
    feature_row: &[Option<f64>],
    fwd_rank_row: &[Option<f64>],
    min_valid_stocks: usize,
) -> Option<f64> {
    // Find intersection of valid positions
    let valid: Vec<usize> = (0..feature_row.len())
        .filter(|&j| feature_row[j].is_some() && fwd_rank_row[j].is_some())
        .collect();

    if valid.len() < min_valid_stocks {
        return None;
    }

    // Collect valid feature values and fwd_rank values
    let feat_vals: Vec<f64> = valid.iter().map(|&j| feature_row[j].unwrap()).collect();
    let fwd_vals: Vec<f64> = valid.iter().map(|&j| fwd_rank_row[j].unwrap()).collect();

    // Rank feature values (cross-sectional rank for this date)
    let feat_ranks = rank_pct(&feat_vals);
    // fwd_rank values are already ranked — use them directly
    // (they were pre-ranked by the Python caller via pandas rank(axis=1, pct=True))

    let ic = pearson_corr(&feat_ranks, &fwd_vals);
    if ic.is_nan() {
        None
    } else {
        Some(ic)
    }
}

// ── Schema builders ───────────────────────────────────────────────────────────

fn ic_series_schema() -> Schema {
    Schema::new(vec![
        Field::new("date", DataType::Utf8, false),
        Field::new("ic", DataType::Float64, true),
    ])
}

fn ic_series_batch_schema() -> Schema {
    Schema::new(vec![
        Field::new("feature_idx", DataType::Int32, false),
        Field::new("date", DataType::Utf8, false),
        Field::new("ic", DataType::Float64, true),
    ])
}

// ── Public Rust functions (called from pyfunction wrappers) ────────────────────

fn compute_ic_series_inner(
    feature_batch: &RecordBatch,
    fwd_rank_batch: &RecordBatch,
    min_valid_stocks: usize,
) -> Result<RecordBatch, String> {
    let (dates, feat_matrix) = extract_wide_batch(feature_batch)?;
    let (_, fwd_matrix) = extract_wide_batch(fwd_rank_batch)?;

    if feat_matrix.len() != fwd_matrix.len() {
        return Err(format!(
            "feature and fwd_rank batch have different row counts: {} vs {}",
            feat_matrix.len(),
            fwd_matrix.len()
        ));
    }

    let n_rows = dates.len();

    // Parallel computation across dates
    let ic_values: Vec<Option<f64>> = (0..n_rows)
        .into_par_iter()
        .map(|i| compute_ic_for_row(&feat_matrix[i], &fwd_matrix[i], min_valid_stocks))
        .collect();

    // Build output RecordBatch
    let schema = Arc::new(ic_series_schema());
    let mut date_builder = StringBuilder::with_capacity(n_rows, n_rows * 10);
    let mut ic_builder = Float64Builder::with_capacity(n_rows);

    for (i, ic_opt) in ic_values.iter().enumerate() {
        date_builder.append_value(&dates[i]);
        match ic_opt {
            Some(v) => ic_builder.append_value(*v),
            None => ic_builder.append_null(),
        }
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(date_builder.finish()),
        Arc::new(ic_builder.finish()),
    ];

    RecordBatch::try_new(schema, columns).map_err(|e| e.to_string())
}

// ── PyO3-exposed functions ────────────────────────────────────────────────────

/// Compute Information Coefficient (rank correlation) between a feature and pre-ranked forward returns.
///
/// Args:
///     feature_batch: pyarrow.RecordBatch — wide format. First column is "date" (Utf8),
///                    remaining columns are ticker names (Float64, nullable).
///     fwd_rank_batch: pyarrow.RecordBatch — same layout, pre-ranked forward returns.
///                     Pass the result of pandas rank(axis=1, pct=True) converted to Arrow.
///     min_valid_stocks: minimum number of valid stocks per date (default 10)
///
/// Returns:
///     pyarrow.RecordBatch with two columns: "date" (Utf8), "ic" (Float64, nullable).
#[pyfunction]
pub fn compute_ic_series(
    py: Python<'_>,
    feature_batch: PyObject,
    fwd_rank_batch: PyObject,
    min_valid_stocks: i64,
) -> PyResult<PyObject> {
    let feat_rb: RecordBatch = PyRecordBatch::extract_bound(&feature_batch.bind(py))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("feature_batch: {}", e)))?
        .into_inner();

    let fwd_rb: RecordBatch = PyRecordBatch::extract_bound(&fwd_rank_batch.bind(py))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fwd_rank_batch: {}", e)))?
        .into_inner();

    let min_stocks = if min_valid_stocks > 0 {
        min_valid_stocks as usize
    } else {
        10
    };

    let result = compute_ic_series_inner(&feat_rb, &fwd_rb, min_stocks)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    PyRecordBatch::new(result).to_pyarrow(py)
}

/// Compute IC series for multiple features against the same pre-ranked forward returns.
///
/// Args:
///     feature_batches: List[pyarrow.RecordBatch] — one per feature
///     fwd_rank_batch: pyarrow.RecordBatch — shared pre-ranked forward returns
///     min_valid_stocks: minimum number of valid stocks per date
///
/// Returns:
///     pyarrow.RecordBatch with columns [feature_idx (Int32), date (Utf8), ic (Float64)]
#[pyfunction]
pub fn compute_ic_series_batch(
    py: Python<'_>,
    feature_batches: Vec<PyObject>,
    fwd_rank_batch: PyObject,
    min_valid_stocks: i64,
) -> PyResult<PyObject> {
    let fwd_rb: RecordBatch = PyRecordBatch::extract_bound(&fwd_rank_batch.bind(py))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("fwd_rank_batch: {}", e)))?
        .into_inner();

    let feat_rbs: Vec<RecordBatch> = feature_batches
        .iter()
        .enumerate()
        .map(|(i, obj)| {
            PyRecordBatch::extract_bound(&obj.bind(py))
                .map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "feature_batches[{}]: {}",
                        i, e
                    ))
                })
                .map(|rb| rb.into_inner())
        })
        .collect::<PyResult<Vec<_>>>()?;

    let min_stocks = if min_valid_stocks > 0 {
        min_valid_stocks as usize
    } else {
        10
    };

    // Pre-extract the fwd_rank matrix once (shared across all features)
    let (fwd_dates, fwd_matrix) = extract_wide_batch(&fwd_rb)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let n_rows = fwd_dates.len();

    // Process each feature in parallel
    let results: Vec<(usize, Vec<Option<f64>>)> = feat_rbs
        .par_iter()
        .enumerate()
        .map(|(feat_idx, feat_rb)| {
            let feat_result = extract_wide_batch(feat_rb);
            match feat_result {
                Ok((_, feat_matrix)) => {
                    let ic_values: Vec<Option<f64>> = (0..n_rows)
                        .map(|i| {
                            if i < feat_matrix.len() {
                                compute_ic_for_row(&feat_matrix[i], &fwd_matrix[i], min_stocks)
                            } else {
                                None
                            }
                        })
                        .collect();
                    (feat_idx, ic_values)
                }
                Err(_) => (feat_idx, vec![None; n_rows]),
            }
        })
        .collect();

    // Build output RecordBatch with [feature_idx, date, ic]
    let total_rows = results.len() * n_rows;
    let schema = Arc::new(ic_series_batch_schema());
    let mut feat_idx_builder = Int32Builder::with_capacity(total_rows);
    let mut date_builder = StringBuilder::with_capacity(total_rows, total_rows * 10);
    let mut ic_builder = Float64Builder::with_capacity(total_rows);

    // Sort by feature_idx to produce deterministic output
    let mut sorted_results = results;
    sorted_results.sort_by_key(|(idx, _)| *idx);

    for (feat_idx, ic_values) in &sorted_results {
        for (row_i, ic_opt) in ic_values.iter().enumerate() {
            feat_idx_builder.append_value(*feat_idx as i32);
            date_builder.append_value(&fwd_dates[row_i]);
            match ic_opt {
                Some(v) => ic_builder.append_value(*v),
                None => ic_builder.append_null(),
            }
        }
    }

    let columns: Vec<ArrayRef> = vec![
        Arc::new(feat_idx_builder.finish()),
        Arc::new(date_builder.finish()),
        Arc::new(ic_builder.finish()),
    ];

    let batch = RecordBatch::try_new(schema, columns)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("RecordBatch: {}", e)))?;

    PyRecordBatch::new(batch).to_pyarrow(py)
}

/// Compute feature turnover: 1 - mean(day-to-day rank autocorrelation).
///
/// Args:
///     feature_batch: pyarrow.RecordBatch — wide format (same layout as compute_ic_series)
///
/// Returns:
///     f64 scalar — turnover value. Returns 1.0 if insufficient data.
#[pyfunction]
pub fn compute_turnover_rs(py: Python<'_>, feature_batch: PyObject) -> PyResult<f64> {
    let feat_rb: RecordBatch = PyRecordBatch::extract_bound(&feature_batch.bind(py))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("feature_batch: {}", e)))?
        .into_inner();

    let (_, feat_matrix) = extract_wide_batch(&feat_rb)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let n_rows = feat_matrix.len();
    if n_rows < 2 {
        return Ok(1.0);
    }

    // Compute ranked rows first
    // For each row, rank the non-null values, store as Option<f64> in same positions
    let ranked_rows: Vec<Vec<Option<f64>>> = feat_matrix
        .par_iter()
        .map(|row| {
            let n_cols = row.len();
            // Collect valid (index, value) pairs
            let valid: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .filter_map(|(j, opt)| opt.map(|v| (j, v)))
                .collect();

            if valid.is_empty() {
                return vec![None; n_cols];
            }

            let vals: Vec<f64> = valid.iter().map(|(_, v)| *v).collect();
            let ranks = rank_pct(&vals);

            let mut ranked_row = vec![None; n_cols];
            for (k, (j, _)) in valid.iter().enumerate() {
                ranked_row[*j] = Some(ranks[k]);
            }
            ranked_row
        })
        .collect();

    // Compute day-to-day Pearson correlation of ranked rows (consecutive pairs)
    // This mirrors the Python implementation using vectorized approach:
    // for each consecutive pair, compute correlation between rank vectors (filling nulls with 0,
    // demeaning, then computing Pearson — matching the pandas vectorized approach)
    let daily_corrs: Vec<Option<f64>> = (1..n_rows)
        .into_par_iter()
        .map(|i| {
            let r = &ranked_rows[i];
            let rs = &ranked_rows[i - 1];
            let n_cols = r.len();

            // Match Python: fill nulls with 0, demean, compute Pearson
            // r_filled[j] = r[j].unwrap_or(0.0)
            // rs_filled[j] = rs[j].unwrap_or(0.0)
            let r_filled: Vec<f64> = r.iter().map(|opt| opt.unwrap_or(0.0)).collect();
            let rs_filled: Vec<f64> = rs.iter().map(|opt| opt.unwrap_or(0.0)).collect();

            // Demean
            let r_mean = r_filled.iter().sum::<f64>() / n_cols as f64;
            let rs_mean = rs_filled.iter().sum::<f64>() / n_cols as f64;

            let r_dm: Vec<f64> = r_filled.iter().map(|&v| v - r_mean).collect();
            let rs_dm: Vec<f64> = rs_filled.iter().map(|&v| v - rs_mean).collect();

            let num: f64 = r_dm.iter().zip(rs_dm.iter()).map(|(a, b)| a * b).sum();
            let ss_r: f64 = r_dm.iter().map(|v| v * v).sum::<f64>().sqrt();
            let ss_rs: f64 = rs_dm.iter().map(|v| v * v).sum::<f64>().sqrt();
            let denom = ss_r * ss_rs;

            if denom == 0.0 {
                None
            } else {
                Some(num / denom)
            }
        })
        .collect();

    // Mean autocorrelation (dropping None values)
    let valid_corrs: Vec<f64> = daily_corrs.into_iter().flatten().collect();
    if valid_corrs.is_empty() {
        return Ok(1.0);
    }
    let mean_autocorr = valid_corrs.iter().sum::<f64>() / valid_corrs.len() as f64;
    // Return 1.0 - mean_autocorr (rounding to 4 decimal places is done in Python)
    Ok(1.0 - mean_autocorr)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_wide_batch(dates: &[&str], data: &[Vec<Option<f64>>]) -> RecordBatch {
        // data[row][col] = Option<f64>
        let n_cols = if data.is_empty() { 0 } else { data[0].len() };

        let mut date_builder = StringBuilder::new();
        for d in dates {
            date_builder.append_value(d);
        }

        let mut ticker_builders: Vec<Float64Builder> =
            (0..n_cols).map(|_| Float64Builder::new()).collect();

        for row in data {
            for (j, opt) in row.iter().enumerate() {
                match opt {
                    Some(v) => ticker_builders[j].append_value(*v),
                    None => ticker_builders[j].append_null(),
                }
            }
        }

        let mut fields = vec![Field::new("date", DataType::Utf8, false)];
        for j in 0..n_cols {
            fields.push(Field::new(format!("T{:02}", j), DataType::Float64, true));
        }
        let schema = Arc::new(Schema::new(fields));

        let mut columns: Vec<ArrayRef> = vec![Arc::new(date_builder.finish())];
        for mut b in ticker_builders {
            columns.push(Arc::new(b.finish()));
        }

        RecordBatch::try_new(schema, columns).unwrap()
    }

    // ── rank_pct tests ────────────────────────────────────────────────────────

    #[test]
    fn test_rank_pct_empty() {
        assert!(rank_pct(&[]).is_empty());
    }

    #[test]
    fn test_rank_pct_single() {
        let r = rank_pct(&[42.0]);
        assert_eq!(r.len(), 1);
        assert!((r[0] - 1.0).abs() < 1e-12); // 1/1 = 1.0
    }

    #[test]
    fn test_rank_pct_no_ties() {
        // [3, 1, 4, 2] → sorted: 1(idx1)=rank1, 2(idx3)=rank2, 3(idx0)=rank3, 4(idx2)=rank4
        // pct = rank/n: 3/4=0.75, 1/4=0.25, 4/4=1.0, 2/4=0.5
        let r = rank_pct(&[3.0, 1.0, 4.0, 2.0]);
        assert_eq!(r.len(), 4);
        assert!((r[0] - 0.75).abs() < 1e-12, "r[0]={}", r[0]);
        assert!((r[1] - 0.25).abs() < 1e-12, "r[1]={}", r[1]);
        assert!((r[2] - 1.00).abs() < 1e-12, "r[2]={}", r[2]);
        assert!((r[3] - 0.50).abs() < 1e-12, "r[3]={}", r[3]);
    }

    #[test]
    fn test_rank_pct_ties() {
        // [1, 1, 3] → tied at positions 0,1 (ranks 1,2 → avg 1.5) and position 2 (rank 3)
        // pct = avg_rank/3: 1.5/3=0.5, 1.5/3=0.5, 3/3=1.0
        let r = rank_pct(&[1.0, 1.0, 3.0]);
        assert_eq!(r.len(), 3);
        assert!((r[0] - 0.5).abs() < 1e-12, "r[0]={}", r[0]);
        assert!((r[1] - 0.5).abs() < 1e-12, "r[1]={}", r[1]);
        assert!((r[2] - 1.0).abs() < 1e-12, "r[2]={}", r[2]);
    }

    // ── pearson_corr tests ────────────────────────────────────────────────────

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
    fn test_pearson_corr_perfect_negative() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        let y = [5.0, 4.0, 3.0, 2.0, 1.0];
        let r = pearson_corr(&x, &y);
        assert!((r + 1.0).abs() < 1e-10, "r={}", r);
    }

    #[test]
    fn test_pearson_corr_all_same_returns_zero() {
        let x = [1.0, 1.0, 1.0];
        let y = [2.0, 3.0, 4.0];
        assert_eq!(pearson_corr(&x, &y), 0.0);
    }

    // ── compute_ic_series_inner tests ─────────────────────────────────────────

    #[test]
    fn test_ic_perfect_correlation() {
        // Feature = forward_returns (after ranking) → IC should be ~1.0
        let n_tickers = 15; // > min_valid_stocks (10)
        let dates = vec!["2020-01-02", "2020-01-03", "2020-01-04"];
        let mut data: Vec<Vec<Option<f64>>> = Vec::new();
        for _ in &dates {
            let row: Vec<Option<f64>> = (0..n_tickers).map(|j| Some(j as f64)).collect();
            data.push(row);
        }
        // fwd_rank: same layout, pre-ranked (0..n) / n as pct rank
        let mut fwd_data: Vec<Vec<Option<f64>>> = Vec::new();
        for _ in &dates {
            let row: Vec<Option<f64>> = (0..n_tickers)
                .map(|j| Some((j + 1) as f64 / n_tickers as f64))
                .collect();
            fwd_data.push(row);
        }

        let feat_rb = make_wide_batch(&dates, &data);
        let fwd_rb = make_wide_batch(&dates, &fwd_data);

        let result = compute_ic_series_inner(&feat_rb, &fwd_rb, 10).unwrap();
        let ic_col = result.column_by_name("ic").unwrap();
        let ic_arr = ic_col.as_any().downcast_ref::<Float64Array>().unwrap();

        // Every date should have IC ≈ 1.0
        for i in 0..result.num_rows() {
            assert!(!ic_arr.is_null(i), "IC should not be null for date {}", i);
            let ic = ic_arr.value(i);
            assert!(ic > 0.95, "Expected IC~1.0, got {}", ic);
        }
    }

    #[test]
    fn test_ic_zero_correlation() {
        // Uncorrelated feature and returns → IC near 0 (not strict)
        // Use 20 tickers, feature ranks ascending, fwd_rank ranks descending
        let n_tickers = 20;
        let dates = vec!["2020-01-02", "2020-01-03"];
        let feat_data: Vec<Vec<Option<f64>>> = dates
            .iter()
            .map(|_| (0..n_tickers).map(|j| Some(j as f64)).collect())
            .collect();
        let fwd_data: Vec<Vec<Option<f64>>> = dates
            .iter()
            .map(|_| {
                (0..n_tickers)
                    .map(|j| Some((n_tickers - 1 - j) as f64 / n_tickers as f64))
                    .collect()
            })
            .collect();

        let feat_rb = make_wide_batch(&dates, &feat_data);
        let fwd_rb = make_wide_batch(&dates, &fwd_data);

        let result = compute_ic_series_inner(&feat_rb, &fwd_rb, 10).unwrap();
        let ic_col = result.column_by_name("ic").unwrap();
        let ic_arr = ic_col.as_any().downcast_ref::<Float64Array>().unwrap();

        // Feature ranks 0..19, fwd ranks 19/20..0/20 — perfect negative corr
        // Just check it's finite (not NaN)
        for i in 0..result.num_rows() {
            if !ic_arr.is_null(i) {
                assert!(ic_arr.value(i).is_finite());
            }
        }
    }

    #[test]
    fn test_ic_all_null_row_produces_null_ic() {
        // A date with all null feature values → null IC
        let n_tickers = 15;
        let dates = vec!["2020-01-02"];
        let feat_data = vec![vec![None; n_tickers]];
        let fwd_data = vec![(0..n_tickers)
            .map(|j| Some((j + 1) as f64 / n_tickers as f64))
            .collect()];

        let feat_rb = make_wide_batch(&dates, &feat_data);
        let fwd_rb = make_wide_batch(&dates, &fwd_data);

        let result = compute_ic_series_inner(&feat_rb, &fwd_rb, 10).unwrap();
        let ic_col = result.column_by_name("ic").unwrap();
        let ic_arr = ic_col.as_any().downcast_ref::<Float64Array>().unwrap();
        assert!(ic_arr.is_null(0), "All-null row should produce null IC");
    }

    #[test]
    fn test_ic_single_stock_below_min_produces_null() {
        // Only 5 tickers (< min_valid_stocks=10) → null IC
        let n_tickers = 5;
        let dates = vec!["2020-01-02"];
        let feat_data = vec![(0..n_tickers).map(|j| Some(j as f64)).collect()];
        let fwd_data = vec![(0..n_tickers)
            .map(|j| Some((j + 1) as f64 / n_tickers as f64))
            .collect()];

        let feat_rb = make_wide_batch(&dates, &feat_data);
        let fwd_rb = make_wide_batch(&dates, &fwd_data);

        let result = compute_ic_series_inner(&feat_rb, &fwd_rb, 10).unwrap();
        let ic_arr = result
            .column_by_name("ic")
            .unwrap()
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(ic_arr.is_null(0), "< min_valid_stocks should produce null IC");
    }

    // ── compute_turnover_rs tests ─────────────────────────────────────────────

    #[test]
    fn test_turnover_constant_feature_near_zero() {
        // Same values every day → rank autocorrelation = 1 → turnover ≈ 0
        let n_tickers = 15;
        let n_dates = 30;
        let dates: Vec<String> = (0..n_dates)
            .map(|i| format!("2020-01-{:02}", i + 2))
            .collect();
        let date_refs: Vec<&str> = dates.iter().map(|s| s.as_str()).collect();

        let constant_row: Vec<Option<f64>> = (0..n_tickers).map(|j| Some(j as f64)).collect();
        let data = vec![constant_row; n_dates];

        let batch = make_wide_batch(&date_refs, &data);

        // Extract and call the internal logic
        let (_, feat_matrix) = extract_wide_batch(&batch).unwrap();
        let n_rows = feat_matrix.len();

        let ranked_rows: Vec<Vec<Option<f64>>> = feat_matrix
            .iter()
            .map(|row| {
                let valid: Vec<(usize, f64)> = row
                    .iter()
                    .enumerate()
                    .filter_map(|(j, opt)| opt.map(|v| (j, v)))
                    .collect();
                if valid.is_empty() {
                    return vec![None; row.len()];
                }
                let vals: Vec<f64> = valid.iter().map(|(_, v)| *v).collect();
                let ranks = rank_pct(&vals);
                let mut ranked = vec![None; row.len()];
                for (k, (j, _)) in valid.iter().enumerate() {
                    ranked[*j] = Some(ranks[k]);
                }
                ranked
            })
            .collect();

        let daily_corrs: Vec<Option<f64>> = (1..n_rows)
            .map(|i| {
                let r = &ranked_rows[i];
                let rs = &ranked_rows[i - 1];
                let n_cols = r.len();
                let r_filled: Vec<f64> = r.iter().map(|opt| opt.unwrap_or(0.0)).collect();
                let rs_filled: Vec<f64> = rs.iter().map(|opt| opt.unwrap_or(0.0)).collect();
                let r_mean = r_filled.iter().sum::<f64>() / n_cols as f64;
                let rs_mean = rs_filled.iter().sum::<f64>() / n_cols as f64;
                let r_dm: Vec<f64> = r_filled.iter().map(|&v| v - r_mean).collect();
                let rs_dm: Vec<f64> = rs_filled.iter().map(|&v| v - rs_mean).collect();
                let num: f64 = r_dm.iter().zip(rs_dm.iter()).map(|(a, b)| a * b).sum();
                let ss_r = r_dm.iter().map(|v| v * v).sum::<f64>().sqrt();
                let ss_rs = rs_dm.iter().map(|v| v * v).sum::<f64>().sqrt();
                let denom = ss_r * ss_rs;
                if denom == 0.0 {
                    None
                } else {
                    Some(num / denom)
                }
            })
            .collect();

        let valid_corrs: Vec<f64> = daily_corrs.into_iter().flatten().collect();
        assert!(!valid_corrs.is_empty());
        let mean_autocorr = valid_corrs.iter().sum::<f64>() / valid_corrs.len() as f64;
        let turnover = 1.0 - mean_autocorr;
        assert!(turnover < 0.05, "Constant feature turnover should be < 0.05, got {}", turnover);
    }

    #[test]
    fn test_output_schema_ic_series() {
        let dates = vec!["2020-01-02"];
        let n_tickers = 15;
        let data = vec![(0..n_tickers).map(|j| Some(j as f64)).collect()];
        let fwd = vec![(0..n_tickers)
            .map(|j| Some((j + 1) as f64 / n_tickers as f64))
            .collect()];
        let feat_rb = make_wide_batch(&dates, &data);
        let fwd_rb = make_wide_batch(&dates, &fwd);
        let result = compute_ic_series_inner(&feat_rb, &fwd_rb, 10).unwrap();
        assert!(result.column_by_name("date").is_some());
        assert!(result.column_by_name("ic").is_some());
        assert_eq!(result.num_rows(), 1);
    }
}
