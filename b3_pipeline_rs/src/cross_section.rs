// Cross-sectional rank and z-score operators.
//
// cross_sectional_rank: computes percentile rank per date (row-wise) matching
//   pandas `rank(axis=1, pct=True)`. NaN/null inputs remain null in output.
//
// cross_sectional_zscore: computes z-score per date (row-wise) matching
//   pandas `df.sub(mean, axis=0).div(std, axis=0)` with ddof=1.
//   Rows with zero variance or fewer than 2 valid values are all-null.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Float64Builder, StringArray, StringBuilder};
use arrow::datatypes::Schema;
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatch;
use rayon::prelude::*;

use crate::util::rank_pct;

// ── Shared extraction helper ──────────────────────────────────────────────────

/// Extract the wide-format Float64 columns from a RecordBatch (skipping "date" column).
/// Returns (date_strings, n_ticker_cols, matrix) where matrix[row][col] is Option<f64>.
fn extract_wide(batch: &RecordBatch) -> Result<(Vec<String>, usize, Vec<Vec<Option<f64>>>), String> {
    let n_rows = batch.num_rows();
    let schema = batch.schema();
    let fields = schema.fields();

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

    let n_ticker_cols = fields.len() - 1; // subtract date column
    let mut matrix: Vec<Vec<Option<f64>>> = vec![vec![None; n_ticker_cols]; n_rows];

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

    Ok((dates, n_ticker_cols, matrix))
}

// ── Build output RecordBatch ──────────────────────────────────────────────────

/// Build a wide RecordBatch with the same schema as the input but with new values.
///
/// `dates`: date strings for the "date" column.
/// `schema`: Arc<Schema> to use for the output (same as input).
/// `result_matrix[row][col]`: Option<f64> output values.
fn build_output_batch(
    dates: &[String],
    schema: Arc<Schema>,
    result_matrix: Vec<Vec<Option<f64>>>,
) -> Result<RecordBatch, String> {
    let n_rows = dates.len();
    let n_ticker_cols = schema.fields().len() - 1;

    let mut date_builder = StringBuilder::with_capacity(n_rows, n_rows * 10);
    for d in dates {
        date_builder.append_value(d);
    }

    let mut ticker_builders: Vec<Float64Builder> =
        (0..n_ticker_cols).map(|_| Float64Builder::with_capacity(n_rows)).collect();

    for row_idx in 0..n_rows {
        let row = &result_matrix[row_idx];
        for (col_idx, opt) in row.iter().enumerate() {
            match opt {
                Some(v) => ticker_builders[col_idx].append_value(*v),
                None => ticker_builders[col_idx].append_null(),
            }
        }
    }

    let mut columns: Vec<ArrayRef> = vec![Arc::new(date_builder.finish())];
    for mut b in ticker_builders {
        columns.push(Arc::new(b.finish()));
    }

    RecordBatch::try_new(schema, columns).map_err(|e| e.to_string())
}

// ── cross_sectional_rank inner ────────────────────────────────────────────────

fn cross_sectional_rank_inner(batch: &RecordBatch) -> Result<RecordBatch, String> {
    let (dates, n_ticker_cols, matrix) = extract_wide(batch)?;
    let n_rows = dates.len();

    // Compute ranked rows in parallel
    let result_matrix: Vec<Vec<Option<f64>>> = (0..n_rows)
        .into_par_iter()
        .map(|row_idx| {
            let row = &matrix[row_idx];
            // Collect (column_index, value) for non-null positions
            let valid: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .filter_map(|(col, opt)| opt.map(|v| (col, v)))
                .collect();

            if valid.is_empty() {
                return vec![None; n_ticker_cols];
            }

            let vals: Vec<f64> = valid.iter().map(|(_, v)| *v).collect();
            let ranks = rank_pct(&vals);

            let mut output_row = vec![None; n_ticker_cols];
            for (k, (col, _)) in valid.iter().enumerate() {
                output_row[*col] = Some(ranks[k]);
            }
            output_row
        })
        .collect();

    build_output_batch(&dates, batch.schema(), result_matrix)
}

// ── cross_sectional_zscore inner ──────────────────────────────────────────────

fn cross_sectional_zscore_inner(batch: &RecordBatch) -> Result<RecordBatch, String> {
    let (dates, n_ticker_cols, matrix) = extract_wide(batch)?;
    let n_rows = dates.len();

    // Compute z-scored rows in parallel
    let result_matrix: Vec<Vec<Option<f64>>> = (0..n_rows)
        .into_par_iter()
        .map(|row_idx| {
            let row = &matrix[row_idx];

            // Collect (column_index, value) for non-null positions
            let valid: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .filter_map(|(col, opt)| opt.map(|v| (col, v)))
                .collect();

            let n = valid.len();
            if n < 2 {
                // Cannot compute std with fewer than 2 values (ddof=1)
                return vec![None; n_ticker_cols];
            }

            let vals: Vec<f64> = valid.iter().map(|(_, v)| *v).collect();

            // Compute mean
            let mean = vals.iter().sum::<f64>() / n as f64;

            // Compute sample std (ddof=1)
            let variance = vals.iter().map(|&v| (v - mean) * (v - mean)).sum::<f64>()
                / (n - 1) as f64;
            let std = variance.sqrt();

            if std == 0.0 {
                // All values identical → z-score undefined
                return vec![None; n_ticker_cols];
            }

            let mut output_row = vec![None; n_ticker_cols];
            for (col, v) in &valid {
                output_row[*col] = Some((v - mean) / std);
            }
            output_row
        })
        .collect();

    build_output_batch(&dates, batch.schema(), result_matrix)
}

// ── PyO3-exposed functions ────────────────────────────────────────────────────

/// Compute cross-sectional percentile rank per date (row-wise).
///
/// Args:
///     feature_batch: pyarrow.RecordBatch — wide format. First column is "date" (Utf8),
///                    remaining columns are ticker names (Float64, nullable).
///
/// Returns:
///     pyarrow.RecordBatch — same schema as input. Values replaced with fractional
///     percentile ranks in [0, 1]. NaN/null inputs remain null in output.
#[pyfunction]
pub fn cross_sectional_rank(py: Python<'_>, feature_batch: PyObject) -> PyResult<PyObject> {
    let batch: RecordBatch = PyRecordBatch::extract_bound(&feature_batch.bind(py))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("feature_batch: {}", e)))?
        .into_inner();

    let result = cross_sectional_rank_inner(&batch)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    PyRecordBatch::new(result).to_pyarrow(py)
}

/// Compute cross-sectional z-score per date (row-wise).
///
/// z = (x - mean) / std, computed independently per date across all tickers.
/// NaN/null inputs are excluded from mean/std computation and remain null in output.
/// Where std == 0 (all identical values on a date), output is null for that entire date.
///
/// Args:
///     feature_batch: pyarrow.RecordBatch — wide format (same layout as cross_sectional_rank)
///
/// Returns:
///     pyarrow.RecordBatch — same schema. Values replaced with z-scores.
#[pyfunction]
pub fn cross_sectional_zscore(py: Python<'_>, feature_batch: PyObject) -> PyResult<PyObject> {
    let batch: RecordBatch = PyRecordBatch::extract_bound(&feature_batch.bind(py))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("feature_batch: {}", e)))?
        .into_inner();

    let result = cross_sectional_zscore_inner(&batch)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    PyRecordBatch::new(result).to_pyarrow(py)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::datatypes::{DataType, Field};

    fn make_batch(dates: &[&str], data: &[Vec<Option<f64>>]) -> RecordBatch {
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

    // ── rank tests ────────────────────────────────────────────────────────────

    /// Helper: read value at (row, ticker_col) from a result batch.
    /// ticker_col is 0-indexed among ticker columns (excludes the "date" column).
    fn get_value(batch: &RecordBatch, row: usize, ticker_col: usize) -> Option<f64> {
        let arr = batch.column(ticker_col + 1).as_any().downcast_ref::<Float64Array>().unwrap();
        if arr.is_null(row) { None } else { Some(arr.value(row)) }
    }

    #[test]
    fn test_rank_3row_4col_known_values() {
        // Row 0: [4, 1, 3, 2] → pct ranks: col0=1.0, col1=0.25, col2=0.75, col3=0.5
        // Row 1: [1, 2, 3, 4] → pct ranks: col0=0.25, col1=0.5, col2=0.75, col3=1.0
        // Row 2: [2, 4, 1, 3] → pct ranks: col0=0.5, col1=1.0, col2=0.25, col3=0.75
        let batch = make_batch(
            &["2020-01-02", "2020-01-03", "2020-01-04"],
            &[
                vec![Some(4.0), Some(1.0), Some(3.0), Some(2.0)],
                vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
                vec![Some(2.0), Some(4.0), Some(1.0), Some(3.0)],
            ],
        );

        let result = cross_sectional_rank_inner(&batch).unwrap();

        // Row 0
        assert!((get_value(&result, 0, 0).unwrap() - 1.00).abs() < 1e-10);
        assert!((get_value(&result, 0, 1).unwrap() - 0.25).abs() < 1e-10);
        assert!((get_value(&result, 0, 2).unwrap() - 0.75).abs() < 1e-10);
        assert!((get_value(&result, 0, 3).unwrap() - 0.50).abs() < 1e-10);

        // Row 1
        assert!((get_value(&result, 1, 0).unwrap() - 0.25).abs() < 1e-10);
        assert!((get_value(&result, 1, 1).unwrap() - 0.50).abs() < 1e-10);
        assert!((get_value(&result, 1, 2).unwrap() - 0.75).abs() < 1e-10);
        assert!((get_value(&result, 1, 3).unwrap() - 1.00).abs() < 1e-10);

        // Row 2
        assert!((get_value(&result, 2, 0).unwrap() - 0.50).abs() < 1e-10);
        assert!((get_value(&result, 2, 1).unwrap() - 1.00).abs() < 1e-10);
        assert!((get_value(&result, 2, 2).unwrap() - 0.25).abs() < 1e-10);
        assert!((get_value(&result, 2, 3).unwrap() - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_rank_all_null_row_remains_all_null() {
        let batch = make_batch(
            &["2020-01-02", "2020-01-03"],
            &[
                vec![None, None, None, None],
                vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
            ],
        );

        let result = cross_sectional_rank_inner(&batch).unwrap();

        // Row 0 should be all null
        for col in 0..4 {
            assert!(get_value(&result, 0, col).is_none(), "row0 col{} should be null", col);
        }

        // Row 1 should be valid
        for col in 0..4 {
            assert!(get_value(&result, 1, col).is_some(), "row1 col{} should not be null", col);
        }
    }

    #[test]
    fn test_rank_single_non_null_value_returns_1() {
        // pandas rank(pct=True) with single non-null value: rank = 1/1 = 1.0
        let batch = make_batch(
            &["2020-01-02"],
            &[vec![Some(5.0), None, None]],
        );

        let result = cross_sectional_rank_inner(&batch).unwrap();

        assert!(get_value(&result, 0, 0).is_some(), "non-null col should have value");
        assert!((get_value(&result, 0, 0).unwrap() - 1.0).abs() < 1e-12,
            "single stock rank={}", get_value(&result, 0, 0).unwrap());
        assert!(get_value(&result, 0, 1).is_none(), "null col should stay null");
        assert!(get_value(&result, 0, 2).is_none(), "null col should stay null");
    }

    // ── zscore tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_zscore_2row_mean_approx_zero() {
        // Row 0: [1, 2, 3, 4] → mean=2.5, std=sqrt(5/3)≈1.291
        // z = [-1.161, -0.387, 0.387, 1.161] — sum should be ~0
        let batch = make_batch(
            &["2020-01-02", "2020-01-03"],
            &[
                vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)],
                vec![Some(10.0), Some(20.0), Some(30.0), Some(40.0)],
            ],
        );

        let result = cross_sectional_zscore_inner(&batch).unwrap();

        // Check row 0: sum of z-scores should be ~0
        let sum0: f64 = (0..4).map(|col| {
            let v = get_value(&result, 0, col);
            assert!(v.is_some(), "row0 col{} should not be null", col);
            v.unwrap()
        }).sum();
        assert!(sum0.abs() < 1e-10, "row0 z-score sum should be ~0, got {}", sum0);

        // Check row 1: same — sum should be ~0
        let sum1: f64 = (0..4).map(|col| get_value(&result, 1, col).unwrap()).sum();
        assert!(sum1.abs() < 1e-10, "row1 z-score sum should be ~0, got {}", sum1);
    }

    #[test]
    fn test_zscore_all_identical_row_is_all_null() {
        // When all values in a row are the same, std=0 → z-score undefined → all null
        let batch = make_batch(
            &["2020-01-02", "2020-01-03"],
            &[
                vec![Some(5.0), Some(5.0), Some(5.0)],     // std=0 → all null
                vec![Some(1.0), Some(2.0), Some(3.0)],     // normal → valid
            ],
        );

        let result = cross_sectional_zscore_inner(&batch).unwrap();

        // Row 0: all null
        for col in 0..3 {
            assert!(get_value(&result, 0, col).is_none(), "row0 col{} should be null (std=0)", col);
        }

        // Row 1: all valid
        for col in 0..3 {
            assert!(get_value(&result, 1, col).is_some(), "row1 col{} should not be null", col);
        }
    }

    #[test]
    fn test_zscore_null_values_stay_null() {
        // Non-null values are z-scored using only valid observations.
        // Null positions remain null in the output.
        let batch = make_batch(
            &["2020-01-02"],
            &[vec![Some(1.0), None, Some(3.0), Some(5.0)]],
        );

        let result = cross_sectional_zscore_inner(&batch).unwrap();

        // Column 1 (None) must stay null
        assert!(get_value(&result, 0, 1).is_none(), "null input must stay null in output");

        // Other columns must be valid
        assert!(get_value(&result, 0, 0).is_some(), "col0 should be valid");
        assert!(get_value(&result, 0, 2).is_some(), "col2 should be valid");
        assert!(get_value(&result, 0, 3).is_some(), "col3 should be valid");

        // z-scores of [1, 3, 5]: mean=3, std=2 (sample), z=[-1, 0, 1]
        assert!((get_value(&result, 0, 0).unwrap() - (-1.0)).abs() < 1e-10,
            "z[0]={}", get_value(&result, 0, 0).unwrap());
        assert!((get_value(&result, 0, 2).unwrap() - 0.0).abs() < 1e-10,
            "z[2]={}", get_value(&result, 0, 2).unwrap());
        assert!((get_value(&result, 0, 3).unwrap() - 1.0).abs() < 1e-10,
            "z[3]={}", get_value(&result, 0, 3).unwrap());
    }
}
