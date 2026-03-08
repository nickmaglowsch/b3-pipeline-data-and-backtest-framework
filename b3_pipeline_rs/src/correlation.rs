// Rust implementation of pairwise Spearman correlation matrix computation.
//
// compute_pairwise_spearman: Task 06
//   Accepts N pre-ranked feature vectors (each of length sampled_dates × n_tickers),
//   computes all N*(N-1)/2 pairwise Pearson correlations of rank vectors (= Spearman)
//   in parallel via rayon, and returns an N×N symmetric correlation matrix as a
//   PyArrow RecordBatch.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Array, Float64Builder, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3_arrow::{PyArray, PyRecordBatch};
use rayon::prelude::*;

use crate::util::pearson_corr;

// ── Internal helpers ──────────────────────────────────────────────────────────

/// Compute pairwise correlation for two feature rank vectors, considering only
/// positions where both vectors are valid (non-NaN).
///
/// Returns NaN if fewer than 10 co-valid positions exist.
fn compute_pair_corr(a: &[f64], b: &[f64], valid_a: &[bool], valid_b: &[bool]) -> f64 {
    let both_valid: Vec<(f64, f64)> = a
        .iter()
        .zip(b.iter())
        .zip(valid_a.iter().zip(valid_b.iter()))
        .filter(|(_, (va, vb))| **va && **vb)
        .map(|((xa, xb), _)| (*xa, *xb))
        .collect();
    if both_valid.len() < 10 {
        return f64::NAN;
    }
    let xs: Vec<f64> = both_valid.iter().map(|(x, _)| *x).collect();
    let ys: Vec<f64> = both_valid.iter().map(|(_, y)| *y).collect();
    pearson_corr(&xs, &ys)
}

/// Extract f64 values and validity mask from a Float64Array.
/// Returns (values, valid_mask) where values[i] = 0.0 if null.
fn extract_float64_array(arr: &dyn Array) -> Result<(Vec<f64>, Vec<bool>), String> {
    let fa = arr
        .as_any()
        .downcast_ref::<Float64Array>()
        .ok_or_else(|| "array is not Float64Array".to_string())?;
    let len = fa.len();
    let mut values = Vec::with_capacity(len);
    let mut valid = Vec::with_capacity(len);
    for i in 0..len {
        if fa.is_null(i) {
            values.push(f64::NAN);
            valid.push(false);
        } else {
            values.push(fa.value(i));
            valid.push(true);
        }
    }
    Ok((values, valid))
}

// ── Inner implementation (testable without PyO3) ──────────────────────────────

fn compute_pairwise_spearman_inner(
    arrays: &[Arc<dyn Array>],
    feature_ids: &[String],
) -> Result<RecordBatch, String> {
    let n = arrays.len();
    if n != feature_ids.len() {
        return Err(format!(
            "feature_vectors length {} != feature_ids length {}",
            n,
            feature_ids.len()
        ));
    }

    // Extract all feature vectors and validity masks up-front
    let extracted: Vec<(Vec<f64>, Vec<bool>)> = arrays
        .iter()
        .enumerate()
        .map(|(i, arr)| {
            extract_float64_array(arr.as_ref())
                .map_err(|e| format!("feature_vectors[{}]: {}", i, e))
        })
        .collect::<Result<Vec<_>, _>>()?;

    // Verify all vectors have the same length
    if n > 0 {
        let expected_len = extracted[0].0.len();
        for (i, (vals, _)) in extracted.iter().enumerate() {
            if vals.len() != expected_len {
                return Err(format!(
                    "feature_vectors[{}] has length {} but expected {}",
                    i,
                    vals.len(),
                    expected_len
                ));
            }
        }
    }

    // Build upper-triangle pair list
    let pairs: Vec<(usize, usize)> = (0..n)
        .flat_map(|i| ((i + 1)..n).map(move |j| (i, j)))
        .collect();

    // Compute pairwise correlations in parallel
    let results: Vec<(usize, usize, f64)> = pairs
        .par_iter()
        .map(|&(i, j)| {
            let (ref a_vals, ref a_valid) = extracted[i];
            let (ref b_vals, ref b_valid) = extracted[j];
            let corr = compute_pair_corr(a_vals, b_vals, a_valid, b_valid);
            (i, j, corr)
        })
        .collect();

    // Fill N×N matrix (diagonal = 1.0)
    let mut matrix = vec![0.0f64; n * n];
    for i in 0..n {
        matrix[i * n + i] = 1.0;
    }
    for (i, j, c) in results {
        matrix[i * n + j] = c;
        matrix[j * n + i] = c;
    }

    // Build output RecordBatch: first column = "feature_id" (Utf8),
    // then one Float64 column per feature ID.
    let mut fields = vec![Field::new("feature_id", DataType::Utf8, false)];
    for fid in feature_ids {
        fields.push(Field::new(fid.as_str(), DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));

    let mut feature_id_builder = StringBuilder::with_capacity(n, n * 20);
    for fid in feature_ids {
        feature_id_builder.append_value(fid);
    }

    let mut columns: Vec<ArrayRef> = vec![Arc::new(feature_id_builder.finish())];

    // Build one Float64 column per feature (each column = one row of the matrix)
    for col_j in 0..n {
        let mut builder = Float64Builder::with_capacity(n);
        for row_i in 0..n {
            let val = matrix[row_i * n + col_j];
            if val.is_nan() {
                builder.append_null();
            } else {
                builder.append_value(val);
            }
        }
        columns.push(Arc::new(builder.finish()));
    }

    RecordBatch::try_new(schema, columns).map_err(|e| e.to_string())
}

// ── PyO3-exposed function ─────────────────────────────────────────────────────

/// Compute pairwise Spearman correlation matrix for N features.
///
/// Each feature is provided as a pyarrow Float64Array of pre-ranked cross-sectional
/// values (length = sampled_dates × n_tickers). Null values are treated as missing
/// and excluded pairwise.
///
/// Args:
///     feature_vectors: List[pyarrow.Array] — one Float64Array per feature,
///                      each of the same length. Null values treated as missing.
///     feature_ids: List[str] — feature ID strings (same length as feature_vectors).
///
/// Returns:
///     pyarrow.RecordBatch — correlation matrix. First column is "feature_id" (Utf8),
///     remaining columns are feature IDs (Float64). Shape: N × (N + 1).
///     Diagonal values are 1.0. Matrix is symmetric.
#[pyfunction]
pub fn compute_pairwise_spearman(
    py: Python<'_>,
    feature_vectors: Vec<PyObject>,
    feature_ids: Vec<String>,
) -> PyResult<PyObject> {
    // Extract Arrow arrays from Python objects
    let arrays: Vec<Arc<dyn Array>> = feature_vectors
        .iter()
        .enumerate()
        .map(|(i, obj)| {
            let py_array = PyArray::extract_bound(&obj.bind(py)).map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "feature_vectors[{}]: {}",
                    i, e
                ))
            })?;
            let (array_ref, _field) = py_array.into_inner();
            Ok(array_ref)
        })
        .collect::<PyResult<Vec<_>>>()?;

    let result = compute_pairwise_spearman_inner(&arrays, &feature_ids)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    PyRecordBatch::new(result).to_pyarrow(py)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_float64_array(values: &[Option<f64>]) -> Arc<dyn Array> {
        let mut builder = Float64Builder::new();
        for v in values {
            match v {
                Some(x) => builder.append_value(*x),
                None => builder.append_null(),
            }
        }
        Arc::new(builder.finish())
    }

    fn make_ids(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("feat_{}", i)).collect()
    }

    // ── Helper: extract f64 value from RecordBatch cell ────────────────────

    fn get_value(batch: &RecordBatch, row: usize, col_name: &str) -> Option<f64> {
        let col = batch.column_by_name(col_name)?;
        let arr = col.as_any().downcast_ref::<Float64Array>()?;
        if arr.is_null(row) {
            None
        } else {
            Some(arr.value(row))
        }
    }

    // ── Tests ──────────────────────────────────────────────────────────────

    #[test]
    fn test_single_feature_returns_1x1_matrix_with_diagonal_1() {
        let arr = make_float64_array(&[Some(1.0), Some(2.0), Some(3.0), Some(4.0), Some(5.0)]);
        let arrays = vec![arr];
        let ids = make_ids(1);
        let batch = compute_pairwise_spearman_inner(&arrays, &ids).unwrap();
        assert_eq!(batch.num_rows(), 1);
        // The only cell (row=0, col="feat_0") should be 1.0
        let val = get_value(&batch, 0, "feat_0").expect("diagonal must not be null");
        assert!((val - 1.0).abs() < 1e-10, "diagonal expected 1.0, got {}", val);
    }

    #[test]
    fn test_identical_features_have_correlation_1() {
        let data: Vec<Option<f64>> = (0..20).map(|i| Some(i as f64)).collect();
        let arr1 = make_float64_array(&data);
        let arr2 = make_float64_array(&data);
        let arrays = vec![arr1, arr2];
        let ids = make_ids(2);
        let batch = compute_pairwise_spearman_inner(&arrays, &ids).unwrap();
        // Off-diagonal: feat_0 row, feat_1 column
        let corr = get_value(&batch, 0, "feat_1").expect("off-diagonal must not be null for identical vectors");
        assert!((corr - 1.0).abs() < 1e-10, "identical features: expected corr=1.0, got {}", corr);
    }

    #[test]
    fn test_matrix_is_symmetric() {
        let n = 5;
        let arrays: Vec<Arc<dyn Array>> = (0..n)
            .map(|k| {
                let data: Vec<Option<f64>> = (0..30)
                    .map(|i| Some(((i + k * 7) % 30) as f64))
                    .collect();
                make_float64_array(&data)
            })
            .collect();
        let ids = make_ids(n);
        let batch = compute_pairwise_spearman_inner(&arrays, &ids).unwrap();
        // Check symmetry: corr(i, j) == corr(j, i)
        for i in 0..n {
            for j in 0..n {
                let col_i = format!("feat_{}", i);
                let col_j = format!("feat_{}", j);
                let c_ij = get_value(&batch, i, &col_j);
                let c_ji = get_value(&batch, j, &col_i);
                match (c_ij, c_ji) {
                    (Some(a), Some(b)) => {
                        assert!((a - b).abs() < 1e-12, "symmetry broken at ({},{}): {} != {}", i, j, a, b);
                    }
                    (None, None) => {}
                    _ => panic!("symmetry broken at ({},{}): one null, other not", i, j),
                }
            }
        }
    }

    #[test]
    fn test_diagonal_is_one() {
        let n = 4;
        let arrays: Vec<Arc<dyn Array>> = (0..n)
            .map(|k| {
                let data: Vec<Option<f64>> = (0..20).map(|i| Some((i * k) as f64)).collect();
                make_float64_array(&data)
            })
            .collect();
        let ids = make_ids(n);
        let batch = compute_pairwise_spearman_inner(&arrays, &ids).unwrap();
        for i in 0..n {
            let col = format!("feat_{}", i);
            let val = get_value(&batch, i, &col).expect("diagonal must not be null");
            assert!((val - 1.0).abs() < 1e-10, "diagonal [{},{}] expected 1.0, got {}", i, i, val);
        }
    }

    #[test]
    fn test_negative_correlation_for_reversed_vectors() {
        // feat_0 = [0, 1, 2, ..., 29], feat_1 = [29, 28, ..., 0] → perfect negative corr
        let n = 30;
        let arr_asc: Vec<Option<f64>> = (0..n).map(|i| Some(i as f64)).collect();
        let arr_desc: Vec<Option<f64>> = (0..n).map(|i| Some((n - 1 - i) as f64)).collect();
        let arrays = vec![
            make_float64_array(&arr_asc),
            make_float64_array(&arr_desc),
        ];
        let ids = make_ids(2);
        let batch = compute_pairwise_spearman_inner(&arrays, &ids).unwrap();
        let corr = get_value(&batch, 0, "feat_1").expect("must be non-null");
        assert!(corr < 0.0, "expected negative correlation, got {}", corr);
    }

    #[test]
    fn test_features_with_some_nan_positions() {
        // 30 values, some null — result should still be finite
        let data_a: Vec<Option<f64>> = (0..30)
            .map(|i| if i % 5 == 0 { None } else { Some(i as f64) })
            .collect();
        let data_b: Vec<Option<f64>> = (0..30)
            .map(|i| if i % 7 == 0 { None } else { Some((i * 2) as f64) })
            .collect();
        let arrays = vec![
            make_float64_array(&data_a),
            make_float64_array(&data_b),
        ];
        let ids = make_ids(2);
        let batch = compute_pairwise_spearman_inner(&arrays, &ids).unwrap();
        // Correlation between feat_0 and feat_1 may be null (< 10 overlap) — just check no panic
        // With 30 elements, 6 nans in a and 5 nans in b, expected overlap >= 10
        if let Some(val) = get_value(&batch, 0, "feat_1") {
            assert!(val.is_finite(), "expected finite correlation, got {}", val);
        }
    }

    #[test]
    fn test_3_feature_known_case() {
        // Three features with known rank structure:
        // feat_0: ascending [0..14]
        // feat_1: same as feat_0 → corr(0,1)=1.0
        // feat_2: descending [14..0] → corr(0,2) = corr(1,2) = -1.0
        let n = 15;
        let asc: Vec<Option<f64>> = (0..n).map(|i| Some(i as f64)).collect();
        let desc: Vec<Option<f64>> = (0..n).map(|i| Some((n - 1 - i) as f64)).collect();
        let arrays = vec![
            make_float64_array(&asc),
            make_float64_array(&asc.clone()),
            make_float64_array(&desc),
        ];
        let ids: Vec<String> = vec!["f0".to_string(), "f1".to_string(), "f2".to_string()];
        let batch = compute_pairwise_spearman_inner(&arrays, &ids).unwrap();

        // corr(f0, f1) == 1.0
        let c01 = get_value(&batch, 0, "f1").unwrap();
        assert!((c01 - 1.0).abs() < 1e-10, "c01={}", c01);

        // corr(f0, f2) < 0
        let c02 = get_value(&batch, 0, "f2").unwrap();
        assert!(c02 < 0.0, "c02={}", c02);

        // diagonal values == 1.0
        assert!((get_value(&batch, 0, "f0").unwrap() - 1.0).abs() < 1e-10);
        assert!((get_value(&batch, 1, "f1").unwrap() - 1.0).abs() < 1e-10);
        assert!((get_value(&batch, 2, "f2").unwrap() - 1.0).abs() < 1e-10);

        // symmetry: c(0,1) == c(1,0)
        let c10 = get_value(&batch, 1, "f0").unwrap();
        assert!((c01 - c10).abs() < 1e-12, "symmetry broken: {} vs {}", c01, c10);
    }

    #[test]
    fn test_mismatched_lengths_returns_error() {
        let arr_short: Vec<Option<f64>> = vec![Some(1.0), Some(2.0)];
        let arr_long: Vec<Option<f64>> = vec![Some(1.0), Some(2.0), Some(3.0)];
        let arrays = vec![
            make_float64_array(&arr_short),
            make_float64_array(&arr_long),
        ];
        let ids = make_ids(2);
        let result = compute_pairwise_spearman_inner(&arrays, &ids);
        assert!(result.is_err(), "mismatched lengths should return an error");
    }
}
