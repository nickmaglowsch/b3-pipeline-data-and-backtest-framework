use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatch;

mod adjustments;
mod parser;
mod pivot;
mod schema;

pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Parse a single COTAHIST ZIP file and return a PyArrow RecordBatch.
///
/// Args:
///     path: Absolute path to a COTAHIST_A<YYYY>.ZIP file.
///
/// Returns:
///     pyarrow.RecordBatch with columns:
///     date, ticker, isin_code, open, high, low, close, volume, quotation_factor
///
/// Raises:
///     RuntimeError: If the file cannot be opened or parsed.
#[pyfunction]
fn parse_zip(py: Python<'_>, path: String) -> PyResult<PyObject> {
    let batch = parser::parse_zip_file(&path)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    PyRecordBatch::new(batch).to_pyarrow(py)
}

/// Parse multiple COTAHIST ZIP files in parallel and return a single PyArrow RecordBatch.
///
/// Uses rayon for inter-file parallelism (each ZIP processed on a separate thread).
/// Individual file errors are logged to stderr; the call never raises even if some
/// paths are invalid.
///
/// Args:
///     paths: List of absolute paths to COTAHIST ZIP files.
///
/// Returns:
///     pyarrow.RecordBatch concatenating all parsed records. Returns an empty
///     RecordBatch (0 rows, correct schema) if the list is empty or all files fail.
#[pyfunction]
fn parse_multiple_zips(py: Python<'_>, paths: Vec<String>) -> PyResult<PyObject> {
    let batch = parser::parse_multiple_zip_files(&paths);
    PyRecordBatch::new(batch).to_pyarrow(py)
}

/// Detect stock splits from consecutive price jumps.
///
/// Args:
///     prices_batch: pyarrow.RecordBatch with columns [isin_code, date (Date32), close,
///                   optionally quotation_factor]
///     existing_batch: pyarrow.RecordBatch with columns [isin_code, ex_date (Date32)]
///                     — already-known split events to suppress
///     detect_nonstandard: if True, non-standard ratios are returned with
///                         source="DETECTED_NONSTANDARD"
///     threshold_high: price ratio above this is a candidate split (default 1.8)
///     threshold_low: price ratio below this is a candidate split (default 0.55)
///
/// Returns:
///     (result_batch, log_messages) where result_batch is a pyarrow.RecordBatch with
///     columns [isin_code, ex_date, action_type, factor, source]
#[pyfunction]
fn detect_splits(
    py: Python<'_>,
    prices_batch: PyObject,
    existing_batch: PyObject,
    detect_nonstandard: bool,
    threshold_high: f64,
    threshold_low: f64,
) -> PyResult<(PyObject, Vec<String>)> {
    adjustments::detect_splits_from_prices_rs(
        py,
        prices_batch,
        existing_batch,
        detect_nonstandard,
        threshold_high,
        threshold_low,
    )
}

/// Compute split adjustment factors and return adjusted OHLC columns.
///
/// Args:
///     prices_batch: pyarrow.RecordBatch with all original price columns including
///                   [isin_code, date (Date32), open, high, low, close, ...]
///     splits_batch: pyarrow.RecordBatch with [isin_code, ex_date (Date32), split_factor]
///
/// Returns:
///     pyarrow.RecordBatch: same as prices_batch + 4 new columns:
///     [split_adj_open, split_adj_high, split_adj_low, split_adj_close]
#[pyfunction]
fn compute_split_adjustment(
    py: Python<'_>,
    prices_batch: PyObject,
    splits_batch: PyObject,
) -> PyResult<PyObject> {
    adjustments::compute_split_adjustment_factors_rs(py, prices_batch, splits_batch)
}

/// Pivot long-format price data into three wide DataFrames and forward-fill close prices.
///
/// Args:
///     long_batch: pyarrow.RecordBatch with columns
///                 [date (Utf8 "YYYY-MM-DD"), ticker, close, adj_close, fin_volume]
///
/// Returns:
///     (adj_close_batch, close_px_batch, fin_vol_batch) — three pyarrow.RecordBatches
///     in wide format. Each batch has a "date" column (Utf8) followed by one Float64
///     column per ticker, sorted lexicographically. adj_close and close_px are
///     forward-filled; fin_vol is NOT forward-filled.
#[pyfunction]
fn pivot_and_ffill(
    py: Python<'_>,
    long_batch: PyObject,
) -> PyResult<(PyObject, PyObject, PyObject)> {
    pivot::pivot_and_ffill_rs(py, long_batch)
}

#[pymodule]
fn cotahist_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", VERSION)?;
    m.add_function(wrap_pyfunction!(parse_zip, m)?)?;
    m.add_function(wrap_pyfunction!(parse_multiple_zips, m)?)?;
    m.add_function(wrap_pyfunction!(detect_splits, m)?)?;
    m.add_function(wrap_pyfunction!(compute_split_adjustment, m)?)?;
    m.add_function(wrap_pyfunction!(pivot_and_ffill, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version_string() {
        assert!(!VERSION.is_empty(), "VERSION must not be empty");
    }

    #[test]
    fn test_module_compiles() {
        // Trivial compilation smoke test — if this test module compiles and runs,
        // all `use` imports and `mod` declarations are wired up correctly.
        assert_eq!(1 + 1, 2);
    }
}
