// Rust implementation of load_b3_data pivot + forward-fill logic.
//
// pivot_and_ffill_rs: Task 03
//   Takes a long-format RecordBatch [date, ticker, close, adj_close, fin_volume]
//   and returns three wide RecordBatches:
//     - adj_close (ffilled)
//     - close_px (ffilled)
//     - fin_vol (NOT ffilled)

use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Float64Builder, StringArray, StringBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use pyo3::prelude::*;
use pyo3_arrow::PyRecordBatch;


pub fn pivot_and_ffill_rs(
    py: Python<'_>,
    long_batch: PyObject,
) -> PyResult<(PyObject, PyObject, PyObject)> {
    let batch: RecordBatch = PyRecordBatch::extract_bound(&long_batch.bind(py))
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("long_batch: {}", e)))?
        .into_inner();

    let n_rows = batch.num_rows();

    if n_rows == 0 {
        // Return three empty RecordBatches (only a "date" column, no ticker columns)
        let empty_schema = Arc::new(Schema::new(vec![Field::new("date", DataType::Utf8, false)]));
        let empty_dates: ArrayRef = Arc::new(StringBuilder::new().finish());
        let empty_batch = RecordBatch::try_new(empty_schema.clone(), vec![empty_dates])
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{}", e)))?;
        let b1 = PyRecordBatch::new(empty_batch.clone()).to_pyarrow(py)?;
        let b2 = PyRecordBatch::new(empty_batch.clone()).to_pyarrow(py)?;
        let b3 = PyRecordBatch::new(empty_batch).to_pyarrow(py)?;
        return Ok((b1, b2, b3));
    }

    // Extract columns
    let date_arr = batch
        .column_by_name("date")
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing date"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("date not StringArray"))?;

    let ticker_arr = batch
        .column_by_name("ticker")
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing ticker"))?
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("ticker not StringArray"))?;

    let close_arr = batch
        .column_by_name("close")
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing close"))?
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("close not Float64Array"))?;

    let adj_close_arr = batch
        .column_by_name("adj_close")
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing adj_close"))?
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("adj_close not Float64Array"))?;

    let fin_vol_arr = batch
        .column_by_name("fin_volume")
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("missing fin_volume"))?
        .as_any()
        .downcast_ref::<arrow::array::Float64Array>()
        .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("fin_volume not Float64Array"))?;

    // Build sorted unique axes
    let mut unique_dates_set: BTreeSet<String> = BTreeSet::new();
    let mut unique_tickers_set: BTreeSet<String> = BTreeSet::new();
    for i in 0..n_rows {
        if !date_arr.is_null(i) {
            unique_dates_set.insert(date_arr.value(i).to_string());
        }
        if !ticker_arr.is_null(i) {
            unique_tickers_set.insert(ticker_arr.value(i).to_string());
        }
    }

    let unique_dates: Vec<String> = unique_dates_set.into_iter().collect();
    let unique_tickers: Vec<String> = unique_tickers_set.into_iter().collect();
    let n_dates = unique_dates.len();
    let n_tickers = unique_tickers.len();

    // Build index lookup maps
    let date_idx: HashMap<&str, usize> = unique_dates
        .iter()
        .enumerate()
        .map(|(i, d)| (d.as_str(), i))
        .collect();
    let ticker_idx: HashMap<&str, usize> = unique_tickers
        .iter()
        .enumerate()
        .map(|(i, t)| (t.as_str(), i))
        .collect();

    // Allocate flat matrices (row-major: matrix[date_i * n_tickers + ticker_j])
    // f64::NAN represents missing values
    let total = n_dates * n_tickers;
    let mut adj_close_mat = vec![f64::NAN; total];
    let mut close_mat = vec![f64::NAN; total];
    let mut fin_vol_mat = vec![f64::NAN; total];

    // Fill matrices in a single pass
    for i in 0..n_rows {
        if date_arr.is_null(i) || ticker_arr.is_null(i) {
            continue;
        }
        let d = date_arr.value(i);
        let t = ticker_arr.value(i);
        let di = date_idx[d];
        let ti = ticker_idx[t];
        let offset = di * n_tickers + ti;

        if !adj_close_arr.is_null(i) {
            adj_close_mat[offset] = adj_close_arr.value(i);
        }
        if !close_arr.is_null(i) {
            close_mat[offset] = close_arr.value(i);
        }
        if !fin_vol_arr.is_null(i) {
            fin_vol_mat[offset] = fin_vol_arr.value(i);
        }
    }

    // Forward-fill adj_close and close (NOT fin_vol) per ticker column (serial; O(n_dates) per ticker)
    for j in 0..n_tickers {
        ffill_column(&mut adj_close_mat, n_dates, n_tickers, j);
        ffill_column(&mut close_mat, n_dates, n_tickers, j);
        // fin_vol_mat is intentionally NOT ffilled
    }

    // Build output RecordBatches
    let adj_close_batch = build_wide_batch(&unique_dates, &unique_tickers, &adj_close_mat, n_dates, n_tickers)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let close_batch = build_wide_batch(&unique_dates, &unique_tickers, &close_mat, n_dates, n_tickers)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;
    let fin_vol_batch = build_wide_batch(&unique_dates, &unique_tickers, &fin_vol_mat, n_dates, n_tickers)
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))?;

    let b1 = PyRecordBatch::new(adj_close_batch).to_pyarrow(py)?;
    let b2 = PyRecordBatch::new(close_batch).to_pyarrow(py)?;
    let b3 = PyRecordBatch::new(fin_vol_batch).to_pyarrow(py)?;
    Ok((b1, b2, b3))
}

fn ffill_column(mat: &mut [f64], n_dates: usize, n_tickers: usize, j: usize) {
    let mut last = f64::NAN;
    for i in 0..n_dates {
        let v = &mut mat[i * n_tickers + j];
        if v.is_nan() {
            *v = last; // last is NaN on first iteration — stays NaN
        } else {
            last = *v;
        }
    }
}

fn build_wide_batch(
    unique_dates: &[String],
    unique_tickers: &[String],
    matrix: &[f64],
    n_dates: usize,
    n_tickers: usize,
) -> Result<RecordBatch, String> {
    let mut fields = vec![Field::new("date", DataType::Utf8, false)];
    for t in unique_tickers {
        fields.push(Field::new(t, DataType::Float64, true));
    }
    let schema = Arc::new(Schema::new(fields));

    // date column
    let mut date_builder = StringBuilder::with_capacity(n_dates, n_dates * 10);
    for d in unique_dates {
        date_builder.append_value(d);
    }
    let date_col: ArrayRef = Arc::new(date_builder.finish());

    // ticker columns
    let mut columns: Vec<ArrayRef> = vec![date_col];
    for j in 0..n_tickers {
        let mut builder = Float64Builder::with_capacity(n_dates);
        for i in 0..n_dates {
            let v = matrix[i * n_tickers + j];
            if v.is_nan() {
                builder.append_null();
            } else {
                builder.append_value(v);
            }
        }
        columns.push(Arc::new(builder.finish()));
    }

    RecordBatch::try_new(schema, columns).map_err(|e| e.to_string())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_batch(rows: &[(&str, &str, f64, f64, f64)]) -> RecordBatch {
        // rows: (date, ticker, close, adj_close, fin_volume)
        let mut date_b = StringBuilder::new();
        let mut ticker_b = StringBuilder::new();
        let mut close_b = Float64Builder::new();
        let mut adj_b = Float64Builder::new();
        let mut vol_b = Float64Builder::new();

        for &(d, t, c, a, v) in rows {
            date_b.append_value(d);
            ticker_b.append_value(t);
            close_b.append_value(c);
            adj_b.append_value(a);
            vol_b.append_value(v);
        }

        let schema = Arc::new(Schema::new(vec![
            Field::new("date", DataType::Utf8, false),
            Field::new("ticker", DataType::Utf8, false),
            Field::new("close", DataType::Float64, false),
            Field::new("adj_close", DataType::Float64, false),
            Field::new("fin_volume", DataType::Float64, false),
        ]));

        let cols: Vec<ArrayRef> = vec![
            Arc::new(date_b.finish()),
            Arc::new(ticker_b.finish()),
            Arc::new(close_b.finish()),
            Arc::new(adj_b.finish()),
            Arc::new(vol_b.finish()),
        ];

        RecordBatch::try_new(schema, cols).unwrap()
    }

    fn get_f64(batch: &RecordBatch, col: &str, row: usize) -> Option<f64> {
        let arr = batch.column_by_name(col)?;
        let fa = arr.as_any().downcast_ref::<arrow::array::Float64Array>()?;
        if fa.is_null(row) {
            None
        } else {
            Some(fa.value(row))
        }
    }

    fn get_str(batch: &RecordBatch, col: &str, row: usize) -> Option<String> {
        let arr = batch.column_by_name(col)?;
        let sa = arr.as_any().downcast_ref::<StringArray>()?;
        if sa.is_null(row) {
            None
        } else {
            Some(sa.value(row).to_string())
        }
    }

    #[test]
    fn test_single_row_pivot() {
        let rows = vec![("2020-01-02", "PETR3", 100.0, 95.0, 1000.0)];
        let batch = make_batch(&rows);
        let (adj, _close, _vol) = build_matrices_for_test(&batch);
        assert_eq!(adj.num_rows(), 1);
        // Only "date" and "PETR3" columns
        assert!(adj.column_by_name("PETR3").is_some());
        let v = get_f64(&adj, "PETR3", 0).unwrap();
        assert!((v - 95.0).abs() < 1e-9);
        let d = get_str(&adj, "date", 0).unwrap();
        assert_eq!(d, "2020-01-02");
    }

    #[test]
    fn test_ffill_fills_gap() {
        // PETR3 has value on day 0 and day 2 but not day 1
        // We need a second ticker so day 1 appears
        let rows = vec![
            ("2020-01-02", "PETR3", 100.0, 95.0, 1000.0),
            ("2020-01-03", "VALE3", 200.0, 190.0, 2000.0),
            ("2020-01-04", "PETR3", 102.0, 97.0, 1050.0),
            ("2020-01-04", "VALE3", 202.0, 192.0, 2050.0),
        ];
        let batch = make_batch(&rows);
        let (adj, _, _) = build_matrices_for_test(&batch);

        // Date order: 2020-01-02, 2020-01-03, 2020-01-04
        // PETR3 on 2020-01-02: 95.0; 2020-01-03: ffilled to 95.0; 2020-01-04: 97.0
        let petr3_row1 = get_f64(&adj, "PETR3", 1); // 2020-01-03
        assert!(petr3_row1.is_some());
        assert!((petr3_row1.unwrap() - 95.0).abs() < 1e-9);
    }

    #[test]
    fn test_fin_vol_not_ffilled() {
        let rows = vec![
            ("2020-01-02", "PETR3", 100.0, 95.0, 1000.0),
            ("2020-01-03", "VALE3", 200.0, 190.0, 2000.0),
        ];
        let batch = make_batch(&rows);
        let (_, _, vol) = build_matrices_for_test(&batch);

        // PETR3 has no row on 2020-01-03; vol should be null (not ffilled)
        let petr3_vol = get_f64(&vol, "PETR3", 1); // 2020-01-03
        assert!(petr3_vol.is_none(), "fin_vol must not be ffilled");
    }

    #[test]
    fn test_two_tickers_independent_ffill() {
        let rows = vec![
            ("2020-01-02", "AAA3", 10.0, 9.0, 100.0),
            ("2020-01-02", "BBB3", 20.0, 18.0, 200.0),
            ("2020-01-03", "BBB3", 21.0, 19.0, 210.0),
            // AAA3 has a gap on 2020-01-03; BBB3 does not
        ];
        let batch = make_batch(&rows);
        let (adj, _, _) = build_matrices_for_test(&batch);

        // AAA3 day 1 (2020-01-03) should be ffilled to 9.0
        let aaa_d1 = get_f64(&adj, "AAA3", 1).unwrap();
        assert!((aaa_d1 - 9.0).abs() < 1e-9);

        // BBB3 day 1 should be its actual value 19.0 (not affected by AAA3 ffill)
        let bbb_d1 = get_f64(&adj, "BBB3", 1).unwrap();
        assert!((bbb_d1 - 19.0).abs() < 1e-9);
    }

    #[test]
    fn test_sorted_dates_and_tickers() {
        // Input in unsorted order
        let rows = vec![
            ("2020-01-04", "ZZZ3", 10.0, 9.0, 100.0),
            ("2020-01-02", "AAA3", 20.0, 18.0, 200.0),
            ("2020-01-03", "MMM3", 15.0, 14.0, 150.0),
        ];
        let batch = make_batch(&rows);
        let (adj, _, _) = build_matrices_for_test(&batch);

        // Dates should be sorted ascending
        let d0 = get_str(&adj, "date", 0).unwrap();
        let d1 = get_str(&adj, "date", 1).unwrap();
        let d2 = get_str(&adj, "date", 2).unwrap();
        assert!(d0 <= d1 && d1 <= d2);

        // Tickers (after "date") should be lexicographically sorted
        let schema = adj.schema();
        let col_names: Vec<&str> = schema.fields().iter().map(|f| f.name().as_str()).collect();
        // col_names[0] = "date", rest are tickers sorted
        let tickers: Vec<&str> = col_names[1..].to_vec();
        let mut sorted = tickers.clone();
        sorted.sort();
        assert_eq!(tickers, sorted);
    }

    #[test]
    fn test_empty_input_returns_empty_batches() {
        let rows: Vec<(&str, &str, f64, f64, f64)> = vec![];
        let batch = make_batch(&rows);
        let (adj, close, vol) = build_matrices_for_test(&batch);

        assert_eq!(adj.num_rows(), 0);
        assert_eq!(close.num_rows(), 0);
        assert_eq!(vol.num_rows(), 0);
    }

    /// Helper that runs the Rust pivot logic on a RecordBatch using the internal functions
    /// (without going through PyO3).
    fn build_matrices_for_test(
        batch: &RecordBatch,
    ) -> (RecordBatch, RecordBatch, RecordBatch) {
        let n_rows = batch.num_rows();

        if n_rows == 0 {
            let schema = Arc::new(Schema::new(vec![Field::new("date", DataType::Utf8, false)]));
            let empty: ArrayRef = Arc::new(StringBuilder::new().finish());
            let empty_batch = RecordBatch::try_new(schema, vec![empty]).unwrap();
            return (empty_batch.clone(), empty_batch.clone(), empty_batch);
        }

        let date_arr = batch.column_by_name("date").unwrap().as_any()
            .downcast_ref::<StringArray>().unwrap();
        let ticker_arr = batch.column_by_name("ticker").unwrap().as_any()
            .downcast_ref::<StringArray>().unwrap();
        let close_arr = batch.column_by_name("close").unwrap().as_any()
            .downcast_ref::<arrow::array::Float64Array>().unwrap();
        let adj_arr = batch.column_by_name("adj_close").unwrap().as_any()
            .downcast_ref::<arrow::array::Float64Array>().unwrap();
        let vol_arr = batch.column_by_name("fin_volume").unwrap().as_any()
            .downcast_ref::<arrow::array::Float64Array>().unwrap();

        let mut unique_dates_set: BTreeSet<String> = BTreeSet::new();
        let mut unique_tickers_set: BTreeSet<String> = BTreeSet::new();
        for i in 0..n_rows {
            unique_dates_set.insert(date_arr.value(i).to_string());
            unique_tickers_set.insert(ticker_arr.value(i).to_string());
        }
        let unique_dates: Vec<String> = unique_dates_set.into_iter().collect();
        let unique_tickers: Vec<String> = unique_tickers_set.into_iter().collect();
        let n_dates = unique_dates.len();
        let n_tickers = unique_tickers.len();

        let date_idx: HashMap<&str, usize> = unique_dates.iter().enumerate().map(|(i, d)| (d.as_str(), i)).collect();
        let ticker_idx: HashMap<&str, usize> = unique_tickers.iter().enumerate().map(|(i, t)| (t.as_str(), i)).collect();

        let total = n_dates * n_tickers;
        let mut adj_mat = vec![f64::NAN; total];
        let mut close_mat = vec![f64::NAN; total];
        let mut vol_mat = vec![f64::NAN; total];

        for i in 0..n_rows {
            let di = date_idx[date_arr.value(i)];
            let ti = ticker_idx[ticker_arr.value(i)];
            let off = di * n_tickers + ti;
            adj_mat[off] = adj_arr.value(i);
            close_mat[off] = close_arr.value(i);
            vol_mat[off] = vol_arr.value(i);
        }

        for j in 0..n_tickers {
            ffill_column(&mut adj_mat, n_dates, n_tickers, j);
            ffill_column(&mut close_mat, n_dates, n_tickers, j);
        }

        let a = build_wide_batch(&unique_dates, &unique_tickers, &adj_mat, n_dates, n_tickers).unwrap();
        let c = build_wide_batch(&unique_dates, &unique_tickers, &close_mat, n_dates, n_tickers).unwrap();
        let v = build_wide_batch(&unique_dates, &unique_tickers, &vol_mat, n_dates, n_tickers).unwrap();
        (a, c, v)
    }
}
