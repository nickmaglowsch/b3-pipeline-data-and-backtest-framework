# B3 Data Pipeline — Task Implementer Memory

## Project Structure
- Python 3.9, venv at `.venv/`
- Test runner: `python -m pytest tests/ -x -q`
- 282 Python tests passing as of task-06; 80 Rust tests passing

## Passing Individual pyarrow Arrays to Rust (task-06 pattern)
- Use `PyArray::extract_bound(&obj.bind(py))` from `pyo3_arrow::PyArray` (has `FromPyObject` impl)
- `PyArray::into_inner()` returns `(ArrayRef, FieldRef)`
- Import: `use pyo3_arrow::PyArray;`
- Then `downcast_ref::<Float64Array>()` on the `ArrayRef` to get values

## Key Discovery Module Files
- `research/discovery/config.py` — constants, now includes `MAX_WORKERS`
- `research/discovery/store.py` — `FeatureStore`: registry (JSON) + Parquet files; `save_feature`, `save_evaluation`, `save_registry` all in main process only
- `research/discovery/evaluator.py` — `evaluate_all_features`, `evaluate_feature`, IC computation; `_RUST_EVAL` flag for Rust path
- `research/discovery/generator.py` — Level 0/1/2 feature generation
- `research/discovery/operators.py` — `UNARY_OPS`, `BINARY_OPS` dicts + individual op functions
- `research/discovery/_worker.py` — (created in task-02) top-level picklable worker functions for ProcessPoolExecutor

## Rust Crate (`b3_pipeline_rs/`)
- Module name (Python import): `cotahist_rs`
- Build: `bash -c 'export PATH="/usr/bin:$HOME/.cargo/bin:$PATH" && .venv/bin/maturin develop --release --manifest-path b3_pipeline_rs/Cargo.toml'`
  (must export PATH in same shell; maturin needs cc from /usr/bin AND rustc from cargo bin)
- Rust tests: `bash -c 'export PATH="/usr/bin:$HOME/.cargo/bin:$PATH" && ~/.cargo/bin/cargo test --manifest-path b3_pipeline_rs/Cargo.toml'`
- Modules: `adjustments.rs`, `parser.rs`, `pivot.rs`, `schema.rs`, `feature_eval.rs`, `util.rs`, `cross_section.rs`, `correlation.rs`
- Arrow pattern: extract via `.column_by_name("col")?.as_any().downcast_ref::<StringArray>()?`
- Rayon: `use rayon::prelude::*; vec.par_iter().map(...).collect()`
- Shared util module: `util.rs` declared as `pub(crate) mod util;` in lib.rs, imported via `use crate::util::...`
- Arrow types (DataType, Field) used only in `#[cfg(test)]`: import inside the test module to avoid unused-import warnings

## Rust–Python Arrow Interop (task-04 pattern)
- When passing a pandas DataFrame to Rust as RecordBatch:
  1. Call `df.reset_index()` then rename the first column to `"date"` explicitly (index name varies)
  2. Convert date to string: `df["date"] = df["date"].astype(str)` (Rust expects Utf8)
  3. Use `pa.RecordBatch.from_pandas(df, preserve_index=False)` to avoid double-index
- When returning IC series from Rust, assign `result_df.index = feature_wide.index` to
  preserve DatetimeIndex frequency metadata (otherwise `pd.testing.assert_series_equal` fails)

## Parallelism Pattern (task-02)
- Workers load Parquet files directly (small); never receive `data` dict (multi-GB)
- Shared inputs (universe_mask, fwd_rank DataFrames) serialized to `store_dir/tmp_eval/` or `tmp_gen_l1/`, `tmp_gen_l2/` before pool starts; cleaned up after
- Main process handles all store writes (registry JSON never written from workers)
- Workers return `dict | None`; `None` on exception (exception logged to stderr)
- `MAX_WORKERS = min(os.cpu_count() or 4, 8)`

## FeatureStore Registry
- `store._registry` is a JSON dict in memory; `store.save_registry()` writes it
- `store.save_feature(id, df, metadata)` updates `_registry` AND writes Parquet
- Workers MUST NOT call `save_registry()` or `save_feature()` — only return dicts
