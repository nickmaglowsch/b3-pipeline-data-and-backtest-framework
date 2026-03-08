# Test-Writer Memory — b3-data-pipeline

## Test Framework & Command
- Framework: pytest
- Test command: `python -m pytest tests/ -v`
- Run single file: `python -m pytest tests/test_discovery_core.py -v`
- Python: 3.9.6 (venv at `.venv/`)

## Test File Conventions
- Location: `tests/` at project root (flat, no subdirectories)
- Naming: `test_<module_name>.py`
- Style: class-free plain functions, `_make_df()` / `_make_wide()` module-level helpers
- No pytest fixtures needed for simple cases; use `pytest.fixture` only for complex shared setup
- `tmp_path` built-in fixture used for tests that write to disk (FeatureStore, SQLite, Parquet)

## Import Pattern
- Production code is importable from project root: `from research.discovery.operators import op_rank`
- No special conftest.py setup required

## Mocking Patterns
- External HTTP: `unittest.mock.patch` or `pytest-mock`'s `mocker` fixture
- DB: use `tmp_path` with real SQLite; no DB mocking
- FeatureStore: instantiate with `FeatureStore(store_dir=tmp_path)` for isolation

## Numeric Test Patterns
- Snapshot constants: module-level `_CONSTANT = value`, assert with `abs(actual - expected) < 1e-5`
- Float comparisons: `abs(val - expected) < 1e-8` for deterministic formulas
- Stochastic tests: use fixed `seed=` and loose tolerances (e.g., `< 0.1` for mean IC ~ 0)

## Key Gotchas
- `compute_turnover` can return slightly > 1.0 (not a bug — vectorized corr with fillna(0) causes
  small negative autocorrelations on pure random data). Do not assert `<= 1.0`.
- `op_rank` with a single non-NaN stock per row returns 1.0 (pandas pct=True, 1 obs → rank 1/1).
  The task spec claimed 0.5 but actual behavior is 1.0.
- `filter_nan_and_variance` uses `config.MAX_NAN_RATE=0.30` and `MIN_VARIANCE_DATES_FRAC=0.90`.
  An all-NaN feature has nan_rate=1.0 > 0.30, so it gets removed.
- `compute_evaluation_summary` includes `mean_ic_5y` key (not listed in task spec keys) — do not
  assert exact key equality, use `issubset` instead.

## Discovery Module Locations
- `research/discovery/operators.py` — op_rank, op_zscore, op_delta, op_ratio_to_mean, op_ratio, op_product
- `research/discovery/evaluator.py` — compute_ic_series_fast, compute_turnover, compute_decay, compute_evaluation_summary, evaluate_feature
- `research/discovery/pruning.py` — filter_nan_and_variance, filter_by_ic, deduplicate_by_correlation, enforce_cap
- `research/discovery/store.py` — FeatureStore class (Parquet + JSON registry)
- `research/discovery/config.py` — MIN_IC_THRESHOLD=0.005, MAX_CORRELATION=0.90, MAX_FEATURES=500
