# Code Reviewer Memory -- B3 Data Pipeline

## Project Structure
- SQLite DB: `b3_market_data.sqlite` at project root
- Pipeline: `b3_pipeline/` with parser.py, storage.py, adjustments.py, b3_corporate_actions.py, config.py, main.py
- Tests: `tests/` with test_parser_fatcot.py, test_split_detection.py, test_label_handling.py, test_failure_tracking.py
- Backtests: `backtests/` with `core/data.py` for data loading, `core/metrics.py`, `core/plotting.py`, `core/simulation.py`
- New core modules (Feb 2026): `core/strategy_returns.py`, `core/portfolio_opt.py`, `core/param_scanner.py`
- Strategy plugins (Feb 2026): `backtests/strategies/` (13 classes), `backtests/core/strategy_base.py`, `strategy_registry.py`, `shared_data.py`
- Streamlit UI (Feb 2026): `ui/` with `app.py`, `pages/`, `components/`, `services/`
- Research module: `research/` -- feature importance discovery (added Feb 2026)
- Tasks/PRDs: `tasks/` directory

## Key Conventions
- sys.path manipulation for imports (both backtests and research use this)
- ATR true_range: `pd.concat([tr1,tr2,tr3]).groupby(level=0).max()` -- established pattern
- Volatility: mostly uses `ret.shift(1).rolling(n).std()` BUT LowVol uses `ret.rolling(n).std()` (no shift) -- inconsistency from original compare_all.py
- Dark theme plots: PALETTE dict with bg=#0D1117, panel=#161B22, etc. + fmt_ax() helper
- Volume from COTAHIST has 2 implied decimals: true BRL volume = volume / 100
- Simulation: `run_simulation()` requires `monthly_sales_exemption=20_000` param for R$20K tax exemption

## Common Patterns to Check
- Lookahead bias: features must not use future data; targets use shift(-N)
- Time-series split must be by sorted unique dates, no shuffling
- Universe filtering is rolling (per stock per day), not retroactive
- adj_close for returns, split_adj_* for ATR/technical indicators
- Cross-sectional operations use axis=1 (across tickers per date)
- Private symbol imports: `_equal_weights` and `_REGIME_EQUITY_BUDGET` are imported by 4+ scripts despite underscore prefix

## Known Issues / Recurring Patterns
- min_periods often set lower than window size -- check if intentional
- pickle caching with only time-based invalidation (no config hash)
- gini/gain importance column unification logic duplicated between modeling and viz
- `compute_portfolio_returns()` and `compute_regime_portfolio()` duplicated in 5 files -- should be extracted
- `vol_60d = ret.rolling(5).std()` comment says "weekly periods" but data is monthly -- misleading
- `build_strategy_returns()` type annotation says 2-tuple but returns 3-tuple
- **CRITICAL (Mar 2026)**: fatcot normalization + B3 API split double-adjustment not mitigated. No FATCOT_REDUNDANT filtering exists.
- `COMMON_SPLIT_RATIOS` in config.py is dead code -- `detect_splits_from_prices` uses its own local `_common_ratios` list
- `retry_failed_companies()` in main.py only processes stockDividends, not cashDividends
- parser.py close_price reads line[108:121] which config.py says is preco_medio (average), not preco_ultimo_negocio (close at 121:134) -- pre-existing, needs investigation
- `fetch_company_data` failure tracking only covers RequestException, not JSON/parse errors

## Streamlit UI Review Notes (Feb 2026)
- `@st.cache_resource` for shared data dict is fragile -- strategies must `.copy()` before mutation
- `monthly_sales_exemption` not passed in backtest_service.py `run_simulation()` call
- PALETTE dict duplicated in `ui/pages/4_research.py` (should import from charts.py)
- Strategy registry silently swallows import/instantiation errors -- hard to debug
- `result_store.py` has module-level `RESULTS_DIR.mkdir()` side effect
- `log_stream.py` uses blocking while-loop; consider `st.fragment` for partial reruns
- Streamlit version: 1.50.0. `st.status` state values: "running", "complete", "error"

## Feature Discovery Engine Review Notes (Mar 2026)
- Module: `research/discovery/` (12 files), modification to `research/targets.py`
- Feature store: `research/feature_store/` with registry.json + features/*.parquet
- **FIXED**: `evaluate_all_features()` now returns `store.get_all_evaluations()` -- incremental re-runs work correctly
- **FIXED**: Pruning summary counts now distinguish `after_nan_filter` vs `after_ic_filter`
- **FIXED**: EWM variants and Mean Reversion signals implemented in `base_signals.py`
- **FIXED**: `ratio_to_mean` integrated as Level 2 operator; `--incremental` flag wired
- **FIXED**: IC timeseries persisted as consolidated Parquet; `mean_ic_5y` metric added
- **FIXED**: IC timeseries line chart and correlation heatmap plots added
- REMAINING: `compute_evaluation_summary()` crashes on empty IC series (no guard for `ic.index.max()` on empty)
- REMAINING: IC timeseries batch write only at end of evaluation -- data loss on interruption + incremental resume
- REMAINING: `compute_turnover()` uses fillna(0) on ranks -- introduces bias from stable NaN patterns
- REMAINING: `op_diff` and `compute_forward_returns_wide()` are dead code
- REMAINING: Config module-level `mkdir()` side effects (same pattern as `result_store.py` in UI)
- REMAINING: IC decay plot still doesn't show actual decay bars (shows IC_IR bar instead)

## Discovery UI Review Notes (Mar 2026)
- Files: `ui/services/discovery_service.py`, `ui/components/discovery_charts.py`, `ui/pages/5_discovery.py`, modified `ui/app.py`
- Catalog JSON keys: {generated_at, pipeline_version, evaluation_date_range, primary_horizon, total_generated, total_after_pruning, features, category_summary}
- Each feature has BOTH `feature["decay"]` (feature-level) and `feature["metrics"][horizon]["decay"]` (per-horizon)
- `pct_positive_ic` stored as fraction (0.55 = 55%), multiplied by 100 in table display
- BUG: `plot_pruning_funnel` regex for "After Correlation Dedup" fails -- `[:\s]` can't match "Dedup" word
- `_catalog_mtime` underscore prefix means Streamlit ignores it for cache key (only TTL invalidates)
- `load_feature_data` silently swallows exceptions via bare `except Exception: return None`
- Redundant `import os as _os` mid-file in 5_discovery.py (line 60) when `os` already imported at line 8
