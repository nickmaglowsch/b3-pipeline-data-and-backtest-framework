# Code Reviewer Memory -- B3 Data Pipeline

## Project Structure
- SQLite DB: `b3_market_data.sqlite` at project root
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

## Streamlit UI Review Notes (Feb 2026)
- `@st.cache_resource` for shared data dict is fragile -- strategies must `.copy()` before mutation
- `monthly_sales_exemption` not passed in backtest_service.py `run_simulation()` call
- PALETTE dict duplicated in `ui/pages/4_research.py` (should import from charts.py)
- Strategy registry silently swallows import/instantiation errors -- hard to debug
- `result_store.py` has module-level `RESULTS_DIR.mkdir()` side effect
- `log_stream.py` uses blocking while-loop; consider `st.fragment` for partial reruns
- Streamlit version: 1.50.0. `st.status` state values: "running", "complete", "error"
