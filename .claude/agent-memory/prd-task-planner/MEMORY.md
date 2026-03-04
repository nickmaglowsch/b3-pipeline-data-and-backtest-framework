# PRD Task Planner - Agent Memory

## Project Architecture
- **b3_pipeline/**: Data pipeline (downloader, parser, adjustments, storage, config)
- **backtests/**: 43 strategy backtests, each a standalone `*_backtest.py` script
- **backtests/core/**: Shared modules (data.py, simulation.py, metrics.py, plotting.py, strategy_returns.py, portfolio_opt.py, param_scanner.py, shared_data.py, strategy_base.py, strategy_registry.py)
- **backtests/strategies/**: 11 registered StrategyBase plugins (auto-discovered by registry)
- **research/**: ML feature importance study (RF+XGB, 19 features, outputs in research/output/)
- **research/discovery/**: Feature discovery engine (541 generated, 186 after pruning)
- **ui/**: Streamlit app with pages: pipeline, backtest_runner, dashboard, research, discovery
- **DB**: `b3_market_data.sqlite` at project root
- **Data**: 33 COTAHIST ZIPs in `data/raw/` covering 1994-2026

## Database Schema (prices table)
- Columns: ticker, isin_code, date, open, high, low, close, volume, split_adj_open/high/low/close, adj_close
- PK: (ticker, date)
- Volume is COTAHIST VOLTOT (financial volume), parsed as int with 2 implied decimals. Real BRL volume = volume/100
- Tables: prices, corporate_actions, stock_actions, detected_splits

## Key Conventions
- Standard lot tickers: length 5 ending in 3/4/5/6, or length 6 ending in 11
- Backtests use wide-format DataFrames (date x ticker) pivoted from SQLite long format
- Liquidity filter: typically ADTV >= R$1M, price >= R$1.0
- Returns computed from adj_close (dividend+split adjusted); ATR/technicals from split_adj_* columns
- Cross-sectional ranking via `.rank(axis=1, pct=True)`
- CDI downloaded from BCB SGS API series 12; IBOV from Yahoo via yfinance
- Plotting: dark theme PALETTE in backtests/core/plotting.py; all matplotlib
- `from __future__ import annotations` required for modern type hints

## Strategy Patterns
- **Standalone scripts**: `*_backtest.py` with own `main()`, produce PNG tear sheets
- **Plugins**: extend `StrategyBase` in `backtests/strategies/`, `generate_signals(shared_data, params) -> (ret, tw)`
- **shared_data dict**: precomputed by `build_shared_data()` with price, return, regime, composite keys
- **Simulation**: `run_simulation()` supports long AND short (negative weights), tax, slippage
- **Short example**: `bull_trap_short_backtest.py` uses -0.20/N weights with CDI collateral

## Feature Store
- Parquet files in `research/feature_store/features/` (long format: date, ticker, value)
- Key features for mean-reversion: Autocorr_20d/60d, High_low_range_5d/20d/60d, Rolling_vol_*, CDI_cumulative_*, Mean_reversion_5d/10d/20d
- Top IC_IR features: ratio(HLR_20d/Win_rate_120d) -0.1616, ratio(Autocorr_60d/ATR_14) -0.1479, CDI ratios ~0.1238

## Dependencies (as of 2026-03-02)
- requirements.txt: requests, pandas, numpy, matplotlib, scikit-learn, xgboost, yfinance, python-dateutil, scipy, hmmlearn, streamlit>=1.30, plotly>=5.18, pyarrow>=14
- Python 3.9 with venv at .venv/
- `from __future__ import annotations` required for modern type hints

## Key Files for Backtest Architecture
- `backtests/core/strategy_returns.py` -- centralizes 8 core strategies with shared data dict pattern
- `backtests/core/portfolio_opt.py` -- inverse-vol, ERC, HRP, rolling Sharpe, regime-conditional weights
- `backtests/core/param_scanner.py` -- 2D parameter sweep framework
- `backtests/compare_all.py` -- runs 8 strategies side-by-side
- `backtests/correlation_matrix.py` -- strategy return correlation heatmap

## Streamlit UI (IMPLEMENTED as of 2026-03-02)
- Entry: `streamlit run ui/app.py` from project root
- Pages: 1_pipeline, 2_backtest_runner, 3_dashboard, 4_research, 5_discovery
- Services: job_runner.py (bg thread+queue), backtest_service.py, pipeline_service.py, research_service.py, discovery_service.py, result_store.py
- Components: charts.py, log_stream.py, metrics_table.py, parameter_form.py, discovery_charts.py

## Feature Discovery Engine (2026-03-02) -- IMPLEMENTED
- Subpackage: `research/discovery/` (12 files)
- Entry point: `python -m research.discovery.main` (--incremental, --force-recompute)
- 541 features generated (registry.json), 186 survive pruning (feature_catalog.json)
- Feature store: Parquet files in research/feature_store/features/, JSON registry
- Catalog JSON structure: features[] with rank, id, category, level, formula_human, metrics per horizon, turnover, decay
- EWM and Mean Reversion base signals implemented (all 17+ categories present)

## Tests (pytest introduced 2026-03-03)
- pytest>=7.0 added to requirements.txt as part of fundamentals feature
- Test directory: `tests/` at project root, with `tests/__init__.py`
- Convention: `tests/test_<module_name>.py`
- Run command: `python -m pytest tests/ -v` from project root

## Fundamentals Pipeline (IN PROGRESS as of 2026-03-03)
- New tables in `b3_market_data.sqlite`: `cvm_companies`, `cvm_filings`, `fundamentals_pit`
- New modules: `b3_pipeline/cvm_downloader.py`, `cvm_parser.py`, `cvm_storage.py`, `cvm_main.py`
- CNPJ mapping: B3 API `GetListedSupplementCompany` response contains CNPJ field (verify field name at impl time)
- CVM data directory: `data/cvm/` (created by `config.CVM_DATA_DIR`)
- CVM data starts 2010; backtests using fundamentals should start 2012+
- CVM CSV encoding: latin-1, semicolon-separated; `ORDEM_EXERC == 'ÚLTIMO'` filter for current period
- Account codes: 3.01=revenue, 3.05=EBIT(EBITDA proxy), 3.11=net_income, 1=total_assets, 2.03=equity
- Fundamentals in shared_data: `include_fundamentals=True` param; keys prefixed `f_` (e.g. `f_pe_ratio`)
- Strategy plugin `ValueQuality` in `backtests/strategies/value_quality.py` uses `needs_fundamentals = True`
- UI: `ui/pages/6_fundamentals.py` + `ui/services/fundamentals_service.py`
- CVM URL constants in config.py: `CVM_DFP_BASE_URL`, `CVM_ITR_BASE_URL`, `CVM_FRE_BASE_URL`
