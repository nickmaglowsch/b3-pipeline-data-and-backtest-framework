# PRD Task Planner - Agent Memory

## Project Architecture
- **b3_pipeline/**: Data pipeline (downloader, parser, adjustments, storage, config)
- **backtests/**: 43 strategy backtests, each a standalone `*_backtest.py` script
- **backtests/core/**: Shared modules (data.py, simulation.py, metrics.py, plotting.py, strategy_returns.py, portfolio_opt.py, param_scanner.py)
- **research/**: ML feature importance study (RF+XGB, 19 features, outputs in research/output/)
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
- Pages: 1_pipeline, 2_backtest_runner, 3_dashboard, 4_research
- Services: job_runner.py (bg thread+queue), backtest_service.py, pipeline_service.py, research_service.py, result_store.py
- Components: charts.py (Plotly, PALETTE, _apply_dark_theme), log_stream.py (st.fragment), metrics_table.py, parameter_form.py
- All pages use pattern: _PROJECT_ROOT path insert, set_page_config, try/except ImportError for graceful degradation
- Dark theme: bg=#0D1117, panel=#161B22, grid=#21262D, text=#E6EDF3, sub=#8B949E

## Feature Discovery Engine (2026-03-02) -- IMPLEMENTED
- Subpackage: `research/discovery/` (12 files)
- Entry point: `python -m research.discovery.main` (--incremental, --force-recompute)
- 541 features generated (registry.json), 186 survive pruning (feature_catalog.json)
- Feature store: Parquet files in research/feature_store/features/, JSON registry
- IC time series Parquet NOT persisted (evaluations/ dir empty)
- 4 of 6 static PNG plots generated (missing: IC time series, correlation heatmap)
- Catalog JSON structure: features[] with rank, id, category, level, formula_human, metrics per horizon, turnover, decay
- Current 4_research.py shows OLD ML study, NOT discovery engine
- EWM and Mean Reversion base signals ARE now implemented (all 17+ categories present)
- op_ratio_to_mean IS wired into Level 2 generation (not in UNARY_OPS/BINARY_OPS dicts but called directly)

## No Tests
- No test suite exists in the codebase
