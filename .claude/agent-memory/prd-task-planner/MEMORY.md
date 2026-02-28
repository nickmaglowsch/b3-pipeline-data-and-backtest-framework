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

## Dependencies (as of 2026-02-28)
- requirements.txt: requests, pandas, numpy, matplotlib, scikit-learn, xgboost, yfinance, python-dateutil, scipy
- Python 3.9 with venv at .venv/
- No web framework installed yet

## Key Files for Backtest Architecture
- `backtests/core/strategy_returns.py` -- centralizes 8 core strategies with shared data dict pattern
- `backtests/core/portfolio_opt.py` -- inverse-vol, ERC, HRP, rolling Sharpe, regime-conditional weights
- `backtests/core/param_scanner.py` -- 2D parameter sweep framework
- `backtests/compare_all.py` -- runs 8 strategies side-by-side
- `backtests/correlation_matrix.py` -- strategy return correlation heatmap

## Streamlit UI Plan (2026-02-28)
- User chose: Streamlit, Plotly charts, background jobs with streaming, full parameter editor, plugin architecture, local-only
- New dirs: ui/ (Streamlit app), backtests/strategies/ (plugin classes), results/ (saved backtest results)
- 13 tasks in tasks/ directory
- Strategy base class with ParameterSpec for dynamic form generation
- Python 3.9 requires `from __future__ import annotations` for modern type hints

## No Tests
- No test suite exists in the codebase
