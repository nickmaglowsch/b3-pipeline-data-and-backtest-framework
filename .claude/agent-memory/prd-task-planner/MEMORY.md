# PRD Task Planner - Agent Memory

## Project Architecture
- **b3_pipeline/**: Data pipeline (downloader, parser, adjustments, storage, config)
- **backtests/**: 30+ strategy backtests, each a standalone script
- **backtests/core/**: Shared modules (data.py, simulation.py, metrics.py, plotting.py)
- **DB**: `b3_market_data.sqlite` at project root (also symlinked/copied to backtests/)
- **Data**: COTAHIST ZIPs in `data/raw/` covering 1994-2026

## Database Schema (prices table)
- Columns: ticker, isin_code, date, open, high, low, close, volume, split_adj_open/high/low/close, adj_close
- PK: (ticker, date)
- Volume is actually COTAHIST VOLTOT (financial volume), parsed as int with 2 implied decimals. Real BRL volume = volume/100
- Tables: prices, corporate_actions, stock_actions, detected_splits

## Key Conventions
- Standard lot tickers: length 5 ending in 3/4/5/6, or length 6 ending in 11
- Backtests use wide-format DataFrames (date x ticker) pivoted from SQLite long format
- Liquidity filter: typically ADTV >= R$1M, price >= R$1.0
- Returns computed from adj_close (dividend+split adjusted); ATR/technicals from split_adj_* columns
- Cross-sectional ranking via `.rank(axis=1, pct=True)`
- CDI downloaded from BCB SGS API series 12; IBOV from Yahoo via yfinance

## Dependencies
- requirements.txt only has requests, pandas
- Venv also has: numpy, matplotlib, yfinance, hmmlearn, sklearn (not pinned)
- No scikit-learn or xgboost in requirements.txt
- Python 3.9

## No Tests
- No test suite exists in the codebase
