# Code Reviewer Memory -- B3 Data Pipeline

## Project Structure
- SQLite DB: `b3_market_data.sqlite` at project root
- Backtests: `backtests/` with `core/data.py` for data loading, `core/metrics.py`, `core/plotting.py`, `core/simulation.py`
- Research module: `research/` -- feature importance discovery (added Feb 2026)
- Tasks/PRDs: `tasks/` directory

## Key Conventions
- sys.path manipulation for imports (both backtests and research use this)
- ATR true_range: `pd.concat([tr1,tr2,tr3]).groupby(level=0).max()` -- established pattern
- Volatility in backtests always uses `ret.shift(1).rolling(n).std()` (1-day lag)
- Dark theme plots: PALETTE dict with bg=#0D1117, panel=#161B22, etc.
- Volume from COTAHIST has 2 implied decimals: true BRL volume = volume / 100

## Common Patterns to Check
- Lookahead bias: features must not use future data; targets use shift(-N)
- Time-series split must be by sorted unique dates, no shuffling
- Universe filtering is rolling (per stock per day), not retroactive
- adj_close for returns, split_adj_* for ATR/technical indicators
- Cross-sectional operations use axis=1 (across tickers per date)

## Known Issues / Recurring Patterns
- min_periods often set lower than window size -- check if intentional
- pickle caching with only time-based invalidation (no config hash)
- gini/gain importance column unification logic duplicated between modeling and viz
