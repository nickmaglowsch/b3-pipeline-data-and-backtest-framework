# B3 Historical Market Data Pipeline

A complete, production-ready Python data pipeline for downloading, parsing, and processing historical equity data from B3 (Brazilian Stock Exchange).

**B3 is the single authoritative source for all data:**
- Price data (COTAHIST)
- Cash dividends and JCP (Juros sobre Capital Próprio)
- Stock splits (Desdobramento)
- Reverse splits (Grupamento)
- Bonus shares (Bonificação)

This tool automatically downloads historical daily price data from B3's official systems, fetches corporate actions directly from B3's listedCompaniesProxy API, and calculates split-adjusted OHLC and total-return adjusted close prices (Yahoo Finance style).

All data is stored in a clean SQLite database ready for quantitative analysis or backtesting.

## Features

- **Automated Downloads**: Fetches B3 COTAHIST annual files (from 1994 to present) automatically.
- **Fixed-width Parsing**: Parses the notoriously complex COTAHIST fixed-width format, filtering only for standard lot equities (BDI `02`).
- **Corporate Actions from B3**: Fetches dividends, JCP, splits, reverse splits, and bonus shares directly from B3's official API.
- **Accurate Split Data**: Uses B3's official split factors instead of heuristic detection, correctly handling:
  - Stock splits (e.g., 100:1 split in 2008)
  - Reverse splits (e.g., 0.01 factor in 2000)
  - Bonus shares (Bonificação)
- **Data Adjustments**:
  - `split_adj_*`: OHLC prices adjusted for splits/reverse splits (backward cumulative factor).
  - `split_adj_volume`: Volume inversely adjusted for splits.
  - `adj_close`: Total-return adjusted close price accounting for both splits and dividends/JCP.
- **Idempotent**: Safe to run multiple times. Uses `INSERT OR REPLACE` and checks for existing files.
- **Resilient**: Handles missing data, gracefully skips rate-limited responses, and handles partial years.

## Architecture

- `main.py` - CLI orchestrator and pipeline execution.
- `downloader.py` - Manages fetching ZIPs from B3 COTAHIST.
- `b3_corporate_actions.py` - Fetches corporate actions from B3 listedCompaniesProxy API.
- `parser.py` - Extracts and normalizes the `.TXT` files within the downloaded ZIPs.
- `adjustments.py` - Core logic for split and dividend adjustments using B3 official data.
- `storage.py` - SQLite schema definition and fast batch upsert operations.
- `config.py` - Configuration, schema offsets, and URL templates.

## Installation

Requirements: Python 3.11+

```bash
# Clone the repository
git clone <repository_url>
cd b3-data-pipeline

# Install dependencies
pip install -r requirements.txt
```

Dependencies are intentionally kept light: only `requests` and `pandas` are required.

## Usage

Run the pipeline using the main module:

```bash
# Run the standard pipeline (downloads any missing years and updates DB)
python -m b3_pipeline.main

# Rebuild the database from scratch (drops tables and recompiles adjustments)
python -m b3_pipeline.main --rebuild

# Process a specific year only (useful for testing)
python -m b3_pipeline.main --year 2024

# Process data but skip fetching corporate actions (faster runs)
python -m b3_pipeline.main --skip-corporate-actions
```

## Database Schema

The pipeline produces a SQLite database file named `b3_market_data.sqlite` with the following schema:

### Table: `prices`
Primary Key: `(ticker, date)`

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | TEXT | Stock symbol (e.g., PETR4) |
| `date` | DATE | Trading date (YYYY-MM-DD) |
| `open` | REAL | Raw open price |
| `high` | REAL | Raw high price |
| `low` | REAL | Raw low price |
| `close` | REAL | Raw close price |
| `volume` | INTEGER | Raw traded volume |
| `split_adj_open` | REAL | Open price adjusted for splits |
| `split_adj_high` | REAL | High price adjusted for splits |
| `split_adj_low` | REAL | Low price adjusted for splits |
| `split_adj_close`| REAL | Close price adjusted for splits |
| `adj_close` | REAL | Close price adjusted for splits AND dividends |

### Table: `corporate_actions`
Primary Key: `(ticker, event_date, event_type)`

Stores dividends, JCP, and stock action events with ISIN codes for traceability.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | TEXT | Stock symbol |
| `event_date` | DATE | Ex-date |
| `event_type` | TEXT | CASH_DIVIDEND, JCP, STOCK_SPLIT, REVERSE_SPLIT, BONUS_SHARES |
| `value` | REAL | Dividend/JCP amount per share |
| `isin_code` | TEXT | ISIN code from B3 |
| `factor` | REAL | Split/bonus factor |
| `source` | TEXT | Data source (always "B3") |

### Table: `stock_actions`
Primary Key: `(ticker, ex_date, action_type)`

Stores split, reverse split, and bonus share events separately for clarity.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | TEXT | Stock symbol |
| `ex_date` | DATE | Ex-date |
| `action_type` | TEXT | STOCK_SPLIT, REVERSE_SPLIT, BONUS_SHARES |
| `factor` | REAL | Split/bonus factor |
| `isin_code` | TEXT | ISIN code from B3 |
| `source` | TEXT | Data source (always "B3") |

### Table: `detected_splits`
Primary Key: `(ticker, ex_date)`

Legacy table for storing split factors derived from B3 stock_actions data.

| Column | Type | Description |
|--------|------|-------------|
| `ticker` | TEXT | Stock symbol |
| `ex_date` | DATE | Ex-date |
| `split_factor`| REAL | Calculated multiplier (old_shares / new_shares) |
| `description` | TEXT | Text description of the split |

## B3 Corporate Action Labels

The pipeline maps B3's Portuguese labels to standardized types:

| B3 Label | Event Type | Description |
|----------|------------|-------------|
| DIVIDENDO | CASH_DIVIDEND | Cash dividend |
| JRS CAP PROPRIO | JCP | Interest on own capital |
| RENDIMENTO | CASH_DIVIDEND | Yield (treated as dividend) |
| DESDOBRAMENTO | STOCK_SPLIT | Stock split (factor > 1) |
| GRUPAMENTO | REVERSE_SPLIT | Reverse split (factor < 1) |
| BONIFICACAO | BONUS_SHARES | Bonus shares |

## Split Factor Interpretation

B3 provides factors as localized strings (e.g., "100,00000000000"):

- **Stock Split (DESDOBRAMENTO)**: factor > 1
  - Example: factor=100 means 100 new shares for each 1 old share
  - Split factor for adjustment = 1/100 = 0.01

- **Reverse Split (GRUPAMENTO)**: factor < 1
  - Example: factor=0.01 means 1 new share for each 100 old shares
  - Split factor for adjustment = 1/0.01 = 100

- **Bonus Shares (BONIFICACAO)**: factor represents percentage
  - Example: factor=33.33 means 33.33% bonus (get 133 shares for each 100 held)
  - Split factor for adjustment = 1/(1+33.33/100) ≈ 0.75

## Example Query

To fetch a clean, Yahoo-style historical price series for Petrobras:

```sql
SELECT 
    date, 
    ticker, 
    open, 
    high, 
    low, 
    close as raw_close, 
    adj_close
FROM prices 
WHERE ticker = 'PETR4' 
ORDER BY date DESC
LIMIT 10;
```

## Backtesting Framework

The `backtests/` directory contains a full quantitative backtesting framework built on top of the data pipeline. It includes 30+ individual strategy backtests, portfolio optimization tools, and research utilities.

### Core Modules (`backtests/core/`)

| Module | Description |
|--------|-------------|
| `data.py` | Loads B3 data from SQLite + downloads CDI (BCB API) and IBOV (Yahoo Finance) |
| `simulation.py` | Tax-aware portfolio simulator (15% CGT, loss carryforward, slippage, R$20K monthly sales exemption) |
| `metrics.py` | Performance metrics: Sharpe, Calmar, max drawdown, annualized return/volatility |
| `plotting.py` | Standardized dark-theme tear sheets and equity curve plots |
| `strategy_returns.py` | Runs all 8 core strategies in one call, returns a clean DataFrame of monthly after-tax returns |
| `portfolio_opt.py` | Portfolio weight functions: equal-weight, inverse-vol, ERC, HRP, rolling Sharpe, regime-conditional |
| `param_scanner.py` | Generic 2D parameter sweep framework with heatmap visualization |

### Quick Start

```bash
# Install backtesting dependencies
pip install -r requirements.txt

# Run the portfolio comparison dashboard (all 7 optimization methods)
cd backtests && python portfolio_compare_all.py

# Run sub-period stability analysis
cd backtests && python portfolio_stability_analysis.py

# Run parameter sensitivity heatmaps
cd backtests && python param_sensitivity_analysis.py
```

### Building a Custom Portfolio

```python
from core.strategy_returns import build_strategy_returns
from core.portfolio_opt import hrp_weights, compute_portfolio_returns
from core.metrics import build_metrics

# Load all 8 strategy return streams + IBOV/CDI benchmarks
returns_df, sim_results, regime_signals = build_strategy_returns()

# Select liquid strategies (ADTV >= R$1M)
liquid = [c for c in returns_df.columns if c not in ("IBOV", "CDI", "SmallcapMom")]

# Build an HRP-weighted portfolio
port_ret, weights_df = compute_portfolio_returns(
    returns_df, liquid, lambda w: hrp_weights(w, 36)
)

# Check performance
print(build_metrics(port_ret, "My HRP Portfolio", 12))
```

### Available Strategies

**Stock-selection signals:** Momentum (12M), Smooth Momentum / Sharpe, Low Volatility, MultiFactor (mom + low-vol), Research MultiFactor (5 factors + regime), Smallcap Momentum, Anti-Lottery, Frog-in-the-Pan, Volume Breakout, Mean Reversion, Dividend Yield, and more.

**Regime/allocation filters:** COPOM easing/tightening, IBOV trend (10M SMA), IBOV volatility percentile, dual momentum, global flight (IBOV vs IVVB11).

**Portfolio optimization methods:** Equal Weight, Inverse Volatility, Equal Risk Contribution (ERC), Hierarchical Risk Parity (HRP), Rolling Sharpe, Regime-Conditional, and Combined (regime budget + rolling Sharpe within equity).

### Research Scripts

| Script | Description |
|--------|-------------|
| `compare_all.py` | Side-by-side comparison of all individual strategies |
| `correlation_matrix.py` | Strategy return correlation heatmap |
| `portfolio_risk_parity_backtest.py` | Equal-weight vs inverse-vol vs ERC |
| `portfolio_hrp_backtest.py` | Hierarchical Risk Parity + dendrogram |
| `portfolio_dynamic_backtest.py` | Rolling Sharpe, regime, and combined dynamic allocation |
| `portfolio_compare_all.py` | All 7 portfolio methods: 4-panel dashboard + CSV export |
| `portfolio_stability_analysis.py` | Sub-period metrics (4 eras) + rolling 36-month Sharpe |
| `param_sensitivity_analysis.py` | 2D parameter sweep heatmaps for key strategies |
| `seasonal_effects_backtest.py` | Turn-of-month, monthly seasonality, Dec/Jan, Sell-in-May |
| `earnings_proxy_backtest.py` | Volume-confirmed momentum around B3 reporting windows |
| `sector_rotation_backtest.py` | Sector-level momentum with heuristic ticker classification |

## License

MIT License
