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

Requirements: Python 3.9+

```bash
# Clone the repository
git clone <repository_url>
cd b3-data-pipeline

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
# .venv\Scripts\activate    # Windows

# Install all dependencies
pip install -r requirements.txt
```

This installs everything needed for the data pipeline, backtesting framework, ML research, and the Streamlit web UI.

## Rust Extension (COTAHIST Parser)

The COTAHIST fixed-width parsing loop is implemented as a compiled Rust extension (`cotahist_rs`)
for performance. It uses `pyo3` + `maturin` to produce a Python `.so` and `rayon` for parallel
processing of line records and multiple annual files. Parsing 33 years of data takes ~100ms
instead of ~60s.

### Prerequisites

Install the Rust stable toolchain (one-time, system-wide):

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env   # or restart your shell
```

Install `maturin` into the active Python venv (already in `requirements.txt`):

```bash
pip install maturin
```

### Building the Extension

```bash
# Development build — fast compile, installs into active venv
make dev-rust

# Release build — optimised, produces a .whl in b3_pipeline_rs/target/wheels/
make build-rust
```

The `make dev-rust` command must be run once after cloning, and again whenever
the Rust source files in `b3_pipeline_rs/src/` are changed.

### Running Tests

```bash
make test-rust   # Rust unit tests (cargo test)
make test        # Python test suite (pytest)
make all         # build + test in sequence
```

### Troubleshooting

If you see:

```
ImportError: The cotahist_rs Rust extension is not compiled.
Run `make dev-rust` (or `cd b3_pipeline_rs && maturin develop`)
to build it before running the pipeline.
```

Run `make dev-rust` to compile the extension.

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

## Web UI

A Streamlit-based management UI for running the pipeline, executing backtests, browsing results, and viewing ML research -- all from the browser.

### Running the UI

```bash
# Option 1: using streamlit directly
streamlit run ui/app.py

# Option 2: using the convenience launcher
python run_ui.py
```

Then open http://localhost:8501 in your browser.

### Pages

| Page | Description |
|------|-------------|
| **Pipeline Manager** | View database stats, browse raw COTAHIST files, explore table data, and trigger pipeline runs with real-time log streaming |
| **Backtest Runner** | Select from 13 registered strategies, configure all parameters via dynamic forms, run backtests in background with live logs, view interactive Plotly results |
| **Results Dashboard** | Browse all saved results (new + legacy PNGs), compare multiple strategies side-by-side with equity overlays and correlation matrices |
| **Research Viewer** | View ML feature importance rankings, model performance metrics, and trigger the research pipeline |

### Strategy Plugin System

Strategies are registered via a plugin architecture. Each strategy extends `StrategyBase` and is auto-discovered from `backtests/strategies/`. To add a new strategy:

```python
# backtests/strategies/my_strategy.py
from backtests.core.strategy_base import StrategyBase, ParameterSpec, COMMON_START_DATE, COMMON_END_DATE

class MyStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "My Strategy"

    @property
    def description(self) -> str:
        return "Description shown in the UI."

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [COMMON_START_DATE, COMMON_END_DATE]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        # ... compute target weights ...
        return returns_matrix, target_weights
```

The strategy will automatically appear in the Backtest Runner dropdown on the next page load.

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

## Feature Discovery Engine

The `research/discovery/` module is an **automatic feature discovery engine** that generates, evaluates, and ranks hundreds of candidate alpha features using Information Coefficient (IC) analysis. It replaces manual feature selection with a systematic pipeline that sweeps across signal categories, applies mathematical operators, and prunes the result set to a compact, uncorrelated feature catalog.

### Quick Start

```bash
# Full run (generates all features, evaluates, prunes, exports catalog + plots)
python -m research.discovery.main

# Incremental run (skips already-computed features and evaluations)
python -m research.discovery.main --incremental

# Force recompute (wipes feature store and starts fresh)
python -m research.discovery.main --force-recompute
```

### Pipeline Steps

The discovery pipeline runs 12 steps in sequence:

| Step | What Happens |
|------|-------------|
| 1 | **Load data** from SQLite + IBOV + CDI (reuses `research.data_loader`) |
| 2 | **Initialize feature store** -- checks data hash, invalidates cache if source data changed |
| 3 | **Compute universe mask** -- filters to liquid stocks (ADTV >= R$1M, price >= R$1, 200+ days) |
| 4 | **Generate Level 0 + Level 1** features (base signals + rank/zscore transforms) |
| 5 | **Evaluate** Level 0+1 features (IC computation across 4 forward horizons) |
| 6 | **Select top features** for Level 2 generation (top-50 for delta/ratio_to_mean, top-20 for binary ops) |
| 7 | **Generate Level 2** features (delta, ratio_to_mean, ratio, product operators on top features) |
| 8 | **Evaluate** Level 2 features |
| 9 | **Prune** -- NaN filter → IC threshold → correlation dedup → cap at 500 |
| 10 | **Export feature catalog** JSON for backtest consumption |
| 11 | **Generate plots** and text report |
| 12 | **Save feature store** registry |

### Three-Level Feature Generation

Features are built in layers, each more selective than the last:

```
Level 0: Base Signals (~120-150 features)
  Parametric sweep across 15 signal categories with multiple windows each.

Level 1: Unary Transforms (~240-300 features)
  Apply rank() and zscore() cross-sectionally to every Level 0 signal.

Level 2: Composite Features (variable count)
  - delta(20) on top-50 features by |IC_IR|
  - ratio_to_mean(10/20/60) on top-50 features
  - ratio(A, B) and product(A, B) on top-20 cross-category pairs
```

### Base Signal Categories

| Category | Signals | Windows |
|----------|---------|---------|
| Momentum | Return, Distance-to-MA | 1, 2, 3, 5, 10, 15, 20, 30, 60, 120, 250 |
| Volatility | Rolling vol, ATR, Drawdown | 5, 10, 20, 40, 60, 120 |
| Volume | Z-score, Ratio | 5, 10, 20, 40, 60 |
| Beta | Rolling beta vs IBOV | 60, 120, 252 |
| Skewness | Rolling return skewness | 20, 60, 120 |
| Kurtosis | Rolling return kurtosis | 20, 60, 120 |
| Max/Min Return | Extreme single-day returns | 20, 60, 120 |
| Win-rate | Fraction of positive-return days | 20, 60, 120 |
| Amihud Illiquidity | abs(return) / volume | 20, 60 |
| Autocorrelation | Return lag-1 autocorrelation | 20, 60 |
| Mean Reversion | Normalized distance from rolling mean | 5, 10, 20 |
| EWM Variants | EWM mean/std of returns | spans: 5, 10, 20, 40 |
| High-Low Range | Normalized (H-L)/C average | 5, 20, 60 |
| Market (IBOV) | IBOV return and volatility | 10, 20, 40, 63 |
| Market (CDI) | CDI cumulative and change | 21, 42, 63, 126 |

### IC-Based Evaluation

Every feature is evaluated using **Spearman rank correlation** (Information Coefficient) against forward returns:

```
IC(t) = spearman_corr(feature_ranks(t), forward_return_ranks(t))
```

Computed cross-sectionally (across all stocks) for each date, then aggregated:

| Metric | Description |
|--------|-------------|
| `mean_ic` | Average IC across all dates |
| `ic_ir` | IC Information Ratio = mean_ic / ic_std (primary ranking metric) |
| `ic_t_stat` | Statistical significance of IC |
| `pct_positive_ic` | Fraction of dates with IC > 0 |
| `mean_ic_5y` | Mean IC over last 5 years (recency check) |
| `turnover` | 1 - avg rank autocorrelation (trading cost proxy) |
| `decay_1d/5d/20d` | IC at lagged feature values (signal persistence) |
| `train/test split` | Separate IC on first 70% vs last 30% of dates (overfit detection) |

**Forward return horizons**: 5, 10, 20, and 60 trading days.

### Pruning Pipeline

After evaluation, features pass through four filters:

1. **NaN/Variance filter** -- Drop features with > 30% NaN rate or zero variance on > 10% of dates
2. **IC threshold** -- Drop features where |mean_ic| < 0.005 on the 20-day horizon
3. **Correlation deduplication** -- If two features have Spearman correlation > 0.90, keep the one with higher |IC_IR|
4. **Cap enforcement** -- Keep top 500 features by |IC_IR| if more remain

### Feature Store

Computed features and evaluations are persisted for incremental re-runs:

```
research/feature_store/
├── registry.json              # Feature metadata + evaluation summaries
├── features/                  # One Parquet file per feature (long format: date, ticker, value)
└── evaluations/
    └── ic_timeseries.parquet  # Consolidated IC time series for all features
```

The registry tracks a **data hash** of the source data. If the underlying price data changes (e.g., new trading days added), the store is automatically invalidated and all features are recomputed.

### Output Artifacts

After a full run, the pipeline produces:

```
research/output/
├── feature_catalog.json               # Ranked feature catalog for backtest consumption
├── discovery_report.txt               # Text report with top features and category breakdown
├── discovery_ic_top30.png             # Top 30 features by IC_IR (bar chart)
├── discovery_ic_timeseries.png        # Rolling 1-year IC for top 10 features (line chart)
├── discovery_ic_decay.png             # IC decay across top 20 features
├── discovery_turnover_scatter.png     # IC vs turnover trade-off scatter
├── discovery_correlation_heatmap.png  # Hierarchical clustering of feature correlations
└── discovery_train_test_scatter.png   # Train vs test IC (overfit detection)
```

#### Feature Catalog JSON

The catalog is designed for consumption by the backtest framework:

```json
{
  "generated_at": "2026-03-02T...",
  "evaluation_horizon": "fwd_20d",
  "features": [
    {
      "id": "ratio__Return_60d__Rolling_vol_60d",
      "rank": 1,
      "formula_human": "ratio(Return_60d, Rolling_vol_60d)",
      "category": "composite",
      "mean_ic": 0.045,
      "ic_ir": 1.23,
      "turnover": 0.12
    }
  ]
}
```

### Module Structure

```
research/discovery/
├── __init__.py        # Module definition
├── config.py          # All configuration constants (windows, thresholds, paths)
├── base_signals.py    # 15 signal category compute functions + parametric sweep generator
├── operators.py       # Unary (rank, zscore, delta, ratio_to_mean) + binary (ratio, product) operators
├── generator.py       # Three-level feature generation pipeline
├── evaluator.py       # Vectorized IC computation, decay, turnover analysis
├── store.py           # Parquet + JSON feature store with data hash validation
├── pruning.py         # NaN filter, IC threshold, correlation dedup, cap enforcement
├── catalog.py         # JSON catalog export with human-readable formula parsing
├── report.py          # Text report generation
├── plots.py           # 6 discovery plots (IC bar, timeseries, decay, scatter, heatmap, train/test)
└── main.py            # Pipeline orchestrator with CLI argument parsing
```

## License

MIT License
