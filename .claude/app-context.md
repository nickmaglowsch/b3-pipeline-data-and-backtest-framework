# B3 Data Pipeline — Application Context

## Project Overview

A production-ready Python data pipeline for downloading, parsing, and processing historical equity data from B3 (Brazilian Stock Exchange). Built on a single SQLite database, it includes price data ingestion, corporate actions handling, CVM fundamentals, a backtesting framework (30+ strategies), an ML research engine, and a Streamlit web UI.

---

## How to Run the App

### Prerequisites

- Python 3.9+
- Virtual environment at `.venv/` (activate before running any commands)

```bash
source .venv/bin/activate
```

### Price Data Pipeline (COTAHIST)

```bash
# Standard run — downloads missing years and updates the DB
python -m b3_pipeline.main

# Rebuild DB from scratch (drops all tables)
python -m b3_pipeline.main --rebuild

# Process a single year only
python -m b3_pipeline.main --year 2024

# From a specific year onwards
python -m b3_pipeline.main --start-year 2010

# Skip re-fetching corporate actions (faster)
python -m b3_pipeline.main --skip-corporate-actions

# Also detect non-standard splits from price heuristics
python -m b3_pipeline.main --detect-nonstandard-splits

# Retry previously failed company fetches
python -m b3_pipeline.main --retry-failures

# Populate CNPJ → ticker map from B3 API (fast, no price data needed)
python -m b3_pipeline.main --update-cnpj-map

# Verbose/debug logging
python -m b3_pipeline.main --verbose
```

### CVM Fundamentals Pipeline

```bash
# Run all years (default start: 2010)
python -m b3_pipeline.cvm_main

# From a specific year
python -m b3_pipeline.cvm_main --start-year 2020

# Specific year range
python -m b3_pipeline.cvm_main --start-year 2023 --end-year 2023

# Rebuild CVM tables from scratch
python -m b3_pipeline.cvm_main --rebuild

# Skip valuation ratio computation (P/E, P/B, EV/EBITDA)
python -m b3_pipeline.cvm_main --skip-ratios

# Skip B3 API ticker fetch (use when already populated)
python -m b3_pipeline.cvm_main --skip-ticker-fetch

# Force re-download of ZIPs even if cached
python -m b3_pipeline.cvm_main --force-download
```

### Feature Discovery Engine

```bash
# Full run
python -m research.discovery.main

# Incremental (skip already-computed features)
python -m research.discovery.main --incremental

# Force full recompute (wipes feature store)
python -m research.discovery.main --force-recompute
```

### Web UI (Streamlit)

```bash
# Option 1 — convenience launcher
python run_ui.py

# Option 2 — direct streamlit
streamlit run ui/app.py
```

Open http://localhost:8501 in a browser.

UI pages:
1. **Pipeline** (`1_pipeline.py`) — DB stats, trigger pipeline runs, real-time logs
2. **Backtest Runner** (`2_backtest_runner.py`) — Select strategy, configure params, run backtests
3. **Dashboard** (`3_dashboard.py`) — Browse and compare saved backtest results
4. **Research** (`4_research.py`) — ML feature importance and model metrics
5. **Discovery** (`5_discovery.py`) — IC-based alpha feature discovery
6. **Fundamentals** (`6_fundamentals.py`) — CVM fundamentals pipeline runner and coverage stats

---

## How to Run Tests

```bash
# Run all 60 tests
pytest

# Verbose output
pytest -v

# Run a specific test file
pytest tests/test_cvm_storage.py
pytest tests/test_cvm_parser.py
pytest tests/test_fundamentals_pit.py
pytest tests/test_cvm_downloader.py
pytest tests/test_cvm_companies.py

# Run a single test by name
pytest tests/test_cvm_storage.py::test_function_name -v
```

Test files live in `/tests/`. All 60 tests should pass on a clean run.

---

## Key Files and Structure

```
b3-data-pipeline/
├── b3_market_data.sqlite          # Single SQLite database (all data)
├── requirements.txt               # Python dependencies
├── run_ui.py                      # Streamlit UI launcher
│
├── b3_pipeline/                   # Core data pipeline
│   ├── config.py                  # Constants: URLs, paths, DB_PATH, schema offsets
│   ├── main.py                    # COTAHIST pipeline CLI entry point
│   ├── downloader.py              # Downloads COTAHIST ZIP files from B3
│   ├── parser.py                  # Fixed-width COTAHIST parser
│   ├── storage.py                 # SQLite schema (SCHEMA_* constants), init_db(), upsert functions
│   ├── adjustments.py             # Split and dividend adjustment logic
│   ├── b3_corporate_actions.py    # Fetches dividends/splits from B3 listedCompaniesProxy API
│   ├── cvm_main.py                # CVM fundamentals pipeline CLI entry point
│   ├── cvm_downloader.py          # Downloads DFP/ITR/FRE ZIP files from CVM portal
│   ├── cvm_parser.py              # Parses CVM financial statement ZIP files
│   └── cvm_storage.py             # CVM-specific DB operations (companies, filings, fundamentals)
│
├── backtests/
│   ├── core/
│   │   ├── data.py                # load_b3_data(), load_b3_hlc_data(), download_cdi_daily(), load_fundamentals_pit()
│   │   ├── shared_data.py         # build_shared_data() — precomputes all DataFrames for strategies
│   │   ├── strategy_base.py       # StrategyBase ABC + ParameterSpec
│   │   ├── strategy_registry.py   # Auto-discovers strategy plugins; get_registry()
│   │   ├── simulation.py          # Tax-aware portfolio simulator
│   │   ├── metrics.py             # Sharpe, Calmar, drawdown, annualized return/vol
│   │   ├── plotting.py            # Dark-theme tear sheets and equity curves
│   │   ├── portfolio_opt.py       # equal-weight, inverse-vol, ERC, HRP, rolling Sharpe
│   │   ├── param_scanner.py       # 2D parameter sweep with heatmap
│   │   └── strategy_returns.py    # Runs all strategies, returns monthly return DataFrame
│   │
│   └── strategies/                # 16 strategy plugins (auto-discovered)
│       ├── momentum_sharpe.py
│       ├── low_volatility.py
│       ├── multifactor.py
│       ├── research_multifactor.py
│       ├── value_quality.py       # P/B + ROE composite (uses fundamentals)
│       └── ...                    # 11 more strategies
│
├── ui/
│   ├── app.py                     # Streamlit entry point
│   ├── pages/                     # 6 Streamlit pages (numbered 1–6)
│   ├── services/                  # Backend services for UI (pipeline_service, job_runner, etc.)
│   └── components/                # Reusable UI components
│
├── research/
│   ├── data_loader.py             # Research-specific data loading
│   ├── features.py                # Feature computation helpers
│   ├── feature_store/             # Parquet-based feature cache
│   │   ├── registry.json          # Feature metadata + evaluation summaries
│   │   ├── features/              # One Parquet per feature
│   │   └── evaluations/
│   │       └── ic_timeseries.parquet
│   ├── discovery/                 # Automatic feature discovery engine
│   │   ├── main.py                # Discovery pipeline CLI entry point
│   │   ├── base_signals.py        # 15 signal categories
│   │   ├── evaluator.py           # IC computation (Spearman rank correlation)
│   │   ├── pruning.py             # NaN filter, IC threshold, correlation dedup
│   │   └── ...
│   └── output/                    # Discovery artifacts (JSON catalog, plots, reports)
│
├── tests/                         # pytest test suite (60 tests)
│   ├── test_cvm_storage.py
│   ├── test_cvm_companies.py
│   ├── test_cvm_downloader.py
│   ├── test_cvm_parser.py
│   ├── test_fundamentals_pit.py
│   └── ...
│
└── data/
    ├── raw/                       # Downloaded COTAHIST ZIP files
    └── cvm/                       # Downloaded CVM DFP/ITR/FRE ZIP files
```

---

## Database

### Location

`/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_market_data.sqlite`

Defined in `b3_pipeline/config.py` as `DB_PATH = PROJECT_ROOT / "b3_market_data.sqlite"`.

### Getting a Connection (Python)

```python
from b3_pipeline import storage
conn = storage.get_connection()
# Always use WAL mode + normal sync (set automatically)
```

Or directly:

```python
import sqlite3
conn = sqlite3.connect("b3_market_data.sqlite")
```

### Tables

| Table | Primary Key | Description |
|---|---|---|
| `prices` | `(ticker, date)` | OHLCV + split-adj + adj_close prices |
| `corporate_actions` | `(isin_code, event_date, event_type)` | Dividends, JCP |
| `stock_actions` | `(isin_code, ex_date, action_type)` | Splits, reverse splits, bonus shares |
| `detected_splits` | `(isin_code, ex_date)` | Splits detected from price heuristics |
| `skipped_events` | `(isin_code, event_date, label)` | Unrecognized B3 labels (manual review) |
| `fetch_failures` | `(company_code, endpoint)` | API fetch failures with retry tracking |
| `cvm_companies` | `cnpj` | CNPJ → ticker + company metadata |
| `cvm_filings` | `filing_id` | CVM filing index (DFP, ITR, FRE) |
| `fundamentals_pit` | `filing_id` | Point-in-time fundamentals (revenue, NI, EBITDA, equity, ratios) |

### Key Columns in `prices`

| Column | Notes |
|---|---|
| `ticker` | e.g. `PETR4`, `VALE3` |
| `date` | `YYYY-MM-DD` |
| `close` | Raw close (unadjusted) |
| `split_adj_close` | Split-adjusted only |
| `adj_close` | Total-return adjusted (splits + dividends) — use this for backtesting |
| `volume` | Financial volume with 2 implied decimal places — divide by 100 to get BRL |

### Key Columns in `fundamentals_pit`

All financial values in **thousands of BRL** as reported by CVM. Multiply by 1000 when computing ratios against market cap.

| Column | Notes |
|---|---|
| `cnpj` | Company identifier |
| `ticker` | 4-char root (e.g. `PETR`) |
| `filing_date` | Date the filing became public — use for PIT queries |
| `period_end` | Financial period end date |
| `doc_type` | `DFP` (annual), `ITR` (quarterly), `FRE` (reference form) |
| `revenue`, `net_income`, `ebitda`, `equity`, `total_assets`, `net_debt` | In thousands BRL |
| `pe_ratio`, `pb_ratio`, `ev_ebitda` | Pre-computed ratios |

### Common Queries

```sql
-- Adjusted price history for a ticker
SELECT date, adj_close, close, volume / 100.0 AS fin_volume_brl
FROM prices
WHERE ticker = 'PETR4'
ORDER BY date DESC
LIMIT 20;

-- All standard lot equity tickers (suffix 3/4/5/6 or 11)
SELECT DISTINCT ticker FROM prices
WHERE (LENGTH(ticker) = 5 AND SUBSTR(ticker, 5, 1) IN ('3', '4', '5', '6'))
   OR (LENGTH(ticker) = 6 AND SUBSTR(ticker, 5, 2) = '11')
ORDER BY ticker;

-- Latest fundamentals for a ticker (PIT-safe: latest filing on or before a date)
SELECT period_end, filing_date, doc_type, revenue, net_income, pe_ratio, pb_ratio
FROM fundamentals_pit
WHERE ticker = 'PETR'
  AND filing_date <= '2025-12-31'
ORDER BY filing_date DESC, filing_version DESC
LIMIT 5;

-- DB summary stats (replicates storage.get_summary_stats)
SELECT
  (SELECT COUNT(*) FROM prices) AS total_prices,
  (SELECT COUNT(DISTINCT ticker) FROM prices) AS tickers,
  (SELECT MIN(date) FROM prices) AS first_date,
  (SELECT MAX(date) FROM prices) AS last_date,
  (SELECT COUNT(*) FROM fundamentals_pit) AS fundamentals_rows;
```

### Schema Initialization / Migration

Schema is managed in `b3_pipeline/storage.py` via `SCHEMA_*` string constants. Migrations are applied non-destructively through `_migrate_schema()` which runs on every `init_db()` call. To add a new column: add an `ALTER TABLE` statement inside `_migrate_schema()`.

---

## Adding a New Strategy

1. Create `backtests/strategies/my_strategy.py`
2. Subclass `StrategyBase`, implement `name`, `description`, `get_parameter_specs()`, and `generate_signals()`
3. The strategy is auto-discovered by `StrategyRegistry.discover()` on next run — no registration needed

```python
from backtests.core.strategy_base import StrategyBase, ParameterSpec, COMMON_START_DATE, COMMON_END_DATE

class MyStrategy(StrategyBase):
    @property
    def name(self) -> str:
        return "My Strategy"

    @property
    def description(self) -> str:
        return "What this strategy does."

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [COMMON_START_DATE, COMMON_END_DATE]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        # ... compute target_weights DataFrame ...
        return returns_matrix, target_weights
```

If the strategy uses fundamentals, add `needs_fundamentals = True` as a class attribute.

---

## Key Conventions

- **No ORM** — raw `sqlite3` everywhere
- **CVM values in thousands of BRL** — multiply by 1000 when comparing with market cap
- **`adj_close` for backtesting** — dividend + split adjusted; `split_adj_close` for ATR/technical signals
- **`volume / 100.0`** — COTAHIST financial volume has 2 implied decimal places
- **Strict PIT** — never deduplicate fundamentals across all time; forward-fill handles restatements
- **Account codes**: `3.01` = Revenue, `3.11` = Net Income, `6.01` = Total Assets, `6.02` = Equity
- **`include_fundamentals=False`** default in `build_shared_data()` — opt-in for strategies that need it
