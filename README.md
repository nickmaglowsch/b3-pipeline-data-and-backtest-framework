# B3 Historical Market Data Pipeline

A complete, production-ready Python data pipeline for downloading, parsing, and processing historical equity data from the B3 (Brazilian Stock Exchange).

This tool automatically downloads historical daily price data from B3's official systems, fetches corporate actions (dividends, JCP), detects splits, and calculates split-adjusted OHLC and total-return adjusted close prices (Yahoo Finance style).

All data is stored in a clean SQLite database ready for quantitative analysis or backtesting.

## Features

- **Automated Downloads**: Fetches B3 COTAHIST annual files (from 1994 to present) automatically.
- **Fixed-width Parsing**: Parses the notoriously complex COTAHIST fixed-width format, filtering only for standard lot equities (BDI `02`).
- **Corporate Actions**: Fetches historical dividend and JCP events automatically via the StatusInvest public API.
- **Split Detection**: Heuristically detects stock splits, reverse splits, and bonuses based on price discontinuities, mapping them to common split ratios.
- **Data Adjustments**:
  - `split_adj_*`: OHLC prices adjusted for splits/reverse splits (backward cumulative factor).
  - `split_adj_volume`: Volume inversely adjusted for splits.
  - `adj_close`: Total-return adjusted close price accounting for both splits and dividends/JCP (calculated using the industry-standard Yahoo Finance backward cumulative method based on the ex-date).
- **Idempotent**: Safe to run multiple times. Uses `INSERT OR REPLACE` and checks for existing files.
- **Resilient**: Handles missing data, gracefully skips rate-limited responses, and handles partial years.

## Architecture

- `main.py` - CLI orchestrator and pipeline execution.
- `downloader.py` - Manages fetching ZIPs from B3 and JSON payloads from StatusInvest.
- `parser.py` - Extracts and normalizes the `.TXT` files within the downloaded ZIPs.
- `adjustments.py` - Core logic for split detection and mathematical price adjustments.
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
Stores raw dividend, JCP, and taxed yield events.

### Table: `detected_splits`
Primary Key: `(ticker, ex_date)`
Stores heuristically detected splits and reverse-splits.

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

## License

MIT License
