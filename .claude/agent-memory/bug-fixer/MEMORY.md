# Bug Fixer Memory — B3 Data Pipeline

## Project Overview
- Python 3.9 + SQLite (NOT DuckDB, despite diagnosis mention)
- Test framework: pytest (installed manually into .venv with pip install pytest)
- Run tests: `.venv/bin/python -m pytest tests/ -v`
- No pytest.ini or pyproject.toml at project root

## Key Architectural Facts
- `parse_stock_dividends` returns a 3-tuple: (corp_df, stock_df, skipped_df) after Phase 3 fix
- `fetch_all_corporate_actions` returns a 3-tuple: (corp_df, stock_df, skipped_df) and accepts `conn=` kwarg
- `fetch_company_data` accepts `conn=` kwarg for failure tracking
- Prices are normalized to per-share basis IN the parser (divide OHLC by quotation_factor)
- Volume is NOT divided by quotation_factor (it's monetary BRL, not share count)

## Database Schema (after fix)
- `prices` table: added `quotation_factor INTEGER DEFAULT 1`
- `skipped_events` table: new (isin_code, event_date, label, factor, source, reason)
- `fetch_failures` table: new (company_code, endpoint, error_message, failed_at, retry_count, resolved)
- Migration: `_migrate_schema()` adds quotation_factor column to existing DBs

## Key File Locations
- Parser: `b3_pipeline/parser.py` -- COTAHIST fixed-width parsing (positions 210-217 = fator_cotacao)
- Corporate actions: `b3_pipeline/b3_corporate_actions.py`
- Adjustments: `b3_pipeline/adjustments.py` (includes `detect_splits_from_prices`)
- Storage: `b3_pipeline/storage.py`
- Config: `b3_pipeline/config.py` (label constants live here)
- Backtest split detector: `backtests/core/shared_data.py` lines 24-109

## Split Detection Logic
- Threshold: ratio > 1.8 OR ratio < 0.55 triggers investigation
- Ratios tried: [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50, 100] with 8% tolerance
- Detected splits have source='DETECTED' (not 'B3')
- Fatcot transitions are skipped (normalized prices should be continuous)

## Patching sqlite3.connect inside function-local imports
When a function does `import sqlite3` inside its body (not at module level),
patch `sqlite3.connect` directly, not `b3_pipeline.main.sqlite3.connect`.
  Correct: `patch("sqlite3.connect", ...)`
  Wrong:   `patch("b3_pipeline.main.sqlite3.connect", ...)`  # ModuleNotFoundError

## CVM pipeline ticker-fetch step (added 2026-03-04)
- `_fetch_ticker_mappings(conn)` in `cvm_main.py` -- fetches B3 API per ticker root, upserts cvm_companies
- `run_fundamentals_pipeline()` now has `skip_ticker_fetch=False` param; step 5c calls `_fetch_ticker_mappings`
- `--skip-ticker-fetch` CLI flag added to `cvm_main.py main()`
- `main.py _run_update_cnpj_map()` now delegates to `cvm_main._fetch_ticker_mappings(conn)`

## Non-Split Labels (stored in skipped_events, never applied as splits)
- RESG TOTAL RV: total share redemption / delisting (reason: delisting_event)
- CIS RED CAP: spin-off with capital reduction (reason: needs_manual_review)
- INCORPORACAO: merger (reason: needs_manual_review)
See: `b3_pipeline/config.py` B3_LABEL_* constants
