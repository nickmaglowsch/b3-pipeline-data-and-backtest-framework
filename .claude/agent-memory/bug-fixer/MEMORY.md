# Bug Fixer Memory — B3 Data Pipeline

## Project Overview
- Python 3.9 + SQLite (NOT DuckDB, despite diagnosis mention)
- Test framework: pytest (installed manually into .venv with pip install pytest)
- Run tests: `source .venv/bin/activate && pytest tests/ -v`
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
- `company_isin_map` table: new (cnpj, isin_code, ticker, share_class, is_primary, first_seen, last_seen)
- Migration: `_migrate_schema()` adds quotation_factor column and company_isin_map indexes to existing DBs

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

## CVM pipeline ticker-fetch step (fixed 2026-03-04)
- `_fetch_ticker_mappings(conn)` -- bulk path: `fetch_all_b3_listed_companies()` then
  `update_ticker_by_cvm_code()`; fallback: per-company `fetch_company_data()` + `update_ticker_by_cvm_code()`
- NEVER use `extract_cnpj_from_company_data` for ticker mapping -- `GetListedSupplementCompany` returns cnpj=None always
- `fetch_all_b3_listed_companies()` in `b3_corporate_actions.py` -- paginates GetInitialCompanies
- `populate_company_isin_map(conn)` in `cvm_storage.py` -- fills company_isin_map from prices+cvm_companies join
- `materialize_valuation_ratios()` -- ISIN join (company_isin_map) primary, LIKE fallback secondary
- Match rate warning fires when < 10% of ticker roots matched cvm_companies
- `run_fundamentals_pipeline()` calls `populate_company_isin_map` between ticker propagation and ratio materialization
- `--skip-ticker-fetch` CLI flag in `cvm_main.py main()`
- `main.py _run_update_cnpj_map()` delegates to `cvm_main._fetch_ticker_mappings(conn)`

## B3 API behavior (confirmed)
- `GetListedSupplementCompany`: returns `cnpj: null` ALWAYS -- never use CNPJ from this endpoint
- `GetListedSupplementCompany`: DOES return `codeCVM` as integer (e.g., 9512)
- `GetInitialCompanies`: returns cnpj, codeCVM (as string), issuingCompany (ticker root)
- `update_ticker_by_cvm_code` zero-pads codeCVM integers to 6 chars to match CVM CSV format

## Non-Split Labels (stored in skipped_events, never applied as splits)
- RESG TOTAL RV: total share redemption / delisting (reason: delisting_event)
- CIS RED CAP: spin-off with capital reduction (reason: needs_manual_review)
- INCORPORACAO: merger (reason: needs_manual_review)
See: `b3_pipeline/config.py` B3_LABEL_* constants

## Test Patterns
- When mocking `_fetch_ticker_mappings` tests: ALWAYS mock both
  `fetch_all_b3_listed_companies` AND `fetch_company_data` to avoid real HTTP calls
- Tests that only mock the fallback path must also mock `fetch_all_b3_listed_companies`
  with `return_value=[]` to force the fallback code path
- `caplog.at_level(logging.WARNING, logger="b3_pipeline.cvm_main")` to capture warnings
