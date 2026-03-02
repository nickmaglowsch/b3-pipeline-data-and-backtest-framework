# Bug Investigator Memory

## Project Overview
- B3 (Brazilian stock exchange) data pipeline: COTAHIST price files + B3 API corporate actions
- SQLite database at `b3_market_data.sqlite` (~560MB, 2.5M price records, 2578 ticker roots)
- Pipeline code in `b3_pipeline/` (main.py, adjustments.py, b3_corporate_actions.py, storage.py, parser.py, config.py)
- Backtests in `backtests/core/` with heuristic split detection in `shared_data.py`

## Key Architecture
- Splits fetched from B3 API endpoint `GetListedSupplementCompany` using 4-char ticker root
- Only labels DESDOBRAMENTO, GRUPAMENTO, BONIFICACAO are recognized; others silently dropped
- Additional labels exist: RESG TOTAL RV (delisting), CIS RED CAP (capital reduction), INCORPORACAO (merger)
- COTAHIST FATCOT field (pos 210-217) is NOT parsed -- 20% of records have fatcot=1000 (per-lot pricing)
- Backtest layer has heuristic split detector for ratios [2,3,4,5,8,10] with 4% tolerance

## Confirmed Issues (diagnosed 2026-03-02)
- B3 API returns empty stockDividends for many companies (confirmed: EQPA, ADMF, CCTY, GOLL, etc.)
- 520 Server Errors and timeouts in fetch.log (no retry logic) -- 31 companies failed
- ~1,119 unrecorded splits since 2000 (677 match standard ratios, 442 non-standard)
- ~114 unrecorded splits since 2020 across 81 tickers
- fatcot=1000 affects 76% of records in 2000, ~70-87% in 1995-2003, declining to 0% by 2017
- Double-adjustment risk: fatcot normalization + API split on same date
- See [b3-split-gaps.md](b3-split-gaps.md) for detailed findings

## Database Tables
- prices: (ticker, isin_code, date, open/high/low/close, volume, split_adj_*, adj_close) -- NO quotation_factor column
- stock_actions: (isin_code, ex_date, action_type, factor, source) - 1007 records, only 446 unique company codes
- corporate_actions: (isin_code, event_date, event_type, value, factor, source)
- detected_splits: (isin_code, ex_date, split_factor, description)

## Debugging Patterns
- To check if a split is in the API: encode {issuingCompany, language} as base64, query GetListedSupplementCompany
- To check COTAHIST fatcot: read positions 210-217 from fixed-width line
- API returns JSON as string that needs json.loads() double-parse
- 520 errors from B3 are transient server errors, not auth issues
