# B3 Split Data Gaps - Detailed Findings

## Confirmed Missing Splits (B3 API returns empty stockDividends)

| Ticker | ISIN | Jump Date | Close Before -> After | Ratio | Likely Event |
|--------|------|-----------|----------------------|-------|-------------|
| EQPA7 | BREQPAACNPC0 | 2026-01-22 | 5.51 -> 10.00 | 1.81 | ~2:1 reverse |
| TKNO3 | BRTKNOACNOR8 | 2025-07-28 | 76.01 -> 180.00 | 2.37 | ~2:1 reverse |
| TKNO4 | BRTKNOACNPR5 | 2025-07-28 | 77.71 -> 163.19 | 2.10 | 2:1 reverse |
| ADMF3 | BRADMFACNOR3 | 2025-10-03 | 70.00 -> 35.02 | 0.50 | 2:1 forward |
| CCTY3 | BRCCTYACNOR7 | 2025-05 (multiple) | multiple jumps | varies | complex |
| GOLL54 | BRGOLLA01PR5 | 2025-06/07 (multiple) | multiple jumps | varies | complex |
| BSLI3 | BRBSLIACNOR5 | 2025-03-31 | 7.49 -> 13.74 | 1.83 | ~2:1 reverse |
| BSLI4 | BRBSLIACNPR2 | 2025-03-31 | 6.83 -> 13.00 | 1.90 | 2:1 reverse |

## Unrecognized B3 Labels

Found in API but dropped by parser:
- `RESG TOTAL RV` - Total redemption of shares (CCTY has this with factor=100)
- `CIS RED CAP` - Capital reduction via share cancellation
- `INCORPORACAO` - Share incorporation/merger

## Relevant Code Locations

- Parser label filter: `b3_pipeline/b3_corporate_actions.py` lines 345-350
- Adjustment logic: `b3_pipeline/adjustments.py` lines 86-108
- Backtest heuristic: `backtests/core/shared_data.py` lines 24-109
- COTAHIST FATCOT position: lines 210-217 in fixed-width format
- Config split thresholds: `b3_pipeline/config.py` lines 77-78

## Statistics (as of 2026-03-02)
- 1,007 stock_actions in DB
- 114 unrecorded large jumps since 2020 (split_adj == raw close)
- 81 unique tickers affected
- 26 match standard ratios, ~24 non-standard
