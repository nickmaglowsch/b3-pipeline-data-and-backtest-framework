# TODO: COTAHIST Quotation Factor (FATCOT) Support

## Problem

B3's COTAHIST format includes a field called **FATCOT** (fator de cotacao) that indicates how a stock's price is quoted — per share, per lot of 1000, per lot of 1M, etc. Our parser ignores this field, which causes incorrect price adjustments for stocks that changed their quotation basis over time.

### How it manifests

When a stock like VSPT3 (Ferrovia Centro-Atlantica) had a grupamento (reverse split) of 3,333,333:1 in 2017, B3's API correctly reports `factor: "0,00000030000"`. The adjustment code multiplies all pre-grupamento prices by 3,333,333x. But the pre-grupamento prices in COTAHIST were quoted **per lot**, not per share, so the multiplication is wrong:

- Raw close in 2010: R$2.80 (per lot, NOT per share)
- After adjustment: R$2.80 * 3,333,333 = R$9,333,333 (nonsensical)
- B3 API confirms: `"quotedPerSharSince": "23/10/2017"`

This affects many stocks that went through quotation basis changes, especially:
- Hyperinflation-era stocks (1980s-1990s) with massive bonus shares (99,900%)
- Stocks that transitioned from per-lot to per-share quotation
- Penny stocks that underwent extreme grupamentos

### Current workaround

Sanity capping in `b3_pipeline/adjustments.py`:
- Individual split factors capped to 1000:1 max
- Cumulative factors capped to 100,000x
- Final adj_close/close ratio check with fallback to split_adj_close

This prevents extreme values from reaching backtests but silently discards legitimate adjustments for affected stocks.

### Affected stocks (examples)

| Ticker | Event | B3 Factor | Result |
|--------|-------|-----------|--------|
| VSPT3 | Grupamento 2017 | 0.0000003 | 3,333,333x adjustment on lot-priced data |
| BEES3/4 | Desdobramento 2012 | 99,900 | 99,900x split on old quotation basis |
| MERC3/4 | Desdobramento 2023 | 99,900 | Combined with 0.001 grupamento same day |
| Various | Hyperinflation bonuses | 99,900-351,900 | 1980s/90s bonus shares |

## Proposed Fix

### 1. Parse FATCOT from COTAHIST

The COTAHIST fixed-width format has a quotation factor field. Add it to the parser:

```
Position 210-217 (0-indexed): FATCOT - Fator de Cotacao (quotation factor)
Format: integer
Value: number of shares per quoted unit (e.g., 1 = per share, 1000 = per lot of 1000)
```

In `b3_pipeline/parser.py`, add:
```python
quotation_factor = _parse_int(line[210:217])
```

### 2. Normalize prices to per-share

Before storing or adjusting prices, divide by the quotation factor:
```python
per_share_price = raw_price / quotation_factor
```

### 3. Track quotation basis changes

Store the quotation factor per ticker per date in the database. When applying split/grupamento adjustments, only adjust prices that are already on the same quotation basis, or normalize first.

### 4. Alternative approach: only adjust post-quotation-change data

For stocks with `quotedPerSharSince` dates from B3's API, only apply corporate action adjustments to data after that date. For earlier data, use the FATCOT field to normalize.

## References

- B3 COTAHIST layout specification (positions and field formats)
- B3 API field: `quotedPerSharSince` in GetListedSupplementCompany response
- FATCOT field documentation in B3's historical data manual
