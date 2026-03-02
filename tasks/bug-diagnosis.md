# Bug Diagnosis

## Bug Summary

Many stock splits and share-count-changing corporate events are not being applied to adjust prices in the `prices` table, resulting in incorrect `split_adj_close` and `adj_close` values. The root cause is threefold: (1) the B3 API (`GetListedSupplementCompany`) simply does not return split data for a large number of companies, (2) the parser silently drops events with unrecognized labels like `RESG TOTAL RV`, `CIS RED CAP`, and `INCORPORACAO`, and (3) API fetch failures (520 errors and timeouts) are not retried or tracked. Since 2000, there are approximately **1,119 large unexplained price jumps** across the database where `split_adj_close` equals the raw `close` (no adjustment applied). The COTAHIST `fator_cotacao` field, which could serve as a primary fallback data source, is completely ignored by the parser -- this affects 76% of records in 2000, declining to near-zero by 2017.

## Root Cause

There are four distinct root causes that compound each other:

### Root Cause 1: B3 API data gaps (PRIMARY)

The B3 `GetListedSupplementCompany` endpoint returns an empty `stockDividends` array for many companies that clearly have had splits. Out of 2,578 unique ticker roots in the prices table, only 446 have any stock_actions records at all.

**Verified examples of API returning empty stockDividends:**
- `EQPA` (Equatorial Para): API returns `stockDividends: []`, but EQPA7 had a ~2:1 reverse split on 2026-01-22 (close jumped 5.51 -> 10.00)
- `ADMF` (Adm Participacoes): API returns `stockDividends: []`, but ADMF3 had a 2:1 split on 2025-10-03 (close dropped 70.00 -> 35.02)
- `TKNO` (Tekno): API returns only old events (2007, 2014) but TKNO3/TKNO4 had a ~2:1 reverse split on 2025-07-28 (close jumped 76.01 -> 180.00)
- `BSLI` (BRB Seguradora): API returns only old events (1994, 2005, 2021) but BSLI3/BSLI4 had a ~2:1 reverse split on 2025-03-31 (close jumped 7.49 -> 13.74)

The code at `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/b3_corporate_actions.py` line 430 simply checks `if stock_divs:` and moves on if empty -- there is no warning logged, no fallback attempted, and no tracking of which companies returned empty data.

### Root Cause 2: Unrecognized B3 API labels

The parser at `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/b3_corporate_actions.py` lines 345-352 only recognizes three labels:
- `DESDOBRAMENTO` (forward split)
- `GRUPAMENTO` (reverse split)
- `BONIFICACAO` (bonus shares)

The `else: continue` at line 352 silently drops all other labels. A survey of the B3 API across 250 companies found three additional labels that affect share count:

| Label | Meaning | Example | Factor |
|-------|---------|---------|--------|
| `RESG TOTAL RV` | Total redemption of shares | CCTY (factor=100), AHEB, GOLL | Always 100 |
| `CIS RED CAP` | Spin-off with capital reduction | ARMT (factor=100, factor=2.5), ARNC (factor=25) | Varies |
| `INCORPORACAO` | Merger/incorporation | Various | Varies |

`RESG TOTAL RV` with factor=100 means 100% of shares are redeemed (delisting event, not a split -- should NOT be treated as a split). `CIS RED CAP` and `INCORPORACAO` have variable factors and DO affect outstanding share counts, but their price impact semantics differ from standard splits and need individual investigation before mapping.

### Root Cause 3: COTAHIST `fator_cotacao` field is ignored

The COTAHIST fixed-width format includes a quotation factor field at positions 210-217. This field indicates how many shares one quoted unit represents:
- `0000001` = price is per 1 share (standard, modern)
- `0001000` = price is per lot of 1,000 shares
- `0010000` = price is per lot of 10,000 shares
- etc.

The parser at `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/parser.py` does NOT read this field. The data shows:

| Period | Records with fatcot != 1 | Percentage |
|--------|-------------------------|-----------|
| 1994 | 33,798 | 50.7% |
| 1995-2003 | ~300,000 | 70-87% |
| 2004-2007 | ~95,000 | 20-63% |
| 2008-2016 | ~12,000 | 0-5% |
| 2017+ | ~450 | <1% (GOLL54, IBOV11, SMLL11 only) |

For the user's target period (2000+), this means hundreds of thousands of price records are stored at lot-level prices (e.g., R$2,800 per lot of 1000 instead of R$2.80 per share). When a grupamento later converts the stock to per-share pricing, the B3 API factor (e.g., 3,333,333:1) is "correct" for lot-to-share conversion but produces nonsensical adjusted prices because the historical data was never normalized to per-share.

There are also 1,251 fatcot transitions across the full history where a ticker's quotation basis changes (almost all are `1000 -> 1`). Each transition acts like an implicit split in the price series that is invisible to the current adjustment logic.

### Root Cause 4: API fetch failures with no retry or tracking

The `fetch.log` at `/Users/nickmaglowsch/person-projects/b3-data-pipeline/fetch.log` shows:
- **22** 520 Server Errors
- **17** Read timeouts
- **31** unique company codes with failed `fetch_company_data` calls

Several of these are actively traded companies: `ARZZ` (Arezzo, 3,345 price records through 2024-07), `BMGB` (2 tickers, data through 2026-02), `BMI` (4 tickers, data through 2026-02), `DMVF` (data through 2026-02), `MEAL` (data through 2026-02).

The code at `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/b3_corporate_actions.py` lines 159-167 catches all exceptions and logs a warning, then returns `None`. The caller at line 423 does `if company_data is None: continue` -- no retry, no failure tracking.

## Evidence

### Database statistics
```
Total price records: 2,557,027
Records with fatcot=1 (per-share): 2,045,327 (80%)
Records with fatcot=1000 (per-lot): 511,464 (20%)
Stock actions in database: 1,007
Stock actions with NO matching prices (wrong ISIN): 115
Unrecorded large price jumps since 2000 (split_adj == raw close): 1,119
  - Standard ratio matches (detectable by heuristic): 677 (60.5%)
  - Non-standard ratios (not catchable by heuristic): 442 (39.5%)
Unrecorded large price jumps since 2020: 114 across 81 tickers
API fetch failures (last run): 31 unique companies
```

### Verified API emptiness (live queries)
```python
# EQPA: stockDividends=[], but EQPA7 split on 2026-01-22
# ADMF: stockDividends=[], but ADMF3 split on 2025-10-03
# CCTY: stockDividends=[{label: "RESG TOTAL RV", ...}], not a split
# GOLL: stockDividends=[{label: "RESG TOTAL RV", ...}], not a split
# TKNO: only old events (2007, 2014), missing recent 2025 splits
```

### Price jump verification
```
SEQL3: 2025-12-01 close=0.42, split_adj=4.20 (10:1 reverse split correctly applied)
EQPA7: 2026-01-22 close=10.00, split_adj=10.00 (NO adjustment, should be ~5.00)
ADMF3: 2025-10-03 close=35.02, split_adj=35.02 (NO adjustment, should be ~70.00)
```

## Affected Files

- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/parser.py` -- Must parse `fator_cotacao` field (positions 210-217) and store it, or normalize prices to per-share basis immediately
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/b3_corporate_actions.py` -- Must handle unrecognized labels (log them, investigate CIS RED CAP / INCORPORACAO), add retry logic, track failures
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/adjustments.py` -- Must add price-jump-based split detection as fallback, handle fatcot-based normalization
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/storage.py` -- Must add a `fetch_failures` table for tracking, may need to add `quotation_factor` column to prices
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/config.py` -- Must add new label constants, retry settings, split detection config
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/main.py` -- Must integrate the new detection step and failure tracking into the pipeline

## Fix Recommendations

The fix should be implemented in four phases, each self-contained and testable:

### Phase 1: Parse COTAHIST `fator_cotacao` and normalize prices (PRIMARY FIX)

This addresses Root Cause 3 and is the highest-impact change for 2000+ data quality.

**Step 1a: Add `quotation_factor` to parser output**

In `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/parser.py`, after line 108 (`isin_code = line[230:242].strip()`), add:
```python
quotation_factor = _parse_int(line[210:217])
if quotation_factor <= 0:
    quotation_factor = 1
```

Include it in the records dict (after the `volume` entry at line 119):
```python
"quotation_factor": quotation_factor,
```

**Step 1b: Normalize prices to per-share basis in the parser**

After reading the raw prices but before storing them, divide OHLC prices by the quotation factor. In the same records dict, modify the price fields:
```python
"open": open_price / quotation_factor,
"high": high_price / quotation_factor,
"low": low_price / quotation_factor,
"close": close_price / quotation_factor,
```

Keep the raw (unnormalized) `volume` as-is since it represents financial volume in BRL, not share count.

**Step 1c: Store quotation_factor in the database**

In `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/storage.py`, add a `quotation_factor INTEGER DEFAULT 1` column to the `prices` schema. Update `upsert_prices()` to include the new column. This preserves the original quotation basis for auditability.

**Important nuance:** When a stock transitions from fatcot=1000 to fatcot=1 (e.g., `BEES3` in 2012), the COTAHIST prices naturally go from lot-level to share-level. After normalization (dividing by fatcot), the prices will be continuous across the transition WITHOUT needing a separate split adjustment for that transition. However, B3's API may ALSO report a "DESDOBRAMENTO" for that transition -- so there is a risk of double-adjusting. The code must detect and skip API splits that correspond to quotation factor changes.

To handle this: after Phase 1, scan for cases where (a) a fatcot transition exists on the same date as (b) a B3 API stock_action. If the transition ratio matches the API factor, mark the API event as redundant and skip it during adjustment.

### Phase 2: Add pipeline-level price-jump split detection (SECONDARY FIX)

This addresses Root Cause 1 by detecting splits from price data when the API has no data.

**Step 2a: Move the heuristic detector from backtests into the pipeline**

Port the logic from `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/shared_data.py` lines 24-109 into a new function in `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/adjustments.py`. The algorithm should:

1. For each ISIN, compute consecutive-day price ratios from the `close` column.
2. Identify jumps where `ratio > 1.8` or `ratio < 0.55` within a 5-trading-day window.
3. Check if a `stock_actions` entry already exists for that ISIN and date range. If so, skip.
4. Check if a `quotation_factor` transition explains the jump. If so, skip.
5. For remaining jumps, attempt to match against common split ratios: `[2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50, 100]` with 8% tolerance (wider than the backtest's 4% to catch more splits, since pipeline detection can be manually audited).
6. For matching jumps, create a `stock_actions` entry with `source='DETECTED'` (not `'B3'`) so they can be distinguished.
7. For non-matching jumps that exceed a threshold (e.g., ratio > 3.0 or < 0.33), log a warning for manual review.

**Step 2b: Integrate detection into the pipeline**

In `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/main.py`, add a new step between step 6 (fetch corporate actions) and step 7 (compute adjustments):

```python
# Step 6b: Detect missing splits from price data
detected = adjustments.detect_splits_from_prices(prices_from_db, stock_actions)
if not detected.empty:
    storage.upsert_stock_actions(conn, detected)
    stock_actions = storage.get_all_stock_actions(conn)  # reload
```

### Phase 3: Log and investigate non-standard API labels

This addresses Root Cause 2.

**Step 3a: Log dropped labels instead of silently continuing**

In `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/b3_corporate_actions.py`, replace the `else: continue` block at line 352 with:
```python
else:
    logger.info(
        f"Unrecognized stockDividend label '{label}' for {isin_code} "
        f"on {date_str}, factor={factor_str}"
    )
    continue
```

**Step 3b: Add a `skipped_events` table**

In `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/storage.py`, add a new table:
```sql
CREATE TABLE IF NOT EXISTS skipped_events (
    isin_code TEXT NOT NULL,
    event_date DATE NOT NULL,
    label TEXT NOT NULL,
    factor REAL,
    source TEXT DEFAULT 'B3',
    reason TEXT,
    PRIMARY KEY (isin_code, event_date, label)
);
```

Store dropped events here for later manual review. This is especially important for `CIS RED CAP` and `INCORPORACAO` events, which may need custom handling.

**Step 3c: Do NOT auto-include `RESG TOTAL RV`**

Based on investigation, `RESG TOTAL RV` (total redemption) always has factor=100, meaning 100% share cancellation. This is a delisting event, not a price-affecting split. These should be logged to `skipped_events` but NOT treated as splits.

**Step 3d: Flag `CIS RED CAP` and `INCORPORACAO` for manual review**

These events have variable factors and DO affect share count. However, their price impact semantics are complex (spin-offs create new tickers, mergers involve share conversion ratios). For now, store them in `skipped_events` with `reason='needs_manual_review'`. A future enhancement could handle specific subtypes.

### Phase 4: Add API failure tracking and selective re-fetching

This addresses Root Cause 4.

**Step 4a: Add a `fetch_failures` table**

In `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/storage.py`:
```sql
CREATE TABLE IF NOT EXISTS fetch_failures (
    company_code TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    error_message TEXT,
    failed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    retry_count INTEGER DEFAULT 0,
    resolved BOOLEAN DEFAULT 0,
    PRIMARY KEY (company_code, endpoint)
);
```

**Step 4b: Track failures in `fetch_company_data`**

In `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/b3_corporate_actions.py`, modify the `fetch_company_data` function to accept and use a database connection for failure tracking. On catch of `RequestException`:
```python
if conn:
    storage.record_fetch_failure(conn, trading_name, "GetListedSupplementCompany", str(e))
```

**Step 4c: Add a `--retry-failures` CLI flag**

In `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/main.py`, add a mode that:
1. Reads `fetch_failures` where `resolved=0`
2. Re-attempts those specific companies
3. On success, marks `resolved=1`

**Step 4d: Add exponential backoff to HTTP requests (optional enhancement)**

If desired, wrap the `requests.get()` calls in a retry decorator (e.g., `tenacity` library or manual retry loop with `time.sleep(2 ** attempt)`). Keep max retries at 2 to avoid excessive load on B3's API.

## Test Strategy

### Unit tests

1. **Parser fatcot normalization test:**
   - Create a mock COTAHIST line with `fatcot=0001000` and close=280000 (representing R$2,800.00 per lot)
   - Verify the parsed close is 2.80 (R$2,800.00 / 1,000)
   - Test edge cases: fatcot=1 (no change), fatcot=0 (default to 1), fatcot=10000

2. **Split detection test:**
   - Create a DataFrame with a known 2:1 split pattern (close drops 50% overnight)
   - Verify the detector finds it, creates the right factor, and assigns `source='DETECTED'`
   - Test that it does NOT fire when a stock_action already exists for that date
   - Test that it does NOT fire when a fatcot transition explains the jump

3. **Label handling test:**
   - Pass a `stockDividends` record with `label='RESG TOTAL RV'` to `parse_stock_dividends`
   - Verify it is logged and stored in `skipped_events`, NOT treated as a split
   - Same for `CIS RED CAP` and `INCORPORACAO`

4. **Failure tracking test:**
   - Simulate a 520 error from the API
   - Verify a row is inserted into `fetch_failures`
   - Verify `--retry-failures` re-fetches and marks resolved on success

### Integration tests

5. **End-to-end pipeline test on a subset of tickers:**
   - Pick 5-10 tickers with known issues: EQPA7, ADMF3, TKNO3, BSLI3, GOLL54
   - Run the improved pipeline on just those tickers
   - Verify that `split_adj_close` is continuous across all known split dates
   - Verify that the adjusted prices match Yahoo Finance or other reference data (within 5% tolerance for dividend adjustment differences)

6. **Fatcot normalization regression test:**
   - For a ticker that transitioned from fatcot=1000 to fatcot=1 (e.g., `CBEE3` in 2016), verify:
     - Pre-transition prices are divided by 1000
     - Post-transition prices are unchanged
     - The price series is continuous (no 1000x jump at the transition)
     - If B3 API also reports a split for that date, it is NOT double-applied

7. **Full rebuild comparison:**
   - Run `--rebuild` with the improved pipeline
   - Compare adj_close values against a known-good reference (e.g., Yahoo Finance) for 20 liquid tickers
   - Verify total stock_actions count increased from 1,007 to approximately 1,500-2,000

### Manual spot checks

8. **Verify the 31 previously failed companies** now have data (or are tracked in `fetch_failures`)
9. **Verify GOLL54** (fatcot=1000 in 2025/2026) has correct per-share prices after normalization
10. **Check that the `skipped_events` table** is populated with `RESG TOTAL RV`, `CIS RED CAP`, `INCORPORACAO` entries

## Risk Assessment

### High risk: Double-adjustment from fatcot normalization + API splits

When a stock transitions from per-lot to per-share pricing, B3's API may report a corresponding `DESDOBRAMENTO` or `GRUPAMENTO`. If the pipeline normalizes the lot-level prices AND also applies the API split, the pre-transition prices would be adjusted twice (e.g., divided by 1000 from fatcot AND multiplied by the API split factor).

**Mitigation:** After fatcot normalization, compare the normalized price series for continuity. If a fatcot transition on date D produces a continuous series AND a B3 API split exists on the same date with a matching factor, mark the API split as `source='FATCOT_REDUNDANT'` and skip it.

### Medium risk: Heuristic false positives

The price-jump detector could flag legitimate large price moves (e.g., penny stocks, judicial recovery announcements) as splits. The BRPR3 (BR Properties) example from the data shows 3 large jumps in 2023 that may be real corporate events, not data errors.

**Mitigation:** Use `source='DETECTED'` for heuristically detected splits so they can be easily queried and audited. Add a conservative tolerance (8%) and only match standard ratios. Log non-matching jumps to a warnings file instead of auto-correcting.

### Medium risk: Breaking existing backtest results

The adjusted prices will change for many tickers after the fix, which means backtest results will differ from previous runs. Users may need to re-evaluate strategy performance.

**Mitigation:** Since the user requested a full rebuild, this is expected. The key is that the NEW results are correct. Consider generating a comparison report (old vs new adj_close) for the top 50 most liquid tickers.

### Low risk: API behavior changes

B3 may change their API response format or start returning more/fewer events. The fix should be resilient to this.

**Mitigation:** The `skipped_events` table captures anything the parser does not recognize, providing a natural audit trail for API changes.

### Low risk: Performance impact of price-jump detection

Scanning all 2.5M price records for jumps adds computation time.

**Mitigation:** The detection only needs to run once per pipeline execution (not per-ticker), and uses vectorized Pandas operations. Expected impact: <30 seconds additional on a full rebuild.
