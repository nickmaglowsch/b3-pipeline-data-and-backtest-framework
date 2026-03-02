# Code Review Report: Bug Fix for Missing Split Adjustments

## Summary

This fix addresses all four root causes identified in the bug diagnosis across all six target files and adds four well-structured test modules (25/25 passing). The implementation is well-organized, follows existing codebase patterns, and handles the most impactful issues. However, there is one **critical gap**: the double-adjustment risk when a fatcot transition coincides with a B3 API split is identified in the diagnosis as "High risk" and recommended for explicit mitigation, but **no deduplication logic was implemented**. This could cause incorrect adjusted prices for stocks that transitioned from lot-based to share-based pricing. The fix should not ship without at least a documented plan for this scenario, and ideally an implementation of the `FATCOT_REDUNDANT` filtering described in the diagnosis.

## PRD/Diagnosis Compliance

| # | Requirement (from bug-diagnosis.md) | Status | Notes |
|---|-------------------------------------|--------|-------|
| 1 | **Phase 1a**: Parse `fator_cotacao` from positions 210-217 | Complete | `parser.py:117` correctly reads `line[210:217]` via `_parse_int`, defaults to 1 if <= 0 |
| 2 | **Phase 1b**: Normalize OHLC prices by dividing by quotation_factor | Complete | `parser.py:121-125` divides open/high/low/close; volume correctly excluded |
| 3 | **Phase 1c**: Store quotation_factor in database | Complete | New column in `SCHEMA_PRICES`, included in `upsert_prices`, migration via `_migrate_schema` |
| 4 | **Phase 1 nuance**: Detect and skip API splits that correspond to fatcot transitions (double-adjustment prevention) | Missing | The diagnosis explicitly recommends scanning for cases where a fatcot transition matches a B3 API split on the same date and marking the API event as `FATCOT_REDUNDANT`. No such logic exists anywhere in the codebase. See Critical issue #1. |
| 5 | **Phase 2a**: Move heuristic split detector into `adjustments.py` | Complete | `detect_splits_from_prices()` at line 352, 165 lines, well-documented algorithm |
| 6 | **Phase 2b**: Integrate detection into pipeline (between step 6 and step 7) | Complete | `main.py:159-173` runs detection, upserts results, reloads stock_actions |
| 7 | **Phase 3a**: Log dropped labels instead of silently continuing | Complete | `b3_corporate_actions.py:383-410` logs both known non-split labels and truly unrecognized labels |
| 8 | **Phase 3b**: Add `skipped_events` table | Complete | `storage.py:72-81`, exact schema as specified in diagnosis |
| 9 | **Phase 3c**: Do NOT auto-include RESG TOTAL RV | Complete | Stored with `reason='delisting_event'`, not treated as split |
| 10 | **Phase 3d**: Flag CIS RED CAP and INCORPORACAO for manual review | Complete | Stored with `reason='needs_manual_review'` |
| 11 | **Phase 4a**: Add `fetch_failures` table | Complete | `storage.py:84-93`, matches diagnosis specification |
| 12 | **Phase 4b**: Track failures in `fetch_company_data` | Complete | `b3_corporate_actions.py:164-170` records failures on `RequestException` |
| 13 | **Phase 4c**: Add `--retry-failures` CLI flag | Complete | `main.py:301-305` arg, `retry_failed_companies()` implementation at line 214 |
| 14 | **Phase 4d**: Exponential backoff (optional) | Not implemented | Diagnosis marked this as optional. Acceptable to defer. |
| 15 | **Test 1**: Parser fatcot normalization test | Complete | 7 tests in `test_parser_fatcot.py` covering fatcot=1000, 1, 0, 10000, OHLC, volume, storage |
| 16 | **Test 2**: Split detection test | Complete | 6 tests in `test_split_detection.py` covering forward/reverse detection, existing-action skip, fatcot-transition skip, normal-movement no-fire, source label |
| 17 | **Test 3**: Label handling test | Complete | 7 tests in `test_label_handling.py` covering all three non-split labels, 3-tuple return, existing label regression, mixed records |
| 18 | **Test 4**: Failure tracking test | Complete | 5 tests in `test_failure_tracking.py` covering table creation, insertion, filtering, resolution, upsert-on-duplicate |

**Compliance Score**: 15/18 requirements fully met (1 critical missing, 1 optional deferred, 1 pre-existing not in scope)

## Debug-Specific Review Criteria

### 1. Was the root cause identified in the diagnosis actually addressed by the fix?

**Yes, all four root causes are addressed:**
- Root Cause 1 (API data gaps): Split detection from price data fills the gap.
- Root Cause 2 (Unrecognized labels): Known non-split labels are captured in `skipped_events` rather than silently dropped.
- Root Cause 3 (Ignored `fator_cotacao`): Parser now reads and normalizes by the quotation factor.
- Root Cause 4 (No failure tracking): `fetch_failures` table + `--retry-failures` CLI flag.

However, the **mitigation** for the highest-risk interaction between Root Cause 3's fix and existing B3 API splits is missing (see Critical #1 below).

### 2. Are there any regressions introduced by the fix?

No functional regressions were identified in the changed code. All call sites for `parse_stock_dividends` and `fetch_all_corporate_actions` have been updated to handle the new 3-tuple return value. The backward-compatible handling in `test_label_handling.py` (lines 96-100, 114-117) suggests awareness that other code might still use 2-tuple destructuring, but a full search confirms no remaining 2-tuple callers exist -- this is defensive test code, not a sign of missed callers.

### 3. Was a test written for the bug?

**Yes.** Four test files covering all four phases, 25 tests total, all passing. The tests are well-structured with clear GIVEN/WHEN/THEN docstrings and test realistic scenarios including edge cases. One missing test scenario is noted in Important #3 below.

### 4. Are the changes minimal and focused on the bug fix?

**Yes.** All changes directly address the four root causes and their interactions. There are no unrelated refactors, no cosmetic changes, and no feature additions beyond what the diagnosis prescribed. The only unused artifact is `COMMON_SPLIT_RATIOS` in `config.py` (see Minor #1).

## Issues Found

### Critical (must fix before shipping)

1. **Double-adjustment risk: fatcot + B3 API split overlap not mitigated**
   - The diagnosis (section "High risk: Double-adjustment from fatcot normalization + API splits") explicitly states: "After fatcot normalization, compare the normalized price series for continuity. If a fatcot transition on date D produces a continuous series AND a B3 API split exists on the same date with a matching factor, mark the API split as `source='FATCOT_REDUNDANT'` and skip it."
   - **No such logic exists.** The `detect_splits_from_prices` function (adjustments.py:444-458) handles fatcot transitions for its own heuristic detection, but the pre-existing B3 API splits in the `stock_actions` table are never filtered against fatcot transitions. This means `compute_split_adjustment_factors` will apply the B3 API split factor to prices that have already been normalized by the parser's fatcot division.
   - **Concrete scenario**: A stock like BEES3 transitions from fatcot=1000 to fatcot=1 in 2012. The parser now normalizes all pre-2012 prices by dividing by 1000, making the series continuous. But if B3's API also reports a GRUPAMENTO with factor matching 1/1000, that factor will be applied by `compute_split_adjustment_factors`, dividing all pre-2012 prices by an additional 1000x. The resulting `split_adj_close` would be 1,000,000x too small.
   - **Recommendation**: Before `compute_all_adjustments`, add a filtering step that cross-references `stock_actions` entries against `quotation_factor` transitions in the prices table. When a fatcot transition on date D has a ratio matching a B3 API split on the same date (within tolerance), remove or flag that B3 split as redundant. The diagnosis recommended marking these as `source='FATCOT_REDUNDANT'`.

### Important (should fix)

2. **`retry_failed_companies()` does not fetch cash dividends**
   - **File**: `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/main.py:252-262`
   - The retry function only processes `stockDividends` from the company data. The main pipeline (lines 115-120 in the normal flow) also calls `fetch_cash_dividends_paginated` with the `full_trading_name` to get cash dividends. The retry path skips this entirely.
   - **Impact**: Companies whose initial fetch failed will have their stock splits recovered but not their cash dividends. This means `adj_close` (which incorporates dividend adjustments) will still be incorrect for these companies even after retry succeeds.
   - **Recommendation**: Extract the full processing logic (stock dividends + cash dividends) into a shared helper, or add the cash dividend fetch to `retry_failed_companies`.

3. **No test for the double-adjustment scenario**
   - The diagnosis recommends (Test Strategy, item 6): "For a ticker that transitioned from fatcot=1000 to fatcot=1, verify: If B3 API also reports a split for that date, it is NOT double-applied."
   - No such test exists. Even if the deduplication logic in Critical #1 is implemented, a regression test specifically for this scenario is essential to prevent future breakage.

4. **`detect_splits_from_prices` runs on all 2.5M records in an O(N) Python loop**
   - **File**: `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/adjustments.py:404-503`
   - The function iterates per-ISIN with a Python for-loop over every row. While the diagnosis estimates <30 seconds, the actual time depends on the number of ISINs (2,578 unique ticker roots). The inner loop at line 414 (`for i in range(1, len(closes))`) is pure Python on NumPy arrays, which is fast, but the existing-key lookup at lines 431-435 does a linear scan over a 5-day window for each candidate -- this is O(N * 5) per ISIN per jump. For 2.5M records across ~2,500 ISINs, this should be acceptable but is worth monitoring.
   - **Recommendation**: The existing implementation is likely fast enough. Consider profiling on real data to confirm. If it becomes slow, vectorize with `groupby().pct_change()` and boolean masking.

### Minor (nice to fix)

5. **`COMMON_SPLIT_RATIOS` in config.py is unused dead code**
   - **File**: `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/config.py:80-93`
   - This tuple-of-tuples list was added but is never referenced. The `detect_splits_from_prices` function defines its own `_common_ratios = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50, 100]` locally. Either use the config constant or remove the dead code.

6. **`fetch_company_data` failure tracking does not cover JSON/parse errors**
   - **File**: `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/b3_corporate_actions.py:172-177`
   - Only `requests.RequestException` triggers `record_fetch_failure`. The `json.JSONDecodeError`, `ValueError`, `AttributeError`, and `TypeError` handlers at lines 172-177 log warnings but do not record failures. If B3 returns a 200 OK with malformed JSON, the failure is not tracked and `--retry-failures` will not know to re-attempt.

7. **Step numbering in main.py log output is confusing**
   - **File**: `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/main.py:159`
   - The split detection step is labeled "Step 6b" but it appears inside what is ostensibly "Step 7/9: Computing adjustments..." (line 150). The log output would show "Step 7/9: Computing adjustments..." followed by "Step 6b: Detecting missing splits..." which is temporally backwards. Consider renumbering to 10 steps or placing the log message before the step 7 header.

8. **Pre-existing observation (not a regression): close price field position**
   - **File**: `/Users/nickmaglowsch/person-projects/b3-data-pipeline/b3_pipeline/parser.py:107`
   - The parser reads `close_price = _parse_price(line[108:121])`. According to the COTAHIST layout in `config.py`, position 108-121 is `preco_medio` (average price), while `preco_ultimo_negocio` (last trade / close) is at position 121-134. The config comment says "This layout dictionary is for reference only and may not exactly match the field positions used in parser.py" -- one of these is wrong. This predates the current fix and is noted only for awareness.

## What Looks Good

- **Thorough 4-phase implementation** that directly maps to the diagnosis. Each root cause has a clear, targeted fix.
- **`_migrate_schema()` function** (storage.py:143-154) ensures backward compatibility with existing databases by adding the `quotation_factor` column non-destructively via `ALTER TABLE ... ADD COLUMN`. This is the right approach and prevents data loss during upgrades.
- **Defensive coding in parser** (`quotation_factor <= 0` defaults to 1, and `if quotation_factor != 1` avoids unnecessary division).
- **`source='DETECTED'` labeling** provides clear auditability -- pipeline-detected splits can be distinguished from B3 API splits in queries and manual review.
- **Skipped events with structured reasons** (`delisting_event`, `needs_manual_review`, `unrecognized_label`) enable future automation and systematic manual review.
- **Failure tracking with retry_count increment** on duplicate failures (storage.py:507) is a nice touch for monitoring persistent failures.
- **Test quality is high**: tests use real-format COTAHIST lines, cover edge cases (fatcot=0, fatcot=10000), and use clear assertion messages. The `_build_cotahist_line` helper in `test_parser_fatcot.py` is well-constructed and reusable.
- **No secrets, no injection risks** -- the changes are all internal data processing with no new user-facing inputs beyond the existing `--retry-failures` CLI flag.
- **All existing tests still pass** (25/25) confirming no regressions in the test suite.

## Recommendations

1. **[Critical]** Implement fatcot/API-split deduplication before shipping. Add a function (e.g., `filter_fatcot_redundant_splits`) that runs after loading `stock_actions` and before `compute_all_adjustments`. It should cross-reference each stock_action against the `quotation_factor` column in prices for the same ISIN/date and remove actions where the split factor matches the fatcot transition ratio. Add a test for this scenario using the BEES3 or CBEE3 examples from the diagnosis.

2. **[Important]** Add cash dividend fetching to `retry_failed_companies()` or document why it is intentionally omitted.

3. **[Important]** Add a specific unit test for the double-adjustment scenario (fatcot transition + matching B3 API split on same date/ISIN).

4. **[Minor]** Remove `COMMON_SPLIT_RATIOS` from `config.py` or wire it into `detect_splits_from_prices` to avoid dead code.

5. **[Minor]** Consider extending failure tracking to cover JSON parse errors, not just HTTP transport errors.

6. **[Minor]** Fix the step numbering in pipeline log output so "Step 6b" does not appear after "Step 7/9".
