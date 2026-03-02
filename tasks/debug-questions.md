# Debug Questions

## Investigation Summary

The investigation confirmed the user's suspicion: many stock splits are genuinely absent from the B3 API. Here is what was found:

### Confirmed root causes for missing splits

**1. B3 API does not return all split events.** Of 2,578 unique ticker roots in the prices table, only 446 have any stock_actions at all (from B3's `GetListedSupplementCompany` endpoint). Since 2020, there are **114 large price jumps** (ratio > 1.8x or < 0.55x) across **81 unique tickers** where the `split_adj_close` equals the raw `close` -- meaning no adjustment was applied because no split data exists. Examples include EQPA7 (2:1 reverse split Jan 2026), TKNO3/TKNO4 (2:1 reverse splits in 2025), CCTY3 (multiple jumps in 2025), ADMF3, GOLL54, BSLI3/BSLI4, and many more.

**2. Unrecognized B3 API labels.** The B3 API `stockDividends` field returns events with labels like `"RESG TOTAL RV"` (total redemption), `"CIS RED CAP"` (capital reduction), and `"INCORPORACAO"` (merger/incorporation). The parser in `b3_corporate_actions.py` (line 345-350) only recognizes `DESDOBRAMENTO`, `GRUPAMENTO`, and `BONIFICACAO`, silently dropping these other event types.

**3. API fetch failures (520 errors and timeouts).** The `fetch.log` shows many 520 Server Error and read timeout failures. The code catches these exceptions and moves on with a `logger.warning()` -- meaning splits for those companies are permanently lost unless the pipeline is rerun.

**4. No retry logic.** Failed API calls are not retried, so transient B3 server errors cause permanent gaps.

### What works correctly

- When splits ARE present in the B3 API, the factor calculation and backward cumulative adjustment logic in `adjustments.py` works correctly (verified for SEQL3, PDGR3, IFCM3, etc.).
- The `lastDatePrior` date interpretation is correct (used as the last day to apply the adjustment factor, with the actual price change happening on the next trading day).
- The backtest layer (`shared_data.py` lines 24-109) has a heuristic detector that catches some unrecorded splits, but it only works for standard ratios (2, 3, 4, 5, 8, 10) and the detection happens at backtest time, not pipeline time.

### Scale of the problem

- 1,007 stock_actions in the database (from B3 API)
- ~114 unrecorded large price jumps since 2020 alone (where split_adj = raw close)
- Total large jumps since 2020: 334 across 249 tickers (some have stock_actions for OTHER dates but not the specific jump date)
- The unrecorded jumps break down roughly as: 26 matching standard ratios (2:1, 3:1, etc.), ~24 with non-standard ratios

---

## Questions

### Q1: Which stocks are you specifically noticing problems with?
**Context:** The investigation found 81 tickers with unrecorded splits since 2020. Some are highly liquid (like EQPA7, GOLL54), while others are micro/small caps. Knowing which specific stocks matter most will help prioritize the fix approach.
**Question:** Can you name the specific tickers where you noticed incorrect adjusted prices? Are these mostly liquid, actively-traded stocks, or smaller/less liquid names?
**Options:**
1. Mostly liquid blue-chips and mid-caps (e.g., EQPA, GOLL, BSLI)
2. Mostly small-cap and illiquid stocks
3. A mix of both
4. I don't have specific tickers -- I noticed it as a general issue in backtest results

### Q2: How should non-standard stock dividend labels be handled?
**Context:** The B3 API returns `stockDividends` entries with labels beyond DESDOBRAMENTO/GRUPAMENTO/BONIFICACAO. Specifically: `"RESG TOTAL RV"` (total redemption of shares), `"CIS RED CAP"` (capital reduction via share cancellation), and `"INCORPORACAO"` (share incorporation/merger). Currently, the parser in `b3_pipeline/b3_corporate_actions.py` silently ignores these. For example, CCTY3 has a "RESG TOTAL RV" event with factor=100, which was dropped.
**Question:** Should these additional label types be treated as stock splits and applied as price adjustments?
**Options:**
1. Yes -- treat them all as split-like events and include them
2. Only include specific ones (specify which)
3. No -- these are fundamentally different events and should not be treated as splits
4. I'm not sure -- need to investigate each type individually

### Q3: Should the pipeline detect splits from price data as a fallback?
**Context:** Currently the pipeline only uses B3 API data for splits. A heuristic split detector already exists in the backtest layer (`backtests/core/shared_data.py` lines 24-109) that catches unrecorded splits by looking for overnight price jumps matching common ratios (2:1, 3:1, etc.). This detector could be moved into the pipeline itself to catch the ~114 missing splits. The COTAHIST `fator_cotacao` field (position 210-217) is also available but currently ignored by the parser.
**Question:** What approach do you prefer for filling in the gaps left by the B3 API?
**Options:**
1. Move the heuristic split detector from backtests into the pipeline (detect from price jumps, store in stock_actions)
2. Parse the COTAHIST `fator_cotacao` field to detect quotation factor changes (more authoritative but complex)
3. Both: use `fator_cotacao` as primary, price-jump detection as secondary fallback
4. Use a third-party data source (e.g., Yahoo Finance split data) to cross-reference
5. Keep detection at the backtest level only (current approach)

### Q4: How should API fetch failures be handled?
**Context:** The `fetch.log` shows many 520 Server Errors and timeouts from the B3 API. The current code catches exceptions and continues without retrying, meaning companies that fail during a pipeline run permanently miss having their splits recorded (until the next full rebuild). There is also no tracking of which companies failed so they can be retried selectively.
**Question:** What level of retry/resilience should be added?
**Options:**
1. Add retry with exponential backoff (e.g., 3 retries per request)
2. Track failed companies and allow selective re-fetching
3. Both retry logic and failure tracking
4. The current approach is fine -- I just rerun the pipeline when needed

### Q5: What time period matters most for correctness?
**Context:** The missing-splits problem affects all time periods, but the impact differs: pre-2010 data has additional complications from quotation-factor changes (the FATCOT issue documented in `docs/todo.md`), while post-2010 data mostly just needs the missing splits filled in. The backtest heuristic can partially compensate at runtime.
**Question:** Which time period is most critical for your backtests?
**Options:**
1. Only recent data (2015+) -- older data quality is not critical
2. Full history (2000+) -- I need accurate adjustments all the way back
3. Full history (1994+) -- including pre-Real plan era
4. Only the last 5 years (2021+)

### Q6: Should the fix be applied retroactively to existing data?
**Context:** The current database has 2.5M price records. Fixing the split issue requires either (a) re-running the pipeline with improved logic (`--rebuild`), or (b) running a targeted fix script that detects and applies missing adjustments to existing data without re-downloading everything.
**Question:** What is your preference for applying the fix?
**Options:**
1. Full rebuild with improved pipeline logic (cleaner but takes longer)
2. Targeted fix script that patches existing data (faster but may miss edge cases)
3. Both: targeted fix now, then rebuild when convenient
