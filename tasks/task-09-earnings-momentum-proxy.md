# Task 9: Earnings Momentum / Revision Proxy Strategy

## Objective
Build a strategy that captures earnings surprise / revision effects using only price and volume data (no fundamental data source needed). The key insight: stocks that gap up on abnormally high volume likely reported positive earnings surprises. This "volume-confirmed momentum" is a different signal from standard momentum and may add diversification benefit.

## Context
The repo has no fundamental data (no earnings, book value, P/E ratios). However, B3's quarterly reporting windows are clustered in time (March-April for Q4 results, May for Q1, August for Q2, November for Q3). Price-volume behavior during these windows can serve as a proxy for earnings quality:
- A stock that rises 5% on 3x normal volume during reporting season likely had a positive earnings surprise
- A stock that drops 10% on high volume likely disappointed

This is conceptually different from the existing signals because it focuses on EVENT-DRIVEN behavior rather than trailing averages. The existing `volume_breakout_backtest.py` looks at volume acceleration unconditionally, but this task specifically targets reporting windows.

The existing ML research (`research/output/research_summary.txt`) found that Return_60d was a top-10 feature in XGBoost -- this aligns with the idea that medium-term price reactions encode fundamental information.

## Requirements
- Create a new file `backtests/earnings_proxy_backtest.py`
- Implement three variants:

### Variant 1: Volume-Confirmed Momentum
Rank stocks by their return during reporting months (March-May, August, November) weighted by the volume ratio (actual volume / 6-month average volume). Higher return on higher relative volume gets a higher score.

```
signal = return_1m * max(1, volume_ratio)
```

Hold the top decile for the next month. During non-reporting months, use standard 6-month momentum.

### Variant 2: Post-Earnings Drift
After a reporting month, stocks with the highest returns tend to continue drifting upward for 2-3 months (post-earnings announcement drift, or PEAD). Buy the top decile of reporting-month winners and hold for 3 months, regardless of intermediate signals. This reduces turnover.

### Variant 3: Combined Earnings + Regime
Apply the earnings proxy signal only during COPOM easing cycles. During tightening, sit in CDI. This combines the earnings signal with the proven regime filter.

### For all variants:
- Universe: ADTV >= R$1M, price >= R$1.0, standard lot tickers
- Rebalance: Monthly
- Tax: 15% CGT, 0.1% slippage, R$20K monthly exemption
- Period: 2005-present
- Compare against: MultiFactor, COPOM Easing, IBOV, CDI

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/data.py` -- `load_b3_data()` returns adj_close, close_px, fin_vol
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/volume_breakout_backtest.py` -- Pattern for volume-based signals. It computes volume acceleration as `period_vol.shift(1) / period_vol.shift(1).rolling(lookback).mean()` (around line 60). Adapt this for earnings windows.
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/copom_momentum_backtest.py` -- Pattern for combining a stock-level signal with COPOM regime filtering
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/simulation.py` -- `run_simulation()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `build_metrics()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/plotting.py` -- `plot_tax_backtest()` for the standard 4-panel tear sheet

## Implementation Details

### B3 Reporting Season Calendar
B3 companies typically report quarterly results following this schedule:
- **Q4 annual results**: March-April (deadline: March 31 for DFP filing)
- **Q1 results**: May (ITR filing)
- **Q2 results**: August (ITR filing)
- **Q3 results**: November (ITR filing)

So the "reporting months" are approximately: March, April, May, August, November.

Define reporting months:
```python
REPORTING_MONTHS = {3, 4, 5, 8, 11}
```

### Volume Ratio Computation
```python
# Monthly volume
period_vol = fin_vol.resample("ME").sum()
avg_vol_6m = period_vol.shift(1).rolling(6).mean()
volume_ratio = period_vol.shift(1) / avg_vol_6m
volume_ratio = volume_ratio.clip(upper=5.0)  # cap to avoid extreme outliers
```

### Signal Construction (Variant 1)
```python
ret_1m = px.pct_change()  # last month's return
is_reporting = pd.Series(px.index.month.isin(REPORTING_MONTHS), index=px.index)

# In reporting months: volume-weighted return
# In non-reporting months: standard 6-month momentum
signal = pd.DataFrame(index=px.index, columns=px.columns)
for i in range(7, len(px)):
    if is_reporting.iloc[i-1]:  # last month was a reporting month
        signal.iloc[i] = ret_1m.iloc[i-1] * volume_ratio.iloc[i-1].clip(lower=1.0)
    else:
        signal.iloc[i] = log_ret.shift(1).rolling(6).sum().iloc[i]
```

### Post-Earnings Drift (Variant 2)
Track a "holding period" for each selected stock. When a stock enters the portfolio during a reporting month, it is held for 3 months regardless of signal changes. This requires maintaining a "hold_until" dict:
```python
hold_until = {}  # {ticker: expiry_month_index}
```

### Output Files
- `earnings_proxy_backtest.png` -- Standard 4-panel tear sheet for the best variant
- Console: comparison table of all 3 variants + benchmarks

## Acceptance Criteria
- [ ] File `backtests/earnings_proxy_backtest.py` exists and runs successfully
- [ ] All 3 variants produce valid equity curves
- [ ] Volume-confirmed momentum signal is meaningfully different from standard momentum (correlation < 0.85)
- [ ] Results clearly show whether earnings proxy adds value vs standard MultiFactor
- [ ] If any variant achieves Sharpe > 0.3, report its correlation with existing strategies
- [ ] Tear sheet saved for the best variant
- [ ] The reporting month calendar is correct for B3

## Dependencies
- Depends on: Task 1 (data loading patterns), general familiarity with strategy structure from Tasks 2-5
- Blocks: None
