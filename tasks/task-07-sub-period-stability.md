# Task 7: Sub-Period Stability Analysis

## Objective
Extend the portfolio comparison framework to report metrics across sub-periods (2005-2010, 2010-2015, 2015-2020, 2020-2026) and test whether portfolio optimization benefits are consistent across market regimes or concentrated in one era.

## Context
A strategy or portfolio that achieves a 0.70 Sharpe over 20 years might have achieved 1.40 in one decade and 0.00 in the other. This is critical information for deciding whether to deploy capital. Brazilian markets had very different regimes across these sub-periods:
- **2005-2010**: Commodity supercycle, pre/post-GFC, high CDI rates (~12-15%)
- **2010-2015**: Dilma government, Petrobras scandal, rising inflation, IBOV declined
- **2015-2020**: Impeachment, Temer/Bolsonaro reforms, COVID crash
- **2020-2026**: Post-COVID recovery, high CDI returning, global inflation

If portfolio optimization (HRP, risk parity, dynamic) only works in one sub-period, that is a warning sign.

## Requirements
- Create a new file `backtests/portfolio_stability_analysis.py`
- For each portfolio method from Task 5 (EqualWeight, InvVol, ERC, HRP, Dynamic Rolling Sharpe, Dynamic Regime, Dynamic Combined):
  1. Compute full-period metrics (2005-present)
  2. Compute metrics for each sub-period: 2005-2010, 2010-2015, 2015-2020, 2020-present
  3. Compute a "consistency score": how many sub-periods have positive excess return over CDI?
  4. Compute rolling 36-month Sharpe and report the minimum/maximum/median across time

### Outputs Required

#### 1. Sub-Period Table (console + CSV)
One table per portfolio method showing Ann.Return, Sharpe, MaxDD for each sub-period and the full period. Example format:
```
Method: HRP
Period          Ann.Ret%  Sharpe  MaxDD%
------          --------  ------  ------
2005-2010          12.3    0.45   -32.1
2010-2015           5.2    0.21   -18.4
2015-2020           9.8    0.62   -25.6
2020-2026          14.1    0.81   -12.3
Full (2005-2026)   10.2    0.52   -32.1
```

#### 2. Consistency Comparison Table
Compare all methods by their worst sub-period Sharpe. A robust method has a high minimum sub-period Sharpe.

#### 3. Heatmap Plot
A heatmap where rows = portfolio methods, columns = sub-periods, cell values = Sharpe ratio. Color-coded so you can visually see which methods are consistently good vs. inconsistent.

#### 4. Rolling Sharpe Plot
Rolling 36-month Sharpe for the top 3-4 portfolio methods overlaid on a single chart, with shaded recession/crisis periods.

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/strategy_returns.py` -- `build_strategy_returns()` for input data
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/portfolio_opt.py` -- All optimization functions from Tasks 2-4
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `build_metrics()`, `ann_return()`, `sharpe()`, `max_dd()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/plotting.py` -- `PALETTE`, `fmt_ax()` for consistent styling
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/portfolio_compare_all.py` -- Created in Task 5; reuse the portfolio construction code

## Implementation Details

### Sub-Period Computation
```python
SUB_PERIODS = [
    ("2005-2010", "2005-01-01", "2009-12-31"),
    ("2010-2015", "2010-01-01", "2014-12-31"),
    ("2015-2020", "2015-01-01", "2019-12-31"),
    ("2020-present", "2020-01-01", None),  # None = today
]
```

For each period, slice the portfolio return series and compute metrics using `build_metrics()` from `core/metrics.py`.

Important: For time-varying methods (HRP, dynamic), the weights at month t are computed using a trailing window. When analyzing the 2010-2015 sub-period, the weights at the start of 2010 should still use the 36-month trailing window starting from ~2007. So the weight computation should NOT be re-started at each sub-period boundary. Instead, run the full simulation once (2005-present) and then slice the resulting equity curves into sub-periods for metrics.

### Heatmap Implementation
Use matplotlib `imshow` with:
- Rows: ["EqualWeight", "InvVol", "ERC", "HRP", "DynRollSharpe", "DynRegime", "DynCombined"]
- Columns: ["2005-2010", "2010-2015", "2015-2020", "2020-present", "Full"]
- Values: Sharpe ratio
- Color map: "RdYlGn" (red=negative, green=positive)
- Annotate each cell with the numeric value
- Dark background matching PALETTE

### Rolling Sharpe Plot
```python
rolling_sharpe = returns.rolling(36).apply(
    lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
)
```

Add shaded regions for known stress periods:
- GFC: 2008-01 to 2009-03
- Dilma crisis: 2014-01 to 2016-06
- COVID: 2020-02 to 2020-06

### Output Files
- `portfolio_stability_heatmap.png` -- Sub-period Sharpe heatmap
- `portfolio_stability_rolling.png` -- Rolling 36-month Sharpe comparison
- `portfolio_stability_results.csv` -- Full results table

## Acceptance Criteria
- [ ] File `backtests/portfolio_stability_analysis.py` exists and runs successfully
- [ ] Sub-period metrics computed for all 7 portfolio methods across 4 sub-periods + full period
- [ ] Heatmap shows clear visual pattern of which methods are most consistent
- [ ] Rolling Sharpe plot covers 2005-present with stress period shading
- [ ] CSV with all results saved
- [ ] Console output includes "consistency score" ranking
- [ ] The analysis clearly identifies whether portfolio optimization benefits are regime-dependent

## Dependencies
- Depends on: Task 1, Task 2, Task 3, Task 4, Task 5 (uses the same portfolio construction code)
- Blocks: None (this is an analysis task)
