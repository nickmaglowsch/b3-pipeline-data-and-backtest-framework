# Task 8: Seasonal / Calendar Effects Strategy

## Objective
Test whether B3 exhibits seasonal or calendar effects (turn-of-month, January/December effect, sell-in-May, pre-holiday) and build a strategy that exploits any robust effects found. Calendar anomalies are well-documented in academic literature for emerging markets and have NOT been tested in this repo.

## Context
Seasonal effects in Brazil could be particularly strong due to:
- **13th salary ("decimo terceiro")**: Brazilian workers receive an extra month's salary in November/December, driving retail inflows
- **Year-end portfolio window dressing**: Institutional fund managers rebalance in December
- **Dividend distribution cycles**: Many companies pay dividends in Q1 (after annual results)
- **Turn-of-month effect**: Index fund rebalancing and salary-driven investment at month boundaries

None of the 30+ existing backtests test calendar-based signals. All existing signals are price/volume-based (momentum, volatility, volume patterns).

## Requirements
- Create a new file `backtests/seasonal_effects_backtest.py`
- Test the following calendar effects on B3 data (2005-present):

### Effect 1: Turn-of-Month
Buy the market (top decile by momentum) on the last 2 trading days and first 3 trading days of each month. Sit in CDI otherwise. Academic literature shows this captures a disproportionate share of monthly returns.

### Effect 2: Monthly Seasonality
Compute the average return for each calendar month (Jan-Dec) across all years. Identify months with statistically significant positive returns. Build a strategy that is invested in equities only during historically positive months and in CDI otherwise.

### Effect 3: December/January Effect
Be invested in equities during November-January (13th salary / year-end effects) and CDI during other months. Test variants with different windows (Nov-Jan, Dec-Feb, etc.).

### Effect 4: Sell-in-May
Classic "sell in May, go away": invest in equities October-April, CDI May-September. Test whether this holds for B3.

### For each effect:
- Run through the standard simulation framework (15% CGT, 0.1% slippage, R$20K exemption)
- Compare against buy-and-hold IBOV, CDI, and the best existing strategy (COPOM Easing)
- Report metrics: Ann.Return, Sharpe, MaxDD, Calmar
- Report the calendar pattern itself (average return by month/day)

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/data.py` -- `load_b3_data()`, `download_benchmark()`, `download_cdi_daily()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/simulation.py` -- `run_simulation()` with monthly_sales_exemption parameter
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `build_metrics()`, `value_to_ret()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/plotting.py` -- `PALETTE`, `fmt_ax()`, `plot_tax_backtest()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/copom_easing_backtest.py` -- Pattern for a simple binary switch strategy (IBOV vs CDI). The seasonal strategies follow a similar structure: be in equities during favorable calendar periods, CDI otherwise.
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/compare_all.py` -- Pattern for running multiple strategy variants and comparing

## Implementation Details

### Data Handling
The turn-of-month effect requires daily-frequency analysis. Use daily adj_close data directly (not resampled to monthly). For the monthly strategies (seasonality, December effect, sell-in-May), use the standard monthly resampling.

### Turn-of-Month Implementation
```python
# Identify turn-of-month days (last 2 + first 3 trading days of each month)
daily_ret = adj_close.pct_change()
is_month_end = adj_close.index.to_series().groupby(adj_close.index.to_period('M')).transform('max') - adj_close.index <= pd.Timedelta(days=4)
is_month_start = adj_close.index - adj_close.index.to_series().groupby(adj_close.index.to_period('M')).transform('min') <= pd.Timedelta(days=4)
is_tom = is_month_end | is_month_start
```

For this effect, since we need daily data, either:
1. Use daily returns with weekly/monthly rebalancing to keep it compatible with the simulation engine
2. Compute the monthly return from only TOM days vs non-TOM days and use the standard monthly framework

Option 2 is simpler: compute the aggregate monthly return earned during TOM days vs non-TOM days, then weight the strategy accordingly in a monthly framework.

### Monthly Seasonality Analysis
```python
# Average return by calendar month
monthly_returns = ibov_px.resample("ME").last().pct_change()
monthly_returns.index = monthly_returns.index.month
seasonal_avg = monthly_returns.groupby(monthly_returns.index).mean()
seasonal_std = monthly_returns.groupby(monthly_returns.index).std()
seasonal_tstat = seasonal_avg / (seasonal_std / np.sqrt(monthly_returns.groupby(monthly_returns.index).count()))
```

### Stock Selection During Equity Months
For calendar strategies that are "in equities", use the existing MultiFactor signal (50% momentum + 50% low-vol, top 10%) for stock selection. This combines the calendar timing with proven stock selection.

Also test a simpler variant: just hold IBOV during favorable months (similar to COPOM Easing's approach).

### Output Files
- `seasonal_effects_analysis.png` -- Bar chart of average IBOV return by calendar month with significance markers
- `seasonal_effects_backtest.png` -- Equity curves of all 4 seasonal strategies vs benchmarks
- Console: metrics comparison table

## Acceptance Criteria
- [ ] File `backtests/seasonal_effects_backtest.py` exists and runs successfully
- [ ] Monthly seasonality bar chart clearly shows which months are historically positive/negative for B3
- [ ] All 4 calendar effects are tested with full simulation (tax, slippage, exemption)
- [ ] Results compared against IBOV, CDI, and COPOM Easing
- [ ] At least the analysis is complete -- even if no calendar effect beats COPOM Easing, the negative result is documented
- [ ] If any seasonal strategy has Sharpe > 0.3, its correlation with existing strategies is reported
- [ ] Both plots saved

## Dependencies
- Depends on: Task 1 (for data loading patterns and MultiFactor signal logic)
- Blocks: None
