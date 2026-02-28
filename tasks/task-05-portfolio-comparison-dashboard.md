# Task 5: Portfolio Comparison Dashboard

## Objective
Create a comprehensive comparison script that runs ALL portfolio optimization methods side-by-side with full metrics, equity curves, drawdowns, turnover analysis, tax impact, and rolling Sharpe comparison. This becomes the new canonical "portfolio analysis" script, superseding `portfolio_low_corr_backtest.py`.

## Context
After Tasks 2-4, we will have multiple portfolio construction methods (equal-weight, inverse-vol, ERC, HRP, three dynamic variants). We need a single script that runs them all, compares performance, and produces a definitive answer on which approach maximizes risk-adjusted return for a Brazilian individual investor (long-only, tax-aware, ADTV >= R$1M).

## Requirements
- Create a new file `backtests/portfolio_compare_all.py`
- Run all portfolio methods on the same strategy return data:
  1. **Equal Weight** -- Baseline 1/N allocation
  2. **Inverse Volatility** -- From Task 2
  3. **Equal Risk Contribution** -- From Task 2
  4. **HRP** -- From Task 3
  5. **Dynamic Rolling Sharpe** -- From Task 4
  6. **Dynamic Regime** -- From Task 4
  7. **Dynamic Combined** -- From Task 4
- Include benchmarks: IBOV and CDI
- All methods use: 2005-present, monthly rebalancing, R$20K monthly sales exemption applied at the individual strategy level (already embedded in the strategy returns from Task 1)

### Outputs Required

#### 1. Summary Table (console + saved to CSV)
For each portfolio method:
- Ann. Return (%)
- Ann. Volatility (%)
- Sharpe Ratio
- Max Drawdown (%)
- Calmar Ratio
- Avg Monthly Turnover (%)
- Total Months in Equity vs CDI (for regime methods)
- Final NAV (starting from R$100K)

#### 2. Multi-Panel Plot (saved to PNG)
Panel 1: Cumulative equity curves (log scale) for all methods + benchmarks
Panel 2: Drawdown comparison
Panel 3: Rolling 12-month Sharpe for each method
Panel 4: Metrics table

#### 3. Correlation Matrix of Portfolio Returns
Compute and display the correlation between the different portfolio methods' returns. This shows how much the methods actually differ in practice.

#### 4. Weight Analysis
For each time-varying method (InvVol, ERC, HRP, dynamic):
- Final-period weights
- Average weights over the full period
- Weight standard deviation (stability measure)

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/strategy_returns.py` -- `build_strategy_returns()` for input data
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/portfolio_opt.py` -- All optimization functions from Tasks 2-4
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `build_metrics()`, `ann_return()`, `sharpe()`, `max_dd()`, `calmar()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/plotting.py` -- `PALETTE`, `fmt_ax()`. Follow the dark-theme styling established here.
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/portfolio_low_corr_backtest.py` -- Pattern for the equity curve + metrics table layout (lines 298-440)
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/compare_all.py` -- Pattern for console output formatting (lines 361-380)

## Implementation Details

### Portfolio Construction Loop
```python
from core.strategy_returns import build_strategy_returns
from core.portfolio_opt import (
    inverse_vol_weights,
    equal_risk_contribution_weights,
    hrp_weights,
    rolling_sharpe_weights,
    regime_conditional_weights,
)

# 1. Load strategy returns
returns_df, sim_results = build_strategy_returns(monthly_sales_exemption=20_000)

# 2. Select strategies (exclude SmallcapMom for liquid-only, keep as a variant)
equity_strats = [col for col in returns_df.columns if col not in ("IBOV", "CDI")]

# 3. For each month t with enough history:
#    - compute weights for each method
#    - portfolio_return[method][t] = sum(w_i * returns_df[strategy_i][t])
```

### Rolling Sharpe Computation
For the rolling 12-month Sharpe panel, compute for each portfolio method:
```python
rolling_sharpe = portfolio_returns.rolling(12).apply(
    lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0
)
```

### Plot Styling
Follow the established dark-theme convention. Use the `PALETTE` dict from `core/plotting.py`:
- bg: "#0D1117"
- panel: "#161B22"
- grid: "#21262D"
- text: "#E6EDF3"

Assign distinct colors to each portfolio method. Use solid lines for the top 3-4 methods and dashed/dotted for the rest.

### CSV Export
Save the metrics table to `portfolio_comparison_results.csv` in the backtests directory for easy reference.

### Output Files
- `portfolio_compare_all.png` -- Main comparison plot (4 panels)
- `portfolio_compare_corr.png` -- Correlation heatmap of portfolio method returns
- `portfolio_comparison_results.csv` -- Metrics table

## Acceptance Criteria
- [ ] File `backtests/portfolio_compare_all.py` exists and runs successfully
- [ ] All 7 portfolio methods + 2 benchmarks are compared in a single run
- [ ] Console output shows a clear ranked table sorted by Sharpe
- [ ] 4-panel plot saved to `portfolio_compare_all.png`
- [ ] Correlation matrix of portfolio returns saved to `portfolio_compare_corr.png`
- [ ] CSV results saved to `portfolio_comparison_results.csv`
- [ ] The best portfolio method is clearly identifiable from the outputs
- [ ] Weight analysis section shows final, average, and stability metrics for time-varying methods
- [ ] Script runs end-to-end without errors

## Dependencies
- Depends on: Task 1, Task 2, Task 3, Task 4
- Blocks: Task 7 (sub-period stability uses this framework)
