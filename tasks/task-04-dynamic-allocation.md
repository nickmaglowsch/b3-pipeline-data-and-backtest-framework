# Task 4: Dynamic Allocation Portfolio

## Objective
Implement time-varying portfolio weights that adapt to market conditions. Unlike the static risk parity and HRP approaches (Tasks 2-3) that use a fixed lookback window, this task explores allocation schemes that explicitly respond to regime changes and rolling performance metrics.

## Context
The B3 market is highly cyclical with distinct regimes (COPOM easing/tightening, high/low IBOV volatility). The existing COPOM Easing strategy already demonstrates that a simple binary regime switch (equity vs CDI) achieves the best risk-adjusted equity Sharpe (~0.79). This task extends that insight to the meta-portfolio level: dynamically shift weight toward strategies that perform well in the current regime and toward CDI during stress.

From the correlation matrix, we know:
- CDI is negatively correlated with equity strategies (-0.03 to -0.21)
- COPOM Easing captures regime timing at the asset-allocation level
- SmallcapMom is the most uncorrelated equity strategy

## Requirements
- Create a new file `backtests/portfolio_dynamic_backtest.py`
- Implement three dynamic allocation approaches and compare them:

### Approach 1: Rolling Sharpe Momentum
Weight each strategy proportional to its trailing Sharpe ratio over the last N months. Strategies with negative trailing Sharpe get zero weight (cash goes to CDI). This naturally rotates toward strategies that are currently working.

### Approach 2: Regime-Conditional Allocation
Use the existing COPOM easing signal (CDI 3m change) and IBOV volatility regime to select pre-defined allocation mixes:
- **Easing + Calm**: Aggressive equity mix (higher weight to momentum strategies)
- **Easing + Stressed**: Moderate equity mix (higher weight to low-vol strategies)
- **Tightening + Calm**: Defensive (higher CDI weight, keep some equity exposure)
- **Tightening + Stressed**: Full CDI (or near-full)

### Approach 3: Combined (Rolling Sharpe within Regime)
Use Approach 2 to determine the equity/CDI budget, then use rolling Sharpe weights from Approach 1 to allocate within the equity portion.

- All approaches: monthly rebalancing, 2005-present, 15% CGT, 0.1% slippage, R$20K exemption
- Compare against: static equal-weight, static risk parity, static HRP, IBOV, CDI

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/strategy_returns.py` -- `build_strategy_returns()` for input data
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/portfolio_opt.py` -- Created in Tasks 2-3; add dynamic allocation functions here
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `build_metrics()`, `sharpe()`, `value_to_ret()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/copom_easing_backtest.py` -- Reference for the COPOM easing regime signal: `is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)` (line 79 of copom_momentum_backtest.py)
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/compare_all.py` -- Reference for IBOV volatility regime signal: `ibov_vol_pctrank <= 0.70` (lines 122-126)
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/plotting.py` -- `PALETTE`, `fmt_ax()`

## Implementation Details

### Add to `core/portfolio_opt.py`

```python
def rolling_sharpe_weights(
    returns_df: pd.DataFrame,
    lookback: int = 12,
    min_sharpe: float = 0.0,
) -> pd.Series:
    """
    Weight proportional to trailing Sharpe ratio.
    Strategies with Sharpe < min_sharpe get zero weight.
    Remaining weight goes to CDI if it exists in the DataFrame.
    """


def regime_conditional_weights(
    is_easing: bool,
    is_calm: bool,
    strategy_names: list,
) -> pd.Series:
    """
    Pre-defined weight mixes based on macro regime.
    Returns weights for each strategy.
    """
```

### Regime Signal Computation
The dynamic backtest script needs to recompute the regime signals (easing, calm) from raw data. These are available inside `build_strategy_returns()` intermediate data but not currently exposed. Two options:
1. Have `build_strategy_returns()` also return the regime signals (preferred, cleaner API)
2. Recompute them in the dynamic backtest script

Choose option 1: modify `build_strategy_returns()` to optionally return a dict of regime indicators:
```python
regime_signals = {
    "is_easing": is_easing,  # Series[bool], monthly
    "ibov_calm": ibov_calm,  # Series[bool], monthly
    "ibov_uptrend": ibov_uptrend,  # Series[bool], monthly
}
```

### Rolling Sharpe Parameters
- Default lookback: 12 months (1 year of trailing returns)
- Also test 6-month and 24-month lookbacks to see sensitivity
- Minimum Sharpe threshold: 0.0 (strategies with negative Sharpe get zero)

### Regime Allocation Mixes
Define sensible starting allocations per regime. These are not optimized -- they are heuristic starting points:

```python
REGIME_MIXES = {
    ("easing", "calm"): {
        # Aggressive: high equity exposure, favor momentum/multifactor
        "equity_budget": 0.80,
        "cdi_budget": 0.20,
    },
    ("easing", "stressed"): {
        # Moderate: some equity, favor low-vol and regime-filtered strategies
        "equity_budget": 0.50,
        "cdi_budget": 0.50,
    },
    ("tightening", "calm"): {
        # Defensive: low equity
        "equity_budget": 0.30,
        "cdi_budget": 0.70,
    },
    ("tightening", "stressed"): {
        # Full risk-off
        "equity_budget": 0.00,
        "cdi_budget": 1.00,
    },
}
```

Within each regime, allocate the equity budget using rolling Sharpe weights among the equity-oriented strategies.

### Turnover Tracking
Dynamic strategies have potentially high turnover. Track and report:
- Average monthly weight change (L1 norm of weight differences)
- Number of full regime switches per year
- Tax impact comparison (with R$20K exemption)

### Output
- Console: comparison table of all portfolio methods (Dynamic Rolling Sharpe / Dynamic Regime / Dynamic Combined / EqualWeight / InvVol / HRP / IBOV / CDI)
- Plot: equity curves, saved to `portfolio_dynamic.png`
- Plot: weight evolution over time for the best dynamic approach (stacked area chart), saved to `portfolio_dynamic_weights.png`
- Print regime summary (how many months in each regime state)

## Acceptance Criteria
- [ ] Dynamic allocation functions added to `backtests/core/portfolio_opt.py`
- [ ] File `backtests/portfolio_dynamic_backtest.py` exists and runs successfully
- [ ] Rolling Sharpe weights are non-negative and sum to 1.0 each month
- [ ] Regime-conditional allocation correctly identifies COPOM easing/tightening and IBOV calm/stressed states
- [ ] The combined dynamic approach produces reasonable equity curves (no extreme concentration)
- [ ] Weight evolution plot shows meaningful regime-driven shifts (not noise)
- [ ] At least one dynamic variant outperforms static equal-weight on Sharpe
- [ ] Turnover metrics are reported and reasonable (not excessively high)
- [ ] Both plots saved (portfolio_dynamic.png, portfolio_dynamic_weights.png)

## Dependencies
- Depends on: Task 1 (strategy returns), Task 2 (portfolio_opt.py, risk parity), Task 3 (HRP for comparison)
- Blocks: Task 5 (portfolio comparison dashboard)
