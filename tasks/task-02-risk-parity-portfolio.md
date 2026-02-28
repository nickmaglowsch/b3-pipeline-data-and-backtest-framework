# Task 2: Risk Parity Portfolio

## Objective
Implement an inverse-variance (risk parity) portfolio allocation across the 8 strategy return streams. Each strategy should contribute approximately equally to total portfolio risk, producing a more balanced risk profile than naive equal-weight.

## Context
The current `portfolio_low_corr_backtest.py` uses naive 25% equal-weight across 4 hand-picked strategies. Risk parity is a well-established improvement that accounts for volatility differences: low-vol strategies (CDI, COPOM Easing) get higher weights, high-vol strategies (SmallcapMom, MultiFactor) get lower weights. This naturally produces better Sharpe ratios without forecasting returns.

The input data comes from the `build_strategy_returns()` function created in Task 1.

## Requirements
- Create a new file `backtests/portfolio_risk_parity_backtest.py`
- Implement two risk parity variants:
  1. **Naive Risk Parity (Inverse Volatility)**: weight_i = (1/vol_i) / sum(1/vol_j). Uses only each strategy's own volatility, ignoring correlations.
  2. **Full Risk Parity (Equal Risk Contribution)**: Numerically solve for weights where each strategy's marginal contribution to portfolio variance is equal. This accounts for correlations.
- Use a rolling estimation window (default 36 months) to compute volatilities and correlations, re-estimating monthly
- For the first 36 months (before enough history), fall back to equal-weight
- Run both variants through the existing simulation framework with 15% CGT, 0.1% slippage, and R$20K monthly exemption
- Compare against: equal-weight portfolio, IBOV, CDI
- Produce a comparison table and equity curve plot

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/strategy_returns.py` -- Created in Task 1. Call `build_strategy_returns()` to get the strategy return DataFrame
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `build_metrics()`, `value_to_ret()`, `cumret()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/plotting.py` -- `PALETTE`, `fmt_ax()` for consistent styling
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/portfolio_low_corr_backtest.py` -- Pattern for combining strategy returns and plotting. The current approach simply averages returns: `portfolio_ret = sub_df.mean(axis=1)` (line 248). This task replaces that with principled weighting.

## Implementation Details

### Add scipy to requirements.txt
The full risk parity solver needs `scipy.optimize.minimize`. Add `scipy>=1.10.0` to `/Users/nickmaglowsch/person-projects/b3-data-pipeline/requirements.txt`.

### Risk Parity Core Functions
Create these in a utility module `backtests/core/portfolio_opt.py` so they are reusable by Tasks 3 and 4:

```python
def inverse_vol_weights(returns_df: pd.DataFrame, lookback: int = 36) -> pd.Series:
    """
    Naive risk parity: weight inversely proportional to trailing volatility.
    Returns a Series of weights summing to 1.0.
    """
    vols = returns_df.tail(lookback).std()
    inv_vols = 1.0 / vols
    return inv_vols / inv_vols.sum()


def equal_risk_contribution_weights(
    returns_df: pd.DataFrame, lookback: int = 36
) -> pd.Series:
    """
    Full risk parity: solve for weights where each asset's risk contribution is equal.
    Uses scipy.optimize.minimize with SLSQP.
    """
    # ... use trailing lookback months of returns
    # ... compute covariance matrix
    # ... minimize sum((RC_i - RC_target)^2) where RC_i = w_i * (Sigma @ w)_i / sqrt(w' Sigma w)
```

### Portfolio Construction Approach
Since these are strategy-level returns (not stock-level), the portfolio is simulated at the meta-strategy level:
1. Get monthly returns from `build_strategy_returns()` -> DataFrame with 8 strategy columns + IBOV + CDI
2. Select which strategies to include (all 8, or exclude SmallcapMom since it uses ADTV < R$1M)
3. Each month, compute weights using trailing returns
4. Portfolio return = weighted sum of strategy returns for that month
5. Compound into an equity curve
6. Compute metrics using `build_metrics()` from `core/metrics.py`

This is a simpler approach than running a full `run_simulation()` on stock-level weights because the individual strategy simulations already account for tax and slippage. The meta-portfolio just combines their after-tax return streams. This means we do NOT need to pass through the simulation engine again at the portfolio level -- just weighted return aggregation.

### Strategy Selection
By default, include all 8 strategies. Also run a variant excluding SmallcapMom (liquid-only) to honor the ADTV >= R$1M constraint. Compare both.

### Output
- Console: comparison table of EqualWeight / InvVol / ERC / IBOV / CDI with Ann.Return, Ann.Vol, Sharpe, MaxDD, Calmar
- Plot: equity curves for all variants, saved to `portfolio_risk_parity.png`
- Print the monthly weight allocations at the end of the backtest period to verify they look reasonable

## Acceptance Criteria
- [ ] File `backtests/core/portfolio_opt.py` exists with `inverse_vol_weights()` and `equal_risk_contribution_weights()` functions
- [ ] `scipy>=1.10.0` is added to requirements.txt
- [ ] File `backtests/portfolio_risk_parity_backtest.py` exists and runs successfully
- [ ] Both risk parity variants produce valid weight vectors (all positive, sum to 1.0)
- [ ] Equity curves are plotted and saved to `portfolio_risk_parity.png`
- [ ] The ERC portfolio achieves a better Sharpe than naive equal-weight over the full period
- [ ] Weights for low-vol strategies (CDI-heavy ones like COPOM Easing) are meaningfully higher than for high-vol strategies (SmallcapMom)
- [ ] Terminal output shows final weights and full metrics comparison table

## Dependencies
- Depends on: Task 1 (strategy return extraction module)
- Blocks: Task 5 (portfolio comparison dashboard)
