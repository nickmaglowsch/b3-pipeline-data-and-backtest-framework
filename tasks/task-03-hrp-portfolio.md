# Task 3: Hierarchical Risk Parity (HRP) Portfolio

## Objective
Implement Lopez de Prado's Hierarchical Risk Parity (HRP) algorithm for portfolio allocation across the strategy return streams. HRP uses hierarchical clustering on the correlation matrix to group similar strategies, then allocates weights top-down through the tree. It avoids matrix inversion entirely, making it more numerically stable than traditional mean-variance optimization.

## Context
The correlation matrix (visible in `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/correlation_matrix.png`) shows clear clustering: MultiFactor and MomSharpe are nearly redundant (0.89 corr), CDI+MA200 and COPOM Easing cluster together (0.84 corr), while SmallcapMom is the most isolated. HRP naturally handles this structure by first clustering correlated strategies, then allocating between clusters before within clusters. This prevents over-allocation to redundant strategies.

## Requirements
- Add the HRP algorithm to `backtests/core/portfolio_opt.py` (created in Task 2)
- Create a new backtest file `backtests/portfolio_hrp_backtest.py`
- Implement the full HRP pipeline:
  1. Compute the correlation and covariance matrices from trailing returns
  2. Compute distance matrix: d(i,j) = sqrt(0.5 * (1 - corr(i,j)))
  3. Apply hierarchical clustering (single-linkage or Ward) using scipy
  4. Quasi-diagonalize the covariance matrix by reordering along the dendrogram
  5. Recursively bisect the sorted assets, allocating weight proportional to inverse variance at each split
- Use a rolling estimation window (default 36 months), re-estimating monthly
- Compare HRP against: equal-weight, inverse-vol risk parity (from Task 2), IBOV, CDI
- Produce comparison metrics and equity curve plot

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/strategy_returns.py` -- `build_strategy_returns()` for input data
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/portfolio_opt.py` -- Created in Task 2; add HRP functions here
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `build_metrics()`, `value_to_ret()`, `cumret()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/plotting.py` -- `PALETTE`, `fmt_ax()` for consistent dark-theme styling
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/correlation_matrix.py` -- Reference for how the existing correlation matrix is built and visualized

## Implementation Details

### HRP Algorithm (add to `core/portfolio_opt.py`)

```python
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

def hrp_weights(returns_df: pd.DataFrame, lookback: int = 36) -> pd.Series:
    """
    Hierarchical Risk Parity allocation.

    Steps:
      1. Correlation -> distance matrix
      2. Hierarchical clustering (Ward linkage)
      3. Quasi-diagonalization (reorder by dendrogram leaves)
      4. Recursive bisection with inverse-variance allocation

    Returns a Series of weights summing to 1.0.
    """
```

Key implementation notes:
- Use `scipy.cluster.hierarchy.linkage(distance_condensed, method='ward')` for clustering
- Use `scipy.cluster.hierarchy.leaves_list()` to get the quasi-diagonal ordering
- The recursive bisection step splits the sorted list of assets in half at each node, then allocates proportional to cluster inverse-variance
- Handle edge cases: if lookback window has fewer than 2 observations for any strategy, fall back to equal-weight

### Dendrogram Visualization
In the backtest script, add an optional dendrogram plot showing how the strategies cluster. This provides visual insight into strategy groupings. Save to `portfolio_hrp_dendrogram.png`.

### Portfolio Simulation
Same approach as Task 2 -- combine after-tax strategy returns with time-varying HRP weights:
1. For each month `t` with enough history (>= 36 months):
   - Compute HRP weights using returns from `t-36` to `t-1`
   - Portfolio return at `t` = sum(weight_i * return_i_at_t)
2. For early months, use equal-weight fallback
3. Compound into equity curve, compute metrics

### Strategy Selection
Include all 8 strategies from Task 1. Also run a liquid-only variant excluding SmallcapMom.

### Output
- Console: comparison table (HRP vs EqualWeight vs InvVol vs ERC vs IBOV vs CDI)
- Plot 1: equity curves for all variants, saved to `portfolio_hrp.png`
- Plot 2: dendrogram showing strategy clustering, saved to `portfolio_hrp_dendrogram.png`
- Print the final-month weight allocation for HRP and the clustering order

## Acceptance Criteria
- [ ] `hrp_weights()` function added to `backtests/core/portfolio_opt.py`
- [ ] File `backtests/portfolio_hrp_backtest.py` exists and runs successfully
- [ ] HRP weights are all positive and sum to 1.0
- [ ] The dendrogram shows sensible clustering (e.g., MultiFactor and MomSharpe should cluster together given their 0.89 correlation)
- [ ] HRP allocates lower combined weight to the MultiFactor/MomSharpe cluster than to uncorrelated strategies like SmallcapMom
- [ ] Equity curve plot saved to `portfolio_hrp.png`
- [ ] Dendrogram plot saved to `portfolio_hrp_dendrogram.png`
- [ ] Metrics comparison printed to console

## Dependencies
- Depends on: Task 1 (strategy returns), Task 2 (portfolio_opt.py module and scipy dependency)
- Blocks: Task 5 (portfolio comparison dashboard)
