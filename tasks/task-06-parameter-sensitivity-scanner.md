# Task 6: Parameter Sensitivity Scanner

## Objective
Build a lightweight parameter sensitivity scanner that sweeps 2-3 key parameters for each of the top strategies, producing heatmaps of Sharpe / Return / MaxDD across the parameter space. This reveals whether strategy performance is robust or highly sensitive to specific parameter choices.

## Context
Every existing backtest uses hard-coded parameters (e.g., LOOKBACK=12, TOP_DECILE=0.10, MIN_ADTV=1_000_000). There is no evidence these are optimal or even robust. If a strategy's Sharpe drops from 0.79 to 0.20 when the lookback changes from 12 to 10 months, that is a red flag for overfitting. Conversely, if performance is stable across a wide range of parameters, we have more confidence the alpha is real.

This does NOT need to be a full walk-forward optimizer. It is a diagnostic tool: run each parameter combination over the full 2005-present period and plot the results as a heatmap.

## Requirements
- Create a new file `backtests/core/param_scanner.py` containing the reusable scanning framework
- Create a new file `backtests/param_sensitivity_analysis.py` that applies the scanner to the top 4 strategies
- Strategies to scan:
  1. **MultiFactor** (the workhorse): sweep LOOKBACK (6, 9, 12, 15, 18 months) x TOP_PCT (0.05, 0.10, 0.15, 0.20, 0.25)
  2. **COPOM Easing** (best risk-adjusted): sweep COPOM lookback (2, 3, 4, 6 months for the "shift" comparison) x the choice of what to hold during easing (IBOV vs top-decile multifactor stocks vs equal-weight all liquid stocks)
  3. **MomSharpe** (momentum variant): sweep LOOKBACK (6, 9, 12, 15, 18) x TOP_PCT (0.05, 0.10, 0.15, 0.20)
  4. **Res.MultiFactor** (regime-filtered): sweep regime threshold (1-of-3, 2-of-3, 3-of-3 signals required) x TOP_PCT (0.05, 0.10, 0.15, 0.20)

### For each parameter combination, compute:
- After-tax Sharpe ratio
- After-tax annualized return
- Max drawdown
- Average monthly turnover

### Output for each strategy:
- Heatmap of Sharpe across the 2D parameter grid
- Heatmap of annualized return
- Heatmap of max drawdown
- A text summary identifying the "robust zone" (parameter region where Sharpe is within 80% of the peak)

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/compare_all.py` -- Contains inline implementations of all 4 strategies. The scanner needs to parameterize these loops.
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/simulation.py` -- `run_simulation()` with tax-aware parameters
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `build_metrics()`, `sharpe()`, `max_dd()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/data.py` -- Data loaders
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/plotting.py` -- `PALETTE` for consistent dark-theme heatmap styling

## Implementation Details

### Scanner Framework (`core/param_scanner.py`)

```python
def scan_parameters(
    signal_fn,       # callable(params_dict) -> (returns_matrix, target_weights)
    param_grid: dict, # {"param_name": [val1, val2, ...], ...}
    shared_data: dict, # precomputed data (ret, adtv, etc.)
    sim_config: dict,  # capital, tax_rate, slippage, etc.
) -> pd.DataFrame:
    """
    Run a strategy over all combinations of parameters.

    Args:
        signal_fn: A function that takes a dict of parameters and shared_data,
                   and returns (returns_matrix, target_weights) for run_simulation()
        param_grid: Dict mapping parameter names to lists of values to try
        shared_data: Dict of precomputed DataFrames (ret, adtv, log_ret, etc.)
        sim_config: Dict with simulation parameters

    Returns:
        DataFrame with columns: param1, param2, ..., sharpe, ann_return, max_dd, avg_turnover
    """


def plot_param_heatmap(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    metric: str,   # "sharpe", "ann_return", "max_dd"
    title: str,
    out_path: str,
):
    """
    Plot a 2D heatmap of a metric across two parameter dimensions.
    Uses the dark-theme styling from core/plotting.py PALETTE.
    """
```

### Strategy Signal Functions
For each strategy, create a parameterized signal function that can be called by the scanner. These functions should accept a params dict and the shared precomputed data, returning (returns_matrix, target_weights). For example:

```python
def multifactor_signal(params: dict, data: dict) -> tuple:
    """
    Parameterized MultiFactor strategy.
    params: {"lookback": int, "top_pct": float}
    """
    lookback = params["lookback"]
    top_pct = params["top_pct"]
    # ... signal generation using data["log_ret"], data["ret"], data["adtv"], etc.
    return ret_matrix, target_weights
```

### Shared Data Precomputation
Load data once at the top of the analysis script:
```python
adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START, END)
cdi_daily = download_cdi_daily(START, END)
ibov_px = download_benchmark("^BVSP", START, END)
# ... resample, compute signals, etc.
shared_data = {
    "ret": ret,
    "log_ret": log_ret,
    "adtv": adtv,
    "raw_close": raw_close,
    "has_glitch": has_glitch,
    "cdi_monthly": cdi_monthly,
    "ibov_ret": ibov_ret,
    # ... all regime signals
}
```

### Heatmap Styling
Use matplotlib's `imshow` or `pcolormesh` with:
- `cmap="RdYlGn"` for Sharpe (green = good)
- `cmap="RdYlGn"` for return (green = good)
- `cmap="RdYlGn_r"` for max drawdown (green = small drawdown)
- Dark background matching PALETTE["bg"] = "#0D1117"
- Annotate each cell with the value

### Output Files
- `param_sensitivity_multifactor.png` -- 3 heatmaps (Sharpe, Return, MaxDD) for MultiFactor
- `param_sensitivity_copom.png` -- Heatmaps for COPOM Easing
- `param_sensitivity_momsharpe.png` -- Heatmaps for MomSharpe
- `param_sensitivity_resmultifactor.png` -- Heatmaps for Res.MultiFactor
- Console: summary of robust parameter zones for each strategy

### Performance Note
Each parameter combination requires a full simulation run (~0.5-2 seconds). With ~25 combinations per strategy and 4 strategies, expect ~200-400 seconds total runtime. This is acceptable for a diagnostic tool.

## Acceptance Criteria
- [ ] File `backtests/core/param_scanner.py` exists with `scan_parameters()` and `plot_param_heatmap()`
- [ ] File `backtests/param_sensitivity_analysis.py` exists and runs successfully
- [ ] Heatmaps produced for all 4 strategies (4 PNG files)
- [ ] Each heatmap clearly shows parameter sensitivity (cells are annotated with values)
- [ ] Console output identifies the robust parameter zones for each strategy
- [ ] The current default parameters (LOOKBACK=12, TOP_PCT=0.10) are shown in the heatmaps and are not outliers (i.e., they are in a reasonable zone, not at an isolated peak)
- [ ] Runtime is under 10 minutes total

## Dependencies
- Depends on: Task 1 (for shared data loading patterns and strategy signal logic)
- Blocks: None directly, but results inform Task 7 (sub-period stability) and Tasks 8-10 (new signal parameter choices)
