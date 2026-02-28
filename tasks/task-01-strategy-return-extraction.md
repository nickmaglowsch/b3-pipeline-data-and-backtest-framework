# Task 1: Strategy Return Extraction Module

## Objective
Create a reusable module that generates monthly after-tax return series for all 8 core strategies, producing a single clean DataFrame. This decouples signal generation from portfolio construction and eliminates the massive code duplication currently spread across `compare_all.py`, `correlation_matrix.py`, and `portfolio_low_corr_backtest.py`.

## Context
Currently, the 8 strategy signal-generation loops are copy-pasted into at least 3 different scripts (compare_all.py, correlation_matrix.py, portfolio_low_corr_backtest.py). Each copy is slightly different and must be kept in sync manually. The portfolio optimization tasks (Tasks 2-5) all need the same input: a DataFrame of monthly after-tax return series indexed by date with strategy names as columns. Building this extraction module first makes all downstream tasks cleaner.

The 8 strategies and their signal logic are defined inline in `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/compare_all.py` (lines 83-350) and `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/correlation_matrix.py` (lines 122-360).

## Requirements
- Create a new file `backtests/core/strategy_returns.py` containing a single main function `build_strategy_returns()` that:
  1. Loads B3 data, CDI, and IBOV once
  2. Runs all 8 strategies with consistent configuration
  3. Returns a `pd.DataFrame` with columns = strategy names, rows = monthly dates, values = after-tax monthly returns
  4. Also returns a dict of full simulation results (pretax_values, aftertax_values, tax_paid, etc.) for each strategy, so downstream code can access equity curves
- The function should accept configuration parameters: `start`, `end`, `db_path`, `capital`, `tax_rate`, `slippage`, `monthly_sales_exemption`
- Default configuration: START="2005-01-01", CAPITAL=100_000, TAX=0.15, SLIP=0.001, monthly_sales_exemption=20_000
- Include IBOV and CDI as benchmarks in the returned DataFrame
- Handle the SmallcapMom special case: it uses ADTV >= 100K (below the R$1M liquid-only floor). Include it in the output but add a note/flag so downstream code can choose to exclude it.

## Existing Code References
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/compare_all.py` -- Contains all 8 strategy definitions inline (the canonical source to extract from)
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/correlation_matrix.py` -- Same 8 strategies with `run_and_get_returns()` helper
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/data.py` -- `load_b3_data()`, `download_benchmark()`, `download_cdi_daily()`
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/simulation.py` -- `run_simulation()` function
- `/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/metrics.py` -- `value_to_ret()`, `build_metrics()`

## Implementation Details

### Function signature
```python
def build_strategy_returns(
    db_path: str = "b3_market_data.sqlite",
    start: str = "2005-01-01",
    end: str = None,  # defaults to today
    capital: float = 100_000,
    tax_rate: float = 0.15,
    slippage: float = 0.001,
    monthly_sales_exemption: float = 20_000,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all 8 core strategies and return their after-tax monthly return series.

    Returns:
        returns_df: DataFrame with columns = strategy names + "IBOV" + "CDI",
                    rows = monthly dates, values = after-tax simple returns
        sim_results: dict of {strategy_name: simulation_result_dict} for equity curve access
    """
```

### Strategy definitions to extract
Each strategy should be a private function `_run_<name>(...)` that receives the shared precomputed data and returns target_weights and a returns matrix. The 8 strategies to include:

1. **CDI+MA200** -- compare_all.py lines 83-117
2. **Res.MultiFactor** -- compare_all.py lines 119-175
3. **RegimeSwitching** -- compare_all.py lines 177-217
4. **COPOM Easing** -- compare_all.py lines 219-237
5. **MultiFactor** -- compare_all.py lines 239-266
6. **SmallcapMom** -- compare_all.py lines 268-295
7. **LowVol** -- compare_all.py lines 297-321
8. **MomSharpe** -- compare_all.py lines 323-350

### Shared precomputed data (compute once, pass to all)
Extract the shared data preparation from compare_all.py lines 37-57:
- adj_close, close_px, fin_vol from load_b3_data()
- Monthly resampled: px, ret, raw_close, adtv
- log_ret, has_glitch
- CDI monthly, IBOV monthly returns
- MA200, above_ma200, is_easing
- IBOV regime signals (ibov_calm, ibov_uptrend, ibov_above, etc.)
- Multifactor composite signals

### Key implementation note
The `monthly_sales_exemption=20_000` parameter is new -- it should be passed through to `run_simulation()`. This uses the R$20K monthly sales exemption that already exists in the simulation engine (`_execute_rebalance()` line 178-180 in simulation.py).

### DB path resolution
Use the pattern from correlation_matrix.py line 38:
```python
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "b3_market_data.sqlite")
```

## Acceptance Criteria
- [ ] File `backtests/core/strategy_returns.py` exists and is importable
- [ ] `build_strategy_returns()` returns a DataFrame with 10 columns (8 strategies + IBOV + CDI) and ~240 monthly rows (2005-2026)
- [ ] All 8 strategies produce the same after-tax return values as compare_all.py (within floating-point tolerance, except for the R$20K exemption effect)
- [ ] The function runs in under 60 seconds on typical hardware
- [ ] A simple test script at the bottom (`if __name__ == "__main__"`) demonstrates usage and prints the returned DataFrame shape and column names
- [ ] SmallcapMom is included but clearly documented as using ADTV < R$1M

## Dependencies
- Depends on: None (this is the foundation task)
- Blocks: Task 2, Task 3, Task 4, Task 5, Task 6, Task 7
