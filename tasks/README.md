# B3 Strategy Research & Portfolio Optimization -- Task Plan

## Summary

This task plan implements a systematic improvement to the B3 backtesting framework across three phases:

1. **Portfolio Optimization (Tasks 1-5)**: Extract strategy return streams into a reusable module, then apply Risk Parity, HRP, and Dynamic allocation methods to find the best portfolio of existing strategies.
2. **Robustness Testing (Tasks 6-7)**: Validate that strategy performance is not an artifact of specific parameter choices or concentrated in one historical era.
3. **New Signal Discovery (Tasks 8-10)**: Explore fundamentally new signal types (seasonal effects, earnings proxy, sector rotation) not yet tested in the repo.

**Total tasks**: 10
**Estimated complexity**: Medium-High. Tasks 1-5 form a tightly coupled chain. Tasks 6-10 are largely independent once Task 1 is complete.

## Dependency Graph

```
Task 1: Strategy Return Extraction (FOUNDATION -- blocks everything)
  |
  +---> Task 2: Risk Parity Portfolio
  |       |
  |       +---> Task 3: HRP Portfolio
  |               |
  |               +---> Task 4: Dynamic Allocation
  |                       |
  |                       +---> Task 5: Portfolio Comparison Dashboard
  |                               |
  |                               +---> Task 7: Sub-Period Stability Analysis
  |
  +---> Task 6: Parameter Sensitivity Scanner (independent of Tasks 2-5)
  |
  +---> Task 8: Seasonal / Calendar Effects (independent of Tasks 2-7)
  |
  +---> Task 9: Earnings Momentum Proxy (independent of Tasks 2-7)
  |
  +---> Task 10: Sector Rotation (independent of Tasks 2-7)
```

### Parallel execution opportunities

After Task 1 is complete, the following can run in parallel:
- **Track A**: Tasks 2 -> 3 -> 4 -> 5 -> 7 (portfolio optimization chain)
- **Track B**: Task 6 (parameter sensitivity)
- **Track C**: Task 8 (seasonal effects)
- **Track D**: Task 9 (earnings proxy)
- **Track E**: Task 10 (sector rotation)

Tracks B-E are fully independent of each other and of Track A (except Task 7 which depends on Track A completing).

## Task Summary Table

| Task | Name | New Files Created | Depends On | Phase |
|------|------|-------------------|------------|-------|
| 1 | Strategy Return Extraction | `backtests/core/strategy_returns.py` | None | Portfolio |
| 2 | Risk Parity Portfolio | `backtests/core/portfolio_opt.py`, `backtests/portfolio_risk_parity_backtest.py` | 1 | Portfolio |
| 3 | HRP Portfolio | (extends `portfolio_opt.py`), `backtests/portfolio_hrp_backtest.py` | 1, 2 | Portfolio |
| 4 | Dynamic Allocation | (extends `portfolio_opt.py`), `backtests/portfolio_dynamic_backtest.py` | 1, 2, 3 | Portfolio |
| 5 | Portfolio Comparison Dashboard | `backtests/portfolio_compare_all.py` | 1, 2, 3, 4 | Portfolio |
| 6 | Parameter Sensitivity Scanner | `backtests/core/param_scanner.py`, `backtests/param_sensitivity_analysis.py` | 1 | Robustness |
| 7 | Sub-Period Stability Analysis | `backtests/portfolio_stability_analysis.py` | 1-5 | Robustness |
| 8 | Seasonal / Calendar Effects | `backtests/seasonal_effects_backtest.py` | 1 | Signals |
| 9 | Earnings Momentum Proxy | `backtests/earnings_proxy_backtest.py` | 1 | Signals |
| 10 | Sector Rotation | `backtests/sector_rotation_backtest.py` | 1 | Signals |

## How to Use These Files

These task files are prompts for AI agents. Each file is a self-contained specification that an agent can pick up and execute independently, given that its dependencies have been completed.

**Workflow:**
1. Start with Task 1. It is the foundation for everything else.
2. After Task 1 is done, proceed with Tasks 2-5 in order (portfolio optimization chain), and optionally run Tasks 6, 8, 9, 10 in parallel.
3. Task 7 requires Tasks 1-5 to be complete.
4. Delete each task file after the task is completed and verified.
5. When all task files are deleted, the initiative is complete.

**Important**: Each task file contains all the context an executing agent needs, including specific file paths, code patterns to follow, and acceptance criteria. The agent does not need to read other task files unless explicitly stated in the Dependencies section.

## Key Technical Context

- **Database**: `b3_market_data.sqlite` at project root
- **Backtest period**: 2005-01-01 to present
- **Universe**: Standard lot tickers, ADTV >= R$1M, price >= R$1.0
- **Tax model**: 15% CGT, 0.1% slippage, R$20K monthly sales exemption, loss carryforward
- **New dependency needed**: `scipy>=1.10.0` (added in Task 2)
- **Plotting**: Dark theme via `backtests/core/plotting.py` PALETTE

## Open Questions / Decisions for Human Review

1. **SmallcapMom inclusion**: This strategy uses ADTV >= 100K (below the R$1M liquid floor). Task 1 includes it but flags it. The portfolio optimization tasks should test with and without it.
2. **Rebalancing frequency for dynamic allocation**: Monthly is the default, but Task 4 notes that quarterly rebalancing may reduce turnover. Results should be compared.
3. **Sector map coverage**: Task 10's heuristic ticker-to-sector mapping may miss some tickers. If coverage drops below 70% of the liquid universe, the map needs expansion.

## Reference Documents

- `tasks/updated-prd.md` -- Full product requirements document with all decisions and context
- `tasks/planning-questions.md` -- Original planning questions and codebase summary from the discovery phase
