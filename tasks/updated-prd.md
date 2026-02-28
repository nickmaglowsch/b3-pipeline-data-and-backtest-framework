# Updated PRD: Strategy Research & Portfolio Optimization for B3 Data Pipeline

## 1. Overview

This initiative systematically improves the risk-adjusted returns of the B3 backtesting framework through three phases, in priority order:

1. **Portfolio-level optimization** -- Apply principled allocation techniques (Risk Parity, Hierarchical Risk Parity, Dynamic allocation) to existing strategy return streams.
2. **Robustness testing** -- Build a lightweight parameter sensitivity scanner to validate that existing strategy performance is not an artifact of specific parameter choices.
3. **New signal discovery** -- Explore fundamentally new signal types not yet tested, guided by robustness insights.

## 2. Decisions & Constraints

Based on user inputs:

| Decision | Choice | Rationale |
|---|---|---|
| Research priority | Portfolio -> Robustness -> Signals | Portfolio optimization gives quickest ROI from existing work |
| Portfolio optimization | Risk Parity + HRP + Dynamic | Skip fragile MVO; focus on robust, modern methods |
| Universe scope | Liquid only (ADTV >= R$1M) | Realistic, higher capacity, less survivorship bias |
| Walk-forward approach | Hybrid (lightweight param scanner) | Heatmaps for 2-3 key params per strategy, not full walk-forward |
| Risk management | Neither (no vol-targeting, no drawdown control) | Keep simple; focus on signal & portfolio improvements |
| Backtest period | 2005-present | Matches compare_all.py, 20+ years of data |
| Constraints | Long-only, tax-aware | Minimize turnover, use R$20K monthly exemption |

## 3. What Already Exists

### Infrastructure (no changes needed)
- `backtests/core/data.py` -- Data loaders for B3 SQLite, CDI (BCB API), IBOV/benchmarks (Yahoo)
- `backtests/core/simulation.py` -- Tax-aware simulator with 15% CGT, loss carryforward, slippage, deferred DARF, R$20K exemption
- `backtests/core/metrics.py` -- Sharpe, Calmar, MaxDD, ann. return/vol, build_metrics()
- `backtests/core/plotting.py` -- Standardized 4-panel tear sheets
- `b3_market_data.sqlite` -- Prices table with OHLCV, split_adj_*, adj_close for ~620 tickers (1994-2026)

### Existing Strategies (8 strategies already in compare_all.py)
These produce monthly after-tax return series that will serve as inputs to portfolio optimization:
1. **CDI+MA200** -- COPOM easing filter + MA200 trend filter on individual stocks
2. **Res.MultiFactor** -- 5-factor composite (dist_MA200, low_vol_60d, low_ATR, vol_rank_20d, liquidity) + 2-of-3 regime filter
3. **RegimeSwitching** -- IBOV > 10M SMA -> multifactor stock selection, else CDI
4. **COPOM Easing** -- Simple IBOV vs CDI binary switch on easing/tightening
5. **MultiFactor** -- 50% momentum rank + 50% low-vol rank, no regime filter
6. **SmallcapMom** -- 6M momentum on below-median-ADTV stocks (NOTE: uses ADTV >= 100K, outside our R$1M floor)
7. **LowVol** -- Unconditional low-volatility decile
8. **MomSharpe** -- Risk-adjusted momentum (return / volatility)

### Existing Portfolio Combinations (to be superseded)
- `portfolio_low_corr_backtest.py` -- Naive 25% equal-weight across 4 strategies
- `combined_strategy_backtest.py` -- 50/50 Momentum + Low Vol
- `triple_blend_backtest.py` -- 33/33/33 SmallCapMom + Multifactor + CDI
- `balanced_smallcap_cdi_backtest.py` -- 80/20 SmallCapMom + CDI

### Key Performance Observations (from correlation matrix, 2005-2026)
- CDI: Sharpe ~5.95, Ann.Return ~10.4%, Vol ~1.6%
- COPOM Easing: Sharpe ~0.79, Ann.Return ~12.5%
- Res.MultiFactor: Sharpe ~0.70
- SmallcapMom: Highest absolute return but most volatile, most uncorrelated (negative with COPOM Easing)
- MultiFactor & MomSharpe: corr 0.89 (nearly redundant)
- CDI negatively correlated with equity strategies

## 4. What Needs to Be Built

### Phase 1: Portfolio Optimization (Tasks 1-5)

**Task 1: Strategy Return Extraction Module**
Create a reusable module (`backtests/core/strategy_returns.py`) that runs all 8 strategies from compare_all.py and returns a clean DataFrame of monthly after-tax return series. This decouples signal generation from portfolio construction. Currently each strategy is embedded inline in compare_all.py and correlation_matrix.py with massive code duplication.

**Task 2: Risk Parity Portfolio**
Implement inverse-variance (risk parity) allocation across the strategy return streams. Each strategy contributes equally to total portfolio risk. More robust than equal-weight since it accounts for volatility differences.

**Task 3: Hierarchical Risk Parity (HRP) Portfolio**
Implement Lopez de Prado's HRP algorithm using the strategy return covariance matrix. No matrix inversion required, more stable than MVO, naturally handles correlated strategies by clustering them first.

**Task 4: Dynamic Allocation Portfolio**
Implement time-varying weights that adapt to market conditions. Use rolling windows (trailing 12-36 months) to re-estimate correlations and volatilities, then re-optimize weights monthly. Also explore regime-conditional allocation (increase CDI weight in high-vol regimes).

**Task 5: Portfolio Comparison Dashboard**
Create a comprehensive comparison script that runs all portfolio optimization methods (equal-weight, risk parity, HRP, dynamic) side-by-side with full metrics, equity curves, drawdowns, turnover analysis, and tax impact. This supersedes portfolio_low_corr_backtest.py.

### Phase 2: Robustness Testing (Tasks 6-7)

**Task 6: Parameter Sensitivity Scanner**
Build a lightweight framework (`backtests/core/param_scanner.py`) that sweeps 2-3 key parameters for each strategy and generates heatmaps of Sharpe / Return / MaxDD across the parameter space. This reveals whether strategy performance is robust or driven by specific parameter choices.

**Task 7: Sub-Period Stability Analysis**
Extend the comparison framework to report metrics across sub-periods (2005-2010, 2010-2015, 2015-2020, 2020-2026) and test whether portfolio optimization benefits are consistent across market regimes, or concentrated in one era.

### Phase 3: New Signal Discovery (Tasks 8-10)

**Task 8: Seasonal / Calendar Effects Strategy**
Test whether B3 exhibits turn-of-month, January effect, sell-in-May, or pre-holiday effects. These are well-documented in academic literature for emerging markets and have not been tested in this repo.

**Task 9: Earnings Momentum / Revision Proxy Strategy**
Without direct earnings data, use price-volume behavior around quarterly reporting windows as a proxy for earnings surprise. Stocks that gap up on high volume during reporting season and continue trending likely had positive earnings surprises.

**Task 10: Sector Rotation Strategy**
Test whether rotating among B3 sectors (financials, commodities, utilities, consumer) based on relative momentum improves risk-adjusted returns compared to stock-level selection. Requires mapping tickers to sectors using the first 2-3 characters of the ticker code as a heuristic.

## 5. Technical Requirements

- **Language**: Python 3.9+
- **New dependency**: `scipy` (for optimization routines in risk parity / HRP). Must be added to requirements.txt.
- **Backtest period**: 2005-01-01 to present, matching compare_all.py
- **Rebalance frequency**: Monthly (ME)
- **Universe**: Standard lot tickers with ADTV >= R$1M, price >= R$1.0
- **Tax**: 15% CGT with loss carryforward, 0.1% slippage
- **Tax optimization**: Use monthly_sales_exemption=20_000 parameter (R$20K threshold) in run_simulation()
- **All scripts**: Must be standalone, runnable from the `backtests/` directory
- **Plots**: Use existing PALETTE and plotting conventions from `backtests/core/plotting.py`

## 6. Success Criteria

1. At least one optimized portfolio achieves higher after-tax Sharpe than the naive equal-weight LowCorr Portfolio (~0.54) over the full 2005-present period.
2. Parameter sensitivity heatmaps exist for the top 4 strategies, confirming robustness.
3. Sub-period analysis shows consistent improvement (not concentrated in one era).
4. At least one new signal (seasonal, earnings proxy, or sector rotation) produces a Sharpe > 0.3 standalone and adds diversification benefit (correlation < 0.5 with existing strategies).
5. All new code follows existing conventions: uses core/ modules, produces standard tear sheets, runs from 2005.

## 7. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Overfitting to 2005-2026 period | Parameter scanner + sub-period analysis in Tasks 6-7 |
| Scipy not available in venv | Task 2 explicitly adds it to requirements.txt |
| SmallcapMom uses ADTV < R$1M | Exclude from optimization or replace with liquid-universe variant |
| High turnover from monthly reoptimization | Use turnover penalty in dynamic allocation; compare with quarterly rebalancing |
| Strategy return streams have different start dates | Align to common start date (latest start across all strategies) |
