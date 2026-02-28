# Planning Questions

## Codebase Summary

### Existing Infrastructure
The repo has a mature B3 (Brazilian stock exchange) data pipeline and backtesting framework:
- **Data pipeline** (`b3_pipeline/`): Downloads, parses, and stores COTAHIST data in SQLite with full split/dividend adjustment.
- **Core backtesting** (`backtests/core/`): `data.py` (B3 data + CDI + IBOV loaders), `simulation.py` (tax-aware portfolio simulator with 15% CGT, loss carryforward, slippage, deferred DARF), `metrics.py` (Sharpe, Calmar, MaxDD, ann. return/vol), `plotting.py` (standardized tear sheets).
- **Database**: `b3_market_data.sqlite` with `prices` (ticker, date, OHLCV, split_adj_*, adj_close), `corporate_actions`, `stock_actions`, `detected_splits`. Data from 1994-2026, ~620 unique tickers.
- **External data**: CDI daily from BCB API, IBOV/IVVB11 from Yahoo Finance.

### Existing Strategies (30+ backtests)
**Stock-selection signals explored:**
1. **Momentum** (12M, 6M, 3M lookbacks, with/without 1M skip)
2. **Smooth Momentum / Information Ratio** (return/volatility ratio)
3. **Low Volatility** (12M rolling std, inverted rank)
4. **Multifactor** (50% momentum rank + 50% low-vol rank)
5. **Research Multifactor** (5 factors: dist_to_MA200, low_vol_60d, low_ATR, vol_rank_20d, liquidity)
6. **Anti-Lottery / Low Skewness** (lowest max single-month return)
7. **Frog-in-the-Pan** (win rate + momentum)
8. **Volume Breakout** (volume acceleration + positive trend)
9. **Volume Z-Score** (anomalous volume spikes)
10. **Liquidity Shock Reversal** (price drop on low volume)
11. **Mean Reversion** (short-term 1M losers)
12. **Dividend Yield + Momentum** (>6-8% TTM yield, positive trend)
13. **Momentum of Momentum** (acceleration)
14. **Bull Trap Short** (weekly, shorting blow-off tops)

**Regime/allocation approaches explored:**
- COPOM easing/tightening (CDI 3m change) -> equity vs CDI
- IBOV > 10M SMA -> equity vs CDI
- IBOV volatility percentile (calm vs stressed)
- IBOV 20d return (uptrend vs downtrend)
- 2-of-3 regime composite (easing + calm + uptrend)
- HMM regime switching (GaussianHMM on IBOV features)
- Beta rotation (high-beta in calm, low-beta in stress)
- Dual Momentum (IBOV vs IVVB11 vs CDI)
- Global Flight (IBOV SMA -> equities vs IVVB11)

**Weighting schemes explored:**
- Equal weight (most common)
- Inverse volatility (risk parity momentum)
- Sharpe-weighted (top 20 stocks)

**Portfolio combinations explored:**
- 50/50 Momentum + Low Volatility (`combined_strategy_backtest.py`)
- Triple Blend: 33% SmallCap Mom + 33% Multifactor + 33% CDI
- 80% SmallCap Mom + 20% CDI (`balanced_smallcap_cdi_backtest.py`)
- 33% Multifactor + 33% CDI + 34% IVVB11
- Low-Correlation Portfolio: 25% each of SmallcapMom + Res.MultiFactor + MomSharpe + COPOM Easing

### Current Performance (from correlation matrix & compare_all.py results)
From the correlation matrix image (2005-2026):
- **CDI**: Sharpe ~5.95, Ann.Return ~10.4%, Vol ~1.62% -- the dominant "Sharpe" strategy but low absolute return
- **COPOM Easing**: Sharpe ~0.79, Ann.Return ~12.5%, Vol ~16.51%
- **Res.MultiFactor**: Sharpe ~0.70, Ann.Return ~7.55%
- **LowCorr Portfolio**: Sharpe ~0.54, Ann.Return ~7.47%, Vol ~10.28%, MaxDD ~-40.2%
- **SmallcapMom**: Highest return (~100%+ cumulative) but very volatile and negative Sharpe over full period
- **MultiFactor, RegimeSwitching, LowVol, MomSharpe**: Sharpe ~0.2-0.5 range

**Key correlations (from the matrix):**
- SmallcapMom is the most uncorrelated to everything (negative corr with CDI and COPOM Easing at -0.25)
- CDI is negatively correlated with most equity strategies (-0.03 to -0.21)
- MultiFactor and MomSharpe are highly correlated (0.89)
- COPOM Easing and CDI+MA200 are highly correlated (0.84)
- Res.MultiFactor offers moderate diversification from most strategies (0.49-0.67)

### Key Gaps Identified
1. **No systematic parameter optimization / walk-forward analysis** -- all parameters are hand-picked
2. **No proper portfolio optimization** -- combinations are naive equal-weight, no mean-variance, Black-Litterman, or risk-budgeting
3. **No sector/industry exposure analysis** -- strategies may be accidentally concentrated
4. **No transaction cost sensitivity analysis** -- 0.1% slippage assumed, not validated
5. **No out-of-sample validation** -- no train/test splits for signal parameters
6. **No fundamental data** -- everything is price/volume based; no earnings, book value, etc.
7. **No intraday data** -- all strategies are daily or monthly frequency
8. **No position sizing beyond equal-weight or simple inverse-vol** -- no Kelly, no volatility targeting
9. **ML research was inconclusive** -- AUC ~0.48 (worse than random) on binary classification

## Questions

### Q1: Research Priority -- New Signals vs. Portfolio Optimization vs. Infrastructure
**Context:** The repo has 30+ individual strategy backtests but the portfolio combination work is primitive (naive equal-weight blends). Meanwhile, the feature research showed ML models have no predictive edge on binary return classification. The biggest immediate alpha likely comes from smarter portfolio construction using existing signals rather than discovering entirely new signals.
**Question:** Which research direction should we prioritize?
**Options:**
- A) **Portfolio-level optimization** -- Use existing strategy return streams to build an optimal portfolio (mean-variance, risk-parity, minimum-variance, hierarchical risk parity, dynamic allocation). This leverages the existing low-correlation insights.
- B) **New factor/signal discovery** -- Explore fundamentally different signal types not yet tested (e.g., earnings surprise, insider activity proxies from volume patterns, cross-sectional momentum with industry-neutral adjustment, seasonal effects, event-driven signals around corporate actions).
- C) **Parameter robustness & walk-forward optimization** -- Build infrastructure to systematically test parameter sensitivity (lookback windows, top-N%, liquidity thresholds) with proper in-sample/out-of-sample splits to find robust configurations.
- D) **All of the above, prioritized in order A -> C -> B** -- Start with portfolio optimization (quickest ROI using existing signals), then robustness testing, then new signals.

### Q2: Portfolio Optimization Approach
**Context:** The `portfolio_low_corr_backtest.py` currently uses naive 25% equal-weight across 4 strategies picked by visual inspection of the correlation matrix. The correlation matrix shows SmallcapMom is most uncorrelated (even negative) with other strategies, and CDI dominates Sharpe but has minimal absolute return. There is a clear opportunity to do principled optimization.
**Question:** If we pursue portfolio optimization, which techniques should we implement?
**Options:**
- A) **Mean-Variance / Markowitz** -- Classic efficient frontier using strategy return streams. Risk: highly sensitive to estimation error.
- B) **Risk Parity / Risk Budgeting** -- Allocate so each strategy contributes equally to portfolio risk. More robust than MVO.
- C) **Hierarchical Risk Parity (HRP)** -- Lopez de Prado's tree-based approach; no matrix inversion, more stable.
- D) **Dynamic allocation** -- Time-varying weights based on rolling correlations, trailing Sharpe, or regime state (e.g., increase CDI allocation in high-vol regimes).
- E) **Multiple approaches** -- Implement B, C, and D and compare. Skip pure MVO as it is known to be fragile.

### Q3: Universe Scope -- Large-Cap Only or Include Small-Cap?
**Context:** SmallcapMom is the highest absolute return strategy but also the most volatile and has the lowest Sharpe. It uses stocks with ADTV 100K-median (very illiquid). The balanced SmallCap+CDI strategy (80/20) was tested on a short window (2022-2026). Meanwhile, the main compare_all.py uses R$1M ADTV minimum. There is a real tension between capturing the small-cap premium and dealing with illiquidity/capacity constraints.
**Question:** Should new research include small-cap / micro-cap stocks, or focus on the liquid universe only?
**Options:**
- A) **Liquid only** (ADTV >= R$1M) -- More realistic, higher capacity, less survivorship bias risk
- B) **Include small-caps** (ADTV >= R$100K) -- Higher alpha potential, but needs careful handling of slippage and capacity
- C) **Tiered approach** -- Run strategies separately for large-cap and small-cap universes, then combine at portfolio level with capacity-aware weights (e.g., limit small-cap allocation to 20-30%)

### Q4: Walk-Forward / Robustness Infrastructure
**Context:** Currently every backtest is a single-run script with hard-coded parameters. There is no framework for parameter sweeps, walk-forward optimization, or bootstrap resampling. The research_multifactor_backtest.py derived its 5 factors from a one-shot ML study. Without robustness testing, we cannot distinguish genuine alpha from overfitting to the 2005-2026 backtest period.
**Question:** Should we invest in building proper walk-forward / robustness testing infrastructure before running more backtests?
**Options:**
- A) **Yes, build it first** -- Create a reusable framework for: (1) parameter grid search, (2) rolling walk-forward optimization, (3) bootstrap confidence intervals on Sharpe/returns, (4) regime-conditional performance breakdown. This delays new strategy output by ~2-3 tasks but makes all future results more trustworthy.
- B) **No, iterate fast** -- Keep the current single-run approach but add a simple in-sample/out-of-sample split (e.g., 2005-2015 train, 2016-2026 test) to each new backtest. Less infrastructure overhead.
- C) **Hybrid** -- Build a lightweight parameter sensitivity scanner (not full walk-forward) that can sweep 2-3 key parameters per strategy and report a heatmap. Cheap to implement, adds some rigor.

### Q5: Risk Management & Position Sizing
**Context:** All current strategies use equal-weight positions or simple inverse-vol. There is no volatility targeting (e.g., "target 10% annual vol"), no dynamic position sizing based on conviction or market conditions, and no drawdown-triggered de-risking. The simulation engine supports slippage and tax but not position-level stop-losses or portfolio-level risk limits.
**Question:** Should we add position sizing / risk management overlays to existing strategies?
**Options:**
- A) **Volatility targeting** -- Scale total equity exposure so the portfolio targets a fixed annual volatility (e.g., 10-12%). When realized vol is high, reduce exposure; when low, lever up (or just go 100% invested). This is the most impactful single improvement for Sharpe.
- B) **Drawdown control** -- Add a circuit-breaker that shifts to CDI after a certain portfolio drawdown threshold (e.g., -15%) and waits for recovery.
- C) **Both A and B** -- Implement vol-targeting as the primary overlay, with a drawdown circuit-breaker as a backstop.
- D) **Neither** -- Keep it simple, focus on signal and portfolio-level improvements instead.

### Q6: Backtest Start Date & Period
**Context:** `compare_all.py` runs from 2005, while individual backtests start from 2000, 2003, 2012, 2015, or 2022. The earlier data (pre-2005) includes periods with very few liquid stocks and different market microstructure. The more recent data (2017+) better reflects current market conditions. Different start dates make cross-strategy comparison unreliable.
**Question:** What should be the standard backtest period for new research?
**Options:**
- A) **2005-present** -- Matches compare_all.py. Post-Real Plan stability, enough data for 20+ years.
- B) **2010-present** -- More relevant to current market structure, avoids the 2008 crisis "one-off" event.
- C) **Keep flexible** -- Run from earliest available data but report metrics for sub-periods (2005-2015, 2016-2026) to check stability.

### Q7: Constraints -- Long-Only, Tax Optimization, Capacity
**Context:** The simulation engine fully supports short positions (bull_trap_short_backtest.py exists), deferred DARF tax payments, and the R$20K monthly sales exemption. However, most strategies are long-only. The tax drag is significant -- the LowCorr portfolio went from ~7.5% pre-tax to ~7.5% after-tax only because the COPOM Easing component avoids frequent rebalancing. High-turnover strategies lose 1-3% annually to tax.
**Question:** What practical constraints should guide the research?
**Options:**
- A) **Long-only, tax-aware** -- Focus exclusively on long-only strategies but actively optimize for tax efficiency (minimize turnover, use the R$20K monthly exemption, consider deferred DARF). This is realistic for an individual Brazilian investor.
- B) **Long-only, ignore tax** -- Optimize purely for pre-tax Sharpe/return. Tax is already modeled in the simulation engine -- let it be a measurement, not a constraint.
- C) **Allow long/short** -- Include market-neutral and long/short strategies (already have the short infrastructure). Higher complexity but potentially much better Sharpe since you can hedge market beta.

