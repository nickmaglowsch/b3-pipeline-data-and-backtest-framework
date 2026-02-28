# Strategy Ideas from Feature Importance Research

Based on the B3 Feature Importance Discovery (2026-02-28).

Top drivers: **CDI regime > Volatility > Trend position > Liquidity rank**

AUC < 0.50 means direction prediction fails, but cross-sectional sorting by these factors works.

---

## 1. Volatility-Regime Adaptive Low Vol (highest conviction)

Existing `low_volatility_backtest` doesn't condition on anything. Research shows `Ibovespa_vol_20d` and `CDI_3m_change` are the two strongest regime features.

- In **low IBOV vol + falling CDI**: go aggressive -- overweight low-vol stocks (they outperform in calm bull markets)
- In **high IBOV vol + rising CDI**: go defensive -- shift to CDI or tighten the vol filter to the lowest 5%
- Uses the #1 and #2 ranked features as regime switches on top of the #3-5 ranked features for stock selection

## 2. Cross-Sectional Volatility-Adjusted Momentum (long-short or long-only)

Research shows `Rolling_vol_60d`, `ATR_14`, and `Rank_volatility_20d` are all more important than raw momentum returns. Current momentum strategies use raw returns.

- Rank stocks by **momentum / volatility** (Sharpe-like ratio) instead of raw returns
- Select top quintile long (bottom quintile short if long-short)
- Similar to `momentum_sharpe_backtest` but research validates that normalizing by vol is where the actual signal is

## 3. CDI Carry Regime + MA200 Trend Filter (simplest, most novel) **[BUILT]**

The two strongest individual features are `CDI_3m_change` (#1) and `Distance_to_MA200` (#5). No existing strategy combines these two directly.

- **Stock selection**: only buy stocks trading above their MA200 (positive trend)
- **Regime filter**: when CDI 3-month cumulative is rising (tightening cycle), reduce equity exposure or go 100% CDI
- When CDI is falling (easing) AND stock is above MA200 -> full allocation
- Simplest expression of the two strongest signals

## 4. Multi-Factor Composite Score (most complete)

Research shows 8 features stable across both models and all 3 targets. Build a single composite score:

- `z(Rank_volume) + z(1/Rank_volatility_20d) + z(Distance_to_MA200) + z(1/ATR_14) + z(Rolling_vol_60d inverted)`
- Condition entry/exit on `CDI_3m_change` and `Ibovespa_vol_20d`
- Different from existing `multifactor_backtest` which only uses momentum + low vol

## 5. Liquidity Rotation (niche)

`Rank_volume` is consistently top-10 across all models/targets.

- In high IBOV vol: rotate into the **most liquid** stocks (flight to liquidity)
- In low IBOV vol: rotate into **mid-liquidity** stocks (liquidity premium harvesting)
- Existing strategies filter on liquidity but none trade on it as a signal
