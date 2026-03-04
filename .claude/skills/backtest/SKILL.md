---
name: backtest
description: Scaffold and create a new backtest plugin strategy for the B3 data pipeline. Use when the user asks to create a new strategy, backtest, or trading signal.
argument-hint: "<describe the strategy>"
disable-model-invocation: true
---

# Backtest Strategy Builder

Build a new backtest strategy from the description below and write it to disk.

## Input

$ARGUMENTS

---

## Defaults (apply unless the description overrides)

- **Mode**: Plugin strategy (extends `StrategyBase`, auto-registered for UI)
- **File**: `backtests/strategies/<snake_case_name>.py`
- **Rebalance**: Monthly (`ME`)
- **Universe filters**: ADTV >= 1M BRL, raw close >= 1.0 BRL, no glitch rows
- **Portfolio**: Equal weight, top 10% of valid universe

---

## Framework Reference

### Key imports for plugin strategy
```python
import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV,
    COMMON_REBALANCE_FREQ, COMMON_MONTHLY_SALES_EXEMPTION,
)
```

### `shared_data` keys
```python
shared_data["ret"]            # Period returns (date x ticker)
shared_data["log_ret"]        # Log returns
shared_data["px"]             # Adj close resampled
shared_data["raw_close"]      # Raw close resampled
shared_data["adtv"]           # Avg daily traded volume (BRL)
shared_data["has_glitch"]     # 1 where return > 100% or < -45%
shared_data["adj_close"]      # Daily adj close
shared_data["close_px"]       # Daily raw close
shared_data["ibov_ret"]       # IBOV monthly returns (Series)
shared_data["ibov_px"]        # IBOV price (Series)
shared_data["cdi_monthly"]    # CDI monthly returns (Series)
shared_data["cdi_daily"]      # CDI daily returns (Series)
shared_data["is_easing"]      # COPOM easing flag 1/0 (Series)
shared_data["ibov_calm"]      # IBOV vol percentile <= 70% (Series)
shared_data["ibov_uptrend"]   # IBOV monthly return > 0 (Series)
shared_data["ibov_above"]     # IBOV above MA10 (Series)
shared_data["above_ma200"]    # Stocks above 200-day MA (DataFrame)
shared_data["dist_ma200"]     # Distance to MA200 (DataFrame)
shared_data["ma200_m"]        # Resampled MA200 (DataFrame)
shared_data["mf_composite"]   # 50% momentum + 50% low-vol ranks (DataFrame)
shared_data["vol_5m"]         # 5-period rolling volatility (DataFrame)
shared_data["atr_m"]          # ATR proxy (DataFrame)
shared_data["autocorr_20d"]           # 20-day autocorrelation (DataFrame)
shared_data["autocorr_60d"]           # 60-day autocorrelation (DataFrame)
shared_data["high_low_range_20d"]     # 20-day high/low range (DataFrame)
shared_data["rolling_vol_20d_daily"]  # 20-day rolling vol daily (DataFrame)
shared_data["rolling_vol_60d_daily"]  # 60-day rolling vol daily (DataFrame)
```

### `ParameterSpec` constants
```python
COMMON_START_DATE          # default "2005-01-01"
COMMON_END_DATE            # default today
COMMON_INITIAL_CAPITAL     # default 100_000 BRL
COMMON_TAX_RATE            # default 0.15
COMMON_SLIPPAGE            # default 0.001
COMMON_MIN_ADTV            # default 1_000_000 BRL
COMMON_REBALANCE_FREQ      # choices: "ME", "QE", "W-FRI"
COMMON_MONTHLY_SALES_EXEMPTION  # default 20_000 BRL
```

### `ParameterSpec` constructor
```python
ParameterSpec(name, label, type, default, min_value=None, max_value=None, step=None, choices=None)
# type: "int" | "float" | "str" | "date" | "choice"
```

### Plugin strategy template
```python
class <ClassName>Strategy(StrategyBase):

    @property
    def name(self) -> str:
        return "<ShortName>"

    @property
    def description(self) -> str:
        return "<One-sentence description.>"

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV,
            COMMON_REBALANCE_FREQ, COMMON_MONTHLY_SALES_EXEMPTION,
            # strategy-specific params here
        ]

    def generate_signals(self, shared_data: dict, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        has_glitch = shared_data["has_glitch"]

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)

        for i in range(<lookback> + 1, len(ret)):
            sig_row = <signal>.iloc[i - 1]
            adtv_row = adtv.iloc[i - 1]
            raw_close_row = raw_close.iloc[i - 1]
            has_glitch_row = has_glitch.iloc[i - 1]

            valid_mask = (adtv_row >= params["min_adtv"]) & (raw_close_row >= 1.0) & (has_glitch_row == 0)
            valid = sig_row[valid_mask].dropna()
            if valid.empty:
                continue

            n = max(5, int(len(valid) * params["top_pct"]))
            selected = valid.nlargest(n).index.tolist()
            w = 1.0 / len(selected)
            for t in selected:
                tw.iloc[i, tw.columns.get_loc(t)] = w

        return ret, tw
```

### Signal patterns by type
- **Momentum**: `signal = ret.rolling(lookback).sum()` → `nlargest` (top performers)
- **Mean Reversion**: `signal = -ret.rolling(lookback).sum()` → `nlargest` (biggest losers)
- **Low Volatility**: `signal = -ret.rolling(lookback).std()` → `nlargest` (lowest vol)
- **Regime-based**: gate the selection on `shared_data["is_easing"].iloc[i-1]` or similar; return empty weights when regime is off
- **Multifactor**: rank each signal 0–1 with `.rank(pct=True)`, combine: `signal = w1 * rank1 + w2 * rank2`

---

## Instructions

1. Read `$ARGUMENTS` and infer: strategy name, signal logic, any non-default parameters or portfolio construction rules.
2. Generate the complete plugin strategy file following the template above. Fill in all signal logic — no TODOs left.
3. Write the file to `backtests/strategies/<snake_case_name>.py` using the Write tool.
4. Report: file path created, class name, and a one-line summary of the signal logic.

The strategy is auto-discovered by `StrategyRegistry` — no registration step needed.
