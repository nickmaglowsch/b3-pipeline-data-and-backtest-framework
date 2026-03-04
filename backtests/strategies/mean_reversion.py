"""Strategy: Mean Reversion variants.

SimpleMeanReversionStrategy -- original trivial 1-period reversal (buy biggest losers).
MeanReversionCompositeStrategy -- 4-layer composite alpha with regime filter, vol-parity,
                                   and IC signal stability guard.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


class SimpleMeanReversionStrategy(StrategyBase):
    """Original trivial mean-reversion: buy biggest recent losers, equal-weight."""

    @property
    def name(self) -> str:
        return "SimpleMeanReversion"

    @property
    def description(self) -> str:
        return (
            "Short-Term Mean Reversion. Buys the biggest recent losers (1-month negative return). "
            "Top decile of the liquid universe ranked by most negative 1-period return."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec(
                "lookback", "Lookback Periods", "int", 1,
                description="Periods for mean-reversion signal (1 = last month)",
                min_value=1, max_value=6, step=1,
            ),
            ParameterSpec(
                "top_pct", "Top Percentile", "float", 0.10,
                description="Fraction of eligible universe to select (biggest losers)",
                min_value=0.01, max_value=0.50, step=0.01,
            ),
            ParameterSpec(
                "min_price", "Min Price (BRL)", "float", 1.0,
                min_value=0.0, max_value=10.0, step=0.5,
            ),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]

        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        top_pct = params.get("top_pct", 0.10)
        lookback = params.get("lookback", 1)

        # Signal = negative return (bigger loser = higher signal)
        signal = -ret
        has_glitch = (ret > 1.0) | (ret < -0.90)
        signal[has_glitch] = np.nan

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        r = ret.copy()

        prev_sel: set = set()
        start_idx = lookback + 1

        for i in range(start_idx, len(ret)):
            sig_row = signal.iloc[i - 1]
            adtv_row = adtv.iloc[i - 1]
            raw_close_row = raw_close.iloc[i - 1]

            valid_mask = (adtv_row >= min_adtv) & (raw_close_row >= min_price)
            valid = sig_row[valid_mask].dropna()

            if len(valid) < 5:
                sel = prev_sel
            else:
                n_sel = max(5, int(len(valid) * top_pct))
                sel = set(valid.nlargest(n_sel).index)

            if not sel:
                continue

            w = 1.0 / len(sel)
            for t in sel:
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w
            prev_sel = sel

        return r, tw


class MeanReversionCompositeStrategy(StrategyBase):
    """Mean-Reversion Composite Alpha with regime filter, vol-parity, and IC guard."""

    @property
    def name(self) -> str:
        return "MeanRevComposite"

    @property
    def description(self) -> str:
        return (
            "Cross-sectional mean-reversion composite alpha. "
            "Layer 1: CDI/IBOV regime filter. "
            "Layer 2: Volatility reversal + macro-relative momentum + autocorrelation scoring. "
            "Layer 3: Vol-parity portfolio construction (long-only or long-short). "
            "Layer 4: Rolling IC signal stability guard."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
            ParameterSpec("min_price", "Min Price (BRL)", "float", 1.0,
                          min_value=0.0, max_value=10.0, step=0.5),
            ParameterSpec("w_vol", "Weight: Volatility Reversal", "float", 0.25,
                          description="Sub-signal A weight", min_value=0.0, max_value=1.0, step=0.01),
            ParameterSpec("w_macro", "Weight: Macro Momentum", "float", 0.25,
                          description="Sub-signal B weight", min_value=0.0, max_value=1.0, step=0.01),
            ParameterSpec("w_autocorr", "Weight: Autocorrelation", "float", 0.25,
                          description="Sub-signal C weight", min_value=0.0, max_value=1.0, step=0.01),
            ParameterSpec("w_vol_ratio", "Weight: Vol/Mean60 Ratio", "float", 0.25,
                          description="Sub-signal D weight", min_value=0.0, max_value=1.0, step=0.01),
            ParameterSpec("long_pct", "Long Percentile", "float", 0.20,
                          description="Top percentile of alpha score for long leg",
                          min_value=0.05, max_value=0.50, step=0.05),
            ParameterSpec("enable_short", "Enable Short Leg", "choice", "No",
                          choices=["No", "Yes"]),
            ParameterSpec("short_pct", "Short Percentile", "float", 0.20,
                          description="Bottom percentile for short leg (if enabled)",
                          min_value=0.05, max_value=0.50, step=0.05),
            ParameterSpec("short_gross", "Short Gross Exposure", "float", 0.20,
                          description="Total short exposure as fraction of NAV",
                          min_value=0.05, max_value=0.50, step=0.05),
            ParameterSpec("risk_off_exposure", "Risk-Off Equity Exposure", "float", 0.0,
                          description="Equity exposure in Risk-Off (0=100% CDI, 1=ignore regime)",
                          min_value=0.0, max_value=1.0, step=0.1),
            ParameterSpec("enable_vol_parity", "Vol-Parity Sizing", "choice", "Yes",
                          choices=["Yes", "No"]),
            ParameterSpec("enable_stability_guard", "Signal Stability Guard", "choice", "Yes",
                          choices=["Yes", "No"]),
            ParameterSpec("ic_check_freq", "IC Check Frequency (months)", "int", 3,
                          min_value=1, max_value=12, step=1),
            ParameterSpec("ic_trailing_months", "IC Trailing Window (months)", "int", 12,
                          min_value=6, max_value=36, step=3),
            ParameterSpec("ic_flip_consecutive", "IC Flip Threshold", "int", 2,
                          description="Consecutive flipped IC windows to disable signal",
                          min_value=1, max_value=5, step=1),
            ParameterSpec("ibov_drawdown_gate", "IBOV Drawdown Gate", "choice", "No",
                          choices=["No", "Yes"]),
            ParameterSpec("ibov_drawdown_threshold", "IBOV Drawdown Threshold", "float", -0.10,
                          min_value=-0.30, max_value=0.0, step=0.01),
            ParameterSpec("enable_win_rate_filter", "Win Rate Filter", "choice", "Yes",
                          choices=["Yes", "No"]),
            ParameterSpec("win_rate_threshold", "Win Rate Ratio Threshold", "float", 1.5,
                          description="Exclude stocks trending above this win rate ratio",
                          min_value=1.0, max_value=2.5, step=0.1),
            ParameterSpec("atr_ratio_gate", "ATR Ratio Gate", "float", 2.0,
                          description="Exclude stocks with ATR > N x their 60d mean",
                          min_value=1.2, max_value=3.0, step=0.1),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        from backtests.core.mean_rev_helpers import (
            compute_regime_filter,
            compute_alpha_score,
            compute_signal_stability,
        )

        freq = params.get("rebalance_freq", "ME")
        min_adtv = params.get("min_adtv", 1_000_000)
        min_price = params.get("min_price", 1.0)
        long_pct = params.get("long_pct", 0.20)
        enable_short = params.get("enable_short", "No") == "Yes"
        short_pct = params.get("short_pct", 0.20)
        short_gross = params.get("short_gross", 0.20)
        risk_off_exposure = params.get("risk_off_exposure", 0.0)
        enable_vol_parity = params.get("enable_vol_parity", "Yes") == "Yes"
        enable_guard = params.get("enable_stability_guard", "Yes") == "Yes"

        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        cdi_monthly = shared_data["cdi_monthly"]
        ibov_ret = shared_data["ibov_ret"]
        rolling_vol_20d_daily = shared_data["rolling_vol_20d_daily"]
        has_glitch = shared_data.get("has_glitch")

        # Layer 1: Regime filter
        risk_on = compute_regime_filter(shared_data, params, freq)

        # Layer 2: Composite alpha score
        alpha, sub_signals = compute_alpha_score(shared_data, params, freq)

        # Apply glitch mask to alpha
        if has_glitch is not None:
            glitch_m = has_glitch.resample(freq).last()
            alpha[glitch_m == 1] = np.nan

        # Layer 4: Signal stability guard
        if enable_guard:
            fwd_ret = ret.shift(-1)
            base_weights = {
                "sub_A": params.get("w_vol", 0.25),
                "sub_B": params.get("w_macro", 0.25),
                "sub_C": params.get("w_autocorr", 0.25),
                "sub_D": params.get("w_vol_ratio", 0.25),
            }
            stability_weights = compute_signal_stability(
                sub_signals, fwd_ret, base_weights, params
            )
        else:
            stability_weights = None

        # Pre-compute win rate ratio (daily → monthly, aligned to ret columns)
        enable_win_rate_filter = params.get("enable_win_rate_filter", "Yes") == "Yes"
        if enable_win_rate_filter:
            daily_ret_d = shared_data["adj_close"].pct_change()
            win_rate_20d = (daily_ret_d > 0).rolling(20, min_periods=10).mean()
            win_rate_ratio_d = win_rate_20d / (win_rate_20d.rolling(60, min_periods=30).mean() + 1e-8)
            win_rate_ratio_m = (
                win_rate_ratio_d
                .resample(freq).last()
                .reindex(columns=ret.columns)
            )
        else:
            win_rate_ratio_m = None

        # ATR ratio gate (from sub_signals if sub_E was computed)
        atr_ratio_m = sub_signals.get("atr_ratio_m")

        # Vol-parity weights (resample to monthly)
        rolling_vol_20d_m = rolling_vol_20d_daily.resample(freq).last()

        # Build returns and target weights matrices
        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        tw["CDI_ASSET"] = 0.0
        r = ret.copy()
        r["CDI_ASSET"] = cdi_monthly
        r["IBOV"] = ibov_ret

        # Warmup: at minimum 14 periods (12 months + buffer)
        start_idx = 14

        for i in range(start_idx, len(ret)):
            date = ret.index[i]

            # Layer 1: Check regime
            if date in risk_on.index:
                is_risk_on = bool(risk_on.loc[date])
            else:
                is_risk_on = False

            if not is_risk_on and risk_off_exposure == 0.0:
                # Full CDI allocation
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                continue

            # Layer 2+4: Get alpha score for this month
            if stability_weights is not None and date in stability_weights.index:
                alpha_idx = i - 1
                if alpha_idx < 0 or alpha_idx >= len(sub_signals["sub_A"]):
                    tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                    continue
                # Build alpha_row dynamically over all monitored sub-signals
                alpha_row = sum(
                    stability_weights.loc[date, f"w_{s}"] * sub_signals[s].iloc[alpha_idx]
                    for s in ["sub_A", "sub_B", "sub_C", "sub_D"]
                    if s in sub_signals and f"w_{s}" in stability_weights.columns
                )
            else:
                alpha_idx = i - 1
                if alpha_idx < 0 or alpha_idx >= len(alpha):
                    tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                    continue
                alpha_row = alpha.iloc[alpha_idx]

            # Edge case: all signals degenerate
            if alpha_row.notna().sum() < 5 or alpha_row.abs().sum() == 0:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                continue

            # Layer 3: Portfolio construction
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            adtv_rank = adtv_r.rank(pct=True)
            mask = (adtv_r >= min_adtv) & (raw_r >= min_price) & (adtv_rank >= 0.20)
            valid = alpha_row[mask].dropna()

            if len(valid) < 5:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                continue

            # Win rate filter: exclude stocks trending strongly upward
            # (high recent win rate vs own history → momentum, not mean-reversion)
            if win_rate_ratio_m is not None and date in win_rate_ratio_m.index:
                wr_threshold = params.get("win_rate_threshold", 1.5)
                wr_row = win_rate_ratio_m.loc[date].reindex(valid.index).fillna(1.0)
                valid = valid[wr_row < wr_threshold]

            # ATR ratio gate: exclude stocks with volatility expanding far above own norm
            # (structural vol expansion period — mean-reversion unlikely)
            if atr_ratio_m is not None and date in atr_ratio_m.index:
                atr_gate_threshold = params.get("atr_ratio_gate", 2.0)
                atr_row = atr_ratio_m.loc[date].reindex(valid.index).fillna(1.0)
                valid = valid[atr_row < atr_gate_threshold]

            if len(valid) < 5:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
                continue

            # Long leg: top quintile by alpha score
            n_long = max(5, int(len(valid) * long_pct))
            long_tickers = valid.nlargest(n_long).index.tolist()

            # Determine gross exposure
            long_gross = risk_off_exposure if not is_risk_on else 1.0

            # Vol-parity sizing for long leg
            if enable_vol_parity and i - 1 < len(rolling_vol_20d_m):
                vol_row = rolling_vol_20d_m.iloc[i - 1]
                inv_vol = (
                    1.0 / vol_row.reindex(long_tickers).replace(0, np.nan).dropna()
                )
                if len(inv_vol) == 0:
                    long_weights = pd.Series(
                        1.0 / len(long_tickers), index=long_tickers
                    )
                else:
                    long_weights = inv_vol / inv_vol.sum()
            else:
                long_weights = pd.Series(
                    1.0 / len(long_tickers), index=long_tickers
                )

            # Assign long weights
            for t, w in long_weights.items():
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w * long_gross

            # Short leg (optional)
            if enable_short and is_risk_on:
                n_short = max(3, int(len(valid) * short_pct))
                short_tickers = valid.nsmallest(n_short).index.tolist()

                # Remove overlap with long leg
                short_tickers = [t for t in short_tickers if t not in long_tickers]
                if short_tickers:
                    short_w_per = short_gross / len(short_tickers)
                    for t in short_tickers:
                        if t in tw.columns:
                            tw.iloc[i, tw.columns.get_loc(t)] = -short_w_per

            # If risk-off with partial exposure, allocate remainder to CDI
            if not is_risk_on and risk_off_exposure > 0.0:
                cdi_alloc = 1.0 - long_gross
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = max(0.0, cdi_alloc)

        return r, tw
