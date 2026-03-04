import numpy as np
import pandas as pd
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV,
    COMMON_REBALANCE_FREQ, COMMON_MONTHLY_SALES_EXEMPTION,
)


class WinRateMeanReversionStrategy(StrategyBase):

    @property
    def name(self) -> str:
        return "WinRateMeanRev"

    @property
    def description(self) -> str:
        return (
            "Mean-reversion strategy with a win-rate momentum filter: excludes stocks whose "
            "recent win rate is significantly above their 60-day historical average (trending stocks), "
            "then selects the biggest recent losers from the remaining universe."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE,
            COMMON_SLIPPAGE,
            COMMON_MIN_ADTV,
            COMMON_REBALANCE_FREQ,
            COMMON_MONTHLY_SALES_EXEMPTION,
            ParameterSpec("lookback", "Mean-Rev Lookback (months)", "int", 1,
                         min_value=1, max_value=6, step=1),
            ParameterSpec("win_rate_window", "Win Rate Window (days)", "int", 20,
                         min_value=10, max_value=60, step=5),
            ParameterSpec("win_rate_high_threshold", "Win Rate Ratio Upper Bound (exclude trending up)", "float", 1.25,
                         min_value=1.00, max_value=2.00, step=0.05),
            ParameterSpec("win_rate_low_threshold", "Win Rate Ratio Lower Bound (exclude trending down)", "float", 0.75,
                         min_value=0.25, max_value=1.00, step=0.05),
            ParameterSpec("top_pct", "Selection Percentile", "float", 0.10,
                         min_value=0.01, max_value=0.30, step=0.01),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]
        has_glitch = shared_data["has_glitch"]
        adj_close = shared_data["adj_close"]

        lookback = params["lookback"]
        win_rate_window = params["win_rate_window"]
        win_rate_high_threshold = params["win_rate_high_threshold"]
        win_rate_low_threshold = params["win_rate_low_threshold"]
        top_pct = params["top_pct"]
        min_adtv = params["min_adtv"]
        freq = params.get("rebalance_freq", "ME")

        # --- Mean-reversion signal: buy biggest recent losers ---
        signal = -ret.rolling(lookback).sum()
        signal[has_glitch == 1] = np.nan

        # --- Win rate ratio from daily data ---
        # win_rate: fraction of up days over the rolling window
        # ratio: recent win rate relative to its own 60-day mean
        # High ratio → stock in consistent uptrend → likely to keep trending, exclude from mean-rev
        daily_ret = adj_close.pct_change()
        win_rate = (daily_ret > 0).rolling(win_rate_window).mean()
        win_rate_60d_mean = win_rate.rolling(60).mean()
        ratio_win_rate = win_rate / (win_rate_60d_mean + 1e-8)

        # Resample to rebalance frequency (last value of each period) and align to ret index
        ratio_m = (
            ratio_win_rate
            .resample(freq)
            .last()
            .reindex(columns=ret.columns)
            .reindex(ret.index, method="ffill")
        )

        # Warm-up: need at least lookback months + enough daily data for win_rate_window + 60d mean
        min_lookback = lookback + (win_rate_window + 60) // 21 + 2

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)

        for i in range(min_lookback, len(ret)):
            sig_row = signal.iloc[i - 1]
            adtv_row = adtv.iloc[i - 1]
            raw_close_row = raw_close.iloc[i - 1]
            has_glitch_row = has_glitch.iloc[i - 1]
            ratio_row = ratio_m.iloc[i - 1]

            # Standard universe filters
            valid_mask = (
                (adtv_row >= min_adtv) &
                (raw_close_row >= 1.0) &
                (has_glitch_row == 0)
            )

            # Win rate filter: exclude stocks trending strongly in either direction.
            # Keep only stocks with win rate ratio near 1.0 — normal consistency, not chronically down.
            in_neutral_zone = (ratio_row >= win_rate_low_threshold) & (ratio_row <= win_rate_high_threshold)
            valid_mask = valid_mask & in_neutral_zone

            valid = sig_row[valid_mask].dropna()
            if valid.empty:
                continue

            n = max(5, int(len(valid) * top_pct))
            selected = valid.nlargest(n).index.tolist()
            w = 1.0 / len(selected)
            for t in selected:
                tw.iloc[i, tw.columns.get_loc(t)] = w

        return ret, tw
