"""
Compute split and dividend adjustments for price data.

Implements:
1. Split detection from price gaps
2. Split adjustment factors (backward cumulative)
3. Dividend/JCP adjustment factors (Yahoo-style backward cumulative)
"""

import logging
from datetime import date, datetime
from typing import List, Tuple

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def _normalize_date(val):
    """Convert various date formats to datetime.date object."""
    if val is None:
        return None
    if isinstance(val, date) and not isinstance(val, datetime):
        return val
    if isinstance(val, datetime):
        return val.date()
    if isinstance(val, str):
        try:
            if "T" in val:
                return datetime.fromisoformat(val.replace("Z", "+00:00")).date()
            return datetime.strptime(val[:10], "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _find_nearest_split_ratio(price_ratio: float) -> Tuple[float, str]:
    """
    Find the nearest common split ratio for a given price ratio.

    Args:
        price_ratio: The ratio close[t] / close[t-1]

    Returns:
        Tuple of (split_factor, description)
        split_factor = old_shares / new_shares
    """
    best_diff = float("inf")
    best_factor = 1.0
    best_desc = "1:1"

    for old_shares, new_shares in config.COMMON_SPLIT_RATIOS:
        expected_ratio = new_shares / old_shares
        diff = abs(price_ratio - expected_ratio)

        if diff < best_diff:
            best_diff = diff
            best_factor = old_shares / new_shares
            best_desc = f"{old_shares}:{new_shares}"

    return best_factor, best_desc


def detect_splits(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Detect stock splits from price discontinuities.

    Algorithm:
    1. For each ticker, compute daily close ratio: close[t] / close[t-1]
    2. Flag potential split if ratio > SPLIT_DETECTION_THRESHOLD_HIGH
       or ratio < SPLIT_DETECTION_THRESHOLD_LOW
    3. Infer split factor from nearest round ratio

    Args:
        prices: DataFrame with columns [date, ticker, close]

    Returns:
        DataFrame with columns [ticker, ex_date, split_factor, description]
    """
    if prices.empty:
        return pd.DataFrame(
            columns=["ticker", "ex_date", "split_factor", "description"]
        )

    logger.info("Detecting splits from price gaps...")

    splits = []
    tickers = prices["ticker"].unique()

    for ticker in tickers:
        ticker_data = prices[prices["ticker"] == ticker].copy()
        ticker_data = ticker_data.sort_values("date").reset_index(drop=True)

        if len(ticker_data) < 2:
            continue

        ticker_data["prev_close"] = ticker_data["close"].shift(1)
        ticker_data["price_ratio"] = ticker_data["close"] / ticker_data["prev_close"]

        potential_splits = ticker_data[
            (ticker_data["price_ratio"] > config.SPLIT_DETECTION_THRESHOLD_HIGH)
            | (ticker_data["price_ratio"] < config.SPLIT_DETECTION_THRESHOLD_LOW)
        ].copy()

        for _, row in potential_splits.iterrows():
            price_ratio = row["price_ratio"]
            split_factor, description = _find_nearest_split_ratio(price_ratio)

            if split_factor != 1.0:
                splits.append(
                    {
                        "ticker": ticker,
                        "ex_date": _normalize_date(row["date"]),
                        "split_factor": round(split_factor, 6),
                        "description": description,
                    }
                )

    if splits:
        result = pd.DataFrame(splits)
        result = result.drop_duplicates(subset=["ticker", "ex_date"], keep="first")
        result = result.sort_values(["ticker", "ex_date"]).reset_index(drop=True)
        logger.info(
            f"Detected {len(result)} potential splits across {len(tickers)} tickers"
        )
        return result

    logger.info("No splits detected")
    return pd.DataFrame(columns=["ticker", "ex_date", "split_factor", "description"])


def compute_split_adjustment_factors(
    prices: pd.DataFrame, splits: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute backward cumulative split adjustment factors.

    For each ticker:
    1. Sort splits by date descending
    2. For each trading date, cumulative_factor = product of all split_factors
       for splits on or after that date
    3. Apply to open, high, low, close
    4. Adjust volume inversely: volume / cumulative_factor

    Args:
        prices: DataFrame with raw price data
        splits: DataFrame with detected splits

    Returns:
        DataFrame with split-adjusted columns added
    """
    result = prices.copy()

    result["split_adj_open"] = result["open"].astype(float)
    result["split_adj_high"] = result["high"].astype(float)
    result["split_adj_low"] = result["low"].astype(float)
    result["split_adj_close"] = result["close"].astype(float)
    result["split_adj_volume"] = result["volume"].astype(float)

    if splits.empty:
        logger.info("No splits to apply")
        return result

    logger.info("Applying split adjustments...")

    tickers_with_splits = splits["ticker"].unique()

    for ticker in tickers_with_splits:
        ticker_mask = result["ticker"] == ticker
        ticker_prices = result[ticker_mask].copy()

        if ticker_prices.empty:
            continue

        ticker_splits = splits[splits["ticker"] == ticker].copy()
        ticker_splits = ticker_splits.sort_values("ex_date", ascending=False)

        cumulative_factor = 1.0
        split_idx = 0
        n_splits = len(ticker_splits)

        ticker_prices = ticker_prices.sort_values("date", ascending=False)

        for idx in ticker_prices.index:
            row_date = _normalize_date(result.loc[idx, "date"])
            if row_date is None:
                continue

            while split_idx < n_splits:
                split_row = ticker_splits.iloc[split_idx]
                split_date = _normalize_date(split_row["ex_date"])
                if split_date is None:
                    split_idx += 1
                    continue

                if split_date >= row_date:
                    cumulative_factor *= split_row["split_factor"]
                    split_idx += 1
                else:
                    break

            result.loc[idx, "split_adj_open"] = (
                result.loc[idx, "open"] * cumulative_factor
            )
            result.loc[idx, "split_adj_high"] = (
                result.loc[idx, "high"] * cumulative_factor
            )
            result.loc[idx, "split_adj_low"] = (
                result.loc[idx, "low"] * cumulative_factor
            )
            result.loc[idx, "split_adj_close"] = (
                result.loc[idx, "close"] * cumulative_factor
            )

            if cumulative_factor > 0:
                result.loc[idx, "split_adj_volume"] = (
                    result.loc[idx, "volume"] / cumulative_factor
                )

    logger.info("Split adjustments applied")
    return result


def compute_dividend_adjustment_factors(
    prices: pd.DataFrame, corporate_actions: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute Yahoo-style backward cumulative dividend adjustment factors.

    For each dividend/JCP event:
    1. Get previous close (trading day before ex_date)
    2. factor_on_exdate = 1 - (dividend_amount / prev_close)
    3. Build cumulative product backward from most recent to oldest
    4. adj_close = split_adj_close * cumulative_factor

    Note: Only adjusts close price (adj_close), not open/high/low.

    Args:
        prices: DataFrame with split-adjusted prices
        corporate_actions: DataFrame with dividend/JCP events

    Returns:
        DataFrame with adj_close column added
    """
    result = prices.copy()
    result["adj_close"] = result["split_adj_close"].copy()

    if corporate_actions.empty:
        logger.info("No corporate actions to apply")
        return result

    dividend_actions = corporate_actions[
        corporate_actions["event_type"].isin(["Dividendo", "JCP", "Rend. Tributado"])
    ].copy()

    if dividend_actions.empty:
        logger.info("No dividend/JCP events to apply")
        return result

    logger.info("Applying dividend/JCP adjustments...")

    result["date_normalized"] = result["date"].apply(_normalize_date)
    dividend_actions["event_date_normalized"] = dividend_actions["event_date"].apply(
        _normalize_date
    )

    tickers_with_dividends = dividend_actions["ticker"].unique()
    n_tickers = len(tickers_with_dividends)

    for i, ticker in enumerate(tickers_with_dividends, 1):
        if i % 100 == 0 or i == n_tickers:
            logger.info(f"Applying dividend adjustments: {i}/{n_tickers}")

        ticker_mask = result["ticker"] == ticker
        ticker_prices = result.loc[ticker_mask].copy()

        if ticker_prices.empty:
            continue

        ticker_prices = ticker_prices.sort_values("date_normalized").reset_index(
            drop=True
        )

        ticker_dividends = dividend_actions[dividend_actions["ticker"] == ticker].copy()
        ticker_dividends = ticker_dividends.dropna(subset=["event_date_normalized"])

        if ticker_dividends.empty:
            continue

        div_factors = []
        for _, div_row in ticker_dividends.iterrows():
            ex_date = div_row["event_date_normalized"]
            div_amount = div_row["value"]

            prev_mask = ticker_prices["date_normalized"] < ex_date
            prev_data = ticker_prices[prev_mask]

            if prev_data.empty:
                continue

            prev_close = prev_data.iloc[-1]["split_adj_close"]

            if prev_close <= 0:
                continue

            factor = 1.0 - (div_amount / prev_close)
            factor = max(0.0, min(1.0, factor))

            div_factors.append({"ex_date": ex_date, "factor": factor})

        if not div_factors:
            continue

        div_factors_df = pd.DataFrame(div_factors)
        div_factors_df = div_factors_df.sort_values("ex_date", ascending=True)

        ticker_prices = ticker_prices.sort_values(
            "date_normalized", ascending=True
        ).reset_index(drop=True)

        div_factors_df["cumulative_factor"] = div_factors_df["factor"][::-1].cumprod()[
            ::-1
        ]
        div_factors_df = div_factors_df.sort_values("ex_date", ascending=True)

        last_cumulative = 1.0
        adj_close_values = ticker_prices["split_adj_close"].values.copy()
        dates = ticker_prices["date_normalized"].values

        div_idx = len(div_factors_df) - 1
        for j in range(len(dates) - 1, -1, -1):
            row_date = dates[j]

            while div_idx >= 0 and div_factors_df.iloc[div_idx]["ex_date"] <= row_date:
                last_cumulative *= div_factors_df.iloc[div_idx]["factor"]
                div_idx -= 1

            adj_close_values[j] = adj_close_values[j] * last_cumulative

        result.loc[ticker_prices.index, "adj_close"] = adj_close_values

    result = result.drop(columns=["date_normalized"])
    logger.info("Dividend/JCP adjustments applied")
    return result

    dividend_actions = corporate_actions[
        corporate_actions["event_type"].isin(["Dividendo", "JCP", "Rend. Tributado"])
    ].copy()

    if dividend_actions.empty:
        logger.info("No dividend/JCP events to apply")
        return result

    logger.info("Applying dividend/JCP adjustments...")

    tickers_with_dividends = dividend_actions["ticker"].unique()

    for ticker in tickers_with_dividends:
        ticker_mask = result["ticker"] == ticker
        ticker_prices = result[ticker_mask].copy()

        if ticker_prices.empty:
            continue

        ticker_prices = ticker_prices.sort_values("date").reset_index(drop=True)
        ticker_prices["date_normalized"] = ticker_prices["date"].apply(_normalize_date)

        ticker_dividends = dividend_actions[dividend_actions["ticker"] == ticker].copy()
        ticker_dividends = ticker_dividends.sort_values("event_date", ascending=False)

        ticker_dividends["event_date_normalized"] = ticker_dividends[
            "event_date"
        ].apply(_normalize_date)

        adj_factors = []

        for _, div_row in ticker_dividends.iterrows():
            ex_date = div_row["event_date_normalized"]
            if ex_date is None:
                continue

            div_amount = div_row["value"]

            prev_day_mask = ticker_prices["date_normalized"] < ex_date
            prev_day_data = ticker_prices[prev_day_mask]

            if prev_day_data.empty:
                continue

            prev_close = prev_day_data.iloc[-1]["split_adj_close"]

            if prev_close <= 0:
                continue

            factor = 1.0 - (div_amount / prev_close)
            factor = max(0.0, min(1.0, factor))

            adj_factors.append(
                {
                    "ex_date": ex_date,
                    "factor": factor,
                    "dividend": div_amount,
                    "prev_close": prev_close,
                }
            )

        if not adj_factors:
            continue

        adj_factors_df = pd.DataFrame(adj_factors)
        adj_factors_df = adj_factors_df.sort_values("ex_date", ascending=False)

        cumulative_factor = 1.0
        factor_idx = 0
        n_factors = len(adj_factors_df)

        ticker_prices_desc = ticker_prices.sort_values("date", ascending=False)

        for _, row in ticker_prices_desc.iterrows():
            row_date = row["date_normalized"]
            if row_date is None:
                continue

            while factor_idx < n_factors:
                factor_row = adj_factors_df.iloc[factor_idx]
                factor_date = factor_row["ex_date"]

                if factor_date >= row_date:
                    cumulative_factor *= factor_row["factor"]
                    factor_idx += 1
                else:
                    break

            idx_mask = (result["ticker"] == ticker) & (
                result["date"].apply(lambda x: x.date() if hasattr(x, "date") else x)
                == row_date
            )
            result.loc[idx_mask, "adj_close"] = (
                row["split_adj_close"] * cumulative_factor
            )

    logger.info("Dividend/JCP adjustments applied")
    return result


def compute_all_adjustments(
    prices: pd.DataFrame, corporate_actions: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all adjustments: split detection, split adjustment, dividend adjustment.

    Args:
        prices: Raw price DataFrame
        corporate_actions: Corporate actions DataFrame

    Returns:
        Tuple of (adjusted_prices, detected_splits)
    """
    logger.info("Starting adjustment computation...")

    splits = detect_splits(prices)

    prices = compute_split_adjustment_factors(prices, splits)

    prices = compute_dividend_adjustment_factors(prices, corporate_actions)

    logger.info("Adjustment computation complete")

    return prices, splits
