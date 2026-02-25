"""
Compute split and dividend adjustments for price data.

Implements:
1. Split adjustment factors from B3 official data (backward cumulative)
2. Dividend/JCP adjustment factors (Yahoo-style backward cumulative)
"""

import logging
from datetime import date, datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config

logger = logging.getLogger(__name__)


def _normalize_date(val) -> Optional[date]:
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


def convert_stock_actions_to_splits(stock_actions: pd.DataFrame) -> pd.DataFrame:
    """
    Convert B3 stock actions to the format needed for split adjustments.

    B3 factors:
    - STOCK_SPLIT (DESDOBRAMENTO): factor > 1 means new_shares = old_shares * factor
      -> split_factor = 1/factor (e.g., 100x split -> factor=0.01)
    - REVERSE_SPLIT (GRUPAMENTO): factor < 1 means new_shares = old_shares * factor
      -> split_factor = 1/factor (e.g., 0.01 reverse -> factor=100)
    - BONUS_SHARES (BONIFICACAO): factor represents bonus ratio
      -> split_factor = 1/(1+factor) (e.g., 33.33% bonus -> factor=0.75)

    The split_factor represents: old_shares / new_shares
    - split_factor < 1: price increases (stock split)
    - split_factor > 1: price decreases (reverse split)

    Args:
        stock_actions: DataFrame from B3 with columns [isin_code, ex_date, action_type, factor]

    Returns:
        DataFrame with columns [isin_code, ex_date, split_factor, description]
    """
    if stock_actions.empty:
        return pd.DataFrame(
            columns=["isin_code", "ex_date", "split_factor", "description"]
        )

    splits = []
    for _, row in stock_actions.iterrows():
        isin_code = row["isin_code"]
        ex_date = _normalize_date(row["ex_date"])
        action_type = row["action_type"]
        b3_factor = row["factor"]

        if ex_date is None:
            continue

        if action_type == config.EVENT_TYPE_STOCK_SPLIT:
            split_factor = 1.0 / b3_factor
            description = f"Split {b3_factor}:1"
        elif action_type == config.EVENT_TYPE_REVERSE_SPLIT:
            split_factor = 1.0 / b3_factor
            description = (
                f"Reverse split 1:{1 / b3_factor:.0f}"
                if b3_factor < 1
                else f"Reverse split {1 / b3_factor:.0f}:1"
            )
        elif action_type == config.EVENT_TYPE_BONUS_SHARES:
            split_factor = 1.0 / (1.0 + b3_factor / 100.0)
            description = f"Bonus {b3_factor}%"
        else:
            continue

        splits.append(
            {
                "isin_code": isin_code,
                "ex_date": ex_date,
                "split_factor": round(split_factor, 10),
                "description": description,
            }
        )

    if splits:
        return pd.DataFrame(splits)
    return pd.DataFrame(columns=["isin_code", "ex_date", "split_factor", "description"])


def compute_split_adjustment_factors(
    prices: pd.DataFrame, splits: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute backward cumulative split adjustment factors based on ISIN codes.

    For each ISIN:
    1. Sort splits by date descending
    2. For each trading date, cumulative_factor = product of all split_factors
       for splits on or after that date
    3. Apply to open, high, low, close
    4. Adjust volume inversely: volume / cumulative_factor

    Args:
        prices: DataFrame with raw price data
        splits: DataFrame with splits (from B3 stock_actions)

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

    logger.info(f"Applying {len(splits)} split adjustments...")

    isins_with_splits = splits["isin_code"].unique()

    for isin_code in isins_with_splits:
        isin_mask = result["isin_code"] == isin_code
        isin_prices = result[isin_mask].copy()

        if isin_prices.empty:
            continue

        isin_splits = splits[splits["isin_code"] == isin_code].copy()
        isin_splits = isin_splits.sort_values("ex_date", ascending=False)

        cumulative_factor = 1.0
        split_idx = 0
        n_splits = len(isin_splits)

        isin_prices = isin_prices.sort_values("date", ascending=False)

        for idx in isin_prices.index:
            row_date = _normalize_date(result.loc[idx, "date"])
            if row_date is None:
                continue

            while split_idx < n_splits:
                split_row = isin_splits.iloc[split_idx]
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
    Compute Yahoo-style backward cumulative dividend adjustment factors based on ISIN.

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

    dividend_types = [
        config.EVENT_TYPE_CASH_DIVIDEND,
        config.EVENT_TYPE_JCP,
        "Dividendo",
        "JCP",
        "Rend. Tributado",
    ]
    dividend_actions = corporate_actions[
        corporate_actions["event_type"].isin(dividend_types)
    ].copy()

    if dividend_actions.empty:
        logger.info("No dividend/JCP events to apply")
        return result

    logger.info(
        f"Applying dividend/JCP adjustments ({len(dividend_actions)} events)..."
    )

    result["date_normalized"] = result["date"].apply(_normalize_date)
    dividend_actions["event_date_normalized"] = dividend_actions["event_date"].apply(
        _normalize_date
    )

    isins_with_dividends = dividend_actions["isin_code"].unique()
    n_isins = len(isins_with_dividends)

    for i, isin_code in enumerate(isins_with_dividends, 1):
        if i % 100 == 0 or i == n_isins:
            logger.info(f"Applying dividend adjustments: {i}/{n_isins}")

        isin_mask = result["isin_code"] == isin_code
        isin_prices = result.loc[isin_mask].copy()

        if isin_prices.empty:
            continue

        isin_prices = isin_prices.sort_values("date_normalized").reset_index(drop=True)

        isin_dividends = dividend_actions[
            dividend_actions["isin_code"] == isin_code
        ].copy()
        isin_dividends = isin_dividends.dropna(subset=["event_date_normalized"])

        if isin_dividends.empty:
            continue

        div_factors = []
        for _, div_row in isin_dividends.iterrows():
            ex_date = div_row["event_date_normalized"]
            div_amount = div_row["value"]

            if div_amount is None or div_amount <= 0:
                continue

            prev_mask = isin_prices["date_normalized"] < ex_date
            prev_data = isin_prices[prev_mask]

            if prev_data.empty:
                continue

            # IMPORTANT: We must use the RAW close price to calculate the yield factor,
            # because B3 dividend amounts are raw historical amounts (not split-adjusted).
            prev_close = prev_data.iloc[-1]["close"]

            if prev_close <= 0:
                continue

            factor = 1.0 - (div_amount / prev_close)
            factor = max(0.0, min(1.0, factor))

            div_factors.append({"ex_date": ex_date, "factor": factor})

        if not div_factors:
            continue

        div_factors_df = pd.DataFrame(div_factors)
        div_factors_df = div_factors_df.sort_values("ex_date", ascending=True)

        isin_prices = isin_prices.sort_values(
            "date_normalized", ascending=True
        ).reset_index(drop=True)

        div_factors_df["cumulative_factor"] = div_factors_df["factor"][::-1].cumprod()[
            ::-1
        ]
        div_factors_df = div_factors_df.sort_values("ex_date", ascending=True)

        last_cumulative = 1.0
        adj_close_values = isin_prices["split_adj_close"].values.copy()
        dates = isin_prices["date_normalized"].values

        div_idx = len(div_factors_df) - 1
        for j in range(len(dates) - 1, -1, -1):
            row_date = dates[j]

            while div_idx >= 0 and div_factors_df.iloc[div_idx]["ex_date"] > row_date:
                last_cumulative *= div_factors_df.iloc[div_idx]["factor"]
                div_idx -= 1

            adj_close_values[j] = adj_close_values[j] * last_cumulative

        result.loc[isin_prices.index, "adj_close"] = adj_close_values

    result = result.drop(columns=["date_normalized"])
    logger.info("Dividend/JCP adjustments applied")
    return result


def compute_all_adjustments(
    prices: pd.DataFrame,
    corporate_actions: pd.DataFrame,
    stock_actions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute all adjustments using B3 official data.

    Args:
        prices: Raw price DataFrame
        corporate_actions: Corporate actions DataFrame (dividends, JCP)
        stock_actions: Stock actions DataFrame (splits, reverse splits, bonuses)

    Returns:
        Tuple of (adjusted_prices, splits)
    """
    logger.info("Starting adjustment computation...")

    splits = convert_stock_actions_to_splits(stock_actions)
    logger.info(f"Converted {len(stock_actions)} stock actions to {len(splits)} splits")

    prices = compute_split_adjustment_factors(prices, splits)

    prices = compute_dividend_adjustment_factors(prices, corporate_actions)

    logger.info("Adjustment computation complete")

    return prices, splits
