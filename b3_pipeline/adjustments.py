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

# Sanity bounds for split adjustment factors.
# Individual event: allow up to 1000:1 splits or 1:1000 reverse splits.
# Anything beyond this is almost certainly bad data from B3's API.
_MAX_INDIVIDUAL_SPLIT_FACTOR = 1000.0
_MIN_INDIVIDUAL_SPLIT_FACTOR = 1.0 / 1000.0

# Cumulative factor: allow up to 100,000x total adjustment across all events.
_MAX_CUMULATIVE_FACTOR = 100_000.0
_MIN_CUMULATIVE_FACTOR = 1.0 / 100_000.0


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

        # Sanity check: skip obviously bad factors from B3's API
        if split_factor > _MAX_INDIVIDUAL_SPLIT_FACTOR or split_factor < _MIN_INDIVIDUAL_SPLIT_FACTOR:
            logger.warning(
                f"Skipping extreme split factor for {isin_code} on {ex_date}: "
                f"{description} (factor={split_factor:.6f})"
            )
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
    1. Sort splits by date ascending, build suffix-product array
    2. Use np.searchsorted to assign each price row its factor in O(n log m)
    3. Apply to open, high, low, close via vectorized column writes

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

    if splits.empty:
        logger.info("No splits to apply")
        return result

    logger.info(f"Applying {len(splits)} split adjustments...")

    # Pre-normalize date column once for all ISINs
    result["_date_ts"] = pd.to_datetime(result["date"], errors="coerce")

    isins_with_splits = splits["isin_code"].unique()

    for isin_code in isins_with_splits:
        isin_mask = result["isin_code"] == isin_code
        isin_prices = result[isin_mask]

        if isin_prices.empty:
            continue

        isin_splits = splits[splits["isin_code"] == isin_code].copy()
        isin_splits = isin_splits.dropna(subset=["ex_date"])
        isin_splits = isin_splits.sort_values("ex_date", ascending=True)

        if isin_splits.empty:
            continue

        split_dates = pd.to_datetime(isin_splits["ex_date"].values, errors="coerce").astype("datetime64[ns]")
        split_factors = isin_splits["split_factor"].values.astype(float)
        n_splits = len(split_factors)

        # Build suffix-product array: suffix[i] = product of split_factors[i:]
        # suffix[n_splits] = 1.0 (no splits remaining)
        suffix = np.ones(n_splits + 1)
        for i in range(n_splits - 1, -1, -1):
            raw = split_factors[i] * suffix[i + 1]
            suffix[i] = max(_MIN_CUMULATIVE_FACTOR, min(_MAX_CUMULATIVE_FACTOR, raw))

        # For each price date, find first split index where split_date >= price_date
        # np.searchsorted(split_dates, price_date, side='left') gives that index
        price_dates = isin_prices["_date_ts"].values.astype("datetime64[ns]")
        indices = np.searchsorted(split_dates, price_dates, side="left")
        factors = suffix[indices]

        # Four vectorized column writes per ISIN (not per row)
        result.loc[isin_mask, "split_adj_open"] = isin_prices["open"].values.astype(float) * factors
        result.loc[isin_mask, "split_adj_high"] = isin_prices["high"].values.astype(float) * factors
        result.loc[isin_mask, "split_adj_low"] = isin_prices["low"].values.astype(float) * factors
        result.loc[isin_mask, "split_adj_close"] = isin_prices["close"].values.astype(float) * factors

    result = result.drop(columns=["_date_ts"])
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

    # Pre-normalize dates once for all ISINs
    result["_date_ts"] = pd.to_datetime(result["date"], errors="coerce")
    dividend_actions["_event_date_ts"] = pd.to_datetime(
        dividend_actions["event_date"], errors="coerce"
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

        isin_prices = isin_prices.sort_values("_date_ts")

        isin_dividends = dividend_actions[
            dividend_actions["isin_code"] == isin_code
        ].copy()
        isin_dividends = isin_dividends.dropna(subset=["_event_date_ts"])

        if isin_dividends.empty:
            continue

        # Sorted ascending price arrays for searchsorted — force datetime64[ns]
        price_dates = isin_prices["_date_ts"].values.astype("datetime64[ns]")
        price_closes = isin_prices["close"].values.astype(float)

        div_factors = []
        for _, div_row in isin_dividends.iterrows():
            ex_date = div_row["_event_date_ts"]
            div_amount = div_row["value"]

            if div_amount is None or div_amount <= 0:
                continue

            # Find last price before ex_date via searchsorted (avoids boolean mask)
            ex_date_np = np.datetime64(pd.Timestamp(ex_date), "ns")
            pos = int(np.searchsorted(price_dates, ex_date_np, side="left")) - 1
            if pos < 0:
                continue

            # IMPORTANT: We must use the RAW close price to calculate the yield factor,
            # because B3 dividend amounts are raw historical amounts (not split-adjusted).
            prev_close = price_closes[pos]

            if prev_close <= 0:
                continue

            factor = 1.0 - (div_amount / prev_close)
            factor = max(0.0, min(1.0, factor))

            div_factors.append({"ex_date": ex_date, "factor": factor})

        if not div_factors:
            continue

        div_factors_df = pd.DataFrame(div_factors)
        div_factors_df = div_factors_df.sort_values("ex_date", ascending=True)

        last_cumulative = 1.0
        adj_close_values = isin_prices["split_adj_close"].values.copy()
        dates = isin_prices["_date_ts"].values.astype("datetime64[ns]")

        # Use numpy arrays for the backward sweep (avoids iloc per step)
        div_dates_arr = pd.to_datetime(div_factors_df["ex_date"]).values.astype("datetime64[ns]")
        div_individual_factors = div_factors_df["factor"].values

        div_idx = len(div_factors_df) - 1
        for j in range(len(dates) - 1, -1, -1):
            row_date = dates[j]

            while div_idx >= 0 and div_dates_arr[div_idx] > row_date:
                last_cumulative *= div_individual_factors[div_idx]
                div_idx -= 1

            adj_close_values[j] = adj_close_values[j] * last_cumulative

        result.loc[isin_prices.index, "adj_close"] = adj_close_values

    result = result.drop(columns=["_date_ts"])
    logger.info("Dividend/JCP adjustments applied")
    return result


def detect_splits_from_prices(
    prices: pd.DataFrame,
    existing_stock_actions: pd.DataFrame,
    detect_nonstandard: bool = False,
) -> pd.DataFrame:
    """
    Detect missing stock splits from price data by looking for overnight price jumps.

    This is a fallback heuristic for splits not returned by B3's API. It compares
    consecutive-day close prices and flags jumps matching common split ratios.

    Algorithm:
    1. For each ISIN, compute consecutive-day price ratios from the close column.
    2. Identify jumps where ratio > SPLIT_DETECTION_THRESHOLD_HIGH (e.g., 1.8) or
       ratio < SPLIT_DETECTION_THRESHOLD_LOW (e.g., 0.55) within a 5-trading-day window.
    3. Skip if a stock_actions entry already exists for that ISIN and date range.
    4. Skip if a quotation_factor transition on that date explains the jump.
    5. Attempt to match remaining jumps against common split ratios with 8% tolerance.
    6. Return matched splits as stock_actions with source='DETECTED'.

    Args:
        prices: DataFrame with raw price data (must include isin_code, date, close,
                and optionally quotation_factor columns).
        existing_stock_actions: DataFrame of already-known stock actions (to avoid
                                duplicates).

    Returns:
        DataFrame of newly detected stock actions with source='DETECTED'.
        Columns: [isin_code, ex_date, action_type, factor, source]
    """
    # Common split ratios to try: (N, direction) where N is the integer multiple
    _common_ratios = [2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 50, 100]
    _tolerance = 0.08  # 8% tolerance -- wider than backtest's 4% for pipeline auditing

    detected = []

    if prices.empty:
        return pd.DataFrame(
            columns=["isin_code", "ex_date", "action_type", "factor", "source"]
        )

    # Normalize date column
    prices = prices.copy()
    prices["date"] = prices["date"].apply(_normalize_date)

    # Build a lookup set of (isin_code, ex_date) for existing actions
    existing_keys: set = set()
    if not existing_stock_actions.empty:
        for _, row in existing_stock_actions.iterrows():
            ex = _normalize_date(row["ex_date"])
            if ex is not None:
                existing_keys.add((row["isin_code"], ex))

    for isin_code, group in prices.groupby("isin_code"):
        group = group.sort_values("date").reset_index(drop=True)

        if len(group) < 2:
            continue

        closes = group["close"].values
        dates = group["date"].values
        factors = group["quotation_factor"].values if "quotation_factor" in group.columns else None

        for i in range(1, len(closes)):
            prev_close = closes[i - 1]
            curr_close = closes[i]

            if prev_close <= 0 or curr_close <= 0:
                continue

            ratio = curr_close / prev_close

            # Only flag large jumps
            if config.SPLIT_DETECTION_THRESHOLD_LOW <= ratio <= config.SPLIT_DETECTION_THRESHOLD_HIGH:
                continue

            jump_date = dates[i]

            # Skip if existing stock_action already covers this date (within 5-day window)
            already_recorded = False
            for lookback_i in range(max(0, i - 5), i + 1):
                candidate_date = dates[lookback_i]
                if (isin_code, candidate_date) in existing_keys:
                    already_recorded = True
                    break
            if already_recorded:
                continue

            # Skip if this is a quotation_factor transition that explains the jump.
            # After normalization, a fatcot transition produces a continuous series --
            # prices are already per-share on both sides. However, if the fatcot
            # column is not yet normalized (pre-normalization pipeline), we check
            # whether the fatcot changed on this date.
            if factors is not None:
                prev_factor = factors[i - 1]
                curr_factor = factors[i]
                if prev_factor != curr_factor and prev_factor > 0 and curr_factor > 0:
                    # The "jump" is accounted for by the quotation factor change.
                    # Since the parser already normalizes prices by dividing by
                    # quotation_factor, if prices are still jumping here it means
                    # something else is going on. We skip if the ratio of factors
                    # explains the price ratio within tolerance.
                    factor_ratio = curr_factor / prev_factor
                    if abs(ratio - (1.0 / factor_ratio)) / max(abs(1.0 / factor_ratio), 0.001) < _tolerance:
                        continue
                    # Also skip if prices are already per-share and the fatcot
                    # transition just changes the metadata (no jump expected)
                    if abs(ratio - 1.0) < _tolerance:
                        continue

            # Try to match against common forward split ratios (close drops by 1/N)
            matched = False
            for n in _common_ratios:
                # Forward split: close drops to 1/N of previous
                target_forward = 1.0 / n
                if abs(ratio - target_forward) / target_forward < _tolerance:
                    detected.append({
                        "isin_code": isin_code,
                        "ex_date": jump_date,
                        "action_type": config.EVENT_TYPE_STOCK_SPLIT,
                        "factor": float(n),
                        "source": "DETECTED",
                    })
                    matched = True
                    logger.info(
                        f"Detected forward split {n}:1 for {isin_code} on {jump_date} "
                        f"(ratio={ratio:.4f})"
                    )
                    break

                # Reverse split: close rises to N times previous
                target_reverse = float(n)
                if abs(ratio - target_reverse) / target_reverse < _tolerance:
                    detected.append({
                        "isin_code": isin_code,
                        "ex_date": jump_date,
                        "action_type": config.EVENT_TYPE_REVERSE_SPLIT,
                        "factor": 1.0 / float(n),
                        "source": "DETECTED",
                    })
                    matched = True
                    logger.info(
                        f"Detected reverse split 1:{n} for {isin_code} on {jump_date} "
                        f"(ratio={ratio:.4f})"
                    )
                    break

            if not matched and detect_nonstandard:
                # Use raw ratio as factor for non-standard splits.
                # Convention matches standard detection:
                #   STOCK_SPLIT factor = split multiple (e.g., 2.37 for a 2.37:1 split)
                #   REVERSE_SPLIT factor = 1/multiple (e.g., 0.42 for a 1:2.37 reverse)
                if ratio < 1.0:
                    # Price dropped -> forward split
                    split_multiple = 1.0 / ratio
                    action_type = config.EVENT_TYPE_STOCK_SPLIT
                    factor = split_multiple
                    label = f"forward {split_multiple:.2f}:1"
                else:
                    # Price rose -> reverse split
                    action_type = config.EVENT_TYPE_REVERSE_SPLIT
                    factor = 1.0 / ratio
                    label = f"reverse 1:{ratio:.2f}"

                detected.append({
                    "isin_code": isin_code,
                    "ex_date": jump_date,
                    "action_type": action_type,
                    "factor": float(factor),
                    "source": "DETECTED_NONSTANDARD",
                })
                matched = True
                logger.info(
                    f"Detected nonstandard split {label} for {isin_code} on {jump_date} "
                    f"(ratio={ratio:.4f})"
                )

            if not matched and (ratio > 3.0 or ratio < 0.33):
                logger.warning(
                    f"Large unmatched price jump for {isin_code} on {jump_date}: "
                    f"ratio={ratio:.4f} (prev={prev_close:.4f}, curr={curr_close:.4f}). "
                    f"Manual review recommended."
                )

    if detected:
        df = pd.DataFrame(detected)
        df = df.drop_duplicates(subset=["isin_code", "ex_date", "action_type"])
        logger.info(
            f"Split detection: found {len(df)} new potential splits from price data"
        )
        return df

    return pd.DataFrame(
        columns=["isin_code", "ex_date", "action_type", "factor", "source"]
    )


def filter_fatcot_redundant_splits(
    stock_actions: pd.DataFrame,
    prices: pd.DataFrame,
    tolerance: float = 0.15,
    date_window: int = 5,
) -> pd.DataFrame:
    """
    Remove B3 API stock_actions that duplicate a quotation_factor transition.

    When a stock transitions from lot-based pricing (fatcot=1000) to per-share
    pricing (fatcot=1), B3's API sometimes also reports a DESDOBRAMENTO (split)
    for the same date and ratio. Since the parser already normalizes prices by
    dividing by quotation_factor, applying the API split would double-adjust.

    This function identifies and removes such redundant splits by checking whether
    a fatcot transition on or near the ex_date explains the reported split factor.

    Args:
        stock_actions: DataFrame of stock actions (must include source, isin_code,
                       ex_date, factor, action_type columns).
        prices: DataFrame of price data (must include isin_code, date,
                quotation_factor columns).
        tolerance: Relative tolerance for matching the fatcot ratio to the split
                   factor (default 15%).
        date_window: Number of trading days around the ex_date to search for a
                     fatcot transition (default 5).

    Returns:
        Filtered stock_actions DataFrame with redundant entries removed.
    """
    if stock_actions.empty or prices.empty:
        return stock_actions

    if "quotation_factor" not in prices.columns:
        return stock_actions

    # Only check B3 API splits (not DETECTED ones)
    b3_mask = stock_actions["source"] == "B3"
    if not b3_mask.any():
        return stock_actions

    prices_copy = prices.copy()
    prices_copy["date"] = prices_copy["date"].apply(_normalize_date)

    redundant_indices = []

    for idx, row in stock_actions[b3_mask].iterrows():
        isin = row["isin_code"]
        ex_date = _normalize_date(row["ex_date"])
        b3_factor = row["factor"]
        action_type = row["action_type"]

        if ex_date is None:
            continue

        # Get prices for this ISIN around the ex_date
        isin_prices = prices_copy[prices_copy["isin_code"] == isin].sort_values("date")
        if len(isin_prices) < 2:
            continue

        # Look for a fatcot transition within the date window
        dates = isin_prices["date"].values
        factors = isin_prices["quotation_factor"].values

        for i in range(1, len(dates)):
            d = dates[i]
            if d is None:
                continue

            # Check if this date is within the window of the ex_date
            day_diff = abs((d - ex_date).days) if hasattr(d, 'days') else abs((pd.Timestamp(d) - pd.Timestamp(ex_date)).days)
            if day_diff > date_window:
                continue

            prev_fatcot = factors[i - 1]
            curr_fatcot = factors[i]

            if prev_fatcot == curr_fatcot or prev_fatcot <= 0 or curr_fatcot <= 0:
                continue

            # The fatcot ratio: e.g., 1000->1 gives fatcot_ratio = 1000
            fatcot_ratio = prev_fatcot / curr_fatcot

            # For a STOCK_SPLIT with factor N, the expected fatcot ratio is N
            # For a REVERSE_SPLIT with factor < 1, the expected fatcot ratio is 1/factor
            if action_type == config.EVENT_TYPE_STOCK_SPLIT:
                expected_ratio = b3_factor
            elif action_type == config.EVENT_TYPE_REVERSE_SPLIT:
                expected_ratio = 1.0 / b3_factor if b3_factor > 0 else 0
            else:
                continue

            if expected_ratio <= 0:
                continue

            rel_diff = abs(fatcot_ratio - expected_ratio) / expected_ratio
            if rel_diff < tolerance:
                redundant_indices.append(idx)
                logger.info(
                    f"Filtering FATCOT_REDUNDANT split for {isin} on {ex_date}: "
                    f"B3 factor={b3_factor}, fatcot transition={prev_fatcot}->{curr_fatcot} "
                    f"(ratio={fatcot_ratio:.1f}, expected={expected_ratio:.1f})"
                )
                break  # Found the matching transition, no need to check more dates

    if redundant_indices:
        logger.info(
            f"Filtered {len(redundant_indices)} FATCOT_REDUNDANT splits "
            f"(would have caused double-adjustment)"
        )
        return stock_actions.drop(index=redundant_indices).reset_index(drop=True)

    return stock_actions


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

    # Filter out B3 API splits that are redundant with fatcot transitions
    stock_actions = filter_fatcot_redundant_splits(stock_actions, prices)

    splits = convert_stock_actions_to_splits(stock_actions)
    logger.info(f"Converted {len(stock_actions)} stock actions to {len(splits)} splits")

    prices = compute_split_adjustment_factors(prices, splits)

    prices = compute_dividend_adjustment_factors(prices, corporate_actions)

    # Final sanity check: cap adj_close / close ratio to prevent extreme values
    # from surviving through to backtests
    close_vals = prices["close"].replace(0, np.nan)
    ratio = prices["adj_close"] / close_vals
    extreme_mask = (ratio.abs() > _MAX_CUMULATIVE_FACTOR) | ratio.isna()
    n_extreme = extreme_mask.sum()
    if n_extreme > 0:
        logger.warning(
            f"Clamping {n_extreme} rows with extreme adj_close/close ratio "
            f"(>{_MAX_CUMULATIVE_FACTOR}x)"
        )
        # Fall back to split_adj_close for extreme rows
        prices.loc[extreme_mask, "adj_close"] = prices.loc[
            extreme_mask, "split_adj_close"
        ]

    logger.info("Adjustment computation complete")

    return prices, splits
