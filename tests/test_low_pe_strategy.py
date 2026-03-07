"""
Tests for the LowPE strategy plugin (Task 07 TDD).

All tests use synthetic shared_data dicts — no DB, no network calls.
"""
from __future__ import annotations

import pandas as pd
import numpy as np
import pytest


def _make_shared_data(
    tickers: list[str],
    pe_values: dict[str, float],
    n_periods: int = 3,
    adtv_value: float = 5_000_000.0,
    price_value: float = 10.0,
    use_dyn_key: bool = True,
) -> dict:
    """Build a minimal shared_data dict for testing LowPEStrategy."""
    dates = pd.date_range("2023-01-31", periods=n_periods, freq="ME")

    # Build returns (all zeros — just for index/column shape)
    ret = pd.DataFrame(0.0, index=dates, columns=tickers)

    # ADTV above threshold for all tickers
    adtv = pd.DataFrame(adtv_value, index=dates, columns=tickers)

    # Raw close above threshold for all tickers
    raw_close = pd.DataFrame(price_value, index=dates, columns=tickers)

    # PE ratio DataFrame
    pe_df = pd.DataFrame(index=dates, columns=tickers, dtype=float)
    for ticker, pe in pe_values.items():
        pe_df[ticker] = pe

    shared: dict = {
        "ret": ret,
        "adtv": adtv,
        "raw_close": raw_close,
    }
    if use_dyn_key:
        shared["f_pe_ratio_dyn"] = pe_df
    else:
        shared["f_pe_ratio"] = pe_df

    return shared


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Strategy is registered and has correct attributes
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_strategy_registered():
    """LowPEStrategy should be importable with correct name and needs_fundamentals."""
    from backtests.strategies.low_pe import LowPEStrategy

    assert LowPEStrategy().name == "LowPE"
    assert LowPEStrategy.needs_fundamentals is True


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Selects lowest P/E stocks
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_selects_lowest_pe():
    """Strategy should select the N tickers with the lowest P/E ratio."""
    from backtests.strategies.low_pe import LowPEStrategy

    shared = _make_shared_data(
        tickers=["TICK1", "TICK2", "TICK3"],
        pe_values={"TICK1": 5.0, "TICK2": 15.0, "TICK3": 25.0},
    )
    _, tw = LowPEStrategy().generate_signals(
        shared, {"n_stocks": 2, "min_pe": 1.0, "max_pe": 30.0, "min_stocks": 2}
    )

    # At period index 1 (using prev_dt = period 0)
    row = tw.iloc[1]
    assert row["TICK1"] == pytest.approx(0.5), f"TICK1 should have weight 0.5, got {row['TICK1']}"
    assert row["TICK2"] == pytest.approx(0.5), f"TICK2 should have weight 0.5, got {row['TICK2']}"
    assert row["TICK3"] == pytest.approx(0.0), f"TICK3 should have weight 0.0, got {row['TICK3']}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Excludes P/E above max
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_excludes_pe_above_max():
    """Tickers with P/E > max_pe should be excluded."""
    from backtests.strategies.low_pe import LowPEStrategy

    shared = _make_shared_data(
        tickers=["TICK1", "TICK2", "TICK3"],
        pe_values={"TICK1": 5.0, "TICK2": 35.0, "TICK3": 20.0},
    )
    _, tw = LowPEStrategy().generate_signals(
        shared, {"n_stocks": 2, "min_pe": 1.0, "max_pe": 30.0, "min_stocks": 2}
    )

    row = tw.iloc[1]
    # TICK2 excluded (35 > max_pe=30); TICK1 and TICK3 selected
    assert row["TICK2"] == pytest.approx(0.0), f"TICK2 (pe=35 > max) should be excluded"
    assert row["TICK1"] == pytest.approx(0.5)
    assert row["TICK3"] == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Excludes P/E below min
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_excludes_pe_below_min():
    """Tickers with P/E < min_pe should be excluded."""
    from backtests.strategies.low_pe import LowPEStrategy

    shared = _make_shared_data(
        tickers=["TICK1", "TICK2", "TICK3"],
        pe_values={"TICK1": 0.5, "TICK2": 8.0, "TICK3": 10.0},
    )
    _, tw = LowPEStrategy().generate_signals(
        shared, {"n_stocks": 2, "min_pe": 1.0, "max_pe": 30.0, "min_stocks": 2}
    )

    row = tw.iloc[1]
    # TICK1 excluded (0.5 < min_pe=1.0)
    assert row["TICK1"] == pytest.approx(0.0), f"TICK1 (pe=0.5 < min) should be excluded"
    assert row["TICK2"] == pytest.approx(0.5)
    assert row["TICK3"] == pytest.approx(0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Falls back to stored P/E when dyn key is absent
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_falls_back_to_stored_pe():
    """Strategy should use f_pe_ratio when f_pe_ratio_dyn is absent."""
    from backtests.strategies.low_pe import LowPEStrategy

    shared = _make_shared_data(
        tickers=["TICK1", "TICK2", "TICK3"],
        pe_values={"TICK1": 5.0, "TICK2": 15.0, "TICK3": 25.0},
        use_dyn_key=False,  # only f_pe_ratio, no f_pe_ratio_dyn
    )
    # Should not raise; should use f_pe_ratio
    ret, tw = LowPEStrategy().generate_signals(
        shared, {"n_stocks": 2, "min_pe": 1.0, "max_pe": 30.0, "min_stocks": 2}
    )
    row = tw.iloc[1]
    assert row["TICK1"] == pytest.approx(0.5), "Fallback to f_pe_ratio should select TICK1"


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Too few stocks gives zero weight (no crash)
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_too_few_stocks_gives_zero_weight():
    """When fewer than min_stocks pass filters, all weights for that period are zero."""
    from backtests.strategies.low_pe import LowPEStrategy

    # Only 2 tickers but min_stocks=3
    shared = _make_shared_data(
        tickers=["TICK1", "TICK2"],
        pe_values={"TICK1": 5.0, "TICK2": 10.0},
    )
    _, tw = LowPEStrategy().generate_signals(
        shared, {"n_stocks": 2, "min_pe": 1.0, "max_pe": 30.0, "min_stocks": 3}
    )

    # The period should be all zeros
    row = tw.iloc[1]
    assert row.sum() == pytest.approx(0.0), f"Expected all-zero weights, got {row.to_dict()}"


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Equal weights
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_equal_weights():
    """4 selected tickers should each have weight = 1/4 = 0.25."""
    from backtests.strategies.low_pe import LowPEStrategy

    shared = _make_shared_data(
        tickers=["TICK1", "TICK2", "TICK3", "TICK4"],
        pe_values={"TICK1": 5.0, "TICK2": 8.0, "TICK3": 12.0, "TICK4": 15.0},
    )
    _, tw = LowPEStrategy().generate_signals(
        shared, {"n_stocks": 4, "min_pe": 1.0, "max_pe": 30.0, "min_stocks": 2}
    )

    row = tw.iloc[1]
    for ticker in ["TICK1", "TICK2", "TICK3", "TICK4"]:
        assert row[ticker] == pytest.approx(0.25), (
            f"{ticker} should have weight 0.25, got {row[ticker]}"
        )
