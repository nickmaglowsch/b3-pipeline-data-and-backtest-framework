"""
Tests for the LowPE strategy — now ``backtests/strategies/specs/low_pe.yaml``
driven by the ``RankAndHold`` config engine (the hand-written class was deleted
after exact target-weight + after-tax-curve parity on the real DB).

All tests use synthetic shared_data dicts — no DB, no network calls.

Note on tickers: every fixture uses **distinct 4-char company roots**. Real B3
share classes of one company share a root (PETR3/PETR4) and both the engine's
`dedup_roots` and the global `dedup_target_weights` collapse them onto the most
liquid class — so a fixture like TICK1..TICK4 is one company, not four.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from backtests.core.config_strategy import RankAndHold

SPEC_PATH = (Path(__file__).resolve().parent.parent
             / "backtests" / "strategies" / "specs" / "low_pe.yaml")


def _spec() -> dict:
    return yaml.safe_load(SPEC_PATH.read_text())


def _strategy() -> RankAndHold:
    return RankAndHold(_spec())


def _make_shared_data(
    pe_values: dict[str, float],
    n_periods: int = 3,
    adtv: dict[str, float] | None = None,
    price_value: float = 10.0,
    empty_fundamentals: bool = False,
) -> dict:
    """Minimal shared_data for the LowPE spec: a fully-liquid universe plus a
    P/E frame. Keys mirror build_shared_data(include_fundamentals=True)."""
    tickers = list(pe_values)
    dates = pd.date_range("2023-01-31", periods=n_periods, freq="ME")

    adtv_df = pd.DataFrame(5_000_000.0, index=dates, columns=tickers)
    for t, v in (adtv or {}).items():
        adtv_df[t] = v

    pe_df = (pd.DataFrame(index=dates, dtype=float) if empty_fundamentals
             else pd.DataFrame([pe_values] * n_periods, index=dates, columns=tickers))

    return {
        "ret": pd.DataFrame(0.0, index=dates, columns=tickers),
        "adtv": adtv_df,
        "raw_close": pd.DataFrame(price_value, index=dates, columns=tickers),
        "f_pe_ratio_dyn": pe_df,
    }


def _held(row: pd.Series) -> list[str]:
    return sorted(row[row > 0].index)


# ──────────────────────────────────────────────────────────────────────────────
# Test 1: Strategy is registered and declares its fundamentals dependency
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_strategy_registered():
    from backtests.core.strategy_registry import get_registry

    strat = get_registry().get("LowPE")
    assert strat.name == "LowPE"
    assert strat.needs_fundamentals is True


def test_low_pe_band_is_the_inverted_pe_range():
    """The [min_pe, max_pe] band is expressed as an earnings-yield range
    (higher = better, so selection is always `nlargest`)."""
    lo, hi = _spec()["signal"]["range"]
    assert lo == pytest.approx(1 / 30.0)     # max_pe 30
    assert hi == pytest.approx(1 / 1.0)      # min_pe 1


# ──────────────────────────────────────────────────────────────────────────────
# Test 2: Selects lowest P/E stocks
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_selects_lowest_pe():
    """Strategy should select the N tickers with the lowest P/E ratio."""
    shared = _make_shared_data(
        {"AAAA3": 5.0, "BBBB3": 15.0, "CCCC3": 25.0, "DDDD3": 28.0})
    _, tw = _strategy().generate_signals(shared, {"top_n": 2})

    row = tw.iloc[1]                     # period 1 trades off period 0's P/E
    assert _held(row) == ["AAAA3", "BBBB3"]
    assert np.allclose(row[_held(row)].to_numpy(), 0.5)


# ──────────────────────────────────────────────────────────────────────────────
# Test 3: Excludes P/E above max
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_excludes_pe_above_max():
    """Tickers with P/E > max_pe (30) should be excluded."""
    shared = _make_shared_data(
        {"AAAA3": 5.0, "BBBB3": 35.0, "CCCC3": 20.0, "DDDD3": 25.0})
    _, tw = _strategy().generate_signals(shared, {"top_n": 2})

    row = tw.iloc[1]
    assert row["BBBB3"] == 0.0           # 35 > max_pe
    assert _held(row) == ["AAAA3", "CCCC3"]


# ──────────────────────────────────────────────────────────────────────────────
# Test 4: Excludes P/E below min
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_excludes_pe_below_min():
    """Tickers with P/E < min_pe (1.0) should be excluded — a 0.5 P/E is a
    one-off gain or a data error, not the cheapest company on the exchange."""
    shared = _make_shared_data(
        {"AAAA3": 0.5, "BBBB3": 8.0, "CCCC3": 10.0, "DDDD3": 12.0})
    _, tw = _strategy().generate_signals(shared, {"top_n": 2})

    row = tw.iloc[1]
    assert row["AAAA3"] == 0.0
    assert _held(row) == ["BBBB3", "CCCC3"]


# ──────────────────────────────────────────────────────────────────────────────
# Test 5: Only f_pe_ratio_dyn is read — no fallback to a stored ratio column
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_no_fallback_to_stored_pe():
    """When the monthly snapshot is missing, build_shared_data hands over an
    EMPTY f_pe_ratio_dyn. The strategy must then hold nothing, never fall back
    to a stored f_pe_ratio column."""
    shared = _make_shared_data(
        {"AAAA3": 5.0, "BBBB3": 15.0, "CCCC3": 25.0}, empty_fundamentals=True)
    shared["f_pe_ratio"] = pd.DataFrame(
        5.0, index=shared["ret"].index, columns=shared["ret"].columns)

    _, tw = _strategy().generate_signals(shared, {})
    assert tw.to_numpy().sum() == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Test 6: Too few candidates gives zero weight (no crash)
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_too_few_stocks_gives_zero_weight():
    """Fewer than min_names (3) candidates -> all weights for that period stay 0."""
    shared = _make_shared_data({"AAAA3": 5.0, "BBBB3": 10.0})
    _, tw = _strategy().generate_signals(shared, {})

    assert tw.iloc[1].sum() == pytest.approx(0.0)


# ──────────────────────────────────────────────────────────────────────────────
# Test 7: Equal weights
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_equal_weights():
    """4 selected tickers should each have weight = 1/4 = 0.25."""
    shared = _make_shared_data(
        {"AAAA3": 5.0, "BBBB3": 8.0, "CCCC3": 12.0, "DDDD3": 15.0})
    _, tw = _strategy().generate_signals(shared, {"top_n": 4})

    row = tw.iloc[1]
    for ticker in ["AAAA3", "BBBB3", "CCCC3", "DDDD3"]:
        assert row[ticker] == pytest.approx(0.25), (
            f"{ticker} should have weight 0.25, got {row[ticker]}"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Test 8: One share class per company
# ──────────────────────────────────────────────────────────────────────────────

def test_low_pe_keeps_one_share_class_per_company():
    """Fundamentals are per-company, so PETR3 and PETR4 carry the same P/E.
    Only the most liquid class may be held — otherwise the book is double
    exposed to one company."""
    shared = _make_shared_data(
        {"PETR3": 4.0, "PETR4": 4.0, "VALE3": 6.0, "ITUB4": 9.0},
        adtv={"PETR4": 9_000_000.0},
    )
    _, tw = _strategy().generate_signals(shared, {"top_n": 2})

    row = tw.iloc[1]
    assert _held(row) == ["PETR4", "VALE3"]
    assert row["PETR3"] == 0.0
