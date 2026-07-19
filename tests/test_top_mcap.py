"""Tests for the Top-N-by-market-cap strategy (backtests/strategies/top_mcap.py).

Synthetic in-memory frames, no database. Covers the require_earnings=False
flag on select_constituents and the top-N selection + weighting logic.
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtests.strategies.sp500_b3 import compute_weights, select_constituents
from tests.test_sp500_b3 import make_actions, make_fund

T = pd.Timestamp("2020-09-30")


def _universe():
    # BIGG: R$30bn cap but NEGATIVE net income; MIDD: R$20bn; SMLL: R$10bn.
    fund = make_fund([
        ("BIGG", "2020-06-30", "2020-08-01", 1, -50_000.0, 1_000_000.0, 3e9),
        ("MIDD", "2020-06-30", "2020-08-01", 1, 100_000.0, 1_000_000.0, 2e9),
        ("SMLL", "2020-06-30", "2020-08-01", 1, 100_000.0, 1_000_000.0, 1e9),
    ])
    close = pd.Series({"BIGG3": 10.0, "MIDD3": 10.0, "SMLL3": 10.0})
    vol = pd.Series({"BIGG3": 5e6, "MIDD3": 5e6, "SMLL3": 5e6})
    return fund, close, vol


def test_require_earnings_false_admits_negative_ni():
    fund, close, vol = _universe()
    sel = select_constituents(
        T, fund, close, vol, make_actions(),
        min_market_cap=0.0, min_adtv=0.0, require_earnings=False,
    )
    assert set(sel["root"]) == {"BIGG", "MIDD", "SMLL"}
    # default (require_earnings=True) still excludes the loss-maker
    sel_strict = select_constituents(
        T, fund, close, vol, make_actions(), min_market_cap=0.0, min_adtv=0.0,
    )
    assert set(sel_strict["root"]) == {"MIDD", "SMLL"}


def test_corrupted_share_count_excluded():
    # Real-world failure: TELB/TOYB/COCE filings carry ~1e11-1e12 shares
    # (fake trillion-BRL companies) while the stock trades ~R$1M/day. The
    # turnover sanity guard must drop them even though they rank #1 by cap.
    fund = make_fund([
        ("FAKE", "2020-06-30", "2020-08-01", 1, 100_000.0, 1_000_000.0, 1e12),
        ("REAL", "2020-06-30", "2020-08-01", 1, 100_000.0, 1_000_000.0, 2e9),
    ])
    close = pd.Series({"FAKE4": 0.6, "REAL3": 30.0})   # FAKE "cap" = R$600bn
    vol = pd.Series({"FAKE4": 1e6, "REAL3": 3e8})
    sel = select_constituents(
        T, fund, close, vol, make_actions(),
        min_market_cap=0.0, min_adtv=0.0, require_earnings=False,
    )
    assert set(sel["root"]) == {"REAL"}


def test_per_class_market_cap_avoids_unit_inflation():
    # BPAC-style: most liquid listing is the unit (1 ON + 2 PN). Old formula
    # (total shares x unit price) tripled the cap; per-class pricing must use
    # BTGU3 (ON) and BTGU5 (PN) closes instead of the BTGU11 unit close.
    fund = make_fund([("BTGU", "2020-06-30", "2020-08-01", 1, 100_000.0, 1_000_000.0, 3e9)])
    fund["shares_on"] = 1e9
    fund["shares_pn"] = 2e9
    close = pd.Series({"BTGU11": 30.0, "BTGU3": 10.0, "BTGU5": 10.0})
    vol = pd.Series({"BTGU11": 5e8, "BTGU3": 1e6, "BTGU5": 1e6})
    sel = select_constituents(
        T, fund, close, vol, make_actions(),
        min_market_cap=0.0, min_adtv=0.0, require_earnings=False,
    )
    assert len(sel) == 1
    assert sel["ticker"].iloc[0] == "BTGU11"        # unit still the traded instrument
    assert sel["market_cap"].iloc[0] == pytest.approx(3e10)   # not 9e10


def test_top_n_and_weightings():
    fund, close, vol = _universe()
    sel = select_constituents(
        T, fund, close, vol, make_actions(),
        min_market_cap=0.0, min_adtv=0.0, require_earnings=False,
    )
    top2 = sel.nlargest(2, "market_cap")
    assert set(top2["root"]) == {"BIGG", "MIDD"}

    caps = top2.set_index("ticker")["market_cap"]
    w_cap = compute_weights(caps, "market_cap")
    assert abs(w_cap.sum() - 1.0) < 1e-12
    assert abs(w_cap["BIGG3"] - 3.0 / 5.0) < 1e-12

    w_eq = compute_weights(caps, "equal")
    assert (w_eq == 0.5).all()
