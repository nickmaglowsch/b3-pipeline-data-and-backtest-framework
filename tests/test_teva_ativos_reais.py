"""Tests for the Teva Dividendos Ativos Reais replica (pure helpers only).

Synthetic in-memory series, no database. Covers the Dividend Score windowing +
DY cap, the 10% cap redistribution (including the infeasible all-capped case
that must stay fully invested), and the 50/50 mcap+score blend.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtests.core.data import sector_membership, sector_membership_asof
from backtests.strategies.teva_ativos_reais import (
    dividend_score,
    cap_weights,
    blend_weights,
)

IDX = pd.date_range("2018-01-31", periods=48, freq="ME")


def test_dividend_score_constant_dy():
    dy = pd.Series(0.06, index=IDX)
    assert abs(dividend_score(dy, IDX[-1]) - 0.06) < 1e-9


def test_dividend_score_caps_outlier_window():
    dy = pd.Series(0.05, index=IDX).copy()
    dy.iloc[-1] = 0.50  # one wild window
    # cap (mean+1std over 5y) pulls the 0.50 window down below the raw mean
    assert dividend_score(dy, IDX[-1]) < (0.05 + 0.50 + 0.05) / 3


def test_dividend_score_needs_36m_history():
    assert np.isnan(dividend_score(pd.Series(0.06, index=IDX[:5]), IDX[4]))


def test_cap_feasible_redistributes():
    w = cap_weights(pd.Series({"A": 8.0, "B": 1.0, "C": 1.0}), cap=0.50)
    assert abs(w["A"] - 0.50) < 1e-9
    assert abs(w.sum() - 1.0) < 1e-9


def test_cap_infeasible_stays_fully_invested():
    # 5 names, 10% cap -> can't sum to 1; must fall back to fully invested,
    # not strand half the book in (untracked) cash.
    w = cap_weights(pd.Series({c: 1.0 for c in "ABCDE"}), cap=0.10)
    assert abs(w.sum() - 1.0) < 1e-9
    assert np.allclose(w.values, 0.2)


def test_sector_membership_substring_and_holdings():
    # substring match keeps both the direct sector and the "Emp. Adm. Part."
    # holding variant, and excludes unrelated sectors (no survivorship pruning —
    # a delisted utility still appears).
    pit = pd.DataFrame({
        "ticker": ["EGIE3", "SBSP3", "PETR4", "ELPL3"],
        "ref_date": pd.to_datetime(["2015-12-31"] * 4),
        "sector": [
            "Energia Elétrica",
            "Saneamento, Serv. Água e Gás",
            "Petróleo e Gás",
            "Emp. Adm. Part. - Energia Elétrica",  # delisted holding
        ],
    })
    m = sector_membership(pit, ["Energia Elétrica", "Saneamento"])
    assert set(m["ticker"]) == {"EGIE3", "SBSP3", "ELPL3"}


def test_sector_membership_asof_revokes_after_reclassification():
    # AAAA3 is a utility in 2015 but reclassifies to oil in 2018: eligible at
    # 2016 rebalances, NOT eligible after the 2018 filing ("ever classified"
    # would keep it forever). BBBB3 stays eligible; before its first filing
    # nothing is eligible.
    pit = pd.DataFrame({
        "ticker": ["AAAA3", "AAAA3", "BBBB3"],
        "ref_date": pd.to_datetime(["2015-12-31", "2018-12-31", "2015-12-31"]),
        "sector": ["Energia Elétrica", "Petróleo e Gás", "Energia Elétrica"],
    })
    dates = pd.DatetimeIndex(["2015-06-30", "2016-06-30", "2019-06-30"])
    m = sector_membership_asof(pit, ["Energia Elétrica"], dates)
    assert not m.loc["2015-06-30"].any()                    # no filing yet
    assert bool(m.loc["2016-06-30", "AAAA3"])
    assert not bool(m.loc["2019-06-30", "AAAA3"])           # reclassified out
    assert bool(m.loc["2019-06-30", "BBBB3"])


def test_blend_tilts_toward_higher_score():
    w = blend_weights(
        pd.Series({"A": 100.0, "B": 100.0}),  # equal mcap
        pd.Series({"A": 3.0, "B": 1.0}),      # A scores higher
        cap=1.0,
    )
    assert abs(w.sum() - 1.0) < 1e-9
    assert w["A"] > w["B"]
