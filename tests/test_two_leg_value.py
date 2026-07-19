"""Tests for the Two-Leg Value strategy pure functions (no DB)."""
from __future__ import annotations

import pandas as pd

from backtests.strategies.two_leg_value import (
    select_value_portfolio,
    value_rebalance_dates,
)

T = pd.Timestamp("2026-07-15")
FRESH = pd.Timestamp("2026-03-31")   # 106 days -> fresh
STALE = pd.Timestamp("2025-09-30")   # 288 days -> stale


def _company(ticker, mcap, ni, eq, ebitda, net_debt, rev, pe=None):
    """One row; financials in thousands BRL, mcap in BRL. pe unused (derived)."""
    return dict(
        ticker=ticker, market_cap=mcap, net_income_ttm=ni, equity=eq,
        roe=(ni / eq if eq else float("nan")),
        ebitda=ebitda, net_debt=net_debt, revenue=rev, period_end=FRESH,
    )


def _df(rows):
    return pd.DataFrame(rows)


def test_freshness_drops_stale_names():
    rows = [_company("GOOD3", 1e9, 1e5, 5e5, 2e5, 1e5, 1e6)]
    rows[0]["period_end"] = STALE
    out = select_value_portfolio(_df(rows), T, target_n=5)
    assert out.empty, "a balance older than 180d must be dropped pre-rank"


def test_health_drops_loss_makers_and_negative_margin():
    rows = [
        _company("LOSS3", 1e9, -1e5, 5e5, 2e5, 0, 1e6),      # negative earnings -> P/L<0
        _company("NEGM3", 1e9, 1e4, 5e5, -3e4, 0, 1e6),      # negative EBITDA margin (non-fin)
        _company("FINE3", 1e9, 2e5, 5e5, 4e5, 0, 1e6),       # clean
    ]
    out = select_value_portfolio(_df(rows), T, target_n=5)
    assert set(out["ticker"]) == {"FINE3"}


def test_bank_and_operating_names_rank_together():
    # BANK4 has no EBITDA (financial leg, cheap on P/L+P/VP); INDU3 is a cheap
    # operating name (high EBITDA/EV). Both should be selected and coexist.
    rows = [
        _company("BANK4", 1e9, 5e5, 1e6, float("nan"), float("nan"), float("nan")),  # PL=2, PVP=1
        _company("INDU3", 1e9, 2e5, 2e6, 3e5, 0, 1e6),   # PL=5, high EBITDA/EV
        _company("EXPN3", 5e9, 2e5, 2e6, 1e5, 0, 1e6),   # expensive: low EBITDA/EV
    ]
    out = select_value_portfolio(_df(rows), T, target_n=3)
    assert "BANK4" in set(out["ticker"]), "cheap bank must surface via P/L+P/VP leg"
    assert "INDU3" in set(out["ticker"])
    assert bool(out.set_index("ticker").loc["BANK4", "is_fin"]), "bank flagged financial"


def test_zero_revenue_holding_routes_to_financial_leg():
    # BBSE-style filer: real EBIT line but revenue = 0 (earnings via equity
    # method). Must rank on P/L+P/VP instead of dying at the revenue>0 gate.
    rows = [
        _company("HOLD3", 1e9, 5e5, 1e6, 8e5, -1e5, 0.0),  # PL=2, PVP=1, rev=0
        _company("INDU3", 1e9, 2e5, 2e6, 3e5, 0, 1e6),
    ]
    out = select_value_portfolio(_df(rows), T, target_n=3)
    assert "HOLD3" in set(out["ticker"]), "zero-revenue holding must not be dropped"
    assert bool(out.set_index("ticker").loc["HOLD3", "is_fin"])


def test_value_trap_filter_and_backfill():
    # Two cheap-but-trap names (low ROE / high P/L) rank top, then clean names.
    # target_n=2 must skip the traps and backfill the two clean names.
    rows = [
        _company("TRAP1", 1e9, 1e4, 5e6, 6e5, 0, 1e6),   # ROE 0.2% -> trap, but very cheap
        _company("TRAP2", 5e10, 3e6, 1e6, 6e5, 0, 1e6),  # P/L=16.7? actually high P/L trap
        _company("OKAY3", 1e9, 2e5, 5e5, 4e5, 0, 1e6),   # ROE 40%, P/L 5 -> clean
        _company("OKAY4", 1e9, 2e5, 6e5, 35e4, 0, 1e6),  # clean
    ]
    # force TRAP2 to be a P/L trap
    rows[1]["market_cap"] = 1e11  # P/L = 1e11/(3e6*1000)=33 -> >20
    out = select_value_portfolio(_df(rows), T, min_roe=0.08, max_pe=20.0,
                                 pool_size=2, target_n=2)
    assert set(out["ticker"]) == {"OKAY3", "OKAY4"}, "traps dropped, clean backfilled"


def test_rebalance_calendar():
    days = pd.bdate_range("2024-01-01", "2024-12-31")
    reb = value_rebalance_dates(days)
    assert list(reb.month) == [4, 6, 9, 12]
    for d in reb:
        assert d.day >= 15 and d.day <= 21, "first trading day on/after the 15th"
