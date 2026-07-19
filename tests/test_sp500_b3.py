"""
Tests for the SP500-style B3 Index selection logic (backtests/strategies/sp500_b3.py).

All tests use small SYNTHETIC in-memory DataFrames — no database required.
Covers:
  (a) PIT join uses only filing_date <= t and respects the 400-day staleness rule
  (b) equity > 0 requirement (negative NI / negative equity must not pass)
  (c) split adjustment of shares between period_end and t
  (d) market-cap weights sum to 1 and match hand-computed values
  (e) a delisted ticker drops out at the next rebalance
  (+) mid-quarter delisting renormalization in the index accumulator
"""
from __future__ import annotations

import pandas as pd
import pytest

from backtests.strategies.sp500_b3 import (
    build_index_series,
    compute_weights,
    pit_snapshot,
    quarter_end_rebalance_dates,
    select_constituents,
    share_multiplier,
)

T = pd.Timestamp("2020-09-30")


def make_fund(rows):
    """rows: list of (ticker, period_end, filing_date, version, ni_ttm, equity, shares)."""
    df = pd.DataFrame(
        rows,
        columns=["ticker", "period_end", "filing_date", "filing_version",
                 "net_income_ttm", "equity", "shares_outstanding"],
    )
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def make_actions(rows=()):
    """rows: list of (ticker, ex_date, action_type, factor)."""
    df = pd.DataFrame(rows, columns=["ticker", "ex_date", "action_type", "factor"])
    df["ex_date"] = pd.to_datetime(df["ex_date"]) if len(df) else df["ex_date"]
    return df


# A company that passes every filter with room to spare:
# shares 1e9 x close 10 = R$10bn cap; med vol R$5M; NI 100k (thousands), equity 1M.
GOOD_ROW = ("GOOD", "2020-06-30", "2020-08-01", 1, 100_000.0, 1_000_000.0, 1e9)


def run_select(fund, close=None, vol=None, actions=None, **kw):
    close = close if close is not None else pd.Series({"GOOD3": 10.0})
    vol = vol if vol is not None else pd.Series({"GOOD3": 5e6})
    return select_constituents(T, fund, close, vol, actions if actions is not None else make_actions(), **kw)


# ── (a) PIT join: filing_date <= t, staleness, max (filing_date, version) ────

class TestPitSnapshot:
    def test_only_filings_on_or_before_t_visible(self):
        fund = make_fund([
            ("GOOD", "2020-06-30", "2020-08-01", 1, 100.0, 1000.0, 1e9),
            ("GOOD", "2020-09-30", "2020-10-15", 1, 999.0, 9999.0, 1e9),  # filed after t
        ])
        snap = pit_snapshot(fund, T)
        assert len(snap) == 1
        assert snap.iloc[0]["net_income_ttm"] == 100.0

    def test_max_filing_date_then_max_version_wins(self):
        fund = make_fund([
            ("GOOD", "2020-03-31", "2020-05-01", 1, 100.0, 1000.0, 1e9),
            ("GOOD", "2020-06-30", "2020-08-01", 1, 200.0, 1000.0, 1e9),
            ("GOOD", "2020-06-30", "2020-08-01", 2, 250.0, 1000.0, 1e9),  # restatement
        ])
        snap = pit_snapshot(fund, T)
        assert len(snap) == 1
        assert snap.iloc[0]["net_income_ttm"] == 250.0

    def test_stale_filings_ignored(self):
        # T - 400d = 2019-08-27. A filing from 2019-08-01 is stale; 2019-09-10 is not.
        fund = make_fund([
            ("OLDC", "2019-06-30", "2019-08-01", 1, 100.0, 1000.0, 1e9),
            ("FRSH", "2019-06-30", "2019-09-10", 1, 100.0, 1000.0, 1e9),
        ])
        snap = pit_snapshot(fund, T)
        assert list(snap["ticker"]) == ["FRSH"]

    def test_stale_company_excluded_from_selection(self):
        fund = make_fund([
            ("GOOD", "2019-06-30", "2019-08-01", 1, 100_000.0, 1_000_000.0, 1e9),
        ])
        sel = run_select(fund)
        assert sel.empty


# ── (b) earnings/equity filters ───────────────────────────────────────────────

class TestEarningsFilters:
    def _sel(self, ni, equity):
        fund = make_fund([("GOOD", "2020-06-30", "2020-08-01", 1, ni, equity, 1e9)])
        return run_select(fund)

    def test_positive_ni_positive_equity_passes(self):
        sel = self._sel(100_000.0, 1_000_000.0)
        assert list(sel["ticker"]) == ["GOOD3"]
        assert sel.iloc[0]["roe"] == pytest.approx(0.1)

    def test_negative_net_income_excluded(self):
        assert self._sel(-100_000.0, 1_000_000.0).empty

    def test_negative_equity_excluded(self):
        assert self._sel(100_000.0, -1_000_000.0).empty

    def test_negative_over_negative_must_not_pass(self):
        # NI < 0 and equity < 0 gives a numerically positive ROE — must be rejected.
        assert self._sel(-100_000.0, -1_000_000.0).empty

    def test_null_ttm_excluded(self):
        # net_income_ttm is NULL when there is insufficient filing history.
        assert self._sel(float("nan"), 1_000_000.0).empty


# ── (c) split adjustment of shares between period_end and t ──────────────────

class TestShareMultiplier:
    def test_split_between_period_end_and_t_counted(self):
        actions = make_actions([("GOOD3", "2020-08-15", "STOCK_SPLIT", 2.0)])
        assert share_multiplier(actions, "GOOD3", pd.Timestamp("2020-06-30"), T) == 2.0

    def test_action_before_period_end_ignored(self):
        actions = make_actions([("GOOD3", "2020-05-15", "STOCK_SPLIT", 2.0)])
        assert share_multiplier(actions, "GOOD3", pd.Timestamp("2020-06-30"), T) == 1.0

    def test_action_after_t_ignored(self):
        actions = make_actions([("GOOD3", "2020-10-15", "STOCK_SPLIT", 2.0)])
        assert share_multiplier(actions, "GOOD3", pd.Timestamp("2020-06-30"), T) == 1.0

    def test_reverse_split_and_bonus(self):
        actions = make_actions([
            ("GOOD3", "2020-07-10", "REVERSE_SPLIT", 0.25),   # 1-for-4: count x0.25
            ("GOOD3", "2020-08-10", "BONUS_SHARES", 25.0),    # 25% bonus: count x1.25
        ])
        mult = share_multiplier(actions, "GOOD3", pd.Timestamp("2020-06-30"), T)
        assert mult == pytest.approx(0.25 * 1.25)

    def test_other_ticker_actions_ignored(self):
        actions = make_actions([("BADX3", "2020-08-15", "STOCK_SPLIT", 10.0)])
        assert share_multiplier(actions, "GOOD3", pd.Timestamp("2020-06-30"), T) == 1.0

    def test_market_cap_uses_split_adjusted_shares(self):
        # 1e9 shares as of period_end, 2:1 split in August, close 10:
        # market cap = 1e9 * 2 * 10 = 2e10.
        fund = make_fund([GOOD_ROW])
        actions = make_actions([("GOOD3", "2020-08-15", "STOCK_SPLIT", 2.0)])
        sel = run_select(fund, actions=actions)
        assert sel.iloc[0]["market_cap"] == pytest.approx(2e10)


# ── (d) weights ───────────────────────────────────────────────────────────────

class TestWeights:
    def test_market_cap_weights_sum_to_one_and_match_hand_computed(self):
        caps = pd.Series({"AAAA3": 6e9, "BBBB4": 3e9, "CCCC3": 1e9})
        w = compute_weights(caps, "market_cap")
        assert w.sum() == pytest.approx(1.0)
        assert w["AAAA3"] == pytest.approx(0.6)
        assert w["BBBB4"] == pytest.approx(0.3)
        assert w["CCCC3"] == pytest.approx(0.1)

    def test_equal_weights(self):
        caps = pd.Series({"AAAA3": 6e9, "BBBB4": 3e9, "CCCC3": 1e9})
        w = compute_weights(caps, "equal")
        assert w.sum() == pytest.approx(1.0)
        assert w.tolist() == pytest.approx([1 / 3] * 3)

    def test_end_to_end_selection_weights(self):
        fund = make_fund([
            ("AAAA", "2020-06-30", "2020-08-01", 1, 100_000.0, 1_000_000.0, 8e8),
            ("BBBB", "2020-06-30", "2020-08-01", 1, 100_000.0, 1_000_000.0, 2e8),
        ])
        close = pd.Series({"AAAA3": 10.0, "BBBB4": 10.0})
        vol = pd.Series({"AAAA3": 5e6, "BBBB4": 5e6})
        sel = run_select(fund, close, vol)   # caps: 8e9 and 2e9... 2e9 not > 2e9
        # BBBB cap == exactly 2e9 -> strictly-greater filter drops it
        assert list(sel["ticker"]) == ["AAAA3"]

        sel = run_select(fund, close, vol, min_market_cap=1e9)
        w = compute_weights(sel.set_index("ticker")["market_cap"], "market_cap")
        assert w["AAAA3"] == pytest.approx(0.8)
        assert w["BBBB4"] == pytest.approx(0.2)

    def test_most_liquid_share_class_selected_per_root(self):
        fund = make_fund([GOOD_ROW])
        close = pd.Series({"GOOD3": 10.0, "GOOD4": 12.0})
        vol = pd.Series({"GOOD3": 3e6, "GOOD4": 9e6})
        sel = run_select(fund, close, vol)
        assert list(sel["ticker"]) == ["GOOD4"]  # higher 63d median volume
        assert sel.iloc[0]["market_cap"] == pytest.approx(1e9 * 12.0)


# ── (e) delisting ─────────────────────────────────────────────────────────────

class TestDelisting:
    def test_delisted_ticker_drops_out_at_next_rebalance(self):
        fund = make_fund([
            GOOD_ROW,
            ("DEAD", "2020-06-30", "2020-08-01", 1, 100_000.0, 1_000_000.0, 1e9),
        ])
        close = pd.Series({"GOOD3": 10.0, "DEAD3": 10.0})
        vol_t1 = pd.Series({"GOOD3": 5e6, "DEAD3": 5e6})
        vol_t2 = pd.Series({"GOOD3": 5e6, "DEAD3": 0.0})  # stopped trading

        sel_t1 = select_constituents(T, fund, close, vol_t1, make_actions())
        assert set(sel_t1["ticker"]) == {"GOOD3", "DEAD3"}

        t2 = pd.Timestamp("2020-12-30")
        sel_t2 = select_constituents(t2, fund, close, vol_t2, make_actions())
        assert list(sel_t2["ticker"]) == ["GOOD3"]

    def test_mid_quarter_delist_renormalizes(self):
        # X compounds 10%/day; Y trades until d2 (price 40), then delists.
        # The index sells Y at its last price and redistributes into X.
        dates = pd.date_range("2021-01-04", periods=5, freq="B")
        d0, d2, d4 = dates[0], dates[2], dates[4]
        adj_close = pd.DataFrame({
            "XXXX3": [100.0, 110.0, 121.0, 133.1, 146.41],
            "YYYY3": [50.0, 50.0, 40.0, 40.0, 40.0],  # ffilled after delist
        }, index=dates)
        holdings = {d0: pd.Series({"XXXX3": 0.5, "YYYY3": 0.5})}
        last_trade = pd.Series({"XXXX3": d4, "YYYY3": d2})

        level = build_index_series(adj_close, holdings, last_trade, end_date=d4)

        # Hand-computed: units X=0.005, Y=0.01.
        # V(d1)=0.005*110+0.01*50=1.05 ; V(d2)=0.605+0.4=1.005 (Y sold here)
        # X units scaled by 1 + 0.4/0.605 -> V(d3)=1.10550, V(d4)=1.216056
        assert level.loc[dates[1]] == pytest.approx(1.05)
        assert level.loc[d2] == pytest.approx(1.005)       # value conserved at delist
        assert level.loc[dates[3]] == pytest.approx(0.6655 * (1 + 0.4 / 0.605))
        assert level.loc[d4] == pytest.approx(0.6655 * 1.1 * (1 + 0.4 / 0.605))


# ── rebalance calendar ────────────────────────────────────────────────────────

def test_quarter_end_rebalance_dates_last_trading_day_of_quarter():
    # Business days Jan–mid-Jul 2021: only Mar and Jun quarter-ends qualify.
    days = pd.date_range("2021-01-04", "2021-07-15", freq="B")
    qe = quarter_end_rebalance_dates(days)
    assert list(qe) == [pd.Timestamp("2021-03-31"), pd.Timestamp("2021-06-30")]


# ── UI plugin weight timing (regression: one-quarter execution lag) ──────────

def test_generate_signals_weights_land_on_selection_quarter(monkeypatch):
    """Weights selected at quarter end t must be written to tw row t.

    run_simulation buys row-t weights at the close of t and applies the NEXT
    row's returns to them, so any shift to row t+1 executes a full quarter late
    relative to the CLI runner (build_index_series).
    """
    import backtests.strategies.sp500_b3 as mod

    # GOOD3 is liquid only through Q1 2021, so exactly one quarter selects it:
    # Q4-2020 has < 63 volume observations (med63 NaN), Q2/Q3 volume is zero.
    days = pd.date_range("2020-12-28", "2021-09-30", freq="B")
    close_px = pd.DataFrame({"GOOD3": 10.0}, index=days)
    fin_vol = pd.DataFrame({"GOOD3": 1e8}, index=days)
    fin_vol.loc[fin_vol.index > "2021-03-31"] = 0.0
    ret = close_px.resample("QE").last().pct_change()

    fund = make_fund([("GOOD", "2020-12-31", "2021-02-01", 1,
                       100_000.0, 1_000_000.0, 1e9)])
    monkeypatch.setattr(mod, "load_fundamentals_pit_raw", lambda db: fund)
    monkeypatch.setattr(mod, "load_stock_actions", lambda db: make_actions())

    shared = {"ret": ret, "close_px": close_px, "fin_vol": fin_vol}
    _, tw = mod.SP500B3Strategy().generate_signals(
        shared, {"min_constituents": 1, "db_path": "unused"}
    )

    q1 = pd.Timestamp("2021-03-31")
    assert tw.loc[q1, "GOOD3"] == pytest.approx(1.0)
    assert tw.drop(index=q1).to_numpy().sum() == 0.0  # not lagged into Q2
