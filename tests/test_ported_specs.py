"""
Behaviour tests for the 8 specs ported from hand-written strategy classes, plus
the signal-DSL primitives that port needed (`where`, `nan_add`, `at_rebalance`,
comparisons and `&`/`|`).

Each was golden-tested against its Python original — exact `target_weights` AND
after-tax equity-curve parity on the real DB over 2005-2025 via
``scripts/golden_config_parity.py`` — and the class then deleted. These tests are
the permanent guard: synthetic data only, no DB, no network.

The LowPE spec has its own file (``tests/test_low_pe_strategy.py``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pandas.testing as pdt
import pytest
import yaml

from backtests.core import signal_dsl
from backtests.core.config_strategy import RankAndHold

SPECS = Path(__file__).resolve().parent.parent / "backtests" / "strategies" / "specs"


def _spec(fname: str) -> dict:
    return yaml.safe_load((SPECS / fname).read_text())


def _grid(n_periods: int, tickers: list[str]) -> pd.DatetimeIndex:
    return pd.date_range("2010-01-31", periods=n_periods, freq="ME")


def _base(idx, cols, ret) -> dict:
    """shared_data with a fully-eligible universe (liquid, above the price floor,
    no glitches) so a test only has to shape the factor it cares about."""
    return {
        "ret": ret,
        "log_ret": np.log1p(ret),
        "adtv": pd.DataFrame(5e6, index=idx, columns=cols),
        "raw_close": pd.DataFrame(50.0, index=idx, columns=cols),
        "has_glitch": pd.DataFrame(0.0, index=idx, columns=cols),
    }


def _weighted(row: pd.Series) -> list[str]:
    return sorted(row[row > 0].index)


# ── SimpleMeanReversion ───────────────────────────────────────────────────────

def test_simple_mean_reversion_buys_losers_and_skips_glitch_prints():
    cols = [f"T{i:02d}" for i in range(10)]
    idx = _grid(4, cols)
    ret = pd.DataFrame(0.0, index=idx, columns=cols)
    #        T00     T01    T02    T03    T04    T05    T06   T07   T08   T09
    ret.iloc[1] = [-0.95, -0.30, -0.25, -0.20, -0.15, -0.10, 0.05, 0.10, 0.15, 0.20]

    _, tw = RankAndHold(_spec("simple_mean_reversion.yaml")).generate_signals(
        _base(idx, cols, ret), {})

    row = tw.iloc[2]                       # warmup = lookback(1) + pad(1) = 2
    # top_pct 0.10 of 10 names -> the min_names floor of 5 applies
    assert _weighted(row) == ["T01", "T02", "T03", "T04", "T05"]
    assert np.allclose(row[_weighted(row)].to_numpy(), 0.2)
    # T00 is the biggest loser but -95% is the strategy's glitch band, not alpha
    assert row["T00"] == 0.0


def test_simple_mean_reversion_sticky_holds_book_through_a_thin_month():
    """The deleted class carried `prev_sel` when < 5 names qualified; without
    `sticky` the thin month empties the book and the next rebalance re-enters
    from cash, changing weights, turnover and realised tax from then on."""
    cols = [f"T{i:02d}" for i in range(10)]
    idx = _grid(5, cols)
    ret = pd.DataFrame(0.0, index=idx, columns=cols)
    ret.iloc[1] = [-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20]

    shared = _base(idx, cols, ret)
    # row 2 reads a fully liquid month; row 3 reads a month where only 2 names
    # clear min_adtv -> too few to rank
    shared["adtv"].iloc[2, 2:] = 0.0

    _, tw = RankAndHold(_spec("simple_mean_reversion.yaml")).generate_signals(shared, {})

    assert tw.iloc[2].sum() == pytest.approx(1.0), "baseline month must be invested"
    assert tw.iloc[3].sum() == pytest.approx(1.0), "thin month must hold, not go to cash"
    pdt.assert_series_equal(tw.iloc[3], tw.iloc[2], check_names=False)


# ── FrogInPan ─────────────────────────────────────────────────────────────────

def test_frog_in_pan_ranks_momentum_plus_win_rate_and_masks_glitches():
    cols = [f"T{i:02d}" for i in range(20)]
    idx = _grid(20, cols)
    # constant per-ticker drift: momentum AND win-rate both increase with i
    ret = pd.DataFrame(
        np.tile(np.arange(20) * 0.01 - 0.05, (20, 1)), index=idx, columns=cols)
    ret.iloc[5, 19] = -0.95            # T19: a glitch print inside the lookback

    _, tw = RankAndHold(_spec("frog_in_pan.yaml")).generate_signals(
        _base(idx, cols, ret), {})

    row = tw.iloc[14]                  # warmup = lookback(12) + pad(2)
    assert _weighted(row) == ["T14", "T15", "T16", "T17", "T18"]
    assert np.allclose(row[_weighted(row)].to_numpy(), 0.2)
    assert row["T19"] == 0.0           # highest drift, but glitch-masked


# ── VolumeBreakout ────────────────────────────────────────────────────────────

def _volume_breakout_shared():
    cols = [f"T{i:02d}" for i in range(10)]
    idx = _grid(16, cols)
    ret = pd.DataFrame(
        [[0.01] * 6 + [-0.01] * 4] * 16, index=idx, columns=cols)
    ret.iloc[10] = -0.01               # washout month: nothing qualifies at i=12
    shared = _base(idx, cols, ret)
    # distinct, drifting volume so vol_accel has no ties
    shared["adtv"] = pd.DataFrame(
        2e6 * (1 + 0.01 * np.outer(np.arange(16), np.arange(1, 11))),
        index=idx, columns=cols)
    return shared, cols


def test_volume_breakout_requires_positive_prior_return():
    shared, cols = _volume_breakout_shared()
    _, tw = RankAndHold(_spec("volume_breakout.yaml")).generate_signals(shared, {})

    active = tw[(tw != 0).any(axis=1)]
    assert len(active) > 0
    # only the six always-rising names can ever be held
    assert (active[cols[6:]] == 0.0).all().all()
    assert (active[cols[:6]] > 0).sum(axis=1).eq(5).all()   # min_names floor of 5


def test_volume_breakout_sticky_holds_book_through_a_thin_month():
    shared, _ = _volume_breakout_shared()
    _, tw = RankAndHold(_spec("volume_breakout.yaml")).generate_signals(shared, {})

    # i=12 reads the signal built off the all-negative month -> zero candidates.
    # `sticky: true` carries the previous book instead of going to cash.
    assert tw.iloc[12].sum() == pytest.approx(1.0)
    pdt.assert_series_equal(tw.iloc[12], tw.iloc[11], check_names=False)


def test_volume_breakout_without_sticky_goes_flat():
    """Guard the option itself: drop `sticky` and the thin month empties."""
    shared, _ = _volume_breakout_shared()
    spec = _spec("volume_breakout.yaml")
    spec["selection"].pop("sticky")
    _, tw = RankAndHold(spec).generate_signals(shared, {})
    assert tw.iloc[12].sum() == 0.0


# ── ValueQuality ──────────────────────────────────────────────────────────────

def test_value_quality_composite_survives_a_missing_leg_and_dedups_roots():
    cols = ["PETR3", "PETR4", "VALE3", "ITUB4", "BBAS3", "ABEV3", "WEGE3"]
    idx = _grid(3, cols)
    ret = pd.DataFrame(0.0, index=idx, columns=cols)
    shared = _base(idx, cols, ret)
    shared["adtv"] = pd.DataFrame(5e6, index=idx, columns=cols)
    shared["adtv"]["PETR4"] = 9e6                    # the liquid PETR class

    pb = {"PETR3": 1.0, "PETR4": 1.0, "VALE3": 1.5,
          "ITUB4": 8.0, "BBAS3": 2.0, "ABEV3": 3.0, "WEGE3": 4.0}
    shared["f_pb_ratio_dyn"] = pd.DataFrame(
        [pb] * 3, index=idx, columns=cols)
    # ROE covers ONE ticker only: a plain `+` composite would NaN out every other
    # name and the strategy would never invest (the historical bug nan_add fixes).
    ni = pd.DataFrame(np.nan, index=idx, columns=cols)
    eq = pd.DataFrame(np.nan, index=idx, columns=cols)
    ni["WEGE3"], eq["WEGE3"] = 100.0, 100.0
    shared["f_net_income"], shared["f_equity"] = ni, eq

    _, tw = RankAndHold(_spec("value_quality.yaml")).generate_signals(shared, {})

    row = tw.iloc[1]
    # top_pct 0.20 of 6 deduped names -> min_names floor of 3
    assert _weighted(row) == ["PETR4", "VALE3", "WEGE3"]
    assert np.allclose(row[_weighted(row)].to_numpy(), 1 / 3)
    assert row["ITUB4"] == 0.0        # P/B 8 > max_pb 5
    assert row["PETR3"] == 0.0        # collapsed onto the higher-ADTV class


# ── WinRateMeanRev ────────────────────────────────────────────────────────────

def test_win_rate_mean_rev_excludes_names_trending_above_their_own_average():
    kept, trend = ["KAAA3", "KBBB3"], ["XAAA3", "XBBB3", "XCCC3", "XDDD3"]
    cols = kept + trend
    days = pd.bdate_range("2021-01-01", periods=300)
    alt = np.where(np.arange(len(days)) % 2 == 0, 0.01, -0.01)   # win rate ~0.5

    adj = pd.DataFrame({t: 100 * np.cumprod(1 + alt) for t in cols}, index=days)
    # the row the last rebalance reads: 20 straight up days for the trend names,
    # so their 20d win rate (1.0) sits way above its own 60d mean -> ratio > 1.25
    eval_dt = adj.resample("ME").last().index[-2]
    t = adj.index.get_indexer([eval_dt], method="ffill")[0]
    for name in trend:
        r = alt.copy()
        r[t - 19:t + 1] = 0.01
        adj[name] = 100 * np.cumprod(1 + r)

    ret = adj.resample("ME").last().pct_change()
    shared = _base(ret.index, cols, ret)
    shared["adj_close"] = adj

    _, tw = RankAndHold(_spec("win_rate_mean_rev.yaml")).generate_signals(shared, {})

    row = tw.iloc[-1]
    assert _weighted(row) == kept
    # min_names 1 lets it invest on 2 survivors; min_hold 5 caps nothing here
    assert np.allclose(row[kept].to_numpy(), 0.5)
    assert (row[trend] == 0.0).all()


# ── Res.MultiFactor ───────────────────────────────────────────────────────────

def test_research_multifactor_needs_two_of_three_regime_signals():
    cols = [f"T{i:02d}" for i in range(20)]
    idx = _grid(30, cols)
    ret = pd.DataFrame(0.01, index=idx, columns=cols)
    shared = _base(idx, cols, ret)
    ramp = pd.DataFrame(np.tile(np.arange(20, dtype=float), (30, 1)),
                        index=idx, columns=cols)
    shared["dist_ma200"] = ramp
    shared["vol_60d"] = -ramp
    shared["atr_m"] = -ramp
    shared["vol_20d"] = -ramp
    shared["cdi_monthly"] = pd.Series(0.01, index=idx)
    shared["ibov_ret"] = pd.Series(0.005, index=idx)
    shared["is_easing"] = pd.Series(True, index=idx)
    calm = pd.Series(True, index=idx)
    up = pd.Series(True, index=idx)
    calm.iloc[20:] = False              # from row 20 only `easing` is left = 1 of 3
    up.iloc[20:] = False
    shared["ibov_calm"], shared["ibov_uptrend"] = calm, up

    _, tw = RankAndHold(_spec("research_multifactor.yaml")).generate_signals(shared, {})

    stocks = [c for c in tw.columns if c not in ("CDI_ASSET", "IBOV")]
    on = tw.iloc[15]                   # 3 of 3 -> invested, top decile of 20 = 5
    assert on["CDI_ASSET"] == 0.0
    assert _weighted(on[stocks]) == ["T15", "T16", "T17", "T18", "T19"]
    off = tw.iloc[21]                  # 1 of 3 -> 100% CDI
    assert off["CDI_ASSET"] == 1.0
    assert (off[stocks] == 0.0).all()


# ── CDI+MA200 ─────────────────────────────────────────────────────────────────

def test_cdi_ma200_holds_every_name_above_its_ma_while_easing():
    cols = [f"T{i:02d}" for i in range(10)]
    idx = _grid(20, cols)
    ret = pd.DataFrame(0.01, index=idx, columns=cols)
    shared = _base(idx, cols, ret)
    shared["cdi_monthly"] = pd.Series(0.01, index=idx)
    shared["ibov_ret"] = pd.Series(0.005, index=idx)
    above = pd.DataFrame(False, index=idx, columns=cols)
    above.iloc[:, :7] = True           # 7 of 10 names above their MA200
    shared["above_ma200"] = above
    easing = pd.Series(True, index=idx)
    easing.iloc[16:] = False
    shared["is_easing"] = easing

    _, tw = RankAndHold(_spec("cdi_ma200.yaml")).generate_signals(shared, {})

    row = tw.iloc[15]                  # easing: every qualifying name, equal weight
    assert _weighted(row) == cols[:7]
    assert np.allclose(row[cols[:7]].to_numpy(), 1 / 7)
    assert row["CDI_ASSET"] == 0.0
    assert (tw.loc[idx[16:], "CDI_ASSET"] == 1.0).all()        # tightening -> cash


# ── signal-DSL primitives added for the port ──────────────────────────────────

def _dsl_ns():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    cols = ["A", "B"]
    return {
        "ret": pd.DataFrame([[0.1, -0.2], [np.nan, 0.3], [0.5, -0.6],
                             [0.7, 0.8], [-0.9, 1.5], [0.0, 0.0]],
                            index=idx, columns=cols),
        "lo": pd.DataFrame([[1.0, np.nan]] * 6, index=idx, columns=cols),
        "hi": pd.DataFrame([[np.nan, 2.0]] * 6, index=idx, columns=cols),
    }


def test_where_masks_on_a_comparison():
    ns = _dsl_ns()
    got = signal_dsl.evaluate("where(ret, (ret <= 1.0) & (ret >= -0.9))", ns)
    pdt.assert_frame_equal(got, ns["ret"].where((ns["ret"] <= 1.0) & (ns["ret"] >= -0.9)))
    assert np.isnan(got.iloc[4, 1])          # 1.5 > 1.0 -> masked
    assert got.iloc[4, 0] == -0.9            # boundary is inclusive


def test_boolean_or_and_ne():
    ns = _dsl_ns()
    got = signal_dsl.evaluate("where(ret, ((ret > 1.0) | (ret < -0.9)) != 1)", ns)
    pdt.assert_frame_equal(got, ns["ret"].where(~((ns["ret"] > 1.0) | (ns["ret"] < -0.9))))


def test_nan_add_keeps_a_lone_leg():
    ns = _dsl_ns()
    got = signal_dsl.evaluate("nan_add(lo, hi)", ns)
    assert got.iloc[0, 0] == 1.0             # only the `lo` leg present
    assert got.iloc[0, 1] == 2.0             # only the `hi` leg present
    # ...whereas a plain `+` would NaN both out
    assert signal_dsl.evaluate("lo + hi", ns).isna().all().all()


def test_at_rebalance_pulls_a_daily_frame_onto_the_grid():
    idx = pd.date_range("2020-01-31", periods=3, freq="ME")
    daily = pd.bdate_range("2020-01-01", "2020-03-31")
    ns = {
        "ret": pd.DataFrame(0.0, index=idx, columns=["A", "B"]),
        "d": pd.DataFrame(np.arange(len(daily) * 3, dtype=float).reshape(-1, 3),
                          index=daily, columns=["A", "B", "C"]),
    }
    got = signal_dsl.evaluate("at_rebalance(d)", ns)
    assert list(got.columns) == ["A", "B"]   # extra ticker dropped, grid aligned
    assert got.index.equals(idx)
    pdt.assert_frame_equal(got, ns["d"].resample("ME").last()[["A", "B"]])


@pytest.mark.parametrize("expr", [
    "1 < ret < 2",              # chained comparison
    "where(ret, ret is None)",  # `is` comparison is not whitelisted
])
def test_rejects_disallowed_comparisons(expr):
    with pytest.raises((ValueError, KeyError)):
        signal_dsl.evaluate(expr, _dsl_ns())
