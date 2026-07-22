"""Assembly contract for build_strategy_returns().

The 9 strategies it runs are the registry's, not private copies — these tests
pin the wiring (registry lookup, simulation call, benchmark columns, failure
degradation) on stubs, so they stay fast and need neither the DB nor network.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests.core import strategy_returns as sr

NAMES = [
    "CDI+MA200", "Res.MultiFactor", "RegimeSwitching", "COPOM Easing",
    "MultiFactor", "SmallcapMom", "LowVol", "MomSharpe", "MeanRevComposite",
]


class _StubStrategy:
    """Registry stand-in: records the params it was handed, emits a 1-name book."""

    def __init__(self, name: str, boom: bool = False) -> None:
        self.name = name
        self.boom = boom
        self.seen_params: dict | None = None

    def get_default_parameters(self) -> dict:
        return {"lookback": 12}

    def generate_signals(self, shared_data, params):
        self.seen_params = params
        if self.boom:
            raise RuntimeError(f"{self.name} exploded")
        ret = shared_data["ret"]
        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        tw.iloc[1:, 0] = 1.0
        return ret, tw


class _StubRegistry:
    def __init__(self, strategies: dict) -> None:
        self._s = strategies

    def get(self, name):
        return self._s[name]


def _shared(n: int = 6) -> dict:
    idx = pd.date_range("2010-01-31", periods=n, freq="ME")
    ret = pd.DataFrame(0.01, index=idx, columns=["AAAA3", "BBBB4"])
    return {
        "ret": ret,
        "ibov_ret": pd.Series(0.02, index=idx),
        "cdi_monthly": pd.Series(0.005, index=idx),
        "is_easing": pd.Series(True, index=idx),
        "ibov_calm": pd.Series(True, index=idx),
        "ibov_uptrend": pd.Series(False, index=idx),
        "ibov_above": pd.Series(True, index=idx),
    }


@pytest.fixture
def wired(monkeypatch):
    """Patch out the DB/network and the registry; hand back the stubs."""
    shared = _shared()
    stubs = {n: _StubStrategy(n) for n in NAMES}
    monkeypatch.setattr(sr, "build_shared_data", lambda *a, **k: shared)
    monkeypatch.setattr(sr, "get_registry", lambda: _StubRegistry(stubs))

    calls = []

    def _fake_sim(**kw):
        calls.append(kw)
        idx = shared["ret"].index
        return {"aftertax_values": pd.Series(np.linspace(100.0, 150.0, len(idx)), index=idx)}

    monkeypatch.setattr(sr, "run_simulation", _fake_sim)
    return shared, stubs, calls


def test_returns_df_has_every_strategy_plus_benchmarks(wired):
    shared, _, _ = wired
    returns_df, _, _ = sr.build_strategy_returns(db_path="ignored")

    assert list(returns_df.columns) == NAMES + ["IBOV", "CDI"]
    assert (returns_df["IBOV"].dropna() == 0.02).all()
    assert (returns_df["CDI"].dropna() == 0.005).all()


def test_each_strategy_comes_from_the_registry(wired):
    _, stubs, _ = wired
    sr.build_strategy_returns(db_path="ignored", start="2011-02-01", end="2019-09-30")

    for name, s in stubs.items():
        assert s.seen_params is not None, f"{name} was never invoked"
        # the strategy's OWN defaults must survive — passing a bare {} here is
        # how the deleted private copies silently drifted from the registry
        assert s.seen_params["lookback"] == 12, f"{name} lost its defaults"
        assert s.seen_params["start_date"] == "2011-02-01"
        assert s.seen_params["end_date"] == "2019-09-30"
        assert s.seen_params["rebalance_freq"] == "ME"


def test_sim_config_is_threaded_through(wired):
    _, _, calls = wired
    sr.build_strategy_returns(
        db_path="ignored", capital=250_000.0, tax_rate=0.20,
        slippage=0.002, monthly_sales_exemption=35_000.0,
    )

    assert len(calls) == len(NAMES)
    for kw in calls:
        assert kw["initial_capital"] == 250_000.0
        assert kw["tax_rate"] == 0.20
        assert kw["slippage"] == 0.002
        assert kw["monthly_sales_exemption"] == 35_000.0
    assert {kw["name"] for kw in calls} == set(NAMES)


def test_sim_results_keyed_by_strategy_name(wired):
    _, _, _ = wired
    _, sim_results, _ = sr.build_strategy_returns(db_path="ignored")

    assert set(sim_results) == set(NAMES)
    assert all("aftertax_values" in r for r in sim_results.values())


def test_regime_signals_passed_through(wired):
    shared, _, _ = wired
    _, _, regime = sr.build_strategy_returns(db_path="ignored")

    assert set(regime) == {"is_easing", "ibov_calm", "ibov_uptrend", "ibov_above"}
    for k, v in regime.items():
        pd.testing.assert_series_equal(v, shared[k])


def test_one_broken_strategy_does_not_sink_the_rest(monkeypatch, wired):
    shared, stubs, _ = wired
    stubs["LowVol"].boom = True

    returns_df, sim_results, _ = sr.build_strategy_returns(db_path="ignored")

    assert "LowVol" in returns_df.columns          # column survives, empty
    assert returns_df["LowVol"].isna().all()
    assert "LowVol" not in sim_results             # but no bogus result recorded
    assert returns_df["MomSharpe"].notna().any()   # siblings unaffected
