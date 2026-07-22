"""
Behaviour tests for the config-driven strategy engines.

The Python strategy classes these specs replaced were golden-tested (exact
pre-tax AND after-tax equity-curve parity on the real DB via
``scripts/golden_config_parity.py``) and then deleted. These tests guard the
config engines against regression on synthetic data — no dependency on the
removed classes.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtests.core.config_strategy import RankAndHold, FixedWeight
import backtests.core.data as core_data

SPECS = Path(__file__).resolve().parent.parent / "backtests" / "strategies" / "specs"

RANK_SPECS = ["low_vol.yaml", "multifactor.yaml", "smallcap_momentum.yaml", "momentum_sharpe.yaml"]


def _spec(fname: str) -> dict:
    import yaml
    return yaml.safe_load((SPECS / fname).read_text())


def _synthetic_shared(n_periods: int = 72, n_tickers: int = 40, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n_periods, freq="ME")
    cols = [f"T{i:02d}" for i in range(n_tickers)]
    ret = pd.DataFrame(rng.normal(0.01, 0.08, (n_periods, n_tickers)), index=idx, columns=cols)
    adtv = pd.DataFrame(rng.uniform(5e4, 5e6, (n_periods, n_tickers)), index=idx, columns=cols)
    raw_close = pd.DataFrame(rng.uniform(2.0, 100.0, (n_periods, n_tickers)), index=idx, columns=cols)
    has_glitch = pd.DataFrame(0, index=idx, columns=cols)
    has_glitch.iloc[10, 3] = 1
    return {
        "ret": ret, "log_ret": np.log1p(ret), "adtv": adtv,
        "raw_close": raw_close, "has_glitch": has_glitch,
    }


@pytest.mark.parametrize("spec_file", RANK_SPECS)
def test_rank_and_hold_places_equal_weights(spec_file):
    shared = _synthetic_shared()
    _, tw = RankAndHold(_spec(spec_file)).generate_signals(shared, {})

    active = tw[(tw != 0).any(axis=1)]
    assert len(active) > 0, "strategy never invested"
    # equal weighting: the nonzero weights within each active row are all equal
    for _, row in active.iterrows():
        nz = row[row > 0]
        assert np.allclose(nz.to_numpy(), nz.iloc[0])
    # never over-allocate
    assert (tw.sum(axis=1) <= 1.0 + 1e-9).all()


def test_regime_overlay_parks_in_cdi_when_gate_off():
    # RegimeSwitching: hold equities only when IBOV is above its MA (ibov_above),
    # else 100% CDI. Gate reads ibov_above at row i-1.
    shared = _synthetic_shared()
    idx = shared["ret"].index
    shared["mf_composite"] = pd.DataFrame(
        np.random.default_rng(1).normal(0, 1, shared["ret"].shape), index=idx, columns=shared["ret"].columns)
    shared["cdi_monthly"] = pd.Series(0.01, index=idx)
    shared["ibov_ret"] = pd.Series(0.005, index=idx)
    above = pd.Series(True, index=idx)
    above.iloc[30:] = False           # regime turns "off" from row 30
    shared["ibov_above"] = above

    _, tw = RankAndHold(_spec("regime_switching.yaml")).generate_signals(shared, {})
    # rows whose gate (i-1) is off -> 100% CDI, no equity
    off_rows = tw.index[31:]          # ibov_above.iloc[i-1] False for i>=31
    assert (tw.loc[off_rows, "CDI_ASSET"] == 1.0).all()
    stock_cols = [c for c in tw.columns if c not in ("CDI_ASSET", "IBOV")]
    assert (tw.loc[off_rows, stock_cols] == 0.0).all().all()
    # some earlier (gate on) row is invested in equities, not CDI
    on = tw.iloc[20]
    assert on[stock_cols].sum() > 0 and on["CDI_ASSET"] == 0.0


def _blend_shared(n: int = 24):
    idx = pd.date_range("2010-03-31", periods=n, freq="QE")
    return idx, {
        "ret": pd.DataFrame(0.0, index=idx, columns=["A", "B"]),
        "cdi_monthly": pd.Series(0.02, index=idx),
    }


def test_fixed_weight_per_sleeve_parks_dead_etf(monkeypatch):
    idx, shared = _blend_shared()
    divo = pd.Series(np.linspace(20.0, 60.0, len(idx)), index=idx)
    ivvb = pd.Series(np.nan, index=idx)
    ivvb.iloc[12:] = np.linspace(100.0, 150.0, len(idx) - 12)  # lists at row 12
    prices = {"DIVO11.SA": divo, "IVVB11.SA": ivvb}
    monkeypatch.setattr(core_data, "download_benchmark", lambda t, s, e: prices[t])

    _, tw = FixedWeight(_spec("divo_cdi_ivvb.yaml")).generate_signals(shared, {})
    third = 1.0 / 3.0
    # before IVVB lists: its third is parked in CDI, DIVO keeps its third
    assert np.allclose(tw["IVVB11"].iloc[:12], 0.0)
    assert np.allclose(tw["DIVO11"].iloc[:12], third)
    assert np.allclose(tw["CDI_ASSET"].iloc[:12], 2 * third)
    # once both live: equal thirds
    assert np.allclose(tw[["DIVO11", "IVVB11", "CDI_ASSET"]].iloc[12:], third)


def test_fixed_weight_parks_nested_strategy_warmup_residual(monkeypatch):
    # A nested `strategy` sleeve emits all-zero weights during its own warmup.
    # That share must be parked in CDI (weights sum to 1), not left unallocated —
    # run_simulation vaporizes uninvested weight, faking a compounding drawdown.
    from backtests.core import strategy_registry as sr
    from backtests.core.strategy_base import StrategyBase

    idx, shared = _blend_shared()  # tickers A, B

    class _Stub(StrategyBase):
        @property
        def name(self): return "_StubSleeve"
        @property
        def description(self): return "stub"
        def get_parameter_specs(self): return []
        def generate_signals(self, shared_data, params):
            ret = shared_data["ret"]
            tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
            tw.iloc[6:, tw.columns.get_loc("A")] = 1.0   # all-zero warmup, then 100% A
            return ret, tw

    reg = sr.StrategyRegistry()
    reg.register(_Stub())
    monkeypatch.setattr(sr, "get_registry", lambda: reg)

    ivvb = pd.Series(np.nan, index=idx)
    ivvb.iloc[12:] = np.linspace(100.0, 150.0, len(idx) - 12)  # lists at row 12
    monkeypatch.setattr(core_data, "download_benchmark", lambda t, s, e: ivvb)

    spec = {
        "name": "stub blend", "kind": "fixed_weight", "rebalance": "QE",
        "park_in_cdi_until_live": True,
        "sleeves": [{"strategy": "_StubSleeve", "weight": 0.5},
                    {"ticker": "IVVB11.SA", "weight": 0.5}],
    }
    _, tw = FixedWeight(spec).generate_signals(shared, {})

    rowsum = tw.sum(axis=1)
    live = rowsum > 0
    assert np.allclose(rowsum[live], 1.0), f"live rows must sum to 1, got {rowsum[live].unique()}"
    # sleeve-warmup + dead ETF -> everything in CDI (nothing vaporized)
    assert np.allclose(tw["CDI_ASSET"].iloc[:6], 1.0)


def test_fixed_weight_all_or_nothing_holds_cdi_until_all_live(monkeypatch):
    idx, shared = _blend_shared()
    bova = pd.Series(np.linspace(50.0, 80.0, len(idx)), index=idx)
    ivvb = pd.Series(np.nan, index=idx)
    ivvb.iloc[12:] = np.linspace(100.0, 120.0, len(idx) - 12)
    prices = {"BOVA11.SA": bova, "IVVB11.SA": ivvb}
    monkeypatch.setattr(core_data, "download_benchmark", lambda t, s, e: prices[t])

    _, tw = FixedWeight(_spec("baseline_thirds.yaml")).generate_signals(shared, {})
    # before both ETFs are live: 100% CDI, no equity
    assert (tw["CDI_ASSET"].iloc[:13] == 1.0).all()
    assert (tw[["BOVA11", "IVVB11"]].iloc[:13] == 0.0).all().all()
    # after both live: equal thirds
    assert np.allclose(tw[["BOVA11", "IVVB11", "CDI_ASSET"]].iloc[13:], 1.0 / 3.0)
