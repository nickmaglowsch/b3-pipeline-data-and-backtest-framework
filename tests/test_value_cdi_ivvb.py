"""Test the thirds split of Value / CDI / IVVB11 (synthetic, no network/DB)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtests.strategies.value_cdi_ivvb as mod
from backtests.strategies.value_cdi_ivvb import ValueCdiIvvbStrategy


def test_thirds_split(monkeypatch):
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    tickers = ["AAAA3", "BBBB4"]
    ret = pd.DataFrame(0.01, index=idx, columns=tickers)
    sd = {"ret": ret, "cdi_monthly": pd.Series(0.01, index=idx)}

    # stub the value sleeve: uninvested first row, equal-weight basket after
    tw_value = pd.DataFrame(0.0, index=idx, columns=tickers)
    tw_value.iloc[1:] = 0.5  # two names, sum to 1.0 when invested
    monkeypatch.setattr(mod.TwoLegValueStrategy, "generate_signals",
                        lambda self, s, p: (ret, tw_value))
    monkeypatch.setattr(mod, "download_benchmark",
                        lambda *a, **k: pd.Series(100.0, index=idx))

    r, tw = ValueCdiIvvbStrategy().generate_signals(sd, {})
    assert "CDI_ASSET" in r.columns and "IVVB11" in r.columns

    # pre-basket row parks in CDI
    assert tw.iloc[0]["CDI_ASSET"] == 1.0
    # invested rows: exact thirds, sum to 1
    inv = tw.iloc[-1]
    assert np.isclose(inv["CDI_ASSET"], 1 / 3)
    assert np.isclose(inv["IVVB11"], 1 / 3)
    assert np.isclose(inv[tickers].sum(), 1 / 3)
    assert np.isclose(inv.sum(), 1.0)
