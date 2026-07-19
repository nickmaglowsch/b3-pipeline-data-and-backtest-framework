"""Test the thirds split of DIVO11 / CDI / IVVB11 (synthetic, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtests.strategies.divo_cdi_ivvb as mod
from backtests.strategies.divo_cdi_ivvb import DivoCdiIvvbStrategy


def test_thirds_split(monkeypatch):
    idx = pd.date_range("2020-01-31", periods=4, freq="ME")
    ret = pd.DataFrame(0.01, index=idx, columns=["AAAA3"])
    sd = {"ret": ret, "cdi_monthly": pd.Series(0.01, index=idx)}

    # DIVO11 not priced until row 2; IVVB11 priced throughout
    def fake_dl(ticker, *a, **k):
        s = pd.Series(100.0, index=idx)
        if ticker == mod.DIVO11_TICKER:
            s.iloc[:2] = np.nan
        return s
    monkeypatch.setattr(mod, "download_benchmark", fake_dl)

    r, tw = DivoCdiIvvbStrategy().generate_signals(sd, {})
    assert {"DIVO11", "CDI_ASSET", "IVVB11"} <= set(r.columns)

    # every row is fully invested
    assert np.allclose(tw.sum(axis=1), 1.0)
    # pre-listing rows: DIVO third goes to CDI (2/3 CDI + 1/3 IVVB)
    assert np.isclose(tw.iloc[0]["DIVO11"], 0.0)
    assert np.isclose(tw.iloc[0]["CDI_ASSET"], 2 / 3)
    # live rows: exact thirds
    live = tw.iloc[-1]
    assert np.isclose(live["DIVO11"], 1 / 3)
    assert np.isclose(live["CDI_ASSET"], 1 / 3)
    assert np.isclose(live["IVVB11"], 1 / 3)
