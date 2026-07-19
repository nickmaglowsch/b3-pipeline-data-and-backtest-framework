"""Test the 40/40/20 split of DIVO11 / IVVB11 / CDI (synthetic, no network)."""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtests.strategies.divo_ivvb_cdi_40_40_20 as mod
from backtests.strategies.divo_ivvb_cdi_40_40_20 import DivoIvvbCdi404020Strategy


def test_weights_split(monkeypatch):
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

    r, tw = DivoIvvbCdi404020Strategy().generate_signals(sd, {})
    assert {"DIVO11", "CDI_ASSET", "IVVB11"} <= set(r.columns)

    # every row is fully invested
    assert np.allclose(tw.sum(axis=1), 1.0)
    # pre-listing rows: DIVO weight goes to CDI (0.20 + 0.40 CDI, 0.40 IVVB)
    assert np.isclose(tw.iloc[0]["DIVO11"], 0.0)
    assert np.isclose(tw.iloc[0]["CDI_ASSET"], 0.60)
    assert np.isclose(tw.iloc[0]["IVVB11"], 0.40)
    # live rows: exact weights
    live = tw.iloc[-1]
    assert np.isclose(live["DIVO11"], 0.40)
    assert np.isclose(live["IVVB11"], 0.40)
    assert np.isclose(live["CDI_ASSET"], 0.20)
