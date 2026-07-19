"""Test for BaselineThirdsStrategy: no allocation to a sleeve before it lists."""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtests.strategies.baseline_thirds as bt

IDX = pd.date_range("2013-01-31", periods=24, freq="ME")


def test_thirds_wait_for_both_etfs(monkeypatch):
    # BOVA11 trades the whole window; IVVB11 only lists at month 12. Before
    # both are live the book must sit 100% in CDI — not park a dead third at 0%.
    bova = pd.Series(np.linspace(50.0, 80.0, 24), index=IDX)
    ivvb = pd.Series(np.nan, index=IDX)
    ivvb.iloc[12:] = np.linspace(100.0, 120.0, 12)

    monkeypatch.setattr(
        bt, "download_benchmark",
        lambda ticker, start, end: bova if ticker == bt.BOVA11_TICKER else ivvb,
    )
    shared = {
        "ret": pd.DataFrame(0.0, index=IDX, columns=["X"]),
        "cdi_monthly": pd.Series(0.01, index=IDX),
    }
    r, tw = bt.BaselineThirdsStrategy().generate_signals(shared, {})

    pre, post = tw.iloc[:13], tw.iloc[13:]
    assert (pre["CDI_ASSET"] == 1.0).all()
    assert (pre[["BOVA11", "IVVB11"]] == 0.0).all().all()
    assert np.allclose(post[["BOVA11", "IVVB11", "CDI_ASSET"]], 1.0 / 3.0)
