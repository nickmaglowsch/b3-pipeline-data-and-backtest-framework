"""Tests for the 1/3 Teva / 1/3 IVVB11 / 1/3 CDI blend.

Stubs the Teva sleeve and the Yahoo download so no DB or network is touched.
Checks the thirds split and that CDI absorbs residual weight (fully invested).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import backtests.strategies.teva_ivvb_cdi as mod
from backtests.strategies.teva_ivvb_cdi import TevaIvvbCdiStrategy, W_EACH

IDX = pd.date_range("2020-03-31", periods=4, freq="QE")


def _strategy(monkeypatch, teva_rows):
    ret = pd.DataFrame(0.0, index=IDX, columns=["AAAA3", "BBBB3"])

    class _StubTeva:
        def generate_signals(self, sd, p):
            tw = pd.DataFrame(0.0, index=IDX, columns=["AAAA3", "BBBB3"])
            for i, w in teva_rows.items():
                tw.iloc[i] = w
            return ret, tw

    monkeypatch.setattr(mod, "download_benchmark", lambda *a, **k: pd.Series(100.0, index=IDX))
    s = TevaIvvbCdiStrategy()
    s._teva = _StubTeva()
    shared = {"ret": ret, "cdi_monthly": pd.Series(0.01, index=IDX)}
    return s.generate_signals(shared, {})


def test_weights_always_sum_to_one(monkeypatch):
    _, tw = _strategy(monkeypatch, {2: [0.6, 0.4]})
    assert np.allclose(tw.sum(axis=1), 1.0)


def test_active_row_is_thirds(monkeypatch):
    _, tw = _strategy(monkeypatch, {2: [0.6, 0.4]})
    row = tw.iloc[2]
    assert abs(row[["AAAA3", "BBBB3"]].sum() - W_EACH) < 1e-9
    assert abs(row["IVVB11"] - W_EACH) < 1e-9
    assert abs(row["CDI_ASSET"] - W_EACH) < 1e-9


def test_empty_teva_sleeve_parks_in_cdi(monkeypatch):
    # no Teva rows active -> equity is just IVVB11 (1/3), CDI takes the other 2/3
    _, tw = _strategy(monkeypatch, {})
    assert abs(tw.iloc[0]["CDI_ASSET"] - 2 * W_EACH) < 1e-9
    assert abs(tw.iloc[0]["IVVB11"] - W_EACH) < 1e-9


def test_pre_ivvb_listing_parks_in_cdi(monkeypatch):
    # IVVB11 not yet listed on the first two rows -> that third goes to CDI too
    ret = pd.DataFrame(0.0, index=IDX, columns=["AAAA3", "BBBB3"])

    class _StubTeva:
        def generate_signals(self, sd, p):
            return ret, pd.DataFrame(0.0, index=IDX, columns=["AAAA3", "BBBB3"])

    px = pd.Series([np.nan, np.nan, 100.0, 101.0], index=IDX)
    monkeypatch.setattr(mod, "download_benchmark", lambda *a, **k: px)
    s = TevaIvvbCdiStrategy()
    s._teva = _StubTeva()
    _, tw = s.generate_signals({"ret": ret, "cdi_monthly": pd.Series(0.01, index=IDX)}, {})
    assert abs(tw.iloc[0]["CDI_ASSET"] - 1.0) < 1e-9  # all cash pre-listing
    assert np.allclose(tw.sum(axis=1), 1.0)
