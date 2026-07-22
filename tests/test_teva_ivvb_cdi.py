"""Tests for the 1/3 Teva / 1/3 IVVB11 / 1/3 CDI blend.

The bespoke ``TevaIvvbCdiStrategy`` class was replaced by
``backtests/strategies/specs/teva_ivvb_cdi.yaml`` on the generic ``FixedWeight``
engine, after an exact target-weight parity check against the old class on the
real DB (2010-2020, QE: max abs weight diff 1.1e-16, 0/44 rows differing).

Stubs the Teva sleeve and the Yahoo download so no DB or network is touched.
Checks the thirds split and that CDI absorbs residual weight (fully invested).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import backtests.core.data as core_data
from backtests.core import strategy_registry as sr
from backtests.core.config_strategy import FixedWeight
from backtests.core.strategy_base import StrategyBase

SPEC_PATH = (Path(__file__).resolve().parent.parent / "backtests" / "strategies"
             / "specs" / "teva_ivvb_cdi.yaml")
IDX = pd.date_range("2020-03-31", periods=4, freq="QE")
TICKERS = ["AAAA3", "BBBB3"]
W_EACH = 1.0 / 3.0


def _run(monkeypatch, teva_rows: dict, ivvb_px: pd.Series | None = None):
    """Run the spec with a stubbed Teva sleeve and a stubbed IVVB11 price."""
    ret = pd.DataFrame(0.0, index=IDX, columns=TICKERS)

    class _StubTeva(StrategyBase):
        @property
        def name(self) -> str:
            return "TevaAtivosReais"

        @property
        def description(self) -> str:
            return "stub"

        def get_parameter_specs(self):
            return []

        def generate_signals(self, shared_data, params):
            tw = pd.DataFrame(0.0, index=IDX, columns=TICKERS)
            for i, w in teva_rows.items():
                tw.iloc[i] = w
            return ret, tw

    reg = sr.StrategyRegistry()
    reg.register(_StubTeva())
    monkeypatch.setattr(sr, "get_registry", lambda: reg)
    px = pd.Series(100.0, index=IDX) if ivvb_px is None else ivvb_px
    monkeypatch.setattr(core_data, "download_benchmark", lambda *a, **k: px)

    shared = {"ret": ret, "cdi_monthly": pd.Series(0.01, index=IDX)}
    return FixedWeight(yaml.safe_load(SPEC_PATH.read_text())).generate_signals(shared, {})


def test_registered_name_is_unchanged():
    # other code looks this strategy up by name — it must stay byte-identical
    assert yaml.safe_load(SPEC_PATH.read_text())["name"] == "Teva / IVVB11 / CDI (1/3 each)"


def test_weights_always_sum_to_one(monkeypatch):
    _, tw = _run(monkeypatch, {2: [0.6, 0.4]})
    assert np.allclose(tw.sum(axis=1), 1.0)


def test_active_row_is_thirds(monkeypatch):
    _, tw = _run(monkeypatch, {2: [0.6, 0.4]})
    row = tw.iloc[2]
    assert abs(row[TICKERS].sum() - W_EACH) < 1e-9
    assert abs(row["IVVB11"] - W_EACH) < 1e-9
    assert abs(row["CDI_ASSET"] - W_EACH) < 1e-9


def test_empty_teva_sleeve_parks_in_cdi(monkeypatch):
    # no Teva rows active -> equity is just IVVB11 (1/3), CDI takes the other 2/3
    _, tw = _run(monkeypatch, {})
    assert abs(tw.iloc[0]["CDI_ASSET"] - 2 * W_EACH) < 1e-9
    assert abs(tw.iloc[0]["IVVB11"] - W_EACH) < 1e-9


def test_pre_ivvb_listing_parks_in_cdi(monkeypatch):
    # IVVB11 not yet listed on the first two rows -> that third goes to CDI too
    px = pd.Series([np.nan, np.nan, 100.0, 101.0], index=IDX)
    _, tw = _run(monkeypatch, {}, ivvb_px=px)
    assert abs(tw.iloc[0]["CDI_ASSET"] - 1.0) < 1e-9  # all cash pre-listing
    assert np.allclose(tw.sum(axis=1), 1.0)
