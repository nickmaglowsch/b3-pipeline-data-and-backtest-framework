"""keep_most_liquid_per_root collapses multi-class companies to one ticker
(backtests/core/strategy_base.py)."""
from __future__ import annotations

import pandas as pd

from backtests.core.strategy_base import keep_most_liquid_per_root, dedup_target_weights


def test_keeps_highest_adtv_class_per_company():
    scores = pd.Series({"PETR3": 0.8, "PETR4": 0.9, "ABEV3": 0.5, "VALE3": 0.7})
    adtv = pd.Series({"PETR3": 1e6, "PETR4": 5e6, "ABEV3": 3e6})  # VALE3 ADTV missing
    out = keep_most_liquid_per_root(scores, adtv)
    assert set(out.index) == {"PETR4", "ABEV3", "VALE3"}   # PETR3 dropped (lower ADTV)
    assert out["PETR4"] == 0.9
    assert out["VALE3"] == 0.7                             # single class kept despite no ADTV


def test_empty_passthrough():
    assert keep_most_liquid_per_root(pd.Series(dtype=float), pd.Series(dtype=float)).empty


def test_dedup_target_weights_merges_dual_class_and_preserves_sum():
    idx = pd.date_range("2020-03-31", periods=2, freq="QE")
    tw = pd.DataFrame(
        {"PETR3": [0.3, 0.0], "PETR4": [0.2, 0.5], "ABEV3": [0.5, 0.5], "CDI_ASSET": [0.0, 0.0]},
        index=idx,
    )
    adtv = pd.DataFrame(
        {"PETR3": [1e6, 1e6], "PETR4": [9e6, 9e6], "ABEV3": [5e6, 5e6], "CDI_ASSET": [0.0, 0.0]},
        index=idx,
    )
    out = dedup_target_weights(tw, adtv)
    # row 0 held both PETR3+PETR4 -> merged onto PETR4 (higher ADTV), weight summed
    assert out.loc[idx[0], "PETR4"] == 0.5 and out.loc[idx[0], "PETR3"] == 0.0
    assert out.loc[idx[0], "ABEV3"] == 0.5
    # row 1 held only PETR4 -> untouched
    assert out.loc[idx[1], "PETR4"] == 0.5
    # weight sums preserved on every row
    assert (out.sum(axis=1) == tw.sum(axis=1)).all()


def test_dedup_target_weights_noop_when_one_class_per_company():
    idx = pd.date_range("2020-03-31", periods=1, freq="QE")
    tw = pd.DataFrame({"PETR4": [0.5], "ABEV3": [0.5]}, index=idx)
    adtv = pd.DataFrame({"PETR4": [9e6], "ABEV3": [5e6]}, index=idx)
    assert dedup_target_weights(tw, adtv).equals(tw)
