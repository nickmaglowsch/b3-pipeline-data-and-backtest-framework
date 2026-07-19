"""Tests for _detect_and_fix_unrecorded_splits (backtests/core/shared_data.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtests.core.shared_data import _detect_and_fix_unrecorded_splits


def _frames(prices, vols=None):
    idx = pd.date_range("2020-01-01", periods=len(prices), freq="B")
    px = pd.DataFrame({"X": prices}, index=idx)
    fv = pd.DataFrame({"X": vols}, index=idx) if vols is not None else None
    return px, fv


def test_unrecorded_split_still_fixed():
    # Exact 2:1 drop, calm volume, price stays in the new regime -> a genuine
    # unrecorded split: all prior history divided by 2.
    px, fv = _frames([20.0] * 15 + [10.0] * 15, [1e6] * 30)
    fixed = _detect_and_fix_unrecorded_splits(px.copy(), px, fv)
    assert np.isclose(fixed["X"].iloc[0], 10.0)


def test_real_crash_with_volume_spike_untouched():
    # A genuine -50% crash prints anomalous volume — history must NOT be
    # rewritten (this erased real crashes before).
    vols = [1e6] * 15 + [2e7] + [5e6] * 14
    px, fv = _frames([20.0] * 15 + [10.0] * 15, vols)
    fixed = _detect_and_fix_unrecorded_splits(px.copy(), px, fv)
    assert np.isclose(fixed["X"].iloc[0], 20.0)


def test_one_day_glitch_untouched():
    # A one-day bad print that reverts the next day is not a split (neither is
    # the recovery day a reverse split).
    px, fv = _frames([20.0] * 15 + [10.0] + [20.0] * 14, [1e6] * 30)
    fixed = _detect_and_fix_unrecorded_splits(px.copy(), px, fv)
    assert np.isclose(fixed["X"].iloc[0], 20.0)
