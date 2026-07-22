"""Tests for the generic simulation engine (backtests/core/simulation.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests.core.simulation import run_simulation, _apply_returns, _execute_rebalance

IDX = pd.date_range("2020-01-31", periods=5, freq="ME")
COLS = ["A", "B"]


def _run(weights_rows, returns_rows, **kw):
    rm = pd.DataFrame(returns_rows, index=IDX[: len(returns_rows)], columns=COLS)
    tw = pd.DataFrame(weights_rows, index=rm.index, columns=COLS)
    return run_simulation(rm, tw, initial_capital=100.0, tax_rate=0.0, **kw)


def test_zero_weight_row_after_inception_holds_positions():
    # Month 3 emits an all-zero row (e.g. <5 valid names that period) — the
    # engine must hold current positions, not liquidate to nothing (which
    # zeroed NAV and faked a margin call).
    tw_rows = [[1.0, 0.0], [1.0, 0.0], [0.0, 0.0], [1.0, 0.0], [1.0, 0.0]]
    rets = [[0.0, 0.0], [0.10, 0.0], [0.10, 0.0], [0.10, 0.0], [0.10, 0.0]]
    res = _run(tw_rows, rets)
    vals = res["aftertax_values"]
    assert len(vals) == 5, "must not truncate with a bogus margin call"
    assert np.allclose(vals.values, 100.0 * 1.1 ** np.arange(5))
    assert res["turnover"].iloc[2] == 0.0


def test_underinvested_row_warns():
    # A live row summing < 1 must warn (its uninvested weight is dropped from NAV).
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run([[0.5, 0.0], [0.5, 0.0]], [[0.0, 0.0], [0.0, 0.0]])
    assert "summing < 1" in buf.getvalue()


def test_fully_invested_row_does_not_warn():
    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        _run([[0.6, 0.4], [0.6, 0.4]], [[0.0, 0.0], [0.0, 0.0]])
    assert "summing < 1" not in buf.getvalue()


def test_initial_deployment_charged_slippage():
    res = _run([[1.0, 0.0], [1.0, 0.0]], [[0.0, 0.0], [0.0, 0.0]], slippage=0.01)
    assert res["aftertax_values"].iloc[0] == pytest.approx(99.0)
    assert res["pretax_values"].iloc[0] == pytest.approx(99.0)


def test_excess_drag_deducted_from_nav():
    # Sell 10, buy 10, slippage 60% -> drag 12 exceeds buy liquidity 10.
    # Buys scale to zero and the residual 2 must come out of NAV (previously
    # it silently vanished).
    positions = {"A": {"cost_basis": 100.0, "current_value": 100.0}}
    tw = pd.Series({"A": 0.9, "B": 0.1})
    _execute_rebalance(
        positions, tw, 100.0, slippage=0.6, tax_rate=0.0, loss_carryforward=0.0
    )
    nav = sum(p["current_value"] for p in positions.values())
    assert nav == pytest.approx(90.0 - 2.0)


def test_apply_returns_allows_large_genuine_gains():
    pos = {"A": {"cost_basis": 10.0, "current_value": 10.0}}
    _apply_returns(pos, pd.Series({"A": 2.0}))  # +200% must not be clipped
    assert pos["A"]["current_value"] == pytest.approx(30.0)
