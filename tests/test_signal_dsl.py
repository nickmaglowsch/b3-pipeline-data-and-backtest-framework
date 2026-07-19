"""
Tests for the signal expression interpreter (backtests/core/signal_dsl.py):
correctness of the primitives and — critically — that the safe evaluator rejects
anything outside the whitelist (no code execution from a spec).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import pandas.testing as pdt

from backtests.core import signal_dsl


def _frames(seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-31", periods=40, freq="ME")
    cols = [f"T{i}" for i in range(8)]
    ret = pd.DataFrame(rng.normal(0.01, 0.06, (40, 8)), index=idx, columns=cols)
    log_ret = np.log1p(ret)
    glitch = pd.DataFrame(0, index=idx, columns=cols)
    glitch.iloc[5, 2] = 1
    return {"ret": ret, "log_ret": log_ret, "has_glitch": glitch, "lookback": 6}


# ── correctness: expressions reproduce the equivalent pandas ops ──────────────

def test_momentum_matches_pandas():
    ns = _frames()
    got = signal_dsl.evaluate("roll_sum(shift(log_ret, 1), lookback)", ns)
    want = ns["log_ret"].shift(1).rolling(6).sum()
    pdt.assert_frame_equal(got, want)


def test_sharpe_matches_pandas():
    ns = _frames()
    got = signal_dsl.evaluate(
        "roll_sum(shift(log_ret,1), lookback) / roll_std(shift(ret,1), lookback)", ns)
    want = ns["log_ret"].shift(1).rolling(6).sum() / ns["ret"].shift(1).rolling(6).std()
    pdt.assert_frame_equal(got, want)


def test_rank_is_cross_sectional_pct():
    ns = _frames()
    got = signal_dsl.evaluate("rank(ret)", ns)
    pdt.assert_frame_equal(got, ns["ret"].rank(axis=1, pct=True))


def test_unary_minus_and_scalar_arithmetic():
    ns = _frames()
    got = signal_dsl.evaluate("-(0.5 * ret + 0.5 * log_ret)", ns)
    want = -(0.5 * ns["ret"] + 0.5 * ns["log_ret"])
    pdt.assert_frame_equal(got, want)


def test_mask_glitch_nans_flagged_cells():
    ns = _frames()
    got = signal_dsl.evaluate("mask_glitch(ret)", ns)
    assert np.isnan(got.iloc[5, 2])                       # the flagged cell
    assert not np.isnan(got.iloc[6, 2])                   # an unflagged one


# ── safety: everything outside the whitelist is rejected ──────────────────────

@pytest.mark.parametrize("expr", [
    "ret.values",                 # attribute access
    "ret['T0']",                  # subscript
    "__import__('os')",           # unknown name (would-be import)
    "unknown_frame + 1",          # unknown name
    "nope(ret)",                  # unknown function
    "roll_std(ret, w=3)",         # keyword argument
    "(lambda x: x)(ret)",         # lambda
    "'a string'",                 # non-numeric constant
])
def test_rejects_disallowed(expr):
    with pytest.raises(ValueError):
        signal_dsl.evaluate(expr, _frames())


if __name__ == "__main__":
    test_momentum_matches_pandas()
    test_sharpe_matches_pandas()
    test_mask_glitch_nans_flagged_cells()
    print("OK: signal DSL correctness + safety")
