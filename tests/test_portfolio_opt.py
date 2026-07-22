"""Weight builders must survive pandas copy-on-write.

Under pandas 3.x, ``DataFrame.cov().values`` / ``.corr().values`` hand back a
READ-ONLY view, so the in-place ``cov += eye*eps`` and ``np.fill_diagonal(corr, 1)``
these functions do raise ``ValueError: output array is read-only`` — which took
out every portfolio_*.py runner. These pin the contract on real inputs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtests.core.portfolio_opt import (
    equal_risk_contribution_weights,
    hrp_weights,
    inverse_vol_weights,
)

BUILDERS = [inverse_vol_weights, equal_risk_contribution_weights, hrp_weights]


def _returns(n_months: int = 48, n_strats: int = 6, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-31", periods=n_months, freq="ME")
    cols = [f"S{i}" for i in range(n_strats)]
    # correlated blocks so HRP's clustering has something real to bite on
    base = rng.normal(0.01, 0.04, (n_months, 1))
    data = base + rng.normal(0.0, 0.03, (n_months, n_strats))
    return pd.DataFrame(data, index=idx, columns=cols)


@pytest.mark.parametrize("weight_fn", BUILDERS, ids=lambda f: f.__name__)
def test_weights_are_valid_simplex(weight_fn):
    df = _returns()
    w = weight_fn(df, 36)

    assert list(w.index) == list(df.columns)
    assert w.notna().all(), f"{weight_fn.__name__} produced NaN weights"
    assert (w >= 0).all(), f"{weight_fn.__name__} produced negative weights"
    assert w.sum() == pytest.approx(1.0)


@pytest.mark.parametrize("weight_fn", BUILDERS, ids=lambda f: f.__name__)
def test_read_only_covariance_does_not_raise(weight_fn):
    """The pandas-3 regression: .cov()/.corr() views are not writeable."""
    df = _returns()
    assert not df.cov().values.flags.writeable, "fixture no longer reproduces the hazard"

    weight_fn(df, 36)  # must not raise


@pytest.mark.parametrize("weight_fn", BUILDERS, ids=lambda f: f.__name__)
def test_short_history_falls_back_to_equal_weight(weight_fn):
    # 1 observation: too few for a std/cov under every builder's own threshold
    df = _returns(n_months=1)
    w = weight_fn(df, 36)

    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w.to_numpy(), 1.0 / len(df.columns))


def test_erc_equalises_risk_contributions():
    df = _returns(n_strats=6, seed=11)
    cov = df.tail(36).cov().to_numpy()

    def spread(w):
        """Max-minus-min risk contribution, as a fraction of the mean."""
        rc = w * (cov @ w) / np.sqrt(w @ cov @ w)
        return (rc.max() - rc.min()) / rc.mean()

    erc = equal_risk_contribution_weights(df, 36).to_numpy()
    equal = np.full(len(df.columns), 1.0 / len(df.columns))

    # the solver lands near-but-not-exactly equal (SLSQP ftol), so assert both a
    # loose absolute bound AND that it genuinely beats naive equal-weighting
    assert spread(erc) < 0.05
    assert spread(erc) < spread(equal)
