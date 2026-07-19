import numpy as np
import pandas as pd

from backtests.strategies.ibov_low_vol import ewma_annualized_vol, inverse_vol_capped


def test_inverse_vol_capped_respects_cap_and_sums_to_one():
    # 15 names (cap*n=1.5 >= 1, so feasible). One tiny-vol name blows past the
    # cap and must be clipped to 10%.
    vol = pd.Series({c: v for c, v in zip("ABCDEFGHIJKLMNO",
                     [0.01] + list(np.linspace(0.20, 0.50, 14)))})
    w = inverse_vol_capped(vol, cap=0.10)
    assert np.isclose(w.sum(), 1.0)
    assert (w <= 0.10 + 1e-9).all()
    assert np.isclose(w["A"], 0.10)  # lowest vol pinned at the cap


def test_ewma_vol_positive_and_annualized():
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    px = pd.DataFrame({"X": 100 * np.exp(np.cumsum(rng.normal(0, 0.02, 300)))}, index=idx)
    vol = ewma_annualized_vol(px, n=252)
    v = vol["X"].dropna()
    assert len(v) > 0
    assert (v > 0).all()
    # ~2% daily -> ~32% annualized, allow wide band
    assert 0.1 < v.iloc[-1] < 0.6


if __name__ == "__main__":
    test_inverse_vol_capped_respects_cap_and_sums_to_one()
    test_ewma_vol_positive_and_annualized()
    print("ok")
