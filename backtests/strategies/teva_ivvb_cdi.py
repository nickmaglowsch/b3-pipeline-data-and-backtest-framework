"""
Strategy: TevaAtivosReais / IVVB11 / CDI — 1/3 each, quarterly rebalance
=======================================================================
Three-sleeve portfolio:

  1/3  TevaAtivosReais  (Utilities dividend replica — its own multi-stock book)
  1/3  IVVB11           (S&P 500 in BRL — the B3 ETF, pulled from Yahoo)
  1/3  CDI              (Brazilian cash rate)

The Teva sleeve delegates to TevaAtivosReaisStrategy for stock selection and
weighting, then that book is scaled to 1/3. CDI is the residual, so any weight
the equity sleeves can't place (empty Teva universe early on, or before IVVB11
lists ~2014) parks in cash rather than going dead-flat.
"""
from __future__ import annotations

import pandas as pd

from backtests.core.data import download_benchmark
from backtests.core.strategy_base import (
    StrategyBase,
    ParameterSpec,
    COMMON_START_DATE,
    COMMON_END_DATE,
    COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE,
    COMMON_SLIPPAGE,
    COMMON_MONTHLY_SALES_EXEMPTION,
)
from backtests.strategies.teva_ativos_reais import TevaAtivosReaisStrategy

IVVB11_TICKER = "IVVB11.SA"
W_EACH = 1.0 / 3.0


class TevaIvvbCdiStrategy(StrategyBase):
    """1/3 TevaAtivosReais / 1/3 IVVB11 / 1/3 CDI, rebalanced quarterly."""

    # Inherits the Teva sleeve's data needs.
    needs_fundamentals: bool = True

    def __init__(self) -> None:
        self._teva = TevaAtivosReaisStrategy()

    @property
    def name(self) -> str:
        return "Teva / IVVB11 / CDI (1/3 each)"

    @property
    def description(self) -> str:
        return (
            "Quarterly-rebalanced thirds: 1/3 TevaAtivosReais (Utilities dividend "
            "replica), 1/3 IVVB11 (S&P 500 in BRL), 1/3 CDI (cash). CDI absorbs "
            "any weight the equity sleeves can't place. IVVB11 from Yahoo."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        # Reuse the Teva sleeve's parameters (min cap, ADTV, eligible sectors,
        # issuer cap …) so the equity book stays configurable, plus the common
        # simulation knobs.
        teva_only = {
            spec.name: spec
            for spec in self._teva.get_parameter_specs()
            if spec.name not in {
                "start_date", "end_date", "initial_capital", "tax_rate",
                "slippage", "monthly_sales_exemption", "rebalance_freq",
            }
        }
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE,
            COMMON_SLIPPAGE,
            COMMON_MONTHLY_SALES_EXEMPTION,
            ParameterSpec(
                "rebalance_freq", "Rebalance Frequency", "choice", "QE",
                description="Rebalance back to thirds (quarterly by default)",
                choices=["ME", "QE", "W-FRI"],
            ),
            *teva_only.values(),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        cdi = shared_data["cdi_monthly"]

        # ── Teva equity sleeve (its own multi-stock target weights) ───────────
        teva_ret, teva_tw = self._teva.generate_signals(shared_data, params)

        start, end = str(ret.index[0].date()), str(ret.index[-1].date())
        ivvb_px = download_benchmark(IVVB11_TICKER, start, end).reindex(
            ret.index, method="ffill"
        )

        r = teva_ret.copy()
        r["CDI_ASSET"] = cdi
        r["IVVB11"] = ivvb_px.pct_change().fillna(0.0)

        tw = pd.DataFrame(0.0, index=r.index, columns=r.columns)
        # Teva book scaled to a third (rows sum to 1/3 when the sleeve is active,
        # 0 when it has no eligible names).
        tw[teva_tw.columns] = teva_tw.values * W_EACH
        # IVVB11 third, dropped to CDI before the ETF lists.
        tw["IVVB11"] = W_EACH
        tw.loc[ivvb_px.isna(), "IVVB11"] = 0.0
        # CDI is the residual → keeps the book fully invested regardless of which
        # equity sleeves are live.
        equity = tw.drop(columns=["CDI_ASSET"]).sum(axis=1)
        tw["CDI_ASSET"] = (1.0 - equity).clip(lower=0.0)

        return r, tw


if __name__ == "__main__":
    # ── self-check: weights sum to 1 and thirds are respected ─────────────────
    import numpy as np

    idx = pd.date_range("2020-03-31", periods=4, freq="QE")
    s = TevaIvvbCdiStrategy()
    # fake a shared_data with a two-stock Teva book on the middle rows
    ret = pd.DataFrame(0.0, index=idx, columns=["AAAA3", "BBBB3"])

    class _StubTeva:
        def generate_signals(self, sd, p):
            tw = pd.DataFrame(0.0, index=idx, columns=["AAAA3", "BBBB3"])
            tw.iloc[2] = [0.6, 0.4]  # active sleeve on one row
            return ret, tw

    s._teva = _StubTeva()
    # stub Yahoo download to avoid network in the self-check
    import backtests.strategies.teva_ivvb_cdi as mod
    mod.download_benchmark = lambda *a, **k: pd.Series(100.0, index=idx)

    shared = {"ret": ret, "cdi_monthly": pd.Series(0.01, index=idx)}
    r, tw = s.generate_signals(shared, {})
    row_sums = tw.sum(axis=1)
    assert np.allclose(row_sums, 1.0), row_sums
    # on the active row: equity = 1/3 (teva) + 1/3 (ivvb) = 2/3, CDI = 1/3
    active = tw.iloc[2]
    assert abs(active[["AAAA3", "BBBB3"]].sum() - W_EACH) < 1e-9
    assert abs(active["IVVB11"] - W_EACH) < 1e-9
    assert abs(active["CDI_ASSET"] - W_EACH) < 1e-9
    # inactive Teva row: CDI = 2/3 (Teva third parked in cash)
    assert abs(tw.iloc[0]["CDI_ASSET"] - 2 * W_EACH) < 1e-9
    print("teva_ivvb_cdi self-check OK")
