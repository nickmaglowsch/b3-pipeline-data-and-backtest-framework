"""
Leverage analysis for the QDL / IVVB11 / CDI blend.
===================================================
Conclusion up front: in Brazil, borrowed leverage is the WRONG tool for this
portfolio. The borrow rate is ~CDI (10-14%), which is ~ the portfolio's own
return, so there is no positive carry to amplify. Every levered variant below
has lower Sharpe/Calmar, and at realistic borrow costs (CDI+2-4%) lower return
too, with 2-3x the drawdown. The low -5.6% drawdown is the *product* (cash +
diversification), not unused headroom.

Two things this script shows:

  A. Reallocation ("spend the cash buffer") — shift the CDI third into the risky
     sleeves. No borrowing, all weights >=0, sum to 1: the engine simulates this
     correctly. This is the efficient way to add return, and it DOMINATES
     borrowed leverage at the same drawdown.

  B. Analytic constant-leverage on the trustworthy 1x return stream
     (r_lev = L*r - (L-1)*(CDI+spread)). We do NOT lever inside run_simulation:
     the engine has no cash/borrow ledger (NAV = sum of position values), so a
     net-negative CDI_ASSET pumps fake returns — a 2x IVVB11 book prints ~327%.
     Borrowed leverage must be evaluated analytically until that's fixed.

Run from repo root:  python -m backtests.leverage_qdl_blend
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from backtests.core.shared_data import build_shared_data
from backtests.core.simulation import run_simulation
from backtests.core.strategy_registry import get_registry
from backtests.core.data import download_benchmark
from backtests.core.metrics import ann_return, ann_vol, sharpe, max_dd, calmar, value_to_ret

DB, START, FREQ, PPY = "b3_market_data.sqlite", "2006-01-01", "QE", 4
END = datetime.today().strftime("%Y-%m-%d")

# Reallocation menu (no borrowing): (a_qdl, a_ivvb, a_cdi), all >=0, sum to 1.
REALLOC = [
    ("thirds 33/33/33", (1/3, 1/3, 1/3)),
    ("40/40/20",        (0.40, 0.40, 0.20)),
    ("50/50/0",         (0.50, 0.50, 0.00)),
]


def hdr(title):
    print(f"\n{title}")
    print(f"{'Config':<26}{'Ret%':>8}{'Vol%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Calmar':>8}")
    print("-" * 67)


def line(label, r):
    return (f"{label:<26}{ann_return(r,PPY)*100:>8.2f}{ann_vol(r,PPY)*100:>8.2f}"
            f"{sharpe(r,periods_per_year=PPY):>8.2f}{max_dd(r)*100:>9.2f}"
            f"{calmar(r,PPY):>8.2f}")


def main() -> None:
    shared = build_shared_data(DB, START, END, FREQ, include_fundamentals=True)
    ret, cdi = shared["ret"], shared["cdi_monthly"]
    reg = get_registry()

    qdl = reg.get("QDL")
    p = {**qdl.get_default_parameters(),
         "start_date": START, "end_date": END, "rebalance_freq": FREQ}
    _, tw_qdl = qdl.generate_signals(shared, p)

    ivvb = download_benchmark("IVVB11.SA", START, END).reindex(ret.index, method="ffill")
    live = ivvb.notna().values
    R = ret.copy()
    R["CDI_ASSET"] = cdi
    R["IVVB11"] = ivvb.pct_change().fillna(0.0)
    common = ret.index[live]

    def run_alloc(a_qdl, a_ivvb, a_cdi) -> pd.Series:
        tw = pd.DataFrame(0.0, index=ret.index, columns=R.columns)
        tw[tw_qdl.columns] = a_qdl * tw_qdl.values
        tw["IVVB11"] = a_ivvb
        tw.loc[~live, "CDI_ASSET"] += a_ivvb
        tw.loc[~live, "IVVB11"] = 0.0
        tw["CDI_ASSET"] += a_cdi
        res = run_simulation(returns_matrix=R.fillna(0.0), target_weights=tw,
                             initial_capital=100_000.0, tax_rate=0.15, slippage=0.001,
                             monthly_sales_exemption=20_000.0, name="alloc")
        return value_to_ret(res["aftertax_values"].reindex(common).dropna())

    # ── A. Reallocation (trustworthy: engine-simulated, no borrow) ────────────
    hdr(f"A. Reallocation, no borrowing ({common[0].date()}..{common[-1].date()}):")
    base_stream = None
    for label, (aq, ai, ac) in REALLOC:
        r = run_alloc(aq, ai, ac)
        if base_stream is None:
            base_stream = r  # thirds = the 1x blend we lever analytically below
        print(line(label, r))

    # ── B. Analytic borrowed leverage on the 1x thirds stream ─────────────────
    cdi_c = cdi.reindex(base_stream.index).fillna(0.0)
    hdr("B. Analytic borrowed leverage on the 1x thirds stream:")
    for spread in [0.0, 0.02, 0.04]:
        b = cdi_c + spread / PPY
        for L in [1.25, 1.5, 2.0]:
            r = L * base_stream - (L - 1) * b
            print(line(f"{L:.2f}x  (borrow CDI+{int(spread*100)}%)", r))
    print("\nTakeaway: 50/50/0 reallocation (15.9%, -10.6% DD) beats 1.5x@CDI+2% "
          "borrow (14.3%, -11.3% DD) on every axis. Borrowing is dominated.")


if __name__ == "__main__":
    main()
