"""
Validate QDL vs the three known-edge B3 strategies — with overfit checks.
========================================================================
Runs QDL, TevaAtivosReais, BESST Quality and a DIVO11 buy-hold through the
same tax+slippage engine and reports:
  1. Head-to-head metrics on the common overlapping window.
  2. Sub-period stability (a strategy that only wins in one regime is overfit).
  3. QDL parameter sensitivity — one knob swept at a time around its a-priori
     value. A flat response = robust; a sharp spike at the default = overfit.

Run from the repo root:  python -m backtests.validate_qdl
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd

from backtests.core.shared_data import build_shared_data
from backtests.core.simulation import run_simulation
from backtests.core.strategy_registry import get_registry
from backtests.core.data import download_benchmark
from backtests.core.metrics import (
    ann_return, ann_vol, sharpe, max_dd, calmar, value_to_ret,
)

DB = "b3_market_data.sqlite"
START, FREQ = "2006-01-01", "QE"
PPY = {"ME": 12, "QE": 4, "W-FRI": 52}[FREQ]
END = datetime.today().strftime("%Y-%m-%d")


def metrics_row(name: str, values: pd.Series) -> dict:
    """Full metric set from an equity-value curve."""
    r = value_to_ret(values)
    return {
        "name": name,
        "ret": round(ann_return(r, PPY) * 100, 2),
        "vol": round(ann_vol(r, PPY) * 100, 2),
        "sharpe": round(sharpe(r, periods_per_year=PPY), 2),
        "mdd": round(max_dd(r) * 100, 2),
        "calmar": round(calmar(r, PPY), 2),
    }


def print_table(title: str, rows: list[dict]) -> None:
    print(f"\n{title}")
    print(f"{'Strategy':<18}{'Ret%':>8}{'Vol%':>8}{'Sharpe':>8}{'MaxDD%':>9}{'Calmar':>8}")
    print("-" * 59)
    for m in rows:
        print(f"{m['name']:<18}{m['ret']:>8}{m['vol']:>8}{m['sharpe']:>8}"
              f"{m['mdd']:>9}{m['calmar']:>8}")


def run_strategy(strat, shared, params) -> pd.Series:
    """Run one strategy -> after-tax equity curve (Series on rebalance dates)."""
    ret_matrix, tw = strat.generate_signals(shared, params)
    res = run_simulation(
        returns_matrix=ret_matrix.fillna(0.0),
        target_weights=tw,
        initial_capital=100_000.0,
        tax_rate=0.15,
        slippage=0.001,
        monthly_sales_exemption=20_000.0,
        name=strat.name,
    )
    return res["aftertax_values"]


def main() -> None:
    print(f"Building shared data {START}..{END} ({FREQ}, +fundamentals)...")
    shared = build_shared_data(DB, START, END, FREQ, include_fundamentals=True)

    reg = get_registry()
    base = {"start_date": START, "end_date": END, "rebalance_freq": FREQ,
            "initial_capital": 100_000.0, "tax_rate": 0.15, "slippage": 0.001,
            "monthly_sales_exemption": 20_000.0}

    curves: dict[str, pd.Series] = {}
    for nm in ["QDL", "QDL-Equity", "TevaAtivosReais", "BESST Quality"]:
        strat = reg.get(nm)
        p = dict(base)
        p.update(strat.get_default_parameters())
        p.update(base)  # keep our common dates/costs
        curves[nm] = run_strategy(strat, shared, p)

    # DIVO11 buy-hold ETF (Yahoo total-return via auto_adjust); no tax/turnover.
    divo_px = download_benchmark("DIVO11.SA", START, END).resample(FREQ).last().dropna()
    curves["DIVO11"] = divo_px / divo_px.iloc[0] * 100_000.0

    # IBOV reference.
    ibov = (1 + shared["ibov_ret"]).cumprod() * 100_000.0
    curves["IBOV"] = ibov

    # ── Each strategy on its OWN full available window ────────────────────────
    print_table("Own full window (each since first live signal):",
                [metrics_row(nm, c.dropna()) for nm, c in curves.items()])

    # ── Common overlapping window (strict apples-to-apples) ───────────────────
    common = None
    for nm in ["QDL", "TevaAtivosReais", "BESST Quality", "DIVO11"]:
        idx = curves[nm].dropna()
        idx = idx[idx > 0].index
        common = idx if common is None else common.intersection(idx)
    common = common.sort_values()
    c0, c1 = common[0], common[-1]
    print(f"\nCommon window: {c0.date()} .. {c1.date()}  ({len(common)} periods)")

    def on(nm, idx):  # slice + rebase to 100k at the window start
        s = curves[nm].reindex(idx).dropna()
        return s / s.iloc[0] * 100_000.0

    order = ["QDL", "QDL-Equity", "TevaAtivosReais", "BESST Quality", "DIVO11", "IBOV"]
    print_table("Common window:", [metrics_row(nm, on(nm, common)) for nm in order])

    # ── Sub-period stability ──────────────────────────────────────────────────
    subs = [("2011-2015", "2011-01-01", "2015-12-31"),
            ("2015-2020", "2015-01-01", "2019-12-31"),
            ("2020-now",   "2020-01-01", END)]
    for label, a, b in subs:
        idx = common[(common >= a) & (common <= b)]
        if len(idx) < 4:
            continue
        print_table(f"Sub-period {label}:",
                    [metrics_row(nm, on(nm, idx)) for nm in order])

    # ── QDL parameter sensitivity (one knob at a time; center = a-priori) ─────
    qdl = reg.get("QDL")
    sweeps = {
        "top_n": [10, 15, 20, 25, 30],
        "target_vol": [0.08, 0.10, 0.12, 0.15, 0.20],
        "min_div_years": [1, 2, 3],
        "ewma_n": [126, 252, 378],
    }
    print("\n\n=== QDL parameter sensitivity (common window) ===")
    print("(default: top_n=20, target_vol=0.12, min_div_years=2, ewma_n=252)")
    for knob, vals in sweeps.items():
        rows = []
        for v in vals:
            p = dict(base)
            p.update(qdl.get_default_parameters())
            p.update(base)
            p[knob] = v
            cv = run_strategy(qdl, shared, p)
            rows.append(metrics_row(f"{knob}={v}", on_series(cv, common)))
        print_table(f"sweep {knob}:", rows)


def on_series(s: pd.Series, idx) -> pd.Series:
    s = s.reindex(idx).dropna()
    return s / s.iloc[0] * 100_000.0 if len(s) else s


if __name__ == "__main__":
    main()
