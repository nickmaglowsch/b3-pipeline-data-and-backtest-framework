"""
Risk Parity Portfolio Backtest
================================
Implements two risk parity variants over the 8 core strategy return streams:
  1. Naive Risk Parity (Inverse Volatility)
  2. Full Risk Parity (Equal Risk Contribution / ERC)

Compares against equal-weight portfolio, IBOV, and CDI.

Usage:
    python3 backtests/portfolio_risk_parity_backtest.py
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_BACKTESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKTESTS_DIR)
for _p in [_PROJECT_ROOT, _BACKTESTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.strategy_returns import build_strategy_returns, SMALLCAP_MOM_NOTE
from core.portfolio_opt import inverse_vol_weights, equal_risk_contribution_weights, compute_portfolio_returns
from core.metrics import build_metrics, cumret, ann_return, ann_vol, sharpe, max_dd, calmar
from core.plotting import PALETTE, fmt_ax

# ─── Configuration ───────────────────────────────────────────────────────────
LOOKBACK = 36          # rolling window for vol/cov estimation (months)
CAPITAL = 100_000      # starting capital (BRL)
OUT_PATH = os.path.join(_BACKTESTS_DIR, "portfolio_risk_parity.png")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def compute_eq_weight_returns(
    returns_df: pd.DataFrame,
    strategy_cols: list,
) -> pd.Series:
    """Simple equal-weight portfolio (no lookback needed)."""
    sub_df = returns_df[strategy_cols].fillna(0.0)
    return sub_df.mean(axis=1).rename("EqualWeight")


def get_final_weights(
    returns_df: pd.DataFrame,
    strategy_cols: list,
    weight_fn,
    lookback: int = LOOKBACK,
) -> pd.Series:
    """Get the most recent weight allocation."""
    if len(returns_df) >= lookback:
        window = returns_df[strategy_cols].iloc[-lookback:]
        return weight_fn(window)
    return pd.Series(1.0 / len(strategy_cols), index=strategy_cols)


def _metrics_row(name: str, ret_series: pd.Series) -> dict:
    m = build_metrics(ret_series.dropna(), name, 12)
    return m


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  Risk Parity Portfolio Backtest")
    print("=" * 70)

    # 1. Load strategy returns
    print("\nLoading strategy returns...")
    returns_df, sim_results, regime_signals = build_strategy_returns()

    # 2. Strategy universe: all 8 strategies
    all_strats = [c for c in returns_df.columns if c not in ("IBOV", "CDI")]
    liquid_strats = [c for c in all_strats if c != "SmallcapMom"]

    print(f"All strategies ({len(all_strats)}): {all_strats}")
    print(f"Liquid-only    ({len(liquid_strats)}): {liquid_strats}")
    print(f"\nNote: {SMALLCAP_MOM_NOTE}")

    # 3. Build portfolios
    print("\nComputing portfolios...")

    # Equal weight (both universes)
    ew_all = compute_eq_weight_returns(returns_df, all_strats)
    ew_liq = compute_eq_weight_returns(returns_df, liquid_strats)

    # Inverse vol
    iv_all, _ = compute_portfolio_returns(
        returns_df, all_strats,
        lambda w: inverse_vol_weights(w, LOOKBACK)
    )
    iv_liq, _ = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: inverse_vol_weights(w, LOOKBACK)
    )

    # ERC
    erc_all, _ = compute_portfolio_returns(
        returns_df, all_strats,
        lambda w: equal_risk_contribution_weights(w, LOOKBACK)
    )
    erc_liq, _ = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: equal_risk_contribution_weights(w, LOOKBACK)
    )

    ibov_ret = returns_df["IBOV"].dropna()
    cdi_ret = returns_df["CDI"].dropna()

    # 4. Metrics comparison
    print("\nComputing metrics...")
    metrics_list = [
        _metrics_row("EqualWeight (all)",     ew_all),
        _metrics_row("EqualWeight (liquid)",  ew_liq),
        _metrics_row("InvVol (all)",           iv_all),
        _metrics_row("InvVol (liquid)",        iv_liq),
        _metrics_row("ERC (all)",              erc_all),
        _metrics_row("ERC (liquid)",           erc_liq),
        _metrics_row("IBOV",                   ibov_ret),
        _metrics_row("CDI",                    cdi_ret),
    ]
    metrics_list.sort(key=lambda m: float(m.get("Sharpe", 0)), reverse=True)

    # Print metrics table
    print(f"\n{'='*90}")
    print(f"  RISK PARITY COMPARISON -- After-Strategy-Level Tax (15% CGT)")
    print(f"{'='*90}")
    hdr = f"  {'Method':<25s} {'Ann.Ret%':>8s} {'Ann.Vol%':>8s} {'Sharpe':>8s} {'MaxDD%':>8s} {'Calmar':>8s}"
    print(hdr)
    print(f"  {'-'*68}")
    for m in metrics_list:
        name = str(m.get("Strategy", "?"))[:25]
        print(
            f"  {name:<25s} "
            f"{str(m.get('Ann. Return (%)', '?')):>8s} "
            f"{str(m.get('Ann. Volatility (%)', '?')):>8s} "
            f"{str(m.get('Sharpe', '?')):>8s} "
            f"{str(m.get('Max Drawdown (%)', '?')):>8s} "
            f"{str(m.get('Calmar', '?')):>8s}"
        )
    print(f"{'='*90}\n")

    # 5. Final weight allocations
    print("--- Final weight allocations ---")
    for name, strats, w_fn in [
        ("InvVol (all)",    all_strats,     lambda w: inverse_vol_weights(w, LOOKBACK)),
        ("InvVol (liquid)", liquid_strats,  lambda w: inverse_vol_weights(w, LOOKBACK)),
        ("ERC (all)",       all_strats,     lambda w: equal_risk_contribution_weights(w, LOOKBACK)),
        ("ERC (liquid)",    liquid_strats,  lambda w: equal_risk_contribution_weights(w, LOOKBACK)),
    ]:
        fw = get_final_weights(returns_df, strats, w_fn)
        print(f"\n{name}:")
        for s, v in fw.sort_values(ascending=False).items():
            print(f"  {s:<20s}: {v*100:.1f}%")

    # 6. Plot equity curves
    print("\nPlotting equity curves...")
    _plot_equity_curves(
        {
            "EqualWeight (all)":  ew_all,
            "EqualWeight (liq)":  ew_liq,
            "InvVol (all)":       iv_all,
            "InvVol (liq)":       iv_liq,
            "ERC (all)":          erc_all,
            "ERC (liq)":          erc_liq,
            "IBOV":               ibov_ret,
            "CDI":                cdi_ret,
        },
        out_path=OUT_PATH,
    )

    print("\nDone.")


def _plot_equity_curves(series_dict: dict, out_path: str):
    """Plot cumulative return equity curves for all methods."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    colors = [
        "#00D4AA", "#7B61FF", "#FF6B35", "#FFC947",
        "#4FC3F7", "#F48FB1", "#A8B2C1", "#81C784",
    ]
    linestyles = ["-", "-", "-.", "-.", "--", "--", ":", ":"]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor(PALETTE["panel"])

    for idx, (name, ret) in enumerate(series_dict.items()):
        ret_clean = ret.dropna()
        if len(ret_clean) == 0:
            continue
        curve = cumret(ret_clean)
        ax.plot(
            curve.index,
            curve.values,
            label=name,
            color=colors[idx % len(colors)],
            ls=linestyles[idx % len(linestyles)],
            lw=2.0 if idx < 6 else 1.4,
        )

    ax.set_title("Risk Parity Portfolio Comparison", color=PALETTE["text"],
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of R$1", color=PALETTE["sub"])
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=8.5, ncol=2, loc="upper left")
    fmt_ax(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Plot saved -> {out_path}")


if __name__ == "__main__":
    main()
