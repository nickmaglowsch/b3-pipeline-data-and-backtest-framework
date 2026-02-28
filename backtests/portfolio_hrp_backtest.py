"""
Hierarchical Risk Parity (HRP) Portfolio Backtest
====================================================
Implements Lopez de Prado's HRP algorithm over the 8 core strategy return
streams, using hierarchical clustering to group correlated strategies and
allocate weights top-down through the dendrogram.

Compares HRP against: equal-weight, inverse-vol, ERC, IBOV, CDI.

Usage:
    python3 backtests/portfolio_hrp_backtest.py
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
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform

_BACKTESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKTESTS_DIR)
for _p in [_PROJECT_ROOT, _BACKTESTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.strategy_returns import build_strategy_returns, SMALLCAP_MOM_NOTE
from core.portfolio_opt import (
    inverse_vol_weights,
    equal_risk_contribution_weights,
    hrp_weights,
    compute_portfolio_returns,
)
from core.metrics import build_metrics, cumret
from core.plotting import PALETTE, fmt_ax

# ─── Configuration ────────────────────────────────────────────────────────────
LOOKBACK = 36
OUT_DIR = _BACKTESTS_DIR


def _metrics_row(name: str, ret: pd.Series) -> dict:
    return build_metrics(ret.dropna(), name, 12)


def main():
    print("\n" + "=" * 70)
    print("  HRP Portfolio Backtest")
    print("=" * 70)

    print("\nLoading strategy returns...")
    returns_df, sim_results, regime_signals = build_strategy_returns()

    all_strats = [c for c in returns_df.columns if c not in ("IBOV", "CDI")]
    liquid_strats = [c for c in all_strats if c != "SmallcapMom"]
    print(f"All strategies: {all_strats}")
    print(f"Liquid-only:    {liquid_strats}")
    print(f"\nNote: {SMALLCAP_MOM_NOTE}")

    # Build portfolio returns for all methods
    print("\nComputing portfolios...")
    ew_all = returns_df[all_strats].fillna(0.0).mean(axis=1).rename("EqualWeight (all)")
    ew_liq = returns_df[liquid_strats].fillna(0.0).mean(axis=1).rename("EqualWeight (liq)")

    iv_liq, _ = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: inverse_vol_weights(w, LOOKBACK),
    )
    erc_liq, _ = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: equal_risk_contribution_weights(w, LOOKBACK),
    )
    hrp_all, _ = compute_portfolio_returns(
        returns_df, all_strats,
        lambda w: hrp_weights(w, LOOKBACK),
    )
    hrp_liq, _ = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: hrp_weights(w, LOOKBACK),
    )

    ibov_ret = returns_df["IBOV"].dropna()
    cdi_ret = returns_df["CDI"].dropna()

    # Metrics
    print("\nComputing metrics...")
    metrics_list = [
        _metrics_row("EqualWeight (all)",    ew_all),
        _metrics_row("EqualWeight (liquid)", ew_liq),
        _metrics_row("InvVol (liquid)",      iv_liq),
        _metrics_row("ERC (liquid)",         erc_liq),
        _metrics_row("HRP (all)",            hrp_all),
        _metrics_row("HRP (liquid)",         hrp_liq),
        _metrics_row("IBOV",                 ibov_ret),
        _metrics_row("CDI",                  cdi_ret),
    ]
    metrics_list.sort(key=lambda m: float(m.get("Sharpe", 0)), reverse=True)

    print(f"\n{'='*90}")
    print(f"  HRP COMPARISON -- After-Strategy-Level Tax (15% CGT)")
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

    # ── Final HRP weights ─────────────────────────────────────────────────────
    window = returns_df[liquid_strats].iloc[-LOOKBACK:]
    final_hrp_liq = hrp_weights(window, LOOKBACK)
    print("Final HRP weights (liquid):")
    for s, v in final_hrp_liq.sort_values(ascending=False).items():
        print(f"  {s:<22s}: {v*100:.1f}%")

    window_all = returns_df[all_strats].iloc[-LOOKBACK:]
    final_hrp_all = hrp_weights(window_all, LOOKBACK)
    print("\nFinal HRP weights (all strategies):")
    for s, v in final_hrp_all.sort_values(ascending=False).items():
        print(f"  {s:<22s}: {v*100:.1f}%")

    # Show clustering order
    corr_m = window.corr().values
    np.fill_diagonal(corr_m, 1.0)
    dist_m = np.sqrt(np.maximum(0.5 * (1.0 - corr_m), 0.0))
    np.fill_diagonal(dist_m, 0.0)
    condensed = squareform(dist_m, checks=False)
    lm = linkage(condensed, method="ward")
    order_idx = leaves_list(lm)
    print(f"\nDendrogram leaf order (liquid): {[liquid_strats[i] for i in order_idx]}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    series_dict = {
        "EqualWeight (all)":  ew_all,
        "EqualWeight (liq)":  ew_liq,
        "InvVol (liq)":       iv_liq,
        "ERC (liq)":          erc_liq,
        "HRP (all)":          hrp_all,
        "HRP (liq)":          hrp_liq,
        "IBOV":               ibov_ret,
        "CDI":                cdi_ret,
    }
    _plot_equity_curves(series_dict, os.path.join(OUT_DIR, "portfolio_hrp.png"))
    _plot_dendrogram(returns_df, liquid_strats, os.path.join(OUT_DIR, "portfolio_hrp_dendrogram.png"))

    print("\nDone.")


def _plot_equity_curves(series_dict: dict, out_path: str):
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
    fig.patch.set_facecolor(PALETTE["bg"])

    for idx, (name, ret) in enumerate(series_dict.items()):
        r = ret.dropna()
        if len(r) == 0:
            continue
        curve = cumret(r)
        ax.plot(
            curve.index, curve.values,
            label=name,
            color=colors[idx % len(colors)],
            ls=linestyles[idx % len(linestyles)],
            lw=2.2 if "HRP" in name else 1.5,
        )

    ax.set_title("HRP Portfolio Comparison", color=PALETTE["text"],
                 fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of R$1", color=PALETTE["sub"])
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=8.5, ncol=2, loc="upper left")
    fmt_ax(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Equity curves saved -> {out_path}")


def _plot_dendrogram(returns_df: pd.DataFrame, strategy_cols: list, out_path: str):
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    window = returns_df[strategy_cols].iloc[-LOOKBACK:]
    corr_m = window.corr().values
    np.fill_diagonal(corr_m, 1.0)
    dist_m = np.sqrt(np.maximum(0.5 * (1.0 - corr_m), 0.0))
    np.fill_diagonal(dist_m, 0.0)
    condensed = squareform(dist_m, checks=False)
    lm = linkage(condensed, method="ward")

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor(PALETTE["panel"])
    fig.patch.set_facecolor(PALETTE["bg"])

    dend = dendrogram(
        lm,
        labels=strategy_cols,
        ax=ax,
        leaf_rotation=30,
        color_threshold=0.3 * max(lm[:, 2]),
        above_threshold_color=PALETTE["sub"],
    )

    ax.set_title(
        "HRP Strategy Dendrogram (Ward linkage, 36-month trailing)",
        color=PALETTE["text"], fontsize=11, fontweight="bold",
    )
    ax.set_ylabel("Distance", color=PALETTE["sub"])
    ax.tick_params(colors=PALETTE["sub"])
    ax.spines[:].set_color(PALETTE["grid"])
    ax.set_facecolor(PALETTE["panel"])
    ax.yaxis.label.set_color(PALETTE["sub"])
    for label in ax.get_xticklabels():
        label.set_color(PALETTE["text"])
    for label in ax.get_yticklabels():
        label.set_color(PALETTE["sub"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Dendrogram saved -> {out_path}")


if __name__ == "__main__":
    main()
