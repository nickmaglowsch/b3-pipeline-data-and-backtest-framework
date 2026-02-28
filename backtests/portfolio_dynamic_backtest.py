"""
Dynamic Allocation Portfolio Backtest
========================================
Implements three dynamic allocation approaches over the 8 core strategy return streams:
  1. Rolling Sharpe Momentum    -- weight strategies by trailing Sharpe ratio
  2. Regime-Conditional         -- pre-defined equity/CDI mixes per COPOM x IBOV regime
  3. Combined                   -- regime sets equity/CDI budget; rolling Sharpe within equity

Compares against: equal-weight, inverse-vol, ERC, HRP, IBOV, CDI.

Usage:
    python3 backtests/portfolio_dynamic_backtest.py
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
from core.portfolio_opt import (
    inverse_vol_weights,
    equal_risk_contribution_weights,
    hrp_weights,
    equal_weights,
    compute_portfolio_returns,
    compute_regime_portfolio,
)
from core.metrics import build_metrics, cumret
from core.plotting import PALETTE, fmt_ax

# ─── Configuration ────────────────────────────────────────────────────────────
LOOKBACK_RISK = 36    # for static risk parity fallback comparison
LOOKBACK_SHARPE = 12  # for rolling Sharpe weighting
OUT_DIR = _BACKTESTS_DIR


def compute_turnover(weights_df: pd.DataFrame) -> pd.Series:
    """Compute monthly L1-norm weight change (portfolio turnover)."""
    diff = weights_df.diff().abs().sum(axis=1)
    return diff


def _metrics_row(name: str, ret: pd.Series) -> dict:
    return build_metrics(ret.dropna(), name, 12)


def main():
    print("\n" + "=" * 70)
    print("  Dynamic Allocation Portfolio Backtest")
    print("=" * 70)

    print("\nLoading strategy returns...")
    returns_df, sim_results, regime_signals = build_strategy_returns()

    all_strats = [c for c in returns_df.columns if c not in ("IBOV", "CDI")]
    liquid_strats = [c for c in all_strats if c != "SmallcapMom"]
    print(f"Using liquid strategies: {liquid_strats}")
    print(f"\nNote: {SMALLCAP_MOM_NOTE}")

    # ── Static baselines for comparison ──────────────────────────────────────
    print("\nComputing static baselines...")
    ew_liq, _ = compute_portfolio_returns(
        returns_df, liquid_strats, lambda w: equal_weights(len(liquid_strats), liquid_strats)
    )
    iv_liq, _ = compute_portfolio_returns(
        returns_df, liquid_strats, lambda w: inverse_vol_weights(w, LOOKBACK_RISK)
    )
    hrp_liq, _ = compute_portfolio_returns(
        returns_df, liquid_strats, lambda w: hrp_weights(w, LOOKBACK_RISK)
    )

    # ── Dynamic portfolios ────────────────────────────────────────────────────
    print("\nComputing dynamic portfolios...")

    # Add CDI to strategy list for regime methods
    strats_with_cdi = liquid_strats + ["CDI"]

    dyn_rs, dyn_rs_w = compute_regime_portfolio(
        returns_df, liquid_strats, regime_signals,
        mode="rolling_sharpe", lookback_sharpe=LOOKBACK_SHARPE,
    )
    dyn_regime, dyn_regime_w = compute_regime_portfolio(
        returns_df, liquid_strats, regime_signals,
        mode="regime_only",
    )
    dyn_combined, dyn_combined_w = compute_regime_portfolio(
        returns_df, liquid_strats, regime_signals,
        mode="combined", lookback_sharpe=LOOKBACK_SHARPE,
    )

    # Also try different Sharpe lookbacks
    dyn_rs_6, _ = compute_regime_portfolio(
        returns_df, liquid_strats, regime_signals,
        mode="rolling_sharpe", lookback_sharpe=6,
    )
    dyn_rs_24, _ = compute_regime_portfolio(
        returns_df, liquid_strats, regime_signals,
        mode="rolling_sharpe", lookback_sharpe=24,
    )

    ibov_ret = returns_df["IBOV"].dropna()
    cdi_ret = returns_df["CDI"].dropna()

    # ── Metrics ───────────────────────────────────────────────────────────────
    print("\nComputing metrics...")
    metrics_list = [
        _metrics_row("EqualWeight",           ew_liq),
        _metrics_row("InvVol",                iv_liq),
        _metrics_row("HRP",                   hrp_liq),
        _metrics_row("DynRollSharpe (12m)",   dyn_rs),
        _metrics_row("DynRollSharpe (6m)",    dyn_rs_6),
        _metrics_row("DynRollSharpe (24m)",   dyn_rs_24),
        _metrics_row("DynRegime",             dyn_regime),
        _metrics_row("DynCombined",           dyn_combined),
        _metrics_row("IBOV",                  ibov_ret),
        _metrics_row("CDI",                   cdi_ret),
    ]
    metrics_list.sort(key=lambda m: float(m.get("Sharpe", 0)), reverse=True)

    print(f"\n{'='*95}")
    print(f"  DYNAMIC ALLOCATION COMPARISON -- After-Strategy-Level Tax (15% CGT)")
    print(f"{'='*95}")
    hdr = f"  {'Method':<25s} {'Ann.Ret%':>8s} {'Ann.Vol%':>8s} {'Sharpe':>8s} {'MaxDD%':>8s} {'Calmar':>8s}"
    print(hdr)
    print(f"  {'-'*70}")
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
    print(f"{'='*95}\n")

    # ── Turnover analysis ─────────────────────────────────────────────────────
    print("\n--- Turnover Analysis ---")
    for name, w_df in [
        ("DynRollSharpe (12m)", dyn_rs_w),
        ("DynRegime",           dyn_regime_w),
        ("DynCombined",         dyn_combined_w),
    ]:
        to = compute_turnover(w_df)
        print(f"  {name:<25s}: avg monthly weight change = {to.mean()*100:.1f}%")

    # ── Regime summary ────────────────────────────────────────────────────────
    is_easing = regime_signals["is_easing"].reindex(returns_df.index, method="ffill")
    ibov_calm = regime_signals["ibov_calm"].reindex(returns_df.index, method="ffill")

    regime_states = pd.DataFrame({
        "is_easing": is_easing,
        "ibov_calm": ibov_calm,
    }).dropna()

    print("\n--- Regime Summary ---")
    for (easing, calm), group in regime_states.groupby(["is_easing", "ibov_calm"]):
        label_e = "Easing" if easing else "Tightening"
        label_c = "Calm" if calm else "Stressed"
        budget = REGIME_EQUITY_BUDGET.get((easing, calm), 0.0)
        print(f"  {label_e} + {label_c}: {len(group)} months ({len(group)/len(regime_states)*100:.1f}%), "
              f"equity budget={budget*100:.0f}%")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    series_dict = {
        "EqualWeight":         ew_liq,
        "InvVol":              iv_liq,
        "HRP":                 hrp_liq,
        "DynRollSharpe (12m)": dyn_rs,
        "DynRegime":           dyn_regime,
        "DynCombined":         dyn_combined,
        "IBOV":                ibov_ret,
        "CDI":                 cdi_ret,
    }
    _plot_equity_curves(series_dict, os.path.join(OUT_DIR, "portfolio_dynamic.png"))

    # Weight evolution for best dynamic approach
    # Find best dynamic method
    dyn_metrics = {
        "DynRollSharpe": dyn_rs,
        "DynRegime":     dyn_regime,
        "DynCombined":   dyn_combined,
    }
    best_dyn = max(dyn_metrics, key=lambda k: float(
        _metrics_row(k, dyn_metrics[k]).get("Sharpe", 0)
    ))
    best_w_dict = {
        "DynRollSharpe": dyn_rs_w,
        "DynRegime":     dyn_regime_w,
        "DynCombined":   dyn_combined_w,
    }
    print(f"\nBest dynamic approach: {best_dyn}")
    _plot_weight_evolution(
        best_w_dict[best_dyn],
        title=f"Weight Evolution: {best_dyn}",
        out_path=os.path.join(OUT_DIR, "portfolio_dynamic_weights.png"),
    )

    print("\nDone.")


def _plot_equity_curves(series_dict: dict, out_path: str):
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })
    colors = [
        "#00D4AA", "#7B61FF", "#FF6B35", "#4FC3F7",
        "#F48FB1", "#FFC947", "#A8B2C1", "#81C784",
    ]
    linestyles = ["-", "-.", "--", "-", "-.", "--", ":", ":"]

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
            lw=2.2 if "Dyn" in name else 1.5,
        )

    ax.set_title("Dynamic Allocation Portfolio Comparison",
                 color=PALETTE["text"], fontsize=13, fontweight="bold")
    ax.set_ylabel("Growth of R$1", color=PALETTE["sub"])
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=8.5, ncol=2, loc="upper left")
    fmt_ax(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Equity curves saved -> {out_path}")


def _plot_weight_evolution(weights_df: pd.DataFrame, title: str, out_path: str):
    """Stacked area chart of portfolio weight evolution over time."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_facecolor(PALETTE["panel"])
    fig.patch.set_facecolor(PALETTE["bg"])

    colors = [
        "#00D4AA", "#7B61FF", "#FF6B35", "#FFC947",
        "#4FC3F7", "#F48FB1", "#81C784", "#A8B2C1", "#E0E0E0",
    ]

    # Only plot columns with meaningful weight
    cols = weights_df.columns.tolist()
    # Drop CDI from stacked area if it's almost always 1 (regime method)
    data = weights_df[cols].fillna(0.0)

    ax.stackplot(
        data.index,
        [data[c].values for c in cols],
        labels=cols,
        colors=colors[:len(cols)],
        alpha=0.75,
    )

    ax.set_title(title, color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Portfolio Weight", color=PALETTE["sub"])
    ax.set_ylim(0, 1)
    ax.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=7.5, ncol=3, loc="upper left",
    )
    fmt_ax(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Weight evolution plot saved -> {out_path}")


if __name__ == "__main__":
    main()
