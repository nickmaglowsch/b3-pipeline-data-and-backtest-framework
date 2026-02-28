"""
Sub-Period Stability Analysis
================================
Computes portfolio metrics across 4 sub-periods to test whether optimization
benefits are consistent or concentrated in one historical era.

Sub-periods:
  2005-2010: Commodity supercycle, pre/post-GFC, high CDI
  2010-2015: Dilma government, Petrobras scandal, IBOV decline
  2015-2020: Impeachment, Temer/Bolsonaro, COVID crash
  2020-present: Post-COVID recovery, high CDI returning

For each portfolio method, the full simulation is run once (2005-present),
then the resulting equity curves are sliced into sub-periods for metrics.
This preserves the time-varying weight history (no re-start bias).

Outputs:
  - portfolio_stability_heatmap.png  -- Sharpe heatmap (methods x sub-periods)
  - portfolio_stability_rolling.png  -- Rolling 36-month Sharpe with stress shading
  - portfolio_stability_results.csv  -- Full results

Usage:
    python3 backtests/portfolio_stability_analysis.py
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
from matplotlib.gridspec import GridSpec

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
from core.metrics import build_metrics, ann_return, ann_vol, sharpe, max_dd
from core.plotting import PALETTE, fmt_ax

# ─── Sub-period definitions ───────────────────────────────────────────────────
SUB_PERIODS = [
    ("2005-2010",    "2005-01-01", "2009-12-31"),
    ("2010-2015",    "2010-01-01", "2014-12-31"),
    ("2015-2020",    "2015-01-01", "2019-12-31"),
    ("2020-present", "2020-01-01", None),
]

# Stress periods for shading on rolling Sharpe plot
STRESS_PERIODS = [
    ("GFC",           "2008-01-01", "2009-03-31"),
    ("Dilma crisis",  "2014-01-01", "2016-06-30"),
    ("COVID",         "2020-02-01", "2020-06-30"),
]

LOOKBACK_RISK = 36
LOOKBACK_SHARPE = 12
OUT_DIR = _BACKTESTS_DIR


def subperiod_metrics(ret: pd.Series, start: str, end: str) -> dict:
    """Compute Sharpe, Ann.Return, MaxDD for a sub-period slice."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else ret.index[-1]
    sub = ret.loc[start_dt:end_dt].dropna()
    if len(sub) < 3:
        return {"sharpe": np.nan, "ann_return": np.nan, "max_dd": np.nan}
    m = build_metrics(sub, "", 12)
    return {
        "sharpe":     float(m.get("Sharpe", np.nan)),
        "ann_return": float(m.get("Ann. Return (%)", np.nan)),
        "max_dd":     float(m.get("Max Drawdown (%)", np.nan)),
    }


def cdi_excess_return(ret: pd.Series, cdi: pd.Series, start: str, end: str) -> float:
    """Return excess return (strategy minus CDI) over sub-period."""
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end) if end else ret.index[-1]
    s_sub = ret.loc[start_dt:end_dt].dropna()
    c_sub = cdi.loc[start_dt:end_dt].dropna()
    common = s_sub.index.intersection(c_sub.index)
    if len(common) < 3:
        return np.nan
    return float(
        ann_return(s_sub.loc[common], 12) - ann_return(c_sub.loc[common], 12)
    )


def main():
    print("\n" + "=" * 70)
    print("  Sub-Period Stability Analysis")
    print("=" * 70)

    # Load strategy returns
    print("\nLoading strategy returns...")
    returns_df, sim_results, regime_signals = build_strategy_returns()

    liquid_strats = [
        c for c in returns_df.columns if c not in ("IBOV", "CDI", "SmallcapMom")
    ]
    ibov_ret = returns_df["IBOV"].dropna()
    cdi_ret = returns_df["CDI"].dropna()

    # Build all portfolio return series (full period)
    print("\nBuilding portfolio return series...")
    portfolio_ret = {}

    portfolio_ret["EqualWeight"], _ = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: equal_weights(len(liquid_strats), w.columns)
    )
    portfolio_ret["InvVol"], _ = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: inverse_vol_weights(w, LOOKBACK_RISK)
    )
    portfolio_ret["ERC"], _ = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: equal_risk_contribution_weights(w, LOOKBACK_RISK)
    )
    portfolio_ret["HRP"], _ = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: hrp_weights(w, LOOKBACK_RISK)
    )
    portfolio_ret["DynRollSharpe"], _ = compute_regime_portfolio(
        returns_df, liquid_strats, regime_signals, mode="rolling_sharpe"
    )
    portfolio_ret["DynRegime"], _ = compute_regime_portfolio(
        returns_df, liquid_strats, regime_signals, mode="regime_only"
    )
    portfolio_ret["DynCombined"], _ = compute_regime_portfolio(
        returns_df, liquid_strats, regime_signals, mode="combined"
    )

    # ── Sub-period analysis ────────────────────────────────────────────────────
    print("\nComputing sub-period metrics...")
    methods = list(portfolio_ret.keys())
    all_periods = [(label, s, e) for (label, s, e) in SUB_PERIODS] + [("Full", "2005-01-01", None)]

    # Build results table
    results = {}
    csv_rows = []

    print(f"\n{'='*95}")
    print(f"  SUB-PERIOD STABILITY ANALYSIS")
    print(f"{'='*95}")

    for method in methods:
        ret = portfolio_ret[method]
        print(f"\n  Method: {method}")
        print(f"  {'Period':<15} {'Ann.Ret%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'vs CDI':>8}")
        print(f"  {'-'*55}")

        results[method] = {}
        for label, s, e in all_periods:
            m = subperiod_metrics(ret, s, e or str(ret.index[-1].date()))
            excess = cdi_excess_return(ret, cdi_ret, s, e or str(ret.index[-1].date()))
            results[method][label] = m
            print(
                f"  {label:<15} "
                f"{m['ann_return']:>8.1f}% "
                f"{m['sharpe']:>8.2f} "
                f"{m['max_dd']:>8.1f}% "
                f"{excess*100:>7.1f}%"
            )
            csv_rows.append({
                "method": method,
                "period": label,
                "ann_return": round(m["ann_return"], 2),
                "sharpe": round(m["sharpe"], 3),
                "max_dd": round(m["max_dd"], 2),
                "excess_vs_cdi": round(excess * 100, 2) if not np.isnan(excess) else np.nan,
            })

    # ── Consistency scores ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  CONSISTENCY SCORES (sub-periods with positive CDI excess return)")
    print(f"{'='*70}")

    consistency = {}
    for method in methods:
        count = sum(
            1 for label, s, e in SUB_PERIODS
            if not np.isnan(
                cdi_excess_return(portfolio_ret[method], cdi_ret, s, e or str(portfolio_ret[method].index[-1].date()))
            ) and cdi_excess_return(portfolio_ret[method], cdi_ret, s, e or str(portfolio_ret[method].index[-1].date())) > 0
        )
        consistency[method] = count

    worst_sharpe = {
        method: min(
            results[method][label]["sharpe"]
            for label, _, _ in SUB_PERIODS
            if not np.isnan(results[method][label]["sharpe"])
        )
        for method in methods
    }

    sorted_methods = sorted(methods, key=lambda m: (consistency[m], worst_sharpe.get(m, -99)), reverse=True)
    print(f"\n  {'Method':<20} {'Consistent (4 periods)':>22} {'Worst sub-period Sharpe':>24}")
    print(f"  {'-'*68}")
    for m in sorted_methods:
        print(
            f"  {m:<20} "
            f"{consistency[m]:>5}/4  "
            f"{'Yes' if consistency[m] == 4 else 'No':<15} "
            f"{worst_sharpe.get(m, np.nan):>10.2f}"
        )

    # ── CSV export ─────────────────────────────────────────────────────────────
    csv_path = os.path.join(OUT_DIR, "portfolio_stability_results.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"\n  Results CSV saved -> {csv_path}")

    # ── Plots ──────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    _plot_stability_heatmap(results, methods, all_periods,
                            os.path.join(OUT_DIR, "portfolio_stability_heatmap.png"))
    _plot_rolling_sharpe(portfolio_ret,
                         os.path.join(OUT_DIR, "portfolio_stability_rolling.png"))

    print("\nDone.")


def _plot_stability_heatmap(results: dict, methods: list, all_periods: list, out_path: str):
    """Heatmap: rows = methods, columns = periods, values = Sharpe."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    period_labels = [label for label, _, _ in all_periods]
    data = np.array([
        [results[m][p]["sharpe"] for p in period_labels]
        for m in methods
    ])

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_facecolor(PALETTE["panel"])
    fig.patch.set_facecolor(PALETTE["bg"])

    vmax = max(abs(np.nanmax(data)), abs(np.nanmin(data)))
    vmax = max(vmax, 1.0)
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    for (yi, xi), val in np.ndenumerate(data):
        if np.isnan(val):
            txt = "n/a"
        else:
            txt = f"{val:.2f}"
        ax.text(xi, yi, txt,
                ha="center", va="center", fontsize=9,
                color="black" if 0.2 < (val + vmax) / (2 * vmax) < 0.9 else "white",
                fontweight="bold")

    ax.set_xticks(range(len(period_labels)))
    ax.set_xticklabels(period_labels, color=PALETTE["sub"], fontsize=9)
    ax.set_yticks(range(len(methods)))
    ax.set_yticklabels(methods, color=PALETTE["sub"], fontsize=9)
    ax.set_title("Sub-Period Sharpe Ratio Heatmap",
                 color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.spines[:].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["sub"])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["sub"])
    cbar.set_label("Sharpe Ratio", color=PALETTE["sub"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Heatmap saved -> {out_path}")


def _plot_rolling_sharpe(portfolio_ret: dict, out_path: str):
    """Rolling 36-month Sharpe for top methods with stress period shading."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    colors = ["#00D4AA", "#7B61FF", "#FF6B35", "#FFC947"]
    linestyles = ["-", "-.", "--", ":"]

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_facecolor(PALETTE["panel"])
    fig.patch.set_facecolor(PALETTE["bg"])

    # Plot top 4 methods only
    top_methods = ["DynCombined", "DynRollSharpe", "HRP", "EqualWeight"]

    for idx, method in enumerate(top_methods):
        if method not in portfolio_ret:
            continue
        ret = portfolio_ret[method].dropna()
        if len(ret) < 37:
            continue
        rolling_sh = ret.rolling(36).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0,
            raw=True,
        )
        ax.plot(rolling_sh.index, rolling_sh.values,
                label=method, color=colors[idx], ls=linestyles[idx], lw=2.0)

    ax.axhline(0, color=PALETTE["sub"], lw=0.8, ls="--", alpha=0.6)

    # Shade stress periods
    stress_colors = ["#FF4C6A", "#FFC947", "#4FC3F7"]
    for i, (label, s, e) in enumerate(STRESS_PERIODS):
        ax.axvspan(pd.Timestamp(s), pd.Timestamp(e),
                   alpha=0.15, color=stress_colors[i], label=label)

    ax.set_title("Rolling 36-Month Sharpe Ratio",
                 color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.set_ylabel("Annualized Sharpe", color=PALETTE["sub"])
    ax.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
              labelcolor=PALETTE["text"], fontsize=8.5, ncol=3)
    fmt_ax(ax)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Rolling Sharpe plot saved -> {out_path}")


if __name__ == "__main__":
    main()
