"""
Portfolio Comparison Dashboard
================================
Comprehensive comparison of all portfolio construction methods:
  1. Equal Weight
  2. Inverse Volatility (InvVol)
  3. Equal Risk Contribution (ERC)
  4. Hierarchical Risk Parity (HRP)
  5. Dynamic Rolling Sharpe (12m)
  6. Dynamic Regime-Conditional
  7. Dynamic Combined (regime budget + rolling Sharpe within equity)

Benchmarks: IBOV, CDI

Outputs:
  - Console: ranked metrics table sorted by Sharpe
  - portfolio_compare_all.png    -- 4-panel plot
  - portfolio_compare_corr.png   -- correlation heatmap of portfolio returns
  - portfolio_comparison_results.csv -- full metrics table

Usage:
    python3 backtests/portfolio_compare_all.py
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
from core.metrics import build_metrics, cumret, ann_return, ann_vol, sharpe, max_dd, calmar
from core.plotting import PALETTE, fmt_ax

# ─── Configuration ────────────────────────────────────────────────────────────
LOOKBACK_RISK = 36
LOOKBACK_SHARPE = 12
OUT_DIR = _BACKTESTS_DIR


def metrics_dict(name: str, ret: pd.Series) -> dict:
    r = ret.dropna()
    m = build_metrics(r, name, 12)
    return m


def weight_stats(w_df: pd.DataFrame, name: str) -> dict:
    """Compute final, average, and std of weights for a time-varying portfolio."""
    final = w_df.iloc[-1]
    avg = w_df.mean()
    std = w_df.std()
    return {"method": name, "final": final, "avg": avg, "std": std}


def main():
    print("\n" + "=" * 70)
    print("  Portfolio Comparison Dashboard")
    print("=" * 70)

    # 1. Load strategy returns
    print("\nLoading strategy returns...")
    returns_df, sim_results, regime_signals = build_strategy_returns()

    liquid_strats = [
        c for c in returns_df.columns if c not in ("IBOV", "CDI", "SmallcapMom")
    ]
    all_strats = [c for c in returns_df.columns if c not in ("IBOV", "CDI")]
    print(f"Liquid strategies ({len(liquid_strats)}): {liquid_strats}")
    print(f"\nNote: {SMALLCAP_MOM_NOTE}")

    ibov_ret = returns_df["IBOV"].dropna()
    cdi_ret = returns_df["CDI"].dropna()

    # 2. Build all portfolios
    print("\nBuilding portfolios...")

    ew, ew_w = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: equal_weights(len(liquid_strats), w.columns)
    )
    iv, iv_w = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: inverse_vol_weights(w, LOOKBACK_RISK)
    )
    erc, erc_w = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: equal_risk_contribution_weights(w, LOOKBACK_RISK)
    )
    hrp, hrp_w = compute_portfolio_returns(
        returns_df, liquid_strats,
        lambda w: hrp_weights(w, LOOKBACK_RISK)
    )
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

    portfolios = {
        "EqualWeight":   (ew, ew_w),
        "InvVol":        (iv, iv_w),
        "ERC":           (erc, erc_w),
        "HRP":           (hrp, hrp_w),
        "DynRollSharpe": (dyn_rs, dyn_rs_w),
        "DynRegime":     (dyn_regime, dyn_regime_w),
        "DynCombined":   (dyn_combined, dyn_combined_w),
    }

    # 3. Compute metrics
    print("\nComputing metrics...")
    metrics_rows = []
    for name, (ret, w_df) in portfolios.items():
        m = metrics_dict(name, ret)
        # Add turnover
        turnover = w_df.diff().abs().sum(axis=1).mean() * 100
        m["Avg Turnover %"] = round(turnover, 1)
        # Add final NAV
        r_clean = ret.dropna()
        final_nav = 100_000 * (1 + r_clean).prod()
        m["Final NAV (R$)"] = round(final_nav, 0)
        metrics_rows.append(m)

    # Benchmarks
    m_ibov = metrics_dict("IBOV", ibov_ret)
    m_ibov["Avg Turnover %"] = 0.0
    m_ibov["Final NAV (R$)"] = round(100_000 * (1 + ibov_ret.dropna()).prod(), 0)
    m_cdi = metrics_dict("CDI", cdi_ret)
    m_cdi["Avg Turnover %"] = 0.0
    m_cdi["Final NAV (R$)"] = round(100_000 * (1 + cdi_ret.dropna()).prod(), 0)
    metrics_rows.extend([m_ibov, m_cdi])

    # Sort by Sharpe
    metrics_rows.sort(key=lambda m: float(m.get("Sharpe", 0)), reverse=True)

    # Console table
    print(f"\n{'='*110}")
    print(f"  PORTFOLIO COMPARISON -- {len(liquid_strats)} Liquid Strategies -- After-Strategy-Level Tax")
    print(f"{'='*110}")
    hdr = (
        f"  {'Method':<20s} {'Ann.Ret%':>8s} {'Ann.Vol%':>8s} "
        f"{'Sharpe':>8s} {'MaxDD%':>8s} {'Calmar':>8s} "
        f"{'Turnover%':>10s} {'Final NAV':>14s}"
    )
    print(hdr)
    print(f"  {'-'*102}")
    for m in metrics_rows:
        name = str(m.get("Strategy", "?"))[:20]
        print(
            f"  {name:<20s} "
            f"{str(m.get('Ann. Return (%)', '?')):>8s} "
            f"{str(m.get('Ann. Volatility (%)', '?')):>8s} "
            f"{str(m.get('Sharpe', '?')):>8s} "
            f"{str(m.get('Max Drawdown (%)', '?')):>8s} "
            f"{str(m.get('Calmar', '?')):>8s} "
            f"{str(m.get('Avg Turnover %', '?')):>10s} "
            f"R${str(m.get('Final NAV (R$)', '?')):>12s}"
        )
    print(f"{'='*110}\n")

    # Save CSV
    csv_path = os.path.join(OUT_DIR, "portfolio_comparison_results.csv")
    pd.DataFrame(metrics_rows).to_csv(csv_path, index=False)
    print(f"  Metrics CSV saved -> {csv_path}")

    # Best method
    best = metrics_rows[0]
    print(f"\n  WINNER: {best['Strategy']} (Sharpe={best['Sharpe']}, "
          f"Ann.Ret={best['Ann. Return (%)']}%, MaxDD={best['Max Drawdown (%)']}%)")

    # 4. Weight analysis for time-varying methods
    print("\n--- Weight Analysis (time-varying methods) ---")
    for name in ["InvVol", "ERC", "HRP", "DynRollSharpe", "DynRegime", "DynCombined"]:
        ret, w_df = portfolios[name]
        ws = weight_stats(w_df, name)
        print(f"\n  {name}:")
        print(f"    Final weights:   {dict(ws['final'].round(3))}")
        print(f"    Average weights: {dict(ws['avg'].round(3))}")
        print(f"    Weight std dev:  {dict(ws['std'].round(3))}")

    # 5. Plots
    print("\nGenerating plots...")
    port_returns_dict = {name: ret for name, (ret, _) in portfolios.items()}
    port_returns_dict["IBOV"] = ibov_ret
    port_returns_dict["CDI"] = cdi_ret

    _plot_4panel(port_returns_dict, metrics_rows,
                 os.path.join(OUT_DIR, "portfolio_compare_all.png"))
    _plot_correlation_heatmap(port_returns_dict,
                              os.path.join(OUT_DIR, "portfolio_compare_corr.png"))

    print("\nDone.")


def _plot_4panel(port_returns: dict, metrics_rows: list, out_path: str):
    """4-panel comparison plot: equity curves, drawdowns, rolling Sharpe, metrics table."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    colors = [
        "#00D4AA", "#7B61FF", "#FF6B35", "#4FC3F7",
        "#F48FB1", "#FFC947", "#81C784", "#A8B2C1", "#E0E0E0",
    ]
    linestyles = ["-", "-", "-.", "-.", "--", "--", ":", "-", ":"]
    methods_order = list(port_returns.keys())

    fig = plt.figure(figsize=(18, 18))
    gs = GridSpec(4, 1, figure=fig, hspace=0.4, left=0.07, right=0.97, top=0.93, bottom=0.05)

    # Panel 1: Cumulative equity curves (log scale)
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PALETTE["panel"])
    for idx, name in enumerate(methods_order):
        ret = port_returns[name].dropna()
        if len(ret) == 0:
            continue
        curve = cumret(ret)
        ax1.plot(curve.index, curve.values,
                 label=name, color=colors[idx % len(colors)],
                 ls=linestyles[idx % len(linestyles)],
                 lw=2.2 if "Dyn" in name or "HRP" in name else 1.4)
    ax1.set_yscale("log")
    ax1.set_title("Cumulative Returns (log scale)", color=PALETTE["text"],
                  fontsize=11, fontweight="bold")
    ax1.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=8, ncol=3)
    fmt_ax(ax1, ylabel="NAV (log)")

    # Panel 2: Drawdowns
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(PALETTE["panel"])
    for idx, name in enumerate(methods_order):
        ret = port_returns[name].dropna()
        if len(ret) == 0:
            continue
        curve = cumret(ret)
        dd = (curve / curve.cummax() - 1) * 100
        ax2.plot(dd.index, dd.values, label=name,
                 color=colors[idx % len(colors)],
                 ls=linestyles[idx % len(linestyles)],
                 lw=1.4, alpha=0.8)
    ax2.axhline(0, color=PALETTE["sub"], lw=0.6)
    ax2.set_title("Drawdown (%)", color=PALETTE["text"], fontsize=11, fontweight="bold")
    ax2.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=8, ncol=3)
    fmt_ax(ax2, ylabel="Drawdown (%)")

    # Panel 3: Rolling 12-month Sharpe
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(PALETTE["panel"])
    for idx, name in enumerate(methods_order[:6]):  # top 6 only to avoid clutter
        ret = port_returns[name].dropna()
        if len(ret) < 13:
            continue
        rolling_sh = ret.rolling(12).apply(
            lambda x: x.mean() / x.std() * np.sqrt(12) if x.std() > 0 else 0,
            raw=True,
        )
        ax3.plot(rolling_sh.index, rolling_sh.values,
                 label=name, color=colors[idx % len(colors)],
                 ls=linestyles[idx % len(linestyles)], lw=1.6)
    ax3.axhline(0, color=PALETTE["sub"], lw=0.8, ls="--")
    ax3.set_title("Rolling 12-Month Sharpe", color=PALETTE["text"],
                  fontsize=11, fontweight="bold")
    ax3.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=8, ncol=3)
    fmt_ax(ax3, ylabel="Sharpe")

    # Panel 4: Metrics table
    ax4 = fig.add_subplot(gs[3])
    ax4.axis("off")
    ax4.set_facecolor(PALETTE["panel"])

    cols_to_show = ["Strategy", "Ann. Return (%)", "Ann. Volatility (%)",
                    "Sharpe", "Max Drawdown (%)", "Calmar", "Avg Turnover %"]
    table_data = [[str(m.get(c, "?")) for c in cols_to_show] for m in metrics_rows]
    col_headers = ["Method", "Ann.Ret%", "Ann.Vol%", "Sharpe", "MaxDD%", "Calmar", "Turnover%"]

    tbl = ax4.table(
        cellText=table_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)

    row_colors = ["#0D2E26", "#1A1230", "#2A1A10", "#1E2A3A", "#332211",
                  "#1B2A1B", "#2A1A2A", "#1E1E2A", "#1A1A1A"]
    txt_colors_list = [
        "#00D4AA", "#7B61FF", "#FF6B35", "#4FC3F7",
        "#F48FB1", "#FFC947", "#81C784", "#A8B2C1", "#E0E0E0",
    ]

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if r == 0:
            cell.set_facecolor("#1F2937")
            cell.get_text().set_color(PALETTE["text"])
            cell.get_text().set_fontweight("bold")
        else:
            idx = (r - 1) % len(row_colors)
            cell.set_facecolor(row_colors[idx])
            cell.get_text().set_color(txt_colors_list[idx])

    ax4.set_title("Performance Summary (sorted by Sharpe)",
                  color=PALETTE["text"], fontsize=10, pad=6, y=0.98)

    fig.suptitle("Portfolio Optimization Comparison Dashboard",
                 fontsize=13, fontweight="bold", color=PALETTE["text"], y=0.97)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  4-panel plot saved -> {out_path}")


def _plot_correlation_heatmap(port_returns: dict, out_path: str):
    """Correlation heatmap of portfolio method returns."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    df = pd.DataFrame({name: ret.dropna() for name, ret in port_returns.items()})
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_facecolor(PALETTE["panel"])
    fig.patch.set_facecolor(PALETTE["bg"])

    n = len(corr)
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    for (yi, xi), val in np.ndenumerate(corr.values):
        ax.text(xi, yi, f"{val:.2f}",
                ha="center", va="center", fontsize=8,
                color="black" if abs(val) < 0.7 else "white",
                fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=30, ha="right", color=PALETTE["sub"], fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(corr.index, color=PALETTE["sub"], fontsize=8)
    ax.set_title("Portfolio Method Return Correlations",
                 color=PALETTE["text"], fontsize=12, fontweight="bold")
    ax.spines[:].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["sub"])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["sub"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Correlation heatmap saved -> {out_path}")


if __name__ == "__main__":
    main()
