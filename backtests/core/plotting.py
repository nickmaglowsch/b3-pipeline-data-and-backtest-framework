"""
Core plotting utilities for backtesting.
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from .metrics import cumret

PALETTE = {
    "pretax": "#00D4AA",
    "aftertax": "#7B61FF",
    "ibov": "#FF6B35",
    "tax": "#FF4C6A",
    "loss_cf": "#FFC947",
    "bg": "#0D1117",
    "panel": "#161B22",
    "grid": "#21262D",
    "text": "#E6EDF3",
    "sub": "#8B949E",
}


def fmt_ax(ax, ylabel=""):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["sub"], labelsize=8.5)
    ax.spines[:].set_color(PALETTE["grid"])
    if ylabel:
        ax.set_ylabel(ylabel, color=PALETTE["sub"], fontsize=9)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.6, ls="--")
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.3, ls=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_tax_backtest(
    title: str,
    pretax_val: pd.Series,
    aftertax_val: pd.Series,
    ibov_ret: pd.Series,
    tax_paid: pd.Series,
    loss_cf: pd.Series,
    turnover: pd.Series,
    metrics: list,
    total_tax_brl: float,
    out_path: str = "backtest_results.png",
    cdi_ret: "pd.Series | None" = None,
):
    """Generate the standard 4-panel tear sheet for tax-aware backtests."""
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "figure.facecolor": PALETTE["bg"],
            "text.color": PALETTE["text"],
            "axes.facecolor": PALETTE["panel"],
        }
    )

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        hspace=0.48,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.92,
        bottom=0.05,
    )

    common = pretax_val.index.intersection(ibov_ret.index)
    pt_val = pretax_val.loc[common]
    ibov = ibov_ret.loc[common]
    at_val = aftertax_val.loc[common]
    tx = tax_paid.loc[common]
    lc = loss_cf.loc[common]
    tv = turnover.loc[common]

    pt_curve = pt_val / pt_val.iloc[0]
    at_curve = at_val / at_val.iloc[0]
    ibov_curve = cumret(ibov)

    cdi_curve = None
    if cdi_ret is not None:
        cdi = cdi_ret.loc[common].fillna(0)
        cdi_curve = cumret(cdi)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        pt_curve.index,
        pt_curve.values,
        color=PALETTE["pretax"],
        lw=2.2,
        label="Pre-Tax",
        zorder=4,
    )
    ax1.plot(
        at_curve.index,
        at_curve.values,
        color=PALETTE["aftertax"],
        lw=2.2,
        label="After-Tax (15% CGT)",
        zorder=3,
        ls="-.",
    )
    ax1.plot(
        ibov_curve.index,
        ibov_curve.values,
        color=PALETTE["ibov"],
        lw=1.8,
        label="Benchmark",
        zorder=2,
        ls="--",
    )

    if cdi_curve is not None:
        ax1.plot(
            cdi_curve.index,
            cdi_curve.values,
            color="#A8B2C1",
            lw=1.5,
            label="CDI",
            zorder=1,
            ls=":",
        )

    ax1.fill_between(
        pt_curve.index,
        pt_curve.values,
        at_curve.values,
        alpha=0.15,
        color=PALETTE["tax"],
        label="Tax Drag",
    )
    ax1.set_title(
        "Cumulative Return",
        color=PALETTE["text"],
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=9,
        ncol=5,
    )
    fmt_ax(ax1, ylabel="Growth of R$1")

    ax2 = fig.add_subplot(gs[1, :])
    dd_pt = (pt_curve / pt_curve.cummax() - 1) * 100
    dd_at = (at_curve / at_curve.cummax() - 1) * 100
    dd_ibov = (ibov_curve / ibov_curve.cummax() - 1) * 100
    ax2.fill_between(
        dd_pt.index,
        dd_pt.values,
        0,
        alpha=0.45,
        color=PALETTE["pretax"],
        label="Pre-Tax DD",
    )
    ax2.fill_between(
        dd_at.index,
        dd_at.values,
        0,
        alpha=0.35,
        color=PALETTE["aftertax"],
        label="After-Tax DD",
    )
    ax2.fill_between(
        dd_ibov.index,
        dd_ibov.values,
        0,
        alpha=0.25,
        color=PALETTE["ibov"],
        label="Benchmark DD",
    )

    if cdi_curve is not None:
        dd_cdi = (cdi_curve / cdi_curve.cummax() - 1) * 100
        ax2.fill_between(
            dd_cdi.index, dd_cdi.values, 0, alpha=0.20, color="#A8B2C1", label="CDI DD"
        )

    ax2.set_title("Drawdown (%)", color=PALETTE["text"], fontsize=11, pad=6)
    ax2.legend(
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=9,
        ncol=4,
    )
    fmt_ax(ax2, ylabel="Drawdown (%)")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(tx.index, tx.values / 1_000, color=PALETTE["tax"], width=20, alpha=0.8)
    ax3.set_title(
        "Tax Paid per Month (R$ k)", color=PALETTE["text"], fontsize=10, pad=6
    )
    fmt_ax(ax3, ylabel="R$ thousands")

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.fill_between(
        lc.index, lc.values / 1_000, 0, alpha=0.6, color=PALETTE["loss_cf"]
    )
    ax4.plot(lc.index, lc.values / 1_000, color=PALETTE["loss_cf"], lw=1.4)
    ax4.set_title(
        "Loss Carryforward Balance (R$ k)", color=PALETTE["text"], fontsize=10, pad=6
    )
    fmt_ax(ax4, ylabel="R$ thousands")

    ax5 = fig.add_subplot(gs[2, 2])
    ax5.bar(tv.index, tv.values * 100, color=PALETTE["pretax"], width=20, alpha=0.7)
    ax5.set_title(
        "Monthly Portfolio Turnover (%)", color=PALETTE["text"], fontsize=10, pad=6
    )
    fmt_ax(ax5, ylabel="%")

    ax6 = fig.add_subplot(gs[3, 0:2])
    spread = (pt_curve - at_curve) * 100
    ax6.fill_between(spread.index, spread.values, 0, alpha=0.55, color=PALETTE["tax"])
    ax6.plot(spread.index, spread.values, color=PALETTE["tax"], lw=1.4)
    ax6.axhline(0, color=PALETTE["sub"], lw=0.8)
    ax6.set_title(
        "Cumulative Tax Drag: Pre-Tax minus After-Tax (percentage points)",
        color=PALETTE["text"],
        fontsize=10,
        pad=6,
    )
    fmt_ax(ax6, ylabel="pp")
    ax6.annotate(
        f"Total tax paid: R$ {total_tax_brl:,.0f}",
        xy=(0.02, 0.88),
        xycoords="axes fraction",
        fontsize=9,
        color=PALETTE["tax"],
        bbox=dict(
            boxstyle="round,pad=0.3", fc=PALETTE["panel"], ec=PALETTE["tax"], alpha=0.8
        ),
    )

    ax7 = fig.add_subplot(gs[3, 2])
    ax7.axis("off")
    col_labels = list(metrics[0].keys())
    row_vals = [[str(m[k]) for k in col_labels] for m in metrics]

    row_bg = ["#0D2E26", "#1A1230", "#2A1A10", "#1E2A3A", "#332211"]

    # Need to match the number of metrics provided
    txt_colors = [
        PALETTE["pretax"],
        PALETTE["aftertax"],
        PALETTE["ibov"],
        "#A8B2C1",
        "#FFC947",
    ]

    tbl = ax7.table(
        cellText=row_vals, colLabels=col_labels, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if r == 0:
            cell.set_facecolor("#1F2937")
            cell.get_text().set_color(PALETTE["text"])
            cell.get_text().set_fontweight("bold")
        else:
            color_idx = (r - 1) % len(row_bg)
            cell.set_facecolor(row_bg[color_idx])
            cell.get_text().set_color(txt_colors[color_idx])

    ax7.set_title(
        "Performance Summary", color=PALETTE["text"], fontsize=10, pad=6, y=0.98
    )

    fig.suptitle(title, fontsize=12, fontweight="bold", color=PALETTE["text"], y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"📊  Plot saved → {out_path}")

    try:
        plt.show()
    except Exception:
        pass
