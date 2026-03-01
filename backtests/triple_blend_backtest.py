"""
Triple Blend: SmallCap Momentum + Multifactor + CDI (33/33/33)
==============================================================
Allocates equally across three sleeves:
- 33% Small-Cap Momentum (6M lookback, 1M skip, bottom 50% ADTV, top 20% mom)
- 33% Multifactor (12M lookback, 1M skip, 50% mom rank + 50% low-vol rank, top 10%)
- 33% CDI (Brazilian fixed income)

Stocks appearing in both equity sleeves have their weights summed.
Three tax scenarios: Pre-Tax / 15% CGT / Deferred DARF.
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import download_benchmark, download_cdi_daily, load_b3_data
from core.metrics import build_metrics, cumret, display_metrics_table, value_to_ret
from core.plotting import PALETTE, fmt_ax
from core.simulation import run_simulation

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
WEIGHT_SMALLCAP = 1 / 3
WEIGHT_MULTIFACTOR = 1 / 3
WEIGHT_CDI = 1 / 3

REBALANCE_FREQ = "ME"

# Small-cap momentum parameters
LOOKBACK_PERIODS_SC = 6   # 6-month momentum
SKIP_PERIODS_SC = 1
TOP_QUINTILE_SC = 0.20    # top 20%

# Multifactor parameters
LOOKBACK_PERIODS_MF = 12  # 12-month lookback
SKIP_PERIODS_MF = 1
TOP_DECILE_MF = 0.10      # top 10%

MIN_ADTV = 1_000_000      # R$1M liquidity floor
TAX_RATE = 0.15
SLIPPAGE = 0.001
INITIAL_CAPITAL = 100_000

START_DATE = "2000-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

PERIODS_PER_YEAR = 12
DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


def generate_signals(adj_close, close_px, fin_vol, cdi_ret):
    """
    Build combined target weights for triple-blend portfolio.

    Two independent stock selectors run on the same resampled data:
    1. SmallCap Momentum sleeve (budget 0.33)
    2. Multifactor sleeve (budget 0.33)
    Weights are summed per stock; CDI_ASSET gets 0.33.
    """
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()

    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    log_ret = np.log1p(ret)

    # ── SmallCap Momentum signal ──────────────────────────────────
    sc_signal = log_ret.shift(SKIP_PERIODS_SC).rolling(LOOKBACK_PERIODS_SC).sum()
    sc_glitch = (
        ((ret > 1.0) | (ret < -0.90))
        .shift(SKIP_PERIODS_SC)
        .rolling(LOOKBACK_PERIODS_SC)
        .max()
    )
    sc_signal[sc_glitch == 1] = np.nan

    # ── Multifactor signal ────────────────────────────────────────
    mom_signal = log_ret.shift(SKIP_PERIODS_MF).rolling(LOOKBACK_PERIODS_MF).sum()
    mom_glitch = (
        ((ret > 1.0) | (ret < -0.90))
        .shift(SKIP_PERIODS_MF)
        .rolling(LOOKBACK_PERIODS_MF)
        .max()
    )
    mom_signal[mom_glitch == 1] = np.nan

    vol_signal = -ret.shift(1).rolling(LOOKBACK_PERIODS_MF).std()
    vol_glitch = (
        ((ret > 1.0) | (ret < -0.90))
        .shift(1)
        .rolling(LOOKBACK_PERIODS_MF)
        .max()
    )
    vol_signal[vol_glitch == 1] = np.nan

    mom_rank = mom_signal.rank(axis=1, pct=True)
    vol_rank = vol_signal.rank(axis=1, pct=True)
    mf_composite = (mom_rank * 0.5) + (vol_rank * 0.5)

    # ── Build combined weights ────────────────────────────────────
    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    target_weights["CDI_ASSET"] = 0.0

    # start_idx uses the longer lookback so both sleeves have enough data
    start_idx = max(
        LOOKBACK_PERIODS_SC + SKIP_PERIODS_SC,
        LOOKBACK_PERIODS_MF + SKIP_PERIODS_MF,
    ) + 1

    prev_sel_sc = set()
    prev_sel_mf = set()

    for i in range(start_idx, len(ret)):
        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]

        # Common liquidity filter
        valid_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)

        # ── SmallCap Momentum selection ───────────────────────────
        sc_row = sc_signal.iloc[i - 1]
        sc_liquid = sc_row[valid_mask].dropna()

        if len(sc_liquid) < 5:
            sel_sc = prev_sel_sc
        else:
            liquid_adtv = adtv_row[sc_liquid.index]
            adtv_median = liquid_adtv.median()
            smallcap_mask = liquid_adtv <= adtv_median
            smallcap_universe = sc_liquid[smallcap_mask]

            if len(smallcap_universe) < 3:
                sel_sc = prev_sel_sc
            else:
                n_sel = max(1, int(len(smallcap_universe) * TOP_QUINTILE_SC))
                sel_sc = set(smallcap_universe.nlargest(n_sel).index)

        # ── Multifactor selection ─────────────────────────────────
        mf_row = mf_composite.iloc[i - 1]
        mf_valid = mf_row[valid_mask].dropna()

        if len(mf_valid) < 5:
            sel_mf = prev_sel_mf
        else:
            n_sel = max(1, int(len(mf_valid) * TOP_DECILE_MF))
            sel_mf = set(mf_valid.nlargest(n_sel).index)

        # ── Combine weights ───────────────────────────────────────
        has_stocks = bool(sel_sc) or bool(sel_mf)
        if not has_stocks:
            continue

        if sel_sc:
            w_sc = WEIGHT_SMALLCAP / len(sel_sc)
            for t in sel_sc:
                col = target_weights.columns.get_loc(t)
                target_weights.iloc[i, col] += w_sc

        if sel_mf:
            w_mf = WEIGHT_MULTIFACTOR / len(sel_mf)
            for t in sel_mf:
                col = target_weights.columns.get_loc(t)
                target_weights.iloc[i, col] += w_mf

        target_weights.iloc[i, target_weights.columns.get_loc("CDI_ASSET")] = WEIGHT_CDI

        prev_sel_sc = sel_sc
        prev_sel_mf = sel_mf

    # Merge CDI returns into the returns matrix
    ret["CDI_ASSET"] = cdi_ret

    return ret, target_weights


def plot_triple_blend(
    title,
    pretax_val,
    aftertax_val,
    deferred_val,
    ibov_ret,
    cdi_ret,
    tax_paid_std,
    tax_paid_deferred,
    loss_cf_std,
    loss_cf_deferred,
    turnover,
    metrics,
    total_tax_std,
    total_tax_deferred,
    out_path="triple_blend_backtest.png",
):
    """Custom 3-way tear sheet with 5 curves."""
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "figure.facecolor": PALETTE["bg"],
            "text.color": PALETTE["text"],
            "axes.facecolor": PALETTE["panel"],
        }
    )

    EXEMPT_COLOR = "#FFD700"  # gold for deferred curve

    fig = plt.figure(figsize=(18, 20))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        hspace=0.48,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.93,
        bottom=0.04,
    )

    # Align all series to common index
    common = pretax_val.index.intersection(ibov_ret.index).intersection(cdi_ret.index)
    pt_val = pretax_val.loc[common]
    at_val = aftertax_val.loc[common]
    df_val = deferred_val.loc[common]
    ibov = ibov_ret.loc[common]
    cdi = cdi_ret.loc[common].fillna(0)
    tx_std = tax_paid_std.loc[common]
    tx_df = tax_paid_deferred.loc[common]
    lc_std = loss_cf_std.loc[common]
    lc_df = loss_cf_deferred.loc[common]
    tv = turnover.loc[common]

    # Normalize to growth-of-R$1
    pt_curve = pt_val / pt_val.iloc[0]
    at_curve = at_val / at_val.iloc[0]
    df_curve = df_val / df_val.iloc[0]
    ibov_curve = cumret(ibov)
    cdi_curve = cumret(cdi)

    # ── Row 0: Cumulative Returns ──────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(
        pt_curve.index,
        pt_curve.values,
        color=PALETTE["pretax"],
        lw=2.2,
        label="Pre-Tax",
        zorder=5,
    )
    ax0.plot(
        at_curve.index,
        at_curve.values,
        color=PALETTE["aftertax"],
        lw=2.2,
        label="15% CGT",
        zorder=4,
        ls="-.",
    )
    ax0.plot(
        df_curve.index,
        df_curve.values,
        color=EXEMPT_COLOR,
        lw=2.2,
        label="Deferred DARF",
        zorder=3,
        ls="--",
    )
    ax0.plot(
        ibov_curve.index,
        ibov_curve.values,
        color=PALETTE["ibov"],
        lw=1.8,
        label="IBOV",
        zorder=2,
        ls="--",
    )
    ax0.plot(
        cdi_curve.index,
        cdi_curve.values,
        color="#A8B2C1",
        lw=1.5,
        label="CDI",
        zorder=1,
        ls=":",
    )

    # Tax drag fills
    ax0.fill_between(
        pt_curve.index,
        pt_curve.values,
        at_curve.values,
        alpha=0.12,
        color=PALETTE["tax"],
        label="Tax Drag (std)",
    )
    ax0.fill_between(
        at_curve.index,
        at_curve.values,
        df_curve.values,
        alpha=0.10,
        color=EXEMPT_COLOR,
        label="Deferral Benefit",
    )

    ax0.set_title(
        "Cumulative Return",
        color=PALETTE["text"],
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax0.legend(
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
        ncol=4,
    )
    fmt_ax(ax0, ylabel="Growth of R$1")

    # ── Row 1: Drawdowns ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    for curve, label, color, alpha in [
        (pt_curve, "Pre-Tax DD", PALETTE["pretax"], 0.45),
        (at_curve, "15% CGT DD", PALETTE["aftertax"], 0.35),
        (df_curve, "Deferred DARF DD", EXEMPT_COLOR, 0.30),
        (ibov_curve, "IBOV DD", PALETTE["ibov"], 0.25),
        (cdi_curve, "CDI DD", "#A8B2C1", 0.20),
    ]:
        dd = (curve / curve.cummax() - 1) * 100
        ax1.fill_between(dd.index, dd.values, 0, alpha=alpha, color=color, label=label)

    ax1.set_title("Drawdown (%)", color=PALETTE["text"], fontsize=11, pad=6)
    ax1.legend(
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
        ncol=5,
    )
    fmt_ax(ax1, ylabel="Drawdown (%)")

    # ── Row 2: Tax, Loss CF, Turnover ─────────────────────────────
    ax2a = fig.add_subplot(gs[2, 0])
    w = 12
    ax2a.bar(
        tx_std.index - pd.Timedelta(days=7),
        tx_std.values / 1_000,
        color=PALETTE["tax"],
        width=w,
        alpha=0.8,
        label="Std 15%",
    )
    ax2a.bar(
        tx_df.index + pd.Timedelta(days=7),
        tx_df.values / 1_000,
        color=EXEMPT_COLOR,
        width=w,
        alpha=0.8,
        label="Deferred DARF",
    )
    ax2a.set_title(
        "Tax Paid per Month (R$ k)", color=PALETTE["text"], fontsize=10, pad=6
    )
    ax2a.legend(
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=8,
    )
    fmt_ax(ax2a, ylabel="R$ thousands")

    ax2b = fig.add_subplot(gs[2, 1])
    ax2b.plot(
        lc_std.index,
        lc_std.values / 1_000,
        color=PALETTE["loss_cf"],
        lw=1.4,
        label="Std 15%",
    )
    ax2b.fill_between(
        lc_std.index, lc_std.values / 1_000, 0, alpha=0.3, color=PALETTE["loss_cf"]
    )
    ax2b.plot(
        lc_df.index,
        lc_df.values / 1_000,
        color=EXEMPT_COLOR,
        lw=1.4,
        ls="--",
        label="Deferred DARF",
    )
    ax2b.fill_between(
        lc_df.index, lc_df.values / 1_000, 0, alpha=0.2, color=EXEMPT_COLOR
    )
    ax2b.set_title(
        "Loss Carryforward (R$ k)", color=PALETTE["text"], fontsize=10, pad=6
    )
    ax2b.legend(
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=8,
    )
    fmt_ax(ax2b, ylabel="R$ thousands")

    ax2c = fig.add_subplot(gs[2, 2])
    ax2c.bar(tv.index, tv.values * 100, color=PALETTE["pretax"], width=20, alpha=0.7)
    ax2c.set_title(
        "Monthly Portfolio Turnover (%)", color=PALETTE["text"], fontsize=10, pad=6
    )
    fmt_ax(ax2c, ylabel="%")

    # ── Row 3: Cumulative Tax Drag + Metrics Table ────────────────
    ax3a = fig.add_subplot(gs[3, 0:2])
    spread_std = (pt_curve - at_curve) * 100
    spread_df = (pt_curve - df_curve) * 100
    ax3a.fill_between(
        spread_std.index,
        spread_std.values,
        0,
        alpha=0.45,
        color=PALETTE["tax"],
        label="Std 15% drag",
    )
    ax3a.plot(spread_std.index, spread_std.values, color=PALETTE["tax"], lw=1.4)
    ax3a.fill_between(
        spread_df.index,
        spread_df.values,
        0,
        alpha=0.30,
        color=EXEMPT_COLOR,
        label="Deferred DARF drag",
    )
    ax3a.plot(spread_df.index, spread_df.values, color=EXEMPT_COLOR, lw=1.4, ls="--")
    ax3a.axhline(0, color=PALETTE["sub"], lw=0.8)
    ax3a.set_title(
        "Cumulative Tax Drag (pp)", color=PALETTE["text"], fontsize=10, pad=6
    )
    ax3a.legend(
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=8.5,
    )
    fmt_ax(ax3a, ylabel="pp")

    nav_benefit = deferred_val.iloc[-1] - aftertax_val.iloc[-1]
    ax3a.annotate(
        f"Std tax: R$ {total_tax_std:,.0f}  |  Deferred tax: R$ {total_tax_deferred:,.0f}  |  NAV benefit: R$ {nav_benefit:,.0f}",
        xy=(0.02, 0.88),
        xycoords="axes fraction",
        fontsize=8.5,
        color=EXEMPT_COLOR,
        bbox=dict(
            boxstyle="round,pad=0.3", fc=PALETTE["panel"], ec=EXEMPT_COLOR, alpha=0.8
        ),
    )

    # Metrics table
    ax3b = fig.add_subplot(gs[3, 2])
    ax3b.axis("off")
    col_labels = list(metrics[0].keys())
    row_vals = [[str(m[k]) for k in col_labels] for m in metrics]

    row_bg = ["#0D2E26", "#1A1230", "#2A2010", "#2A1A10", "#1E2A3A"]
    txt_colors = [
        PALETTE["pretax"],
        PALETTE["aftertax"],
        EXEMPT_COLOR,
        PALETTE["ibov"],
        "#A8B2C1",
    ]

    tbl = ax3b.table(
        cellText=row_vals, colLabels=col_labels, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7)
    tbl.scale(1, 1.7)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if r == 0:
            cell.set_facecolor("#1F2937")
            cell.get_text().set_color(PALETTE["text"])
            cell.get_text().set_fontweight("bold")
        else:
            idx = (r - 1) % len(row_bg)
            cell.set_facecolor(row_bg[idx])
            cell.get_text().set_color(txt_colors[idx])

    ax3b.set_title(
        "Performance Summary", color=PALETTE["text"], fontsize=10, pad=6, y=0.98
    )

    fig.suptitle(title, fontsize=12, fontweight="bold", color=PALETTE["text"], y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"\n  Plot saved -> {out_path}")

    try:
        plt.show()
    except Exception:
        pass


def main():
    print("\n" + "=" * 70)
    print("  TRIPLE BLEND: SmallCap Momentum + Multifactor + CDI (33/33/33)")
    print("  3 Tax Scenarios: Pre-Tax / 15% CGT / Deferred DARF")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"

    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n  Generating triple-blend signals...")
    ret, target_weights = generate_signals(adj_close, close_px, fin_vol, cdi_ret)
    ret = ret.fillna(0.0)

    # ── Run 1: Standard 15% CGT (also gives pre-tax curve) ────────
    print("\n  Running simulation: Standard 15% CGT...")
    result_std = run_simulation(
        returns_matrix=ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="TripleBlend",
        monthly_sales_exemption=20_000,
    )

    # ── Run 2: 15% CGT with deferred DARF payment ───────────────
    print("  Running simulation: 15% CGT + Deferred DARF...")
    result_deferred = run_simulation(
        returns_matrix=ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="TripleBlend (Deferred DARF)",
        monthly_sales_exemption=20_000,
        defer_tax=True,
    )

    # ── Align to common index ─────────────────────────────────────
    common = (
        result_std["pretax_values"]
        .index.intersection(ibov_ret.index)
        .intersection(cdi_ret.index)
    )

    pretax_val = result_std["pretax_values"].loc[common]
    aftertax_val = result_std["aftertax_values"].loc[common]
    deferred_val = result_deferred["aftertax_values"].loc[common]
    ibov_c = ibov_ret.loc[common]
    cdi_c = cdi_ret.loc[common]

    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    deferred_ret = value_to_ret(deferred_val)

    total_tax_std = result_std["tax_paid"].sum()
    total_tax_deferred = result_deferred["tax_paid"].sum()

    # ── Metrics ───────────────────────────────────────────────────
    m_pretax = build_metrics(pretax_ret, "Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(aftertax_ret, "15% CGT", PERIODS_PER_YEAR)
    m_deferred = build_metrics(deferred_ret, "Deferred DARF", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_c, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_c, "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_deferred, m_ibov, m_cdi])

    nav_benefit = deferred_val.iloc[-1] - aftertax_val.iloc[-1]
    print(f"\n  Total Tax (Standard 15%):     R$ {total_tax_std:>12,.2f}")
    print(f"  Total Tax (Deferred DARF):    R$ {total_tax_deferred:>12,.2f}")
    print(f"  NAV Benefit from Deferral:    R$ {nav_benefit:>12,.2f}")

    # ── Plot ──────────────────────────────────────────────────────
    plot_triple_blend(
        title=(
            f"Triple Blend: SmallCap Mom + Multifactor + CDI (33/33/33)\n"
            f"SC: {LOOKBACK_PERIODS_SC}M Mom · Skip {SKIP_PERIODS_SC} · "
            f"Bot 50% ADTV · Top {int(TOP_QUINTILE_SC * 100)}%  |  "
            f"MF: {LOOKBACK_PERIODS_MF}M · Top {int(TOP_DECILE_MF * 100)}%  |  "
            f"R$ {MIN_ADTV / 1_000_000:.0f}M+ Floor · "
            f"{SLIPPAGE * 100:.1f}% Slip\n"
            f"{START_DATE[:4]}-{END_DATE[:4]}"
        ),
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        deferred_val=deferred_val,
        ibov_ret=ibov_c,
        cdi_ret=cdi_c,
        tax_paid_std=result_std["tax_paid"].loc[common],
        tax_paid_deferred=result_deferred["tax_paid"].loc[common],
        loss_cf_std=result_std["loss_carryforward"].loc[common],
        loss_cf_deferred=result_deferred["loss_carryforward"].loc[common],
        turnover=result_std["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_deferred, m_ibov, m_cdi],
        total_tax_std=total_tax_std,
        total_tax_deferred=total_tax_deferred,
        out_path="triple_blend_backtest.png",
    )


if __name__ == "__main__":
    main()
