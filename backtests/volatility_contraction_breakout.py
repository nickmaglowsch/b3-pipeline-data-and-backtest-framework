"""
Volatility Contraction Breakout (Brazil Edition)
=================================================
Daily-frequency backtest that captures the compression -> expansion cycle
common in B3's commodity-heavy market.

Entry: ATR is in the lowest 10th percentile of its own 252-day history AND
       price breaks above a 60-day high.
Exit:  ATR-based Chandelier trailing stop (3x ATR).

State-based position tracking with trailing stops — different from the
rank-and-select pattern used by the monthly backtests.

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

from core.data import load_b3_hlc_data, download_benchmark, download_cdi_daily
from core.metrics import build_metrics, cumret, display_metrics_table, value_to_ret
from core.plotting import PALETTE, fmt_ax
from core.simulation import run_simulation

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
ATR_PERIOD = 20           # 20-day ATR (EWM span)
ATR_HISTORY = 252         # 1-year rolling window for percentile
ATR_PERCENTILE_THR = 0.10 # Lowest 10th percentile = contraction
BREAKOUT_WINDOW = 60      # 60-day high breakout
TRAIL_ATR_MULT = 3.0      # Chandelier-style trailing stop multiplier
MAX_POSITIONS = 20        # Avoid over-dilution
MIN_ADTV = 1_000_000     # R$1M daily liquidity floor
SLIPPAGE = 0.002          # 0.2% — higher than monthly due to daily frequency
TAX_RATE = 0.15
INITIAL_CAPITAL = 100_000
PERIODS_PER_YEAR = 252    # Daily

START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


def generate_signals(adj_close, split_high, split_low, split_close, close_px, fin_vol):
    """
    State-based volatility contraction breakout signal generation.

    Pre-computes ATR, contraction thresholds, breakout levels, and ADTV
    vectorized, then runs a day-by-day loop tracking positions with
    trailing stops.

    Returns:
        (daily_returns, target_weights) — both daily-frequency DataFrames.
    """
    # ── Pre-computation (vectorized) ─────────────────────────────────

    # 1. True Range using split-adjusted HLC (avoids ex-div spikes)
    prev_close = split_close.shift(1)
    tr1 = split_high - split_low
    tr2 = (split_high - prev_close).abs()
    tr3 = (split_low - prev_close).abs()
    true_range = pd.concat([tr1, tr2, tr3]).groupby(level=0).max()
    # Reindex to match the original index (groupby may reorder)
    true_range = true_range.reindex(split_close.index)

    # 2. ATR: Exponential weighted mean
    atr = true_range.ewm(span=ATR_PERIOD, min_periods=ATR_PERIOD).mean()

    # 3. ATR contraction threshold: 10th percentile over 252-day window
    atr_threshold = atr.rolling(ATR_HISTORY, min_periods=ATR_HISTORY).quantile(
        ATR_PERCENTILE_THR
    )

    # 4. 60-day high (prior 60 days, excluding today)
    rolling_high_60 = split_close.shift(1).rolling(BREAKOUT_WINDOW, min_periods=BREAKOUT_WINDOW).max()

    # 5. Rolling ADTV (20-day average)
    adtv = fin_vol.rolling(20, min_periods=10).mean()

    # 6. Daily returns from dividend-adjusted close (for simulation)
    daily_ret = adj_close.pct_change()

    # ── Day-by-day loop (state-based) ────────────────────────────────
    warmup = ATR_HISTORY + ATR_PERIOD + 1
    dates = split_close.index
    tickers = split_close.columns

    target_weights = pd.DataFrame(0.0, index=dates, columns=tickers)

    # Track positions: {ticker: {"trail_stop": float}}
    positions = {}

    print(f"  Running signal generation ({len(dates)} trading days, warmup={warmup})...")

    for t_idx in range(warmup, len(dates)):
        date = dates[t_idx]

        close_row = split_close.iloc[t_idx]
        atr_row = atr.iloc[t_idx]
        atr_thr_row = atr_threshold.iloc[t_idx]
        high60_row = rolling_high_60.iloc[t_idx]
        adtv_row = adtv.iloc[t_idx]
        raw_close_row = close_px.iloc[t_idx]
        fvol_row = fin_vol.iloc[t_idx]

        # ── A. Check exits ───────────────────────────────────────────
        exits = []
        for ticker, pos in positions.items():
            c = close_row.get(ticker, np.nan)
            a = atr_row.get(ticker, np.nan)

            # Exit if delisted (NaN close or ATR)
            if pd.isna(c) or pd.isna(a):
                exits.append(ticker)
                continue

            # Exit if close fell below trailing stop
            if c < pos["trail_stop"]:
                exits.append(ticker)
                continue

            # Still held: ratchet up trailing stop (never down)
            new_stop = c - TRAIL_ATR_MULT * a
            pos["trail_stop"] = max(pos["trail_stop"], new_stop)

        for ticker in exits:
            del positions[ticker]

        # ── B. Check entries ─────────────────────────────────────────
        if len(positions) < MAX_POSITIONS:
            candidates = []

            for ticker in tickers:
                if ticker in positions:
                    continue

                c = close_row.get(ticker, np.nan)
                a = atr_row.get(ticker, np.nan)
                a_thr = atr_thr_row.get(ticker, np.nan)
                h60 = high60_row.get(ticker, np.nan)
                av = adtv_row.get(ticker, np.nan)
                rc = raw_close_row.get(ticker, np.nan)
                fv = fvol_row.get(ticker, np.nan)

                # Skip if any required data is missing
                if pd.isna(c) or pd.isna(a) or pd.isna(a_thr) or pd.isna(h60):
                    continue
                if pd.isna(av) or pd.isna(rc) or pd.isna(fv):
                    continue

                # ATR contraction: current ATR <= 10th percentile threshold
                if a > a_thr:
                    continue

                # Breakout: close above 60-day high
                if c <= h60:
                    continue

                # Liquidity filters
                if av < MIN_ADTV:
                    continue
                if rc < 1.0:
                    continue

                # Must have actually traded (avoids false breakouts from ffill)
                if fv <= 0:
                    continue

                # Compute ATR percentile rank for prioritization
                atr_hist = atr.iloc[max(0, t_idx - ATR_HISTORY):t_idx + 1]
                if ticker in atr_hist.columns:
                    col = atr_hist[ticker].dropna()
                    if len(col) > 0:
                        pct_rank = (col < a).sum() / len(col)
                    else:
                        pct_rank = 1.0
                else:
                    pct_rank = 1.0

                candidates.append((ticker, pct_rank, c, a))

            # Prioritize by lowest ATR percentile (deepest contraction)
            candidates.sort(key=lambda x: x[1])

            slots = MAX_POSITIONS - len(positions)
            for ticker, _, c, a in candidates[:slots]:
                stop = c - TRAIL_ATR_MULT * a
                positions[ticker] = {"trail_stop": stop}

        # ── C. Set weights ───────────────────────────────────────────
        if positions:
            w = 1.0 / len(positions)
            for ticker in positions:
                target_weights.at[date, ticker] = w

    n_signals = (target_weights > 0).any(axis=1).sum()
    print(f"  Signal generation complete. Active days: {n_signals}")

    return daily_ret, target_weights


def plot_volatility_breakout(
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
    out_path="volatility_contraction_breakout.png",
):
    """4-row, 5-curve tear sheet adapted for daily frequency."""
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

    # Resample daily tax/turnover to monthly for bar chart readability
    tx_std = tax_paid_std.loc[common].resample("ME").sum()
    tx_df = tax_paid_deferred.loc[common].resample("ME").sum()
    lc_std = loss_cf_std.loc[common].resample("ME").last()
    lc_df = loss_cf_deferred.loc[common].resample("ME").last()
    tv = turnover.loc[common].resample("ME").mean()

    # Normalize to growth-of-R$1
    pt_curve = pt_val / pt_val.iloc[0]
    at_curve = at_val / at_val.iloc[0]
    df_curve = df_val / df_val.iloc[0]
    ibov_curve = cumret(ibov)
    cdi_curve = cumret(cdi)

    # ── Row 0: Cumulative Returns ──────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(
        pt_curve.index, pt_curve.values,
        color=PALETTE["pretax"], lw=2.2, label="Pre-Tax", zorder=5,
    )
    ax0.plot(
        at_curve.index, at_curve.values,
        color=PALETTE["aftertax"], lw=2.2, label="15% CGT", zorder=4, ls="-.",
    )
    ax0.plot(
        df_curve.index, df_curve.values,
        color=EXEMPT_COLOR, lw=2.2, label="Deferred DARF", zorder=3, ls="--",
    )
    ax0.plot(
        ibov_curve.index, ibov_curve.values,
        color=PALETTE["ibov"], lw=1.8, label="IBOV", zorder=2, ls="--",
    )
    ax0.plot(
        cdi_curve.index, cdi_curve.values,
        color="#A8B2C1", lw=1.5, label="CDI", zorder=1, ls=":",
    )

    ax0.fill_between(
        pt_curve.index, pt_curve.values, at_curve.values,
        alpha=0.12, color=PALETTE["tax"], label="Tax Drag (std)",
    )
    ax0.fill_between(
        at_curve.index, at_curve.values, df_curve.values,
        alpha=0.10, color=EXEMPT_COLOR, label="Deferral Benefit",
    )

    ax0.set_title(
        "Cumulative Return", color=PALETTE["text"],
        fontsize=13, fontweight="bold", pad=10,
    )
    ax0.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8.5, ncol=4,
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
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8.5, ncol=5,
    )
    fmt_ax(ax1, ylabel="Drawdown (%)")

    # ── Row 2: Tax, Loss CF, Turnover (monthly bars) ──────────────
    ax2a = fig.add_subplot(gs[2, 0])
    w = 12
    ax2a.bar(
        tx_std.index - pd.Timedelta(days=7), tx_std.values / 1_000,
        color=PALETTE["tax"], width=w, alpha=0.8, label="Std 15%",
    )
    ax2a.bar(
        tx_df.index + pd.Timedelta(days=7), tx_df.values / 1_000,
        color=EXEMPT_COLOR, width=w, alpha=0.8, label="Deferred DARF",
    )
    ax2a.set_title(
        "Tax Paid per Month (R$ k)", color=PALETTE["text"], fontsize=10, pad=6,
    )
    ax2a.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8,
    )
    fmt_ax(ax2a, ylabel="R$ thousands")

    ax2b = fig.add_subplot(gs[2, 1])
    ax2b.plot(
        lc_std.index, lc_std.values / 1_000,
        color=PALETTE["loss_cf"], lw=1.4, label="Std 15%",
    )
    ax2b.fill_between(
        lc_std.index, lc_std.values / 1_000, 0, alpha=0.3, color=PALETTE["loss_cf"],
    )
    ax2b.plot(
        lc_df.index, lc_df.values / 1_000,
        color=EXEMPT_COLOR, lw=1.4, ls="--", label="Deferred DARF",
    )
    ax2b.fill_between(
        lc_df.index, lc_df.values / 1_000, 0, alpha=0.2, color=EXEMPT_COLOR,
    )
    ax2b.set_title(
        "Loss Carryforward (R$ k)", color=PALETTE["text"], fontsize=10, pad=6,
    )
    ax2b.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8,
    )
    fmt_ax(ax2b, ylabel="R$ thousands")

    ax2c = fig.add_subplot(gs[2, 2])
    ax2c.bar(tv.index, tv.values * 100, color=PALETTE["pretax"], width=20, alpha=0.7)
    ax2c.set_title(
        "Monthly Avg Turnover (%)", color=PALETTE["text"], fontsize=10, pad=6,
    )
    fmt_ax(ax2c, ylabel="%")

    # ── Row 3: Cumulative Tax Drag + Metrics Table ────────────────
    ax3a = fig.add_subplot(gs[3, 0:2])
    spread_std = (pt_curve - at_curve) * 100
    spread_df = (pt_curve - df_curve) * 100
    ax3a.fill_between(
        spread_std.index, spread_std.values, 0,
        alpha=0.45, color=PALETTE["tax"], label="Std 15% drag",
    )
    ax3a.plot(spread_std.index, spread_std.values, color=PALETTE["tax"], lw=1.4)
    ax3a.fill_between(
        spread_df.index, spread_df.values, 0,
        alpha=0.30, color=EXEMPT_COLOR, label="Deferred DARF drag",
    )
    ax3a.plot(spread_df.index, spread_df.values, color=EXEMPT_COLOR, lw=1.4, ls="--")
    ax3a.axhline(0, color=PALETTE["sub"], lw=0.8)
    ax3a.set_title(
        "Cumulative Tax Drag (pp)", color=PALETTE["text"], fontsize=10, pad=6,
    )
    ax3a.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8.5,
    )
    fmt_ax(ax3a, ylabel="pp")

    nav_benefit = deferred_val.iloc[-1] - aftertax_val.iloc[-1]
    ax3a.annotate(
        f"Std tax: R$ {total_tax_std:,.0f}  |  Deferred tax: R$ {total_tax_deferred:,.0f}  |  NAV benefit: R$ {nav_benefit:,.0f}",
        xy=(0.02, 0.88), xycoords="axes fraction", fontsize=8.5, color=EXEMPT_COLOR,
        bbox=dict(
            boxstyle="round,pad=0.3", fc=PALETTE["panel"], ec=EXEMPT_COLOR, alpha=0.8,
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
        cellText=row_vals, colLabels=col_labels, cellLoc="center", loc="center",
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
        "Performance Summary", color=PALETTE["text"], fontsize=10, pad=6, y=0.98,
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
    print("  VOLATILITY CONTRACTION BREAKOUT (BRAZIL EDITION)")
    print("  Daily Frequency | ATR Contraction + 60-Day Breakout")
    print("  3 Tax Scenarios: Pre-Tax / 15% CGT / Deferred DARF")
    print("=" * 70)

    adj_close, split_high, split_low, split_close, close_px, fin_vol = load_b3_hlc_data(
        DB_PATH, START_DATE, END_DATE
    )

    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    # Daily IBOV returns
    ibov_ret = ibov_px.pct_change().dropna()
    ibov_ret.name = "IBOV"

    print("\n  Generating volatility contraction breakout signals...")
    daily_ret, target_weights = generate_signals(
        adj_close, split_high, split_low, split_close, close_px, fin_vol
    )
    daily_ret = daily_ret.fillna(0.0)

    # ── Run 1: Standard 15% CGT (also gives pre-tax curve) ────────
    print("\n  Running simulation: Standard 15% CGT...")
    result_std = run_simulation(
        returns_matrix=daily_ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="Vol Contraction Breakout",
    )

    # ── Run 2: 15% CGT with deferred DARF payment ───────────────
    print("  Running simulation: 15% CGT + Deferred DARF...")
    result_deferred = run_simulation(
        returns_matrix=daily_ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="Vol Contraction Breakout (Deferred DARF)",
        defer_tax=True,
    )

    # ── Align to common index ─────────────────────────────────────
    common = (
        result_std["pretax_values"]
        .index.intersection(ibov_ret.index)
        .intersection(cdi_daily.index)
    )

    pretax_val = result_std["pretax_values"].loc[common]
    aftertax_val = result_std["aftertax_values"].loc[common]
    deferred_val = result_deferred["aftertax_values"].loc[common]
    ibov_c = ibov_ret.loc[common]
    cdi_c = cdi_daily.loc[common]

    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    deferred_ret = value_to_ret(deferred_val)

    total_tax_std = result_std["tax_paid"].sum()
    total_tax_deferred = result_deferred["tax_paid"].sum()

    # ── Metrics (daily frequency) ─────────────────────────────────
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
    plot_volatility_breakout(
        title=(
            f"Volatility Contraction Breakout (Brazil Edition)  ·  "
            f"ATR({ATR_PERIOD}) < P{int(ATR_PERCENTILE_THR * 100)} over {ATR_HISTORY}d\n"
            f"{BREAKOUT_WINDOW}-Day Breakout · {TRAIL_ATR_MULT}x ATR Trail Stop · "
            f"Max {MAX_POSITIONS} Positions  ·  "
            f"{SLIPPAGE * 100:.1f}% Slippage\n"
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
        out_path="volatility_contraction_breakout.png",
    )


if __name__ == "__main__":
    main()
