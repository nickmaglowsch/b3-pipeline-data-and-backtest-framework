"""
Beta Adjusted by Volatility Regime — B3 Backtest
=================================================
Tests whether beta exposure should rotate with IBOV volatility regimes:
- High-vol regime → low-beta stocks (defensive)
- Low-vol regime → high-beta stocks (aggressive)

Compares a regime-switching strategy against two naive baselines
(always-high-beta, always-low-beta) plus IBOV and CDI.
Monthly frequency with dynamic vol threshold (rolling median).
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
from datetime import datetime

import matplotlib.dates as mdates
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
REBALANCE_FREQ = "ME"
BETA_WINDOW = 252          # 252-day rolling beta (daily returns)
IBOV_VOL_WINDOW = 63       # 3-month realized vol window
VOL_REGIME_LOOKBACK = 252  # rolling median of vol for dynamic threshold
TOP_BETA_PCT = 0.20        # top/bottom 20% by beta
MIN_ADTV = 1_000_000
SLIPPAGE = 0.001
TAX_RATE = 0.15
INITIAL_CAPITAL = 100_000

START_DATE = "2005-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")

PERIODS_PER_YEAR = 12
DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"

# Colors
COLOR_REGIME = PALETTE["pretax"]       # teal
COLOR_REGIME_AT = PALETTE["aftertax"]  # purple
COLOR_HIGH_BETA = "#FFD700"            # gold
COLOR_LOW_BETA = "#4FC3F7"             # sky blue
COLOR_IBOV = PALETTE["ibov"]           # orange
COLOR_CDI = "#A8B2C1"                  # grey


def compute_rolling_beta(daily_ret, ibov_daily_ret, window=BETA_WINDOW):
    """
    Compute rolling beta of each stock vs IBOV using daily returns.
    Returns a monthly-sampled DataFrame of betas.
    """
    min_periods = int(window * 0.8)

    # Align IBOV to the stock return index
    ibov_aligned = ibov_daily_ret.reindex(daily_ret.index)

    # Vectorized rolling covariance: per-column cov with ibov
    rolling_cov = daily_ret.rolling(window, min_periods=min_periods).cov(ibov_aligned)

    # Rolling variance of IBOV
    ibov_var = ibov_aligned.rolling(window, min_periods=min_periods).var()

    # Beta = cov(stock, ibov) / var(ibov)
    beta_daily = rolling_cov.div(ibov_var, axis=0)

    # Resample to month-end
    return beta_daily.resample("ME").last()


def compute_vol_regime(ibov_daily_ret, vol_window=IBOV_VOL_WINDOW,
                       regime_lookback=VOL_REGIME_LOOKBACK):
    """
    Classify IBOV volatility regime using a dynamic threshold (rolling median).

    Returns (regime_monthly, realized_vol_monthly) — both shifted(1) to avoid
    look-ahead bias.

    Dynamic threshold rationale: B3 structural vol varies enormously
    (2008 crisis vs 2017 calm). A fixed threshold would over-classify one era.
    Rolling median ensures ~50/50 split over any window.
    """
    # Annualized realized vol
    realized_vol = ibov_daily_ret.rolling(vol_window).std() * np.sqrt(252)

    # Dynamic threshold: rolling median
    vol_median = realized_vol.rolling(regime_lookback).median()

    # High vol = realized vol > its rolling median
    is_high_vol = realized_vol > vol_median

    # Resample to monthly and shift to avoid look-ahead
    regime_monthly = is_high_vol.resample("ME").last().shift(1)
    realized_vol_monthly = realized_vol.resample("ME").last().shift(1)

    return regime_monthly, realized_vol_monthly


def generate_signals(adj_close, close_px, fin_vol, ibov_px):
    """
    Build target weights for 3 strategies:
    1. Regime-Switch: low-beta in HIGH_VOL, high-beta in LOW_VOL
    2. Always High-Beta baseline
    3. Always Low-Beta baseline

    Returns:
        (ret, tw_regime, tw_high_beta, tw_low_beta,
         regime_monthly, ibov_vol, portfolio_betas)
    """
    # Daily returns for beta computation (keep NaNs for beta calc)
    daily_ret = adj_close.pct_change()
    ibov_daily_ret = ibov_px.pct_change().dropna()

    # Rolling beta (monthly sampled)
    print("    Computing rolling betas...")
    beta_monthly = compute_rolling_beta(daily_ret, ibov_daily_ret, window=BETA_WINDOW)

    # Volatility regime
    print("    Computing volatility regime...")
    regime_monthly, ibov_vol = compute_vol_regime(ibov_daily_ret)

    # Resample to monthly
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()
    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    # Align beta_monthly to ret index
    beta_monthly = beta_monthly.reindex(ret.index, method="ffill")

    # Align regime to ret index
    regime_monthly = regime_monthly.reindex(ret.index, method="ffill")
    ibov_vol = ibov_vol.reindex(ret.index, method="ffill")

    # Glitch filter
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(2).max()

    # Initialize target weights
    tw_regime = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_high_beta = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_low_beta = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)

    # Track portfolio betas for diagnostic
    regime_betas = pd.Series(np.nan, index=ret.index)
    high_beta_betas = pd.Series(np.nan, index=ret.index)
    low_beta_betas = pd.Series(np.nan, index=ret.index)

    start_idx = 25  # enough warmup for beta + regime

    for i in range(start_idx, len(ret)):
        beta_row = beta_monthly.iloc[i - 1]  # previous month's beta
        regime = regime_monthly.iloc[i]       # already shifted inside compute_vol_regime
        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]
        glitch_row = has_glitch.iloc[i - 1] if i > 0 else pd.Series(0, index=ret.columns)

        # Liquidity filter
        valid_mask = (
            (adtv_row >= MIN_ADTV) &
            (raw_close_row >= 1.0) &
            (glitch_row != 1)
        )

        beta_valid = beta_row[valid_mask].dropna()

        if len(beta_valid) < 10:
            continue

        n_high = max(1, int(len(beta_valid) * TOP_BETA_PCT))
        n_low = max(1, int(len(beta_valid) * TOP_BETA_PCT))

        high_beta_stocks = beta_valid.nlargest(n_high).index
        low_beta_stocks = beta_valid.nsmallest(n_low).index

        # Always High-Beta
        w_high = 1.0 / len(high_beta_stocks)
        for t in high_beta_stocks:
            tw_high_beta.iloc[i, tw_high_beta.columns.get_loc(t)] = w_high

        # Always Low-Beta
        w_low = 1.0 / len(low_beta_stocks)
        for t in low_beta_stocks:
            tw_low_beta.iloc[i, tw_low_beta.columns.get_loc(t)] = w_low

        # Regime-Switch: low-beta if HIGH_VOL, high-beta if LOW_VOL
        if pd.isna(regime):
            # Default to high-beta if regime not yet available
            selected = high_beta_stocks
        elif regime:  # HIGH_VOL → low-beta
            selected = low_beta_stocks
        else:         # LOW_VOL → high-beta
            selected = high_beta_stocks

        w_regime = 1.0 / len(selected)
        for t in selected:
            tw_regime.iloc[i, tw_regime.columns.get_loc(t)] = w_regime

        # Track median portfolio beta
        high_beta_betas.iloc[i] = beta_row[high_beta_stocks].median()
        low_beta_betas.iloc[i] = beta_row[low_beta_stocks].median()
        regime_betas.iloc[i] = beta_row[selected].median()

    portfolio_betas = {
        "Regime-Switch": regime_betas,
        "High-Beta": high_beta_betas,
        "Low-Beta": low_beta_betas,
    }

    return (ret, tw_regime, tw_high_beta, tw_low_beta,
            regime_monthly, ibov_vol, portfolio_betas)


def plot_beta_regime(
    title,
    regime_val,
    regime_at_val,
    highbeta_val,
    lowbeta_val,
    ibov_ret,
    cdi_ret,
    regime_monthly,
    ibov_vol,
    portfolio_betas,
    turnover_regime,
    turnover_highbeta,
    turnover_lowbeta,
    metrics,
    out_path="beta_volatility_regime_backtest.png",
):
    """Custom 4-row tear sheet for beta-regime backtest."""
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "figure.facecolor": PALETTE["bg"],
            "text.color": PALETTE["text"],
            "axes.facecolor": PALETTE["panel"],
        }
    )

    fig = plt.figure(figsize=(18, 22))
    gs = GridSpec(
        4, 3,
        figure=fig,
        hspace=0.48,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.93,
        bottom=0.04,
    )

    # Normalize to growth-of-R$1
    r_curve = regime_val / regime_val.iloc[0]
    rat_curve = regime_at_val / regime_at_val.iloc[0]
    hb_curve = highbeta_val / highbeta_val.iloc[0]
    lb_curve = lowbeta_val / lowbeta_val.iloc[0]
    ibov_curve = cumret(ibov_ret)
    cdi_curve = cumret(cdi_ret.fillna(0))

    # ── Row 0: Cumulative Returns ──────────────────────────────────
    ax0 = fig.add_subplot(gs[0, :])
    ax0.plot(r_curve.index, r_curve.values, color=COLOR_REGIME,
             lw=2.2, label="Regime-Switch", zorder=6)
    ax0.plot(rat_curve.index, rat_curve.values, color=COLOR_REGIME_AT,
             lw=2.2, label="Regime-Switch (AT)", zorder=5, ls="-.")
    ax0.plot(hb_curve.index, hb_curve.values, color=COLOR_HIGH_BETA,
             lw=1.8, label="Always High-Beta", zorder=4, ls="--")
    ax0.plot(lb_curve.index, lb_curve.values, color=COLOR_LOW_BETA,
             lw=1.8, label="Always Low-Beta", zorder=3, ls="--")
    ax0.plot(ibov_curve.index, ibov_curve.values, color=COLOR_IBOV,
             lw=1.8, label="IBOV", zorder=2, ls="--")
    ax0.plot(cdi_curve.index, cdi_curve.values, color=COLOR_CDI,
             lw=1.5, label="CDI", zorder=1, ls=":")

    # Tax drag fill
    ax0.fill_between(
        r_curve.index, r_curve.values, rat_curve.values,
        alpha=0.12, color=PALETTE["tax"], label="Tax Drag",
    )

    ax0.set_title("Cumulative Return", color=PALETTE["text"],
                   fontsize=13, fontweight="bold", pad=10)
    ax0.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8.5, ncol=4,
    )
    fmt_ax(ax0, ylabel="Growth of R$1")

    # ── Row 1: Drawdowns ──────────────────────────────────────────
    ax1 = fig.add_subplot(gs[1, :])
    for curve, label, color, alpha in [
        (r_curve, "Regime DD", COLOR_REGIME, 0.45),
        (rat_curve, "Regime AT DD", COLOR_REGIME_AT, 0.35),
        (hb_curve, "High-Beta DD", COLOR_HIGH_BETA, 0.30),
        (lb_curve, "Low-Beta DD", COLOR_LOW_BETA, 0.25),
        (ibov_curve, "IBOV DD", COLOR_IBOV, 0.22),
        (cdi_curve, "CDI DD", COLOR_CDI, 0.18),
    ]:
        dd = (curve / curve.cummax() - 1) * 100
        ax1.fill_between(dd.index, dd.values, 0, alpha=alpha, color=color, label=label)

    ax1.set_title("Drawdown (%)", color=PALETTE["text"], fontsize=11, pad=6)
    ax1.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8.5, ncol=6,
    )
    fmt_ax(ax1, ylabel="Drawdown (%)")

    # ── Row 2 Left (2/3): Volatility Regime Panel ─────────────────
    ax2a = fig.add_subplot(gs[2, 0:2])
    vol_common = ibov_vol.dropna()
    regime_common = regime_monthly.reindex(vol_common.index)

    ax2a.plot(vol_common.index, vol_common.values * 100, color=PALETTE["text"],
              lw=1.2, label="IBOV Realized Vol", zorder=3)

    # Background shading for regime
    for i in range(len(vol_common)):
        if pd.isna(regime_common.iloc[i]):
            continue
        c = "#FF4C4C" if regime_common.iloc[i] else "#4CAF50"
        a = 0.15
        if i < len(vol_common) - 1:
            ax2a.axvspan(vol_common.index[i], vol_common.index[i + 1],
                         color=c, alpha=a, lw=0)
        else:
            ax2a.axvspan(vol_common.index[i],
                         vol_common.index[i] + pd.Timedelta(days=30),
                         color=c, alpha=a, lw=0)

    # Invisible patches for legend
    from matplotlib.patches import Patch
    ax2a.legend(
        handles=[
            plt.Line2D([0], [0], color=PALETTE["text"], lw=1.2, label="IBOV Real. Vol"),
            Patch(facecolor="#FF4C4C", alpha=0.3, label="HIGH VOL"),
            Patch(facecolor="#4CAF50", alpha=0.3, label="LOW VOL"),
        ],
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8,
    )
    ax2a.set_title("IBOV Volatility Regime", color=PALETTE["text"], fontsize=10, pad=6)
    fmt_ax(ax2a, ylabel="Annualized Vol (%)")

    # ── Row 2 Right (1/3): Portfolio Beta Over Time ───────────────
    ax2b = fig.add_subplot(gs[2, 2])
    for label, color, series in [
        ("Regime-Switch", COLOR_REGIME, portfolio_betas["Regime-Switch"]),
        ("High-Beta", COLOR_HIGH_BETA, portfolio_betas["High-Beta"]),
        ("Low-Beta", COLOR_LOW_BETA, portfolio_betas["Low-Beta"]),
    ]:
        s = series.dropna()
        ax2b.plot(s.index, s.values, color=color, lw=1.2, label=label, alpha=0.85)

    ax2b.set_title("Portfolio Beta", color=PALETTE["text"], fontsize=10, pad=6)
    ax2b.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=7.5,
    )
    fmt_ax(ax2b, ylabel="Median Beta")

    # ── Row 3 Left (2/3): Monthly Turnover ─────────────────────────
    ax3a = fig.add_subplot(gs[3, 0:2])
    w = 8
    ax3a.bar(turnover_regime.index - pd.Timedelta(days=10),
             turnover_regime.values * 100,
             color=COLOR_REGIME, width=w, alpha=0.8, label="Regime-Switch")
    ax3a.bar(turnover_highbeta.index,
             turnover_highbeta.values * 100,
             color=COLOR_HIGH_BETA, width=w, alpha=0.8, label="High-Beta")
    ax3a.bar(turnover_lowbeta.index + pd.Timedelta(days=10),
             turnover_lowbeta.values * 100,
             color=COLOR_LOW_BETA, width=w, alpha=0.8, label="Low-Beta")
    ax3a.set_title("Monthly Portfolio Turnover (%)", color=PALETTE["text"],
                    fontsize=10, pad=6)
    ax3a.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8,
    )
    fmt_ax(ax3a, ylabel="%")

    # ── Row 3 Right (1/3): Metrics Table ──────────────────────────
    ax3b = fig.add_subplot(gs[3, 2])
    ax3b.axis("off")
    col_labels = list(metrics[0].keys())
    row_vals = [[str(m[k]) for k in col_labels] for m in metrics]

    row_bg = ["#0D2E26", "#1A1230", "#2A2010", "#1A2A3A", "#2A1A10", "#1E2A3A"]
    txt_colors = [
        COLOR_REGIME, COLOR_REGIME_AT, COLOR_HIGH_BETA,
        COLOR_LOW_BETA, COLOR_IBOV, COLOR_CDI,
    ]

    tbl = ax3b.table(
        cellText=row_vals, colLabels=col_labels, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
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

    ax3b.set_title("Performance Summary", color=PALETTE["text"],
                    fontsize=10, pad=6, y=0.98)

    fig.suptitle(title, fontsize=12, fontweight="bold", color=PALETTE["text"], y=0.98)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"\n  Plot saved -> {out_path}")

    try:
        plt.show()
    except Exception:
        pass


def main():
    print("\n" + "=" * 70)
    print("  B3 BETA ADJUSTED BY VOLATILITY REGIME BACKTEST")
    print("  Regime-Switch vs Always-High-Beta vs Always-Low-Beta")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"

    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n  Generating beta-regime signals...")
    (ret, tw_regime, tw_high_beta, tw_low_beta,
     regime_monthly, ibov_vol, portfolio_betas) = generate_signals(
        adj_close, close_px, fin_vol, ibov_px
    )
    ret = ret.fillna(0.0)

    # ── Regime Breakdown ─────────────────────────────────────────
    regime_valid = regime_monthly.dropna()
    n_high = regime_valid.sum()
    n_low = len(regime_valid) - n_high
    print(f"\n  Regime breakdown: HIGH_VOL = {int(n_high)} months, "
          f"LOW_VOL = {int(n_low)} months "
          f"({n_high / len(regime_valid) * 100:.1f}% / {n_low / len(regime_valid) * 100:.1f}%)")

    # ── Run 1: Regime-Switch (Pre-Tax) ────────────────────────────
    print("\n  Running simulation: Regime-Switch (Pre-Tax)...")
    result_regime = run_simulation(
        returns_matrix=ret,
        target_weights=tw_regime,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=0.0,
        slippage=SLIPPAGE,
        name="Regime-Switch",
    )

    # ── Run 2: Always High-Beta (Pre-Tax) ─────────────────────────
    print("  Running simulation: Always High-Beta (Pre-Tax)...")
    result_high = run_simulation(
        returns_matrix=ret,
        target_weights=tw_high_beta,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=0.0,
        slippage=SLIPPAGE,
        name="Always High-Beta",
    )

    # ── Run 3: Always Low-Beta (Pre-Tax) ──────────────────────────
    print("  Running simulation: Always Low-Beta (Pre-Tax)...")
    result_low = run_simulation(
        returns_matrix=ret,
        target_weights=tw_low_beta,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=0.0,
        slippage=SLIPPAGE,
        name="Always Low-Beta",
    )

    # ── Run 4: Regime-Switch (After-Tax) ──────────────────────────
    print("  Running simulation: Regime-Switch (After-Tax 15%)...")
    result_regime_at = run_simulation(
        returns_matrix=ret,
        target_weights=tw_regime,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="Regime-Switch (AT)",
    )

    # ── Align to common index ─────────────────────────────────────
    common = (
        result_regime["pretax_values"].index
        .intersection(result_high["pretax_values"].index)
        .intersection(result_low["pretax_values"].index)
        .intersection(result_regime_at["aftertax_values"].index)
        .intersection(ibov_ret.index)
        .intersection(cdi_ret.index)
    )

    regime_val = result_regime["pretax_values"].loc[common]
    regime_at_val = result_regime_at["aftertax_values"].loc[common]
    highbeta_val = result_high["pretax_values"].loc[common]
    lowbeta_val = result_low["pretax_values"].loc[common]
    ibov_c = ibov_ret.loc[common]
    cdi_c = cdi_ret.loc[common]

    regime_ret = value_to_ret(regime_val)
    regime_at_ret = value_to_ret(regime_at_val)
    highbeta_ret = value_to_ret(highbeta_val)
    lowbeta_ret = value_to_ret(lowbeta_val)

    total_tax = result_regime_at["tax_paid"].sum()

    # ── Metrics ───────────────────────────────────────────────────
    m_regime = build_metrics(regime_ret, "Regime-Switch", PERIODS_PER_YEAR)
    m_regime_at = build_metrics(regime_at_ret, "Regime (AT)", PERIODS_PER_YEAR)
    m_high = build_metrics(highbeta_ret, "High-Beta", PERIODS_PER_YEAR)
    m_low = build_metrics(lowbeta_ret, "Low-Beta", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_c, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_c, "CDI", PERIODS_PER_YEAR)

    all_metrics = [m_regime, m_regime_at, m_high, m_low, m_ibov, m_cdi]
    display_metrics_table(all_metrics)

    print(f"\n  Total Tax (Regime-Switch AT):  R$ {total_tax:>12,.2f}")

    # ── Plot ──────────────────────────────────────────────────────
    plot_beta_regime(
        title=(
            f"Beta Adjusted by Volatility Regime (B3)  ·  "
            f"{BETA_WINDOW}d Beta · {IBOV_VOL_WINDOW}d Vol · "
            f"Top/Bottom {int(TOP_BETA_PCT * 100)}%\n"
            f"R$ {MIN_ADTV / 1_000_000:.0f}M+ Floor  ·  "
            f"{SLIPPAGE * 100:.1f}% Slippage  ·  "
            f"{START_DATE[:4]}–{END_DATE[:4]}"
        ),
        regime_val=regime_val,
        regime_at_val=regime_at_val,
        highbeta_val=highbeta_val,
        lowbeta_val=lowbeta_val,
        ibov_ret=ibov_c,
        cdi_ret=cdi_c,
        regime_monthly=regime_monthly,
        ibov_vol=ibov_vol,
        portfolio_betas=portfolio_betas,
        turnover_regime=result_regime["turnover"].loc[common],
        turnover_highbeta=result_high["turnover"].loc[common],
        turnover_lowbeta=result_low["turnover"].loc[common],
        metrics=all_metrics,
        out_path="beta_volatility_regime_backtest.png",
    )


if __name__ == "__main__":
    main()
