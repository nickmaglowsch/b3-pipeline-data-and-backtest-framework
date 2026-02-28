"""
Seasonal / Calendar Effects Strategy Backtest
===============================================
Tests four calendar anomalies on B3 data (2005-present):
  1. Turn-of-Month (TOM)         -- last 2 + first 3 trading days
  2. Monthly Seasonality         -- invest only in historically positive months
  3. December/January Effect     -- invest Nov-Jan (13th salary / year-end)
  4. Sell in May                 -- invest Oct-Apr, CDI May-Sep

All strategies use MultiFactor signal (top 10%) for stock selection during
equity periods, and CDI otherwise.

Also tests simpler IBOV-based variants for each effect.

Usage:
    python3 backtests/seasonal_effects_backtest.py
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

from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.simulation import run_simulation
from core.metrics import build_metrics, value_to_ret, cumret
from core.plotting import PALETTE, fmt_ax

# ─── Config ──────────────────────────────────────────────────────────────────
START = "2005-01-01"
from datetime import datetime
END = datetime.today().strftime("%Y-%m-%d")
FREQ = "ME"
CAPITAL = 100_000
TAX = 0.15
SLIP = 0.001
EXEMPTION = 20_000
MIN_ADTV = 1_000_000
MIN_PRICE = 1.0
LOOKBACK = 12

DB_PATH = os.path.join(_PROJECT_ROOT, "b3_market_data.sqlite")
OUT_DIR = _BACKTESTS_DIR


def run_and_collect(name, ret_matrix, weights, cdi_monthly, ibov_ret):
    """Run simulation and return metrics + equity curve."""
    result = run_simulation(
        returns_matrix=ret_matrix.fillna(0.0),
        target_weights=weights,
        initial_capital=CAPITAL,
        tax_rate=TAX,
        slippage=SLIP,
        name=name,
        monthly_sales_exemption=EXEMPTION,
    )
    at_val = result["aftertax_values"]
    at_ret = value_to_ret(at_val).dropna()
    m = build_metrics(at_ret, name, 12)
    return m, at_val, at_ret


def build_multifactor_weights(ret, log_ret, adtv, raw_close, has_glitch):
    """Build MultiFactor (mom+vol, top 10%) target weights DataFrame."""
    mom_sig = log_ret.shift(1).rolling(LOOKBACK).sum()
    mom_sig[has_glitch == 1] = np.nan
    vol_sig = -ret.shift(1).rolling(LOOKBACK).std()
    vol_sig[has_glitch == 1] = np.nan
    composite = mom_sig.rank(axis=1, pct=True) * 0.5 + vol_sig.rank(axis=1, pct=True) * 0.5

    tw_mf = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    for i in range(LOOKBACK + 2, len(ret)):
        sig_row = composite.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        valid = sig_row[mask].dropna()
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw_mf.columns:
                tw_mf.iloc[i, tw_mf.columns.get_loc(t)] = w
    return tw_mf


def main():
    print("\n" + "=" * 70)
    print("  Seasonal / Calendar Effects Backtest")
    print("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START, END)
    cdi_daily = download_cdi_daily(START, END)
    ibov_px = download_benchmark("^BVSP", START, END)

    ibov_ret_daily = ibov_px.pct_change()
    ibov_ret = ibov_px.resample(FREQ).last().pct_change().dropna()
    cdi_ret = (1 + cdi_daily).resample(FREQ).prod() - 1
    cdi_monthly = cdi_ret.copy()

    px = adj_close.resample(FREQ).last()
    ret = px.pct_change()
    raw_close = close_px.resample(FREQ).last()
    adtv = fin_vol.resample(FREQ).mean()
    log_ret = np.log1p(ret)
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(LOOKBACK).max()

    # ── Monthly seasonality analysis ─────────────────────────────────────────
    print("\nComputing monthly seasonality statistics for IBOV...")
    ibov_monthly = ibov_px.resample(FREQ).last().pct_change().dropna()
    month_labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    monthly_avgs = ibov_monthly.groupby(ibov_monthly.index.month).mean() * 100
    monthly_stds = ibov_monthly.groupby(ibov_monthly.index.month).std() * 100
    monthly_counts = ibov_monthly.groupby(ibov_monthly.index.month).count()
    monthly_tstat = monthly_avgs / (monthly_stds / np.sqrt(monthly_counts))

    print("\n  IBOV Monthly Seasonality (2005-present):")
    print(f"  {'Month':<6} {'Avg%':>8} {'Std%':>8} {'T-stat':>8} {'Positive%':>10}")
    print(f"  {'-'*45}")
    positive_months = set()
    for m_idx in range(1, 13):
        month_rets = ibov_monthly[ibov_monthly.index.month == m_idx]
        avg = monthly_avgs.get(m_idx, np.nan)
        std = monthly_stds.get(m_idx, np.nan)
        t = monthly_tstat.get(m_idx, np.nan)
        pct_pos = (month_rets > 0).mean() * 100
        marker = " *" if t > 1.5 else ""
        print(f"  {month_labels[m_idx-1]:<6} {avg:>7.2f}% {std:>7.2f}% {t:>7.2f}  {pct_pos:>7.1f}%{marker}")
        if avg > 0 and t > 0.5:
            positive_months.add(m_idx)

    print(f"\n  Historically positive months (avg > 0 and t > 0.5): "
          f"{[month_labels[m-1] for m in sorted(positive_months)]}")

    # ── Precompute MultiFactor weights (used in equity periods) ───────────────
    print("\nPrecomputing MultiFactor weights...")
    tw_mf = build_multifactor_weights(ret, log_ret, adtv, raw_close, has_glitch)

    results = []

    # ── Strategy 1: Turn-of-Month (TOM) ──────────────────────────────────────
    # We compute monthly returns earned only on TOM days vs all days
    print("\n[1/4] Turn-of-Month Strategy...")

    # For each month: return on TOM days vs non-TOM days using daily adj_close
    adj_daily_ret = adj_close.pct_change()
    trading_days_per_month = adj_daily_ret.resample("ME").count()

    # TOM = last 2 + first 3 trading days of each month
    # Identify which daily dates are TOM
    def get_tom_dates(index):
        """Return a boolean Series: True if date is a TOM day."""
        is_tom = pd.Series(False, index=index)
        months = pd.PeriodIndex(index, freq="M")
        for period in months.unique():
            month_dates = index[months == period]
            if len(month_dates) >= 2:
                # Last 2 trading days
                is_tom[month_dates[-2:]] = True
            if len(month_dates) >= 3:
                # First 3 trading days
                is_tom[month_dates[:3]] = True
        return is_tom

    tom_mask = get_tom_dates(adj_daily_ret.index)

    # Compute monthly TOM return vs full month return for IBOV
    ibov_daily = ibov_px.pct_change().dropna()
    tom_ibov_monthly_ret = ibov_daily[tom_mask].resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )
    non_tom_ibov_monthly_ret = ibov_daily[~tom_mask].resample("ME").apply(
        lambda x: (1 + x).prod() - 1
    )

    print(f"  TOM IBOV avg monthly return:     {tom_ibov_monthly_ret.mean()*100:.2f}%")
    print(f"  Non-TOM IBOV avg monthly return: {non_tom_ibov_monthly_ret.mean()*100:.2f}%")

    # Build TOM strategy: in equities (MultiFactor) during TOM months (simple monthly approx)
    # Use IBOV variant for simplicity (monthly framework)
    tw_tom = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_tom["CDI_ASSET"] = 0.0
    tw_tom["IBOV"] = 0.0
    r_tom = ret.copy()
    r_tom["CDI_ASSET"] = cdi_monthly
    r_tom["IBOV"] = ibov_ret

    # NOTE: The monthly framework cannot isolate intra-month TOM timing.
    # This strategy holds MultiFactor equities every month as a proxy, since
    # TOM days dominate monthly returns. The daily TOM analysis above quantifies
    # the actual TOM effect; this backtest just captures general equity exposure.
    for i in range(LOOKBACK + 2, len(ret)):
        # Use multifactor stock weights for the equity portion
        mf_row = tw_mf.iloc[i]
        if mf_row.sum() > 0:
            # Hold equities (TOM effect captured via stock selection)
            tw_tom.iloc[i] = mf_row
        else:
            tw_tom.iloc[i, tw_tom.columns.get_loc("CDI_ASSET")] = 1.0

    m_tom, at_tom, ret_tom = run_and_collect("TOM (MF stocks)", r_tom, tw_tom, cdi_monthly, ibov_ret)
    results.append(m_tom)

    # ── Strategy 2: Monthly Seasonality ──────────────────────────────────────
    print("\n[2/4] Monthly Seasonality Strategy...")
    tw_seas = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_seas["CDI_ASSET"] = 0.0
    tw_seas["IBOV"] = 0.0
    r_seas = ret.copy()
    r_seas["CDI_ASSET"] = cdi_monthly
    r_seas["IBOV"] = ibov_ret

    for i in range(LOOKBACK + 2, len(ret)):
        month = ret.index[i].month
        if month in positive_months:
            mf_row = tw_mf.iloc[i]
            if mf_row.sum() > 0:
                tw_seas.iloc[i] = mf_row
            else:
                tw_seas.iloc[i, tw_seas.columns.get_loc("IBOV")] = 1.0
        else:
            tw_seas.iloc[i, tw_seas.columns.get_loc("CDI_ASSET")] = 1.0

    m_seas, at_seas, ret_seas = run_and_collect("Seasonal (pos months)", r_seas, tw_seas, cdi_monthly, ibov_ret)
    results.append(m_seas)

    # IBOV variant of seasonality
    tw_seas_ibov = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_seas_ibov["CDI_ASSET"] = 0.0
    tw_seas_ibov["IBOV"] = 0.0
    r_seas_ibov = ret.copy()
    r_seas_ibov["CDI_ASSET"] = cdi_monthly
    r_seas_ibov["IBOV"] = ibov_ret

    for i in range(5, len(ret)):
        month = ret.index[i].month
        if month in positive_months:
            tw_seas_ibov.iloc[i, tw_seas_ibov.columns.get_loc("IBOV")] = 1.0
        else:
            tw_seas_ibov.iloc[i, tw_seas_ibov.columns.get_loc("CDI_ASSET")] = 1.0

    m_seas_ibov, at_seas_ibov, ret_seas_ibov = run_and_collect(
        "Seasonal (IBOV)", r_seas_ibov, tw_seas_ibov, cdi_monthly, ibov_ret
    )
    results.append(m_seas_ibov)

    # ── Strategy 3: December/January (13th Salary) ───────────────────────────
    print("\n[3/4] December/January Effect...")
    # Nov-Jan invested in equities, rest CDI
    DECJAN_MONTHS = {11, 12, 1}

    tw_dj = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_dj["CDI_ASSET"] = 0.0
    tw_dj["IBOV"] = 0.0
    r_dj = ret.copy()
    r_dj["CDI_ASSET"] = cdi_monthly
    r_dj["IBOV"] = ibov_ret

    for i in range(LOOKBACK + 2, len(ret)):
        month = ret.index[i].month
        if month in DECJAN_MONTHS:
            mf_row = tw_mf.iloc[i]
            if mf_row.sum() > 0:
                tw_dj.iloc[i] = mf_row
            else:
                tw_dj.iloc[i, tw_dj.columns.get_loc("IBOV")] = 1.0
        else:
            tw_dj.iloc[i, tw_dj.columns.get_loc("CDI_ASSET")] = 1.0

    m_dj, at_dj, ret_dj = run_and_collect("Dec/Jan (Nov-Jan)", r_dj, tw_dj, cdi_monthly, ibov_ret)
    results.append(m_dj)

    # Dec-Feb variant
    DECFEB_MONTHS = {12, 1, 2}
    tw_df_v = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_df_v["CDI_ASSET"] = 0.0
    tw_df_v["IBOV"] = 0.0
    r_df_v = ret.copy()
    r_df_v["CDI_ASSET"] = cdi_monthly
    r_df_v["IBOV"] = ibov_ret

    for i in range(LOOKBACK + 2, len(ret)):
        month = ret.index[i].month
        if month in DECFEB_MONTHS:
            mf_row = tw_mf.iloc[i]
            if mf_row.sum() > 0:
                tw_df_v.iloc[i] = mf_row
            else:
                tw_df_v.iloc[i, tw_df_v.columns.get_loc("IBOV")] = 1.0
        else:
            tw_df_v.iloc[i, tw_df_v.columns.get_loc("CDI_ASSET")] = 1.0

    m_dfv, at_dfv, ret_dfv = run_and_collect("Dec/Jan (Dec-Feb)", r_df_v, tw_df_v, cdi_monthly, ibov_ret)
    results.append(m_dfv)

    # ── Strategy 4: Sell in May ───────────────────────────────────────────────
    print("\n[4/4] Sell in May...")
    # Invest Oct-Apr, CDI May-Sep
    EQUITY_MONTHS = {10, 11, 12, 1, 2, 3, 4}

    tw_sim = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_sim["CDI_ASSET"] = 0.0
    tw_sim["IBOV"] = 0.0
    r_sim = ret.copy()
    r_sim["CDI_ASSET"] = cdi_monthly
    r_sim["IBOV"] = ibov_ret

    for i in range(LOOKBACK + 2, len(ret)):
        month = ret.index[i].month
        if month in EQUITY_MONTHS:
            mf_row = tw_mf.iloc[i]
            if mf_row.sum() > 0:
                tw_sim.iloc[i] = mf_row
            else:
                tw_sim.iloc[i, tw_sim.columns.get_loc("IBOV")] = 1.0
        else:
            tw_sim.iloc[i, tw_sim.columns.get_loc("CDI_ASSET")] = 1.0

    m_sim, at_sim, ret_sim = run_and_collect("Sell-in-May (MF)", r_sim, tw_sim, cdi_monthly, ibov_ret)
    results.append(m_sim)

    # IBOV-only Sell in May
    tw_sim_ibov = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_sim_ibov["CDI_ASSET"] = 0.0
    tw_sim_ibov["IBOV"] = 0.0
    r_sim_ibov = ret.copy()
    r_sim_ibov["CDI_ASSET"] = cdi_monthly
    r_sim_ibov["IBOV"] = ibov_ret

    for i in range(5, len(ret)):
        month = ret.index[i].month
        if month in EQUITY_MONTHS:
            tw_sim_ibov.iloc[i, tw_sim_ibov.columns.get_loc("IBOV")] = 1.0
        else:
            tw_sim_ibov.iloc[i, tw_sim_ibov.columns.get_loc("CDI_ASSET")] = 1.0

    m_sim_ibov, at_sim_ibov, ret_sim_ibov = run_and_collect(
        "Sell-in-May (IBOV)", r_sim_ibov, tw_sim_ibov, cdi_monthly, ibov_ret
    )
    results.append(m_sim_ibov)

    # ── Benchmarks ────────────────────────────────────────────────────────────
    m_ibov = build_metrics(ibov_ret.dropna(), "IBOV", 12)
    m_cdi = build_metrics(cdi_ret.dropna(), "CDI", 12)
    results.extend([m_ibov, m_cdi])

    # ── Results table ─────────────────────────────────────────────────────────
    results.sort(key=lambda m: float(m.get("Sharpe", 0)), reverse=True)
    print(f"\n{'='*90}")
    print(f"  SEASONAL EFFECTS -- After-Tax (15% CGT) -- {START} to {END}")
    print(f"{'='*90}")
    hdr = f"  {'Strategy':<28s} {'Ann.Ret%':>8s} {'Ann.Vol%':>8s} {'Sharpe':>8s} {'MaxDD%':>8s} {'Calmar':>8s}"
    print(hdr)
    print(f"  {'-'*72}")
    for m in results:
        name = str(m.get("Strategy", "?"))[:28]
        print(
            f"  {name:<28s} "
            f"{str(m.get('Ann. Return (%)', '?')):>8s} "
            f"{str(m.get('Ann. Volatility (%)', '?')):>8s} "
            f"{str(m.get('Sharpe', '?')):>8s} "
            f"{str(m.get('Max Drawdown (%)', '?')):>8s} "
            f"{str(m.get('Calmar', '?')):>8s}"
        )
    print(f"{'='*90}\n")

    # Check if any seasonal strategy achieves Sharpe > 0.3
    print("\nCorrelation check for strategies with Sharpe > 0.3:")
    strat_rets = {
        "TOM (MF stocks)":        ret_tom,
        "Seasonal (pos months)":  ret_seas,
        "Seasonal (IBOV)":        ret_seas_ibov,
        "Dec/Jan (Nov-Jan)":      ret_dj,
        "Dec/Jan (Dec-Feb)":      ret_dfv,
        "Sell-in-May (MF)":       ret_sim,
        "Sell-in-May (IBOV)":     ret_sim_ibov,
        "IBOV":                   ibov_ret.dropna(),
        "CDI":                    cdi_ret.dropna(),
    }
    high_sharpe = [
        m["Strategy"] for m in results
        if float(m.get("Sharpe", 0)) > 0.3 and m["Strategy"] not in ("IBOV", "CDI")
    ]
    if high_sharpe:
        print(f"  Strategies with Sharpe > 0.3: {high_sharpe}")
        # Build correlation table
        corr_data = {}
        for name in high_sharpe:
            if name in strat_rets:
                corr_data[name] = strat_rets[name]
        if corr_data:
            corr_df = pd.DataFrame(corr_data).dropna(how="all")
            print("\n  Correlation matrix:")
            print(corr_df.corr().round(3).to_string())
    else:
        print("  No seasonal strategy achieved Sharpe > 0.3")

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")
    _plot_seasonality_bar(monthly_avgs, monthly_stds, monthly_tstat, month_labels)
    _plot_equity_curves(strat_rets)

    print("\nDone.")


def _plot_seasonality_bar(monthly_avgs, monthly_stds, monthly_tstat, month_labels):
    """Bar chart of average IBOV return by calendar month."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    months = list(range(1, 13))
    avgs = [monthly_avgs.get(m, np.nan) for m in months]
    stds = [monthly_stds.get(m, np.nan) for m in months]
    tstats = [monthly_tstat.get(m, np.nan) for m in months]

    colors = ["#00D4AA" if a > 0 else "#FF4C6A" for a in avgs]
    bars = ax.bar(month_labels, avgs, color=colors, alpha=0.8, edgecolor=PALETTE["grid"])

    # Add error bars (1 std)
    ax.errorbar(
        month_labels, avgs,
        yerr=[s / 2 for s in stds],
        fmt="none", color=PALETTE["text"], alpha=0.5, capsize=4,
    )

    # Mark significant months (|t| > 1.5)
    for i, (t, a) in enumerate(zip(tstats, avgs)):
        if abs(t) > 1.5:
            ax.text(
                i, a + (0.3 if a > 0 else -0.5),
                "*", ha="center", va="bottom" if a > 0 else "top",
                color=PALETTE["text"], fontsize=14,
            )

    ax.axhline(0, color=PALETTE["sub"], lw=0.8, ls="--")
    ax.set_title(
        "Average IBOV Monthly Return by Calendar Month (2005-present)\n* = |t-stat| > 1.5",
        color=PALETTE["text"], fontsize=11, fontweight="bold"
    )
    ax.set_ylabel("Avg Monthly Return (%)", color=PALETTE["sub"])
    ax.tick_params(colors=PALETTE["sub"])
    ax.spines[:].set_color(PALETTE["grid"])
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.6, ls="--")

    out = os.path.join(_BACKTESTS_DIR, "seasonal_effects_analysis.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Seasonality bar chart saved -> {out}")


def _plot_equity_curves(strat_rets: dict):
    """Equity curves for all seasonal strategies vs benchmarks."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    colors = [
        "#00D4AA", "#7B61FF", "#FFC947", "#4FC3F7",
        "#F48FB1", "#81C784", "#FF6B35", "#A8B2C1", "#E0E0E0",
    ]
    linestyles = ["-", "-", "-.", "-.", "--", "--", ":", "-", ":"]

    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_facecolor(PALETTE["panel"])
    fig.patch.set_facecolor(PALETTE["bg"])

    for idx, (name, ret) in enumerate(strat_rets.items()):
        r = ret.dropna()
        if len(r) == 0:
            continue
        curve = cumret(r)
        ax.plot(
            curve.index, curve.values,
            label=name,
            color=colors[idx % len(colors)],
            ls=linestyles[idx % len(linestyles)],
            lw=2.0 if name not in ("IBOV", "CDI") else 1.4,
        )

    ax.set_title(
        "Seasonal Effects: Equity Curves (2005-present)",
        color=PALETTE["text"], fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Growth of R$1", color=PALETTE["sub"])
    ax.legend(
        facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"], fontsize=8, ncol=3, loc="upper left"
    )
    fmt_ax(ax)

    out = os.path.join(_BACKTESTS_DIR, "seasonal_effects_backtest.png")
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Equity curves saved -> {out}")


if __name__ == "__main__":
    main()
