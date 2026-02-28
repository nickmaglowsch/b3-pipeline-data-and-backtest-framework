"""
Earnings Momentum Proxy Backtest
==================================
Tests three variants of a volume-confirmed earnings proxy signal:
  1. Volume-Confirmed Momentum -- return * volume_ratio during reporting months
  2. Post-Earnings Drift       -- hold top reporting-month winners for 3 months
  3. Earnings + COPOM Regime   -- earnings signal only during COPOM easing

B3 reporting season: March, April, May, August, November.

Usage:
    python3 backtests/earnings_proxy_backtest.py
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

_BACKTESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKTESTS_DIR)
for _p in [_PROJECT_ROOT, _BACKTESTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.simulation import run_simulation
from core.metrics import build_metrics, value_to_ret, cumret
from core.plotting import PALETTE, fmt_ax, plot_tax_backtest

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
REPORTING_MONTHS = {3, 4, 5, 8, 11}  # Q4/Q1/Q2/Q3 reporting windows on B3

DB_PATH = os.path.join(_PROJECT_ROOT, "b3_market_data.sqlite")
OUT_DIR = _BACKTESTS_DIR


def run_and_collect(name, ret_matrix, weights):
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
    return m, result, at_ret


def main():
    print("\n" + "=" * 70)
    print("  Earnings Momentum Proxy Backtest")
    print("=" * 70)
    print(f"  B3 Reporting months: {sorted(REPORTING_MONTHS)}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\nLoading data...")
    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START, END)
    cdi_daily = download_cdi_daily(START, END)
    ibov_px = download_benchmark("^BVSP", START, END)

    ibov_ret = ibov_px.resample(FREQ).last().pct_change().dropna()
    cdi_ret = (1 + cdi_daily).resample(FREQ).prod() - 1
    cdi_monthly = cdi_ret.copy()

    px = adj_close.resample(FREQ).last()
    ret = px.pct_change()
    raw_close = close_px.resample(FREQ).last()
    adtv = fin_vol.resample(FREQ).mean()
    log_ret = np.log1p(ret)
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(LOOKBACK).max()

    # COPOM easing signal
    is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    # Monthly volume ratio: current month vol / 6-month trailing avg
    period_vol = fin_vol.resample(FREQ).sum()
    avg_vol_6m = period_vol.shift(1).rolling(6).mean()
    volume_ratio = (period_vol.shift(1) / avg_vol_6m).clip(upper=5.0)

    # Standard 6-month momentum for non-reporting periods
    mom_6m = log_ret.shift(1).rolling(6).sum()
    mom_6m[has_glitch == 1] = np.nan

    # MultiFactor signal as reference
    mom_sig = log_ret.shift(1).rolling(LOOKBACK).sum()
    mom_sig[has_glitch == 1] = np.nan
    vol_sig = -ret.shift(1).rolling(LOOKBACK).std()
    vol_sig[has_glitch == 1] = np.nan
    mf_composite = (
        mom_sig.rank(axis=1, pct=True) * 0.5
        + vol_sig.rank(axis=1, pct=True) * 0.5
    )

    results = []

    # ── Variant 1: Volume-Confirmed Momentum ─────────────────────────────────
    print("\n[1/3] Volume-Confirmed Momentum...")
    tw_v1 = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r_v1 = ret.copy()

    for i in range(8, len(ret)):
        date = ret.index[i]
        prev_month = date.month - 1 if date.month > 1 else 12
        was_reporting = prev_month in REPORTING_MONTHS

        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        gl = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()
        if len(gl) > 0:
            mask = mask & (gl != 1)

        if was_reporting:
            # Volume-weighted return: signal = return_1m * volume_ratio
            ret_1m = ret.iloc[i - 1]
            vr = volume_ratio.iloc[i - 1].clip(lower=1.0)
            # Handle alignment: volume_ratio may have different columns
            common_cols = ret_1m.index.intersection(vr.index)
            signal = pd.Series(np.nan, index=ret.columns)
            if len(common_cols) > 0:
                signal[common_cols] = ret_1m[common_cols] * vr[common_cols]
        else:
            # Standard 6-month momentum
            signal = mom_6m.iloc[i - 1] if i - 1 < len(mom_6m) else pd.Series()

        if len(signal) == 0:
            continue

        valid = signal[mask].dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw_v1.columns:
                tw_v1.iloc[i, tw_v1.columns.get_loc(t)] = w

    m_v1, res_v1, ret_v1 = run_and_collect("VolumeConfirmedMom", r_v1, tw_v1)
    results.append(m_v1)

    # ── Variant 2: Post-Earnings Drift (3-month hold) ────────────────────────
    print("\n[2/3] Post-Earnings Drift (3-month hold)...")
    tw_v2 = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r_v2 = ret.copy()

    # Track which stocks were selected in each reporting month
    # Hold for 3 months regardless of signal changes
    hold_until = {}  # {ticker: expiry_index}

    for i in range(8, len(ret)):
        date = ret.index[i]
        prev_month = date.month - 1 if date.month > 1 else 12
        was_reporting = prev_month in REPORTING_MONTHS

        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        gl = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()
        if len(gl) > 0:
            mask = mask & (gl != 1)

        if was_reporting:
            # New selection using volume-confirmed signal
            ret_1m = ret.iloc[i - 1]
            vr = volume_ratio.iloc[i - 1].clip(lower=1.0)
            common_cols = ret_1m.index.intersection(vr.index)
            signal = pd.Series(np.nan, index=ret.columns)
            if len(common_cols) > 0:
                signal[common_cols] = ret_1m[common_cols] * vr[common_cols]
            valid = signal[mask].dropna()
            valid = valid[np.isfinite(valid)]
            if len(valid) >= 5:
                n = max(1, int(len(valid) * 0.10))
                new_sel = valid.nlargest(n).index.tolist()
                # Hold for 3 months
                for t in new_sel:
                    hold_until[t] = i + 3
                # Also clear expired holdings
                hold_until = {t: exp for t, exp in hold_until.items() if exp > i}
        else:
            # Clear expired holdings
            hold_until = {t: exp for t, exp in hold_until.items() if exp > i}

        # Build weights from current holdings
        active = [t for t, exp in hold_until.items() if t in tw_v2.columns and exp > i]
        if active:
            w = 1.0 / len(active)
            for t in active:
                tw_v2.iloc[i, tw_v2.columns.get_loc(t)] = w
        else:
            # No holdings: use standard momentum to stay invested
            if i >= LOOKBACK + 2:
                valid_mf = mf_composite.iloc[i - 1][mask].dropna()
                if len(valid_mf) >= 5:
                    n = max(1, int(len(valid_mf) * 0.10))
                    sel = valid_mf.nlargest(n).index.tolist()
                    ww = 1.0 / len(sel)
                    for t in sel:
                        if t in tw_v2.columns:
                            tw_v2.iloc[i, tw_v2.columns.get_loc(t)] = ww

    m_v2, res_v2, ret_v2 = run_and_collect("PostEarningsDrift", r_v2, tw_v2)
    results.append(m_v2)

    # ── Variant 3: Earnings + COPOM Regime ───────────────────────────────────
    print("\n[3/3] Earnings + COPOM Regime Filter...")
    tw_v3 = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_v3["CDI_ASSET"] = 0.0
    r_v3 = ret.copy()
    r_v3["CDI_ASSET"] = cdi_monthly

    for i in range(8, len(ret)):
        easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        if not easing:
            tw_v3.iloc[i, tw_v3.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        date = ret.index[i]
        prev_month = date.month - 1 if date.month > 1 else 12
        was_reporting = prev_month in REPORTING_MONTHS

        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        gl = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()
        if len(gl) > 0:
            mask = mask & (gl != 1)

        if was_reporting:
            ret_1m = ret.iloc[i - 1]
            vr = volume_ratio.iloc[i - 1].clip(lower=1.0)
            common_cols = ret_1m.index.intersection(vr.index)
            signal = pd.Series(np.nan, index=ret.columns)
            if len(common_cols) > 0:
                signal[common_cols] = ret_1m[common_cols] * vr[common_cols]
        else:
            signal = mom_6m.iloc[i - 1] if i - 1 < len(mom_6m) else pd.Series()

        if len(signal) == 0:
            tw_v3.iloc[i, tw_v3.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        valid = signal[mask].dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < 5:
            tw_v3.iloc[i, tw_v3.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw_v3.columns:
                tw_v3.iloc[i, tw_v3.columns.get_loc(t)] = w

    m_v3, res_v3, ret_v3 = run_and_collect("Earnings+COPOM", r_v3, tw_v3)
    results.append(m_v3)

    # ── Benchmarks ────────────────────────────────────────────────────────────
    # MultiFactor benchmark (standard, for comparison)
    tw_mf = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r_mf = ret.copy()
    for i in range(LOOKBACK + 2, len(ret)):
        sig_row = mf_composite.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        valid = sig_row[mask].dropna()
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        ww = 1.0 / len(sel)
        for t in sel:
            if t in tw_mf.columns:
                tw_mf.iloc[i, tw_mf.columns.get_loc(t)] = ww
    m_mf, _, ret_mf = run_and_collect("MultiFactor", r_mf, tw_mf)
    results.append(m_mf)

    m_ibov = build_metrics(ibov_ret.dropna(), "IBOV", 12)
    m_cdi = build_metrics(cdi_ret.dropna(), "CDI", 12)
    results.extend([m_ibov, m_cdi])

    # ── Results table ─────────────────────────────────────────────────────────
    results.sort(key=lambda m: float(m.get("Sharpe", 0)), reverse=True)
    print(f"\n{'='*90}")
    print(f"  EARNINGS PROXY -- After-Tax (15% CGT) -- {START} to {END}")
    print(f"{'='*90}")
    hdr = f"  {'Strategy':<25s} {'Ann.Ret%':>8s} {'Ann.Vol%':>8s} {'Sharpe':>8s} {'MaxDD%':>8s} {'Calmar':>8s}"
    print(hdr)
    print(f"  {'-'*68}")
    for m in results:
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

    # ── Correlation with existing strategies ──────────────────────────────────
    print("\nCorrelation analysis (earnings variants vs MultiFactor, IBOV, CDI):")
    corr_data = {
        "VolumeConfirmedMom": ret_v1,
        "PostEarningsDrift":  ret_v2,
        "Earnings+COPOM":     ret_v3,
        "MultiFactor":        ret_mf,
        "IBOV":               ibov_ret.dropna(),
    }
    corr_df = pd.DataFrame(corr_data).dropna(how="all")
    print(corr_df.corr().round(3).to_string())

    strat_names = ["VolumeConfirmedMom", "PostEarningsDrift", "Earnings+COPOM"]
    mf_corrs = corr_df.corr()["MultiFactor"]
    for sname in strat_names:
        c = mf_corrs.get(sname, np.nan)
        status = "different signal" if c < 0.85 else "similar to MultiFactor"
        print(f"  {sname} vs MultiFactor corr: {c:.3f} ({status})")

    # ── Best variant tearsheet plot ───────────────────────────────────────────
    print("\nPlotting best variant tearsheet...")
    best_result_name = max(
        [m["Strategy"] for m in results if m["Strategy"] not in ("IBOV", "CDI", "MultiFactor")],
        key=lambda nm: next(float(m.get("Sharpe", 0)) for m in results if m["Strategy"] == nm),
        default="VolumeConfirmedMom",
    )
    print(f"  Best variant: {best_result_name}")

    result_map = {
        "VolumeConfirmedMom": res_v1,
        "PostEarningsDrift":  res_v2,
        "Earnings+COPOM":     res_v3,
    }
    best_res = result_map.get(best_result_name, res_v1)

    metrics_for_plot = [m for m in results if m["Strategy"] in (
        best_result_name, "MultiFactor", "IBOV", "CDI"
    )][:4]

    out_path = os.path.join(OUT_DIR, "earnings_proxy_backtest.png")
    try:
        plot_tax_backtest(
            title=f"Earnings Proxy: {best_result_name} (15% CGT, 0.1% slip)",
            pretax_val=best_res["pretax_values"],
            aftertax_val=best_res["aftertax_values"],
            ibov_ret=ibov_ret,
            tax_paid=best_res["tax_paid"],
            loss_cf=best_res["loss_carryforward"],
            turnover=best_res["turnover"],
            metrics=metrics_for_plot,
            total_tax_brl=float(best_res["tax_paid"].sum()),
            out_path=out_path,
            cdi_ret=cdi_ret,
        )
    except Exception as exc:
        print(f"  Tearsheet plot failed: {exc}. Using simple equity curve plot.")
        _simple_equity_plot(corr_data, out_path)

    print("\nDone.")


def _simple_equity_plot(strat_rets: dict, out_path: str):
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })
    colors = ["#00D4AA", "#7B61FF", "#FFC947", "#FF6B35", "#A8B2C1"]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_facecolor(PALETTE["panel"])
    fig.patch.set_facecolor(PALETTE["bg"])
    for idx, (name, ret) in enumerate(strat_rets.items()):
        r = ret.dropna()
        if len(r) == 0:
            continue
        curve = cumret(r)
        ax.plot(curve.index, curve.values, label=name,
                color=colors[idx % len(colors)], lw=2.0)
    ax.set_title("Earnings Proxy Backtest", color=PALETTE["text"], fontsize=12)
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["text"], fontsize=9)
    fmt_ax(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Plot saved -> {out_path}")


if __name__ == "__main__":
    main()
