"""
Parameter Sensitivity Analysis
================================
Sweeps 2-3 key parameters for the top 4 strategies and produces Sharpe /
Return / MaxDD heatmaps. Identifies robust parameter zones where performance
is stable rather than peaking at a single overfitted setting.

Strategies analyzed:
  1. MultiFactor    -- LOOKBACK x TOP_PCT
  2. COPOM Easing   -- cdi_shift (signal lag) x equity_type
  3. MomSharpe      -- LOOKBACK x TOP_PCT
  4. Res.MultiFactor -- regime_threshold x TOP_PCT

Usage:
    python3 backtests/param_sensitivity_analysis.py

Expected runtime: 5-15 minutes (100-200 simulation runs total).
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
from matplotlib.gridspec import GridSpec

_BACKTESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_BACKTESTS_DIR)
for _p in [_PROJECT_ROOT, _BACKTESTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.simulation import run_simulation
from core.metrics import build_metrics, value_to_ret
from core.param_scanner import scan_parameters, plot_param_heatmap, identify_robust_zone
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

DB_PATH = os.path.join(_PROJECT_ROOT, "b3_market_data.sqlite")
OUT_DIR = _BACKTESTS_DIR

SIM_CONFIG = dict(
    capital=CAPITAL,
    tax_rate=TAX,
    slippage=SLIP,
    monthly_sales_exemption=EXEMPTION,
)

METRICS = ["sharpe", "ann_return", "max_dd"]
METRIC_LABELS = {
    "sharpe": "Sharpe Ratio",
    "ann_return": "Ann. Return (%)",
    "max_dd": "Max Drawdown (%)",
}


# ─── Shared data loader ───────────────────────────────────────────────────────

def load_shared_data() -> dict:
    """Load all market data and precompute signals used across strategies."""
    print("\n  Loading shared market data...")
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
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(12).max()

    # COPOM easing signal
    is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    # MA200
    ma200_daily = adj_close.rolling(200, min_periods=200).mean()
    ma200_m = ma200_daily.resample(FREQ).last()
    above_ma200 = px > ma200_m
    dist_ma200 = px / ma200_m - 1

    # IBOV regime
    ibov_daily_ret = ibov_px.pct_change()
    ibov_vol_20d = ibov_daily_ret.rolling(20).std()
    ibov_vol_m = ibov_vol_20d.resample(FREQ).last()
    ibov_vol_pctrank = ibov_vol_m.expanding(min_periods=12).apply(
        lambda x: (x.iloc[-1] >= x).mean(), raw=False
    )
    ibov_calm = ibov_vol_pctrank <= 0.70
    ibov_ret_20d = ibov_px.pct_change(20)
    ibov_ret_m = ibov_ret_20d.resample(FREQ).last()
    ibov_uptrend = ibov_ret_m > 0

    # IBOV MA10
    ibov_m = ibov_px.resample(FREQ).last()
    ibov_ma10 = ibov_m.rolling(10).mean()
    ibov_above = ibov_m > ibov_ma10

    # Vol/ATR for ResMultifactor
    vol_60d = ret.rolling(5).std()
    daily_ret_abs = adj_close.pct_change().abs()
    atr_proxy = daily_ret_abs.ewm(span=14, min_periods=14).mean()
    atr_m = atr_proxy.resample(FREQ).last()
    vol_20d = ret.rolling(2).std()

    return dict(
        ret=ret, log_ret=log_ret, adtv=adtv, raw_close=raw_close,
        has_glitch=has_glitch, is_easing=is_easing, above_ma200=above_ma200,
        dist_ma200=dist_ma200, ma200_m=ma200_m, ibov_calm=ibov_calm,
        ibov_uptrend=ibov_uptrend, ibov_above=ibov_above,
        vol_60d=vol_60d, atr_m=atr_m, vol_20d=vol_20d,
        cdi_monthly=cdi_monthly, ibov_ret=ibov_ret, ibov_m=ibov_m,
        px=px, adj_close=adj_close, fin_vol=fin_vol,
    )


# ─── Strategy signal functions ────────────────────────────────────────────────

def multifactor_signal(params: dict, data: dict):
    """MultiFactor (Mom + LowVol) with parameterized LOOKBACK and TOP_PCT."""
    lookback = params["lookback"]
    top_pct = params["top_pct"]
    ret = data["ret"]
    log_ret = data["log_ret"]
    adtv = data["adtv"]
    raw_close = data["raw_close"]
    has_glitch = data["has_glitch"]

    mom_sig = log_ret.shift(1).rolling(lookback).sum()
    mom_sig[has_glitch == 1] = np.nan
    vol_sig = -ret.shift(1).rolling(lookback).std()
    vol_sig[has_glitch == 1] = np.nan
    composite = mom_sig.rank(axis=1, pct=True) * 0.5 + vol_sig.rank(axis=1, pct=True) * 0.5

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

    for i in range(lookback + 2, len(ret)):
        sig_row = composite.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= 1_000_000) & (raw_r >= 1.0)
        valid = sig_row[mask].dropna()
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * top_pct))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw.columns:
                tw.iloc[i, tw.columns.get_loc(t)] = w

    return r, tw


def copom_easing_signal(params: dict, data: dict):
    """
    COPOM Easing with parameterized shift lag and equity type.
    cdi_shift: how many months back to compare CDI (2, 3, 4, 6)
    equity_type: "IBOV", "multifactor", "equal_weight"
    """
    cdi_shift = params["cdi_shift"]
    equity_type = params.get("equity_type", "IBOV")

    ret = data["ret"]
    cdi_monthly = data["cdi_monthly"]
    ibov_ret = data["ibov_ret"]
    adtv = data["adtv"]
    raw_close = data["raw_close"]
    log_ret = data["log_ret"]
    has_glitch = data["has_glitch"]

    is_easing_param = cdi_monthly.shift(1) < cdi_monthly.shift(cdi_shift)

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    tw["IBOV"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    # Precompute multifactor signal if needed
    if equity_type == "multifactor":
        mom_sig = log_ret.shift(1).rolling(12).sum()
        mom_sig[has_glitch == 1] = np.nan
        vol_sig = -ret.shift(1).rolling(12).std()
        vol_sig[has_glitch == 1] = np.nan
        composite = mom_sig.rank(axis=1, pct=True) * 0.5 + vol_sig.rank(axis=1, pct=True) * 0.5

    for i in range(cdi_shift + 1, len(ret)):
        easing = bool(is_easing_param.iloc[i]) if i < len(is_easing_param) else False
        if not easing:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        if equity_type == "IBOV":
            tw.iloc[i, tw.columns.get_loc("IBOV")] = 1.0
        elif equity_type == "equal_weight":
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            mask = (adtv_r >= 1_000_000) & (raw_r >= 1.0)
            tickers = mask[mask].index.tolist()
            if tickers:
                w = 1.0 / len(tickers)
                for t in tickers:
                    if t in tw.columns:
                        tw.iloc[i, tw.columns.get_loc(t)] = w
            else:
                tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
        elif equity_type == "multifactor":
            sig_row = composite.iloc[i - 1]
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]
            mask = (adtv_r >= 1_000_000) & (raw_r >= 1.0)
            valid = sig_row[mask].dropna()
            if len(valid) < 5:
                tw.iloc[i, tw.columns.get_loc("IBOV")] = 1.0
            else:
                n = max(1, int(len(valid) * 0.10))
                sel = valid.nlargest(n).index.tolist()
                ww = 1.0 / len(sel)
                for t in sel:
                    if t in tw.columns:
                        tw.iloc[i, tw.columns.get_loc(t)] = ww

    return r, tw


def momsharpe_signal(params: dict, data: dict):
    """MomSharpe (risk-adjusted momentum) with parameterized LOOKBACK and TOP_PCT."""
    lookback = params["lookback"]
    top_pct = params["top_pct"]
    ret = data["ret"]
    log_ret = data["log_ret"]
    adtv = data["adtv"]
    raw_close = data["raw_close"]
    has_glitch = data["has_glitch"]

    mom_12m = log_ret.shift(1).rolling(lookback).sum()
    vol_12m = ret.shift(1).rolling(lookback).std()
    sharpe_sig = mom_12m / vol_12m
    sharpe_sig[has_glitch == 1] = np.nan

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

    for i in range(lookback + 2, len(ret)):
        sig_row = sharpe_sig.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= 1_000_000) & (raw_r >= 1.0)
        valid = sig_row[mask].dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * top_pct))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw.columns:
                tw.iloc[i, tw.columns.get_loc(t)] = w

    return r, tw


def res_multifactor_signal(params: dict, data: dict):
    """
    Res.MultiFactor with parameterized regime_threshold (1/2/3 of 3 signals) and TOP_PCT.
    5-factor composite: dist_ma200, low vol_60d, low ATR, low vol_20d, high ADTV.
    """
    regime_threshold = int(params["regime_threshold"])  # 1, 2, or 3
    top_pct = params["top_pct"]
    ret = data["ret"]
    adtv = data["adtv"]
    raw_close = data["raw_close"]
    has_glitch = data["has_glitch"]
    is_easing = data["is_easing"]
    ibov_calm = data["ibov_calm"]
    ibov_uptrend = data["ibov_uptrend"]
    dist_ma200 = data["dist_ma200"]
    vol_60d = data["vol_60d"]
    atr_m = data["atr_m"]
    vol_20d = data["vol_20d"]
    cdi_monthly = data["cdi_monthly"]
    ibov_ret = data["ibov_ret"]

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    for i in range(14, len(ret)):
        sig_easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        sig_calm = bool(ibov_calm.iloc[i - 1]) if i - 1 < len(ibov_calm) else False
        sig_up = bool(ibov_uptrend.iloc[i - 1]) if i - 1 < len(ibov_uptrend) else False
        if (int(sig_easing) + int(sig_calm) + int(sig_up)) < regime_threshold:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        composite = (
            0.2 * dist_ma200.iloc[i - 1].rank(pct=True)
            + 0.2 * (-vol_60d.iloc[i - 1]).rank(pct=True)
            + 0.2 * (-atr_m.iloc[i - 1]).rank(pct=True)
            + 0.2 * (-vol_20d.iloc[i - 1]).rank(pct=True)
            + 0.2 * adtv.iloc[i - 1].rank(pct=True)
        )
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        gl = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()
        mask = (adtv_r >= 1_000_000) & (raw_r >= 1.0)
        if len(gl) > 0:
            mask = mask & (gl != 1)
        valid = composite[mask].dropna()
        if len(valid) < 5:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        n = max(1, int(len(valid) * top_pct))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw.columns:
                tw.iloc[i, tw.columns.get_loc(t)] = w

    return r, tw


# ─── Plotting helpers ─────────────────────────────────────────────────────────

def plot_strategy_heatmaps(
    results_df: pd.DataFrame,
    param_x: str,
    param_y: str,
    strategy_name: str,
    out_path: str,
):
    """Plot 3 heatmaps (Sharpe, Return, MaxDD) for a strategy in a single PNG."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle(
        f"Parameter Sensitivity: {strategy_name}",
        color=PALETTE["text"], fontsize=13, fontweight="bold", y=1.02
    )

    for ax, metric in zip(axes, METRICS):
        plot_param_heatmap(
            results_df=results_df,
            param_x=param_x,
            param_y=param_y,
            metric=metric,
            title=METRIC_LABELS[metric],
            out_path=out_path,  # not used for subplots
            ax=ax,
        )
        # Add colorbar for each subplot
        im = ax.get_images()
        if im:
            plt.colorbar(im[0], ax=ax, shrink=0.8).ax.yaxis.set_tick_params(
                color=PALETTE["sub"]
            )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Heatmaps saved -> {out_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 70)
    print("  Parameter Sensitivity Analysis")
    print("=" * 70)

    # Load shared data once
    shared = load_shared_data()

    # ── Strategy 1: MultiFactor ──────────────────────────────────────────────
    print("\n[1/4] MultiFactor: LOOKBACK x TOP_PCT")
    grid_mf = {
        "lookback": [6, 9, 12, 15, 18],
        "top_pct":  [0.05, 0.10, 0.15, 0.20, 0.25],
    }
    df_mf = scan_parameters(multifactor_signal, grid_mf, shared, SIM_CONFIG)
    plot_strategy_heatmaps(
        df_mf, "lookback", "top_pct", "MultiFactor",
        os.path.join(OUT_DIR, "param_sensitivity_multifactor.png"),
    )
    print("\n  Robust zone analysis (MultiFactor):")
    print(identify_robust_zone(df_mf, "lookback", "top_pct", "sharpe"))

    # ── Strategy 2: COPOM Easing ────────────────────────────────────────────
    print("\n[2/4] COPOM Easing: cdi_shift x equity_type")
    grid_copom = {
        "cdi_shift":   [2, 3, 4, 6],
        "equity_type": ["IBOV", "equal_weight", "multifactor"],
    }
    df_copom = scan_parameters(copom_easing_signal, grid_copom, shared, SIM_CONFIG)
    plot_strategy_heatmaps(
        df_copom, "cdi_shift", "equity_type", "COPOM Easing",
        os.path.join(OUT_DIR, "param_sensitivity_copom.png"),
    )
    print("\n  Robust zone analysis (COPOM Easing):")
    print(identify_robust_zone(df_copom, "cdi_shift", "equity_type", "sharpe"))

    # ── Strategy 3: MomSharpe ────────────────────────────────────────────────
    print("\n[3/4] MomSharpe: LOOKBACK x TOP_PCT")
    grid_ms = {
        "lookback": [6, 9, 12, 15, 18],
        "top_pct":  [0.05, 0.10, 0.15, 0.20],
    }
    df_ms = scan_parameters(momsharpe_signal, grid_ms, shared, SIM_CONFIG)
    plot_strategy_heatmaps(
        df_ms, "lookback", "top_pct", "MomSharpe",
        os.path.join(OUT_DIR, "param_sensitivity_momsharpe.png"),
    )
    print("\n  Robust zone analysis (MomSharpe):")
    print(identify_robust_zone(df_ms, "lookback", "top_pct", "sharpe"))

    # ── Strategy 4: Res.MultiFactor ──────────────────────────────────────────
    print("\n[4/4] Res.MultiFactor: regime_threshold x TOP_PCT")
    grid_rmf = {
        "regime_threshold": [1, 2, 3],
        "top_pct":          [0.05, 0.10, 0.15, 0.20],
    }
    df_rmf = scan_parameters(res_multifactor_signal, grid_rmf, shared, SIM_CONFIG)
    plot_strategy_heatmaps(
        df_rmf, "regime_threshold", "top_pct", "Res.MultiFactor",
        os.path.join(OUT_DIR, "param_sensitivity_resmultifactor.png"),
    )
    print("\n  Robust zone analysis (Res.MultiFactor):")
    print(identify_robust_zone(df_rmf, "regime_threshold", "top_pct", "sharpe"))

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  PARAMETER SENSITIVITY SUMMARY")
    print("=" * 70)

    for strat_name, df, px, py in [
        ("MultiFactor",    df_mf,   "lookback", "top_pct"),
        ("COPOM Easing",   df_copom,"cdi_shift", "equity_type"),
        ("MomSharpe",      df_ms,   "lookback", "top_pct"),
        ("Res.MultiFactor",df_rmf,  "regime_threshold", "top_pct"),
    ]:
        best_idx = df["sharpe"].idxmax() if not df["sharpe"].isna().all() else None
        if best_idx is not None:
            best = df.iloc[best_idx]
            print(f"\n  {strat_name}:")
            print(f"    Best Sharpe: {best['sharpe']:.3f}")
            print(f"    Best params: {px}={best[px]}, {py}={best[py]}")
            print(f"    Ann. Return: {best['ann_return']:.1f}%, Max DD: {best['max_dd']:.1f}%")
            print(f"    Default params (lookback=12 or shift=3): see heatmap")

    print("\n  Output files:")
    for fname in [
        "param_sensitivity_multifactor.png",
        "param_sensitivity_copom.png",
        "param_sensitivity_momsharpe.png",
        "param_sensitivity_resmultifactor.png",
    ]:
        print(f"    {os.path.join(OUT_DIR, fname)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
