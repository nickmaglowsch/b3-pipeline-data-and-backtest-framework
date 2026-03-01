"""
Run all strategies, collect their after-tax monthly return series,
and produce a correlation matrix heatmap.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.metrics import value_to_ret
from core.simulation import run_simulation

import matplotlib.pyplot as plt

# ─── Shared config ───
START = "2005-01-01"
END = datetime.today().strftime("%Y-%m-%d")
FREQ = "ME"
PERIODS = 12
CAPITAL = 100_000
TAX = 0.15
SLIP = 0.001

print(f"\n{'='*80}")
print(f"  STRATEGY CORRELATION MATRIX -- All from {START}")
print(f"{'='*80}\n")

# ─── Load shared data once ───
print("  Loading shared data...")
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "b3_market_data.sqlite")
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

MIN_ADTV = 1_000_000
MIN_PRICE = 1.0
LOOKBACK = 12

# Precompute common signals
log_ret = np.log1p(ret)
has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(LOOKBACK).max()
ma200_daily = adj_close.rolling(200, min_periods=200).mean()
ma200_m = ma200_daily.resample(FREQ).last()
above_ma200 = px > ma200_m
is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

# IBOV regime signals
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

# Multifactor common signals
dist_ma200 = px / ma200_m - 1
vol_60d = ret.rolling(5).std()
daily_ret_abs = adj_close.pct_change().abs()
atr_proxy = daily_ret_abs.ewm(span=14, min_periods=14).mean()
atr_m = atr_proxy.resample(FREQ).last()
vol_20d = ret.rolling(2).std()

mom_sig_12m = log_ret.shift(1).rolling(LOOKBACK).sum()
mom_sig_12m_clean = mom_sig_12m.copy()
mom_sig_12m_clean[has_glitch == 1] = np.nan
vol_sig_12m = -ret.shift(1).rolling(LOOKBACK).std()
vol_sig_12m_clean = vol_sig_12m.copy()
vol_sig_12m_clean[has_glitch == 1] = np.nan
mf_composite = (
    mom_sig_12m_clean.rank(axis=1, pct=True) * 0.5
    + vol_sig_12m_clean.rank(axis=1, pct=True) * 0.5
)

ibov_m = ibov_px.resample(FREQ).last()
ibov_ma10 = ibov_m.rolling(10).mean()
ibov_above = ibov_m > ibov_ma10


def run_and_get_returns(name, ret_matrix, weights):
    """Run simulation and return the after-tax monthly return series."""
    result = run_simulation(
        returns_matrix=ret_matrix.fillna(0.0),
        target_weights=weights,
        initial_capital=CAPITAL,
        tax_rate=TAX,
        slippage=SLIP,
        name=name,
        monthly_sales_exemption=20_000,
    )
    common = result["pretax_values"].index.intersection(ibov_ret.index)
    at_val = result["aftertax_values"].loc[common]
    at_ret = value_to_ret(at_val)
    at_ret.name = name
    return at_ret


# ═══════════════════════════════════════════════
#  STRATEGY DEFINITIONS
# ═══════════════════════════════════════════════
all_returns = {}

# ── 1. CDI + MA200 ──
print("  [1/8] CDI + MA200 Trend Filter...")
try:
    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    for i in range(13, len(ret)):
        easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        if not easing:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        above = above_ma200.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        gl = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()
        mask = above.fillna(False) & (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        if len(gl) > 0:
            mask = mask & (gl != 1)
        tickers = mask[mask].index.tolist()
        if not tickers:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
        else:
            w = 1.0 / len(tickers)
            for t in tickers:
                tw.iloc[i, tw.columns.get_loc(t)] = w
    all_returns["CDI+MA200"] = run_and_get_returns("CDI+MA200", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 2. Research Multi-Factor ──
print("  [2/8] Research Multi-Factor (5 factors + regime)...")
try:
    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    for i in range(14, len(ret)):
        sig_easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        sig_calm = bool(ibov_calm.iloc[i - 1]) if i - 1 < len(ibov_calm) else False
        sig_up = bool(ibov_uptrend.iloc[i - 1]) if i - 1 < len(ibov_uptrend) else False
        if (int(sig_easing) + int(sig_calm) + int(sig_up)) < 2:
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
        mask = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        if len(gl) > 0:
            mask = mask & (gl != 1)
        valid = composite[mask].dropna()
        if len(valid) < 5:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            tw.iloc[i, tw.columns.get_loc(t)] = w
    all_returns["Res.MultiFactor"] = run_and_get_returns("Res.MultiFactor", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 3. Regime Switching ──
print("  [3/8] Regime Switching (IBOV trend)...")
try:
    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    for i in range(LOOKBACK + 2, len(ret)):
        above = bool(ibov_above.iloc[i - 1]) if i - 1 < len(ibov_above) else False
        if not above:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        sig_row = mf_composite.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        valid = sig_row[mask].dropna()
        if len(valid) < 5:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            tw.iloc[i, tw.columns.get_loc(t)] = w
    all_returns["RegimeSwitching"] = run_and_get_returns("RegimeSwitching", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 4. COPOM Easing ──
print("  [4/8] COPOM Easing (IBOV vs CDI)...")
try:
    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    tw["IBOV"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    for i in range(5, len(ret)):
        easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        if easing:
            tw.iloc[i, tw.columns.get_loc("IBOV")] = 1.0
        else:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
    all_returns["COPOM Easing"] = run_and_get_returns("COPOM Easing", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 5. Multifactor (Mom + Low Vol, no regime) ──
print("  [5/8] Multifactor (Mom + Low Vol)...")
try:
    composite = (
        mom_sig_12m_clean.rank(axis=1, pct=True) * 0.5
        + vol_sig_12m_clean.rank(axis=1, pct=True) * 0.5
    )
    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

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
            tw.iloc[i, tw.columns.get_loc(t)] = w
    all_returns["MultiFactor"] = run_and_get_returns("MultiFactor", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 6. Smallcap Momentum ──
print("  [6/8] Smallcap Momentum...")
try:
    mom_6m = log_ret.shift(1).rolling(6).sum()
    mom_6m[has_glitch == 1] = np.nan
    adtv_median = adtv.median(axis=1)

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

    for i in range(8, len(ret)):
        sig_row = mom_6m.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        med = adtv_median.iloc[i - 1]
        mask = (adtv_r >= 100_000) & (adtv_r < med) & (raw_r >= MIN_PRICE)
        valid = sig_row[mask].dropna()
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * 0.20))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            tw.iloc[i, tw.columns.get_loc(t)] = w
    all_returns["SmallcapMom"] = run_and_get_returns("SmallcapMom", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 7. Low Volatility ──
print("  [7/8] Low Volatility (unconditional)...")
try:
    vol_sig = -ret.rolling(LOOKBACK).std()
    vol_sig[has_glitch == 1] = np.nan

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

    for i in range(LOOKBACK + 1, len(ret)):
        sig_row = vol_sig.iloc[i - 1]
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
            tw.iloc[i, tw.columns.get_loc(t)] = w
    all_returns["LowVol"] = run_and_get_returns("LowVol", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 8. Momentum Sharpe ──
print("  [8/8] Momentum Sharpe...")
try:
    mom_12m = log_ret.shift(1).rolling(LOOKBACK).sum()
    vol_12m = ret.shift(1).rolling(LOOKBACK).std()
    sharpe_sig = mom_12m / vol_12m
    sharpe_sig[has_glitch == 1] = np.nan

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

    for i in range(LOOKBACK + 2, len(ret)):
        sig_row = sharpe_sig.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        valid = sig_row[mask].dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            tw.iloc[i, tw.columns.get_loc(t)] = w
    all_returns["MomSharpe"] = run_and_get_returns("MomSharpe", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── Benchmarks ──
print("  Adding benchmarks...")
all_returns["IBOV"] = ibov_ret
all_returns["CDI"] = cdi_ret

# ═══════════════════════════════════════════════
#  BUILD CORRELATION MATRIX
# ═══════════════════════════════════════════════
print("\n  Building correlation matrix...")

returns_df = pd.DataFrame(all_returns)
# Align all series to common dates
returns_df = returns_df.dropna(how="all")

corr = returns_df.corr()

# Print to console
print(f"\n{'='*80}")
print(f"  MONTHLY RETURN CORRELATION MATRIX -- {START} to {END}")
print(f"{'='*80}")
print(corr.round(2).to_string())
print()

# ═══════════════════════════════════════════════
#  PLOT HEATMAP
# ═══════════════════════════════════════════════
PALETTE_BG = "#0D1117"
PALETTE_TEXT = "#E6EDF3"

plt.rcParams.update({
    "font.family": "monospace",
    "figure.facecolor": PALETTE_BG,
    "text.color": PALETTE_TEXT,
    "axes.facecolor": PALETTE_BG,
})

fig, ax = plt.subplots(figsize=(12, 10))

# Mask upper triangle
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
masked_corr = corr.copy().values.astype(float)
masked_corr[mask] = np.nan

im = ax.imshow(masked_corr, cmap="RdYlGn", vmin=-1, vmax=1, aspect="equal")

labels = corr.columns.tolist()
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10, color=PALETTE_TEXT)
ax.set_yticklabels(labels, fontsize=10, color=PALETTE_TEXT)

# Annotate cells
for i in range(len(labels)):
    for j in range(len(labels)):
        if mask[i, j]:
            continue
        val = corr.iloc[i, j]
        text_color = "black" if abs(val) > 0.5 else PALETTE_TEXT
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=text_color)

# Grid lines
for edge in range(len(labels) + 1):
    ax.axhline(edge - 0.5, color=PALETTE_BG, lw=1.5)
    ax.axvline(edge - 0.5, color=PALETTE_BG, lw=1.5)

ax.set_title(
    f"Strategy Monthly Return Correlations\n{START} to {END}",
    fontsize=14,
    fontweight="bold",
    color=PALETTE_TEXT,
    pad=20,
)

cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.ax.tick_params(colors=PALETTE_TEXT, labelsize=9)
cbar.set_label("Correlation", color=PALETTE_TEXT, fontsize=10)

out_path = "correlation_matrix.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE_BG)
print(f"📊  Heatmap saved → {out_path}")

try:
    plt.show()
except Exception:
    pass

print(f"\n{'='*80}")
print("  Done!")
print(f"{'='*80}\n")
