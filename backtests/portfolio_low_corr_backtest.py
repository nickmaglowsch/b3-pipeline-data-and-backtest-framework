"""
Low-Correlation Portfolio: equal-weight blend of 4 strategies that showed
the lowest pairwise correlations in the correlation matrix analysis.

Strategies (25% each, rebalanced monthly):
  1. SmallcapMom
  2. Res.MultiFactor
  3. MomSharpe
  4. COPOM Easing
"""

import warnings

warnings.filterwarnings("ignore")

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from core.data import download_benchmark, download_cdi_daily, load_b3_data
from core.metrics import build_metrics, cumret, display_metrics_table, value_to_ret
from core.simulation import run_simulation

# ─── Shared config ───
START = "2017-01-01"
END = datetime.today().strftime("%Y-%m-%d")
FREQ = "ME"
PERIODS = 12
CAPITAL = 100_000
TAX = 0.15
SLIP = 0.001

print(f"\n{'=' * 80}")
print(f"  LOW-CORRELATION PORTFOLIO BACKTEST -- from {START}")
print(f"{'=' * 80}\n")

# ─── Load shared data once ───
print("  Loading shared data...")
DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "b3_market_data.sqlite"
)
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

# Multifactor signals
dist_ma200 = px / ma200_m - 1
vol_60d = ret.rolling(60).std()
daily_ret_abs = adj_close.pct_change().abs()
atr_proxy = daily_ret_abs.ewm(span=14, min_periods=14).mean()
atr_m = atr_proxy.resample(FREQ).last()
vol_20d = ret.rolling(20).std()


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
#  RUN THE 4 SUB-STRATEGIES
# ═══════════════════════════════════════════════
sub_returns = {}

# ── 1. SmallcapMom ──
print("  [1/4] Smallcap Momentum...")
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
    sub_returns["SmallcapMom"] = run_and_get_returns("SmallcapMom", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 2. Res.MultiFactor ──
print("  [2/4] Research Multi-Factor (5 factors + regime)...")
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
    sub_returns["Res.MultiFactor"] = run_and_get_returns("Res.MultiFactor", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 3. MomSharpe ──
print("  [3/4] Momentum Sharpe...")
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
    sub_returns["MomSharpe"] = run_and_get_returns("MomSharpe", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ── 4. COPOM Easing ──
print("  [4/4] COPOM Easing (IBOV vs CDI)...")
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
    sub_returns["COPOM Easing"] = run_and_get_returns("COPOM Easing", r, tw)
except Exception as e:
    print(f"    ERROR: {e}")

# ═══════════════════════════════════════════════
#  COMBINE INTO EQUAL-WEIGHT PORTFOLIO
# ═══════════════════════════════════════════════
print("\n  Combining into equal-weight portfolio...")

sub_df = pd.DataFrame(sub_returns)
sub_df = sub_df.dropna(how="all")

# Equal weight: 25% each
n_strats = len(sub_df.columns)
portfolio_ret = sub_df.mean(axis=1)
portfolio_ret.name = "LowCorr Portfolio"

# ═══════════════════════════════════════════════
#  COMPUTE METRICS
# ═══════════════════════════════════════════════
print("  Computing metrics...")

common_idx = portfolio_ret.index.intersection(ibov_ret.index).intersection(
    cdi_ret.index
)
portfolio_ret = portfolio_ret.loc[common_idx]

metrics_portfolio = build_metrics(portfolio_ret, "LowCorr Portfolio", PERIODS)
metrics_ibov = build_metrics(ibov_ret.loc[common_idx], "IBOV", PERIODS)
metrics_cdi = build_metrics(cdi_ret.loc[common_idx], "CDI", PERIODS)

sub_metrics = []
for name, s in sub_returns.items():
    s_aligned = s.loc[s.index.intersection(common_idx)]
    sub_metrics.append(build_metrics(s_aligned, name, PERIODS))

all_metrics = [metrics_portfolio] + sub_metrics + [metrics_ibov, metrics_cdi]
all_metrics.sort(key=lambda m: float(m.get("Sharpe", 0)), reverse=True)

# ═══════════════════════════════════════════════
#  PRINT COMPARISON TABLE
# ═══════════════════════════════════════════════
print(f"\n{'=' * 90}")
print(f"  LOW-CORRELATION PORTFOLIO -- After-Tax (15% CGT) -- {START} to {END}")
print(f"{'=' * 90}")

header = f"  {'Strategy':<20s} {'Ann.Ret%':>8s} {'Ann.Vol%':>8s} {'Sharpe':>8s} {'MaxDD%':>8s} {'Calmar':>8s}"
print(header)
print(f"  {'-' * 68}")
for m in all_metrics:
    name = str(m.get("Strategy", "?"))[:20]
    ann_ret = str(m.get("Ann. Return (%)", "?"))
    ann_vol = str(m.get("Ann. Volatility (%)", "?"))
    sh = str(m.get("Sharpe", "?"))
    maxdd = str(m.get("Max Drawdown (%)", "?"))
    calmar = str(m.get("Calmar", "?"))
    print(
        f"  {name:<20s} {ann_ret:>8s} {ann_vol:>8s} {sh:>8s} {maxdd:>8s} {calmar:>8s}"
    )

print(f"{'=' * 90}\n")

# ═══════════════════════════════════════════════
#  PLOT
# ═══════════════════════════════════════════════
from core.plotting import PALETTE, fmt_ax

plt.rcParams.update(
    {
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
        "axes.facecolor": PALETTE["panel"],
    }
)

fig, axes = plt.subplots(
    2, 1, figsize=(18, 14), gridspec_kw={"height_ratios": [3, 1], "hspace": 0.35}
)

# ── Panel 1: Equity curves ──
ax1 = axes[0]

# Portfolio
port_curve = cumret(portfolio_ret)
ax1.plot(
    port_curve.index,
    port_curve.values,
    color="#00D4AA",
    lw=2.8,
    label="LowCorr Portfolio",
    zorder=10,
)

# Sub-strategies
sub_colors = ["#7B61FF", "#FFC947", "#FF6B35", "#A8B2C1"]
sub_styles = ["-.", "--", "-.", ":"]
for idx, (name, s) in enumerate(sub_returns.items()):
    s_aligned = s.loc[s.index.intersection(common_idx)]
    curve = cumret(s_aligned)
    ax1.plot(
        curve.index,
        curve.values,
        color=sub_colors[idx % len(sub_colors)],
        lw=1.4,
        label=name,
        ls=sub_styles[idx % len(sub_styles)],
        alpha=0.7,
        zorder=5,
    )

# Benchmarks
ibov_curve = cumret(ibov_ret.loc[common_idx])
cdi_curve = cumret(cdi_ret.loc[common_idx])
ax1.plot(
    ibov_curve.index,
    ibov_curve.values,
    color="#FF4C6A",
    lw=1.6,
    label="IBOV",
    ls="--",
    alpha=0.8,
    zorder=3,
)
ax1.plot(
    cdi_curve.index,
    cdi_curve.values,
    color="#8B949E",
    lw=1.4,
    label="CDI",
    ls=":",
    alpha=0.7,
    zorder=2,
)

ax1.set_title(
    f"Low-Correlation Portfolio (Equal Weight)\n{START} to {END}",
    color=PALETTE["text"],
    fontsize=14,
    fontweight="bold",
    pad=12,
)
ax1.legend(
    facecolor=PALETTE["bg"],
    edgecolor=PALETTE["grid"],
    labelcolor=PALETTE["text"],
    fontsize=9,
    ncol=4,
    loc="upper left",
)
fmt_ax(ax1, ylabel="Growth of R$1")
ax1.set_yscale("log")

# ── Panel 2: Metrics table ──
ax2 = axes[1]
ax2.axis("off")

col_labels = list(all_metrics[0].keys())
row_vals = [[str(m[k]) for k in col_labels] for m in all_metrics]

tbl = ax2.table(cellText=row_vals, colLabels=col_labels, cellLoc="center", loc="center")
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.6)

# Color rows
row_colors = [
    "#0D2E26",
    "#1A1230",
    "#2A1A10",
    "#1E2A3A",
    "#332211",
    "#1A2A1A",
    "#2A1A2A",
]
row_text_colors = [
    "#00D4AA",
    "#7B61FF",
    "#FFC947",
    "#FF6B35",
    "#A8B2C1",
    "#FF4C6A",
    "#8B949E",
]

for (r, c), cell in tbl.get_celld().items():
    cell.set_edgecolor(PALETTE["grid"])
    if r == 0:
        cell.set_facecolor("#1F2937")
        cell.get_text().set_color(PALETTE["text"])
        cell.get_text().set_fontweight("bold")
    else:
        ci = (r - 1) % len(row_colors)
        cell.set_facecolor(row_colors[ci])
        cell.get_text().set_color(row_text_colors[ci])

ax2.set_title(
    "Performance Summary (sorted by Sharpe)", color=PALETTE["text"], fontsize=11, pad=8
)

out_path = "portfolio_low_corr.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
print(f"\n  Plot saved -> {out_path}")

try:
    plt.show()
except Exception:
    pass

print(f"\n{'=' * 80}")
print("  Done!")
print(f"{'=' * 80}\n")
