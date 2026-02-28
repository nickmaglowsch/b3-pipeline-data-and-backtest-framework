"""
Sector Rotation Strategy Backtest
====================================
Tests three sector rotation variants for the B3 market:
  1. Sector Momentum          -- long top 2 sectors by 6-month return
  2. Sector Momentum + Stock  -- top 2 sectors + MultiFactor within each sector
  3. Sector + COPOM Regime    -- sector rotation only during COPOM easing

Uses a heuristic ticker-to-sector mapping based on the first 4 characters of
the B3 ticker code.

Usage:
    python3 backtests/sector_rotation_backtest.py
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
SECTOR_MOMENTUM_LOOKBACK = 6  # months for sector momentum signal

DB_PATH = os.path.join(_PROJECT_ROOT, "b3_market_data.sqlite")
OUT_DIR = _BACKTESTS_DIR

# ─── Sector mapping ───────────────────────────────────────────────────────────
SECTOR_MAP = {
    # Financials
    "ITUB": "Financials", "BBDC": "Financials", "BBAS": "Financials",
    "SANB": "Financials", "BPAC": "Financials", "B3SA": "Financials",
    "ITSA": "Financials", "BRSR": "Financials", "ABCB": "Financials",
    "BMGB": "Financials", "BPAN": "Financials",

    # Commodities / Mining / Oil & Gas
    "VALE": "Commodities", "PETR": "Commodities", "CSNA": "Commodities",
    "GGBR": "Commodities", "USIM": "Commodities", "GOAU": "Commodities",
    "CMIN": "Commodities", "BRAP": "Commodities", "SUZB": "Commodities",
    "KLBN": "Commodities", "DXCO": "Commodities",

    # Utilities
    "ELET": "Utilities", "SBSP": "Utilities", "CMIG": "Utilities",
    "CPFE": "Utilities", "EGIE": "Utilities", "ENGI": "Utilities",
    "TAEE": "Utilities", "TRPL": "Utilities", "CPLE": "Utilities",
    "NEOE": "Utilities", "AURE": "Utilities", "SAPR": "Utilities",
    "ENEV": "Utilities",

    # Consumer / Retail
    "MGLU": "Consumer", "LREN": "Consumer", "AMER": "Consumer",
    "VIIA": "Consumer", "PETZ": "Consumer", "SOMA": "Consumer",
    "ARZZ": "Consumer", "GRND": "Consumer", "ALPA": "Consumer",
    "NTCO": "Consumer", "ABEV": "Consumer", "JBSS": "Consumer",
    "BRFS": "Consumer", "MRFG": "Consumer", "BEEF": "Consumer",
    "MDIA": "Consumer", "PCAR": "Consumer", "CRFB": "Consumer",
    "ASAI": "Consumer", "RAIZ": "Consumer",

    # Real Estate
    "CYRE": "RealEstate", "MRVE": "RealEstate", "EZTC": "RealEstate",
    "EVEN": "RealEstate", "DIRR": "RealEstate", "TEND": "RealEstate",
    "MULT": "RealEstate", "IGTI": "RealEstate", "BRML": "RealEstate",
    "ALSO": "RealEstate",

    # Healthcare
    "HAPV": "Healthcare", "RDOR": "Healthcare", "FLRY": "Healthcare",
    "QUAL": "Healthcare", "HYPE": "Healthcare",

    # Telecom / Tech
    "VIVT": "TechTelecom", "TIMS": "TechTelecom", "TOTS": "TechTelecom",
    "LWSA": "TechTelecom", "POSI": "TechTelecom", "INTB": "TechTelecom",

    # Transportation / Infrastructure
    "CCRO": "Infrastructure", "ECOR": "Infrastructure", "RAIL": "Infrastructure",
    "AZUL": "Infrastructure", "GOLL": "Infrastructure", "EMBR": "Infrastructure",
    "RENT": "Infrastructure", "MOVI": "Infrastructure",

    # Insurance
    "BBSE": "Insurance", "PSSA": "Insurance", "SULA": "Insurance",
    "IRBR": "Insurance",
}


def classify_ticker(ticker: str) -> str:
    """Classify a B3 ticker into a sector using the first 4 chars."""
    prefix = ticker[:4]
    return SECTOR_MAP.get(prefix, "Other")


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
    print("  Sector Rotation Strategy Backtest")
    print("=" * 70)

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

    is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    # ── Sector classification ─────────────────────────────────────────────────
    all_tickers = list(ret.columns)
    sector_assignments = {t: classify_ticker(t) for t in all_tickers}

    sectors = [s for s in set(sector_assignments.values()) if s != "Other"]
    print(f"\nSectors identified ({len(sectors)}): {sorted(sectors)}")

    # Coverage statistics on liquid universe
    liquid_mask_sample = (adtv.iloc[-1] >= MIN_ADTV) & (raw_close.iloc[-1] >= MIN_PRICE)
    liquid_tickers = liquid_mask_sample[liquid_mask_sample].index.tolist()
    mapped = [t for t in liquid_tickers if sector_assignments.get(t, "Other") != "Other"]
    coverage = len(mapped) / len(liquid_tickers) if liquid_tickers else 0
    print(f"Liquid universe: {len(liquid_tickers)} tickers")
    print(f"Mapped to named sectors: {len(mapped)} ({coverage*100:.1f}%)")

    if coverage < 0.70:
        print("  WARNING: Sector coverage below 70% -- sector rotation may be unreliable.")

    # ── Compute sector equal-weighted returns ─────────────────────────────────
    print("\nComputing sector returns...")
    sector_ret_data = {}
    for sector in sectors:
        sector_tickers = [t for t, s in sector_assignments.items() if s == sector and t in ret.columns]
        if len(sector_tickers) >= 2:
            sector_ret_data[sector] = ret[sector_tickers].mean(axis=1)

    if len(sector_ret_data) < 5:
        print(f"  WARNING: Only {len(sector_ret_data)} sectors have >= 2 tickers. "
              "Sector rotation may be limited.")

    sector_returns = pd.DataFrame(sector_ret_data)
    active_sectors = list(sector_returns.columns)
    print(f"Active sectors with >= 2 tickers: {active_sectors}")

    # Sector 6-month cumulative momentum
    sector_mom = sector_returns.rolling(SECTOR_MOMENTUM_LOOKBACK).apply(
        lambda x: (1 + x).prod() - 1, raw=False
    )

    # MultiFactor composite signal for within-sector stock selection
    mom_sig = log_ret.shift(1).rolling(LOOKBACK).sum()
    mom_sig[has_glitch == 1] = np.nan
    vol_sig = -ret.shift(1).rolling(LOOKBACK).std()
    vol_sig[has_glitch == 1] = np.nan
    mf_composite = (
        mom_sig.rank(axis=1, pct=True) * 0.5
        + vol_sig.rank(axis=1, pct=True) * 0.5
    )

    results = []

    # ── Variant 1: Sector Momentum (equal weight within top 2 sectors) ────────
    print("\n[1/3] Sector Momentum...")
    tw_v1 = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_v1["CDI_ASSET"] = 0.0
    r_v1 = ret.copy()
    r_v1["CDI_ASSET"] = cdi_monthly

    for i in range(SECTOR_MOMENTUM_LOOKBACK + 1, len(ret)):
        if i - 1 >= len(sector_mom):
            tw_v1.iloc[i, tw_v1.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        mom_row = sector_mom.iloc[i - 1].dropna()
        if len(mom_row) == 0 or mom_row.max() <= 0:
            tw_v1.iloc[i, tw_v1.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        top_sectors = mom_row.nlargest(min(2, len(mom_row))).index.tolist()
        # Only include sectors with positive momentum
        top_sectors = [s for s in top_sectors if mom_row[s] > 0]

        if not top_sectors:
            tw_v1.iloc[i, tw_v1.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask_liq = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)

        # Collect tickers from top sectors
        selected_tickers = []
        for sector in top_sectors:
            sector_tickers = [
                t for t, s in sector_assignments.items()
                if s == sector and t in ret.columns and mask_liq.get(t, False)
            ]
            selected_tickers.extend(sector_tickers)

        if not selected_tickers:
            tw_v1.iloc[i, tw_v1.columns.get_loc("CDI_ASSET")] = 1.0
        else:
            w = 1.0 / len(selected_tickers)
            for t in selected_tickers:
                if t in tw_v1.columns:
                    tw_v1.iloc[i, tw_v1.columns.get_loc(t)] = w

    m_v1, res_v1, ret_v1 = run_and_collect("SectorMomentum", r_v1, tw_v1)
    results.append(m_v1)

    # ── Variant 2: Sector Momentum + MultiFactor within sectors ───────────────
    print("\n[2/3] Sector Momentum + Stock Selection...")
    tw_v2 = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_v2["CDI_ASSET"] = 0.0
    r_v2 = ret.copy()
    r_v2["CDI_ASSET"] = cdi_monthly

    for i in range(max(SECTOR_MOMENTUM_LOOKBACK, LOOKBACK) + 2, len(ret)):
        if i - 1 >= len(sector_mom):
            tw_v2.iloc[i, tw_v2.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        mom_row = sector_mom.iloc[i - 1].dropna()
        if len(mom_row) == 0 or mom_row.max() <= 0:
            tw_v2.iloc[i, tw_v2.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        top_sectors = mom_row.nlargest(min(2, len(mom_row))).index.tolist()
        top_sectors = [s for s in top_sectors if mom_row[s] > 0]

        if not top_sectors:
            tw_v2.iloc[i, tw_v2.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask_liq = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)
        mf_row = mf_composite.iloc[i - 1]

        all_selected = []
        for sector in top_sectors:
            sector_tickers = [
                t for t, s in sector_assignments.items()
                if s == sector and t in ret.columns and mask_liq.get(t, False)
            ]
            if not sector_tickers:
                continue
            # Apply MultiFactor within this sector
            sector_scores = mf_row[sector_tickers].dropna()
            if len(sector_scores) < 3:
                # Too few tickers in sector; use all
                all_selected.extend(sector_tickers)
            else:
                n = max(1, int(len(sector_scores) * 0.30))
                top_in_sector = sector_scores.nlargest(n).index.tolist()
                all_selected.extend(top_in_sector)

        if not all_selected:
            tw_v2.iloc[i, tw_v2.columns.get_loc("CDI_ASSET")] = 1.0
        else:
            w = 1.0 / len(all_selected)
            for t in all_selected:
                if t in tw_v2.columns:
                    tw_v2.iloc[i, tw_v2.columns.get_loc(t)] = w

    m_v2, res_v2, ret_v2 = run_and_collect("SectorMom+MFStock", r_v2, tw_v2)
    results.append(m_v2)

    # ── Variant 3: Sector Rotation + COPOM Regime ────────────────────────────
    print("\n[3/3] Sector Rotation + COPOM Regime...")
    tw_v3 = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw_v3["CDI_ASSET"] = 0.0
    r_v3 = ret.copy()
    r_v3["CDI_ASSET"] = cdi_monthly

    for i in range(max(SECTOR_MOMENTUM_LOOKBACK, LOOKBACK) + 2, len(ret)):
        easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        if not easing:
            tw_v3.iloc[i, tw_v3.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        if i - 1 >= len(sector_mom):
            tw_v3.iloc[i, tw_v3.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        mom_row = sector_mom.iloc[i - 1].dropna()
        if len(mom_row) == 0 or mom_row.max() <= 0:
            tw_v3.iloc[i, tw_v3.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        top_sectors = mom_row.nlargest(min(2, len(mom_row))).index.tolist()
        top_sectors = [s for s in top_sectors if mom_row[s] > 0]

        if not top_sectors:
            tw_v3.iloc[i, tw_v3.columns.get_loc("CDI_ASSET")] = 1.0
            continue

        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask_liq = (adtv_r >= MIN_ADTV) & (raw_r >= MIN_PRICE)

        selected_tickers = []
        for sector in top_sectors:
            sector_tickers = [
                t for t, s in sector_assignments.items()
                if s == sector and t in ret.columns and mask_liq.get(t, False)
            ]
            selected_tickers.extend(sector_tickers)

        if not selected_tickers:
            tw_v3.iloc[i, tw_v3.columns.get_loc("CDI_ASSET")] = 1.0
        else:
            w = 1.0 / len(selected_tickers)
            for t in selected_tickers:
                if t in tw_v3.columns:
                    tw_v3.iloc[i, tw_v3.columns.get_loc(t)] = w

    m_v3, res_v3, ret_v3 = run_and_collect("SectorMom+COPOM", r_v3, tw_v3)
    results.append(m_v3)

    # ── MultiFactor benchmark ─────────────────────────────────────────────────
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
    print(f"  SECTOR ROTATION -- After-Tax (15% CGT) -- {START} to {END}")
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

    # Coverage / correlation notes
    print(f"\nSector coverage: {coverage*100:.1f}% of liquid universe")
    print(f"Sectors with >= 2 liquid tickers: {active_sectors}")

    strat_rets_dict = {
        "SectorMomentum":     ret_v1,
        "SectorMom+MFStock":  ret_v2,
        "SectorMom+COPOM":    ret_v3,
        "MultiFactor":        ret_mf,
        "IBOV":               ibov_ret.dropna(),
    }
    high_sharpe_strats = [
        m["Strategy"] for m in results
        if float(m.get("Sharpe", 0)) > 0.3
        and m["Strategy"] not in ("IBOV", "CDI", "MultiFactor")
    ]
    if high_sharpe_strats:
        print(f"\nStrategies with Sharpe > 0.3: {high_sharpe_strats}")
        corr_df = pd.DataFrame({
            k: v for k, v in strat_rets_dict.items()
        }).dropna(how="all")
        print("\nCorrelation matrix:")
        print(corr_df.corr().round(3).to_string())

    # ── Plots ─────────────────────────────────────────────────────────────────
    print("\nGenerating plots...")

    # Tearsheet for best variant
    best_name = max(
        ["SectorMomentum", "SectorMom+MFStock", "SectorMom+COPOM"],
        key=lambda nm: next(float(m.get("Sharpe", 0)) for m in results if m["Strategy"] == nm),
    )
    res_map = {
        "SectorMomentum":    res_v1,
        "SectorMom+MFStock": res_v2,
        "SectorMom+COPOM":   res_v3,
    }
    best_res = res_map[best_name]
    out_path1 = os.path.join(OUT_DIR, "sector_rotation_backtest.png")
    try:
        metrics_for_plot = [m for m in results if m["Strategy"] in (
            best_name, "MultiFactor", "IBOV", "CDI"
        )][:4]
        plot_tax_backtest(
            title=f"Sector Rotation: {best_name} (15% CGT, 0.1% slip)",
            pretax_val=best_res["pretax_values"],
            aftertax_val=best_res["aftertax_values"],
            ibov_ret=ibov_ret,
            tax_paid=best_res["tax_paid"],
            loss_cf=best_res["loss_carryforward"],
            turnover=best_res["turnover"],
            metrics=metrics_for_plot,
            total_tax_brl=float(best_res["tax_paid"].sum()),
            out_path=out_path1,
            cdi_ret=cdi_ret,
        )
    except Exception as exc:
        print(f"  Tearsheet failed: {exc}")
        _simple_equity_plot(strat_rets_dict, out_path1)

    # Sector momentum heatmap (sectors x years)
    out_path2 = os.path.join(OUT_DIR, "sector_rotation_analysis.png")
    _plot_sector_heatmap(sector_returns, active_sectors, out_path2)

    print("\nDone.")


def _plot_sector_heatmap(sector_returns: pd.DataFrame, sectors: list, out_path: str):
    """Sector momentum heatmap: sectors x years, colored by annual return."""
    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
    })

    # Annual returns per sector
    annual = sector_returns[sectors].resample("YE").apply(
        lambda x: (1 + x).prod() - 1
    ) * 100

    if annual.empty:
        print(f"  No sector data for heatmap; skipping -> {out_path}")
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor(PALETTE["bg"])
    ax.set_facecolor(PALETTE["panel"])

    data = annual.T.values
    years = [str(y.year) for y in annual.index]
    sector_labels = list(annual.columns)

    vmax = np.nanmax(np.abs(data))
    im = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=-vmax, vmax=vmax)

    for (yi, xi), val in np.ndenumerate(data):
        if not np.isnan(val):
            ax.text(xi, yi, f"{val:.0f}%",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(val) < vmax * 0.6 else "white",
                    fontweight="bold")

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels(years, rotation=45, ha="right", color=PALETTE["sub"], fontsize=7)
    ax.set_yticks(range(len(sector_labels)))
    ax.set_yticklabels(sector_labels, color=PALETTE["sub"], fontsize=8)
    ax.set_title("B3 Sector Annual Returns (%)",
                 color=PALETTE["text"], fontsize=11, fontweight="bold")
    ax.spines[:].set_color(PALETTE["grid"])
    ax.tick_params(colors=PALETTE["sub"])

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.yaxis.set_tick_params(color=PALETTE["sub"])
    cbar.set_label("Annual Return (%)", color=PALETTE["sub"])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Sector heatmap saved -> {out_path}")


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
    ax.set_title("Sector Rotation Backtest", color=PALETTE["text"], fontsize=12)
    ax.legend(facecolor=PALETTE["bg"], labelcolor=PALETTE["text"], fontsize=9)
    fmt_ax(ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close()
    print(f"  Plot saved -> {out_path}")


if __name__ == "__main__":
    main()
