"""
SP500-style B3 Index ("Índice SP-B3") — standalone runner
==========================================================
Builds the point-in-time, survivorship-bias-free SP-B3 index in BOTH
weighting schemes (market-cap and equal weight), plots equity curves vs
IBOV and CDI, writes per-rebalance holdings to
backtests/output/sp500_b3_holdings.csv, and prints a metrics table.

Methodology: see backtests/strategies/sp500_b3.py (the selection logic and
index accumulation live there as pure functions; this file only orchestrates
data loading, plotting and reporting).

Usage (after the pipeline has populated the DB):
    python -m b3_pipeline.main          # prices + corporate actions
    python -m b3_pipeline.cvm_main      # fundamentals_pit (net_income_ttm)
    python -m backtests.sp500_b3_index
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pandas as pd

from backtests.core.data import load_b3_data, download_benchmark, download_cdi_daily
from backtests.core.metrics import build_metrics, display_metrics_table, cumret
from backtests.core.plotting import PALETTE, fmt_ax
from backtests.strategies.sp500_b3 import (
    DEFAULT_MIN_ADTV,
    DEFAULT_MIN_CONSTITUENTS,
    DEFAULT_MIN_MARKET_CAP,
    LIQUIDITY_WINDOW,
    build_index_series,
    compute_weights,
    deflated_threshold,
    download_ipca_index,
    infer_volume_scale,
    load_fundamentals_pit_raw,
    load_stock_actions,
    quarter_end_rebalance_dates,
    select_constituents,
)

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE = "1995-01-01"
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")
MIN_MARKET_CAP = DEFAULT_MIN_MARKET_CAP   # R$ 2bn (nominal)
MIN_ADTV = DEFAULT_MIN_ADTV               # R$ 2M 63d median financial volume
IPCA_DEFLATE = False                      # deflate cap threshold with IPCA (SGS 433)
SLIPPAGE = 0.0                            # it's an index — no costs by default

_HERE = Path(__file__).resolve().parent
OUT_PNG = _HERE / "sp500_b3_backtest.png"
OUT_CSV = _HERE / "output" / "sp500_b3_holdings.csv"


def _resolve_db_path() -> Path:
    candidates = [
        _HERE.parent / "b3_market_data.sqlite",  # repo root (pipeline default)
        _HERE / "b3_market_data.sqlite",         # legacy backtests/ location
        Path("b3_market_data.sqlite"),
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 0:
            return p
    _fail_not_ready("database file not found")


def _fail_not_ready(reason: str):
    raise SystemExit(
        f"DB not ready ({reason}) — run `python -m b3_pipeline.main` and "
        f"`python -m b3_pipeline.cvm_main` first."
    )


def _check_db_ready(db_path: Path):
    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        try:
            n_prices = cur.execute("SELECT COUNT(*) FROM prices").fetchone()[0]
        except sqlite3.OperationalError:
            _fail_not_ready("prices table missing")
        if n_prices == 0:
            _fail_not_ready("prices table empty")
        try:
            n_ttm = cur.execute(
                "SELECT COUNT(*) FROM fundamentals_pit WHERE net_income_ttm IS NOT NULL"
            ).fetchone()[0]
        except sqlite3.OperationalError:
            _fail_not_ready("fundamentals_pit table (or net_income_ttm column) missing")
        if n_ttm == 0:
            _fail_not_ready("fundamentals_pit has no net_income_ttm data")


def _last_trade_dates(fin_vol: pd.DataFrame) -> pd.Series:
    """Last actual trading date per ticker (fin_vol is NaN on non-traded days)."""
    traded = fin_vol.where(fin_vol > 0)
    return pd.Series(
        {c: traded[c].last_valid_index() for c in traded.columns}
    )


def _plot(mw_ret, ew_ret, ibov_ret, cdi_ret, metrics, title, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    plt.rcParams.update({
        "font.family": "monospace",
        "figure.facecolor": PALETTE["bg"],
        "text.color": PALETTE["text"],
        "axes.facecolor": PALETTE["panel"],
    })
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 1, figure=fig, hspace=0.4, height_ratios=[2.2, 1.2, 1.0],
                  left=0.07, right=0.97, top=0.90, bottom=0.06)

    curves = {
        "SP-B3 Market-Cap": (mw_ret, PALETTE["pretax"], "-"),
        "SP-B3 Equal-Weight": (ew_ret, PALETTE["aftertax"], "-."),
        "IBOV": (ibov_ret, PALETTE["ibov"], "--"),
        "CDI": (cdi_ret, "#A8B2C1", ":"),
    }

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    for label, (r, color, ls) in curves.items():
        if r is None or r.empty:
            continue
        c = cumret(r.fillna(0))
        ax1.plot(c.index, c.values, color=color, lw=2.0, ls=ls, label=label)
        dd = (c / c.cummax() - 1) * 100
        ax2.fill_between(dd.index, dd.values, 0, alpha=0.30, color=color, label=label)
    ax1.set_yscale("log")
    ax1.set_title("Cumulative Return (log scale)", color=PALETTE["text"],
                  fontsize=13, fontweight="bold", pad=10)
    ax1.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=9, ncol=4)
    fmt_ax(ax1, ylabel="Growth of R$1")
    ax2.set_title("Drawdown (%)", color=PALETTE["text"], fontsize=11, pad=6)
    ax2.legend(facecolor=PALETTE["bg"], edgecolor=PALETTE["grid"],
               labelcolor=PALETTE["text"], fontsize=9, ncol=4)
    fmt_ax(ax2, ylabel="Drawdown (%)")

    ax3 = fig.add_subplot(gs[2])
    ax3.axis("off")
    col_labels = list(metrics[0].keys())
    rows = [[str(m[k]) for k in col_labels] for m in metrics]
    tbl = ax3.table(cellText=rows, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        cell.set_facecolor("#1F2937" if r == 0 else PALETTE["panel"])
        cell.get_text().set_color(PALETTE["text"])
        if r == 0:
            cell.get_text().set_fontweight("bold")
    ax3.set_title("Performance Summary", color=PALETTE["text"], fontsize=10, pad=6)

    fig.suptitle(title, fontsize=12, fontweight="bold", color=PALETTE["text"], y=0.96)
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)
    print(f"📊  Plot saved → {out_path}")


def main():
    print("\n" + "=" * 70)
    print("  ÍNDICE SP-B3 — S&P 500-METHODOLOGY INDEX ON B3")
    print("=" * 70)

    db_path = _resolve_db_path()
    _check_db_ready(db_path)

    adj_close, close_px, fin_vol = load_b3_data(str(db_path), START_DATE, END_DATE)
    fund = load_fundamentals_pit_raw(str(db_path))
    actions = load_stock_actions(str(db_path))

    scale = infer_volume_scale(fin_vol)
    print(f"ℹ  Financial volume scale factor: x{scale:g} "
          f"(see infer_volume_scale docstring)")
    med63 = fin_vol.fillna(0.0).rolling(LIQUIDITY_WINDOW).median() * scale

    ipca_index = pd.Series(dtype=float)
    if IPCA_DEFLATE:
        ipca_index = download_ipca_index(START_DATE, END_DATE)

    # ── Quarterly selection ───────────────────────────────────────────────────
    rebal_dates = quarter_end_rebalance_dates(close_px.index)
    holdings_mw: dict = {}
    holdings_ew: dict = {}
    holdings_rows = []
    started = False

    for t in rebal_dates:
        thr = (deflated_threshold(MIN_MARKET_CAP, ipca_index, t)
               if IPCA_DEFLATE else MIN_MARKET_CAP)
        sel = select_constituents(
            t, fund, close_px.loc[t], med63.loc[t], actions,
            min_market_cap=thr, min_adtv=MIN_ADTV,
        )
        if not started:
            if len(sel) < DEFAULT_MIN_CONSTITUENTS:
                continue
            started = True
            print(f"▶  Index inception: {t.date()} ({len(sel)} constituents)")
        if sel.empty:
            continue
        caps = sel.set_index("ticker")["market_cap"]
        w_mw = compute_weights(caps, "market_cap")
        w_ew = compute_weights(caps, "equal")
        holdings_mw[t] = w_mw
        holdings_ew[t] = w_ew
        for row in sel.itertuples():
            holdings_rows.append({
                "rebalance_date": t.date(),
                "root": row.root,
                "ticker": row.ticker,
                "market_cap": round(row.market_cap, 0),
                "med_volume_63d": round(row.med_vol, 0),
                "roe": round(row.roe, 4),
                "weight_market_cap": round(w_mw[row.ticker], 6),
                "weight_equal": round(w_ew[row.ticker], 6),
            })

    if not holdings_mw:
        raise SystemExit(
            "No rebalance date had >= 20 passing companies. Check that "
            "fundamentals_pit is fully populated (python -m b3_pipeline.cvm_main)."
        )

    # ── Index accumulation (buy-and-hold between rebalances) ─────────────────
    last_trade = _last_trade_dates(fin_vol)
    end = adj_close.index[-1]
    mw_level = build_index_series(adj_close, holdings_mw, last_trade, end, SLIPPAGE)
    ew_level = build_index_series(adj_close, holdings_ew, last_trade, end, SLIPPAGE)
    mw_ret = mw_level.pct_change().dropna()
    ew_ret = ew_level.pct_change().dropna()

    # ── Benchmarks ────────────────────────────────────────────────────────────
    start = str(mw_level.index[0].date())
    ibov_ret = cdi_ret = pd.Series(dtype=float)
    try:
        ibov_ret = download_benchmark("^BVSP", start, END_DATE).pct_change().dropna()
        cdi = download_cdi_daily(start, END_DATE)
        cdi_ret = cdi.reindex(mw_ret.index).fillna(0.0)
        ibov_ret = ibov_ret.reindex(mw_ret.index).dropna()
    except Exception as e:
        print(f"⚠  Benchmark download failed ({e}) — continuing without IBOV/CDI.")

    # ── Metrics (daily returns, 252 periods/year) ─────────────────────────────
    metrics = [
        build_metrics(mw_ret, "SP-B3 MktCap", 252),
        build_metrics(ew_ret, "SP-B3 EqualWt", 252),
    ]
    if not ibov_ret.empty:
        metrics.append(build_metrics(ibov_ret, "IBOV", 252))
    if not cdi_ret.empty:
        metrics.append(build_metrics(cdi_ret, "CDI", 252))
    display_metrics_table(metrics)

    # ── Constituent count stats per year ─────────────────────────────────────
    counts = pd.Series({t: len(w) for t, w in holdings_mw.items()})
    per_year = counts.groupby(counts.index.year).agg(["mean", "min", "max"])
    print("Constituent count per year (avg / min / max):")
    for year, row in per_year.iterrows():
        print(f"  {year}: {row['mean']:6.1f} / {int(row['min']):3d} / {int(row['max']):3d}")
    print(f"  Overall: {counts.mean():.1f} / {int(counts.min())} / {int(counts.max())}\n")

    # ── Outputs ───────────────────────────────────────────────────────────────
    os.makedirs(OUT_CSV.parent, exist_ok=True)
    pd.DataFrame(holdings_rows).to_csv(OUT_CSV, index=False)
    print(f"💾  Holdings saved → {OUT_CSV}")

    _plot(
        mw_ret, ew_ret, ibov_ret, cdi_ret, metrics,
        title=(
            "Índice SP-B3 — S&P 500 Methodology on B3\n"
            f"Cap > R$ {MIN_MARKET_CAP/1e9:.0f}bn · 63d median vol > "
            f"R$ {MIN_ADTV/1e6:.0f}M · NI(TTM) > 0 · ROE > 0 · quarterly\n"
            f"{start} – {END_DATE}"
        ),
        out_path=OUT_PNG,
    )


if __name__ == "__main__":
    main()
