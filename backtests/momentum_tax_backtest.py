"""
B3 Dynamic Momentum Strategy Backtest — with Brazilian Capital Gains Tax
=======================================================================
Strategy: Each month, rank all eligible stocks by their past N-month return.
Buy the top K% performers (winners). Equal-weight, rebalance monthly.

Universe:
  • Sourced natively from local B3 SQLite database (b3_market_data.sqlite).
  • Restricted to standard stocks/units ending in 3, 4, 5, 6, 11.
  • Dynamically filtered each month to only include stocks with an Average
    Daily Traded Volume (ADTV) >= R$ 1,000,000 in the preceding month.

Tax rules (Brazilian renda variável, Lei 11.033/2004):
  • 15% capital gains tax on NET realised gains each rebalance.
  • Only CLOSED positions (stocks leaving the portfolio) trigger a tax event.
  • Losses from closed positions are carried forward indefinitely.
  • Loss carryforward offsets future gains before tax is applied.

Three equity curves are compared:
  1. Momentum — pre-tax (gross performance)
  2. Momentum — after-tax (net of 15% CGT with loss carryforward)
  3. IBOV index (Benchmark from Yahoo Finance)
"""

import warnings

warnings.filterwarnings("ignore")

import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.gridspec import GridSpec
from datetime import datetime

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
LOOKBACK_MONTHS = 12  # momentum signal window (months)
SKIP_MONTHS = 1  # skip most-recent month (avoid short-term reversal)
TOP_DECILE = 0.10  # fraction of valid universe selected each rebalance
TAX_RATE = 0.15  # Brazilian capital gains tax rate
START_DATE = "2012-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000  # BRL
MIN_ADTV = 1_000_000  # Minimum Average Daily Traded Volume in BRL

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


# ─────────────────────────────────────────────
#  DATA LOADING (Local DB)
# ─────────────────────────────────────────────
def load_b3_data(
    db_path: str, start: str, end: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from the local B3 SQLite database.
    Filters for valid tickers (ending in 3,4,5,6,11) and calculates daily financial volume.
    Returns three wide DataFrames: adj_close, close, and daily_fin_volume.
    """
    print(f"⬇  Loading B3 data from {db_path} ({start} to {end})...")

    conn = sqlite3.connect(db_path)

    # Query: standard lot tickers ending in 3, 4, 5, 6, 11
    # BDRs (34, 35, etc.) and other weird assets are excluded using strict length and suffix checks.
    query = f"""
        SELECT date, ticker, close, adj_close, volume
        FROM prices
        WHERE date >= '{start}' AND date <= '{end}'
        AND (
            (LENGTH(ticker) = 5 AND SUBSTR(ticker, 5, 1) IN ('3', '4', '5', '6'))
            OR 
            (LENGTH(ticker) = 6 AND SUBSTR(ticker, 5, 2) = '11')
        )
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])

    # Calculate daily financial volume in BRL.
    # In our DB, the `volume` column comes from COTAHIST VOLTOT (Financial Volume),
    # but it is parsed as an integer with 2 implied decimal places.
    # Therefore, true financial volume = volume / 100
    df["fin_volume"] = df["volume"] / 100.0

    # Pivot to wide format
    print("🔄  Pivoting data to wide format...")
    adj_close = df.pivot(index="date", columns="ticker", values="adj_close")
    close_px = df.pivot(index="date", columns="ticker", values="close")
    fin_vol = df.pivot(index="date", columns="ticker", values="fin_volume")

    # Forward fill prices to handle missing days, but leave volume as NaN/0
    # to accurately calculate averages
    adj_close = adj_close.ffill()
    close_px = close_px.ffill()

    print(f"✅  Loaded {adj_close.shape[1]} unique standard tickers.")
    return adj_close, close_px, fin_vol


def download_index(start: str, end: str) -> pd.Series:
    print("⬇  Downloading IBOV index benchmark from Yahoo...")
    ibov = yf.download(
        IBOV_INDEX, start=start, end=end, auto_adjust=True, progress=False
    )["Close"]

    if isinstance(ibov, pd.DataFrame):
        ibov = ibov.squeeze()

    ibov.index = pd.to_datetime(ibov.index)

    # Remove timezone if present to align with local sqlite dates
    if ibov.index.tz is not None:
        ibov.index = ibov.index.tz_localize(None)

    return ibov


# ─────────────────────────────────────────────
#  SIGNAL
# ─────────────────────────────────────────────
def momentum_signal(
    monthly_ret: pd.DataFrame, lookback: int, skip: int
) -> pd.DataFrame:
    """
    Log-cumulative return from (t-lookback) to (t-skip).
    If any single month return > 100% or < -90%, mark the signal as NaN to protect against data glitches.
    """
    log_ret = np.log1p(monthly_ret)

    # Calculate cumulative signal
    signal = log_ret.shift(skip).rolling(lookback).sum()

    # Detect data glitches (monthly return > 100% or < -90%)
    has_glitch = (
        ((monthly_ret > 1.0) | (monthly_ret < -0.90))
        .shift(skip)
        .rolling(lookback)
        .max()
    )

    # Invalidate signal if there's a glitch in the lookback window
    signal[has_glitch == 1] = np.nan

    return signal


# ─────────────────────────────────────────────
#  BACKTEST ENGINE
# ─────────────────────────────────────────────
def _rebalance_positions(
    positions: dict, exiting: set, entering: set, ret_row: pd.Series, tax: float
) -> None:
    """
    Shared rebalancing logic used by both ledgers.

    1. Remove exiting positions and collect their liquidation cash.
    2. Deduct tax from available cash (after-tax ledger passes tax > 0;
       pre-tax ledger always passes tax = 0).
    3. Deploy cash into new entrants.  If there are no new entrants,
       distribute freed cash pro-rata to surviving continuing positions
       so no capital leaks out.
    """
    exit_cash = 0.0
    for t in exiting:
        if t in positions:
            exit_cash += positions[t]["current_value"]
            del positions[t]

    cash_available = exit_cash - tax

    if cash_available <= 0:
        return

    if entering:
        alloc_each = cash_available / len(entering)
        for t in entering:
            positions[t] = {"cost_basis": alloc_each, "current_value": alloc_each}
    else:
        continuing_total = sum(p["current_value"] for p in positions.values())
        if continuing_total > 0:
            for pos in positions.values():
                injected = cash_available * pos["current_value"] / continuing_total
                pos["cost_basis"] += injected
                pos["current_value"] += injected


def run_backtest(
    adj_close: pd.DataFrame,
    close_px: pd.DataFrame,
    fin_vol: pd.DataFrame,
    top_pct: float = TOP_DECILE,
) -> dict:
    """
    Simulate two portfolios in parallel using dynamic universe filtering.
    """
    # Resample to monthly
    monthly_px = adj_close.resample("ME").last()
    monthly_ret = monthly_px.pct_change()

    # We also need the monthly raw close to filter out penny stocks (< R$ 1.00)
    monthly_raw_close = close_px.resample("ME").last()

    # Calculate Average Daily Traded Volume per month
    # We use ME (Month End) to align with prices
    monthly_adtv = fin_vol.resample("ME").mean()

    # Generate momentum signals
    signal = momentum_signal(monthly_ret, LOOKBACK_MONTHS, SKIP_MONTHS)

    start_idx = LOOKBACK_MONTHS + SKIP_MONTHS + 1

    pretax_positions = {}
    aftertax_positions = {}
    loss_carryforward = 0.0

    pretax_values = []
    aftertax_values = []
    tax_paid_list = []
    loss_cf_list = []
    turnover_list = []
    dates = []
    prev_selected = set()
    initialized = False

    for i in range(start_idx, len(monthly_ret)):
        date = monthly_ret.index[i]

        # Signal is based on data up to i-1
        sig_row = signal.iloc[i - 1]

        # Liquidity filter: ADTV of the PREVIOUS month must be >= MIN_ADTV
        adtv_row = monthly_adtv.iloc[i - 1]

        # Penny stock filter: raw close price of the PREVIOUS month must be >= R$ 1.00
        raw_close_row = monthly_raw_close.iloc[i - 1]

        valid_universe_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)

        # Apply mask to signal
        valid = sig_row[valid_universe_mask].dropna()

        if len(valid) < 5:
            # If not enough liquid stocks, just hold current portfolio (skip rebalance)
            new_selected = prev_selected
        else:
            n_select = max(1, int(len(valid) * top_pct))
            new_selected = set(valid.nlargest(n_select).index.tolist())

        ret_row = monthly_ret.iloc[i]

        # ── First month: open equal-weight positions ───
        if not initialized and len(valid) >= 5:
            alloc = INITIAL_CAPITAL / len(new_selected)
            for t in new_selected:
                pretax_positions[t] = {"cost_basis": alloc, "current_value": alloc}
                aftertax_positions[t] = {"cost_basis": alloc, "current_value": alloc}
            prev_selected = new_selected
            initialized = True

            pretax_values.append(INITIAL_CAPITAL)
            aftertax_values.append(INITIAL_CAPITAL)
            tax_paid_list.append(0.0)
            loss_cf_list.append(0.0)
            turnover_list.append(1.0)
            dates.append(date)
            continue

        if not initialized:
            continue

        # ── Step 1: apply this month's return to ALL open positions ────
        for ledger in (pretax_positions, aftertax_positions):
            for t, pos in ledger.items():
                r = ret_row.get(t, np.nan)
                if pd.isna(r):
                    r = 0.0

                # Conservative cap to protect against unadjusted split data glitches
                r = min(max(r, -0.90), 1.0)

                pos["current_value"] *= 1.0 + r

        # ── Step 2: classify positions ─────────────────────────────────
        exiting = prev_selected - new_selected
        entering = new_selected - prev_selected
        n_universe = len(new_selected | prev_selected)
        turnover = len(exiting | entering) / max(n_universe, 1)
        turnover_list.append(turnover)

        # ── Step 3: compute tax for after-tax ledger ───────────────────
        gross_gain = 0.0
        gross_loss = 0.0

        for t in exiting:
            if t not in aftertax_positions:
                continue
            cv = aftertax_positions[t]["current_value"]
            cb = aftertax_positions[t]["cost_basis"]
            pnl = cv - cb
            if pnl > 0:
                gross_gain += pnl
            else:
                gross_loss += abs(pnl)

        net_pnl = gross_gain - gross_loss

        if net_pnl > 0:
            net_after_cf = net_pnl - loss_carryforward
            if net_after_cf > 0:
                tax = TAX_RATE * net_after_cf
                loss_carryforward = 0.0
            else:
                tax = 0.0
                loss_carryforward = abs(net_after_cf)
        else:
            tax = 0.0
            loss_carryforward += abs(net_pnl)

        # ── Step 4: rebalance both ledgers ─────────────────────────────
        _rebalance_positions(pretax_positions, exiting, entering, ret_row, tax=0.0)
        _rebalance_positions(aftertax_positions, exiting, entering, ret_row, tax=tax)

        # ── Step 5: record state ───────────────────────────────────────
        pretax_total = sum(p["current_value"] for p in pretax_positions.values())
        aftertax_total = sum(p["current_value"] for p in aftertax_positions.values())

        pretax_values.append(pretax_total)
        aftertax_values.append(aftertax_total)
        tax_paid_list.append(tax)
        loss_cf_list.append(loss_carryforward)
        dates.append(date)

        prev_selected = new_selected

    idx = pd.DatetimeIndex(dates)

    return {
        "pretax_values": pd.Series(pretax_values, index=idx, name="Momentum (Pre-Tax)"),
        "aftertax_values": pd.Series(
            aftertax_values, index=idx, name="Momentum (After-Tax)"
        ),
        "tax_paid": pd.Series(tax_paid_list, index=idx, name="Tax Paid (BRL)"),
        "loss_carryforward": pd.Series(
            loss_cf_list, index=idx, name="Loss Carryforward (BRL)"
        ),
        "turnover": pd.Series(turnover_list, index=idx, name="Turnover"),
    }


# ─────────────────────────────────────────────
#  PERFORMANCE METRICS
# ─────────────────────────────────────────────
def cumret(ret: pd.Series) -> pd.Series:
    return (1 + ret).cumprod()


def value_to_ret(values: pd.Series) -> pd.Series:
    return values.pct_change().fillna(0)


def ann_return(ret: pd.Series) -> float:
    n = len(ret) / 12
    return (1 + ret).prod() ** (1 / n) - 1 if n > 0 else 0.0


def ann_vol(ret: pd.Series) -> float:
    return ret.std() * np.sqrt(12)


def sharpe(ret: pd.Series) -> float:
    return (ret.mean() / ret.std()) * np.sqrt(12) if ret.std() != 0 else 0.0


def max_dd(ret: pd.Series) -> float:
    cum = cumret(ret)
    return (cum / cum.cummax() - 1).min()


def calmar(ret: pd.Series) -> float:
    mdd = abs(max_dd(ret))
    return ann_return(ret) / mdd if mdd != 0 else 0.0


def build_metrics(ret: pd.Series, label: str) -> dict:
    return {
        "Strategy": label,
        "Ann. Return (%)": round(ann_return(ret) * 100, 2),
        "Ann. Volatility (%)": round(ann_vol(ret) * 100, 2),
        "Sharpe": round(sharpe(ret), 2),
        "Max Drawdown (%)": round(max_dd(ret) * 100, 2),
        "Calmar": round(calmar(ret), 2),
    }


# ─────────────────────────────────────────────
#  PLOTTING
# ─────────────────────────────────────────────
PALETTE = {
    "pretax": "#00D4AA",
    "aftertax": "#7B61FF",
    "ibov": "#FF6B35",
    "tax": "#FF4C6A",
    "loss_cf": "#FFC947",
    "bg": "#0D1117",
    "panel": "#161B22",
    "grid": "#21262D",
    "text": "#E6EDF3",
    "sub": "#8B949E",
}


def fmt(ax, ylabel=""):
    ax.set_facecolor(PALETTE["panel"])
    ax.tick_params(colors=PALETTE["sub"], labelsize=8.5)
    ax.spines[:].set_color(PALETTE["grid"])
    if ylabel:
        ax.set_ylabel(ylabel, color=PALETTE["sub"], fontsize=9)
    ax.grid(axis="y", color=PALETTE["grid"], lw=0.6, ls="--")
    ax.grid(axis="x", color=PALETTE["grid"], lw=0.3, ls=":")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))


def plot_results(
    pretax_val: pd.Series,
    aftertax_val: pd.Series,
    ibov_ret: pd.Series,
    tax_paid: pd.Series,
    loss_cf: pd.Series,
    turnover: pd.Series,
    metrics: list,
    total_tax_brl: float,
):
    plt.rcParams.update(
        {
            "font.family": "monospace",
            "figure.facecolor": PALETTE["bg"],
            "text.color": PALETTE["text"],
            "axes.facecolor": PALETTE["panel"],
        }
    )

    fig = plt.figure(figsize=(18, 16))
    gs = GridSpec(
        4,
        3,
        figure=fig,
        hspace=0.48,
        wspace=0.35,
        left=0.06,
        right=0.97,
        top=0.92,
        bottom=0.05,
    )

    common = pretax_val.index.intersection(ibov_ret.index)
    pt_val = pretax_val.loc[common]
    ibov = ibov_ret.loc[common]
    at_val = aftertax_val.loc[common]
    tx = tax_paid.loc[common]
    lc = loss_cf.loc[common]
    tv = turnover.loc[common]

    pt_curve = pt_val / pt_val.iloc[0]
    at_curve = at_val / at_val.iloc[0]
    ibov_curve = cumret(ibov)

    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(
        pt_curve.index,
        pt_curve.values,
        color=PALETTE["pretax"],
        lw=2.2,
        label="Momentum Pre-Tax",
        zorder=4,
    )
    ax1.plot(
        at_curve.index,
        at_curve.values,
        color=PALETTE["aftertax"],
        lw=2.2,
        label="Momentum After-Tax",
        zorder=3,
        ls="-.",
    )
    ax1.plot(
        ibov_curve.index,
        ibov_curve.values,
        color=PALETTE["ibov"],
        lw=1.8,
        label="IBOV Index",
        zorder=2,
        ls="--",
    )
    ax1.fill_between(
        pt_curve.index,
        pt_curve.values,
        at_curve.values,
        alpha=0.15,
        color=PALETTE["tax"],
        label="Tax Drag",
    )
    ax1.set_title(
        "Cumulative Return — Pre-Tax vs After-Tax vs IBOV",
        color=PALETTE["text"],
        fontsize=13,
        fontweight="bold",
        pad=10,
    )
    ax1.legend(
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=9,
        ncol=4,
    )
    fmt(ax1, ylabel="Growth of R$1")

    ax2 = fig.add_subplot(gs[1, :])
    dd_pt = (pt_curve / pt_curve.cummax() - 1) * 100
    dd_at = (at_curve / at_curve.cummax() - 1) * 100
    dd_ibov = (ibov_curve / ibov_curve.cummax() - 1) * 100
    ax2.fill_between(
        dd_pt.index,
        dd_pt.values,
        0,
        alpha=0.45,
        color=PALETTE["pretax"],
        label="Pre-Tax DD",
    )
    ax2.fill_between(
        dd_at.index,
        dd_at.values,
        0,
        alpha=0.35,
        color=PALETTE["aftertax"],
        label="After-Tax DD",
    )
    ax2.fill_between(
        dd_ibov.index,
        dd_ibov.values,
        0,
        alpha=0.25,
        color=PALETTE["ibov"],
        label="IBOV DD",
    )
    ax2.set_title("Drawdown (%)", color=PALETTE["text"], fontsize=11, pad=6)
    ax2.legend(
        facecolor=PALETTE["bg"],
        edgecolor=PALETTE["grid"],
        labelcolor=PALETTE["text"],
        fontsize=9,
        ncol=3,
    )
    fmt(ax2, ylabel="Drawdown (%)")

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(tx.index, tx.values / 1_000, color=PALETTE["tax"], width=20, alpha=0.8)
    ax3.set_title(
        "Tax Paid per Month (R$ k)", color=PALETTE["text"], fontsize=10, pad=6
    )
    fmt(ax3, ylabel="R$ thousands")

    ax4 = fig.add_subplot(gs[2, 1])
    ax4.fill_between(
        lc.index, lc.values / 1_000, 0, alpha=0.6, color=PALETTE["loss_cf"]
    )
    ax4.plot(lc.index, lc.values / 1_000, color=PALETTE["loss_cf"], lw=1.4)
    ax4.set_title(
        "Loss Carryforward Balance (R$ k)", color=PALETTE["text"], fontsize=10, pad=6
    )
    fmt(ax4, ylabel="R$ thousands")

    ax5 = fig.add_subplot(gs[2, 2])
    ax5.bar(tv.index, tv.values * 100, color=PALETTE["pretax"], width=20, alpha=0.7)
    ax5.set_title(
        "Monthly Portfolio Turnover (%)", color=PALETTE["text"], fontsize=10, pad=6
    )
    fmt(ax5, ylabel="%")

    ax6 = fig.add_subplot(gs[3, 0:2])
    spread = (pt_curve - at_curve) * 100
    ax6.fill_between(spread.index, spread.values, 0, alpha=0.55, color=PALETTE["tax"])
    ax6.plot(spread.index, spread.values, color=PALETTE["tax"], lw=1.4)
    ax6.axhline(0, color=PALETTE["sub"], lw=0.8)
    ax6.set_title(
        "Cumulative Tax Drag: Pre-Tax minus After-Tax (percentage points)",
        color=PALETTE["text"],
        fontsize=10,
        pad=6,
    )
    fmt(ax6, ylabel="pp")
    ax6.annotate(
        f"Total tax paid: R$ {total_tax_brl:,.0f}",
        xy=(0.02, 0.88),
        xycoords="axes fraction",
        fontsize=9,
        color=PALETTE["tax"],
        bbox=dict(
            boxstyle="round,pad=0.3", fc=PALETTE["panel"], ec=PALETTE["tax"], alpha=0.8
        ),
    )

    ax7 = fig.add_subplot(gs[3, 2])
    ax7.axis("off")
    col_labels = list(metrics[0].keys())
    row_vals = [[str(m[k]) for k in col_labels] for m in metrics]
    row_bg = ["#0D2E26", "#1A1230", "#2A1A10"]
    txt_colors = [PALETTE["pretax"], PALETTE["aftertax"], PALETTE["ibov"]]

    tbl = ax7.table(
        cellText=row_vals, colLabels=col_labels, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.8)

    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(PALETTE["grid"])
        if r == 0:
            cell.set_facecolor("#1F2937")
            cell.get_text().set_color(PALETTE["text"])
            cell.get_text().set_fontweight("bold")
        else:
            cell.set_facecolor(row_bg[r - 1])
            cell.get_text().set_color(txt_colors[r - 1])

    ax7.set_title(
        "Performance Summary", color=PALETTE["text"], fontsize=10, pad=6, y=0.98
    )

    fig.suptitle(
        f"Dynamic Momentum (B3 Native)  ·  {LOOKBACK_MONTHS}M Lookback · Skip {SKIP_MONTHS}M\n"
        f"Top {int(TOP_DECILE * 100)}% of R$ {MIN_ADTV / 1_000_000:.0f}M+ ADTV  ·  15% CGT with Loss Carryforward\n"
        f"{START_DATE[:4]}–{END_DATE[:4]}",
        fontsize=12,
        fontweight="bold",
        color=PALETTE["text"],
        y=0.98,
    )

    out_path = "backtests/momentum_dynamic_backtest.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PALETTE["bg"])
    print(f"📊  Plot saved → {out_path}")

    try:
        plt.show()
    except Exception:
        pass


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  B3 NATIVE DYNAMIC MOMENTUM BACKTEST (15% CGT)")
    print("=" * 70)

    # 1. Load native B3 adjusted data
    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

    # 2. Download benchmark
    ibov_px = download_index(START_DATE, END_DATE)
    ibov_monthly = ibov_px.resample("ME").last().pct_change().dropna()
    ibov_monthly.name = "IBOV"

    # 3. Run Strategy
    print("\n🚀 Running dynamic backtest with tax engine...")
    result = run_backtest(adj_close, close_px, fin_vol)

    pretax_val = result["pretax_values"]
    aftertax_val = result["aftertax_values"]
    tax_paid = result["tax_paid"]
    loss_cf = result["loss_carryforward"]
    turnover = result["turnover"]

    common = pretax_val.index.intersection(ibov_monthly.index)
    pretax_val = pretax_val.loc[common]
    aftertax_val = aftertax_val.loc[common]
    tax_paid = tax_paid.loc[common]
    loss_cf = loss_cf.loc[common]
    turnover = turnover.loc[common]
    ibov_ret = ibov_monthly.loc[common]

    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    total_tax = tax_paid.sum()

    print(f"\n   Period          : {common[0].date()} → {common[-1].date()}")
    print(f"   Total months    : {len(common)}")
    print(f"   Total tax paid  : R$ {total_tax:,.2f}")
    print(f"   Final loss C/F  : R$ {loss_cf.iloc[-1]:,.2f}")
    print(f"   Avg turnover/mo : {turnover.mean() * 100:.1f}%")

    m_pretax = build_metrics(pretax_ret, "Momentum Pre-Tax")
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT")
    m_ibov = build_metrics(ibov_ret, "IBOV")

    print("\n" + "-" * 65)
    print(f"  {'Metric':<26}  {'Pre-Tax':>12}  {'After-Tax':>12}  {'IBOV':>8}")
    print("-" * 65)
    for key in m_pretax:
        if key == "Strategy":
            continue
        print(
            f"  {key:<26}  {str(m_pretax[key]):>12}  "
            f"{str(m_aftertax[key]):>12}  {str(m_ibov[key]):>8}"
        )
    print("-" * 65 + "\n")

    plot_results(
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=tax_paid,
        loss_cf=loss_cf,
        turnover=turnover,
        metrics=[m_pretax, m_aftertax, m_ibov],
        total_tax_brl=total_tax,
    )


if __name__ == "__main__":
    main()
