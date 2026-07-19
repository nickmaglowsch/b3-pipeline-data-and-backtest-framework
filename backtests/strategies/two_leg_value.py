"""
Strategy: Two-Leg Integrated Value
==================================
A cheapness screen that ranks banks and operating companies on the same scale,
then filters out the value traps. Six steps, evaluated point-in-time at each
rebalance:

1. Liquidity floor. Keep only names whose ~2-month average daily financial
   volume clears `min_adtv` (default R$200k/day). Illiquid names have
   unreliable prices and can't be traded into.
2. Balance-sheet freshness. Drop any name whose last filed balance (period_end)
   is older than `max_stale_days` (180) versus the rebalance date. A stale name
   doesn't just misprice itself — every percentile below ranks against it, so it
   corrupts the whole cross-section.
3. Health. P/L > 0 (no loss-makers), and non-financials also need a positive
   EBITDA margin.
4. Two-leg ranking. EV/EBIT is undefined for banks (debt is their raw material),
   so a naive EV screen excludes every financial. Instead:
     - Non-financials rank by EBITDA/EV (earnings yield, capital-structure-neutral).
     - Banks / EBITDA-less names rank by lowest P/L + P/VP (the classic pair).
   Each leg is normalized to a cheapness percentile in [0,1] and merged into one
   integrated ranking, so genuinely cheap banks surface automatically.
5. Value-trap filter. Take the top `pool_size` (30), drop cheap-for-a-reason
   names — ROE < `min_roe` (8%) or P/L > `max_pe` (20) — then backfill from the
   rest of the ranking until `target_n` (20) clean names remain.
6. Cadence. Rebalance on the first trading day on/after the 15th of Apr, Jun,
   Sep and Dec — deliberately weeks after quarter-end so filings reflect current
   numbers. Weights are held (forward-filled) between rebalances.

Substitutions vs. the canonical spec, forced by available data (approved):
  - EBIT is not in the pipeline → EBITDA/EV and EBITDA-margin are used instead.
  - No sector flag → "financial" is proxied by a missing/undefined EBITDA.
  - No dividend data → the DY > 18% value-trap leg is dropped.

Financials in fundamentals_pit are stored in THOUSANDS of BRL; market cap
(shares x price) is in BRL, so multiples scale earnings/equity by _FIN_SCALE.

Selection logic is pure functions over DataFrames (unit-testable without a DB —
see tests/test_two_leg_value.py). Requires b3_pipeline.cvm_main to have
populated fundamentals_pit. Reuses the SP-B3 liquidity / market-cap machinery.
"""
from __future__ import annotations

import sqlite3

import pandas as pd

from backtests.core.strategy_base import (
    StrategyBase,
    ParameterSpec,
    COMMON_START_DATE,
    COMMON_END_DATE,
)
from backtests.strategies.sp500_b3 import (
    DEFAULT_STALENESS_DAYS,
    infer_volume_scale,
    load_stock_actions,
    select_constituents,
)

_FIN_SCALE = 1000.0            # fundamentals_pit financials are in thousands BRL
DEFAULT_MIN_ADTV = 2e5         # R$200k/day
LIQUIDITY_WINDOW = 42          # ~2 trading months
DEFAULT_MAX_STALE_DAYS = 180
DEFAULT_MIN_ROE = 0.08
DEFAULT_MAX_PE = 20.0
DEFAULT_POOL_SIZE = 30
DEFAULT_TARGET_N = 20
DEFAULT_MIN_NAMES = 10         # don't start until a real basket exists
REBALANCE_MONTHS = (4, 6, 9, 12)
REBALANCE_DAY = 15


# ── Pure selection functions (no DB access — unit-testable) ───────────────────

def value_rebalance_dates(
    trading_days: pd.DatetimeIndex,
    months=REBALANCE_MONTHS,
    day: int = REBALANCE_DAY,
) -> pd.DatetimeIndex:
    """First trading day on/after `day` of each `months` month present in the
    data (the rebalance calendar)."""
    td = pd.DatetimeIndex(sorted(set(pd.DatetimeIndex(trading_days))))
    if len(td) == 0:
        return pd.DatetimeIndex([])
    out = []
    for y in range(td[0].year, td[-1].year + 1):
        for m in months:
            target = pd.Timestamp(y, m, day)
            cand = td[(td >= target) & (td.year == y) & (td.month == m)]
            if len(cand):
                out.append(cand[0])
    return pd.DatetimeIndex(sorted(out))


def pit_metrics_snapshot(
    fund: pd.DataFrame,
    t: pd.Timestamp,
    staleness_days: int = DEFAULT_STALENESS_DAYS,
) -> pd.DataFrame:
    """Per-metric point-in-time snapshot of ebitda / net_debt / revenue at t,
    plus the balance date (period_end) of the earnings vintage.

    Same per-metric staleness semantics as sp500_b3.pit_snapshot: filings sparse
    by doc_type, so take the last non-null value per metric among filings with
    filing_date <= t (and >= t - staleness_days). Indexed by (4-char root) ticker.
    """
    t = pd.Timestamp(t)
    cutoff = t - pd.Timedelta(days=staleness_days)
    visible = fund[fund["filing_date"] <= t]
    if visible.empty:
        return pd.DataFrame(columns=["ebitda", "net_debt", "revenue", "period_end"])
    visible = visible.sort_values(["filing_date", "filing_version"])

    parts = {}
    for m in ("ebitda", "net_debt", "revenue"):
        sub = visible.dropna(subset=[m]).groupby("ticker", sort=False).tail(1)
        sub = sub[sub["filing_date"] >= cutoff]
        parts[m] = sub.set_index("ticker")[m]
    # period_end / freshness comes from the earnings (net_income_ttm) vintage
    sub = visible.dropna(subset=["net_income_ttm"]).groupby("ticker", sort=False).tail(1)
    sub = sub[sub["filing_date"] >= cutoff]
    parts["period_end"] = sub.set_index("ticker")["period_end"]
    return pd.DataFrame(parts)


def select_value_portfolio(
    df: pd.DataFrame,
    t: pd.Timestamp,
    max_stale_days: int = DEFAULT_MAX_STALE_DAYS,
    min_roe: float = DEFAULT_MIN_ROE,
    max_pe: float = DEFAULT_MAX_PE,
    pool_size: int = DEFAULT_POOL_SIZE,
    target_n: int = DEFAULT_TARGET_N,
) -> pd.DataFrame:
    """Steps 2–5 of the strategy on a per-company frame.

    df columns: ticker, market_cap (BRL), net_income_ttm, equity, roe, ebitda,
    net_debt, revenue (thousands BRL), period_end (datetime). Returns the chosen
    rows (<= target_n), ordered cheapest-first, with the derived columns added.
    """
    d = df.copy()

    # ── (2) balance-sheet freshness ───────────────────────────────────────────
    age = (pd.Timestamp(t) - pd.to_datetime(d["period_end"])).dt.days
    d = d[age.notna() & (age <= max_stale_days)]
    if d.empty:
        return d

    # ── derived multiples (earnings/equity in thousands -> scale to BRL) ──────
    # Financials have no EBIT line (NULL ebitda). Zero-revenue filers with a
    # real EBIT (equity-method insurer holdings like BBSE) also rank on
    # P/L+P/VP — their EBITDA/EV and margin are meaningless, and the non-fin
    # health gate (revenue > 0) would otherwise silently drop them.
    d["is_fin"] = d["ebitda"].isna() | ~(d["revenue"] > 0)
    d["pl"] = d["market_cap"] / (d["net_income_ttm"] * _FIN_SCALE)
    d["pvp"] = d["market_cap"] / (d["equity"] * _FIN_SCALE)
    d["ev"] = d["market_cap"] + d["net_debt"].fillna(0.0) * _FIN_SCALE
    d["ebit_ev"] = (d["ebitda"] * _FIN_SCALE) / d["ev"]
    d["ebitda_margin"] = d["ebitda"] / d["revenue"]

    # ── (3) health: P/L > 0; non-financials need positive EBITDA margin ───────
    d = d[d["pl"] > 0]
    nonfin_ok = d["is_fin"] | ((d["revenue"] > 0) & (d["ebitda_margin"] > 0))
    d = d[nonfin_ok]
    if d.empty:
        return d

    # ── (4) two-leg cheapness percentile, merged ──────────────────────────────
    non = d[~d["is_fin"]].copy()
    fin = d[d["is_fin"]].copy()
    if len(non):
        non["cheap"] = non["ebit_ev"].rank(pct=True)             # higher yield = cheaper
    if len(fin):
        fin["cheap"] = (-(fin["pl"] + fin["pvp"])).rank(pct=True)  # lower sum = cheaper
    ranked = pd.concat([non, fin]).sort_values(
        ["cheap", "ticker"], ascending=[False, True]
    )

    # ── (5) value-trap filter over the top `pool_size`, backfill to target_n ──
    def _clean(x: pd.DataFrame) -> pd.DataFrame:
        return x[(x["roe"] >= min_roe) & (x["pl"] <= max_pe)]

    pool = ranked.head(pool_size)
    chosen = _clean(pool)
    if len(chosen) < target_n:
        chosen = pd.concat([chosen, _clean(ranked.iloc[pool_size:])])
    return chosen.head(target_n)


# ── DB loader (thin) ──────────────────────────────────────────────────────────

def load_fundamentals_full(db_path: str) -> pd.DataFrame:
    """fundamentals_pit rows with the columns both select_constituents and
    pit_metrics_snapshot need (dates parsed)."""
    sql = """
        SELECT ticker, period_end, filing_date, filing_version,
               net_income_ttm, equity, shares_outstanding, shares_on, shares_pn,
               ebitda, net_debt, revenue
        FROM fundamentals_pit
        WHERE ticker IS NOT NULL
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


# ── Strategy plugin ────────────────────────────────────────────────────────────

class TwoLegValueStrategy(StrategyBase):
    """Integrated two-leg value screen (banks + operating companies)."""

    @property
    def name(self) -> str:
        return "Two-Leg Value"

    @property
    def description(self) -> str:
        return (
            "Integrated cheapness screen: liquidity floor, 180-day balance "
            "freshness, health filter, then a two-leg ranking (non-financials by "
            "EBITDA/EV, banks by P/L+P/VP) merged on cheapness percentiles, with "
            "a value-trap filter (ROE, P/L) down to 20 equal-weighted names. "
            "Quarterly rebalance (mid Apr/Jun/Sep/Dec). Requires fundamentals_pit."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            ParameterSpec(
                "min_adtv", "Min Avg Volume (BRL)", "float", DEFAULT_MIN_ADTV,
                description="Minimum ~2-month average daily financial volume",
                min_value=0.0, step=5e4,
            ),
            ParameterSpec(
                "max_stale_days", "Max Balance Age (days)", "int", DEFAULT_MAX_STALE_DAYS,
                description="Drop names whose last filed balance is older than this",
                min_value=30, max_value=540, step=30,
            ),
            ParameterSpec(
                "min_roe", "Min ROE (value trap)", "float", DEFAULT_MIN_ROE,
                min_value=0.0, max_value=0.30, step=0.01,
            ),
            ParameterSpec(
                "max_pe", "Max P/L (value trap)", "float", DEFAULT_MAX_PE,
                min_value=5.0, max_value=50.0, step=1.0,
            ),
            ParameterSpec(
                "target_n", "Number of Stocks", "int", DEFAULT_TARGET_N,
                min_value=5, max_value=40, step=1,
            ),
            ParameterSpec(
                "pool_size", "Ranking Pool Size", "int", DEFAULT_POOL_SIZE,
                description="Top-N of the integrated ranking screened for value traps",
                min_value=10, max_value=80, step=5,
            ),
            ParameterSpec(
                "db_path", "Database Path", "str", "b3_market_data.sqlite",
                description="Path to the B3 SQLite database (fundamentals_pit / stock_actions)",
            ),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        close_px = shared_data["close_px"]   # daily raw close (ffilled)
        fin_vol = shared_data["fin_vol"]     # daily financial volume

        min_adtv = float(params.get("min_adtv", DEFAULT_MIN_ADTV))
        max_stale = int(params.get("max_stale_days", DEFAULT_MAX_STALE_DAYS))
        min_roe = float(params.get("min_roe", DEFAULT_MIN_ROE))
        max_pe = float(params.get("max_pe", DEFAULT_MAX_PE))
        target_n = int(params.get("target_n", DEFAULT_TARGET_N))
        pool_size = int(params.get("pool_size", DEFAULT_POOL_SIZE))
        min_names = int(params.get("min_names", DEFAULT_MIN_NAMES))
        db_path = str(params.get("db_path", "b3_market_data.sqlite"))

        fund = load_fundamentals_full(db_path)
        actions = load_stock_actions(db_path)

        scale = infer_volume_scale(fin_vol)
        avg_vol = fin_vol.fillna(0.0).rolling(LIQUIDITY_WINDOW).mean() * scale

        holdings: dict[pd.Timestamp, pd.Series] = {}
        daily_idx = close_px.index
        for td in value_rebalance_dates(daily_idx):
            # (1) liquidity + market cap via the SP-B3 machinery (no earnings gate)
            sel = select_constituents(
                td, fund, close_px.loc[td], avg_vol.loc[td], actions,
                min_market_cap=0.0, min_adtv=min_adtv, require_earnings=False,
            )
            if sel.empty:
                continue
            snap = pit_metrics_snapshot(fund, td)
            for col in ("ebitda", "net_debt", "revenue", "period_end"):
                sel[col] = (
                    snap[col].reindex(sel["root"]).values
                    if col in snap.columns else pd.NaT if col == "period_end" else float("nan")
                )
            chosen = select_value_portfolio(
                sel, td, max_stale_days=max_stale, min_roe=min_roe,
                max_pe=max_pe, pool_size=pool_size, target_n=target_n,
            )
            if len(chosen) < min_names:
                continue
            w = 1.0 / len(chosen)
            holdings[td] = pd.Series(w, index=chosen["ticker"].values)

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        if not holdings:
            return ret, tw
        rdates = pd.DatetimeIndex(sorted(holdings))
        for t_cal in ret.index:
            pos = rdates.asof(t_cal)  # most recent rebalance <= t_cal (holdings held)
            if pd.isna(pos):
                continue
            w = holdings[pos]
            cols = w.index.intersection(tw.columns)
            if len(cols) == 0:
                continue
            tw.loc[t_cal, cols] = (w[cols] / w[cols].sum()).values

        return ret, tw
