"""
Strategy: SP500-style B3 Index ("Índice SP-B3")
================================================
S&P 500-methodology index on B3, point-in-time and survivorship-bias-free.

Selection filters, evaluated ONLY with information available at each quarterly
rebalance date t (last trading day of Mar/Jun/Sep/Dec):

1. Market cap > R$ 2 billion (nominal; optionally IPCA-deflated to t).
2. Liquidity: 63-trading-day median daily financial volume > R$ 2 million.
3. Positive trailing-12-month net income (net_income_ttm from fundamentals_pit).
4. ROE > 0: net_income_ttm / equity > 0 AND equity > 0 (negative/negative
   must NOT pass).

Point-in-time rule: at rebalance date t, use the fundamentals_pit row with the
max (filing_date, filing_version) among rows with filing_date <= t, ignoring
rows whose filing_date < t - 400 days (staleness).

Company identity: price tickers grouped by 4-char root; per company the MOST
LIQUID share class (63d median financial volume) at each rebalance is the
traded instrument.

Delisting: a company delisted mid-quarter is held until its last traded price,
then its weight is dropped and the proceeds are redistributed pro-rata across
the surviving constituents (renormalization). It naturally drops out at the
next rebalance because its 63d median volume falls below the threshold.

Requires the CVM fundamentals pipeline:
    python -m b3_pipeline.main
    python -m b3_pipeline.cvm_main

All selection logic is implemented as pure functions over DataFrames so it is
testable without the database (see tests/test_sp500_b3.py).
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

DEFAULT_MIN_MARKET_CAP = 2e9   # R$ 2 billion
DEFAULT_MIN_ADTV = 2e6         # R$ 2 million (63d median daily financial volume)
DEFAULT_STALENESS_DAYS = 400
DEFAULT_MIN_CONSTITUENTS = 20
LIQUIDITY_WINDOW = 63          # trading days
# Data-sanity guard: minimum daily turnover (med_vol / market_cap). CVM filings
# contain corrupted share counts (TELB/TOYB filed ~1e12 shares, COCE 3e11 —
# fake R$0.6-12 TRILLION companies). A real large cap turns over far more than
# 0.001%/day of its cap (PETR ~0.1-0.5%); the fakes sit at ~1e-7. Only bites on
# giants: for a R$2bn cap the implied floor (R$20k/day) is below any min_adtv.
MIN_TURNOVER_RATIO = 1e-5


# ── Pure selection functions (no DB access — unit-testable) ───────────────────

def quarter_end_rebalance_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Last trading day of each completed Mar/Jun/Sep/Dec quarter.

    A quarter only produces a rebalance date if its last trading day falls in
    the quarter's final month (i.e. the quarter is complete in the data).
    """
    if len(trading_days) == 0:
        return pd.DatetimeIndex([])
    s = pd.Series(trading_days, index=trading_days)
    last_per_q = s.groupby(trading_days.to_period("Q")).max()
    qe = pd.DatetimeIndex(last_per_q.values)
    return qe[qe.month.isin([3, 6, 9, 12])]


def pit_snapshot(
    fund: pd.DataFrame,
    t: pd.Timestamp,
    staleness_days: int = DEFAULT_STALENESS_DAYS,
) -> pd.DataFrame:
    """Point-in-time fundamentals snapshot at date t.

    Args:
        fund: fundamentals_pit-shaped frame with at least columns
              [ticker, period_end, filing_date, filing_version] plus metrics.
              filing_date / period_end must be datetime64.
        t:    rebalance date.

    Returns:
        One row per ticker with the last non-null value PER METRIC across all
        filings with filing_date <= t (per-metric staleness: a metric whose
        source filing is older than t - staleness_days is dropped). This
        matters because filings are sparse by doc_type: an FRE row carries only
        shares_outstanding (earnings NULL), so "latest row wins" would blank
        out earnings for every company right after the May/June FRE wave.
        `period_end` in the result is the period_end of the row that supplied
        shares_outstanding (the anchor for split adjustment of share counts).
    """
    t = pd.Timestamp(t)
    cutoff = t - pd.Timedelta(days=staleness_days)
    visible = fund[fund["filing_date"] <= t]
    if visible.empty:
        return visible.head(0)
    visible = visible.sort_values(["filing_date", "filing_version"])

    parts = {}
    for metric in ["net_income_ttm", "equity", "shares_outstanding"]:
        sub = visible.dropna(subset=[metric]).groupby("ticker", sort=False).tail(1)
        sub = sub[sub["filing_date"] >= cutoff]
        parts[metric] = sub.set_index("ticker")[metric]
        if metric == "shares_outstanding":
            parts["period_end"] = sub.set_index("ticker")["period_end"]
            parts["filing_date"] = sub.set_index("ticker")["filing_date"]
            # per-class counts ride along from the SAME row as the total so
            # counts and split-adjustment anchor stay in one vintage
            for extra in ("shares_on", "shares_pn"):
                if extra in sub.columns:
                    parts[extra] = sub.set_index("ticker")[extra]

    snap = pd.DataFrame(parts).reset_index().rename(columns={"index": "ticker"})
    return snap


def share_multiplier(
    actions: pd.DataFrame,
    ticker: str,
    period_end: pd.Timestamp,
    t: pd.Timestamp,
) -> float:
    """Cumulative share-count multiplier from stock_actions ex_dates in
    (period_end, t] for `ticker`.

    Per README semantics:
      STOCK_SPLIT   factor F (>1): 1 old share becomes F shares  -> count x F
      REVERSE_SPLIT factor F (<1): 1 old share becomes F shares  -> count x F
      BONUS_SHARES  factor F (percent): count x (1 + F/100)
    """
    if actions is None or actions.empty:
        return 1.0
    a = actions[
        (actions["ticker"] == ticker)
        & (actions["ex_date"] > pd.Timestamp(period_end))
        & (actions["ex_date"] <= pd.Timestamp(t))
    ]
    mult = 1.0
    for _, row in a.iterrows():
        f = float(row["factor"])
        if row["action_type"] == "BONUS_SHARES":
            mult *= 1.0 + f / 100.0
        else:  # STOCK_SPLIT or REVERSE_SPLIT: factor is directly new/old count
            mult *= f
    return mult


def select_constituents(
    t: pd.Timestamp,
    fund: pd.DataFrame,
    close_row: pd.Series,
    med_vol_row: pd.Series,
    actions: pd.DataFrame,
    min_market_cap: float = DEFAULT_MIN_MARKET_CAP,
    min_adtv: float = DEFAULT_MIN_ADTV,
    staleness_days: int = DEFAULT_STALENESS_DAYS,
    require_earnings: bool = True,
) -> pd.DataFrame:
    """Apply the four SP-B3 selection filters at rebalance date t.

    Args:
        t:            rebalance date (a trading day).
        fund:         fundamentals_pit-shaped frame keyed by 4-char root ticker
                      (columns: ticker, period_end, filing_date, filing_version,
                      net_income_ttm, equity, shares_outstanding).
                      Financials in THOUSANDS of BRL; shares in units.
        close_row:    raw close prices at t, indexed by full price ticker
                      (e.g. PETR3, PETR4).
        med_vol_row:  63d median daily financial volume (BRL) at t, indexed by
                      full price ticker. NaN treated as 0.
        actions:      stock_actions frame (ticker, ex_date, action_type, factor)
                      with ex_date as datetime64.

    Returns:
        DataFrame [root, ticker, market_cap, med_vol, net_income_ttm, equity,
        roe] — one row per passing company, ticker = most liquid share class.
    """
    t = pd.Timestamp(t)
    empty = pd.DataFrame(
        columns=["root", "ticker", "market_cap", "med_vol",
                 "net_income_ttm", "equity", "roe"]
    )

    # ── most liquid share class per 4-char root ───────────────────────────────
    vol = med_vol_row.fillna(0.0)
    cand = pd.DataFrame({"ticker": vol.index.astype(str), "med_vol": vol.values})
    cand["root"] = cand["ticker"].str[:4]
    cand["close"] = close_row.reindex(cand["ticker"]).values
    cand = cand[cand["close"].notna() & (cand["close"] > 0)]
    if cand.empty:
        return empty
    cand = cand.sort_values(["med_vol", "ticker"]).groupby("root").tail(1)

    # ── liquidity filter ──────────────────────────────────────────────────────
    cand = cand[cand["med_vol"] > min_adtv]
    if cand.empty:
        return empty

    # ── PIT fundamentals join (by root) ───────────────────────────────────────
    pit = pit_snapshot(fund, t, staleness_days=staleness_days)
    if pit.empty:
        return empty
    pit = pit.set_index("ticker")
    cand = cand[cand["root"].isin(pit.index)]
    if cand.empty:
        return empty
    for col in ["net_income_ttm", "equity", "shares_outstanding", "period_end",
                "shares_on", "shares_pn"]:
        if col in pit.columns:
            cand[col] = pit[col].reindex(cand["root"]).values
        else:
            cand[col] = float("nan")

    # ── earnings quality filters (skipped when require_earnings=False) ────────
    # 3. positive TTM net income; 4. ROE > 0 with equity explicitly > 0
    #    (negative NI / negative equity must NOT pass).
    if require_earnings:
        cand = cand[
            cand["net_income_ttm"].notna() & (cand["net_income_ttm"] > 0)
            & cand["equity"].notna() & (cand["equity"] > 0)
            & (cand["net_income_ttm"] / cand["equity"] > 0)
        ]
    cand = cand[cand["shares_outstanding"].notna() & (cand["shares_outstanding"] > 0)]
    if cand.empty:
        return empty

    # ── market cap filter ─────────────────────────────────────────────────────
    # Per-class market cap when the FRE per-class counts are available:
    # ON shares x ON close (root+"3") + PN shares x most senior traded PN close
    # (root+"4"/"5"/"6"), a missing class priced at its sibling. This avoids the
    # unit inflation of total shares x unit price (BPAC11 = 3 shares per unit).
    # Splits apply company-wide, so one multiplier (from the selected ticker's
    # actions) adjusts both legs.
    def _class_prices(root: str):
        p_on = close_row.get(root + "3")
        p_on = float(p_on) if pd.notna(p_on) and p_on > 0 else None
        for sfx in ("4", "5", "6"):
            v = close_row.get(root + sfx)
            if pd.notna(v) and v > 0:
                return p_on, float(v)
        return p_on, None

    mcaps = []
    for row in cand.itertuples():
        mult = share_multiplier(actions, row.ticker, row.period_end, t)
        p_on, p_pn = _class_prices(row.root)
        if (pd.notna(row.shares_on) and pd.notna(row.shares_pn)
                and (row.shares_on + row.shares_pn) > 0
                and (p_on is not None or p_pn is not None)):
            po = p_on if p_on is not None else p_pn
            pp = p_pn if p_pn is not None else p_on
            mcaps.append(mult * (row.shares_on * po + row.shares_pn * pp))
        else:
            # ponytail: fallback = total shares x selected class close; if that
            # class is a unit this overstates 2-5x — only hit when no plain
            # share class trades or per-class counts are missing (pre-FRE era).
            mcaps.append(row.shares_outstanding * mult * row.close)
    cand["market_cap"] = mcaps
    # corrupted-share-count guard (see MIN_TURNOVER_RATIO)
    cand = cand[cand["med_vol"] >= cand["market_cap"] * MIN_TURNOVER_RATIO]
    cand = cand[cand["market_cap"] > min_market_cap]
    if cand.empty:
        return empty

    cand["roe"] = cand["net_income_ttm"] / cand["equity"]
    out = cand[["root", "ticker", "market_cap", "med_vol",
                "net_income_ttm", "equity", "roe"]].reset_index(drop=True)
    return out


def compute_weights(market_caps: pd.Series, weighting: str = "market_cap") -> pd.Series:
    """Index weights from a Series of market caps (index = price ticker).

    weighting: "market_cap" (cap-weighted) or "equal" (1/N). Sums to 1.0.
    """
    if len(market_caps) == 0:
        return pd.Series(dtype=float)
    if weighting == "equal":
        return pd.Series(1.0 / len(market_caps), index=market_caps.index)
    if weighting == "market_cap":
        return market_caps / market_caps.sum()
    raise ValueError(f"Unknown weighting {weighting!r}")


def _hold_period(px: pd.DataFrame, weights: pd.Series, last_trade: pd.Series) -> pd.Series:
    """Buy-and-hold value path over one rebalance period, starting at 1.0.

    px:         daily adj_close (total-return, forward-filled) slice covering
                [t0, t1], columns superset of weights.index.
    weights:    weights at t0 (sum to 1).
    last_trade: per-ticker last actual trading date (Series). A ticker whose
                last_trade falls before t1 is DELISTED mid-period: it is held
                until its last traded price, then sold and the proceeds are
                redistributed pro-rata across survivors (documented choice —
                the ffilled price at the sale date equals the last real price).
    """
    idx = px.index
    t0, t1 = idx[0], idx[-1]

    w = weights[weights > 0].copy()
    p0 = px.iloc[0].reindex(w.index)
    valid = p0.notna() & (p0 > 0)
    if not valid.all():
        w = w[valid]
        w = w / w.sum()
    units = w / px.iloc[0][w.index]
    alive = list(w.index)

    # group delist events by (clamped) event date strictly inside the period
    death: dict[pd.Timestamp, list] = {}
    for tk in alive:
        lt = last_trade.get(tk, pd.NaT)
        if pd.isna(lt) or lt >= t1:
            continue
        death.setdefault(max(pd.Timestamp(lt), t0), []).append(tk)

    values = pd.Series(index=idx, dtype=float)
    seg_start = t0
    for event in sorted(death):
        seg = px.loc[seg_start:event, alive]
        values.loc[seg_start:event] = (seg * units[alive]).sum(axis=1)
        v_event = values.loc[event]
        dead = death[event]
        proceeds = float((units[dead] * px.loc[event, dead]).sum())
        survivors = [a for a in alive if a not in dead]
        units = units.drop(dead)
        if not survivors or (v_event - proceeds) <= 0:
            # everything delisted: park as cash (flat) until period end
            values.loc[event:t1] = v_event
            return values / values.iloc[0]
        units[survivors] *= 1.0 + proceeds / (v_event - proceeds)
        alive = survivors
        seg_start = event
    seg = px.loc[seg_start:t1, alive]
    values.loc[seg_start:t1] = (seg * units[alive]).sum(axis=1)
    return values / values.iloc[0]


def build_index_series(
    adj_close: pd.DataFrame,
    holdings: dict,
    last_trade: pd.Series,
    end_date=None,
    slippage: float = 0.0,
) -> pd.Series:
    """Chain buy-and-hold periods into a total-return index level series.

    Args:
        adj_close: daily total-return prices (forward-filled), wide.
        holdings:  {rebalance_date: weight Series (price ticker -> weight)}.
        last_trade: per-ticker last actual trading date.
        end_date:  final date (default: last adj_close date).
        slippage:  optional fractional cost applied at each rebalance
                   (index convention: 0.0 — no costs).

    Returns:
        Series of index levels starting at 1.0 on the first rebalance date.
    """
    dates = sorted(holdings)
    if not dates:
        return pd.Series(dtype=float)
    end = pd.Timestamp(end_date) if end_date is not None else adj_close.index[-1]
    bounds = list(dates[1:]) + [end]
    level = 1.0
    parts = []
    for t0, t1 in zip(dates, bounds):
        if t1 <= t0:
            continue
        px = adj_close.loc[t0:t1]
        path = _hold_period(px, holdings[t0], last_trade)
        level *= (1.0 - slippage)
        seg = path * level
        parts.append(seg if not parts else seg.iloc[1:])
        level = float(seg.iloc[-1])
    return pd.concat(parts)


def infer_volume_scale(fin_vol: pd.DataFrame) -> float:
    """Detect the fin_vol unit convention empirically.

    core/data.py returns fin_vol = prices.volume / 100 (legacy convention from
    the original parser that stored raw VOLTOT centavos). The CURRENT parsers
    (b3_pipeline_rs/src/parser.rs and b3_pipeline/parser.py) already divide
    VOLTOT by 100 when storing, so on a freshly rebuilt DB fin_vol comes back
    understated 100x.

    ponytail: empirical calibration knob — the most liquid B3 stock trades well
    over R$100M/day; if the max recent 63d-median is far below that, the data
    was double-divided and we scale by 100. Upgrade path: fix the convention
    in core/data.py once the pipeline settles.
    """
    if fin_vol is None or fin_vol.empty:
        return 1.0
    top = fin_vol.tail(LIQUIDITY_WINDOW).median().max()
    return 100.0 if pd.notna(top) and top < 5e7 else 1.0


# ── DB loaders (thin — all logic stays in the pure functions above) ──────────

def load_fundamentals_pit_raw(db_path: str) -> pd.DataFrame:
    """Load fundamentals_pit rows needed for selection (dates parsed)."""
    sql = """
        SELECT ticker, period_end, filing_date, filing_version,
               net_income_ttm, equity, shares_outstanding, shares_on, shares_pn
        FROM fundamentals_pit
        WHERE ticker IS NOT NULL
    """
    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query(sql, conn)
        except Exception:
            # DB predates the shares_on/shares_pn migration
            df = pd.read_sql_query(
                sql.replace(", shares_on, shares_pn", ""), conn
            )
            df["shares_on"] = float("nan")
            df["shares_pn"] = float("nan")
    df["period_end"] = pd.to_datetime(df["period_end"])
    df["filing_date"] = pd.to_datetime(df["filing_date"])
    return df


def load_stock_actions(db_path: str) -> pd.DataFrame:
    """Load stock_actions (ticker, ex_date, action_type, factor), dates parsed.

    stock_actions is keyed by isin_code; map to tickers via the distinct
    (isin_code, ticker) pairs observed in prices. An ISIN that traded under
    multiple tickers yields one row per ticker (the caller filters by ticker).
    """
    sql = """
        SELECT DISTINCT p.ticker, s.ex_date, s.action_type, s.factor
        FROM stock_actions s
        JOIN (SELECT DISTINCT isin_code, ticker FROM prices
              WHERE isin_code != 'UNKNOWN') p
          ON p.isin_code = s.isin_code
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    df["ex_date"] = pd.to_datetime(df["ex_date"])
    return df


def download_ipca_index(start: str, end: str) -> pd.Series:
    """Monthly IPCA cumulative index from BCB SGS series 433 (m/m % change).

    Mirrors core.data.download_cdi_daily (batched requests). Returns a
    cumulative index Series (base 1.0 at the first observation), monthly.
    """
    import requests
    from dateutil.relativedelta import relativedelta

    print("⬇  Downloading IPCA (SGS 433) from Brazilian Central Bank...")
    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    series = []
    current_start = start_dt
    while current_start <= end_dt:
        current_end = min(current_start + relativedelta(years=9), end_dt)
        s_str = current_start.strftime("%d/%m/%Y")
        e_str = current_end.strftime("%d/%m/%Y")
        url = (
            "https://api.bcb.gov.br/dados/serie/bcdata.sgs.433/dados"
            f"?formato=json&dataInicial={s_str}&dataFinal={e_str}"
        )
        try:
            data = requests.get(url, timeout=30).json()
            if data:
                df = pd.DataFrame(data)
                df["data"] = pd.to_datetime(df["data"], format="%d/%m/%Y")
                df["valor"] = pd.to_numeric(df["valor"]) / 100.0
                series.append(df.set_index("data")["valor"])
        except Exception as e:
            print(f"Warning: failed to fetch IPCA chunk {s_str}–{e_str}: {e}")
        current_start = current_end + relativedelta(days=1)
    if not series:
        return pd.Series(dtype=float, name="IPCA")
    ipca = pd.concat(series)
    ipca = ipca[~ipca.index.duplicated(keep="first")].sort_index()
    return (1.0 + ipca).cumprod().rename("IPCA")


def deflated_threshold(nominal: float, ipca_index: pd.Series, t: pd.Timestamp) -> float:
    """Deflate a nominal (present-day) BRL threshold back to rebalance date t.

    threshold(t) = nominal * IPCA(t) / IPCA(latest). Falls back to nominal
    when the IPCA index is empty or does not cover t.
    """
    if ipca_index is None or ipca_index.empty:
        return nominal
    at_t = ipca_index.asof(pd.Timestamp(t))
    if pd.isna(at_t):
        return nominal
    return nominal * float(at_t) / float(ipca_index.iloc[-1])


# ── Strategy plugin ───────────────────────────────────────────────────────────

class SP500B3Strategy(StrategyBase):
    """SP500-style B3 Index plugin.

    Quarterly (QE) rebalance. At each rebalance the framework simulator resets
    to the target weights; intra-quarter market-cap drift is only fully modeled
    by the standalone runner (backtests/sp500_b3_index.py), which is the
    reference implementation of the index.
    """

    needs_fundamentals: bool = True

    @property
    def name(self) -> str:
        return "SP500-style B3 Index"

    @property
    def description(self) -> str:
        return (
            "S&P 500-methodology index on B3, point-in-time and survivorship-"
            "bias-free. Quarterly rebalance; filters: market cap > R$2bn, 63d "
            "median financial volume > R$2M, positive TTM net income, ROE > 0 "
            "with equity > 0. Market-cap or equal weighting. Requires "
            "b3_pipeline.cvm_main to have populated fundamentals_pit "
            "(net_income_ttm column)."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            ParameterSpec(
                "weighting", "Weighting Scheme", "choice", "market_cap",
                description="market_cap = cap-weighted; equal = 1/N",
                choices=["market_cap", "equal"],
            ),
            ParameterSpec(
                "min_market_cap", "Min Market Cap (BRL)", "float",
                DEFAULT_MIN_MARKET_CAP,
                description="Minimum market capitalization at rebalance (nominal BRL)",
                min_value=0.0, step=1e8,
            ),
            ParameterSpec(
                "min_adtv", "Min Median Volume (BRL)", "float", DEFAULT_MIN_ADTV,
                description="Minimum 63-trading-day median daily financial volume",
                min_value=0.0, step=1e5,
            ),
            # Rebalance frequency is fixed quarterly by methodology.
            ParameterSpec(
                "rebalance_freq", "Rebalance Frequency", "choice", "QE",
                description="Fixed quarterly (S&P methodology)",
                choices=["QE"],
            ),
            ParameterSpec(
                "ipca_deflate", "IPCA-deflate Cap Threshold", "choice", "No",
                description=(
                    "If Yes, deflate the market-cap threshold to each rebalance "
                    "date using IPCA (BCB SGS 433)"
                ),
                choices=["No", "Yes"],
            ),
            ParameterSpec(
                "db_path", "Database Path", "str", "b3_market_data.sqlite",
                description="Path to the B3 SQLite database (for fundamentals_pit / stock_actions)",
            ),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        close_px = shared_data["close_px"]   # daily raw close (ffilled)
        fin_vol = shared_data["fin_vol"]     # daily financial volume

        weighting = str(params.get("weighting", "market_cap"))
        min_mcap = float(params.get("min_market_cap", DEFAULT_MIN_MARKET_CAP))
        min_adtv = float(params.get("min_adtv", DEFAULT_MIN_ADTV))
        ipca_deflate = str(params.get("ipca_deflate", "No")) == "Yes"
        db_path = str(params.get("db_path", "b3_market_data.sqlite"))
        min_constituents = int(params.get("min_constituents", DEFAULT_MIN_CONSTITUENTS))

        fund = load_fundamentals_pit_raw(db_path)
        actions = load_stock_actions(db_path)

        scale = infer_volume_scale(fin_vol)
        med63 = fin_vol.fillna(0.0).rolling(LIQUIDITY_WINDOW).median() * scale

        ipca_index = pd.Series(dtype=float)
        if ipca_deflate:
            ipca_index = download_ipca_index(
                str(close_px.index[0].date()), str(close_px.index[-1].date())
            )

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        started = False
        daily_idx = close_px.index

        for t_cal in ret.index:
            td = daily_idx.asof(t_cal)  # last trading day <= calendar quarter end
            if pd.isna(td):
                continue
            thr = deflated_threshold(min_mcap, ipca_index, td) if ipca_deflate else min_mcap
            sel = select_constituents(
                td, fund, close_px.loc[td], med63.loc[td], actions,
                min_market_cap=thr, min_adtv=min_adtv,
            )
            if not started:
                if len(sel) < min_constituents:
                    continue
                started = True
            if sel.empty:
                continue
            w = compute_weights(sel.set_index("ticker")["market_cap"], weighting)
            cols = w.index.intersection(tw.columns)
            if len(cols) == 0:
                continue
            w = w[cols] / w[cols].sum()
            # run_simulation buys row-t weights at the close of t and applies the
            # NEXT row's returns to them, so weights selected at quarter end t
            # belong in row t — same trade-at-selection-close timing as the CLI
            # runner (build_index_series).
            tw.loc[t_cal, cols] = w.values

        return ret, tw
