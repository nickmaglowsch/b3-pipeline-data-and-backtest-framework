"""
Strategy: BESST Quality (MarketVector Brazil BESST Quality replica)
===================================================================
Replicates the methodology of the MarketVector(TM) Brazil BESST Quality (BRL)
Index — the reference index of the INVESTO BRAZIL BESST QUALITY FUNDO DE INDICE
(ticker BEST11). See the regulamento / index methodology.

BESST = the five sectors the index is built from:
  Bancos, Energia elétrica, Seguros, serviços de Telecomunicações, Saneamento.

Rules implemented (quarterly index reconstruction):
  1. Universe = companies in the five BESST sectors (BESST_SECTORS map below).
  2. Quality: positive point-in-time TTM net income + positive equity (ROE > 0)
     — the `require_earnings` gate in sp500_b3.select_constituents.
  3. Dividend distribution: paid a dividend/JCP in at least `min_div_years` of the
     last 3 trailing years ("regular dividend payer").
  4. Liquidity: 63d median daily financial volume >= min_adtv.
  5. Weight by market cap, capped at `max_weight` (default 8%) per company.
  6. Rebalance quarterly; require >= `min_names` (default 15) names to start.

Reuses the SP-B3 index machinery (backtests/strategies/sp500_b3.py) for PIT
fundamentals, split-adjusted per-class share counts and market cap — same as
top_mcap.py. Requires b3_pipeline.cvm_main to have populated fundamentals_pit.

ponytail: two data gaps vs the official methodology, both documented here:
  - No sector column in the DB → BESST membership is a curated static map
    (BESST_SECTORS). Upgrade path: a real setor/segment classification table.
  - No free-float column → we weight by FULL market cap, not free-float-adjusted.
    Overweights closely-held names (e.g. state-controlled utilities) vs the index.
  The market-cap >150M USD and free-float >=10% eligibility screens are likewise
  approximated by the min_adtv liquidity floor (no USD FX / float data on hand).
"""
from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd

from backtests.core.strategy_base import (
    StrategyBase,
    ParameterSpec,
    COMMON_START_DATE,
    COMMON_END_DATE,
    COMMON_REBALANCE_FREQ,
)
from backtests.strategies.sp500_b3 import (
    DEFAULT_MIN_ADTV,
    LIQUIDITY_WINDOW,
    compute_weights,
    infer_volume_scale,
    load_fundamentals_pit_raw,
    load_stock_actions,
    select_constituents,
)

# ── The BESST universe: 4-char ticker root -> sector ──────────────────────────
# Curated because the DB has no sector classification. Grounded on the roots that
# actually trade on B3 with CVM fundamentals (verified against prices +
# fundamentals_pit). The PIT quality/liquidity gate drops any root that lacks
# fundamentals in a given quarter, so listing a name that later delists is safe.
BESST_SECTORS: dict[str, str] = {
    # Bancos
    "ITUB": "banks", "BBDC": "banks", "BBAS": "banks", "SANB": "banks",
    "BPAC": "banks", "ABCB": "banks", "BRSR": "banks", "BMGB": "banks",
    "PINE": "banks", "BPAN": "banks", "BEES": "banks", "BAZA": "banks",
    "BNBR": "banks", "BMEB": "banks", "BMIN": "banks", "BGIP": "banks",
    "BRIV": "banks", "BSLI": "banks",
    # Energia elétrica
    "ELET": "power", "CMIG": "power", "CPLE": "power", "CPFE": "power",
    "ENGI": "power", "EQTL": "power", "TAEE": "power", "TRPL": "power",
    "EGIE": "power", "NEOE": "power", "ENEV": "power", "AURE": "power",
    "CLSC": "power", "COCE": "power", "LIGT": "power", "ALUP": "power",
    "GEPA": "power", "EKTR": "power", "REDE": "power", "EMAE": "power",
    "CEBR": "power", "CEEB": "power", "CEED": "power", "CEPE": "power",
    "AFLT": "power", "ENMT": "power", "GPAR": "power", "RNEW": "power",
    "OMGE": "power", "CSRN": "power", "ELPL": "power", "SRNA": "power",
    # Seguros
    "BBSE": "insurance", "PSSA": "insurance", "SULA": "insurance",
    "CXSE": "insurance", "IRBR": "insurance", "WIZC": "insurance",
    "APER": "insurance", "CSAB": "insurance",
    # Telecomunicações
    "VIVT": "telecom", "TIMS": "telecom", "OIBR": "telecom", "TELB": "telecom",
    "FIQE": "telecom", "DESK": "telecom", "BRST": "telecom", "TIMP": "telecom",
    # Saneamento
    "SBSP": "sanitation", "SAPR": "sanitation", "CSMG": "sanitation",
    "CASN": "sanitation",
}
BESST_ROOTS = frozenset(BESST_SECTORS)


def load_dividends(db_path: str) -> dict[str, np.ndarray]:
    """Cash-dividend / JCP ex-dates per 4-char ticker root, sorted ascending.

    corporate_actions is keyed by isin_code; map to tickers via the distinct
    (isin_code, ticker) pairs in prices (same join as load_stock_actions), then
    fold to the 4-char root the selection code uses.
    """
    sql = """
        SELECT DISTINCT p.ticker, ca.event_date
        FROM corporate_actions ca
        JOIN (SELECT DISTINCT isin_code, ticker FROM prices
              WHERE isin_code != 'UNKNOWN') p
          ON p.isin_code = ca.isin_code
        WHERE ca.event_type IN ('CASH_DIVIDEND', 'JCP')
    """
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn)
    df["root"] = df["ticker"].str[:4]
    df["event_date"] = pd.to_datetime(df["event_date"])
    out: dict[str, np.ndarray] = {}
    for root, grp in df.groupby("root"):
        out[root] = np.sort(grp["event_date"].values)
    return out


def paid_regularly(dates: np.ndarray | None, t: pd.Timestamp, min_years: int) -> bool:
    """True if dividends fell in >= min_years of the 3 trailing 12-month windows.

    Faithful to the methodology's "regular dividend payer" screen
    ("pelo menos dois dos últimos três anos").
    """
    if dates is None or len(dates) == 0:
        return False
    edges = [np.datetime64(t - pd.DateOffset(years=k)) for k in range(4)]  # t, t-1y..t-3y
    hits = sum(
        bool(((dates > edges[k + 1]) & (dates <= edges[k])).any())
        for k in range(3)
    )
    return hits >= min_years


def cap_weights(w: pd.Series, cap: float) -> pd.Series:
    """Iterative water-filling cap: no weight exceeds `cap`, weights sum to 1.

    Redistributes the spilled excess pro-rata to the uncapped names, repeating
    until none exceed the cap (feasible when len(w) >= 1/cap; min_names ensures it).
    """
    w = (w / w.sum()).astype(float)
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = float((w[over] - cap).sum())
        w[over] = cap
        under = ~over
        pool = float(w[under].sum())
        if pool <= 0:
            break
        w[under] += excess * w[under] / pool
    return w


class BesstQualityStrategy(StrategyBase):
    """Quarterly BESST-sector, quality + dividend screened, cap-weighted (8% cap)."""

    @property
    def name(self) -> str:
        return "BESST Quality"

    @property
    def description(self) -> str:
        return (
            "Replica of the MarketVector Brazil BESST Quality index (BEST11 ETF): "
            "companies in banking, electric power, insurance, telecom and "
            "sanitation with positive PIT earnings and a regular dividend history, "
            "market-cap weighted with an 8% per-name cap, rebalanced quarterly. "
            "Requires fundamentals_pit. Weights use full (not free-float) market cap."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        rebalance = ParameterSpec(
            "rebalance_freq", "Rebalance Frequency", "choice", "QE",
            description="Index rebalances quarterly (QE)",
            choices=["ME", "QE", "W-FRI"],
        )
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            rebalance,
            ParameterSpec(
                "max_weight", "Max Weight per Name", "float", 0.08,
                description="Per-company weight cap (index caps at 8%)",
                min_value=0.01, max_value=1.0, step=0.01,
            ),
            ParameterSpec(
                "min_names", "Min Constituents", "int", 15,
                description="Index holds a minimum of 15 companies",
                min_value=1, max_value=60, step=1,
            ),
            ParameterSpec(
                "min_div_years", "Min Dividend Years (of last 3)", "int", 2,
                description="Require dividends in >= this many of the last 3 years",
                min_value=0, max_value=3, step=1,
            ),
            ParameterSpec(
                "min_adtv", "Min Median Volume (BRL)", "float", DEFAULT_MIN_ADTV,
                description="Minimum 63-trading-day median daily financial volume",
                min_value=0.0, step=1e5,
            ),
            ParameterSpec(
                "db_path", "Database Path", "str", "b3_market_data.sqlite",
                description="Path to the B3 SQLite database (fundamentals_pit / prices)",
            ),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        close_px = shared_data["close_px"]   # daily raw close (ffilled)
        fin_vol = shared_data["fin_vol"]     # daily financial volume

        max_weight = float(params.get("max_weight", 0.08))
        min_names = int(params.get("min_names", 15))
        min_div_years = int(params.get("min_div_years", 2))
        min_adtv = float(params.get("min_adtv", DEFAULT_MIN_ADTV))
        db_path = str(params.get("db_path", "b3_market_data.sqlite"))

        fund = load_fundamentals_pit_raw(db_path)
        actions = load_stock_actions(db_path)
        divs = load_dividends(db_path)

        scale = infer_volume_scale(fin_vol)
        med63 = fin_vol.fillna(0.0).rolling(LIQUIDITY_WINDOW).median() * scale

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        daily_idx = close_px.index
        started = False  # wait for the first full basket (early years are sparse)

        for t_cal in ret.index:
            td = daily_idx.asof(t_cal)  # last trading day <= calendar rebalance date
            if pd.isna(td):
                continue
            sel = select_constituents(
                td, fund, close_px.loc[td], med63.loc[td], actions,
                min_market_cap=0.0, min_adtv=min_adtv, require_earnings=True,
            )
            # BESST sector membership
            sel = sel[sel["root"].isin(BESST_ROOTS)]
            # regular dividend payer
            if not sel.empty:
                keep = [paid_regularly(divs.get(r), pd.Timestamp(td), min_div_years)
                        for r in sel["root"]]
                sel = sel[keep]

            if not started:
                if len(sel) < min_names:
                    continue
                started = True
            if sel.empty:
                continue

            w = compute_weights(sel.set_index("ticker")["market_cap"], "market_cap")
            w = cap_weights(w, max_weight)
            cols = w.index.intersection(tw.columns)
            if len(cols) == 0:
                continue
            w = w[cols] / w[cols].sum()
            tw.loc[t_cal, cols] = w.values

        return ret, tw
