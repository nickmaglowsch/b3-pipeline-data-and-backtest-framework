"""
Strategy: TevaAtivosReais — Teva Dividendos Ativos Reais Listados (replica)

Replicates the Teva Dividendos Ativos Reais index methodology:
  - Universe filtered point-in-time by CVM sector (Setor_Atividade) sourced from
    CVM FCA data — survivorship-free (covers delisted companies), no hardcoded
    ticker list. Default targets Utilities (energy + saneamento); CVM's taxonomy
    has no clean bucket for Shoppings or toll-road concessions (see
    `eligible_sectors` param).
  - Eligibility: market cap > R$500mm and liquidity (ADTV) floor.
  - Dividend Score = mean of 3 consecutive trailing-12m dividend yields over the
    last 36 months, each capped by DividendYieldCap (mean + 1 std of the monthly
    rolling-12m DYs over the prior 5y).
  - Keep the top 3 quartiles by score (drop the lowest quartile).
  - Weight = 50% market-cap proportion + 50% dividend-score proportion, with a
    10% per-issuer cap applied iteratively.
  - Semiannual rebalance (methodology default; driven by rebalance_freq).

Requires CVM fundamentals (needs_fundamentals=True) for shares outstanding and
reads dividends directly from the corporate_actions table.

ponytail: free-float weighting is approximated by full market cap — B3 has no
free-float series here. Add free_float scaling when a source exists.
ponytail: DY uses split-adjusted per-share dividends / split-adjusted price;
faithful enough for stable payers, revisit if a name splits mid-window often.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from backtests.core.data import sector_membership, sector_membership_asof
from backtests.core.strategy_base import (
    StrategyBase,
    ParameterSpec,
    COMMON_START_DATE,
    COMMON_END_DATE,
    COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE,
    COMMON_SLIPPAGE,
    COMMON_MONTHLY_SALES_EXEMPTION,
)

_DB_PATH = Path(__file__).resolve().parents[2] / "b3_market_data.sqlite"

# Eligible CVM Setor_Atividade classes (substring match, so the "Emp. Adm. Part.
# - <sector>" holding variants are included). CVM's taxonomy cleanly identifies
# Utilities (energy + saneamento); it has no dedicated bucket for Shoppings or
# toll-road concessions, so those Teva sectors can't be reproduced from this
# source without a proprietary mapping. Sourced point-in-time from FCA data,
# which covers delisted companies — no survivorship bias.
_DEFAULT_SECTORS = ["Energia Elétrica", "Saneamento"]


# ── Pure helpers (unit-tested) ────────────────────────────────────────────────

def dividend_score(dy_12m: pd.Series, asof: pd.Timestamp) -> float:
    """
    Teva Dividend Score at `asof` from a monthly series of trailing-12m DYs.

    Score = mean of the 3 consecutive 12m DYs ending at asof, asof-12m, asof-24m,
    each capped at DividendYieldCap = mean + 1 std of the monthly rolling-12m DYs
    over the 5 years prior to asof. Returns NaN if the three windows aren't all
    available.
    """
    dy = dy_12m.dropna()
    if dy.empty:
        return float("nan")

    def at(dt: pd.Timestamp) -> float:
        window = dy.loc[:dt]
        return float(window.iloc[-1]) if len(window) else float("nan")

    yr = pd.DateOffset(years=1)
    windows = [at(asof), at(asof - yr), at(asof - 2 * yr)]
    if any(np.isnan(w) for w in windows):
        return float("nan")

    hist = dy.loc[asof - pd.DateOffset(years=5): asof]
    cap = hist.mean() + hist.std() if len(hist) >= 2 else float("inf")
    return float(np.mean([min(w, cap) for w in windows]))


def cap_weights(w: pd.Series, cap: float = 0.10) -> pd.Series:
    """Normalise to 1 and iteratively enforce a per-name cap, redistributing the
    excess proportionally across the uncapped names."""
    w = w[w > 0].astype(float)
    if w.empty:
        return w
    w = w / w.sum()
    for _ in range(100):
        over = w > cap + 1e-12
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        if not under.any() or w[under].sum() == 0:
            break
        w[under] += excess * w[under] / w[under].sum()
    # If the cap is infeasible (n * cap < 1) every name maxes out and the excess
    # can't be placed — renormalise so the book stays fully invested (equal
    # weight in that limit). No-op when already summing to 1.
    if w.sum() > 0:
        w = w / w.sum()
    return w


def blend_weights(mcap: pd.Series, score: pd.Series, cap: float = 0.10) -> pd.Series:
    """50% market-cap proportion + 50% dividend-score proportion, then 10% cap."""
    mcap = mcap[mcap > 0]
    score = score.reindex(mcap.index).clip(lower=0).fillna(0.0)
    idx = mcap.index
    mc = mcap / mcap.sum()
    sc = score / score.sum() if score.sum() > 0 else pd.Series(1.0 / len(idx), index=idx)
    return cap_weights(0.5 * mc + 0.5 * sc, cap)


# ── DB-backed dividend-yield matrix ───────────────────────────────────────────

def build_dy_12m(db_path, tickers: list[str]) -> dict[str, pd.Series]:
    """
    Return {ticker: monthly series of trailing-12m dividend yield}.

    DY uses split-adjusted per-share dividends (value * split_adj_close/close on
    the event date) summed over 12 months, divided by month-end split_adj_close.
    """
    import sqlite3

    with sqlite3.connect(str(db_path)) as conn:
        px = pd.read_sql_query(
            "SELECT ticker, date, close, split_adj_close, isin_code FROM prices "
            f"WHERE ticker IN ({','.join('?' * len(tickers))})",
            conn, params=tickers, parse_dates=["date"],
        )
        divs = pd.read_sql_query(
            "SELECT isin_code, event_date, value FROM corporate_actions "
            "WHERE event_type IN ('CASH_DIVIDEND','JCP')",
            conn, params=[], parse_dates=["event_date"],
        )

    out: dict[str, pd.Series] = {}
    divs = divs.set_index("isin_code")
    for tkr, g in px.groupby("ticker"):
        g = g.sort_values("date").drop_duplicates("date", keep="last").set_index("date")
        g = g[(g["close"] > 0) & (g["split_adj_close"] > 0)]
        if g.empty:
            continue
        isins = g["isin_code"].dropna().unique().tolist()
        d = divs.loc[divs.index.intersection(isins)]
        if isinstance(d, pd.Series):  # single row edge case
            d = d.to_frame().T
        # split factor at each event date = split_adj_close / close (fwd-filled)
        ev = pd.DatetimeIndex(d["event_date"]) if len(d) else pd.DatetimeIndex([])
        split = (g["split_adj_close"] / g["close"])
        split = split.reindex(g.index.union(ev.unique())).sort_index().ffill()
        if len(d):
            adj_div = pd.Series(
                d["value"].values * split.reindex(ev).values,
                index=ev,
            ).sort_index()
        else:
            adj_div = pd.Series(dtype=float)

        price_me = g["split_adj_close"].resample("ME").last()
        if len(adj_div):
            div_me = adj_div.resample("ME").sum().reindex(price_me.index, fill_value=0.0)
        else:
            div_me = pd.Series(0.0, index=price_me.index)
        dy = div_me.rolling(12, min_periods=12).sum() / price_me
        out[tkr] = dy.replace([np.inf, -np.inf], np.nan)
    return out


class TevaAtivosReaisStrategy(StrategyBase):
    """Replica of the Teva Dividendos Ativos Reais Listados index."""

    needs_fundamentals: bool = True

    @property
    def name(self) -> str:
        return "TevaAtivosReais"

    @property
    def description(self) -> str:
        return (
            "Replica of the Teva Dividendos Ativos Reais index: dividend-focused "
            "portfolio of Utilities (energy + saneamento), weighted 50% by market "
            "cap and 50% by a 36-month capped Dividend Score, with a 10% "
            "per-issuer cap. Universe is filtered point-in-time by CVM sector "
            "(FCA data, survivorship-free). Semiannual rebalance."
        )

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE,
            COMMON_END_DATE,
            COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE,
            COMMON_SLIPPAGE,
            COMMON_MONTHLY_SALES_EXEMPTION,
            ParameterSpec(
                "rebalance_freq", "Rebalance Frequency", "choice", "QE",
                description="Rebalance cadence (methodology is semiannual; QE≈closest quarterly proxy)",
                choices=["ME", "QE", "W-FRI"],
            ),
            ParameterSpec(
                "min_market_cap", "Min Market Cap (BRL)", "float", 500_000_000.0,
                description="Minimum market capitalisation (methodology: R$500mm)",
                min_value=0.0, step=100_000_000.0,
            ),
            ParameterSpec(
                "min_adtv", "Min ADTV (BRL)", "float", 9_500_000.0,
                description="Liquidity floor: ~R$200mm/month secondary volume ÷ 21 trading days",
                min_value=0.0, step=1_000_000.0,
            ),
            ParameterSpec(
                "issuer_cap", "Per-Issuer Cap", "float", 0.10,
                description="Maximum portfolio weight per name",
                min_value=0.01, max_value=1.0, step=0.01,
            ),
            ParameterSpec(
                "eligible_sectors", "Eligible CVM Sectors", "str", ",".join(_DEFAULT_SECTORS),
                description=(
                    "Comma-separated CVM Setor_Atividade keywords (substring match, "
                    "includes 'Emp. Adm. Part.' holdings). Default targets Utilities. "
                    "Sourced point-in-time from CVM FCA data (survivorship-free). "
                    "CVM's taxonomy has no clean Shoppings / toll-road bucket."
                ),
            ),
        ]

    def generate_signals(
        self, shared_data: dict, params: dict
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        ret = shared_data["ret"]
        adtv = shared_data.get("adtv", pd.DataFrame())
        raw_close = shared_data.get("raw_close", pd.DataFrame())
        shares = shared_data.get("f_shares_outstanding_m", pd.DataFrame())

        min_mcap = float(params.get("min_market_cap", 500_000_000.0))
        min_adtv = float(params.get("min_adtv", 9_500_000.0))
        cap = float(params.get("issuer_cap", 0.10))
        keywords = [
            k.strip()
            for k in str(params.get("eligible_sectors", ",".join(_DEFAULT_SECTORS))).split(",")
            if k.strip()
        ]

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        if not keywords or raw_close.empty or shares.empty:
            return ret, tw

        # ── Point-in-time sector universe (survivorship-free) ─────────────────
        sector_pit = shared_data.get("sector_pit", pd.DataFrame())
        member = sector_membership(sector_pit, keywords)
        member = member[member["ticker"].isin(ret.columns)]
        if member.empty:
            return ret, tw
        # full universe ever eligible (for the DY matrix); per-date eligibility
        # uses the classification AS-OF the rebalance date — the latest filing
        # on/before it — so reclassification out of the sector revokes it
        universe = list(member["ticker"].unique())
        elig_asof = sector_membership_asof(sector_pit, keywords, ret.index)
        elig_asof = elig_asof.reindex(columns=universe, fill_value=False)

        dy_by_ticker = build_dy_12m(_DB_PATH, universe)

        for i in range(1, len(ret)):
            prev = ret.index[i - 1]

            # ── Eligible names at `prev` (latest sector filing on/before prev) ─
            elig_row = elig_asof.loc[prev]
            eligible = [t for t in universe if bool(elig_row[t])]
            if not eligible:
                continue

            # ── Dividend Score per eligible name ──────────────────────────────
            scores = {}
            for t in eligible:
                dy = dy_by_ticker.get(t)
                if dy is not None:
                    s = dividend_score(dy, prev)
                    if not np.isnan(s) and s > 0:
                        scores[t] = s
            if not scores:
                continue
            score = pd.Series(scores)

            # ── Market cap (price × shares outstanding) ───────────────────────
            if prev not in raw_close.index or prev not in shares.index:
                continue
            px_r = raw_close.loc[prev].reindex(score.index)
            sh_r = shares.loc[prev].reindex(score.index)
            mcap = (px_r * sh_r).dropna()
            mcap = mcap[mcap > min_mcap]
            score = score.reindex(mcap.index).dropna()
            if score.empty:
                continue

            # ── Liquidity filter ──────────────────────────────────────────────
            if not adtv.empty and prev in adtv.index:
                liq = adtv.loc[prev].reindex(score.index, fill_value=0.0)
                score = score[liq >= min_adtv]
            if len(score) < 2:
                continue

            # ── Keep top 3 quartiles by score (drop lowest quartile) ──────────
            cutoff = score.quantile(0.25)
            score = score[score >= cutoff]

            w = blend_weights(mcap.reindex(score.index), score, cap)
            for t, wt in w.items():
                tw.iloc[i, tw.columns.get_loc(t)] = wt

        return ret, tw


if __name__ == "__main__":
    # ── self-check ────────────────────────────────────────────────────────────
    idx = pd.date_range("2018-01-31", periods=48, freq="ME")

    # constant 6% DY → score 0.06, cap doesn't bind
    dy = pd.Series(0.06, index=idx)
    assert abs(dividend_score(dy, idx[-1]) - 0.06) < 1e-9

    # a spike window gets capped near mean+std, pulling the score below raw mean
    dy2 = pd.Series(0.05, index=idx).copy()
    dy2.iloc[-1] = 0.50
    assert dividend_score(dy2, idx[-1]) < (0.05 + 0.50 + 0.05) / 3

    # missing 36m history → NaN
    assert np.isnan(dividend_score(pd.Series(0.06, index=idx[:5]), idx[4]))

    # cap: a would-be 80% name is clamped and the excess redistributed (sum 1)
    w = cap_weights(pd.Series({"A": 8.0, "B": 1.0, "C": 1.0}), cap=0.50)
    assert abs(w["A"] - 0.50) < 1e-9 and abs(w.sum() - 1.0) < 1e-9

    # blend: equal mcap, skewed score → weights between the two proportions
    bw = blend_weights(
        pd.Series({"A": 100.0, "B": 100.0}),
        pd.Series({"A": 3.0, "B": 1.0}),
        cap=1.0,
    )
    assert abs(bw.sum() - 1.0) < 1e-9 and bw["A"] > bw["B"]
    print("teva_ativos_reais self-check OK")
