"""200-SMA momentum-tilt: DIVO11 / IVVB11 / CDI.

Sisters of the fixed DIVO11/IVVB11/CDI blends (thirds, 40/40/20), but instead of
constant weights the two risk sleeves are tilted by their % distance from a
200-day SMA — allocate more to the asset trending further *above* its SMA, and
route weight to CDI when assets sit *below* it. A momentum / trend-following blend.

Three formulas are registered so they can be compared head-to-head; they differ
only in how CDI earns weight and how "never 100% in one asset" is enforced:

  * cdi_absorb   — risk weight ∝ distance-above-SMA; CDI ∝ the summed shortfall
                   of assets below their SMA. A max cap (max_weight) keeps any
                   single asset — CDI included — below 100%.
  * two_buckets  — each risk asset owns a 1/N bucket, held only while it is above
                   its SMA, else that bucket → CDI. Caps each risk asset at 1/N;
                   CDI can reach 100% in a full downtrend (by design).
  * baseline_floor — 1/N-each anchor, tilted toward above-SMA assets by `tilt`,
                   with a min_weight floor on every sleeve so none hits 0 or 100%
                   (CDI therefore always holds ≥ floor).

Cadence is the standard `rebalance_freq` knob (ME/QE/W-FRI): build_shared_data
already resamples the whole grid to it, so switching monthly↔quarterly is free.
The heavy lifting (rebalancing, tax, slippage) stays in run_simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_REBALANCE_FREQ,
)
from backtests.core.data import download_benchmark

# {portfolio column: Yahoo ticker}. CDI is added separately as CDI_ASSET.
_RISK_ASSETS = {"DIVO11": "DIVO11.SA", "IVVB11": "IVVB11.SA"}


# ── Weight construction ───────────────────────────────────────────────────────

def _bounded_weights(raw: dict, lo: float, hi: float) -> dict:
    """Normalize ``raw`` (non-negative scores) to sum 1 with lo ≤ wᵢ ≤ hi, by
    water-filling: clamp violators to the bound, spread the residual across the
    still-free assets proportionally (equal split if they are all at zero)."""
    keys = list(raw)
    n = len(keys)
    if n == 1:
        return {keys[0]: 1.0}
    lo = min(lo, 1.0 / n)          # keep the bounds feasible for this count
    hi = max(hi, 1.0 / n)
    s = sum(max(raw[k], 0.0) for k in keys)
    w = {k: (max(raw[k], 0.0) / s if s > 0 else 1.0 / n) for k in keys}
    for _ in range(2 * n):
        fixed = {}
        for k in keys:
            if w[k] > hi + 1e-12:
                fixed[k] = hi
            elif w[k] < lo - 1e-12:
                fixed[k] = lo
        if not fixed:
            break
        for k, v in fixed.items():
            w[k] = v
        free = [k for k in keys if k not in fixed]
        resid = 1.0 - sum(w.values())
        if not free or abs(resid) < 1e-12:
            break
        fs = sum(w[k] for k in free)
        for k in free:                       # ponytail: equal split when free legs are all 0
            w[k] += resid * (w[k] / fs if fs > 1e-12 else 1.0 / len(free))
    tot = sum(w.values())
    return {k: v / tot for k, v in w.items()}


def _weights_row(formula: str, dists: dict, cap: float, floor: float, tilt: float) -> dict:
    """Target weights for one rebalance from each risk asset's SMA distance.

    ``dists`` maps risk column -> % distance from its SMA (NaN = not yet live /
    SMA still warming, which parks that sleeve's share in CDI)."""
    live = {k: float(v) for k, v in dists.items() if pd.notna(v)}
    if not live:
        return {"CDI_ASSET": 1.0}            # nothing tradable yet → park

    if formula == "two_buckets":
        share = 1.0 / len(dists)             # each risk asset owns an equal bucket
        w = {"CDI_ASSET": 0.0}
        for k, v in dists.items():
            if pd.notna(v) and v > 0:
                w[k] = w.get(k, 0.0) + share
            else:
                w["CDI_ASSET"] += share       # below SMA or not live → risk-off
        return {k: x for k, x in w.items() if x > 0}

    if formula == "cdi_absorb":
        raw = {k: max(v, 0.0) for k, v in live.items()}          # distance-above only
        raw["CDI_ASSET"] = sum(max(-v, 0.0) for v in live.values())  # shortfall below SMA
        if sum(raw.values()) <= 0:
            return {"CDI_ASSET": 1.0}
        return _bounded_weights(raw, 0.0, cap)

    if formula == "baseline_floor":
        base = 1.0 / (len(live) + 1)         # +1 for the CDI anchor
        raw = {k: max(base + tilt * v, 0.0) for k, v in live.items()}
        raw["CDI_ASSET"] = base
        hi = 1.0 - (len(raw) - 1) * floor
        return _bounded_weights(raw, floor, hi)

    raise ValueError(f"unknown formula {formula!r}")


# ── Shared engine ─────────────────────────────────────────────────────────────

def _common_specs() -> list[ParameterSpec]:
    return [
        COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
        COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_REBALANCE_FREQ,
        ParameterSpec("sma_window", "SMA Window (days)", "int", 200,
                      description="Trailing daily SMA the distance is measured from",
                      min_value=20, max_value=400, step=10),
        ParameterSpec("max_weight", "Max Weight per Asset", "float", 0.60,
                      description="Cap per sleeve (cdi_absorb) — keeps any single asset < 100%",
                      min_value=0.34, max_value=1.0, step=0.05),
        ParameterSpec("min_weight", "Min Weight / Floor", "float", 0.10,
                      description="Floor per sleeve (baseline_floor) — keeps every sleeve > 0",
                      min_value=0.0, max_value=0.33, step=0.05),
        ParameterSpec("tilt", "Momentum Tilt Strength", "float", 3.0,
                      description="Distance-to-weight gain (baseline_floor)",
                      min_value=0.0, max_value=10.0, step=0.5),
    ]


def _grid_signals(shared: dict, win: int) -> tuple[pd.DataFrame, dict]:
    """Return (returns_matrix, {col: SMA-distance signal}) aligned to the grid.

    The distance is computed on daily prices then shifted one grid period so a
    weight for period t only sees data through t-1 (no look-ahead), matching the
    ``signal.iloc[i-1]`` convention used across the codebase."""
    cdi_m = shared["cdi_monthly"]
    grid = cdi_m.index
    dl_start = (grid[0] - pd.DateOffset(days=400)).strftime("%Y-%m-%d")   # SMA warm-up
    end = grid[-1].strftime("%Y-%m-%d")

    cols = {"CDI_ASSET": cdi_m}
    dist = {}
    for col, tk in _RISK_ASSETS.items():
        try:
            px = download_benchmark(tk, dl_start, end).astype(float)
        except Exception:
            px = pd.Series(dtype=float)
        if px.dropna().empty:
            dist[col] = pd.Series(np.nan, index=grid)
            cols[col] = pd.Series(0.0, index=grid)
            continue
        d_daily = px / px.rolling(win).mean() - 1.0
        dist[col] = d_daily.reindex(grid, method="ffill").shift(1)
        cols[col] = px.reindex(grid, method="ffill").pct_change()

    r = pd.DataFrame(cols).reindex(grid)
    r["CDI_ASSET"] = cdi_m                       # CDI is exact, never ffilled/pct_changed
    return r, dist


def _run(formula: str, shared: dict, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    win = int(params.get("sma_window", 200))
    cap = float(params.get("max_weight", 0.60))
    floor = float(params.get("min_weight", 0.10))
    tilt = float(params.get("tilt", 3.0))

    r, dist = _grid_signals(shared, win)
    tw = pd.DataFrame(0.0, index=r.index, columns=r.columns)
    for i, dt in enumerate(r.index):
        row = {c: dist[c].iloc[i] for c in _RISK_ASSETS}
        for k, v in _weights_row(formula, row, cap, floor, tilt).items():
            tw.at[dt, k] = v
    return r.fillna(0.0), tw


# ── Registered strategies (one per formula) ───────────────────────────────────

class _SMATiltBase(StrategyBase):
    FORMULA = ""
    NAME = ""
    DESC = ""

    @property
    def name(self) -> str:
        return self.NAME

    @property
    def description(self) -> str:
        return self.DESC

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return _common_specs()

    def generate_signals(self, shared_data: dict, params: dict):
        return _run(self.FORMULA, shared_data, params)


class SMATiltCDIAbsorb(_SMATiltBase):
    FORMULA = "cdi_absorb"
    NAME = "200SMA Tilt: Momentum Share + CDI Absorb"
    DESC = (
        "DIVO11/IVVB11/CDI tilted by 200-SMA distance. Risk weight ∝ each ETF's "
        "distance ABOVE its SMA; CDI ∝ the summed shortfall of ETFs below their SMA "
        "(so CDI is held only when something is below trend). A max_weight cap keeps "
        "any single asset — CDI included — under 100%."
    )


class SMATiltTwoBuckets(_SMATiltBase):
    FORMULA = "two_buckets"
    NAME = "200SMA Tilt: Two 50% Buckets"
    DESC = (
        "DIVO11 and IVVB11 each own a 50% bucket, held only while above their 200-day "
        "SMA, otherwise that bucket rotates to CDI. Binary trend gate: each ETF is "
        "capped at 50%, and CDI runs 0→100% (full risk-off in a two-sided downtrend)."
    )


class SMATiltBaseline(_SMATiltBase):
    FORMULA = "baseline_floor"
    NAME = "200SMA Tilt: Baseline Thirds + Tilt"
    DESC = (
        "Equal-thirds DIVO11/IVVB11/CDI anchor, tilted toward whichever ETF trades "
        "further above its 200-day SMA (gain = `tilt`), with a min_weight floor on "
        "every sleeve so none reaches 0 or 100% — CDI always keeps a defensive core."
    )


# ── Self-check ────────────────────────────────────────────────────────────────

def _selfcheck() -> None:
    NAN = float("nan")

    def s(w):  # sum
        return round(sum(w.values()), 6)

    # cdi_absorb: caps hold in every regime, CDI only appears with a below-SMA leg
    w = _weights_row("cdi_absorb", {"DIVO11": 0.30, "IVVB11": 0.10}, 0.60, 0.10, 3.0)
    assert s(w) == 1.0 and max(w.values()) <= 0.60 + 1e-9
    assert w.get("CDI_ASSET", 0.0) == 0.0            # both above SMA → no CDI
    w = _weights_row("cdi_absorb", {"DIVO11": -0.10, "IVVB11": -0.20}, 0.60, 0.10, 3.0)
    assert s(w) == 1.0 and w["CDI_ASSET"] <= 0.60 + 1e-9   # never 100% CDI either
    w = _weights_row("cdi_absorb", {"DIVO11": 0.30, "IVVB11": NAN}, 0.60, 0.10, 3.0)
    assert s(w) == 1.0 and w["DIVO11"] <= 0.60 + 1e-9      # dead IVVB → excess parks in CDI
    assert "IVVB11" not in w

    # two_buckets: each risk asset ≤ 0.5; CDI can hit 1.0
    w = _weights_row("two_buckets", {"DIVO11": 0.30, "IVVB11": -0.10}, 0.60, 0.10, 3.0)
    assert w == {"DIVO11": 0.5, "CDI_ASSET": 0.5}
    w = _weights_row("two_buckets", {"DIVO11": -0.1, "IVVB11": -0.1}, 0.60, 0.10, 3.0)
    assert w == {"CDI_ASSET": 1.0}

    # baseline_floor: sums 1, floor respected, nothing ≥ 1-(n-1)*floor
    w = _weights_row("baseline_floor", {"DIVO11": 0.30, "IVVB11": -0.05}, 0.60, 0.10, 3.0)
    assert s(w) == 1.0 and min(w.values()) >= 0.10 - 1e-9 and max(w.values()) <= 0.80 + 1e-9
    assert w["CDI_ASSET"] >= 0.10 - 1e-9

    # both legs dead → parked in CDI for every formula
    for f in ("cdi_absorb", "two_buckets", "baseline_floor"):
        assert _weights_row(f, {"DIVO11": NAN, "IVVB11": NAN}, 0.6, 0.1, 3.0) == {"CDI_ASSET": 1.0}

    print("sma_momentum_tilt self-check OK")


if __name__ == "__main__":
    _selfcheck()
