"""
Config-driven strategies
========================
Two generic ``StrategyBase`` subclasses that turn "portfolio with rebalancing"
strategies into data. Instead of a bespoke Python class per strategy, a YAML
spec (loaded by ``spec_loader.py``) drives one of:

  * ``RankAndHold``  — rank a factor over a liquid universe, take the top
    N / top-pct, weight them, rebalance. Covers the ~16 ranked-equity
    strategies whose only real difference was one signal line.
  * ``FixedWeight``  — hold a fixed set of sleeves at constant weights,
    rebalanced on the grid. Covers the ETF/CDI blend strategies.

The heavy lifting (rebalancing, tax, slippage) stays in ``run_simulation``;
these classes only build ``target_weights``.

Signal convention: every named signal returns a wide (date x ticker) DataFrame
where **higher = better**, so selection is always ``nlargest``. A low-P/E
strategy is expressed as ``earnings_yield`` (higher yield = cheaper), etc.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

from backtests.core import signal_dsl
from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


# ── Signals as data ───────────────────────────────────────────────────────────
# A factor is a string expression (the signal DSL, see signal_dsl.py) over
# shared_data frames. Named signals live in backtests/strategies/signals.yaml;
# specs may also inline an `expr:`. resolve_factor / _build_signal wire it in.

@functools.lru_cache(maxsize=1)
def _signal_defs() -> dict:
    """Load the named-signal definitions from backtests/strategies/signals.yaml
    (factor name -> {expr, <default params>})."""
    import yaml
    path = Path(__file__).resolve().parent.parent / "strategies" / "signals.yaml"
    return yaml.safe_load(path.read_text()) or {}


def _eval_expr(expr: str, scalars: dict, shared: dict) -> pd.DataFrame:
    """Evaluate a signal expression in a namespace of shared_data frames plus any
    numeric scalars (lookback, etc.)."""
    ns = dict(shared)
    ns.update({k: v for k, v in scalars.items() if isinstance(v, (int, float))})
    return signal_dsl.evaluate(expr, ns)


def resolve_factor(cfg: dict, shared: dict, params: dict | None = None) -> pd.DataFrame:
    """Resolve one factor cfg to a wide (higher=better) frame. In precedence:

    1. an inline ``expr:`` string (the signal DSL);
    2. a named signal from ``signals.yaml`` (also a DSL ``expr``);
    3. ``store:<feature_id>`` — the research/discovery IC factor store;
    4. a bare ``shared_data`` key (e.g. ``mf_composite``, ``f_pe_ratio_dyn``).

    ``params`` (UI/default values) override the signal's own defaults, so a
    ``lookback`` slider flows into the expression.
    """
    scalars = {**cfg, **(params or {})}
    if "expr" in cfg:
        return _eval_expr(cfg["expr"], scalars, shared)

    ref = cfg.get("factor")
    defs = _signal_defs()
    if ref in defs:
        entry = defs[ref]
        merged = {**entry, **scalars}       # spec/params override the signal's defaults
        return _eval_expr(entry["expr"], merged, shared)
    if isinstance(ref, str) and ref.startswith("store:"):
        return _load_store_feature(ref[len("store:"):], shared)
    if ref in shared:
        return shared[ref]
    raise KeyError(
        f"Unknown factor {ref!r}: not an expr, a name in signals.yaml, "
        f"a shared_data key, or store:<id>"
    )


def _load_store_feature(feature_id: str, shared: dict) -> pd.DataFrame:
    """Load a wide feature from the research/discovery FeatureStore and align it
    to the rebalance calendar (as-of ffill), mirroring the fundamentals path in
    shared_data.py:360-369."""
    from research.discovery.store import FeatureStore

    df = FeatureStore().load_feature(feature_id)      # wide date x ticker, daily
    rebal = shared["ret"].index
    idx = df.index.union(rebal).sort_values()
    return df.reindex(idx).ffill().reindex(rebal)


def _build_signal(spec: dict, params: dict, shared: dict) -> tuple[pd.DataFrame, int]:
    """Return (signal_df, lookback). The signal is a single factor/expression;
    composite blends are just an ``expr`` using ``rank()``/``zscore()`` (see
    specs/multifactor.yaml)."""
    cfg = {**spec["signal"]}
    cfg["lookback"] = params.get("lookback", cfg.get("lookback", 0))
    sig = resolve_factor(cfg, shared, params)
    if cfg.get("apply_glitch", True):
        glitch = shared.get("has_glitch")
        if glitch is not None:
            sig = sig.where(glitch != 1)
    return sig, int(cfg.get("lookback", 0) or 0)


def _eval_regime(regime: dict, shared: dict, i: int):
    """Evaluate the regime rules at rebalance row i. Each input is a boolean read
    of a shared_data flag at row i-offset (optionally `> gt`). Returns the first
    matching rule dict (carries `top_pct` or `park`), or None (=> park in cash)."""
    vals = {}
    for nm, ic in regime.get("inputs", {}).items():
        flag = shared[ic["flag"]]
        j = i - int(ic.get("offset", 1))
        if 0 <= j < len(flag):
            raw = flag.iloc[j]
            if "gt" in ic:
                vals[nm] = bool(pd.notna(raw) and float(raw) > ic["gt"])
            else:
                vals[nm] = bool(raw) if pd.notna(raw) else False
        else:
            vals[nm] = False
    for rule in regime.get("rules", []):
        if rule.get("default"):
            return rule
        conds = {k: v for k, v in rule.items() if k not in ("top_pct", "park", "default")}
        if all(vals.get(k) == v for k, v in conds.items()):
            return rule
    return None


# ── Ranked-equity engine ──────────────────────────────────────────────────────

class RankAndHold(StrategyBase):
    """Generic rank -> top-N -> weight -> rebalance strategy, driven by a spec."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.needs_fundamentals = bool(spec.get("needs_fundamentals", False))

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def description(self) -> str:
        return self.spec.get("description", self.spec["name"])

    def get_parameter_specs(self) -> list[ParameterSpec]:
        uni = self.spec.get("universe", {})
        sel = self.spec.get("selection", {})
        specs = [
            COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
            COMMON_TAX_RATE, COMMON_SLIPPAGE,
            ParameterSpec("min_adtv", "Min ADTV (BRL)", "float",
                          float(uni.get("min_adtv", 1_000_000.0)),
                          min_value=0.0, step=100_000.0),
            ParameterSpec("min_price", "Min Price (BRL)", "float",
                          float(uni.get("min_price", 1.0)),
                          min_value=0.0, max_value=10.0, step=0.5),
            ParameterSpec("rebalance_freq", "Rebalance Frequency", "choice",
                          self.spec.get("rebalance", "ME"), choices=["ME", "QE", "W-FRI"]),
        ]
        # Expose lookback for whichever the signal uses (single or composite —
        # one shared lookback drives both composite legs, like multifactor.py).
        lb = self.spec.get("signal", {}).get("lookback")
        if lb is None and self.spec.get("composite"):
            lb = self.spec["composite"][0].get("lookback")
        if lb is not None:
            specs.append(ParameterSpec("lookback", "Lookback Periods", "int", int(lb),
                                       min_value=1, max_value=36, step=1))
        # Suppress the top_pct slider when a regime overlay always overrides it
        # (every rule sets its own top_pct or parks) — it would be a dead knob.
        regime = self.spec.get("regime")
        rules = (regime or {}).get("rules", [])
        top_pct_dead = bool(regime and rules and all(
            ("top_pct" in r or r.get("park") or r.get("default")) for r in rules))
        if "top_pct" in sel and not top_pct_dead:
            specs.append(ParameterSpec("top_pct", "Top Percentile", "float",
                                       float(sel["top_pct"]),
                                       min_value=0.01, max_value=0.50, step=0.01))
        if "top_n" in sel:
            specs.append(ParameterSpec("top_n", "Number of Stocks", "int",
                                       int(sel["top_n"]), min_value=1, max_value=100, step=1))
        return specs

    def generate_signals(self, shared_data: dict, params: dict):
        spec = self.spec
        uni = spec.get("universe", {})
        sel = spec.get("selection", {})

        ret = shared_data["ret"]
        adtv = shared_data["adtv"]
        raw_close = shared_data["raw_close"]

        min_adtv = float(params.get("min_adtv", uni.get("min_adtv", 1_000_000)))
        min_price = float(params.get("min_price", uni.get("min_price", 1.0)))
        min_names = int(sel.get("min_names", 5))
        top_pct = params.get("top_pct", sel.get("top_pct"))
        top_n = params.get("top_n", sel.get("top_n"))
        vrange = spec.get("signal", {}).get("range")
        smallcap = bool(uni.get("smallcap_below_median", False))
        weighting = spec.get("weighting", "equal")

        signal, max_lb = _build_signal(spec, params, shared_data)
        start = int(spec.get("warmup", max_lb + int(spec.get("warmup_pad", 1))))
        start = max(start, 1)

        tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
        r = ret.copy()

        # Optional macro-regime overlay: gate the equity book on a regime flag,
        # parking in CDI (and holding IBOV as an available asset) when "off".
        regime = spec.get("regime")
        cash = None
        if regime:
            cash = "CDI_ASSET"
            r[cash] = shared_data["cdi_monthly"]
            if regime.get("add_ibov", True):
                r["IBOV"] = shared_data["ibov_ret"]
            tw[cash] = 0.0
            park_few = bool(regime.get("park_when_too_few", True))
            cash_loc = tw.columns.get_loc(cash)

        for i in range(start, len(ret)):
            eff_top_pct = top_pct
            if regime:
                action = _eval_regime(regime, shared_data, i)
                if action is None or action.get("park"):
                    tw.iloc[i, cash_loc] = 1.0
                    continue
                eff_top_pct = action.get("top_pct", top_pct)

            sig_row = signal.iloc[i - 1]
            adtv_r = adtv.iloc[i - 1]
            raw_r = raw_close.iloc[i - 1]

            mask = (adtv_r >= min_adtv) & (raw_r >= min_price)
            valid = sig_row[mask].dropna()
            valid = valid[np.isfinite(valid)]
            if vrange is not None:
                valid = valid[(valid >= vrange[0]) & (valid <= vrange[1])]
            if smallcap:
                liq_adtv = adtv_r[valid.index]
                valid = valid[liq_adtv <= liq_adtv.median()]
            if len(valid) < min_names:
                if regime and park_few:
                    tw.iloc[i, cash_loc] = 1.0
                continue

            n = int(top_n) if top_n else max(min_names, int(len(valid) * eff_top_pct))
            selected = valid.nlargest(n).index.tolist()
            w = self._weights(selected, adtv_r, signal, i, weighting)
            for t, wt in w.items():
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = wt

        return r, tw

    @staticmethod
    def _weights(selected, adtv_r, signal, i, scheme) -> dict:
        if scheme == "market_cap":
            caps = adtv_r[selected].clip(lower=0)          # ponytail: ADTV proxy for cap weight
            tot = caps.sum()
            if tot > 0:
                return (caps / tot).to_dict()
        # equal (default) and fallback
        w = 1.0 / len(selected)
        return {t: w for t in selected}


# ── Fixed-weight blend engine ─────────────────────────────────────────────────

class FixedWeight(StrategyBase):
    """Constant-weight sleeves rebalanced on the grid. Sleeves are B3/Yahoo
    tickers (downloaded), the special asset ``CDI``, or another registered
    ``strategy`` (its whole book becomes one sleeve). Generic version of
    divo_cdi_ivvb.py:55-86."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        # A strategy sleeve may need CVM fundamentals; the spec declares it so
        # the service loads them (can't query the registry here — we're being
        # constructed *inside* registry discovery).
        self.needs_fundamentals = bool(spec.get("needs_fundamentals", False))

    @property
    def name(self) -> str:
        return self.spec["name"]

    @property
    def description(self) -> str:
        return self.spec.get("description", self.spec["name"])

    def get_parameter_specs(self) -> list[ParameterSpec]:
        return [
            COMMON_START_DATE, COMMON_END_DATE,
            ParameterSpec("rebalance_freq", "Rebalance Frequency", "choice",
                          self.spec.get("rebalance", "QE"), choices=["ME", "QE", "W-FRI"]),
        ]

    def generate_signals(self, shared_data: dict, params: dict):
        from backtests.core.data import download_benchmark

        spec = self.spec
        ret = shared_data["ret"]
        cdi = shared_data["cdi_monthly"]
        park = bool(spec.get("park_in_cdi_until_live", True))
        # per_sleeve: park each dead ETF's weight in CDI individually (divo style).
        # all_or_nothing: hold nothing until every ETF is live, else 100% CDI
        # (baseline_thirds style).
        park_mode = spec.get("park_mode", "per_sleeve")

        start, end = str(ret.index[0].date()), str(ret.index[-1].date())
        r = ret.copy()
        r["CDI_ASSET"] = cdi

        tw = pd.DataFrame(0.0, index=ret.index, columns=r.columns)
        tw["CDI_ASSET"] = 0.0

        etf_cols = []
        for sleeve in spec["sleeves"]:
            weight = float(sleeve["weight"])
            if sleeve.get("asset") == "CDI":
                tw["CDI_ASSET"] += weight
                continue
            if "strategy" in sleeve:
                # Nest a registered strategy as one sleeve: run it, scale its
                # target weights by this sleeve's weight, and fold them into the
                # blend. Its own assets (stocks, CDI_ASSET) already share the
                # blend's return matrix, so no new return streams are needed.
                from backtests.core.strategy_registry import get_registry
                sub = get_registry().get(sleeve["strategy"])
                sub_params = {**sub.get_default_parameters(), **params}
                _, tw_sub = sub.generate_signals(shared_data, sub_params)
                for col in tw_sub.columns:
                    if col not in tw.columns:
                        tw[col] = 0.0
                    if col not in r.columns:
                        r[col] = shared_data["cdi_monthly"] if col == "CDI_ASSET" else 0.0
                    tw[col] = tw[col] + weight * tw_sub[col].reindex(tw.index).fillna(0.0)
                continue
            ticker = sleeve["ticker"]
            col = ticker.split(".")[0]
            px_daily = download_benchmark(ticker, start, end)
            # Stash the daily ETF returns so the daily NAV reconstruction
            # (metrics.strategy_daily_values) can mark this sleeve intra-period;
            # the rebalance-cadence `r[col]` below can't reveal its drawdown.
            shared_data.setdefault("_daily_asset_ret", {})[col] = px_daily.pct_change()
            px = px_daily.reindex(ret.index, method="ffill")
            r[col] = px.pct_change().fillna(0.0)
            tw[col] = weight
            etf_cols.append(col)
            if park and park_mode == "per_sleeve":
                dead = px.isna()
                tw.loc[dead, col] = 0.0
                tw.loc[dead, "CDI_ASSET"] += weight

        if park and park_mode == "all_or_nothing" and etf_cols:
            live = (r[etf_cols] != 0).all(axis=1).cummax()   # baseline_thirds.py:70-71
            tw.loc[~live, etf_cols + ["CDI_ASSET"]] = 0.0
            tw.loc[~live, "CDI_ASSET"] = 1.0

        # Park any residual on live rows into CDI so weights always sum to 1.
        # A nested `strategy` sleeve emits all-zero weights during its own warmup
        # (fundamentals/EWMA), leaving its share unallocated; run_simulation does
        # NOT hold uninvested cash (NAV = sum of positions, simulation.py:347-350),
        # so that share would be vaporized every rebalance — compounding a fake
        # drawdown (a 50/50 blend bled 100k -> 26k through the sleeve warmup).
        live_rows = tw.abs().sum(axis=1) > 0
        resid = (1.0 - tw.sum(axis=1)).clip(lower=0.0)
        tw.loc[live_rows, "CDI_ASSET"] = tw.loc[live_rows, "CDI_ASSET"] + resid[live_rows]

        return r, tw
