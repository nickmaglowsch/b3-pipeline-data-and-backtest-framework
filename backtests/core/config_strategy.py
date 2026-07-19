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

import numpy as np
import pandas as pd

from backtests.core.strategy_base import (
    StrategyBase, ParameterSpec,
    COMMON_START_DATE, COMMON_END_DATE, COMMON_INITIAL_CAPITAL,
    COMMON_TAX_RATE, COMMON_SLIPPAGE, COMMON_MIN_ADTV, COMMON_REBALANCE_FREQ,
)


# ── Signal library ────────────────────────────────────────────────────────────
# Each fn(shared_data, cfg) -> wide DataFrame, higher = better. `cfg` is the
# signal's own spec dict (holds lookback, shift, ...). Formulas are lifted
# verbatim from the strategy each replaces (referenced in comments).

def _sig_momentum(shared: dict, cfg: dict) -> pd.DataFrame:
    lb = int(cfg.get("lookback", 12))
    sh = int(cfg.get("shift", 1))
    return shared["log_ret"].shift(sh).rolling(lb).sum()          # smallcap_momentum.py:63


def _sig_low_vol(shared: dict, cfg: dict) -> pd.DataFrame:
    lb = int(cfg.get("lookback", 12))
    sh = int(cfg.get("shift", 0))
    return -shared["ret"].shift(sh).rolling(lb).std()             # low_volatility.py:57


def _sig_sharpe_mom(shared: dict, cfg: dict) -> pd.DataFrame:
    lb = int(cfg.get("lookback", 12))
    mom = shared["log_ret"].shift(1).rolling(lb).sum()
    vol = shared["ret"].shift(1).rolling(lb).std()
    return mom / vol                                             # momentum_sharpe.py:57-59


def _sig_blended_vol(shared: dict, cfg: dict) -> pd.DataFrame:
    ret = shared["ret"]
    vol_60d = shared.get("vol_60d", ret.rolling(5).std())
    vol_20d = shared.get("vol_20d", ret.rolling(2).std())
    return -(0.5 * vol_60d + 0.5 * vol_20d)                      # adaptive_low_vol.py:82-85


SIGNAL_LIBRARY = {
    "momentum": _sig_momentum,
    "low_vol": _sig_low_vol,
    "sharpe_mom": _sig_sharpe_mom,
    "blended_vol": _sig_blended_vol,
}


def resolve_factor(cfg: dict, shared: dict) -> pd.DataFrame:
    """Resolve one factor cfg {factor: <ref>, ...} to a wide (higher=better) df.

    ``factor`` is either a named signal, a bare ``shared_data`` key, or
    ``store:<feature_id>`` for the research/discovery IC factor store.
    """
    ref = cfg["factor"]
    if ref in SIGNAL_LIBRARY:
        return SIGNAL_LIBRARY[ref](shared, cfg)
    if ref.startswith("store:"):
        return _load_store_feature(ref[len("store:"):], shared)
    if ref in shared:
        return shared[ref]
    raise KeyError(
        f"Unknown factor {ref!r}: not a named signal, shared_data key, or store:<id>"
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
    """Return (signal_df, max_lookback). Single factor -> raw signal; composite
    -> rank-averaged (matches multifactor.py:69)."""
    glitch = shared.get("has_glitch")

    def _mask_glitch(df: pd.DataFrame) -> pd.DataFrame:
        if glitch is not None:
            df = df.copy()
            df[glitch == 1] = np.nan
        return df

    if "composite" in spec:
        parts = spec["composite"]
        composite = None
        max_lb = 0
        for p in parts:
            cfg = {**p, "lookback": params.get("lookback", p.get("lookback", 12))}
            max_lb = max(max_lb, int(cfg.get("lookback", 0)))
            sig = resolve_factor(cfg, shared)
            if p.get("apply_glitch", True):
                sig = _mask_glitch(sig)
            ranked = sig.rank(axis=1, pct=True) * float(p.get("weight", 1.0))
            composite = ranked if composite is None else composite + ranked
        return composite, max_lb

    cfg = {**spec["signal"]}
    cfg["lookback"] = params.get("lookback", cfg.get("lookback", 12))
    sig = resolve_factor(cfg, shared)
    if cfg.get("apply_glitch", True):
        sig = _mask_glitch(sig)
    return sig, int(cfg.get("lookback", 0))


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
    tickers (downloaded) or the special asset ``CDI``. Generic version of
    divo_cdi_ivvb.py:55-86."""

    def __init__(self, spec: dict) -> None:
        self.spec = spec
        self.needs_fundamentals = False

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
            ticker = sleeve["ticker"]
            col = ticker.split(".")[0]
            px = download_benchmark(ticker, start, end).reindex(ret.index, method="ffill")
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

        return r, tw
