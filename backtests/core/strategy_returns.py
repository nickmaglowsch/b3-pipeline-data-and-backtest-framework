"""
Strategy Return Extraction Module
==================================
Runs all 8 core B3 strategies and returns their monthly after-tax return series
as a single clean DataFrame. This decouples signal generation from portfolio
construction and eliminates code duplication across compare_all.py,
correlation_matrix.py, and portfolio_low_corr_backtest.py.

Usage:
    from backtests.core.strategy_returns import build_strategy_returns

    returns_df, sim_results, regime_signals = build_strategy_returns()
    # returns_df: DataFrame[date x strategy], values = after-tax monthly returns
    # sim_results: dict of {strategy_name: simulation_result_dict}
    # regime_signals: dict of {signal_name: pd.Series} for downstream regime use
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# ── Path setup (works whether imported or run directly) ──────────────────────
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BACKTESTS_DIR = os.path.dirname(_THIS_DIR)
_PROJECT_ROOT = os.path.dirname(_BACKTESTS_DIR)

for _p in [_PROJECT_ROOT, _BACKTESTS_DIR]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.metrics import value_to_ret
from core.simulation import run_simulation

# ── Default DB path ──────────────────────────────────────────────────────────
_DEFAULT_DB = os.path.join(_PROJECT_ROOT, "b3_market_data.sqlite")

# ── Constants ────────────────────────────────────────────────────────────────
_FREQ = "ME"
_PERIODS = 12
_MIN_ADTV = 1_000_000
_MIN_PRICE = 1.0
_LOOKBACK = 12

# SmallcapMom uses a lower ADTV threshold -- flagged for downstream exclusion
SMALLCAP_MOM_NOTE = (
    "SmallcapMom uses ADTV >= R$100K (below the R$1M liquid floor). "
    "Downstream portfolio optimization should test with and without this strategy."
)


# ════════════════════════════════════════════════════════════════════════════
#  Private: strategy implementations
# ════════════════════════════════════════════════════════════════════════════

def _run_cdi_ma200(shared: dict, cfg: dict) -> dict:
    """
    Strategy 1: CDI + MA200 Trend Filter
    Hold equal-weight stocks above 200-day MA during COPOM easing; else CDI.
    Source: compare_all.py lines 83-117
    """
    ret = shared["ret"]
    adtv = shared["adtv"]
    raw_close = shared["raw_close"]
    has_glitch = shared["has_glitch"]
    is_easing = shared["is_easing"]
    above_ma200 = shared["above_ma200"]
    cdi_monthly = shared["cdi_monthly"]
    ibov_ret = shared["ibov_ret"]

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    for i in range(13, len(ret)):
        easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        if not easing:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        above = above_ma200.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        gl = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()
        mask = above.fillna(False) & (adtv_r >= _MIN_ADTV) & (raw_r >= _MIN_PRICE)
        if len(gl) > 0:
            mask = mask & (gl != 1)
        tickers = mask[mask].index.tolist()
        if not tickers:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
        else:
            w = 1.0 / len(tickers)
            for t in tickers:
                if t in tw.columns:
                    tw.iloc[i, tw.columns.get_loc(t)] = w

    return _run_sim("CDI+MA200", r, tw, cfg)


def _run_res_multifactor(shared: dict, cfg: dict) -> dict:
    """
    Strategy 2: Research Multi-Factor (5 factors + regime filter)
    Requires 2-of-3 regime signals (easing, calm, uptrend). Composite = distance
    from MA200 + low vol (60d) + low ATR + low vol (20d) + high ADTV.
    Source: compare_all.py lines 119-175
    """
    ret = shared["ret"]
    adtv = shared["adtv"]
    raw_close = shared["raw_close"]
    has_glitch = shared["has_glitch"]
    is_easing = shared["is_easing"]
    ibov_calm = shared["ibov_calm"]
    ibov_uptrend = shared["ibov_uptrend"]
    dist_ma200 = shared["dist_ma200"]
    vol_60d = shared["vol_60d"]
    atr_m = shared["atr_m"]
    vol_20d = shared["vol_20d"]
    cdi_monthly = shared["cdi_monthly"]
    ibov_ret = shared["ibov_ret"]

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    for i in range(14, len(ret)):
        sig_easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        sig_calm = bool(ibov_calm.iloc[i - 1]) if i - 1 < len(ibov_calm) else False
        sig_up = bool(ibov_uptrend.iloc[i - 1]) if i - 1 < len(ibov_uptrend) else False
        if (int(sig_easing) + int(sig_calm) + int(sig_up)) < 2:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        composite = (
            0.2 * dist_ma200.iloc[i - 1].rank(pct=True)
            + 0.2 * (-vol_60d.iloc[i - 1]).rank(pct=True)
            + 0.2 * (-atr_m.iloc[i - 1]).rank(pct=True)
            + 0.2 * (-vol_20d.iloc[i - 1]).rank(pct=True)
            + 0.2 * adtv.iloc[i - 1].rank(pct=True)
        )
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        gl = has_glitch.iloc[i - 1] if i - 1 < len(has_glitch) else pd.Series()
        mask = (adtv_r >= _MIN_ADTV) & (raw_r >= _MIN_PRICE)
        if len(gl) > 0:
            mask = mask & (gl != 1)
        valid = composite[mask].dropna()
        if len(valid) < 5:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw.columns:
                tw.iloc[i, tw.columns.get_loc(t)] = w

    return _run_sim("Res.MultiFactor", r, tw, cfg)


def _run_regime_switching(shared: dict, cfg: dict) -> dict:
    """
    Strategy 3: Regime Switching (IBOV above 10-month MA → equities, else CDI)
    Uses MultiFactor signal (50% mom + 50% low-vol, top 10%) for stock selection.
    Source: compare_all.py lines 177-217
    """
    ret = shared["ret"]
    adtv = shared["adtv"]
    raw_close = shared["raw_close"]
    has_glitch = shared["has_glitch"]
    ibov_above = shared["ibov_above"]
    mf_composite = shared["mf_composite"]
    cdi_monthly = shared["cdi_monthly"]
    ibov_ret = shared["ibov_ret"]

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    for i in range(_LOOKBACK + 2, len(ret)):
        above = bool(ibov_above.iloc[i - 1]) if i - 1 < len(ibov_above) else False
        if not above:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        sig_row = mf_composite.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= _MIN_ADTV) & (raw_r >= _MIN_PRICE)
        valid = sig_row[mask].dropna()
        if len(valid) < 5:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw.columns:
                tw.iloc[i, tw.columns.get_loc(t)] = w

    return _run_sim("RegimeSwitching", r, tw, cfg)


def _run_copom_easing(shared: dict, cfg: dict) -> dict:
    """
    Strategy 4: COPOM Easing (simple IBOV vs CDI binary switch)
    During COPOM easing: 100% IBOV. During tightening: 100% CDI.
    Source: compare_all.py lines 219-237
    """
    ret = shared["ret"]
    is_easing = shared["is_easing"]
    cdi_monthly = shared["cdi_monthly"]
    ibov_ret = shared["ibov_ret"]

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    tw["CDI_ASSET"] = 0.0
    tw["IBOV"] = 0.0
    r = ret.copy()
    r["CDI_ASSET"] = cdi_monthly
    r["IBOV"] = ibov_ret

    for i in range(5, len(ret)):
        easing = bool(is_easing.iloc[i]) if i < len(is_easing) else False
        if easing:
            tw.iloc[i, tw.columns.get_loc("IBOV")] = 1.0
        else:
            tw.iloc[i, tw.columns.get_loc("CDI_ASSET")] = 1.0

    return _run_sim("COPOM Easing", r, tw, cfg)


def _run_multifactor(shared: dict, cfg: dict) -> dict:
    """
    Strategy 5: Multifactor (Mom + Low Vol, no regime filter)
    Composite = 50% momentum + 50% low-vol, top 10% liquid universe, always invested.
    Source: compare_all.py lines 239-266
    """
    ret = shared["ret"]
    adtv = shared["adtv"]
    raw_close = shared["raw_close"]
    log_ret = shared["log_ret"]
    has_glitch = shared["has_glitch"]

    mom_sig = log_ret.shift(1).rolling(_LOOKBACK).sum()
    mom_sig[has_glitch == 1] = np.nan
    vol_sig = -ret.shift(1).rolling(_LOOKBACK).std()
    vol_sig[has_glitch == 1] = np.nan
    composite = mom_sig.rank(axis=1, pct=True) * 0.5 + vol_sig.rank(axis=1, pct=True) * 0.5

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

    for i in range(_LOOKBACK + 2, len(ret)):
        sig_row = composite.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= _MIN_ADTV) & (raw_r >= _MIN_PRICE)
        valid = sig_row[mask].dropna()
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw.columns:
                tw.iloc[i, tw.columns.get_loc(t)] = w

    return _run_sim("MultiFactor", r, tw, cfg)


def _run_smallcap_mom(shared: dict, cfg: dict) -> dict:
    """
    Strategy 6: Smallcap Momentum
    NOTE: Uses ADTV >= R$100K (below the standard R$1M liquid floor).
    Selects stocks below median ADTV (small caps) by 6-month momentum, top 20%.
    Source: compare_all.py lines 268-295
    """
    ret = shared["ret"]
    adtv = shared["adtv"]
    raw_close = shared["raw_close"]
    log_ret = shared["log_ret"]
    has_glitch = shared["has_glitch"]

    mom_6m = log_ret.shift(1).rolling(6).sum()
    mom_6m[has_glitch == 1] = np.nan
    adtv_median = adtv.median(axis=1)

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

    for i in range(8, len(ret)):
        sig_row = mom_6m.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        med = adtv_median.iloc[i - 1]
        # Smallcap: below median ADTV but above 100K minimum
        mask = (adtv_r >= 100_000) & (adtv_r < med) & (raw_r >= _MIN_PRICE)
        valid = sig_row[mask].dropna()
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * 0.20))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw.columns:
                tw.iloc[i, tw.columns.get_loc(t)] = w

    return _run_sim("SmallcapMom", r, tw, cfg)


def _run_low_vol(shared: dict, cfg: dict) -> dict:
    """
    Strategy 7: Low Volatility (unconditional, no regime filter)
    Selects lowest volatility stocks from liquid universe, top 10%.
    Source: compare_all.py lines 297-321
    """
    ret = shared["ret"]
    adtv = shared["adtv"]
    raw_close = shared["raw_close"]
    has_glitch = shared["has_glitch"]

    vol_sig = -ret.rolling(_LOOKBACK).std()
    vol_sig[has_glitch == 1] = np.nan

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

    for i in range(_LOOKBACK + 1, len(ret)):
        sig_row = vol_sig.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= _MIN_ADTV) & (raw_r >= _MIN_PRICE)
        valid = sig_row[mask].dropna()
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw.columns:
                tw.iloc[i, tw.columns.get_loc(t)] = w

    return _run_sim("LowVol", r, tw, cfg)


def _run_mom_sharpe(shared: dict, cfg: dict) -> dict:
    """
    Strategy 8: Momentum Sharpe (risk-adjusted momentum)
    Signal = 12-month log return / 12-month volatility, top 10%.
    Source: compare_all.py lines 323-350
    """
    ret = shared["ret"]
    adtv = shared["adtv"]
    raw_close = shared["raw_close"]
    log_ret = shared["log_ret"]
    has_glitch = shared["has_glitch"]

    mom_12m = log_ret.shift(1).rolling(_LOOKBACK).sum()
    vol_12m = ret.shift(1).rolling(_LOOKBACK).std()
    sharpe_sig = mom_12m / vol_12m
    sharpe_sig[has_glitch == 1] = np.nan

    tw = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    r = ret.copy()

    for i in range(_LOOKBACK + 2, len(ret)):
        sig_row = sharpe_sig.iloc[i - 1]
        adtv_r = adtv.iloc[i - 1]
        raw_r = raw_close.iloc[i - 1]
        mask = (adtv_r >= _MIN_ADTV) & (raw_r >= _MIN_PRICE)
        valid = sig_row[mask].dropna()
        valid = valid[np.isfinite(valid)]
        if len(valid) < 5:
            continue
        n = max(1, int(len(valid) * 0.10))
        sel = valid.nlargest(n).index.tolist()
        w = 1.0 / len(sel)
        for t in sel:
            if t in tw.columns:
                tw.iloc[i, tw.columns.get_loc(t)] = w

    return _run_sim("MomSharpe", r, tw, cfg)


def _run_sim(name: str, r: pd.DataFrame, tw: pd.DataFrame, cfg: dict) -> dict:
    """Helper: call run_simulation with config dict, return result."""
    return run_simulation(
        returns_matrix=r.fillna(0.0),
        target_weights=tw,
        initial_capital=cfg["capital"],
        tax_rate=cfg["tax_rate"],
        slippage=cfg["slippage"],
        name=name,
        monthly_sales_exemption=cfg["monthly_sales_exemption"],
    )


# ════════════════════════════════════════════════════════════════════════════
#  Public API
# ════════════════════════════════════════════════════════════════════════════

def build_strategy_returns(
    db_path: str = None,
    start: str = "2005-01-01",
    end: str = None,
    capital: float = 100_000,
    tax_rate: float = 0.15,
    slippage: float = 0.001,
    monthly_sales_exemption: float = 20_000,
) -> "tuple[pd.DataFrame, dict, dict]":
    """
    Run all 8 core strategies and return their after-tax monthly return series.

    Parameters
    ----------
    db_path : str
        Path to b3_market_data.sqlite. Defaults to project root.
    start : str
        Backtest start date (ISO 8601).
    end : str
        Backtest end date. Defaults to today.
    capital : float
        Initial capital in BRL.
    tax_rate : float
        Capital gains tax rate (default 0.15 = 15%).
    slippage : float
        Round-trip transaction cost fraction (default 0.001 = 0.1%).
    monthly_sales_exemption : float
        Monthly sales exemption amount (BRL) under Brazilian CGT rules.
        R$20K means months where total sales <= 20K are tax-exempt.

    Returns
    -------
    returns_df : pd.DataFrame
        Columns = [strategy_names..., "IBOV", "CDI"]
        Index = monthly dates
        Values = after-tax simple monthly returns
    sim_results : dict
        {"strategy_name": simulation_result_dict, ...}
        Provides access to full equity curves (pretax_values, aftertax_values, etc.)
    regime_signals : dict
        {"is_easing": Series, "ibov_calm": Series, "ibov_uptrend": Series, "ibov_above": Series}
        Monthly boolean Series for regime-conditional downstream use.
    """
    if db_path is None:
        db_path = _DEFAULT_DB
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    cfg = dict(
        capital=capital,
        tax_rate=tax_rate,
        slippage=slippage,
        monthly_sales_exemption=monthly_sales_exemption,
    )

    # ── 1. Load shared data once ─────────────────────────────────────────────
    print("  [strategy_returns] Loading market data...")
    adj_close, close_px, fin_vol = load_b3_data(db_path, start, end)
    cdi_daily = download_cdi_daily(start, end)
    ibov_px = download_benchmark("^BVSP", start, end)

    ibov_ret = ibov_px.resample(_FREQ).last().pct_change().dropna()
    cdi_ret = (1 + cdi_daily).resample(_FREQ).prod() - 1
    cdi_monthly = cdi_ret.copy()

    px = adj_close.resample(_FREQ).last()
    ret = px.pct_change()
    raw_close = close_px.resample(_FREQ).last()
    adtv = fin_vol.resample(_FREQ).mean()

    log_ret = np.log1p(ret)
    has_glitch = ((ret > 1.0) | (ret < -0.90)).rolling(_LOOKBACK).max()

    # ── 2. Precompute regime signals ──────────────────────────────────────────
    # COPOM easing: CDI 3-month ago higher than 1-month ago
    is_easing = cdi_monthly.shift(1) < cdi_monthly.shift(4)

    # MA200 and above-MA200 flag
    ma200_daily = adj_close.rolling(200, min_periods=200).mean()
    ma200_m = ma200_daily.resample(_FREQ).last()
    above_ma200 = px > ma200_m
    dist_ma200 = px / ma200_m - 1

    # IBOV regime signals
    ibov_daily_ret = ibov_px.pct_change()
    ibov_vol_20d = ibov_daily_ret.rolling(20).std()
    ibov_vol_m = ibov_vol_20d.resample(_FREQ).last()
    ibov_vol_pctrank = ibov_vol_m.expanding(min_periods=12).apply(
        lambda x: (x.iloc[-1] >= x).mean(), raw=False
    )
    ibov_calm = ibov_vol_pctrank <= 0.70

    ibov_ret_20d = ibov_px.pct_change(20)
    ibov_ret_m = ibov_ret_20d.resample(_FREQ).last()
    ibov_uptrend = ibov_ret_m > 0

    ibov_m = ibov_px.resample(_FREQ).last()
    ibov_ma10 = ibov_m.rolling(10).mean()
    ibov_above = ibov_m > ibov_ma10

    # Multifactor composite (used by RegimeSwitching)
    mom_sig = log_ret.shift(1).rolling(_LOOKBACK).sum()
    mom_sig[has_glitch == 1] = np.nan
    vol_sig_mf = -ret.shift(1).rolling(_LOOKBACK).std()
    vol_sig_mf[has_glitch == 1] = np.nan
    mf_composite = (
        mom_sig.rank(axis=1, pct=True) * 0.5
        + vol_sig_mf.rank(axis=1, pct=True) * 0.5
    )

    # Additional signals for Res.MultiFactor
    vol_60d = ret.rolling(5).std()  # 5-month rolling window (monthly data)
    daily_ret_abs = adj_close.pct_change().abs()
    atr_proxy = daily_ret_abs.ewm(span=14, min_periods=14).mean()
    atr_m = atr_proxy.resample(_FREQ).last()
    vol_20d = ret.rolling(2).std()

    # ── 3. Package shared data ────────────────────────────────────────────────
    shared = dict(
        ret=ret,
        log_ret=log_ret,
        adtv=adtv,
        raw_close=raw_close,
        has_glitch=has_glitch,
        is_easing=is_easing,
        above_ma200=above_ma200,
        dist_ma200=dist_ma200,
        ma200_m=ma200_m,
        ibov_calm=ibov_calm,
        ibov_uptrend=ibov_uptrend,
        ibov_above=ibov_above,
        mf_composite=mf_composite,
        vol_60d=vol_60d,
        atr_m=atr_m,
        vol_20d=vol_20d,
        cdi_monthly=cdi_monthly,
        ibov_ret=ibov_ret,
        px=px,
        adj_close=adj_close,
        fin_vol=fin_vol,
    )

    regime_signals = dict(
        is_easing=is_easing,
        ibov_calm=ibov_calm,
        ibov_uptrend=ibov_uptrend,
        ibov_above=ibov_above,
    )

    # ── 4. Run all 8 strategies ───────────────────────────────────────────────
    strategy_runners = [
        ("CDI+MA200",       _run_cdi_ma200),
        ("Res.MultiFactor", _run_res_multifactor),
        ("RegimeSwitching", _run_regime_switching),
        ("COPOM Easing",    _run_copom_easing),
        ("MultiFactor",     _run_multifactor),
        ("SmallcapMom",     _run_smallcap_mom),   # NOTE: ADTV < R$1M
        ("LowVol",          _run_low_vol),
        ("MomSharpe",       _run_mom_sharpe),
    ]

    sim_results = {}
    monthly_returns = {}

    for name, runner in strategy_runners:
        print(f"  [strategy_returns] Running {name}...")
        try:
            result = runner(shared, cfg)
            sim_results[name] = result

            # Extract after-tax monthly returns
            at_val = result["aftertax_values"]
            at_ret = at_val.pct_change().fillna(0.0)
            monthly_returns[name] = at_ret

        except Exception as exc:
            print(f"    WARNING: {name} failed -- {exc}")
            import traceback
            traceback.print_exc()
            monthly_returns[name] = pd.Series(dtype=float)

    # ── 5. Add benchmarks ─────────────────────────────────────────────────────
    monthly_returns["IBOV"] = ibov_ret
    monthly_returns["CDI"] = cdi_ret

    # ── 6. Build aligned DataFrame ────────────────────────────────────────────
    # Align to common date index (intersection)
    returns_df = pd.DataFrame(monthly_returns)
    returns_df = returns_df.dropna(how="all")

    print(
        f"  [strategy_returns] Done. Shape={returns_df.shape}, "
        f"Columns={list(returns_df.columns)}"
    )

    return returns_df, sim_results, regime_signals


# ════════════════════════════════════════════════════════════════════════════
#  Quick test / demo
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import time

    t0 = time.time()
    print("\n" + "=" * 70)
    print("  build_strategy_returns() -- Quick Demo")
    print("=" * 70)

    returns_df, sim_results, regime_signals = build_strategy_returns()

    elapsed = time.time() - t0
    print(f"\nReturns DataFrame:")
    print(f"  Shape : {returns_df.shape}")
    print(f"  Columns: {list(returns_df.columns)}")
    print(f"  Index  : {returns_df.index[0]} to {returns_df.index[-1]}")
    print(f"\nSim results keys: {list(sim_results.keys())}")
    print(f"Regime signals  : {list(regime_signals.keys())}")
    print(f"\nElapsed: {elapsed:.1f}s")

    print("\nSmallcapMom note:")
    print(f"  {SMALLCAP_MOM_NOTE}")

    print("\nFirst 3 rows:")
    print(returns_df.head(3).to_string())

    print("\nLast 3 rows:")
    print(returns_df.tail(3).to_string())
