"""
B3 HMM Regime-Switching Strategy
=======================================================================
Strategy: Uses a 2-state Gaussian Hidden Markov Model fitted on market-wide
features (IBOV returns, rolling volatility, market breadth) to classify
each month into one of two regimes:

  - RISK_ON : Favorable conditions → go long top momentum stocks
  - RISK_OFF: Adverse conditions   → park in CDI (risk-free rate)

The HMM is fitted with a 36-month rolling window (minimum 24 months of
history required) and re-fit every month. Shorter windows capture B3's
high volatility and frequent regime shifts. Only past data is used at
each rebalance date — no lookahead bias.

A synthetic "_CDI" column is injected into the returns matrix so the
simulation framework naturally earns CDI when the strategy goes to cash.
"""
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from datetime import datetime
from hmmlearn.hmm import GaussianHMM
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.data import load_b3_data, download_benchmark, download_cdi_daily
from core.metrics import build_metrics, value_to_ret, display_metrics_table
from core.plotting import plot_tax_backtest
from core.simulation import run_simulation

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
REBALANCE_FREQ = "ME"
LOOKBACK_YEARS = 1

period_map = {"ME": 12, "QE": 4, "YE": 1, "W": 52, "W-FRI": 52, "W-MON": 52}
PERIODS_PER_YEAR = period_map.get(REBALANCE_FREQ, 12)
LOOKBACK_PERIODS = int(LOOKBACK_YEARS * PERIODS_PER_YEAR)
SKIP_PERIODS = 1 if REBALANCE_FREQ == "ME" else 0

N_HMM_STATES = 2
HMM_MIN_HISTORY = 24          # Minimum months before first HMM fit
HMM_REFIT_EVERY = 1           # Re-fit HMM every month (B3 is volatile)
HMM_WINDOW = 36               # Rolling window size (months) — shorter for B3
PORTFOLIO_SIZE_PCT = 0.10     # Top 10% of eligible stocks
MIN_PORTFOLIO_SIZE = 10       # Never hold fewer than 10 stocks
TAX_RATE = 0.15
SLIPPAGE = 0.001
START_DATE = "2003-01-01"
END_DATE = datetime.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 100_000
MIN_ADTV = 1_000_000

CDI_TICKER = "_CDI"           # Synthetic ticker for CDI returns

DB_PATH = "b3_market_data.sqlite"
IBOV_INDEX = "^BVSP"


# ─────────────────────────────────────────────
#  HMM HELPERS
# ─────────────────────────────────────────────
def build_market_features(ibov_ret_monthly, breadth_monthly=None):
    """Build feature matrix for HMM from IBOV monthly returns + breadth.

    Features (all computed from past data only):
      1. Monthly return
      2. Rolling 3-month volatility
      3. Market breadth (fraction of stocks with positive monthly return)
    """
    df = pd.DataFrame(index=ibov_ret_monthly.index)
    df["ret"] = ibov_ret_monthly.values
    df["vol_3m"] = ibov_ret_monthly.rolling(3).std()
    if breadth_monthly is not None:
        common = df.index.intersection(breadth_monthly.index)
        df = df.loc[common]
        df["breadth"] = breadth_monthly.loc[common].values
    return df.dropna()


def label_states(model, features):
    """Assign semantic labels (RISK_ON/RISK_OFF) to HMM hidden states
    based on the fitted state means for the return feature."""
    means = model.means_[:, 0]  # mean of the 'ret' feature per state
    order = np.argsort(means)   # lowest → highest mean return
    if len(order) == 2:
        label_map = {order[0]: "RISK_OFF", order[1]: "RISK_ON"}
    else:
        label_map = {order[0]: "RISK_OFF", order[1]: "RISK_OFF", order[2]: "RISK_ON"}
    return label_map


def fit_hmm_and_predict(features_df, n_states=2):
    """Fit GaussianHMM with multiple random seeds, pick best by log-likelihood.
    Features are z-scored internally for numerical stability; the scaler
    params are returned so we can transform new data consistently.
    """
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X = scaler.fit_transform(features_df.values)

    best_model = None
    best_score = -np.inf

    import logging
    logging.getLogger("hmmlearn").setLevel(logging.ERROR)

    for seed in range(10):
        try:
            m = GaussianHMM(
                n_components=n_states,
                covariance_type="full",
                n_iter=500,
                random_state=seed,
                verbose=False,
            )
            m.fit(X)
            score = m.score(X)
            if score > best_score:
                best_score = score
                best_model = m
        except Exception:
            continue

    if best_model is None:
        raise RuntimeError("HMM fitting failed on all seeds")

    states = best_model.predict(X)
    label_map = label_states(best_model, features_df)
    current_regime = label_map[states[-1]]
    return current_regime, best_model, states, label_map, scaler


# ─────────────────────────────────────────────
#  SIGNAL GENERATION
# ─────────────────────────────────────────────
def generate_signals(adj_close, close_px, fin_vol, ibov_px, cdi_monthly):
    px = adj_close.resample(REBALANCE_FREQ).last()
    ret = px.pct_change()

    raw_close = close_px.resample(REBALANCE_FREQ).last()
    adtv = fin_vol.resample(REBALANCE_FREQ).mean()

    # ── Inject synthetic CDI column into the returns matrix ──
    ret[CDI_TICKER] = cdi_monthly.reindex(ret.index).fillna(0.0)

    # ── Momentum signal (12-month log returns) ──
    log_ret = np.log1p(ret.drop(columns=[CDI_TICKER]))
    mom_signal = log_ret.shift(SKIP_PERIODS).rolling(LOOKBACK_PERIODS).sum()
    mom_glitch = ((ret.drop(columns=[CDI_TICKER]) > 1.0) | (ret.drop(columns=[CDI_TICKER]) < -0.90)).shift(SKIP_PERIODS).rolling(LOOKBACK_PERIODS).max()
    mom_signal[mom_glitch == 1] = np.nan

    # ── Market breadth: fraction of stocks with positive monthly return ──
    stock_ret = ret.drop(columns=[CDI_TICKER])
    breadth = (stock_ret > 0).sum(axis=1) / (stock_ret.notna().sum(axis=1) + 1e-6)

    # ── IBOV monthly returns for HMM features ──
    ibov_monthly = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    features = build_market_features(ibov_monthly, breadth)

    target_weights = pd.DataFrame(0.0, index=ret.index, columns=ret.columns)
    start_idx = LOOKBACK_PERIODS + SKIP_PERIODS + 1

    regime_log = {}
    last_regime = "RISK_OFF"
    last_model = None
    last_scaler = None
    months_since_fit = HMM_REFIT_EVERY  # force fit on first eligible month
    prev_sel_weights = {}

    for i in range(start_idx, len(ret)):
        date = ret.index[i]

        # ── Determine regime via HMM ──
        past_features = features.loc[features.index < date]

        if len(past_features) >= HMM_MIN_HISTORY:
            months_since_fit += 1
            # Use rolling window for fitting
            window_features = past_features.iloc[-HMM_WINDOW:]
            if months_since_fit >= HMM_REFIT_EVERY or last_model is None:
                try:
                    regime, last_model, _, _, last_scaler = fit_hmm_and_predict(
                        window_features, N_HMM_STATES
                    )
                    last_regime = regime
                    months_since_fit = 0
                except Exception:
                    pass  # keep last_regime
            else:
                # Use existing model + scaler to predict current state
                try:
                    X = last_scaler.transform(window_features.values)
                    states = last_model.predict(X)
                    lmap = label_states(last_model, window_features)
                    last_regime = lmap[states[-1]]
                except Exception:
                    pass

        regime_log[date] = last_regime

        # ── RISK_OFF → 100% CDI ──
        if last_regime == "RISK_OFF":
            target_weights.iloc[i, target_weights.columns.get_loc(CDI_TICKER)] = 1.0
            prev_sel_weights = {CDI_TICKER: 1.0}
            continue

        # ── RISK_ON → top momentum stocks ──
        signal_row = mom_signal.iloc[i - 1]

        adtv_row = adtv.iloc[i - 1]
        raw_close_row = raw_close.iloc[i - 1]

        valid_mask = (adtv_row >= MIN_ADTV) & (raw_close_row >= 1.0)
        valid = signal_row[valid_mask].dropna()

        if len(valid) < MIN_PORTFOLIO_SIZE:
            # Keep previous weights
            for t, w in prev_sel_weights.items():
                target_weights.iloc[i, target_weights.columns.get_loc(t)] = w
            continue

        n_sel = max(MIN_PORTFOLIO_SIZE, int(len(valid) * PORTFOLIO_SIZE_PCT))
        selected = valid.nlargest(n_sel).index

        weight = 1.0 / len(selected)
        prev_sel_weights = {}
        for t in selected:
            target_weights.iloc[i, target_weights.columns.get_loc(t)] = weight
            prev_sel_weights[t] = weight

    regime_series = pd.Series(regime_log, name="regime")
    return ret, target_weights, regime_series


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "=" * 70)
    print("  B3 HMM REGIME-SWITCHING STRATEGY")
    print("=" * 70)

    adj_close, close_px, fin_vol = load_b3_data(DB_PATH, START_DATE, END_DATE)

    cdi_daily = download_cdi_daily(START_DATE, END_DATE)
    ibov_px = download_benchmark(IBOV_INDEX, START_DATE, END_DATE)

    ibov_ret = ibov_px.resample(REBALANCE_FREQ).last().pct_change().dropna()
    ibov_ret.name = "IBOV"

    cdi_ret = (1 + cdi_daily).resample(REBALANCE_FREQ).prod() - 1

    print("\n  Fitting HMM & generating target weights...")
    ret, target_weights, regimes = generate_signals(
        adj_close, close_px, fin_vol, ibov_px, cdi_ret
    )

    # ── Print regime summary ──
    regime_counts = regimes.value_counts()
    total = len(regimes)
    print(f"\n  Regime breakdown ({total} months):")
    for r in ["RISK_ON", "RISK_OFF"]:
        cnt = regime_counts.get(r, 0)
        print(f"    {r:8s}: {cnt:4d} months ({100*cnt/total:5.1f}%)")

    ret = ret.fillna(0.0)

    print(f"\n  Running simulation ({REBALANCE_FREQ})...")
    result = run_simulation(
        returns_matrix=ret,
        target_weights=target_weights,
        initial_capital=INITIAL_CAPITAL,
        tax_rate=TAX_RATE,
        slippage=SLIPPAGE,
        name="HMM Regime",
    )

    common = result["pretax_values"].index.intersection(ibov_ret.index)
    pretax_val = result["pretax_values"].loc[common]
    aftertax_val = result["aftertax_values"].loc[common]
    ibov_ret = ibov_ret.loc[common]
    cdi_ret = cdi_ret.loc[common]

    pretax_ret = value_to_ret(pretax_val)
    aftertax_ret = value_to_ret(aftertax_val)
    total_tax = result["tax_paid"].sum()

    m_pretax = build_metrics(pretax_ret, "HMM Pre-Tax", PERIODS_PER_YEAR)
    m_aftertax = build_metrics(aftertax_ret, "After-Tax 15% CGT", PERIODS_PER_YEAR)
    m_ibov = build_metrics(ibov_ret, "IBOV", PERIODS_PER_YEAR)
    m_cdi = build_metrics(cdi_ret, "CDI", PERIODS_PER_YEAR)

    display_metrics_table([m_pretax, m_aftertax, m_ibov, m_cdi])

    plot_tax_backtest(
        title=(
            f"HMM Regime-Switching: 2-State (Risk-On\u2192Momentum, Risk-Off\u2192CDI)\n"
            f"R$ {MIN_ADTV/1_000_000:.0f}M+ ADTV  \u00b7  15% CGT + {SLIPPAGE*100}% Slippage\n"
            f"{START_DATE[:4]}\u2013{END_DATE[:4]}"
        ),
        pretax_val=pretax_val,
        aftertax_val=aftertax_val,
        ibov_ret=ibov_ret,
        tax_paid=result["tax_paid"].loc[common],
        loss_cf=result["loss_carryforward"].loc[common],
        turnover=result["turnover"].loc[common],
        metrics=[m_pretax, m_aftertax, m_ibov, m_cdi],
        total_tax_brl=total_tax,
        out_path="hmm_regime_backtest.png",
        cdi_ret=cdi_ret,
    )


if __name__ == "__main__":
    main()
