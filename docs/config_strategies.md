# Config-driven strategies (strategies as data)

Audience: humans **and** AI agents building/backtesting B3 portfolios.

A large class of strategies — "pick a factor, rank the liquid universe, hold the
top N, rebalance" and "hold a fixed set of sleeves and rebalance to weights" —
are now **data, not code**. You create one by dropping a YAML file in
`backtests/strategies/specs/`. No Python, no class, no registration step.

- Engines: `backtests/core/config_strategy.py`
- Loader: `backtests/core/spec_loader.py` (auto-discovers `specs/*.yaml`)
- The engines only build `target_weights`; the shared `run_simulation`
  (`backtests/core/simulation.py`) handles rebalancing, slippage and the
  Brazilian capital-gains tax engine. You never touch that.

---

## Quick start — create a portfolio in one file

Create `backtests/strategies/specs/my_momentum.yaml`:

```yaml
name: MyMomentum                      # unique; shown in the UI and registry
description: Top-decile 6-month momentum, monthly, equal weight.
kind: rank_and_hold
rebalance: ME                         # ME (monthly) | QE (quarterly) | W-FRI (weekly)
universe:
  min_adtv: 1000000                   # min avg daily traded value (BRL). NOTE: plain ints, no 1_000_000
  min_price: 1.0                      # penny-stock floor (BRL)
signal:
  factor: momentum                    # a named signal (see catalog below)
  lookback: 6
selection:
  top_pct: 0.10                       # hold the top 10% of the eligible universe
  min_names: 5                        # need >= this many candidates to invest
weighting: equal
```

That's it. It now appears in the registry and the Streamlit backtest runner:

```python
from backtests.core.strategy_registry import get_registry
get_registry().get("MyMomentum")      # a ready-to-run strategy instance

# Or end-to-end (loads data, runs the sim, computes metrics vs IBOV/CDI):
from ui.services.backtest_service import run_backtest
result = run_backtest("MyMomentum", get_registry().get("MyMomentum").get_default_parameters())
```

---

## Two `kind`s of strategy

### `rank_and_hold` — factor-ranked equity book

Per rebalance date: compute a factor (a wide date×ticker frame, **higher =
better**) → filter to the liquid universe → rank → take the top slice → weight →
hold until the next rebalance.

Full field reference:

```yaml
name: <str>                # required, unique
description: <str>
kind: rank_and_hold        # required
rebalance: ME              # ME | QE | W-FRI
needs_fundamentals: false  # set true to load CVM fundamentals (f_* shared keys)
warmup_pad: 1              # rows skipped at the start = max_lookback + warmup_pad
                           #   (use 2 when the signal has shift:1; see Gotchas)

universe:
  min_adtv: 1000000
  min_price: 1.0
  smallcap_below_median: false   # keep only below-median-ADTV names (small caps)

# --- pick ONE of `signal:` or `composite:` ---
signal:
  factor: low_vol          # named signal | shared_data key | store:<feature_id>
  lookback: 12             # periods; drives both the factor and the warmup
  shift: 0                 # extra lag in periods (signal-specific default)
  apply_glitch: true       # NaN-out flagged bad prints (has_glitch) before ranking
  range: [1.0, 30.0]       # optional: keep only signal values within [lo, hi]

composite:                 # rank-average several factors (each ranked cross-sectionally)
  - {factor: momentum, lookback: 12, shift: 1, weight: 0.5}
  - {factor: low_vol,  lookback: 12, shift: 1, weight: 0.5}

selection:
  top_pct: 0.10            # fraction of eligible universe... OR:
  top_n: 20               # ...a fixed number of names (top_n wins if both set)
  min_names: 5

weighting: equal           # equal | market_cap  (market_cap uses ADTV as a size proxy)

regime: null               # optional macro overlay — see below
```

### `fixed_weight` — constant-weight sleeve blend

Hold a fixed set of sleeves (Yahoo/B3 tickers + the special `CDI` asset) at
constant weights, rebalanced on the grid.

```yaml
name: DIVO / IVVB / CDI (40/40/20)
description: 40% dividend ETF, 40% S&P-in-BRL, 20% cash, quarterly.
kind: fixed_weight
rebalance: QE
park_in_cdi_until_live: true   # before an ETF lists, route its weight to CDI
park_mode: per_sleeve          # per_sleeve: park each dead ETF individually
                               # all_or_nothing: hold 100% CDI until *every* ETF is live
sleeves:
  - {ticker: DIVO11.SA, weight: 0.40}   # downloaded via download_benchmark (Yahoo)
  - {ticker: IVVB11.SA, weight: 0.40}
  - {asset: CDI,        weight: 0.20}   # the cash sleeve (shared cdi_monthly)
```

Weights need not sum to 1 (the simulator holds the rest in nothing), but for a
fully-invested blend make them sum to 1.

---

## Factor catalog — what `factor:` can reference

`resolve_factor` accepts three forms (all return a wide date×ticker frame,
higher = better):

1. **Named signal** from `SIGNAL_LIBRARY` (`config_strategy.py`):

   | name | meaning | notes |
   |------|---------|-------|
   | `momentum` | Σ log-returns over `lookback` | default `shift: 1` |
   | `low_vol` | −rolling std of returns over `lookback` | default `shift: 0` |
   | `sharpe_mom` | momentum ÷ vol (risk-adjusted), both `shift:1` | |
   | `blended_vol` | −(0.5·vol_60d + 0.5·vol_20d) | uses precomputed vol keys |

2. **A bare `shared_data` key** — any precomputed frame from
   `build_shared_data`, e.g. `mf_composite`, `f_pe_ratio_dyn`, `f_pb_ratio_dyn`.
   (Fundamentals `f_*` keys require `needs_fundamentals: true`.)

3. **`store:<feature_id>`** — a factor from the research IC factor store
   (`research/discovery`), e.g. `factor: store:ratio__High_low_range_20d__Win_rate_120d`.
   It's loaded from parquet and as-of-aligned to the rebalance calendar. Browse
   available IDs in `research/output/feature_catalog.json`.

Convention: signals are **higher = better** so selection is always "top". Express
a low-P/E strategy as an earnings-yield (higher = cheaper), a low-vol strategy as
negated volatility, etc.

---

## Regime overlay (rank_and_hold only)

Gate or scale the equity book by a macro regime, parking in `CDI_ASSET` when
"off". Inputs are boolean reads of `shared_data` flags at row `i - offset`;
rules are matched first-to-last.

```yaml
regime:
  add_ibov: true             # also expose IBOV as an available return column
  park_when_too_few: true    # if the filtered universe < min_names, park in CDI (not skip)
  inputs:
    easing:   {flag: is_easing, offset: 0}                 # offset 0 = read at row i
    stressed: {flag: ibov_vol_pctrank, offset: 1, gt: 0.70} # offset 1 = row i-1; True if value > 0.70
  rules:                     # first match wins
    - {easing: true,  stressed: false, top_pct: 0.15}      # override the selection top_pct
    - {easing: false, stressed: false, top_pct: 0.10}
    - {easing: true,  stressed: true,  top_pct: 0.05}
    - {default: true, park: true}                          # else: 100% CDI
```

A **binary gate** is just the degenerate case (one input, one invest rule, one
park default) — see `specs/regime_switching.yaml`. When a rule omits `top_pct`,
it invests using `selection.top_pct`.

Common regime flags in `shared_data`: `is_easing`, `above_ma200`, `ibov_above`,
`ibov_calm`, `ibov_uptrend`, `ibov_vol_pctrank`.

---

## Recipes (copy an existing spec)

| You want | Start from |
|----------|-----------|
| Single technical factor | `specs/low_vol.yaml` |
| Blend of factors (rank-average) | `specs/multifactor.yaml` |
| Small-cap tilt | `specs/smallcap_momentum.yaml` |
| Macro on/off gate | `specs/regime_switching.yaml` |
| Regime-scaled exposure (table) | `specs/adaptive_low_vol.yaml` |
| Fixed ETF/CDI blend | `specs/divo_ivvb_cdi_40_40_20.yaml` |
| Blend that waits for all ETFs | `specs/baseline_thirds.yaml` |

---

## Adding a new named signal (the only time you write Python)

If your factor isn't a `shared_data` key or store feature, add a one-liner to
`SIGNAL_LIBRARY` in `backtests/core/config_strategy.py`:

```python
def _sig_my_factor(shared: dict, cfg: dict) -> pd.DataFrame:
    lb = int(cfg.get("lookback", 12))
    return shared["log_ret"].shift(1).rolling(lb).sum() / shared["adtv"]  # higher = better

SIGNAL_LIBRARY = {
    ...,
    "my_factor": _sig_my_factor,
}
```

Then reference it: `signal: {factor: my_factor, lookback: 12}`.

---

## Testing / trusting a spec

- The permanent guard is `tests/test_config_strategy.py` (behaviour tests on
  synthetic data): `python -m pytest tests/test_config_strategy.py`.
- To sanity-check a new spec end-to-end on the real DB, run it through
  `run_backtest(name, params)` and inspect `result["pretax_values"]`.
- **Porting an existing Python strategy?** Prove *exact* parity before deleting
  the original: run both through `build_shared_data → generate_signals →
  run_simulation` at default params and assert the pre-tax **and** after-tax
  equity curves match (`np.allclose(..., atol=1e-6)`). Watch the parity traps in
  Gotchas below.

---

## Gotchas (these bite parity and correctness)

- **YAML ints have no underscores.** Write `1000000`, not `1_000_000` (YAML
  parses the latter as a string).
- **Warmup / shift.** The loop starts at `max_lookback + warmup_pad`
  (default pad 1). A signal with `shift: 1` consumes one extra row — use
  `warmup_pad: 2` to match a `range(lookback+2, ...)` original.
- **`apply_glitch`.** Defaults true (NaN-out flagged prints before ranking). Set
  `false` when the factor is already clean/pre-masked (e.g. `mf_composite`).
- **Parking modes differ.** `per_sleeve` parks each dead ETF's weight in CDI
  individually; `all_or_nothing` holds 100% CDI until every ETF is live.
- **Regime input offsets.** `offset: 0` reads the flag at the current row `i`;
  `offset: 1` reads `i-1`. A NaN flag is treated as `False`.
- **Regime `top_pct` overrides `selection.top_pct`.** If every rule sets its own
  `top_pct`, the `selection.top_pct` (and its UI slider) is inert.

---

## When NOT to use a spec (keep it as code)

The config engines deliberately cover the homogeneous cases. Write a normal
`StrategyBase` subclass in `backtests/strategies/` when the strategy needs any
of: per-row fundamental cross-sectional ranking, iterative/inverse-vol weighting,
index reconstruction (market-cap constituent selection), vol-targeting, sticky
selection, pure asset-switch rotation, or an embedded sub-strategy. Forcing those
into YAML makes a worse language, not less code. Existing examples kept as code:
`low_pe`, `value_quality`, `top_mcap`, `ibov_low_vol`, `qmv`, `mean_reversion`,
`sp500_b3`, `two_leg_value`, `cdi_ma200`, `copom_easing`, `dual_momentum`.
