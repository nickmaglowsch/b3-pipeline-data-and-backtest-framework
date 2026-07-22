# Config-driven strategies (strategies as data)

Audience: humans **and** AI agents building/backtesting B3 portfolios.

A large class of strategies — "pick a factor, rank the liquid universe, hold the
top N, rebalance" and "hold a fixed set of sleeves and rebalance to weights" —
are now **data, not code**. You create one by dropping a YAML file in
`backtests/strategies/specs/`. No Python, no class, no registration step.

- Engines: `backtests/core/config_strategy.py`
- Loader: `backtests/core/strategy_registry.py` (`discover()` auto-registers `specs/*.yaml`)
- The engines only build `target_weights`; the shared `run_simulation`
  (`backtests/core/simulation.py`) handles rebalancing, slippage and the
  Brazilian capital-gains tax engine. You never touch that.
- Periodic buy-ins (`contribution`, BRL/month) are a **run-level knob**, not a
  spec field — set it in the UI form or on the `run_simulation` call. Every spec
  gets it for free, and the new cash is allocated by the same rebalance logic
  (biggest gap to target gets the most). See `backtests/README.md`.

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

signal:
  # The factor is DATA. Reference a named signal from signals.yaml...
  factor: low_vol          # name in signals.yaml | shared_data key | store:<feature_id>
  lookback: 12             # periods; passed into the expression and drives the warmup
  apply_glitch: true       # NaN-out flagged bad prints (has_glitch) before ranking
  range: [1.0, 30.0]       # optional: keep only signal values within [lo, hi]
  # ...or inline an expression directly (a "composite" is just an expr with rank()):
  # expr: "0.5*rank(mask_glitch(roll_sum(shift(log_ret,1),lookback)))
  #        + 0.5*rank(mask_glitch(-roll_std(shift(ret,1),lookback)))"
  # apply_glitch: false    # set false when the expr masks glitches itself

selection:
  top_pct: 0.10            # fraction of eligible universe... OR:
  top_n: 20               # ...a fixed number of names (top_n wins if both set)
  min_names: 5             # need >= this many candidates to invest at all
  min_hold: 5              # floor on the number held (default: min_names). Split
                           #   them when a strategy invests on 1 survivor but
                           #   always holds >= 5 (see win_rate_mean_rev.yaml).
  dedup_roots: false       # collapse share classes (PETR3/PETR4) to the most
                           #   liquid one BEFORE ranking. Required for any
                           #   fundamentals factor — fundamentals are per-company,
                           #   so both classes carry the same score and a top-N
                           #   would double-book the company.
  sticky: false            # when < min_names qualify, hold the previous book
                           #   instead of going flat (volume_breakout).

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

## Signals are data — the expression language

A factor is a **string expression** (the signal DSL, `backtests/core/signal_dsl.py`)
that evaluates to a wide date×ticker frame, **higher = better**. You never write
Python to add a factor.

`resolve_factor` accepts, in precedence order:

1. An inline `expr:` in the spec's `signal:` block.
2. A **named signal** — a key in `backtests/strategies/signals.yaml`, which is
   itself just an `expr` + default params (this replaced the old code library):

   ```yaml
   momentum:   {expr: "roll_sum(shift(log_ret, 1), lookback)", lookback: 12}
   low_vol:    {expr: "-roll_std(ret, lookback)", lookback: 12}
   sharpe_mom: {expr: "roll_sum(shift(log_ret,1),lookback) / roll_std(shift(ret,1),lookback)", lookback: 12}
   blended_vol:{expr: "-(0.5*vol_60d + 0.5*vol_20d)"}
   ```

3. `store:<feature_id>` — a factor from the research IC store (`research/discovery`),
   e.g. `factor: store:ratio__High_low_range_20d__Win_rate_120d`. Loaded from
   parquet and as-of-aligned to the rebalance calendar. IDs live in
   `research/output/feature_catalog.json`. Reference-only (`factor:`); string
   constants are rejected inside an `expr:`.
4. A bare **`shared_data` key** — any precomputed frame (`mf_composite`,
   `f_pe_ratio_dyn`, …). Fundamentals `f_*` keys need `needs_fundamentals: true`.

### Primitive vocabulary (what an `expr` may call)

Names resolve to `shared_data` frames (`ret`, `log_ret`, `adj_close`, `raw_close`,
`adtv`, `close_px`, `fin_vol`, `vol_60d`, `vol_20d`, `atr_m`, `mf_composite`,
`f_*` …) or numeric scalars — `lookback`, plus **any extra numeric key you put in
the `signal:` block** (`mom_weight`, `max_pb`, … ; a run-time param of the same
name overrides it).

Operators: `+ - * / **`, unary `-`, the comparisons `> < >= <= == !=` and the
elementwise boolean `&` / `|`. Comparisons yield boolean frames — feed them to
`where(...)`. **Parenthesise**: `&` binds tighter than `>` in Python, so write
`(a > 0) & (b < 1)`, never `a > 0 & b < 1`. Chained comparisons (`0 < x < 1`)
are rejected.

| group | functions |
|-------|-----------|
| time-series | `shift(x,n)`, `roll_sum/mean/std/var/min/max/median/skew/kurt(x,w)`, `pct_change(x,w)`, `diff(x,n)`, `ewm_mean(x,span)`, `ewm_std(x,span)` |
| elementwise | `log(x)`, `sign(x)`, `abs(x)`, `clip(x,lo,hi)` |
| cross-sectional | `rank(x)` (pct, per date), `zscore(x)`, `demean(x)` |
| masking | `mask_glitch(x)` (NaN-out `has_glitch` cells), `where(x, cond)` (keep where `cond`, else NaN) |
| combining | `nan_add(a, b)` (sum two legs; NaN only where **both** are NaN) |
| calendar | `at_rebalance(x)` (daily frame → the rebalance grid: last obs per period, columns aligned to the traded universe) |

The evaluator is **safe**: expressions are parsed with `ast` and only this
whitelist runs — no attribute access, subscripts, lambdas, or arbitrary calls,
so a spec can never execute code.

Three of these exist for a specific reason and are easy to get wrong:

- **`where(x, cond)`** is how a strategy carries its *own* glitch band or gate
  rather than the shared `has_glitch`, e.g.
  `where(-roll_sum(ret, lookback), (ret <= 1.0) & (ret >= -0.9))`. Set
  `apply_glitch: false` when you do this, or you get both masks.
- **`nan_add(a, b)`** — use it whenever two legs have *different coverage*. A
  plain `a + b` is NaN wherever either side is NaN, which is how ValueQuality
  once produced zero trades for its whole history (P/B and ROE come from
  different tables). `nan_add` = `pandas.DataFrame.add(fill_value=0)`.
- **`at_rebalance(x)`** lets an expression mix a daily-derived factor with the
  monthly frames (`win_rate_mean_rev` builds a 20d win rate off daily
  `adj_close`). Without it the frames misalign silently.

Convention: **higher = better**. Express low-P/E as an earnings-yield, low-vol as
negated volatility, etc. A "composite" is just an expr:
`0.5*rank(momentum_expr) + 0.5*rank(lowvol_expr)`.

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

An **N-of-M vote** — "invest when at least 2 of these 3 signals are on" — is
`min_true` on the rule, instead of enumerating every winning combination
(`specs/research_multifactor.yaml`):

```yaml
  rules:
    - {min_true: 2}                 # >= 2 of the inputs above are True -> invest
    - {default: true, park: true}
```

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
| N-of-M regime vote | `specs/research_multifactor.yaml` |
| Hold *every* qualifying name (no top slice) | `specs/cdi_ma200.yaml` |
| Fundamentals ranker (needs `dedup_roots`) | `specs/low_pe.yaml`, `specs/value_quality.yaml` |
| Band filter on the factor value | `specs/low_pe.yaml` (`range:`) |
| Daily-derived factor on a monthly grid | `signals.yaml: win_rate_mean_rev` |
| Fixed ETF/CDI blend | `specs/divo_ivvb_cdi_40_40_20.yaml` |
| Blend that waits for all ETFs | `specs/baseline_thirds.yaml` |

---

## Adding a new signal (no Python)

Add a line to `backtests/strategies/signals.yaml`:

```yaml
my_factor:
  expr: "roll_sum(shift(log_ret, 1), lookback) / adtv"   # higher = better
  lookback: 12
```

Then reference it: `signal: {factor: my_factor, lookback: 12}` — or skip the
name and inline the same `expr:` directly in the strategy spec.

You only touch Python for a genuinely new *primitive* (a rolling/cross-sectional
op that can't be composed from the vocabulary above) — add it to the function
table in `backtests/core/signal_dsl.py`. That's rare; factors are data.

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
- **Warmup / shift.** The loop starts at `signal.lookback + warmup_pad`
  (default pad 1). An expression that shifts one extra row (`shift(...,1)`)
  consumes a row — use `warmup_pad: 2` to match a `range(lookback+2, ...)` original.
- **`apply_glitch`.** Defaults true (NaN-out flagged prints before ranking). Set
  `false` when the factor is already clean/pre-masked (`mf_composite`) or when the
  `expr` masks each leg itself with `mask_glitch(...)` (see `multifactor.yaml`).
- **`rank()` uses pandas semantics** (`rank(axis=1, pct=True)`), not the Rust
  cross-sectional op, so folded composites stay bit-identical to the old ones.
- **Parking modes differ.** `per_sleeve` parks each dead ETF's weight in CDI
  individually; `all_or_nothing` holds 100% CDI until every ETF is live.
- **Regime input offsets.** `offset: 0` reads the flag at the current row `i`;
  `offset: 1` reads `i-1`. A NaN flag is treated as `False`.
- **Regime `top_pct` overrides `selection.top_pct`.** If every rule sets its own
  `top_pct`, the `selection.top_pct` (and its UI slider) is inert.
- **`top_pct: 1.0` on a rule = hold everything that qualifies**, no top slice —
  that is how `cdi_ma200` holds every name above its MA200. Put it on the rule
  (not in `selection`) so the 0.01–0.50 UI slider is suppressed.
- **A `range:` filter is on the *signal* value**, after the liquidity mask. A
  low-P/E band `[1, 30]` becomes an earnings-yield range `[1/30, 1]`.
- **Fundamentals frames are wider than the traded universe** (`f_pe_ratio_dyn`
  spans every share class, ~15% more columns than `ret`). The engine reindexes
  the signal row onto the universe, but `rank()` inside your `expr` still ranks
  over the *whole* frame — which is what the original classes did.
- **`min_names` vs `min_hold`.** `min_names` gates whether to invest at all;
  `min_hold` floors how many to hold. They default to the same value; split them
  only when a strategy really does both (`win_rate_mean_rev`).

---

## When NOT to use a spec (keep it as code)

The config engines deliberately cover the homogeneous cases. Write a normal
`StrategyBase` subclass in `backtests/strategies/` when the strategy needs any
of: iterative or inverse-vol weighting, index reconstruction (market-cap
constituent selection), vol-targeting, long/short books, a rolling-IC stability
guard, pure asset-switch rotation, or an embedded sub-strategy. Forcing those
into YAML makes a worse language, not less code. Examples kept as code:
`top_mcap`, `ibov_low_vol`, `qmv`, `mean_reversion` (MeanRevComposite),
`sp500_b3`, `two_leg_value`, `copom_easing`, `dual_momentum`, `sma_momentum_tilt`.

Fundamental cross-sectional ranking and sticky selection are *no longer* on that
list — `low_pe`, `value_quality`, `cdi_ma200`, `frog_in_pan`, `volume_breakout`,
`research_multifactor`, `win_rate_mean_reversion` and the simple mean-reversion
half of `mean_reversion` are all specs now, each proven bit-identical to the
class it replaced before deletion.
