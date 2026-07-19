# B3 Quant Research Platform

An end-to-end Python platform for Brazilian equity research: it **downloads and
cleans** historical B3 market data, **enriches** it with CVM fundamentals,
**backtests** strategies against a tax-accurate simulator, **discovers** alpha
features automatically, and drives all of it from a **Streamlit UI**.

It started life as a COTAHIST price scraper. It is now five layers:

| Layer | Where | What it does |
|-------|-------|--------------|
| **Data pipeline** | `b3_pipeline/` | Prices, corporate actions, and fundamentals into one SQLite DB |
| **Backtesting framework** | `backtests/` | Tax-aware simulator + config/DSL-driven strategies |
| **Feature discovery** | `research/discovery/` | Auto-generates & IC-ranks hundreds of alpha features |
| **ML research** | `research/` | Feature-importance study (RandomForest / XGBoost) |
| **Web UI** | `ui/` | Streamlit app to run every layer from the browser |

All data lands in a single SQLite file (`b3_market_data.sqlite`) with
split/dividend-adjusted prices (Yahoo-style `adj_close`), point-in-time
fundamentals, and ISIN-linked identity so the dataset is **survivorship-bias
free** across ticker changes, mergers, and delistings.

## Data sources & provenance

Earlier versions claimed "B3 is the single source of truth." That is no longer
true — the platform stitches together several official feeds:

| Source | Feed | Provides |
|--------|------|----------|
| **B3** | COTAHIST annual files | Daily OHLC + volume (1994→present) |
| **B3** | `listedCompaniesProxy` API | Dividends, JCP, splits, reverse splits, bonus shares |
| **CVM** | `dados.cvm.gov.br` bulk CSVs (DFP/ITR/FCA) | Structured fundamentals (2010→present) + PIT ticker map |
| **CVM** | "Download Múltiplo" legacy channel | Pre-2010 fundamentals (~2006–2009), *opt-in, credentialed* |
| **BCB** | Central Bank API | CDI risk-free daily series |
| **Yahoo Finance** | `yfinance` | IBOV, ETF sleeves (IVVB11, DIVO11, …) for benchmarks/blends |

Splits use B3's **official factors** (not heuristic jump detection), so
100:1 splits, `0.01` reverse-split factors, and `Bonificação` bonus shares are
all handled exactly.

## Installation

Requirements: **Python 3.9+** and the **Rust stable toolchain** (for the parser
extension, see below).

```bash
git clone <repository_url>
cd b3-pipeline-data-and-backtest-framework

python -m venv .venv
source .venv/bin/activate        # Linux/macOS  (.venv\Scripts\activate on Windows)

pip install -r requirements.txt  # pipeline + backtests + ML + Streamlit UI
make dev-rust                    # build the Rust COTAHIST parser into the venv (one-time)
```

`requirements.txt` covers everything. `make dev-rust` must be run once after
cloning (and again whenever `b3_pipeline_rs/src/` changes).

### Rust extension (COTAHIST parser)

The COTAHIST fixed-width parsing loop is a compiled Rust extension
(`cotahist_rs`, `pyo3` + `maturin`, `rayon` for parallelism). Parsing ~33 years
of data takes **~100 ms** instead of ~60 s.

```bash
# one-time toolchain install
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh && source $HOME/.cargo/env

make dev-rust     # debug build into the active venv (run after clone & on src changes)
make build-rust   # optimised release wheel -> b3_pipeline_rs/target/wheels/
make test-rust    # cargo test
make test         # pytest (Python suite)
make all          # dev-rust + test
```

If you see `ImportError: The cotahist_rs Rust extension is not compiled`, run
`make dev-rust`.

## Running the data pipelines

There are three independent entry points. All write to the same SQLite DB.

### 1. Prices + corporate actions — `b3_pipeline.main`

```bash
python -m b3_pipeline.main                     # download missing years, update DB
python -m b3_pipeline.main --rebuild           # drop tables & recompute adjustments
python -m b3_pipeline.main --year 2024         # a single year (testing)
python -m b3_pipeline.main --start-year 2010   # from a year onward
python -m b3_pipeline.main --skip-corporate-actions   # faster; prices only
python -m b3_pipeline.main --retry-failures    # re-fetch only companies in fetch_failures
python -m b3_pipeline.main --update-cnpj-map   # just refresh ticker→CNPJ map (fast)
```

Pipeline steps: download COTAHIST → parse (Rust) → upsert raw prices → fetch B3
corporate actions → compute split factors → compute dividend factors → write
`split_adj_*` and total-return `adj_close`. It is **idempotent**
(`INSERT OR REPLACE`, existing-file checks) and resilient to rate limits and
partial years.

### 2. CVM fundamentals 2010+ — `b3_pipeline.cvm_main`

```bash
python -m b3_pipeline.cvm_main                       # all available years
python -m b3_pipeline.cvm_main --start-year 2020     # from 2020 onward
python -m b3_pipeline.cvm_main --rebuild             # rebuild fundamentals tables
python -m b3_pipeline.cvm_main --skip-ticker-fetch   # skip B3 ticker-map fetch
python -m b3_pipeline.cvm_main --include-historical  # + FCA/CAD/IPE metadata (register, PIT tickers)
```

Ingests structured DFP/ITR filings into **point-in-time** `fundamentals_pit`
(keyed by filing, so backtests can query "what was known on date X" with no
look-ahead), then materialises a monthly snapshot (`fundamentals_monthly`).
Valuation ratios (P/E, P/B, EV/EBITDA) are computed **dynamically at query
time**, not stored. `--include-historical` adds the CAD company register
(listing/delisting dates) and FCA point-in-time ticker map — which closes the
survivorship hole for delisted names — but does **not** provide pre-2010
financials.

### 3. Pre-2010 fundamentals ("Download Múltiplo") — `b3_pipeline.dm_main`

CVM's bulk CSVs only reach back to 2010. `dm_main` extends fundamentals to
~2006–2009 via CVM's older credentialed delivery channel. It requires
free CVM credentials (`CVM_DM_USER` / `CVM_DM_PASS`) and is **not yet
integration-tested against live files** — see
**[docs/download_multiplo.md](docs/download_multiplo.md)** for the full protocol,
registration steps, DBF layout, and known limitations.

```bash
export CVM_DM_USER=... CVM_DM_PASS=...
python -m b3_pipeline.dm_main --start 2006-01-01 --end 2010-12-31 --types ITR,DFP,IAN
python -m b3_pipeline.dm_main --parse-only    # re-parse files already in data/dm/
```

## Database schema

`b3_market_data.sqlite` holds 12 tables. Identity is **ISIN-linked** so ticker
renames don't fragment a company's history.

**Market data**

| Table | Key | Contents |
|-------|-----|----------|
| `prices` | `(ticker, date)` | Raw OHLCV + `split_adj_*` + total-return `adj_close`; carries `isin_code` |
| `corporate_actions` | `(isin_code, event_date, event_type)` | Cash dividends, JCP, and stock events (`value`, `factor`, `source`) |
| `stock_actions` | `(isin_code, ex_date, action_type)` | Splits / reverse splits / bonus shares with factors |
| `detected_splits` | `(isin_code, ex_date)` | Legacy split factors derived from `stock_actions` |
| `skipped_events` | `(isin_code, event_date, label)` | Corporate events skipped during adjustment, with reason |
| `fetch_failures` | `(company_code, endpoint)` | Failed API fetches for `--retry-failures` |

**Fundamentals & identity**

| Table | Key | Contents |
|-------|-----|----------|
| `fundamentals_pit` | `filing_id` | Point-in-time financials per filing (revenue, net income, EBITDA, assets, equity, net debt, shares, TTM net income) |
| `fundamentals_monthly` | `(month_end, ticker)` | Monthly fundamentals snapshot for fast backtest joins |
| `cvm_filings` | `filing_id` | Filing metadata (doc type, period end, filing date, version) |
| `cvm_companies` | `cnpj` | CNPJ ↔ ticker, CVM code, listing/delisting dates |
| `company_isin_map` | `(cnpj, isin_code)` | CNPJ ↔ ISIN ↔ ticker ↔ share class |
| `company_tickers_pit` | `(cnpj, ticker, start_date)` | Point-in-time ticker map from CVM FCA (covers delisted names) |

All CVM financial values are stored in **thousands of BRL**, as reported.

### Example query — a clean, Yahoo-style series

```sql
SELECT date, ticker, open, high, low, close AS raw_close, adj_close
FROM prices
WHERE ticker = 'PETR4'
ORDER BY date DESC
LIMIT 10;
```

### B3 corporate-action labels & split factors

B3's Portuguese labels map to standard event types, and localised factor strings
(e.g. `"100,00000000000"`) are converted to adjustment multipliers:

| B3 Label | Event Type | Adjustment factor |
|----------|------------|-------------------|
| DIVIDENDO / RENDIMENTO | CASH_DIVIDEND | via total-return `adj_close` |
| JRS CAP PROPRIO | JCP | via total-return `adj_close` |
| DESDOBRAMENTO (split, factor > 1) | STOCK_SPLIT | `1 / factor` (e.g. 100:1 → 0.01) |
| GRUPAMENTO (reverse, factor < 1) | REVERSE_SPLIT | `1 / factor` (e.g. 0.01 → 100) |
| BONIFICACAO (bonus %) | BONUS_SHARES | `1 / (1 + pct/100)` (e.g. 33.33% → ≈0.75) |

## Backtesting framework

`backtests/` is a modular framework built on the pipeline DB. It uses
fully-adjusted `adj_close`, computes daily financial volume natively from
COTAHIST, pulls CDI (BCB) and IBOV/ETFs (Yahoo), and — critically — implements
the **exact mechanics of Brazilian capital-gains tax** for variable income
(15% CGT, infinite loss carryforward, the R$20k/month sale exemption, slippage,
and margin borrow at CDI + spread). The result is a realistic **after-tax**
equity curve, not a gross one. See **[backtests/README.md](backtests/README.md)**.

### Strategies are data — config + a signal DSL

**Most strategies need no Python.** A large class ("rank a factor, hold the top
N, rebalance" and "hold fixed sleeves and rebalance to weights") is defined by
dropping a YAML file in `backtests/strategies/specs/`. Factors themselves are
**string expressions** in a safe DSL (`backtests/core/signal_dsl.py`, parsed
with `ast` — no code execution) evaluated over shared data frames.

```yaml
# backtests/strategies/specs/my_momentum.yaml
name: MyMomentum
kind: rank_and_hold
rebalance: ME                 # ME | QE | W-FRI
universe: {min_adtv: 1000000, min_price: 1.0}
signal:   {factor: momentum, lookback: 6}   # a named signal from signals.yaml
selection: {top_pct: 0.10, min_names: 5}
weighting: equal
```

It auto-registers and appears in the UI's backtest runner. Named signals live in
`backtests/strategies/signals.yaml` (also just `expr` + defaults); research-discovered
features are referenceable as `store:<feature_id>`. Full field reference, the
expression vocabulary, regime overlays, and parity-porting gotchas are in
**[docs/config_strategies.md](docs/config_strategies.md)** — the guide for
humans *and* AI agents.

Write a `StrategyBase` subclass in `backtests/strategies/` only for genuinely
bespoke logic the config engines don't cover (per-row fundamental ranking,
inverse-vol weighting, index reconstruction, vol-targeting, asset rotation):

```python
from backtests.core.strategy_base import StrategyBase, ParameterSpec, COMMON_START_DATE, COMMON_END_DATE

class MyStrategy(StrategyBase):
    name = "My Strategy"
    def get_parameter_specs(self): return [COMMON_START_DATE, COMMON_END_DATE]
    def generate_signals(self, shared_data, params):
        ...  # return returns_matrix, target_weights
```

Both paths flow through one registry (`get_registry()`, auto-discovered) and the
shared `run_simulation`. **28 strategies** are registered today — factor books
(momentum, low-vol, Sharpe-momentum, multifactor, value/quality, low-P/E,
smallcap momentum, frog-in-the-pan, volume breakout, mean reversion), regime
overlays (COPOM easing, IBOV trend/vol, dual momentum), index-style books, and
fixed ETF/CDI blends.

### Core modules (`backtests/core/`)

| Module | Description |
|--------|-------------|
| `data.py` | Loads B3 data from SQLite; downloads CDI (BCB) and IBOV/ETFs (Yahoo) |
| `shared_data.py` | Builds the shared frame bundle (`ret`, `log_ret`, `adj_close`, `adtv`, `f_*`, regime flags…) consumed by every strategy |
| `simulation.py` | Tax-aware multi-asset simulator (CGT, loss carryforward, slippage, margin, R$20k exemption) |
| `signal_dsl.py` | Safe `ast`-parsed expression interpreter for factor signals |
| `config_strategy.py` / `spec_loader.py` | YAML `rank_and_hold` / `fixed_weight` engines + spec auto-discovery |
| `strategy_base.py` / `strategy_registry.py` | Plugin base class + auto-discovery registry |
| `metrics.py` | Sharpe, Calmar, max drawdown, annualised return/vol at any frequency |
| `portfolio_opt.py` | Equal-weight, inverse-vol, ERC, HRP, rolling-Sharpe, regime-conditional weights |
| `strategy_returns.py` | Runs the core strategies in one call → monthly after-tax return frame |
| `plotting.py` / `param_scanner.py` | Dark-theme tear sheets; generic 2D parameter-sweep heatmaps |

### Portfolio construction example

```python
from core.strategy_returns import build_strategy_returns
from core.portfolio_opt import hrp_weights, compute_portfolio_returns
from core.metrics import build_metrics

returns_df, sim_results, regime_signals = build_strategy_returns()
liquid = [c for c in returns_df.columns if c not in ("IBOV", "CDI", "SmallcapMom")]
port_ret, weights = compute_portfolio_returns(returns_df, liquid, lambda w: hrp_weights(w, 36))
print(build_metrics(port_ret, "My HRP Portfolio", 12))
```

### Runnable research scripts (`backtests/`)

```bash
cd backtests
python portfolio_compare_all.py          # all portfolio methods: 4-panel dashboard + CSV
python portfolio_risk_parity_backtest.py # equal-weight vs inverse-vol vs ERC
python portfolio_hrp_backtest.py         # Hierarchical Risk Parity + dendrogram
python portfolio_dynamic_backtest.py     # rolling-Sharpe / regime / combined allocation
python portfolio_stability_analysis.py   # sub-period metrics + rolling 36-month Sharpe
python sp500_b3_index.py                 # SP500-style market-cap B3 index reconstruction
python validate_mean_rev_composite.py    # parity check for the mean-reversion composite
```

## Feature discovery engine

`research/discovery/` automatically generates, evaluates, and ranks hundreds of
candidate alpha features by **Information Coefficient (IC)** — replacing manual
factor selection with a systematic sweep-and-prune pipeline.

```bash
python -m research.discovery.main                 # full run: generate → evaluate → prune → export
python -m research.discovery.main --incremental   # skip already-computed features
python -m research.discovery.main --force-recompute
python -m research.discovery.main --no-fundamentals
```

**How it works.** Level 0 base signals (~15 categories: momentum, volatility,
volume, beta, skew/kurtosis, illiquidity, mean reversion, market/CDI regime, …)
across many windows → Level 1 unary transforms (cross-sectional `rank`/`zscore`)
→ Level 2 composites (`delta`, `ratio_to_mean`, `ratio`, `product` on the
best features). Each feature is scored by IC (Spearman rank correlation vs
forward returns at 5/10/20/60-day horizons), then pruned by NaN/variance filter →
IC threshold → correlation dedup (>0.90) → cap at 500. Computed features and IC
time-series are cached in `research/feature_store/` (Parquet + a `registry.json`
with a source-data hash that auto-invalidates on new data), so re-runs are
incremental.

**Outputs** land in `research/output/`: a ranked `feature_catalog.json` (consumed
by backtests via `store:<id>`), a text report, and diagnostic plots (IC bar
chart, rolling IC time-series, IC decay, turnover scatter, correlation heatmap,
train/test overfit scatter).

## ML research pipeline

`research/main.py` is a separate feature-importance study: it engineers ~19
price/volume/cross-sectional/regime features, builds binary classification
targets, trains RandomForest + XGBoost, and compares feature-importance rankings.

```bash
python -m research.main
```

## Web UI

A Streamlit app to run every layer from the browser.

```bash
streamlit run ui/app.py      # or: python run_ui.py
```

Then open http://localhost:8501.

| Page | Description |
|------|-------------|
| **Pipeline** | DB stats, browse raw COTAHIST files & tables, trigger pipeline runs with live logs |
| **Backtest Runner** | Pick a registered strategy, configure params via dynamic forms, run in background, view interactive Plotly results |
| **Dashboard** | Browse saved results, compare strategies side-by-side (equity overlays, correlation matrices) |
| **Research** | ML feature-importance rankings, model metrics, trigger the research pipeline |
| **Discovery** | Feature-discovery catalog, IC rankings, and diagnostic charts |
| **Fundamentals** | Browse CVM fundamentals and point-in-time filings |

## Testing

```bash
make test                                  # full Python suite (pytest)
python -m pytest tests/test_config_strategy.py   # config-strategy behaviour tests
make test-rust                             # Rust unit tests
```

The `tests/` suite covers the parser, corporate actions, CVM ingestion (PIT,
storage, historical), the config-strategy engines, the data loader, and the
discovery core.

## License

MIT License
