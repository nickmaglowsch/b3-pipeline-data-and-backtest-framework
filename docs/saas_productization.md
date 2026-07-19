# SPEC: Productizing the B3 Pipeline into a SaaS Platform

Status: **Design / not started.** This is a spec-driven roadmap for turning the
current single-user research bench into a multi-tenant product. Nothing here is
built yet. Each phase below is written to be picked up independently and lists
its concrete files, interface contracts, and acceptance criteria.

---

## Problem

Today the repo is a **solo-quant research bench**: local, single-user, no auth,
no web API, no scheduler, and a hardcoded SQLite file.

Verified current state (branch `main`, this repo):
- DB is SQLite, path hardcoded: `b3_pipeline/config.py:23` → `DB_PATH = PROJECT_ROOT / "b3_market_data.sqlite"` (~602 MB).
- No web API: no `fastapi` / `flask` / `uvicorn` anywhere.
- No orchestration: no `.github/`, no `apscheduler` / `airflow` / `prefect` / `celery`. Ingestion is manual CLI (`python -m b3_pipeline.main`, `cvm_main`) or triggered from the Streamlit UI.
- No auth / billing / multi-tenancy.
- Benchmarks pulled via `yfinance` in `backtests/core/data.py` (a scraper — not commercially redistributable).

Despite that, the repo holds three genuinely defensible assets:

1. **Corporate-actions-accurate B3 price history** (1994→present, ~2.6M rows) with split/reverse-split/bonus/dividend/JCP adjustment from B3 official data, not heuristics — `b3_pipeline/adjustments.py`, `b3_pipeline/b3_corporate_actions.py`.
2. **Point-in-time, survivorship-corrected CVM fundamentals** — `fundamentals_pit`, delisting-aware CNPJ↔ticker via FCA (`company_tickers_pit`).
3. **A tax-aware research engine** — Brazilian-CGT-accurate portfolio simulator (`backtests/core/simulation.py`) + automated IC factor-discovery pipeline (`research/discovery/`). Strategies are now **config/spec-driven** (`backtests/strategies/specs/*.yaml` + expression interpreter), so a hosted bench can let users author strategies as *data*, not code.

The goal: expose these assets as a product without discarding the research core.

---

## Target segments & surfaces

Three product surfaces share **one data core**:

| Surface | Segment | Monetizes | ARPU | Regulatory load |
|---|---|---|---|---|
| Investor app | Retail BR | Portfolio + IRPF tax + FII/stock screens | Low, high volume, seasonal (Mar–Apr IRPF) | LGPD (high, handles CPF); CVM advisory risk if it "recommends" |
| Research bench | Pro / serious-retail quant | Hosted backtest + factor discovery + tax-accurate sim | High | Low (self-directed tooling) |
| Data API | B2B (fintechs, researchers) | Adjusted B3 prices + PIT fundamentals feed | High | **B3 redistribution license = hard blocker** |

**Recommended lead:** productize the **data core (Phase 0)** first — it is pure
engineering, gated by no legal review, and unblocks every surface. Then open the
**pro research bench (Phase 2)**, which reuses the most existing in-repo code.

**Important correction:** the retail MCP business logic (`irpf_report`,
`portfolio_status`, `tlh_plan`, `rank_equities`, …) lives in a **separate
project**, not this repo. The retail app (Phase 1) is therefore a *from-scratch*
build here — scope it accordingly. The pro bench reuses `backtests/core/`,
`ui/services/job_runner.py`, and `research/discovery/` directly.

---

## Target architecture

```
   Ingestion workers        ┌─────────────────────────────────────────┐
   (scheduled, Phase 0)      │            SHARED DATA CORE              │
   B3 COTAHIST daily   ────▶ │  Postgres (+ Timescale for prices)      │
   B3 corp-actions           │  • reference data: prices, corp actions, │
   CVM DFP/ITR/FRE           │    fundamentals_pit  (ONE copy, shared)  │
                             │  • tenant data: users, portfolios,       │
                             │    holdings, tax lots (row-level tenant) │
                             └───────────────┬─────────────────────────┘
                                             │  FastAPI (Phase 0/2)
              ┌──────────────────────────────┼──────────────────────────────┐
              ▼                              ▼                              ▼
     Investor app (Phase 1)       Research bench (Phase 2)         Data API (Phase 3)
     Next.js + billing            Streamlit reused as console      metered, licensed
```

**Reference data is shared, single-copy, read-only** — prices and fundamentals
are identical for every tenant. Do NOT partition them per user. Only user
portfolios/holdings/tax-lots are per-tenant. Storage stays a few GB.

---

## Phase 0 — Productionize the data core

Foundation for every surface. Pure engineering, no legal gates, no external
accounts. Do this first.

### 0.1 Parameterize DB connection
- **Target:** remove the hardcoded SQLite path; make the backend env-driven.
- **Files (authoritative way to find them: `grep -rn b3_market_data.sqlite`):** `b3_pipeline/config.py:23` (`DB_PATH` → read `DATABASE_URL`), plus these modules that **independently** hardcode the root SQLite path and are easy to miss: `research/config.py:5` (a *separate* `DB_PATH` — does NOT import `b3_pipeline.config`), `ui/services/fundamentals_service.py:17`, `ui/services/pipeline_service.py:19`, `backtests/core/strategy_returns.py:43` (`_DEFAULT_DB`), `ui/services/backtest_service.py`. (`scripts/audit_db.py` already exposes a `--db` flag — no change needed.)
- **Contract:** `b3_pipeline/storage.py:228` **already** has `get_connection(db_path: Optional[Path] = None)` defaulting to `config.DB_PATH` — *extend* it to accept a `DATABASE_URL` (Postgres) or path (SQLite); do not create a new function. `DATABASE_URL` unset → falls back to the existing SQLite file (keeps local dev working).
- **Note:** `b3_pipeline/cvm_storage.py` and `backtests/core/data.py` take an *injected* connection / `db_path` param rather than a hardcoded path, so they are not 0.1 work. They are SQLite-*dialect*-coupled (`INSERT OR REPLACE`, `sqlite3.connect`) — their change belongs in **0.2** (dialect shim).
- **Acceptance:** existing pytest suite passes unchanged with SQLite; `DATABASE_URL=postgres://…` routes all reads/writes to Postgres.

### 0.2 Postgres backend + schema port
- **Target:** port the schema in `b3_pipeline/storage.py` + `cvm_storage.py` to Postgres. Prices → a Timescale hypertable (or native partitioning by year).
- **Approach:** add a thin dialect shim (`INSERT OR REPLACE` → `INSERT … ON CONFLICT … DO UPDATE`). Keep raw SQL; no ORM (matches current style). *ponytail: Postgres is the one real infra swap multi-tenancy requires — do not add a warehouse/Kafka/etc. yet.*
- **Files:** new `b3_pipeline/db/` (dialect + migrations, e.g. plain `.sql` or `yoyo`/`alembic`); modify `storage.py`, `cvm_storage.py`.
- **Migration:** one-shot script that copies the 602 MB SQLite into Postgres and validates row counts (`prices`, `corporate_actions`, `fundamentals_pit`, `fundamentals_monthly`, …).
- **Acceptance:** row counts match SQLite (`prices` ≈ 2.59M, `fundamentals_pit` ≈ 52K); a sample `PETR4` adjusted series is byte-identical between backends.

### 0.3 Scheduler
- **Target:** run the already-idempotent ingestion CLIs on a schedule.
- **Approach (laziest that works):** a container running APScheduler, or a GitHub Actions / cloud scheduled job. Daily: COTAHIST + corporate actions. Weekly/quarterly: CVM. The pipelines are `INSERT OR REPLACE`-idempotent, so re-runs are safe.
- **Files:** new `scheduler/` (or `.github/workflows/ingest.yml`).
- **Acceptance:** unattended for 2 weeks, daily freshness with zero manual intervention.

### 0.4 Ingestion monitoring
- **Target:** freshness + gap alerting. Repurpose `scripts/audit_db.py`.
- **Checks:** last trading day present? corporate-actions gaps? fundamentals staleness > 400d? Alert on failure (email/webhook).
- **Acceptance:** a simulated stale/failed run fires an alert.

### 0.5 Containerize
- **Target:** reproducible image including the Rust extension build (`make dev-rust` / maturin must run in-image; see `b3_pipeline_rs/`).
- **Files:** new `Dockerfile`, `.dockerignore`.
- **Acceptance:** `docker run` executes an ingestion CLI end-to-end against Postgres.

---

## Phase 1 — Retail investor app (from-scratch here; the revenue wedge)

Framed as a **reporting/tracking tool, not advice**, to stay clear of CVM
advisory rules (see Regulatory). Lead feature = IRPF report.

### 1.1 IRPF tax report engine (NEW — not in this repo)
- **Target:** ingest B3/CEI brokerage reports → cost-basis + realized-gain computation → IRPF-ready output. Correctness depends directly on the corporate-actions accuracy already built (`b3_pipeline/adjustments.py`).
- **Acceptance:** output reconciles against a hand-computed IRPF case (buys, sells, splits, dividends/JCP, R$20k monthly exemption, loss carryforward).

### 1.2 Portfolio tracking
- Holdings + performance vs IBOV/CDI. Reuses metrics (`backtests/core/metrics.py`) and a **licensed** benchmark source (NOT yfinance — see 3.2).

### 1.3 Accounts, multi-tenancy, billing
- Auth (Clerk/Auth0/Supabase or self-hosted) with **row-level tenant isolation** on portfolio tables.
- Billing: Stripe + a BR processor for **Pix/boleto** (Pagar.me / Iugu / Asaas). Price IRPF as a seasonal hook; tracking as recurring base.
- Frontend: **Next.js** (Streamlit is wrong for a polished consumer app).
- **Requires external accounts + secrets** — cannot be built without them.

### 1.4 Screens as *educational* tools
- `rank_equities` / `rank_fiis`-style screens ship as explicitly self-directed, disclaimed screening — behind the advisory line (see Regulatory).

---

## Phase 2 — Pro research bench (reuses the most existing code)

### 2.1 Reuse Streamlit as the hosted console
- Put `ui/app.py` (6 pages) behind the same auth/billing. Pro users tolerate Streamlit; don't rebuild it.

### 2.2 Per-tenant background jobs
- `ui/services/job_runner.py` already runs long jobs in threads. Move to a real worker (RQ/Celery) with **per-tenant resource quotas** so one user can't starve the box.
- **Acceptance:** two tenants run concurrent backtests; neither exceeds its quota.

### 2.3 Sell the differentiated core
- Tax-accurate B3 backtesting + IC factor discovery (`research/discovery/`) is genuinely differentiated for BR quants. Strategy authoring is already spec/YAML-driven (`backtests/strategies/specs/`), so a sandboxed "upload a strategy spec" path is *config*, not arbitrary code — much lower risk than untrusted code execution. This is even safer than it sounds: the expression interpreter `backtests/core/signal_dsl.py` parses signals with Python's `ast` module against a whitelist, with **no `eval()`** — so accepting user strategy specs does not mean executing user code.

---

## Phase 3 — Data API (deepest moat, gated on licensing)

Metered API over adjusted prices + PIT fundamentals. **Do not ship until the B3
redistribution question is legally cleared (see 3.1).** CVM-derived fundamentals
are the safer first API product.

---

## Regulatory, licensing & ops gates (resolve BEFORE the relevant phase)

These can block or reshape the product — not just polish.

### 3.1 B3 market-data redistribution — biggest blocker (gates Phase 3)
COTAHIST serial-history files are downloaded from B3's public site, but
**redistributing B3 market data commercially generally requires a B3
market-data license/agreement.** Serving raw/adjusted prices via API or app =
redistribution. Get a written legal opinion before Phase 3. For Phases 1–2,
derived/computed outputs (tax reports, backtest curves, screens) are lower-risk
than exposing the raw feed — confirm scope.

### 3.2 `yfinance` must go (gates Phases 1–2)
`backtests/core/data.py` pulls IBOV + ETF benchmarks via `yfinance`. Yahoo's ToS
prohibits commercial redistribution and yfinance is a scraper — **not
production-safe.** Replace with a licensed benchmark source (B3 index license or
a data vendor) or compute an in-house index from your own adjusted constituents.

### 3.3 CVM investment-advice regulation (shapes Phase 1 feature set)
Resolução CVM 19/20/179: providing **buy/sell recommendations or portfolio
allocations to the public for compensation** requires registration as a
*consultor/analista de valores mobiliários*. At risk: `suggest_allocation`,
`rank_*` framed as "buy these", arguably `tlh_plan`. Mitigations: frame as
educational/screening/self-directed tooling, add disclaimers, or partner with /
hire a licensed advisor for any true recommendation. Keep the tax/reporting core
cleanly on the "tool, not advice" side — that's why it leads.

### 3.4 LGPD (gates Phase 1)
Storing users' brokerage holdings, portfolio history, and **CPF** (for IRPF) =
sensitive personal + financial data. Requires explicit consent, DPA, encryption
at rest/in transit, data-subject rights, breach process, likely a DPO. **CPF is
high-sensitivity — minimize and isolate it.**

### 3.5 Tax-report liability (Phase 1)
IRPF outputs feed users' real tax filings; errors cause real harm. Need
disclaimers ("verify with your accountant"), an accuracy-validation suite
against known cases, and ideally an accountant-in-the-loop path.

### 3.6 Third-party data provenance
- The "Múltiplo" pre-2010 backfill (`b3_pipeline/dm_downloader.py`) is a third-party source — **check ToS; likely not redistributable.** Quarantine from any customer-facing output.
- CVM open data (`dados.cvm.gov.br`) and BCB CDI are low-risk / redistributable — your safest assets.

### 3.7 Ops
Data-freshness SLAs + ingestion alerting (0.4); per-tenant compute limits (2.2);
reproducible Rust build in CI (0.5); Postgres backups + PITR (tenant financial
data).

---

## De-risking checkpoints (do in order; each gates the next spend)

1. **Legal gate (first, cheap):** written opinion on (a) B3 redistribution scope, (b) whether the tax/portfolio core stays clear of CVM advisory rules. Determines whether Phase 3 and the ranking features are viable. Kills/reshapes early.
2. **Technical proof:** Phase 0 done + one FastAPI endpoint (IRPF report or a backtest run) end-to-end against Postgres, reconciled against a known case.
3. **Demand proof:** paywall the IRPF report for a small cohort during a tax season (or pre-sell) before building the full Next.js app.
4. **Ops proof:** scheduled ingestion unattended for 2 weeks with alerting (0.3 + 0.4).

Only after 1–3 pass is the full-platform spend justified. *ponytail: the full
platform is the destination, but each phase ships something sellable — resist
building Phase 3 infra before Phase 1 has a paying user.*

---

## Dependencies

- Phase 0 blocks everything (multi-tenancy needs Postgres + config parameterization).
- Phase 1 needs 3.2 (benchmark source), 3.3/3.4/3.5 (compliance), external billing/auth accounts.
- Phase 2 needs Phase 0 + 3.2; reuses existing code, lightest lift after Phase 0.
- Phase 3 needs 3.1 (B3 license) — hard-blocked until cleared.
