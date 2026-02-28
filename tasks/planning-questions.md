# Planning Questions

## Codebase Summary

This is a comprehensive B3 (Brazilian Stock Exchange) data pipeline and quantitative research platform. The codebase has three major functional areas:

### 1. Data Pipeline (`b3_pipeline/`)
- **`main.py`** -- CLI orchestrator with 9-step pipeline: download COTAHIST ZIPs from B3, parse fixed-width files, upsert to SQLite, fetch corporate actions (dividends, splits) from B3 API, compute split/dividend adjustments, update adjusted columns
- **`downloader.py`** -- Downloads annual/daily COTAHIST ZIP files from B3
- **`parser.py`** -- Parses fixed-width COTAHIST format into DataFrames
- **`storage.py`** -- SQLite operations (schema: `prices`, `corporate_actions`, `stock_actions`, `detected_splits` tables)
- **`adjustments.py`** -- Split and dividend adjustment computation
- **`b3_corporate_actions.py`** -- B3 API client for corporate actions
- **`config.py`** -- All configuration constants, URLs, schema layouts
- Run via: `python -m b3_pipeline.main [--rebuild] [--year YYYY] [--skip-corporate-actions]`
- Database: `b3_market_data.sqlite` (~33 years of data, 1994-2026)
- Raw data: 33 COTAHIST ZIP files in `data/raw/`

### 2. Backtesting Framework (`backtests/`)
- **`core/`** -- Shared modules:
  - `data.py` -- Load B3 data from SQLite, download benchmarks (IBOV via yfinance, CDI from BCB API)
  - `simulation.py` -- Generic portfolio simulation engine with tax/slippage/loss-carryforward
  - `metrics.py` -- Performance metrics (Sharpe, Calmar, max drawdown, etc.)
  - `plotting.py` -- Dark-themed 4-panel tear sheet plots (cumulative return, drawdown, tax, turnover)
  - `strategy_returns.py` -- Runs all 8 core strategies and returns after-tax monthly return series
  - `portfolio_opt.py` -- Portfolio optimization (inverse-vol, ERC, HRP, rolling Sharpe, regime-conditional)
  - `param_scanner.py` -- Parameter sensitivity analysis framework with heatmap visualization
- **~40+ individual backtest scripts** -- Each is standalone, follows a common pattern:
  1. Load data from SQLite via `core.data`
  2. Compute signals (momentum, low-vol, multifactor, etc.)
  3. Generate target weights DataFrame
  4. Run simulation via `core.simulation.run_simulation()`
  5. Compute metrics and generate PNG tear sheet
- **Cross-strategy analysis scripts**:
  - `compare_all.py` -- Runs 8 strategies side-by-side, prints metrics table
  - `correlation_matrix.py` -- Correlation heatmap of strategy returns
  - `portfolio_compare_all.py` -- Compares 7 portfolio construction methods
  - `portfolio_stability_analysis.py` -- Sub-period stability metrics
  - `param_sensitivity_analysis.py` -- 2D parameter sweep heatmaps

### 3. ML Research (`research/`)
- Feature importance discovery pipeline (RF + XGBoost)
- 19 features, 3 binary targets
- Outputs: `research/output/` (PNG plots, CSV importance, JSON metrics, text summary)
- Run via: `python -m research.main`

### Current Outputs
- **~40 PNG files** scattered across project root and `backtests/` directory (tear sheets, heatmaps, correlation matrices)
- **2 CSV files** in `backtests/` (portfolio comparison results, stability results)
- **Research output** in `research/output/` (PNG, CSV, JSON, TXT)
- **Console output** -- all scripts print metrics tables, progress, etc. to stdout

### Key Technical Details
- Python 3.9 with venv
- Dependencies: requests, pandas, numpy, matplotlib, scikit-learn, xgboost, yfinance, scipy
- No web framework currently installed
- No test suite
- No existing API layer -- everything is direct script execution
- Database is SQLite (single file, ~local only)
- Backtest scripts use relative DB paths and `sys.path` manipulation for imports

---

## Questions

### Q1: Web Framework Choice
**Context:** The project currently has zero web infrastructure -- no Flask, Django, FastAPI, or any frontend framework. The existing Python code is pure scripts with matplotlib for charts. The choice of framework will be the single biggest architectural decision, affecting everything from how scripts are executed to how charts are displayed.
**Question:** Which technology stack do you prefer for the UI?
**Options (if applicable):**
- A) **Streamlit** -- Fastest to build, pure Python, great for data dashboards. Drawback: limited customization, all-Python (no separate frontend), can feel "demo-ish" for complex interactions.
- B) **FastAPI backend + React/Next.js frontend** -- Most flexible and professional, proper API separation. Drawback: significantly more work, requires JS/TS skills, two codebases to maintain.
- C) **FastAPI backend + simple HTML/HTMX frontend** -- Good middle ground, server-rendered, minimal JS. Drawback: less interactive than React but simpler to maintain.
- D) **Dash (Plotly)** -- Python-based like Streamlit but more powerful for interactive charts. Drawback: Dash-specific patterns, heavier than Streamlit.
- E) **Gradio or Panel** -- Other Python-native options, similar to Streamlit.
- F) **No preference -- recommend the best option** given the codebase.

### Q2: Scope of "Manage/Handle All of This"
**Context:** The codebase has three very different functional areas: (1) data pipeline operations (download, parse, adjust), (2) backtest execution and visualization (40+ strategies), and (3) ML research. Each has different UI needs -- the pipeline needs a "run and monitor" interface, backtests need parameter input and chart viewing, research needs a results dashboard. Building full UI coverage for all three is a large project.
**Question:** What is the priority order and minimum viable scope? Which of these capabilities are most important for v1?
**Options (if applicable):**
- A) **Pipeline management** -- Run/monitor the data pipeline, view database stats, trigger downloads
- B) **Backtest runner** -- Select a strategy, configure parameters, run it, view results (tear sheets, metrics)
- C) **Results dashboard** -- Browse and compare already-generated backtest results (PNGs, metrics tables), view the comparison/correlation analyses
- D) **Research viewer** -- View ML research results, feature importance plots
- E) **All of the above** -- full coverage from day one
- F) **Start with B+C** and expand later

### Q3: Script Execution Model
**Context:** Backtest scripts currently take 30 seconds to 15+ minutes to run (they load data from SQLite, download benchmarks from Yahoo/BCB, and run simulations). The compare_all.py and correlation_matrix.py scripts are particularly heavy since they run 8+ strategies. The pipeline itself can take 30+ minutes for a full rebuild. These are long-running processes.
**Question:** How should long-running scripts be executed from the UI?
**Options (if applicable):**
- A) **Synchronous** -- User clicks "Run", browser waits for completion (with loading spinner). Simple but blocks the UI for minutes.
- B) **Background jobs with polling** -- Scripts run in background processes, UI polls for status and displays results when done. More complex but much better UX.
- C) **Background jobs with WebSocket streaming** -- Real-time log streaming to the browser while scripts run. Best UX but most complex.
- D) **Pre-computed results only** -- Don't run scripts from UI at all; just browse results that were generated via CLI. Simplest approach.

### Q4: Chart Rendering Strategy
**Context:** All current charts are static PNGs generated by matplotlib with a custom dark theme (`PALETTE` in `backtests/core/plotting.py`). The tear sheets are complex 4-panel layouts. There are ~40 existing PNGs. Options range from simply displaying these PNGs to re-rendering charts interactively in the browser.
**Question:** How should charts be handled in the UI?
**Options (if applicable):**
- A) **Display existing PNGs** -- Simply serve the matplotlib-generated images. Fastest to implement, no chart rewrite needed. No interactivity (zoom, hover, etc.).
- B) **Interactive browser charts (Plotly/Chart.js)** -- Re-implement charts using a JS charting library for zoom, hover tooltips, etc. Much more work but much better UX.
- C) **Hybrid** -- Display existing PNGs for pre-generated results, but use interactive charts for new backtest runs. Good balance.
- D) **Keep matplotlib but generate on-demand** -- Run matplotlib server-side when user requests, serve as PNG. Allows parameter changes but no interactivity.

### Q5: Parameter Configuration
**Context:** Each backtest script has hardcoded configuration at the top (e.g., `LOOKBACK_YEARS = 1`, `PORTFOLIO_SIZE = 50`, `TAX_RATE = 0.15`, `MIN_ADTV = 1_000_000`, `START_DATE`, `END_DATE`). The `param_scanner.py` framework already supports parameter sweeps over a grid. Currently there is no way to change parameters without editing Python files.
**Question:** How configurable should the backtest runner be?
**Options (if applicable):**
- A) **Fixed** -- Just run scripts as-is with their hardcoded parameters. Simplest.
- B) **Basic overrides** -- Allow changing the most common parameters (start date, end date, initial capital, tax rate) via the UI, pass them as environment variables or CLI args.
- C) **Full parameter editor** -- Expose all strategy-specific parameters (lookback, portfolio size, top percentile, ADTV threshold, etc.) per strategy with sensible defaults.
- D) **Parameter sweep UI** -- Build on the existing `param_scanner.py` to let users define parameter grids and run sensitivity analyses from the UI.

### Q6: Deployment Model
**Context:** This appears to be a personal/local project (single-user, local SQLite database, local file paths). The question of deployment affects security concerns, multi-user support, authentication needs, and infrastructure complexity.
**Question:** Is this UI intended for local-only use (running on your machine), or do you want it deployable to a server (for team access or remote use)?
**Options (if applicable):**
- A) **Local only** -- Runs on localhost, accessed from your browser. No auth, no deployment concerns. Simplest.
- B) **Deployable to a VPS/cloud** -- Could be deployed to a server with basic auth. Moderate complexity.
- C) **Docker-containerized** -- Package everything in Docker for reproducible deployment anywhere. More setup but portable.

### Q7: Modular/Plugin Architecture for New Strategies
**Context:** The project has grown organically to 40+ backtest scripts, and new strategies keep being added. Each script follows a similar pattern but is standalone. The `core/strategy_returns.py` already attempts to centralize the 8 "core" strategies. There is a tension between the current "one script per strategy" flexibility and a more structured approach.
**Question:** Should the UI enforce a structured strategy registration pattern, or just discover and run existing scripts as-is?
**Options (if applicable):**
- A) **Script discovery** -- Automatically scan `backtests/*.py` for files matching `*_backtest.py`, display them as runnable items. No refactoring of existing scripts needed. Metadata (name, description) extracted from docstrings.
- B) **Registry pattern** -- Define a strategy registry (YAML/JSON/Python) that maps strategy names to scripts, parameters, and descriptions. Requires some cataloging work but gives better UI.
- C) **Plugin architecture** -- Refactor strategies into a common interface (e.g., `class Strategy` with `generate_signals()`, `get_parameters()`, etc.). Most work but cleanest long-term.

