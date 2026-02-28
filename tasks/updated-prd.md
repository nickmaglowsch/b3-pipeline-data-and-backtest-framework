# Updated PRD: B3 Data Pipeline -- Streamlit Management UI

## Overview

Build a modular Streamlit-based UI that serves as the single control plane for the entire B3 data pipeline and quantitative research platform. The UI provides four major modules: (1) data pipeline management, (2) backtest runner with full parameter editing, (3) results dashboard with interactive Plotly charts, and (4) ML research viewer. All long-running operations execute as background jobs with real-time WebSocket-style log streaming. Strategies are refactored into a plugin architecture with a common base class.

## What Already Exists

### Data Pipeline (`b3_pipeline/`)
- Complete 9-step CLI pipeline: download, parse, upsert, corporate actions, adjustments
- Entry point: `b3_pipeline/main.py` -- `run_pipeline(rebuild, year, skip_corporate_actions)`
- Config: `b3_pipeline/config.py` -- all constants, URLs, schema layouts
- Storage: `b3_pipeline/storage.py` -- SQLite with `get_summary_stats()`, `get_all_tickers()`, etc.
- Database: `b3_market_data.sqlite` at project root

### Backtesting Framework (`backtests/`)
- **Core modules** (`backtests/core/`):
  - `data.py` -- `load_b3_data()`, `load_b3_hlc_data()`, `download_benchmark()`, `download_cdi_daily()`
  - `simulation.py` -- `run_simulation()` -- generic portfolio simulation with tax/slippage
  - `metrics.py` -- `build_metrics()`, `value_to_ret()`, `display_metrics_table()`
  - `plotting.py` -- matplotlib 4-panel tear sheets with dark PALETTE theme
  - `strategy_returns.py` -- `build_strategy_returns()` runs 8 core strategies centrally
  - `portfolio_opt.py` -- inverse-vol, ERC, HRP, rolling Sharpe, regime-conditional weights
  - `param_scanner.py` -- `scan_parameters()` + `plot_param_heatmap()` for 2D sweeps
- **43 standalone backtest scripts** in `backtests/*_backtest.py` -- each follows:
  1. Hardcoded CONFIG section at top
  2. `generate_signals()` or inline signal logic
  3. Build target weights DataFrame
  4. Call `run_simulation()`
  5. Call `plot_tax_backtest()` to produce PNG
- **Cross-strategy analysis**: `compare_all.py`, `correlation_matrix.py`, `portfolio_compare_all.py`, `portfolio_stability_analysis.py`, `param_sensitivity_analysis.py`

### ML Research (`research/`)
- 6-step pipeline: load data, engineer 19 features, compute targets, train RF+XGBoost, save results, generate report
- Entry point: `research/main.py`
- Outputs in `research/output/`: PNGs, CSV, JSON, TXT

### Current Outputs
- ~40 PNG tear sheets (scattered in project root and `backtests/`)
- 2 CSV files in `backtests/`
- Research output in `research/output/`
- All console output goes to stdout

## Architecture Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| UI Framework | Streamlit | Pure Python, fastest to build, fits Python-heavy codebase |
| Charts | Plotly (via `plotly` + `st.plotly_chart`) | Interactive zoom, hover, tooltips |
| Long-running jobs | Background threads + streaming | Real-time log output in the UI |
| Strategy architecture | Plugin base class | Clean long-term extensibility |
| Deployment | Local only (localhost) | No auth, no Docker, simplest setup |
| Scope | All 4 modules in v1 | Pipeline, backtests, dashboard, research |

## New Directory Structure

```
ui/
  __init__.py
  app.py                    # Streamlit entry point (multipage)
  pages/
    __init__.py
    1_pipeline.py            # Data pipeline management page
    2_backtest_runner.py     # Run backtests with parameter editing
    3_dashboard.py           # Browse/compare results
    4_research.py            # ML research viewer
  components/
    __init__.py
    charts.py                # Plotly chart builders (replaces matplotlib)
    log_stream.py            # Real-time log streaming component
    metrics_table.py         # Performance metrics display
    parameter_form.py        # Dynamic parameter form builder
  services/
    __init__.py
    job_runner.py            # Background job execution engine
    pipeline_service.py      # Pipeline operations wrapper
    backtest_service.py      # Backtest execution wrapper
    research_service.py      # Research pipeline wrapper
    result_store.py          # Read/write backtest results (JSON/pickle)

backtests/
  core/
    strategy_base.py         # NEW: Abstract base class for strategies
    strategy_registry.py     # NEW: Strategy discovery and registration
    ... (existing modules unchanged)
  strategies/                # NEW: Refactored strategy plugins
    __init__.py
    momentum_sharpe.py
    low_volatility.py
    copom_easing.py
    multifactor.py
    smallcap_momentum.py
    ... (one file per strategy)
  ... (existing scripts kept for backward compatibility)
```

## Module Requirements

### Module 1: Data Pipeline Management
- View database summary stats (total records, tickers, date range, tables)
- Trigger pipeline run with options: rebuild, specific year, skip corporate actions
- View raw data files in `data/raw/` with sizes and dates
- Real-time log streaming during pipeline execution
- View sample data from each table (prices, corporate_actions, stock_actions, detected_splits)

### Module 2: Backtest Runner
- Browse registered strategies from the plugin registry
- View strategy description, default parameters, and parameter schema
- Edit all parameters via dynamically generated form (numeric inputs, dropdowns, date pickers)
- Run selected strategy with custom parameters
- Real-time log streaming during execution
- Display results inline: Plotly tear sheet, metrics table
- Save results to disk for later viewing in dashboard

### Module 3: Results Dashboard
- Browse all saved backtest results (from both UI runs and legacy CLI runs)
- View individual result: interactive Plotly tear sheet, metrics table, parameter snapshot
- Compare multiple strategies side-by-side: overlay equity curves, metrics comparison table
- Correlation matrix heatmap of strategy returns
- Display pre-existing PNG results from legacy runs as fallback

### Module 4: ML Research Viewer
- Trigger research pipeline run with streaming logs
- View feature importance rankings (interactive bar charts)
- View model performance metrics (accuracy, AUC, precision, recall, F1)
- View robustness comparison across targets
- Display research summary report

### Cross-cutting: Job Runner
- Execute Python functions in background threads
- Capture stdout/stderr in real-time via queue
- Provide status (pending, running, completed, failed) and progress
- Store job results for retrieval
- Allow only one job of each type at a time (no concurrent pipeline runs)

### Cross-cutting: Strategy Plugin Architecture
- Abstract base class `StrategyBase` with:
  - `name` property
  - `description` property
  - `get_default_parameters() -> dict`
  - `get_parameter_schema() -> dict` (types, ranges, descriptions)
  - `generate_signals(shared_data, params) -> (returns_matrix, target_weights)`
- Registry that discovers strategies from `backtests/strategies/` directory
- Shared data loader that prepares common DataFrames once per session
- Legacy scripts remain runnable from CLI unchanged

## Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| Streamlit's threading model conflicts with background jobs | Use `threading.Thread` + `queue.Queue`; Streamlit reruns are state-safe via `st.session_state` |
| Plotly chart recreation is expensive for 40+ strategies | Lazy-load charts; cache with `@st.cache_data`; only render visible charts |
| Strategy refactoring breaks existing scripts | Keep old scripts untouched; new plugin classes are separate files in `backtests/strategies/` |
| SQLite concurrent access during pipeline runs | Use WAL mode (already configured); read-only connections for UI queries |
| Large session state from cached DataFrames | Use `@st.cache_resource` for shared data; clear cache explicitly when pipeline updates DB |
| Matplotlib -> Plotly chart parity | Start with the most important charts (equity curve, drawdown, metrics); add detail incrementally |

## Dependencies to Add

```
streamlit>=1.30.0
plotly>=5.18.0
```

## Non-Goals (v1)
- Multi-user authentication
- Docker containerization
- Remote deployment
- Real-time market data streaming
- Automated scheduling (cron-like)
- Mobile-responsive design
