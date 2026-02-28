# Orchestrator Memory

## Project: B3 Data Pipeline

### Environment
- Python: python3 (not python)
- Shell: zsh on macOS Darwin
- Working directory: /Users/nickmaglowsch/person-projects/b3-data-pipeline

### macOS Dependencies
- xgboost requires `brew install libomp` before it will load on macOS
- scikit-learn not in requirements.txt but needed; install with pip3

### Research Module (research/)
- Created 2026-02-27
- All 7 modules implemented: config, data_loader, features, targets, modeling, visualization, main
- Entry point: `python3 -m research.main` from project root
- Output: research/output/ (CSV, JSON, PNG, TXT)

### Import Pattern
- backtests/core is imported via sys.path.insert: `sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "backtests"))`
- research modules import each other via `from research import config` (project root must be in sys.path)

### Task Dependency Notes
- Linear tasks 01->07 were executed sequentially
- Tasks 03 (features.py) and 04 (targets.py) touch different files -- safe to parallelize
- Always verify imports after creation before proceeding to dependent tasks

### Portfolio Optimization Module (backtests/core/) -- created 2026-02-28
- strategy_returns.py: build_strategy_returns() returns 3 values (returns_df, sim_results, regime_signals)
  - Note: function signature was expanded beyond the 2-value tuple in the task spec
  - Downstream scripts must unpack 3 values: `returns_df, sim_results, regime_signals = build_strategy_returns()`
- portfolio_opt.py: all weight functions (inverse_vol, ERC, HRP, rolling_sharpe, regime_conditional)
- param_scanner.py: scan_parameters() + plot_param_heatmap()
- scipy added to requirements.txt (needed for ERC + HRP clustering)

### Wave Execution Pattern (tasks 01-10)
- Wave 1 (seq): Task 01 (foundation)
- Wave 2 (parallel): Tasks 02 (portfolio_opt.py), 06, 08, 09, 10 -- all different files
- Wave 3 (seq): Task 03 extends portfolio_opt.py -- cannot parallel with Task 02
- Wave 4 (seq): Task 04 extends portfolio_opt.py -- cannot parallel with 02/03
- Wave 5 (seq): Task 05 uses portfolio_opt.py (read-only)
- Wave 6 (seq): Task 07 uses portfolio_opt.py (read-only)
- Key conflict: Tasks 02/03/04 all write to portfolio_opt.py → must be sequential

### Key Results (B3 Portfolio Optimization, 2005-2026)
- DynCombined: Sharpe 1.62, Ann.Ret 14.61%, Final NAV R$1.79M from R$100K
- DynRollSharpe: Sharpe 2.34 (mostly in CDI), lower equity exposure
- HRP: Sharpe 0.71 (best static method), beats EqualWeight 0.65
- SmallcapMom bankrupt in 2009: excluded from liquid portfolio

### Streamlit UI Module (ui/) -- created 2026-02-28
- Entry point: `streamlit run ui/app.py`
- 4 pages: 1_pipeline.py, 2_backtest_runner.py, 3_dashboard.py, 4_research.py
- Components: charts.py (Plotly), metrics_table.py, parameter_form.py, log_stream.py
- Services: job_runner.py, result_store.py, pipeline_service.py, backtest_service.py, research_service.py
- Strategy plugins: backtests/strategies/ (13 strategies, all discovered by StrategyRegistry)
- New core files: strategy_base.py, strategy_registry.py, shared_data.py
- Install: pip install streamlit plotly (added to requirements.txt)
- JobRunner captures stdout/stderr from threads via _thread_local routing
- Results stored in results/ directory (gitignored)
- Strategy discovery: get_registry().list_all() returns all 13 strategies on first call
- backtest_service uses @st.cache_resource to cache shared_data (heavy, ~200MB)
