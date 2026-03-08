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

### CVM Fundamentals Module -- created 2026-03-03
- b3_pipeline/storage.py: added SCHEMA_CVM_COMPANIES, SCHEMA_CVM_FILINGS, SCHEMA_FUNDAMENTALS_PIT + indexes
- b3_pipeline/cvm_storage.py: upsert_cvm_company(), upsert_cvm_filing(), upsert_fundamentals_pit(), get_cvm_company_map()
- b3_pipeline/config.py: CVM_DFP_BASE_URL, CVM_ITR_BASE_URL, CVM_FRE_BASE_URL, CVM_DATA_DIR, CVM_START_YEAR=2010
- b3_pipeline/b3_corporate_actions.py: extract_cnpj_from_company_data(), build_cnpj_ticker_map() added
- b3_pipeline/cvm_downloader.py: download_dfp_file(), download_itr_file(), download_fre_file(), download_all_cvm_files()
- b3_pipeline/cvm_parser.py: parse_dfp_zip(), parse_itr_zip(), parse_fre_zip() with in-memory ZIP support for testing
- b3_pipeline/cvm_main.py: run_fundamentals_pipeline(), materialize_valuation_ratios() — entry point: python -m b3_pipeline.cvm_main
- backtests/core/data.py: load_fundamentals_pit(), load_all_fundamentals() added
- backtests/core/shared_data.py: build_shared_data() now accepts include_fundamentals=False (backward compat)
- backtests/strategies/value_quality.py: ValueQualityStrategy with needs_fundamentals=True
- ui/services/fundamentals_service.py + ui/pages/6_fundamentals.py: Streamlit page follows 1_pipeline.py pattern
- ui/services/backtest_service.py: detects needs_fundamentals on strategy, passes include_fundamentals flag
- Tests: 36 new tests in tests/test_cvm_*.py and tests/test_fundamentals_pit.py; all 60 tests pass
- Key TDD pattern: tests written FIRST, implementation second; in-memory SQLite fixtures for DB tests
- CVM ZIP test pattern: use io.BytesIO() + zipfile.ZipFile for in-memory synthetic ZIPs (no real files needed)

### fundamentals_monthly Module -- created 2026-03-05
- b3_pipeline/storage.py: SCHEMA_FUNDAMENTALS_MONTHLY + 2 indexes added
- b3_pipeline/cvm_storage.py: upsert_fundamentals_monthly(), truncate_fundamentals_monthly()
- b3_pipeline/cvm_main.py: _build_adtv_ticker_map() helper (shared by materialize_valuation_ratios + materialize_fundamentals_monthly), materialize_fundamentals_monthly(), --skip-monthly CLI flag
- backtests/core/data.py: load_fundamentals_pit() bug fix (removed filing_date >= start; uses extended index union for pre-period ffill seeding), load_fundamentals_monthly(), load_all_fundamentals_monthly(), compute_pe_ratio_dynamic(), compute_pb_ratio_dynamic(), compute_ev_ebitda_dynamic()
- backtests/core/shared_data.py: added f_*_m and f_*_dyn keys when include_fundamentals=True
- backtests/strategies/low_pe.py: LowPEStrategy (name="LowPE"), needs_fundamentals=True
- Tests: 153 total (was 60 → now 153); new test files: test_fundamentals_monthly.py, test_materialize_ratios_window.py, test_shared_data_fundamentals.py, test_low_pe_strategy.py
- Key pattern: existing tests (test_fundamentals_pit.py) needed _insert_company() helper added after ADTV map changed to require cvm_companies rows

### Historical Fundamentals Extension -- created 2026-03-07
- CRITICAL: CVM IPE (/DOC/IPE/) is a DOCUMENT INDEX (metadata + PDF links), NOT financial data
  - No account codes, no VL_CONTA, no structured financials — do NOT attempt to parse as DFP/ITR
  - IPE is useful only for company CNPJ/name/CVM code extraction (populate cvm_companies)
  - Confirmed Path B (Task 05): no shares outstanding in IPE; _propagate_ipe_shares() not implemented
- CAD dataset (/CAD/DADOS/cad_cia_aberta.csv): single bulk CSV with listing/delisting dates
- New files: b3_pipeline/cad_downloader.py, cad_parser.py, ipe_downloader.py, ipe_parser.py
- b3_pipeline/storage.py: SCHEMA_CVM_COMPANIES now has listing_date DATE, delisting_date DATE
- b3_pipeline/cvm_storage.py: upsert_cad_company_dates() (COALESCE on listing_date, overwrite delisting_date)
- b3_pipeline/cvm_main.py: include_historical param + Steps 11-13; --include-historical CLI flag
  - IPE year range: _orig_start_year captured BEFORE start_year defaults to CVM_START_YEAR
  - Without --start-year, IPE defaults to 2003-2009; with --start-year 2006, IPE covers 2006-2009
- backtests/core/data.py: load_active_tickers(), _get_all_known_roots(), _apply_delisted_filter()
  - load_b3_data() and load_b3_hlc_data() get filter_delisted=False param (backward compat)
- backtests/core/shared_data.py: build_shared_data() gets filter_delisted=False param
- Tests: 198 total (was 153); new: test_cad_parser.py, test_cad_downloader.py, test_ipe_parser.py,
  test_ipe_shares.py, test_cvm_main_historical.py, test_survivorship_filter.py

### Rust Extension Module (b3_pipeline_rs/) -- created 2026-03-08
- Crate: cotahist_rs, PyO3 0.22, pyo3-arrow 0.5, arrow 53, rayon 1.8
- New files: src/adjustments.rs, src/pivot.rs (src/parser.rs, src/schema.rs pre-existed)
- Functions added: detect_splits, compute_split_adjustment, pivot_and_ffill
- Python wrappers: _detect_splits_rs, _compute_split_adjustment_rs in b3_pipeline/adjustments.py
- Python wrapper: _pivot_and_ffill_rs in backtests/core/data.py
- Build: `make dev-rust` (maturin develop); Test: `make test-rust` (cargo test)
- CRITICAL: Arrow Date32 casting — pandas datetime64[ns] converts to Timestamp[ns] in Arrow,
  NOT Date32. Must explicitly cast: `old_col.cast(pa.date32())` via _cast_date_col() helper
  before passing date columns to Rust. Applied to both "date" and "ex_date" columns.
- cargo path on macOS: /Users/nickmaglowsch/.cargo/bin/cargo (not in default PATH)
  Use `export PATH="$HOME/.cargo/bin:$PATH"` before cargo commands.
- Tests: 220 Python tests + 45 Rust unit tests all passing
- Integration test files: tests/test_detect_splits_rs.py, test_split_adjustment_rs.py,
  test_load_b3_data_pivot_rs.py — all skip if cotahist_rs not compiled

### MeanRevComposite Strategy -- created 2026-03-02
- backtests/core/mean_rev_helpers.py: 3 helpers -- compute_regime_filter(), compute_alpha_score(), compute_signal_stability()
- backtests/strategies/mean_reversion.py: SimpleMeanReversionStrategy (old) + MeanReversionCompositeStrategy (new, name="MeanRevComposite")
- backtests/core/shared_data.py: now calls load_b3_hlc_data() (not load_b3_data()), adds 8 new feature keys + 3 HLC keys
- backtests/core/strategy_returns.py: now 9 strategies; also calls load_b3_hlc_data() and adds same 8 feature keys + ibov_vol_pctrank + ibov_px to shared dict
- backtests/validate_mean_rev_composite.py: 5-test end-to-end validation script
- All helper functions (regime filter, alpha scoring, stability guard) are in mean_rev_helpers.py -- NOT split across tasks as the task files suggested
- strategy_returns.py imports from `core.mean_rev_helpers` (not `backtests.core`) due to its sys.path setup
