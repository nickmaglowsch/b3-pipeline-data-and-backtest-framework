# Code Review Report -- Streamlit Management UI

## Summary

The implementation is structurally sound and covers the vast majority of PRD requirements across all four modules plus the two cross-cutting concerns (job runner, strategy plugins). The code is well-organized, follows established project conventions, and demonstrates thoughtful architecture decisions. However, there are a handful of bugs that will cause runtime crashes, one security concern, and several behavioral discrepancies versus the legacy backtest scripts that should be addressed before shipping.

## PRD Compliance

| # | Requirement | Status | Notes |
|---|-------------|--------|-------|
| **Module 1: Data Pipeline Management** | | | |
| 1.1 | View database summary stats (total records, tickers, date range, tables) | Complete | `ui/pages/1_pipeline.py` lines 46-56 + `ui/components/metrics_table.py:render_db_stats()` |
| 1.2 | Trigger pipeline run with options: rebuild, year, skip corporate actions | Complete | `ui/pages/1_pipeline.py` lines 93-147. Form with all three options. |
| 1.3 | View raw data files in `data/raw/` with sizes and dates | Complete | `ui/pages/1_pipeline.py` lines 61-71 + `pipeline_service.get_raw_files()` |
| 1.4 | Real-time log streaming during pipeline execution | Complete | `ui/components/log_stream.py` + `ui/services/job_runner.py` |
| 1.5 | View sample data from each table (prices, corporate_actions, stock_actions, detected_splits) | Complete | `ui/pages/1_pipeline.py` lines 73-89 with tabs per table |
| **Module 2: Backtest Runner** | | | |
| 2.1 | Browse registered strategies from the plugin registry | Complete | `ui/pages/2_backtest_runner.py` lines 54-68 |
| 2.2 | View strategy description, default parameters, and parameter schema | Complete | Line 72 shows description; parameter form shows all specs |
| 2.3 | Edit all parameters via dynamically generated form | Complete | `ui/components/parameter_form.py` handles int, float, str, date, choice types |
| 2.4 | Run selected strategy with custom parameters | Complete | Lines 102-108 submit to JobRunner |
| 2.5 | Real-time log streaming during execution | Complete | Line 115 calls `render_log_stream` |
| 2.6 | Display results inline: Plotly tear sheet, metrics table | Complete | Lines 132-180 with tabbed equity/drawdown/tax/metrics views |
| 2.7 | Save results to disk for later viewing in dashboard | Complete | Lines 121-127 use `ResultStore.save()` |
| **Module 3: Results Dashboard** | | | |
| 3.1 | Browse all saved backtest results (UI + legacy) | Complete | `ui/pages/3_dashboard.py` lines 46-67 + `ResultStore.list_results()` |
| 3.2 | View individual result: Plotly tear sheet, metrics table, parameter snapshot | Complete | Lines 121-183 with 5-tab view |
| 3.3 | Compare multiple strategies side-by-side | Complete | Lines 186-238: equity overlay + metrics comparison |
| 3.4 | Correlation matrix heatmap of strategy returns | Complete | Lines 222-234 |
| 3.5 | Display pre-existing PNG results from legacy runs as fallback | Complete | Lines 125-129 for legacy PNG display; `ResultStore._discover_legacy_results()` |
| **Module 4: ML Research Viewer** | | | |
| 4.1 | Trigger research pipeline run with streaming logs | Complete | `ui/pages/4_research.py` lines 125-148 |
| 4.2 | View feature importance rankings (interactive bar charts) | Complete | Lines 159-183 with model/target selectors |
| 4.3 | View model performance metrics (accuracy, AUC, precision, recall, F1) | Complete | Lines 187-213 |
| 4.4 | View robustness comparison across targets | Complete | Lines 234-238 (PNG display) |
| 4.5 | Display research summary report | Complete | Lines 218-223 |
| **Cross-cutting: Job Runner** | | | |
| 5.1 | Execute Python functions in background threads | Complete | `ui/services/job_runner.py` |
| 5.2 | Capture stdout/stderr in real-time via queue | Complete | `_CapturingStream` + `_RedirectingStream` |
| 5.3 | Provide status (pending, running, completed, failed) and progress | Partial | Status is tracked; **progress percentage is not implemented** |
| 5.4 | Store job results for retrieval | Complete | `job.result` field |
| 5.5 | Allow only one job of each type at a time | Complete | Lines 164-169: returns existing job ID if already running |
| **Cross-cutting: Strategy Plugin Architecture** | | | |
| 6.1 | Abstract base class StrategyBase with name, description, get_default_parameters, get_parameter_schema, generate_signals | Complete | `backtests/core/strategy_base.py`. Note: method is `get_parameter_specs()` instead of PRD's `get_parameter_schema()` but functionally equivalent. |
| 6.2 | Registry that discovers strategies from `backtests/strategies/` | Complete | `backtests/core/strategy_registry.py` with auto-discovery |
| 6.3 | Shared data loader that prepares common DataFrames once per session | Complete | `backtests/core/shared_data.py` + `@st.cache_resource` in backtest_service |
| 6.4 | Legacy scripts remain runnable from CLI unchanged | Complete | All 43 `*_backtest.py` files untouched |
| **Infrastructure** | | | |
| 7.1 | Streamlit + Plotly added to requirements.txt | Complete | `streamlit>=1.30.0` and `plotly>=5.18.0` added |
| 7.2 | Directory structure matches PRD | Complete | All specified files present |
| 7.3 | .gitignore updated for results/ and .streamlit/ | Complete | `.gitignore` updated |

**Compliance Score**: 26/27 requirements fully met, 1 partial (progress percentage not implemented in job runner)

## Issues Found

### Critical (must fix before shipping)

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/services/backtest_service.py:101`**: Format string crash. `{metrics[1].get('Sharpe', 'N/A'):.2f}` will raise `ValueError: Unknown format code 'f' for object of type 'str'` if the `'Sharpe'` key is missing from the metrics dict. The fallback `'N/A'` is a string and cannot be formatted with `:.2f`. Fix: either remove the format spec or handle the case, e.g. `f"... {metrics[1].get('Sharpe', 0):.2f}"` or use a conditional.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/pages/1_pipeline.py:127-135`**: Rebuild confirmation logic is broken. When `submitted` is `True` and `rebuild` is `True`, the code calls `st.warning()` and then `st.button("Confirm Rebuild")`. But because Streamlit reruns the entire script on every interaction, the `submitted` variable will be `False` on the next rerun (the form was already submitted), so the confirmation button will never be visible long enough to click. The confirmation flow needs to use `st.session_state` to persist the "rebuild requested" state across reruns.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/components/log_stream.py:53-59`**: Blocking while-loop in Streamlit. The `while runner.get_active_job(job_type).status == JobStatus.RUNNING` loop with `time.sleep(0.5)` will block the Streamlit script thread. While the intention is to show live logs, this will prevent the page from responding to user interactions and may cause Streamlit to appear frozen. Modern Streamlit idiom is to use `st.rerun()` with session state, or use `st.fragment` (Streamlit >=1.33) for partial reruns. On Streamlit 1.50.0 the `log_container.code()` updates inside the `while` loop may work visually because they update `st.empty()`, but the entire page is unresponsive during this time (no sidebar navigation, no other interactions).

### Important (should fix)

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/services/backtest_service.py:75-82`**: Missing `monthly_sales_exemption` parameter. The `run_simulation()` call does not pass `monthly_sales_exemption`. Some legacy scripts use `monthly_sales_exemption=20_000` (the R$20K Brazilian tax exemption for small monthly sales). The UI backtests will produce different after-tax results than the legacy CLI runs. Either add a `monthly_sales_exemption` ParameterSpec to the common parameters, or document this as an intentional deviation.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/services/pipeline_service.py:119`**: SQL injection via f-string. `f"SELECT * FROM {table_name} LIMIT {limit}"` injects `table_name` directly into SQL. While current callers only pass hardcoded table names, the function signature accepts arbitrary strings. Should validate `table_name` against a whitelist (e.g., `ALLOWED_TABLES = {"prices", "corporate_actions", "stock_actions", "detected_splits"}`).

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/services/pipeline_service.py:31-38`**: Import inside cached function. `from b3_pipeline.storage import get_summary_stats` is called inside `get_db_stats()` which is decorated with `@st.cache_data(ttl=60)`. The import itself will execute every time the cache expires. More importantly, `get_summary_stats(conn)` is called, but the `conn` object is created before the import. If the import fails, the function falls through to `_manual_stats(conn)` which is good, but the try/except catches `ImportError` only -- if `get_summary_stats` raises a different exception, it propagates unhandled. Should catch `Exception` or be more explicit.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/pages/4_research.py:41-64`**: Duplicated PALETTE dict and `_apply_dark_theme` function. These are already defined in `ui/components/charts.py`. The research page re-declares them locally, creating maintenance burden and potential drift. Should import from `ui.components.charts`.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/strategy_registry.py:54-68`**: Silent exception swallowing during discovery. When a strategy module fails to import or instantiate, the exception is caught with bare `except Exception: continue/pass`. This makes debugging registration failures extremely difficult. Should at minimum log a warning.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/services/backtest_service.py:20-27`**: `@st.cache_resource` used for mutable DataFrames. The `_get_shared_data_cached` function returns a dict of pandas DataFrames. `@st.cache_resource` does not copy the return value -- all callers share the same object instances. While the current strategy implementations correctly call `.copy()` on `ret`, a future strategy that accidentally mutates a shared DataFrame (e.g., `shared_data["ret"]["NEW_COL"] = ...` without copying first) will corrupt the cache for all subsequent callers. Consider using `@st.cache_data` (which serializes/deserializes, creating independent copies) or adding a deep-copy wrapper.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/services/result_store.py:22-23`**: Module-level side effect. `RESULTS_DIR.mkdir(exist_ok=True)` executes at import time, creating a `results/` directory whenever any module imports `result_store`. This is a side effect that can surprise users. Should be deferred to first use (e.g., inside `save()`).

### Minor (nice to fix)

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/run_ui.py:11`**: `import sys` is unused. Remove the dead import.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/pages/3_dashboard.py:224`**: `import numpy as np` inside a conditional block. Should be at the top of the file with other imports.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/pages/3_dashboard.py:243`**: `from pathlib import Path` inside `st.expander`. Should be at the top of the file.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/pages/1_pipeline.py:65`**: `import pandas as pd` inside `st.expander`. Should be at the top of the file.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/components/charts.py:300`**: `format=[None] + [",.2f"] * (len(df.columns) - 1)` assumes the first column is always non-numeric. If the metrics dict doesn't have a "Strategy" column first, this format specification will be wrong.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/services/pipeline_service.py:118-120`**: Connection not closed in exception case. If `pd.read_sql_query` raises, the `conn.close()` on line 120 will not execute. Should use a `try/finally` or context manager.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/core/shared_data.py:92-96`**: Misleading variable names persist from legacy code. `vol_60d = ret.rolling(5).std()` computes a 5-period rolling std on monthly data, which is ~5 months, not 60 days. Similarly `vol_20d = ret.rolling(2).std()` is a 2-month rolling std, not 20 days. These names are inherited from the original codebase (documented in memory), but propagating them into new shared_data infrastructure cements the confusion.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/ui/components/metrics_table.py:65`**: Type annotation `specs: list = None` should be `specs: list | None = None` or `Optional[list]` for correctness.

- **`/Users/nickmaglowsch/person-projects/b3-data-pipeline/backtests/strategies/copom_easing.py:61`**: `bool(is_easing.iloc[i])` can raise if `is_easing.iloc[i]` is NaN. Should use `pd.notna(is_easing.iloc[i]) and bool(is_easing.iloc[i])` or handle NaN explicitly. Same pattern in `cdi_ma200.py:61`, `research_multifactor.py:72`, `adaptive_low_vol.py:95`.

## What Looks Good

- **Clean separation of concerns.** The `ui/services/`, `ui/components/`, `ui/pages/` layering is well thought out. Services handle business logic, components handle reusable UI rendering, pages compose them together.

- **Strategy plugin architecture is solid.** `ParameterSpec` provides rich metadata for UI form generation. The `StrategyBase` abstract class has a minimal, clean interface. The auto-discovery registry with `pkgutil.iter_modules` is the right approach.

- **Shared data loader centralizes expensive computations.** `build_shared_data()` mirrors what 43 scripts each did independently and avoids redundant DB queries and computations. Caching with `@st.cache_resource` is appropriate.

- **Thread-based stdout capture is well-engineered.** The `_RedirectingStream` / `_CapturingStream` pattern in `job_runner.py` correctly handles per-thread output capture without breaking the main thread's stdout. The thread-local approach is the right design.

- **All 13 strategy plugins correctly copy shared data before mutation.** Every strategy does `r = ret.copy()` before adding synthetic columns, preventing corruption of the cached shared data.

- **Plotly chart library is comprehensive.** 8 chart functions covering equity curves, drawdown, tax detail, tax drag, metrics table, correlation heatmap, strategy comparison, and parameter sensitivity. Dark theme is consistently applied.

- **ResultStore gracefully handles legacy results.** The `_discover_legacy_results()` method finds PNG files from CLI runs and presents them alongside new UI results. The fallback to PNG display for legacy results is a nice touch.

- **Defensive error handling throughout.** Try/except blocks around chart rendering, data loading, and service calls with user-friendly error messages. The graceful degradation when services are not available (import error guards on each page) is well done.

- **Parameter form component handles edge cases.** Date parsing, choice index lookup, float formatting precision, and the "Reset to Defaults" button are all thoughtfully implemented.

## Recommendations

1. **Fix the three critical issues first** -- the format string crash in `backtest_service.py:101`, the broken rebuild confirmation flow in `1_pipeline.py:127-135`, and the blocking while-loop in `log_stream.py:53-59`. The first will crash at runtime, the second silently fails, and the third freezes the UI.

2. **Add `monthly_sales_exemption` as a common parameter** or document the deviation. This is a real-world tax calculation difference that will produce noticeably different after-tax results.

3. **Add input validation to `get_table_sample()`** with a whitelist of allowed table names.

4. **Extract the duplicated PALETTE and `_apply_dark_theme`** from `4_research.py` into imports from `ui.components.charts`.

5. **Add logging to strategy discovery failures** in `strategy_registry.py`. Even a `print()` would be better than silent `pass`.

6. **Consider `@st.cache_data` instead of `@st.cache_resource`** for `_get_shared_data_cached` to prevent accidental cache mutation. The serialization overhead is worth the safety guarantee.

7. **Move module-level `RESULTS_DIR.mkdir()`** into the `save()` method to avoid import-time side effects.

8. **For the log streaming component**, consider using Streamlit's `st.fragment` decorator (available in 1.33+) to create a rerunnable fragment that polls without blocking the full page.
