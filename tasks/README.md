# B3 Data Pipeline -- Streamlit UI Tasks

## Summary

Build a modular Streamlit-based management UI for the B3 data pipeline and quantitative research platform. The UI covers 4 modules: data pipeline management, backtest runner with full parameter editing, results dashboard with interactive Plotly charts, and ML research viewer. Strategies are refactored into a plugin architecture with a common base class. All long-running operations execute as background jobs with real-time log streaming.

## Total Tasks: 13

## Estimated Complexity
- **High complexity**: tasks 05, 07, 09, 13
- **Medium complexity**: tasks 02, 03, 04, 06, 08, 12
- **Lower complexity**: tasks 01, 10, 11

## Dependency Graph

```
task-01 (Project Setup)
  |
  +---> task-02 (Job Runner) --------+---> task-06 (Pipeline Page)
  |                                   |
  |                                   +---> task-07 (Backtest Runner) <--- task-05 <--- task-04
  |                                   |       ^                             ^
  |                                   |       |                             |
  +---> task-03 (Plotly Charts) ------+       +-- task-11 (Param Form) <---+
  |                                   |       |
  |                                   |       +-- task-10 (Metrics) <--- task-01
  |                                   |       |
  +---> task-08 (Result Store) -------+       +-- task-08 (Result Store)
  |                                   |
  |                                   +---> task-09 (Dashboard)
  |                                   |
  |                                   +---> task-12 (Research Page)
  |
  +---> task-04 (Strategy Base) -----> task-05 (Migrate Strategies)

  ALL ------> task-13 (Integration & Polish)
```

## Execution Order

The recommended build order, respecting dependencies:

### Phase 1: Foundation (can be done in parallel)
1. **task-01**: Project setup and dependencies
2. **task-04**: Strategy base class and registry (no UI dependency)

### Phase 2: Core Infrastructure (can be done in parallel after Phase 1)
3. **task-02**: Background job runner with log streaming
4. **task-03**: Plotly chart library
5. **task-05**: Migrate core strategies to plugins (depends on task-04)
6. **task-08**: Result store service
7. **task-10**: Metrics table component
8. **task-11**: Dynamic parameter form component (depends on task-04)

### Phase 3: Pages (can be done in parallel after Phase 2)
9. **task-06**: Pipeline management page
10. **task-07**: Backtest runner page
11. **task-09**: Results dashboard page
12. **task-12**: ML research viewer page

### Phase 4: Integration
13. **task-13**: Integration testing and polish

## Instructions

These task files are prompts for AI agents. Each file contains everything needed to implement that task independently. Delete each file after the task is completed. When all files are deleted, the feature is complete.

## Open Questions / Decisions

1. **Streamlit multipage vs single page with tabs**: The plan uses Streamlit's native multipage pattern (`pages/` directory). If this causes issues with shared state, consider switching to a single-page app with `st.tabs()`.

2. **Shared data caching lifetime**: The shared data dict (loaded from SQLite + Yahoo + BCB) is ~200MB in memory. Using `@st.cache_resource` means it persists for the entire Streamlit session. Consider adding a "Clear Cache" button on the Pipeline page after data updates.

3. **Legacy strategy migration scope**: Task 05 migrates 13 strategies (8 core + 5 standalone). The remaining ~30 standalone scripts can be migrated incrementally as a follow-up effort.

4. **Python 3.9 compatibility**: The `ParameterSpec` class uses `list[ParameterSpec]` type hints which require `from __future__ import annotations` on Python 3.9. Ensure all new files include this import.
