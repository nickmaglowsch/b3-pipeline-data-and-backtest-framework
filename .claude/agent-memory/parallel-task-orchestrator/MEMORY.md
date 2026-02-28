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
