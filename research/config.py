from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DB_PATH = PROJECT_ROOT / "b3_market_data.sqlite"
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Date range
START_DATE = "2005-01-01"
END_DATE = "2026-12-31"

# Universe filters
MIN_ADTV = 1_000_000       # R$1M minimum 20-day average daily financial volume
MIN_PRICE = 1.0             # R$1.00 minimum raw close price
MIN_HISTORY_DAYS = 200      # Minimum trading days for MA200

# Liquidity window
ADTV_WINDOW = 20            # Rolling window for ADTV computation (trading days)

# Target
FORWARD_PERIOD_20D = 20     # Trading days for primary target
FORWARD_PERIOD_60D = 60     # Trading days for robustness target

# Feature windows
MA_WINDOWS = [20, 50, 200]
VOL_WINDOWS = [20, 60]
ATR_SPAN = 14
DRAWDOWN_WINDOW = 60
VOLUME_ZSCORE_WINDOW = 20
VOLUME_RATIO_SHORT = 5
VOLUME_RATIO_LONG = 20
MOMENTUM_WINDOWS = [1, 5, 20, 60]
CDI_CUMULATIVE_WINDOW = 63  # ~3 months of trading days
IBOV_WINDOW = 20

# Train/test split
TRAIN_FRACTION = 0.70

# Model hyperparameters (deliberately minimal -- this is discovery, not tuning)
RF_PARAMS = {
    "n_estimators": 200,       # importance rankings stabilize well before 500
    "max_depth": 15,           # prevents very deep trees, big speedup
    "random_state": 42,
    "n_jobs": 2,               # -1 duplicates the full dataset per worker process
}
XGB_PARAMS = {
    "n_estimators": 500,
    "random_state": 42,
    "eval_metric": "logloss",
    "verbosity": 0,
    "n_jobs": -1,  # XGBoost uses OpenMP threads (not multiprocessing) -- memory-safe
}

# Permutation importance
PERM_N_REPEATS = 10
PERM_RANDOM_STATE = 42

# External data
IBOV_TICKER = "^BVSP"

# Output filenames
IMPORTANCE_CSV = "importance_results.csv"
METRICS_JSON = "model_metrics.json"
IMPORTANCE_PLOT = "feature_importance_top15.png"
ROBUSTNESS_PLOT = "robustness_comparison.png"
SUMMARY_TXT = "research_summary.txt"
