"""
Configuration for B3 Feature Discovery Engine.
"""
from __future__ import annotations

from pathlib import Path

# Import parent config for shared constants
from research import config as parent_config

# ── Paths ────────────────────────────────────────────────────────────────

FEATURE_STORE_DIR = Path(__file__).parent.parent / "feature_store"
FEATURES_DIR = FEATURE_STORE_DIR / "features"
EVALUATIONS_DIR = FEATURE_STORE_DIR / "evaluations"
REGISTRY_PATH = FEATURE_STORE_DIR / "registry.json"
CATALOG_PATH = parent_config.OUTPUT_DIR / "feature_catalog.json"
DISCOVERY_REPORT_PATH = parent_config.OUTPUT_DIR / "discovery_report.txt"

# Ensure directories exist
FEATURE_STORE_DIR.mkdir(parents=True, exist_ok=True)
FEATURES_DIR.mkdir(parents=True, exist_ok=True)
EVALUATIONS_DIR.mkdir(parents=True, exist_ok=True)

# ── Parametric Sweep Windows ──────────────────────────────────────────

MOMENTUM_WINDOWS = [1, 2, 3, 5, 10, 15, 20, 30, 60, 120, 250]
MA_WINDOWS = [5, 10, 20, 50, 100, 200]
VOLATILITY_WINDOWS = [5, 10, 20, 40, 60, 120]
VOLUME_WINDOWS = [5, 10, 20, 40, 60]
BETA_WINDOWS = [60, 120, 252]
SKEW_KURT_WINDOWS = [20, 60, 120]
MAX_MIN_RET_WINDOWS = [20, 60, 120]
WIN_RATE_WINDOWS = [20, 60, 120]
AMIHUD_WINDOWS = [20, 60]
AUTOCORR_WINDOWS = [20, 60]
MEAN_REVERSION_WINDOWS = [5, 10, 20]
EWM_SPANS = [5, 10, 20, 40]
HIGH_LOW_RANGE_WINDOWS = [5, 20, 60]
IBOV_WINDOWS = [10, 20, 40, 63]
CDI_WINDOWS = [21, 42, 63, 126]

# ── Evaluation Config ──────────────────────────────────────────────────

FORWARD_HORIZONS = [5, 10, 20, 60]
PRIMARY_HORIZON = 20
TRAIN_FRACTION = 0.70
IC_DECAY_LAGS = [1, 5, 20]
IC_RECENCY_YEARS = 5
IC_TIMESERIES_FILE = "ic_timeseries.parquet"

# ── Pruning Config ────────────────────────────────────────────────────

MIN_IC_THRESHOLD = 0.005
MAX_CORRELATION = 0.90
MAX_FEATURES = 500
MAX_NAN_RATE = 0.30
MIN_VARIANCE_DATES_FRAC = 0.90

# ── Generation Config ─────────────────────────────────────────────────

TOP_N_FOR_DELTA = 50
TOP_N_FOR_BINARY_OPS = 20
DELTA_PERIODS = [20]
UNARY_OPERATORS = ["rank", "zscore"]
BINARY_OPERATORS = ["ratio", "product"]
RATIO_TO_MEAN_PERIODS = [10, 20, 60]
TOP_N_FOR_RATIO_TO_MEAN = 50

# ── Universe Config (reuse from parent) ─────────────────────────────

START_DATE = parent_config.START_DATE
END_DATE = parent_config.END_DATE
DB_PATH = parent_config.DB_PATH
MIN_ADTV = parent_config.MIN_ADTV
MIN_PRICE = parent_config.MIN_PRICE
MIN_HISTORY_DAYS = parent_config.MIN_HISTORY_DAYS
ADTV_WINDOW = parent_config.ADTV_WINDOW
OUTPUT_DIR = parent_config.OUTPUT_DIR
