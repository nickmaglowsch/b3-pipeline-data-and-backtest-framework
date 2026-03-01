"""
Result Store Service
====================
Persists backtest results to disk and discovers legacy CLI results (PNGs/CSVs).
Each new result is saved as a directory: results/{timestamp}_{strategy_name}/
  - metadata.json  -- strategy name, parameters, timestamp, metrics
  - data/          -- parquet files for each pandas Series (equity curves, etc.)

Legacy results that were saved with data.pkl are still loadable (read-only).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"

# Keys that are persisted as individual parquet files inside the data/ subdirectory.
_DATA_KEYS = [
    "pretax_values",
    "aftertax_values",
    "ibov_ret",
    "cdi_ret",
    "tax_paid",
    "loss_carryforward",
    "turnover",
]


def _serialize_params(params: dict) -> dict:
    """Convert params dict to JSON-serialisable types."""
    safe: dict = {}
    for k, v in params.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            safe[k] = v
        elif hasattr(v, "isoformat"):
            safe[k] = v.isoformat()
        else:
            safe[k] = str(v)
    return safe


def _validate_inside_results_dir(path: Path) -> None:
    """Raise ValueError if *path* resolves outside the results directory.

    This prevents path-traversal attacks where a crafted result_id such as
    ``../../etc`` could trick ``shutil.rmtree`` or file-loading operations
    into acting on arbitrary filesystem locations.
    """
    resolved = os.path.realpath(path)
    results_base = os.path.realpath(RESULTS_DIR)
    if not resolved.startswith(results_base + os.sep) and resolved != results_base:
        raise ValueError(f"Invalid result path (outside results directory): {path}")


class BacktestResult:
    """Represents one saved (or discovered legacy) backtest result."""

    def __init__(
        self,
        result_id: str,
        strategy_name: str,
        timestamp: str,
        params: dict,
        metrics: list[dict],
        data_path: Optional[Path] = None,
        legacy_png_path: Optional[Path] = None,
        is_legacy: bool = False,
    ) -> None:
        self.result_id = result_id
        self.strategy_name = strategy_name
        self.timestamp = timestamp
        self.params = params
        self.metrics = metrics
        self.data_path = data_path
        self.legacy_png_path = legacy_png_path
        self.is_legacy = is_legacy

    def __repr__(self) -> str:
        return (
            f"BacktestResult(id={self.result_id!r}, "
            f"strategy={self.strategy_name!r}, ts={self.timestamp!r})"
        )


class ResultStore:
    """Persist and retrieve backtest results."""

    # ── Save ──────────────────────────────────────────────────────────────────

    def save(self, result_dict: dict) -> str:
        """
        Save a backtest result dict (from run_backtest()) to disk.

        Data is persisted as individual parquet files (one per Series) instead
        of a single pickle blob, avoiding arbitrary-code-execution risks.

        Returns:
            The result_id string.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        strategy_name = result_dict["strategy_name"]
        # Sanitise name for filesystem use
        safe_name = strategy_name.replace("/", "_").replace(" ", "_")
        result_id = f"{timestamp}_{safe_name}"
        RESULTS_DIR.mkdir(exist_ok=True)
        result_dir = RESULTS_DIR / result_id
        result_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "result_id": result_id,
            "strategy_name": strategy_name,
            "timestamp": timestamp,
            "params": _serialize_params(result_dict.get("params", {})),
            "metrics": result_dict.get("metrics", []),
        }
        with open(result_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2, default=str)

        # Persist each data series as a parquet file.
        data_dir = result_dir / "data"
        data_dir.mkdir(exist_ok=True)
        for key in _DATA_KEYS:
            value = result_dict.get(key)
            if value is None:
                continue
            if isinstance(value, pd.Series):
                value.to_frame(name=key).to_parquet(data_dir / f"{key}.parquet")
            elif isinstance(value, pd.DataFrame):
                value.to_parquet(data_dir / f"{key}.parquet")

        return result_id

    # ── List ──────────────────────────────────────────────────────────────────

    def list_results(self) -> list[BacktestResult]:
        """
        List all saved results (newest first) plus legacy CLI results.
        """
        results: list[BacktestResult] = []

        if RESULTS_DIR.exists():
            for d in sorted(RESULTS_DIR.iterdir(), reverse=True):
                if d.is_dir() and (d / "metadata.json").exists():
                    try:
                        with open(d / "metadata.json") as f:
                            meta = json.load(f)
                        results.append(BacktestResult(
                            result_id=meta["result_id"],
                            strategy_name=meta.get("strategy_name", d.name),
                            timestamp=meta.get("timestamp", ""),
                            params=meta.get("params", {}),
                            metrics=meta.get("metrics", []),
                            data_path=d,
                        ))
                    except Exception:
                        pass

        results.extend(self._discover_legacy_results())
        return results

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_data(self, result: BacktestResult) -> Optional[dict]:
        """Load the full data dict for a new-format result.

        Tries the new parquet-based layout first (``data/*.parquet``), then
        falls back to the legacy ``data.pkl`` pickle file for backward
        compatibility.  Path validation is performed before any I/O to guard
        against path-traversal attacks.
        """
        if result.data_path is None or not result.data_path.exists():
            return None

        _validate_inside_results_dir(result.data_path)

        # --- New format: data/ directory with individual parquet files --------
        data_dir = result.data_path / "data"
        if data_dir.is_dir():
            data: dict = {}
            for key in _DATA_KEYS:
                pq_path = data_dir / f"{key}.parquet"
                if pq_path.exists():
                    df = pd.read_parquet(pq_path)
                    # If the parquet was saved from a Series via to_frame(),
                    # convert it back to a Series.
                    if len(df.columns) == 1:
                        data[key] = df.iloc[:, 0]
                    else:
                        data[key] = df
                else:
                    data[key] = None
            return data

        # --- Legacy format: data.pkl (pickle) --------------------------------
        # SECURITY WARNING: pickle.load can execute arbitrary code.  This
        # fallback exists only for backward compatibility with results saved
        # before the migration to parquet.  The path has already been validated
        # above so at least we know the file lives inside the results directory.
        import pickle  # noqa: S403 – intentional, guarded by path validation

        pkl_path = result.data_path / "data.pkl"
        if pkl_path.exists():
            _validate_inside_results_dir(pkl_path)
            try:
                with open(pkl_path, "rb") as f:
                    return pickle.load(f)  # noqa: S301
            except Exception:
                return None

        return None

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, result_id: str) -> None:
        """Delete a saved result directory.

        Validates that the resolved path is inside the results directory
        before deletion to prevent path-traversal attacks via crafted
        ``result_id`` values (e.g. ``../../important_dir``).
        """
        result_dir = RESULTS_DIR / result_id
        _validate_inside_results_dir(result_dir)
        if result_dir.exists():
            shutil.rmtree(result_dir)

    # ── Legacy discovery ──────────────────────────────────────────────────────

    def _discover_legacy_results(self) -> list[BacktestResult]:
        """Find existing PNG/CSV results from previous CLI runs."""
        legacy: list[BacktestResult] = []

        # Project root PNGs
        for png in sorted(PROJECT_ROOT.glob("*_backtest.png")):
            name = png.stem.replace("_backtest", "").replace("_", " ").title()
            mtime = datetime.fromtimestamp(png.stat().st_mtime)
            legacy.append(BacktestResult(
                result_id=f"legacy_{png.stem}",
                strategy_name=name,
                timestamp=mtime.strftime("%Y%m%d_%H%M%S"),
                params={},
                metrics=[],
                legacy_png_path=png,
                is_legacy=True,
            ))

        # backtests/ directory PNGs
        backtests_dir = PROJECT_ROOT / "backtests"
        if backtests_dir.exists():
            for png in sorted(backtests_dir.glob("*.png")):
                name = png.stem.replace("_", " ").title()
                mtime = datetime.fromtimestamp(png.stat().st_mtime)
                legacy.append(BacktestResult(
                    result_id=f"legacy_{png.stem}",
                    strategy_name=name,
                    timestamp=mtime.strftime("%Y%m%d_%H%M%S"),
                    params={},
                    metrics=[],
                    legacy_png_path=png,
                    is_legacy=True,
                ))

        return legacy
