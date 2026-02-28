"""
Result Store Service
====================
Persists backtest results to disk and discovers legacy CLI results (PNGs/CSVs).
Each new result is saved as a directory: results/{timestamp}_{strategy_name}/
  - metadata.json  -- strategy name, parameters, timestamp, metrics
  - data.pkl       -- serialised pandas objects (equity curves, tax series, etc.)
"""
from __future__ import annotations

import json
import pickle
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


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

        data = {
            "pretax_values": result_dict.get("pretax_values"),
            "aftertax_values": result_dict.get("aftertax_values"),
            "ibov_ret": result_dict.get("ibov_ret"),
            "cdi_ret": result_dict.get("cdi_ret"),
            "tax_paid": result_dict.get("tax_paid"),
            "loss_carryforward": result_dict.get("loss_carryforward"),
            "turnover": result_dict.get("turnover"),
        }
        with open(result_dir / "data.pkl", "wb") as f:
            pickle.dump(data, f)

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
                            data_path=d / "data.pkl",
                        ))
                    except Exception:
                        pass

        results.extend(self._discover_legacy_results())
        return results

    # ── Load ──────────────────────────────────────────────────────────────────

    def load_data(self, result: BacktestResult) -> Optional[dict]:
        """Load the full data dict for a new-format result."""
        if result.data_path and result.data_path.exists():
            try:
                with open(result.data_path, "rb") as f:
                    return pickle.load(f)
            except Exception:
                return None
        return None

    # ── Delete ────────────────────────────────────────────────────────────────

    def delete(self, result_id: str) -> None:
        """Delete a saved result directory."""
        result_dir = RESULTS_DIR / result_id
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
