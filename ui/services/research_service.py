"""
Research Service
================
Loads existing research results from research/output/ and provides a function
to run the research pipeline inside a JobRunner thread.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
RESEARCH_OUTPUT = PROJECT_ROOT / "research" / "output"


def get_research_results() -> Optional[dict]:
    """
    Load existing research results from research/output/.
    Returns None if no results exist yet.
    """
    metrics_path = RESEARCH_OUTPUT / "model_metrics.json"
    importance_path = RESEARCH_OUTPUT / "importance_results.csv"
    summary_path = RESEARCH_OUTPUT / "research_summary.txt"

    if not metrics_path.exists():
        return None

    results: dict = {}

    # Load metrics JSON
    try:
        with open(metrics_path) as f:
            results["metrics"] = json.load(f)
    except Exception:
        results["metrics"] = {}

    # Load feature importance CSV
    if importance_path.exists():
        try:
            results["importance"] = pd.read_csv(importance_path)
        except Exception:
            pass

    # Load summary text
    if summary_path.exists():
        try:
            results["summary"] = summary_path.read_text()
        except Exception:
            pass

    # PNG plot paths
    for fname in ["feature_importance_top15.png", "robustness_comparison.png"]:
        path = RESEARCH_OUTPUT / fname
        if path.exists():
            results[f"png_{fname}"] = str(path)

    # Last run timestamp (from metrics file mtime)
    results["last_run"] = metrics_path.stat().st_mtime

    return results


def run_research_pipeline() -> None:
    """
    Run the research pipeline. Designed to be called inside a JobRunner thread.
    """
    sys.path.insert(0, str(PROJECT_ROOT))
    from research.main import main as research_main
    research_main()
