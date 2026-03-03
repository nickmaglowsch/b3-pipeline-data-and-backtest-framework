"""
Feature catalog export for consumption by backtest framework.
"""
from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime

import pandas as pd

from research.discovery import config
from research.discovery.store import FeatureStore


def parse_feature_id(feature_id: str) -> dict:
    """
    Parse a feature ID into a human-readable formula and component tree.

    Examples:
        "Return_20d" -> {"formula_human": "Return_20d", "formula_components": {...}}
        "rank__Return_60d" -> {"formula_human": "rank(Return_60d)", ...}
        "ratio__Return_60d__Rolling_vol_60d" -> {"formula_human": "Return_60d / Rolling_vol_60d", ...}
    """
    parts = feature_id.split("__")

    if len(parts) == 1:
        # Base signal
        return {
            "formula_human": feature_id,
            "formula_components": {"base": feature_id},
        }

    # Check if first part is a known operator
    unary_ops = {"rank", "zscore"}
    binary_ops = {"ratio", "product", "delta5", "delta20"}

    first = parts[0]

    # Handle delta operators (delta20__...)
    if first.startswith("delta"):
        period = first[5:]  # e.g., "20" from "delta20"
        operand = "__".join(parts[1:])
        return {
            "formula_human": f"delta{period}({operand})",
            "formula_components": {
                "operator": f"delta{period}",
                "operand": operand,
            },
        }

    if first in unary_ops:
        operand = "__".join(parts[1:])
        return {
            "formula_human": f"{first}({operand})",
            "formula_components": {
                "operator": first,
                "operand": operand,
            },
        }

    if first in binary_ops:
        if len(parts) >= 3:
            a = parts[1]
            b = "__".join(parts[2:])
            op_symbol = "/" if first == "ratio" else "*"
            return {
                "formula_human": f"{a} {op_symbol} {b}",
                "formula_components": {
                    "operator": first,
                    "operand_a": a,
                    "operand_b": b,
                },
            }

    # Fallback
    return {
        "formula_human": feature_id,
        "formula_components": {"raw": feature_id},
    }


def compute_category_summary(
    store: FeatureStore,
    kept_feature_ids: list[str],
    evaluations_df: pd.DataFrame,
) -> dict:
    """
    Compute per-category summary: count, avg IC_IR, best feature per category.
    """
    registry = store.get_registry()
    summary = {}

    for feature_id in kept_feature_ids:
        if feature_id not in registry["features"]:
            continue

        category = registry["features"][feature_id].get("category", "unknown")

        if category not in summary:
            summary[category] = {
                "count": 0,
                "features": [],
            }

        summary[category]["count"] += 1
        summary[category]["features"].append(feature_id)

    # Compute avg IC_IR
    if not evaluations_df.empty:
        ic_col = None
        for col in evaluations_df.columns:
            if "ic_ir" in col and "fwd_20d" in col:
                ic_col = col
                break

        if ic_col:
            for category in summary:
                cat_features = summary[category]["features"]
                cat_evals = evaluations_df[evaluations_df["feature_id"].isin(cat_features)]
                if not cat_evals.empty:
                    summary[category]["avg_ic_ir"] = round(
                        cat_evals[ic_col].abs().mean(), 4
                    )

    # Simplify output
    return {
        cat: {
            "count": summary[cat]["count"],
            "avg_ic_ir": summary[cat].get("avg_ic_ir", 0),
        }
        for cat in summary
    }


def validate_catalog(path: Path) -> bool:
    """Load the catalog JSON and verify it has the expected structure."""
    with open(path) as f:
        catalog = json.load(f)

    assert "features" in catalog, "Missing 'features' key"
    assert len(catalog["features"]) > 0, "No features in catalog"
    assert all("id" in f and "metrics" in f for f in catalog["features"]), \
        "Features missing 'id' or 'metrics'"
    return True


def export_catalog(
    store: FeatureStore,
    kept_feature_ids: list[str],
    pruning_summary: dict,
    output_path: Path = None,
    adj_close: pd.DataFrame = None,
) -> Path:
    """
    Export the feature catalog JSON.

    Args:
        store: FeatureStore with evaluation results
        kept_feature_ids: feature IDs that survived pruning
        pruning_summary: dict from pruning pipeline
        output_path: where to write the JSON (default: config.CATALOG_PATH)

    Returns:
        Path to the written file.
    """
    output_path = output_path or config.CATALOG_PATH

    registry = store.get_registry()
    evaluations_df = store.get_all_evaluations()

    # Sort by IC_IR on primary horizon
    ic_col = f"ic_ir_fwd_{config.PRIMARY_HORIZON}d"
    if ic_col not in evaluations_df.columns:
        ic_cols = [c for c in evaluations_df.columns if "ic_ir" in c]
        if ic_cols:
            ic_col = ic_cols[0]

    if not evaluations_df.empty:
        evaluations_df = evaluations_df.sort_values(
            ic_col, ascending=False, key=abs
        )

    # Get date range (use passed-in adj_close to avoid reloading)

    start_date = str(adj_close.index.min().date()) if adj_close is not None else "unknown"
    end_date = str(adj_close.index.max().date()) if adj_close is not None else "unknown"

    # Compute train cutoff
    if adj_close is not None:
        dates = adj_close.index.sort_values()
        n_dates = len(dates)
        cutoff_idx = int(n_dates * config.TRAIN_FRACTION)
        train_end = str(dates[cutoff_idx].date())
        test_start = str(dates[cutoff_idx + 1].date()) if cutoff_idx + 1 < n_dates else train_end
    else:
        train_end = "unknown"
        test_start = "unknown"

    # Build catalog
    catalog = {
        "generated_at": datetime.now().isoformat(),
        "pipeline_version": "1.0",
        "evaluation_date_range": {
            "start": start_date,
            "end": end_date,
            "train_end": train_end,
            "test_start": test_start,
        },
        "primary_horizon": f"fwd_{config.PRIMARY_HORIZON}d",
        "total_generated": pruning_summary.get("initial_count", 0),
        "total_after_pruning": len(kept_feature_ids),
        "features": [],
        "category_summary": {},
    }

    # Build feature entries
    for rank, feature_id in enumerate(kept_feature_ids, start=1):
        if feature_id not in registry["features"]:
            continue

        feature_data = registry["features"][feature_id]
        eval_data = feature_data.get("evaluation", {})

        # Parse formula
        formula_info = parse_feature_id(feature_id)

        # Build metrics dict (per-horizon)
        metrics = {}
        for horizon, horizon_metrics in eval_data.items():
            if isinstance(horizon_metrics, dict) and "mean_ic" in horizon_metrics:
                metrics[horizon] = horizon_metrics

        # Collect decay and turnover
        decay_values = {}
        turnover_value = 0.0

        for horizon, horizon_metrics in eval_data.items():
            if isinstance(horizon_metrics, dict):
                if "decay" in horizon_metrics:
                    decay_values = horizon_metrics["decay"]
                if "turnover" in horizon_metrics:
                    turnover_value = horizon_metrics["turnover"]

        feature_entry = {
            "rank": rank,
            "id": feature_id,
            "category": feature_data.get("category", "unknown"),
            "level": feature_data.get("level", 0),
            "formula_human": formula_info.get("formula_human", feature_id),
            "formula_components": formula_info.get("formula_components", {}),
            "params": feature_data.get("params", {}),
            "metrics": metrics,
            "turnover": turnover_value,
            "decay": decay_values,
        }

        catalog["features"].append(feature_entry)

    # Add category summary
    catalog["category_summary"] = compute_category_summary(
        store, kept_feature_ids, evaluations_df
    )

    # Write JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(catalog, f, indent=2)

    # Validate
    validate_catalog(output_path)

    return output_path
